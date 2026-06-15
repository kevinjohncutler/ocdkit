"""Canonical WGSL for the tileserve tile pipelines — one source of truth.

The browser viewer (:mod:`ocdkit.tileserve.viewer`) and the headless wgpu-native
renderer (:mod:`ocdkit.tileserve.headless`) must run *identical* shader math so a
script-side render matches what the browser paints. These strings are that shared
source; ``headless`` imports them directly and a drift test asserts the viewer's
embedded copies stay byte-identical (see ``tests`` / the repro harness).

Three tile pipelines mirror the viewer's WebGPU backend:

- ``RGB``            — sample an 8-bit/linear RGB texture, passthrough.
- ``HDR``            — peak-normalized linear-P3 [0,1] → ``OETF(d * headroom)``.
                       At ``headroom == 1`` this collapses to ordinary sRGB, so it
                       byte-matches the 8-bit path (the reproducible headless case).
- ``int_wgsl()``     — R32F/R16F scalar → ``normalize(lo,hi)`` → LUT lookup, with
                       an optional extended-sRGB OETF for the HDR-colormap variant.

Uniform byte layouts (must match ``viewer.py`` ``frameEnd`` writes):
  RGB  U = vp:vec4f, tr:vec4f                       (8 f32 = 32 B)
  HDR  U = vp:vec4f, hr:vec4f, tr:vec4f             (12 f32 = 48 B; hr.x = headroom)
  INT  U = vp:vec4f, lohi:vec4f, tr:vec4f           (12 f32 = 48 B; lohi.xy = lo,hi)
"""
from __future__ import annotations

# ── RGB pipeline (viewer.py: const RGB) ──────────────────────────────────────
RGB = """
struct U{ vp:vec4f, tr:vec4f };
@group(0)@binding(0) var t:texture_2d<f32>;
@group(0)@binding(1) var s:sampler;
@group(1)@binding(0) var<uniform> u:U;
struct VO{ @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VO{
  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var o:VO; o.pos=vec4f(p[i],0,1); o.uv=vec2f(p[i].x*0.5+0.5, p[i].y*0.5+0.5); return o; }
@fragment fn fs(in:VO)->@location(0) vec4f{
  let tc=u.vp.xy+vec2f(in.uv.x,1.0-in.uv.y)*u.vp.zw;
  let st=(tc-u.tr.xy)/u.tr.zw;
  let oob=st.x<0.0||st.x>1.0||st.y<0.0||st.y>1.0;
  let c=textureSample(t,s,clamp(st,vec2f(0.0),vec2f(1.0)));   // sample in uniform ctrl flow
  return select(vec4f(c.rgb,1.0), vec4f(0,0,0,0), oob); }"""

# ── HDR-RGB pipeline (viewer.py: const HDR) ──────────────────────────────────
HDR = """
struct U{ vp:vec4f, hr:vec4f, tr:vec4f };
@group(0)@binding(0) var t:texture_2d<f32>;
@group(0)@binding(1) var s:sampler;
@group(1)@binding(0) var<uniform> u:U;
struct VO{ @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VO{
  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var o:VO; o.pos=vec4f(p[i],0,1); o.uv=vec2f(p[i].x*0.5+0.5, p[i].y*0.5+0.5); return o; }
// extended P3 transfer (same curve sRGB uses); >1 allowed for HDR highlights
fn oetf(v:vec3f)->vec3f{
  let a=max(v,vec3f(0.0));
  return select(12.92*a, 1.055*pow(a,vec3f(1.0/2.4))-0.055, a>vec3f(0.0031308)); }
@fragment fn fs(in:VO)->@location(0) vec4f{
  let tc=u.vp.xy+vec2f(in.uv.x,1.0-in.uv.y)*u.vp.zw;
  let st=(tc-u.tr.xy)/u.tr.zw;
  let oob=st.x<0.0||st.x>1.0||st.y<0.0||st.y>1.0;
  let d=textureSample(t,s,clamp(st,vec2f(0.0),vec2f(1.0))).rgb;
  let lin=d*u.hr.x;                                        // P3-linear, peak→headroom
  return select(vec4f(oetf(lin),1.0), vec4f(0,0,0,0), oob); }"""

# ── intensity pipeline (viewer.py: const INT + HDR_CMAP runtime concat) ───────
# Base is assembled exactly as the JS does: the fragment return and a trailing
# oetf() fn are appended when the HDR-colormap variant is active. Keeping the
# assembly identical here means int_wgsl(False) byte-matches the viewer's plain
# (uint8-LUT) INT shader and int_wgsl(True) matches its hdr_cmap form.
_INT_HEAD = """
struct U{ vp:vec4f, lohi:vec4f, tr:vec4f };
@group(0)@binding(0) var t:texture_2d<f32>;
@group(0)@binding(1) var s:sampler;
@group(1)@binding(0) var lut:texture_2d<f32>;
@group(1)@binding(1) var lsmp:sampler;
@group(2)@binding(0) var<uniform> u:U;
struct VO{ @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VO{
  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var o:VO; o.pos=vec4f(p[i],0,1); o.uv=vec2f(p[i].x*0.5+0.5, p[i].y*0.5+0.5); return o; }
@fragment fn fs(in:VO)->@location(0) vec4f{
  let tc=u.vp.xy+vec2f(in.uv.x,1.0-in.uv.y)*u.vp.zw;
  let st=(tc-u.tr.xy)/u.tr.zw;
  let oob=st.x<0.0||st.x>1.0||st.y<0.0||st.y>1.0;
  let v=textureSample(t,s,clamp(st,vec2f(0.0),vec2f(1.0))).r;   // sample in uniform ctrl flow
  let n=clamp((v-u.lohi.x)/max(u.lohi.y-u.lohi.x,1e-12),0.0,1.0);
  let col=textureSample(lut,lsmp,vec2f(n,0.5));
  return select("""

_INT_OETF = "\nfn oetf(v:vec3f)->vec3f{let a=max(v,vec3f(0.0));return select(12.92*a,1.055*pow(a,vec3f(1.0/2.4))-0.055,a>vec3f(0.0031308));}"


def int_wgsl(hdr_cmap: bool = False) -> str:
    """Intensity shader. ``hdr_cmap`` lifts the LUT colour through the extended
    sRGB OETF (HDR-colormap mode); ``False`` is the plain SDR uint8-LUT path."""
    ret = "vec4f(oetf(col.rgb),1.0)" if hdr_cmap else "vec4f(col.rgb,1.0)"
    return _INT_HEAD + ret + ", vec4f(0,0,0,0), oob); }" + (_INT_OETF if hdr_cmap else "")


# Convenience: the plain SDR intensity shader (matches the de-risk INT_WGSL).
INT = int_wgsl(False)
