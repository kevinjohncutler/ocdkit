"""Generic browser viewer for :mod:`ocdkit.tileserve` — a zoomable, GPU-colormap
tile grid with a pluggable LinkedPanel (SpectraGL density / ScatterGL points),
served by ``/grid``. A host injects its data via the engine attachments and
parameterizes the viewer per request (panel kind, HDR colormap, an optional
reference-token regex). Nothing here is application-specific.
"""

def _spectra_gl_js() -> str:
    """Read the SpectraGL WebGPU density renderer JS from the ocdkit package, to
    inject into the grid viewer (mirrors the SVG figure's injection). Read fresh
    so edits to the JS show up without reinstalling; degrades to a no-op shim if
    ocdkit isn't importable so the rest of the viewer still works."""
    try:
        import os as _os
        import ocdkit.plot as _op
        p = _os.path.join(_os.path.dirname(_op.__file__), "web", "spectra_density_gl.js")
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:                       # pragma: no cover - environment dependent
        return f"/* SpectraGL unavailable: {e} */ self.SpectraGL=self.SpectraGL||null;"

def _scatter_gl_js() -> str:
    """Read the ScatterGL panel JS from ocdkit (the linked discrete-object-scatter
    panel, interchangeable with SpectraGL). Same fresh-read + no-op-shim policy."""
    try:
        import os as _os
        import ocdkit.plot as _op
        p = _os.path.join(_os.path.dirname(_op.__file__), "web", "scatter_gl.js")
        with open(p, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:                       # pragma: no cover - environment dependent
        return f"/* ScatterGL unavailable: {e} */ self.ScatterGL=self.ScatterGL||null;"

_LUTS_JSON = _LUTS_HDR_JSON = _LUTS_SDRF_JSON = None

def _luts_for(kind: str) -> str:
    import json
    from ocdkit.plot.luts import colormap_luts, DEFAULT_COLORMAPS
    return json.dumps(colormap_luts(DEFAULT_COLORMAPS, kind))

def _luts_json() -> str:
    global _LUTS_JSON
    if _LUTS_JSON is None:
        _LUTS_JSON = _luts_for("uint8")
    return _LUTS_JSON

def _luts_hdr_json() -> str:
    global _LUTS_HDR_JSON
    if _LUTS_HDR_JSON is None:
        _LUTS_HDR_JSON = _luts_for("hdr_float")
    return _LUTS_HDR_JSON

def _luts_sdr_float_json() -> str:
    global _LUTS_SDRF_JSON
    if _LUTS_SDRF_JSON is None:
        _LUTS_SDRF_JSON = _luts_for("sdr_float")
    return _LUTS_SDRF_JSON

_VIEWER_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>tile viewer</title>
<style>
 html,body{margin:0;height:100%;background:#0b0c0e;overflow:hidden}
 #c{width:100vw;height:100vh;display:block;touch-action:none}
 #hud{position:fixed;left:8px;top:8px;color:#8fd;font:12px/1.4 monospace;
   background:#000a;padding:5px 9px;border-radius:5px;white-space:pre;pointer-events:none}
 #tabs{position:fixed;right:8px;top:8px}
 #tabs button{font:12px monospace;margin-left:5px;background:#222;color:#ddd;
   border:1px solid #444;border-radius:4px;padding:3px 8px;cursor:pointer}
 #tabs button.on{background:#2a6;color:#000;border-color:#2a6}
</style></head><body>
<canvas id="c"></canvas><div id="hud">init…</div><div id="tabs"></div>
<script>
const SID="__SID__"; let LAYER="__LAYER__";
// Base path for this viewer's own requests, derived from its page URL — so the
// absolute routes work whether served at the origin root (/grid/<sid>) OR behind
// a sub-path proxy (…/proxy/<port>/grid/<sid>, e.g. jupyter-server-proxy).
const VBASE=location.pathname.replace(/(grid|viewgl|view)\/[^/]+\/?$/,'');
const canvas=document.getElementById('c'), hud=document.getElementById('hud'), tabs=document.getElementById('tabs');
let info=null, device=null, ctx=null, format=null, pipeline=null, sampler=null, ubuf=null;
const texCache=new Map();                 // "layer/level" -> {tex,bg,w,h}
let view={vx:0, vy:0, vw:1, vh:1};         // visible region in image-normalised [0,1]

async function init(){
  if(!navigator.gpu){ hud.textContent='no WebGPU in this browser'; return; }
  const adapter=await navigator.gpu.requestAdapter();
  if(!adapter){ hud.textContent='no WebGPU adapter (headless?)'; return; }
  device=await adapter.requestDevice();
  ctx=canvas.getContext('webgpu');
  format=navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({device, format, alphaMode:'opaque'});
  sampler=device.createSampler({magFilter:'nearest', minFilter:'linear'});
  ubuf=device.createBuffer({size:16, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
  const code=`
struct U { vp: vec4f };
@group(0) @binding(0) var t: texture_2d<f32>;
@group(0) @binding(1) var s: sampler;
@group(0) @binding(2) var<uniform> u: U;
struct VO{ @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VO{
  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var uv=array<vec2f,3>(vec2f(0,0),vec2f(2,0),vec2f(0,2));
  var o:VO; o.pos=vec4f(p[i],0,1); o.uv=uv[i]; return o;
}
@fragment fn fs(in:VO)->@location(0) vec4f{
  let tc=u.vp.xy + in.uv*u.vp.zw;
  if(tc.x<0.0||tc.x>1.0||tc.y<0.0||tc.y>1.0){ return vec4f(0,0,0,1); }
  return textureSample(t,s,tc);
}`;
  const mod=device.createShaderModule({code});
  pipeline=device.createRenderPipeline({layout:'auto',
    vertex:{module:mod,entryPoint:'vs'},
    fragment:{module:mod,entryPoint:'fs',targets:[{format}]},
    primitive:{topology:'triangle-list'}});
  info=await (await fetch(VBASE+'info/'+SID)).json();
  if(!LAYER||!info.layers[LAYER]) LAYER=Object.keys(info.layers)[0];
  for(const L of Object.keys(info.layers)){
    const b=document.createElement('button'); b.textContent=L; b.dataset.l=L;
    b.onclick=()=>{ LAYER=L; syncTabs(); draw(); };
    tabs.appendChild(b);
  }
  syncTabs(); resize(); draw();
}
function syncTabs(){ for(const b of tabs.children) b.className=(b.dataset.l===LAYER)?'on':''; }
function resize(){ const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(canvas.clientWidth*dpr); canvas.height=Math.round(canvas.clientHeight*dpr); }
window.addEventListener('resize',()=>{resize();draw();});

function pickLevel(){
  const dims=info.layers[LAYER];                 // [[h,w],...] coarse->fine
  const need=canvas.width / Math.max(view.vw,1e-6);   // texels needed across the view
  for(let i=0;i<dims.length;i++){ if(dims[i][1]>=need) return i; }
  return dims.length-1;
}
async function getTex(layer, level){
  const key=layer+'/'+level;
  if(texCache.has(key)) return texCache.get(key);
  const r=await fetch(VBASE+'tile/'+SID+'/'+layer+'/'+level+'?fmt=raw');
  const w=+r.headers.get('X-Level-Width'), h=+r.headers.get('X-Level-Height');
  const ch=+r.headers.get('X-Channels'), dt=r.headers.get('X-Dtype');
  const buf=await r.arrayBuffer();
  let bytes;
  if(dt==='uint8' && ch===4){ bytes=new Uint8Array(buf); }
  else if(dt==='float32'){                          // RGB float -> srgb-ish u8 (prototype)
    const f=new Float32Array(buf); bytes=new Uint8Array(w*h*4);
    for(let i=0;i<w*h;i++){ for(let k=0;k<3;k++){
      let v=f[i*ch+k]; v=v<=0.0031308?12.92*v:1.055*Math.pow(Math.max(v,0),1/2.4)-0.055;
      bytes[i*4+k]=Math.max(0,Math.min(255,v*255)); } bytes[i*4+3]=255; }
  } else { bytes=new Uint8Array(buf); }
  const tex=device.createTexture({size:[w,h], format:'rgba8unorm',
    usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
  device.queue.writeTexture({texture:tex}, bytes, {bytesPerRow:w*4, rowsPerImage:h}, [w,h]);
  const bg=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
    {binding:0,resource:tex.createView()},{binding:1,resource:sampler},{binding:2,resource:{buffer:ubuf}}]});
  const e={tex,bg,w,h}; texCache.set(key,e); return e;
}
async function draw(){
  if(!device) return;
  const level=pickLevel();
  let e=null;                                       // progressive: show best cached now
  for(let l=level;l>=0;l--){ if(texCache.has(LAYER+'/'+l)){ e=texCache.get(LAYER+'/'+l); break; } }
  if(e) render(e);
  if(!texCache.has(LAYER+'/'+level)){ render(await getTex(LAYER, level)); }
}
function render(e){
  device.queue.writeBuffer(ubuf,0,new Float32Array([view.vx,view.vy,view.vw,view.vh]));
  const enc=device.createCommandEncoder();
  const pass=enc.beginRenderPass({colorAttachments:[{view:ctx.getCurrentTexture().createView(),
    loadOp:'clear',clearValue:{r:0,g:0,b:0,a:1},storeOp:'store'}]});
  pass.setPipeline(pipeline); pass.setBindGroup(0,e.bg); pass.draw(3); pass.end();
  device.queue.submit([enc.finish()]);
  hud.textContent=`layer ${LAYER}\nlevel ${pickLevel()}/${info.layers[LAYER].length-1}  tex ${e.w}x${e.h}\nview ${(view.vw*100).toFixed(1)}%  (drag=pan, wheel=zoom)`;
}
let drag=null;
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY}; canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{ if(!drag)return;
  view.vx-=(e.clientX-drag.x)/canvas.clientWidth*view.vw;
  view.vy-=(e.clientY-drag.y)/canvas.clientHeight*view.vh;
  drag={x:e.clientX,y:e.clientY}; draw(); });
canvas.addEventListener('pointerup',()=>{drag=null;});
canvas.addEventListener('wheel',e=>{e.preventDefault();
  const f=Math.exp(e.deltaY*0.0015);
  const cx=e.clientX/canvas.clientWidth, cy=e.clientY/canvas.clientHeight;
  const ax=view.vx+cx*view.vw, ay=view.vy+cy*view.vh;
  view.vw=Math.min(1,view.vw*f); view.vh=Math.min(1,view.vh*f);
  view.vx=ax-cx*view.vw; view.vy=ay-cy*view.vh; draw();
},{passive:false});
init();
</script></body></html>"""

_VIEWER_GL_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>tile viewer (gl)</title>
<style>
 html,body{margin:0;height:100%;background:#0b0c0e;overflow:hidden}
 #c{width:100vw;height:100vh;display:block;touch-action:none}
 #hud{position:fixed;left:8px;top:8px;color:#8fd;font:12px/1.4 monospace;
   background:#000a;padding:5px 9px;border-radius:5px;white-space:pre;pointer-events:none}
 #tabs{position:fixed;right:8px;top:8px}
 #tabs button{font:12px monospace;margin-left:5px;background:#222;color:#ddd;
   border:1px solid #444;border-radius:4px;padding:3px 8px;cursor:pointer}
 #tabs button.on{background:#2a6;color:#000;border-color:#2a6}
</style></head><body>
<canvas id="c"></canvas><div id="hud">init…</div><div id="tabs"></div>
<script>
const SID="__SID__"; let LAYER="__LAYER__";
// Base path for this viewer's own requests, derived from its page URL — so the
// absolute routes work whether served at the origin root (/grid/<sid>) OR behind
// a sub-path proxy (…/proxy/<port>/grid/<sid>, e.g. jupyter-server-proxy).
const VBASE=location.pathname.replace(/(grid|viewgl|view)\/[^/]+\/?$/,'');
const canvas=document.getElementById('c'), hud=document.getElementById('hud'), tabs=document.getElementById('tabs');
let info=null, gl=null, prog=null, uVp=null, texCache=new Map();
let view={vx:0, vy:0, vw:1, vh:1};
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);
  if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))console.warn(gl.getShaderInfoLog(x));return x;}
async function init(){
  gl=canvas.getContext('webgl2',{antialias:false});
  if(!gl){ hud.textContent='no WebGL2'; return; }
  const VS=`#version 300 es
in vec2 p; out vec2 uv; void main(){ uv=vec2(p.x*0.5+0.5, p.y*0.5+0.5); gl_Position=vec4(p,0,1); }`;
  const FS=`#version 300 es
precision highp float; in vec2 uv; out vec4 o; uniform sampler2D tex; uniform vec4 u_vp;
void main(){ vec2 tc=u_vp.xy+vec2(uv.x,1.0-uv.y)*u_vp.zw;
  if(tc.x<0.0||tc.x>1.0||tc.y<0.0||tc.y>1.0){ o=vec4(0,0,0,1); return; }
  o=texture(tex,tc); }`;
  prog=gl.createProgram(); gl.attachShader(prog,sh(gl.VERTEX_SHADER,VS));
  gl.attachShader(prog,sh(gl.FRAGMENT_SHADER,FS)); gl.linkProgram(prog); gl.useProgram(prog);
  const vbo=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,vbo);
  gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
  const lp=gl.getAttribLocation(prog,'p'); gl.enableVertexAttribArray(lp);
  gl.vertexAttribPointer(lp,2,gl.FLOAT,false,0,0);
  uVp=gl.getUniformLocation(prog,'u_vp'); gl.uniform1i(gl.getUniformLocation(prog,'tex'),0);
  info=await (await fetch(VBASE+'info/'+SID)).json();
  if(!LAYER||!info.layers[LAYER]) LAYER=Object.keys(info.layers)[0];
  for(const L of Object.keys(info.layers)){ const b=document.createElement('button');
    b.textContent=L; b.dataset.l=L; b.onclick=()=>{LAYER=L;syncTabs();draw();}; tabs.appendChild(b); }
  syncTabs(); resize(); draw();
}
function syncTabs(){ for(const b of tabs.children) b.className=(b.dataset.l===LAYER)?'on':''; }
function resize(){ const dpr=window.devicePixelRatio||1;
  canvas.width=Math.round(canvas.clientWidth*dpr); canvas.height=Math.round(canvas.clientHeight*dpr);
  gl.viewport(0,0,canvas.width,canvas.height); }
window.addEventListener('resize',()=>{resize();draw();});
function pickLevel(){ const dims=info.layers[LAYER]; const need=canvas.width/Math.max(view.vw,1e-6);
  for(let i=0;i<dims.length;i++){ if(dims[i][1]>=need) return i; } return dims.length-1; }
async function getTex(layer,level){ const key=layer+'/'+level;
  if(texCache.has(key)) return texCache.get(key);
  const r=await fetch(VBASE+'tile/'+SID+'/'+layer+'/'+level+'?fmt=raw');
  const w=+r.headers.get('X-Level-Width'), h=+r.headers.get('X-Level-Height');
  const ch=+r.headers.get('X-Channels'), dt=r.headers.get('X-Dtype'); const buf=await r.arrayBuffer();
  let bytes;
  if(dt==='uint8'&&ch===4){ bytes=new Uint8Array(buf); }
  else if(dt==='float32'){ const f=new Float32Array(buf); bytes=new Uint8Array(w*h*4);
    for(let i=0;i<w*h;i++){ for(let k=0;k<3;k++){ let v=f[i*ch+k];
      v=v<=0.0031308?12.92*v:1.055*Math.pow(Math.max(v,0),1/2.4)-0.055;
      bytes[i*4+k]=Math.max(0,Math.min(255,v*255)); } bytes[i*4+3]=255; } }
  else { bytes=new Uint8Array(buf); }
  const tex=gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D,tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,false);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,bytes);
  const e={tex,w,h}; texCache.set(key,e); return e;
}
async function draw(){ if(!gl) return; const level=pickLevel(); let e=null;
  for(let l=level;l>=0;l--){ if(texCache.has(LAYER+'/'+l)){ e=texCache.get(LAYER+'/'+l); break; } }
  if(e) render(e);
  if(!texCache.has(LAYER+'/'+level)){ render(await getTex(LAYER,level)); }
}
function render(e){ gl.bindTexture(gl.TEXTURE_2D,e.tex); gl.uniform4f(uVp,view.vx,view.vy,view.vw,view.vh);
  gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT); gl.drawArrays(gl.TRIANGLES,0,3);
  hud.textContent=`[webgl2] layer ${LAYER}\nlevel ${pickLevel()}/${info.layers[LAYER].length-1}  tex ${e.w}x${e.h}\nview ${(view.vw*100).toFixed(1)}%`;
}
let drag=null;
canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{ if(!drag)return;
  view.vx-=(e.clientX-drag.x)/canvas.clientWidth*view.vw; view.vy-=(e.clientY-drag.y)/canvas.clientHeight*view.vh;
  drag={x:e.clientX,y:e.clientY}; draw(); });
canvas.addEventListener('pointerup',()=>{drag=null;});
canvas.addEventListener('wheel',e=>{e.preventDefault(); const f=Math.exp(e.deltaY*0.0015);
  const cx=e.clientX/canvas.clientWidth, cy=e.clientY/canvas.clientHeight;
  const ax=view.vx+cx*view.vw, ay=view.vy+cy*view.vh;
  view.vw=Math.min(1,view.vw*f); view.vh=Math.min(1,view.vh*f);
  view.vx=ax-cx*view.vw; view.vy=ay-cy*view.vh; draw(); },{passive:false});
init();
</script></body></html>"""

_GRID_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>key slices</title>
<style>
 html,body{margin:0;height:100%;background:transparent;overflow:hidden;color-scheme:dark}
 #wrap{position:relative} #c{display:block;touch-action:none}
 /* SVG overlay: vector annotation layer (cell frames + labels) ON TOP of the
    zooming canvas. pointer-events:none so pan/zoom still reaches the canvas. */
 #ovl{position:absolute;left:0;top:0;pointer-events:none;overflow:visible}
 /* spectra-density raster — sits in the panel box, BEHIND the #ovl axes/refs */
 #speccv{position:absolute;left:0;top:0;pointer-events:none;display:block}
 /* 2D hover-highlight overlay over the density (pointer-events ON to pick lines) */
 #specovl{position:absolute;left:0;top:0;display:block}
 #sgtip{position:fixed;pointer-events:none;background:rgba(0,0,0,0.8);color:#fff;
   font:11px system-ui,sans-serif;padding:3px 6px;border-radius:4px;display:none;
   z-index:20;white-space:nowrap;line-height:1.3}
 .lab{position:absolute;color:#fff;font:12px/1 system-ui,sans-serif;
   text-shadow:0 0 3px #000,0 0 3px #000;pointer-events:none}
 /* ctl (cmap picker etc.) is a STRIP ABOVE the figure and hud (debug info) a
    STRIP BELOW — normal document flow (ctl, #wrap, hud), never overlaying the
    tiles/plot. Heights/fonts are scaled by the layout's k (set in layout()) and
    the embed aspect reserves CTL_H0+HUD_H0, so nothing is clipped. */
 #hud{display:flex;align-items:center;box-sizing:border-box;color:#8fd;
   font:11px monospace;padding:2px 10px;white-space:pre;pointer-events:none}
 #ctl{display:flex;gap:4px;align-items:center;box-sizing:border-box;color:#ccc;
   font:12px system-ui,sans-serif;padding:2px 8px}
 #ctl select,#ctl button{background:#222;color:#ddd;border:1px solid #555;
   border-radius:3px;font:inherit;padding:1px 6px;cursor:pointer}
 /* Global figure title — a strip at the very TOP (figtitle → ctl → #wrap → hud),
    normal document flow so it never overlays the tiles; height/font scale with k
    in _sizeBars() and the Python embed aspect reserves TITLE_H0 when present. */
 #figtitle{display:flex;align-items:center;justify-content:center;box-sizing:border-box;
   color:#ddd;font:600 16px system-ui,sans-serif;padding:0 10px;white-space:nowrap;
   overflow:hidden;text-overflow:ellipsis;pointer-events:none}
 /* HDR/save/copy action buttons (icons), bottom-right; mirrors ocdkit's
    .ocd-svgfig-actions styling. Outside #wrap → excluded from the capture. */
 #acts{position:fixed;right:10px;bottom:10px;display:flex;gap:8px;justify-content:flex-end;
   opacity:0.8;transition:opacity .15s;z-index:30}
 #acts:hover{opacity:1}
 #acts button{background:none;border:none;cursor:pointer;padding:0;color:#9a9a9a;
   transition:transform .15s ease,color .15s ease}
 #acts button:hover{transform:scale(1.2);color:#e0e0e0}
 #acts button:disabled{opacity:0.5;cursor:default}
 #acts button svg{width:20px;height:20px;display:block;fill:currentColor}
 #acts button.hdr-off{color:#c97a3a}    /* warm tint = SDR mode active */
</style></head><body>
<div id="wrap"><canvas id="c"></canvas><canvas id="speccv" data-spectra-density="1"></canvas><canvas id="specovl" data-spectra-overlay="1"></canvas><svg id="ovl" xmlns="http://www.w3.org/2000/svg"></svg><div id="labs"></div></div><div id="hud">init…</div><div id="sgtip"></div><div id="acts"><button id="hdrbtn" title="HDR: on"><svg width="20" height="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 28 16"><text x="14" y="13" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-weight="700" font-size="12" fill="currentColor">HDR</text></svg></button><button id="savebtn" title="Save as PNG"><svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 640 640"><path fill="currentColor" d="M160 144C151.2 144 144 151.2 144 160L144 480C144 488.8 151.2 496 160 496L480 496C488.8 496 496 488.8 496 480L496 237.3C496 233.1 494.3 229 491.3 226L416 150.6L416 240C416 257.7 401.7 272 384 272L224 272C206.3 272 192 257.7 192 240L192 144L160 144zM240 144L240 224L368 224L368 144L240 144zM96 160C96 124.7 124.7 96 160 96L402.7 96C419.7 96 436 102.7 448 114.7L525.3 192C537.3 204 544 220.3 544 237.3L544 480C544 515.3 515.3 544 480 544L160 544C124.7 544 96 515.3 96 480L96 160zM256 384C256 348.7 284.7 320 320 320C355.3 320 384 348.7 384 384C384 419.3 355.3 448 320 448C284.7 448 256 419.3 256 384z"/></svg></button><button id="copybtn" title="Copy as PNG"><svg width="20" height="20" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24"><path fill="currentColor" fill-rule="evenodd" d="M4.75 3A1.75 1.75 0 003 4.75v9.5c0 .966.784 1.75 1.75 1.75h1.5a.75.75 0 000-1.5h-1.5a.25.25 0 01-.25-.25v-9.5a.25.25 0 01.25-.25h9.5a.25.25 0 01.25.25v1.5a.75.75 0 001.5 0v-1.5A1.75 1.75 0 0014.25 3h-9.5zm5 5A1.75 1.75 0 008 9.75v9.5c0 .966.784 1.75 1.75 1.75h9.5A1.75 1.75 0 0021 19.25v-9.5A1.75 1.75 0 0019.25 8h-9.5zM9.5 9.75a.25.25 0 01.25-.25h9.5a.25.25 0 01.25.25v9.5a.25.25 0 01-.25.25h-9.5a.25.25 0 01-.25-.25v-9.5z"/></svg></button></div>
<script>__SPECTRA_GL__</script>
<script>__SCATTER_GL__</script>
<script>
const SID="__SID__";
// Base path for this viewer's own requests (see _VIEWER_HTML) — root or proxied.
const VBASE=location.pathname.replace(/(grid|viewgl|view)\/[^/]+\/?$/,'');
const canvas=document.getElementById('c'), labs=document.getElementById('labs'), hud=document.getElementById('hud');
const ovl=document.getElementById('ovl'), SVGNS='http://www.w3.org/2000/svg';
let info=null, dpr=window.devicePixelRatio||1;
// {name:[256*4 uint8]} colormaps, injected server-side. Guarded so a stale
// in-kernel server (serving new HTML through an old /grid that didn't
// substitute the placeholder) degrades to no-colormap instead of a silent
// ReferenceError that hangs init — restart the kernel to refresh the server.
const PANEL_KIND="__PANEL_KIND__";
// LinkedPanel: the bottom panel is pluggable — SpectraGL (density lines) or
// ScatterGL (discrete object points). Both expose decodeAttrs/render/highlight/
// highlightById/clearHighlight; the grid drives them identically (hover->id->
// snap+outline, cell->highlightById). Default spectra so the key-slice view is
// unchanged.
const PANEL=(PANEL_KIND==="scatter"&&self.ScatterGL)?self.ScatterGL:(self.SpectraGL||self.ScatterGL);
// Optional reference-token regex (host-injected): when a cell label matches, the
// matching reference overlays / top-axis entries light up. null disables it, so
// the viewer stays generic — the host supplies the label scheme (e.g. /R\d+/g).
const REF_TOKEN_RE=__REF_TOKEN_RE__;
const refTokens=(lab)=>(lab && REF_TOKEN_RE) ? (lab.match(REF_TOKEN_RE)||[]) : [];
let LUTS={}; try{ LUTS=__LUTS__; }catch(e){ console.warn('LUTS not injected — stale tile server? restart the kernel'); }
// SDR float LUTs (linear-P3 ≤1, same INT_HDR pipeline + OETF → reproduces the
// plain sRGB colormap exactly) — the SDR half of the HDR toggle for the
// hdr_cmap intensity tiles. Empty when hdr_cmap is off (plain uint8 LUTs).
let LUTS_SDR={}; try{ LUTS_SDR=__LUTS_SDR__; }catch(e){}
// intensity normalization mode: self (per-tile) | global (pooled across the
// reference layers) | bitdepth (0..bit_max). Reduction layers always self-scale.
let CMAP="magma", NORMMODE="self", RGLO=Infinity, RGHI=-Infinity, BITMAX=65535;
// HDR colormap mode (server-substituted): "" = SDR uint8 LUT (default); a name =
// the WebGPU layer uploads the ocdkit HDR-lifted float LUT and the INT pipeline
// OETFs it on the extended canvas, so colormapped tiles glow into the headroom.
const HDR_CMAP=(typeof Float16Array!=='undefined')?"__HDR_CMAP__":""; if(HDR_CMAP) CMAP=HDR_CMAP;
// Live display HDR headroom (max HDR luminance / SDR white): 1 on SDR displays,
// up to ~several on HDR. RGB tiles are peak-normalized linear-P3 [0,1] (1.0 =
// XDR peak); the WebGPU HDR path renders OETF(d * HEADROOM) so the peak adapts
// to the display and never clips. Polled live; a change triggers a redraw.
let HEADROOM=1;
// Manual HDR gain override: 0 = auto (track display headroom); >0 = forced
// multiplier on the peak-normalized RGB tile (1.0=SDR white → gain pushes the
// peak into HDR). Initialized from ?hdr_gain= (server-substituted), also live
// via the "HDR gain" slider. Auto is unusable on Chrome builds with no numeric
// headroom API (most), so the explicit gain is the practical HDR control.
let HRMANUAL=__HRGAIN__;
let AUTOHR=1, HRSRC='?';   // last auto headroom reading + which signal gave it
const texCache=new Map(); const _filled=new Set();
// Masks-tile ncolor toggle: click the "Masks" label to swap the underlying tile
// from the seg/mean image to the ncolor pixel-segmentation raster (lazy layer —
// computed server-side on first request). _texOf() remaps which texture a cell
// renders; everything else (outline, white highlight, layout) is unchanged.
let _masksNcolor=false;
function _texOf(label){ return (label==='Masks' && _masksNcolor) ? 'ncolor' : label; }
// Click handler for the "Masks" label. ncolor is a LAZY layer: the first toggle
// triggers the server-side ncolor.label compute (~270 ms; we 204-retry here),
// keeping the OLD Masks frame on screen until the coarse level is ready so the
// swap is clean (no blank flash). Refine then sharpens it. Toggling back to the
// seg/mean image is instant (Masks is already cached).
async function toggleMasksNcolor(){
  _masksNcolor=!_masksNcolor;
  if(_masksNcolor){
    for(let tries=0; tries<60 && _masksNcolor; tries++){
      if(await getTex('ncolor', 0)) break;        // computed + uploaded → swap
      await new Promise(r=>setTimeout(r, 250));    // still computing — retry
    }
  }
  draw();   // ncolor (coarse → refine sharpens), or back to the cached Masks image
}
// Idle pyramid WARM: after first paint, prefetch+upload every (cell,level) during
// idle so the FIRST zoom finds finer levels already on the GPU. A cold-cache zoom
// otherwise hiccups crossing a level threshold (the 2nd run is smooth only because
// the kernel cached the stacks → layers attach instantly). Pauses on interaction.
let _warmDone=false, _warmStop=false, _warmRunning=false, _warmResumeT=0;
let cells=[];                              // {label, x,y,w,h} CSS px in #wrap
let excState=null, excChipsEl=null;        // live RGB compose (per-excitation toggles)
let view={cx:0.5, cy:0.5, s:1.0};          // shared: centre (FOV-norm) + scale (FOV fraction across a cell)
// Seg-outline stroke: constant in IMAGE px (scales WITH zoom — thicker as you
// zoom into the cells) rather than constant screen px. ``OUTLINE_IMG_PX`` is the
// full stroke width in source/image px; a small device-px floor keeps it from
// vanishing when zoomed all the way out (the whole FOV downsampled to one cell).
const OUTLINE_IMG_PX=0.75, OUTLINE_MIN_DPX=0.25;
// ctl/hud strip heights in Wref units (scaled by k; the Python embed aspect
// reserves CTL_H0+HUD_H0 too, so flow-stacking ctl+figure+hud fits exactly).
const CTL_H0=30, HUD_H0=22, TITLE_H0=26;
let _kNow=1;
function _sizeBars(){
  const c=document.getElementById('ctl');
  if(c){ c.style.height=(CTL_H0*_kNow)+'px'; c.style.fontSize=Math.max(9,12*_kNow)+'px'; }
  hud.style.height=(HUD_H0*_kNow)+'px'; hud.style.fontSize=Math.max(8,11*_kNow)+'px';
  const t=document.getElementById('figtitle');
  if(t){ const h=(TITLE_H0*_kNow)+'px'; t.style.height=h; t.style.lineHeight=h;
         t.style.fontSize=Math.max(11,16*_kNow)+'px'; }
}
// Tile / plot-box corner radius (CSS px), exposed via ?rx=. The tile DATA is
// clipped to this radius with a CSS mask (rounded white rects over the canvas),
// so the rounded frame and the image corners match. 0 = sharp.
const RX=(()=>{ const v=parseFloat(new URLSearchParams(location.search).get('rx')); return isNaN(v)?2:Math.max(0,v); })();
// Tile-title position — mirrors ocdkit image_grid._label_position (same names +
// pad semantics) so the option set is unified: top_middle (default) / top_left /
// bottom_middle / bottom_left / above_middle. Returns [tx,ty,anchor,baseline].
const LPOS=(new URLSearchParams(location.search).get('lpos'))||'top_middle';
function _labelPos(x,y,w,h,lpos,pad){
  switch(lpos){
    case 'top_left':      return [x+pad,   y+pad,   'start', 'hanging'];
    case 'bottom_middle': return [x+w/2,   y+h-pad, 'middle','alphabetic'];
    case 'bottom_left':   return [x+pad,   y+h-pad, 'start', 'alphabetic'];
    case 'above_middle':  return [x+w/2,   y-pad,   'middle','alphabetic'];
    case 'top_middle': default: return [x+w/2, y+pad, 'middle','hanging'];
  }
}
// Axis tick stroke width: ?tickw= (CSS px at Wref scale); 0/absent = match the
// axes/frame stroke width (bw) — the default per user preference.
const TICKW=(()=>{ const v=parseFloat(new URLSearchParams(location.search).get('tickw')); return isNaN(v)?0:Math.max(0,v); })();
// Snap-to-cell padding (image px around the cell bbox when zooming); ?snap_pad=.
const SNAP_PAD=(()=>{ const v=parseFloat(new URLSearchParams(location.search).get('snap_pad')); return isNaN(v)?10:Math.max(0,v); })();
// Snap-to-cell INITIAL state (?snap=1 → start enabled); the ctl button still toggles.
const SNAP0=(new URLSearchParams(location.search).get('snap'))==='1';
// ── render backend (API-agnostic seam) ───────────────────────────────────
// The core below (layout / pan-zoom / cache / poll / controls / level pick /
// normalization decision) is graphics-API-agnostic. A `backend` implements the
// GPU half behind a fixed interface:
//   init(canvas) -> bool          acquire context, compile shaders, alloc geometry
//   setColormap(name, uint8[256*4]) upload the active LUT
//   createTile(meta, arrayBuffer) -> entry {w,h,mode,lo,hi,kind,bitmax,...gpu}
//   setOutline(arrayBuffer)        upload seg-outline segments (nSeg = bytes/16)
//   hasOutline() -> bool
//   frameBegin()                   clear, prep for per-cell scissored draws
//   paint(entry, vp, rect, lo, hi) draw one tile into rect=[x,y,wpx,hpx] (GL px, y-up)
//   paintOutline(vp, rect, hwpx, rgba) draw the seg outline into rect
//   frameEnd()                     present (no-op on GL; submits on WebGPU)
// `vp` is the shared viewport rect {x,y,w,h} in FOV-norm [0,1]; `rect` is in
// device px with y measured from the canvas bottom (GL convention) so backends
// can flip as needed.
let backend=null;

function GL2Backend(){
  let gl=null, prog=null, uVp=null, uTr=null, cprog=null, CU=null, lutTex=null, FLOAT_LINEAR=false;
  // Named outline sets: 'default' = the full seg outline; hosts may upload extra
  // per-group geometry (e.g. pass/fail cells) drawn with their own colours.
  let lprog=null, lineCorner=null, LU=null, lineSets={};
  function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);
    if(!gl.getShaderParameter(x,gl.COMPILE_STATUS))console.warn(gl.getShaderInfoLog(x));return x;}
  function mkprog(vs,fs){ const pr=gl.createProgram(); gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));
    gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs)); gl.bindAttribLocation(pr,0,'p');
    gl.linkProgram(pr); return pr; }
  return {
  name:'webgl2',
  init(canvas){
    gl=canvas.getContext('webgl2',{antialias:false});
    if(!gl){ return false; }
    const VS=`#version 300 es
in vec2 p; out vec2 uv; void main(){ uv=vec2(p.x*0.5+0.5,p.y*0.5+0.5); gl_Position=vec4(p,0,1); }`;
    // u_tr = the FOV-norm rect (x,y,w,h) this texture COVERS — (0,0,1,1) for a
    // full-FOV tile, the crop rect for a detail crop. Sampling maps the FOV coord
    // tc into the texture: st=(tc-u_tr.xy)/u_tr.zw; outside [0,1] → transparent so
    // the coarse base shows through (a crop only paints its own sub-rect).
    const FS=`#version 300 es
precision highp float; in vec2 uv; out vec4 o; uniform sampler2D tex; uniform vec4 u_vp; uniform vec4 u_tr;
void main(){ vec2 tc=u_vp.xy+vec2(uv.x,1.0-uv.y)*u_vp.zw; vec2 st=(tc-u_tr.xy)/u_tr.zw;
  if(st.x<0.0||st.x>1.0||st.y<0.0||st.y>1.0){ o=vec4(0,0,0,0); return; } o=vec4(texture(tex,st).rgb,1.0); }`;
    // colormap program: raw scalar intensity -> normalize(lo,hi) -> LUT lookup.
    // lo/hi/cmap are uniforms → contrast, colormap, and per-tile/global
    // normalization all toggle live with no re-colormap or re-upload.
    const CFS=`#version 300 es
precision highp float; in vec2 uv; out vec4 o;
uniform sampler2D tex; uniform sampler2D lut; uniform vec4 u_vp, u_tr; uniform float u_lo,u_hi;
void main(){ vec2 tc=u_vp.xy+vec2(uv.x,1.0-uv.y)*u_vp.zw; vec2 st=(tc-u_tr.xy)/u_tr.zw;
  if(st.x<0.0||st.x>1.0||st.y<0.0||st.y>1.0){ o=vec4(0,0,0,0); return; }
  float v=texture(tex,st).r; float n=clamp((v-u_lo)/max(u_hi-u_lo,1e-12),0.0,1.0);
  o=vec4(texture(lut, vec2(n,0.5)).rgb,1.0); }`;
    prog=mkprog(VS,FS); cprog=mkprog(VS,CFS);   // shared attrib loc 0 -> one vbo
    const vbo=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,vbo);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
    gl.useProgram(prog); uVp=gl.getUniformLocation(prog,'u_vp');
    uTr=gl.getUniformLocation(prog,'u_tr');
    gl.uniform1i(gl.getUniformLocation(prog,'tex'),0);
    gl.useProgram(cprog);
    CU={vp:gl.getUniformLocation(cprog,'u_vp'),lo:gl.getUniformLocation(cprog,'u_lo'),
        hi:gl.getUniformLocation(cprog,'u_hi'),tr:gl.getUniformLocation(cprog,'u_tr')};
    gl.uniform1i(gl.getUniformLocation(cprog,'tex'),0);
    gl.uniform1i(gl.getUniformLocation(cprog,'lut'),1);
    FLOAT_LINEAR=!!gl.getExtension('OES_texture_float_linear');
    lutTex=gl.createTexture(); gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,lutTex);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    gl.activeTexture(gl.TEXTURE0);
    // ── seg-outline line program — MITER-JOIN ribbon (instanced, shader-AA) ──
    // Each instance is one edge carrying (prev,p0,p1,next). The shader offsets
    // each END along the JOINT miter (bisector of the two adjacent edges) instead
    // of the edge normal, so neighbouring edges share the exact mitered vertex →
    // they abut with NO gap and NO overlap → a continuous, smooth outline with
    // uniform coverage (correct under alpha; no joint double-blend / end-extension
    // overdraw — that was the jagged/bloated look). Miter clamped (scl≥… via the
    // 0.25 floor) so sharp turns truncate instead of spiking.
    const LVS=`#version 300 es
in vec2 a_corner; in vec2 a_prev; in vec2 a_p0; in vec2 a_p1; in vec2 a_next;
uniform vec4 u_vp; uniform vec2 u_cpx; uniform float u_hw; out float v_perp;
vec2 toPx(vec2 s){ vec2 uv=(s-u_vp.xy)/u_vp.zw; return uv*u_cpx; }  // FOV-norm → cell device px
void main(){
  bool atP0=(a_corner.x<0.5);
  vec2 cur=toPx(atP0?a_p0:a_p1);
  vec2 aa=toPx(atP0?a_prev:a_p0); vec2 bb=toPx(atP0?a_p1:a_next);
  vec2 dIn=cur-aa, dOut=bb-cur;
  float lIn=length(dIn), lOut=length(dOut);
  vec2 tIn=lIn>1e-5?dIn/lIn:vec2(0.0), tOut=lOut>1e-5?dOut/lOut:vec2(0.0);
  if(lIn<=1e-5)tIn=tOut; if(lOut<=1e-5)tOut=tIn;
  vec2 nIn=vec2(-tIn.y,tIn.x), nOut=vec2(-tOut.y,tOut.x);
  vec2 mit=nIn+nOut; float ml=length(mit);
  float hwAA=u_hw+0.5; vec2 mdir; float scl;
  if(ml<1e-3){mdir=nOut;scl=1.0;}else{mdir=mit/ml;scl=1.0/max(dot(mdir,nOut),0.25);}
  vec2 outpx=cur+a_corner.y*hwAA*scl*mdir;
  v_perp=a_corner.y*hwAA;
  vec2 uv=outpx/u_cpx;
  gl_Position=vec4(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0);
}`;
    const LFS=`#version 300 es
precision highp float; in float v_perp; out vec4 o; uniform vec4 u_color; uniform float u_hw;
void main(){ float a=clamp(u_hw+0.5-abs(v_perp),0.0,1.0); o=vec4(u_color.rgb,u_color.a*a); }`;
    lprog=gl.createProgram(); gl.attachShader(lprog,sh(gl.VERTEX_SHADER,LVS));
    gl.attachShader(lprog,sh(gl.FRAGMENT_SHADER,LFS)); gl.linkProgram(lprog);
    lineCorner=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,lineCorner);
    gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([0,-1, 1,-1, 0,1, 1,1]),gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER,null);
    LU={vp:gl.getUniformLocation(lprog,'u_vp'),cpx:gl.getUniformLocation(lprog,'u_cpx'),
        hw:gl.getUniformLocation(lprog,'u_hw'),color:gl.getUniformLocation(lprog,'u_color')};
    return true;
  },
  setColormap(name,data){
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,lutTex);
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,256,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array(data));
    gl.activeTexture(gl.TEXTURE0);
  },
  createTile(m,buf){
    const {w,h,ch,dt,mode,lo,hi,kind,bitmax,downsample}=m;
    // 'nearest' (label/ncolor) layers must NOT blend across edges — force a
    // NEAREST min-filter so a single-level (no-pyramid) mask stays crisp when
    // minified at zoom-out (a blended group index → a meaningless LUT colour).
    const NEAR=(downsample==='nearest');
    const tex=gl.createTexture(); gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,tex);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);
    if(mode==='intensity' && (dt==='float32'||dt==='float16')){
      // raw scalar → R32F / R16F texture; colormap + normalize happen in the
      // shader (it samples a float either way). R16F halves the wire bytes and is
      // core-filterable in WebGL2 (no float-linear extension needed).
      const f16=(dt==='float16');
      const filt=NEAR?gl.NEAREST:((f16||FLOAT_LINEAR)?gl.LINEAR:gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,filt);
      gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
      if(f16) gl.texImage2D(gl.TEXTURE_2D,0,gl.R16F,w,h,0,gl.RED,gl.HALF_FLOAT,new Uint16Array(buf));
      else    gl.texImage2D(gl.TEXTURE_2D,0,gl.R32F,w,h,0,gl.RED,gl.FLOAT,new Float32Array(buf));
      return {tex,w,h,mode:'intensity',lo,hi,kind,bitmax};
    }
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,NEAR?gl.NEAREST:gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);
    let bytes;
    if(dt==='uint8'&&ch===4){ bytes=new Uint8Array(buf); }
    else if(dt==='float32'){ const f=new Float32Array(buf); bytes=new Uint8Array(w*h*4);
      for(let i=0;i<w*h;i++){ for(let k=0;k<3;k++){ let v=f[i*ch+k];
        v=v<=0.0031308?12.92*v:1.055*Math.pow(Math.max(v,0),1/2.4)-0.055;
        bytes[i*4+k]=Math.max(0,Math.min(255,v*255)); } bytes[i*4+3]=255; } }
    else { bytes=new Uint8Array(buf); }
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA8,w,h,0,gl.RGBA,gl.UNSIGNED_BYTE,bytes);
    return {tex,w,h,mode:'rgb'};
  },
  setOutline(buf,name){
    // one VAO per named set: shared corner strip + this set's instance buffer
    const vao=gl.createVertexArray(); gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER,lineCorner);
    const aC=gl.getAttribLocation(lprog,'a_corner'); gl.enableVertexAttribArray(aC);
    gl.vertexAttribPointer(aC,2,gl.FLOAT,false,0,0);
    const ibuf=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER,ibuf);
    gl.bufferData(gl.ARRAY_BUFFER, buf, gl.STATIC_DRAW);
    [['a_prev',0],['a_p0',8],['a_p1',16],['a_next',24]].forEach(([nm,off])=>{
      const loc=gl.getAttribLocation(lprog,nm); if(loc<0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc,2,gl.FLOAT,false,32,off); gl.vertexAttribDivisor(loc,1); });
    gl.bindVertexArray(null);
    lineSets[name||'default']={vao, n:(buf.byteLength/32)|0};   // 8 floats/edge
  },
  hasOutline(name){ const s=lineSets[name||'default']; return !!(lprog && s && s.n>0); },
  frameBegin(){
    gl.disable(gl.SCISSOR_TEST); gl.clearColor(0,0,0,0); gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.SCISSOR_TEST); gl.bindVertexArray(null);
  },
  paint(e,vp,rect,lo,hi,texRect){
    const [x,y,wpx,hpx]=rect;
    gl.viewport(x,y,wpx,hpx); gl.scissor(x,y,wpx,hpx);
    // texRect = the FOV-norm sub-rect this texture covers; a full-FOV tile passes
    // nothing → [0,0,1,1] → st==tc → identical to the un-cropped path. A detail
    // crop carries its X-Crop rect on e.rect so the shader paints only that
    // sub-region (alpha 0 outside → the coarse base blends through).
    const tr=(e.rect||texRect||[0,0,1,1]);
    if(e.mode==='intensity'){
      gl.useProgram(cprog); gl.uniform4f(CU.vp, vp.x,vp.y,vp.w,vp.h);
      gl.uniform4f(CU.tr, tr[0],tr[1],tr[2],tr[3]);
      gl.uniform1f(CU.lo, lo); gl.uniform1f(CU.hi, hi);
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D,lutTex);
      gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D,e.tex);
      gl.drawArrays(gl.TRIANGLES,0,3);
    } else {
      gl.useProgram(prog); gl.uniform4f(uVp, vp.x,vp.y,vp.w,vp.h);
      gl.uniform4f(uTr, tr[0],tr[1],tr[2],tr[3]);
      gl.bindTexture(gl.TEXTURE_2D,e.tex); gl.drawArrays(gl.TRIANGLES,0,3);
    }
  },
  paintOutline(vp,rect,hwpx,color,name){
    const s=lineSets[name||'default']; if(!s||!s.n) return;
    const [x,y,wpx,hpx]=rect;
    gl.useProgram(lprog); gl.bindVertexArray(s.vao);
    gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.viewport(x,y,wpx,hpx); gl.scissor(x,y,wpx,hpx);
    gl.uniform4f(LU.vp, vp.x,vp.y,vp.w,vp.h); gl.uniform2f(LU.cpx, wpx,hpx);
    gl.uniform1f(LU.hw, hwpx); gl.uniform4f(LU.color, color[0],color[1],color[2],color[3]);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, s.n);
    gl.disable(gl.BLEND); gl.bindVertexArray(null);
  },
  frameEnd(){},
  };
}

function WebGPUBackend(){
  // Same seam as GL2Backend, WebGPU impl. Draws are accumulated in paint()/
  // paintOutline() and recorded as one render pass in frameEnd() (WebGPU has no
  // immediate-mode draw — viewport/scissor live inside a pass). All shader math
  // mirrors the GL2 shaders exactly so the rendered image is identical.
  let device=null, ctx=null, format=null, FLOAT_FILT=false;
  let rgbPipe=null, hdrPipe=null, intPipe=null, linePipe=null, composePipe=null, smp=null, lutSmp=null;
  let excTex=null, excBG=null, excUB=null, excN=0, excW=0, excH=0;   // RGB-cell live compose (per-excitation)
  let lutTex=null, lutBG=null;                    // group(1) for intensity
  let cornerVB=null, lineSets={};   // named outline sets (see GL2Backend)
  let draws=[];                                   // per-frame draw list
  let uPool=[], bgPool=[];                        // reused uniform buf + bindgroups
  const ALIGN=256;
  function padUpload(tex, w, h, bytes, bpp){
    const rowBytes=w*bpp, padded=Math.ceil(rowBytes/ALIGN)*ALIGN;
    let src=bytes;
    if(padded!==rowBytes){ const u8=new Uint8Array(padded*h);
      const sb=new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for(let y=0;y<h;y++) u8.set(sb.subarray(y*rowBytes,(y+1)*rowBytes), y*padded);
      src=u8; }
    device.queue.writeTexture({texture:tex}, src, {bytesPerRow:padded, rowsPerImage:h}, {width:w,height:h});
  }
  return {
  name:'webgpu',
  async init(canvas){
    if(!navigator.gpu) return false;
    const _t0=performance.now();
    const adapter=await navigator.gpu.requestAdapter();
    if(!adapter) return false;
    FLOAT_FILT=adapter.features.has('float32-filterable');
    device=await adapter.requestDevice(FLOAT_FILT?{requiredFeatures:['float32-filterable']}:{});
    const _tDev=performance.now();
    ctx=canvas.getContext('webgpu'); if(!ctx) return false;
    format=navigator.gpu.getPreferredCanvasFormat();
    // Prefer an HDR canvas: rgba16float + extended tone mapping lets RGB tiles
    // carry highlights >1.0 that the browser maps into the display's HDR
    // headroom (adaptive, no clipping). SDR values ≤1 render identically, so
    // this is a safe superset. Falls back to the 8-bit preferred format if the
    // runtime rejects the HDR config.
    // Display-P3 canvas: RGB tiles are scene-linear Display-P3, rendered in P3
    // end-to-end (no sRGB gamut squeeze). rgba16float + extended tone mapping
    // carry the peak-normalized data into the display's HDR headroom.
    this.HDR=false;
    try{ ctx.configure({device, format:'rgba16float', alphaMode:'premultiplied',
                        colorSpace:'display-p3', toneMapping:{mode:'extended'}});
         format='rgba16float'; this.HDR=true; }
    catch(e){ ctx.configure({device, format, alphaMode:'premultiplied'});
              console.warn('HDR canvas unavailable, SDR fallback:', e&&e.message||e); }
    smp=device.createSampler({magFilter:'nearest', minFilter:'linear'});
    lutSmp=device.createSampler({magFilter:'linear', minFilter:'linear'});
    const intSmp=device.createSampler({magFilter:'nearest',
      minFilter:FLOAT_FILT?'linear':'nearest'});
    this._intSmp=intSmp;
    // crisp label/ncolor sampler: NEAREST min too, so a single-level (no-pyramid)
    // mask doesn't blend group colours when minified at zoom-out.
    this._nearSmp=device.createSampler({magFilter:'nearest', minFilter:'nearest'});
    // r16float is ALWAYS filterable (unlike r32float, which needs the
    // 'float32-filterable' feature) → a linear sampler for float16 intensity tiles.
    this._lin16Smp=device.createSampler({magFilter:'nearest', minFilter:'linear'});
    // ── RGB pipeline: textureSample → out (group0 tex+samp, group1 uniform vp)
    // u.tr = FOV-norm sub-rect this texture covers (0,0,1,1 for a full-FOV tile,
    // the X-Crop rect for a detail crop). st maps the FOV coord into the texture;
    // outside [0,1] → transparent so the coarse base blends through.
    const RGB=`
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
  return select(vec4f(c.rgb,1.0), vec4f(0,0,0,0), oob); }`;
    // ── HDR-RGB pipeline: peak-normalized linear-P3 [0,1] (1.0 = XDR peak) →
    // scale by display headroom → extended-sRGB OETF. On an SDR display
    // (headroom 1) this collapses to ordinary sRGB, byte-matching the 8-bit
    // path; on HDR it pushes the peak into the headroom (adaptive, no clip).
    const HDR=`
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
  return select(vec4f(oetf(lin),1.0), vec4f(0,0,0,0), oob); }`;
    // ── intensity pipeline: R32F scalar → normalize(lo,hi) → LUT lookup
    const INT=`
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
  return select(`+(HDR_CMAP?'vec4f(oetf(col.rgb),1.0)':'vec4f(col.rgb,1.0)')+`, vec4f(0,0,0,0), oob); }`
    +(HDR_CMAP?'\nfn oetf(v:vec3f)->vec3f{let a=max(v,vec3f(0.0));return select(12.92*a,1.055*pow(a,vec3f(1.0/2.4))-0.055,a>vec3f(0.0031308));}':'');
    // ── seg-outline: MITER-JOIN ribbon (instanced, shader-AA; mirrors GL LVS) ──
    // Edge carries (prev,p0,p1,next); each end offsets along the joint bisector so
    // neighbours share the mitered vertex → no gap / no overlap (vs butt-cap +
    // end-extension double-blend). Miter clamped via the 0.25 floor on dot(mdir,nOut).
    const LINE=`
struct LU{ vp:vec4f, cpx:vec2f, hw:f32, _p:f32, color:vec4f };
@group(0)@binding(0) var<uniform> u:LU;
struct VI{ @location(0) corner:vec2f, @location(1) prev:vec2f, @location(2) p0:vec2f, @location(3) p1:vec2f, @location(4) nxt:vec2f };
struct VO{ @builtin(position) pos:vec4f, @location(0) vperp:f32 };
fn toPx(s:vec2f)->vec2f{ let uv=(s-u.vp.xy)/u.vp.zw; return uv*u.cpx; }
@vertex fn vs(i:VI)->VO{
  let atP0 = i.corner.x < 0.5;
  let cur = toPx(select(i.p1, i.p0, atP0));
  let aa  = toPx(select(i.p0, i.prev, atP0));
  let bb  = toPx(select(i.nxt, i.p1, atP0));
  let dIn=cur-aa; let dOut=bb-cur;
  let lIn=length(dIn); let lOut=length(dOut);
  var tIn=select(vec2f(0.0), dIn/lIn, lIn>1e-5);
  var tOut=select(vec2f(0.0), dOut/lOut, lOut>1e-5);
  if(lIn<=1e-5){tIn=tOut;} if(lOut<=1e-5){tOut=tIn;}
  let nIn=vec2f(-tIn.y,tIn.x); let nOut=vec2f(-tOut.y,tOut.x);
  let mit=nIn+nOut; let ml=length(mit);
  let hwAA=u.hw+0.5; var mdir:vec2f; var scl:f32;
  if(ml<1e-3){mdir=nOut;scl=1.0;}else{mdir=mit/ml;scl=1.0/max(dot(mdir,nOut),0.25);}
  let outpx=cur+i.corner.y*hwAA*scl*mdir;
  let uv=outpx/u.cpx;
  var o:VO; o.pos=vec4f(uv.x*2.0-1.0, 1.0-uv.y*2.0, 0.0, 1.0); o.vperp=i.corner.y*hwAA; return o; }
@fragment fn fs(in:VO)->@location(0) vec4f{
  let a=clamp(u.hw+0.5-abs(in.vperp),0.0,1.0); return vec4f(u.color.rgb,u.color.a*a); }`;
    // ── compose pipeline: per-excitation texture_2d_array → sum enabled
    // layers (texel·scale) → /total → /clipHigh white-point → ×headroom → OETF.
    // The classification-debugger live-RGB toggle, composited on the GPU. Same
    // math as the SVG path's compose shader; sampling matches the grid's HDR vp.
    const COMPOSE=`
struct U{ vp:vec4f, misc:vec4f, scales:array<vec4f,4>, ints:vec4u };
@group(0)@binding(0) var t:texture_2d_array<f32>;
@group(0)@binding(1) var s:sampler;
@group(1)@binding(0) var<uniform> u:U;
struct VO{ @builtin(position) pos:vec4f, @location(0) uv:vec2f };
@vertex fn vs(@builtin(vertex_index) i:u32)->VO{
  var p=array<vec2f,3>(vec2f(-1,-1),vec2f(3,-1),vec2f(-1,3));
  var o:VO; o.pos=vec4f(p[i],0,1); o.uv=vec2f(p[i].x*0.5+0.5, p[i].y*0.5+0.5); return o; }
fn oetf(v:vec3f)->vec3f{ let a=max(v,vec3f(0.0));
  return select(12.92*a, 1.055*pow(a,vec3f(1.0/2.4))-0.055, a>vec3f(0.0031308)); }
@fragment fn fs(in:VO)->@location(0) vec4f{
  let tc=u.vp.xy+vec2f(in.uv.x,1.0-in.uv.y)*u.vp.zw;
  let oob=tc.x<0.0||tc.x>1.0||tc.y<0.0||tc.y>1.0;
  let cc=clamp(tc,vec2f(0.0),vec2f(1.0));
  var lin=vec3f(0.0); let n=u.ints.x; let mask=u.ints.y;
  for(var k:u32=0u;k<16u;k=k+1u){ if(k>=n){break;} if((mask&(1u<<k))==0u){continue;}
    lin=lin+textureSampleLevel(t,s,cc,k,0.0).rgb*u.scales[k/4u][k%4u]; }
  lin=lin/f32(u.ints.z);              // /total → mean linear
  lin=lin/max(u.misc.y,1e-6);         // /clipHigh → brightest visible → 1
  lin=lin*u.misc.x;                   // ×headroom (EDR)
  return select(vec4f(oetf(lin),1.0), vec4f(0,0,0,0), oob); }`;
    const ct=[{format}];
    // Compile all pipelines CONCURRENTLY (createRenderPipelineAsync) instead of
    // serially — WGSL→Metal compilation is the dominant cold-start cost on real
    // hardware (hundreds of ms each), and the sync form blocks the JS thread one
    // after another. Async + Promise.all overlaps them and keeps first paint
    // snappy. The grid FRAME paints before any tile, so a tiny extra await here
    // is hidden behind the /info fetch.
    const rgbDesc={layout:'auto',
      vertex:{module:device.createShaderModule({code:RGB}),entryPoint:'vs'},
      fragment:{module:device.createShaderModule({code:RGB}),entryPoint:'fs',targets:ct},
      primitive:{topology:'triangle-list'}};
    const hdrDesc={layout:'auto',
      vertex:{module:device.createShaderModule({code:HDR}),entryPoint:'vs'},
      fragment:{module:device.createShaderModule({code:HDR}),entryPoint:'fs',targets:ct},
      primitive:{topology:'triangle-list'}};
    const intDesc={layout:'auto',
      vertex:{module:device.createShaderModule({code:INT}),entryPoint:'vs'},
      fragment:{module:device.createShaderModule({code:INT}),entryPoint:'fs',targets:ct},
      primitive:{topology:'triangle-list'}};
    const lineDesc={layout:'auto',
      vertex:{module:device.createShaderModule({code:LINE}),entryPoint:'vs',
        buffers:[{arrayStride:8,stepMode:'vertex',attributes:[{shaderLocation:0,offset:0,format:'float32x2'}]},
                 {arrayStride:32,stepMode:'instance',attributes:[   // (prev,p0,p1,next)
                   {shaderLocation:1,offset:0,format:'float32x2'},
                   {shaderLocation:2,offset:8,format:'float32x2'},
                   {shaderLocation:3,offset:16,format:'float32x2'},
                   {shaderLocation:4,offset:24,format:'float32x2'}]}]},
      fragment:{module:device.createShaderModule({code:LINE}),entryPoint:'fs',targets:[{format,
        blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},
               alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},
      primitive:{topology:'triangle-strip'}};
    const composeDesc={layout:'auto',
      vertex:{module:device.createShaderModule({code:COMPOSE}),entryPoint:'vs'},
      fragment:{module:device.createShaderModule({code:COMPOSE}),entryPoint:'fs',targets:ct},
      primitive:{topology:'triangle-list'}};
    [rgbPipe, hdrPipe, intPipe, linePipe, composePipe] = await Promise.all([
      device.createRenderPipelineAsync(rgbDesc),
      this.HDR ? device.createRenderPipelineAsync(hdrDesc) : Promise.resolve(null),
      device.createRenderPipelineAsync(intDesc),
      device.createRenderPipelineAsync(lineDesc),
      device.createRenderPipelineAsync(composeDesc)]);
    excUB=device.createBuffer({size:112,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
    cornerVB=device.createBuffer({size:32,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(cornerVB,0,new Float32Array([0,-1, 1,-1, 0,1, 1,1]));
    lutTex=device.createTexture({size:[256,1],
      format:(HDR_CMAP&&typeof Float16Array!=='undefined')?'rgba16float':'rgba8unorm',
      usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    console.log('WEBGPU init: device '+Math.round(_tDev-_t0)+'ms, pipelines '
      +Math.round(performance.now()-_tDev)+'ms, total '+Math.round(performance.now()-_t0)+'ms');
    return true;
  },
  setColormap(name,data){
    if(HDR_CMAP&&typeof Float16Array!=='undefined'){
      // HDR LUT: linear-P3 floats (1.0=SDR white, >1=HDR) → rgba16float texture;
      // the INT pipeline OETFs them on the extended canvas.
      const f=Float32Array.from(data); const half=new Float16Array(f.length);
      for(let i=0;i<f.length;i++) half[i]=f[i];
      padUpload(lutTex,256,1,new Uint8Array(half.buffer),8);
    } else {
      const bytes=new Uint8Array(data);
      padUpload(lutTex,256,1,bytes,4);
    }
    lutBG=device.createBindGroup({layout:intPipe.getBindGroupLayout(1),
      entries:[{binding:0,resource:lutTex.createView()},{binding:1,resource:lutSmp}]});
  },
  createTile(m,buf){
    const {w,h,ch,dt,mode,downsample}=m;
    const NEAR=(downsample==='nearest');   // label/ncolor → NEAREST sampler (no blend)
    if(mode==='intensity' && (dt==='float32'||dt==='float16')){
      // R16F halves the wire vs R32F and is always filterable in WebGPU, so the
      // intensity sampler (linear) binds fine even without 'float32-filterable'.
      const f16=(dt==='float16');
      const tex=device.createTexture({size:[w,h],format:f16?'r16float':'r32float',
        usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      padUpload(tex,w,h,new Uint8Array(buf),f16?2:4);
      const smp=NEAR?this._nearSmp:(f16?(this._lin16Smp||this._intSmp):this._intSmp);
      const bg=device.createBindGroup({layout:intPipe.getBindGroupLayout(0),
        entries:[{binding:0,resource:tex.createView()},{binding:1,resource:smp}]});
      return {mode:'intensity',w,h,bg,lo:m.lo,hi:m.hi,kind:m.kind,bitmax:m.bitmax};
    }
    // HDR path: float RGB from make_rgb is peak-normalized *linear* Display-P3
    // ([0,1], 1.0 = XDR peak). Store it LINEAR in rgba16float; the HDR pipeline
    // does OETF(d * headroom) at draw, adapting the peak to the live display
    // headroom. On an SDR display (headroom 1) the output matches the 8-bit
    // sRGB path. Only float RGB on the HDR canvas takes this route; uint8 RGBA
    // (already sRGB-encoded) stays on the byte path below.
    if(dt==='float32' && this.HDR && hdrPipe && typeof Float16Array!=='undefined'){
      const f=new Float32Array(buf); const half=new Float16Array(w*h*4);
      for(let i=0;i<w*h;i++){ for(let k=0;k<3;k++){ half[i*4+k]=f[i*ch+k]; } half[i*4+3]=1; }
      const tex=device.createTexture({size:[w,h],format:'rgba16float',
        usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      padUpload(tex,w,h,new Uint8Array(half.buffer),8);
      const bg=device.createBindGroup({layout:hdrPipe.getBindGroupLayout(0),
        entries:[{binding:0,resource:tex.createView()},{binding:1,resource:NEAR?this._nearSmp:smp}]});
      return {mode:'rgb',w,h,bg,hdr:true};
    }
    let bytes;
    if(dt==='uint8'&&ch===4){ bytes=new Uint8Array(buf); }
    else if(dt==='float32'){ const f=new Float32Array(buf); bytes=new Uint8Array(w*h*4);
      for(let i=0;i<w*h;i++){ for(let k=0;k<3;k++){ let v=f[i*ch+k];
        v=v<=0.0031308?12.92*v:1.055*Math.pow(Math.max(v,0),1/2.4)-0.055;
        bytes[i*4+k]=Math.max(0,Math.min(255,v*255)); } bytes[i*4+3]=255; } }
    else { bytes=new Uint8Array(buf); }
    const tex=device.createTexture({size:[w,h],format:'rgba8unorm',
      usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    padUpload(tex,w,h,bytes,4);
    const bg=device.createBindGroup({layout:rgbPipe.getBindGroupLayout(0),
      entries:[{binding:0,resource:tex.createView()},{binding:1,resource:NEAR?this._nearSmp:smp}]});
    return {mode:'rgb',w,h,bg};
  },
  setOutline(buf,name){
    const vb=device.createBuffer({size:buf.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
    device.queue.writeBuffer(vb,0,new Uint8Array(buf));
    lineSets[name||'default']={vb, n:(buf.byteLength/32)|0};   // 8 floats/edge
  },
  hasOutline(name){ const s=lineSets[name||'default']; return !!(s && s.n>0); },
  // Upload the per-excitation layers into ONE texture_2d_array for the live RGB
  // compose. ``layers`` = array of Uint8Array (rgba8, w*h*4 each, one per exc).
  setExcArray(layers,w,h){
    if(!composePipe||!layers||!layers.length) return;
    excN=layers.length; excW=w; excH=h;
    excTex=device.createTexture({size:[w,h,excN],format:'rgba8unorm',
      usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
    for(let k=0;k<excN;k++) device.queue.writeTexture({texture:excTex,origin:[0,0,k]},
      layers[k],{bytesPerRow:w*4,rowsPerImage:h},[w,h,1]);
    excBG=device.createBindGroup({layout:composePipe.getBindGroupLayout(0),
      entries:[{binding:0,resource:excTex.createView({dimension:'2d-array'})},{binding:1,resource:smp}]});
  },
  hasExc(){ return excN>0 && !!excBG; },
  frameBegin(){ draws=[]; },
  paint(e,vp,rect,lo,hi,texRect){ draws.push({kind:'tile',e,vp,rect,lo,hi,texRect}); },
  paintOutline(vp,rect,hwpx,color,name){ draws.push({kind:'line',vp,rect,hw:hwpx,color,name}); },
  // Live RGB composite for the RGB cell: ``scales`` (per-exc linear), ``mask``
  // (bit per visible exc), ``total`` (channel count), ``clipHigh`` (white point).
  paintExc(vp,rect,scales,mask,total,clipHigh){
    draws.push({kind:'exc',vp,rect,scales,mask,total,clipHigh}); },
  frameEnd(){
    // grow the uniform-buffer + bindgroup pool to cover this frame's draws
    while(uPool.length<draws.length){
      const ub=device.createBuffer({size:64,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      uPool.push(ub); bgPool.push({rgb:null,hdr:null,int:null,line:null,exc:null}); }
    const enc=device.createCommandEncoder();
    const pass=enc.beginRenderPass({colorAttachments:[{view:ctx.getCurrentTexture().createView(),
      clearValue:{r:0,g:0,b:0,a:0}, loadOp:'clear', storeOp:'store'}]});
    const CH=canvas.height;
    draws.forEach((d,i)=>{
      const ub=uPool[i], slot=bgPool[i];
      const [x,y,wpx,hpx]=d.rect; const yTop=CH-y-hpx;   // GL y-up → WebGPU y-down
      pass.setViewport(x,yTop,wpx,hpx,0,1);
      pass.setScissorRect(Math.max(0,x|0),Math.max(0,yTop|0),Math.max(1,wpx|0),Math.max(1,hpx|0));
      if(d.kind==='exc'){
        // U: vp(16) + misc(16: headroom,clipHigh,_,_) + scales array<vec4f,4>(64)
        // + ints(16: n,mask,total,_) = 112 bytes. Dedicated excUB (pool is 64B).
        const fb=new Float32Array(28);
        fb[0]=d.vp.x; fb[1]=d.vp.y; fb[2]=d.vp.w; fb[3]=d.vp.h;
        fb[4]=HEADROOM; fb[5]=Math.max(d.clipHigh,1e-6);
        for(let k=0;k<d.scales.length&&k<16;k++) fb[8+k]=d.scales[k];   // offset 32B = float idx 8
        device.queue.writeBuffer(excUB,0,fb);
        device.queue.writeBuffer(excUB,96,new Uint32Array([excN,d.mask>>>0,d.total>>>0,0]));
        if(!slot.exc) slot.exc=device.createBindGroup({layout:composePipe.getBindGroupLayout(1),
          entries:[{binding:0,resource:{buffer:excUB}}]});
        pass.setPipeline(composePipe); pass.setBindGroup(0,excBG); pass.setBindGroup(1,slot.exc); pass.draw(3);
      } else if(d.kind==='line'){
        const ls=lineSets[d.name||'default']; if(!ls||!ls.n) return;
        device.queue.writeBuffer(ub,0,new Float32Array([d.vp.x,d.vp.y,d.vp.w,d.vp.h,
          wpx,hpx, d.hw,0, d.color[0],d.color[1],d.color[2],d.color[3]]));
        if(!slot.line) slot.line=device.createBindGroup({layout:linePipe.getBindGroupLayout(0),
          entries:[{binding:0,resource:{buffer:ub}}]});
        pass.setPipeline(linePipe); pass.setBindGroup(0,slot.line);
        pass.setVertexBuffer(0,cornerVB); pass.setVertexBuffer(1,ls.vb);
        pass.draw(4,ls.n);
      } else if(d.e.mode==='intensity'){
        const tr=(d.e.rect||d.texRect||[0,0,1,1]);
        device.queue.writeBuffer(ub,0,new Float32Array([d.vp.x,d.vp.y,d.vp.w,d.vp.h, d.lo,d.hi,0,0, tr[0],tr[1],tr[2],tr[3]]));
        if(!slot.int) slot.int=device.createBindGroup({layout:intPipe.getBindGroupLayout(2),
          entries:[{binding:0,resource:{buffer:ub}}]});
        pass.setPipeline(intPipe); pass.setBindGroup(0,d.e.bg);
        pass.setBindGroup(1,lutBG); pass.setBindGroup(2,slot.int); pass.draw(3);
      } else if(d.e.hdr){
        // peak-normalized linear RGB → OETF(d*headroom) in the HDR pipeline
        const tr=(d.e.rect||d.texRect||[0,0,1,1]);
        device.queue.writeBuffer(ub,0,new Float32Array([d.vp.x,d.vp.y,d.vp.w,d.vp.h, HEADROOM,0,0,0, tr[0],tr[1],tr[2],tr[3]]));
        if(!slot.hdr) slot.hdr=device.createBindGroup({layout:hdrPipe.getBindGroupLayout(1),
          entries:[{binding:0,resource:{buffer:ub}}]});
        pass.setPipeline(hdrPipe); pass.setBindGroup(0,d.e.bg); pass.setBindGroup(1,slot.hdr); pass.draw(3);
      } else {
        const tr=(d.e.rect||d.texRect||[0,0,1,1]);
        device.queue.writeBuffer(ub,0,new Float32Array([d.vp.x,d.vp.y,d.vp.w,d.vp.h, tr[0],tr[1],tr[2],tr[3]]));
        if(!slot.rgb) slot.rgb=device.createBindGroup({layout:rgbPipe.getBindGroupLayout(1),
          entries:[{binding:0,resource:{buffer:ub}}]});
        pass.setPipeline(rgbPipe); pass.setBindGroup(0,d.e.bg); pass.setBindGroup(1,slot.rgb); pass.draw(3);
      }
    });
    pass.end(); device.queue.submit([enc.finish()]);
  },
  };
}

async function init(){ try{
  // Kick off /info AND the GPU init CONCURRENTLY so the device + pipeline cold
  // start (~300ms on real hardware) overlaps the network + the server-side
  // projection instead of blocking before them. The labeled grid FRAME (pure
  // DOM) paints the instant the FOV dims arrive — before any pipeline finishes
  // compiling — so first paint no longer waits on the GPU at all.
  const infoP = fetch(VBASE+'info/'+SID).then(r=>r.json());
  let gpuP = Promise.resolve(false);
  try{ if(navigator.gpu){ const b=WebGPUBackend();
    gpuP = b.init(canvas).then(ok=>{ if(ok) backend=b; return ok; })
             .catch(e=>{ console.warn('WebGPU init failed, falling back to WebGL2:', e); return false; }); } }
  catch(e){}

  // FRAME first: layout + labels the moment dims are known (no GPU needed).
  info = await infoP;
  await refreshLayoutGeom();
  layout();
  console.log('FIRSTPAINT '+Math.round(performance.now()));

  // Settle the backend (GL2 fallback if WebGPU is absent or failed compiling).
  const useGpu = await gpuP;
  if(!useGpu){ backend=GL2Backend(); if(!backend.init(canvas)){ hud.textContent='no WebGL2 context'; return; } }
  console.log('BACKEND '+backend.name+(backend.HDR?' (HDR)':''));
  // Probe whatever HDR signals this context exposes — useful because notebook
  // iframes and older Chrome builds often hide the numeric headroom.
  try{ console.log('HDR probe', JSON.stringify({
    screenHeadroom:('highDynamicRangeHeadroom' in screen)?screen.highDynamicRangeHeadroom:'unsupported',
    dynamicRangeHigh:matchMedia('(dynamic-range: high)').matches,
    p3:matchMedia('(color-gamut: p3)').matches})); }catch(e){}
  // Effective HDR multiplier = auto display headroom UNLESS the user sets a
  // manual gain (HRMANUAL>0). Peak-normalized RGB is 1.0=SDR white, so a gain
  // >1 is what pushes highlights into the HDR range.
  function readAutoHR(){
    // Mirror the rgb_live controller (ocdkit figure.py): probe the four screen
    // keys browsers have used for numeric EDR headroom; if none expose a number
    // but the display is HDR-capable, default to a generous fill (4.0) so HDR is
    // bright out of the box when live RGB is on. SDR → 1.0.
    const sc=window.screen||{};
    for(const k of ['highDynamicRangeHeadroom','dynamicRangeHeadroom','hdrHeadroom','currentEDRHeadroom']){
      try{ const v=sc[k]; if(typeof v==='number'&&v>0){ HRSRC=k; return v; } }catch(e){}
    }
    if(matchMedia('(dynamic-range: high)').matches){ HRSRC='default(4)'; return 4.0; }
    HRSRC='sdr'; return 1;
  }
  function pollHeadroom(){ AUTOHR=readAutoHR();
    let eff = HRMANUAL>0 ? HRMANUAL : AUTOHR;
    if(_sdrMode) eff=1;                       // HDR toggle off → SDR (no headroom boost)
    if(eff!==HEADROOM){ HEADROOM=eff; if(info) render(); } }
  window.__poll=pollHeadroom;
  pollHeadroom(); setInterval(pollHeadroom, 1000);
  // headroom is often event-driven, not poll-readable — listen for changes too
  try{ screen.addEventListener && screen.addEventListener('change', pollHeadroom); }catch(e){}
  try{ matchMedia('(dynamic-range: high)').addEventListener('change', pollHeadroom); }catch(e){}
  setCmap(CMAP); buildControls(); buildTitle();
  fetchOutline();
  fetchOutlineGroups();    // no-op unless info.has_outline_groups
  fetchHighlightGroups();  // no-op unless info.has_highlight_groups
  loadSpectra();          // live spectra-density raster in the panel (static)
  loadExc();              // rgb_live per-excitation compose (retries until projected)
  loadCellInfo();         // cell bboxes + id map (snap-to-cell / hover-highlight)
  // GPU render now that pipelines are live; tiles fill in as the background
  // projection finishes (ncolor first, then mean/max/rgb).
  render();
  poll();
  }catch(err){ hud.textContent='INIT ERROR: '+(err&&err.message||err);
    console.error('grid init failed:', err); }
}
let _paintedAll=false;
async function poll(){
  // exc* are fetched as a texture array by loadExc(), not as individual cell
  // tiles — skip them here so poll doesn't build textures the grid never paints.
  const order=Object.keys(info.layers).filter(l=>!/^exc\d+$/.test(l)); let allReady=true;
  await Promise.all(order.map(async l=>{ if(!texCache.has(l+'/0')){
    const e=await getTex(l,0); if(!e) allReady=false; } }));
  render();
  if(allReady){ if(!_paintedAll){ _paintedAll=true; warmPyramid(); } draw(); } else { setTimeout(poll, 100); }
}
// ── Figure GEOMETRY: single source of truth = the Python compute_layout served
// at /layout?w= (mirrors this file's layout() math, parity-tested ≤1px). Fetched
// into _G for the current container width; layout() consumes it and falls back to
// its inline math only when the fetch is unavailable (older server / offline), so
// the viewer can never go blank. [the inline fallback is slated for deletion once
// real-browser validation confirms the served geometry across notebook/proxy.]
let _G=null;
async function refreshLayoutGeom(){
  try{
    const r = await fetch(VBASE+'layout/'+SID+'?w='+Math.round(window.innerWidth)+'&label_pos='+encodeURIComponent(LPOS));
    _G = r.ok ? await r.json() : null;
  }catch(e){ _G=null; }
}
function layout(){
  // ``exc*`` layers are AUXILIARY data for the RGB cell's live composite
  // (toggleable per-excitation), NOT their own cells — skip them in layout.
  const W0=window.innerWidth;
  // Prefer the 2D LAYOUT grid (info.grid) so the arrangement matches the host's
  // intended layout (label cells, reference columns, reduction layers in the
  // gaps, blank cells for absent entries). null = blank cell. Fall back to a
  // flat wrap when no grid is supplied.
  let cols, rows, placed;
  if(info.grid && info.grid.length){
    rows=info.grid.length;
    cols=info.grid.reduce((m,r)=>Math.max(m,r.length),0);
    placed=[];
    info.grid.forEach((grow,r)=>grow.forEach((lbl,col)=>{
      if(typeof lbl==='string') placed.push({label:lbl,r,col});
      else if(lbl && lbl.empty) placed.push({empty:lbl.empty,r,col}); }));   // absent readout
  } else {
    const order=Object.keys(info.layers).filter(l=>!/^exc\d+$/.test(l));
    cols=order.length<=6?order.length:7; rows=Math.ceil(order.length/cols);
    placed=order.map((lbl,i)=>({label:lbl,r:Math.floor(i/cols),col:i%cols}));
  }
  // ── SCALE-TO-FIT (matches the SVG figure, which scales the whole grid+spectra
  // unit to the container at a FIXED aspect). The panel keeps a proper aspect
  // (PANEL_ASPECT × plot width) instead of "whatever height is left", which
  // collapsed it to a too-short strip when the iframe was short / the grid tall.
  // Compute the natural full-width layout, then a uniform scale ``k`` so the
  // grid + panel together fit the iframe height; center the grid horizontally.
  // panel axis spec — generic ``panel_axes`` (kind:'xy' = continuous numeric
  // x/y) with back-compat for the spectra ``spectra_axes`` (bands/refs/top-axis).
  const AX=info.panel_axes||info.spectra_axes, isXY=!!(AX&&AX.kind==='xy');
  const hasAx=!!(AX&&(isXY||(AX.bands&&AX.bands.length)));
  // ── WIDTH-DRIVEN scale (aspect set by the COMPOSED ELEMENTS, not a fixed box).
  // The whole figure (tiles, gaps, axes, FONTS) is designed at a reference width
  // ``Wref`` and uniformly scaled by ``k=W0/Wref`` to the actual container width.
  // Consequences: (1) the layout ASPECT is constant — set by cols/rows + whether
  // the panel is present — so the embedding iframe (aspect-ratio in scene_layers)
  // fits with NO letterbox / zoom-out; (2) the figure grows/shrinks with the
  // output-box width; (3) text scales WITH the figure (wider box → bigger text)
  // instead of shrinking when the panel is added (the OLD H-fit shrank k as
  // fullH0 grew, which is why the fonts got smaller once the spectra plot landed).
  const TA=(hasAx&&!isXY)?AX.top_axis:null;     // readout top-axis layout (data, not geometry)
  // GEOMETRY — prefer the server-computed layout (_G, Python compute_layout =
  // single source of truth); fall back to the inline width-driven math otherwise.
  let k, cw, gap, totH, vgap=0, XLAB_H=0, TOPAX_H, contentLeft, contentW, topAxTop, panelTop, plotW, plotH, fullH;
  let SERVERCELLS=null;
  if(_G && _G.cols===cols && _G.rows===rows && (!!_G.has_panel)===hasAx){
    k=_G.k; cw=_G.cw; gap=_G.gap; totH=_G.canvas_h; TOPAX_H=_G.top_ax_h;
    contentLeft=_G.content_left; contentW=_G.content_w; topAxTop=_G.top_ax_top;
    panelTop=_G.panel_top; plotW=_G.panel_w; plotH=_G.panel_h; fullH=_G.full_h;
    SERVERCELLS=_G.cells;
  } else {
    // FALLBACK: inline reference-width math (identical to compute_layout). The
    // figure is designed at Wref and uniformly scaled by k=W0/Wref.
    const Wref=1000, PAD=0.05, YAXW0=50, RM0=8, XLAB_H0=46, TOPAX0=22;
    k=W0/Wref;
    const padL0=hasAx?YAXW0:0, padR0=hasAx?RM0:0;
    const contentW0=Math.max(20, Wref-padL0-padR0);
    const cw0=contentW0/(cols+(cols+1)*PAD), gap0=PAD*cw0, totH0=rows*cw0+gap0*(rows+1);
    const gridSpanH0=rows*cw0+Math.max(0,rows-1)*gap0;
    const vgap0=(TA||isXY)?0:Math.max(6,PAD*cw0), plotW0=Math.max(20, contentW0-2*gap0);
    const plotH0=hasAx?gridSpanH0:0;
    const TOPAX_H0=(TA&&TA.top_axis_h!=null)?TA.top_axis_h:((hasAx&&!isXY)?TOPAX0:0);
    const fullH0=hasAx?(totH0+vgap0+TOPAX_H0+plotH0+XLAB_H0):totH0;
    cw=cw0*k; gap=gap0*k; totH=totH0*k; vgap=vgap0*k; XLAB_H=XLAB_H0*k; TOPAX_H=TOPAX_H0*k;
    contentLeft=(hasAx?YAXW0:0)*k; contentW=contentW0*k;
    topAxTop=hasAx?totH+vgap:0; panelTop=hasAx?totH+vgap+TOPAX_H:0; plotW=hasAx?plotW0*k:0;
    plotH=hasAx?plotH0*k:0; fullH=fullH0*k;
  }
  const fs=px=>Math.max(7, px*k);
  // Canvas spans the FULL container width (dpr maps cleanly); the grid sits in
  // [contentLeft, contentLeft+contentW].
  canvas.style.width=W0+'px'; canvas.style.height=totH+'px';
  canvas.width=Math.round(W0*dpr); canvas.height=Math.round(totH*dpr);
  const _wrap=canvas.parentNode; if(_wrap&&_wrap.style) _wrap.style.height=fullH+'px';
  _kNow=k; _sizeBars();   // ctl strip above + hud strip below scale with the figure
  // SVG OVERLAY — vector annotation layer (cell frames + labels AND the spectra
  // axes) on TOP of the canvas, spanning grid + panel. Built ONCE here (on
  // layout/resize), NOT per zoom frame → zero per-frame cost.
  cells=[]; labs.innerHTML='';
  ovl.setAttribute('width', W0); ovl.setAttribute('height', fullH);
  ovl.setAttribute('viewBox', '0 0 '+W0+' '+fullH);
  ovl.style.width=W0+'px'; ovl.style.height=fullH+'px';
  while(ovl.firstChild) ovl.removeChild(ovl.firstChild);
  const _svg=(tag,attrs)=>{ const e=document.createElementNS(SVGNS,tag);
    for(const kk in attrs) e.setAttribute(kk, attrs[kk]); return e; };
  const _txt=(x,y,str,a)=>{ const t=_svg('text',Object.assign({x:x,y:y},a||{})); t.textContent=str; ovl.appendChild(t); return t; };
  // tile-box / plot-box stroke (shared so the spectra box matches the tiles).
  const bw=Math.max(1, cw*0.012);
  // cell frames + labels (grid in the same x-extent as the plot)
  // Unified cell list: server geometry (_G.cells, with label positions) when
  // present, else the inline placed→rect math. Same drawing path for both.
  const _cells = SERVERCELLS ? SERVERCELLS.map(c=>({
      x:c.x, y:c.y, w:c.w, h:c.h, lx:c.label_x, ly:c.label_y,
      anchor:c.anchor, baseline:c.baseline, label:c.label, empty:c.empty }))
    : placed.map(p=>{
      const x=contentLeft+gap+p.col*(cw+gap), y=gap+p.r*(cw+gap);
      const LP=_labelPos(x,y,cw,cw,LPOS,4*k);   // unified ocdkit-style title position
      return {x, y, w:cw, h:cw, lx:LP[0], ly:LP[1], anchor:LP[2], baseline:LP[3],
              label:p.label, empty:p.empty}; });
  _cells.forEach(c=>{
    const bl=c.baseline==='hanging'?'hanging':'auto';
    if(c.empty){
      // ABSENT readout: translucent frame + faded label, no tile/data.
      ovl.appendChild(_svg('rect',{x:c.x+bw/2,y:c.y+bw/2,width:Math.max(0,c.w-bw),
        height:Math.max(0,c.h-bw),fill:'none',stroke:'#888','stroke-width':bw,rx:RX,opacity:0.3}));
      _txt(c.lx, c.ly, c.empty, {fill:'#fff','font-family':'system-ui,sans-serif',
        'font-size':fs(12),opacity:0.3,'text-anchor':c.anchor,'dominant-baseline':bl});
      return;
    }
    cells.push({label:c.label,x:c.x,y:c.y,w:c.w,h:c.h});
    ovl.appendChild(_svg('rect',{x:c.x+bw/2,y:c.y+bw/2,width:Math.max(0,c.w-bw),
      height:Math.max(0,c.h-bw),fill:'none',stroke:'#888','stroke-width':bw,rx:RX}));
    const _lt=_txt(c.lx, c.ly, c.label, {fill:'#fff','font-family':'system-ui,sans-serif',
      'font-size':fs(12),'text-anchor':c.anchor,'dominant-baseline':bl});
    if(c.label==='Masks'){   // clickable → toggle the ncolor pixel-segmentation underlay
      _lt.style.cursor='pointer'; _lt.style.pointerEvents='auto';
      _lt.addEventListener('click', toggleMasksNcolor);
    }
  });
  // Clip the canvas TILE DATA to the rounded cell corners (so the image corners
  // match the rounded frames). CSS mask = white rounded rects at the cell rects;
  // static (rebuilt on layout/resize only, never per zoom frame). RX=0 → no mask.
  if(RX>0 && cells.length){
    // Clip the DATA to the FRAME PATH (the stroke's centerline: inset bw/2,
    // radius RX) — IDENTICAL geometry to the frame rect drawn above. The opaque
    // stroke (centered on this path, extending bw/2 each side) then covers the
    // image's edge, so the visible rounding is the frame's and nothing pokes past
    // it. The old mask used the OUTER cell rect (inset 0, radius RX), whose corner
    // bulged beyond the frame's rounder outer corner (RX+bw/2) → image poked out.
    const mr=cells.map(c=>'<rect x="'+(c.x+bw/2).toFixed(2)+'" y="'+(c.y+bw/2).toFixed(2)+'" width="'+Math.max(0,c.w-bw).toFixed(2)+'" height="'+Math.max(0,c.h-bw).toFixed(2)+'" rx="'+RX+'" fill="#fff"/>').join('');
    const msvg='<svg xmlns="http://www.w3.org/2000/svg" width="'+W0+'" height="'+totH+'">'+mr+'</svg>';
    const murl='url("data:image/svg+xml,'+encodeURIComponent(msvg)+'")';
    canvas.style.webkitMaskImage=murl; canvas.style.maskImage=murl;
    canvas.style.webkitMaskSize=canvas.style.maskSize=W0+'px '+totH+'px';
    canvas.style.webkitMaskRepeat=canvas.style.maskRepeat='no-repeat';
  } else {
    canvas.style.webkitMaskImage=canvas.style.maskImage='none';
  }
  // spectra axes (static; matches the SVG key-slice panel). Plot left = grid left
  // (contentLeft) → x-axis aligns with the grid columns; full box (4 spines).
  if(hasAx){
    const PX=contentLeft+gap;   // = first tile's left edge (plot aligns to tiles)
    const ylo=(AX.ylo!=null?AX.ylo:-0.05), yhi=(AX.yhi!=null?AX.yhi:1.05);
    const yPix=v=>panelTop+(yhi-v)/((yhi-ylo)||1)*plotH;
    const SPN='#888', TXT='#bbb', FSF='system-ui,sans-serif', MONO='ui-monospace,Menlo,monospace';
    const tkw=TICKW>0?TICKW*k:bw;     // tick stroke width: configurable, default = axes width
    if(isXY){
      // ── generic continuous x/y axes (scatter / xy panel) — no bands, refs,
      // top-axis or norm-toggle: a numeric x-axis (ticks+label, bottom) and a
      // plain numeric y-axis (ticks+label, left), inside the shared plot box.
      const _fmt=v=>{ const a=Math.abs(v);
        return (a!==0&&(a<1e-2||a>=1e4))?v.toExponential(1)
             : (Number.isInteger(v)?String(v):parseFloat(v.toFixed(3)).toString()); };
      // plot box (4 spines) — same stroke/inset as the tile frames + spectra box.
      ovl.appendChild(_svg('rect',{x:PX+bw/2,y:panelTop+bw/2,width:Math.max(0,plotW-bw),
        height:Math.max(0,plotH-bw),fill:'none',stroke:SPN,'stroke-width':bw,rx:RX}));
      // y ticks + plain (rotated) y-label, placed just left of the widest tick label.
      const _yt=(AX.yticks||[ylo,(ylo+yhi)/2,yhi]);
      let _ytw=0; _yt.forEach(tv=>{ _ytw=Math.max(_ytw,_measW(_fmt(tv),fs(10))); });
      _yt.forEach(tv=>{ const yy=yPix(tv);
        ovl.appendChild(_svg('line',{x1:PX-3*k,y1:yy,x2:PX,y2:yy,stroke:SPN,'stroke-width':tkw}));
        _txt(PX-6*k, yy+fs(10)*0.35, _fmt(tv), {fill:TXT,'font-family':FSF,'font-size':fs(10),'text-anchor':'end'}); });
      if(AX.ylabel){ const lx=Math.max(fs(11)*0.6, PX-6*k-_ytw-4*k-fs(11)*0.5), ly=panelTop+plotH/2;
        const yl=_txt(lx, ly, AX.ylabel, {fill:TXT,'font-family':FSF,'font-size':fs(11),'text-anchor':'middle'});
        yl.setAttribute('transform','rotate(-90 '+lx+' '+ly+')');
        if(AX.scales&&AX.scales.y&&AX.scales.y.log){   // label = linear↔log toggle
          yl.setAttribute('class','yscale'); yl.style.cursor='pointer';
          yl.style.userSelect='none'; yl.style.pointerEvents='auto';  // #ovl is none
          yl.addEventListener('click',()=>toggleAxisScale('y')); } }
      // x ticks (bottom) + numeric labels, then a centered x-label below them.
      const xlo=(AX.xlo!=null?AX.xlo:0), xhi=(AX.xhi!=null?AX.xhi:1);
      const xPix=v=>PX+(v-xlo)/((xhi-xlo)||1)*plotW;
      const _xt=(AX.xticks||[xlo,(xlo+xhi)/2,xhi]);
      _xt.forEach(tv=>{ const xx=xPix(tv);
        ovl.appendChild(_svg('line',{x1:xx,y1:panelTop+plotH,x2:xx,y2:panelTop+plotH+3*k,stroke:SPN,'stroke-width':tkw}));
        _txt(xx, panelTop+plotH+fs(10)+6*k, _fmt(tv), {fill:TXT,'font-family':FSF,'font-size':fs(10),'text-anchor':'middle'}); });
      if(AX.xlabel){ const xl=_txt(PX+plotW/2, panelTop+plotH+fs(10)+fs(11)+13*k, AX.xlabel,
        {fill:TXT,'font-family':FSF,'font-size':fs(11),'text-anchor':'middle'});
        if(AX.scales&&AX.scales.x&&AX.scales.x.log){   // label = linear↔log toggle
          xl.setAttribute('class','xscale'); xl.style.cursor='pointer';
          xl.style.userSelect='none'; xl.style.pointerEvents='auto';  // #ovl is none
          xl.addEventListener('click',()=>toggleAxisScale('x')); } }
      // optional decision-boundary overlays: shaded pass-region + dashed cutoffs
      // (data coords). shade={x0,x1,y0,y1}; vlines/hlines = arrays of x/y cutoffs.
      const DSH=(4*k).toFixed(1)+','+(3*k).toFixed(1);
      if(AX.shade){ const s=AX.shade;
        const sx0=xPix(s.x0!=null?s.x0:xlo), sx1=xPix(s.x1!=null?s.x1:xhi);
        const sy0=yPix(s.y0!=null?s.y0:ylo), sy1=yPix(s.y1!=null?s.y1:yhi);
        ovl.appendChild(_svg('rect',{x:Math.min(sx0,sx1),y:Math.min(sy0,sy1),
          width:Math.abs(sx1-sx0),height:Math.abs(sy1-sy0),fill:'#888',opacity:0.12})); }
      (AX.vlines||[]).forEach(xv=>{ const xx=xPix(xv);
        ovl.appendChild(_svg('line',{x1:xx,y1:panelTop,x2:xx,y2:panelTop+plotH,
          stroke:SPN,'stroke-width':tkw,'stroke-dasharray':DSH,opacity:0.85})); });
      (AX.hlines||[]).forEach(yv=>{ const yy=yPix(yv);
        ovl.appendChild(_svg('line',{x1:PX,y1:yy,x2:PX+plotW,y2:yy,
          stroke:SPN,'stroke-width':tkw,'stroke-dasharray':DSH,opacity:0.85})); });
    } else {
    AX.bands.forEach((b,i)=>{                     // x: excitation/channel bands
      const bx0=PX+b.x0*plotW, bx1=PX+b.x1*plotW, bmid=(bx0+bx1)/2;
      ovl.appendChild(_svg('rect',{x:bx0,y:panelTop,width:Math.max(0,bx1-bx0),height:plotH,
        fill:'#888',opacity:(i%2===0?0.16:0.08)}));
      ovl.appendChild(_svg('line',{x1:bmid,y1:panelTop+plotH,x2:bmid,y2:panelTop+plotH+3*k,stroke:SPN,'stroke-width':tkw}));
      if(b.exc) _txt(bmid, panelTop+plotH+fs(11)+6*k, b.exc, {fill:TXT,'font-family':FSF,'font-size':fs(11),'font-weight':'bold','text-anchor':'middle'});
      if(b.ch)  _txt(bmid, panelTop+plotH+fs(11)+fs(10)+12*k, b.ch, {fill:'#999','font-family':MONO,'font-size':fs(10),'text-anchor':'middle'});
    });
    // full plot box (4 spines) — SAME stroke width + inset side as the tile
    // frames (rect inset by bw/2 so the stroke sits INSIDE, outer edge exactly at
    // PX = first tile's outer edge; right edge at PX+plotW = last tile's outer edge).
    ovl.appendChild(_svg('rect',{x:PX+bw/2,y:panelTop+bw/2,width:Math.max(0,plotW-bw),
      height:Math.max(0,plotH-bw),fill:'none',stroke:SPN,'stroke-width':bw,rx:RX}));
    const _ticks=(AX.yticks||[0,0.2,0.4,0.6,0.8,1]);
    let _tickW=0; _ticks.forEach(tv=>{ _tickW=Math.max(_tickW, _measW(tv.toFixed(1), fs(10))); });
    _ticks.forEach(tv=>{ const yy=yPix(tv);
      ovl.appendChild(_svg('line',{x1:PX-3*k,y1:yy,x2:PX,y2:yy,stroke:SPN,'stroke-width':tkw}));
      _txt(PX-6*k, yy+fs(10)*0.35, tv.toFixed(1), {fill:TXT,'font-family':FSF,'font-size':fs(10),'text-anchor':'end'}); });
    // y-axis label sits just LEFT of the widest tick label (dynamic spacing), not
    // floating at the figure edge. tick labels span [PX-6k-_tickW, PX-6k]; place
    // the rotated label a small gap left of that (fs(11)*0.5 = its glyph height/2).
    const lx=Math.max(fs(11)*0.6, PX - 6*k - _tickW - 4*k - fs(11)*0.5), ly=panelTop+plotH/2;
    // y-axis label doubles as the NORMALIZATION toggle: click cycles
    // self-norm → global → bit-depth (swaps the density yLines + redraws).
    const yl=_txt(lx, ly, '', {fill:TXT,'font-family':FSF,'font-size':fs(11),
      'text-anchor':'middle','class':'ynorm'});
    yl.setAttribute('transform','rotate(-90 '+lx+' '+ly+')');
    yl.style.cursor='pointer'; yl.style.userSelect='none'; yl.style.pointerEvents='auto';  // #ovl is none
    // Two click zones: the underlined scope chip (tspan.ynorm-scope) toggles
    // all⇄visible; clicking anywhere else on the title cycles the mode.
    yl.addEventListener('click', function(e){
      const tg=e.target;
      if(tg && tg.getAttribute && tg.getAttribute('class')==='ynorm-scope') toggleScope();
      else cycleMode();
    });
    _renderNormLabel();
    // ── reference spectra (TOGGLEABLE) + readout TOP-AXIS labels/ticks ──
    // Refs start HIDDEN; each readout's top-axis label (at its reference peak x)
    // toggles it on click (pinned), and hovering a cell in the density below
    // temporarily shows that cell's classification readouts (temp). A ref shows
    // if pinned||temp; its label colors when on, grays when off. The pinned/temp
    // STATE persists across layout rebuilds (declared at module scope).
    _refPaths={}; _refLabels={}; _refColors={};
    // dashed reference path(s) per readout — hidden by default, tagged + grouped.
    if(AX.refs && AX.refs.length){
      AX.refs.forEach(rf=>{
        const ro=rf.readout; const c=rf.color||[0.7,0.7,0.7];
        const col='rgb('+Math.round(c[0]*255)+','+Math.round(c[1]*255)+','+Math.round(c[2]*255)+')';
        _refColors[ro]=col;
        const paths=[];
        (rf.segs||[]).forEach(seg=>{
          if(!seg||seg.length<2) return;
          const d='M '+seg.map(p=>(PX+p[0]*plotW).toFixed(1)+','+yPix(p[1]).toFixed(1)).join(' L ');
          const p=_svg('path',{d:d,fill:'none',stroke:col,'stroke-width':Math.max(1,1.5*k),
            'stroke-dasharray':(5*k).toFixed(1)+','+(4*k).toFixed(1),'stroke-linejoin':'round',
            'stroke-linecap':'round','data-ref-ro':ro});
          p.style.display='none'; ovl.appendChild(p); paths.push(p);
        });
        _refPaths[ro]=paths;
      });
    }
    // readout TOP-AXIS: band-grouped, no-overlap-stacked labels + ticks from the
    // ported layout (TA.labels in plot-natural px: x∈[0,plotW0], y strip-relative,
    // baseline). Each label is a clickable toggle (gray default). Scale all by k.
    if(TA && TA.labels){
      TA.labels.forEach(L=>{
        const lx=PX+L.x*k, ly=topAxTop+L.y*k;
        ovl.appendChild(_svg('line',{x1:lx,y1:panelTop-3*k,x2:lx,y2:panelTop,stroke:SPN,'stroke-width':tkw}));
        if(L.color){ const c=L.color;
          _refColors[L.ro]='rgb('+Math.round(c[0]*255)+','+Math.round(c[1]*255)+','+Math.round(c[2]*255)+')'; }
        const lab=_txt(lx, ly, L.text, {fill:'#666','font-family':FSF,
          'font-size':(TA.font_px*k).toFixed(2),'text-anchor':'middle','dominant-baseline':'central',
          'font-weight':'bold','class':'rlabel','data-ro':L.ro});
        lab.style.cursor='pointer'; lab.style.userSelect='none'; lab.style.pointerEvents='auto';  // #ovl is none
        lab.addEventListener('click', ()=>{ _refPinned[L.ro]=!_refPinned[L.ro]; _refApply(L.ro); });
        (_refLabels[L.ro]||(_refLabels[L.ro]=[])).push(lab);
      });
    }
    _refApplyAll();   // re-apply persisted pinned/temp state after the rebuild
    }
    // Live spectra-density raster fills the plot box (behind the axes/refs). Set
    // its rect (CSS px) and (re)render ONCE here — layout runs on init/resize,
    // NOT on zoom, so the density adds zero per-zoom-frame cost.
    _specRect=[PX, panelTop, plotW, plotH];
    if(_specReady) drawSpectra();
  } else { _specRect=null; }
  positionExcChips();
}
window.addEventListener('resize',()=>{dpr=window.devicePixelRatio||1; refreshLayoutGeom().then(()=>{layout(); draw();});});

function vpFor(){                            // shared viewport rect in FOV-norm [0,1]
  const vw=Math.min(1, view.s), vh=Math.min(1, view.s);
  return {x:view.cx-vw/2, y:view.cy-vh/2, w:vw, h:vh};
}
function pickLevel(label, cellWpx){
  const dims=info.layers[label]; const need=cellWpx/Math.max(view.s,1e-6);
  for(let i=0;i<dims.length;i++){ if(dims[i][1]>=need) return i; } return dims.length-1;
}
function setCmap(name){
  // SDR mode + hdr_cmap → upload the non-lifted float LUT so the colormapped
  // tiles drop to plain SDR through the same INT_HDR pipeline.
  const src=(_sdrMode && HDR_CMAP && LUTS_SDR[name])?LUTS_SDR:LUTS;
  if(!src[name]) return; CMAP=name;
  backend.setColormap(name, src[name]); if(info) render();
  applySpecCmap(name);   // spectra density follows the selected cmap too
}
// Re-colorize the spectra density to the selected cmap (same per-cmap LUTs as the
// tiles), so the spectra follows the picker. SpectraGL's SDR path indexes a uint8
// LUT, its HDR path (cfg.hdr) a float LUT — match each from the viewer's LUTS.
function applySpecCmap(name){
  if(!_specReady || !_specCfg || !PANEL) return;
  if(_specFixedLut){ drawSpectra(); return; }   // categorical scatter keeps its own LUT
  if(_specCfg.hdr){
    if(LUTS[name]) _specLutHdr=Float32Array.from(LUTS[name]);
    if(LUTS_SDR[name]) _specLutSdr=Float32Array.from(LUTS_SDR[name]);
    if(_specLutHdr) _specCfg.lut=(_sdrMode && _specLutSdr)?_specLutSdr:_specLutHdr;
  } else if(LUTS[name]){
    _specCfg.lut=Uint8Array.from(LUTS[name]);
  }
  drawSpectra();
}
// Global figure title strip (info.title). Inserted at the very top of the flow
// (before ctl if present, else before #wrap) so it sits above everything without
// overlaying the tiles. No-op when /info carries no title.
function buildTitle(){
  const t = (typeof info!=='undefined' && info && info.title) ? String(info.title) : '';
  if(!t) return;
  let el=document.getElementById('figtitle');
  if(!el){ el=document.createElement('div'); el.id='figtitle';
    const ref=document.getElementById('ctl')||document.getElementById('wrap');
    document.body.insertBefore(el, ref); }
  el.textContent=t; _sizeBars();
}
function buildControls(){
  const bar=document.createElement('div'); bar.id='ctl';
  const sel=document.createElement('select');
  for(const k of Object.keys(LUTS)){ const o=document.createElement('option');
    o.value=k; o.textContent=k; if(k===CMAP)o.selected=true; sel.appendChild(o); }
  sel.onchange=()=>setCmap(sel.value);
  const btn=document.createElement('button');
  const MODES=['self','global','bitdepth'], NAMES={self:'self',global:'global',bitdepth:'bit-depth'};
  const lbl=()=>'norm: '+NAMES[NORMMODE]; btn.textContent=lbl();
  btn.onclick=()=>{ NORMMODE=MODES[(MODES.indexOf(NORMMODE)+1)%MODES.length]; btn.textContent=lbl(); render(); };
  bar.appendChild(document.createTextNode('cmap ')); bar.appendChild(sel);
  bar.appendChild(document.createTextNode(' ')); bar.appendChild(btn);
  // snap-to-cell toggle: hovering a spectra line zooms the grid to that cell's
  // bbox (+SNAP_PAD image px). Off by default (hover-zoom is opinionated).
  const snapBtn=document.createElement('button'); snapBtn.id='snapbtn';
  snapBtn.textContent='snap: '+(_snap?'on':'off'); snapBtn.title='hover a spectrum to zoom the grid to that cell';
  snapBtn.onclick=()=>{ _snap=!_snap; snapBtn.textContent='snap: '+(_snap?'on':'off'); };
  bar.appendChild(document.createTextNode(' ')); bar.appendChild(snapBtn);
  // HDR gain (WebGPU HDR canvas only): forces the RGB-tile peak above SDR white
  // so HDR is visible even when the display headroom can't be auto-read. 1.0 =
  // SDR-safe (auto). Drag up on an HDR display to see highlights exceed white.
  if(backend.HDR){
    const gl=document.createElement('span'); gl.textContent=' HDR gain ';
    const sld=document.createElement('input'); sld.type='range';
    sld.min='1'; sld.max='12'; sld.step='0.1'; sld.style.width='90px';
    sld.value=String(HRMANUAL>0?HRMANUAL:1);
    const gv=document.createElement('span'); gv.textContent=HRMANUAL>0?HRMANUAL.toFixed(1)+'×':'auto';
    sld.oninput=()=>{ HRMANUAL=(+sld.value<=1.0001)?0:+sld.value;
      gv.textContent=HRMANUAL?(+sld.value).toFixed(1)+'×':'auto';
      HEADROOM=-1; /* force pollHeadroom to re-apply */
      if(typeof window.__poll==='function') window.__poll(); render(); };
    bar.appendChild(gl); bar.appendChild(sld); bar.appendChild(gv);
  }
  // ctl lives ABOVE the figure in document flow (ctl → #wrap → hud), so it never
  // overlays the tiles. Sized by the layout scale (re-applied on resize).
  document.body.insertBefore(bar, document.getElementById('wrap'));
  _sizeBars();
}
// FETCH ONLY (network; no GPU upload) → a pending {key,label,level,meta,buf} or
// null. Split from the upload so a burst of new levels (e.g. all cells crossing
// a zoom threshold at the same view.s) can fetch in PARALLEL but UPLOAD
// one-per-frame — N synchronous texture uploads in one frame is the zoom
// "hiccup" at the level switch.
async function fetchTile(label, level){
  const key=label+'/'+level; if(texCache.has(key)) return null;
  const r=await fetch(VBASE+'tile/'+SID+'/'+label+'/'+level+'?fmt=raw');
  if(r.status===204 || !r.ok) return null;     // not projected yet — retry later
  const w=+r.headers.get('X-Level-Width'), h=+r.headers.get('X-Level-Height');
  const ch=+r.headers.get('X-Channels'), dt=r.headers.get('X-Dtype');
  const mode=r.headers.get('X-Mode')||'rgb';
  const lo=parseFloat(r.headers.get('X-Lo')||'0'), hi=parseFloat(r.headers.get('X-Hi')||'1');
  const kind=r.headers.get('X-Kind')||'reduction', bitmax=parseFloat(r.headers.get('X-Bitmax')||'1');
  const downsample=r.headers.get('X-Downsample')||'mean';   // 'nearest' label layers → NEAREST min-filter
  const buf=await r.arrayBuffer();
  return {key, label, level, meta:{w,h,ch,dt,mode,lo,hi,kind,bitmax,downsample}, buf};
}
// GPU UPLOAD (synchronous) of a pending tile → caches + returns the entry.
function uploadTile(p){
  if(!p) return null; if(texCache.has(p.key)) return texCache.get(p.key);
  const m=p.meta; const e=backend.createTile(m, p.buf); texCache.set(p.key, e);
  if(p.level===0 && !_filled.has(p.label)){ _filled.add(p.label); console.log('FILL '+p.label+' '+Math.round(performance.now())); }
  // pool the global range across READOUTS only (Mean/Max excluded — they
  // always self-scale, so they must not stretch the readout comparison range)
  if(m.mode==='intensity' && m.kind==='readout'){ RGLO=Math.min(RGLO,m.lo); RGHI=Math.max(RGHI,m.hi); BITMAX=m.bitmax; }
  return e;
}
// fetch + upload immediately — used by the startup poll (level 0), where there's
// no in-flight zoom to stall so spreading the upload buys nothing.
async function getTex(label, level){
  const key=label+'/'+level; if(texCache.has(key)) return texCache.get(key);
  return uploadTile(await fetchTile(label, level));
}
// ── Detail-crop fast-path (server ?crop=) ────────────────────────────────────
// Zoomed in, the finest FULL tile is large but only a sub-rect is on screen.
// Fetch JUST that rect at the target level so the visible region sharpens fast,
// without waiting for the whole finest tile to stream over the wire. The crop is
// PURELY ADDITIVE: it overlays the coarse pyramid base (painted only while it
// fully covers the view — see render), and the normal refine() still pulls the
// full tile in the background and supersedes it. Zoomed out, or once the full
// target level is cached, requestDetail does NOTHING → zero extra work, so the
// common grid view and small (prefetched) images are completely unaffected.
let _detail=new Map();              // label -> {reqKey, level, e}  (e.rect/e.cover set)
let _detailReq=new Set();           // in-flight request keys (dedup concurrent fetches)
const CROP_S=0.6;                   // only crop when showing <60% of the FOV across a cell
const CROP_PAD=0.25, CROP_Q=64;     // pad the window, then quantize corners to 1/Q (pan-stable key)
function _detailKeyFor(){
  if(view.s>=CROP_S) return null;                  // zoomed out → no crop (full tile is small enough)
  const vp=vpFor();
  let x0=vp.x-vp.w*CROP_PAD, y0=vp.y-vp.h*CROP_PAD;
  let x1=vp.x+vp.w*(1+CROP_PAD), y1=vp.y+vp.h*(1+CROP_PAD);
  x0=Math.max(0,x0); y0=Math.max(0,y0); x1=Math.min(1,x1); y1=Math.min(1,y1);
  // quantize OUTWARD so the rect always ⊇ the padded window AND the key is stable
  // across small pans (same crop reused → no refetch / no GPU re-upload).
  x0=Math.floor(x0*CROP_Q)/CROP_Q; y0=Math.floor(y0*CROP_Q)/CROP_Q;
  x1=Math.ceil(x1*CROP_Q)/CROP_Q;  y1=Math.ceil(y1*CROP_Q)/CROP_Q;
  if(x1-x0>=1 && y1-y0>=1) return null;             // (near-)whole FOV → no benefit
  return {x0,y0,x1,y1};
}
function requestDetail(){
  if(!backend||!info) return;
  const kk=_detailKeyFor();
  for(const cell of cells){
    if(cell.label==='RGB' && excState && backend.hasExc && backend.hasExc()) continue;  // live-compose: no pyramid crop
    const L=_texOf(cell.label);
    if(!info.layers[L]) continue;
    const target=pickLevel(L, cell.w*dpr);
    if(!kk || texCache.has(L+'/'+target)){ if(_detail.has(L)) _detail.delete(L); continue; }
    const reqKey=L+'/'+target+'/'+kk.x0+','+kk.y0+','+kk.x1+','+kk.y1;
    const cur=_detail.get(L);
    if((cur && cur.reqKey===reqKey) || _detailReq.has(reqKey)) continue;   // already showing / in flight
    _detailReq.add(reqKey);
    (async()=>{
      try{
        const r=await fetch(VBASE+'tile/'+SID+'/'+L+'/'+target+'?fmt=raw&crop='+kk.x0+','+kk.y0+','+kk.x1+','+kk.y1);
        if(r.status===204 || !r.ok) return;
        const w=+r.headers.get('X-Level-Width'), h=+r.headers.get('X-Level-Height');
        const ch=+r.headers.get('X-Channels'), dt=r.headers.get('X-Dtype');
        const mode=r.headers.get('X-Mode')||'rgb';
        const lo=parseFloat(r.headers.get('X-Lo')||'0'), hi=parseFloat(r.headers.get('X-Hi')||'1');
        const kind=r.headers.get('X-Kind')||'reduction', bitmax=parseFloat(r.headers.get('X-Bitmax')||'1');
        const downsample=r.headers.get('X-Downsample')||'mean';
        const cr=(r.headers.get('X-Crop')||'0,0,1,1').split(',').map(Number);   // x0,y0,x1,y1 (snapped)
        const buf=await r.arrayBuffer();
        const e=backend.createTile({w,h,ch,dt,mode,lo,hi,kind,bitmax,downsample}, buf);
        e.rect=[cr[0],cr[1],cr[2]-cr[0],cr[3]-cr[1]];   // FOV-norm origin+size → shader u_tr
        e.cover=cr;                                      // x0,y0,x1,y1 → coverage test in render
        _detail.set(L, {reqKey, level:target, e});
        render();
      } finally { _detailReq.delete(reqKey); }
    })();
  }
}
// ── Live spectra-density raster (rendered by SpectraGL into the panel box) ──
// The density is INDEPENDENT of the grid zoom (it's the per-cell spectra, fixed),
// so it renders ONCE on load and again only on layout/resize — NEVER per zoom
// frame. ``_specRect`` is the plot-box rect (CSS px) set by layout(); SpectraGL
// fills it via a one-shot WebGPU compute+render. Zero per-zoom-frame cost → grid
// zoom timings are unchanged whether or not the spectra are present.
let _specReady=false, _specCfg=null, _specRect=null;
// Categorical scatter (e.g. pass/fail) ships its own fixed LUT via data-lut and
// must NOT follow the cmap picker (which would re-index a binary value into the
// active colormap → e.g. all points at magma[0]=black). Set from data-fixed-lut.
let _specFixedLut=false;
// xy-panel axis scales: clicking an axis LABEL toggles linear ↔ log (legacy
// convention: natural-log VALUES on a linear axis). The host ships per-axis
// variants in panel_axes.scales = {x:{linear:{lo,hi,ticks,label,lines,shade},
// log:{...}|null}, y:{...}} plus log point coords (data-x-log / data-y-log;
// NaN for non-positive values → those points don't draw in log mode).
let _xyLog={x:false,y:false}, _xyArrs=null;
function applyXYScatter(){
  if(!_specCfg) return;
  const cv=document.getElementById('speccv'); if(!cv) return;
  const AX=info&&(info.panel_axes||info.spectra_axes); if(!AX) return;
  if(!_xyArrs) _xyArrs={x:_specCfg.x, y:_specCfg.y,
    xlog:cv.getAttribute('data-x-log')?_b64f32(cv.getAttribute('data-x-log')):null,
    ylog:cv.getAttribute('data-y-log')?_b64f32(cv.getAttribute('data-y-log')):null};
  _specCfg.x=(_xyLog.x&&_xyArrs.xlog)?_xyArrs.xlog:_xyArrs.x;
  _specCfg.y=(_xyLog.y&&_xyArrs.ylog)?_xyArrs.ylog:_xyArrs.y;
  _specCfg.xLo=AX.xlo; _specCfg.xHi=AX.xhi; _specCfg.yLo=AX.ylo; _specCfg.yHi=AX.yhi;
}
function toggleAxisScale(ax){
  const AX=info&&(info.panel_axes||info.spectra_axes); if(!AX||!AX.scales) return;
  const sc=AX.scales[ax]; if(!sc||!sc.log) return;
  _xyLog[ax]=!_xyLog[ax];
  const v=_xyLog[ax]?sc.log:sc.linear;
  if(ax==='x'){ AX.xlo=v.lo; AX.xhi=v.hi; AX.xticks=v.ticks; AX.xlabel=v.label; AX.vlines=v.lines||[]; }
  else        { AX.ylo=v.lo; AX.yhi=v.hi; AX.yticks=v.ticks; AX.ylabel=v.label; AX.hlines=v.lines||[]; }
  const vx=_xyLog.x?AX.scales.x.log:AX.scales.x.linear,
        vy=_xyLog.y?AX.scales.y.log:AX.scales.y.linear;
  AX.shade=(vx&&vy&&vx.shade&&vy.shade)
    ?{x0:vx.shade[0],x1:vx.shade[1],y0:vy.shade[0],y1:vy.shade[1]}:null;
  applyXYScatter();
  layout();           // rebuilds the axes overlay + re-renders the panel
}
// Spectra y-axis normalization: a MODE (self/global/bit-depth) × a SCOPE
// (all/visible channels). The y-axis label is two click zones — the title cycles
// the mode, the underlined parenthetical chip toggles the scope (only present when
// a channel is hidden). Swapping is a yLines swap + redraw.
let _specMode='self', _specScope='all', _specY=null;
// HDR/SDR display toggle (the HDR action button). SDR forces tile headroom→1 and
// swaps the density to its non-lifted LUT. Both density LUTs shipped when hdr_cmap.
let _sdrMode=false, _specLutHdr=null, _specLutSdr=null;
function _b64f32(s){ const bin=atob(s); const u=new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) u[i]=bin.charCodeAt(i); return new Float32Array(u.buffer); }
let _measCtx=null;   // text width measurement (for dynamic y-axis-label spacing)
function _measW(str, px){ if(!_measCtx) _measCtx=document.createElement('canvas').getContext('2d');
  _measCtx.font=px+'px system-ui,sans-serif'; return _measCtx.measureText(str).width; }
const _MODE_NAME={self:'self-norm',global:'global',bitdepth:'bit-depth'};
// data key for (mode,scope): bit-depth has no scope; 'vis' adds the '-vis' suffix
// (the all-channel variant is the bare mode key). _hasScope = a visible variant
// shipped for this mode (i.e. some channel is hidden — else scope is a no-op).
function _normKey(){ return _specMode==='bitdepth' ? 'bitdepth'
  : _specMode+(_specScope==='vis'?'-vis':''); }
function _hasScope(m){ return !!(_specY && _specY[m+'-vis']); }
// Render the y-label as two click zones (one <text>, tspans): the title (cycles
// the mode) + an optional underlined scope chip (toggles all↔visible).
function _renderNormLabel(){
  const yl=document.querySelector('text.ynorm'); if(!yl) return;
  while(yl.firstChild) yl.removeChild(yl.firstChild);
  const NS='http://www.w3.org/2000/svg';
  const t1=document.createElementNS(NS,'tspan');
  t1.textContent='Intensity ('+(_MODE_NAME[_specMode]||_specMode);
  yl.appendChild(t1);
  if(_specMode!=='bitdepth' && _hasScope(_specMode)){
    const sc=document.createElementNS(NS,'tspan');
    sc.setAttribute('class','ynorm-scope');       // own click zone (cursor inherits pointer)
    sc.textContent=', '+(_specScope==='vis'?'visible':'all');
    yl.appendChild(sc);
  }
  const t2=document.createElementNS(NS,'tspan'); t2.textContent=')'; yl.appendChild(t2);
}
function _applyNorm(){
  const key=_normKey();
  if(_specY && _specY[key] && _specCfg){ _specCfg.yLines=_specY[key]; drawSpectra(); }
  _renderNormLabel();
}
function cycleMode(){    // title click → next mode whose data shipped
  const order=['self','global','bitdepth'].filter(m=>_specY && _specY[m]);
  if(!order.length) return;
  _specMode=order[(order.indexOf(_specMode)+1+order.length)%order.length];
  if(!_hasScope(_specMode)) _specScope='all';   // mode w/o a visible variant → all
  _applyNorm();
}
function toggleScope(){  // parenthetical click → all ⇄ visible
  if(_hasScope(_specMode)){ _specScope=(_specScope==='vis'?'all':'vis'); _applyNorm(); }
}
// Reference-spectra toggle state (persists across layout rebuilds). A readout's
// dashed ref shows if pinned (its top-axis label clicked) OR temp (a cell with
// that readout is hovered in the density). ``_refPaths``/``_refLabels``/
// ``_refColors`` are rebuilt by layout(); the pinned/temp sets survive.
let _refPinned={}, _refTemp={}, _refPaths={}, _refLabels={}, _refColors={};
function _refApply(ro){
  const on=!!(_refPinned[ro]||_refTemp[ro]);
  (_refPaths[ro]||[]).forEach(p=>{ p.style.display=on?'':'none'; });
  (_refLabels[ro]||[]).forEach(l=>{ l.style.fill=on?(_refColors[ro]||'#bbb'):'#666'; });
}
function _refApplyAll(){ const seen={};
  Object.keys(_refPaths).forEach(r=>seen[r]=1); Object.keys(_refLabels).forEach(r=>seen[r]=1);
  Object.keys(seen).forEach(_refApply); }
function refSetTemp(roList){ const next={}; (roList||[]).forEach(r=>{ next[r]=true; });
  const ch={}; Object.keys(_refTemp).forEach(r=>ch[r]=1); Object.keys(next).forEach(r=>ch[r]=1);
  _refTemp=next; Object.keys(ch).forEach(_refApply); }
function refClearTemp(){ const old=_refTemp; _refTemp={}; Object.keys(old).forEach(_refApply); }
async function loadSpectra(){
  const cv=document.getElementById('speccv'); if(!cv) return;
  const r=await fetch(VBASE+'attach/'+SID+'/'+PANEL_KIND);
  if(r.status===204 || !r.ok){ setTimeout(loadSpectra, 400); return; }   // not ready — retry
  const a=await r.json();
  for(const k in a) cv.setAttribute(k, a[k]);
  _specFixedLut=(a['data-fixed-lut']==='1');   // categorical scatter → don't follow the picker
  if(!PANEL){ console.warn('SpectraGL not injected'); return; }
  try{ _specCfg=PANEL.decodeAttrs(cv); }
  catch(e){ console.warn('spectra decode:', e); return; }
  // Norm variants for the y-axis toggle: 3 modes (self/global/bit-depth) +
  // optional visible-scope variants (self-vis/global-vis, present only when a
  // channel is hidden). Default the live view to self-norm, all-channels.
  _specY={}; ['self','self-vis','global','global-vis','bitdepth'].forEach(m=>{ const s=a['data-ylines-'+m];
    if(s){ try{ _specY[m]=_b64f32(s); }catch(e){} } });
  _specScope='all';
  if(_specY['self']){ _specMode='self'; _specCfg.yLines=_specY['self']; }
  else if(a['data-norm-mode'] && _specY[a['data-norm-mode']]){ _specMode=a['data-norm-mode']; }
  // HDR density: decodeAttrs only loads the uint8 SDR LUT; for HDR swap in the
  // float lifted LUT (values >1) + set cfg.hdr so SpectraGL renders through its
  // rgba16float/extended-canvas OETF path (glows into the headroom, like tiles).
  if(a['data-hdr']==='1' && a['data-lut-hdr']){
    try{ _specLutHdr=_b64f32(a['data-lut-hdr']);
         _specLutSdr=a['data-lut-sdr']?_b64f32(a['data-lut-sdr']):null;
         _specCfg.lut=(_sdrMode&&_specLutSdr)?_specLutSdr:_specLutHdr; _specCfg.hdr=true; }
    catch(e){ console.warn('spectra HDR lut:', e); }
  }
  _specReady=true; console.log('SPECTRA loaded ('+(a['data-num-lines']||'?')+' lines'+(_specCfg.hdr?', HDR':'')+')');
  _renderNormLabel();
  applySpecCmap(CMAP);   // start on the currently-selected cmap (follows the picker), then draw
}
function drawSpectra(){
  const cv=document.getElementById('speccv'), ov=document.getElementById('specovl');
  if(!cv || !_specReady || !_specCfg || !_specRect || !PANEL) return;
  const x=_specRect[0], y=_specRect[1], w=_specRect[2], h=_specRect[3];
  cv.style.left=x+'px'; cv.style.top=y+'px'; cv.style.width=w+'px'; cv.style.height=h+'px';
  if(ov){ ov.style.left=x+'px'; ov.style.top=y+'px'; ov.style.width=w+'px'; ov.style.height=h+'px';
          try{ PANEL.clearHighlight(ov); }catch(_){} }   // drop stale highlight on re-render
  if(w<2 || h<2) return;
  Promise.resolve(PANEL.render(cv, _specCfg, w, h))
    .then(ok=>{ if(ok===false) console.warn('SpectraGL: WebGPU unavailable (iframe needs allow="webgpu")'); })
    .catch(e=>console.warn('spectra render:', e));
}
// Live-selectable: hover the density to highlight the nearest cell's spectrum on
// the 2D overlay + show its id/label. Pointer events land on #specovl (it's above
// the density; the SVG axes overlay above it is pointer-events:none). The grid
// canvas is in the grid region only, so this never steals pan/zoom events.
(function(){
  const ov=document.getElementById('specovl'), cv=document.getElementById('speccv'),
        tip=document.getElementById('sgtip');
  if(!ov || !cv) return;
  ov.addEventListener('pointermove', e=>{
    const st=cv.__sgState; if(!st || !PANEL) return;
    const r=ov.getBoundingClientRect(); if(r.width<2 || r.height<2) return;
    const mx=(e.clientX-r.left)/r.width*st.W, my=(e.clientY-r.top)/r.height*st.H;
    let line=-1; try{ line=PANEL.highlight(cv, ov, mx, my, 'rgba(255,80,80,0.95)'); }catch(_){}
    const cfg=_specCfg;
    if(line>=0 && cfg && cfg.cellIds){
      const lab=(cfg.cellLabels && cfg.cellLabels[line]) || '';
      tip.innerHTML='Cell '+cfg.cellIds[line]+(lab?'<br>'+lab:'');
      tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px';
      refSetTemp(refTokens(lab));   // light up this cell's matching reference overlays
      const _cid=cfg.cellIds[line];
      if(_snap) snapToCell(_cid);              // snap (toggle): zoom grid to this cell
      highlightCellInGrid(_cid);               // ALWAYS: white cell outline on every tile
    } else { tip.style.display='none'; refClearTemp(); clearCellHighlight(); }
  });
  ov.addEventListener('pointerleave', ()=>{ try{ PANEL.clearHighlight(ov); }catch(_){}
    if(tip) tip.style.display='none'; refClearTemp(); clearCellHighlight(); });
})();
// ── Cell info: per-cell bboxes + downsampled id map (snap-to-cell + reverse
// spectrum highlight). Loaded once (204-retries while the seg processes).
let _cellBoxes=null, _idMap=null, _idW=0, _idH=0, _snap=SNAP0, _id2line=null, _cellContours=null;
async function loadCellInfo(){
  const r=await fetch(VBASE+'attach/'+SID+'/cellboxes');
  if(r.status===204 || !r.ok){ setTimeout(loadCellInfo, 600); return; }
  const a=new Int32Array(await r.arrayBuffer());
  _cellBoxes=new Map();
  for(let i=0;i+4<a.length;i+=5) _cellBoxes.set(a[i],[a[i+1],a[i+2],a[i+3],a[i+4]]);
  try{ const r2=await fetch(VBASE+'attach/'+SID+'/cellids');
    if(r2.ok && r2.status!==204){
      _idW=+r2.headers.get('X-Map-W'); _idH=+r2.headers.get('X-Map-H');
      _idMap=new Int32Array(await r2.arrayBuffer()); } }catch(e){}
  // per-cell outline contours: [n int32][index n×(id,off,count)][verts M×2 f32]
  try{ const r3=await fetch(VBASE+'attach/'+SID+'/cellcontours');
    if(r3.ok && r3.status!==204){
      const buf=await r3.arrayBuffer(), n=new DataView(buf).getInt32(0,true);
      const idx=new Int32Array(buf, 4, n*3), verts=new Float32Array(buf, 4+n*12);
      _cellContours=new Map();
      for(let k=0;k<n;k++){ const id=idx[k*3], off=idx[k*3+1], cnt=idx[k*3+2];
        _cellContours.set(id, verts.subarray(off*2, (off+cnt)*2)); } } }catch(e){}
  console.log('CELLINFO loaded '+_cellBoxes.size+' boxes, idmap '+_idW+'x'+_idH
              +', contours '+(_cellContours?_cellContours.size:0));
}
// Snap-to-cell: zoom the (shared) grid view to a SQUARE crop centred on the cell
// + SNAP_PAD image px. The cell is ALWAYS centred — for edge cells the crop runs
// off the image (transparent beyond the FOV) rather than shifting inward, so the
// cell never drifts from the middle. (Deliberately NOT ocdkit make_square, which
// clamps the box to stay in bounds and would off-centre edge cells.)
function snapToCell(id){
  if(!_cellBoxes || !info) return; const b=_cellBoxes.get(id); if(!b) return;
  const W=info.width, H=info.height;
  const side=Math.max(b[2]-b[0], b[3]-b[1])+2*SNAP_PAD;  // square crop side (px)
  view.cx=((b[0]+b[2])/2)/W; view.cy=((b[1]+b[3])/2)/H;  // exact cell centre, no clamp
  view.s=Math.min(1, Math.max(side/W, side/H));
  stopAutoZoom(); draw();
}
// Draw the highlighted cell's actual OUTLINE (its contour — the SAME verts as the
// gray seg outline, keyed by id) in WHITE on EVERY non-empty tile (RGB/Mean/Max/
// readouts/Masks), so it stands out from the gray. Drawn whether snap is on or
// off (snap only controls the zoom). Like the GPU masks layer, the geometry is
// built ONCE (raw FOV-norm contour points) and each render() only updates a
// per-tile transform — it rides the view on zoom/pan with no per-frame DOM churn.
let _hlEls=[], _hlCellId=null, _hlBuiltId=null, _hlPolys=null;
function _clearHlEls(){ _hlEls.forEach(e=>{ try{e.remove();}catch(_){}}); _hlEls=[]; _hlPolys=null; _hlBuiltId=null; }
function clearCellHighlight(){ _clearHlEls(); _hlCellId=null; }
function highlightCellInGrid(id){ _hlCellId=id; _drawCellHighlight(); }
// build the static geometry once: points are the raw contour in FOV-norm [0,1];
// a per-tile <g clip> keeps it inside its tile. vector-effect=non-scaling-stroke
// pins the stroke width to CSS px (immune to the per-tile transform scale); the
// width itself is set per-frame in _positionCellHighlight so it scales WITH zoom
// (constant in IMAGE px, like the seg outline) instead of a fixed screen width.
function _buildCellHighlight(){
  _clearHlEls();
  if(_hlCellId==null || !_cellContours || !info || !cells) return;
  const cv=_cellContours.get(_hlCellId); if(!cv || cv.length<6) return;   // ≥3 verts
  let raw=''; for(let i=0;i<cv.length;i+=2) raw+=cv[i].toFixed(5)+','+cv[i+1].toFixed(5)+' ';
  raw=raw.trim();
  _hlPolys=[];
  cells.forEach((c,ti)=>{
    const clipId='hlclip_'+ti, clip=document.createElementNS(SVGNS,'clipPath');
    clip.setAttribute('id',clipId);
    const cr=document.createElementNS(SVGNS,'rect');
    cr.setAttribute('x',c.x); cr.setAttribute('y',c.y);
    cr.setAttribute('width',c.w); cr.setAttribute('height',c.h);
    clip.appendChild(cr); ovl.appendChild(clip); _hlEls.push(clip);
    const g=document.createElementNS(SVGNS,'g'); g.setAttribute('clip-path','url(#'+clipId+')');
    const poly=document.createElementNS(SVGNS,'polygon');
    poly.setAttribute('points',raw);
    // stroke colour per TILE: a host highlight group (e.g. pass/fail on readout
    // slices) colours this cell's contour there; default white elsewhere.
    poly.setAttribute('fill','none'); poly.setAttribute('stroke',_hlColorFor(c.label,_hlCellId));
    poly.setAttribute('stroke-linejoin','round');
    poly.setAttribute('vector-effect','non-scaling-stroke');   // width set per-frame (CSS px)
    poly.setAttribute('pointer-events','none');
    g.appendChild(poly); ovl.appendChild(g); _hlEls.push(g);
    _hlPolys.push({poly, c});
  });
  _hlBuiltId=_hlCellId;
}
// cheap per-frame: map FOV-norm → this tile's screen rect via an affine transform
// (the only thing that changes with zoom/pan). Geometry is untouched.
function _positionCellHighlight(){
  if(!_hlPolys) return; const vp=vpFor();
  // stroke width tracks zoom in IMAGE px (same convention as the seg outline:
  // OUTLINE_IMG_PX source px → CSS px via sx/imgW, floored at OUTLINE_MIN_DPX).
  const imgW=(info&&info.width)||1;
  for(const p of _hlPolys){ const c=p.c, sx=c.w/vp.w, sy=c.h/vp.h;
    p.poly.setAttribute('stroke-width', Math.max(OUTLINE_MIN_DPX, OUTLINE_IMG_PX*sx/imgW).toFixed(3));
    p.poly.setAttribute('transform','translate('+(c.x-vp.x*sx).toFixed(2)+' '
      +(c.y-vp.y*sy).toFixed(2)+') scale('+sx.toFixed(5)+' '+sy.toFixed(5)+')'); }
}
function _drawCellHighlight(){
  if(_hlCellId==null){ if(_hlPolys) _clearHlEls(); return; }
  // rebuild geometry only if the cell changed or a relayout wiped our elements
  if(_hlBuiltId!==_hlCellId || !_hlPolys || !_hlPolys[0] || !_hlPolys[0].poly.isConnected)
    _buildCellHighlight();
  _positionCellHighlight();
}
// Reverse link: stroke a KNOWN cell's spectrum on the 2D overlay (mirrors
// SpectraGL.highlight's stroke, but by line index instead of nearest-to-cursor).
function strokeSpectrumById(id){
  const cv=document.getElementById('speccv'), ov=document.getElementById('specovl');
  if(!cv || !ov || !_specReady || !_specCfg) return -1;
  const st=cv.__sgState; if(!st) return -1;
  if(!_id2line){ _id2line=new Map(); const ci=_specCfg.cellIds||[];
    for(let i=0;i<ci.length;i++) _id2line.set(ci[i],i); }
  const line=_id2line.get(id);
  const W=st.W, H=st.H;
  if(ov.width!==W) ov.width=W; if(ov.height!==H) ov.height=H;
  const ctx=ov.getContext('2d'); ctx.clearRect(0,0,W,H);
  if(line==null) return -1;
  const cfg=st.cfg, P=cfg.numPoints, sx=W/cfg.plotW, ySpan=(cfg.yHi-cfg.yLo)||1;
  ctx.strokeStyle='rgba(255,80,80,0.95)';
  ctx.lineWidth=Math.max(3,(self.devicePixelRatio||1)*2.5);
  ctx.lineJoin='round'; ctx.lineCap='round';
  for(const iv of cfg.intervals){ const s0=iv[0]|0, s1=iv[1]|0; if(s1-s0<2) continue;
    ctx.beginPath();
    ctx.moveTo(cfg.xPix[s0]*sx,(cfg.yHi-cfg.yLines[line*P+s0])/ySpan*H);
    for(let j=s0+1;j<s1;j++) ctx.lineTo(cfg.xPix[j]*sx,(cfg.yHi-cfg.yLines[line*P+j])/ySpan*H);
    ctx.stroke(); }
  return line;
}
function clearCellSpectrum(){
  const ov=document.getElementById('specovl');
  if(ov && ov.width){ try{ ov.getContext('2d').clearRect(0,0,ov.width,ov.height); }catch(e){} }
  const tip=document.getElementById('sgtip'); if(tip) tip.style.display='none';
  refClearTemp(); clearCellHighlight();
}
// Grid-cell hover → look up the cell id under the cursor (downsampled id map at
// the current view) and highlight its spectrum + its readouts' refs + a tooltip.
function hoverCellSpectrum(cell, e){
  if(!cell || !_idMap || !info){ clearCellSpectrum(); return; }
  const vp=vpFor();
  const fx=(e.offsetX-cell.x)/cell.w, fy=(e.offsetY-cell.y)/cell.h;
  const u=vp.x+fx*vp.w, v=vp.y+fy*vp.h;
  if(u<0||u>=1||v<0||v>=1){ clearCellSpectrum(); return; }
  const mx=Math.min(_idW-1,(u*_idW)|0), my=Math.min(_idH-1,(v*_idH)|0);
  const id=_idMap[my*_idW+mx];
  if(!(id>0)){ clearCellSpectrum(); return; }
  const line=PANEL.highlightById(document.getElementById('speccv'),document.getElementById('specovl'),id,'rgba(255,80,80,0.95)');
  const tip=document.getElementById('sgtip');
  const lab=(line>=0 && _specCfg && _specCfg.cellLabels && _specCfg.cellLabels[line])||'';
  if(tip){ tip.innerHTML='Cell '+id+(lab?'<br>'+lab:'');
    tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px'; }
  refSetTemp(refTokens(lab));
  highlightCellInGrid(id);   // also outline the hovered cell (with or without snap)
}
// hover wiring: chips visibility (RGB cell) + reverse spectrum highlight
canvas.addEventListener('pointermove',e=>{ if(drag) return;
  const c=cellAt(e.offsetX,e.offsetY);
  _setExcVis(!!(c&&c.label==='RGB'));
  hoverCellSpectrum(c,e); });
canvas.addEventListener('pointerleave',e=>{
  if(e.relatedTarget && excChipsEl && excChipsEl.contains(e.relatedTarget)) return;
  _setExcVis(false); clearCellSpectrum(); });
// ── Copy / Save the composited figure (GPU tiles + density + SVG overlay) ──
// Supersample 2×, drawImage the WebGPU canvases, then rasterize the SVG overlay
// (frames/labels/axes/refs/top-axis) on top. HUD / controls / action buttons
// live OUTSIDE #wrap so they're never captured. Output is a transparent PNG (the
// figure is transparent); the WebGPU-HDR canvases tonemap to SDR via drawImage.
async function compositeFigure(){
  const wrap=document.getElementById('wrap'); const fR=wrap.getBoundingClientRect();
  const SS=(window.devicePixelRatio||1)*2;
  const W=Math.max(1,Math.round(fR.width*SS)), H=Math.max(1,Math.round(fR.height*SS));
  const out=document.createElement('canvas'); out.width=W; out.height=H;
  const ctx=out.getContext('2d');
  const drawCv=cv=>{ if(!cv||!cv.width||!cv.height) return; const r=cv.getBoundingClientRect();
    if(r.width<1||r.height<1) return;
    try{ ctx.drawImage(cv,(r.left-fR.left)*SS,(r.top-fR.top)*SS,r.width*SS,r.height*SS); }
    catch(e){ console.warn('composite drawImage:',e); } };
  try{ render(); }catch(e){}                          // fresh tile backing for drawImage
  if(_specReady && _specCfg && _specRect){            // fresh density backing
    try{ await PANEL.render(document.getElementById('speccv'), _specCfg, _specRect[2], _specRect[3]); }catch(e){}
  }
  drawCv(document.getElementById('c'));               // tile grid (z0)
  drawCv(document.getElementById('speccv'));          // spectra density
  const clone=document.getElementById('ovl').cloneNode(true);   // SVG annotation layer ON TOP
  clone.setAttribute('width',W); clone.setAttribute('height',H);
  const xml=new XMLSerializer().serializeToString(clone);
  const img=new Image();
  await new Promise((res,rej)=>{ img.onload=res; img.onerror=rej;
    img.src='data:image/svg+xml;charset=utf-8,'+encodeURIComponent(xml); });
  ctx.drawImage(img,0,0,W,H);
  return out;
}
(function(){
  // HDR toggle — flips EVERYTHING: RGB tile (headroom→1), hdr_cmap intensity
  // tiles (LUT swap to the non-lifted SDR floats via setCmap), and the spectra
  // density (LUT swap). The density's async compute is awaited FIRST, then the
  // tile LUT + headroom + paint happen in one shot, so all pieces switch
  // together instead of RGB-first-then-spectra.
  const hb=document.getElementById('hdrbtn');
  if(hb) hb.addEventListener('click', async ()=>{
    _sdrMode=!_sdrMode;
    hb.classList.toggle('hdr-off', _sdrMode);
    hb.title = _sdrMode ? 'HDR: off (SDR)' : 'HDR: on';
    if(_specCfg && _specCfg.hdr && _specLutHdr && _specRect && PANEL){
      _specCfg.lut = (_sdrMode && _specLutSdr) ? _specLutSdr : _specLutHdr;
      try{ await PANEL.render(document.getElementById('speccv'), _specCfg,
                                       _specRect[2], _specRect[3]); }catch(e){}
    }
    if(typeof window.__poll==='function') window.__poll();   // HEADROOM (renders if changed)
    setCmap(CMAP);                                            // LUT variant swap + render()
  });
  const sb=document.getElementById('savebtn'), cb=document.getElementById('copybtn');
  if(sb) sb.addEventListener('click', async e=>{ const b=e.currentTarget; b.disabled=true;
    try{ const out=await compositeFigure(); const png=await new Promise(r=>out.toBlob(r,'image/png'));
      const url=URL.createObjectURL(png); const a=document.createElement('a');
      a.href=url; a.download='key_slices.png'; document.body.appendChild(a); a.click();
      document.body.removeChild(a); URL.revokeObjectURL(url);
    }catch(err){ console.error('save failed:',err); alert('Save failed: '+(err&&err.message||err)); }
    finally{ b.disabled=false; } });
  if(cb) cb.addEventListener('click', async e=>{ const b=e.currentTarget; b.disabled=true;
    try{ const out=await compositeFigure(); const png=await new Promise(r=>out.toBlob(r,'image/png'));
      await navigator.clipboard.write([new ClipboardItem({[png.type]:png})]);
      const t=b.textContent; b.textContent='Copied'; setTimeout(()=>{b.textContent=t;},1200);
    }catch(err){ console.error('copy failed:',err); alert('Copy failed: '+(err&&err.message||err)); }
    finally{ b.disabled=false; } });
})();
// ── Live RGB compose (per-excitation toggles) ───────────────────────────
// Fetch the per-excitation layers (finest level) into a texture_2d_array so the
// RGB cell composites + toggles them on the GPU. WebGL2 backend has no compose
// (no setExcArray) → silently falls back to the baked RGB tile.
async function loadExc(){
  if(!backend || !backend.setExcArray) return;
  const labels=Object.keys(info.layers).filter(l=>/^exc\d+$/.test(l))
    .sort((a,b)=>(+a.slice(3))-(+b.slice(3)));
  if(!labels.length) return;
  const layers=[]; let W=0,H=0;
  for(const l of labels){
    const dims=info.layers[l]; const lvl=dims.length-1;     // finest
    const r=await fetch(VBASE+'tile/'+SID+'/'+l+'/'+lvl+'?fmt=raw');
    if(r.status===204 || !r.ok){ setTimeout(loadExc, 300); return; }   // not projected yet
    W=+r.headers.get('X-Level-Width'); H=+r.headers.get('X-Level-Height');
    layers.push(new Uint8Array(await r.arrayBuffer()));
  }
  // Meta (scale/name/total) MUST be re-fetched now: the init-time ``info``
  // snapshot predates the exc bg projection (meta lands WITH the fill), so on a
  // cold kernel the stale snapshot gave scale=1 / total=0 → recomputeClip's
  // ``/total`` → clipHigh=Infinity → a BLACK composite. (The SVG figure path
  // re-reads /info after the layers land for exactly this reason.)
  let meta=(info.meta||{});
  try{ const fresh=await fetch(VBASE+'info/'+SID).then(r=>r.json());
       if(fresh && fresh.meta) meta=fresh.meta; }catch(e){}
  const scales=[], names=[]; let total=0;
  labels.forEach(l=>{ const m=meta[l]||{};
    scales.push(+m.scale||1); names.push(m.name||l.slice(3)); total=+m.total||total; });
  if(!(total>0)){ total=layers.length;   // last-resort exposure (never /0 → black)
    console.warn('EXC meta missing total — using n layers as exposure fallback'); }
  backend.setExcArray(layers, W, H);
  excState={layers,W,H,scales,names,total,n:layers.length,
            mask:(1<<layers.length)-1, clipHigh:1};
  recomputeClip(); buildExcChips(); console.log('EXC loaded '+layers.length); render();
}
// White point = max LINEAR luminance over the VISIBLE excitations (subsampled),
// so toggling re-exposes instead of dimming. Mirrors the SVG path's recomputeClip.
function recomputeClip(){
  if(!excState) return; const {layers,W,H,scales,n,mask}=excState;
  const total=Math.max(1e-6, +excState.total||0);   // never /0 → Infinity → black
  let ch=1e-6; const np=W*H, stp=Math.max(1,Math.floor(np/40000));
  for(let px=0;px<np;px+=stp){ let r=0,g=0,b=0; const base=px*4;
    for(let k=0;k<n;k++){ if((mask&(1<<k))===0) continue; const L=layers[k], sc=scales[k]/255;
      r+=L[base]*sc; g+=L[base+1]*sc; b+=L[base+2]*sc; }
    const mm=Math.max(r,g,b)/total; if(mm>ch) ch=mm; }
  excState.clipHigh=ch;
}
function buildExcChips(){
  if(!excState) return;
  if(excChipsEl) excChipsEl.remove();
  const bar=document.createElement('div'); excChipsEl=bar;
  bar.style.cssText='position:absolute;z-index:6;display:flex;gap:1px;flex-wrap:wrap;pointer-events:none;';
  excState.names.forEach((nm,k)=>{
    const chip=document.createElement('button'); chip.textContent=nm; chip.title='toggle '+nm+' nm';
    chip.style.cssText='pointer-events:auto;font:9px/1.1 system-ui;padding:1px 3px;margin:0;border:0;'+
      'border-radius:2px;cursor:pointer;background:rgba(0,0,0,0.5);color:#fff;';
    chip.style.opacity=(excState.mask&(1<<k))?'1':'0.3';
    chip.onclick=(e)=>{ e.stopPropagation(); excState.mask^=(1<<k);
      chip.style.opacity=(excState.mask&(1<<k))?'1':'0.3'; recomputeClip(); render(); };
    bar.appendChild(chip);
  });
  // inside #wrap (position:relative): positionExcChips uses wrap-relative cell
  // coords, and the ctl strip above #wrap would offset body-anchored absolutes.
  bar.style.opacity='0'; bar.style.transition='opacity .15s'; bar.style.pointerEvents='none';
  bar.addEventListener('mouseenter', ()=>_setExcVis(true));   // keep visible while on the chips
  bar.addEventListener('mouseleave', ()=>_setExcVis(false));
  document.getElementById('wrap').appendChild(bar);
  _excVis=true; _setExcVis(false);   // start hidden (buttons unclickable until hover)
  positionExcChips();
}
// Chips appear ONLY while hovering the RGB image (or the chips themselves),
// centered at the BOTTOM of the cell.
let _excVis=false;
function _setExcVis(v){ if(v===_excVis||!excChipsEl) return; _excVis=v;
  excChipsEl.style.opacity=v?'1':'0';
  excChipsEl.querySelectorAll('button').forEach(b=>{ b.style.pointerEvents=v?'auto':'none'; }); }
function positionExcChips(){
  if(!excChipsEl) return; const cell=cells.find(c=>c.label==='RGB');
  if(!cell){ excChipsEl.style.display='none'; return; }
  excChipsEl.style.display='flex'; excChipsEl.style.justifyContent='center';
  excChipsEl.style.left=(cell.x+cell.w/2)+'px';
  excChipsEl.style.transform='translateX(-50%)';
  excChipsEl.style.top=(cell.y+cell.h-16)+'px';   // bottom-middle, inside the cell
  excChipsEl.style.maxWidth=cell.w+'px';
}
// Render-only: paint the best-cached level for every cell at the current
// view. Never fetches → always instant, so pan/zoom is smooth from frame 1.
function cellRectPx(cell){
  const x=Math.round(cell.x*dpr), wpx=Math.round(cell.w*dpr), hpx=Math.round(cell.h*dpr);
  const y=canvas.height-Math.round((cell.y+cell.h)*dpr); return [x,y,wpx,hpx];
}
async function fetchOutline(){
  const r=await fetch(VBASE+'attach/'+SID+'/outline');
  if(r.status===204 || !r.ok){ setTimeout(fetchOutline, 300); return; }
  const buf=await r.arrayBuffer(); backend.setOutline(buf); render();
}
// Optional per-group outline colouring (e.g. pass/fail cells): the host attaches
// ``outline_groups`` = JSON {groups:[{attach,color:[r,g,b,a],on:[labels]}],
// default_on:[labels]} plus one geometry attachment per group (same miter-instance
// format as ``outline``). Each group draws on its ``on`` tile labels in its colour;
// the full white outline draws on ``default_on`` (default: Masks only). Polled only
// when /info advertises it (info.has_outline_groups) — no blind retry loop.
let OUTLINE_GROUPS=null;
async function fetchOutlineGroups(){
  if(!(info&&info.has_outline_groups)) return;
  const r=await fetch(VBASE+'attach/'+SID+'/outline_groups');
  if(r.status===204 || !r.ok){ setTimeout(fetchOutlineGroups, 500); return; }
  const spec=await r.json();
  for(const g of (spec.groups||[])){
    try{ const rb=await fetch(VBASE+'attach/'+SID+'/'+g.attach);
         if(rb.ok && rb.status!==204) backend.setOutline(await rb.arrayBuffer(), g.attach); }
    catch(e){ console.warn('outline group', g.attach, e); }
  }
  OUTLINE_GROUPS=spec; render();
}
// Highlight-contour colouring per cell GROUP (e.g. pass/fail): the host attaches
// ``highlight_groups`` = JSON {groups:[{ids:[...],color:[r,g,b,a],on:[labels]}]}.
// Unlike outline_groups (whole-population geometry) this colours only the
// HIGHLIGHTED cell's contour: on a tile in ``on``, a cell in ``ids`` strokes in
// the group colour; elsewhere it stays white. Polled when /info advertises it.
let HL_GROUPS=null;
async function fetchHighlightGroups(){
  if(!(info&&info.has_highlight_groups)) return;
  const r=await fetch(VBASE+'attach/'+SID+'/highlight_groups');
  if(r.status===204 || !r.ok){ setTimeout(fetchHighlightGroups, 500); return; }
  const spec=await r.json();
  HL_GROUPS=(spec.groups||[]).map(g=>{ const c=g.color||[1,1,1,1];
    return {ids:new Set(g.ids||[]), on:new Set(g.on||[]),
            color:'rgba('+Math.round(c[0]*255)+','+Math.round(c[1]*255)+','+Math.round(c[2]*255)+','+(c[3]!=null?c[3]:1)+')'}; });
  _clearHlEls(); _drawCellHighlight();   // rebuild any live highlight with colours
}
function _hlColorFor(label,id){
  if(HL_GROUPS) for(const g of HL_GROUPS){ if(g.on.has(label)&&g.ids.has(id)) return g.color; }
  return '#fff';
}
function render(){
  if(!backend) return;
  backend.frameBegin();
  const vp=vpFor();
  for(const cell of cells){
    // RGB cell with live per-excitation compose (WebGPU only) → composite from
    // the texture_2d_array with the current toggle mask, instead of the baked tile.
    if(cell.label==='RGB' && excState && backend.hasExc && backend.hasExc()){
      backend.paintExc(vp, cellRectPx(cell), excState.scales, excState.mask,
                       excState.total, excState.clipHigh);
      continue;
    }
    const _TL=_texOf(cell.label);   // 'ncolor' for the Masks tile when toggled
    const lvl=pickLevel(_TL, cell.w*dpr);
    let e=null;
    for(let l=lvl;l>=0;l--){ if(texCache.has(_TL+'/'+l)){ e=texCache.get(_TL+'/'+l); break; } }
    if(!e) continue;
    // GPU colormap: (lo,hi) is a uniform swap. Mean/Max ('reduction') always
    // self-scale; readouts follow NORMMODE: self / pooled-global / bit-depth.
    let lo=e.lo, hi=e.hi;
    if(e.mode==='intensity' && e.kind==='readout'){
      if(NORMMODE==='global' && RGLO<RGHI){ lo=RGLO; hi=RGHI; }
      else if(NORMMODE==='bitdepth'){ lo=0; hi=e.bitmax; }
    }
    backend.paint(e, vp, cellRectPx(cell), lo, hi);
    // Detail-crop overlay: sharper visible region streamed via ?crop=. Painted
    // ONLY while it fully covers the current window (with a tiny epsilon) — a pan
    // that outruns the cached crop falls back to the coarse base rather than
    // flashing transparent (the crop is opaque inside its rect, so no blend is
    // needed; outside, the oob-discard would clobber the base, hence the gate).
    const det=_detail.get(_TL);
    if(det && det.e){ const cv=det.e.cover, eps=1e-4;
      if(cv && cv[0]<=vp.x+eps && cv[1]<=vp.y+eps && cv[2]>=vp.x+vp.w-eps && cv[3]>=vp.y+vp.h-eps)
        backend.paint(det.e, vp, cellRectPx(cell), lo, hi, det.e.rect);
    }
  }
  // seg outline (GPU lines) on the Masks cell, synced to the shared viewport.
  // Half-width is constant in IMAGE px (scales with zoom): 1 image px spans
  // ``rw/(RAS*vw)`` device px (rw = cell device width, RAS = source width, vw =
  // FOV-fraction shown), so the stroke tracks the pixels — thin at the grid view,
  // thicker as you zoom into cells — with a small device-px floor so it stays
  // visible when the whole FOV is downsampled to one cell.
  {
    const OG=OUTLINE_GROUPS;
    const defOn=(OG&&OG.default_on)||['Masks'];   // tiles drawing the full white outline
    const vw=Math.max(1e-6, Math.min(1, view.s));
    for(const cell of cells){
      const wantDef=defOn.includes(cell.label)&&backend.hasOutline();
      const grps=OG?(OG.groups||[]).filter(g=>(g.on||[]).includes(cell.label)&&backend.hasOutline(g.attach)):[];
      if(!wantDef && !grps.length) continue;
      const rpx=cellRectPx(cell);                 // [x,y,wpx,hpx] device px
      const RAS=(info&&info.width)||rpx[2];
      const hw=Math.max(OUTLINE_MIN_DPX*dpr, (OUTLINE_IMG_PX*0.5)*(rpx[2]/(RAS*vw)));
      if(wantDef) backend.paintOutline(vp, rpx, hw, [1,1,1,0.85]);
      for(const g of grps) backend.paintOutline(vp, rpx, hw, g.color||[1,1,1,0.85], g.attach);
    }
  }
  backend.frameEnd();
  // backend + HDR diagnostics so it's obvious in-notebook whether the HDR path
  // is live: e.g. "webgpu HDR · headroom 3.00" (boosting) vs "webgl2 · SDR"
  // (no HDR — WebGPU unavailable, common inside a notebook iframe) vs
  // "webgpu HDR · headroom 1.00" (HDR canvas but display/iframe reports no
  // headroom). Open the URL in a real Chrome tab if the iframe shows headroom 1.
  const be=backend.name+(backend.HDR
    ?' HDR · headroom '+HEADROOM.toFixed(2)+' (auto '+AUTOHR.toFixed(2)+' via '+HRSRC+')'
    :' · SDR');
  hud.textContent=`${cells.length} tiles  zoom ${(100/view.s).toFixed(0)}%  ·  ${be}  (drag pan, wheel zoom)`;
  _drawCellHighlight();   // reposition the white cell outline at the current view (tracks zoom/pan)
}
// Refinement: fetch the target levels for the current view, then repaint.
// Debounced so a fast zoom doesn't fetch every level it flies past — it
// loads the level you actually land on (climbing one notch first for a
// progressive sharpen).
let _refineT=0, _refining=false, _refineLast=0;
function draw(){
  render();
  // Sharpen DURING the motion, not only on the pause: throttle a refine while
  // the view is changing (≈8/s) so the resolution climbs as you zoom, PLUS a
  // debounced final refine 60 ms after motion stops so the last frame is always
  // full-detail. (The old impl only had the debounce → refine fired solely on
  // the pause, which is why high-res appeared only when you stopped zooming.)
  const _now=(self.performance&&performance.now)?performance.now():Date.now();
  if(_now-_refineLast>120){ _refineLast=_now; refine(); requestDetail(); }
  if(_refineT) clearTimeout(_refineT);
  _refineT=setTimeout(()=>{ _refineLast=(self.performance&&performance.now)?performance.now():Date.now(); refine(); requestDetail(); }, 60);
}
async function refine(){
  if(_refining) return; _refining=true;
  try{
    // one progressive step toward each cell's target (so it sharpens, not pops)
    const want=new Map();
    for(const cell of cells){
      const L=_texOf(cell.label);
      if(!texCache.has(L+'/0')) continue;   // not projected yet (poll handles it)
      const tgt=pickLevel(L, cell.w*dpr);
      let have=-1; for(let l=tgt;l>=0;l--){ if(texCache.has(L+'/'+l)){ have=l; break; } }
      const step=Math.min(tgt, have+1);
      if(!texCache.has(L+'/'+step)) want.set(L+'/'+step, [L, step, tgt]);
    }
    if(want.size){
      console.log('REFINE @'+Math.round(performance.now())+'ms s='+view.s.toFixed(3)+' → '+[...want.keys()].join(' '));
      // Fetch every target level in PARALLEL (network, cheap, in-kernel server),
      // then UPLOAD one tile per animation frame, repainting after each. This
      // spreads the synchronous GPU texture uploads across frames instead of
      // stalling one frame with N uploads — the visible hiccup when a zoom
      // crosses a level threshold and all cells want the next level at once.
      const pend=(await Promise.all([...want.values()].map(([l,v])=>fetchTile(l,v)))).filter(Boolean);
      for(const p of pend){
        uploadTile(p); render();
        await new Promise(r=>requestAnimationFrame(r));   // yield a frame between uploads
      }
      render();
      const more=cells.some(c=>{ const L=_texOf(c.label); if(!texCache.has(L+'/0')) return false;
        const t=pickLevel(L,c.w*dpr); return !texCache.has(L+'/'+t); });
      _refining=false; if(more) draw(); return;
    }
  } finally { _refining=false; }
}

// Idle pyramid warm — prefetch+upload every (cell,level) one-per-frame so a cold
// first zoom is as smooth as a warm second one. Retries layers that aren't
// projected yet (NAS load), bails on interaction (re-kicked after via pauseWarm).
async function warmPyramid(){
  if(_warmRunning || _warmDone || !backend) return; _warmRunning=true;
  try{
    // BUDGET for background full-res prefetch. Prefetching every layer's whole-FOV
    // FINEST level makes a single-slice tile view buttery (zoom + snap are instant
    // — full-res already cached), but for a big image grid (many layers) it's tens
    // to hundreds of MB of needless traffic. So only when the finest-level total
    // fits the budget do we prefetch full-res (FINEST FIRST, so a snap/zoom finds
    // it ready); a big grid warms coarse→mid only and lets refine pull full-res on
    // demand. (Cheap: re-visited tiles are browser-cached now, so no refetch.)
    const BUDGET=32<<20;
    let finestBytes=0;
    for(const c of cells){ const d=info.layers[c.label]; if(d){ const f=d[d.length-1];
      const mt=info.meta&&info.meta[c.label]; const bpp=(mt&&mt.mode==='intensity')?2:4;
      finestBytes+=f[0]*f[1]*bpp; } }
    const prefetchFinest=finestBytes<=BUDGET;
    const buildWork=()=>{ const w=[], seen=new Set();
      const push=(lbl,l)=>{ const k=lbl+'/'+l; if(!seen.has(k)&&!texCache.has(k)){ seen.add(k); w.push([lbl,l]); } };
      if(prefetchFinest) for(const c of cells){ const d=info.layers[c.label]; if(d) push(c.label, d.length-1); }
      for(const c of cells){ const d=info.layers[c.label]; if(!d) continue;
        const top=prefetchFinest?d.length:Math.max(1,d.length-1);   // skip finest when over budget
        for(let l=0;l<top;l++) push(c.label,l); }
      return w; };
    for(let pass=0; pass<80 && !_warmDone; pass++){
      const work=buildWork();
      if(!work.length){ _warmDone=true; break; }
      let pending=false, uploaded=false;
      for(const [lbl,l] of work){
        if(_warmStop) return;                          // interaction → bail
        const p=await fetchTile(lbl,l);
        if(p){ uploadTile(p); uploaded=true; await new Promise(r=>requestAnimationFrame(r)); }
        else { pending=true; }                          // not projected yet
      }
      if(!pending){ _warmDone=true; break; }
      if(!uploaded) await new Promise(r=>setTimeout(r, 400));   // wait for projections, retry
    }
    if(_warmDone) console.log('WARM done @'+Math.round(performance.now())+'ms ('+texCache.size+' tiles, '+(prefetchFinest?'full-res prefetched':'coarse-only')+')');
  } finally { _warmRunning=false; }
}
// Interaction does NOT pause warming — panning/zooming/hovering means the user
// wants those full-res tiles SOONER, not later — so keep prefetching (GPU uploads
// stay throttled to one/frame, and the browser caps in-flight fetches). This only
// CLEARS a stop set elsewhere (the autozoom diagnostic) and ensures warm is live.
function pauseWarm(){ _warmStop=false; if(_warmResumeT) clearTimeout(_warmResumeT);
  if(!_warmRunning && !_warmDone) _warmResumeT=setTimeout(warmPyramid, 0); }

// ── Auto-zoom diagnostic (?autozoom=1) ───────────────────────────────────
// Self-runs a continuous zoom-IN → pause → zoom-OUT → pause loop so you can SEE
// whether the resolution refines DURING the motion or only on the pause —
// ``refine`` is debounced off ``draw`` (60 ms after the last frame), so while
// the zoom is animating (a draw every frame) it never fires; it fires on the
// PAUSE. So: blurry while moving, sharpens on the hold = "updates on pause".
// Any wheel/drag stops it. Console logs each phase + when refine actually runs.
let _az=null;
function stopAutoZoom(){ if(_az){ if(_az.raf) cancelAnimationFrame(_az.raf);
  if(_az.to) clearTimeout(_az.to); _az=null; console.log('autozoom: STOPPED (user input)'); pauseWarm(); } }
function startAutoZoom(){
  _warmStop=true;   // don't let warming compete with the auto-zoom animation
  const Tin=10000, Tout=1200, hold=1000, sMin=0.06, sMax=1.0, cx=0.5, cy=0.5;
  const ease=t=>t*t*(3-2*t);
  _az={raf:0, to:0};
  function ramp(t0, a, b, dur, next, label){
    function f(ts){ if(!_az) return; const u=Math.min(1,(ts-t0)/dur);
      view.s=a+(b-a)*ease(u); view.cx=cx; view.cy=cy; draw();
      if(u<1){ _az.raf=requestAnimationFrame(f); }
      else { console.log('autozoom: '+label+' done @s='+view.s.toFixed(2)+' — HOLD '+hold+'ms (watch it sharpen)');
             _az.to=setTimeout(next, hold); } }
    _az.raf=requestAnimationFrame(f);
  }
  const zin =()=>{ if(_az) ramp(performance.now(), sMax, sMin, Tin,  zout, 'zoom IN'); };
  const zout=()=>{ if(_az) ramp(performance.now(), sMin, sMax, Tout, zin,  'zoom OUT'); };
  view.cx=cx; view.cy=cy;
  console.log('autozoom: starting (zoom in '+Tin+'ms, hold '+hold+'ms, out '+Tout+'ms, repeat). Scroll/drag to stop.');
  _az.to=setTimeout(zin, 600);
}
try{ if(new URLSearchParams(location.search).get('autozoom')){
  // Wait for the pyramid WARM so the diagnostic reflects the warmed (smooth)
  // state, but cap the wait so a slow/never-finishing projection can't hang it.
  const _t0=(self.performance&&performance.now)?performance.now():Date.now();
  const _now=()=>(self.performance&&performance.now)?performance.now():Date.now();
  const _wait=()=>{ if((_warmDone && _paintedAll) || (_now()-_t0)>20000) startAutoZoom();
                    else setTimeout(_wait, 200); };
  setTimeout(_wait, 500);
} }catch(e){}

// pan/zoom in shared FOV-norm coords; anchored at the cursor's cell
function cellAt(px,py){ for(const c of cells){ if(px>=c.x&&px<c.x+c.w&&py>=c.y&&py<c.y+c.h) return c; } return null; }
let drag=null;
canvas.addEventListener('pointerdown',e=>{stopAutoZoom();pauseWarm();drag={x:e.clientX,y:e.clientY,c:cellAt(e.offsetX,e.offsetY)};canvas.setPointerCapture(e.pointerId);});
canvas.addEventListener('pointermove',e=>{ if(!drag||!drag.c)return;
  view.cx-=(e.clientX-drag.x)/drag.c.w*view.s; view.cy-=(e.clientY-drag.y)/drag.c.h*view.s;
  drag.x=e.clientX; drag.y=e.clientY; draw(); });
canvas.addEventListener('pointerup',()=>{drag=null;});
canvas.addEventListener('wheel',e=>{e.preventDefault(); stopAutoZoom(); pauseWarm(); const c=cellAt(e.offsetX,e.offsetY); if(!c)return;
  const fx=(e.offsetX-c.x)/c.w, fy=(e.offsetY-c.y)/c.h;       // cursor frac within the cell
  const vp=vpFor(); const ax=vp.x+fx*vp.w, ay=vp.y+fy*vp.h;  // anchor in FOV-norm
  // macOS trackpad PINCH comes through as ctrl+wheel with much smaller deltaY
  // than scroll → amplify it so a pinch zooms at the same rate as a scroll.
  const zk = e.ctrlKey ? 0.01 : 0.0015;
  view.s=Math.min(1, view.s*Math.exp(e.deltaY*zk));
  const nvp={w:Math.min(1,view.s)};
  view.cx=ax-(fx-0.5)*nvp.w; view.cy=ay-(fy-0.5)*nvp.w; draw();
},{passive:false});
init();
</script></body></html>"""


def grid_html(sid: str, *, panel: str = "spectra", hdr_cmap: str = "",
              hdr_gain: str = "auto", ref_token_re: str = "") -> str:
    """Render the grid viewer for ``sid`` with the per-request injections.

    ``panel``: "spectra" | "scatter" (which LinkedPanel the bottom box hosts).
    ``hdr_cmap``: a colormap name -> intensity tiles glow through the HDR lift.
    ``ref_token_re``: optional regex (token chars only) -> when a cell label
    matches, the matching reference overlays light up; "" disables it.
    """
    import re as _re
    try:
        g = 0.0 if hdr_gain == "auto" else max(0.0, float(hdr_gain))
    except (TypeError, ValueError):
        g = 0.0
    hc = hdr_cmap if hdr_cmap and hdr_cmap.isidentifier() else ""
    # Allow only token chars + regex metacharacters that are safe to inline into
    # a JS regex literal /…/g (word chars, a literal backslash for \d etc., and
    # the common metacharacters) — blocks `/`, `<`, `;`, quotes → no injection.
    safe = ref_token_re if ref_token_re and _re.fullmatch(r"[\w\\+*?.|()\[\]^$-]+", ref_token_re) else ""
    re_js = ("/%s/g" % safe) if safe else "null"
    return (_GRID_HTML.replace("__SPECTRA_GL__", _spectra_gl_js())
            .replace("__SCATTER_GL__", _scatter_gl_js())
            .replace("__PANEL_KIND__", panel if panel in ("spectra", "scatter") else "spectra")
            .replace("__REF_TOKEN_RE__", re_js)
            .replace("__SID__", sid)
            .replace("__LUTS__", _luts_hdr_json() if hc else _luts_json())
            .replace("__LUTS_SDR__", _luts_sdr_float_json() if hc else "{}")
            .replace("__HDR_CMAP__", hc)
            .replace("__HRGAIN__", repr(g)))


def view_html(sid: str, layer: str = "") -> str:
    return _VIEWER_HTML.replace("__SID__", sid).replace("__LAYER__", layer)


def viewgl_html(sid: str, layer: str = "") -> str:
    return _VIEWER_GL_HTML.replace("__SID__", sid).replace("__LAYER__", layer)
