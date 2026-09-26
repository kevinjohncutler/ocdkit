"""Build icon_options.html: candidate icons for the viewer's panel labels.

Searches four open-source icon sets (unpacked npm tarballs under ICON_ROOT) by
keyword for each control and lays the matches out on one self-contained page,
shown at panel size in the viewer's gray. Click a tile to pick it; the picks
are listed at the top so they can be copied back.
Usage: ICON_ROOT=<dir with the unpacked packages> build.py
"""
import html
import os
import re
from pathlib import Path

ROOT = Path(os.environ["ICON_ROOT"])
HERE = Path(__file__).resolve().parent

SETS = [  # (library label, variant, directory, filename suffix to strip)
    ("Tabler", "outline", "tabler-icons-*/package/icons/outline", ""),
    ("Tabler", "filled", "tabler-icons-*/package/icons/filled", ""),
    ("Phosphor", "regular", "phosphor-icons-core-*/package/assets/regular", ""),
    ("Phosphor", "bold", "phosphor-icons-core-*/package/assets/bold", "-bold"),
    ("Phosphor", "fill", "phosphor-icons-core-*/package/assets/fill", "-fill"),
    ("Lucide", "", "lucide-static-*/package/icons", ""),
    ("Material Design", "", "mdi-svg-*/package/svg", ""),
]

SLOTS = [
    ("clip", "Percentile clipping (the new double-ended slider row; the old GUI used Phosphor scissors fill)",
     r"scissor|(^|-)cut(-|$)|histogram|percent|collapse-horizontal|arrows?-(in|horizontal|left-right)|fold-horizontal|"
     r"brackets|sliders-horizontal|adjustments-horizontal|(^|-)crop|clamp|(^|-)range|arrow-bar"),
    ("gamma", "Gamma (brightness curve)",
     r"^gamma|contrast|brightness|circle-half|half-full|sun-dim|spline|(^|-)ease|curve|exposure|tone|chart-line|"
     r"wave-sine|math-function|bell"),
    ("alpha", "Alpha (transparent low end of the colormap)",
     r"^alpha$|opacity|transparen|checker|(^|-)drop(let)?(-|$)|droplet-half|blend|ghost|gradient|background|fade|"
     r"layers-(difference|intersect|subtract)|square-half|circle-dashed"),
    ("density", "Density (EA absorption)",
     r"fog|haze|cloud(-fog)?$|blur|grain|density|dots-(nine|six)|weight|atom|cube(-3d)?-?(sphere|scan)?$|smoke|mist"),
    ("spin", "Spin (continuous rotation)",
     r"rotate|refresh|360|orbit|(^|-)spin|clockwise|loop|sync|3d-rotation|repeat|cached|autorenew|turntable"),
]
EXCLUDE = re.compile(r"-off$|^alpha-[a-z]|^alphabet|^alphabetical|-slash$|^format-|^chart-(pie|donut)")


def load(slot_re):
    rx = re.compile(slot_re)
    out = []
    for lib, var, pattern, suffix in SETS:
        dirs = list(ROOT.glob(pattern))
        if not dirs:
            continue
        for f in sorted(dirs[0].glob("*.svg")):
            name = f.stem.removesuffix(suffix)
            if EXCLUDE.search(name) or not rx.search(name):
                continue
            svg = f.read_text()
            svg = re.sub(r"<\?xml.*?\?>|<!--.*?-->", "", svg, flags=re.S).strip()
            svg = re.sub(r'\s(width|height|class)="[^"]*"', "", svg, count=3)
            if lib == "Material Design" or lib == "Phosphor":
                svg = svg.replace("<svg", '<svg fill="currentColor"', 1)
            out.append((lib, var, name, svg))
    return out


def main():
    parts = ["""<!doctype html><html><head><meta charset="utf-8"><title>Icon Options</title><style>
:root{--bg:#fafafa;--panel:#f0f0f0;--fg:#171717;--muted:#525252;--icon:#737373;--line:#d4d4d4;--pick:#171717}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171717;--panel:#262626;--fg:#e5e5e5;--muted:#a3a3a3;--icon:#a3a3a3;--line:#404040;--pick:#e5e5e5}}
body{background:var(--bg);color:var(--fg);font:13px -apple-system,system-ui,sans-serif;margin:20px}
h2{margin:0 0 4px}h3{margin:26px 0 4px}p{color:var(--muted);margin:4px 0 10px}
#picks{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:18px;flex-wrap:wrap;align-items:center}
#pickList{display:flex;gap:16px;flex-wrap:wrap}#picks .slot{display:flex;gap:6px;align-items:center;color:var(--muted)}#picks .slot b{color:var(--fg);font-weight:600}
#picks .prev{width:16px;height:16px;color:var(--icon);display:inline-flex}
#copy{padding:5px 12px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font:inherit;cursor:pointer}
#q{padding:5px 9px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:var(--fg);font:inherit;width:220px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(96px,1fr));gap:6px}
.tile{background:var(--panel);border:1px solid transparent;border-radius:10px;padding:8px 4px 6px;text-align:center;cursor:pointer}
.tile:hover{border-color:var(--line)}.tile.picked{border-color:var(--pick)}
.icons{display:flex;justify-content:center;align-items:center;gap:12px;color:var(--icon);height:26px}
.icons svg{display:block}.s16{width:16px;height:16px}.s24{width:24px;height:24px}
.name{font-size:10.5px;color:var(--fg);margin-top:5px;word-break:break-word}.lib{font-size:9.5px;color:var(--muted)}
.lib-head{font-size:11px;color:var(--muted);margin:10px 0 4px;text-transform:uppercase;letter-spacing:.04em}
</style></head><body>
<h2>Icon options for the panel controls</h2>
<p>Each tile shows the icon at 16 px (the panel's label size) and 24 px, in the panel's gray. Click one per section to pick it; your picks collect in the bar below, with a button to copy them. Libraries: Tabler (what the viewer already uses), Phosphor (the old Qt GUI's scissors), Lucide, Material Design Icons. All are MIT or Apache licensed.</p>
<div id="picks"><input id="q" placeholder="Filter by name" /><span id="pickList"></span><button id="copy" type="button">Copy picks</button></div>
"""]
    for key, title, rx in SLOTS:
        icons = load(rx)
        parts.append(f"<h3 id='{key}'>{html.escape(title)} <span class='lib'>({len(icons)} matches)</span></h3>")
        last = None
        for lib, var, name, svg in icons:
            head = f"{lib} {var}".strip()
            if head != last:
                if last is not None:
                    parts.append("</div>")
                parts.append(f"<div class='lib-head'>{head}</div><div class='grid'>")
                last = head
            s16 = svg.replace("<svg", '<svg class="s16"', 1)
            s24 = svg.replace("<svg", '<svg class="s24"', 1)
            parts.append(f"<div class='tile' data-slot='{key}' data-id='{html.escape(head)}: {name}'>"
                         f"<div class='icons'>{s16}{s24}</div><div class='name'>{name}</div></div>")
        if last is not None:
            parts.append("</div>")
    parts.append("""<script>
const picks = {};
const order = [...new Set([...document.querySelectorAll('.tile')].map(t => t.dataset.slot))];
function draw() {
  document.getElementById('pickList').innerHTML = order.map(s => {
    const p = picks[s];
    return '<span class="slot">' + s + ': ' + (p ? '<span class="prev">' + p.svg + '</span><b>' + p.id + '</b>' : '<i>none</i>') + '</span>';
  }).join(' ');
  document.querySelectorAll('#pickList .prev svg').forEach(sv => { sv.setAttribute('width', 16); sv.setAttribute('height', 16); });
}
document.addEventListener('click', e => {
  const t = e.target.closest('.tile'); if (!t) return;
  document.querySelectorAll('.tile[data-slot="' + t.dataset.slot + '"]').forEach(x => x.classList.remove('picked'));
  t.classList.add('picked');
  picks[t.dataset.slot] = {id: t.dataset.id, svg: t.querySelector('svg').outerHTML};
  draw();
});
document.getElementById('copy').addEventListener('click', () => {
  const txt = order.map(s => s + ': ' + (picks[s] ? picks[s].id : 'none')).join('\\n');
  navigator.clipboard && navigator.clipboard.writeText(txt);
});
document.getElementById('q').addEventListener('input', e => {
  const q = e.target.value.toLowerCase();
  document.querySelectorAll('.tile').forEach(t => { t.style.display = t.dataset.id.toLowerCase().includes(q) ? '' : 'none'; });
});
draw();
</script></body></html>""")
    out = HERE / "icon_options.html"
    out.write_text("\n".join(parts))
    print(out, f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
