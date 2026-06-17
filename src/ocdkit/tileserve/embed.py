"""Embed a tileserve viewer in a notebook.

Two host-neutral helpers used by any tileserve consumer:

* :func:`figure_embed_height` — the figure height (Wref units) for the iframe's
  self-sizing ``aspect-ratio`` box, read straight from :func:`compute_layout`
  (geometry + chrome strip heights) so callers never re-specify a layout default.
* :func:`embed_viewer` — the viewer ``<iframe>`` + the Jupyter-proxy bootstrap
  script (remote-safe: probes the ``ocdkit-tiles`` proxy, then jupyter-server-
  proxy, falling back to direct ``127.0.0.1`` only on a local page).

Domain-agnostic: nothing here knows about scenes, scopes, barcodes, etc. — the
caller supplies the source id, the server base URL, and the viewer URL.
"""
from __future__ import annotations

from .layout import compute_layout, WREF


def figure_embed_height(grid, layers, panel_axes, *, has_title, wref=WREF):
    """Wref-unit height of the embedded figure = ``compute_layout`` geometry +
    its chrome-strip heights (ctl + hud, plus title when present). The ONE place
    an embed computes its aspect, so no layout default is specified twice.

    ``layers`` is the ``info.layers`` mapping (only its keys matter, for the
    grid-less flat-wrap fallback); pass ``{}`` for a panel-only figure.
    """
    g = compute_layout(grid, layers or {}, panel_axes, wref)
    full_h = float(g["full_h"]) + float(g["ctl_h"]) + float(g["hud_h"])
    if has_title:
        full_h += float(g["title_h"])
    return full_h


def embed_viewer(sid, base, url, full_h, *, background="transparent", wref=WREF):
    """Return an IPython ``HTML`` embedding the tileserve viewer at ``url``.

    The iframe is self-sizing (``width:100%`` + ``aspect-ratio: wref/full_h``)
    so it grows/shrinks with the notebook output box with no letterbox. A small
    bootstrap script rewrites ``iframe.src`` to a remote-safe path: it probes the
    ``ocdkit-tiles`` Jupyter proxy then ``jupyter-server-proxy`` (so the figure
    works when the notebook is opened on another machine, inheriting Jupyter's
    HTTPS + auth), and only falls back to direct ``http://127.0.0.1:{port}`` when
    the page itself is local (or a VS Code webview). ``full_h`` is in Wref units.
    """
    from IPython.display import HTML
    import json as _json
    _bg = str(background or "transparent")
    _port = base.rsplit(":", 1)[1]
    _path = url[len(base):]                       # /grid/{sid}?…
    _iid = f"ocdtile-{sid}"
    _script = (
        "<script>(function(){var f=document.getElementById(" + _json.dumps(_iid) + ");if(!f)return;"
        "function jbase(){"
        "try{var el=document.getElementById('jupyter-config-data');"
        "if(el){var c=JSON.parse(el.textContent||'{}');if(c.baseUrl)return c.baseUrl;}}catch(e){}"
        "var b=document.body&&document.body.dataset&&document.body.dataset.baseUrl;if(b)return b;"
        "var m=location.pathname.match(/^(.*?\\/)(lab|notebooks|tree|voila|files|nbclassic)(\\/|$)/);"
        "return m?m[1]:'/';}"
        "var jb=jbase();if(jb.slice(-1)!=='/')jb+='/';"
        "var port=" + _json.dumps(str(_port)) + ",pathabs=" + _json.dumps(_path) + ",sid=" + _json.dumps(sid) + ";"
        "var directUrl='http://127.0.0.1:'+port+pathabs;"
        "var bases=[jb+'ocdkit-tiles/'+port, jb+'proxy/'+port];"
        "var local=['localhost','127.0.0.1','::1',''].indexOf(location.hostname)>=0"
        "||location.protocol==='vscode-webview:';"
        "function fail(code){var d=document.createElement('div');"
        "d.style.cssText='padding:10px 14px;font:12px ui-monospace,monospace;color:#f87171;"
        "border:1px solid #f87171;border-radius:6px;background:#0008';"
        "d.textContent='live figure unavailable from this machine: enable the tile proxy "
        "on the Jupyter server (jupyter server extension enable ocdkit.tileserve.jupyter_ext), "
        "then RESTART the Jupyter server. probe '+jb+'ocdkit-tiles/'+port+'/ failed ('+(code||'network')+')';"
        "f.parentNode.replaceChild(d,f);}"
        "(function tryNext(i,last){"
        "if(i>=bases.length){if(local){f.src=directUrl;}else{fail(last);}return;}"
        "fetch(bases[i]+'/info/'+sid,{credentials:'same-origin'})"
        ".then(function(r){if(r.ok){f.src=bases[i]+pathabs;}else{tryNext(i+1,r.status);}})"
        ".catch(function(){tryNext(i+1,0);});})(0,0);})();</script>"
    )
    return HTML(
        f'<div style="line-height:0">'
        f'<iframe id="{_iid}" src="{url}" scrolling="no" frameborder="0" allowtransparency="true" '
        f'allow="webgpu; clipboard-write" '
        f'style="display:block;border:0;width:100%;height:auto;color-scheme:dark;'
        f'background:{_bg};aspect-ratio:{wref:.0f} / {full_h:.1f};"></iframe></div>'
        f'{_script}')
