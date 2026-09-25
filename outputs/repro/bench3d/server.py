"""Bench server for the 3D renderer A/B harness.

Serves the harness page, the SHIPPED viewer JS/WGSL (read straight from
src/ocdkit/viewer/web/js, never copied), the prepared volumes, and endpoints
that run the SHIPPED server-side code paths:
  POST /pick_server/{ds}   SessionManager._march_ray (today's 3D pick), per ray
  GET  /tx/current/{ds}    session._encode_array bundle (today's /api/volume_bundle)
  GET  /tx/f16/{ds}        candidate: normalized float16 + groups as raw binary
Results posted by the page are appended to results/*.jsonl.

Usage: python server.py DATA_DIR [PORT]
"""
import gzip
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from ocdkit.viewer.session import SessionManager, _encode_array

HERE = Path(__file__).resolve().parent
JS = HERE.parents[2] / "src" / "ocdkit" / "viewer" / "web" / "js"
DATA = Path(sys.argv[1])
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

app = FastAPI()
_cache = {}


def ds_arrays(ds):
    if ds not in _cache:
        d = DATA / ds
        m = json.loads((d / "meta.json").read_text())
        shp = (m["NZ"], m["NY"], m["NX"])
        _cache[ds] = dict(
            meta=m,
            src=np.load(d / "src_img.npy"),
            grp=np.fromfile(d / "grp_u8.bin", np.uint8).reshape(shp),
            lab=np.fromfile(d / "lab.bin", np.dtype(m["lab_dtype"])).reshape(shp),
        )
    return _cache[ds]


@app.post("/result/{kind}")
async def result(kind: str, request: Request):
    body = await request.body()
    with open(RESULTS / f"{kind}.jsonl", "ab") as fh:
        fh.write(body.strip() + b"\n")
    return {"ok": True}


@app.post("/png/{name}")
async def png(name: str, request: Request):
    (RESULTS / "png").mkdir(exist_ok=True)
    (RESULTS / "png" / name).write_bytes(await request.body())
    return {"ok": True}


@app.post("/ping")
async def ping(request: Request):
    await request.body()
    return {"ok": True}


@app.post("/pick_server/{ds}")
async def pick_server(ds: str, request: Request):
    """Run the shipped SessionManager._march_ray for each ray; report hit + time."""
    rays = await request.json()
    st = SimpleNamespace(current_mask_volume=ds_arrays(ds)["lab"])
    out = []
    for r in rays:
        t0 = time.perf_counter()
        v = SessionManager._march_ray(None, st, r["ro"], r["rd"], r["boxMin"], r["boxMax"])
        dt = (time.perf_counter() - t0) * 1e3
        lab = int(st.current_mask_volume[v]) if v is not None else 0
        out.append({"label": lab, "voxel": list(v) if v is not None else None, "ms": dt})
    return out


@app.get("/tx/current/{ds}")
def tx_current(ds: str):
    a = ds_arrays(ds)
    t0 = time.perf_counter()
    bundle = {"meta": {"dim": 3, "depth": a["meta"]["NZ"], "height": a["meta"]["NY"],
                       "width": a["meta"]["NX"]},
              "image": _encode_array(a["src"]), "mask": _encode_array(a["grp"])}
    body = json.dumps(bundle).encode()          # FastAPI would do this for the dict return
    ms = (time.perf_counter() - t0) * 1e3
    return Response(body, media_type="application/json",
                    headers={"X-Enc-Ms": f"{ms:.2f}", "Cache-Control": "no-store"})


@app.get("/tx/f16/{ds}")
def tx_f16(ds: str, gz: int = 0):
    """Candidate: normalize once on the server with the viewer's exact math, send
    float16 intensity + uint8 groups as one binary body (optionally gzip level 1
    as HTTP Content-Encoding, which the browser inflates natively)."""
    a = ds_arrays(ds)
    t0 = time.perf_counter()
    src = a["src"].astype(np.float64, copy=False)
    lo, hi = float(src.min()), float(src.max())
    f16 = ((src - lo) * (1.0 / (hi - lo))).astype(np.float32).astype(np.float16)
    body = f16.tobytes() + a["grp"].tobytes()
    headers = {"Cache-Control": "no-store"}
    if gz:
        body = gzip.compress(body, 1)
        headers["Content-Encoding"] = "gzip"
    ms = (time.perf_counter() - t0) * 1e3
    headers["X-Enc-Ms"] = f"{ms:.2f}"
    return Response(body, media_type="application/octet-stream", headers=headers)


# Pinned baseline: the benchmark's "Shipped" shader is the pre-port version, read
# from git, so the A/B pages keep working after the source shader changes.
BASELINE_REV = "c53d245"


@app.get("/baseline/{name}")
def baseline(name: str):
    import subprocess
    code = subprocess.run(["git", "-C", str(HERE), "show", f"{BASELINE_REV}:src/ocdkit/viewer/web/js/{name}"],
                          capture_output=True, check=True).stdout
    return Response(code, media_type="text/plain")


app.mount("/js", StaticFiles(directory=JS), name="js")
app.mount("/data", StaticFiles(directory=DATA), name="data")
app.mount("/", StaticFiles(directory=HERE, html=True), name="bench")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(sys.argv[2]) if len(sys.argv) > 2 else 8765,
                log_level="warning")
