# Tile server → out-of-process (design + phasing)

## Problem
The tileserve HTTP server runs as a **daemon thread inside the Jupyter kernel**
(`tileserve/server.py`, FastAPI/uvicorn on stable ports 8137–8140). When the
kernel computes (Run-All), the server thread is GIL-starved, so tile/attach
fetches stall — surfacing as `SvgFigure hi-res upgrade gave up`, soft tiles, and
a ~22× TTFB spike under load. Moving serving to its own process removes the GIL
coupling entirely.

## Current architecture (factual map)
- **Server**: FastAPI + uvicorn, daemon thread; `ensure_server()` picks a stable
  port (8137–8140) and blocks until the socket accepts; `_SOURCES: dict[sid,
  TileSource]` is the registry.
- **Source data** lives in kernel memory: `TileSource._pyr` (numpy pyramid
  arrays), `.attachments` (pre-encoded byte blobs), `.meta` (small dicts).
- **`/tile`** (`server.py` ~321): crop the pyramid array + dtype-convert +
  `_encode_level` (raw `tobytes()` / jxl / png) — **per-request compute, holds
  the GIL**. **`/attach`**: returns pre-computed bytes (zero compute). `/info`,
  `/layout`, `/grid`: dict/HTML assembly.
- **Production stays in the kernel**: `register_lazy` producers, `ArraySource`/
  `_RawF16Source.get_bytes` (can touch a Scene via `resolve_linear_p3`) run once
  in a kernel daemon thread. Only the *repeated serving* is what contends.
- **Proxy**: `jupyter_ext.py` maps `/ocdkit-tiles/<port>/…` → `127.0.0.1:<port>`;
  `_ALLOWED_PORTS = {8137..8140}` mirrors `server._STABLE_PORTS`.

## Phase 0 findings (done — this de-risks the rest)
1. **Domain-decoupled**: serving a tile never calls a live Scene/torch object;
   all needed data is snapshotted at register/fill/attach time. ✓ feasible.
2. **Import graph is heavy**: `import ocdkit.tileserve.server` pulls **torch,
   dask, pandas, scipy, PIL** — NOT because serving needs them, but because
   `plot/__init__.py:11,14` eagerly does `from .figure import figure` /
   `from .image_grid import image_grid`. `plot/pyramid.py` itself is torch-free.
   ⇒ a subprocess that imports the server as-is duplicates torch (~300MB, ~2s).
3. Importing the server spawns **no threads** and does **not** load FastAPI/
   uvicorn at import (lazy) — good; the process boundary is clean.

**Consequence**: add an import-decoupling step. Two options:
- **A (lazy `plot/__init__`)**: convert the eager figure/image_grid imports to
  `__getattr__` lazy access (the package already uses `enable_submodules`).
  Speeds up *all* ocdkit imports; risk = call sites that expect eager
  `ocdkit.plot.figure` at import.
- **B (move the torch-free bits)**: relocate `pyramid.py` + the jxl/png byte
  encoders into a torch-free module (e.g. `tileserve/_raster.py`) the server
  imports directly, bypassing `plot/__init__`. Contained, lower-risk; updates
  the few `from ..plot.pyramid import` sites.
- **C (accept torch first)**: ship Phase 1 importing the server as-is (heavy
  subprocess); the baselined `setup()` pre-warm hides startup. Correctness-
  complete (serving is off-GIL regardless); optimize imports in 1.5. ← recommended
  for the first cut, since it fixes the symptom without import surgery.

## Phase 0 findings — host extensions + the RPC surface (also done)
4. **The child needs host route extensions.** `make_app()` carries the generic
   routes (`/info /tile /attach /grid /view /viewgl`), but hosts mount more via
   `register_extension(fn)` — e.g. hostpkg's `serve/tiles.py:140`
   `register_extension(_hostpkg_routes)` adds `/spectra/{sid}/{row}`. The child must
   register the SAME extensions, so the spawn handshake passes the kernel's list of
   extension-registering module names (e.g. `['hostpkg.serve.tiles']`); the child
   imports them before `make_app()`. (Per Option C this also pulls the host's
   weight — acceptable for the first cut.)
5. **The RPC/command surface spans BOTH packages — and all of it is snapshot-safe.**
   The route handlers read from module-global stores populated at register time:
   `_hostpkg_routes`'s `/spectra` serves from `_SPECTRA` (filled by
   `register_spectra`), with **no live Scene access at request time** (verified;
   its docstring notes all domain overlays ride the generic `/attach`). So the
   data-population functions that must become RPC-to-child are:
   - ocdkit: `register / register_pending / fill / register_lazy / register_array /
     attach / drop`
   - hostpkg (`serve/tiles.py`): `register_spectra / set_spectra_axes /
     set_panel_axes / set_spectra_data / set_outline / set_cellinfo /
     set_cell_contours`
   Generalize as a **command registry**: each package registers its population
   functions as "child commands"; the kernel-side proxies route them over the
   control socket; the child applies them to its own stores. `register_lazy`
   producers still run in the kernel (they need live objects); only their RESULT
   is pushed.

   Implication: **Phase 1 touches both ocdkit and hostpkg.** It's a well-scoped but
   substantial cross-package refactor of the data-population path, not a drop-in.

## Target architecture
A lightweight **child process** runs the same FastAPI app and binds the same
ports. **Nothing client-side changes** — `embed.py`, `jupyter_ext.py`, the proxy,
and all baked URLs keep working because the port contract is preserved. The
kernel becomes a *producer + pusher*; the child is the *server*.

| Concern | Process |
|---|---|
| HTTP serving, `/tile` crop+encode, `/attach`, `/info`, `/grid` | Child (off-GIL) |
| Source construction, lazy producers (Scene/torch), attachment encode | Kernel (once) |
| `_SOURCES` | Kernel = source of truth; child holds a serving mirror |

### Data transfer
- **Pyramid arrays** (`_pyr`, tens of MB, served by crop-on-demand): a
  `multiprocessing.shared_memory` / mmap'd `/dev/shm` buffer — zero-copy. Kernel
  writes once, sends `{shm_name, shape, dtype, meta}`; child maps + crops +
  encodes there. (Phase 4: allocate the pyramid *directly* in shm — no copy.)
- **Attachment bytes** (already small/compressed): over the control channel.
- **Metadata** (`/info`, `/layout`): JSON over the control channel.
- **Control channel**: a Unix-domain socket; kernel→child commands
  `register / fill / attach / lazy_result / evict`, mirroring `server.py`'s API.

### Lifecycle
- `ensure_server()` spawns the child (daemon, dies with the kernel) + handshakes
  the control socket; `reset_server()` kills + respawns.
- Open decisions: (a) adopt an existing child across kernel restarts vs. spawn
  fresh; (b) shm cleanup on source evict + on exit (resource_tracker quirks);
  (c) child-crash → respawn + kernel re-pushes live sources (kernel is the truth).

## Phasing (each independently shippable)
- **Phase 0 — DONE**: feasibility audit (above). Lock the kernel↔server API
  surface (`register_pending/fill/register_lazy/attach/get_source/evict`).
- **Phase 1**: extract the FastAPI app into a standalone entry
  (`python -m ocdkit.tileserve._server_proc`) driven by a control socket;
  push data **by value** (socket copy) to prove the boundary. Use option **C**
  (accept heavy imports) — this alone kills the GIL contention. `TileSource`
  becomes a thin client; `ensure_server`/`reset_server` manage the child.
- **Phase 1.5**: import decoupling (option **A** or **B**) → light child.
- **Phase 2**: swap array transfer to shared memory (zero-copy).
- **Phase 3**: lifecycle hardening (crash respawn, shm cleanup, restart adoption,
  Windows `spawn` + shm naming).
- **Phase 4**: allocate pyramids directly in shm (drop the copy).

Phase 1 resolves the user-visible symptom; 1.5–4 are perf/robustness.

## Verification
Re-run the original GIL-starvation harnesses (`outputs/repro/server_ttfb_
contention.py`, `contention_harness.py`) with a GIL-holding kernel loop and
confirm TTFB stays flat (was 0.6→13ms under load). Add a child-crash/respawn
test in Phase 3.

## Alternatives rejected
- **Sub-interpreters (PEP 684)**: uvicorn/FastAPI aren't sub-interpreter-safe;
  cross-interpreter array sharing is restricted. Too immature.
- **Release the GIL in the encoder only**: the uvicorn loop + routing is still
  Python and contends — partial fix.
