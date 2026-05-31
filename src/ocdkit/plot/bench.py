"""Phase-level profiling for ``image_grid`` renders.

Monkey-patches the hot helpers inside ``ocdkit.plot.image_grid`` and
``ocdkit.io.figure_server`` for the duration of a single render and
prints a per-phase breakdown. Use this to (a) verify that the
``opencodecs.jxl.read(downsample=...)`` native path is firing for each
scene, and (b) see where the remaining ms are going (decode vs. PQ
invert vs. SVG emit vs. HTTP register).

Usage::

    from ocdkit.plot.bench import bench_image_grid

    fig = bench_image_grid(
        split_list(scenes, ncol),
        plot_labels=split_list(plot_labels, ncol),
        lpos='top_middle', dpi=dpi, fontcolor='lightgray',
        fontsize=4, figsize=ncol, vmin=0, vmax=1,
    )

    # ↳ prints something like:
    #
    # image_grid: 632.4 ms total  (14 tiles)
    #
    # opencodecs.jxl.read     14 calls   438.2 ms total   31.3 ms mean
    #     downsample=8         12          362.1 ms total   30.2 ms mean
    #     downsample=4          2           76.1 ms total   38.0 ms mean
    # _peek_jxl_size          14 calls    21.5 ms total    1.5 ms mean
    # _pq_uint16_to_p3_linear 14 calls    98.7 ms total    7.0 ms mean
    # _resize_nearest         14 calls    33.2 ms total    2.4 ms mean
    # register (hires URL)    14 calls     5.1 ms total    0.4 ms mean
    #
    # SVG emit + other:       35.7 ms
"""
from __future__ import annotations

import time
from contextlib import contextmanager


@contextmanager
def profile_image_grid():
    """Context manager that records timings of the hot phases."""
    import importlib
    import opencodecs.jxl
    from ..io import figure_server
    # ocdkit.plot re-exports the `image_grid` function from the
    # submodule of the same name (via ``from .image_grid import
    # image_grid`` in __init__.py), which shadows the submodule
    # attribute on the package. Use importlib to grab the actual
    # module so we can monkey-patch its helpers.
    ig_module = importlib.import_module("ocdkit.plot.image_grid")
    svg_module = importlib.import_module("ocdkit.plot.svg")

    logs: dict[str, list] = {
        "jxl_read": [],
        "peek": [],
        "pq_invert": [],
        "resize": [],
        "register": [],
        "encode": [],
    }
    saved = {
        "jxl_read": opencodecs.jxl.read,
        "peek": figure_server._peek_jxl_size,
        "resize": ig_module._resize_nearest,
        "pq_invert": svg_module._pq_uint16_to_p3_linear,
        "register": figure_server.register,
        "encode": ig_module._encode_thumb_url,
    }

    def _trace(name, orig, extra_fn=None):
        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            r = orig(*a, **kw)
            entry = {"dt_ms": (time.perf_counter() - t0) * 1000.0}
            if extra_fn is not None:
                try:
                    entry.update(extra_fn(a, kw, r))
                except Exception:
                    pass
            logs[name].append(entry)
            return r
        return wrapped

    def _read_extra(a, kw, r):
        return {
            "downsample": kw.get("downsample", 1),
            "shape": tuple(r.shape),
            "dtype": str(r.dtype),
        }

    def _peek_extra(a, kw, r):
        return {"size": r}

    def _resize_extra(a, kw, r):
        return {"src": a[0].shape[:2], "dst": (a[1], a[2])}

    def _pq_extra(a, kw, r):
        return {"shape": a[0].shape}

    opencodecs.jxl.read = _trace("jxl_read", saved["jxl_read"], _read_extra)
    figure_server._peek_jxl_size = _trace("peek", saved["peek"], _peek_extra)
    ig_module._resize_nearest = _trace("resize", saved["resize"], _resize_extra)
    svg_module._pq_uint16_to_p3_linear = _trace(
        "pq_invert", saved["pq_invert"], _pq_extra)
    figure_server.register = _trace("register", saved["register"])
    ig_module._encode_thumb_url = _trace("encode", saved["encode"])

    t_total = time.perf_counter()
    try:
        yield logs
    finally:
        total_ms = (time.perf_counter() - t_total) * 1000.0
        # Restore originals before we print (so the printer's own
        # imports don't accidentally trip the patched code).
        opencodecs.jxl.read = saved["jxl_read"]
        figure_server._peek_jxl_size = saved["peek"]
        ig_module._resize_nearest = saved["resize"]
        svg_module._pq_uint16_to_p3_linear = saved["pq_invert"]
        figure_server.register = saved["register"]
        ig_module._encode_thumb_url = saved["encode"]
        logs["__total_ms__"] = total_ms
        _print_summary(logs)


def _phase_total(entries):
    return sum(e["dt_ms"] for e in entries)


def _print_summary(logs):
    total = logs["__total_ms__"]
    n_tiles = len(logs["resize"]) or len(logs["jxl_read"]) or 0
    print(f"image_grid: {total:.1f} ms total  ({n_tiles} tile(s))")
    print()

    # opencodecs.jxl.read — broken down by downsample ratio.
    rd = logs["jxl_read"]
    if rd:
        by_ds: dict[int, list[dict]] = {}
        for e in rd:
            by_ds.setdefault(e["downsample"], []).append(e)
        tot = _phase_total(rd)
        print(
            f"{'opencodecs.jxl.read':<24} "
            f"{len(rd):>3} calls  {tot:>7.1f} ms total  "
            f"{tot/len(rd):>5.1f} ms mean"
        )
        for ds in sorted(by_ds.keys()):
            es = by_ds[ds]
            t = _phase_total(es)
            shapes = {e["shape"] for e in es}
            shape_str = (str(next(iter(shapes))) if len(shapes) == 1
                         else f"{len(shapes)} distinct shapes")
            print(
                f"  {'downsample=' + str(ds):<22} "
                f"{len(es):>3}        {t:>7.1f} ms        "
                f"{t/len(es):>5.1f} ms       → {shape_str}"
            )

    # Other instrumented phases.
    for name, label in (
        ("peek", "_peek_jxl_size"),
        ("pq_invert", "_pq_uint16_to_p3_linear"),
        ("resize", "_resize_nearest"),
        ("register", "register (hires URL)"),
        ("encode", "_encode_thumb_url"),
    ):
        es = logs[name]
        if not es:
            continue
        t = _phase_total(es)
        print(
            f"{label:<24} "
            f"{len(es):>3} calls  {t:>7.1f} ms total  "
            f"{t/len(es):>5.1f} ms mean"
        )

    accounted = sum(_phase_total(logs[k])
                    for k in ("jxl_read", "peek", "pq_invert",
                              "resize", "register", "encode"))
    print()
    print(f"SVG emit + other:        {total - accounted:>7.1f} ms")

    # Verify the native path is firing.
    ds_used = [e["downsample"] for e in rd]
    native = sum(1 for d in ds_used if d > 1)
    full = sum(1 for d in ds_used if d == 1)
    print()
    print(
        f"opencodecs downsample API: "
        f"{native} tile(s) used downsample>1 (native progressive), "
        f"{full} tile(s) used downsample=1 (full decode)"
    )


def bench_image_grid(items, **kwargs):
    """Thin wrapper: call ``image_grid(items, **kwargs)`` under
    :func:`profile_image_grid` and print the breakdown.

    Returns the resulting :class:`SvgFigure` unchanged.
    """
    from .image_grid import image_grid
    with profile_image_grid():
        return image_grid(items, **kwargs)


__all__ = ["profile_image_grid", "bench_image_grid"]
