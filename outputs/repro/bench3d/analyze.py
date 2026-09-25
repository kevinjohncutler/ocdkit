"""Summarize bench results (results/*.jsonl) into markdown tables.

Speed is reported as variant GPU median / shipped GPU median per scenario
(<1 = faster). A scenario only counts as a real difference when the variant's
p10..p90 frame-time range does not overlap the shipped shader's range.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"


def load(kind):
    p = R / f"{kind}.jsonl"
    return [json.loads(line) for line in open(p)] if p.exists() else []


# which scenarios each variant can affect (others are reported separately as a
# no-regression check)
RELEVANT = {
    "ea_exp": lambda r: r["mode"] == "EA" and r["cfg"] != "lab",
    "lab_first": lambda r: r["cfg"] in ("imglab", "imglab50"),
    "clip": lambda r: r["cfg"] in ("imglab", "imglab50"),
    "skip4": lambda r: True, "skip8": lambda r: True, "skip16": lambda r: True,
    "lut": lambda r: r["cfg"] != "img",
    "idbuf": lambda r: r["cfg"] != "img",
    "override": lambda r: True,
    "combo8": lambda r: True, "combo16": lambda r: True,
    "skipx8": lambda r: True, "skipx16": lambda r: True,
    "combo16m": lambda r: True, "ported": lambda r: True,
    "combo8x": lambda r: True, "combo16x": lambda r: True,
}


def gmean(xs):
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def verdict(v, b):
    if v["gpu_p90"] < b["gpu_p10"]:
        return "faster"
    if v["gpu_p10"] > b["gpu_p90"]:
        return "slower"
    return "same"


def render_tables(kind="render"):
    rows = load(kind)
    if not rows:
        return
    print(f"## Render: {len(rows)} scenarios, {rows[0]['W']}x{rows[0]['H']}, "
          f"{rows[0]['K']} frames x {rows[0]['ROUNDS']} rounds per variant\n")
    variants = [v for v in rows[0]["variants"] if v != "base"]
    print("| Variant | Scenarios | GPU time vs shipped (geo mean) | Best | Worst | Faster | Same | Slower | Max pixel diff |")
    print("|---|---|---|---|---|---|---|---|---|")
    for v in variants:
        rel = [r for r in rows if RELEVANT[v](r)]
        rel = [r for r in rel if r["variants"][v]["gpu_med"] > 0]
        ratios = [r["variants"][v]["gpu_med"] / r["variants"]["base"]["gpu_med"] for r in rel]
        vd = defaultdict(int)
        for r in rel:
            vd[verdict(r["variants"][v], r["variants"]["base"])] += 1
        mx = max((r["variants"][v]["fid"] or {}).get("maxAbs", 0) for r in rel)
        print(f"| {v} | {len(rel)} | {gmean(ratios):.3f} | {min(ratios):.3f} | {max(ratios):.3f} | "
              f"{vd['faster']} | {vd['same']} | {vd['slower']} | {mx:.3f} |")
    print()
    # per-mode / per-config breakdown for each variant
    for v in variants:
        print(f"### {v}\n")
        print("| Dataset | Config | Mode | Shipped ms | Variant ms | Ratio | Verdict | Pixels changed |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if not RELEVANT[v](r):
                continue
            b, x = r["variants"]["base"], r["variants"][v]
            fid = x["fid"] or {}
            print(f"| {r['ds']} {r['view']} | {r['cfg']} | {r['mode']} | {b['gpu_med']:.2f} | "
                  f"{x['gpu_med']:.2f} | {x['gpu_med'] / b['gpu_med']:.3f} | {verdict(x, b)} | "
                  f"{fid.get('fracGt1of255', 0) * 100:.3f}% |")
        print()


def simple(kind):
    rows = load(kind)
    if rows:
        print(f"## {kind}\n")
        for r in rows:
            print(json.dumps(r))
        print()


if __name__ == "__main__":
    render_tables()
    render_tables("render_combo")
    for k in ["bricks", "pick", "update", "tx"]:
        simple(k)
