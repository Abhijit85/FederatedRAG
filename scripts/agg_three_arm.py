#!/usr/bin/env python3
"""
agg_three_arm.py — aggregate the three-arm typing-isolation run (Result 5) into the
rebuttal table plus the two paired isolation contrasts.

Expects one JSON per run: {"arm": "full"|"structured_untyped"|"typed_generic",
                           "rate": 0|20|40|60, "seed": 1, "score": 0.986}
Usage:
    python agg_three_arm.py runs_threearm/*.json
    python agg_three_arm.py runs_threearm/*.json --md
"""
import json
import glob
import math
import argparse
import statistics as st
from collections import defaultdict

ARMS = ["full", "structured_untyped", "typed_generic"]
LABEL = {"full": "Full SYNAPSE", "structured_untyped": "Structured-untyped",
         "typed_generic": "Typed-generic-merge"}


def load(paths):
    rows = []
    for p in paths:
        for g in glob.glob(p):
            rows.append(json.load(open(g)))
    return rows


def ms(xs):
    m = sum(xs) / len(xs)
    sd = st.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd


def paired(a, b):
    """paired mean diff (a-b), sd, t, n, status over common seeds.
    status: 'ok'         -> normal (nonzero within-pair variance)
            'identical'  -> every seed identical, Δ=0  (NO effect; t is n/a, never significant)
            'degenerate' -> every seed identical, Δ!=0 (perfectly consistent, but t is UNDEFINED;
                            evaluate with a sign/exact test, not a t-test)"""
    common = sorted(set(a) & set(b))
    d = [a[s] - b[s] for s in common]
    if not d:
        return None
    m, sd = ms(d)
    n = len(d)
    if sd == 0:
        return m, sd, None, n, ("identical" if abs(m) < 1e-12 else "degenerate")
    return m, sd, m / (sd / math.sqrt(n)), n, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--md", action="store_true")
    args = ap.parse_args()
    rows = load(args.paths)
    cell = defaultdict(dict)                       # (arm,rate) -> {seed: score}
    for r in rows:
        cell[(r["arm"], r["rate"])][r["seed"]] = r["score"]
    rates = sorted({r["rate"] for r in rows})

    print("\n=== per-arm means (mean ± SD) ===")
    for arm in ARMS:
        line = f"  {LABEL[arm]:22s} " + "  ".join(
            f"{rate}%={ms(list(cell[(arm,rate)].values()))[0]:.3f}±{ms(list(cell[(arm,rate)].values()))[1]:.3f}"
            for rate in rates if cell[(arm,rate)])
        print(line)

    tcrit = 2.776  # df=4, two-sided 0.05
    for other, what in [("structured_untyped", "TYPE DECLARATIONS"),
                        ("typed_generic", "CONFLICT LOG")]:
        print(f"\n=== full - {other}  (isolates the {what}) ===")
        for rate in rates:
            pr = paired(cell[("full", rate)], cell[(other, rate)])
            if not pr:
                continue
            m, sd, t, n, status = pr
            if status == "ok":
                flag = "*" if abs(t) > tcrit else " "
                ts = f"t={t:6.2f}"
            elif status == "identical":
                flag = " "
                ts = "t=n/a  (Δ=0 — no difference; a zero-variance row is never significant)"
            else:  # degenerate: zero within-pair variance, nonzero Δ
                flag = "!"
                ts = "t=undef (all seeds identical Δ — report via a sign/exact test, NOT a t-test)"
            print(f"  {rate:>2}%  Δ={m:+.3f} ± {sd:.3f}  {ts} (df={n-1}) {flag}")
        print(f"       * = |t|>{tcrit} significant at 0.05 (df=4);  ! = zero within-pair variance, t undefined")

    if args.md:
        print("\n=== markdown (Result 5 table) ===")
        hdr = "| Arm | " + " | ".join(f"{r}%" for r in rates) + " |"
        print(hdr)
        print("|" + "---|" * (len(rates) + 1))
        for arm in ARMS:
            cells = []
            for rate in rates:
                xs = list(cell[(arm,rate)].values())
                cells.append(f"{ms(xs)[0]:.3f} ± {ms(xs)[1]:.3f}" if xs else "—")
            print(f"| {LABEL[arm]} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
