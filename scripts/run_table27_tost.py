#!/usr/bin/env python3
"""
Run the Table 27 paired TOST equivalence check with an explicit sanity gate.

This wrapper exists to make the ZkZQ-W4 / §D.2 workflow hard to misuse:

1. Load paired per-seed accuracies for SYNAPSE and Centralized-SYNAPSE.
2. Confirm they reproduce the intended Table 27 run closely enough.
3. Report a paired null-hypothesis test p-value for the same seed pairing.
4. Run the paired TOST at a pre-specified margin only after the sanity gate.

Accepted inputs:
- plain text / CSV file with one accuracy per line
- summary.json containing {"per_seed_accuracy": {"1": 0.91, ...}}
- directory containing per-seed JSON files with an "accuracy" field

Examples:
    python scripts/run_table27_tost.py \
      --syn artifacts/.../summary.json \
      --cen artifacts/.../summary.json \
      --margin 0.03

    python scripts/run_table27_tost.py \
      --syn syn.csv --cen cen.csv --margin 0.03 \
      --anchor-syn-mean 0.92 --anchor-syn-sd 0.02 \
      --anchor-cen-mean 0.92 --anchor-cen-sd 0.02 \
      --anchor-p 0.31
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path
from typing import Any

from tost_paired import t_crit, t_sf


def read_scalar_lines(path: Path, col: str | None) -> list[float]:
    values: list[float] = []
    with path.open() as handle:
        if path.suffix.lower() == ".csv" and col is not None:
            for row in csv.DictReader(handle):
                values.append(float(row[col]))
            return values
        for line in handle:
            line = line.strip()
            if not line:
                continue
            token = line.split(",")[-1]
            try:
                values.append(float(token))
            except ValueError:
                continue
    return values


def ordered_seed_values(mapping: dict[str, Any]) -> list[float]:
    pairs: list[tuple[int, float]] = []
    for seed, value in mapping.items():
        try:
            pairs.append((int(seed), float(value)))
        except (TypeError, ValueError):
            continue
    return [value for _, value in sorted(pairs)]


def read_summary_json(path: Path) -> list[float] | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("per_seed_accuracy"), dict):
        return ordered_seed_values(payload["per_seed_accuracy"])
    return None


def read_seed_dir(path: Path) -> list[float] | None:
    files = sorted(path.glob("*seed_*.json"))
    if not files:
        return None
    pairs: list[tuple[int, float]] = []
    for item in files:
        try:
            payload = json.loads(item.read_text())
        except Exception:
            continue
        seed = payload.get("seed")
        acc = payload.get("accuracy")
        try:
            pairs.append((int(seed), float(acc)))
        except (TypeError, ValueError):
            continue
    if not pairs:
        return None
    return [value for _, value in sorted(pairs)]


def load_values(spec: str, col: str | None) -> list[float]:
    path = Path(spec)
    if path.is_dir():
        values = read_seed_dir(path)
        if values is None:
            raise ValueError(f"{path} does not contain readable per-seed JSON files")
        return values
    if not path.exists():
        raise FileNotFoundError(path)
    summary_values = read_summary_json(path)
    if summary_values is not None:
        return summary_values
    values = read_scalar_lines(path, col)
    if not values:
        raise ValueError(f"{path} did not yield any numeric accuracy values")
    return values


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean_value = sum(values) / len(values)
    sd_value = st.stdev(values) if len(values) > 1 else 0.0
    return mean_value, sd_value


def paired_t_test(x: list[float], y: list[float]) -> dict[str, float]:
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    mean_diff, sd_diff = mean_sd(diffs)
    se = sd_diff / math.sqrt(n)
    if se == 0.0:
        t_value = 0.0 if abs(mean_diff) < 1e-12 else math.copysign(math.inf, mean_diff)
        p_value = 1.0 if abs(mean_diff) < 1e-12 else 0.0
    else:
        t_value = mean_diff / se
        p_value = min(1.0, 2.0 * t_sf(abs(t_value), n - 1))
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "se_diff": se,
        "t": t_value,
        "p": p_value,
    }


def paired_tost(x: list[float], y: list[float], margin: float, alpha: float) -> dict[str, float | bool]:
    diffs = [a - b for a, b in zip(x, y)]
    n = len(diffs)
    mean_diff, sd_diff = mean_sd(diffs)
    se = sd_diff / math.sqrt(n)
    if se == 0.0:
        inside = (-margin <= mean_diff <= margin)
        return {
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "se_diff": se,
            "ci_lo": mean_diff,
            "ci_hi": mean_diff,
            "p_lower": 0.0 if mean_diff > -margin else 1.0,
            "p_upper": 0.0 if mean_diff < margin else 1.0,
            "p_tost": 0.0 if inside else 1.0,
            "equivalent": inside,
        }
    t_lower = (mean_diff + margin) / se
    t_upper = (mean_diff - margin) / se
    p_lower = t_sf(t_lower, n - 1)
    p_upper = t_sf(-t_upper, n - 1)
    crit = t_crit(n - 1, alpha)
    ci_lo = mean_diff - crit * se
    ci_hi = mean_diff + crit * se
    equivalent = (p_lower < alpha) and (p_upper < alpha)
    return {
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "se_diff": se,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": max(p_lower, p_upper),
        "equivalent": equivalent,
    }


def within_tolerance(value: float, anchor: float | None, tol: float) -> bool:
    if anchor is None:
        return True
    return abs(value - anchor) <= tol


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanity-gated paired TOST for Table 27.")
    parser.add_argument("--syn", required=True, help="SYNAPSE vector file, summary JSON, or seed directory")
    parser.add_argument("--cen", required=True, help="Centralized-SYNAPSE vector file, summary JSON, or seed directory")
    parser.add_argument("--margin", type=float, required=True, help="Equivalence margin, e.g. 0.03")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--col", default=None, help="CSV column name when passing CSV inputs")
    parser.add_argument("--anchor-syn-mean", type=float, default=0.92)
    parser.add_argument("--anchor-syn-sd", type=float, default=0.02)
    parser.add_argument("--anchor-cen-mean", type=float, default=0.92)
    parser.add_argument("--anchor-cen-sd", type=float, default=0.02)
    parser.add_argument("--anchor-p", type=float, default=0.31, help="Expected paired null-test p-value")
    parser.add_argument("--mean-tol", type=float, default=0.015)
    parser.add_argument("--sd-tol", type=float, default=0.015)
    parser.add_argument("--p-tol", type=float, default=0.10)
    parser.add_argument("--allow-san-gate-fail", action="store_true", help="Still run TOST even if the sanity gate fails")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    syn = load_values(args.syn, args.col)
    cen = load_values(args.cen, args.col)
    if len(syn) != len(cen) or len(syn) < 2:
        raise SystemExit("Need equal-length paired seed vectors with at least two seeds.")

    syn_mean, syn_sd = mean_sd(syn)
    cen_mean, cen_sd = mean_sd(cen)
    paired = paired_t_test(syn, cen)

    sanity_checks = {
        "syn_mean": within_tolerance(syn_mean, args.anchor_syn_mean, args.mean_tol),
        "syn_sd": within_tolerance(syn_sd, args.anchor_syn_sd, args.sd_tol),
        "cen_mean": within_tolerance(cen_mean, args.anchor_cen_mean, args.mean_tol),
        "cen_sd": within_tolerance(cen_sd, args.anchor_cen_sd, args.sd_tol),
        "paired_p": within_tolerance(float(paired["p"]), args.anchor_p, args.p_tol),
    }
    sanity_pass = all(sanity_checks.values())

    print("=== Table 27 sanity gate ===")
    print(f"SYNAPSE seeds:      {', '.join(f'{v:.3f}' for v in syn)}")
    print(f"Centralized seeds:  {', '.join(f'{v:.3f}' for v in cen)}")
    print(f"SYNAPSE mean ± SD:      {syn_mean:.3f} ± {syn_sd:.3f}")
    print(f"Centralized mean ± SD:  {cen_mean:.3f} ± {cen_sd:.3f}")
    print(
        "paired null test: "
        f"mean diff={float(paired['mean_diff']):+.4f}, "
        f"t={float(paired['t']):+.4f}, p={float(paired['p']):.4f}"
    )
    print(
        "anchors: "
        f"SYN={args.anchor_syn_mean:.3f}±{args.anchor_syn_sd:.3f}, "
        f"CEN={args.anchor_cen_mean:.3f}±{args.anchor_cen_sd:.3f}, "
        f"paired p≈{args.anchor_p:.3f}"
    )
    print(
        "sanity checks: "
        + ", ".join(f"{name}={'PASS' if ok else 'FAIL'}" for name, ok in sanity_checks.items())
    )
    print(f"SANITY GATE: {'PASS' if sanity_pass else 'FAIL'}")

    if not sanity_pass and not args.allow_san_gate_fail:
        print()
        print("Refusing to report TOST because these vectors do not reproduce the intended Table 27 run.")
        print("Use the actual per-seed Table 27 logs, or rerun with the correct harness, then try again.")
        return 2

    result = paired_tost(syn, cen, args.margin, args.alpha)
    ci_level = int((1 - 2 * args.alpha) * 100)
    print()
    print("=== Paired TOST ===")
    print(
        f"n={len(syn)}  mean paired diff (SYN-CEN) = {float(result['mean_diff']):+.4f}  "
        f"sd={float(result['sd_diff']):.4f}  se={float(result['se_diff']):.4f}"
    )
    print(f"margin = ±{args.margin:.4f}")
    print(f"{ci_level}% CI (paired): [{float(result['ci_lo']):+.4f}, {float(result['ci_hi']):+.4f}]")
    print(
        f"one-sided p (> -m): {float(result['p_lower']):.4f}   "
        f"one-sided p (< +m): {float(result['p_upper']):.4f}   "
        f"TOST p = {float(result['p_tost']):.4f}"
    )
    if bool(result["equivalent"]):
        print(f"RESULT: EQUIVALENT within ±{args.margin:.3f} (paired TOST, alpha={args.alpha:.2f})")
    else:
        print("RESULT: NOT established — CI not fully inside the margin")
        print(
            f"  -> report: 'no significant difference at n={len(syn)}; "
            f"equivalence not established at margin {args.margin:.3f}.'"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
