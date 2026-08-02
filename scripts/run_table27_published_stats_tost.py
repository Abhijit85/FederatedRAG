#!/usr/bin/env python3
"""
Reconstruct a paired TOST from the published Table 27 summary statistics.

This is for the case where the paper reports:
- mean +- SD for both arms,
- paired p-value,
- effect size d,
- number of paired seeds,

but the raw per-seed values are unavailable.

The key assumption must be stated explicitly: by default, this script interprets
the published `d` using the reported arm SD scale, so:

    mean_diff = d * reported_sd

That interpretation is consistent with the Table 27 numbers (`p=0.31`, `d=0.2`,
`n=5`) and yields a non-degenerate TOST. If a different effect-size definition
was used in the manuscript, pass `--mean-diff` directly instead of `--d`.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - fallback only
    scipy_stats = None

from scripts.tost_paired import t_crit as fallback_t_crit, t_sf as fallback_t_sf


def t_crit(alpha: float, df: int) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.isf(alpha, df))
    return fallback_t_crit(df, alpha)


def t_sf(value: float, df: int) -> float:
    if scipy_stats is not None:
        return float(scipy_stats.t.sf(value, df))
    return float(fallback_t_sf(value, df))


def reconstruct_two_sided_t_from_p(p_value: float, df: int) -> float:
    if not (0.0 < p_value <= 1.0):
        raise ValueError(f"paired p-value must be in (0, 1], got {p_value}")
    if p_value == 1.0:
        return 0.0
    if scipy_stats is not None:
        return float(scipy_stats.t.isf(p_value / 2.0, df))
    return fallback_t_crit(df, p_value / 2.0)


def pooled_sd(sd_left: float, sd_right: float) -> float:
    return math.sqrt((sd_left * sd_left + sd_right * sd_right) / 2.0)


def mean_diff_from_d(
    *,
    effect_size_d: float,
    sd_left: float,
    sd_right: float,
    d_mode: str,
) -> float:
    if d_mode == "reported_sd":
        if abs(sd_left - sd_right) > 1e-12:
            raise ValueError("reported_sd mode requires equal arm SDs or an explicit --mean-diff")
        return effect_size_d * sd_left
    if d_mode == "pooled_sd":
        return effect_size_d * pooled_sd(sd_left, sd_right)
    raise ValueError(f"Unsupported d_mode: {d_mode}")


def reconstruct_from_published_stats(
    *,
    mean_left: float,
    sd_left: float,
    mean_right: float,
    sd_right: float,
    paired_p: float,
    n_pairs: int,
    effect_size_d: float | None,
    mean_diff: float | None,
    d_mode: str,
    margin: float,
    alpha: float,
) -> dict[str, Any]:
    if n_pairs < 2:
        raise ValueError("n_pairs must be at least 2")
    df = n_pairs - 1

    if mean_diff is None:
        if effect_size_d is None:
            raise ValueError("Need either effect_size_d or mean_diff")
        mean_diff = mean_diff_from_d(
            effect_size_d=effect_size_d,
            sd_left=sd_left,
            sd_right=sd_right,
            d_mode=d_mode,
        )

    t_abs = reconstruct_two_sided_t_from_p(paired_p, df)
    if t_abs == 0.0:
        raise ValueError("paired_p=1.0 does not identify a finite paired SE from summary stats alone")

    se_diff = abs(mean_diff) / t_abs
    sd_diff = se_diff * math.sqrt(n_pairs)

    tcrit = t_crit(alpha, df)
    ci_lo = mean_diff - tcrit * se_diff
    ci_hi = mean_diff + tcrit * se_diff
    t_lower = (mean_diff + margin) / se_diff
    t_upper = (mean_diff - margin) / se_diff
    p_lower = t_sf(t_lower, df)
    p_upper = t_sf(-t_upper, df)
    p_tost = max(p_lower, p_upper)
    equivalent = (p_lower < alpha) and (p_upper < alpha)

    return {
        "inputs": {
            "mean_left": mean_left,
            "sd_left": sd_left,
            "mean_right": mean_right,
            "sd_right": sd_right,
            "paired_p": paired_p,
            "n_pairs": n_pairs,
            "effect_size_d": effect_size_d,
            "mean_diff": mean_diff,
            "d_mode": d_mode,
            "margin": margin,
            "alpha": alpha,
        },
        "reconstruction": {
            "df": df,
            "abs_t_from_p": t_abs,
            "mean_diff": mean_diff,
            "se_diff": se_diff,
            "sd_diff": sd_diff,
        },
        "tost": {
            "mean_diff": mean_diff,
            "sd_diff": sd_diff,
            "se_diff": se_diff,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "p_lower": p_lower,
            "p_upper": p_upper,
            "p_tost": p_tost,
            "equivalent": equivalent,
        },
        "note": (
            "This is a re-analysis of published paired summary statistics, not a rerun from raw per-seed logs. "
            "The result depends on the stated d interpretation."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reconstruct a paired Table 27 TOST from published summary stats.")
    parser.add_argument("--mean-left", type=float, default=0.92)
    parser.add_argument("--sd-left", type=float, default=0.02)
    parser.add_argument("--mean-right", type=float, default=0.92)
    parser.add_argument("--sd-right", type=float, default=0.02)
    parser.add_argument("--paired-p", type=float, default=0.31)
    parser.add_argument("--n-pairs", type=int, default=5)
    parser.add_argument("--d", type=float, default=0.2, help="Published effect size d")
    parser.add_argument("--mean-diff", type=float, default=None, help="Optional direct paired mean difference override")
    parser.add_argument(
        "--d-mode",
        choices=("reported_sd", "pooled_sd"),
        default="reported_sd",
        help="How to convert d into a paired mean difference when --mean-diff is not supplied.",
    )
    parser.add_argument("--margin", type=float, default=0.03)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("artifacts/verification/table27_published_stats_tost/summary.json"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = reconstruct_from_published_stats(
        mean_left=args.mean_left,
        sd_left=args.sd_left,
        mean_right=args.mean_right,
        sd_right=args.sd_right,
        paired_p=args.paired_p,
        n_pairs=args.n_pairs,
        effect_size_d=args.d,
        mean_diff=args.mean_diff,
        d_mode=args.d_mode,
        margin=args.margin,
        alpha=args.alpha,
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    tost = summary["tost"]
    recon = summary["reconstruction"]
    print(
        f"reconstructed: mean_diff={recon["mean_diff"]:+.4f}, se_diff={recon["se_diff"]:.4f}, "
        f"sd_diff={recon["sd_diff"]:.4f}, |t|={recon["abs_t_from_p"]:.4f}"
    )
    print(
        f"tost: margin=±{args.margin:.3f}, 90% CI=[{tost["ci_lo"]:+.4f}, {tost["ci_hi"]:+.4f}], "
        f"p_lower={tost["p_lower"]:.4f}, p_upper={tost["p_upper"]:.4f}, equivalent={tost["equivalent"]}"
    )
    print(f"wrote: {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
