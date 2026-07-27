#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover
    scipy_stats = None


DEFAULT_INPUT = Path("artifacts/verification/experiment2_typed_vs_flat_seeds.json")
DEFAULT_CI_LEVEL = 0.95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Experiment 2 (typed vs flat merge) over shared seeds and "
            "compute a paired CI for the 0%%->60%% degradation-gap reduction."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ci-level", type=float, default=DEFAULT_CI_LEVEL)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def load_input(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "typed" not in data or "flat" not in data:
        raise ValueError("Input JSON must contain top-level 'typed' and 'flat' objects.")
    return data


def ordered_rates(data: dict[str, Any]) -> list[str]:
    shared = set(data["typed"].keys()) & set(data["flat"].keys())
    if not shared:
        raise ValueError("No shared conflict-rate keys between typed and flat.")
    return sorted(shared, key=lambda value: float(str(value).rstrip("%")))


def as_float_list(values: list[Any], label: str) -> list[float]:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{label} must be a non-empty list.")
    return [float(value) for value in values]


def mean_sd(values: list[float]) -> tuple[float, float]:
    mean = sum(values) / len(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd


def ci_for_mean(values: list[float], level: float) -> tuple[float, float]:
    mean, sd = mean_sd(values)
    n = len(values)
    if n < 2 or sd == 0.0:
        return mean, mean
    se = sd / math.sqrt(n)
    alpha = 1.0 - level
    if scipy_stats is not None:
        tcrit = float(scipy_stats.t.ppf(1.0 - alpha / 2.0, n - 1))
    else:
        # Good enough fallback for the intended n=5 case.
        tcrit = 2.776 if abs(level - 0.95) < 1e-9 and n == 5 else 1.96
    half = tcrit * se
    return mean - half, mean + half


def summarize_condition(condition: dict[str, Any], rates: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"per_rate": {}}
    for rate in rates:
        values = as_float_list(condition[rate], rate)
        mean, sd = mean_sd(values)
        summary["per_rate"][rate] = {
            "seeds": values,
            "mean": mean,
            "sd": sd,
        }
    return summary


def paired_degradation_analysis(
    typed: dict[str, Any],
    flat: dict[str, Any],
    rate_0: str,
    rate_60: str,
    ci_level: float,
) -> dict[str, Any]:
    typed_0 = as_float_list(typed[rate_0], f"typed[{rate_0}]")
    typed_60 = as_float_list(typed[rate_60], f"typed[{rate_60}]")
    flat_0 = as_float_list(flat[rate_0], f"flat[{rate_0}]")
    flat_60 = as_float_list(flat[rate_60], f"flat[{rate_60}]")
    lengths = {len(typed_0), len(typed_60), len(flat_0), len(flat_60)}
    if len(lengths) != 1:
        raise ValueError("All 0% and 60% seed lists must have the same length for paired analysis.")

    typed_deg = [a - b for a, b in zip(typed_0, typed_60)]
    flat_deg = [a - b for a, b in zip(flat_0, flat_60)]
    reduction = [b - a for a, b in zip(typed_deg, flat_deg)]
    reduction_mean, reduction_sd = mean_sd(reduction)
    ci_low, ci_high = ci_for_mean(reduction, ci_level)
    typed_mean, typed_sd = mean_sd(typed_deg)
    flat_mean, flat_sd = mean_sd(flat_deg)
    relative_reduction = (reduction_mean / flat_mean) if flat_mean else None

    return {
        "typed_degradation_points": {
            "per_seed": typed_deg,
            "mean": typed_mean,
            "sd": typed_sd,
        },
        "flat_degradation_points": {
            "per_seed": flat_deg,
            "mean": flat_mean,
            "sd": flat_sd,
        },
        "paired_reduction_points": {
            "per_seed": reduction,
            "mean": reduction_mean,
            "sd": reduction_sd,
            "ci_level": ci_level,
            "ci": [ci_low, ci_high],
            "relative_vs_flat": relative_reduction,
        },
    }


def fmt_pct_points(value: float) -> str:
    return f"{value * 100:.1f}"


def fmt_mean_sd(mean: float, sd: float) -> str:
    return f"{fmt_pct_points(mean)} ± {fmt_pct_points(sd)}"


def render_text(report: dict[str, Any], rates: list[str]) -> str:
    typed = report["typed"]
    flat = report["flat"]
    degr = report["degradation_gap"]
    ci_level_pct = int(round(degr["paired_reduction_points"]["ci_level"] * 100))
    ci_low, ci_high = degr["paired_reduction_points"]["ci"]
    relative = degr["paired_reduction_points"]["relative_vs_flat"]

    lines = [
        "Experiment 2 (typed vs flat merge, paired over shared seeds)",
        "",
        "Per-rate mean ± SD (percentage points):",
    ]
    for rate in rates:
        typed_row = typed["per_rate"][rate]
        flat_row = flat["per_rate"][rate]
        lines.append(
            f"- {rate}: typed {fmt_mean_sd(typed_row['mean'], typed_row['sd'])}; "
            f"flat {fmt_mean_sd(flat_row['mean'], flat_row['sd'])}"
        )

    lines.extend(
        [
            "",
            "0% -> 60% degradation:",
            (
                f"- Absolute first: typed degrades by "
                f"{fmt_mean_sd(degr['typed_degradation_points']['mean'], degr['typed_degradation_points']['sd'])} "
                f"vs flat {fmt_mean_sd(degr['flat_degradation_points']['mean'], degr['flat_degradation_points']['sd'])}, "
                f"for a {fmt_pct_points(degr['paired_reduction_points']['mean'])}-point smaller drop with typed merge."
            ),
            (
                f"- Paired {ci_level_pct}% CI on the degradation-gap reduction: "
                f"[{fmt_pct_points(ci_low)}, {fmt_pct_points(ci_high)}] points."
            ),
        ]
    )
    if relative is not None:
        lines.append(
            f"- Relative second: that is a {relative * 100:.1f}% reduction in degradation versus flat merge."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    data = load_input(args.input)
    rates = ordered_rates(data)
    if "0%" not in rates or "60%" not in rates:
        raise ValueError("Input must include both '0%' and '60%' conflict-rate entries.")

    report = {
        "input_file": str(args.input),
        "typed": summarize_condition(data["typed"], rates),
        "flat": summarize_condition(data["flat"], rates),
    }
    report["degradation_gap"] = paired_degradation_analysis(
        typed=data["typed"],
        flat=data["flat"],
        rate_0="0%",
        rate_60="60%",
        ci_level=args.ci_level,
    )

    print(render_text(report, rates))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
