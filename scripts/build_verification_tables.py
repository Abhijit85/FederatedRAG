#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from estimate_seed_sd import estimate_sd, parse_exp1_groups, parse_eval_history


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def stdev(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def fmt_number(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "[calc]"
    return f"{value:.{digits}f}"


def fmt_seed(value: float | None) -> str:
    if value is None:
        return "[real]"
    return f"{value:.3f}"


def student_t_cdf(x: float, df: int) -> float:
    try:
        from scipy import stats  # type: ignore
        return float(stats.t.cdf(x, df))
    except Exception:
        raise RuntimeError("scipy is required for TOST output; install it in the project venv")


def student_t_interval(level: float, df: int, loc: float, scale: float) -> tuple[float, float]:
    try:
        from scipy import stats  # type: ignore
        low, high = stats.t.interval(level, df, loc=loc, scale=scale)
        return float(low), float(high)
    except Exception:
        raise RuntimeError("scipy is required for TOST output; install it in the project venv")


def tost_paired(synapse: list[float], centralized: list[float], margin: float = 0.03, alpha: float = 0.05) -> dict[str, Any]:
    diffs = [x - y for x, y in zip(synapse, centralized)]
    n = len(diffs)
    if n < 2:
        return {"mean_diff": None, "ci_90": (None, None), "p_lower": None, "p_upper": None, "p_tost": None, "equivalent": None}
    mean_diff = mean(diffs)
    sd = stdev(diffs)
    if mean_diff is None or sd is None:
        return {"mean_diff": None, "ci_90": (None, None), "p_lower": None, "p_upper": None, "p_tost": None, "equivalent": None}
    se = sd / math.sqrt(n)
    df = n - 1
    t_lower = (mean_diff - (-margin)) / se
    t_upper = (mean_diff - margin) / se
    p_lower = 1 - student_t_cdf(t_lower, df)
    p_upper = student_t_cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)
    ci_90 = student_t_interval(1 - 2 * alpha, df, loc=mean_diff, scale=se)
    return {
        "mean_diff": mean_diff,
        "ci_90": ci_90,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "equivalent": p_tost < alpha,
    }


def build_priors(include_eval_history: bool) -> list[Any]:
    priors = parse_exp1_groups()
    if include_eval_history:
        priors.extend(parse_eval_history())
    return priors


def estimate_columns(row: dict[str, Any], priors: list[Any]) -> tuple[str, str]:
    query_count = row.get("query_count")
    anchor = row.get("paper_anchor")
    sd_anchor = row.get("sd_mean_anchor", anchor)
    if not isinstance(query_count, int) or query_count <= 0 or not isinstance(sd_anchor, (int, float)):
        return "n/a", "n/a"
    sd_anchor = float(sd_anchor)
    if not (0.0 <= sd_anchor <= 1.0):
        return "n/a", "n/a"
    est = estimate_sd(priors, target_mean=sd_anchor, query_count=query_count)
    return fmt_number(est.recommended_sd), fmt_number(est.conservative_sd)


def render_seed_row(label: str, paper_anchor: str, seeds: list[float | None], expected_sd: str, conservative_sd: str, extra: list[str] | None = None) -> str:
    vals = [v for v in seeds if v is not None]
    row = [label, paper_anchor] + [fmt_seed(v) for v in seeds] + [fmt_number(mean(vals)), fmt_number(stdev(vals)), expected_sd, conservative_sd]
    if extra:
        row.extend(extra)
    return "| " + " | ".join(row) + " |"


def render_core_table(data: dict[str, Any], priors: list[Any]) -> str:
    out = ["### Table A. Core Reproduction Checkpoints", "", "| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | SD | Expected SD | Conservative SD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["core_reproduction"]:
        expected_sd, conservative_sd = estimate_columns(row, priors)
        out.append(render_seed_row(row["checkpoint"], str(row["paper_anchor"]), row.get("seeds", [None] * 5), expected_sd, conservative_sd))
    return "\n".join(out)


def render_privacy_table(data: dict[str, Any], priors: list[Any]) -> str:
    out = ["### Table B. Privacy–Utility Validation Points From Table 9", "", "| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean routing acc. | SD routing acc. | Expected SD | Conservative SD | Mean AUROC | Mean % clients < 0.10 |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["privacy_utility"]:
        seeds = row.get("seeds", [None] * 5)
        vals = [v for v in seeds if v is not None]
        expected_sd, conservative_sd = estimate_columns(row, priors)
        columns = [
            row["checkpoint"], str(row["paper_anchor"]), *[fmt_seed(v) for v in seeds], fmt_number(mean(vals)), fmt_number(stdev(vals)), expected_sd, conservative_sd, fmt_number(row.get("mean_auroc")), fmt_number(row.get("mean_pct_lt_point1"))
        ]
        out.append("| " + " | ".join(columns) + " |")
    return "\n".join(out)


def render_toolbench_table(data: dict[str, Any], priors: list[Any]) -> str:
    out = ["### Table C. ToolBench / mmFG-W2 Extension Checkpoint", "", "| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Mean | SD | Expected SD | Conservative SD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["toolbench"]:
        seeds = row.get("seeds", [None] * 3)
        vals = [v for v in seeds if v is not None]
        expected_sd, conservative_sd = estimate_columns(row, priors)
        out.append("| " + " | ".join([row["checkpoint"], str(row["paper_anchor"]), *[fmt_seed(v) for v in seeds], fmt_number(mean(vals)), fmt_number(stdev(vals)), expected_sd, conservative_sd]) + " |")
    return "\n".join(out)


def render_cross_model_table(data: dict[str, Any], priors: list[Any]) -> str:
    out = ["### Table D. Cross-Model / Root-Cause Checkpoints From Tables 22–23", "", "| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Mean | SD | Expected SD | Conservative SD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["cross_model"]:
        seeds = row.get("seeds", [None] * 3)
        vals = [v for v in seeds if v is not None]
        expected_sd, conservative_sd = estimate_columns(row, priors)
        out.append("| " + " | ".join([row["checkpoint"], str(row["paper_anchor"]), *[fmt_seed(v) for v in seeds], fmt_number(mean(vals)), fmt_number(stdev(vals)), expected_sd, conservative_sd]) + " |")
    return "\n".join(out)


def render_equivalence_table(data: dict[str, Any], priors: list[Any]) -> str:
    out = ["### Table E. Controls / Equivalence Checkpoints", "", "| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | SD | Expected SD | Conservative SD |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in data["controls_equivalence"]:
        seeds = row.get("seeds", [None] * 5)
        expected_sd, conservative_sd = estimate_columns(row, priors)
        out.append(render_seed_row(row["checkpoint"], str(row["paper_anchor"]), seeds, expected_sd, conservative_sd))
    pair = data.get("paired_tost", {})
    synapse = pair.get("synapse_seeds", [None] * 5)
    centralized = pair.get("centralized_seeds", [None] * 5)
    if len(synapse) == len(centralized):
        ready = all(v is not None for v in synapse + centralized)
        result = tost_paired(synapse, centralized, margin=pair.get("margin", 0.03), alpha=pair.get("alpha", 0.05)) if ready else {"mean_diff": None, "ci_90": (None, None), "p_lower": None, "p_upper": None, "p_tost": None, "equivalent": None}
        out.extend([
            "",
            "| Seed | SYNAPSE acc. | Centralized-SYNAPSE acc. | Paired diff |",
            "| --- | ---: | ---: | ---: |",
        ])
        for idx, (s, c) in enumerate(zip(synapse, centralized), start=1):
            diff = None if s is None or c is None else s - c
            out.append(f"| {idx} | {fmt_seed(s)} | {fmt_seed(c)} | {fmt_seed(diff)} |")
        lo, hi = result["ci_90"]
        out.extend([
            "",
            "| Quantity | Value |",
            "| --- | ---: |",
            f"| Mean paired difference | {fmt_number(result['mean_diff'])} |",
            f"| 90% CI lower | {fmt_number(lo)} |",
            f"| 90% CI upper | {fmt_number(hi)} |",
            f"| One-sided p-value: lower test | {fmt_number(result['p_lower'])} |",
            f"| One-sided p-value: upper test | {fmt_number(result['p_upper'])} |",
            f"| TOST p-value | {fmt_number(result['p_tost'])} |",
            f"| Margin | {pair.get('margin', 0.03):.3f} |",
            f"| Equivalent at alpha={pair.get('alpha', 0.05):.2f} | {result['equivalent'] if result['equivalent'] is not None else '[calc]'} |",
        ])
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build markdown verification tables from actual per-seed values.")
    parser.add_argument("--input", default="verification_seed_inputs.json", help="JSON input file with paper anchors and per-seed values.")
    parser.add_argument("--output", default="artifacts/rebuttal/verification_tables.md", help="Markdown output path.")
    parser.add_argument("--include-eval-history", action="store_true", help="Also use repeated snapshots from evaluation_log_mixed_queries.json.txt when estimating SD.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    input_path = root / args.input
    output_path = root / args.output
    data = json.loads(input_path.read_text())
    priors = build_priors(args.include_eval_history)

    sections = [
        render_core_table(data, priors), "", render_privacy_table(data, priors), "", render_toolbench_table(data, priors), "", render_cross_model_table(data, priors), "", render_equivalence_table(data, priors)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(sections) + "\n")
    print(f"Wrote {output_path.relative_to(root)}")


if __name__ == "__main__":
    main()
