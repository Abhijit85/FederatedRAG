#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent.parent
EXP1_GLOB = "logs/exp1_*.txt"
EVAL_LOG = ROOT / "evaluation_log_mixed_queries.json.txt"


@dataclass
class PriorGroup:
    source: str
    label: str
    query_count: int
    seed_values: list[float]
    mean: float
    sd: float
    binomial_sd: float
    overdispersion_ratio: float


@dataclass
class Estimate:
    target_mean: float
    query_count: int
    historical_group_count: int
    median_prior_sd: float | None
    median_overdispersion_ratio: float | None
    binomial_sd: float
    recommended_sd: float | None
    conservative_sd: float | None


def binomial_sd(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, p))
    return math.sqrt(p * (1.0 - p) / n)


def parse_exp1_groups() -> list[PriorGroup]:
    groups: dict[str, list[tuple[int, float]]] = {}
    for path in ROOT.glob(EXP1_GLOB):
        text = path.read_text(errors="ignore")
        match = re.search(r"Overall Accuracy:\s*(\d+)/(\d+)\s*\(([0-9.]+)%\)", text)
        if not match:
            continue
        total = int(match.group(2))
        acc = float(match.group(3)) / 100.0
        label = re.sub(r"_seed\d+\.txt$", "", path.name)
        groups.setdefault(label, []).append((total, acc))

    priors: list[PriorGroup] = []
    for label, rows in sorted(groups.items()):
        if len(rows) < 2:
            continue
        counts = {total for total, _ in rows}
        if len(counts) != 1:
            continue
        query_count = counts.pop()
        seed_values = sorted(acc for _, acc in rows)
        mean_val = statistics.mean(seed_values)
        sd_val = statistics.stdev(seed_values)
        bino = binomial_sd(mean_val, query_count)
        ratio = (sd_val / bino) if bino > 0 else 1.0
        priors.append(
            PriorGroup(
                source="exp1_logs",
                label=label,
                query_count=query_count,
                seed_values=seed_values,
                mean=mean_val,
                sd=sd_val,
                binomial_sd=bino,
                overdispersion_ratio=ratio,
            )
        )
    return priors


def parse_eval_history() -> list[PriorGroup]:
    if not EVAL_LOG.exists():
        return []
    rows = []
    for line in EVAL_LOG.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        overall = (((payload.get("central") or {}).get("overall") or {}))
        acc = overall.get("accuracy")
        total = overall.get("total")
        if isinstance(acc, (int, float)) and isinstance(total, int) and total > 0:
            rows.append((payload.get("dataset", "unknown"), total, float(acc)))
    grouped: dict[tuple[str, int], list[float]] = {}
    for dataset, total, acc in rows:
        grouped.setdefault((dataset, total), []).append(acc)

    priors: list[PriorGroup] = []
    for (dataset, total), values in sorted(grouped.items()):
        if len(values) < 5:
            continue
        mean_val = statistics.mean(values)
        sd_val = statistics.stdev(values)
        bino = binomial_sd(mean_val, total)
        ratio = (sd_val / bino) if bino > 0 else 1.0
        priors.append(
            PriorGroup(
                source="evaluation_log_history",
                label=dataset,
                query_count=total,
                seed_values=values,
                mean=mean_val,
                sd=sd_val,
                binomial_sd=bino,
                overdispersion_ratio=ratio,
            )
        )
    return priors


def select_comparable_priors(priors: Iterable[PriorGroup], query_count: int, tolerance_ratio: float = 0.5) -> list[PriorGroup]:
    selected = []
    lower = max(1, int(query_count * (1.0 - tolerance_ratio)))
    upper = int(query_count * (1.0 + tolerance_ratio))
    for prior in priors:
        if lower <= prior.query_count <= upper:
            selected.append(prior)
    return selected


def estimate_sd(priors: list[PriorGroup], target_mean: float, query_count: int) -> Estimate:
    comparable = select_comparable_priors(priors, query_count)
    pool = comparable or priors
    if not pool:
        bino = binomial_sd(target_mean, query_count)
        return Estimate(
            target_mean=target_mean,
            query_count=query_count,
            historical_group_count=0,
            median_prior_sd=None,
            median_overdispersion_ratio=None,
            binomial_sd=bino,
            recommended_sd=bino,
            conservative_sd=bino * 1.25,
        )

    median_prior_sd = statistics.median(group.sd for group in pool)
    median_ratio = statistics.median(group.overdispersion_ratio for group in pool)
    bino = binomial_sd(target_mean, query_count)
    recommended = max(bino, bino * median_ratio)
    if comparable:
        conservative = max(recommended, median_prior_sd)
    else:
        conservative = recommended * 1.5
    return Estimate(
        target_mean=target_mean,
        query_count=query_count,
        historical_group_count=len(pool),
        median_prior_sd=median_prior_sd,
        median_overdispersion_ratio=median_ratio,
        binomial_sd=bino,
        recommended_sd=recommended,
        conservative_sd=conservative,
    )


def print_prior_summary(priors: list[PriorGroup]) -> None:
    print("Historical prior groups used for SD calibration")
    print("=")
    for group in priors:
        print(
            f"- {group.source}:{group.label} | n={group.query_count} | "
            f"mean={group.mean:.3f} | sd={group.sd:.3f} | "
            f"binomial_sd={group.binomial_sd:.3f} | ratio={group.overdispersion_ratio:.3f} | "
            f"seeds={','.join(f'{v:.3f}' for v in group.seed_values)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate a plausible seed-level SD from prior repo runs.")
    parser.add_argument("--mean", type=float, required=True, help="Target mean accuracy/proportion for the new run.")
    parser.add_argument("--query-count", type=int, required=True, help="Number of evaluation items per seed.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument("--include-eval-history", action="store_true", help="Also use repeated snapshots from evaluation_log_mixed_queries.json.txt.")
    parser.add_argument("--show-priors", action="store_true", help="Print the historical prior groups used for calibration.")
    args = parser.parse_args()

    priors = parse_exp1_groups()
    if args.include_eval_history:
        priors.extend(parse_eval_history())

    estimate = estimate_sd(priors, target_mean=args.mean, query_count=args.query_count)

    if args.show_priors:
        print_prior_summary(priors)
        print()

    if args.json:
        payload = {
            "estimate": asdict(estimate),
            "priors": [asdict(p) for p in priors],
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"Target mean: {estimate.target_mean:.3f}")
    print(f"Query count per seed: {estimate.query_count}")
    print(f"Historical prior groups: {estimate.historical_group_count}")
    print(f"Binomial SD floor: {estimate.binomial_sd:.4f}")
    if estimate.median_prior_sd is not None:
        print(f"Median historical SD: {estimate.median_prior_sd:.4f}")
    if estimate.median_overdispersion_ratio is not None:
        print(f"Median overdispersion ratio vs binomial: {estimate.median_overdispersion_ratio:.4f}")
    print(f"Recommended SD: {estimate.recommended_sd:.4f}")
    print(f"Conservative SD: {estimate.conservative_sd:.4f}")
    print()
    print("Interpretation")
    print("- Recommended SD uses repo-derived overdispersion relative to plain binomial noise.")
    print("- Conservative SD is the safer upper-side checkpoint for hand-checking whether new seed spread looks plausible.")


if __name__ == "__main__":
    main()
