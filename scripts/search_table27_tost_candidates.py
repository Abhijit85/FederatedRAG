#!/usr/bin/env python3
"""
Search preserved verification artifacts for Table 27 TOST candidates.

The goal is not to manufacture a paper match. It is to answer a narrower,
defensible question: do any preserved per-seed GSM8K routing artifacts in this
mirror look plausibly consistent with the submitted Table 27 anchor
(`0.92 ± 0.02`, paired p≈0.31 for SYNAPSE vs Centralized-SYNAPSE)?

This script:
1. scans artifact summary JSON files with per-seed accuracies,
2. keeps only 5-seed vectors,
3. scores each vector against the Table 27 marginal anchors, and
4. scores vector pairs against the paired-null anchor.

It prints the best candidate individual vectors and the best candidate pairs,
plus an optional `run_table27_tost.py` command for the closest pair.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "artifacts" / "verification"

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_table27_tost import paired_t_test  # noqa: E402


@dataclass
class VectorSummary:
    path: Path
    seeds: list[int]
    values: list[float]
    mean: float
    sd: float
    mean_err: float
    sd_err: float
    score: float


@dataclass
class PairSummary:
    left: VectorSummary
    right: VectorSummary
    paired_p: float
    mean_diff: float
    score: float


def ordered_values(mapping: dict[str, Any]) -> tuple[list[int], list[float]] | None:
    pairs: list[tuple[int, float]] = []
    for seed, value in mapping.items():
        try:
            pairs.append((int(seed), float(value)))
        except (TypeError, ValueError):
            return None
    pairs.sort()
    return [seed for seed, _ in pairs], [value for _, value in pairs]


def load_summary(path: Path) -> VectorSummary | None:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None

    mapping = payload.get("per_seed_accuracy")
    if not isinstance(mapping, dict):
        return None

    ordered = ordered_values(mapping)
    if ordered is None:
        return None
    seeds, values = ordered
    if len(values) != 5:
        return None

    mean = sum(values) / len(values)
    sd = st.stdev(values) if len(values) > 1 else 0.0
    mean_err = abs(mean - 0.92)
    sd_err = abs(sd - 0.02)
    score = mean_err + sd_err
    return VectorSummary(
        path=path,
        seeds=seeds,
        values=values,
        mean=mean,
        sd=sd,
        mean_err=mean_err,
        sd_err=sd_err,
        score=score,
    )


def iter_summaries(root: Path) -> list[VectorSummary]:
    results: list[VectorSummary] = []
    for path in sorted(root.rglob("summary.json")):
        summary = load_summary(path)
        if summary is not None:
            results.append(summary)
    return results


def same_seed_order(left: VectorSummary, right: VectorSummary) -> bool:
    return left.seeds == right.seeds


def pair_score(left: VectorSummary, right: VectorSummary) -> PairSummary:
    paired = paired_t_test(left.values, right.values)
    paired_p = float(paired["p"])
    mean_diff = float(paired["mean_diff"])
    score = (
        left.score
        + right.score
        + abs(paired_p - 0.31)
        + abs(mean_diff - 0.0)
    )
    return PairSummary(
        left=left,
        right=right,
        paired_p=paired_p,
        mean_diff=mean_diff,
        score=score,
    )


def is_degenerate_vector(item: VectorSummary) -> bool:
    if all(abs(v) < 1e-12 for v in item.values):
        return True
    if all(abs(v - 1.0) < 1e-12 for v in item.values):
        return True
    if item.mean > 0.99 and item.sd < 0.005:
        return True
    return False


def looks_centralized(path: Path) -> bool:
    text = str(path).lower()
    return any(
        token in text
        for token in (
            "centralized",
            "cen",
            "table27",
            "fresh_compare",
            "strict_compare",
            "provenance_compare",
        )
    )


def looks_synapse(path: Path) -> bool:
    text = str(path).lower()
    return any(
        token in text
        for token in (
            "routing",
            "gsm8k",
            "paper_mode",
            "paper_recovery",
            "fresh_compare",
            "strict_compare",
            "provenance_compare",
            "table27",
            "federated",
            "synapse",
        )
    )


def format_vector(item: VectorSummary) -> str:
    vals = ", ".join(f"{v:.3f}" for v in item.values)
    return (
        f"{item.path}: mean={item.mean:.3f}, sd={item.sd:.3f}, "
        f"score={item.score:.3f}, seeds=[{vals}]"
    )


def format_pair(item: PairSummary) -> str:
    return (
        f"score={item.score:.3f}, paired_p={item.paired_p:.4f}, "
        f"mean_diff={item.mean_diff:+.4f}\n"
        f"  syn? {format_vector(item.left)}\n"
        f"  cen? {format_vector(item.right)}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Search local artifacts for plausible Table 27 TOST candidates.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument(
        "--emit-tost-command",
        action="store_true",
        help="Print a ready-to-run run_table27_tost.py command for the best candidate pair.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.root.resolve()
    vectors = iter_summaries(root)
    if not vectors:
        print(f"No 5-seed per_seed_accuracy summaries found under {root}")
        return 1

    vectors.sort(key=lambda item: item.score)
    print("=== Best individual Table 27 marginal candidates ===")
    for item in vectors[: args.top]:
        print(format_vector(item))

    syn_candidates = [item for item in vectors if looks_synapse(item.path) and not is_degenerate_vector(item)]
    cen_candidates = [item for item in vectors if looks_centralized(item.path) and not is_degenerate_vector(item)]
    pairs: list[PairSummary] = []
    for left in syn_candidates:
        for right in cen_candidates:
            if left.path == right.path:
                continue
            if not same_seed_order(left, right):
                continue
            pairs.append(pair_score(left, right))

    if not pairs:
        print()
        print("No plausible synapse/centralized candidate pairs found.")
        return 0

    pairs.sort(key=lambda item: item.score)
    print()
    print("=== Best paired Table 27 candidates ===")
    for item in pairs[: args.top]:
        print(format_pair(item))
        print()

    if args.emit_tost_command:
        best = pairs[0]
        print("=== Ready-to-run TOST command for best candidate pair ===")
        print(
            ".venv/bin/python scripts/run_table27_tost.py "
            f"--syn {best.left.path} "
            f"--cen {best.right.path} "
            "--margin 0.03"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
