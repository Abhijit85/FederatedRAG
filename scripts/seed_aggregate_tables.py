#!/usr/bin/env python3
"""
Seed-aware aggregation for the rebuttal reproduction tables.

WHY
---
`artifacts/rebuttal/reviewer_gap_response.md` currently reports each
reproduction checkpoint as a single "Actual Rerun number" (one run). A
reproducibility rebuttal is far stronger when each number is reported as a
mean over multiple seeds with a standard deviation and confidence interval.

WHAT THIS DOES
--------------
* Reads per-seed measured accuracies from `scripts/seed_values.json`.
* For every checkpoint it computes: n (number of seeds), mean, SD (ddof=1),
  SEM, and a 95% t-based confidence interval.
* Compares the seed mean against the published "Paper anchor".
* Regenerates the Table A / B / C / D / E markdown blocks with the seed
  statistics filled in.

WHAT THIS DOES NOT DO
---------------------
It does not invent per-seed variance. Any checkpoint whose list in
`seed_values.json` still has a single value is reported as `n=1` with no SD /
CI, and flagged, so nothing here fabricates spread you did not measure. Paste
your real per-seed numbers into `seed_values.json` to populate SD / CI.

USAGE
-----
    python3 scripts/seed_aggregate_tables.py                 # print report
    python3 scripts/seed_aggregate_tables.py --emit-md OUT.md # also write md
    python3 scripts/seed_aggregate_tables.py --seed-json PATH # custom values

Requires: scipy (already a dependency of scripts/tost_equivalence.py).
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from scipy import stats as _stats
except Exception:  # pragma: no cover - scipy optional at import time
    _stats = None

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED_JSON = REPO_ROOT / "scripts" / "seed_values.json"


@dataclass
class Checkpoint:
    """One row of a rebuttal table."""

    label: str
    anchor: str          # paper anchor exactly as displayed (may be text)
    decimals: int        # display precision for numeric columns
    signed: bool = False  # render numeric values with an explicit +/- sign
    numeric: bool = True  # False => pass-through textual row (e.g. TOST)
    note: str = ""        # text shown for non-numeric rows

    # populated at runtime
    seeds: list[float] = field(default_factory=list)


@dataclass
class Table:
    name: str
    header: str
    rows: list[Checkpoint]


def _n(label: str, anchor: str, decimals: int, **kw) -> Checkpoint:
    return Checkpoint(label=label, anchor=anchor, decimals=decimals, **kw)


TABLES: list[Table] = [
    Table(
        name="A",
        header="Table A. Core Reproduction Checkpoints",
        rows=[
            _n("Table 2, typed condition, 0% contradiction", "0.92", 2),
            _n("Table 2, typed condition, 20% contradiction", "0.89", 2),
            _n("Table 2, typed condition, 40% contradiction", "0.86", 2),
            _n("Table 2, typed condition, 60% contradiction", "0.81", 2),
            _n("Table 14, TextGrad S=3", "0.92", 2),
            _n("Table 14, TextGrad S=1", "0.89", 2),
            _n("Table 14, TextGrad S=5", "0.92", 2),
            _n("Table 14, extractive centroid", "0.85", 2),
            _n("Table 14, single-shot summarize", "0.87", 2),
            _n("Table 14, no summarization", "0.78", 2),
        ],
    ),
    Table(
        name="B",
        header="Table B. Privacy-Utility Validation Points From Table 9",
        rows=[
            _n("No privacy", "0.935", 3),
            _n("eps=2.0, lambda=0.5", "0.928", 3),
            _n("eps=2.0, lambda=1.0", "0.914", 3),
            _n("eps=2.0, lambda=1.5", "0.897", 3),
            _n("eps=1.0, lambda=0.5", "0.909", 3),
            _n("eps=1.0, lambda=1.0", "0.902", 3),
            _n("eps=1.0, lambda=1.5", "0.881", 3),
            _n("eps=0.5, lambda=0.5", "0.884", 3),
            _n("eps=0.5, lambda=1.0", "0.866", 3),
            _n("eps=0.5, lambda=1.5", "0.851", 3),
        ],
    ),
    Table(
        name="C",
        header="Table C. ToolBench / mmFG-W2 Extension Checkpoint",
        rows=[
            _n("ToolBench overall, 250-query baseline", "0.728", 3),
            _n(
                "ToolBench extension to 600-750 queries, same protocol",
                "0.728 reference",
                3,
            ),
        ],
    ),
    Table(
        name="D",
        header="Table D. Cross-Model / Root-Cause Checkpoints From Tables 22-23",
        rows=[
            _n("Table 22, tau-bench retail, SYNAPSE main", "0.453", 3),
            _n("Table 22, tau-bench retail, centralized", "0.511", 3),
            _n("Table 22, tau-bench retail, Fed-ICL", "0.301", 3),
            _n("Table 23, LLaMA-3.2-3B delta vs main", "-0.022", 3, signed=True),
            _n("Table 23, Mistral-7B delta vs main", "-0.009", 3, signed=True),
            _n("Table 23, GPT-4o delta vs main", "+0.085", 3, signed=True),
        ],
    ),
    Table(
        name="E",
        header="Table E. Controls / Equivalence Checkpoints",
        rows=[
            _n(
                "Field-preserving structured-but-untyped control",
                "between 0.82 and 0.92",
                2,
            ),
            _n(
                "Paired TOST mean difference (SYNAPSE - centralized)",
                "parity claim",
                3,
                numeric=False,
                note="0.00 (see scripts/tost_equivalence.py)",
            ),
            _n(
                "Paired TOST 90% CI containment margin",
                "+/-0.03 margin",
                3,
                numeric=False,
                note="inside +/-0.03 (see scripts/tost_equivalence.py)",
            ),
        ],
    ),
]


@dataclass
class SeedStats:
    n: int
    mean: float
    sd: Optional[float]
    sem: Optional[float]
    ci_low: Optional[float]
    ci_high: Optional[float]


def compute_stats(values: list[float], alpha: float = 0.05) -> SeedStats:
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return SeedStats(n=n, mean=mean, sd=None, sem=None, ci_low=None, ci_high=None)
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(var)
    sem = sd / math.sqrt(n)
    if _stats is not None:
        tcrit = _stats.t.ppf(1 - alpha / 2, n - 1)
    else:  # fallback to normal approx if scipy missing
        tcrit = 1.96
    half = tcrit * sem
    return SeedStats(n=n, mean=mean, sd=sd, sem=sem, ci_low=mean - half, ci_high=mean + half)


def fmt(value: float, decimals: int, signed: bool = False) -> str:
    s = f"{value:.{decimals}f}"
    if signed and value >= 0:
        s = "+" + s
    return s


def load_seed_values(path: Path) -> dict[str, list[float]]:
    data = json.loads(path.read_text())
    return {k: v for k, v in data.items() if not k.startswith("_") and isinstance(v, list)}


def attach_seeds(seed_values: dict[str, list[float]]) -> list[str]:
    """Attach per-seed lists to checkpoints. Returns list of warnings."""
    warnings: list[str] = []
    for table in TABLES:
        for row in table.rows:
            if not row.numeric:
                continue
            vals = seed_values.get(row.label)
            if vals is None:
                warnings.append(f"[missing] no seed values for: {row.label!r}")
                continue
            row.seeds = [float(v) for v in vals]
    return warnings


def anchor_numeric(anchor: str) -> Optional[float]:
    """Extract a single float from an anchor string, else None."""
    token = anchor.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


def render_table_md(table: Table) -> str:
    lines = [
        f"### {table.header}",
        "",
        "| Checkpoint | Paper anchor | Seeds (n) | Mean rerun | SD | 95% CI | Δ vs anchor |",
        "| --- | ---: | ---: | ---: | ---: | :---: | ---: |",
    ]
    for row in table.rows:
        if not row.numeric:
            lines.append(
                f"| {row.label} | {row.anchor} | — | {row.note} | — | — | — |"
            )
            continue
        if not row.seeds:
            lines.append(
                f"| {row.label} | {row.anchor} | 0 | _no data_ | — | — | — |"
            )
            continue
        st = compute_stats(row.seeds)
        mean_s = fmt(st.mean, row.decimals, row.signed)
        if st.sd is None:
            sd_s = "n=1"
            ci_s = "—"
        else:
            sd_s = fmt(st.sd, row.decimals)
            ci_s = f"[{fmt(st.ci_low, row.decimals, row.signed)}, {fmt(st.ci_high, row.decimals, row.signed)}]"
        anc = anchor_numeric(row.anchor)
        if anc is None:
            delta_s = "—"
        else:
            delta = st.mean - anc
            delta_s = fmt(delta, row.decimals, signed=True)
        lines.append(
            f"| {row.label} | {row.anchor} | {st.n} | {mean_s} | {sd_s} | {ci_s} | {delta_s} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_all_md() -> str:
    blocks = [
        "<!-- Auto-generated by scripts/seed_aggregate_tables.py -->",
        "<!-- Fill scripts/seed_values.json with real per-seed values, then rerun. -->",
        "",
    ]
    for table in TABLES:
        blocks.append(render_table_md(table))
    return "\n".join(blocks)


def print_console_report(warnings: list[str]) -> None:
    print("Seed aggregation report")
    print("=" * 72)
    single_run: list[str] = []
    for table in TABLES:
        for row in table.rows:
            if row.numeric and row.seeds and len(row.seeds) < 2:
                single_run.append(row.label)
    if warnings:
        print(f"\n{len(warnings)} checkpoint(s) missing from seed_values.json:")
        for w in warnings:
            print("  " + w)
    if single_run:
        print(
            f"\n{len(single_run)} checkpoint(s) still n=1 (no SD/CI). "
            "Add more seeds in scripts/seed_values.json:"
        )
        for label in single_run:
            print("  [n=1] " + label)
    print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seed-json",
        type=Path,
        default=DEFAULT_SEED_JSON,
        help="Path to per-seed values JSON (default: scripts/seed_values.json).",
    )
    ap.add_argument(
        "--emit-md",
        type=Path,
        default=None,
        help="Optional path to write the regenerated markdown tables.",
    )
    args = ap.parse_args()

    seed_values = load_seed_values(args.seed_json)
    warnings = attach_seeds(seed_values)

    md = render_all_md()
    print(md)
    print_console_report(warnings)

    if args.emit_md is not None:
        args.emit_md.write_text(md + "\n")
        print(f"Wrote markdown tables to {args.emit_md}")


if __name__ == "__main__":
    main()
