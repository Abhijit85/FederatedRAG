#!/usr/bin/env python3
"""
Utility to export a subset of the BIG-Bench Hard object counting task into the
JSON format consumed by --mixed-queries.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from third_party.textgrad.tasks.big_bench_hard import BigBenchHard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export BBH task samples to mixed-query format.")
    parser.add_argument("--task", default="BBH_object_counting", help="Name of the BBH task to load.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test", help="Dataset split to sample from.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum number of samples to export (0 = all).")
    parser.add_argument("--output", type=Path, default=Path("bbh_object_counting_eval.json"), help="Destination JSON file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    task_name = args.task.replace("BBH_", "")
    dataset = BigBenchHard(task_name, split=args.split)

    samples = []
    for idx, (question, answer) in enumerate(dataset):
        if args.limit and idx >= args.limit:
            break
        samples.append(
            {
                "question": question,
                "answer": answer,
                "dataset": args.task,
                "domain": "math",
            }
        )

    args.output.write_text(json.dumps(samples, indent=2), encoding="utf-8")
    print(f"✅ Wrote {len(samples)} samples to {args.output}")


if __name__ == "__main__":
    main()
