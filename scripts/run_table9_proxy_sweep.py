#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "verification" / "table9_proxy_sweep"

CONFIGS = [
    {"label": "no_privacy", "epsilon": None, "lambda_scale": None},
    {"label": "eps_2_0_lambda_0_5", "epsilon": 2.0, "lambda_scale": 0.5},
    {"label": "eps_2_0_lambda_1_0", "epsilon": 2.0, "lambda_scale": 1.0},
    {"label": "eps_2_0_lambda_1_5", "epsilon": 2.0, "lambda_scale": 1.5},
    {"label": "eps_1_0_lambda_0_5", "epsilon": 1.0, "lambda_scale": 0.5},
    {"label": "eps_1_0_lambda_1_0", "epsilon": 1.0, "lambda_scale": 1.0},
    {"label": "eps_1_0_lambda_1_5", "epsilon": 1.0, "lambda_scale": 1.5},
    {"label": "eps_0_5_lambda_0_5", "epsilon": 0.5, "lambda_scale": 0.5},
    {"label": "eps_0_5_lambda_1_0", "epsilon": 0.5, "lambda_scale": 1.0},
    {"label": "eps_0_5_lambda_1_5", "epsilon": 0.5, "lambda_scale": 1.5},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Table 9 proxy privacy-utility sweep.")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "train_new.json")
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def _run_one(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    label = str(config["label"])
    outdir = args.output_dir / label
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "scripts/run_privacy_utility_eval.py",
        "--dataset",
        str(args.dataset),
        "--sample-count",
        str(args.sample_count),
        "--seeds",
        args.seeds,
        "--rounds",
        str(args.rounds),
        "--label",
        label,
        "--output-dir",
        str(outdir),
    ]
    if args.client_count is not None:
        cmd.extend(["--client-count", str(args.client_count)])
    if config["epsilon"] is not None:
        cmd.extend(["--epsilon", str(config["epsilon"])])
    if config["lambda_scale"] is not None:
        cmd.extend(["--lambda-scale", str(config["lambda_scale"])])

    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    summary_path = outdir / f"{label}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "label": label,
        "epsilon": config["epsilon"],
        "lambda_scale": config["lambda_scale"],
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [_run_one(config, args) for config in CONFIGS]
    payload = {
        "dataset": str(args.dataset),
        "sample_count": args.sample_count,
        "seeds": args.seeds,
        "rounds": args.rounds,
        "client_count": args.client_count,
        "results": results,
    }
    out_path = args.output_dir / "combined_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
