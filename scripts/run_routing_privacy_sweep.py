#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "artifacts" / "verification" / "routing_privacy_sweep"

CONFIGS = [
    {"label": "no_privacy", "enable_dp": "0", "epsilon": None},
    {"label": "eps_2_0", "enable_dp": "1", "epsilon": "2.0"},
    {"label": "eps_1_0", "enable_dp": "1", "epsilon": "1.0"},
    {"label": "eps_0_5", "enable_dp": "1", "epsilon": "0.5"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real routing verification sweeps across privacy settings.")
    parser.add_argument("--sample-count", type=int, default=50)
    parser.add_argument("--seeds", type=str, default="1,2,3,4,5")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--client-count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def run_one(config: dict[str, str | None], args: argparse.Namespace) -> dict[str, Any]:
    label = str(config["label"])
    outdir = args.output_dir / label
    outdir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["SYNAPSE_ENABLE_DP"] = str(config["enable_dp"])
    if config["epsilon"] is None:
        env.pop("SYNAPSE_DP_EPSILON", None)
    else:
        env["SYNAPSE_DP_EPSILON"] = str(config["epsilon"])

    cmd = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        "scripts/run_routing_verification.py",
        "--sample-count", str(args.sample_count),
        "--seeds", args.seeds,
        "--rounds", str(args.rounds),
        "--output-dir", str(outdir),
    ]
    if args.client_count is not None:
        cmd.extend(["--client-count", str(args.client_count)])

    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    summary = json.loads((outdir / "summary.json").read_text(encoding="utf-8"))
    return {
        "label": label,
        "enable_dp": env["SYNAPSE_ENABLE_DP"],
        "epsilon": env.get("SYNAPSE_DP_EPSILON"),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_one(config, args) for config in CONFIGS]
    payload = {"sample_count": args.sample_count, "seeds": args.seeds, "rounds": args.rounds, "results": results}
    out_path = args.output_dir / "combined_summary.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
