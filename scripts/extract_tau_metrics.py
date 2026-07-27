#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Table 22/23-relevant metrics from a tau-bench results JSON file."
    )
    parser.add_argument("results_file", type=Path, help="Path to a tau-bench JSON results file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the extracted metrics as JSON.",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _count_role(traj: list[dict[str, Any]], role: str) -> int:
    return sum(1 for message in traj if message.get("role") == role)


def extract_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards: list[float] = []
    r_actions_all: list[float] = []
    r_actions_present: list[float] = []
    assistant_turns: list[int] = []
    user_turns: list[int] = []
    tool_turns: list[int] = []
    missing_reward_info = 0

    for row in rows:
        rewards.append(float(row.get("reward", 0.0)))

        info = row.get("info") or {}
        reward_info = info.get("reward_info") or {}
        reward_meta = reward_info.get("info") or {}
        if "r_actions" in reward_meta:
            action_score = float(reward_meta["r_actions"])
            r_actions_present.append(action_score)
        else:
            missing_reward_info += 1
            action_score = 0.0
        r_actions_all.append(action_score)

        traj = row.get("traj") or []
        assistant_turns.append(_count_role(traj, "assistant"))
        user_turns.append(_count_role(traj, "user"))
        tool_turns.append(_count_role(traj, "tool"))

    return {
        "row_count": len(rows),
        "task_success_mean_reward": _safe_mean(rewards),
        "task_success_rate_reward_eq_1": _safe_mean([1.0 if value >= 0.999 else 0.0 for value in rewards]),
        "tool_call_accuracy_mean_all_rows": _safe_mean(r_actions_all),
        "tool_call_accuracy_mean_present_rows": _safe_mean(r_actions_present),
        "reward_info_presence_coverage": _safe_mean([1.0 if row.get("info", {}).get("reward_info") else 0.0 for row in rows]),
        "missing_reward_info_rows": missing_reward_info,
        "avg_assistant_turns": _safe_mean([float(value) for value in assistant_turns]),
        "avg_user_turns": _safe_mean([float(value) for value in user_turns]),
        "avg_tool_turns": _safe_mean([float(value) for value in tool_turns]),
        "assistant_turns_min": min(assistant_turns) if assistant_turns else 0,
        "assistant_turns_max": max(assistant_turns) if assistant_turns else 0,
        "user_turns_min": min(user_turns) if user_turns else 0,
        "user_turns_max": max(user_turns) if user_turns else 0,
        "tool_turns_min": min(tool_turns) if tool_turns else 0,
        "tool_turns_max": max(tool_turns) if tool_turns else 0,
    }


def main() -> None:
    args = parse_args()
    rows = json.loads(args.results_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of run rows in {args.results_file}")

    payload = {
        "results_file": str(args.results_file),
        "metrics": extract_metrics(rows),
    }

    rendered = json.dumps(payload, indent=2)
    print(rendered)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
