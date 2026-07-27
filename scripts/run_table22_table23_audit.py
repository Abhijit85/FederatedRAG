#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = (
    REPO_ROOT
    / "external_datasets"
    / "tau_bench"
    / "runs_100"
    / "tool-calling-gpt-4o-mini-0.0_range_0-100_user-openai"
    / "gpt-4o-mini-llm_0724002801.json"
)
DEFAULT_JSON = REPO_ROOT / "artifacts" / "verification" / "table22_table23_audit.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "rebuttal" / "table22_table23_audit.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Table 22/23 against a tau-bench result file using manuscript-relevant metrics."
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
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
    missing_r_actions = 0

    for row in rows:
        rewards.append(float(row.get("reward", 0.0)))

        info = row.get("info") or {}
        reward_info = info.get("reward_info") or {}
        reward_meta = reward_info.get("info") or {}
        if "r_actions" in reward_meta:
            action_score = float(reward_meta["r_actions"])
            r_actions_present.append(action_score)
        else:
            missing_r_actions += 1
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
        "reward_info_presence_coverage": _safe_mean([1.0 if (row.get("info") or {}).get("reward_info") else 0.0 for row in rows]),
        "missing_r_actions_rows": missing_r_actions,
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


def render_markdown(results_file: Path, metrics: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Table 22/23 Audit",
            "",
            "This audit uses the manuscript-relevant metrics from a `tau-bench` result file.",
            "",
            f"- Results file: `{results_file}`",
            "- Paper Table 22 targets: task success, tool-call accuracy, average turns.",
            "- Current checked file is a stock `tau-bench` `gpt-4o-mini` run, not a SYNAPSE retail run.",
            "",
            "## Extracted Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Row count | {metrics['row_count']} |",
            f"| Task success (mean reward) | {metrics['task_success_mean_reward']:.3f} |",
            f"| Task success rate (`reward == 1`) | {metrics['task_success_rate_reward_eq_1']:.3f} |",
            f"| Tool-call accuracy (all rows) | {metrics['tool_call_accuracy_mean_all_rows']:.3f} |",
            f"| Tool-call accuracy (rows with `r_actions`) | {metrics['tool_call_accuracy_mean_present_rows']:.3f} |",
            f"| Reward-info presence coverage | {metrics['reward_info_presence_coverage']:.3f} |",
            f"| Missing `r_actions` rows | {metrics['missing_r_actions_rows']} |",
        f"| `r_actions` coverage | {1.0 - (metrics['missing_r_actions_rows'] / metrics['row_count'] if metrics['row_count'] else 0.0):.3f} |",
            f"| Avg. user turns | {metrics['avg_user_turns']:.2f} |",
            f"| Avg. assistant turns | {metrics['avg_assistant_turns']:.2f} |",
            f"| Avg. tool turns | {metrics['avg_tool_turns']:.2f} |",
            "",
            "## Comparison to Paper Scale",
            "",
            "| Quantity | Paper Table 22 | Current stock path |",
            "| --- | ---: | ---: |",
            "| Task success | 0.453 / 0.511 / 0.301 | "
            f"{metrics['task_success_mean_reward']:.3f} |",
            "| Tool-call accuracy | 0.540 / 0.608 / 0.432 | "
            f"{metrics['tool_call_accuracy_mean_all_rows']:.3f} all-row, {metrics['tool_call_accuracy_mean_present_rows']:.3f} covered-row |",
            "| Avg. turns | 5.8 / 5.5 / 6.7 | "
            f"{metrics['avg_user_turns']:.2f} user, {metrics['avg_assistant_turns']:.2f} assistant |",
            "",
            "Interpretation: this confirms evaluator/provenance mismatch rather than a small arithmetic discrepancy.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    rows = json.loads(args.results_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected list rows in {args.results_file}")

    metrics = extract_metrics(rows)
    payload = {
        "results_file": str(args.results_file),
        "metrics": metrics,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(args.results_file, metrics), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
