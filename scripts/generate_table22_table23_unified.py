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
DEFAULT_JSON = REPO_ROOT / "artifacts" / "verification" / "table22_table23_unified.json"
DEFAULT_MD = REPO_ROOT / "artifacts" / "rebuttal" / "table22_table23_unified.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Table 22-style and Table 23-style metrics from one tau-bench "
            "results artifact using a single extraction pipeline."
        )
    )
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_MD)
    parser.add_argument(
        "--label",
        type=str,
        default="stock tau-bench path",
        help="Short label for the run being summarized.",
    )
    return parser.parse_args()


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _count_role(traj: list[dict[str, Any]], role: str) -> int:
    return sum(1 for message in traj if message.get("role") == role)


def extract_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rewards: list[float] = []
    r_actions_zero_filled: list[float] = []
    r_actions_present_only: list[float] = []
    assistant_turns: list[int] = []
    user_turns: list[int] = []
    tool_turns: list[int] = []
    dialogue_turns_user_assistant: list[int] = []
    missing_r_actions = 0

    for row in rows:
        rewards.append(float(row.get("reward", 0.0)))
        info = row.get("info") or {}
        reward_info = info.get("reward_info") or {}
        reward_meta = reward_info.get("info") or {}
        if "r_actions" in reward_meta:
            action_score = float(reward_meta["r_actions"])
            r_actions_present_only.append(action_score)
        else:
            action_score = 0.0
            missing_r_actions += 1
        r_actions_zero_filled.append(action_score)

        traj = row.get("traj") or []
        assistant = _count_role(traj, "assistant")
        user = _count_role(traj, "user")
        tool = _count_role(traj, "tool")
        assistant_turns.append(assistant)
        user_turns.append(user)
        tool_turns.append(tool)
        dialogue_turns_user_assistant.append(user + assistant)

    return {
        "row_count": len(rows),
        "task_success_mean_reward": _safe_mean(rewards),
        "task_success_rate_reward_eq_1": _safe_mean([1.0 if value >= 0.999 else 0.0 for value in rewards]),
        "tool_call_accuracy_zero_filled": _safe_mean(r_actions_zero_filled),
        "tool_call_accuracy_present_only": _safe_mean(r_actions_present_only),
        "reward_info_presence_coverage": _safe_mean([1.0 if ((row.get("info") or {}).get("reward_info")) else 0.0 for row in rows]),
        "missing_r_actions_rows": missing_r_actions,
        "avg_user_turns": _safe_mean([float(v) for v in user_turns]),
        "avg_assistant_turns": _safe_mean([float(v) for v in assistant_turns]),
        "avg_tool_turns": _safe_mean([float(v) for v in tool_turns]),
        "avg_dialogue_turns_user_assistant": _safe_mean([float(v) for v in dialogue_turns_user_assistant]),
    }


def build_views(metrics: dict[str, Any]) -> dict[str, Any]:
    # Table 22-style: one explicit convention chosen and reported.
    table22_view = {
        "task_success": metrics["task_success_mean_reward"],
        "tool_call_accuracy": metrics["tool_call_accuracy_zero_filled"],
        "avg_turns_user_only": metrics["avg_user_turns"],
        "avg_turns_assistant_only": metrics["avg_assistant_turns"],
    }
    # Table 23-style: same source run, but exposes the other plausible post-processing choices.
    table23_view = {
        "task_success": metrics["task_success_mean_reward"],
        "tool_call_accuracy": metrics["tool_call_accuracy_present_only"],
        "avg_turns_user_plus_assistant": metrics["avg_dialogue_turns_user_assistant"],
    }
    return {
        "table22_view": table22_view,
        "table23_view": table23_view,
    }


def render_markdown(results_file: Path, label: str, metrics: dict[str, Any], views: dict[str, Any]) -> str:
    t22 = views["table22_view"]
    t23 = views["table23_view"]
    return "\n".join([
        "# Unified Table 22/23 Regeneration",
        "",
        "This artifact regenerates both table views from one source run artifact and one extraction pipeline.",
        "",
        f"- Run label: `{label}`",
        f"- Results file: `{results_file}`",
        "- Single source of truth: one per-task JSON artifact containing `reward`, `reward_info`, and `traj`.",
        "",
        "## Canonical Extracted Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Row count | {metrics['row_count']} |",
        f"| Task success (mean reward) | {metrics['task_success_mean_reward']:.3f} |",
        f"| Tool-call accuracy, zero-filled missing rows | {metrics['tool_call_accuracy_zero_filled']:.3f} |",
        f"| Tool-call accuracy, present rows only | {metrics['tool_call_accuracy_present_only']:.3f} |",
        f"| Reward-info presence coverage | {metrics['reward_info_presence_coverage']:.3f} |",
        f"| Missing `r_actions` rows | {metrics['missing_r_actions_rows']} |",
        f"| `r_actions` coverage | {1.0 - (metrics['missing_r_actions_rows'] / metrics['row_count'] if metrics['row_count'] else 0.0):.3f} |",
        f"| Avg. user turns | {metrics['avg_user_turns']:.2f} |",
        f"| Avg. assistant turns | {metrics['avg_assistant_turns']:.2f} |",
        f"| Avg. user+assistant turns | {metrics['avg_dialogue_turns_user_assistant']:.2f} |",
        "",
        "## Regenerated Table Views From The Same Source Run",
        "",
        "| View | Task success | Tool-call accuracy | Turns convention |",
        "| --- | ---: | ---: | ---: |",
        f"| Table 22-style | {t22['task_success']:.3f} | {t22['tool_call_accuracy']:.3f} | {t22['avg_turns_user_only']:.2f} user-only / {t22['avg_turns_assistant_only']:.2f} assistant-only |",
        f"| Table 23-style | {t23['task_success']:.3f} | {t23['tool_call_accuracy']:.3f} | {t23['avg_turns_user_plus_assistant']:.2f} user+assistant |",
        "",
        "## Procedural Conclusion",
        "",
        "Both table views above are generated from the same source artifact. Any remaining discrepancy is therefore attributable to post-processing convention choices, not to using different underlying runs.",
        "",
    ])


def main() -> None:
    args = parse_args()
    rows = json.loads(args.results_file.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a list of run rows in {args.results_file}")

    metrics = extract_metrics(rows)
    views = build_views(metrics)
    payload = {
        "results_file": str(args.results_file),
        "label": args.label,
        "metrics": metrics,
        "views": views,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(args.results_file, args.label, metrics, views), encoding="utf-8")
    print(args.output_json)
    print(args.output_md)


if __name__ == "__main__":
    main()
