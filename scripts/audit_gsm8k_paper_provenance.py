#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAPER_TEXT = Path("/home/ad.asu.edu/achakr40/.codex/attachments/0026c74a-9bd5-4306-aa47-9461b2188243/pasted-text.txt")
DEFAULT_OUTPUT_MD = REPO_ROOT / "artifacts" / "provenance" / "gsm8k_paper_provenance_audit.md"
DEFAULT_OUTPUT_JSON = REPO_ROOT / "artifacts" / "provenance" / "gsm8k_paper_provenance_audit.json"

LIVE_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "routing_math_only_paperlike_100" / "summary.json"
PAPER_MODE_SUMMARY = REPO_ROOT / "artifacts" / "verification" / "gsm8k_paper_mode_local_run3" / "summary.json"
RECOVERY_SWEEP = REPO_ROOT / "artifacts" / "verification" / "gsm8k_paper_recovery_sweep_run2" / "combined_summary.json"
RUNLOG_EVOLUTION = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_routing_evolution.json"
COMPENDIUM_EVOLUTION = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_compendium_evolution.json"
UNIFIED_CLIENT = REPO_ROOT / "synapse" / "clients" / "unified_client.py"


@dataclass(frozen=True)
class EvidenceRow:
    name: str
    path: Path
    metric: str
    mean: float | None
    sd: float | None
    note: str
    status: str


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_extract_paper_anchor(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path else None, "anchor_lines": [], "anchor_mean": None, "anchor_sd": None}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    anchor_lines = [
        line.strip()
        for line in lines
        if "0.92" in line and ("GSM8" in line or "routing accuracy" in line)
    ]
    match = re.search(r"0\.92\s*[±\+\-]\s*0\.02", text)
    return {
        "path": str(path),
        "anchor_lines": anchor_lines[:8],
        "anchor_mean": 0.92 if match else None,
        "anchor_sd": 0.02 if match else None,
    }


def find_method(summary: dict[str, Any], method: str) -> dict[str, Any]:
    for item in summary.get("methods", []):
        if item.get("method") == method:
            return item
    raise KeyError(f"method {method!r} not found")


def detect_math_only_fix(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    return {
        "path": str(path),
        "has_included_tools_env": "SYNAPSE_INCLUDED_TOOLS" in text,
        "guards_mathqa": '"mathqa" in self.included_tools' in text,
        "guards_scienceqa": '"scienceqa" in self.included_tools' in text,
    }


def build_report(paper_text_path: Path | None) -> dict[str, Any]:
    live = load_json(LIVE_SUMMARY)
    paper_mode = load_json(PAPER_MODE_SUMMARY)
    recovery = load_json(RECOVERY_SWEEP)
    runlog = load_json(RUNLOG_EVOLUTION)
    compendium = load_json(COMPENDIUM_EVOLUTION)
    best = find_method(recovery, "cv_svm")
    aux = find_method(recovery, "cv_aux_collapse")
    paper_anchor = maybe_extract_paper_anchor(paper_text_path)
    fix = detect_math_only_fix(UNIFIED_CLIENT)

    evidence = [
        EvidenceRow(
            name="Submitted paper anchor",
            path=paper_text_path if paper_text_path else Path("<missing>"),
            metric="GSM8K routing accuracy",
            mean=paper_anchor["anchor_mean"],
            sd=paper_anchor["anchor_sd"],
            note="Reported in the submitted manuscript as the headline IID GSM8K result.",
            status="claimed",
        ),
        EvidenceRow(
            name="Current live math-only verifier",
            path=LIVE_SUMMARY,
            metric="100-query, 5-seed routing accuracy under current unified runtime",
            mean=live.get("mean_accuracy"),
            sd=live.get("sd_accuracy"),
            note="Uses the current verifier after restricting the runtime to math-only artifacts.",
            status="measured",
        ),
        EvidenceRow(
            name="Historical paper-mode prototype reconstruction",
            path=PAPER_MODE_SUMMARY,
            metric="100-query, 5-seed local prototype routing accuracy",
            mean=paper_mode.get("mean_accuracy"),
            sd=paper_mode.get("sd_accuracy"),
            note="Six historical paper-time labels with exemplar-enriched prototypes.",
            status="measured",
        ),
        EvidenceRow(
            name="Best preserved local recovery",
            path=Path(best["output_dir"]) / "summary.json",
            metric="100-query, 5-seed local reconstruction accuracy",
            mean=best.get("mean_accuracy"),
            sd=best.get("sd_accuracy"),
            note="Cross-validated SVM over the preserved six-label paper-time universe.",
            status="measured",
        ),
        EvidenceRow(
            name="Historical April 3 runlog evolution",
            path=RUNLOG_EVOLUTION,
            metric="500-record routing accuracy in preserved runlog evolution artifact",
            mean=runlog.get("overall", {}).get("routing_accuracy"),
            sd=None,
            note="This preserved runlog is nearly perfect, so it is not the same benchmark object as the submitted 0.92 +- 0.02 table claim.",
            status="measured",
        ),
    ]

    conclusions = [
        "The anonymous mirror preserves the paper-time six-scenario GSM8K artifact space.",
        "The current repo head now supports math-only routing runs via SYNAPSE_INCLUDED_TOOLS, which removes one real runtime drift caused by mixed math/science artifact emission.",
        "Even after that fix, the current live verifier measures 0.328 +- 0.018 on the 100-query reconstruction, far below the submitted 0.92 +- 0.02.",
        "The strongest local historical reconstruction recovered from preserved artifacts is 0.770 +- 0.042 using cv_svm; that is materially better than the live verifier, but still well below the submitted anchor.",
        "The preserved April 3 runlog evolution artifact reports 0.998 routing accuracy over 500 records, which is too high to be the same evaluation object as the submitted 0.92 +- 0.02 benchmark.",
        "Therefore the repo preserves enough state to recover the historical six-label universe, but not the exact historical scorer / benchmark / label-generation path that produced the submitted GSM8K headline.",
    ]

    rebuttal_safe = (
        "The anonymous mirror preserves the paper-time six-scenario GSM8K artifact space and supports local "
        "reconstruction up to 0.770 +- 0.042 over 5 seeds, after fixing one runtime drift in the current verifier "
        "(math/science artifact mixing). However, the mirror does not preserve the exact historical evaluation path "
        "that yielded the submitted 0.92 +- 0.02 GSM8K routing-accuracy result, so we do not present that headline "
        "as reproduced from the current repo state."
    )

    return {
        "paper_anchor": paper_anchor,
        "math_only_fix": fix,
        "paper_labels": compendium.get("overall_unique_scenarios", []),
        "paper_label_count": len(compendium.get("overall_unique_scenarios", [])),
        "best_recovery_method": {
            "method": best.get("method"),
            "mean_accuracy": best.get("mean_accuracy"),
            "sd_accuracy": best.get("sd_accuracy"),
            "per_seed_accuracy": best.get("per_seed_accuracy"),
        },
        "aux_collapse_method": {
            "method": aux.get("method"),
            "mean_accuracy": aux.get("mean_accuracy"),
            "sd_accuracy": aux.get("sd_accuracy"),
            "per_seed_accuracy": aux.get("per_seed_accuracy"),
        },
        "historical_runlog_overall": runlog.get("overall", {}),
        "evidence_rows": [
            {
                "name": row.name,
                "path": str(row.path),
                "metric": row.metric,
                "mean": row.mean,
                "sd": row.sd,
                "note": row.note,
                "status": row.status,
            }
            for row in evidence
        ],
        "conclusions": conclusions,
        "rebuttal_safe_wording": rebuttal_safe,
    }


def fmt_metric(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "not found"
    if sd is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} +- {sd:.3f}"


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# GSM8K Paper-Claim Provenance Audit")
    lines.append("")
    lines.append("## Claim")
    lines.append("")
    lines.append("Submitted-paper anchor under audit: `0.92 +- 0.02` GSM8K routing accuracy.")
    if report["paper_anchor"]["anchor_lines"]:
        lines.append("")
        lines.append("Preserved manuscript lines:")
        for line in report["paper_anchor"]["anchor_lines"]:
            lines.append(f"- `{line}`")
    lines.append("")
    lines.append("## Preserved Evidence")
    lines.append("")
    lines.append("| Artifact | Metric | Value | Status | Note |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["evidence_rows"]:
        lines.append(
            f"| {row['name']} | {row['metric']} | {fmt_metric(row['mean'], row['sd'])} | {row['status']} | "
            f"`{row['path']}`; {row['note']} |"
        )
    lines.append("")
    lines.append("## Recovered Historical Structure")
    lines.append("")
    lines.append(
        f"The preserved compendium evolution still exposes the paper-time six-scenario GSM8K universe "
        f"(`n={report['paper_label_count']}`):"
    )
    for label in report["paper_labels"]:
        lines.append(f"- `{label}`")
    lines.append("")
    lines.append("## Current Runtime Check")
    lines.append("")
    fix = report["math_only_fix"]
    lines.append(
        f"`{fix['path']}` now includes `SYNAPSE_INCLUDED_TOOLS` support: "
        f"env gate={fix['has_included_tools_env']}, math guard={fix['guards_mathqa']}, science guard={fix['guards_scienceqa']}."
    )
    lines.append(
        "This fixes one real drift in the live verifier by allowing math-only artifact emission, but it does not close the full gap to the submitted paper number."
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    for item in report["conclusions"]:
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Rebuttal-Safe Wording")
    lines.append("")
    lines.append(f"> {report['rebuttal_safe_wording']}")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the provenance of the submitted GSM8K 0.92 +- 0.02 claim.")
    parser.add_argument("--paper-text-path", type=Path, default=DEFAULT_PAPER_TEXT)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args.paper_text_path)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
