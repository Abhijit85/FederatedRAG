#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_FILE = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_samples.json"
DEFAULT_RUNLOG = REPO_ROOT / "GSM8K_500_rebuttal_run" / "GSM8K_500_runlog.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "verification" / "gsm8k_historical_artifact_reconstruction"

ALIASES = {
    "geometry and measurement": "geometry shapes and measurement",
    "geometry shapes and measurement": "geometry shapes and measurement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct the preserved April 3, 2026 GSM8K routing artifact by aligning the 500-sample "
            "snapshot to the historical runlog. This recreates the surviving paper-time routing object, "
            "not a fresh rerun of the manuscript benchmark."
        )
    )
    parser.add_argument("--sample-file", type=Path, default=DEFAULT_SAMPLE_FILE)
    parser.add_argument("--runlog", type=Path, default=DEFAULT_RUNLOG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def normalize_label(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return ALIASES.get(text, text)


def labels_match(left: str | None, right: str | None) -> bool:
    return normalize_label(left) == normalize_label(right)


def load_sample_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [row for row in payload["records"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    raise ValueError(f"Unsupported sample file format: {path}")


def load_runlog_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("source_kind") == "gsm8k_derived":
                rows.append(obj)
    return rows


def query_id_for(record: dict[str, Any]) -> str:
    return str(record.get("query_id") or record.get("sample_id") or "")


def query_text_for(record: dict[str, Any]) -> str:
    for key in ("query_text", "question", "Problem"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def gold_label_for(record: dict[str, Any]) -> str:
    router = record.get("router")
    if isinstance(router, dict):
        value = router.get("ground_truth_domain")
        if isinstance(value, str) and value.strip():
            return value.strip()
    for key in ("ground_truth_domain", "domain", "scenario"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def top_candidates_for(runlog_row: dict[str, Any]) -> list[dict[str, Any]]:
    router = runlog_row.get("router") or {}
    candidates = router.get("top_candidates") or []
    result: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        result.append(
            {
                "domain": candidate.get("domain"),
                "domain_normalized": normalize_label(candidate.get("domain") if isinstance(candidate.get("domain"), str) else None),
                "cosine_score": candidate.get("cosine_score"),
                "rank": candidate.get("rank"),
            }
        )
    return result


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "### GSM8K Historical Artifact Reconstruction",
        "",
        f"- Source runlog: `{summary['runlog']}`",
        f"- Source sample snapshot: `{summary['sample_file']}`",
        f"- Aligned records: {summary['aligned_record_count']}",
        f"- Routing accuracy in preserved artifact: {summary['routing_accuracy']:.3f}",
        f"- Ambiguity rate in preserved artifact: {summary['ambiguity_rate']:.3f}",
        f"- Mean top-gap in preserved artifact: {summary['mean_top_gap']:.3f}",
        "",
        "This is a reconstructed historical routing artifact aligned from preserved April 3, 2026 runlog records.",
        "It is not a fresh rerun and should not be presented as a reproduced manuscript benchmark endpoint.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sample_records = load_sample_records(args.sample_file)
    runlog_rows = load_runlog_rows(args.runlog)
    runlog_by_qid = {query_id_for(row): row for row in runlog_rows if query_id_for(row)}

    reconstructed: list[dict[str, Any]] = []
    missing_query_ids: list[str] = []
    routed_correct_flags: list[int] = []
    ambiguous_flags: list[int] = []
    top_gaps: list[float] = []

    for sample_index, sample_record in enumerate(sample_records, start=1):
        query_id = query_id_for(sample_record)
        runlog_row = runlog_by_qid.get(query_id)
        if runlog_row is None:
            missing_query_ids.append(query_id)
            continue

        router = runlog_row.get("router") or {}
        evaluation = runlog_row.get("evaluation") or {}
        predicted_domain = router.get("predicted_domain") if isinstance(router.get("predicted_domain"), str) else ""
        ground_truth_domain = gold_label_for(runlog_row)
        top_candidates = top_candidates_for(runlog_row)
        top_gap = evaluation.get("top_gap")
        ambiguous = bool(evaluation.get("ambiguous"))
        routed_correctly = labels_match(predicted_domain, ground_truth_domain)

        if isinstance(top_gap, (int, float)):
            top_gaps.append(float(top_gap))
        ambiguous_flags.append(int(ambiguous))
        routed_correct_flags.append(int(routed_correctly))

        reconstructed.append(
            {
                "sample_index": sample_index,
                "query_id": query_id,
                "sample_id": runlog_row.get("sample_id") or sample_record.get("sample_id"),
                "headline": runlog_row.get("headline"),
                "timestamp": runlog_row.get("timestamp"),
                "query_text": query_text_for(runlog_row) or query_text_for(sample_record),
                "expected_answer": runlog_row.get("expected_answer"),
                "ground_truth_domain": ground_truth_domain,
                "ground_truth_domain_normalized": normalize_label(ground_truth_domain),
                "predicted_domain": predicted_domain,
                "predicted_domain_normalized": normalize_label(predicted_domain),
                "routed_correctly": routed_correctly,
                "route_confidence": evaluation.get("route_confidence"),
                "top_gap": top_gap,
                "ambiguous": ambiguous,
                "top_candidates": top_candidates,
                "source_dataset": runlog_row.get("source_dataset"),
                "source_file": runlog_row.get("source_file"),
                "source_kind": runlog_row.get("source_kind"),
                "reconstruction_note": (
                    "Aligned from preserved April 3, 2026 runlog metadata. "
                    "This row reconstructs a historical routing artifact rather than re-executing the router."
                ),
            }
        )

    routing_accuracy = sum(routed_correct_flags) / len(routed_correct_flags) if routed_correct_flags else 0.0
    ambiguity_rate = sum(ambiguous_flags) / len(ambiguous_flags) if ambiguous_flags else 0.0
    mean_top_gap = statistics.mean(top_gaps) if top_gaps else 0.0

    summary = {
        "sample_file": str(args.sample_file),
        "runlog": str(args.runlog),
        "aligned_record_count": len(reconstructed),
        "missing_record_count": len(missing_query_ids),
        "missing_query_ids": missing_query_ids,
        "routing_accuracy": routing_accuracy,
        "ambiguity_rate": ambiguity_rate,
        "mean_top_gap": mean_top_gap,
        "note": (
            "This output reconstructs the preserved historical routing artifact from runlog metadata. "
            "It is not a fresh rerun and should not be treated as a reproduced manuscript benchmark."
        ),
    }

    artifact = {
        "metadata": summary,
        "records": reconstructed,
    }

    (args.output_dir / "reconstructed_routing_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "summary.md").write_text(render_markdown(summary), encoding="utf-8")

    print(f"aligned_record_count={len(reconstructed)}")
    print(f"routing_accuracy={routing_accuracy:.3f}")
    print(f"ambiguity_rate={ambiguity_rate:.3f}")
    print(f"mean_top_gap={mean_top_gap:.3f}")


if __name__ == "__main__":
    main()
