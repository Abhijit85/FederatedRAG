#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AuditTarget:
    claim: str
    path: str
    required_patterns: tuple[str, ...]
    support_patterns: tuple[str, ...] | None = None
    disconfirming_patterns: tuple[str, ...] = ()


TARGETS = (
    AuditTarget(
        claim="Per-field clipping before DP noise",
        path="synapse/privacy/policies.py",
        required_patterns=("clip", "clamp", "bound"),
    ),
    AuditTarget(
        claim="Cosine-based clustering / conflict logging in the edge aggregator",
        path="synapse/edge/aggregator.py",
        required_patterns=("cosine", "cluster", "conflict", "similarity_threshold", "0.85"),
        support_patterns=("cosine", "conflict", "similarity_threshold", "0.85"),
        disconfirming_patterns=("future versions will incorporate semantic similarity checks",),
    ),
)


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True)


def current_file_text(path: str) -> str:
    return (REPO_ROOT / path).read_text()


def file_history(path: str) -> list[str]:
    output = run_git(["log", "--follow", "--format=%H", "--", path])
    return [line.strip() for line in output.splitlines() if line.strip()]


def branch_refs() -> list[str]:
    output = run_git(["for-each-ref", "--format=%(refname:short)", "refs/heads", "refs/remotes"])
    return [line.strip() for line in output.splitlines() if line.strip()]


def rev_file_text(rev: str, path: str) -> str | None:
    proc = subprocess.run(
        ["git", "show", f"{rev}:{path}"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return None
    return proc.stdout


def find_matches(text: str, patterns: Iterable[str]) -> list[str]:
    hits: list[str] = []
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            hits.append(pattern)
    return hits


def short_commit(commit: str) -> str:
    return commit[:8]


def commit_subject(commit: str) -> str:
    return run_git(["show", "-s", "--format=%s", commit]).strip()


def earliest_commit_with_patterns(path: str, patterns: tuple[str, ...]) -> dict[str, str] | None:
    commits = list(reversed(file_history(path)))
    for commit in commits:
        text = rev_file_text(commit, path)
        if text is None:
            continue
        if find_matches(text, patterns):
            return {"commit": commit, "subject": commit_subject(commit)}
    return None


def audit_target(target: AuditTarget) -> dict:
    text = current_file_text(target.path)
    current_hits = find_matches(text, target.required_patterns)
    current_disconfirming = find_matches(text, target.disconfirming_patterns)
    support_patterns = target.support_patterns or target.required_patterns
    current_support_hits = find_matches(text, support_patterns)
    commits = file_history(target.path)

    matching_revisions = []
    for commit in commits:
        rev_text = rev_file_text(commit, target.path)
        if rev_text is None:
            continue
        matched = find_matches(rev_text, target.required_patterns)
        support_hits = find_matches(rev_text, support_patterns)
        if support_hits:
            matching_revisions.append(
                {
                    "commit": commit,
                    "subject": commit_subject(commit),
                    "matched_patterns": matched,
                    "support_hits": support_hits,
                }
            )

    return {
        "claim": target.claim,
        "path": target.path,
        "current_required_hits": current_hits,
        "current_support_hits": current_support_hits,
        "current_disconfirming_hits": current_disconfirming,
        "history_commits": [
            {"commit": commit, "subject": commit_subject(commit)}
            for commit in commits
        ],
        "matching_revisions": matching_revisions,
        "supports_claim_in_reachable_history": bool(matching_revisions),
    }


def render_markdown(report: dict) -> str:
    lines: list[str] = []
    lines.append("# Privacy / Conflict-Handling Provenance Audit")
    lines.append("")
    lines.append(f"Generated on {report['generated_on']} from repository state `{report['head']}`.")
    lines.append("")
    lines.append("## Reachable refs checked")
    lines.append("")
    lines.append("| Ref |")
    lines.append("| --- |")
    for ref in report["refs_checked"]:
        lines.append(f"| `{ref}` |")
    lines.append("")
    lines.append("## Claim audit")
    lines.append("")
    lines.append("| Claim | Source file | Current file evidence | Reachable-history evidence | Status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in report["targets"]:
        current = (
            "hits: " + ", ".join(f"`{x}`" for x in item["current_required_hits"])
            if item["current_required_hits"]
            else "no required mechanism terms found"
        )
        if item["current_required_hits"] and not item["current_support_hits"]:
            current += "; no support-bearing mechanism terms found"
        if item["current_disconfirming_hits"]:
            current += "; disconfirming text: " + ", ".join(f"`{x}`" for x in item["current_disconfirming_hits"])
        history = (
            f"{len(item['matching_revisions'])} matching revision(s)"
            if item["matching_revisions"]
            else "no matching reachable revision"
        )
        status = "supported" if item["supports_claim_in_reachable_history"] else "not evidenced"
        lines.append(
            f"| {item['claim']} | `{item['path']}` | {current} | {history} | **{status}** |"
        )
    lines.append("")
    lines.append("## File lineage")
    lines.append("")
    for item in report["targets"]:
        lines.append(f"### `{item['path']}`")
        lines.append("")
        lines.append("| Commit | Subject |")
        lines.append("| --- | --- |")
        for commit in item["history_commits"]:
            lines.append(f"| `{short_commit(commit['commit'])}` | {commit['subject']} |")
        lines.append("")
    lines.append("## Rebuttal-safe wording")
    lines.append("")
    lines.append("Use this only if it is accurate after author verification:")
    lines.append("")
    lines.append("> We audited the anonymous repository history for the two implementation points implicated by the reviewer concern: `synapse/privacy/policies.py` and `synapse/edge/aggregator.py`. In the currently reachable history of the anonymous mirror, we do not find code evidence for per-field clipping before DP noise or cosine-clustering-based conflict logging. Accordingly, we do not rely on those stronger mechanism claims in the rebuttal unless the camera-ready artifact is tied to an author-verified commit that contains them.")
    lines.append("")
    lines.append("If the authors identify a different provenance commit outside the current mirror, replace the sentence above with the exact commit hash and implementation location.")
    lines.append("")
    return "\n".join(lines)


def build_report() -> dict:
    head = run_git(["rev-parse", "HEAD"]).strip()
    refs = branch_refs()
    return {
        "generated_on": subprocess.check_output(["date", "-Iseconds"], text=True).strip(),
        "head": head,
        "refs_checked": refs,
        "targets": [audit_target(target) for target in TARGETS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit whether implementation claims are evidenced in reachable git history.")
    parser.add_argument(
        "--output-dir",
        default="artifacts/provenance",
        help="Directory for JSON and Markdown audit outputs.",
    )
    args = parser.parse_args()

    output_dir = REPO_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_report()
    json_path = output_dir / "privacy_conflict_audit.json"
    md_path = output_dir / "privacy_conflict_audit.md"

    json_path.write_text(json.dumps(report, indent=2) + "\n")
    md_path.write_text(render_markdown(report))

    print(f"Wrote {os.path.relpath(json_path, REPO_ROOT)}")
    print(f"Wrote {os.path.relpath(md_path, REPO_ROOT)}")


if __name__ == "__main__":
    main()
