# Reviewer Concern: Claimed Privacy / Conflict-Handling Mechanism vs. Anonymous Mirror

This concern should be answered with provenance, not rhetoric. The repository evidence currently supports a narrow, defensible response:

| Question | Evidence from the anonymous mirror | Safe conclusion |
| --- | --- | --- |
| Does the current mirror implement per-field clipping before DP noise? | [`synapse/privacy/policies.py`](/mnt/data1/achakr40/FederatedRAG/synapse/privacy/policies.py:86) applies Laplace noise directly to numeric metadata and payload values. The reachable history for that file starts with commit `06b18f6b` ("Add differential privacy support and update privacy policy handling") and does not show clipping terms. | Do not claim clipping from the current mirror. |
| Does the current mirror implement cosine-based clustering / conflict logging in the edge aggregator? | [`synapse/edge/aggregator.py`](/mnt/data1/achakr40/FederatedRAG/synapse/edge/aggregator.py:31) deduplicates by exact artifact signature and explicitly says semantic similarity checks are for a future version. The reachable history for that file does not show clustering/conflict logic. | Do not claim cosine-clustering conflict handling from the current mirror. |
| Is a benign version mismatch still possible? | Yes, but only if the authors can identify the exact commit or local state that produced Tables 2, 9, and 11. The current anonymous mirror does not itself evidence that implementation. | Ask for the producing commit hash directly and cite it precisely if it exists. |

## Rebuttal-safe response

> We checked this concern at the implementation level rather than responding rhetorically. In the currently reachable history of the anonymous mirror, `synapse/privacy/policies.py` applies Laplace perturbation directly to numeric metadata/payload fields, and `synapse/edge/aggregator.py` performs signature-based deduplication rather than cosine-clustering-based conflict logging. Therefore, in the rebuttal we avoid stronger wording that would imply those mechanisms are evidenced by the present mirror. The most likely benign explanation is a provenance mismatch between the mirror and the artifact-producing code path; we are therefore verifying the exact commit that produced Tables 2, 9, and 11, and if that commit contains the clipping and clustering logic we will cite that implementation precisely rather than state the mechanism abstractly.

## Author action items

| Priority | Action | Why it matters |
| --- | --- | --- |
| 1 | Identify the exact commit, branch, or local snapshot that produced Tables 2, 9, and 11. | This is the only path that converts the issue into a provenance mismatch instead of a mechanism gap. |
| 2 | If that commit exists, cite the exact hash and source locations in the rebuttal/camera-ready. | A reviewer can verify it quickly, which materially lowers risk. |
| 3 | If no such commit exists, narrow Theorem 1 / mechanism wording to match the shipped implementation. | This avoids overclaiming a guarantee the visible code does not support. |

## Audit artifact

Run:

```bash
python3 scripts/audit_claim_provenance.py
```

This writes:

| Artifact | Purpose |
| --- | --- |
| [`artifacts/provenance/privacy_conflict_audit.json`](/mnt/data1/achakr40/FederatedRAG/artifacts/provenance/privacy_conflict_audit.json:1) | Machine-readable audit of refs, file lineage, and keyword hits. |
| [`artifacts/provenance/privacy_conflict_audit.md`](/mnt/data1/achakr40/FederatedRAG/artifacts/provenance/privacy_conflict_audit.md:1) | Human-readable provenance summary and rebuttal-safe wording. |
