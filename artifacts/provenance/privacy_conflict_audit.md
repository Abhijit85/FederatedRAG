# Privacy / Conflict-Handling Provenance Audit

Generated on 2026-07-23T15:47:06-07:00 from repository state `72c676403f9e877914142f0824a16545d1b317e7`.

## Reachable refs checked

| Ref |
| --- |
| `main` |
| `origin/Final-Repo` |
| `origin` |
| `origin/SyNAPSE` |
| `origin/SynapseLora` |
| `origin/gemini` |
| `origin/gsm8k-500-rebuttal-run` |
| `origin/main` |
| `origin/my-math-branch` |
| `origin/synapse-TextGrad` |

## Claim audit

| Claim | Source file | Current file evidence | Reachable-history evidence | Status |
| --- | --- | --- | --- | --- |
| Per-field clipping before DP noise | `synapse/privacy/policies.py` | no required mechanism terms found | no matching reachable revision | **not evidenced** |
| Cosine-based clustering / conflict logging in the edge aggregator | `synapse/edge/aggregator.py` | hits: `cluster`; no support-bearing mechanism terms found; disconfirming text: `future versions will incorporate semantic similarity checks` | no matching reachable revision | **not evidenced** |

## File lineage

### `synapse/privacy/policies.py`

| Commit | Subject |
| --- | --- |
| `26599273` | Add multi-step arithmetic benchmark dataset for evaluation |
| `06b18f6b` | Add differential privacy support and update privacy policy handling |
| `147bb102` | Refactor privacy policy handling and remove unused components |
| `3330b563` | First cimmit |

### `synapse/edge/aggregator.py`

| Commit | Subject |
| --- | --- |
| `7087aafc` | First textgrad committ |
| `147bb102` | Refactor privacy policy handling and remove unused components |
| `3330b563` | First cimmit |

## Rebuttal-safe wording

Use this only if it is accurate after author verification:

> We audited the anonymous repository history for the two implementation points implicated by the reviewer concern: `synapse/privacy/policies.py` and `synapse/edge/aggregator.py`. In the currently reachable history of the anonymous mirror, we do not find code evidence for per-field clipping before DP noise or cosine-clustering-based conflict logging. Accordingly, we do not rely on those stronger mechanism claims in the rebuttal unless the camera-ready artifact is tied to an author-verified commit that contains them.

If the authors identify a different provenance commit outside the current mirror, replace the sentence above with the exact commit hash and implementation location.
