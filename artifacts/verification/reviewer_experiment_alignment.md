### Reviewer Experiment Alignment (`C1`, `C3`, `C5`)

This note aligns the reviewer-requested experiments with the **submitted paper's claims** and the
**current branch's actual evidence**. It is intended to prevent accidental over-claiming in the rebuttal.

## Paper Anchor

The submitted paper's main claims are:

- the exchanged federated unit is a **typed compendium artifact**
- the type signature makes **per-field DP**, **schema-aware merge/conflict resolution**, and
  **cross-model transfer** well-defined operations
- the submitted-paper empirical anchors are the paper's own GSM8K, cross-model-transfer, and
  LiveBench tables, not the replacement diagnostics on this branch

Implication: reviewer-requested experiments are valid rebuttal evidence **only if they stay scoped to
the same mechanism claim**. They should not silently replace the paper's headline tables unless they
actually reproduce the paper configuration.

## Status Matrix

| ID | Reviewer intent | Paper-side claim being tested | Current branch evidence | Safe status | Safe rebuttal wording |
| --- | --- | --- | --- | --- | --- |
| `C1` | reranker capability ladder | asks whether reranker capacity alone explains the routing result | historical-paper ladder is **not reproduced** on this branch; best current support run is the replacement learned router at `0.780 ± 0.077` over 5 seeds; the 70B OpenRouter label classifier is worse at `0.668 ± 0.105` | `support-only` | "As a reviewer-requested sensitivity check, we evaluated stronger and weaker replacement routers on the current reconstruction. The resulting ladder does not reproduce the submitted paper's GSM8K anchor, so we report it only as a diagnostic; notably, a local learned router outperforms the 70B label-classifier baseline, indicating that model size alone does not explain the routing result." |
| `C3` | schema coarsening ablation | asks whether coarsening typed structure reduces utility | completed current-branch routing-only support run gives `full = 0.164 ± 0.022`, `merge_up = 0.164 ± 0.022`, `drop_annex = 0.164 ± 0.022` over 5 seeds | `do not claim schema cost from this branch` | "On the current reconstruction, the reviewer-requested coarsening variants are indistinguishable from full schema, which indicates that this branch is not expressing a separable schema effect. We therefore do not use this run to strengthen the submitted paper's schema-ablation claim." |
| `C5` | stronger-reranker LiveBench utility check | asks whether a stronger reranker improves utility on the disclosed hard subset where contraction fails | only a smoke run exists so far (`reasoning:spatial`, 1 example, both baseline and strong at `0/1`); no rebuttal-grade baseline reproduction of Table 20 on this branch | `not yet established` | "We implemented the reviewer-requested stronger-reranker utility runner, but do not yet treat it as rebuttal evidence because the current branch has not reproduced the paper's Table 20 baseline configuration. Until that baseline gate is met, we keep the LiveBench response anchored to the submitted paper's disclosed limitation." |

## Concrete Evidence

### `C1`

- Replacement learned router:
  - artifact: `artifacts/verification/gsm8k_learned_router_run1/summary.json`
  - mean: `0.780`
  - SD: `0.077`
  - per-seed: `0.86, 0.72, 0.80, 0.84, 0.68`
- 70B OpenRouter label classifier:
  - artifact: `artifacts/verification/gsm8k_openrouter_router_70b_run1/summary.json`
  - mean: `0.668`
  - SD: `0.105`
  - per-seed: `0.76, 0.56, 0.68, 0.78, 0.56`

Interpretation: the reviewer question is answered directionally, but not in paper-anchor form.
The current evidence supports "stronger inference model does not automatically recover the historical
paper number," not "the paper ladder is verified."

### `C3`

- Support run artifact:
  - `artifacts/verification/gsm8k_schema_support_run1/combined_summary.json`
- Results:
  - `full`: `0.164 ± 0.022`
  - `merge_up`: `0.164 ± 0.022`
  - `drop_annex`: `0.164 ± 0.022`

Interpretation: the branch-local artifact/retrieval pipeline is currently not sensitive to the requested
schema coarsenings. That is a valid diagnostic result, but it cannot be used to claim the ordered
schema costs described in the rebuttal draft.

### `C5`

- Smoke artifact:
  - `artifacts/verification/livebench_support_smoke/summary.json`
- Current status:
  - runner exists and executes
  - only smoke evidence exists so far
  - no rebuttal-grade four-task table has completed on this branch
  - baseline has **not** been shown to match the submitted paper's Table 20 configuration

Interpretation: this remains implementation-ready but evidentially incomplete.

## Recommended Rebuttal Position

- Keep the **main paper claims** anchored to the submitted manuscript.
- Treat `C1` as a **diagnostic sensitivity check**, not a replacement of the paper's GSM8K number.
- Treat `C3` as **non-supportive on the current branch** unless a different reconstruction begins to
  express the schema effect.
- Treat `C5` as **pending / not yet established** unless the Table 20 baseline gate is matched.

## One-Sentence Summary

`C1`, `C3`, and `C5` are valid reviewer-requested experiments, but on the current branch only `C1`
provides usable support evidence, while `C3` is a null diagnostic and `C5` is not yet rebuttal-grade.
