# GSM8K Paper-Claim Provenance Audit

## Claim

Submitted-paper anchor under audit: `0.92 +- 0.02` GSM8K routing accuracy.

Preserved manuscript lines:
- `(§5) showing 0.92±0.02 routing accuracy on GSM8k, statistically indistinguishable from centralized`
- `extractive summarization drops routing accuracy 0.92 → 0.85 (Tab. 3); removing merge while keeping`
- `achieves 0.92 ± 0.02 on GSM8k (5 IID clients), statistically indistinguishable from Centralized-`
- `non-IID splits, GSM8k drops only 0.96 → 0.92 and BBH benchmarks drop ≤ 2 pts; deduplication`
- `four LLM families on GSM8k – LLaMA-3.1-8B (0.92, native), LLaMA-3.2-3B (0.90), Mistral-7B`
- `or weight sharing. On 5-seed GSM8k: SYNAPSE 0.92, FedLoRA 0.89, C-FedRAG 0.84, Fed-ICL`
- `log drops routing accuracy 0.92 → 0.49. To verify this is a stable component effect rather than`
- `from 0.85 over-merges genuinely-distinct scenarios and hurts routing accuracy (0.92 → 0.87) – and`

## Preserved Evidence

| Artifact | Metric | Value | Status | Note |
| --- | --- | --- | --- | --- |
| Submitted paper anchor | GSM8K routing accuracy | 0.920 +- 0.020 | claimed | `/home/ad.asu.edu/achakr40/.codex/attachments/0026c74a-9bd5-4306-aa47-9461b2188243/pasted-text.txt`; Reported in the submitted manuscript as the headline IID GSM8K result. |
| Current live math-only verifier | 100-query, 5-seed routing accuracy under current unified runtime | 0.328 +- 0.018 | measured | `/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_math_only_paperlike_100/summary.json`; Uses the current verifier after restricting the runtime to math-only artifacts. |
| Historical paper-mode prototype reconstruction | 100-query, 5-seed local prototype routing accuracy | 0.602 +- 0.018 | measured | `/mnt/data1/achakr40/FederatedRAG/artifacts/verification/gsm8k_paper_mode_local_run3/summary.json`; Six historical paper-time labels with exemplar-enriched prototypes. |
| Best preserved local recovery | 100-query, 5-seed local reconstruction accuracy | 0.770 +- 0.042 | measured | `artifacts/verification/gsm8k_paper_recovery_sweep_run2/cv_svm/summary.json`; Cross-validated SVM over the preserved six-label paper-time universe. |
| Historical April 3 runlog evolution | 500-record routing accuracy in preserved runlog evolution artifact | 0.998 | measured | `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_routing_evolution.json`; This preserved runlog is nearly perfect, so it is not the same benchmark object as the submitted 0.92 +- 0.02 table claim. |

## Recovered Historical Structure

The preserved compendium evolution still exposes the paper-time six-scenario GSM8K universe (`n=6`):
- `Algebraic Word Problem Solver`
- `Financial and Banking Calculator`
- `General Logic and Counting`
- `Geometry: Shapes and Measurement`
- `Percentage and Proportion Solver`
- `Work, Rate, and Time Analyzer`

## Current Runtime Check

`/mnt/data1/achakr40/FederatedRAG/synapse/clients/unified_client.py` now includes `SYNAPSE_INCLUDED_TOOLS` support: env gate=True, math guard=True, science guard=True.
This fixes one real drift in the live verifier by allowing math-only artifact emission, but it does not close the full gap to the submitted paper number.

## Conclusion

- The anonymous mirror preserves the paper-time six-scenario GSM8K artifact space.
- The current repo head now supports math-only routing runs via SYNAPSE_INCLUDED_TOOLS, which removes one real runtime drift caused by mixed math/science artifact emission.
- Even after that fix, the current live verifier measures 0.328 +- 0.018 on the 100-query reconstruction, far below the submitted 0.92 +- 0.02.
- The strongest local historical reconstruction recovered from preserved artifacts is 0.770 +- 0.042 using cv_svm; that is materially better than the live verifier, but still well below the submitted anchor.
- The preserved April 3 runlog evolution artifact reports 0.998 routing accuracy over 500 records, which is too high to be the same evaluation object as the submitted 0.92 +- 0.02 benchmark.
- Therefore the repo preserves enough state to recover the historical six-label universe, but not the exact historical scorer / benchmark / label-generation path that produced the submitted GSM8K headline.

## Rebuttal-Safe Wording

> The anonymous mirror preserves the paper-time six-scenario GSM8K artifact space and supports local reconstruction up to 0.770 +- 0.042 over 5 seeds, after fixing one runtime drift in the current verifier (math/science artifact mixing). However, the mirror does not preserve the exact historical evaluation path that yielded the submitted 0.92 +- 0.02 GSM8K routing-accuracy result, so we do not present that headline as reproduced from the current repo state.
