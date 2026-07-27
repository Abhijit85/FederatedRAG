# Table 22/23 Final Audit

These tables separate three things that had been conflated earlier:
1. the manuscript-reported values,
2. the current stock `tau-bench` rerun path that is available in this repo, and
3. the targeted diagnostic reruns on the problematic retail tasks.

## Table 22. Retail Main Scoreboard: Manuscript vs Current Measured Repo Path

| Row | Task success | Tool-call accuracy | Avg. turns | Provenance |
| --- | ---: | ---: | ---: | --- |
| SYNAPSE (paper Table 22) | 0.453 ± 0.023 | 0.540 | 5.8 | Manuscript |
| Centralized (paper Table 22) | 0.511 ± 0.020 | 0.608 | 5.5 | Manuscript |
| Fed-ICL (paper Table 22) | 0.301 ± 0.018 | 0.432 | 6.7 | Manuscript |
| Current stock repo path (`gpt-4o-mini`, 100-task audit) | 0.340 | 0.250 all-row / 0.417 covered-row | 9.77 user / 16.25 assistant | [`runs_100/.../gpt-4o-mini-llm_0724002801.json`](</mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_100/tool-calling-gpt-4o-mini-0.0_range_0-100_user-openai/gpt-4o-mini-llm_0724002801.json:1>) |

Interpretation: the currently available stock repo path is not a reproduction of the manuscript's SYNAPSE retail row. The gap is too large in both scale and turn accounting, so this remains a provenance/evaluator mismatch rather than a small arithmetic discrepancy.

## Table 23. Internal Consistency Check Between Paper Table 22 and Paper Table 23

| Quantity for the SYNAPSE / LLaMA-3.1-8B (main) row | Table 22 | Table 23 | Status |
| --- | ---: | ---: | --- |
| Task success | 0.453 ± 0.023 | 0.453 ± 0.023 | Matches exactly |
| Tool-call accuracy | 0.540 | 0.631 | Mismatch |
| Avg. turns | 5.8 | 5.4 | Mismatch |

Interpretation: because task success and its uncertainty match exactly while tool-call accuracy and turns do not, the most likely failure mode is metric extraction / aggregation mismatch on the same underlying run rather than two genuinely different experiments being mislabeled as one.

## Targeted Diagnostic Slice For The Main Failure Cluster

| Task id | Latest targeted result | Notes |
| --- | ---: | --- |
| 9 | 1.0 | Fixed after exchange-selection and variant-visibility changes |
| 10 | 1.0 | Fixed after refund-policy / human-transfer handling change |
| 20 (`gpt-4o-mini` user simulator) | 0.0 | Failed on authentication / preference-following branches |
| 20 (`gpt-4o` user simulator) | 1.0 | Fixed after auth steering + wearable-size preservation + gift-card preference patch |

Relevant targeted artifacts:
- [`runs_patchcheck_diag4/.../gpt-4o-mini-llm_0724113003.json`](</mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_patchcheck_diag4/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0724113003.json:1>)
- [`runs_patchcheck_task20_fix4/.../gpt-4o-mini-llm_0724114503.json`](</mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_patchcheck_task20_fix4/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0724114503.json:1>)
- [`runs_patchcheck_task20_fix6_user4o/.../gpt-4o-llm_0724114816.json`](</mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_patchcheck_task20_fix6_user4o/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-llm_0724114816.json:1>)
- [`artifacts/rebuttal/table22_table23_audit.md`](</mnt/data1/achakr40/FederatedRAG/artifacts/rebuttal/table22_table23_audit.md:1>)
