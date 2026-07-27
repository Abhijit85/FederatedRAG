### Table A. Core Reproduction Checkpoints

| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | SD | Expected SD | Conservative SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Table 2, typed condition, 0% contradiction | 0.92 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.031 | 0.046 |
| Table 2, typed condition, 20% contradiction | 0.89 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.035 | 0.053 |
| Table 2, typed condition, 40% contradiction | 0.86 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.039 | 0.059 |
| Table 2, typed condition, 60% contradiction | 0.81 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.044 | 0.066 |
| Table 14, TextGrad S=3 | 0.92 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.031 | 0.046 |
| Table 14, TextGrad S=1 | 0.89 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.035 | 0.053 |
| Table 14, TextGrad S=5 | 0.92 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.031 | 0.046 |
| Table 14, extractive centroid | 0.85 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.040 | 0.060 |
| Table 14, single-shot summarize | 0.87 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.038 | 0.057 |
| Table 14, no summarization | 0.78 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.047 | 0.070 |

### Table B. Privacy–Utility Validation Points From Table 9

| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean routing acc. | SD routing acc. | Expected SD | Conservative SD | Mean AUROC | Mean % clients < 0.10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No privacy | 0.935 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.012 | 0.019 | [calc] | [calc] |
| ε=2.0, λ=0.5 | 0.928 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.013 | 0.020 | [calc] | [calc] |
| ε=2.0, λ=1.0 | 0.914 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.014 | 0.021 | [calc] | [calc] |
| ε=2.0, λ=1.5 | 0.897 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.015 | 0.023 | [calc] | [calc] |
| ε=1.0, λ=0.5 | 0.909 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.014 | 0.022 | [calc] | [calc] |
| ε=1.0, λ=1.0 | 0.902 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.015 | 0.022 | [calc] | [calc] |
| ε=1.0, λ=1.5 | 0.881 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.016 | 0.024 | [calc] | [calc] |
| ε=0.5, λ=0.5 | 0.884 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.016 | 0.024 | [calc] | [calc] |
| ε=0.5, λ=1.0 | 0.866 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.017 | 0.026 | [calc] | [calc] |
| ε=0.5, λ=1.5 | 0.851 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.018 | 0.027 | [calc] | [calc] |

### Table C. ToolBench / mmFG-W2 Extension Checkpoint

| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Mean | SD | Expected SD | Conservative SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ToolBench overall, 250-query baseline | 0.728 | [real] | [real] | [real] | [calc] | [calc] | 0.032 | 0.048 |
| ToolBench extension to 600–750 queries, same protocol | 0.728 | [real] | [real] | [real] | [calc] | [calc] | 0.020 | 0.031 |

### Table D. Cross-Model / Root-Cause Checkpoints From Tables 22–23

| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Mean | SD | Expected SD | Conservative SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Table 22, τ-bench retail, SYNAPSE main | 0.453 | [real] | [real] | [real] | [calc] | [calc] | 0.035 | 0.053 |
| Table 22, τ-bench retail, centralized | 0.511 | [real] | [real] | [real] | [calc] | [calc] | 0.036 | 0.053 |
| Table 22, τ-bench retail, Fed-ICL | 0.301 | [real] | [real] | [real] | [calc] | [calc] | 0.033 | 0.049 |
| Table 23, LLaMA-3.2-3B delta vs main | -0.022 | [real] | [real] | [real] | [calc] | [calc] | 0.035 | 0.053 |
| Table 23, Mistral-7B delta vs main | -0.009 | [real] | [real] | [real] | [calc] | [calc] | 0.035 | 0.053 |
| Table 23, GPT-4o delta vs main | 0.085 | [real] | [real] | [real] | [calc] | [calc] | 0.035 | 0.053 |

### Table E. Controls / Equivalence Checkpoints

| Checkpoint | Paper anchor | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Seed 5 | Mean | SD | Expected SD | Conservative SD |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Field-preserving structured-but-untyped control | 0.86 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | 0.039 | 0.059 |
| Paired TOST mean difference (SYNAPSE - centralized) | 0.0 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | n/a | n/a |
| Paired TOST 90% CI containment margin | 0.03 | [real] | [real] | [real] | [real] | [real] | [calc] | [calc] | n/a | n/a |

| Seed | SYNAPSE acc. | Centralized-SYNAPSE acc. | Paired diff |
| --- | ---: | ---: | ---: |
| 1 | [real] | [real] | [real] |
| 2 | [real] | [real] | [real] |
| 3 | [real] | [real] | [real] |
| 4 | [real] | [real] | [real] |
| 5 | [real] | [real] | [real] |

| Quantity | Value |
| --- | ---: |
| Mean paired difference | [calc] |
| 90% CI lower | [calc] |
| 90% CI upper | [calc] |
| One-sided p-value: lower test | [calc] |
| One-sided p-value: upper test | [calc] |
| TOST p-value | [calc] |
| Margin | 0.030 |
| Equivalent at alpha=0.05 | [calc] |
