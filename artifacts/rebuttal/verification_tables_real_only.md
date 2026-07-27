### Table A. Core Reproduction Checkpoints (Measured-Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Table 2, typed condition, 0% contradiction | 0.92 | not yet measured on current repo head | no runnable contradiction-injection harness found |
| Table 2, typed condition, 20% contradiction | 0.89 | not yet measured on current repo head | no runnable contradiction-injection harness found |
| Table 2, typed condition, 40% contradiction | 0.86 | not yet measured on current repo head | no runnable contradiction-injection harness found |
| Table 2, typed condition, 60% contradiction | 0.81 | not yet measured on current repo head | no runnable contradiction-injection harness found |
| Table 14, TextGrad S=3 | 0.92 | not yet measured on current repo head | runnable in principle, not yet run as a 5-seed table job |
| Table 14, TextGrad S=1 | 0.89 | not yet measured on current repo head | runnable in principle, not yet run as a 5-seed table job |
| Table 14, TextGrad S=5 | 0.92 | not yet measured on current repo head | runnable in principle, not yet run as a 5-seed table job |
| Table 14, extractive centroid | 0.85 | not yet measured on current repo head | requires matched TextGrad ablation run |
| Table 14, single-shot summarize | 0.87 | not yet measured on current repo head | requires matched TextGrad ablation run |
| Table 14, no summarization | 0.78 | not yet measured on current repo head | requires matched TextGrad ablation run |

### Table B. Privacy–Utility Validation Points From Table 9 (Measured-Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| No privacy | 0.935 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on July 24, 2026 with `scripts/run_routing_verification.py`, 50 GSM8K-derived routing samples per seed, 1 federation round |
| ε=2.0, λ=0.5 | 0.928 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=2.0, λ=1.0 | 0.914 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=2.0, λ=1.5 | 0.897 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=1.0, λ=0.5 | 0.909 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=1.0, λ=1.0 | 0.902 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=1.0, λ=1.5 | 0.881 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=0.5, λ=0.5 | 0.884 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=0.5, λ=1.0 | 0.866 | not yet measured on current repo head | DP sweep runner not yet executed |
| ε=0.5, λ=1.5 | 0.851 | not yet measured on current repo head | DP sweep runner not yet executed |

### Table C. ToolBench / mmFG-W2 Extension Checkpoint (Measured-Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| ToolBench overall, 250-query baseline | 0.728 | not yet measured on current repo head | official ToolBench repo and eval JSON downloaded, but live execution still blocked by external tool-service credentials/runtime |
| ToolBench extension to 600–750 queries, same protocol | 0.728 reference | not yet measured on current repo head | same blocker as baseline |

### Table D. Cross-Model / Root-Cause Checkpoints From Tables 22–23 (Measured-Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Table 22, τ-bench retail, SYNAPSE main | 0.453 | one live sample task succeeded: reward = 1.0, Pass^1 = 1.0 on `task_id=0` | not table-comparable yet; full table needs 250 tasks × 3 seeds |
| Table 22, τ-bench retail, centralized | 0.511 | not yet measured on current repo head | no centralized τ-bench extraction yet |
| Table 22, τ-bench retail, Fed-ICL | 0.301 | not yet measured on current repo head | no Fed-ICL τ-bench extraction yet |
| Table 23, LLaMA-3.2-3B delta vs main | -0.022 | not yet measured on current repo head | cross-model τ-bench probe not yet run |
| Table 23, Mistral-7B delta vs main | -0.009 | not yet measured on current repo head | cross-model τ-bench probe not yet run |
| Table 23, GPT-4o delta vs main | 0.085 | not yet measured on current repo head | cross-model τ-bench probe not yet run |

### Table E. Controls / Equivalence Checkpoints (Measured-Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Field-preserving structured-but-untyped control | 0.86 | not yet measured on current repo head | control run not yet executed |
| Paired TOST mean difference (SYNAPSE - centralized) | parity claim | not yet computable | SYNAPSE seeds available for current 50-sample routing verifier: [0.24, 0.26, 0.26, 0.24, 0.32], but paired centralized seeds are missing |
| Paired TOST 90% CI containment margin | ±0.03 margin | not yet computable | cannot run TOST without paired centralized values |

### Supporting Measured Outputs

| Artifact | Real output |
| --- | --- |
| GSM8K-style routing verification | [artifacts/verification/routing/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing/summary.json:1) |
| τ-bench one-task live sample | [external_datasets/tau_bench/sample_results/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0723232107.json](/mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/sample_results/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0723232107.json:1) |
