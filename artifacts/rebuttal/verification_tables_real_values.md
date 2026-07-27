### Table B. Privacy–Utility Validation Points From Table 9 (Executed Checkpoints Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| No privacy | 0.935 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=2.0, λ=0.5 | 0.928 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=2.0, λ=1.0 | 0.914 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=2.0, λ=1.5 | 0.897 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=1.0, λ=0.5 | 0.909 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=1.0, λ=1.0 | 0.902 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=1.0, λ=1.5 | 0.881 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=0.5, λ=0.5 | 0.884 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=0.5, λ=1.0 | 0.866 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |
| ε=0.5, λ=1.5 | 0.851 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | measured on 50 GSM8K-derived routing samples per seed, 1 federation round(s) |

### Table D. Cross-Model / Root-Cause Checkpoints From Tables 22–23 (Executed Checkpoints Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Table 22, τ-bench retail, SYNAPSE main | 0.453 | one live sample task succeeded: task_id = 0, reward = 1.0, Pass^1 = 1.0 | measured via tau-bench retail one-task sample; user_cost = 0.00012255; file = external_datasets/tau_bench/sample_results/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0723232107.json |

### Table E. Controls / Equivalence Checkpoints (Executed Checkpoints Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Paired TOST mean difference (SYNAPSE - centralized) | parity claim | not computed | invalid paired seed file: synapse and centralized vectors are exactly identical, which strongly suggests a self-copy artifact rather than a real paired comparison; centralized vector does not match artifacts/verification/centralized_routing/summary.json ([0.0, 0.0, 0.0, 0.0, 0.0] != [0.24, 0.26, 0.26, 0.24, 0.32]) |
| Paired TOST 90% CI containment margin | ±0.03 margin | not computed | resolve paired-input inconsistencies before running TOST |

### Supporting Artifacts

| Artifact | Path |
| --- | --- |
| Routing privacy sweep | [artifacts/verification/routing_privacy_sweep/combined_summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_privacy_sweep/combined_summary.json:1) |
| One-client routing summary | [artifacts/verification/routing_client_count_1/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_client_count_1/summary.json:1) |
| Latest τ-bench sample result | [external_datasets/tau_bench/sample_results/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0723232107.json](/mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/sample_results/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0723232107.json:1) |
| Paired TOST input | [artifacts/verification/paired_tost_one_client_runtime.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/paired_tost_one_client_runtime.json:1) |
