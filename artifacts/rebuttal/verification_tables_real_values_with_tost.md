### Table B. Privacy–Utility Validation Points From Table 9 (Executed Checkpoints Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| No privacy | 0.935 | seeds = [0.21, 0.23, 0.23, 0.23, 0.31]; mean = 0.242; SD = 0.039 | measured on 100 GSM8K-derived routing samples per seed, 1 federation round |
| ε=0.5, λ=1.5 | 0.851 | seeds = [0.21, 0.23, 0.23, 0.23, 0.31]; mean = 0.242; SD = 0.039 | measured on 100 GSM8K-derived routing samples per seed, 1 federation round; identical to no-privacy row on current repo head |

### Table D. Cross-Model / Root-Cause Checkpoints From Tables 22–23 (Executed Checkpoints Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Table 22, τ-bench retail, SYNAPSE main | 0.453 | 100-task final run: mean reward = 0.370; mean r_actions = 0.417; mean user-turn proxy = 9.77 | measured on the completed 100-task tau-bench retail run on July 24, 2026 |

### Table E. Controls / Equivalence Checkpoints (Executed Checkpoints Only)

| Checkpoint | Paper anchor | Real measured values | Status |
| --- | ---: | --- | --- |
| Paired design seed values | parity claim | SYNAPSE = [0.21, 0.23, 0.23, 0.23, 0.31]; centralized = [0.21, 0.23, 0.23, 0.23, 0.31] | measured on 100 GSM8K-derived routing samples per seed |
| Paired TOST summary | ±0.03 margin | mean_diff = 0.000; all 5 paired differences = 0.000 | degenerate zero-variance case on current repo head, so the paired rerun does not provide an informative finite-SE TOST estimate |

### Diagnosis

The current caveats are useful because they expose the failure mode rather than hiding it. They also make the next debugging step narrower.

**Table B reflects two separate problems.** The privacy setting is not changing the output, as shown by the identical no-privacy and epsilon=0.5, lambda=1.5 rows. More importantly, the unprivatized baseline itself does not reproduce Table 9: 0.242 is far from the paper's 0.935 no-privacy anchor. That means the issue is not just DP wiring. The base routing and evaluation harness is not yet matching the manuscript setup even before privacy noise enters. Until the unprotected baseline lands near the paper value with real seed-to-seed variation, the privacy sweep is not interpretable.

**Table D is not yet a like-for-like rerun of Tables 22-23.** The completed artifact comes from tau-bench's tool-calling harness with gpt-4o-mini, not from the paper's SYNAPSE compendium and routing pipeline with the manuscript backbone. The reported outputs also differ from the manuscript metrics: mean reward, r_actions, and user-turn proxy are not the same quantities as task success, tool-call accuracy, and average turns in Tables 22-23. So this run should be treated as a harness check, not as a paper-comparable reproduction.

**Table E is downstream of the same Table B failure mode.** If the routing harness returns the same output across seeds and settings, the paired differences collapse to zero by construction. That makes the paired result uninformative rather than supportive.

The next step should therefore be narrow: rerun the unprivatized routing baseline alone, confirm that it lands near the Table 9 no-privacy anchor, and confirm that different seeds produce non-identical outputs. Once that is true, the privacy sweep, paired TOST, and tau-bench reconciliation become meaningful to revisit.

### Supporting Artifacts

| Artifact | Path |
| --- | --- |
| 100-query no-privacy routing summary | [artifacts/verification/routing_100_no_privacy/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_100_no_privacy/summary.json:1) |
| 100-query epsilon=0.5 routing summary | [artifacts/verification/routing_100_eps_0_5/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_100_eps_0_5/summary.json:1) |
| 100-query one-client routing summary | [artifacts/verification/routing_100_client_count_1/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_100_client_count_1/summary.json:1) |
| 100-query paired TOST input | [artifacts/verification/paired_tost_one_client_runtime_100.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/paired_tost_one_client_runtime_100.json:1) |
| Live 100-task tau-bench checkpoint | [external_datasets/tau_bench/runs_100/tool-calling-gpt-4o-mini-0.0_range_0-100_user-openai/gpt-4o-mini-llm_0724002801.json](/mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_100/tool-calling-gpt-4o-mini-0.0_range_0-100_user-openai/gpt-4o-mini-llm_0724002801.json:1) |
