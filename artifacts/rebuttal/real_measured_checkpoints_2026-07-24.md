### Real Measured Checkpoints (2026-07-24)

| Checkpoint | Real values | Source |
| --- | --- | --- |
| Routing, no privacy, 5 seeds x 50 queries | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | `artifacts/verification/routing/summary.json` |
| Routing, DP enabled, epsilon = 2.0, adaptive text noise = 0 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | `artifacts/verification/routing_dp_eps_2_0/summary.json` |
| Routing, DP enabled, epsilon = 1.0, adaptive text noise = 0 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | `artifacts/verification/routing_dp_eps_1_0/summary.json` |
| Routing, DP enabled, epsilon = 0.5, adaptive text noise = 0 | seeds = [0.24, 0.26, 0.26, 0.24, 0.32]; mean = 0.264; SD = 0.033 | `artifacts/verification/routing_dp_eps_0_5/summary.json` |
| τ-bench retail sample | task_id = 0; reward = 1.0; Pass^1 = 1.0; user_cost = 0.00012255 | `external_datasets/tau_bench/sample_results/tool-calling-gpt-4o-mini-0.0_range_0--1_user-openai/gpt-4o-mini-llm_0723232107.json` |

### Not Yet Measurable From Current Repo State

| Checkpoint family | Why still missing |
| --- | --- |
| Table A contradiction curve | no contradiction harness / typed-vs-untyped control runner exists in this checkout |
| Table C ToolBench extension | dataset is downloaded, but live execution still needs the original tool-service runtime/credentials |
| Table D full Tables 22–23 | only a one-task τ-bench sample is runnable from current setup; seeded full-table extraction is not wired |
| Table E paired TOST | `scripts/tost_equivalence.py` exists, but there are no paired centralized seed values in the repo to feed it |
