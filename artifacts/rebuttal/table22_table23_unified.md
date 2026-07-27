# Unified Table 22/23 Regeneration

This artifact regenerates both table views from one source run artifact and one extraction pipeline.

- Run label: `stock tau-bench path unified regeneration`
- Results file: `/mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_100/tool-calling-gpt-4o-mini-0.0_range_0-100_user-openai/gpt-4o-mini-llm_0724002801.json`
- Single source of truth: one per-task JSON artifact containing `reward`, `reward_info`, and `traj`.

## Canonical Extracted Metrics

| Metric | Value |
| --- | ---: |
| Row count | 100 |
| Task success (mean reward) | 0.340 |
| Tool-call accuracy, zero-filled missing rows | 0.250 |
| Tool-call accuracy, present rows only | 0.417 |
| Reward-info presence coverage | 0.920 |
| Missing `r_actions` rows | 40 |
| `r_actions` coverage | 0.600 |
| Avg. user turns | 9.77 |
| Avg. assistant turns | 16.25 |
| Avg. user+assistant turns | 26.02 |

## Regenerated Table Views From The Same Source Run

| View | Task success | Tool-call accuracy | Turns convention |
| --- | ---: | ---: | ---: |
| Table 22-style | 0.340 | 0.250 | 9.77 user-only / 16.25 assistant-only |
| Table 23-style | 0.340 | 0.417 | 26.02 user+assistant |

## Procedural Conclusion

Both table views above are generated from the same source artifact. Any remaining discrepancy is therefore attributable to post-processing convention choices, not to using different underlying runs.
