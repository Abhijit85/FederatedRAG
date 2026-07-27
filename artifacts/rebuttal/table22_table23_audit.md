# Table 22/23 Audit

This audit uses the manuscript-relevant metrics from a `tau-bench` result file.

- Results file: `/mnt/data1/achakr40/FederatedRAG/external_datasets/tau_bench/runs_100/tool-calling-gpt-4o-mini-0.0_range_0-100_user-openai/gpt-4o-mini-llm_0724002801.json`
- Paper Table 22 targets: task success, tool-call accuracy, average turns.
- Current checked file is a stock `tau-bench` `gpt-4o-mini` run, not a SYNAPSE retail run.

## Extracted Metrics

| Metric | Value |
| --- | ---: |
| Row count | 100 |
| Task success (mean reward) | 0.340 |
| Task success rate (`reward == 1`) | 0.340 |
| Tool-call accuracy (all rows) | 0.250 |
| Tool-call accuracy (rows with `r_actions`) | 0.417 |
| Reward-info presence coverage | 0.920 |
| Missing `r_actions` rows | 40 |
| `r_actions` coverage | 0.600 |
| Avg. user turns | 9.77 |
| Avg. assistant turns | 16.25 |
| Avg. tool turns | 7.48 |

## Comparison to Paper Scale

| Quantity | Paper Table 22 | Current stock path |
| --- | ---: | ---: |
| Task success | 0.453 / 0.511 / 0.301 | 0.340 |
| Tool-call accuracy | 0.540 / 0.608 / 0.432 | 0.250 all-row, 0.417 covered-row |
| Avg. turns | 5.8 / 5.5 / 6.7 | 9.77 user, 16.25 assistant |

Interpretation: this confirms evaluator/provenance mismatch rather than a small arithmetic discrepancy.
