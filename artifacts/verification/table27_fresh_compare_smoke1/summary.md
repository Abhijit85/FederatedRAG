### Fresh Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `20`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- centralized_mode: `direct`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| federated | 0.225 | 0.035 | 1=0.200, 2=0.250 |
| centralized | 0.225 | 0.035 | 1=0.200, 2=0.250 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | +0.000 |
| SD diff | 0.000 |
| SE diff | 0.000 |
| t statistic | +0.000 |

This is a current-codebase paired rerun. It does not by itself establish provenance for the submitted paper's Table 27 path.
