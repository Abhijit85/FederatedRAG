### Fresh Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- centralized_mode: `direct`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| federated | 0.164 | 0.015 | 1=0.140, 2=0.170, 3=0.170, 4=0.160, 5=0.180 |
| centralized | 0.164 | 0.015 | 1=0.140, 2=0.170, 3=0.170, 4=0.160, 5=0.180 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | +0.000 |
| SD diff | 0.000 |
| SE diff | 0.000 |
| t statistic | +0.000 |

This is a current-codebase paired rerun. It does not by itself establish provenance for the submitted paper's Table 27 path.
