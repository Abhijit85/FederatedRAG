### Strict Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1,2,3,4,5`
- synapse_arm: `runtime_federated`
- centralized_arm: `historical_cv_logreg`
- rounds: `1`
- client_count: `5`
- max_items: `5`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.164 | 0.015 | 1=0.140, 2=0.170, 3=0.170, 4=0.160, 5=0.180 |
| historical_cv_logreg | 0.752 | 0.037 | 1=0.780, 2=0.760, 3=0.700, 4=0.790, 5=0.730 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - historical_cv_logreg) | -0.588 |
| SD diff | 0.048 |
| SE diff | 0.022 |
| t statistic | -27.297 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
