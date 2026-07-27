### Strict Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1,2,3,4,5`
- synapse_arm: `runtime_federated`
- centralized_arm: `historical_cv_svm`
- rounds: `1`
- client_count: `5`
- max_items: `5`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.164 | 0.015 | 1=0.140, 2=0.170, 3=0.170, 4=0.160, 5=0.180 |
| historical_cv_svm | 0.770 | 0.042 | 1=0.800, 2=0.790, 3=0.720, 4=0.810, 5=0.730 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - historical_cv_svm) | -0.606 |
| SD diff | 0.053 |
| SE diff | 0.024 |
| t statistic | -25.472 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
