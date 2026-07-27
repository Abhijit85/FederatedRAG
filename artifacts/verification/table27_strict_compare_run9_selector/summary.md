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
| runtime_federated | 0.928 | 0.013 | 1=0.920, 2=0.930, 3=0.940, 4=0.910, 5=0.940 |
| historical_cv_svm | 0.770 | 0.042 | 1=0.800, 2=0.790, 3=0.720, 4=0.810, 5=0.730 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - historical_cv_svm) | +0.158 |
| SD diff | 0.054 |
| SE diff | 0.024 |
| t statistic | +6.538 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
