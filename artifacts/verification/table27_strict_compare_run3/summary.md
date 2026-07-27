### Strict Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1,2,3,4,5`
- synapse_arm: `runtime_federated`
- centralized_arm: `historical_prototype_60`
- rounds: `1`
- client_count: `5`
- max_items: `5`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.164 | 0.015 | 1=0.140, 2=0.170, 3=0.170, 4=0.160, 5=0.180 |
| historical_prototype_60 | 0.598 | 0.036 | 1=0.580, 2=0.590, 3=0.660, 4=0.590, 5=0.570 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - historical_prototype_60) | -0.434 |
| SD diff | 0.036 |
| SE diff | 0.016 |
| t statistic | -26.610 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
