### Strict Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1,2,3,4,5`
- synapse_arm: `runtime_federated`
- centralized_arm: `runtime_centralized_direct_sourceaware`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- runtime_include_training_artifacts: `True`
- runtime_training_sample_limit: `5`
- runtime_included_tools: `mathqa`
- runtime_label_selector: `historical_cv_svm`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.928 | 0.013 | 1=0.920, 2=0.930, 3=0.940, 4=0.910, 5=0.940 |
| runtime_centralized_direct_sourceaware | 0.334 | 0.013 | 1=0.320, 2=0.350, 3=0.320, 4=0.340, 5=0.340 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - runtime_centralized_direct_sourceaware) | +0.594 |
| SD diff | 0.019 |
| SE diff | 0.009 |
| t statistic | +68.136 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
