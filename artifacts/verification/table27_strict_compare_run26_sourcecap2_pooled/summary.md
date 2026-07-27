### Strict Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1,2,3,4,5`
- synapse_arm: `runtime_federated`
- centralized_arm: `runtime_centralized_direct_sourcecap2_pooled`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- runtime_include_training_artifacts: `False`
- runtime_training_sample_limit: `0`
- runtime_included_tools: `mathqa`
- runtime_label_selector: `historical_cv_svm`
- runtime_training_shard_mode: ``

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.928 | 0.013 | 1=0.920, 2=0.930, 3=0.940, 4=0.910, 5=0.940 |
| runtime_centralized_direct_sourcecap2_pooled | 0.328 | 0.018 | 1=0.300, 2=0.340, 3=0.320, 4=0.340, 5=0.340 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - runtime_centralized_direct_sourcecap2_pooled) | +0.600 |
| SD diff | 0.021 |
| SE diff | 0.009 |
| t statistic | +63.246 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
