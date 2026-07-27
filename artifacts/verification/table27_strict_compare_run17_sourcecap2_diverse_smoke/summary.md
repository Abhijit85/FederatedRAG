### Strict Table 27 Comparison

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1`
- synapse_arm: `runtime_federated`
- centralized_arm: `runtime_centralized_direct_sourcecap2`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- runtime_include_training_artifacts: `True`
- runtime_training_sample_limit: `10`
- runtime_included_tools: `mathqa`
- runtime_label_selector: `historical_cv_svm`
- runtime_training_shard_mode: `client_stride`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.920 | 0.000 | 1=0.920 |
| runtime_centralized_direct_sourcecap2 | 0.920 | 0.000 | 1=0.920 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (runtime_federated - runtime_centralized_direct_sourcecap2) | +0.000 |
| SD diff | 0.000 |
| SE diff | 0.000 |
| t statistic | +0.000 |

This is a current-repo paired comparator with stricter historical-arm options. It is not automatically paper provenance.
