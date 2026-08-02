### Table 27 Runtime Reproduction

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `50`
- seeds: `42,123,456,789,1024`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- runtime_included_tools: `mathqa`
- runtime_label_selector: `historical_cv_svm`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.884 | 0.050 | 42=0.840, 123=0.900, 456=0.840, 789=0.960, 1024=0.880 |
| runtime_centralized_direct | 0.884 | 0.050 | 42=0.840, 123=0.900, 456=0.840, 789=0.960, 1024=0.880 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | +0.000 |
| SD diff | 0.000 |
| SE diff | 0.000 |
| t statistic | +0.000 |

Headline sanity: FAIL (target 0.92 +- 0.02, mean tol 0.02, sd tol 0.02)

This reproduces the current-runtime comparator only. It does not establish provenance for the submitted historical Table 27 harness.
