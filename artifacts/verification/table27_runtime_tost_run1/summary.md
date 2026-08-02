### Table 27 Runtime TOST

- sample_file: `/mnt/data1/achakr40/FederatedRAG/GSM8K_500_rebuttal_run/GSM8K_500_samples.json`
- sample_count: `100`
- seeds: `1,2,3,4,5`
- rounds: `1`
- client_count: `5`
- max_items: `5`
- runtime_included_tools: `mathqa`
- runtime_label_selector: `historical_cv_svm`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| runtime_federated | 0.928 | 0.013 | 1=0.920, 2=0.930, 3=0.940, 4=0.910, 5=0.940 |
| runtime_centralized_direct | 0.928 | 0.013 | 1=0.920, 2=0.930, 3=0.940, 4=0.910, 5=0.940 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | +0.000 |
| SD diff | 0.000 |
| SE diff | 0.000 |
| t statistic | +0.000 |

TOST margin: `±0.020`
90% CI: `[+0.0000, +0.0000]`
One-sided p-values: `p(> -m)=0.0000`, `p(< +m)=0.0000`
Verdict: `EQUIVALENT`

This is a fresh current-runtime TOST on the near-paper runtime path.
