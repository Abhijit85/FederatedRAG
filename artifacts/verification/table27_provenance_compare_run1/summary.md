### Table 27 Provenance-Faithful Fresh Comparator

- sample_count: `100`
- seeds: `1,2,3,4,5`
- client_count: `5`
- shard_seed: `0`
- classifier: `logreg(C=3.0)`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| federated_hist_iid5 | 0.698 | 0.047 | 1=0.770, 2=0.690, 3=0.640, 4=0.700, 5=0.690 |
| centralized_hist_pool | 0.752 | 0.049 | 1=0.800, 2=0.710, 3=0.720, 4=0.810, 5=0.720 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | -0.054 |
| SD diff | 0.039 |
| SE diff | 0.017 |
| t statistic | -3.087 |

This comparator is provenance-faithful to the preserved April 3 paper-space GSM8K assets and the 5-IID-client design, but it reconstructs client membership by deterministic stratified sharding because the mirror does not preserve original client IDs.
