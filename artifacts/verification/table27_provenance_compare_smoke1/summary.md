### Table 27 Provenance-Faithful Fresh Comparator

- sample_count: `100`
- seeds: `1`
- client_count: `5`
- shard_seed: `0`
- classifier: `logreg(C=3.0)`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| federated_hist_iid5 | 0.770 | 0.000 | 1=0.770 |
| centralized_hist_pool | 0.800 | 0.000 | 1=0.800 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | -0.030 |
| SD diff | 0.000 |
| SE diff | 0.000 |
| t statistic | -inf |

This comparator is provenance-faithful to the preserved April 3 paper-space GSM8K assets and the 5-IID-client design, but it reconstructs client membership by deterministic stratified sharding because the mirror does not preserve original client IDs.
