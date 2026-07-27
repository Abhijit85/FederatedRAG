### Table 27 Provenance-Faithful Fresh Comparator

- sample_count: `100`
- seeds: `1,2,3,4,5`
- client_count: `5`
- shard_seed: `0`
- classifier: `logreg(C=0.3)`

| Arm | Mean acc. | SD | Seeds |
| --- | ---: | ---: | --- |
| federated_hist_iid5 | 0.530 | 0.047 | 1=0.600, 2=0.540, 3=0.470, 4=0.520, 5=0.520 |
| centralized_hist_pool | 0.688 | 0.044 | 1=0.760, 2=0.690, 3=0.640, 4=0.670, 5=0.680 |

| Paired quantity | Value |
| --- | ---: |
| Mean diff (federated - centralized) | -0.158 |
| SD diff | 0.008 |
| SE diff | 0.004 |
| t statistic | -42.227 |

This comparator is provenance-faithful to the preserved April 3 paper-space GSM8K assets and the 5-IID-client design, but it reconstructs client membership by deterministic stratified sharding because the mirror does not preserve original client IDs.
