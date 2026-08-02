# ToolBench Retrieval-at-Scale

- Query source: `/mnt/data1/achakr40/FederatedRAG/external_datasets/toolbench/toolllama_G123_dfs_eval.json`
- Query format: `toolllama_eval`
- Query count: `250`
- Embedder: `jina-embeddings-v2-base-en`
- Search: `exact cosine`
- Top-K: `5`
- Query embeddings: `precomputed_amortized`

| Catalog size | Recall@5 | Recall | p50 ms | p95 ms | Index KB | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 32 | 89/250 | 0.356 | 27.0 | 27.4 | 414.8 | red flag: 32-tool recall < 0.95 |
| 100 | seed 1: 148/250, seed 2: 135/250, seed 3: 132/250 | 0.553 | 50.7 | 54.0 | 1167.7 | mean+-sd recall = 0.553 +- 0.034 |

## Provenance

- Base catalog provenance: Fixed 32-tool base assembled from the earliest locally observed query-relevant tools, because the paper-time 32-tool manifest is not preserved in this repo.
- Added-tool provenance: Local tool inventory merged from `/mnt/data1/achakr40/FederatedRAG/external_datasets/toolbench/toolllama_G123_dfs_eval.json` and `/mnt/data1/achakr40/FederatedRAG/external_datasets/toolbench/ToolBench-master/data_example/toolenv/tools`
- Hardware: Runs on the local machine using Jina embeddings and exact cosine search in Python.
