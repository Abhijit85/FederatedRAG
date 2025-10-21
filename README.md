# FederatedRAG

## Overview
This repository provides a compendium-aware retrieval-augmented agent that can answer math and science questions. Structured knowledge about each tool is stored in JSON "compendiums". The agent indexes the usage scenarios from these compendiums with Jina embeddings, reranks candidates with a Lambda LLM, and routes the query to the most appropriate tool.

The project has been refactored toward **SYNAPSE** (Structured federated knowledge exchange), a hierarchical framework where clients, edge aggregators, and a central server collaborate by sharing curated knowledge artifacts instead of model weights. The SYNAPSE runtime now powers the main evaluation script by running a client → edge → server round, exporting a global knowledge snapshot, and using it to inject federated context into downstream tools.

The project includes two main tools:
- **MathQATool** – a RAG pipeline built on Jina embeddings, Jina reranker, and ChromaDB for mathematical word problems.
- **ScienceQATool** – a vision-language system that analyzes images and text to solve ScienceQA-style questions.

## Requirements
- Python 3.9 or later
- Python packages: `requests`, `numpy`, `pandas`, `chromadb`, `openai`, `datasets`, `pillow`, `python-dotenv`, `pymongo`

### Environment Variables
Create a `.env` file in the repository root:

```env
API_KEY=your_openrouter_api_key
VLM_MODEL=openai/gpt-4o-mini              # vision-capable model
JINA_API_KEY=your_jina_api_key
MONGO_URI=mongodb://localhost:27017       # or your Atlas connection string
MATHQA_COLLECTION=math_problems           # use 'vectors' if your Atlas collection has that name
SYNAPSE_SECRET=shared_encryption_secret
OPENROUTER_SITE_URL=https://your-app.example   # optional – used for OpenRouter rankings
OPENROUTER_SITE_NAME=FederatedRAG             # optional – used for OpenRouter rankings
# Optional privacy controls
SYNAPSE_ENABLE_DP=1
SYNAPSE_DP_EPSILON=1.0
```

The agent sends chat completions to `https://openrouter.ai/api/v1/chat/completions`. After editing `.env`, run `set -a; source .env; set +a` (or the equivalent in your shell) before launching any scripts.

### MongoDB Atlas Vector Search
MathQA retrieval expects a MongoDB Atlas vector index on the collection that stores embeddings (`math_problems` by default, or `vectors` if you prefer). Create the index from the Atlas UI (Search/Vector Search tab) with this JSON definition:

```json
{
  "name": "vector_index",
  "type": "vectorSearch",
  "definition": {
    "fields": [
      {
        "type": "vector",
        "path": "embedding",
        "numDimensions": 1536,
        "similarity": "cosine"
      }
    ]
  }
}
```

Ensure `numDimensions` matches the embedding size returned by Jina (`1536` for `jina-embeddings-v2-base-en`). Set `MATHQA_COLLECTION` in `.env` to the collection name that holds these documents. If the vector index is unavailable, the code automatically falls back to a manual cosine-similarity search.

## Data and Compendiums
The repository contains example resources used by the agent:
- `train_new.json` – training data for MathQATool.
- `mathqa_tools_compendium.json` and `scienceqa_tools_compendium.json` – structured tool descriptions.
- `mixed_queries.json` – evaluation set with math and science questions.

Regenerate the evaluation set to match your needs:

```bash
# Math + Science (default)
python scripts/build_mixed_queries.py --math-count 10 --science-count 10 --seed 123

# Math-only benchmark
python scripts/build_mixed_queries.py --datasets math --math-count 20

# Science-only benchmark
python scripts/build_mixed_queries.py --datasets science --science-count 15

# Math-only, non-IID slice (e.g., only "geometry" problems)
python scripts/build_mixed_queries.py --datasets math --math-count 20 --distribution noniid --math-category geometry

# Science-only, non-IID slice (e.g., focus on "earth science")
python scripts/build_mixed_queries.py --datasets science --science-count 20 --distribution noniid --science-topic "earth science"

# Mixed non-IID slice (Math: geometry, Science: physics)
python scripts/build_mixed_queries.py \
  --datasets math science \
  --math-count 15 --science-count 15 \
  --distribution noniid \
  --math-category geometry \
  --science-topic physics
```

`--datasets` accepts `math`, `science`, or both. Any `--*-count` option for an omitted dataset is ignored automatically. Use `--distribution noniid` to focus each dataset on a single dominant category/topic (`--math-category` or `--science-topic` can pin that choice). Outputs are written to `mixed_queries.json` unless you override `--output`.

## Running the Agent
1. Install dependencies:
   ```bash
   pip install requests numpy pandas chromadb openai datasets pillow python-dotenv
   ```
2. Ensure the `.env` file and compendium JSON files are present.
3. Optionally rebuild `mixed_queries.json` (see above).
4. Execute the evaluation script:
   ```bash
   python main.py
   ```
   This runs a SYNAPSE federation round (clients → edges → server), exports `synapse_global_snapshot.json`, and evaluates the federated agent on `mixed_queries.json`. Output is written to `evaluation_log.txt`.
5. Review the accuracy metrics printed near the end of the run. You can also recompute them offline:
   ```bash
   python scripts/eval_log_metrics.py
   ```

## Project Structure
- `main.py` – runs the SYNAPSE federation round, instantiates tools, and evaluates the federated agent.
- `CompendiumAwareAgent.py` – legacy single-node routing agent (kept for reference).
- `vector_search.py` – Jina-based embedding client storing scenario vectors in memory.
- `math_qa.py` – RAG pipeline for math word problems using Jina embeddings and ChromaDB.
- `science_qa.py` – image and text reasoning with Lambda's vision-language models.
- `CompendiumBuilder.py` – generates structured compendiums and filters similar tools.
- `agenttools.py` – base classes and helper tools.
- `synapse/` – SYNAPSE implementation (clients, edge aggregators, server orchestrator, knowledge abstractions, retrieval planner, privacy policies, runtime coordinator, and agent wrapper).

## License
MIT License

## Contact
For questions or contributions, reach out to **achakr40@asu.edu** or open an issue in this repository.

## Citation
If this project contributes to your research, please cite it:

```bibtex
@misc{FederatedRAG2025,
  author       = {Abhijit Chakraborty},
  title        = {FederatedRAG: Compendium-Aware Federated Retrieval-Augmented Generation},
  year         = {2025},
  howpublished = {\url{https://github.com/abhijit-chakraborty/FederatedRAG}}
}
```
