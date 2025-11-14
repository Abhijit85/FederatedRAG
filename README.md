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
EVAL_MODEL=gpt-4o-mini                    # text model for Math/Science tools
MODEL_NAME=gpt-4o-mini                    # auto-syncs evaluation/test engines
TEXTGRAD_EVAL_ENGINE=gpt-4o               # LLM that supplies textual gradients
JINA_API_KEY=your_jina_api_key
MONGO_URI=mongodb://localhost:27017       # or your Atlas connection string
MATHQA_COLLECTION=math_problems           # use 'vectors' if your Atlas collection has that name
SYNAPSE_SECRET=shared_encryption_secret
OPENROUTER_SITE_URL=https://your-app.example   # optional – used for OpenRouter rankings
OPENROUTER_SITE_NAME=FederatedRAG             # optional – used for OpenRouter rankings
SYNAPSE_CLIENT_COUNT=4.    # modify to set the number of clients for federation.
# Optional privacy controls
SYNAPSE_ENABLE_DP=1
SYNAPSE_DP_EPSILON=1.0
TEXTGRAD_SAMPLE_WITH_REPLACEMENT=0
```

The agent sends chat completions to `https://openrouter.ai/api/v1/chat/completions`. After editing `.env`, run `set -a; source .env; set +a` (or the equivalent in your shell) before launching any scripts. Toggle `TEXTGRAD_SAMPLE_WITH_REPLACEMENT` to `1` when you want TextGrad’s mini-batches to sample with replacement (allowing repeated questions within an epoch); leave it `0` for the default shuffle-without-replacement behavior.

### Differential privacy & adaptive text noise

When `SYNAPSE_ENABLE_DP=1`, every client applies differential privacy before sharing artifacts. Numeric metadata is perturbed with Laplace noise controlled by `SYNAPSE_DP_EPSILON`, and (optionally) the textual content goes through adaptive token-level masking. Lower epsilon values increase both metadata noise and the aggressiveness of text masking.

#### Adaptive token-level Laplace noise

Prompt-reconstruction attacks succeed when the server sees long, literal excerpts of the client’s original prompts. To counter this, every artifact’s text is now replaced with a compact template (JSON describing the role/tool/skills) and each token in that template is scored for “saliency.” Whenever a token looks sensitive (contains digits, is long, is uppercase, etc.), Laplace noise is used to decide whether to mask part of it. Only the highest-saliency tokens get obfuscated, so the structured template stays useful while sensitive identifiers are scrambled. You can fine-tune the scoring and masking via environment variables:

```env
SYNAPSE_ADAPTIVE_TEXT_NOISE=1          # set to 0 to disable token masking
SYNAPSE_ADAPTIVE_DIGIT_WEIGHT=0.6      # contribution when a token contains digits
SYNAPSE_ADAPTIVE_LENGTH_WEIGHT=0.3     # contribution for long tokens (>=6 chars)
SYNAPSE_ADAPTIVE_UPPER_WEIGHT=0.2      # contribution for ALL-CAPS tokens
SYNAPSE_ADAPTIVE_TITLE_WEIGHT=0.1      # contribution when a token starts uppercase
SYNAPSE_ADAPTIVE_PROBABILITY_MULT=1.0  # scales the probability of masking
SYNAPSE_ADAPTIVE_DISTORT_MULT=1.0      # scales how many characters get replaced
```

Increase the weights or multipliers to mask more aggressively; decrease them for higher fidelity.

To further limit leakage, each client now shares condensed artifacts. You can tune the summariser via:

```env
SYNAPSE_ARTIFACT_MAX_CHARS=280        # max characters kept per artifact text
SYNAPSE_ARTIFACT_MAX_SENTENCES=1      # number of leading sentences preserved
SYNAPSE_ARTIFACT_INCLUDE_SKILLS=1     # set to 0 to omit the "skills" tag suffix
```

Lowering the character/sentence limits shortens every shared exemplar, mirroring Fed-ICL’s minimal-context approach.

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
# Math-only benchmark (alternate)
python scripts/build_mixed_queries.py --math-count 20 --science-count 0

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

### Client-Specific Evaluation Splits
Create per-client benchmark files (IID or non-IID) with the new generator:

```bash
# Two clients, IID sampling, 10 math + 10 science each
python scripts/build_client_datasets.py --clients 2 --math-per-client 10 --science-per-client 10

# Four clients, math-only IID splits
python scripts/build_client_datasets.py --clients 4 --datasets math --math-per-client 8
# Four clients, math-only IID splits (full-size example)
python scripts/build_client_datasets.py --clients 4 --datasets math --math-per-client 20 --science-per-client 0

# Three clients, non-IID categories/topics provided explicitly
python scripts/build_client_datasets.py \
  --clients 3 \
  --distribution noniid \
  --math-per-client 12 --science-per-client 12 \
  --math-categories algebra,geometry,probability \
  --science-topics physics,chemistry,earth\ science
```

Outputs land in `client_datasets/` by default (`summary.json` lists the allocation for reproducibility). Each `client_k_mixed_queries.json` can be fed directly into the evaluation pipeline.

## Running the Agent
1. Install dependencies:
   ```bash
   pip install requests numpy pandas chromadb openai datasets pillow python-dotenv
   ```
2. Ensure the `.env` file and compendium JSON files are present.
3. Optionally rebuild `mixed_queries.json` or generate client-specific datasets.
4. Execute the evaluation script (CLI options shown below):
   ```bash
   # Default run uses mixed_queries.json and the SYNAPSE_CLIENT_COUNT env (defaults to 2)
   python main.py

   # Spawn 4 federated clients and use a custom global benchmark
   python main.py --client-count 4 --test-file custom_mixed.json

   # Evaluate per-client datasets alongside the global run
   python main.py --client-count 4 --client-data-dir client_datasets
   ```
   This runs a SYNAPSE federation round (clients → edges → server), exports `synapse_global_snapshot.json`, and evaluates the federated agent on `mixed_queries.json`. Output is written to `evaluation_log.txt`.
5. When `--client-data-dir` is supplied, the script reports per-client accuracy, macro averages, and fairness dispersion (spread and σ) to highlight cross-client performance. Review the accuracy metrics printed near the end of the run. You can also recompute them offline:
   ```bash
   python scripts/eval_log_metrics.py
   ```

### Example Runs

- **Baseline federation + MathQA focus**
  ```bash
  # Use the bundled mixed benchmark (math+science)
  python main.py --client-count 2

  # Math-only evaluation after rebuilding a custom dataset
  python scripts/build_mixed_queries.py --datasets math --math-count 20 --output math_only.json
  python main.py --test-file math_only.json --client-count 4

  # Per-client math splits (requires client_datasets/*.json)
  python main.py --skip-global-eval --evaluate-clients --client-data-dir client_datasets
  ```

- **FedTextGrad-enabled run**
  ```bash
  # Optimise prompts with TextGrad, federate, then evaluate on math-only data
python scripts/run_fed_textgrad.py \
    --task BBH_object_counting \
    --client-count 4 \
    --rounds 1 \
    --aggregate-method summarization \
    --mixed-queries math_only.json
  ```

- **Heterogeneous TextGrad training (per-client data)**
  ```bash
  # Point each client to its own training set (e.g., MathQA, ScienceQA, BBH)
  python scripts/run_fed_textgrad.py \
      --task BBH_object_counting \
      --client-count 3 \
      --rounds 1 \
      --mixed-queries bbh_object_counting_eval_v3.json \
      --client-train-dir client_datasets/heterogeneous_train \
      --client-data-dir client_datasets/heterogeneous_eval \
      --evaluate-clients
  ```
Each TextGrad run appends its evaluation summary (central benchmark + any per-client datasets) to `evaluation_on_textgrad_log.txt`, so you can track metrics across runs without rerunning the script.
  Set `TEXTGRAD_EVAL_ENGINE`, `SYNAPSE_TEXTGRAD_TEST_ENGINE`, and other knobs in `.env` (or via CLI flags) to choose the LLMs that supply textual gradients and client-side inference.

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
