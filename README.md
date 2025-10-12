# FederatedRAG

## Overview
This repository provides a compendium-aware retrieval-augmented agent that can answer math and science questions. Structured knowledge about each tool is stored in JSON "compendiums". The agent indexes the usage scenarios from these compendiums with Jina embeddings, reranks candidates with a Lambda LLM, and routes the query to the most appropriate tool.

The project has been refactored toward **SYNAPSE** (Structured federated knowledge exchange), a hierarchical framework where clients, edge aggregators, and a central server collaborate by sharing curated knowledge artifacts instead of model weights. The SYNAPSE runtime now powers the main evaluation script by running a client → edge → server round, exporting a global knowledge snapshot, and using it to inject federated context into downstream tools.

The project includes two main tools:
- **MathQATool** – a RAG pipeline built on Jina embeddings, Jina reranker, and ChromaDB for mathematical word problems.
- **ScienceQATool** – a vision-language system that analyzes images and text to solve ScienceQA-style questions.

## Requirements
- Python 3.9 or later
- Environment variables in a `.env` file:
  ```env
  LAMDA_API_KEY=your_lambda_api_key
  JINA_API_KEY=your_jina_api_key
  SYNAPSE_SECRET=shared_encryption_secret
  # Optional privacy controls
  SYNAPSE_ENABLE_DP=1         # set to 0/false to disable differential privacy noise
  SYNAPSE_DP_EPSILON=1.0      # override epsilon (ignored if DP disabled)
  ```
- Python packages: `requests`, `numpy`, `pandas`, `chromadb`, `openai`, `datasets`, `pillow`, `python-dotenv`

## Data and Compendiums
The repository contains example resources used by the agent:
- `train_new.json` – training data for MathQATool.
- `mathqa_tools_compendium.json` and `scienceqa_tools_compendium.json` – structured tool descriptions.
- `mixed_queries.json` – evaluation set with math and science questions.

## Running the Agent
1. Install dependencies:
   ```bash
   pip install requests numpy pandas chromadb openai datasets pillow python-dotenv
   ```
2. Ensure the `.env` file and compendium JSON files are present.
3. Execute the evaluation script:
   ```bash
   python main.py
   ```
   This runs a SYNAPSE federation round (clients → edges → server), exports `synapse_global_snapshot.json`, and evaluates the federated agent on `mixed_queries.json`. Output is written to `evaluation_log.txt`.

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
