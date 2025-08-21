# FederatedRAG

## Overview
This repository provides a compendium-aware retrieval-augmented agent that can answer math and science questions. Structured knowledge about each tool is stored in JSON "compendiums". The agent indexes the usage scenarios from these compendiums with Jina embeddings, reranks candidates with a Lambda LLM, and routes the query to the most appropriate tool.

The project includes two main tools:
- **MathQATool** – a RAG pipeline built on Jina embeddings, Jina reranker, and ChromaDB for mathematical word problems.
- **ScienceQATool** – a vision-language system that analyzes images and text to solve ScienceQA-style questions.

## Requirements
- Python 3.9 or later
- Environment variables in a `.env` file:
  ```env
  LAMDA_API_KEY=your_lambda_api_key
  JINA_API_KEY=your_jina_api_key
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
   This loads compendiums, builds a unified vector store, creates the MathQA and ScienceQA tools, and evaluates the agent on `mixed_queries.json`. Output is written to `evaluation_log.txt`.

## Project Structure
- `main.py` – loads compendiums, initializes `CompendiumAwareAgent`, and runs evaluation.
- `CompendiumAwareAgent.py` – builds the unified vector store and reranks tools using a Lambda LLM before routing.
- `vector_search.py` – Jina-based embedding client storing scenario vectors in memory.
- `math_qa.py` – RAG pipeline for math word problems using Jina embeddings and ChromaDB.
- `science_qa.py` – image and text reasoning with Lambda's vision-language models.
- `CompendiumBuilder.py` – generates structured compendiums and filters similar tools.
- `agenttools.py` – base classes and helper tools.

## License
MIT License

## Contact
For questions or contributions, reach out to **achakr40@asu.edu** or open an issue in this repository.
