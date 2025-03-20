# FediRAG: Federated Retrieval-Augmented Generation with Structured Relational Embeddings and Hybrid Search

## Overview
FediRAG is a federated retrieval-augmented generation (RAG) framework that integrates **structured relational embeddings** with **hybrid retrieval** and **secure federated learning**. It enables privacy-preserving knowledge integration while leveraging both unstructured and structured relational knowledge representations.

## Features
- **Federated Knowledge Graph Embeddings**: Utilizes the **RelatE model** for structured embeddings, ensuring fine-grained relational pattern learning.
- **Hybrid Retrieval & Reranking**: Combines **LLama3-based dense text embeddings** with **RelatE-based structured embeddings** for accurate and contextually aware retrieval.
- **Privacy-Preserving Aggregation**: Implements **secure masking, differential privacy, and encrypted model updates** to ensure data confidentiality.
- **Cross-Modal Reranking**: Uses a **transformer-based cross-encoder** to jointly optimize textual and relational signals for relevance.
- **Retrieval-Augmented Generation (RAG)**: Generates responses enriched with structured knowledge, enhancing contextual understanding and reasoning.

## Architecture
FediRAG consists of the following components:

1. **Federated RelatE-Based Knowledge Graph Embeddings**  
   - Each client trains a **local RelatE model** on private knowledge graph data.
   - Secure aggregation combines relational embeddings without sharing raw data.
   
2. **Hybrid Search and Reranking Module**  
   - **Textual Encoding**: Encodes queries using **Llama 3** to generate dense embeddings.
   - **Relational Encoding**: Uses **RelatE embeddings** to capture structured dependencies.
   - **Hybrid Candidate Retrieval**: Combines dense and structured retrieval using weighted rank fusion.
   - **Cross-Modal Reranking**: Ensures retrieved candidates are both **semantically similar** and **relationally coherent**.

3. **Federated Secure Aggregation**  
   - Implements **secure masking** and **differential privacy (DP)** for encrypted model updates.
   - Utilizes **pairwise random masking** and **secure multi-party computation** to protect sensitive knowledge graphs.
   
4. **Retrieval-Augmented Generation (RAG) with LLMs**  
   - Retrieved context is fed into **Llama 3**, enhancing text generation with structured knowledge.

## Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/FediRAG.git
cd FediRAG

# Install dependencies
pip install -r requirements.txt
```

## Usage
### 1. Train Federated Clients
```python
from model import RelatEModel
from dataloader import load_data
from fediRAG import FediRAGClient

num_clients = 5
data_splits = load_data(num_clients)
clients = [FediRAGClient(RelatEModel(), data=data_splits[i], client_id=i, num_clients=num_clients) for i in range(num_clients)]

for client in clients:
    local_updates = client.train_local_model()
```

### 2. Securely Aggregate Updates
```python
from fediRAG import FediRAGServer
server = FediRAGServer(num_clients=num_clients, global_model=RelatEModel())
aggregated_updates = server.aggregate_updates([client.get_model_updates() for client in clients])
server.update_global_model(aggregated_updates)
```

### 3. Hybrid Retrieval and Generation
```python
query_text = "What are the key impacts of AI in healthcare?"
response = server.generate_response(query_text, knowledge_base)
print(response)
```

## Privacy and Security
- **Local Differential Privacy (DP)**: Clients add noise to updates before sharing.
- **Secure Multi-Party Computation (MPC)**: Ensures encrypted data sharing.
- **Private Query Obfuscation**: Uses dummy queries and encryption for enhanced privacy.
- **Secure Reranking**: Runs cross-encoder inside a secure enclave to protect sensitive knowledge.

## Benchmark Datasets
FediRAG is evaluated on multiple datasets:
- **Knowledge Graph Embeddings**: FB15k-237, WN18RR, YAGO3-10
- **Retrieval-Augmented Generation**: Natural Questions (NQ), TriviaQA, SQuAD, HotpotQA

## Future Improvements
- Enhancing **federated optimization techniques** for knowledge graph embeddings.
- Improving **adaptive retrieval mechanisms** for multi-domain applications.
- Extending **privacy-preserving protocols** with advanced homomorphic encryption.

## License
MIT License

## Contact
For questions and contributions, reach out to **achakr40@asu.edu** or open an issue in this repository.

---
### Citation
If you use **FediRAG** in your research, please cite:
```bibtex
@article{yourpaper2025,
  title={FediRAG: Federated Retrieval-Augmented Generation with Structured Relational Embeddings and Hybrid Search},
  author={Your Name et al.},
  journal={ACL 2025},
  year={2025}
}
```


