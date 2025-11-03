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
- Optional for real TexGrad backpropagation: `torch`, `peft`, `accelerate`, `bitsandbytes`
- Optional for production DP/optimisation: `opacus` (required when `SYNAPSE_CLIENT_USE_PEFT` or `SYNAPSE_CENTRAL_USE_PEFT` is enabled)

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
SYNAPSE_CLIENT_COUNT=4.    # modify to set the number of clients for federation.
# Optional privacy controls
SYNAPSE_ENABLE_DP=1
SYNAPSE_DP_EPSILON=1.0
```

The agent sends chat completions to `https://openrouter.ai/api/v1/chat/completions`. After editing `.env`, run `set -a; source .env; set +a` (or the equivalent in your shell) before launching any scripts.

### Environment Overrides

All major TexGrad/LoRA settings can be changed through `.env` without code edits. Common overrides:
- `SYNAPSE_CLIENT_USE_PEFT` (default `false`) – enable PEFT/transformers LoRA on each client; pair with `SYNAPSE_CLIENT_BASE_MODEL`, `SYNAPSE_CLIENT_QUANTIZATION`, and `SYNAPSE_CLIENT_LORA_TARGETS`/`SYNAPSE_CLIENT_LORA_RANKS` to select the base LLaMA and adapter layout.
- `VLM_MODEL` – global base-model hint used by both federated clients and the centralized trainer; override `SYNAPSE_CLIENT_BASE_MODEL` or `SYNAPSE_CENTRAL_BASE_MODEL` only when you need per-mode differences.
- `SYNAPSE_CLIENT_BACKPROP` (default `true`) – turn on the PyTorch-based TexGrad LoRA trainer that performs real backpropagation before packaging Secure Aggregation updates. Requires `torch` to be installed.
- `SYNAPSE_CENTRAL_USE_PEFT` (default `false`) – run Synapse-Central with a true PEFT/transformers fine-tuning loop; falls back to the lightweight trainer if dependencies are unavailable.
- `SYNAPSE_CLIENT_DP_CLIP`, `SYNAPSE_CLIENT_DP_NOISE`, `SYNAPSE_CLIENT_DP_SAMPLE_RATE` – tune client DP clipping and Gaussian noise.
- `SYNAPSE_CLIENT_SECAGG_PROTOCOL`, `SYNAPSE_CLIENT_SECAGG_KEY_ROTATION` – swap secure aggregation protocol details.
- `SYNAPSE_SECAGG_PROVIDER` (`simple`, `tee`) / `SYNAPSE_SECAGG_SECRET` / `SYNAPSE_SECAGG_ATTESTATION` – choose the secure aggregation backend and provide shared secret or attestation handle; same variables with the `SYNAPSE_CENTRAL_` prefix control centralized training.
- `SYNAPSE_CLIENT_HEARTBEAT_INTERVAL`, `SYNAPSE_CLIENT_RETRY_QUEUE_LIMIT`, `SYNAPSE_CLIENT_OFFLINE_GRACE` – adjust health-monitor heartbeat cadence and retries.
- `SYNAPSE_CENTRAL_BASE_MODEL`, `SYNAPSE_CENTRAL_QUANTIZATION`, `SYNAPSE_CENTRAL_LORA_TARGETS`, `SYNAPSE_CENTRAL_LORA_RANKS` – control the centralized trainer’s base model and adapter footprint.
- `SYNAPSE_CENTRAL_EPOCHS`, `SYNAPSE_CENTRAL_STEPS_PER_EPOCH`, `SYNAPSE_CENTRAL_BATCH_SIZE`, `SYNAPSE_CENTRAL_TRAIN_CORPORA` – drive epochs, steps, batch size, and corpus paths for Synapse-Central.
- `SYNAPSE_CENTRAL_DP_ENABLED`, `SYNAPSE_CENTRAL_DP_CLIP`, `SYNAPSE_CENTRAL_DP_NOISE`, `SYNAPSE_CENTRAL_DP_DELTA`, `SYNAPSE_CENTRAL_DP_EPS_CAP` – toggle DP-SGD and budget caps for centralized runs.
- `SYNAPSE_CENTRAL_ROBUST_*`, `SYNAPSE_CENTRAL_AGG_*`, `SYNAPSE_CENTRAL_ROUTER_*` – fine-tune robustness thresholds, spectral filters, and ARR/MoA routing policies.
- `SYNAPSE_SERVER_AGG_MODE` (`robust` or `sum_only`) – choose between enclave-style robust aggregation (spectral + median) and SecAgg sum-only averaging.

Any override omitted from `.env` falls back to the defaults hard-coded in the dataclass constructors, so you can opt in incrementally.

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

## HyFICAL Control Plane & Federated Training

The runtime now includes a lightweight implementation of the HyFICAL control plane:
- **TexGrad-LoRA Clients** – each `UnifiedQAClient` generates deterministic LoRA adapter deltas using the new `synapse.training` toolkit (TexGrad heuristics, DP clipping/noise, SecAgg masking, and a HealthAgent for heartbeats/retries). Adapter updates follow the `AdapterUpdate` contract in `synapse.hyfical.contracts`.
- **HyFICAL Server** – `SynapseServer` buffers updates with an asynchronous window scheduler, applies Spectral Robust Aggregation + geometric median (trust-weighted), and routes adapters via the ARR/MoA router. Privacy loss is tracked with an RDP accountant and recorded alongside aggregation telemetry in the compliance ledger.
- **Aggregation modes** – set `SYNAPSE_SERVER_AGG_MODE=robust` (default) to run the spectral+median pipeline inside a TEE-style facade, or `sum_only` to emulate pure SecAgg (sum-only) aggregation when you need maximum privacy and can accept simpler robustness.
- **Global Adapter Bundle** – after each round, clients receive a `GlobalAdapterBundle` containing the selected experts per layer, routing hints, and the remaining privacy budget. Clients automatically adjust their target LoRA rank based on the broadcast bundle.

Key files:
- `synapse/training/` – TexGrad, LoRA planning, DP guard, SecAgg stub, and health monitoring utilities.
- `synapse/hyfical/` – adapter payload contracts, async scheduler, robust aggregator, and router policy.
- `synapse/server/orchestrator.py` – orchestrates HyFICAL aggregation, updates the compliance ledger, and exposes bundle + trust score summaries through the runtime.

### Adapter Contracts

Client → server payload (`AdapterUpdate`):

```json
{
  "client_id": "opaque",
  "round_hint": 12,
  "layer_updates": [
    {
      "layer": "attn.qkv.11",
      "format": "LoRA",
      "rank": 16,
      "delta_hash": "<sha256>",
      "masked_delta": "<hex bytes>",
      "norm": 2.31
    }
  ],
  "telemetry": {
    "freshness_ts": 1730478123,
    "steps": 240,
    "loss_lm": 1.92,
    "texgrad": {
      "entailment": 0.84,
      "citation_coverage": 0.71,
      "contrastive_margin": 0.36,
      "retrieval_entropy": 0.52,
      "semantic_fingerprint": [0.02, -0.11, 0.08]
    }
  },
  "dp_local": {"clipping": 0.5, "sigma": 1.2, "epsilon_local": 0.9}
}
```

Server → client broadcast (`GlobalAdapterBundle`):

```json
{
  "version": "v12",
  "adapters": {
    "attn.qkv.11": [
      {"id": "finance", "rank": 8},
      {"id": "general", "rank": 4}
    ]
  },
  "router_hints": {"attn.qkv.11": "grounding"},
  "privacy_budget_remaining": {"clipping": 0.5, "sigma": 1.2, "epsilon_local": 5.6},
  "release_notes": "robust-agg: spectral_k=20, dp_sigma=1.2"
}
```

### Federation Checklist

HyFICAL enforces the following guardrails inside the codebase (see `synapse/checks/checklist.py`):

- Base model remains frozen, quantized (4/8-bit) per client, and license compliant.
- LoRA adapter ranks cover `{4,8,16}` targeting attention `q_proj/k_proj/v_proj` plus selected MLP paths.
- TexGrad losses capture entailment, citation coverage, and contrastive attribution signals.
- Differential privacy guard applies per-layer clipping, Gaussian noise with multiplier σ, and tracks subsampling rate `q` using an RDP accountant.
- Secure aggregation is mandatory with key rotation and a quarantine queue for anomalous updates.
- Aggregation schedules use window `W`, `spectral_k`, cosine divergence threshold `τ`, geometric-median iterations, and freshness half-life weighting.
- Observability surfaces DP budget dashboards, adapter-norm z-scores, poisoning flags, and ledger-backed rollback points.

## Centralized TexGrad-LoRA (Synapse-Central)

When you want a conventional single-cluster fine-tune that still honors the privacy, faithfulness, and robustness guarantees, use the new **Synapse-Central** trainer (`synapse.central`). The centralized path keeps TexGrad losses, ARR/MoA routing, differential privacy accounting, and ledgering, but runs the LoRA optimization directly on your cluster.

Key highlights:

- **TexGrad losses** (LM + entailment + citation + contrastive) are computed batch-wise using retrieved contexts; gradients are projected toward semantic directions before the optimizer step.
- **Robust gradient layer** trims high-loss batches, enforces cosine agreement with an EMA baseline, and applies spectral + geometric-median aggregation across synthetic workers to mimic Byzantine resilience.
- **Differential privacy** is optional; enable it via `CentralPrivacyConfig` to get DP-SGD clipping, Gaussian noise, and an RDP accountant.
- **ARR/MoA router** continues to adapt LoRA ranks `{4,8,16}` and expert mixes based on retrieval entropy, citation coverage, and domain tags.
- **Audit ledger** records every centralized adapter release with ε/δ, spectral parameters, and TexGrad telemetry so you can ship immutable provenance reports.

Example usage:

```python
from pathlib import Path
from synapse.central import (
    CentralTexGradTrainer,
    CentralTrainingConfig,
    CentralPrivacyConfig,
)

config = CentralTrainingConfig(
    epochs=2,
    steps_per_epoch=150,
    batch_size=16,
    training_corpora=[
        Path("train_new.json"),
        Path("scienceqa_dataset.json"),
    ],
    privacy=CentralPrivacyConfig(enabled=True),
)

trainer = CentralTexGradTrainer(config)
summary = trainer.run()

print("Adapter versions:", summary.adapter_versions)
print("Latest privacy budget:", summary.privacy_budget)
print("Ledger entries:", summary.ledger_entries[-1])
```

The summary exposes poisoning flags, adapter norm z-scores, and audit records you can stream into dashboards. Once a centralized run converges, you can export the adapter bank and resume **federated** fine-tuning on remote sites for privacy-preserving adaptation.

## Project Structure
- `main.py` – runs the SYNAPSE federation round, instantiates tools, and evaluates the federated agent.
- `CompendiumAwareAgent.py` – legacy single-node routing agent (kept for reference).
- `vector_search.py` – Jina-based embedding client storing scenario vectors in memory.
- `math_qa.py` – RAG pipeline for math word problems using Jina embeddings and ChromaDB.
- `science_qa.py` – image and text reasoning with Lambda's vision-language models.
- `CompendiumBuilder.py` – generates structured compendiums and filters similar tools.
- `agenttools.py` – base classes and helper tools.
- `synapse/` – SYNAPSE implementation (clients, edge aggregators, server orchestrator, knowledge abstractions, retrieval planner, privacy policies, runtime coordinator, and agent wrapper).
- `synapse/central/` – centralized TexGrad-LoRA trainer with ARR/MoA routing, DP hooks, and robust gradient aggregation.
- `synapse/training/backprop.py` – reference TexGrad backpropagation loop that optimizes LoRA adapters with PyTorch before DP/SecAgg packaging.
- `synapse/server/aggregation.py` – aggregation facade supporting TEE-style robust mode and SecAgg sum-only fallback.
- `synapse/secure/` – secure aggregation provider implementations (simple masking and TEE-style proxy hooks).

## Deployment & Ops Guidance

### Dependencies
- Base stack: Python >=3.9, `pip install -r requirements.txt`.
- GPU training: add `torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `opacus`.
- Secure aggregation:
  * **Simple mode** – just set secrets through env variables.
  * **TEE mode** – provision a confidential-compute enclave (e.g., AWS Nitro Enclaves, Azure Confidential VM, Intel SGX) capable of receiving masked payloads, performing spectral+median aggregation, and returning aggregated adapters.
- Monitoring: Prometheus/Grafana (or Datadog) to ingest runtime summaries (DP budget, poisoning flags, aggregation mode).

### Environment Presets
- **Local development**
  ```env
  SYNAPSE_CLIENT_USE_PEFT=false
  SYNAPSE_CLIENT_BACKPROP=false
  SYNAPSE_SECAGG_PROVIDER=simple
  SYNAPSE_SERVER_AGG_MODE=robust
  ```
- **GPU workstation / staging**
  ```env
  VLM_MODEL=/models/llama-3-8b
  SYNAPSE_CLIENT_USE_PEFT=true
  SYNAPSE_CLIENT_BACKPROP=true
  SYNAPSE_CLIENT_DP_CLIP=0.5
  SYNAPSE_CLIENT_DP_NOISE=1.2
  SYNAPSE_CENTRAL_USE_PEFT=true
  SYNAPSE_SECAGG_PROVIDER=simple
  SYNAPSE_SERVER_AGG_MODE=robust
  ```
- **Production (TEE + SecAgg sum-only)**
  ```env
  SYNAPSE_SECAGG_PROVIDER=tee
  SYNAPSE_SECAGG_SECRET=base64sharedsecret
  SYNAPSE_SECAGG_ATTESTATION=https://enclave-attestor.example
  SYNAPSE_SERVER_AGG_MODE=sum_only
  SYNAPSE_CENTRAL_SECAGG_PROVIDER=tee
  SYNAPSE_CENTRAL_SECAGG_SECRET=centralsecret
  SYNAPSE_CENTRAL_USE_PEFT=true
  SYNAPSE_CLIENT_USE_PEFT=true
  SYNAPSE_CLIENT_DP_CLIP=0.5
  SYNAPSE_CLIENT_DP_NOISE=1.5
  ```

### Secure Aggregation Deployment Checklist
1. **Secret and key management**
   - Create a symmetric mask secret per environment using KMS/Vault.
   - Distribute via secure configuration (Kubernetes Secrets, AWS Secrets Manager). Rotate regularly; update env vars accordingly.
2. **Enclave provisioning (TEE mode)**
   - Build the robust aggregation service (spectral filtering + geometric median) as a small RPC server.
   - Enable attestation and publish an attestation endpoint. Clients set `SYNAPSE_SECAGG_ATTESTATION` so `SecAggAdapter` can validate attestation before sending payloads.
   - Expose secure gRPC/HTTPS endpoint only reachable from trusted networks; enforce mTLS.
3. **Client wiring**
   - Ensure each client uses `SecAggAdapter.mask(...)` which now includes metadata (shape, round, attestation token).
   - For TEE mode, verify the attestation response before transmitting masked deltas; implement retries/backoff when attestation fails.
4. **Server-side decoding**
   - The server’s aggregation pipeline (`SynapseServer._decode_layer`) now delegates to the secure provider; plug in the enclave RPC or sum-only decrypt path as needed.

### Monitoring & Auditing
- Collect `runtime.summarize_round()` output; track `privacy_budget`, `aggregation_mode`, `secagg_attestation`, `adapter_norm_zscores`, and `poisoning_flags`.
- Persist ledger entries (`ComplianceLedger`) to a database; surface dashboard panels for ε/δ over time, participant counts, anomaly rejections, and adapter versions.
- Alert on:
  * Missing/invalid attestation tokens (indicates fallback or compromise).
  * Rapid ε growth or DP noise multipliers dropping below policy thresholds.
  * High z-score counts or repeated quarantine reasons (potential poisoning).

### Recommended Rollout Sequence
1. Start with simple SecAgg in staging; validate TexGrad metrics and DP budgets.
2. Deploy enclave service in sum-only mode; confirm attestation and key distribution.
3. Switch server to `SYNAPSE_SERVER_AGG_MODE=robust` once enclave throughput + monitoring look healthy.
4. Gradually enable PEFT+DP on clients; monitor ledger and runtime summaries.
5. Integrate dashboards and automate ledger archival for compliance reporting.

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
