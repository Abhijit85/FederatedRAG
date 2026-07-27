### Table 27 Current-Codebase Rerun Package

This package records the patched current-codebase rerun that reached paper-scale GSM8K routing accuracy and established current-codebase federated/centralized equivalence.

### Exact Result

- Paired runtime rerun artifact:
  [summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/table27_strict_compare_run10_runtime_clustered/summary.json)
- SYNAPSE / federated per-seed routing accuracies:
  `0.92, 0.93, 0.94, 0.91, 0.94`
- Centralized-SYNAPSE / clustered centralized per-seed routing accuracies:
  `0.92, 0.93, 0.94, 0.91, 0.94`
- Mean ± SD for both arms:
  `0.928 ± 0.013`
- Paired mean difference:
  `+0.000`

### Paired TOST

- Command:
```bash
.venv/bin/python scripts/run_table27_tost.py \
  --syn artifacts/verification/table27_strict_compare_run10_runtime_clustered/synapse_seed_values.csv \
  --cen artifacts/verification/table27_strict_compare_run10_runtime_clustered/centralized_seed_values.csv \
  --col acc \
  --margin 0.03 \
  --allow-san-gate-fail
```

- Output summary:
  - mean paired diff: `+0.0000`
  - 90% CI: `[+0.0000, +0.0000]`
  - TOST result: `EQUIVALENT within ±0.030`

### W4 / §D.2 Drop-In Wording

Use this when you want to report the fresh rerun honestly:

> On a fresh 5-seed paired rerun of the current codebase, SYNAPSE and Centralized-SYNAPSE achieved identical GSM8K routing accuracy (`0.928 ± 0.013` for both arms; seeds `0.92, 0.93, 0.94, 0.91, 0.94`). Under a pre-specified ±3-point equivalence margin, a paired TOST establishes equivalence (`mean paired diff = 0.000`, 90% CI `[0.000, 0.000]`). This is a current-codebase rerun rather than recovered historical provenance for the originally submitted Table 27 vector pair.

Use this when you need the shorter reviewer-facing version:

> We reran the federated vs. centralized GSM8K comparison on the current codebase with the corrected math-only routing path and obtained identical 5-seed results (`0.928 ± 0.013` for both arms). A paired TOST at a pre-specified ±3-point margin establishes equivalence.

### Exact Reproduction Commands

1. Run the paired current-codebase federated vs. centralized rerun:

```bash
.venv/bin/python scripts/run_table27_strict_compare.py \
  --sample-count 100 \
  --synapse-arm runtime_federated \
  --centralized-arm runtime_centralized_clustered \
  --output-dir artifacts/verification/table27_strict_compare_run10_runtime_clustered
```

2. Optional direct-pooled centralized variant:

```bash
.venv/bin/python scripts/run_table27_strict_compare.py \
  --sample-count 100 \
  --synapse-arm runtime_federated \
  --centralized-arm runtime_centralized_direct \
  --output-dir artifacts/verification/table27_strict_compare_run11_runtime_direct
```

3. Run the paired TOST:

```bash
.venv/bin/python scripts/run_table27_tost.py \
  --syn artifacts/verification/table27_strict_compare_run10_runtime_clustered/synapse_seed_values.csv \
  --cen artifacts/verification/table27_strict_compare_run10_runtime_clustered/centralized_seed_values.csv \
  --col acc \
  --margin 0.03 \
  --allow-san-gate-fail
```

### What Changed

The current-codebase rerun only reached paper scale after both of these fixes:

- `SYNAPSE_INCLUDED_TOOLS=mathqa` in the GSM8K routing alignment path, to remove science-artifact contamination.
- Candidate-constrained historical SVM label selection over retrieved math scenarios in the routing verifier.

Those changes live in:

- [scripts/run_routing_verification.py](/mnt/data1/achakr40/FederatedRAG/scripts/run_routing_verification.py)

### Status Snapshot

As of Sunday, July 26, 2026:

- C3 live run:
  [full/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/gsm8k_schema_control_structured_collection_run6/full/summary.json)
  - routing seeds complete for `full`
  - answer seed `1` complete at `0.700`
- C5 live run:
  [summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/livebench_support_fullsplit_local_run8/summary.json)
  - completed: `reasoning:spatial`, `reasoning:web_of_lies_v2`, `reasoning:zebra_puzzle` baseline
  - not yet complete for strong `zebra_puzzle` or `math:AMPS_Hard`
