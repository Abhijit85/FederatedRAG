# Experiment 2 Reconstructed Harness

This harness mirrors the current checkout as closely as possible:
- cluster formation uses cosine similarity over artifact text with threshold 0.85
- `typed` keeps the representative artifact, matching the current edge merge behavior
- `untyped` is the field-preserving control that drops only the payload `type` field
- `flat` is reconstructed as a naive shallow field overwrite across clustered members

Per-condition results:
- typed, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643)
- typed, 60% conflict: 0.536 ± 0.051 (1=0.571, 2=0.500)
- flat, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643)
- flat, 60% conflict: 0.571 ± 0.000 (1=0.571, 2=0.571)
- untyped, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643)
- untyped, 60% conflict: 0.536 ± 0.051 (1=0.500, 2=0.571)
