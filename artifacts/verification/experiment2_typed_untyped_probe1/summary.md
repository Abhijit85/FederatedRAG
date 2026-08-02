# Experiment 2 Reconstructed Harness

This harness mirrors the current checkout as closely as possible:
- cluster formation uses cosine similarity over artifact text with threshold 0.85
- `typed` keeps the representative artifact, matching the current edge merge behavior
- `untyped` is the field-preserving control that drops only the payload `type` field
- `flat` is reconstructed as a naive shallow field overwrite across clustered members

Per-condition results:
- typed, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 20% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 40% conflict: 0.971 ± 0.039 (1=1.000, 2=1.000, 3=0.929, 4=0.929, 5=1.000)
- typed, 60% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- untyped, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- untyped, 20% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- untyped, 40% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- untyped, 60% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
