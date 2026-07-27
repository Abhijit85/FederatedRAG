# Experiment 2 Reconstructed Harness

This harness mirrors the current checkout as closely as possible:
- cluster formation uses cosine similarity over artifact text with threshold 0.85
- `typed` keeps the representative artifact, matching the current edge merge behavior
- `untyped` is the field-preserving control that drops only the payload `type` field
- `flat` is reconstructed as a naive shallow field overwrite across clustered members

Per-condition results:
- typed, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- typed, 20% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- typed, 40% conflict: 0.657 ± 0.032 (1=0.643, 2=0.643, 3=0.643, 4=0.714, 5=0.643)
- typed, 60% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- flat, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- flat, 20% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- flat, 40% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- flat, 60% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- untyped, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- untyped, 20% conflict: 0.700 ± 0.060 (1=0.714, 2=0.643, 3=0.786, 4=0.643, 5=0.714)
- untyped, 40% conflict: 0.743 ± 0.064 (1=0.714, 2=0.714, 3=0.857, 4=0.714, 5=0.714)
- untyped, 60% conflict: 0.757 ± 0.081 (1=0.714, 2=0.786, 3=0.643, 4=0.786, 5=0.857)
