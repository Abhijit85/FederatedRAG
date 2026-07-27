# Experiment 2 Reconstructed Harness

This harness mirrors the current checkout as closely as possible:
- cluster formation uses cosine similarity over artifact text with threshold 0.85
- `typed` keeps the representative artifact, matching the current edge merge behavior
- `untyped` is the field-preserving control that drops only the payload `type` field
- `flat` is reconstructed as a naive shallow field overwrite across clustered members

Per-condition results:
- typed, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- typed, 20% conflict: 0.586 ± 0.078 (1=0.643, 2=0.500, 3=0.643, 4=0.643, 5=0.500)
- typed, 40% conflict: 0.557 ± 0.060 (1=0.500, 2=0.500, 3=0.643, 4=0.571, 5=0.571)
- typed, 60% conflict: 0.529 ± 0.039 (1=0.571, 2=0.500, 3=0.571, 4=0.500, 5=0.500)
- flat, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- flat, 20% conflict: 0.543 ± 0.064 (1=0.500, 2=0.500, 3=0.500, 4=0.643, 5=0.571)
- flat, 40% conflict: 0.543 ± 0.039 (1=0.500, 2=0.571, 3=0.500, 4=0.571, 5=0.571)
- flat, 60% conflict: 0.557 ± 0.032 (1=0.571, 2=0.571, 3=0.571, 4=0.500, 5=0.571)
- untyped, 0% conflict: 0.643 ± 0.000 (1=0.643, 2=0.643, 3=0.643, 4=0.643, 5=0.643)
- untyped, 20% conflict: 0.571 ± 0.071 (1=0.500, 2=0.643, 3=0.571, 4=0.643, 5=0.500)
- untyped, 40% conflict: 0.529 ± 0.064 (1=0.500, 2=0.500, 3=0.643, 4=0.500, 5=0.500)
- untyped, 60% conflict: 0.586 ± 0.060 (1=0.500, 2=0.571, 3=0.643, 4=0.571, 5=0.643)
