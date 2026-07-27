# Experiment 2 Reconstructed Harness

This harness mirrors the current checkout as closely as possible:
- cluster formation uses cosine similarity over artifact text with threshold 0.85
- `typed` keeps the representative artifact, matching the current edge merge behavior
- `untyped` is the field-preserving control that drops only the payload `type` field
- `flat` is reconstructed as a naive shallow field overwrite across clustered members

Per-condition results:
- typed, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 20% conflict: 0.986 ± 0.032 (1=1.000, 2=1.000, 3=0.929, 4=1.000, 5=1.000)
- typed, 40% conflict: 0.900 ± 0.081 (1=1.000, 2=0.929, 3=0.786, 4=0.929, 5=0.857)
- typed, 60% conflict: 0.871 ± 0.093 (1=0.714, 2=0.929, 3=0.929, 4=0.857, 5=0.929)
- flat, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 20% conflict: 0.843 ± 0.117 (1=0.786, 2=0.929, 3=0.786, 4=1.000, 5=0.714)
- flat, 40% conflict: 0.743 ± 0.081 (1=0.714, 2=0.857, 3=0.786, 4=0.714, 5=0.643)
- flat, 60% conflict: 0.671 ± 0.081 (1=0.571, 2=0.714, 3=0.643, 4=0.643, 5=0.786)
- untyped, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- untyped, 20% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- untyped, 40% conflict: 0.929 ± 0.051 (1=1.000, 2=0.929, 3=0.857, 4=0.929, 5=0.929)
- untyped, 60% conflict: 0.943 ± 0.032 (1=0.929, 2=0.929, 3=1.000, 4=0.929, 5=0.929)
