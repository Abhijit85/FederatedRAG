# Experiment 2 Question-Level Reconstruction

- Records come directly from compendium scenario `context` fields.
- Each scenario expands into three queries: named, context-only, and exemplar-style.
- `typed` keeps a representative conflicting artifact; `flat` shallow-merges fields across the cluster.

Per-condition results:
- typed, 0% conflict: 0.881 ± 0.000 (1=0.881, 2=0.881, 3=0.881, 4=0.881, 5=0.881)
- typed, 20% conflict: 0.876 ± 0.011 (1=0.881, 2=0.881, 3=0.881, 4=0.857, 5=0.881)
- typed, 40% conflict: 0.857 ± 0.024 (1=0.857, 2=0.881, 3=0.833, 4=0.833, 5=0.881)
- typed, 60% conflict: 0.862 ± 0.020 (1=0.857, 2=0.881, 3=0.833, 4=0.881, 5=0.857)
- flat, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 20% conflict: 0.929 ± 0.071 (1=0.857, 2=1.000, 3=0.857, 4=1.000, 5=0.929)
- flat, 40% conflict: 0.805 ± 0.074 (1=0.690, 2=0.857, 3=0.810, 4=0.786, 5=0.881)
- flat, 60% conflict: 0.705 ± 0.060 (1=0.619, 2=0.714, 3=0.786, 4=0.714, 5=0.690)
