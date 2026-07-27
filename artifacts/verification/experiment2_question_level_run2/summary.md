# Experiment 2 Question-Level Reconstruction

- Records come directly from compendium scenario `context` fields.
- Each scenario expands into three queries: named, context-only, and exemplar-style.
- `typed` keeps a representative conflicting artifact; `flat` shallow-merges fields across the cluster.

Per-condition results:
- typed, 0% conflict: 0.976 ± 0.000 (1=0.976, 2=0.976, 3=0.976, 4=0.976, 5=0.976)
- typed, 20% conflict: 0.905 ± 0.056 (1=0.833, 2=0.929, 3=0.857, 4=0.952, 5=0.952)
- typed, 40% conflict: 0.733 ± 0.020 (1=0.738, 2=0.714, 3=0.738, 4=0.714, 5=0.762)
- typed, 60% conflict: 0.662 ± 0.046 (1=0.595, 2=0.690, 3=0.643, 4=0.667, 5=0.714)
- flat, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 20% conflict: 0.986 ± 0.032 (1=0.929, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 40% conflict: 0.976 ± 0.029 (1=1.000, 2=0.929, 3=0.976, 4=1.000, 5=0.976)
- flat, 60% conflict: 0.976 ± 0.024 (1=0.952, 2=1.000, 3=0.976, 4=0.952, 5=1.000)
