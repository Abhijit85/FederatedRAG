# Experiment 2 Family4 Reconstruction

- Same-tool nearest-neighbor conflicts with query-time typed representative exposure.
- `typed` preserves cluster identity but query scoring can surface a conflicted member within the cluster.
- `flat` uses last-write field overwrite across clustered members.

Per-condition results:
- typed, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 20% conflict: 0.993 ± 0.016 (1=1.000, 2=0.964, 3=1.000, 4=1.000, 5=1.000)
- typed, 40% conflict: 0.986 ± 0.020 (1=1.000, 2=1.000, 3=0.964, 4=1.000, 5=0.964)
- typed, 60% conflict: 0.993 ± 0.016 (1=1.000, 2=1.000, 3=1.000, 4=0.964, 5=1.000)
- flat, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 20% conflict: 0.971 ± 0.039 (1=0.929, 2=0.929, 3=1.000, 4=1.000, 5=1.000)
- flat, 40% conflict: 0.893 ± 0.051 (1=0.857, 2=0.964, 3=0.857, 4=0.929, 5=0.857)
- flat, 60% conflict: 0.893 ± 0.087 (1=0.857, 2=1.000, 3=0.857, 4=0.786, 5=0.964)
