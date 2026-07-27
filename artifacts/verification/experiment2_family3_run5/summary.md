# Experiment 2 Family3 Reconstruction

- Conflicts are nearest-neighbor same-tool scenarios rather than arbitrary contradictions.
- Queries emphasize one decisive cue plus a weak exemplar/context signal.
- `typed` preserves cluster identity but can drift on the decisive field; `flat` uses last-write field overwrite.

Per-condition results:
- typed, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 20% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 40% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 60% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 20% conflict: 0.971 ± 0.039 (1=0.929, 2=0.929, 3=1.000, 4=1.000, 5=1.000)
- flat, 40% conflict: 0.907 ± 0.048 (1=0.929, 2=0.964, 3=0.857, 4=0.929, 5=0.857)
- flat, 60% conflict: 0.900 ± 0.085 (1=0.857, 2=1.000, 3=0.893, 4=0.786, 5=0.964)
