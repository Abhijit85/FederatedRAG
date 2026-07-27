# Experiment 2 Family3 Reconstruction

- Conflicts are nearest-neighbor same-tool scenarios rather than arbitrary contradictions.
- Queries emphasize one decisive cue plus a weak exemplar/context signal.
- `typed` preserves cluster identity but can drift on the decisive field; `flat` uses last-write field overwrite.

Per-condition results:
- typed, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- typed, 20% conflict: 0.979 ± 0.020 (1=0.964, 2=1.000, 3=0.964, 4=1.000, 5=0.964)
- typed, 40% conflict: 0.971 ± 0.016 (1=1.000, 2=0.964, 3=0.964, 4=0.964, 5=0.964)
- typed, 60% conflict: 0.936 ± 0.053 (1=0.929, 2=0.929, 3=0.857, 4=1.000, 5=0.964)
- flat, 0% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 20% conflict: 1.000 ± 0.000 (1=1.000, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 40% conflict: 0.971 ± 0.064 (1=0.857, 2=1.000, 3=1.000, 4=1.000, 5=1.000)
- flat, 60% conflict: 0.986 ± 0.020 (1=0.964, 2=1.000, 3=1.000, 4=0.964, 5=1.000)
