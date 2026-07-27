# Experiment 2 Family3 Reconstruction

- Conflicts are nearest-neighbor same-tool scenarios rather than arbitrary contradictions.
- Queries emphasize one decisive cue plus a weak exemplar/context signal.
- `typed` preserves cluster identity but can drift on the decisive field; `flat` uses last-write field overwrite.

Per-condition results:
- typed, 0% conflict: 0.750 ± 0.000 (1=0.750, 2=0.750, 3=0.750, 4=0.750, 5=0.750)
- typed, 20% conflict: 0.750 ± 0.000 (1=0.750, 2=0.750, 3=0.750, 4=0.750, 5=0.750)
- typed, 40% conflict: 0.750 ± 0.000 (1=0.750, 2=0.750, 3=0.750, 4=0.750, 5=0.750)
- typed, 60% conflict: 0.750 ± 0.000 (1=0.750, 2=0.750, 3=0.750, 4=0.750, 5=0.750)
- flat, 0% conflict: 0.750 ± 0.000 (1=0.750, 2=0.750, 3=0.750, 4=0.750, 5=0.750)
- flat, 20% conflict: 0.729 ± 0.032 (1=0.714, 2=0.679, 3=0.750, 4=0.750, 5=0.750)
- flat, 40% conflict: 0.707 ± 0.030 (1=0.714, 2=0.750, 3=0.679, 4=0.714, 5=0.679)
- flat, 60% conflict: 0.693 ± 0.065 (1=0.643, 2=0.750, 3=0.714, 4=0.607, 5=0.750)
