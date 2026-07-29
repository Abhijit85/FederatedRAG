### Structured-but-Untyped Ablation

This run uses the existing family-4 conflict harness and isolates the field-preserving structured-untyped control against Full SYNAPSE on the same records, queries, conflict construction, and success metric.

| Arm | 0% | 20% | 40% | 60% |
| --- | ---: | ---: | ---: | ---: |
| Full SYNAPSE | 1.000 ± 0.000 | 0.993 ± 0.016 | 0.979 ± 0.032 | 0.921 ± 0.047 |
| Structured-untyped | 1.000 ± 0.000 | 0.943 ± 0.041 | 0.836 ± 0.070 | 0.664 ± 0.060 |
| Typed-generic-merge | 1.000 ± 0.000 | 0.986 ± 0.032 | 0.943 ± 0.060 | 0.814 ± 0.081 |

| Contrast | 0% | 20% | 40% | 60% |
| --- | ---: | ---: | ---: | ---: |
| full - structured_untyped | +0.000 (all-zero) | +0.050 | +0.143 | +0.257 |
| full - typed_generic | +0.000 (all-zero) | +0.007 | +0.036 | +0.107 |
