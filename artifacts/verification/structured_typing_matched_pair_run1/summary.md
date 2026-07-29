### Clean Matched-Pair Typing Ablation

The `typed_same_merge` and `untyped_same_merge` arms share the same generic field-wise merge. The only changed mechanism is whether query-time scoring uses the structured field roles.

| Arm | 0% | 20% | 40% | 60% |
| --- | ---: | ---: | ---: | ---: |
| Full SYNAPSE | 1.000 ± 0.000 | 0.993 ± 0.016 | 0.979 ± 0.032 | 0.921 ± 0.047 |
| Typed, same merge | 1.000 ± 0.000 | 0.950 ± 0.032 | 0.836 ± 0.070 | 0.671 ± 0.059 |
| Untyped, same merge | 1.000 ± 0.000 | 0.950 ± 0.032 | 0.836 ± 0.070 | 0.671 ± 0.059 |

| Contrast | 0% | 20% | 40% | 60% |
| --- | ---: | ---: | ---: | ---: |
| typed_same_merge - untyped_same_merge | +0.000 (all-zero) | +0.000 (all-zero) | +0.000 (all-zero) | +0.000 (all-zero) |
| full - typed_same_merge | +0.000 (all-zero) | +0.043 | +0.143 | +0.250 |
