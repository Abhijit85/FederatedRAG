### Table 14 vs. Section C.4 Audit

| Question | Current repo evidence |
| --- | --- |
| Current TextGrad runner defaults | aggregate=`summarization`, batch=`3`, steps=`3`, rounds=`1` |
| Exact `0.901` preserved in TextGrad log | False |
| Exact `0.901` present in rebuttal expected-ranges note | True |
| Exact `0.92` preserved in TextGrad log | True |

| Preserved central TextGrad evaluations on `bbh_object_counting_eval_v3.json` | Accuracy |
| --- | ---: |
| 2025-11-09T10:35:37.375409Z | 0.980 (49/50) |
| 2025-11-14T05:43:05.836879Z | 0.920 (46/50) |

Interpretation:
The current checkout preserves a real `0.920` TextGrad result on `bbh_object_counting_eval_v3.json` but does not preserve a raw `0.901` TextGrad run artifact. The exact `0.901` value is present in the rebuttal expectations note, not in the committed TextGrad log. That means the repo supports the Table 14 headline as a real rounded measurement, but it does not preserve sufficient provenance to prove that Section C.4's `0.901` came from a committed batch=3 sweep artifact rather than a manual transcription or an external aggregation sheet.

A second limitation is that `evaluation_on_textgrad_log.txt` records benchmark outcomes but not the per-run CLI arguments, so even though the runner source defaults to batch=3 and max_steps=3, the log alone cannot prove which non-default flags were or were not used for a historical run.
