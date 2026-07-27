### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.240, 2=0.260, 3=0.260, 4=0.240, 5=0.320 | 0.264 | 0.033 | 1=0.660, 2=0.660, 3=0.560, 4=0.700, 5=0.660 | 0.648 | 0.052 |
| untyped | 1=0.240, 2=0.260, 3=0.260, 4=0.240, 5=0.320 | 0.264 | 0.033 | 1=0.680, 2=0.620, 3=0.620, 4=0.600, 5=0.680 | 0.640 | 0.037 |
| no_payload | 1=0.240, 2=0.260, 3=0.260, 4=0.240, 5=0.320 | 0.264 | 0.033 | 1=0.700, 2=0.660, 3=0.640, 4=0.660, 5=0.740 | 0.680 | 0.040 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed structured payload
- `untyped` = identical payload with the top-level `type` field removed
- `no_payload` = structured payload removed entirely
