### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.240, 2=0.260, 3=0.260, 4=0.240, 5=0.320 | 0.264 | 0.033 | 1=0.600, 2=0.500, 3=0.580, 4=0.620, 5=0.700 | 0.600 | 0.072 |
| untyped | 1=0.240, 2=0.260, 3=0.260, 4=0.240, 5=0.320 | 0.264 | 0.033 | 1=0.740, 2=0.680, 3=0.680, 4=0.620, 5=0.640 | 0.672 | 0.046 |
| no_payload | 1=0.240, 2=0.260, 3=0.260, 4=0.240, 5=0.320 | 0.264 | 0.033 | 1=0.580, 2=0.660, 3=0.780, 4=0.660, 5=0.760 | 0.688 | 0.082 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed structured payload
- `untyped` = identical payload with the top-level `type` field removed
- `no_payload` = structured payload removed entirely
