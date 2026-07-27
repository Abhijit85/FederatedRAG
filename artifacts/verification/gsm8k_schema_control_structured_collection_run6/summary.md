### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.700, 2=0.680, 3=0.740, 4=0.560, 5=0.860 | 0.708 | 0.108 |
| merge_up | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.840, 2=0.640, 3=0.660, 4=0.680, 5=0.780 | 0.720 | 0.086 |
| drop_annex | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.760, 2=0.800, 3=0.680, 4=0.720, 5=0.760 | 0.744 | 0.046 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
