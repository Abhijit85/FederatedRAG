### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.320, 2=0.420, 3=0.300, 4=0.340, 5=0.340 | 0.344 | 0.046 | 1=0.660, 2=0.620, 3=0.660, 4=0.620, 5=0.620 | 0.636 | 0.022 |
| merge_up | 1=0.320, 2=0.420, 3=0.300, 4=0.340, 5=0.340 | 0.344 | 0.046 | 1=0.700, 2=0.680, 3=0.580, 4=0.640, 5=0.740 | 0.668 | 0.061 |
| drop_annex | 1=0.320, 2=0.420, 3=0.300, 4=0.340, 5=0.340 | 0.344 | 0.046 | 1=0.620, 2=0.680, 3=0.620, 4=0.620, 5=0.720 | 0.652 | 0.046 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
