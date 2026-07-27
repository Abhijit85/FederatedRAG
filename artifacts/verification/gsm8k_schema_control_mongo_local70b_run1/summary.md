### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.020, 2=0.020, 3=0.040, 4=0.060, 5=0.000 | 0.028 | 0.023 |
| merge_up | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.020, 2=0.020, 3=0.040, 4=0.060, 5=0.000 | 0.028 | 0.023 |
| drop_annex | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.020, 2=0.020, 3=0.040, 4=0.060, 5=0.000 | 0.028 | 0.023 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
