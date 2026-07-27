### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.200 | 0.200 | 0.000 | 1=0.800 | 0.800 | 0.000 |
| merge_up | 1=0.200 | 0.200 | 0.000 | 1=0.800 | 0.800 | 0.000 |
| drop_annex | 1=0.200 | 0.200 | 0.000 | 1=0.900 | 0.900 | 0.000 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
