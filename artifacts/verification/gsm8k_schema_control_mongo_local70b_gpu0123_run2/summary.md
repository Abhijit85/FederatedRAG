### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.980, 2=0.960, 3=0.980, 4=0.940, 5=0.940 | 0.960 | 0.020 |
| merge_up | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.960, 2=0.980, 3=0.960, 4=0.940, 5=0.940 | 0.956 | 0.017 |
| drop_annex | 1=0.140, 2=0.180, 3=0.140, 4=0.180, 5=0.180 | 0.164 | 0.022 | 1=0.960, 2=0.960, 3=0.980, 4=0.940, 5=0.960 | 0.960 | 0.014 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
