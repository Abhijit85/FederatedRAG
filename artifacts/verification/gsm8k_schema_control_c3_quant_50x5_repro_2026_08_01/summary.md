### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.920, 2=0.920, 3=1.000, 4=0.920, 5=0.940 | 0.940 | 0.035 | 1=0.620, 2=0.740, 3=0.700, 4=0.680, 5=0.660 | 0.680 | 0.045 |
| merge_up | 1=0.920, 2=0.920, 3=1.000, 4=0.920, 5=0.940 | 0.940 | 0.035 | 1=0.860 | 0.860 | 0.000 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
