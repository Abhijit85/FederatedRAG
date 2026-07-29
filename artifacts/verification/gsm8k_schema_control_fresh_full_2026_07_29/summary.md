### GSM8K Schema Control

| Condition | Routing seeds | Mean routing acc. | SD | Answer seeds | Mean answer acc. | SD |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| full | 1=0.920, 2=0.920, 3=1.000, 4=0.920, 5=0.940 | 0.940 | 0.035 | 1=0.680, 2=0.640, 3=0.640, 4=0.600, 5=0.640 | 0.640 | 0.028 |
| untyped | 1=0.240, 2=0.260, 3=0.260, 4=0.260, 5=0.320 | 0.268 | 0.030 | 1=0.680, 2=0.540, 3=0.640, 4=0.700, 5=0.680 | 0.648 | 0.064 |

Conditions are the runtime-supported controls on this branch:
- `full` = typed payload with distinct scenario, precaution, and annex channels
- `merge_up` = scenario context and precautions merged into one undifferentiated scenario-notes field
- `drop_annex` = structured annex removed while keeping scenario and precaution channels
- `untyped` = typed payload with only the top-level `type` field removed
- `no_payload` = structured payload removed entirely
