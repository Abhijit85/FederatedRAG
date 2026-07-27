### GSM8K Schema Support

| Condition | Routing acc. | Routing SD | Answer acc. | Answer SD | Δ answer vs full |
| --- | ---: | ---: | ---: | ---: | ---: |
| full | 0.833 | 0.000 | 0.833 | 0.000 | +0.000 |
| merge_up | 0.833 | 0.000 | 0.750 | 0.000 | -0.083 |

Modes on this branch:
- `full`: typed payload with separate scenario, precaution, and annex channels
- `merge_up`: merges scenario context and precautions into a single notes channel
- `drop_annex`: removes annex fields while keeping scenario and precaution channels
- `untyped`: removes only the top-level type label
- `no_payload`: removes the structured payload entirely

This report is support evidence from the current branch. It is not a claim that the numbers match the paper's historical table.
