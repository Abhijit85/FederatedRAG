### GSM8K Schema Support

| Condition | Routing acc. | Routing SD | Δ routing vs full |
| --- | ---: | ---: | ---: |
| full | 0.164 | 0.022 | +0.000 |
| merge_up | 0.164 | 0.022 | +0.000 |
| drop_annex | 0.164 | 0.022 | +0.000 |

Modes on this branch:
- `full`: typed payload with separate scenario, precaution, and annex channels
- `merge_up`: merges scenario context and precautions into a single notes channel
- `drop_annex`: removes annex fields while keeping scenario and precaution channels
- `untyped`: removes only the top-level type label
- `no_payload`: removes the structured payload entirely

This report is support evidence from the current branch. It is not a claim that the numbers match the paper's historical table.
