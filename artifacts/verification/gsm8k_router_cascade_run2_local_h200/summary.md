### GSM8K Two-Tier Router Cascade

- Small router: `llama32_1b`
- Large router: `llama31_8b`
- Confidence signal: `option_margin_logprob`
- Sample count: 100
- Seeds: 42, 123, 456

| Baseline router | Mean routing acc. | SD |
| --- | ---: | ---: |
| llama32_1b | 0.277 | 0.012 |
| llama31_8b | 0.463 | 0.015 |

| Threshold | Mean routing acc. | SD | Mean deferral rate | Mean compute reduction vs full large |
| --- | ---: | ---: | ---: | ---: |
| 0.500 | 0.433 | 0.023 | 0.007 | 0.868 |
| 1.000 | 0.420 | 0.020 | 0.160 | 0.715 |
| 1.500 | 0.443 | 0.035 | 0.897 | -0.022 |
