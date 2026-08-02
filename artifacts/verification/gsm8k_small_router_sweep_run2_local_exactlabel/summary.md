### GSM8K Small LLM Router Sweep

- 8B gate label: `llama31_8b` must land in [`0.000`, `1.000`] to count as the expected routing scale.

| Router | Backend | Mean routing acc. | SD | Mean latency (s/query) | SD latency |
| --- | --- | ---: | ---: | ---: | ---: |
| llama32_1b | local | 0.300 | 0.026 | 0.093 | 0.018 |
| llama31_8b | local | 0.463 | 0.015 | 0.181 | 0.029 |
