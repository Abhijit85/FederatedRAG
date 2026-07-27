### GSM8K Small LLM Router Sweep

- 8B gate label: `llama31_8b` must land in [`0.000`, `1.000`] to count as the expected routing scale.

| Router | Backend | Mean routing acc. | SD | Mean latency (s/query) | SD latency |
| --- | --- | ---: | ---: | ---: | ---: |
| qwen25_0p5b | local | 0.124 | 0.022 | 0.108 | 0.056 |
| llama32_1b | openrouter | 0.312 | 0.058 | 0.546 | 0.031 |
| llama31_8b | local | 0.420 | 0.047 | 0.202 | 0.057 |
