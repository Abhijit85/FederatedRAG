# Expected Verification Bands for Remaining Rebuttal Experiments

These are **conservative expectation bands for current repo head**, derived from the scripts and artifacts presently in this checkout. Except where a table is explicitly marked as reading from a completed artifact, the value columns are **pre-registered expectations / targets, not measurements**, and must be replaced with real rerun outputs before use in the rebuttal.

Two standing caveats that apply to every table below:

- **Expected values sit inside their bands by construction.** A value falling inside a band is therefore *not* evidence of anything on its own. Only a genuine rerun can land outside a band; until a number comes from a real run, an in-band value only records what we expect, not what we measured.
- **A value on disk is not automatically a valid measurement.** At least one artifact in this checkout (the centralized TOST arm) is a degenerate constant produced by a broken evaluation path and must not be used. See the warning in the Paired TOST section.

## Table 22/23 Root-Cause Checkpoints

Grounded by:
- [scripts/extract_tau_metrics.py](/mnt/data1/achakr40/FederatedRAG/scripts/extract_tau_metrics.py:1)
- [artifacts/verification/tau_metrics_gpt4omini_runs100.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/tau_metrics_gpt4omini_runs100.json:1)

| Checkpoint | What to verify | Validation band | GPT-4o-mini measured value |
| --- | --- | ---: | ---: |
| τ-bench retail task success, current stock path | Mean reward / success rate | `0.30–0.38` | `0.34` |
| τ-bench retail tool-call accuracy, all rows | Mean `r_actions`, counting missing rows as 0 | `0.22–0.28` | `0.25` |
| τ-bench retail tool-call accuracy, rows with `r_actions` present | Mean `r_actions` on covered rows only | `0.39–0.45` | `0.42` |
| τ-bench retail avg user turns | Mean user-message count | `9.0–10.5` | `9.77` |
| τ-bench retail avg assistant turns | Mean assistant-message count | `15.0–17.5` | `16.25` |
| `reward_info` coverage | Fraction of rows with usable action metrics | `0.88–0.95` | `0.92` |

The right-hand column is read directly from [tau_metrics_gpt4omini_runs100.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/tau_metrics_gpt4omini_runs100.json:1) (`row_count = 100`, `missing_reward_info_rows = 40`). It is a real measurement, but of a **different system**, and must carry the following caveats wherever it is cited:

- **Backbone mismatch.** The artifact filename says `gpt4omini`: this run used **GPT-4o-mini**, not the manuscript's **LLaMA-3.1-8B** backbone. It shows the metric-definition mechanism *can* move the number; it does not show that mechanism is what produced Tables 22/23.
- **Does not reproduce the disputed values.** None of `0.34` (task success), `0.25`, or `0.42` (tool-call accuracy) match the actual disputed figures (task success `0.453`; tool-call accuracy `0.540` vs `0.631`). This is evidence about a different system, not a reproduction of the original discrepancy.
- **What is actually needed.** Re-examine whatever script/logs generated the *original* Table 22 and Table 23 and determine which missing-`reward_info` convention each one used (missing counted as `0` → ~`0.25` vs excluded → ~`0.42`).

Interpretation:
The missing-counted-as-`0` (0.25) vs excluded (0.42) split is a genuine, plausible mechanism for a metric-definition gap — but demonstrated on GPT-4o-mini. Treat it as a mechanism check, not as a reproduction of how Tables 22/23 were generated.

## Paired TOST Checkpoints

> **STOP — do not use the on-disk TOST result in the rebuttal.**
>
> The centralized arm on disk is `[0, 0, 0, 0, 0]` — exactly zero, with zero variance, across all five seeds. Inspecting the per-row detail in [centralized_routing_seed_1.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/centralized_routing/centralized_routing_seed_1.json:1) shows every prediction is an empty string (`"pred": ""`, `"hit": false`), so the router output is never being captured and every row defaults to a blank label. This is a **broken evaluation path returning a constant default**, not a measured baseline.
>
> Building a "paired difference = `+0.242`, not equivalent at `±0.03`" conclusion on top of this would read to reviewers as strong evidence *against* the paper's equivalence claim, when it is actually an artifact of a dead baseline arm. **It must not go in front of reviewers.**
>
> The manuscript's Table 27 reports **Centralized-SYNAPSE = 0.92 ± 0.02** (genuine spread), so a correct rerun should look like that, not a flat zero. Separately, the on-disk SYNAPSE arm (`0.242`, from a 100-query GSM8K task-accuracy run in [routing_100_client_count_1/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_100_client_count_1/summary.json:1)) is the **wrong artifact** for a routing-accuracy comparison against Table 27 and must also be replaced.

Grounded by:
- [artifacts/verification/routing_100_client_count_1/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/routing_100_client_count_1/summary.json:1)
- [artifacts/verification/centralized_routing/summary.json](/mnt/data1/achakr40/FederatedRAG/artifacts/verification/centralized_routing/summary.json:1)
- [scripts/tost_equivalence.py](/mnt/data1/achakr40/FederatedRAG/scripts/tost_equivalence.py:1)

**On-disk artifact (DEGENERATE — for diagnosis only, NOT for the rebuttal):**

| Field | On-disk value | Status |
| --- | ---: | --- |
| SYNAPSE per-seed | `[0.21, 0.23, 0.23, 0.23, 0.31]` | wrong metric (GSM8K task accuracy, not routing accuracy) |
| Centralized per-seed | `[0, 0, 0, 0, 0]` | broken arm — empty predictions, constant default |
| Paired diff / equivalence | `+0.242` / `False` | invalid — built on a dead baseline; discard |

**Conservative expectation once both arms produce real numbers (targets, NOT measured):**

| Checkpoint | What to verify | Expected band | Conservative target |
| --- | --- | ---: | ---: |
| SYNAPSE per-seed accuracy mean | Mean of 5 seeds | `0.90–0.93` | `0.92` |
| SYNAPSE per-seed SD | Across 5 seeds | `0.01–0.03` | `0.02` |
| Centralized per-seed accuracy mean | Mean of 5 seeds | `0.90–0.93` | `0.92` |
| Centralized per-seed SD | Across 5 seeds | `0.01–0.03` | `0.02` |
| Paired mean difference | SYNAPSE − centralized | `−0.02–+0.02` | `≈0.00` |
| 90% CI on the difference | Straddles 0 within margin | within `±0.03` | `[−0.02, +0.02]` |
| Equivalence at margin `±0.03` | TOST decision | expected `True` | `True` (confirm on real rerun) |

Interpretation:
The claim in Table 27 is **parity** with centralized routing, so a correct rerun should show a **near-zero paired difference with equivalence supported at `±0.03`** — not the large positive difference the degenerate artifact currently produces. Fix the centralized eval path first (the empty-prediction bug above), so it returns real, varying per-seed numbers near `0.92 ± 0.02`; only then run the TOST. These targets are expectations, not measurements, and must be replaced with the real rerun.

## Structured-but-Untyped Control Checkpoints

Current status:
No committed contradiction-injection harness exists in this checkout yet, so the safest validation target is the curve shape rather than an absolute claimed mean.

| Checkpoint | What to verify | Expected band | Conservative target (mean ± SD, ≥3 seeds) |
| --- | --- | ---: | ---: |
| 0% contradiction | Best point on curve | to be measured | `0.74 ± 0.02` |
| 20% contradiction | Drop from 0% | `0–4 pts lower` | `0.72 ± 0.02` (−2 pts) |
| 40% contradiction | Drop from 0% | `3–8 pts lower` | `0.69 ± 0.03` (−5 pts) |
| 60% contradiction | Drop from 0% | `6–14 pts lower` | `0.64 ± 0.03` (−10 pts) |
| Curve shape | Monotonicity of seed means | non-increasing or nearly non-increasing | non-increasing |

Interpretation:
These are expectations, not measurements. The `± SD` columns are placeholders for the spread a real run should report — each conflict level needs **at least 3 seeds** so the drop between levels can be compared against seed-to-seed noise, not read off single points. Build the contradiction-injection harness, run ≥3 seeds per level, and replace the targets with per-seed means and SDs before use.

## 0.5B / Embedder Checkpoints

Current status:
No completed run-backed artifact for this experiment was found on current head.

| Checkpoint | Config (backbone / retrieval) | Expected band | Conservative target (mean ± SD, ≥3 seeds) |
| --- | --- | ---: | ---: |
| Main backbone (reference) | LLaMA-3.1-8B | — | `0.90 ± 0.02` |
| Smaller backbone | Qwen2.5-0.5B-Instruct | `−2 to −10 pts` vs main | `0.84 ± 0.03` (−6 pts) |
| Weak-retrieval baseline | all-MiniLM-L6-v2, no reranker | reference for embedder swap | `0.84 ± 0.03` |
| Better embedder + reranker | bge-base-en-v1.5 + bge-reranker-base | `+1 to +6 pts` vs weak retrieval | `0.87 ± 0.02` (+3 pts) |

Interpretation:
These are expectations with named intended configs, not measurements. The point of the row is to force an **absolute** number (e.g. "0.5B backbone scored `0.84`") plus the model names actually swapped in, rather than a bare delta. Run each config with **≥3 seeds** and replace the targets with per-seed means and SDs; the deltas (`−6`, `+3`) are only the expected direction and rough magnitude.

## Table 14 / §C.4 Discrepancy Checkpoints

The open question is **not** Table 14's headline values themselves (restating those proves nothing). It is why **Experiment 4's own batch=3 / S=3 sweep point (`0.901`)** does not match **Table 14's S=3 entry (`0.92`)**. The table below therefore tracks the sweep configuration, including batch size, and must be reconciled against the headline rather than copied from it.

| Checkpoint | Config (batch × steps) | Expected band | Conservative target |
| --- | --- | ---: | ---: |
| Table 14 headline `S=3` | reported / aggregated rollup | `0.90–0.93` | `0.92` |
| Exp-4 sweep `S=3`, batch=3 | the disputed point | `0.89–0.91` | `0.901` |
| Exp-4 sweep `S=1`, batch=3 | below `S=3` | `0.86–0.90` | `0.88` |
| Exp-4 sweep `S=5`, batch=3 | near `S=3` | `0.89–0.92` | `0.905` |
| Extractive centroid | no TextGrad | `0.83–0.87` | `0.85` |
| Single-shot summarize | above none | `0.85–0.88` | `0.86` |
| No summarization | lowest | `0.76–0.80` | `0.78` |

Interpretation:
Two targets, not one: (a) the ordering `S=3 ≈ S=5 > S=1 > single-shot/extractive > none`, and (b) **reconciling the `0.901` sweep point against the `0.92` headline** — most plausibly a batch-size or aggregation-convention difference between the batch×step sweep and the Table 14 rollup. Identify which batch size and averaging convention Table 14 used; do not present the headline `0.92` as if it were the sweep result.

## ToolBench Extension Checkpoints

Current status:
No completed extension rerun is on disk, so only a conservative consistency band is safe.

Table 16's own methodology uses bootstrap confidence intervals for this kind of check, so the extension should report the same rather than a single point.

| Checkpoint | What to verify | Expected band | Conservative target (mean, bootstrap 95% CI) |
| --- | --- | ---: | ---: |
| 250-query baseline | Overall score | `0.70–0.75` | `0.72` (CI `0.66–0.77`) |
| 600–750 query extension | Same protocol | within `±0.03` of baseline | `0.72` (CI `0.68–0.76`) |
| Extension drift | Baseline vs extension, CI overlap | overlapping CIs, drift `0–3 pts` | overlapping |

Interpretation:
These are expectations, not measurements. Report each score with a **bootstrap 95% CI** (≥1,000 resamples, per Table 16), and check that the 250-query and extension CIs overlap. A large jump on the longer run, or non-overlapping CIs, usually indicates protocol drift rather than a pure sample-size effect.
