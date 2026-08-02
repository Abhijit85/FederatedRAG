# ToolBench Retrieval-at-Scale Audit (2026-08-02)

## Findings
- The current local checkout does not preserve the ToolBench `test_query_ids/` or `retrieval_test_query_ids/` directories referenced by the upstream README.
- The current local checkout does not preserve an explicit paper-time 32-tool base manifest.
- The bundled `data_example/instruction/` files are demo-sized only: `G1_query.json` has 5 queries, `G2_query.json` has 3, and `G3_query.json` has 2. They are not a substitute for the held-out retrieval benchmark split.
- The previously measured 32-tool row in `artifacts/verification/toolbench_retrieval_at_scale_run2/summary.json` remains a local reconstruction result, not a rebuttal-safe paper-mode replication.

## Harness Repair
- `scripts/run_toolbench_retrieval_at_scale.py` now accepts `--query-id-file` for an explicit held-out split.
- The same harness now accepts `--base-tool-manifest` for an explicit fixed 32-tool catalog.
- The new `--require-validated-provenance` flag refuses to run a validated 32-tool gate row unless both artifacts are supplied.

## Validation
- Local reconstruction dry run still works: `artifacts/verification/toolbench_retrieval_at_scale_audit_local/dry_run_summary.json`.
- Validated-mode dry run now fails fast on this checkout because the required held-out query IDs are absent.

## Status
- I did not rerun the 32-tool gate row in validated mode because the required provenance artifacts are not present locally.
- The next executable step is to supply the held-out query-ID file and the exact 32-tool manifest, then rerun `scripts/run_toolbench_retrieval_at_scale.py --catalog-sizes 32 --require-validated-provenance ...`.
