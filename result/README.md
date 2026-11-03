# Result bundle

This folder contains the aggregate outputs produced for the review follow-up.

- `grid_sensitivity.{json,csv}`: coarse 8-point versus fine training-fixed grid comparison.
- `fixed_gate_comparator.json`: existing fixed cleared-similarity gate comparison.
- `lamp3_calibrated_redaction_qwen25_7b.json`: matched calibrated redaction replay on the fresh 150-user LaMP-3 cohort.
- `lamp3_oracle_exhaustive_sensitivity.json`: exhaustive 2^8/cap-3 oracle slice with epsilon sensitivity.
- `lamp3_secondary_exposure.json`: fresh grid/envelope exposure and CRC summaries; fine-grid reader utility is explicitly marked incomplete where applicable.
- `redaction_baseline_status.json`: status summary for the completed calibrated-redaction run plus legacy baseline context.
- `oracle_sensitivity_status.json`: status summary for the completed exhaustive/tolerance run plus legacy aggregate context.
- `parser_accounting_status.json`: parser-contract check over the actual auditor caches.
- `preexisting_aggregate/`: snapshot of the checked-in aggregate JSON files.
- `sim_crc_rerun.json`: deterministic synthetic CRC sanity check rerun.
- `experiment_status.json`: per-experiment completion/skip/block status.

The human audit was not rerun, and the 70B inference was postponed after downloading the model. No raw benchmark text,
row-level labels, prompts, model caches, or private annotations were copied here.
