# Result bundle

This folder contains the aggregate outputs produced for the review follow-up.

- `grid_sensitivity.{json,csv}`: coarse 8-point versus fine training-fixed grid comparison.
- `fixed_gate_comparator.json`: existing fixed cleared-similarity gate comparison.
- `redaction_baseline_status.json`: existing uncalibrated field-level baselines and the status of the missing calibrated-redaction run.
- `oracle_sensitivity_status.json`: existing epsilon=0.05 aggregate and the status of the missing tolerance curve.
- `parser_accounting_status.json`: parser-contract check and the missing full-run parse counts.
- `preexisting_aggregate/`: snapshot of the checked-in aggregate JSON files.
- `sim_crc_rerun.json`: deterministic synthetic CRC sanity check rerun.
- `experiment_status.json`: per-experiment completion/skip/block status.

The human audit was not rerun, and the 70B run was skipped. No raw benchmark text,
row-level labels, prompts, model caches, or private annotations were copied here.
