# Review-requested experiment results

Updated: 2026-11-03

Human audit was skipped because it was already completed. The 70B model was downloaded but inference was postponed by the latest instruction.

Completed additional runs are listed in `experiment_status.json`; parser response counts are in `parser_accounting.json`.

Completed standalone fine-grid reader utilities are in the `lamp3_fine_grid_utility_qwen25_7b.json`, `lamp7_fine_grid_utility_qwen25_7b.json`, and `lampqa_fine_grid_utility_qwen25_7b.json` files; each recorded coverage has zero missing contexts. The LaMP-7 and LaMP-QA files use a separate 150-user union-auditor cohort and do not overwrite the legacy 300-user exposure aggregates.

The result folder contains aggregate outputs only. Raw benchmark rows, prompts, row-level auditor labels, and LLM caches remain in the artifact workspace.
