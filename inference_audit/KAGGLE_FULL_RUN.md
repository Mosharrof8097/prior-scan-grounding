# Full audit on Kaggle: ONE file, all epochs, no flags

Paste `inference_audit/ALL_EPOCHS_FULL_AUDIT.py` into one notebook cell and run it. Nothing to edit.
(Built by `build_all_epochs.py`; it embeds the audit script with both fixes, `with_item_ids` and
`majority_answer`, sha256 `294fc43d6d8c`.)

## Attach to the notebook
`medi-data-3d`, `inference-audit-data` (epoch 1 + manifests), `audit-upload-epochs-2-3` (epochs 2 and 3).
GPU T4 x1, Internet ON. For an unattended run: Save Version -> Save & Run All (Commit), GPU on.

## What it does
1. Finds epoch_1/2/3 among the attached datasets; verifies SHA-256 of projector, adapter, adapter config and the
   split manifests, and the epoch number stored inside each projector. A missing or corrupted epoch is reported
   and skipped; the others run.
2. Preflight on CPU. 3. For each epoch, val then test, 7 conditions: real, blank, mean, noise, swap, zero_prior,
   swap_prior (prior controls only on the Task 5 / Task 6 items: 476 val, 680 test).
4. After every split: checks (7 conditions, full row counts 3,254 / 3,459, temporal items all paired, intended
   epoch loaded) and rewrites `full_audit_results.zip`. A failed split stops the run so no GPU time is wasted.
5. Stops cleanly before a split that would not finish inside the 11-hour budget.

## Time (estimate, not measured on the full set)
35,877 forward passes per epoch at 0.28 to 0.5 s each = about 3 to 5 hours per epoch. Three epochs are
107,631 passes, about 8 to 15 hours, so they may not fit one 12-hour session. If the budget stops the run:
upload `full_audit_results.zip` as a Kaggle dataset, attach it in a NEW session with the other datasets, run the same
file again; finished (epoch, split) pairs that pass every check are reused.

## The one download
`full_audit_results.zip`: per_item / per_task / json / run_log for every epoch and split, `summary.csv`,
`status.json`, `run_info.json` (versions, GPU, hashes, timings), `console_log.txt`, and the audit script that ran.
Unzip into a NEW folder such as `inference_audit/full_run/`, not into `interface audit result/`.
Nothing in this procedure modifies README.md, RESULTS.md or any existing result file.
