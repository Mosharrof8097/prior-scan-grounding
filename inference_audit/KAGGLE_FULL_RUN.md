# Full audit on Kaggle: one epoch per run, val and test, 7 conditions

File to paste into ONE notebook cell: `inference_audit/FULL_RUN_ALL_IN_ONE.py`
(built by `build_full_run.py`; it embeds the current audit script with both fixes: `with_item_ids`
and `majority_answer`, audit sha256 `294fc43d6d8c`).

Conditions per split: `real`, `blank`, `mean`, `noise`, `swap`, `zero_prior`, `swap_prior`
(the two prior controls run on the Task 5 and Task 6 items only: 476 in val, 680 in test).

## What to upload (once)

| Dataset | Folder on the laptop | Holds |
| :-- | :-- | :-- |
| already on Kaggle: `medi-data-3d` | | the 3D-RAD volumes and Task CSVs |
| already on Kaggle: `inference-audit-data` | `retrain/audit_upload/` | epoch_1 + split manifests |
| NEW, 163 MB | `retrain/audit_upload_epochs_2_3/` | epoch_2, epoch_3 + split manifests (`SHA256SUMS` inside) |

## Per run

1. Notebook: GPU T4 x1, Internet ON. Attach `medi-data-3d` plus the dataset that holds the epoch:
   epoch 1 -> `inference-audit-data`; epochs 2 and 3 -> the new epochs-2-3 dataset.
2. Paste `FULL_RUN_ALL_IN_ONE.py` into one cell. The only line to edit is `EPOCH = "epoch_1"`.
3. Run it. Unattended: Save Version -> Save & Run All (Commit) with GPU on.
4. Download `full_run_<EPOCH>.zip` from the Output panel. It is rewritten after each split, so a dropped
   session still leaves the val results.

Time (estimate, not measured on the full set): 35,877 forward passes per epoch, 0.28 to 0.5 s each, so
about 3 to 5 hours per epoch; three epochs about 9 to 15 hours of the weekly Kaggle GPU quota.

## What it checks before spending GPU time

SHA-256 of the projector, adapter weights, adapter config and the three split manifests against embedded
hashes; the epoch number stored inside the projector file; then the audit's own preflight (split matches the
trained one, no volume missing, all Task 5 and Task 6 items paired with an earlier scan).

## What it checks after each split, and stops on failure

All 7 conditions present; each covers the same items as `real` (prior controls: the temporal items only);
`real` has exactly 3,254 (val) or 3,459 (test) rows and 476 / 680 temporal rows; no missing predictions or
non-finite log-probabilities; every temporal item `paired_ok`; the audit's log shows the intended epoch loaded.

## Bringing results back

Unzip into a NEW folder, for example `inference_audit/full_run_epoch_1/`. Do not put them in
`inference_audit/interface audit result/`, which holds the earlier published audit. Nothing in this
procedure modifies README.md, RESULTS.md or any existing result file.
