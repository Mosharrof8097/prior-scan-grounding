# Kaggle smoke test (first GPU run of the new controls)

Goal: prove that the whole audit, including the five new controls (`mean`, `noise`, `swap`,
`zero_prior`, `swap_prior`), runs end to end on a T4 and writes outputs that pass the checks.
**The numbers from this run mean nothing** (10 items per task). Do not quote them.

The controls have only ever run in CPU unit tests (`tests/test_controls.py`), so this is their
first real test. Expect that something may need a fix; that is the reason for the smoke test.

Already checked on the laptop (2026-11-04): `--preflight` passes with the exact upload below.
Split matches the trained one (train 27,197 / val 3,254 / test 3,459), 0 volumes missing, and
all 238 val and 340 test Task 5 and Task 6 items are paired with a genuine earlier scan.

## 1. Upload (once)

1. **Checkpoint dataset.** Upload the folder `retrain/audit_upload/` as a new private Kaggle
   dataset (e.g. `v7-epoch1-audit`). It is 87 MB and holds `retrain_v7/epoch_1/` (projector and
   `lora_adapters/`), the three split manifests, and a copy of `kaggle_inference_audit.py`
   (identical to `inference_audit/kaggle_inference_audit.py`).
2. **Data dataset.** Attach the same 3D-RAD dataset you used for the earlier audit (the `.npy`
   volumes and the `Task*/` question CSVs).

## 2. Notebook settings

- Accelerator: **GPU T4 x1**. Persistence: off is fine.
- Internet: **On**, because the base model `Qwen/Qwen2.5-1.5B-Instruct` is downloaded from the
  Hugging Face Hub. (If Internet is unavailable, add that model as a dataset and pass
  `--llm_model /kaggle/input/<name>`.)
- Add both datasets as inputs.

## 3. Cells

**Easiest:** paste the whole of `inference_audit/kaggle_smoke_test.py` into one notebook cell and run it. It does Cells 1-3 below, checks the result, and zips the outputs to `/kaggle/working/smoke_out.zip`. The manual cells below do the same thing step by step.

```python
# Cell 1
!pip install -q -U peft
```
```python
# Cell 2: preflight, CPU only, seconds, loads no model
S = "/kaggle/input/v7-epoch1-audit/kaggle_inference_audit.py"   # adjust to the dataset slug
!python {S} --preflight --controls mean noise swap zero_prior swap_prior
```
Expect: `manifest counts match the trained split exactly`, `volumes referenced but missing on
disk: 0`, `paired_ok 238/238` (val) and `340/340` (test) for Tasks 5 and 6, and the checkpoint
picked as `epoch_1`. If any of that is different, stop and read the log.

```python
# Cell 3: the smoke test. 60 items per split = 10 per task; val only to halve the time.
!python {S} --limit 60 --splits val --controls mean noise swap zero_prior swap_prior
```
Expect a few minutes (an earlier 40-item real+blank run took about 2 minutes; this has seven
conditions; this is an estimate, not a measurement). If it passes 20 minutes, stop it and tell me.
If it dies with a CUDA out-of-memory error, restart the session so no earlier cell holds GPU memory, and send me the log. (This script has no free-memory check of its own; only the training script does.)

## 4. Bring back and check

Download from `/kaggle/working/inference_audit/` (individual files, not zipped): at least
`per_item_val.csv`, `per_task_val.csv`, `inference_audit_summary.json`, `run_log.txt`.
Put them in a folder (e.g. `inference_audit/smoke_out/`) and run on the laptop:

```bash
python inference_audit/check_smoke_output.py inference_audit/smoke_out \
    --splits val --controls mean noise swap zero_prior swap_prior
```
`PASS` means: every condition is present, each covers the same items as `real` (the two prior
controls only the 20 temporal items), no missing predictions, finite log-probabilities, and every
temporal item paired with a genuine earlier scan. The script also prints how often each control's
prediction equals the `real` prediction. That is information only; with 10 items per task it is
not evidence.

## 5. If it passes: the full run (next step, not part of this)

- Upload `retrain/results/retrain_v7/epoch_2` and `epoch_3` (adapters + projector; copies are in
  `backups/retrain_v7_2026-11-04/`) so epochs 1 to 3 can be audited. The script prefers `epoch_1`
  when several are mounted; use `--ckpt_dir` for each epoch.
- Full command: `python {S} --controls mean noise swap zero_prior swap_prior` (val and test).
- The reported epoch-1 audit used hash-order T0, so this rerun with the fixed T0 is what makes
  finding 4 final.
