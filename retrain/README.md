# Retrain v7

The v6 checkpoint is gone: its artefacts were written only to `/kaggle/working` of a session that
was never committed. `kaggle_retrain_v7.py` retrains the **same model on the same split with the
same hyperparameters**, so the result stays comparable to the 0.2853 val_loss already written up —
and makes losing it again very hard.

## Upload both files

```
kaggle_retrain_v7.py
kaggle_inference_audit.py     <- must sit next to it; the retrain calls it when training ends
```

## Run

```python
!pip install -q -U peft

!python kaggle_retrain_v7.py --preflight    # CPU, seconds. Confirms the split reproduces.
!python kaggle_retrain_v7.py --smoke        # ~5 min GPU. 2 epochs on 12 items/task. Proves saving works.
!python kaggle_retrain_v7.py                # the real run

!python kaggle_retrain_v7.py --resume       # if the session died partway
```

> **Commit the notebook: Save Version → Save & Run All.** An interactive session's
> `/kaggle/working` is deleted when the session ends. That is exactly how the v6 LoRA adapters and
> metrics were lost. Running `--smoke` first and committing once proves the output survives before
> you spend hours on the real run.

## What changed from v6

| | v6 | v7 |
| :-- | :-- | :-- |
| Saving | once, at the very end | **every epoch**: `last/`, `best/`, `epoch_N/` |
| LoRA adapters | best epoch only | every epoch, alongside optimizer state |
| Crash cost | the entire run | one epoch — `--resume` continues |
| Validation | pooled "single-scan" vs "longitudinal" | **per task, 1–6** |
| Accuracy metric | free-text exact match | **MCQ log-likelihood** over real options, plus EM for open tasks |
| Recency ratio | one pooled number | **Task 5 and Task 6 separately**, paired items only |
| Split | assumed | **verified against the v6 counts, aborts on mismatch** |
| Packaging | auto-zip | individual files |

## Why per-task is the point

Tasks 5 and 6 are the **same 16,031 clinical items** (same volumes, same 18 findings, 100%
identical answers) worded differently: Task 6 states the past presence/absence bit inside the
question, Task 5 does not. Pooling them averages away the only effect under study — in the val
split those 476 "longitudinal" items are 238 questions counted twice, once leaked and once not.

So **no separate de-leaked training run is needed for the core experiment.** Task 5 is already the
un-leaked twin and is already in the training data. One ordinary run yields the comparison; the
synthetic `3D-RAD-T` variants are only a later robustness check.

The two rows to read in every epoch report are Task 5 and Task 6. Text-only floors for reference:
Task 5 **0.618**, Task 6 **0.807**, Task 4 **0.826** (val split).

## Settings

Defaults reproduce v6: `--epochs 3 --batch_size 2 --grad_accum_steps 8` (effective 16),
`--lr 2e-4 --lora_r 16 --lora_alpha 32 --max_seq_len 128`, Qwen2.5-1.5B-Instruct, LoRA on
q/k/v/o_proj, frozen ImageNet ResNet-18 slice encoder, gradient checkpointing on.

`--val_mcq_budget 600` scores a stratified MCQ subsample each epoch to keep validation cheap; the
end-of-run audit scores every item on both val and test.

If the session limit looks tight, `--epochs 2` still lands near the v6 loss and halves the run.

## Expected runtime

~3 h/epoch on a T4 for 13,599 batches, so roughly 9–10 h for 3 epochs plus validation — close to
Kaggle's session ceiling, which is why per-epoch saving and `--resume` exist. The end-of-run audit
adds 25–40 min per split.

## Outputs

Individual files under `/kaggle/working/retrain_v7/`:

| file | contents |
| :-- | :-- |
| `training_history.json` | every epoch's losses, per-task validation, OOM count — **rewritten each epoch** |
| `val_per_task_by_epoch.csv` | the per-task table across epochs |
| `best/`, `last/`, `epoch_N/` | `multimodal_projector_best.pt` + `lora_adapters/` + `trainer_state.pt` |
| `split_{train,val,test}_qa.csv` | the manifests |
| `audit/` | the full per-task audit on val and test, including the blank-image ablation |
| `train_log.txt` | the whole console log |

## Caveats carried forward deliberately

- **T0 reconstruction is ambiguous.** Where a patient has several reconstructions of one study
  (`test_423_b_1`, `test_423_b_2`), the prior-scan lookup keeps whichever comes last. Reproduced
  from v6 on purpose so training and evaluation agree; state it rather than fix it silently.
- **Task 4 dominates**: 18,774 of 27,197 training items (69%), binary, with an 0.826 text-only
  floor. Any aggregate number is mostly Task 4, which is why per-task reporting matters.
- Only 3D-RAD's released image subset is usable — HuggingFace publishes `Images/test.zip` alone, so
  the 136,195 train-split QA rows have no volumes. This is the official test set, re-split by patient.
