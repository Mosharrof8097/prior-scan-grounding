# Checkpoint inference audit

`kaggle_inference_audit.py` loads the trained checkpoint and runs **inference only** — nothing is
trained. One pass produces the numbers the project is missing, broken down **per task** instead of
pooled.

## Why per-task

Tasks 5 and 6 are the same 16,031 clinical items with different wording: Task 6 hands the past
presence/absence bit to the model inside the question text, Task 5 does not. The earlier evaluation
pooled them into one "longitudinal" bucket, which averages away the one effect this thesis is
about. In the val split those 476 "longitudinal" items are 238 questions counted twice.

## Run it on Kaggle

```python
# Cell 1 — peft is required; the LoRA adapters cannot load without it
!pip install -q -U peft

# Cell 2 — PREFLIGHT. CPU only, a few seconds, loads no model.
# Checks the split reproduces, the adapters exist, MCQ coverage, T0/T1 pairing, missing files.
!python kaggle_inference_audit.py --preflight

# Cell 3 — smoke test on GPU, ~2 minutes, confirms scoring works end to end
!python kaggle_inference_audit.py --limit 40

# Cell 4 — the real run
!python kaggle_inference_audit.py
```

Paths are auto-detected (`/kaggle/input/*` holding `.npy` files, and the folder holding
`multimodal_projector_best.pt`). Override with `--data_dir` / `--ckpt_dir` if detection picks wrong.

**Run the preflight first.** It is free and it catches the two failure modes that would otherwise
waste a GPU session: missing LoRA adapters, and a split that does not match the trained one.

## Hard requirement: `lora_adapters/`

The checkpoint folder must contain **both**:

```
multimodal_projector_best.pt      <- vision side (you have this)
lora_adapters/                    <- the 4.3M trained LLM parameters
```

With the projector alone the LLM is the untouched base model, so every number would describe a
model that was never trained. The script refuses to run rather than report that. If the adapters
are gone, retraining is the only route.

## Outputs

Written as individual files to `/kaggle/working/inference_audit/` — **not zipped**, so each is one
click to download:

| file | contents |
| :-- | :-- |
| `per_task_<split>.csv` | the headline table: accuracy, text-only floor, gain, recency, per task |
| `per_item_<split>.csv` | every row's prediction, gold, pairing status — the substrate for the taxonomy work |
| `inference_audit_<split>.json` | the same aggregates as JSON |
| `inference_audit_summary.json` | all splits plus checkpoint metadata |
| `split_{train,val,test}_qa.csv` | the manifests, so the split is never lost again |
| `run_log.txt` | the full console log |

## What it measures

| metric | applies to | why |
| :-- | :-- | :-- |
| MCQ accuracy by length-normalised log-likelihood | Tasks 4, 5, 6 | the paper-comparable closed-ended number; replaces free-text exact match |
| Generation exact match + token F1 | Tasks 1, 2, 3 | these are `Open` with no real choices; F1 is informative where EM floors at ~0 |
| Teacher-forced token accuracy, per-item loss | all | continuity with the training-time metrics |
| Recency ratio T1/T0 | Tasks 5, 6, paired items only | reported per task, so Task 6's text-fed past no longer contaminates Task 5's |
| All of the above with the volume zeroed | all | the blank-image ablation |
| Text-only floor | all | computed from the train split, in the same table, so gains are readable at a glance |

The blank pass doubles as the **control for the recency ratio**: with a zero volume there is no
visual signal, so whatever ratio the aggregator reports is its no-information baseline. Comparing
the real ratio against it is what tells you whether 0.956 ever meant anything.

## Implementation notes that matter for correctness

- **Split safety.** The manifests were never saved, so the split is regenerated with the same
  algorithm and seed. That depends on CSV discovery order, so the script tries several orderings and
  keeps the one reproducing the reference counts (27,197 / 3,254 / 3,459 QA and 2,181 / 255 / 260
  volumes). If none matches it **aborts** rather than evaluate on training patients.
  Verified locally: "filesystem order" reproduces it exactly, with zero patient leakage.
- **Variable option count.** Task 4 is binary (`Choice A=Yes`, `B=No`, `C`/`D` are `-`); Tasks 5/6
  have four. Only real options are scored and the gold index is taken within that reduced list.
  Assuming four would silently make all 2,340 Task 4 items unscorable.
- **Identical prompt across options.** The prompt is truncated once, reserving room for the longest
  option (up to 32 tokens, the rule the training script used), and no option text is ever clipped —
  a clipped option would corrupt its own likelihood and bias the argmax.
- **Memory.** Option logits are cast to fp32 one at a time; doing all four at once would allocate
  ~350 MB twice on a 14.5 GiB T4.
- **Patient shuffle.** `np.array(list(df["PatientID"].unique()))`, matching the training script
  exactly. Passing a pandas extension array to `rng.shuffle` is not guaranteed to permute correctly
  on pandas ≥ 3, which would produce a different split.
- **I/O.** Volumes are iterated in filename order behind a small LRU cache, so each is read about
  once instead of once per QA row. The blank pass never touches disk at all.

## Known caveats to carry into the write-up

- **T0 reconstruction is ambiguous.** Where a patient has several reconstructions of the same study
  (`test_423_b_1`, `test_423_b_2`), the prior-scan lookup keeps whichever comes last, so the choice
  of T0 reconstruction is arbitrary. This is inherited from the training script and reproduced here
  deliberately, so evaluation matches training. Worth stating rather than fixing silently.
- **The earlier 0.7667 / 0.7733 figures** were strict exact match over 150 generated samples per
  bucket. The floors computed here are accuracy over every item. Once this run produces per-task
  MCQ accuracy, the comparison becomes like-for-like; until then, read the direction, not the gap.
- Only 3D-RAD's released image subset is usable: HuggingFace publishes `Images/test.zip` alone, so
  the 136,195 train-split QA rows have no volumes. Everything here is the official test set,
  re-split at patient level.

## Expected runtime

Roughly 25–40 minutes per split on a T4 with both passes (val 3,254 items, test 3,459). Use
`--no_blank` to halve it, or `--splits test` for the one split that matters most.
