# prior-scan-grounding

Does a 3D medical vision-language model actually read the **prior** CT scan when asked how a
finding has changed over time?

Work towards a B.Sc. thesis on longitudinal visual question answering over 3D chest CT, built on
the [3D-RAD](https://huggingface.co/datasets/Tang-xiaoxiao/3D-RAD) benchmark (NeurIPS 2025 Datasets
and Benchmarks). The starting hypothesis was *temporal recency bias* — that models over-attend the
current scan and under-use the prior one. Measuring it led somewhere more interesting.

## Findings so far

All of the following are reproducible on CPU from `leak_analysis/`, with no GPU and no model
weights. Numbers are on a patient-level 80/10/10 split (seed 42, zero leakage verified).

**1. Task 6 is handed its own label.** Every Task 6 question embeds the past presence/absence
sequence as a literal bracket — `Past sequences of lung opacity: [1]. (Note: 0 indicates absence,
and 1 indicates presence.)`. That bracket equals the ground-truth past bit on all 16,031 rows. It
eliminates two of the four answer options with no exceptions, leaving **75.4%** of Task 6 solvable
without reading a single voxel.

**2. Tasks 5 and 6 are the same items, worded differently.** Same 943 volumes, same 18 findings,
**100.00% identical answers** across every matched (volume, finding) pair. Task 6 reveals the past
bit in text; Task 5 does not. 3D-RAD therefore ships its own matched leaked / un-leaked control
pair — no synthetic de-leaking is needed to study the effect.

**3. What is achievable is fixed by the label structure.** The four answers are a product of two
bits — was the finding present before, is it present now:

| input available | best possible accuracy |
| :-- | --: |
| text only (Task 5 wording) | 0.5552 |
| text only (Task 6 bracket) | 0.7539 |
| Task 5 + a perfect read of the single provided volume | 0.7392 |
| Task 5 + the prior scan as well | 1.0000 |

Task 5 as the benchmark specifies it supplies one volume, so the past bit has no source at all: a
flawless reader cannot exceed **73.9%**. Supplying the prior scan lifts the ceiling to 1.0 and
makes both bits image-derived.

**4. A trained 3D VLM adds little over the text-only floor.** A LoRA-tuned Qwen2.5-1.5B with a
2D-slice encoder and cross-attention depth aggregator (epoch 3, val_loss 0.2853) scored 0.7667 on
single-scan and 0.7733 on longitudinal questions, against text-only floors of 0.6969 and 0.7122 —
**+7.0 and +6.1 points**. Per-task floors: Task 4 **0.826** (binary, class imbalance), Task 6
**0.807** (bracket), Task 5 **0.618**, Tasks 1–3 ≈ 0.01.

**5. Split integrity.** Patient-level split with zero patient, volume or QA-pair leakage. The
bitwise volume-hash audit now completes: 3,039 volumes hashed, 3,039 unique MD5s, zero identical
clusters, zero cross-split hash leakage.

## Layout

| directory | contents |
| :-- | :-- |
| `leak_analysis/` | the CPU-only audits behind every number above, one JSON per script |
| `inference_audit/` | per-task evaluation of a trained checkpoint, with a blank-image ablation |
| `retrain/` | resumable baseline training with per-task validation |
| `dataset preparetion/` | download, patient-level split and duplicate-audit scripts and write-up |
| `baseline_training_pipeline/` | the original v6 training script |
| `literature review /` | literature review and a comparison table of 3D medical VLMs |
| `split_manifests/` | split summary (the manifests themselves are regenerable) |

Each directory has its own README.

## Reproducing the analysis

Needs only `pandas`, `numpy`, `scikit-learn`, and the 3D-RAD dataset in `3D-RAD_Dataset/`.

```bash
python leak_analysis/text_only_leak_audit.py     # per-task text-only accuracy
python leak_analysis/task5_task6_twins.py        # the matched twin check
python leak_analysis/information_ceilings.py     # floors and ceilings
python leak_analysis/verify_deleak.py            # that de-leaking removes the shortcut
python leak_analysis/md5_volume_audit.py         # bitwise volume hash audit
python leak_analysis/save_split_manifests.py     # regenerate the split
```

## Status

The baseline's per-task numbers are pending: the v6 checkpoint's LoRA adapters were lost with an
uncommitted Kaggle session, so `retrain/kaggle_retrain_v7.py` reproduces the same model on the same
verified split, saving every epoch and resumable. The open question is whether that +6.1 points on
longitudinal questions is real visual reasoning or better template fitting — which the per-task
split of Task 5 against Task 6, plus the blank-image ablation, answers directly.

## Data and licence

3D-RAD QA data is Apache 2.0; the CT volumes derive from CT-RATE (CC-BY-NC-SA, non-commercial
academic use with attribution). No dataset files, model weights or reference papers are tracked
here. Note that HuggingFace publishes `Images/test.zip` only, so the 136,195 train-split QA rows
have no volumes — everything here uses the released test set, re-split at patient level, and that
limitation is stated wherever results are reported.

## Author

Md Mosharrof Hossain · B.Sc. CSE, Mymensingh Engineering College, Bangladesh
· <mdmosharrofhossain455@gmail.com> · https://mosharrof8097.github.io
