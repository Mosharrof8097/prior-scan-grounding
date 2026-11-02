# Do 3D medical VLMs actually read the scan?

An audit of visual grounding in 3D medical vision-language models (VLMs) on chest/whole-body CT.
B.Sc. thesis, planned as a paper. Roadmap and checklist: [`PLAN.md`](PLAN.md).

The project has two parts that share one method (counterfactual input probes scored against a
text-only floor):

| part | question | model | data | state |
| :-- | :-- | :-- | :-- | :-- |
| **A. 3D-RAD audit** | Does a trained baseline use the (prior) CT on longitudinal VQA? | own baseline: LoRA Qwen2.5-1.5B + frozen ResNet-18 | 3D-RAD (CT-RATE) | done for epoch 1; controls and statistics open |
| **B. Mask-verified audit** | Does M3D-LaMed (Phi-3-4B) read the volume, and can training-free fixes recover grounding? | M3D-LaMed | TotalSegmentator v2 masks, ~18.7k questions (proposed) | **not started**, nothing in this repo yet |

Part A is the pilot and motivation for part B: it shows that a benchmark score can be reached
with the image removed. Part B is the main contribution.

## Part A: what is done

Findings 1–3 reproduce on CPU from `leak_analysis/` with no GPU and no weights. They are label
statistics over the released CSVs. The CSVs hold 16,031 Task 5 / Task 6 rows, but only 2,873 rows
(169 volumes, the official `test/` directory) have CT volumes; 13,158 rows (774 volumes) have none.
Numbers are given for all rows and for the evaluable rows. Quote the evaluable column for anything
about models.

**1. Task 6 is handed its own label.** Every Task 6 question embeds the past presence/absence
sequence as a bracket (`Past sequences of lung opacity: [1].`). It equals the ground-truth past bit
on all 16,031 rows and removes two of the four options. A rule that reads only the bracket scores
**75.4%** (all rows) and **78.9%** (evaluable rows).

**2. Tasks 5 and 6 are the same items.** Same 943 volumes, same 18 findings, 100.00% identical
answers on every matched pair. Task 5 is therefore a built-in un-leaked control for Task 6.

**3. Label-structure bounds.** Upper bounds assuming noise-free labels; they are not human ceilings.

| input available | all 16,031 rows | evaluable 2,873 rows |
| :-- | --: | --: |
| text only (Task 5 wording) | 0.5552 | 0.6411 |
| text only (Task 6 bracket) | 0.7539 | 0.7894 |
| Task 5 + perfect read of the current volume | 0.7392 | 0.7894 |
| Task 5 + perfect read of current and nearest prior | 0.8679 | 0.9558 |
| Task 5 + perfect read of every prior scan | 1.0000 | 1.0000 |

The bracket holds the whole history, so one prior scan reveals only its last entry; only the full
history reaches 1.0. Several other "Task 6 text-only" figures appear in the scripts (0.789 to
0.807) because scripts use different splits; they are not yet reconciled (see PLAN, phase 0).

**4. The audited baseline's answers do not depend on the CT.** Epoch-1 checkpoint (val_loss
0.3115) scored on full val (3,254) and test (3,459), once with real volumes and once with every
volume zeroed. Test, closed tasks:

| task | n | text-only floor | real CT | zeroed CT |
| :-- | --: | --: | --: | --: |
| 4 Existence Detection | 2,358 | 0.8355 | 0.8198 | 0.8155 |
| 5 Static Temporal | 340 | 0.6471 | 0.6353 | 0.6353 |
| 6 Longitudinal Temporal | 340 | 0.8059 | 0.7941 | 0.7941 |

The image reaches the model (pooler attention differs on 476/476 paired val items) but moves the
gold answer's log-probability by about 0.0055 nats; 0 of 238 val Task 6 predictions change. The
gaps to the floor are 0 to 4 items out of 340, inside sampling noise, so the supportable
statement is **indistinguishable from the text-only floor**, not below it. Limits: one seed, one
epoch, one perturbation (zeros), and only 14 val / 20 test temporal volumes. This is a statement
about this baseline, not about whether VLMs can do longitudinal CT reasoning. Details:
`inference_audit/RESULTS.md`.

The older "+7.0 / +6.1 points over text-only" claim came from a lost checkpoint, a different
metric and n=150. It is withdrawn.

**5. Split integrity.** Patient-level 80/10/10 split (seed 42): zero patient and zero QA-pair
overlap; 3,039 volumes hashed, 3,039 unique MD5s. Train has 9 exact duplicate QA rows (harmless).

**Does the baseline receive the prior scan?** Yes. Task 5/6 inputs stack the nearest earlier
same-patient study (T0) with the referenced study (T1); all 2,873 temporal rows have an earlier
study. Caveat: when the earlier study has several reconstructions, which one is T0 depends on
set iteration order (`retrain/kaggle_retrain_v7.py`, `build_timeline`).

## Part B: the plan in one paragraph

Build a benchmark of questions whose answers come from TotalSegmentator voxel masks, then audit
M3D-LaMed with four input probes (real, black volume, swapped patient, shuffled options) against a
text-only floor, and test training-free fixes (decoding calibration, zoomed re-reading first;
occlusion attribution and frozen-encoder probing if the first two show gain). A prior-art search
(report NC-6) found the probes themselves already exist in related work (2D counterfactual
grounding, blank-image ablations, tumour-centric 3D VQA). The open ground is the training-free
fixes in a 3D setting and the voxel-mask-grounded benchmark at scale, so the fixes must show a
measured gain or the work is incremental. Occlusion maps and encoder probing are known techniques;
claim them as applications.

Two things to verify before building on TotalSegmentator:
- whether M3D-LaMed's training data (M3D-Data / M3D-Seg) already contains TotalSegmentator
  volumes, which would be contamination;
- whether TotalSegmentator's metadata really carries the pathology labels the proposal assumes.

## Layout

| directory | contents |
| :-- | :-- |
| `PLAN.md` | phased roadmap and checklist |
| `leak_analysis/` | CPU-only audits behind findings 1–3, one JSON per script |
| `inference_audit/` | per-task evaluation of a checkpoint, with a blank-image ablation |
| `retrain/` | resumable baseline training with per-task validation |
| `dataset preparetion/` | download, patient-level split and duplicate-audit scripts |
| `baseline_training_pipeline/` | the original v6 training script |
| `literature review /` | literature review and 3D medical VLM comparison table |
| `split_manifests/` | split summary (manifests are regenerable) |
| `reports/`, `research_notes/` | external review of the project and the notes behind it |
| `novelty_report_NC-6_with_guide.pdf` | prior-art report for part B |

## Reproducing part A

Needs `pandas`, `numpy`, `scikit-learn` and the 3D-RAD data in `3D-RAD_Dataset/`.

```bash
python leak_analysis/text_only_leak_audit.py
python leak_analysis/task5_task6_twins.py
python leak_analysis/information_ceilings.py
python leak_analysis/verify_deleak.py
python leak_analysis/md5_volume_audit.py
python leak_analysis/save_split_manifests.py
```

## Data and licence

3D-RAD QA data is Apache 2.0; its CT volumes derive from CT-RATE (CC-BY-NC-SA, non-commercial
academic use with attribution). HuggingFace publishes `Images/test.zip` only, so the 136,195
train-split QA rows have no volumes; everything here uses the released test set re-split at
patient level. No dataset files, weights or reference papers are tracked.

## Author

Md Mosharrof Hossain · B.Sc. CSE, Mymensingh Engineering College, Bangladesh
· <mdmosharrofhossain455@gmail.com> · https://mosharrof8097.github.io
