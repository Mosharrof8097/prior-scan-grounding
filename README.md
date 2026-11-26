# Do 3D medical VLMs actually read the scan?

An audit of visual grounding in 3D medical vision-language models (VLMs) on chest/whole-body CT.
B.Sc. thesis, planned as a paper. Roadmap and checklist: [`PLAN.md`](PLAN.md).

The project has two parts that share one method (counterfactual input probes scored against a
text-only floor):

| part | question | model | data | state |
| :-- | :-- | :-- | :-- | :-- |
| **A. 3D-RAD audit** | Does a trained baseline use the (prior) CT on longitudinal VQA? | own baseline: LoRA Qwen2.5-1.5B + frozen ResNet-18 | 3D-RAD (CT-RATE) | done for all epochs (1–3) with 7 controls and statistics |
| **B. Mask-verified audit** | Does M3D-LaMed (Phi-3-4B) read the volume, and can training-free fixes recover grounding? | M3D-LaMed | 3,039 preprocessed genuine 3D CT scans (`medi data 3d`) | Phase 2, 3, 4 completed on Kaggle Dual T4 (7,746 questions on 100% genuine CT scans; Real CT 21.09% vs Blank 23.43%, Swap Invariance 91.51%, Slice-Shuffling Invariance 94.53%) |

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
history reaches 1.0. Other "Task 6 text-only" figures in the scripts (0.785 to 0.807) are the same rule on different
splits; see `leak_analysis/TASK6_TEXT_ONLY_FIGURES.md` for which to quote.

**4. The audited baseline's answers do not depend on the CT.** Checkpoints across all three epochs (epoch 1 val_loss 0.3115, epoch 2 0.2999, epoch 3 / best 0.2986) were scored on full val (3,254) and test (3,459) across 7 input conditions (real, blank, mean, noise, swap, zero_prior, swap_prior). Test, closed tasks (epoch 3 / best checkpoint):

| task | n | text-only floor | real CT | zeroed CT | swapped CT |
| :-- | --: | --: | --: | --: | --: |
| 4 Existence Detection | 2,358 | 0.8355 | 0.8359 | 0.8321 | 0.8372 |
| 5 Static Temporal | 340 | 0.6471 | 0.6471 | 0.6441 | 0.6471 |
| 6 Longitudinal Temporal | 340 | 0.8059 | 0.8059 | 0.8118 | 0.8059 |

The image reaches the model (pooler attention differs on all items) but moves the gold answer's log-probability by only ~0.0055 nats. On Task 6, zeroing the CT flips 0 of 238 val predictions in epoch 1, and on test in epoch 3 blank CT actually scores higher (0.8118) than real CT (0.8059, p = 0.77). Swapping in an unrelated patient's CT leaves accuracy identical or slightly higher (0.8372 on Task 4, p = 0.66). Across all tasks, the model's performance is **indistinguishable from the text-only floor**. Limits: one baseline architecture, one seed, and 14 val / 20 test temporal volumes. This is a statement about this baseline and benchmark structure, not about whether VLMs can ever do longitudinal CT reasoning. Details: `inference_audit/RESULTS.md`.

The older "+7.0 / +6.1 points over text-only" claim came from a lost checkpoint, a different
metric and n=150. It is withdrawn.

**5. Split integrity.** Patient-level 80/10/10 split (seed 42): zero patient and zero QA-pair
overlap; 3,039 volumes hashed, 3,039 unique MD5s. Train has 9 exact duplicate QA rows (harmless).

**Does the baseline receive the prior scan?** Yes. Task 5/6 inputs stack the nearest earlier
same-patient study (T0) with the referenced study (T1); all 2,873 temporal rows have an earlier
study. Caveat: 92% of (patient, letter) scans have several reconstructions, and in the run that produced
finding 4 the one used as T0 depended on set iteration order (hash seed). The three pairing
scripts now break ties by sorted path (`tests/test_t0_determinism.py` passes); the rerun with fixed T0 is completed and verified on full val and test (Task 5 real CT accuracy moved slightly from 0.6353 to 0.6324 [1 item change], with all findings and lack of visual grounding replicating exactly; see `inference_audit/RESULTS.md`).

## Part B: Genuine 3D CT Full Audit Results (M3D-LaMed-Phi-3-4B)

Phase 2 (benchmark construction), Phase 3 (pilot verification), and Phase 4 (full canonical audit on genuine 3D CT scans) are **completed**:
1. **Benchmark Construction (`partB_benchmark/`):** Evaluated across 7,746 clinical questions spanning 3 tasks (Existence Detection, Static Temporal Diagnosis, Longitudinal Temporal Diagnosis) on 3,039 preprocessed real 3D CT volumes (`.npy`) from `medi data 3d` with zero dummy data and fail-fast validation.
2. **Phase 4 Full Canonical Audit on Kaggle Dual Tesla T4 (completed 2026-11-26, archived in `partB_benchmark/results/`):** Evaluated `GoodBaiBai88/M3D-LaMed-Phi-3-4B` under 6 counterfactual probe conditions (`real`, `blank`, `swap`, `shuffle_slices`, `shuffle_options`, `text_only`):

| Task / Group | N | Real CT | Blank CT (Zeros) | Swap Donor CT | Slice-Shuffled | Text-Only Floor | Δ (Real − Blank) | McNemar p (vs Blank) | McNemar p (vs Swap) | Prediction Invariance (Swap) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall (Canonical)** | **7,746** | **0.2109** | **0.2343** | **0.2098** | **0.2156** | **0.1936** | **-0.0234** | **p = 5.69e-14** | **p = 0.684** | **91.51%** |
| Task 4 (Existence Detection) | 2,000 | 0.4135 | 0.5000 | 0.4090 | 0.4310 | 0.3165 | -0.0865 | p = 1.57e-21 | p = 0.603 | 88.05% |
| Task 6 (Longitudinal Temporal) | 2,873 | 0.1462 | 0.1511 | 0.1455 | 0.1469 | 0.1542 | -0.0049 | p = 0.275 | p = 0.913 | 92.17% |
| Task 5 (Static Temporal) | 2,873 | 0.1347 | 0.1326 | 0.1354 | 0.1344 | 0.1476 | +0.0021 | p = 0.617 | p = 0.901 | 93.18% |

**Key Empirical Findings Across 7,746 Real CT Scans:**
* **Negative Visual Dependency:** Across all 7,746 items, Real CT accuracy (21.09%) is lower than Blank zero-volume accuracy (23.43%) with statistical significance ($p = 5.69 	imes 10^{-14}$, discordant pairs $b=201$ vs $c=382$). Visual input reduces overall accuracy by 2.34 percentage points, indicating that volume tokens act as non-informative distraction rather than clinical evidence.
* **Anatomical Swap Invariance:** Replacing the patient's genuine CT scan with an unrelated donor patient's CT volume leaves accuracy virtually identical (21.09% vs 20.98%, exact McNemar $p = 0.684$, discordant pairs $b=197$ vs $c=188$). The model emits identical clinical predictions on 91.51% of items regardless of donor identity.
* **3D Volumetric Invariance:** Permuting the 32 axial slices along the z-axis yields identical predictions on 94.53% of items ($p = 0.032$ overall, $p = 0.906$ on Task 6, $p = 1.000$ on Task 5), indicating lack of vertical 3D spatial structure grounding.
* **Option Selection Bias & Label Collapse:** Under real CT volumes, the model predicted Option 'B' on 74.76% of items, Option 'A' on 25.24%, and Options 'C' and 'D' on 0.00% of items, despite the ground truth distribution having Option 'D' at 47.56% and 'B' at 30.97%. The predictions reflect strong linguistic prior collapse over candidate option tokens.
* **Archive & Full Artifacts:** Full per-item scores, candidate logits, model outputs, and summary tables are persisted at `partB_benchmark/results/` (`m3d_full_real_ct_per_item.csv`, `M3D_REAL_CT_FULL_AUDIT_REPORT.md`).

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
| `partB_benchmark/` | TotalSegmentator v2 benchmark generator, manifests, pilot scripts and audit results |
| `novelty_report_NC-6_with_guide.pdf` | prior-art report for part B |

## Reproducing part A

Needs `pandas`, `numpy`, `scikit-learn` and the 3D-RAD data in `3D-RAD_Dataset/`.

```bash
python leak_analysis/text_only_leak_audit.py
python leak_analysis/task5_task6_twins.py
python leak_analysis/information_ceilings.py
python leak_analysis/verify_deleak.py          # TF-IDF; digit-blind, kept for the record
python leak_analysis/verify_deleak_digits.py   # digit-aware check of 3D-RAD-T
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
