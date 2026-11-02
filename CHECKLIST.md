# Checklist: what is done, what is left

`[x]` done · `[ ]` open · **CPU** = laptop is enough · **GPU** = Kaggle T4 or lab machine.
Phases and reasoning are in [`PLAN.md`](PLAN.md); results are in [`README.md`](README.md).

## Done

### Data and split
- [x] 3D-RAD downloaded; QA data and the released test CT volumes in place
      (`dataset preparetion/`, `3D-RAD_Dataset/`)
- [x] Patient-level 80/10/10 split, seed 42 (`split_manifests/`)
- [x] Duplicate audit: 3,039 volumes, 3,039 unique MD5s, zero cross-split leakage
      (`leak_analysis/md5_volume_audit.py`)

### Leak analysis (CPU, `leak_analysis/`)
- [x] Task 6 bracket equals the past bit on all 16,031 rows (`text_only_leak_audit.py`)
- [x] Tasks 5 and 6 are the same items, 100% identical answers (`task5_task6_twins.py`)
- [x] Floors and ceilings computed on pooled and evaluable rows (`information_ceilings.py`); rerun reproduces the README table
- [x] First de-leaked variant built (`build_3drad_t.py`), not yet verified

### Model and audit
- [x] Baseline built and retrained, v7 (`retrain/kaggle_retrain_v7.py`), final val_loss 0.2986
- [x] Blank-CT audit of the epoch-1 checkpoint on full val and test (`inference_audit/`):
      zeroing the volume changes nothing, 0 of 238 Task 6 predictions flip
- [x] Wrong "+7.0 / +6.1" claim found and withdrawn

### Writing and review
- [x] Literature review and 3D VLM comparison table (`literature review /`)
- [x] Proposal v1 and v2 (`*.docx`)
- [x] External project review (`reports/`, `research_notes/`)
- [x] Prior-art report for the new idea (`novelty_report_NC-6_with_guide.pdf`)
- [x] README rewritten, PLAN and this checklist written

## Not done yet

### Phase 0: clean-up (CPU, do first)
- [x] Ceilings on the 2,873 evaluable rows, "all prior scans" wording (script already did this; rerun matches README)
- [x] Reconcile the different "Task 6 text-only" numbers (`leak_analysis/TASK6_TEXT_ONLY_FIGURES.md`)
- [x] Digit-aware de-leak check (`verify_deleak_digits.py`): leak recovered 0.989 on the original, 0.715 / 0.711 on T-mask / T-strip, finding-only baseline 0.717
- [x] Deterministic earlier scan (T0) in all three pairing scripts (`tests/test_t0_determinism.py`)
- [ ] Rerun the epoch-1 audit with the fixed T0 (GPU); reported run used hash-order T0
- [ ] `git commit` the uncommitted files
- [ ] Back up `retrain/results/` (v7 adapters, only copy)
- [x] Add `requirements.txt` (CPU analysis)
- [x] Authorship: sole author for now; no co-authors listed
- [ ] Ask the senior: venue, deadline, GPU access, and how they are credited

### Part A: stronger evidence (GPU, inference only)
- [ ] Audit epochs 2 and 3
- [ ] Bootstrap confidence intervals per patient; McNemar test, Task 6 vs Task 5
- [ ] Extra controls: mean image, noise, another patient's volume, zeroed or swapped prior scan
- [ ] Fine-tuned text-only model as the fair floor; three seeds
- [x] Confirm that 3D-RAD-T really removes the bracket (2,873 test-dir rows only)

### Part B: new benchmark (CPU)
- [ ] Check if M3D-LaMed's training data already contains TotalSegmentator
- [ ] Check if TotalSegmentator metadata has pathology labels
- [ ] Download TotalSegmentator v2 (check disk space first)
- [ ] Question templates, then the ~18.7k question set from the masks
- [ ] Quality checks: duplicates, answer balance, text-only floor
- [ ] Shuffled-option and patient-swap versions

### Part B: model runs (GPU)
- [ ] M3D-LaMed runs on a T4
- [ ] Pilot on 100–200 questions: real / black / swapped / shuffled
- [ ] Full audit on all questions
- [ ] Same probes on 3D-RAD (second dataset, no new download)
- [ ] 2D baseline on slices from the same volumes
- [ ] Fix 1: decoding calibration
- [ ] Fix 2: zoomed re-reading
- [ ] Fix 3 and 4 (occlusion, encoder probing) only if fixes 1–2 show gain
- [ ] Per-method ablation

### Writing
- [ ] Related-work sections on blind baselines and longitudinal benchmarks
- [ ] Novelty stated precisely: fixes plus voxel-mask benchmark; probes are prior art
- [ ] Release plan for masks, questions and scripts
- [ ] Thesis chapter and paper draft

## What can be done without a GPU

Everything in Phase 0, all "Part B: new benchmark" items, and the writing. The GPU is needed only
for model inference (Part A controls, all Part B runs).

## Next three things

1. Back up the v7 adapters (Phase 0); venue, deadline, GPU access and credit for the senior can wait.
2. Check M3D contamination and the TotalSegmentator metadata (two quick lookups).
3. Write the question generator while waiting for GPU access.
