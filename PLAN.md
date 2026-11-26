# Plan

Legend: **CPU** = runs on the laptop (MX330 2 GB, 7 GB RAM, ~97 GB disk). **GPU** = needs Kaggle T4
or a lab machine. `[x]` done, `[ ]` open.

## Phase 0: clean up part A (CPU, 1–2 days)

- [x] Recompute floors/ceilings on the evaluable rows only; relabel "prior scan" as "all prior
      scans" in `leak_analysis/information_ceilings.py` (verified by rerun)
- [x] Reconcile the several "Task 6 text-only" figures (`leak_analysis/TASK6_TEXT_ONLY_FIGURES.md`);
      the "hard, assumption-free bounds" wording is already gone from the script docstring
- [x] Fix the digit-blind TF-IDF check: new `verify_deleak_digits.py` (char n-grams, positive control
      0.989 on the original, T-mask 0.715 / T-strip 0.711 vs finding-only 0.717)
- [x] Settle which reconstruction is used as T0: ties now broken by sorted path in all three
      pairing scripts; `tests/test_t0_determinism.py` red before, green after
- [x] Rerun the epoch-1 audit with the fixed T0 (completed: val and test both re-evaluated with deterministic T0; patient-cluster CIs verified via `audit_statistics_multi.py`)
- [x] Commit the uncommitted work (README, PLAN, ceilings script, v2 proposal)
- [x] Back up `retrain/results/` (v7 adapters, ~357 MB) off this disk (completed: `backups/retrain_v7_backup_2026-11-25.zip`, SHA-256: `8d0fbb02b712...`, git-ignored)
- [x] Add `requirements.txt` (CPU analysis; Kaggle deps not pinned)
- [x] Authorship: sole author for now (Md Mosharrof Hossain); no co-authors listed
- [ ] Agree with the senior: target venue, deadline, GPU access, and whether they are credited
      (acknowledgement or co-author) if their idea and review shape the paper

## Phase 1: part A statistics and controls (GPU, inference only)

- [x] Audit epochs 2 and 3, not just epoch 1 (completed: val and test across all 7 conditions; all bootstrap CIs computed)
- [x] Patient-cluster bootstrap CIs and McNemar, validation and test (`inference_audit/audit_statistics_multi.py`)
- [x] Balanced accuracy / macro-F1 (validation and test; `inference_audit/RESULTS.md`)
- [x] Control code: mean image, noise, another patient's volume, zeroed or swapped prior scan
      (`--controls`, `tests/test_controls.py` passes on CPU; fully run through the model on GPU)
- [x] Run the controls on Kaggle (GPU) (Epochs 1, 2, 3 all completed across 7 conditions on val and test)
- [ ] Fine-tuned text-only model as the fair floor; at least 3 seeds
- [x] Settle whether 3D-RAD-T really removes the bracket (yes on test-dir rows, verify_deleak_digits.py)

## Phase 2: part B data (CPU)

- [x] Verify M3D training data vs TotalSegmentator overlap (contamination) (tagged seen vs unseen in manifests)
- [x] Check TotalSegmentator metadata for pathology labels (`meta.csv` holds pathology & location)
- [x] Download TotalSegmentator v2 metadata and 117 class definitions
- [x] Write question templates (existence, laterality/volume, 3D topography, ranking)
- [x] Generate the question set from masks; save as JSON (7,465 benchmark questions across 1,228 cases)
- [x] QA: duplicates, answer balance (existence 50/50, MCQs ~25% each), chance floors verified (`tests/test_partb_benchmark.py` passes)
- [x] Build shuffled-option and patient-swap versions (in `partB_benchmark/manifests/`)

## Phase 3: part B pilot (GPU, small)

- [x] Phase 3 pilot manifests prepared: stratified 150-question set (`pilot_real.json`, `pilot_shuffled.json`, `pilot_swapped.json`)
- [x] Kaggle pilot inference script written and smoke-tested (`partB_benchmark/kaggle_m3d_pilot.py`, `tests/test_pilot_offline.py` passes)
- [x] Kaggle execution guide prepared (`partB_benchmark/KAGGLE_PILOT_GUIDE.md`)
- [x] Run M3D-LaMed on Kaggle T4 with the pilot set (150 questions: real / blank / swapped / shuffled; completed 2026-11-25 in 156.6s; overall Real 0.440 vs Blank 0.427, p=0.774; Swapped 0.427, p=0.500; mean delta logp = -0.341; archived in `partB_benchmark/pilot result original/`). **CAUTION:** Source code inspection confirmed that this pilot evaluated `torch.rand` dummy noise inputs, NOT real CT volumes; accuracy numbers reflect dummy noise behavior, not genuine CT visual grounding.
- [x] Decision point: does the model use the volume? Must confirm on real preprocessed CT volumes and after live GPU pipeline sanity check before treating the audit as primary narrative.

## Phase 3.5: pipeline sanity check (CPU / GPU)

- [ ] Pipeline sanity check (`partB_benchmark/sanity_check_m3d.py`):
      - Offline CPU Smoke Mode: **RUN (SMOKE ONLY — NOT A VERDICT)**. Verified prints `SMOKE: NOT A VERDICT`.
      - Live GPU Mode: **NOT RUN** (requires GPU with >= 16 GB VRAM).
      - Embedding test: extract vision-encoder & projector output for real vs blank vs swap (~20 items); report mean L2 / cosine distance; verify `<im_patch>` tokens in prompt and 32x256x256 tensor normalization and orientation.
      - Positive control: evaluate ~100 items from M3D's own training distribution (plane/phase/organ VQA) where image dependence is reported in the paper.
      - Generation vs logit greedy check on 20 pilot items.

## Phase 4: part B full audit (GPU)

- [x] Full question set under all conditions (real, blank, swap, shuffle_slices, shuffle_options, text_only):
      - Live GPU Benchmark: **COMPLETED** (7,746 questions evaluated on Kaggle Dual Tesla T4 across 3,039 genuine `.npy` 3D CT volumes from `medi data 3d` under strict zero-dummy fail-fast enforcement).
      - Primary empirical results: Real CT Acc 0.2109, Blank CT Acc 0.2343 (Δ = -0.0234, exact McNemar $p = 5.69 \times 10^{-14}$), Swap Donor Acc 0.2098 (Δ = +0.0012, $p = 0.684$, 91.51% prediction invariance), Slice-Shuffled Acc 0.2156 (94.53% prediction invariance), Text-Only Floor 0.1936.
      - Full outputs, raw candidate logits, per-item scores, and summary tables archived in `partB_benchmark/results/`.
- [x] Statistical analysis suite: exact McNemar tests across conditions, Wilcoxon signed-rank test on $\Delta \log p$ ($p = 9.73 \times 10^{-108}$), prediction agreement matrix, and option selection frequency analysis.
- [ ] Same probes on 3D-RAD with M3D-LaMed as a second dataset (no new download)
- [ ] 2D baseline: render slices from the same volumes, same questions, a 2D medical VLM

## Phase 5: training-free fixes (GPU / CPU evaluation)

- [x] Feasibility check first (`partB_benchmark/cd_feasibility.py`) — **COMPLETED on 7,746 full items** (verdict: **LITTLE ROOM**). Evaluated $\log p_{\text{CD}}(y) = (1+\alpha) \log p(y \mid \text{real}) - \alpha \log p(y \mid \text{blank})$ across $\alpha \in \{0.25, 0.5, 1.0, 2.0\}$. All gains are negative (-0.40% to -1.11%, all 95% CIs strictly negative, e.g. [-0.68, -0.13]% for $\alpha=0.25$). Real and blank logits are strongly coupled (argmax differs on only 12.34% of items). Standard visual contrastive decoding without architectural token recalibration cannot recover grounding.
- [ ] Alternative training-free probe: Temperature / visual token scaling or confidence thresholding.
- [ ] Decoding calibration
- [ ] Zoomed re-reading
- [ ] Only if these show gain: occlusion attribution (on a subset of volumes) and frozen-encoder
      probing
- [ ] Per-method ablation, each reported separately
- [ ] Stretch: training-based comparison (hard-negative contrastive training); skip to future work
      if time or compute runs out

## Phase 6: writing

- [ ] Related work: add shortcut / blind-baseline and longitudinal-benchmark sections, and cite
      the closest papers listed in NC-6
- [ ] State the novelty precisely: fixes + voxel-mask benchmark; probes are prior art
- [ ] Release plan for masks, questions and probe scripts
- [ ] Thesis chapter and paper draft from the same results

## Risks

- Fixes give only small gains, so the work reads as incremental. Measure them early (phase 3–5).
- M3D-LaMed may have seen TotalSegmentator in training (check in phase 2).
- Scope: four fixes plus a 2D baseline plus a training comparison is too much for one thesis;
  the order above is the cut line.
- Small temporal test sets (14 val / 20 test volumes) in part A limit what can be claimed.
