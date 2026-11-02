# Plan

Legend: **CPU** = runs on the laptop (MX330 2 GB, 7 GB RAM, ~97 GB disk). **GPU** = needs Kaggle T4
or a lab machine. `[x]` done, `[ ]` open.

## Phase 0: clean up part A (CPU, 1–2 days)

- [ ] Recompute floors/ceilings on the evaluable rows only; relabel "prior scan" as "all prior
      scans" in `leak_analysis/information_ceilings.py`
- [ ] Reconcile the several "Task 6 text-only" figures to the manifest split; remove the "hard,
      assumption-free bounds" wording
- [ ] Fix the digit-blind TF-IDF check in `verify_deleak.py` (character-level or digit-aware)
- [ ] Settle which reconstruction is used as T0 (deterministic ordering in `build_timeline`)
- [ ] Commit the uncommitted work (README, PLAN, ceilings script, v2 proposal)
- [ ] Back up `retrain/results/` (v7 adapters, ~357 MB) off this disk
- [ ] Add `requirements.txt` so the environment can be rebuilt
- [ ] Agree with the senior: authorship order (first author), target venue, deadline, GPU access

## Phase 1: part A statistics and controls (GPU, inference only)

- [ ] Audit epochs 2 and 3, not just epoch 1
- [ ] Patient-cluster bootstrap CIs; McNemar for Task 6 vs Task 5 on the same items;
      balanced accuracy / macro-F1
- [ ] More controls: mean image, noise, another patient's volume; for Task 5 a zeroed or swapped
      prior scan
- [ ] Fine-tuned text-only model as the fair floor; at least 3 seeds
- [ ] Settle whether 3D-RAD-T really removes the bracket

## Phase 2: part B data (CPU)

- [ ] Verify M3D training data vs TotalSegmentator overlap (contamination)
- [ ] Check TotalSegmentator metadata for pathology labels
- [ ] Download TotalSegmentator v2 (check disk first)
- [ ] Write question templates (existence, organ size, position, laterality, comparison)
- [ ] Generate the question set from masks; save as JSON
- [ ] QA: duplicates, answer balance, text-only majority floor per template
- [ ] Build shuffled-option and patient-swap versions (swap pairs matched by scanner / region)

## Phase 3: part B pilot (GPU, small)

- [ ] M3D-LaMed loads and runs on Kaggle T4 (weights, transformers version)
- [ ] 100–200 questions under real / black / swapped / shuffled
- [ ] Decision point: does the model use the volume? If not, the audit is the story; if yes,
      the fixes carry it

## Phase 4: part B full audit (GPU)

- [ ] Full question set under all four conditions
- [ ] Bootstrap CIs, per-organ error vs organ size / volume / position
- [ ] Same probes on 3D-RAD with M3D-LaMed as a second dataset (no new download)
- [ ] 2D baseline: render slices from the same volumes, same questions, a 2D medical VLM

## Phase 5: training-free fixes (GPU)

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
