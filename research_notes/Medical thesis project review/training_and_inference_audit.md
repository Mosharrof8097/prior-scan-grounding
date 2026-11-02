# Baseline 3D VLM: training, retraining and the blank-image inference audit

Paths are under /home/mosharrof/medical_thesis (abbrev. `P/`). Source type for all findings: local files (no web sources used). Confidence tags: [H]igh / [M]edium / [L]ow. Note: the project is a git repo per `git log` but the harness reported it as non-git; commits are cc4615c (2026-10-08, initial) and 0515556 (2026-10-17, "Blank-image audit...").

## 1. What the blank-image audit measured, and is the conclusion justified?

### Takeaway
The audit scored the epoch-1 retrain_v7 checkpoint (not the best/final epoch) on full val (3,254) and test (3,459), twice each: real volume vs an all-zero volume. Accuracy is unchanged within 0.0042 on every task and exactly 0.0000 on Task 6, so for THIS checkpoint the answers are effectively image-independent. The conclusion "this score is not evidence of grounding" is justified [H]; stronger wording ("provably independent", "the epoch-1 result generalises to the trained baseline") is not [M].

### Cited Findings
- Design: closed Tasks 4/5/6 scored by length-normalised log-likelihood over real options (text of the option, not the letter); open Tasks 1-3 by greedy generation, strict normalised EM plus token F1; every metric run with `torch.zeros_like(img)` fed to the same model — `P/inference_audit/kaggle_inference_audit.py` (`score_item`, lines ~645-760; `__getitem__` blank branch ~lines 453-457) and `P/inference_audit/RESULTS.md`.
- Checkpoint audited: `{"epoch": 1, "val_loss": 0.31154956...}` from `P/inference_audit/interface audit result/inference_audit_summary.json`; ckpt_dir `.../retrain_v7/epoch_1`. Epochs 2 and 3 (val_loss 0.2999 / 0.2986) were never audited; only epoch_1 is in `P/retrain/audit_upload/`; all epochs exist in `P/retrain/results.zip` (and unzipped `P/retrain/results/retrain_v7/`).
- Test (n=3,459), real / blank / text-only floor (`P/inference_audit/interface audit result/per_task_test.csv`):
  - T1 (93): 0.1398 / 0.1398 / 0.0108 (F1 0.4451 vs 0.4423)
  - T2 (244): 0.2131 / 0.2090 / 0.0123
  - T3 (84): 0.0000 / 0.0000 / 0.0000 (F1 0.4127 both)
  - T4 (2,358): 0.8198 / 0.8155 / 0.8355
  - T5 (340): 0.6353 / 0.6353 / 0.6471
  - T6 (340): 0.7941 / 0.7941 / 0.8059
- Val (n=3,254), real / blank / floor (RESULTS.md): T1 0.2159/0.2159/0.0114; T2 0.1612/0.1612/0.0083; T3 0.0093/0.0093/0; T4 0.8115/0.8103/0.8256; T5 0.6218/0.6176/0.6176; T6 0.8025/0.8025/0.8067.
- "Not an artefact" checks (RESULTS.md, val): pooler attention differs on 476/476 paired items; gold log-prob differs on 2,738/2,816 closed items, mean 0.0055 nats, max 0.0624; 35/2,816 (1.24%) closed predictions flip; 0/238 on Task 6. My recomputation from `per_item_val.csv`: T4 33 flips of 2,340; T5 2 flips; T6 0 flips (matches 35 total).
- Pooler attention is near-uniform over slices: recency T1/T0 = 1.0009 real vs 0.99997 blank on test (per_task_test.csv); 1.0019-1.0114 across training epochs (`P/retrain/metrics/val_per_task_by_epoch.csv`). The README's earlier "0.956" recency figure (inference_audit/README.md) is therefore not meaningful as a signal.
- The model's closed predictions are near-degenerate (recomputed from per_item_val.csv, real condition): T4 predicts "No" (B) 2,157/2,340 (92%) vs gold 82.6% No; T5 predicts only A or D (D 217/238) vs gold D 147, A 45, B 26, C 20; T6 predicts only A, B(9), D — never C; every gold-C item (20) is predicted D, 22/26 gold-B are predicted A. T6 confusion is identical in real and blank. I.e. T6 accuracy is explained by a lookup of the in-question bracket, which is exactly the leak documented in `P/README.md` (finding 1: bracket equals past bit on 16,031 rows; 75.4% solvable without pixels).
- Teacher-forced token accuracy on T6 = 0.983 vs 0.794 MCQ accuracy (RESULTS.md) — TF accuracy is dominated by trivial tokens/shared answer text and is not a usable accuracy proxy.
- Open tasks: above-floor margin (T1 +0.129, T2 +0.201 test) is also blank-invariant, so it comes from LM priors (common answers like "Right lung lower lobe": 65/71 T2 predictions on val are that string). Note T1-3 "floor" is ~0.01 because it is majority-answer on a high-cardinality target; a stronger text-only baseline (a text-only fine-tuned LLM, or question-type nearest-neighbour) would be a fairer comparison.

### Inferences
- The blank ablation tests only "does zeroing the input change the output". It cannot show the image is useless in principle, only that this checkpoint's output distribution is insensitive (Δ≈0.005 nats) [H].
- Statistical power: the headline test deltas for T5/T6 are 0 and −0.0118, i.e. 0-4 items of 340. Binomial 95% CI half-width at val n=238 is ±0.05-0.06 (T6 ±0.051, T5 ±0.062); at n=340 about ±0.04. The "below the floor" claims (−0.0118 on T5/T6, −0.0157 on T4 test) are within noise; the commit message/RESULTS phrase "below the text-only floor on all three closed tasks" is directionally true but not statistically supported [H]. Better wording: "indistinguishable from the floor".
- Effective sample size is smaller than n: test T5/T6 cover only 20 patients (340 items; T5 and T6 are the same items worded differently), T4 131 patients (computed from `split_test_qa.csv`). Items within a patient are correlated; use patient-level bootstrap [H].
- The floor uses train-majority answer per (task, bracket-group) (`text_only_floor`, line ~797). For T4 the model (0.8198) is below always-"No" (0.8355): the baseline never learned to out-predict the prior on T4, a stronger statement than "ignores the image" — it is also not extracting useful visual signal (balanced-accuracy / AUROC on T4 not reported) [M].
- Epoch 1 vs 3: the per-epoch MCQ numbers on 100-item subsamples show no upward trend in T4/T5/T6 (T4 0.87→0.85→0.82; T5 0.62→0.64→0.58; T6 0.80→0.76→0.77; `val_per_task_by_epoch.csv`), so there is no sign that more training would reveal visual use, but the subsample CI is ±0.07-0.10 so this is weak evidence. The audit should be re-run on epoch 3/best [M].
- Zero-volume is a single particular off-distribution input; a mean-image or shuffled-real-volume control is stronger evidence (see section 5) [M].

### Gaps
- `per_task_val.csv` and `per_item_test.csv` were not retrieved (RESULTS.md says so); per-item test predictions are therefore unavailable for taxonomy/patient-level CIs.
- No audit of epochs 2/3; no seed replicates.

## 2. Training setup, curves, and v6 vs v7

### Takeaway
v7 faithfully re-trains v6's configuration (Qwen2.5-1.5B-Instruct + LoRA r16/α32 on q/k/v/o, frozen ImageNet ResNet-18 slice encoder, cross-attention pooler to 16 tokens, 2-layer GELU projector, 3 epochs, eff. batch 16, lr 2e-4) on the identical verified split, but did not reproduce v6's val_loss (0.2986 vs 0.2853) and the original v6 adapters remain lost. Validation loss is dominated by Task 4; per-task curves are flat after epoch 1.

### Cited Findings
- v7 setup (`P/retrain/metrics/train_log.txt`, `P/retrain/README.md`): Tesla T4, bf16, peft 0.20.0, 1,563,718,080 params, trainable 8,833,536 (0.56%) (LoRA adapters ≈4.36M by file size 17,462,432 B at fp32, rest aggregator+projector), batch 2 × accum 8 = 16, 13,599 batches/epoch, ~0.36 s/batch (~85 min/epoch per log ETA; README estimates ~3 h/epoch), seed 42 set (`kaggle_retrain_v7.py` lines 752-754), max_seq_len 128, 0 OOM-skipped batches. Split: train 27,197 / val 3,254 / test 3,459, 3,039 volumes, 1,304 patients, "MATCHES the v6 split", zero patient leakage.
- Loss curve (`P/retrain/metrics/training_history.json`): train 0.4864 / 0.3017 / 0.2666; val 0.3115 / 0.2999 / 0.2986 (ppl 1.37/1.35/1.35). Training loss keeps falling while val plateaus (gap 0.032 at epoch 3), mild overfit. Best = epoch 3 (0.2986).
- Per-task val (`val_per_task_by_epoch.csv`; epoch 1/2/3): loss T1 1.027/0.924/0.916, T2 1.088/0.985/0.998, T3 1.196/1.133/1.175, T4 0.214/0.215/0.211, T5 0.089/0.092/0.087, T6 0.0385/0.0395/0.0381. TF acc T4 0.907/0.913/0.913, T5 0.970/0.970/0.971, T6 0.983/0.981/0.982. MCQ acc on 100-item subsamples: T4 0.87/0.85/0.82, T5 0.62/0.64/0.58, T6 0.80/0.76/0.77 (T5 floor 0.618, T6 0.807, T4 0.826 from the README). The pooled val_loss is thus dominated by T4 (69% of items) and open tasks; T5/T6 losses are tiny because the answers are one of a few fixed long strings.
- v6 vs v7 differences: v6 script `P/baseline_training_pipeline/3d_rad_training_pipeline_fixed_v6.py` (README there still names `train_baseline_kaggle.py` and shows batch 4/accum 4 — stale; script itself defaults 2×8, seed 42). v6 saved only at end of run (best by val loss); v7 saves each epoch, resumable. v6 validated pooled "single-scan" vs "longitudinal" with free-text EM over 150 (script default gen_max_samples 200; README says 150) samples; v7 per-task with MCQ log-likelihood. v6 val_loss 0.2853 (epoch 3) is cited only in `P/README.md` and `P/leak_analysis/*`, with no log in the repo.
- Surviving v6 artefact: `P/baseline_training_pipeline/baseline result/multimodal_projector_best.pt` (62,668,667 B, md5 2e905b70…), no LoRA adapters. Its md5 differs from v7 epoch_1 projector (4b139bad…) — consistent with it being v6's separate run. A projector without LoRA adapters is the untrained-LLM case; the audit script refuses to run on it (inference_audit/README.md "Hard requirement").
- "What is new" at 2026-10-24: `retrain/audit_upload/retrain_v7` directory shows 2026-10-24 16:40, but all files inside are dated 2026-10-16/17. I found no content new after 2026-10-17 (last commit 0515556, 2026-10-17 01:19). The 10-24 mtime is a directory touch (likely a copy/rename); the mtime is not evidence of new work [M-H].
- Duplicates: `P/retrain/kaggle_inference_audit.py` (Oct 8, 44,996 B) is the older audit; `P/inference_audit/kaggle_inference_audit.py` (Oct 16, 49,657 B) has the discovery fixes. The audit run's results used the newer one.

### Inferences
- Gap 0.2986 vs 0.2853 (+0.0133, 4.7% rel.) is within plausible run-to-run variation for a 1.5B LoRA run with bf16, gradient checkpointing and shuffled batches, or may reflect a different loss aggregation (v6 "per-item" mean; v7 per-item mean also but a different validate function). The README's "same split, same hyperparameters" is true on paper but the claim of comparability is not proven because v6 produced no recoverable logs [M]. Nothing in the audit hinges on this difference.
- The headline README claims about v6 (0.7667/0.7733) cannot be re-checked and v7 does not replace them; they should be dropped or labelled as unverified in the thesis [H].

### Gaps
- v6's exact data loader/seed behaviour vs v7 (e.g. whether any v7 change altered augmentation/shuffling) was not diffed line by line.
- Whether any hyperparameter schedule (warmup/cosine) differs between v6 and v7 was not checked.

## 3. README claims vs audit outputs

### Takeaway
The README's "adds +7.0/+6.1 points over text-only" (0.7667 vs 0.6969; 0.7733 vs 0.7122) is not supported by the audit: on the retrained checkpoint, per-task accuracy sits at or slightly below the text-only floor, and the "gain" disappears under the blank control.

### Cited Findings
- README (`P/README.md` finding 4; `P/leak_analysis/README.md` finding 5, `compare_to_baseline.json`): epoch-3 val_loss 0.2853; 0.7667 single-scan, 0.7733 longitudinal; floors 0.6969 / 0.7122; "strict exact match over 150 generated samples per bucket" while floors are accuracy over all items in the bucket (leak_analysis/README.md caveats; inference_audit/README.md).
- Audit val closed results: T4 0.8115 (floor 0.8256), T5 0.6218 (0.6176), T6 0.8025 (0.8067). Pooled "longitudinal" = T5+T6 at val ≈ (0.6218+0.8025)/2 ≈ 0.712, i.e. ≈ the README's 0.7122 floor, not 0.7733. Pooled single-scan (T1-4) is dominated by T4 (0.81).
- The two pairs of numbers are not like-for-like: different metric (generation EM on 150 samples vs MCQ log-lik on all), different checkpoint (v6 epoch 3 vs v7 epoch 1), different floors (0.6969/0.7122 pooled vs per-task 0.826/0.618/0.807).

### Inferences
- A 150-sample EM estimate has ±0.07 CI, so +6-7 points over the floor was already statistically fragile before the audit [H]. The README line should be rewritten as "no evidence of gain over text-only; blank-volume control shows no image dependence" [H].
- Consistency of v7 with v6 numbers: only the val_loss (0.2986 vs 0.2853) is directly comparable and is close but not equal; accuracy numbers are not comparable [H].

### Gaps
- Pooled recomputation above is my arithmetic on equal weights; I did not reproduce 0.6969 / 0.7122 floors from the manifests.

## 4. Methodological risks

### Cited Findings and Inferences
- Single run, single seed (42), single epoch audited: no variance estimate [H]. Task 5/6 val has 238 items (T5/T6 same items), test 340 over 20 patients.
- Image pathway exercised but weak: frozen ImageNet ResNet-18 on 2D slices (1-ch conv reinit), 32 slices (16 T0 + 16 T1, each resampled), pooled to 16 visual tokens prepended as an embedding prefix; slice-attention is near-uniform (T1/T0 ≈ 1.00). Pooler attention differing on all items only shows the pathway is connected [H]. It is plausible that LoRA'd LLM learned to ignore a 16-token prefix that carries little discriminative signal for a 0.56%-trainable model with a frozen ImageNet encoder; the audit conclusion is therefore about this architecture, as RESULTS.md says [H].
- Task 6 leak makes its high score uninformative regardless of vision: model output depends on the in-question bracket (confusion identical real vs blank; never predicts class C) [H]. T5 (no bracket) is the real test: 0.6353 test vs floor 0.6471.
- Task 5 as specified supplies only one volume, ceiling 0.7392 without the prior (leak_analysis README), but the audit's "paired" T5 input does include the prior scan (pairing `paired_ok` 476/476 on val; in `_pair` T0 = previous letter scan, T1 current). So T5 here is a prior-scan-aided task and still shows no effect [H]. Caveat: T0 selection is ambiguous when several reconstructions exist (`test_423_b_1/_2` — whichever comes last), acknowledged in READMEs [M].
- Scoring: MCQ by length-normalised log-likelihood of the option text (not letter). For T4 "Yes"/"No" this is dominated by prior token probabilities; sumlogp vs lennorm agree on T5/T6 test to within 0.003 (per_task_test.csv). Strict exact match for open tasks collapses to ~0 on T3 (numeric/units strings) — F1 0.41 on T3 is for near-miss text, not correct measurement. Open-task floors (~0.01) are weak baselines [H].
- Train/test overlap: patient-level 80/10/10 split, "zero patient leakage verified" and 3,039 unique MD5 (no bitwise duplicates, no cross-split hash leakage) — but different reconstructions of the same study differ in bytes, so near-duplicate leakage is controlled only by patient grouping [H, per leak_analysis README]. Only 3D-RAD's official test images are available; the data is re-split from it, so numbers are not comparable to the paper's train/test protocol.
- Class imbalance: T4 82.6% No (train 81.7%); T5/T6 "No abnormality (always absent)" 64.3% of train; floors make this explicit. T4 + T6 = 78% of test; aggregate scores mostly measure prior-matching. Balanced accuracy/AUROC/F1 for T4 and macro-F1 for T5/T6 are not reported [H].
- Sample sizes: test T1/T3 n=93/84; the audit's open-task deltas of 0.004 are 1 item.
- Calibration of the "no dependence" claim: only a zero-volume perturbation; no shuffled/mismatched-volume or noise control; no check of gradients/saliency [M].
- Teacher-forced token accuracy misleads (see section 1).

### Gaps
- Whether the images were intensity-normalised so that zeros are in-distribution (not inspected in detail).
- No external check that the 3D-RAD answers for T4 are visually answerable at this resolution (256×256, 32 slices).

## 5. Concrete next experiments (ranked)
1. Re-run the audit on epoch 3 / `best` (adapters in `P/retrain/results/retrain_v7/`), plus per-item test CSV; patient-cluster bootstrap CIs on all deltas. Cheap, inference only.
2. Task 5 prior-scan ablations on the same checkpoint: (a) T0 zeroed, T1 real; (b) T1 zeroed, T0 real; (c) T0 replaced by another patient's scan (mismatched prior); (d) T0 = T1 duplicated (the existing fallback path); (e) shuffled slice order. Report accuracy and gold log-prob delta paired by item. Because T5 is un-leaked, this is the direct answer to "does it read the prior".
3. Replace zero-volume with mean-image, Gaussian noise, and a random other-patient volume (mismatch) for T4 and T5 to rule out a degenerate-zero artefact.
4. T5 vs T6 split analysis: report per-task confusion matrices and a paired test (McNemar) of T6-minus-T5 on the same items to quantify how much of T6's score is the bracket (expected ≈ the 0.75-0.81 floor).
5. Strengthen the text-only baseline: fine-tune the same Qwen+LoRA with no image (identical hyperparameters, 3 epochs) and compare with the VLM; this is the fair "text-only floor" and also a seed-variance proxy.
6. Seed variance: ≥3 seeds (even 1 epoch each, ~85 min on T4) for val loss and T5/T6/T4 accuracy.
7. Probe whether the encoder carries label information: linear probe on frozen pooled 16-token features (or slice features) for T4 findings, with balanced accuracy/AUROC; if the probe also fails, the bottleneck is the frozen ImageNet encoder/pooler rather than the LLM ignoring visuals. If it succeeds, train with unfrozen encoder, more visual tokens, or an auxiliary vision-only loss, then repeat the blank test.
8. Report balanced accuracy / macro-F1 for T4/T5/T6 and per-finding accuracy (18 findings) to expose majority-class prediction.
9. Train on a de-leaked T6 (bracket removed) or T5-only with the paired scan, which was the original thesis design (3D-RAD-T in `P/3D-RAD-T`), as a robustness check after the above.

Confidence summary: audit numbers and their internal consistency [H]; "model ignores image" for epoch-1 checkpoint [H]; same for epoch-3 [M, unmeasured]; v6/v7 equivalence [M]; README +7/+6 point claim unsupported [H]; "below floor" statements are noise-level [H].
