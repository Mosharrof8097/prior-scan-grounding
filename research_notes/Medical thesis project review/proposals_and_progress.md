# Medical thesis project: proposals, direction, and recent progress (local-file review, 2026-10-27)

Scope: local inspection only (docx XML, README/RESULTS files, git, timestamps, CV text). No web sources used, so "Source" below means local file path. Project files were not modified.

## 1. How the research question evolved (v1 -> v2) and what each proposal contains

### Takeaway
v1 (4 Oct) asked "does temporal recency bias cause longitudinal hallucination?" and proposed a detector plus mitigation. v2 (20 Oct) replaces it with "does the model read the prior scan at all?", driven by a benchmark audit that found label leakage in 3D-RAD Task 6 and a blank-image ablation showing the trained baseline ignores the CT volume. The pivot is well motivated by evidence on disk; the downstream detector/mitigation tail is carried over almost unchanged and is now the least-supported part. Confidence: high on the content summary, medium-high on the "why".

### Cited Findings
**v1: `/home/mosharrof/medical_thesis/Research_Proposal_Longitudinal_3D_CT_VQA.docx` (file mtime 2026-10-04, tracked in git)**
- Title: "Recency Bias and Hallucination in Longitudinal 3D CT VQA". One-page proposal.
- Hypothesis: models over-attend the current scan (T1) and under-use the prior (T0); recency-bias degree correlates with longitudinal hallucination rate, and a paired-scan training objective reduces it.
- Work done (as claimed): 3,039 volumes, 33,910 QA; patient-level 80/10/10 (27,197/3,254/3,459); five-algorithm duplicate audit (0.033% exact duplicates, image-hash check "being re-run"); baseline = 2D slice encoder + cross-attention aggregator + LoRA Qwen2.5-1.5B; attention-based T1/T0 recency ratio; training "in progress" on single Kaggle T4.
- Planned contributions: (a) hallucination taxonomy on a stratified sample, (b) detector combining recency signal and T0-only/T1-only/paired consistency, (c) mitigation (scan-order and counterfactual-swap augmentation, temporal contrastive loss).
- Evaluation: task accuracy vs RadFM, M3D, CT-CHAT and own baseline; T1/T0 attention ratio before/after; taxonomy rate, detector AUROC/F1 vs random and text-only.
- Timeline: 6 months (M1-2 baselines + recency analysis; M3-4 taxonomy/annotation/detector; M5-6 mitigation, ablations, "Q1-journal manuscript"). Asks for supervision, compute, stipend.

**v2: `/home/mosharrof/medical_thesis/Research_Proposal_v2_Prior_Scan_Grounding.docx` (mtime 2026-10-20, untracked)**
- Title: "Does the Model Read the Prior Scan? Visual grounding in multi-temporal 3D CT medical VQA". Longer (about 3 pages), dated 20 Oct 2026, lists a GitHub repo `github.com/Mosharrof8097/prior-scan-grounding`.
- States outright that the recency-bias explanation "was not supported by measurement" and the pivot is to benchmark-validity and grounding.
- Gap claimed: hallucination detection (RadFlag, ReXTrust, discrete semantic entropy) and mitigation (DPO prior-exam suppression, latent steering) are 2D-radiography/report-only; 3D-RAD has multi-timepoint structure but reports only aggregate accuracy and no text-only floor.
- Findings 1-4 (all reproducible on CPU per the proposal):
  1. Every Task 6 question embeds the past bit (`Past sequences of lung opacity: [1]...`); equals ground truth on all 16,031 rows; 75.4% of the task solvable with no scan.
  2. Tasks 5 and 6 are the same 16,031 items (943 volumes, 18 findings, 100.00% identical answers); only Task 6 states the past bit -> built-in matched control.
  3. Information ceilings: text-only Task 5 0.555; text-only Task 6 0.754; Task 5 + perfect read of the single supplied volume 0.739; Task 5 + prior scan 1.000.
  4. Blank-volume ablation on held-out test: Task 4 0.836 floor / 0.820 real / 0.816 blank; Task 5 0.647/0.635/0.635; Task 6 0.806/0.794/0.794. Model sits below text-only floor on all closed tasks; 35/2,816 closed predictions flip, 0 on Task 6; teacher-forced token accuracy (0.983) is uninformative.
- Narrowed claim: "this score does not evidence grounding", explicitly not "VLMs cannot reason over longitudinal CT".
- Next steps (6): diagnose blindness (visual-token variance across volumes); repair (unfreeze encoder, widen visual channel, train on un-leaked wording); radiologist human ceiling (~60 paired items); hallucination taxonomy with radiologist check; temporal-consistency detector; mitigation (order swap, counterfactual pairing, temporal contrastive).
- Evaluation table adds "Visual dependence" (blank-image delta) and "Answerability" (radiologist vs model) rows; baselines are now text-only floor and structural ceiling rather than RadFM/M3D/CT-CHAT.
- Timeline: no calendar; "diagnosis and repair a matter of weeks", annotation/detector gated on radiologist availability (named main scheduling risk). Limitations section is candid (only test-split images released, re-partitioned so other models' published numbers are not comparable; one lightweight baseline on one GPU; 32-slice downsampling; ambiguous T0 lookup).
- Supports requested: supervision, radiologist collaborator, compute beyond a single consumer GPU, assistantship.

**What changed v1 -> v2 (diff summary)**
| Aspect | v1 | v2 |
| :-- | :-- | :-- |
| Core question | Does recency bias cause hallucination? | Does the model read the prior scan / does the benchmark measure grounding? |
| Evidence | none yet (training in progress) | 4 completed findings incl. leakage, twin tasks, ceilings, blank ablation |
| Recency hypothesis | central | dropped; per-epoch recency T1/T0 ratio in `retrain/metrics/val_per_task_by_epoch.csv` is 1.002-1.011 (about 1, i.e. no bias detectable) |
| Comparators | RadFM, M3D, CT-CHAT | text-only floor, structural ceiling, radiologist |
| New items | none | human ceiling, repair step, de-leaked wording, explicit limitations |
| Kept | taxonomy, consistency detector, mitigation | same three, now framed as downstream of the grounding finding |
| Timeline | 6 months, Q1-journal | no dates; radiologist-gated |

Sources: both docx files above; `/home/mosharrof/medical_thesis/inference_audit/RESULTS.md`; `/home/mosharrof/medical_thesis/retrain/metrics/val_per_task_by_epoch.csv`; git log (`0515556` 2026-10-17 "Blank-image audit..." message).

### Inferences
- The "why" of the pivot is documented in the commit message of `0515556` and README ("Measuring it led somewhere more interesting"), consistent with the task brief. Recency ratios of about 1.00 (and blank-image insensitivity) mean the recency hypothesis could not be tested meaningfully: a model that ignores the image has no attention asymmetry to find. v2 does not state this logical point explicitly; worth stating in the thesis (confidence medium).
- v2 is a stronger thesis story than v1: a rigorous negative/validity result with CPU-reproducible numbers is much more feasible for a B.Sc. than a novel detector plus mitigation.

### Gaps
- No tracked record of the exact moment/reason for abandoning v1 beyond commit messages; no supervisor feedback on disk.

## 2. Consistency between proposal v2, README, and code/results

### Takeaway
Numbers in v2 match `inference_audit/RESULTS.md` exactly. The top-level README is stale relative to v2: its Finding 4 and Status section pre-date the blank-image audit and still say the +6.1 pt longitudinal gain is "pending" and may be "real visual reasoning". Confidence: high.

### Cited Findings
- v2 table values (0.836/0.820/0.816; 0.647/0.635/0.635; 0.806/0.794/0.794; n 2,358/340/340) match `/home/mosharrof/medical_thesis/inference_audit/RESULTS.md` test table. v2's attention/log-prob statistics (476/476, 2,738/2,816, 35/2,816, ~0.0055 nats) match the "not an artefact" table, which is labelled **val**, while v2 presents the table as test and describes the checks generally; minor labelling ambiguity.
- `/home/mosharrof/medical_thesis/README.md` (mtime 2026-10-08, pre-dates audit): Finding 4 cites v6 epoch 3 val_loss 0.2853 and +7.0/+6.1 pt over text-only floors (0.7667/0.7733 vs 0.6969/0.7122); README floors per task are val-split (Task 4 0.826, Task 5 0.618, Task 6 0.807). The README "Status" says per-task numbers are pending and v7 retrain is needed. The later audit (on v7 epoch_1, val_loss 0.3115) shows model below floor on test; the README never mentions the blank-image result.
- Checkpoint mismatch: the audit evaluates `retrain_v7/epoch_1`, not the epoch-3/best checkpoint, though the headline claim in v2 is about "the trained baseline". Epoch-by-epoch MCQ (n=100 subsample, val) in `val_per_task_by_epoch.csv` for Task 6 is 0.80, 0.76, 0.77 and for Task 5 0.62, 0.64, 0.58, i.e. no improvement with training, consistent with blindness, but blank ablation was not run on epoch 2/3 (confidence high that this is a gap).
- v2 says "no dataset files or model weights are tracked"; consistent with `.gitignore` and `git ls-files`.
- v2 mentions the CPU-reproducible `leak_analysis/` JSONs: present (text_only_leak_audit, task5_task6_twins, information_ceilings, verify_deleak, md5_volume_audit, build_3drad_t, compare_to_baseline, task6_task4_linkage).
- v2 says the pooler uses "frozen ImageNet ResNet-18, 32 slices, 16 learned queries" and README/retrain README confirm the architecture and "16 visual tokens".
- The README lists `retrain/` but its Layout does not mention `research_notes/` or `reports/` (both empty or just created 2026-10-27).
- `leak_analysis/README.md` lists `build_3drad_t.py` (synthetic de-leaked variants T-mask, T-strip); `retrain/README.md` says Task 5 is already the un-leaked twin, so a separate de-leaked run is unnecessary for the core experiment. v2 section 5 "train on the un-leaked wording" is consistent with that.

### Inferences
- Task 5 as supplied has one volume, so a model trained on it cannot exceed 73.9% by the proposal's own ceiling; the "prior-scan" input exists only inside Task 6/baseline stacking. See section 4.
- README should be updated to include the blank-image headline before sharing the repo (the v2 proposal links to the repo as evidence).

### Gaps
- Did not re-run any scripts or recompute the numbers (read-only, no dataset check); consistency was checked doc-to-doc, not by recomputation.
- `per_item_test.csv` and `per_task_val.csv` were not retrieved from the Kaggle run (per RESULTS.md), so row-level test predictions for the taxonomy are missing.

## 3. What is new since the last commit (0515556, 2026-10-17 01:19 +0600)

### Takeaway
Git sees only two untracked files: the v2 proposal and the CV. Everything else in the "new" list is either already tracked or is ignored and old. Nothing important is lost, but the v2 proposal and CV are uncommitted, and 1.2 GB of Kaggle outputs exist only locally. Confidence: high.

### Cited Findings
- `git status --short`: untracked only `Md Mosharrof Hossain — CV.pdf` (2026-10-21, 88 KB, 3 pages) and `Research_Proposal_v2_Prior_Scan_Grounding.docx` (2026-10-20, 16.6 KB). No modified tracked files (git diff empty). Commits: `cc4615c` 2026-10-08 initial; `0515556` 2026-10-17 blank-image audit.
- `perper ref/` (163 MB, 20 reference PDFs incl. `literature_review/` subfolder): top directory mtime 2026-10-24 16:40, but every PDF is dated 2026-09-24 (so the directory mtime is just a folder change, probably a file move/hardlink: some files show link count 3, e.g. `s41467-025-62385-7.pdf` appears in both the root and `literature_review/`). It is gitignored (publisher licences).
- `preview_images/`: directory mtime 2026-10-24; PNGs are 2026-09-23 and were committed in `cc4615c`; the two `.npy` files (8 MB each, from 2025) are gitignored. Not new content.
- `retrain/`: contents dated 2026-10-17 or earlier. `retrain/metrics/` (28 KB) and scripts/README tracked (cc4615c / 0515556). `retrain/results/` (357 MB: epoch_1..3, best, last, LoRA adapters, trainer state; plus `split_*_qa.csv`) and `retrain/audit_upload/` (78 MB) are gitignored; `retrain/results.zip` (759 MB) is ignored via `*.zip`.
- `find -newermt 2026-10-17` (excluding thesis_env, .git, .cache, 3D-RAD_Dataset): `.gitignore` (10-17), the audit result files and retrain metrics (10-17, committed in 0515556 except the ignored split CSVs), v2 proposal (10-20), CV (10-21). `reports/` and `research_notes/` (directories created 2026-10-27) are empty/new.
- `.gitignore` sanity: sensible (data, env, weights, 3D-RAD-T regenerable, bulk split CSVs, Kaggle outputs, paywalled PDFs). Issues: (a) `*.zip` would also silently ignore any future zip deliverables; (b) `inference_audit/interface audit result/` folder name has a typo/spaces ("interface" for "inference") but is tracked fine; the ignored `split_*_qa.csv` copies there total about 10 MB, regenerable; (c) `retrain/results/` contains the only trained LoRA adapters (v7), already burned once (v6 adapters lost per `retrain/README.md`): no backup outside the local disk; (d) `thesis_env/` ignored, but no `requirements.txt`/env file tracked (not seen in `git ls-files`) so the environment is not reproducible from the repo (confidence medium: I only checked tracked file list).
- Repo has no remote configured as far as inspected (not checked with `git remote`); v2 cites a GitHub URL as the public location of code. Confidence on whether the repo is actually pushed: low.

### CV (presentation of the thesis only)
- `Md Mosharrof Hossain — CV.pdf` lists research interests "Trustworthy medical VLMs, visual grounding and shortcut learning, evaluation of 3D medical imaging AI, hallucination detection"; a "Research Experience" entry "Independent Research Project: Visual Grounding in Multi-Temporal 3D CT VQA, 2026-present" restates the 75.4% leakage, the leak-free split, the ResNet-18/cross-attention/LoRA-1.5B baseline on a 16 GB GPU and the blank-volume ablation (0.794 unchanged) plus "derived upper bounds". Author is a 3rd-year B.Sc. CSE student (expected 2028), with prior small applied-ML papers and a research-assistant role at a lab; no thesis supervisor named.
- Inference: CV and v2 proposal appear designed together (same framing, same numbers), probably for research-position/assistantship applications; the project is presented as a grounding/benchmark-validity study, not the original recency-bias study. Confidence medium-high.

### Inferences
- The "perper ref" and "preview_images" are not new content; only their folder mtimes moved on 2026-10-24.
- Highest-risk uncommitted/unbacked items: v2 docx and CV (untracked), and `retrain/results/` adapters (ignored, local-only).

### Gaps
- Did not check `git remote`/push state or whether the claimed GitHub repo exists.

## 4. Gaps between what v2 promises and what exists; feasibility and scoping

### Takeaway
Findings 1-4 are complete, reproducible and thesis-worthy on their own. Steps 1-2 (diagnose and repair) are plausible within weeks; steps 3-6 (radiologist, taxonomy, detector, mitigation) have no code, data, or collaborator yet, and the repair step has a structural data problem. For a B.Sc. thesis the plan is over-scoped as written but has a coherent, defensible core. Confidence: medium-high (feasibility judgments are my inference).

### Cited Findings (what exists vs promised)
- Exists: split/audit scripts (`dataset preparetion/`), training script v6/v7 (`baseline_training_pipeline/`, `retrain/kaggle_retrain_v7.py`, 893 lines, resumable, per-task validation, recency ratio), audit script (`retrain/kaggle_inference_audit.py`, 939 lines), CPU leakage analyses (`leak_analysis/*.py` with JSON outputs), `build_3drad_t.py` synthetic de-leaked task, literature review docx (IEEE style, `literature review /`), results for epoch-1 audit.
- Not found (searched file list and READMEs): no code for visual-token variance/collapse diagnosis; no encoder-unfreezing or widened-visual-channel training config; no hallucination taxonomy/annotation files; no consistency detector (prior-only/current-only/paired inference); no mitigation (order swap, counterfactual pairing, contrastive loss); no radiologist protocol/item sample; no blank-image ablation on epoch 2/3; no re-run of comparison models (RadFM, M3D, CT-CHAT) on the re-partitioned split.
- Prior-scan input pipeline: partially exists. The baseline stacks the referenced scan and nearest earlier study into one volume for temporal questions (v2 section 3.2; `retrain/README.md` caveat: "T0 reconstruction is ambiguous... keeps whichever comes last"; v2: "all 5,746 temporal items find a genuine earlier scan"). Hence prior volumes do exist in the released test subset for the Task 5/6 items, but v2's ceiling table says Task 5 itself supplies a single volume; a clean "prior-only vs current-only vs paired" inference mode is not implemented as a separate evaluation path (confidence medium; I did not read the 900-line scripts in full).
- Data constraint (v2 limitations; README): only `Images/test.zip` is public, so the 136,195 train-split QA rows have no volumes; whole study uses the test set re-split at patient level (3,039 volumes, 33,910 QA). Temporal items: Tasks 5 and 6 are only 16,031 rows over 943 volumes, and the held-out partition has just 340 items per temporal task (238 val). Statistically the temporal evaluation is small (a 95% CI on 340 items is roughly +/-4 points; my calculation).
- Compute: ~3 h/epoch on a T4 for 3 epochs (about 9-10 h, near the Kaggle session ceiling) per `retrain/README.md`; v2 itself names a single consumer GPU as the binding constraint on repair/mitigation.

### Inferences (coherence, feasibility, scope)
- Coherence: strong for sections 1-4 (leakage -> twin control -> ceilings -> blank ablation). Weaker in 5-6: the "temporal consistency detector" only has content on un-leaked wording and only if the model actually uses images; with a blind model all three inference modes agree trivially, so the detector depends on the repair step succeeding first. v2 acknowledges the dependency partially.
- Risk of the repair step: the baseline is blind possibly because the frozen ImageNet ResNet-18 on 32 downsampled slices, 16 tokens and 3 epochs gives the LLM no usable signal while text priors are an easy shortcut; unfreezing and widening may still fail on one T4, and a failure should be written up as a valid result (diagnosis of collapse). Task 4 floor of 0.826 (69% of training items, binary) also makes accuracy gains hard to detect; repair success should be judged on Tasks 5/6 with blank deltas and McNemar-style tests.
- Ceiling logic: Task 5 ceiling of 0.739 for a perfect reader of the single volume is a derived bound under assumptions (label structure), correct by construction if the four answers are products of two bits; its practical relevance depends on 32-slice resolution (v2 states this).
- Human ceiling: requires a radiologist; v2 flags it. A fallback is needed (e.g. a clinician-collaborator via the lab, or dropping the taxonomy). The B.Sc. thesis should not depend on it.
- Scope recommendation (my judgement): a B.Sc. thesis centred on (i) benchmark audit + ceilings + twin control, (ii) blank-image grounding audit, applied to a baseline plus one repaired variant, with (iii) the radiologist/taxonomy/detector/mitigation listed as stretch/future work, is coherent and feasible. Promising all six steps plus a manuscript is over-scoped given 3rd-year status, single T4 and no named supervisor. v2 dropped v1's "Q1-journal" and 6-month calendar, which makes the timeline vague; a milestone table with go/no-go gates would help.
- Novelty/claims risk: "Every published number on this benchmark is reported without a text-only floor" and "Nobody has asked whether ..." are strong universal claims; they were not verified against the literature here (the lit review exists but I did not audit it). Claims about RadFlag/ReXTrust/etc. being 2D-only likewise unchecked. Also the benchmark test set used here was re-partitioned, so audit findings about the leak are on the public test subset; the leak is in question text so it likely generalises, but only test-split Task 6 rows were audited (16,031 rows matches Task 6's full released count per README).

### Gaps
- Did not read `Hallucination_3D_MedVQA_Literature_Review_IEEE.docx` or the PDFs; no check on related-work completeness.
- Did not verify the "prior volumes available in 3D-RAD" beyond the proposal's and README's statements (no access to dataset audit by me; the dataset dir `3D-RAD_Dataset/` was not opened).
- No information on supervisor, thesis deadline, or university requirements, so "feasible for a B.Sc." is judged generically.
- Whether the project is public on GitHub is unverified.

## Suggested actions (for report writer, all low effort)
1. Commit v2 proposal (and decide whether the CV belongs in the repo; it contains personal contact details).
2. Update README Finding 4 and Status to include the blank-image result and point to `inference_audit/RESULTS.md`.
3. Back up `retrain/results/` adapters off-disk (ignored by git, 357 MB; v6 adapters were already lost once).
4. Run the blank-image ablation on epoch 2/3 (cheap, inference only) and pull `per_item_test.csv`.
5. Add a milestone table with go/no-go gates and a radiologist fallback to the proposal.
