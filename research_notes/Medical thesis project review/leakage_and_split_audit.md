# Audit of dataset, split and label-leakage claims in /home/mosharrof/medical_thesis

Sources are all local files (paths absolute under /home/mosharrof/medical_thesis). "Re-run" means I executed copies of the scripts from a scratchpad (project files untouched; outputs not written into the project). `scikit-learn` is NOT installed in thesis_env or system python, so `text_only_leak_audit.py` and `verify_deleak.py` could not be re-run; I replicated their majority/bracket parts in pandas (scratch scripts `/tmp/claude-1000/-home-mosharrof/c1240d51-14b9-41e3-af10-7b19f253aa9e/scratchpad/{a,b}.py`). Dataset is present locally (3,039 .npy under 3D-RAD_Dataset/Images/test; QA csvs for test and train dirs).

Git state: two commits: cc4615c "Initial commit" (2026-10-08; contains all of leak_analysis/, dataset preparetion/, split_manifests/summary only) and 0515556 (2026-10-17, blank-image audit). Untracked/new: `Research_Proposal_v2_Prior_Scan_Grounding.docx` (2026-10-20), CV pdf, `research_notes/`. 3D-RAD-T/ (CSV outputs) is not tracked (not in `git ls-files`), nor are split_manifests/*.csv. All leak_analysis files are dated 2026-10-08 (15:45-16:48) - i.e. committed, not new.

---

## Q1. Do the JSON outputs support each README number?

### Takeaway
Every headline number is reproduced by the code and JSON exactly (so the arithmetic is honest), but all of the ceiling/floor numbers (0.5552, 0.7539, 0.7392) are computed on all 16,031 Task 5/6 rows pooled across the official train and test directories, 82% of which have no CT volume and are not in the 80/10/10 split. README says "Numbers are on a patient-level 80/10/10 split" - that is false for the ceilings table. Confidence high (re-run + independent pandas recomputation).

### Cited Findings
- (1) bracket == past bit on all 16,031 rows: `leak_analysis/information_ceilings.json` has `"n_items": 16031`, `"bracket_equals_past_bit": true`; script asserts `(t6["bracket_past"] == t6["past"]).all()`. Re-run passed. The crosstab in `task5_task6_twins.py` output is perfectly block-structured: past=False -> only New (997) or No Abnormality (8901); past=True -> only Refractory (2949) or Resolved (3184); zero off-block cells. `text_only_leak_audit.json` Task 6: `"answers_reachable_per_bracket_group": {"False": 2, "True": 2}` supports "eliminates 2 of 4". Supported.
- 75.4%: `information_ceilings.json` `"task6_text_floor_bracket": 0.7538519119206537`; `task5_task6_twins.json` `"bracket_floor": 0.7538519119206537`. Supported as a pooled number. The README wording "75.4% solvable without reading a voxel" is really "accuracy of the best bracket-conditioned majority rule", not "solvable" (the remaining bit still needs a guess).
- (2) 943 volumes / 18 findings / 100% identical: `task5_task6_twins.json`: `"n_matched_pairs": 16031, "answer_agreement_rate": 1.0`. Re-run: Task 5 and 6 both "16,031 rows | 943 volumes | 16,031 findings matched | 4 distinct answers"; 16,031/16,031 identical. Supported. Caveat: join key is (NpyName, finding) with `drop_duplicates(key)`; the finding is extracted by substring search of the 18 Task-4 file names in the question (longest first) - 16,031/16,031 matched, so no evident failure, but it was not checked that 18 findings all appear (README says 18; linkage JSON `"n_findings": 18` derives from Task 4 file names, not Task 5/6). Also "identical answers" is near-tautological: Task 5 and 6 are the same underlying label rows reworded (13 file letters b..q = scan index). It shows they are twins, which is a genuinely useful observation, but 100% is expected for a templated generator, not a measured agreement between independent annotations.
- (3) Ceilings table, `information_ceilings.json`: `"task5_text_floor": 0.5552367288378766`, `"task6_text_floor_bracket": 0.7538519119206537`, `"task5_ceiling_single_volume": 0.7391928139230242`, `"task5_ceiling_with_prior_scan": 1.0` (hard-coded 1.0 in script, `rows` list; not computed). All four match README digit for digit. BUT:
  - The 16,031 rows = 2,873 rows in `3D-RAD_Dataset/test/Task5*` + 13,158 rows in `train/Task5*` (my count; same for Task 6). Local volumes exist only for the 2,873 test-dir rows (169 volumes). The 13,158 train-dir rows (774 volumes) have no CT. `information_ceilings.py` loads with `glob("3D-RAD_Dataset/*/Task5*/*.csv")` and never filters on local volumes or the split.
  - On the usable subset (test dir = the only rows in split_manifests): Task 5 majority floor 0.6411, single-volume perfect-read ceiling 0.7894, Task 6 bracket floor 0.7894 (my recomputation). Train-dir: 0.5365 / 0.7282 / 0.7486. So 0.5552/0.7392/0.7539 are blends of two different label distributions (P(past) = 0.285 in test dir vs 0.404 in train dir; P(No Abnormality) 0.641 vs 0.536).
- (4) 80/10/10 seed-42 patient split, zero leakage: `split_manifests/split_summary.json`: train 27,197 QA / 2,181 vol / 1,043 patients; val 3,254 / 255 / 130; test 3,459 / 260 / 131; `"patient_leakage": "0 overlap verified across all three split pairs"`. Matches `dataset preparetion/.../duplicate_audit_report.json` (same counts; algo2 all zeros). Supported (patient overlap is asserted in code, `save_split_manifests.py`).
- (4b) MD5 audit: `leak_analysis/md5_volume_audit.json`: `"total_hashed_volumes": 3039, "unique_md5": 3039, "bitwise_identical_scan_clusters": 0`, cross_split all 0; `md5_audit.log` shows completed run to 3,039/3,039. Supported. Note it hashes all 3,039 files, but only 2,696 volumes are referenced by QA (my count; README caveat also says so).
- Other README numbers: compare_to_baseline.json: single n=2778 text_only 0.6969 vlm 0.7667 gain 0.0698; longitudinal n=476 text_only 0.7122 vlm 0.7733 gain 0.0611; per-task Task4 0.8256, Task5 0.6176, Task6 0.8067, Tasks 1-3 0.0114/0.0124/0.0. All match README ("+7.0 / +6.1", 0.826, 0.807, 0.618, "~0.01").
- Three different "Task 6 leaks X%" figures coexist: 0.7539 (pooled, ceilings), 0.8067 (val, compare_to_baseline), 0.8062 (80/20 local, text_only_leak_audit.json `bracket_only_acc`), 0.7958 (verify_deleak.json), 0.7894 (test-dir, my recomputation). `build_3drad_t.py` docstring says "80.7%". README item 1 quotes 75.4%, README item 4 quotes 0.807 for the same thing.
- Task 5 text floor likewise: 0.5552 (pooled), 0.6176 (val), 0.6349 (text_only_leak_audit), 0.6540 (verify_deleak), 0.6411 (test-dir all rows).

### Inferences
- The ceilings are internally correct arithmetic on a mismatched population. For the thesis claim about the evaluated population the right numbers are the val/test ones (Task 5 floor ~0.62-0.64, single-volume ceiling ~0.79). The "perfect reader cannot exceed 73.9%" number is therefore too low for the data the model actually sees (0.79). Direction of the error: README understates the achievable ceiling by about 5 points.
- Different scripts split differently (see Q2), so the numbers across files are not comparable even though all cite "seed 42".

### Gaps
- `text_only_leak_audit.py`, `verify_deleak.py` TF-IDF numbers not re-run (no sklearn); I verified their JSON only by reading code and by independently recomputing majority and bracket rules.

---

## Q2. Are the scripts methodologically sound? Bugs, circularity, over-claims

### Takeaway
The core leak finding (bracket = past bit; Tasks 5/6 twins) is solid. The "information ceiling" framing is partly tautological and partly over-claimed, the TF-IDF-based "de-leak verification" has no statistical power because of a tokenisation bug, and different scripts use different, non-manifest splits. Confidence high on the bugs listed (each reproduced or read directly in code).

### Cited Findings
- **Ceilings are definitional.** The 4 answers are a deterministic function of (past, now) (`BITS` dict in information_ceilings.py; zero off-block cells). "Task 5 + prior scan = 1.0" and "Task 6 + perfect current CT = 1.0" are hard-coded 1.0 and follow from label definition, not a measurement. They also assume a perfect, noise-free reader and that labels are exactly image-derivable (labels originate from CT-RATE report-derived annotations, not from the images; label noise is ignored).
- **"+ the prior scan" is mis-stated.** The bracket length equals the number of earlier scans (letters: b=1 prior ... up to 16; bracket-length distribution in Task 6: 1:1989, 2:7939, 3:3145, 4:1377, 5:782 ...), and `past = any(bracket)` is "present at ANY earlier time". Perfect read of the current scan plus ONE prior scan gives a lower ceiling: now + last-prior = 0.8679 pooled / 0.9558 on test dir; now + first-prior = 0.8659 / 0.9558 (my recomputation). 1.0 requires ALL prior scans (mean ~1.5, max 7 priors needed for the 169 test volumes; all of those priors are on disk: for all 169 Task 5/6 test volumes every earlier lettered scan of the same patient is present locally). Only the 1,989 bracket-length-1 rows (12%) are fully resolved by a single prior scan.
- **"A flawless reader cannot exceed 73.9% on a single volume" is not a hard bound.** `best_given(t5, ["now"])` conditions on the current bit only, so it is the ceiling for "image gives the present bit, past bit guessed from the population prior P(past|now)". A single CT can carry evidence about chronicity/the past (e.g. calcification, fibrotic sequela, emphysema are persistent by nature), and the finding identity changes P(past|now): conditioning on finding gives 0.7912 vs 0.7894 on the test dir (small, but nonzero). Docstring claim "hard, assumption-free bounds" is therefore overstated; it is a bound conditional on stated assumptions.
- **Text floor is not the best text-only classifier.** `task5_text_floor` is plain majority. On train-dir->test-dir, finding-conditioned majority gives 0.6377 vs majority 0.6411 (no gain) so the floor is practically fine here; but the README calls them "text-only" and the only learned text-only model (TF-IDF+LR) scores below majority, see next point.
- **TF-IDF bug (power loss).** `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)` with the default token pattern `(?u)\b\w\w+\b` drops single-character tokens: I confirmed `re.findall` on "Past: [1, 0]. 0 indicates" returns only ['Past','indicates']. So the 0/1 bracket digits are invisible to the TF-IDF model. Evidence in outputs: `verify_deleak.json` `"Task 6 original"` tfidf_lr 0.6332 equals Task 5 control tfidf_lr 0.6332 exactly and is below majority (0.6540); `text_only_leak_audit.json` Task 6 `text_only_acc` 0.602 < majority 0.635 while `bracket_only_acc` is 0.806. So the learned text-only model did not find the leak that exists, and the "T-mask/T-strip tfidf_lr 0.6315/0.6280" values prove nothing about de-leaking.
- **De-leak verification is circular.** In `verify_deleak.py` `bracket_only` is `None` whenever <99% of train rows contain a bracket, then `best_text_only = max(majority, bracket_only, tfidf)` over non-None entries. For T-mask/T-strip the bracket rule is switched off by construction (the brackets are gone / `[MASKED]`), so "best_text_only == majority" is guaranteed regardless of residual leakage; the only independent test is TF-IDF, which has the digit-blindness bug (and was blind to the leak even before de-leaking). A real check would use digit-aware or char-level features, or exact-question lookup. My exact-question lookup (fit on train-dir, test on test-dir): original 0.6411, T-mask 0.6342, T-strip 0.6136 (vs majority 0.6411) - i.e. no residual lexical leak found by that cheap probe, which is weak but suggestive evidence that masking works. T-strip leaves just 289 unique questions (all test questions appear in train), so by construction it carries only finding/phrasing.
- **Splits differ between scripts though all cite "seed 42".** `text_only_leak_audit.py` / `verify_deleak.py` do an 80/20 per-script split with `df["PatientID"].unique()` of that script's own row set (Task 5 only for the control, Task 6 only, etc.), so the patient array and shuffle differ per script; `compare_to_baseline.py` uses the 80/10/10 over all tasks (matches manifests: val 3,254, 476 longitudinal + 2,778 single). Hence Task 5 control majority is 0.654 in verify_deleak but 0.635 in text_only_leak_audit (both n_test=578). The README's "patient-level 80/10/10 split (seed 42)" does not apply to the audits or the ceilings.
- **md5_volume_audit.py uses a different split than the manifests.** It builds `pats = sorted({...})` for all 3,039 files, while the manifest/training code uses `df["PatientID"].unique()` in QA order; both have 1,304 patients but only 65.1% of patients get the same split label (my recomputation). Practically moot because there are zero duplicates (no cluster can straddle any split), but the "cross_split_hash_leakage: 0" is evaluated on a split that was not the one used for training.
- **Linkage script cannot test its hypothesis.** `task6_task4_linkage.json`: `"n_joined": 0, "exact_reconstruction_rate": NaN, "current_bit_balance": {}` (NaN is not valid strict JSON). The join is on (NpyName, finding) and Task 4 contains only scan letter 'a' (all 123,642 Task 4 rows) while Task 5/6 contain only letters b..q (my count). So the script's stated goal ("Task 4's accuracy is a lower bound on Task 6 answerability") is not achievable with this join, and the printed line "majority-guessing that bit caps text-only at 0.000" is a degenerate message. The README table claim "Task 4 covers first timepoints only, disjoint from Task 5/6" is right, but only my independent check supports it, not the JSON.
- **Extra finding (not in project):** joined at patient level instead (patient + finding), Task 4's scan-'a' answer equals the FIRST element of the Task 6 bracket on 2,873/2,873 test-dir rows (0/0: 2,181; 1/1: 692; no mismatches). So Task 4 labels and the Task 6 sequences come from the same per-patient timeline; within one patient split this is not cross-split leakage, but it means Task 4 training rows for the first scan encode part of the bracket/past bit for the same patient.

### Inferences
- The strongest, defensible claim is: the Task 6 bracket exposes the past bit (verified, 100%); consequently Task 6 is a weaker test than Task 5 of image use. The ceiling table should be restated on the evaluated rows, and the "1.0 with the prior scan" row should say "with all prior scans".
- The T-mask claim "removes the shortcut" is plausible but the committed verification does not establish it (digit-blind TF-IDF + switched-off rule).

### Gaps
- Could not run sklearn-based scripts; the exact TF-IDF digit-blindness effect on accuracy is inferred from identical outputs (0.6332 on both datasets) rather than from a rerun.

---

## Q3. 3D-RAD-T, compare_to_baseline, task6_task4_linkage

### Takeaway
3D-RAD-T is a purely textual derivative of the released CSVs (no new images, no new labels) that rewrites only Task 6 questions; it is built correctly but is, by the project's own argument, equivalent to Task 5 and has not yet produced any model result. compare_to_baseline's "+7.0/+6.1" is a comparison of unlike quantities with tiny, clustered samples. Confidence high.

### Cited Findings
- `leak_analysis/build_3drad_t.json`: T-mask and T-strip each `"files": 92, "rows": 170105, "task6_questions_rewritten": 16031`. 170,105 = 33,910 test-dir + 136,195 train-dir QA rows (my arithmetic: sum of train-dir counts 2,081+6,055+1,573+100,170+13,158+13,158 = 136,195), so 3D-RAD-T rewrites Task 6 in BOTH the test and train directories (13,158 train rows for which no images exist). Output at `3D-RAD-T/T-mask/{test,train}/Task*/` (dirs verified, letter csvs b..q), untracked in git.
- T-mask: `BRACKET_RE.sub("[MASKED]", q)`; T-strip: drops everything through the "(Note: ...)" gloss and keeps the last sentence (289 distinct questions left in my read of the output). Answers/VolumeName copied verbatim. The mask keeps the "(Note: 0 indicates absence...)" sentence, so the prompt now refers to a masked sequence.
- Produced results: only `verify_deleak.json`: Task 6 original best text-only 0.7958 vs T-mask 0.6540 and T-strip 0.6540, equal to the Task 5 control 0.6540. No VLM has been evaluated on 3D-RAD-T yet (retrain/inference_audit not part of this scope).
- compare_to_baseline.py: text-only "floor" is per-(task, bracket-group) majority from the training split, applied to val rows; VLM number 0.7667/0.7733 are hard-coded constants (`REPORTED`) from a lost Kaggle session, and per its own README caveat are exact-match "over 150 generated samples per bucket" while floors are over all 2,778 / 476 val rows. The v6 LoRA adapters are lost (root README "Status"). At n=150, p~0.77 the binomial 95% CI is about +/-6.7 points, the same size as the claimed +6.1/+7.0 gains. Also the longitudinal bucket pools Tasks 5 and 6 (238 rows each) which are the same 238 items twice, and the 238 val rows come from only 14 patients/volumes (val), 20 patients/volumes (test) (my count from split_manifests), so effective sample size for temporal items is ~14 clusters, not 238.
- Interpretation in `compare_to_baseline.py` bucket "longitudinal" regex `longitudinal|temporal` also matches Task 5 "Static_Temporal", consistent with training code, ok.

### Inferences
- 3D-RAD-T adds no information beyond Task 5 (Task 5 is already the un-leaked twin, the project says so in leak_analysis/README); its value is only as a pipeline-compatible prompt variant (same Task 6 phrasing, bracket removed) enabling a clean A/B inside the existing loader. The risk is that the thesis presents it as a new benchmark; it is a wording ablation on one task.
- Because a de-leaked Task 6 still provides one volume, its single-volume ceiling is the same ~0.79 (test) / 0.74 (pooled) as Task 5, so it is not a "longitudinal" task unless prior scans are fed in.
- The +6/+7 gains are not statistically distinguishable from zero with the described evaluation protocol (moderate confidence; the exact protocol is not documented beyond the README sentence).

### Gaps
- Original baseline per-task accuracy, VLM sample selection and seed for the 150 samples not available (checkpoint lost).

---

## Q4. Split construction and MD5/duplicate audit

### Takeaway
Patient-level grouping is correctly implemented and the zero-patient-overlap assertion is genuine. The "zero QA leakage" is true across splits, but there are 9 exact duplicate (VolumeName, Question) pairs inside train. The split is a re-split of the official test set only, not stratified (despite README claiming it), and the MD5 audit, while now complete, only rules out byte-identical files. Confidence high.

### Cited Findings
- `dataset preparetion/dataset split script/split_dataset_kaggle.py`: PatientID = first two underscore tokens of the volume name; `RandomState(42).shuffle(unique_patients)`; train=int(n*.8) -> 1,043 train / 130 val / 131 test of 1,304 patients; reads only `**/test/*/*.csv` (hence 33,910 rows). Verified consistent with split_manifests/split_summary.json and with `save_split_manifests.py` (same logic). Its README claims "Task Stratification ... balances the proportions of all 6 clinical tasks" and header says "Patient-Level Stratified Split"; the code contains no stratification (a pure shuffle). Actual per-split Task 5/6 rows: train 2,295, val 238, test 340 (manifest), volumes 135 / 14 / 20.
- Patient IDs are all prefixed `test_` (33,910/33,910 rows), i.e. all material comes from the official test partition.
- `duplicate_audit_report.json`: train 9 exact duplicate (VolumeName, Question) records, 0 label conflicts (`duplicate_rate_pct` 0.033), val/test 0; algo2 patient/volume/VQA leakage all 0; algo4 all-zero with `"total_hashed_volumes": 0` (interrupted run, flagged in the docx §3.2 "should not be cited as a passed integrity check"). The docx and .odt text agree (both contain the same §3.2). `leak_analysis/md5_volume_audit.json` supersedes it with a completed run (3,039 unique of 3,039). README item 5 ("zero patient, volume or QA-pair leakage") is accurate for cross-split leakage; it omits the 9 intra-train duplicates (harmless to evaluation).
- Patient-level split is the correct unit, and since the longitudinal letters a..q of one patient share the patient ID, the prior/current scans of one patient never straddle splits (verified for all T5/T6 test volumes: priors are on disk and in the same patient group).
- MD5 limits (also stated in leak_analysis/README caveat): different reconstructions (`_1`, `_2`) or re-sampled versions of the same study differ in bytes; "unique MD5" does not imply visually distinct. Patient-level grouping, not hashing, is the real control (agreed).
- Docx §1.3 reports "Total QA records 33,910; QA records with verified image match 33,910 (100%)" - true only because the 136,195 train-dir rows were excluded before matching; the docx does not mention the exclusion. The 136,195 figure appears in README.md, leak_analysis/README.md, inference_audit/README.md and retrain/README.md (I verified it: sum of train-dir rows = 136,195).

### Inferences
- The "zero leakage" statement is fully defensible for patients/volumes/QA pairs inside the 33,910-row universe. It should not be read as certifying anything about the original 3D-RAD train/test boundary: the training set here is a subset of the official TEST set, so results are not comparable to published 3D-RAD baselines (moderate confidence about CT-RATE-level overlap with the official train set: unknown, because the 136,195 train volumes are unreleased; patient IDs `test_*` suggest disjointness from `train_*` but that was not verified).
- Row-level independence is weak: 33,910 QA rows share 2,696 volumes; Task 4 alone has 23,472 rows (69%). Any aggregate accuracy is dominated by Task 4 yes/no (82.6% majority).

### Gaps
- Whether the QA-referenced vs 343 unreferenced volumes (3,039 - 2,696) are used anywhere: not investigated (they have no questions).

---

## Q5. Limitations and how serious they are

### Takeaway
The biggest threat to the thesis claims is sample size and population mismatch, not the leakage logic: only 169 Task 5/6 volumes (14 val, 20 test, 135 train) have images; every number about "943 volumes / 16,031 items" describes data the model never sees. Confidence high.

### Cited Findings
- Local images exist only for the official test directory: Task 5/6: 2,873 of 16,031 rows (17.9%), 169 of 943 volumes (17.9%); Task 4: 23,472 of 123,642 rows, 1,304 of 6,869 volumes; Tasks 1-3 similarly ~1/3 of rows. Train-dir Task 5/6 volumes (774) do not overlap the test-dir ones (0 shared).
- The held-out temporal evaluation is 14 volumes (val, 238 items per task) and 20 volumes (test, 340 items); items per volume ~17 correlated (each volume x up to 18 findings; each patient has exactly one Task 5/6 volume, 169 patients in 169 volumes). Item-level CIs (e.g. +/-4 points at n=340 noted in research_notes/.../proposals_and_progress.md line 111) understate uncertainty; cluster-level (patient-level) CIs would be about +/-3-4x wider in relative terms (my estimate, not computed).
- Class distribution shift: test dir vs train dir (see Q1), so train-set priors miscalibrate test floors; even val vs test differ in count (238 vs 340).
- Label provenance: answers are deterministic functions of report-derived presence bits; "perfect reader" ceilings of 1.0 ignore that label noise bounds real achievable accuracy. 3D-RAD itself offers no human-validated temporal labels (the project's argument that "no radiologist is required" rests on the benchmark authors' assertion; reasonable but unverified).
- Text-only leakage in Tasks 1-3 is small (majority 0-2%; the audit's TF-IDF 0.13/0.11/0.01 in text_only_leak_audit.json, so Task 1 text-only 13% > majority 1%: a non-trivial text prior for Task 1/2 that is not mentioned in the README headline).
- Task 4 (72% of the held-out QA: 2,340/3,254 val) is 82.6% solvable by "No" (`compare_to_baseline.json` 0.8256; `text_only_leak_audit.json` 0.8306 majority vs 0.8308 TF-IDF) - correctly flagged.

### Inferences
- Seriousness ranking (my judgement): (1) very small, clustered temporal test set - high; (2) pooled-population ceilings quoted for unseen data - medium-high, easily fixed by recomputing on test-dir rows; (3) unsupported "+7/+6 points" claim from a lost checkpoint with n=150 - high for any claim built on it; (4) tfidf/verification flaws - medium, fixable; (5) test-only re-split / non-comparability with benchmark - medium, only matters if comparing to published numbers.
- None of these invalidate finding (1)-(2) of the README (leak exists, twins exist), which are structural and verified at 100% on every row.

### Gaps
- No statistical test was done by the project; my +/-6.7 CI for n=150 is a normal-approximation estimate I computed by hand.

---

## Summary table of README claims (verdict / confidence)
| README claim | JSON value | Verdict |
| :-- | :-- | :-- |
| Bracket = past bit, 16,031 rows | `bracket_equals_past_bit: true` | Correct on pooled train+test rows; holds on test-dir rows too (high) |
| 75.4% text-only solvable on Task 6 | 0.7539 | Arithmetic correct but pooled; test-dir is 0.7894, val 0.8067 (high) |
| 943 volumes, 18 findings, 100% identical | `agreement 1.0`, 16,031 pairs | Reproduced; only 169 volumes have images (high) |
| T5 text 0.5552 / T6 0.7539 | matches | Pooled, not the split; test-dir T5 floor 0.6411 (high) |
| T5 + perfect single read 0.7392 | matches | Not a hard bound; 0.7894 on test dir; ignores image evidence of past (high) |
| T5 + prior scan 1.0 | hard-coded | Needs ALL prior scans; one prior gives 0.868 pooled / 0.956 test (high) |
| Split 80/10/10 seed 42, zero leakage | summary json | True for patient/volume/QA pair; audits use other splits; 9 intra-train dups (high) |
| MD5 3,039 unique, no clusters | md5 json/log | True; moot-split inconsistency; byte-identical only (high) |
| +7.0 / +6.1 vs floor | compare_to_baseline.json | Computed, but unlike metrics, n=150, 14 clusters, checkpoint lost (high) |
