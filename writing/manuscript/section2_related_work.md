# Related work (draft)

Status: draft built from `reports/Blind baselines longitudinal VQA.md` (as of 2026-11-04).
Only sources marked verified there are cited. Before submission: check every quote against the
PDF, add author lists, and run the open searches listed at the end of this file. Every novelty
statement below is bounded by search coverage and reads "to our knowledge".

## 1. Language priors and answers without the image

Visual question answering has long been known to reward language priors. Goyal et al. balanced
VQA with complementary image pairs so that question-only guessing stops paying off [VQA v2,
arXiv 1612.00837], and Agrawal et al. changed the answer priors between train and test to expose
models that rely on the question alone [VQA-CP, arXiv 1712.00377].

The same problem is now documented for medical VQA. MIRAGE shows that frontier models give
fluent answers about images they never received, and its B-Clean filter removes 74-77% of the
questions in three medical benchmarks because candidate models answer them without the image
[arXiv 2603.21687]. RadImageNet-VQA reports text-only accuracy of 11.5-33% on VQA-RAD and SLAKE
but near chance on its own data [arXiv 2512.17396], and MedVIGIL measures a no-image floor of up
to 20.6% on radiologist-authored 2D cases [arXiv 2605.07919]. For 3D data, a report-derived
oncology benchmark finds lung CT questions "essentially solvable without the image", while liver
CT questions are not (blind 0.159 against sighted 0.406) [arXiv 2606.02809]; the shortcut is
therefore specific to the question family and should not be stated for CT in general.

The blank-image and shuffled-image diagnostic we use has 2D precedents: Zafar et al. compare real,
blank and shuffled images on four medical VQA sets and find that a text-only model keeps about
81% of its VQA-RAD performance with blank images [arXiv 2603.03437], and Ask4VG probes original,
perturbed, blank and mismatched images [arXiv 2606.01044]. Our contribution is not the method.

## 2. 3D-RAD and what is already known about it

3D-RAD is a six-task 3D medical VQA benchmark built on CT-RATE (NeurIPS 2025 Datasets and
Benchmarks) [arXiv 2506.11147]. Task 5 asks for the lesion's temporal status from a single current
scan. Task 6 "extends Task 5 by introducing explicit historical diagnostic context", supplied as a
sequence of binary labels because "current VLMs cannot directly process multiple 3D scans". The
authors call this a "stabilizing prior" (Appendix I), and report 49.30% (Task 5) and 74.77%
(Task 6) for a fine-tuned Phi3-4B. The design of Task 6 is therefore known and intended; what the
paper does not report is a text-only floor or any measure of how much the label sequence reveals.

The closest published number is from NeuroQA, which lists 3D-RAD with an image-free score of
68.6% (random 45.1%, n = 29,218), defined as the maximum of Gemini-2.5-Pro text-only accuracy and
a per-template majority baseline [arXiv 2605.20525]. NeuroQA does not say which tasks make up the
29,218 items. The count equals the 3D-RAD benchmark sizes for Task 4 + Task 5 + Task 6
(23,472 + 2,873 + 2,873), and 45.1% is exactly the random baseline for that mix (50% on binary
Task 4, 25% on four-class Tasks 5 and 6). This is our inference, not a statement in NeuroQA, but
two independent numbers agree. If it is right, NeuroQA's figure is about 80% Task 4 and pools in
Task 6, so it is an aggregate and says nothing separate about the longitudinal tasks. We treat it
as an independent confirmation that 3D-RAD is largely answerable without images, and our
per-task analysis as the decomposition it does not provide.

## 3. Longitudinal medical VQA

Longitudinal benchmarks are dense for chest X-ray: MIMIC-Diff-VQA has 700,703 question-answer
pairs over 164,324 image pairs [arXiv 2307.11986], and TemMed-Bench covers temporal VQA, report
generation and image-pair selection on CheXpert Plus [arXiv 2509.25143]. TemMed-Bench's
"closed-book" setting means no retrieved knowledge, not no images, so it is not a blind baseline;
only its pair-selection task sits near chance (about 33%).

For CT the work is recent. CT-ΔBench builds CT-RATE follow-up pairs (2,638 train, 169 validation)
with generated difference reports and reports zero-shot Change-F1 of at most 0.0175, including
0.0051 for M3D-LaMed-Phi-3-4B; it has no text-only, shuffled or swapped-prior condition
[arXiv 2608.11534]. ALTER feeds a prior CT and report into CT report generation
[arXiv 2608.05615]. The only verified study that tests whether a longitudinal model uses the
prior is BrainDiff, a factorial test of prior-report availability against image identity on brain
MRI [arXiv 2609.00593]. We found no swapped-prior or zeroed-prior test on CT.

## 4. Position of this work (Part A)

Not new: that medical and 3D medical VQA can be answered without the image; that 3D-RAD has a high
image-free aggregate score; the blank and shuffled image diagnostic; and that Task 6 supplies the
prior as labels.

To our knowledge new, as of 2026-11-04 and from a negative search only: (a) the Task 6 bracket
equals the past-finding bit on 16,031 of 16,031 rows, which removes two of the four answer
classes; (b) Tasks 5 and 6 are the same 16,031 items with identical answers, so Task 5 is a
built-in un-leaked twin (same items: Task 6 minus Task 5 = +18.1 points on validation, patient-level
CI 10.9-26.5, exact McNemar p = 2e-13); (c) a zeroed-CT control on a fine-tuned 3D-RAD model; and
(d) the model in (c) was given both the earlier and the current scan for Tasks 5 and 6 (checked in
the loader for every row, `leak_analysis/PRIOR_SCAN_INPUT.md`) and its Task 6 predictions did not
change when the volume was zeroed (0 of 238 on validation).

Scope limits that belong in the text: one baseline, one seed, the epoch-1 checkpoint, one
perturbation (zeros), validation per-item data only, and 14 validation and 20 test temporal
volumes. The statement supported is that this baseline is indistinguishable from the text-only
floor, not that VLMs cannot do longitudinal CT reasoning. Controls with mean, noise, swapped and
zeroed-prior inputs, and epochs 2-3, are still to run.

## 5. Mask-derived 3D CT VQA and training-free fixes (Part B)

Questions generated deterministically from segmentation masks are prior art. DeepTumorVQA
generates about 395K questions over 9,262 CTs [arXiv 2505.18915]; a later version reports 476K
questions from organ volumes, HU values and lesion counts over 43 structures and evaluates
M3D-Llama2 and M3D-Phi3 [arXiv 2605.09679] (whether the two IDs are one lineage or two papers is
still to be checked). RadGPT turns mask data into reports [arXiv 2501.04678].

M3D-LaMed has already seen TotalSegmentator: M3D-Seg includes it as dataset 0011 (1,203 scans),
and the authors annotated 210 of its images for M3D-RefSeg [arXiv 2404.00578]. M3D-VQA and M3D-Cap
come from Radiopaedia. TotalSegmentator provides anatomy masks only (v2: 117 CT classes, 1,228
exams, CC BY 4.0) [arXiv 2208.05868 for v1; Zenodo 10047292 for v2], so questions are limited to
presence, size, laterality, relative position and counts. A TotalSegmentator question set is
therefore not new, and its value is as an audit: a seen-source against unseen-source split for
M3D-LaMed with blind and zeroed-volume controls, using the same protocol as Part A.

Training-free methods are prior art: visual contrastive decoding [arXiv 2311.16922], PAI
[arXiv 2407.21771], OPERA [arXiv 2311.17911] and contextual calibration [arXiv 2102.09690]. We
found none applied to a 3D medical VLM, so the claim to make is an evaluation of known methods on
a 3D model, not a new method.

## Still to do before this goes in the thesis

- Check quotes against the PDFs (the notes used summarising fetches).
- Run the searches the report could not: Semantic Scholar "cited by" for 2506.11147, 3D-RAD GitHub
  issues and OpenReview, and the unread tail of NeuroQA.
- Resolve DeepTumorVQA's two IDs; compare M3D-Seg split case IDs with TotalSegmentator v2.
- Do not cite until their abstracts are fetched: EHRXQA, SURE-VQA, Med-BiasX, CEDO, SurgCheck, the
  two 2607 audits, RadThinking, and the MS-CXR-T origin.
- Add author lists and venues to every entry.
