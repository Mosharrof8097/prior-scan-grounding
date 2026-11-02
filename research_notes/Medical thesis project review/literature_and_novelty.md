# Literature landscape and novelty of the 3D-RAD Task 5/6 leakage audit (as of 2026-10-27)

Method note: ~17 tool calls. Search snippets were checked against arXiv abstract/HTML pages where marked "fetched". Items marked "snippet only" were not opened. Local PDFs/docx were read directly.

## Q1. What exactly is 3D-RAD, and has anyone reported the Task 6 leakage or text-only baselines?

### Takeaway
3D-RAD is a CT-RATE-derived six-task 3D Med-VQA benchmark (NeurIPS 2025 D&B). Its own paper openly says Task 6 gives the past label sequence as text and that this makes Task 6 "more amenable" even zero-shot, but it never quantifies the shortcut or runs a text-only baseline. I found no paper that reports the Task 6 bracket leakage specifically. One 2026 paper (NeuroQA) reports an image-free score on 3D-RAD overall (68.6%), which is the closest overlap.

### Cited Findings
- Six tasks: anomaly detection, image observation, medical computation, existence detection, static temporal diagnosis (T5), longitudinal temporal diagnosis (T6); open and closed ended; ~170K QA total; benchmark 33,910 QA pairs and training set 3D-RAD-T of 136,195 — [3D-RAD arXiv v2 (local PDF 2506.11147v2, read)](https://arxiv.org/abs/2506.11147)
- Table 2 sizes. Bench: T1 2,666 / T2 1,024 / T3 1,002 / T4 23,472 / T5 2,873 / T6 2,873 QA; T5 and T6 each 169 images/patients. Train: 6,055 / 2,081 / 1,573 / 100,170 / 13,158 / 13,158 — same paper (local PDF, Table 2).
- Task 5 "does not provide prior image or label history"; four classes (Refractory, Resolved, New, No Abnormality) over 18 labels; single current scan only — same paper.
- Task 6 "extends Task 5": "prior information is provided as a sequence of binary labels" (e.g. "Arterial wall calcification progression: [1, 0]") because "current VLMs cannot directly process multiple 3D scans"; same four classes; "Task 6 adopts the same answer construction strategy as Task 5" — same paper.
- Authors' reported numbers: Phi3-4B fine-tuned gets 49.30% on T5 and 74.77% on T6; zero-shot ~24-25% on both; LLaMA2-7B, M3D, RadFM similar (Table in appendix; local PDF).
- The authors acknowledge the effect qualitatively: Task 6 "provides more external structure, making it more robust even without fine-tuning"; "longitudinal cues (label sequences) serve as a stabilizing prior" — same paper, failure-case appendix. Limitations section admits the task uses labels only and no model takes multiple volumes; no open-ended format for T6.
- Dataset built on CT-RATE (16,188 scans/11,255 patients per a later description) — [Photon, arXiv 2603.25155 (snippet only)](https://arxiv.org/pdf/2603.25155)
- GitHub repo README: no errata, one open issue (content not retrieved), last news entry 2025-10-23, MIT licence, CT-RATE CC-BY-NC-SA — [GitHub Tang-xiaoxiao/M3D-RAD (fetched)](https://github.com/Tang-xiaoxiao/M3D-RAD). HF dataset card fetch failed; not verified.
- NeuroQA (Stanford, 19 May 2026) Table 4: 3D-RAD (3D chest CT, n=29,218 closed items) random 45.1%, image-free 68.6% [68.0, 69.1], margin +23.5 pp. "Image-free" = max(Gemini-2.5-Pro text-only, majority-per-template). No per-task breakdown, and it does not mention Task 6 bracket or T5/T6 twins — [NeuroQA arXiv 2605.20525 (fetched)](https://arxiv.org/html/2605.20525)
- Papers citing 3D-RAD (CT-ΔBench, RadThinking) use it only as related work, no audit — [CT-ΔBench](https://arxiv.org/html/2608.11534v1), [RadThinking](https://arxiv.org/html/2605.10761v1) (both fetched).
- Searches for 3D-RAD + leakage/text-only/label-sequence returned nothing about 3D-RAD — WebSearch, no source.

### Inferences
- The project's 75.4% Task 6 text-only figure, the exact T5/T6 100% answer identity, and the 73.9% single-volume ceiling appear novel as quantified, task-level findings; NeuroQA's 68.6% is aggregate and uses an LLM, so it is complementary and should be cited and reconciled (different method: Gemini + template majority, 29,218 items).
- Because the 3D-RAD authors themselves describe the label sequence as a "stabilizing prior", reviewers will say "this is by design". The framing must be: the design makes the task's headline metric uninformative about image use, and the paper's claim that VLMs "generalise poorly on multi-temporal tasks" is confounded.
- Paper numbers are consistent with leakage: fine-tuned T6 74.77% is close to the project's 75.4% text-only ceiling (inference, not stated by authors).
- Check whether the paper's Task 6 test split matches the project's patient-level split numbers; project's 16,031 T6 rows vs paper's 2,873 bench + 13,158 train = 16,031. Matches exactly (arithmetic from Table 2).

### Gaps
- HF dataset card and GitHub open issue content not retrieved; OpenReview reviews of 3D-RAD not checked (may contain reviewer remarks on Task 6).
- Could not confirm whether 3D-RAD authors released a v3/erratum after Oct 2025.

## Q2. Language-prior / shortcut / blind-baseline literature (2D and 3D medical VQA)

### Takeaway
There is now substantial 2025-2026 evidence that medical VQA benchmarks are solvable without images (MIRAGE, RadImageNet-VQA, MedVIGIL, oncology-VQA, NeuroQA). A text-only audit of a 3D benchmark is no longer novel in itself; novelty must come from the structural, task-level mechanism (answer-bit in the prompt, matched leaked/un-leaked twin) and the longitudinal setting.

### Cited Findings
- MIRAGE (Asadi, ... Fei-Fei Li, Adeli, Ashley; 23 Mar 2026, v3 2 Apr 2026): frontier models without images score highly on general and medical benchmarks; one ranked first on a chest X-ray QA benchmark with no images; calls for private benchmarks removing textual cues; proposes B-Clean — [arXiv 2603.21687 (fetched)](https://arxiv.org/abs/2603.21687). Specific figures (70-80% retained accuracy, B-Clean removing 74-77% of questions, 3B text-only "super-guesser" on ReXVQA) come from secondary coverage only — [The Decoder](https://the-decoder.com/ai-models-confidently-describe-images-they-never-saw-and-benchmarks-fail-to-catch-it/) (snippet only, unverified).
- RadImageNet-VQA (Dec 2025): text-only open-ended 11.5-33% on VQA-RAD and SLAKE vs 2-10% on its own benchmark; "textual shortcuts" in existing benchmarks — [arXiv 2512.17396 (snippet)](https://arxiv.org/pdf/2512.17396)
- MedVIGIL (May 2026): 300-case radiologist-checked suite, 16-model audit with no-image ablation; HuatuoGPT-V 7B highest no-image floor 20.6% — [arXiv 2605.07919 (snippet)](https://arxiv.org/html/2605.07919)
- Report-derived oncology 3D VQA (Liu et al., 1 Jun 2026): blind ablation; lung CT "nearly solvable without" the image, a leading closed model scored higher blinded; private in-house data, no 3D-RAD mention — [arXiv 2606.02809 (fetched)](https://arxiv.org/abs/2606.02809)
- Med-BiasX (MICCAI 2025): SLAKE-BIAS / VQA-RAD-BIAS protocols for question-only shortcuts — [MICCAI 2025 paper (snippet)](https://papers.miccai.org/miccai-2025/paper/5135_paper.pdf)
- Contamination audit of public medical VL benchmarks (2026): SLAKE-En image overlap, OmniMedVQA mirror flagged — [arXiv 2606.10066 (snippet)](https://arxiv.org/html/2606.10066)
- Whole-slide VQA data-leakage audit (2026) found 92.3-100% train-test overlaps — [arXiv 2607.12278 (snippet)](https://arxiv.org/abs/2607.12278); pathology VLM text-only comparison — [arXiv 2607.21065 (snippet)](https://arxiv.org/pdf/2607.21065)
- NeuroQA also reports image-free accuracy for M3D-VQA: 52.0% vs 27.0 pp margin (see Q1 link).
- CT-SpatialVQA "Lost in Volume" (MBZUAI, May 2026) evaluates 3D VLMs (Merlin, M3D, RadFM, Med3DVLM) on CT-RATE; PDF could not be parsed, so text-only content unverified — [arXiv 2605.08787](https://arxiv.org/pdf/2605.08787)

### Inferences
- Reviewers will likely ask the project to cite MIRAGE and B-Clean and to position against them: MIRAGE is LLM-based and benchmark-agnostic; this project gives an exact information-theoretic ceiling from label structure, which is a stronger, deterministic argument.
- I found no pre-2026 shortcut audit of M3D, RadGenome-ChestCT, CT-RATE/CT-CLIP, RadFM, Merlin, Med3DVLM (from searches; not exhaustive). NeuroQA's M3D-VQA row is the only one found.
- PMC-VQA blind-baseline critiques: none found in my searches.

### Gaps
- Original VQA-RAD/SLAKE language-prior papers from 2018-2023 (e.g., VQA-RAD's own question-only analysis) not looked up; rely on your own knowledge or fetch before citing.
- MedVIGIL, RadImageNet-VQA, 2606.10066 only seen as snippets.

## Q3. Longitudinal / temporal medical VQA and report generation: prior-scan use and findings on ignoring priors

### Takeaway
Pairwise/longitudinal benchmarks exist for CXR (MIMIC-Diff-VQA, TemMed-Bench, MI-CXR) and appeared for CT in 2026 (CT-ΔBench, RadThinking). Evidence that models ignore or mis-handle priors is mostly "near-random closed-book" and "hallucinated prior comparison" rather than a controlled prior-ablation; none uses a matched leaked/un-leaked twin pair.

### Cited Findings
- TemMed-Bench (Zhang et al., Sep 2025; v2 10 Aug 2026; arXiv-only): VQA, report generation, image-pair selection; 12 LVLMs; "most LVLMs lack the ability to analyze patients' condition changes"; many near random in closed-book; multimodal retrieval +2.59% on VQA — [arXiv 2509.25143 (fetched)](https://arxiv.org/abs/2509.25143)
- MIMIC-Diff-VQA: 700,703 QA pairs from 164,324 longitudinal CXR pairs — [Emergent Mind summary (secondary)](https://www.emergentmind.com/topics/longitudinal-medical-visual-question-answering-diff-vqa)
- MI-CXR (ACL Findings 2026): multi-interval CXR; models "fail to enforce temporal constraints" — [arXiv 2605.15574 (snippet)](https://arxiv.org/pdf/2605.15574)
- CT-ΔBench (Tang et al., Univ. Tennessee Chattanooga, 12 Aug 2026): CT-RATE prior/follow-up pairs, Gemini-synthesised difference reports, 2,638 train / 169 val pairs, zero-shot Change-F1 <= 0.0175; does not claim models ignore priors; no text-only baseline — [arXiv 2608.11534 (fetched)](https://arxiv.org/html/2608.11534v1). NOTE: this is the closest CT-RATE longitudinal competitor; 169 val pairs echoes 3D-RAD's 169 T5/T6 volumes (possible shared source; unverified).
- RadThinking (JHU, 11 May 2026): 20,362 CTs/9,131 patients, 10 European sites; temporal labels (new/growing/stable/shrinking/resolved); explicitly "no experimental results", no text-only baselines — [arXiv 2605.10761 (fetched)](https://arxiv.org/html/2605.10761v1)
- CT2Rep / CT2RepLong (MICCAI 2024, CT-RATE): longitudinal fusion with prior visits improves NLG and CE metrics — [CT2Rep listing (snippet)](https://portal.fis.tum.de/en/publications/ct2rep-automated-radiology-report-generation-for3d-medical-imagin/)
- Prior-comparison hallucination work (DPO on MIMIC-CXR with prior refs removed; SDLS steering attributing it to grounding failure) — summarised in the user's local review docx (refs [5],[6]); original papers not re-fetched.

### Inferences
- The "models ignore priors" claim is mostly untested with controlled ablations; the project's blank-image ablation on single-scan 3D plus matched twin is a methodologically distinct contribution, but note the project's VLM never receives the prior scan, so the thesis title ("prior-scan grounding") is currently ahead of the evidence: what was shown is that Task 6 is label-leaked and the 1.5B model barely uses the CT, not that models ignore a prior scan.
- CT-ΔBench and RadThinking can be called concurrent work (Aug and May 2026).

### Gaps
- No search for CT follow-up/RECIST longitudinal VLMs, Chest ImaGenome temporal, or Libra/MLRG-style longitudinal CXR RG papers; add if needed.
- No paper found that supplies both volumes in 3D-RAD (i.e., fixing the single-volume cap).

## Q4. Venues and reviewer expectations

### Takeaway
(Mostly inference; I did not look up 2026-27 deadlines.) Best fits for a benchmark-audit negative result: NeurIPS D&B 2027 (the original venue, so a "corrected benchmark" angle is natural), TMLR, MICCAI/MIDL workshops, ML4H, and JBHI/Scientific Data style journals for a de-leaked release.

### Cited Findings
- 3D-RAD itself: NeurIPS 2025 D&B — [NeurIPS page](https://papers.neurips.cc/paper_files/paper/2025/hash/f8250afec8305ed4a6b27ae405507442-Abstract-Datasets_and_Benchmarks_Track.html)
- Comparable audit/negative-result papers are mostly arXiv preprints in 2026 (MIRAGE, NeuroQA, oncology-VQA); TemMed-Bench has been arXiv-only for ~13 months — fetched pages above.
- Local reference set includes ML4H 2023 and Scientific Data papers (see local files section).

### Inferences
- Reviewers will expect: multiple seeds/CI (patients 169 in bench, so bootstrap CIs by patient); stronger baselines (a frontier LLM text-only run, plus open 3D VLMs such as RadFM, M3D-LaMed, Med3DVLM, and a properly trained model given both volumes); human/radiologist or at least a larger-model reference; a released de-leaked Task 6 (e.g. prompt without bracket) with a matched twin protocol; a model that actually consumes the prior scan; and contact with the 3D-RAD authors. The small 1.5B LoRA with lost checkpoint (README: adapters lost) is weak evidence on its own. A 1.5B model "beating floors by 6-7 points" and blank-image ablation need seeds and a no-leak CT-dependence test (e.g., shuffled-image).
- An undergraduate (B.Sc.) thesis is well served by a workshop/arXiv paper; TMLR is realistic for a rigorous negative result.

### Gaps
- No verified 2026-27 call-for-papers dates or acceptance norms; verify on official sites.

## Q5. Possible scoops/overlap (2025-2026)

### Cited Findings
- NeuroQA, 19 May 2026: image-free 3D-RAD 68.6% (+23.5 pp) — https://arxiv.org/html/2605.20525
- MIRAGE, 23 Mar 2026: general text-only exploitation of medical benchmarks, B-Clean — https://arxiv.org/abs/2603.21687
- CT-ΔBench, 12 Aug 2026 (CT-RATE longitudinal) — https://arxiv.org/html/2608.11534v1
- RadThinking, 11 May 2026 — https://arxiv.org/html/2605.10761v1
- TemMed-Bench v2, Aug 2026 — https://arxiv.org/abs/2509.25143
- Oncology report-derived 3D VQA with blind ablation, 1 Jun 2026 — https://arxiv.org/abs/2606.02809
- CT-SpatialVQA, May 2026 — https://arxiv.org/pdf/2605.08787
- RadImageNet-VQA, Dec 2025 — https://arxiv.org/pdf/2512.17396

### Inferences
- No direct scoop found of "3D-RAD Task 6 bracket leakage / T5-T6 twin / 73.9% ceiling". Highest overlap risk: NeuroQA (aggregate image-free number) and MIRAGE (conceptual). Moderate: CT-ΔBench (same data, 169 pairs).
- Search coverage was limited (Google-like search, US-only); a Semantic Scholar "cited by 3D-RAD" listing should be run before submission.

## Local reference material (what it covers and misses)

- `literature review /Hallucination_3D_MedVQA_Literature_Review_IEEE.docx` (read): hallucination detection (RadFlag, ReXTrust, DSE), prior-comparison hallucination mitigation (DPO, SDLS, visual contrastive decoding), medical hallucination benchmarks, 3D-RAD description, positioning table. Frames the thesis as hallucination detection for multi-temporal VQA. Misses: language-prior/blind-baseline/shortcut literature (VQA-RAD/SLAKE/PMC-VQA critiques, MIRAGE, RadImageNet-VQA, NeuroQA), longitudinal VQA benchmarks (MIMIC-Diff-VQA, TemMed-Bench, MI-CXR), CT longitudinal RG (CT2RepLong), all 2026 CT longitudinal work; its statement that 3D-RAD "multi-temporal task weakest" is the leakage-confounded claim.
- `literature review /3d_image_papers_table.docx` (read, partial): table of 3D VLMs (Lian et al. Sci Rep 2026 2D-encoder model, Med3DVLM JBHI 2026, Reg2RG TMI 2025, Merlin Nature 2026 etc.) with datasets/methods/results; all on report generation and M3D-VQA closed accuracy. Misses: any evaluation of image dependence of those numbers (M3D-VQA closed 79.95-82.9% vs NeuroQA's 52.0% image-free).
- `perper ref/` PDFs (titles only checked): 3D-RAD v2 (read in detail), M3D, RadFM (Nat Commun s41467-025-62385-7), CT-RATE/CT-CLIP (2403.17834v4), SegVol, SLAKE, VQA-RAD (sdata2018251), an ML4H 2023 paper (PMLR 225, likely a med VQA/VL paper; not identified), Merlin v2 (Mar 2026), Med3DVLM, Reg2RG, foundation-model perspectives (Nature Medicine s41591-024-03185-2), npj AI review, Sci Rep s41598-026-39526-z. These cover the models and datasets but none is a shortcut/audit/longitudinal paper.
