# Longitudinal / temporal / change-detection benchmarks for medical VQA and radiology (as of 2026-11-04)

Method note: ~12 tool calls. All "fetched" items were opened on arXiv abs/HTML pages via WebFetch (a small model summarised pages, so quotes are as returned; exact wording should be re-checked against the PDF before quoting in the thesis). Items marked UNVERIFIED were not confirmed.

## TemMed-Bench (arXiv 2509.25143)

### Takeaway
TemMed-Bench is a CheXpert Plus-based chest X-ray benchmark of temporal (two-visit) reasoning with three tasks (VQA, report generation, image-pair selection). Many of 12 LVLMs are near random closed-book on VQA (50%) and pair selection (33.3%), but several models are well above chance on VQA (o4-mini 79.15), so "near random" is not universal. It is an arXiv-only preprint (no venue stated).

### Cited Findings
- Authors Zhang, Gu, Hu, Zhou, Piramuthu, Peng; v1 29 Sep 2025, v2 10 Aug 2026; cs.CV; no venue listed on the abstract page — [arXiv abs](https://arxiv.org/abs/2509.25143)
- Abstract: 12 LVLMs (6 proprietary, 6 open); "most LVLMs lack the ability to analyze patients' condition changes"; many at random-guess level in the closed-book setting; multimodal retrieval gives larger gains than none or text-only retrieval; average VQA gain 2.59% — [arXiv abs](https://arxiv.org/abs/2509.25143)
- Test sizes: VQA 2,000 questions from 1,000 image pairs; report generation 1,000; image-pair selection 862 questions with three options; knowledge corpus 17,144 instances (text also says 18,144 collected, 1,000 held out). Raw data from CheXpert Plus (not MIMIC-CXR) — [arXiv HTML](https://arxiv.org/html/2509.25143)
- Random baselines: VQA 50%, pair selection 33.3% — [arXiv HTML](https://arxiv.org/html/2509.25143)
- Closed-book VQA acc / pair-selection acc (Table 3): LLaVA-Med 51.65/n.r.; HuatuoGPT-Vision 53.00/33.29; HealthGPT 46.30/33.64; Qwen2.5-VL 59.90/33.87; Llama3.2-Vision 45.65/33.06; LLaVA-OneVision 63.90/32.83; Gemini 2.5 Flash 47.30/39.33; Claude 3.5 Sonnet 69.90/33.53; GPT-4o 51.65/32.60; GPT-4.1 57.90/35.38; o4-mini 79.15/35.03; o3 64.40/38.05 — [arXiv HTML](https://arxiv.org/html/2509.25143)
- No single-image / prior-removal ablation seen in the first ~100k of 142k chars; Table 5 compares retrieval variants (image-to-image retrieval uses only the current-visit image). Appendix unread — [arXiv HTML](https://arxiv.org/html/2509.25143)

### Inferences
- Pair selection (about 33%) is the cleanest "near random" evidence; VQA closed-book spread (45.7 to 79.2) suggests some language-prior exploitation on VQA, not tested by the authors as far as read.
- "Closed-book" here means no retrieved knowledge, NOT no images; it is not a text-only/blind baseline. Do not cite it as one.

### Gaps
- Appendix (last 42k chars) not read; a blind or one-image ablation may exist there.
- Peer-reviewed venue: none found.

## CT-ΔBench (arXiv 2608.11534) and the "169" question

### Takeaway
CT-ΔBench is a CT-RATE-derived paired-scan difference-report benchmark (2,638 train / 169 val pairs; zero-shot Change-F1 at most 0.0175). It has no text-only baseline and no prior-ablation. The "169" is most likely a coincidence of different units (169 pairs vs 169 3D-RAD volumes/patients), but a shared origin cannot be excluded; UNVERIFIED either way.

### Cited Findings
- Title "CT-ΔBench: A Benchmark for Longitudinal 3D Medical Imaging Difference Reporting with Vision-Language Models"; Tang, Wang, Ren, Wang; v1 12 Aug 2026, v2 4 Sep 2026; abstract page states acceptance to COLM 2026 (from fetch summary; re-check) — [arXiv abs](https://arxiv.org/abs/2608.11534)
- Data: built on CT-RATE; "identify patients with multiple CT studies and form longitudinal pairs by selecting two scans acquired at different time points for the same patient"; only Findings and Impression used; reference difference reports synthesised by Gemini-2.5-Flash — [arXiv HTML v2](https://arxiv.org/html/2608.11534v2)
- Split: 2,638 training paired studies, 169 validation; patient-level split ("To prevent subject leakage..."). The paper does NOT say the val set is CT-RATE's own validation split and does not give how patients were assigned — [arXiv HTML v2](https://arxiv.org/html/2608.11534v2)
- Zero-shot validation Change-F1 (Table 2): MedGemma-1.5-4B 0.0175; M3D-LaMed-Phi-3-4B 0.0051; RadFM-13B 0.0000; Med3DVLM-Qwen2.5-7B 0.0138; Merlin-RadLLaMA-7B 0.0034 — [arXiv HTML v2](https://arxiv.org/html/2608.11534v2)
- DeltaMed (trained baseline) Change-F1 0.0909 / 0.1313 / 0.1980 at 1% / 10% / 100% train; MedGemma fine-tuned 0.0010 / 0.0649 / 0.1577; at 100% DeltaMed missing rate 0.8301, hallucination rate 0.8057 — [arXiv HTML v2](https://arxiv.org/html/2608.11534v2)
- No text-only, shuffle, swap or prior-ablation condition. Closest is a two-stage pipeline (single-timepoint reports then text comparison), which is not a control for prior use — [arXiv HTML v2](https://arxiv.org/html/2608.11534v2)
- 3D-RAD side: the benchmark has 169 images for each of Task 5 and 6 (train 774 images each), and follows CT-RATE's original train/validation split, drawing on its 1,304 validation cases; the page does not state outright that the 169 come from validation — [3D-RAD arXiv HTML](https://arxiv.org/html/2506.11147)

### Inferences
- Different units (169 patients/volumes vs 169 pairs) and different construction rules (3D-RAD: label-derived QA on single current scan; CT-ΔBench: multi-study patients, own patient-level split) point to coincidence. Only a check of case IDs against each other would settle it.
- Both are on CT-RATE, so any patient overlap would be through CT-RATE; compare IDs if the thesis wants to claim anything.

### Gaps
- Per-patient/volume ID lists of the 169 in each paper not compared.
- COLM acceptance only from the abstract-page fetch; not cross-checked on COLM site.

## 3D-RAD Task 5 / Task 6 (arXiv 2506.11147)

### Takeaway
Task 5 is single-scan temporal status diagnosis; Task 6 adds the prior as a text sequence of binary labels (not a prior scan), and the authors themselves call that sequence a "stabilizing prior". Bench 2,873 QA per task over 169 images; train 13,158 per task over 774 images; fine-tuned Phi3-4B: 49.30% on T5, 74.77% on T6.

### Cited Findings
- Task 5 (Static Temporal Diagnosis): meant "to evaluate a model's ability to infer temporal lesion status based solely on a single current 3D scan" — [arXiv HTML](https://arxiv.org/html/2506.11147)
- Task 6 (Longitudinal Temporal Diagnosis): "extends Task 5 by introducing explicit historical diagnostic context to support more accurate temporal reasoning"; Section 3.1: "As current VLMs cannot directly process multiple 3D scans, prior information is provided as a sequence of binary labels" (1 presence, 0 absence) — [arXiv HTML](https://arxiv.org/html/2506.11147)
- "Stabilizing prior": Appendix I, Case 4: longitudinal cues "serve as a stabilizing prior" — [arXiv HTML](https://arxiv.org/html/2506.11147)
- Sizes (Section 3.2.1, Table 2): bench 2,873 QA each for T5 and T6, 169 images each; train (3D-RAD-T) 13,158 QA each, 774 images each. 2,873 + 13,158 = 16,031 — [arXiv HTML](https://arxiv.org/html/2506.11147) (the 2,873/13,158 figures and 169 are also in the earlier local note, which read the local PDF Table 2)
- Table 3 (Section 4.2): M3D-RAD (Phi3-4B) fine-tuned 49.30% T5, 74.77% T6 — [arXiv HTML](https://arxiv.org/html/2506.11147)
- Venue: NeurIPS 2025 Datasets and Benchmarks — [NeurIPS page](https://papers.neurips.cc/paper_files/paper/2025/hash/f8250afec8305ed4a6b27ae405507442-Abstract-Datasets_and_Benchmarks_Track.html); arXiv v2 dated 27 Oct 2025 (the arXiv HTML itself names no venue)
- Built on CT-RATE following its original train/validation split — [arXiv HTML](https://arxiv.org/html/2506.11147)

### Inferences
- Because the prior is a label string, a text-only model can read the answer-relevant bit; the 74.77% vs 49.30% gap is consistent with that (inference; the paper does not say it).

### Gaps
- Last ~2.6k chars of the HTML not read. Exact Section/Table of the "more robust even without fine-tuning" quote relied on the earlier local note, not re-verified. Whether 169 is exactly the CT-RATE validation subset: not confirmed.

## Other longitudinal medical VQA / report work

### Takeaway
Longitudinal work is dense for chest X-ray (MIMIC-Diff-VQA, TemMed-Bench) and new for CT in Aug-Sep 2026 (CT-ΔBench, ALTER); no 3D CT longitudinal VQA other than 3D-RAD T5/T6 was found.

### Cited Findings
- MIMIC-Diff-VQA / Medical-Diff-VQA task (Hu et al., arXiv 2307.11986; v1 22 Jul 2023, v2 27 Aug 2024; KDD 2023 per search result): 700,703 QA pairs from 164,324 main/reference image pairs; MIMIC-CXR; 8:1:1 split; seven question types incl. "difference"; code github.com/Holipori/MIMIC-Diff-VQA — [arXiv abs](https://arxiv.org/abs/2307.11986); split and venue from search summary [search result](https://arxiv.org/pdf/2307.11986)
- MS-CXR-T: not on the Boecking et al. (BioViL, ECCV 2022, arXiv 2204.09817) abstract page; page lists only MS-CXR phrase grounding. MS-CXR-T is from the later BioViL-T paper (Bannur et al., CVPR 2023); UNVERIFIED here, from memory — [BioViL abs](https://arxiv.org/abs/2204.09817)
- ALTER (Li, Liang, Li; arXiv 2608.05615, 6 Aug 2026): 3D CT report generation with prior CT and report as context; evaluated on RadGenome-ChestCT validation and CTRG-Chest-548K test; abstract page gave no prior-availability findings (search summary says there is such an analysis; UNVERIFIED) — [arXiv abs](https://arxiv.org/abs/2608.05615)
- BrainDiff (arXiv 2609.00593, 1 Sep 2026, brain MRI): factorial test varying prior-report availability and image identity; +0.0387 rg_er attributed to visual contribution, larger when the prior report is withheld; prior-report dropout counterfactual objective raises image reliance about 47% — [arXiv abs](https://arxiv.org/abs/2609.00593)
- Dalla Serra et al., prior scan as extra input for CXR report generation, EMNLP 2023, arXiv 2310.05881 — [arXiv](https://arxiv.org/abs/2310.05881) (from search snippet only)
- Earlier local note items, NOT re-verified here: RadThinking (arXiv 2605.10761, 11 May 2026, 20,362 CTs, temporal labels, no experiments), MI-CXR (arXiv 2605.15574), CT2RepLong (MICCAI 2024), NeuroQA (arXiv 2605.20525; image-free 68.6% on 3D-RAD overall) — [literature_and_novelty.md]
- Chest ImaGenome temporal: not searched; UNVERIFIED.

### Inferences
- BrainDiff is the clearest example of a prior-image-identity ablation in this literature (MRI, not CT-RATE).

### Gaps
- Chest ImaGenome temporal, MS-CXR-T primary source, CT RECIST-style follow-up VLMs: not researched.

## Does any benchmark check use of the prior scan?

### Takeaway
Among those verified: BrainDiff does (factorial prior-report x image-identity). TemMed-Bench (as far as read) and CT-ΔBench do not; 3D-RAD does not and cannot (prior is labels). No swapped-prior test found for CT.

### Cited Findings
- CT-ΔBench: no text-only/shuffle/swap condition — [arXiv HTML v2](https://arxiv.org/html/2608.11534v2)
- TemMed-Bench: closed-book (no retrieval) and retrieval variants only in what was read — [arXiv HTML](https://arxiv.org/html/2509.25143)
- 3D-RAD: no text-only baseline in the paper per earlier local note (not re-verified here) — [literature_and_novelty.md]
- BrainDiff factorial test — [arXiv abs](https://arxiv.org/abs/2609.00593)

### Inferences
- Positioning: thesis contribution = a text-only/blind and swapped-label test on 3D-RAD T5/T6, which no verified paper has done; NeuroQA gives only an aggregate image-free number (from earlier note).

### Gaps
- Other works may have such ablations; coverage is limited.
