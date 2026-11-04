# Blind (text-only) baselines and shortcut audits in medical VQA and 3D medical VQA

Method: arXiv abstract pages fetched directly (WebFetch) on 2026-11-04 for all papers marked CONFIRMED; full-text HTML fetched for NeuroQA, MIRAGE, MedVIGIL, RadImageNet-VQA, oncology-VQA (WebFetch summaries of long pages; the tail of NeuroQA, MIRAGE and MedVIGIL pages, beyond ~100k chars, was not read). Earlier notes at research_notes/Medical thesis project review/literature_and_novelty.md were re-verified; corrections are flagged. Venue: all 2026 items below are arXiv preprints; none showed a peer-reviewed venue on its arXiv page.

## Q1. Verification of the named papers

### Takeaway
All six named items exist and say roughly what the earlier notes claimed, with two corrections. (1) NeuroQA is a brain-MRI benchmark paper; its 68.6% image-free figure for 3D-RAD is a single aggregate row in a comparison table (Table 4), not a study of 3D-RAD. (2) The "contamination audit" 2606.10066 is about pretraining contamination of 2D benchmarks and has nothing to do with 3D-RAD or text-only floors. RadImageNet-VQA's text-only result is about its own dataset plus VQA-RAD/SLAKE, and MedVIGIL is a 2D no-image audit; neither touches 3D-RAD.

### Cited Findings
- NeuroQA: "NeuroQA: A Large-Scale Image-Grounded Benchmark for 3D Brain MRI Understanding", Abbasi, Nerrise, Ghosh, Yesiloglu, Mao, Trang, Asadi, Daniel, Chau Loo Kung, Chang, Shah, Turnbull, Younes, Dehkharghani, Adeli (Stanford). arXiv 2605.20525 v1, 19 May 2026, cs.CV, "30 pages, dataset and benchmark release", no venue. 56,953 QA pairs / 12,977 subjects / 12 datasets; reduced closed-format text-only accuracy from over 80% to 44.6%; best zero-shot VLM 47.5% and a supervised 3D CNN 43.7%, both below a 49.4% text-only majority-template floor — [abstract](https://arxiv.org/abs/2605.20525)
- NeuroQA Table 4 (HTML full text): 3D-RAD, "3D chest CT", n = 29,218, random 45.1%, image-free 68.6% [68.0, 69.1], shortcut margin +23.5 pp. M3D-VQA, "3D mixed", n = 5,000, random 25.0%, image-free 52.0% [50.6, 53.4], margin +27.0 pp. Caption: "Image-free accuracy is the maximum of Gemini-2.5-Pro text-only accuracy and the majority-per-template baseline; CIs are 95% Wilson intervals." No per-task breakdown; the text does not mention Task 5, Task 6 or label sequences; 3D-RAD is described as targeting CT, with a temporal column check in Table 1 — [NeuroQA HTML](https://arxiv.org/html/2605.20525) (fetched; the last ~44k characters of the page were not read, so a later appendix mention is not excluded)
- MIRAGE: "MIRAGE: The Illusion of Visual Understanding", Asadi, O'Sullivan, Cao, Nedaee, Rajabalifardi, Fei-Fei Li, Adeli, Ashley. arXiv 2603.21687, v1 23 Mar 2026, v3 2 Apr 2026, cs.AI, no venue. Claims: frontier models describe images never given ("mirage reasoning"); with no image they score strikingly well on general and medical benchmarks, including top rank on a chest X-ray QA benchmark; calls for private benchmarks removing textual cues; introduces B-Clean — [abstract](https://arxiv.org/abs/2603.21687)
- B-Clean details (full text): removes questions any of three candidate models (GPT-5.1, Gemini-2.5-Pro, Gemini-3-Pro) answers correctly without images. Removal rates: MicroVQA 77.0% (1,042 to 240), MedXpertQA-MM 74.3% (2,000 to 514), MMMU-Pro 75.3% (1,730 to 428). Accuracy after cleaning e.g. MicroVQA GPT-5.1 61.5% to 15.4%. Mirage-mode accuracy averages 70-80% of image-enabled accuracy across four models and six benchmarks (range 60-99%, medical at the high end). ReXVQA (~696k MCQ, 160k studies): Qwen2.5-3B-Instruct LoRA-fine-tuned on 100k image-free samples outperformed all frontier multimodal models and, per the authors, radiologists by >10% on average; exact numbers only in a chart. No mention of 3D, CT or 3D-RAD in the visible text (reference list truncated) — [MIRAGE HTML](https://arxiv.org/html/2603.21687)
- Oncology 3D VQA: "Automated Report-Derived Oncology VQA Benchmark for Evaluating Vision-Language Models on 3D Medical Imaging", Liu, Gu, Li, Zhu, Ellison, Wang, Lupo, Yang, Lin. arXiv 2606.02809 v1, 1 Jun 2026, cs.CV, no venue. Four private in-house cohorts (liver CT, liver MRI, lung CT, brain MRI), 6 VLMs zero-shot — [abstract](https://arxiv.org/abs/2606.02809)
- Oncology blind-ablation numbers (GPT-5, N=2000/cell): Lung CT RADS-style sighted 0.498 vs blind 0.520; Lung CT report-derived 0.662 vs 0.688 (blind higher); Liver CT report-derived 0.406 vs 0.159. MedGemma1.0-27B lung CT 0.524/0.455 and 0.473/0.440. Paper states "Lung CT is essentially solvable without the image"; authors caution LLM-generated distractors may inflate blind scores. 3D-RAD, CT-RATE and NeuroQA are not mentioned; MIRAGE is cited as prior work — [HTML](https://arxiv.org/html/2606.02809). Note: "nearly solvable blind" holds for lung CT only, not liver (the abstract says liver needs the image).
- Contamination audit: "Measuring Pretraining Contamination in Public Medical Vision-Language Benchmarks", Bruce Changlong Xu, Lan Wu, Alexander Ryu. arXiv 2606.10066 v1 8 Jun 2026, v2 21 Oct 2026, cs.CV, 28 pages, no venue. Tests SLAKE-En, PathVQA, VQA-RAD, an OmniMedVQA mirror with four detectors; ~19.8% of SLAKE-En images have an unusually close PMC-OA-beta match (SigLIP-B-16; 4.2% with SigLIP-SO400M), likely shared image sources; text-exchangeability signals are unreliable (BLIP-2 reproduces Min-K%++ and top-K overlap signals) so they should not be used standalone. Not about 3D, CT or text-only floors — [abstract](https://arxiv.org/abs/2606.10066)
- RadImageNet-VQA: Butsanets, Corbière, Khlaut, Manceron, Dancette. arXiv 2512.17396 v1 19 Dec 2025, v2 30 Mar 2026, cs.CV, "Preprint, 33 pages", no venue. ~750K CT/MRI images, 7.5M QA. Claims text-only models perform near random on it — [abstract](https://arxiv.org/abs/2512.17396). Section 4.2: "models reach 11.5-33% text-only accuracy" on VQA-RAD and SLAKE; open-ended RadImageNet-VQA "2-10%"; multiple-choice "collapses to the expected 25% baseline"; per-model values only in Figure 4. 3D-RAD appears only in related work/Table 1 (16K images, 136K train, 34K test) and references — [HTML](https://arxiv.org/html/2512.17396). It audits its own dataset (2D slices from CT/MRI), not 3D-RAD.
- MedVIGIL: "MedVIGIL: Evaluating Trustworthy Medical VLMs Under Broken Visual Evidence", Jiang et al. (Hanqi Jiang ... Xiang Li; 15 authors incl. Quanzheng Li, Tianming Liu). arXiv 2605.07919 v1 8 May 2026, v2 22 May 2026, cs.CV, no venue. 300 radiologist-authored cases built from VQA-RAD (120), SLAKE (60), ROCO (60), chest X-ray collections (60); perturbations (false premises, rewording, knowledge-only rewrites, corrupted regions); 16 vision models + 2 text-only baselines; HuatuoGPT-V 7B no-image floor 20.6% (about 4x GPT-4o). Neither 3D nor CT — [abstract](https://arxiv.org/abs/2605.07919), [HTML](https://arxiv.org/html/2605.07919) (only the first ~100k of ~177k chars read; Appendix I no-image tables not seen)

### Inferences
- The earlier note's "contamination audit ... SLAKE-En image overlap, OmniMedVQA mirror flagged" is correct but cite it only for 2D pretraining-contamination methodology, not as evidence about leakage inside 3D-RAD.
- NeuroQA's 68.6% is robust enough to cite as "an independent aggregate image-free estimate for 3D-RAD (n=29,218 closed items; Gemini-2.5-Pro text-only or template-majority)", but it hides per-task structure. The thesis's per-task result (Task 6 alone, structural cause) is not contained in it.
- The 29,218 figure is likely the closed-ended subset (Task 4 alone is 23,472 bench items per the 3D-RAD paper); this is my arithmetic inference, not stated by NeuroQA, and the dominance of Task 4 means the 68.6% is probably driven by Task 4 and not Task 6.

### Gaps
- NeuroQA text beyond char 100,000 and MIRAGE/MedVIGIL tails not read; NeuroQA appendix could contain per-task 3D-RAD numbers.
- MIRAGE ReXVQA exact accuracies unavailable (chart only).

## Q2. Text-only floor, blank/zeroed-image, or mismatched-image control on 3D-RAD or CT-RATE-based VQA

### Takeaway
Only NeuroQA reports an image-free number for 3D-RAD (and M3D-VQA), as one aggregate row, with a text-only/majority-template method. I found no paper reporting a blank/zeroed-image or mismatched/shuffled-image control on 3D-RAD or any CT-RATE-derived VQA set.

### Cited Findings
- NeuroQA is the only hit for 3D-RAD image-free (see Q1).
- Searches for blank-image ablation on CT-RATE VQA returned: a UCSF brain-tumor MRI VQA paper (UCSF-PDGM-VQA, Ghosh et al., arXiv 2605.17140, 16 May 2026) reporting "modality collapse" and language-prior reliance on 3D MRI, not CT-RATE — [abstract](https://arxiv.org/abs/2605.17140). Web search snippet attributes a blank-image Lingshu-32B 66.04% result to a paper in that result set; I did not verify which paper or number — UNVERIFIED.
- "Beyond Accuracy: Evaluating Visual Grounding In Multimodal Medical Reasoning", Zafar, Murali, Vashist, arXiv 2603.03437, 3 Mar 2026, cs.CV: real vs blank vs shuffled images on PathVQA, PMC-VQA, SLAKE, VQA-RAD (all 2D); text-only RLVR on PathVQA has Visual Reliance Score -0.09 (better with mismatched images); VQA-RAD keeps ~81% performance with blank images — [abstract](https://arxiv.org/abs/2603.03437). Closest methodological precedent for the blank/shuffled control, but 2D.
- Ask4VG, Zhu, Li, Xu, Wang, Nie, arXiv 2606.01044, 31 May 2026: probes with original, perturbed, blank, mismatched images on VQA-RAD / PMC-VQA (Qwen2-VL-2B); no 3D — [abstract](https://arxiv.org/abs/2606.01044)
- DeepTumorVQA (Chen ... Zhou, Yuille; arXiv 2605.09679, 10 May 2026; 476K questions, 9,262 CT volumes): no text-only/blind baseline reported on the abstract page — [abstract](https://arxiv.org/abs/2605.09679). Other earlier-flagged 3D CT papers (CT-SpatialVQA 2605.08787, CT-DeltaBench 2608.11534, RadThinking 2605.10761) were seen previously; the earlier notes say none reports a text-only baseline; not re-fetched today.

### Inferences
- A blank/zeroed-CT and shuffled-CT control on a fine-tuned model on 3D-RAD appears unreported; this part of the thesis audit looks novel, qualified by search coverage (see queries).

### Gaps
- Did not check HF/GitHub issues of 3D-RAD, OpenReview discussion (https://openreview.net/forum?id=VB2cgrlikN surfaced but not opened), or Semantic Scholar "cited by" list.

## Q3. Task 6 bracket/prior-label leak and Task 5/6 twin identity

### Takeaway
No paper found reports either finding. Negative result with stated queries.

### Cited Findings
- Queries run (WebSearch, 2026-11-04): (a) "3D-RAD benchmark text-only baseline leakage Task 6 label sequence CT-RATE VQA"; (b) "CT-RATE VQA blank image ablation language prior 3D medical VQA shortcut"; (c) "3D-RAD arXiv 2506.11147 cited text-only shortcut audit 3D medical VQA benchmark image-free"; plus earlier-session queries on 3D-RAD + leakage/text-only/label-sequence (per prior note). The search results returned the 3D-RAD paper/GitHub/HF/OpenReview pages, RadImageNet-VQA, NeuroQA, Photon, DeepTumorVQA, none with Task 6 leakage analysis.
- Full-text fetches of NeuroQA, RadImageNet-VQA and oncology-VQA contain no mention of Task 5/Task 6 or label sequences (3D-RAD cited only as related work/table row).
- The 3D-RAD paper itself (arXiv 2506.11147, Gai et al., NeurIPS 2025 D&B) states Task 6 supplies prior labels as a binary sequence and calls it a "stabilizing prior" (earlier notes, read from local PDF; not re-fetched today); search snippet confirms abstract claim of limited VLM generalisation "particularly in multi-temporal tasks" — [arXiv](https://arxiv.org/abs/2506.11147)

### Inferences
- Quantifying the leak (text-only ceiling, bracket-bit information, T5/T6 identity) is novel as far as findable; but the leak is acknowledged qualitatively by the benchmark authors, so the thesis should frame it as "quantification and consequence for the headline claim", not "discovery that priors are given".

### Gaps
- Not exhaustive: Google Scholar/Semantic Scholar citation lists and GitHub issues not checked; absence of evidence only.

## Q4. Classic foundations and 2024-2026 medical-VQA shortcut/robustness work

### Takeaway
Cite Goyal et al. (VQA v2) and Agrawal et al. (VQA-CP) for the language-prior principle, Ramakrishnan et al. for adversarial regularisation / question-only baselines, and medical-specific bias work (Med-BiasX, CEDO, EHRXQA question-only prior, SURE-VQA no-image baseline, RadImageNet-VQA text-only, MIRAGE).

### Cited Findings
- Goyal, Khot, Summers-Stay, Batra, Parikh, "Making the V in VQA Matter", arXiv 1612.00837, 2 Dec 2016 (v3 15 May 2017); argues language priors let models ignore the image; balanced dataset pairs each question with two images with different answers; all models perform significantly worse on the balanced set. Conference venue not on arXiv page (published at CVPR 2017 per general knowledge; not verified here) — [arXiv](https://arxiv.org/abs/1612.00837)
- Agrawal, Batra, Parikh, Kembhavi, "Don't Just Assume; Look and Answer: Overcoming Priors for VQA", arXiv 1712.00377, 1 Dec 2017, CVPR 2018; VQA-CP v1/v2 with different train/test answer priors per question type; existing models degrade; proposes GVQA — [arXiv](https://arxiv.org/abs/1712.00377)
- Ramakrishnan, Agrawal, Lee, "Overcoming Language Priors in VQA with Adversarial Regularization", arXiv 1810.03649 (listed in search; abstract not fetched; characterised via search snippet as noting the question-only model is a standard blind baseline) — [PDF](https://arxiv.org/pdf/1810.03649)
- Med-BiasX (MICCAI 2025): question-only branch + free-energy penalty; SLAKE-BIAS and VQA-RAD-BIAS — [paper](https://papers.miccai.org/miccai-2025/paper/5135_paper.pdf) (via search snippet). CEDO, arXiv 2506.17903: SLAKE-CP and VQA-RAD-CP language-bias protocols — [arXiv](https://arxiv.org/pdf/2506.17903) (snippet only).
- EHRXQA (arXiv 2310.18652): reported a question-only "Prior (Question)" baseline around 30% on its MIMIC-CXR-VQA — snippet only [PDF](https://arxiv.org/pdf/2310.18652); SURE-VQA (arXiv 2411.19688) lists a "No Img. Baseline" and most-frequent-answer baseline as sanity baselines — snippet only [PDF](https://arxiv.org/pdf/2411.19688v1). Both UNVERIFIED beyond snippets; confirm title/authors before citing.
- Medical VQA survey: arXiv 2111.10056 — [PDF](https://arxiv.org/pdf/2111.10056) (listed, not fetched).
- Counterfactual/blank-image audits 2026: Zafar et al. 2603.03437 and Ask4VG 2606.01044 (above); SurgCheck "Do Vision-Language Models Really Look at Images in Surgical VQA?" arXiv 2605.01911 (title from search only, UNVERIFIED) — [PDF](https://arxiv.org/pdf/2605.01911)
- Earlier-note items not re-fetched: Whole-slide VQA leakage audit 2607.12278 (92.3-100% overlaps) and pathology text-only comparison 2607.21065: snippet only, UNVERIFIED.

### Inferences
- VQA-RAD and SLAKE original data papers (Lau et al., Sci Data 2018; Liu et al., ISBI 2021) are in the project's local ref folder but I did not check whether they report blind baselines; do not claim so. No primary PMC-VQA question-only number found.

### Gaps
- Original VQA-RAD / SLAKE / PathVQA / PMC-VQA question-only analyses not located; Ramakrishnan, EHRXQA, SURE-VQA, Med-BiasX, CEDO require abstract fetch before formal citation.

## Positioning summary for the thesis (inference)
- Not novel: that medical VQA benchmarks can be solved without images (MIRAGE, RadImageNet-VQA, MedVIGIL, oncology-VQA, NeuroQA); that 3D-RAD has a high image-free aggregate (NeuroQA 68.6%, +23.5 pp, Gemini or template-majority); blank/shuffled-image diagnostic methodology (Zafar et al. 2026 in 2D).
- Appears novel (no source found): task-level 3D-RAD analysis; Task 6 bracket carrying the past-finding bit; exact Task 5/Task 6 item identity; blank/zeroed-CT invariance of a fine-tuned 3D-RAD model.
