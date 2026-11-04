# Part B prior art and data facts: auditing M3D-LaMed on a TotalSegmentator mask-verified 3D CT question benchmark (as of 2026-11-04)

Note: WebFetch answers come from a small summarising model; items marked UNVERIFIED could not be confirmed from a primary page. Local files (PLAN.md, literature_and_novelty.md) were only grepped; PLAN.md lists these same checks as open Phase-2 items.

## Q1. Does M3D-Data (M3D-Seg etc.) contain TotalSegmentator volumes/masks?

### Takeaway
Yes. TotalSegmentator is one of the 25 CT segmentation datasets aggregated into M3D-Seg (dataset ID 0011, 1,203 scans per the paper). Any TotalSegmentator-derived Part B benchmark therefore has a training-contamination risk for M3D-LaMed (at least for segmentation/positioning tasks). M3D-VQA/M3D-Cap come from Radiopaedia, not TotalSegmentator.

### Cited Findings
- M3D-Seg = 25 public 3D CT segmentation datasets, IDs 0000-0024: CHAOS, HaN-Seg, AMOS22, AbdomenCT-1k, KiTS23, KiPA22, KiTS19, BTCV, Pancreas-CT, 3D-IRCADB, FLARE22, TotalSegmentator (0011), CT-ORG, WORD, VerSe19, VerSe20, SLIVER07, QUBIQ, MSD-Colon, MSD-HepaticVessel, MSD-Liver, MSD-lung, MSD-pancreas, MSD-spleen, LUNA16 — [HF dataset card GoodBaiBai88/M3D-Seg](https://huggingface.co/datasets/GoodBaiBai88/M3D-Seg)
- Paper: M3D-Seg has 5,772 labeled 3D CT scans from 25 datasets (4,610 train / 1,162 test, Appendix Table 2); TotalSegmentator listed with 1,203 scans (Appendix Table 2); authors annotated a subset, M3D-RefSeg (210 images), from TotalSegmentator for referring-expression segmentation (Sec 3.2.2, Table 1) — [M3D arXiv HTML 2404.00578](https://arxiv.org/html/2404.00578)
- M3D-Cap: 120,092 images / 42,496 texts, from Radiopaedia (not TotalSegmentator) — [same](https://arxiv.org/html/2404.00578)
- M3D-Data headline: 120K image-text pairs, 662K instruction-response pairs; M3D-Bench has eight tasks — [arXiv abs](https://arxiv.org/abs/2404.00578)
- GitHub README lists M3D-Seg as 5,772 images / 149,196 category texts and masks; does not name sources or mention TotalSegmentator — [GitHub BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D)
- Training used 8x A100 80GB, bf16 via DeepSpeed — [M3D arXiv HTML](https://arxiv.org/html/2404.00578)

### Inferences
- Whether the specific TotalSegmentator cases in M3D-Seg train vs test split overlap your benchmark cases is UNVERIFIED; M3D's 1,203 count differs from v1 (1,204) / v2 (1,228) CT counts, so M3D likely used v1. The overlap must be checked by case ID (Data/process scripts in the repo; not inspected).
- Contamination is a stated limitation to report; also a design choice: use v2-only new cases (1,228 minus 1,204 v1 cases is roughly 24, which is too few) or treat M3D as "seen-domain" and compare with an unseen-source control set (e.g., AbdomenAtlas/other volumes not in the 25 list).
- Even if volumes were seen, M3D-Seg trains segmentation/positioning (SegVol-like), not organ-volume VQA text, so seen-ness mainly affects image-side familiarity.

### Gaps
- Exact case-level split overlap with TotalSegmentator v2 not checked (needs M3D-Seg split files and meta.csv).
- Whether M3D-VQA includes any TotalSegmentator images: not found; VQA derived from M3D-Cap (Radiopaedia) per paper summary, not fully confirmed from text.

## Q2. TotalSegmentator facts (v1, v2): labels, counts, classes, license, size, Zenodo IDs

### Takeaway
TotalSegmentator is anatomy-only segmentations (117 classes in v2 CT), CC BY 4.0, with images sampled from clinical routine across "a range of pathologies". Pathology is NOT a labelled class set in the main CT release; meta.csv holds split plus some patient/acquisition fields, but exact column list is only partially confirmed.

### Cited Findings
- v1 paper: 1,204 CT exams, 104 structures (27 organs, 59 bones, 10 muscles, 8 vessels); "major pathologies" in test set; a separate 4,004-exam set for an aging study — [arXiv 2208.05868](https://arxiv.org/abs/2208.05868)
- v1 Zenodo record 6802614 (DOI 10.5281/zenodo.6802613 concept), v1.0 published 2022-07-06, Totalsegmentator_dataset.zip 28.4 GB, CC BY 4.0 — [Zenodo 6802614](https://zenodo.org/records/6802614)
- v2.0.1 (2023-10-27): 117 anatomical structures, 1,228 CT images, Totalsegmentator_dataset_v201.zip 23.6 GB, CC BY 4.0, DOI 10.5281/zenodo.10047292; 102-subject exploration subset exists; images "randomly sampled from clinical routine... range of pathologies, scanners, sequences, institutions"; University Hospital Basel — [Zenodo 10047292](https://zenodo.org/records/10047292)
- Software repo: default `total` task = 117 classes, `total_mr` = 50 classes; CT dataset 1,228 subjects, MR dataset 616 subjects (MR DOI 10.5281/zenodo.11367004); code Apache-2.0; some subtasks (e.g. intracerebral hemorrhage, pleural/pericardial effusion, liver lesions, kidney cysts) are separate smaller datasets/models; dataset license not stated on README page — [GitHub wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- `meta.csv` holds train/val/test split — [train_nnunet.md](https://github.com/wasserth/TotalSegmentator/blob/master/resources/train_nnunet.md)
- Secondary source (search snippet): meta.csv columns image_id, age, gender, institute (code), study_type (examination range), split; v2 reportedly adds scanner and pathology metadata — [Awesome-Medical-Dataset mirror](https://github.yxlr123.workers.dev/mehrnia/Awesome-Medical-Dataset/blob/main/resources/TotalSegmentator_v2.md) (WebFetch of this URL later returned 404; snippet only, UNVERIFIED)
- HF mirror TotalSegmentator-CT-Lite (CC-BY-4.0, 23.4 GB, Images.zip 22.6 GB, Masks.zip 823 MB, meta.csv 113 kB) — [HF](https://huggingface.co/datasets/YongchengYAO/TotalSegmentator-CT-Lite/tree/main)

### Inferences
- No diagnosis labels means "mask-verified" questions are limited to anatomy-derived facts (organ volume, laterality, presence/absence, relative position, count of segmented structures) -- not pathology.
- v2 CT is 1,228 volumes; for ~16GB-GPU audit this is ~23.6 GB download, feasible with disk check.

### Gaps
- Exact meta.csv header (esp. scanner/pathology/image_description) UNVERIFIED: download the 102-subject subset or meta.csv to confirm.
- Zenodo concept-vs-version IDs: 6802613 (concept DOI, resolves to latest per search listing), 6802614 (v1.0), 10047292 (v2.0.1); a newer version exists per Zenodo note; its ID not retrieved.
- Zenodo page for MR record ID (11367004) not opened.

## Q3. Existing mask-derived 3D CT VQA / benchmarks (novelty of a voxel-mask benchmark)

### Takeaway
The generic idea is NOT novel: DeepTumorVQA (2025, extended 2026) and DeepChestVQA deterministically generate measurement/recognition questions from segmentation masks, and RadGPT converts masks to reports. What remains open (UNVERIFIED but not found): a benchmark built on TotalSegmentator's 117-class whole-body masks that audits M3D-LaMed specifically for organ-volume/size/location/count with seen/unseen-source control and blind (image-free) baselines.

### Cited Findings
- DeepTumorVQA (Chen/Zhou line): 9,262 CT volumes, 17 public datasets, 395K questions in 4 categories (Recognition, Measurement, Visual Reasoning, Medical Reasoning); benchmarked RadFM, M3D, Merlin, CT-CHAT — [arXiv 2505.18915](https://arxiv.org/abs/2505.18915)
- DeepTumorVQA revised/2026 version (2605.09679): 476K questions (428K train / 48K test), 42 subtypes; QA fully deterministic from "organ volumes, HU values, lesion counts computed from consensus segmentation masks covering 43 anatomical structures"; 10 templates per subtype; distractors 0.7x-1.3x truth; evaluates M3D-Llama2 and M3D-Phi3 plus RadFM, Merlin, MedGemma, Qwen3-VL, etc.; TotalSegmentator used only for "predicted"/tool-agent mode, not to produce benchmark volumes/masks — [arXiv HTML 2605.09679v1](https://arxiv.org/html/2605.09679v1)
- RadGPT / AbdomenAtlas 3.0: 9,262 CT-mask-report triplets (3,955 with tumors), reports from radiologist-revised tumor masks covering size, location, attenuation, resectability; no VQA in abstract — [arXiv 2501.04678](https://arxiv.org/abs/2501.04678)
- DeepChestVQA (in 3DMedAgent): 1,020 VQA pairs, 17 capability dimensions; masks partly from TotalSegmentator and VISTA3D; measurement questions excluded (search-result summary only) — [arXiv 2602.18064](https://arxiv.org/pdf/2602.18064)
- MIRP relative-position benchmark (2508.00549): tests relative-position questions in medical images; whether 3D or TotalSegmentator-based NOT confirmed — [arXiv 2508.00549](https://arxiv.org/abs/2508.00549)
- Contamination audit of 2D medical VLM benchmarks (not 3D/M3D): [arXiv 2606.10066](https://arxiv.org/abs/2606.10066)
- A search snippet claimed some paper builds 3D questions from TotalSegmentator masks; the paper could not be identified/confirmed (UNVERIFIED) — [search listing](https://arxiv.org/abs/2606.10066)
- Local prior notes: 3D-RAD (arXiv 2506.11147) is CT-RATE-derived, not mask-derived (from /home/mosharrof/medical_thesis/research_notes/Medical thesis project review/literature_and_novelty.md).

### Inferences
- Novelty claim should be narrowed: "mask-verified" benchmark itself is prior art (DeepTumorVQA); novelty would be (a) whole-body 117-class TotalSegmentator, (b) audit of M3D-LaMed seen-source contamination, (c) blind/ablation baselines. Cite DeepTumorVQA as closest and explain differences.
- DeepTumorVQA already shows M3D (Llama2 and Phi3) performs adequately on measurement, per abstract ("adequate on measurement, struggle on recognition/reasoning" is for the four-model set).

### Gaps
- SAT (Segment-and-Talk), MedVInT, OmniMedVQA, SegVol: not checked; MedVInT and OmniMedVQA are 2D (UNVERIFIED here).
- DeepTumorVQA full question list for organ-volume/count/location not read (HTML 275K chars; only first 100K read); GitHub (Schuture/DeepTumorVQA) not opened.
- No systematic search of Semantic Scholar citing-works; other 2025-26 papers may exist.

## Q4. Training-free inference methods and where they have been applied

### Takeaway
All named methods exist with arXiv IDs, but I found NO application to 3D medical VLMs (or M3D-LaMed). Medical 2D analogues exist (Med-VCD, ARCD); occlusion/image-ablation probing in 3D medical VQA not found. This is a viable gap but also means the methods are unvalidated for 3D.

### Cited Findings
- VCD (Leng et al.), contrast original vs distorted image: arXiv 2311.16922 — [arXiv](https://arxiv.org/pdf/2311.16922v1)
- PAI, amplify image attention and subtract text-only logits: arXiv 2407.21771 — [arXiv](https://arxiv.org/pdf/2407.21771)
- OPERA (Huang et al., CVPR 2024), over-trust penalty plus retrospection-allocation in beam search: arXiv 2311.17911 — [arXiv](https://arxiv.org/abs/2311.17911)
- Contextual calibration, "Calibrate Before Use" (Zhao et al. 2021, content-free input like "N/A"): arXiv 2102.09690 — [arXiv](https://arxiv.org/pdf/2102.09690)
- Delve into VCD (downsampling/edited contrastive inputs, entropy criterion): arXiv 2412.06775 — [arXiv](https://arxiv.org/html/2412.06775v1)
- Med-VCD: sparse visual contrastive decoding for medical LVLMs, 8 datasets (radiology, ophthalmology, pathology); 3D CT listed as future work — [arXiv 2512.01922 via listing](https://www.emergentmind.com/topics/med-vcd)
- ARCD, anatomical-region-guided contrastive decoding (AAAI 2026): training-free; tested on CXR, CT, brain MRI, ocular B-ultrasound; needs region masks; CT part is not volumetric-focused — [arXiv 2512.17189](https://arxiv.org/html/2512.17189v1)
- Zoom/crop-reread (general domain): RADAR (training-free attention-guided zoom, remote sensing) [arXiv 2603.02754](https://arxiv.org/pdf/2603.02754v1); uncertainty-guided re-attention with multi-scale crops (title not retrieved) [arXiv 2602.04304](https://www.arxiv.org/pdf/2602.04304); LASER query-aware localization [arXiv 2511.19820](https://arxiv.org/abs/2511.19820v1) (which paper maps to which ID is UNVERIFIED; the search listing did not pair them)
- Medical VQA language-bias work: Med-BiasX (MICCAI 2025) [paper](https://papers.miccai.org/miccai-2025/paper/5135_paper.pdf); post-hoc confidence calibration of medical VLMs, Byun et al. [arXiv 2604.02543](https://arxiv.org/abs/2604.02543)

### Inferences
- Applying VCD/PAI/contextual calibration to M3D-LaMed (3D tokens via Perceiver, 256 image tokens) would be new for 3D, but the method itself is prior art; claim "evaluation", not "method".
- Occlusion probing in 3D: M3D-LaMed uses ViT 3D patches so occlude by sub-volume masking; use mask-guided occlusion (organ masks from TotalSegmentator) as the novel link (ARCD uses masks for decoding in 2D).

### Gaps
- No paper found applying contextual calibration to medical VQA; none found for VCD/PAI/OPERA on any 3D VLM; none on occlusion probing for 3D medical VQA (single search each, US web search).
- OPERA and ARCD/Med-VCD details not read in full.

## Q5. Can M3D-LaMed run in 16 GB T4 VRAM?

### Takeaway
Likely yes with fp16 for Phi-3-4B (about 8 GB weights) and with fp16 Llama-2-7B (about 14 GB weights, tight) or 8-bit; but NO official memory figures exist, so this is arithmetic plus UNVERIFIED until tested. T4 lacks native bf16, so use fp16 (README lists float16 as allowed).

### Cited Findings
- README quickstart: dtype bfloat16/float16/float32, `device_map='auto'`, input 1x32x256x256 normalised 0-1, 256 image tokens; no memory or quantization notes — [GitHub BAAI-DCAI/M3D](https://github.com/BAAI-DCAI/M3D), [HF Llama-2-7B card](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Llama-2-7B)
- Phi-3-4B variant: 4B params, weights stored F32 on HF, no stated hardware requirement; README says Phi-3-4B outperformed Llama-2-7B in simple testing and the Llama-2-7B had segmentation problems (fixed) — [HF Phi-3-4B card](https://huggingface.co/GoodBaiBai88/M3D-LaMed-Phi-3-4B), [GitHub](https://github.com/BAAI-DCAI/M3D)
- Checkpoints are F32 on HF: Llama-2-7B about 28 GB and Phi-3-4B about 16 GB to download/load unless converted (card's own rough estimate) — same HF cards
- General quantization: bitsandbytes 8-bit/4-bit supported through `quantization_config` in `from_pretrained`; LLM.int8() roughly halves memory — [LLM.int8() NeurIPS 2022](https://nips.cc/virtual/2022/poster/53018)
- No source found showing M3D-LaMed run on T4 or in 8-bit; DeepTumorVQA evaluated M3D-Llama2/Phi3 (hardware not captured) — [arXiv 2605.09679](https://arxiv.org/html/2605.09679v1)

### Inferences
- Load with `torch_dtype=torch.float16`, `low_cpu_mem_usage=True` (host RAM must handle F32 shards, about 28 GB for 7B unless sharded loading) -- not verified.
- bitsandbytes 8-bit on custom `trust_remote_code` M3D classes is untested; the 3D ViT and Perceiver should stay in fp16 (skip modules).
- Start with Phi-3-4B fp16 for the pilot; 7B only if memory test passes.

### Gaps
- Real peak VRAM measurement not available; must be tested on the T4.
- Whether the repo's custom code supports 8-bit loading: UNVERIFIED.
