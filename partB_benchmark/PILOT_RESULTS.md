# Pilot Audit Results: M3D-LaMed-Phi-3-4B on TotalSegmentator v2 Benchmark

> [!CAUTION]
> **CRITICAL PREPROCESSING PIPELINE AUDIT FINDING:**
> Source code audit of `partB_benchmark/kaggle_m3d_pilot.py` (lines 285–296) confirms that the Kaggle T4 pilot evaluated dummy synthetic tensors:
> ```python
> default_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype)
> blank_vol = torch.zeros((1, 1, 32, 256, 256), dtype=dtype)
> donor_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype)
> ```
> Real CT NIfTI volumes were **never loaded or processed** during this pilot. The evaluated inputs were uniform random noise tensors $[0, 1)$ vs. all-zero tensors. Therefore, all pilot accuracy statistics below (Real 44.00%, Blank 42.67%, Swapped 42.67%) are **results from invalid dummy-volume input**, NOT genuine evidence of CT understanding or grounding.

**Evaluation Date:** 2026-11-25  
**Hardware Platform:** Kaggle T4 GPU (16 GB VRAM)  
**Execution Wall Time:** 156.6 seconds (~2.5 minutes)  
**Evaluated Model:** `GoodBaiBai88/M3D-LaMed-Phi-3-4B`  
**Data Subset:** 150 Stratified Questions across 134 distinct cases (TotalSegmentator v2 mask-verified benchmark)  
**Evaluated Conditions (Audited Input Identity):** 
1. `real`: Uniform random noise tensor `torch.rand((1, 1, 32, 256, 256))` (DUMMY NOISE, NOT REAL CT)
2. `blank`: All-zero volume tensor `torch.zeros((1, 1, 32, 256, 256))`
3. `swapped`: Independent random noise tensor `torch.rand((1, 1, 32, 256, 256))` (DUMMY NOISE, NOT DONOR SCAN)
4. `shuffled`: Option choices permuted in the prompt text and gold letter remapped (option position bias probe)

---

## 1. Master Results Table ($N = 150$) [INVALID DUMMY-VOLUME INPUT — NOT GENUINE CT EVIDENCE]

| Group / Category | $N$ | Real CT Acc | Blank CT Acc | Swapped CT Acc | Shuffled Options Acc | $\Delta_{\text{Blank}}$ ($\text{Real}-\text{Blank}$) | 95% CI of $\Delta_{\text{Blank}}$ | Discordant $(b, c)$ | McNemar $p_{\text{blank}}$ | McNemar $p_{\text{swap}}$ | Mean $\Delta \log p$ [95% CI] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Overall** | **150** | **0.4400** | **0.4267** | **0.4267** | **0.3800** | **+0.0133** | **[-0.0333, +0.0600]** | **(7, 5)** | **0.7744** | **0.5000** | **-0.3406 [-0.773, +0.079]** |
| Existence | 50 | 0.3800 | 0.4000 | 0.3800 | 0.4200 | -0.0200 | [-0.1000, +0.0400] | (1, 2) | 1.0000 | 1.0000 | -1.0592 [-2.219, +0.045] |
| Laterality & Volume | 40 | 0.3000 | 0.3250 | 0.2750 | 0.3250 | -0.0250 | [-0.0750, +0.0000] | (0, 1) | 1.0000 | 1.0000 | -0.2904 [-0.860, +0.284] |
| 3D Spatial Topography | 30 | 0.4333 | 0.2667 | 0.4333 | 0.3667 | +0.1667 | [+0.0000, +0.3333] | (6, 1) | 0.1250 | 1.0000 | +0.5839 [+0.052, +1.065] |
| Volume Ranking (4-MCQ) | 30 | 0.7333 | 0.7667 | 0.7000 | 0.4000 | -0.0333 | [-0.1000, +0.0000] | (0, 1) | 1.0000 | 1.0000 | -0.1344 [-0.482, +0.154] |
| Seen Domains (train/val) | 111 | 0.4595 | 0.4324 | 0.4414 | 0.4054 | +0.0270 | [-0.0180, +0.0721] | (5, 2) | 0.4531 | 0.5000 | -0.2902 [-0.796, +0.203] |
| Unseen Novel Domains | 39 | 0.3846 | 0.4103 | 0.3846 | 0.3077 | -0.0256 | [-0.1538, +0.0769] | (2, 3) | 1.0000 | 1.0000 | -0.4841 [-1.404, +0.283] |

*Notes on metrics:*
* Discordant pairs $(b, c)$: $b$ is the number of items correct under Real only; $c$ is correct under Blank only.
* 95% CIs are derived from 10,000 non-parametric bootstrap replications.

---

## 2. Key Empirical Findings & Nuances

1. **Overall Difference Not Statistically Distinguishable from Zero ($p = 0.7744$):**
   Overall accuracy with the real CT scan is 44.00% versus 42.67% when the volume is replaced by zeros (difference +1.33 pp, 95% CI [-3.33, +6.00] pp). McNemar's exact two-sided test yields $p = 0.7744$ (discordant pairs $b=7, c=5$). Real and blank predictions are identical on 137 of 150 items (91.33%).

2. **Patient Scan Swap Invariance:**
   In 3D Spatial Topography ($n=30$), pairing the questions with an unrelated donor patient's CT volume produces the exact same accuracy (43.33%) as pairing with the true patient scan ($\Delta_{\text{swap}} = 0.00\%$, McNemar $p = 1.0000$, discordant pairs 0 vs 0). Similarly, on unseen domains ($n=39$), swapped CT and real CT both yield exactly 38.46%. Real and swapped scan correctness are identical on 148 of 150 items (98.67%). While spatial topography exhibits an apparent signal over blank (+16.67 pp, 95% CI [0.00, +33.33] pp, $p=0.1250$), this signal is non-patient-specific.

3. **Log-Probability Distribution Shift:**
   The mean shift in gold-token log-probability between real and blank is $-0.3406$, but its 95% bootstrap CI is $[-0.773, +0.079]$ (crossing zero) with a median of $0.0000$. The negative mean is primarily driven by outliers in Existence items (mean $\Delta \log p = -1.0592$, median $-0.297$).

4. **Consistency with Language Priors and Option Ordering in Volume Ranking:**
   On 4-choice Volume Ranking ($n=30$), the model achieves 76.67% with no image input, compared to 73.33% with real CT. However, permuting the option choices in the prompt text reduces accuracy to 40.00%, demonstrating sensitivity to option letter order and position bias rather than visual structure.

5. **Distributional and Pipeline Caveats:**
   - **Label and Prediction Bias:** The pilot sample exhibits gold label imbalance (A=53, B=50, C=30, D=17) and model predictions are strongly biased toward option A (62 items) and B (55 items).
   - **Invalid Dummy Volume Input (Root Cause Identified):** The high agreement (91.33%) and indistinguishable performance between "real" and blank inputs in this pilot is directly explained by the fact that `kaggle_m3d_pilot.py` passed synthetic uniform noise `torch.rand((1, 1, 32, 256, 256))` instead of real CT scans. Genuine visual understanding of CT anatomy cannot be evaluated from these pilot numbers. A live GPU sanity check (`partB_benchmark/sanity_check_m3d.py`) and evaluation with real preprocessed NIfTI/NumPy volumes remain mandatory before drawing substantive conclusions.
