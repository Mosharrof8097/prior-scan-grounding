# Genuine 3D CT Counterfactual Audit Report: M3D-LaMed-Phi-3-4B

**Total Real CT Questions Evaluated:** 7,746
**Environment:** Kaggle Dual Tesla T4 GPUs | Zero-Dummy Protocol Enforced

## 1. Primary Empirical Findings (Table 1 for Paper)

| Task | N | Real CT Acc | Blank CT Acc | Swap Donor Acc | Slice-Shuf Acc | Text-Only Acc | Δ(Real - Blank) | McNemar p (vs Blank) | McNemar p (vs Swap) | Prediction Invariance (Swap) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **OVERALL** | **7,746** | **21.09%** | **23.43%** | **20.98%** | **21.56%** | **19.36%** | **-2.34%** | **5.69e-14** | **0.684** | **91.5%** |
| Task4_Existence_Detection | 2,000 | 41.35% | 50.00% | 40.90% | 43.10% | 31.65% | -8.65% | 1.57e-21 | 0.603 | 88.1% |
| Task5_Static_Temporal_Diagnosis | 2,873 | 13.47% | 13.26% | 13.54% | 13.44% | 14.76% | +0.21% | 6.17e-01 | 0.901 | 93.2% |
| Task6_Longitudinal_Temporal_Diagnosis | 2,873 | 14.62% | 15.11% | 14.55% | 14.69% | 15.42% | -0.49% | 2.75e-01 | 0.913 | 92.2% |

## 2. Invariance & Visual Grounding Breakdown

- **Real vs Swap Donor CT Agreement:** 91.51% identical clinical predictions across all 7,746 questions.
- **Real vs Slice-Shuffled CT Agreement:** 94.53% identical predictions when all 32 z-slices are permuted.
- **Real vs Blank Zero-Tensor Agreement:** 87.66% identical predictions when the CT scan is completely erased.
- **Visual Dependency Ratio (VDR):** -11.08% (Negative visual grounding: image voxels act as distracting noise).
