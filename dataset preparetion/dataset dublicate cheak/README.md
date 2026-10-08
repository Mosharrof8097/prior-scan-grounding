# 3D Medical VQA Duplicate & Contamination Audit Pipeline (Kaggle Ready)

## 📌 Overview
This directory contains the multi-algorithm dataset audit script: [`check_duplicates_kaggle.py`](check_duplicates_kaggle.py).

It is designed to be executed directly inside your **Kaggle Notebook** on the generated `split_dataset` (Train, Val, Test). It inspects both the **3D CT volume images (`.npy`)** and the **question-answer text (`.csv`)** from multiple mathematical and algorithmic angles.

---

## 🔬 Implemented Duplicate & Contamination Algorithms

| Algorithm | Scope | Description & Mathematical Criteria |
| :--- | :--- | :--- |
| **Algo 1: Exact VQA Duplicate Detection** | Intra-split (Train, Val, Test) | Detects identical `(VolumeName, Question)` tuples. Audits for **conflicting labels** (where the same question on the same scan has two different answers). |
| **Algo 2: Cross-Split Contamination & Leakage** | Inter-split (Train vs Val vs Test) | Enforces strict independence: <br>1. $\text{Train Patients} \cap \text{Val Patients} = \emptyset$<br>2. $\text{Train VQA} \cap \text{Test VQA} = \emptyset$<br>3. $\text{Train Volumes} \cap \text{Val/Test Volumes} = \emptyset$ |
| **Algo 3: Normalized & Fuzzy Text Duplication** | Intra-split | Strips punctuation, lowercase casing, and whitespace normalization to catch near-duplicate questions asked on the same scan (e.g., *"Is there a lesion?"* vs *"is there a lesion ?"*). |
| **Algo 4: 3D CT Volume Image Hash (Bitwise MD5)** | Image scans (`.npy`) | Computes cryptographic MD5 hashes of all 3D image files to verify whether two different filenames actually represent the exact same 3D scan bit-for-bit, and ensures no duplicate scans cross split boundaries. |
| **Algo 5: Global Template & QA Redundancy** | Entire dataset | Quantifies the concentration of clinical template questions and answer distribution to identify high-frequency patterns and class imbalances. |

---

## 🚀 How to Run in Kaggle

In your Kaggle notebook (in a new code cell right below your split code):

```python
# Option A: Run as command
!python "/kaggle/working/check_duplicates_kaggle.py" \
    --data_dir "/kaggle/working/split_dataset" \
    --output_dir "/kaggle/working/duplicate_audit_results"
```

Or paste the contents of `check_duplicates_kaggle.py` directly into a Kaggle cell and click Run.

---

## 📊 Output
- **Console Log:** Real-time breakdown of all 5 algorithm checks with clear `[PASSED]` / `[FLAGGED]` statuses.
- **JSON Audit Report:** Automatically saved to `/kaggle/working/duplicate_audit_results/duplicate_audit_report.json` for inclusion in thesis methodology / appendix reports.
