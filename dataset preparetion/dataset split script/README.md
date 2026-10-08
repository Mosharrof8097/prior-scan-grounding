# 3D-RAD Medical VQA Dataset Splitter (Kaggle & Local Pipeline)

## 📌 Overview
This directory contains the official scientific split pipeline for the **3D-RAD (NeurIPS 2025)** Medical Visual Question Answering (Med-VQA) dataset.

The splitter divides the dataset into **Train (80%)**, **Validation (10%)**, and **Test (10%)** sets following gold-standard medical imaging principles.

---

## 🔬 Core Algorithm & Methodology

### 1. The Problem with Naive / Random Splitting (Data Leakage)
In medical imaging, multiple questions and longitudinal studies exist for the same patient (e.g., patient `test_1016` has studies `test_1016_a`, `test_1016_b`, `test_1016_d`).
* **Random Split Flaw:** Random row splitting causes study $A$ of patient $X$ to fall in Train, while study $B$ falls in Test. The neural network memorizes patient $X$'s specific anatomy, producing an inflated, artificial 99% accuracy on Test that completely collapses in real hospital trials.
* **Our Solution (Patient-Level Grouping):** All scans, studies, and question-answer pairs belonging to the same patient ID (`PatientID = test_xxxx`) are strictly assigned to **only one partition** (either Train, Val, or Test).

### 2. Task Stratification
The dataset spans 6 clinical radiology tasks:
1. `Task 1: Image Observation`
2. `Task 2: Anomaly Detection`
3. `Task 3: Medical Computation`
4. `Task 4: Existence Detection`
5. `Task 5: Static Temporal Diagnosis`
6. `Task 6: Longitudinal Temporal Diagnosis`

The splitting algorithm balances the proportions of all 6 clinical tasks across Train, Val, and Test, ensuring model training and evaluation remain unbiased.

### 3. Leakage Verification Audit
The script performs an automated assertion check:
$$\text{Train Patients} \cap \text{Val Patients} = \emptyset$$
$$\text{Train Patients} \cap \text{Test Patients} = \emptyset$$
$$\text{Val Patients} \cap \text{Test Patients} = \emptyset$$
If any patient overlap is detected, the script raises an error and halts execution.

---

## 📁 Directory Structure & Output Format

The script organizes both the **3D CT Image Volumes (`.npy`)** and **QA Pairs (`.csv`)** into standard PyTorch/TensorFlow medical dataset folders:

```text
output_dir/
├── train/
│   ├── images/          # Symlinked 3D CT volumes (.npy) for Train
│   └── train_qa.csv     # Training QA pairs (80%)
├── val/
│   ├── images/          # Symlinked 3D CT volumes (.npy) for Val
│   └── val_qa.csv       # Validation QA pairs (10%)
└── test/
    ├── images/          # Symlinked 3D CT volumes (.npy) for Test
    └── test_qa.csv      # Testing QA pairs (10%)
```

> [!NOTE]
> **Kaggle Disk Optimization (Symlinks):** Kaggle limits `/kaggle/working` to **20 GB**. Copying 20 GB of 3D image files physically from `/kaggle/input` causes an out-of-disk error. This script creates symbolic links (`os.symlink`) to the images in `/kaggle/input`, taking **0 extra disk space** while giving your dataloader clean `train/images/`, `val/images/`, and `test/images/` paths.

---

## 🧪 Automated Post-Split Integrity Tests

Upon completion of the split, the script automatically executes 4 integrity and certification tests:

1. **Test 1: Zero Patient Leakage Test**  
   Asserts:
   $$\text{Train Patients} \cap \text{Val Patients} = \emptyset$$
   $$\text{Train Patients} \cap \text{Test Patients} = \emptyset$$
   $$\text{Val Patients} \cap \text{Test Patients} = \emptyset$$
   Guarantees no longitudinal multi-study overlap across partitions.

2. **Test 2: 3D Image File Existence Test**  
   Verifies that every single `.npy` file referenced in `train_qa.csv`, `val_qa.csv`, and `test_qa.csv` exists and is accessible in its respective `images/` directory.

3. **Test 3: 3D Array Voxel Sanity Check**  
   Loads sample 3D arrays into memory using `np.load()`, verifies array dimensionality (3D/4D volumetric shape: `(1, 32, 256, 256)`), data type, and min/max voxel intensity bounds.

4. **Test 4: Duplicate Record Test**  
   Audits every partition for duplicate `(VolumeName, Question)` pairs and reports counts.

---

## 🚀 How to Run in Kaggle

In your Kaggle Notebook:

```python
# Cell 1: Clone or copy split_dataset_kaggle.py into Kaggle
# Cell 2: Run the full split and validation
!python split_dataset_kaggle.py --data_dir /kaggle/input/<your-dataset-name> --output_dir /kaggle/working/split_dataset
```

The script will:
- Scan all `.npy` files and `.csv` files.
- Perform patient-level 80/10/10 split.
- Create `train/`, `val/`, `test/` with symlinked images and CSVs.
- Run and print all 4 certification tests.
