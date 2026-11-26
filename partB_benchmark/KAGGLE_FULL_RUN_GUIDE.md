# Kaggle T4 Execution Guide: Full M3D-LaMed Benchmark Audit

This guide provides the exact notebook setup and commands to run the Part B Audit on a **Kaggle Tesla T4 GPU (16 GB VRAM)**.

---

## 1. Kaggle Notebook Configuration

1. Go to [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook**.
2. In the right-hand panel (**Notebook options**):
   * **Accelerator**: Select **GPU T4 x1** (16 GB VRAM).
   * **Internet**: Turn **ON** (Required to download HuggingFace model weights: `GoodBaiBai88/M3D-LaMed-Phi-3-4B`).
   * **Language**: Python.

---

## 2. Uploading the Code Package

Upload the pre-packaged zip file:
- File path on local machine: `medical_thesis/partB_benchmark/kaggle_partB_audit.zip` (Size: ~583 KB).

In a notebook cell, extract the package:
```bash
!unzip -o kaggle_partB_audit.zip
```

---

## 3. Notebook Cells to Execute

### Cell 1: Install Dependencies
```bash
!pip install -q -U transformers accelerate timm einops scipy pandas numpy monai
```

### Cell 2: Verify GPU Allocation
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

---

### Step 1: Run Pipeline Sanity Check (~10–12 minutes)
Verifies vision encoder embeddings, runs 100 positive controls from M3D-VQA, and tests 20 free text generations:
```bash
!python sanity_check_m3d.py \
    --manifest manifests/pilot_real.json \
    --output_json sanity_check_results.json
```

---

### Step 2: Run 6-Condition Pilot Run (~12–15 minutes)
Runs all 6 counterfactual conditions (`real`, `blank`, `swap`, `shuffle_slices`, `shuffle_options`, `text_only`) on the 150 stratified questions:
```bash
!python kaggle_m3d_full.py \
    --pilot \
    --manifest_dir manifests \
    --image_dir /kaggle/input/totalseg_v2_volumes \
    --output_dir results_pilot
```

---

### Step 3: Run Full Benchmark Audit (~2 to 2.5 hours)
Evaluates all 7,465 questions across 1,228 cases across all 6 conditions, saving checkpoints every 500 items.
*(Note: `--image_dir` is required; the script asserts that all CT volumes are genuine non-dummy tensors).*
```bash
!python kaggle_m3d_full.py \
    --manifest_dir manifests \
    --image_dir /kaggle/input/totalseg_v2_volumes \
    --output_dir results_full \
    --checkpoint_step 500
```
*(Note: If disconnected, re-running the exact same command will automatically resume from the last 500-item checkpoint).*

---

## 4. Download Outputs

In a final cell, zip and download the results:
```python
import shutil
from IPython.display import FileLink

# Zip results
shutil.make_archive("m3d_audit_results", "zip", "results_full")

# Download link
FileLink("m3d_audit_results.zip")
```
