# Kaggle T4 Pilot Guide: Auditing M3D-LaMed (Phi-3-4B)

This guide explains how to run the Part B Pilot on a single **Kaggle Tesla T4 GPU (16 GB)** using our TotalSegmentator mask-verified benchmark pilot set (150 questions across 4 counterfactual conditions: Real, Blank, Swapped, Shuffled).

---

## 1. Preparing the Kaggle Dataset / Uploads

You only need to upload the `partB_benchmark/` folder (or just the `manifests/` and script) to Kaggle as a private Dataset or directly into your notebook.

Files needed:
* `kaggle_m3d_pilot.py`
* `manifests/pilot_real.json` (150 questions)
* `manifests/pilot_shuffled.json` (150 questions)
* `manifests/pilot_swapped.json` (150 questions)

---

## 2. Setting Up the Kaggle Notebook

1. **Create a new Kaggle Notebook**:
   * Navigate to [kaggle.com/code](https://www.kaggle.com/code) -> **New Notebook**.
2. **Configure Environment Settings** (Right Sidebar):
   * **Accelerator**: **GPU T4 x1** (16 GB VRAM).
   * **Internet**: **ON** (Required to download model weights from Hugging Face: `GoodBaiBai88/M3D-LaMed-Phi-3-4B`).
   * **Language**: Python.

---

## 3. Running the Pilot Script in Kaggle

In the notebook cells, run:

### Cell 1: Install Required Dependencies
```bash
!pip install -q -U transformers accelerate timm einops scipy pandas numpy
```

### Cell 2: Verify GPU Allocation & VRAM
```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
```

### Cell 3: Execute the M3D-LaMed Pilot
```bash
!python kaggle_m3d_pilot.py \
    --model_id "GoodBaiBai88/M3D-LaMed-Phi-3-4B" \
    --manifest_dir "manifests" \
    --output_dir "pilot_results"
```

*Expected Runtime*: **12 to 18 minutes** for 150 questions $\times$ 4 conditions ($600$ forward passes). Peak VRAM will sit around **~9.6 GB**, well within the 16 GB limit.

---

## 4. Output Artifacts & Download

The script saves three output files in `pilot_results/`:
1. `m3d_pilot_summary.csv`: Aggregated accuracy per family and condition, plus McNemar $p$-values.
2. `m3d_pilot_summary.json`: Detailed machine-readable metrics.
3. `m3d_pilot_per_item.csv`: Item-by-item predictions, gold answers, and log-probability shifts.

To download in Kaggle:
```python
from IPython.display import FileLink
FileLink("pilot_results/m3d_pilot_summary.csv")
```

---

## 5. Interpreting the Results & Scientific Decision Point

| Metric | Outcome A (Blind Model) | Outcome B (Visually Grounded Model) |
| :-- | :-- | :-- |
| **Real vs Blank Delta ($\Delta$)** | $\Delta \le 0.02$ ($p > 0.10$) | $\Delta > 0.05$ ($p < 0.05$) |
| **Swapped Patient Accuracy** | Equal to Real CT ($p = 1.0$) | Significantly lower than Real CT |
| **Log-Prob Shift ($\Delta \log p$)** | $\approx 0.005$ nats | $> 0.5$ nats |
| **Scientific Action** | Indicates visual insensitivity on this task set; proceed to pipeline sanity check and full benchmark audit. | Proceed to Phase 5: test training-free mitigations (Visual Contrastive Decoding, Zoomed Re-reading) if feasible. |
