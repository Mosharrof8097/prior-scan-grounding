# 3D-RAD Medical Multimodal VLM: Automated Split & Baseline Training Pipeline

## 📌 Overview
This directory contains the production-grade, end-to-end training pipeline: [`train_baseline_kaggle.py`](train_baseline_kaggle.py).

It takes your **raw 3D-RAD dataset**, automatically executes the **80/10/10 Patient-Level Split** (if not already split), builds the 3D PyTorch DataLoaders, and trains the **Baseline 3D Medical Multimodal Vision-Language Model** with **XAI Attention Hooks** to track temporal hallucinations and recency bias.

---

## 🔬 Scientific Architecture (Informed by Literature Review)

| Component | Technical Selection | Grounding from Literature Review |
| :--- | :--- | :--- |
| **Vision Backbone** | 2D Medical Slice Feature Extractor | Based on **Lian et al. (*Scientific Reports 2026*)**: Avoids training heavy 3D convolutions from scratch; extracts features from each of the 32 slices with high efficiency. |
| **Depth Aggregator** | Cross-Attention Slice Pooler | Aggregates the 32 slice features into concise 3D visual tokens (`num_tokens=16`). |
| **Multimodal Projector** | 2-Layer GeLU MLP | Maps visual token representations into the LLM embedding space. |
| **LLM Backbone** | `Qwen/Qwen2.5-1.5B-Instruct` (or `Llama-3.2-1B`) | Lightweight, high-accuracy instruction-following model that comfortably fits within Kaggle's 16 GB VRAM. |
| **PEFT Adapter** | LoRA (`r=16`, `alpha=32`) | Keeps the base LLM frozen, training only ~0.5%–1% parameters for fast convergence and zero OOM crashes. |
| **XAI Hook** | Recency-Bias Ratio Tracker ($T_1/T_0$) | Measures attention weight distribution across earlier slices ($T_0$) vs. later slices ($T_1$) to prove your thesis hypothesis on longitudinal degradation. |

---

## 🚀 How to Run on Kaggle

### Option A: Direct Notebook Command (Recommended)
Paste this in a Kaggle notebook cell:

```bash
!python "/kaggle/working/train_baseline_kaggle.py" \
    --data_dir "/kaggle/input/medi-data-3d" \
    --output_dir "/kaggle/working/baseline_training_output" \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum_steps 4 \
    --lr 2e-4
```

### Option B: Copy & Paste Code directly into a Notebook Cell
The script includes dynamic IPython kernel detection, so you can paste the entire contents of [`train_baseline_kaggle.py`](train_baseline_kaggle.py) into a single code cell and click **Run**.

---

## 📊 Detailed Per-Epoch Reporting

At the end of every epoch, the pipeline prints a detailed clinical report:

```text
======================================================================
  [VALIDATION REPORT - EPOCH 1]
  Overall Val Loss     : 1.4820 (Perplexity: 4.40)
  Single-Scan Task Loss: 1.2150 (Tasks 1-4)
  Longitudinal Loss    : 1.9430 (Tasks 5-6)
  XAI Recency-Bias (T1/T0): 1.642 (ALERT: Recency Bias Detected)
======================================================================
```

---

## 📦 Auto-Zip Packaging
Upon completion of training, the script automatically packages all outputs into:
`/kaggle/working/baseline_training_output/baseline_training_results.zip`

Contents of the zip file:
- `best_checkpoint/multimodal_projector_best.pt` (Best projector weights)
- `best_checkpoint/lora_adapters/` (Best LoRA adapter weights)
- `training_history_metrics.json` (Full epoch loss, perplexity, and XAI recency metrics)
- `split_dataset/` (Split CSV manifests and symlinked structures)

You can download this `.zip` directly with 1 click from Kaggle's Output panel.
