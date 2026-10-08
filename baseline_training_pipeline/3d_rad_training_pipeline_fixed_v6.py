#!/usr/bin/env python3
"""
====================================================================================================
3D-RAD Medical Multimodal VLM: Baseline Training Pipeline — FIXED v6
====================================================================================================
Carried forward UNCHANGED (confirmed correct):
  - Patient-level 80/10/10 split with leakage assertion.
  - [FIX 5] Longitudinal T0/T1 pairing (diagnostic + filename-letter hypothesis).
  - [FIX 6] PEFT / LoRA availability check that fails loudly instead of silently.
  - [FIX 7] Multi-target torchao dispatcher patch (confirmed working — LoRA now applies:
            "trainable params: 4,358,144 || all params: 1,548,072,448").

Fixed in this version (v5), based on your last run's traceback — training now starts and reaches
the first forward/backward pass, then hits:

    OutOfMemoryError: CUDA out of memory. Tried to allocate 334.00 MiB. GPU 0 has a total
    capacity of 14.56 GiB ... 13.90 GiB is allocated by PyTorch ...

  [FIX 8] T4 (14.56 GiB usable) is too small to hold, in one go: the frozen Qwen2.5-1.5B weights
          (~3 GB in fp16) PLUS the forward activations of all 28 transformer layers kept around for
          backprop — even though the LLM's own weights are frozen, `combined_embeds` (built from
          the trainable visual projector) requires grad, so autograd must still save every layer's
          intermediate activations to backprop into the projector/LoRA. That's the actual memory
          hog, not the LoRA adapters themselves (only 4.3M trainable params). This is a best-effort,
          HONESTLY LABELLED hypothesis based on where the OOM fires (first forward pass, not later);
          if these changes don't fully fix it, the diagnostic prints added below will show exactly
          how much memory is used right after model construction, before any batch runs, so we can
          narrow it down further.

          Changes made, from most to least impactful:
            (a) Gradient checkpointing enabled on the base LLM (`gradient_checkpointing_enable`),
                which recomputes activations during backward instead of storing all of them —
                typically the single biggest activation-memory win for a frozen-backbone + LoRA
                setup like this. `use_cache` is also forced to False during training (required for
                checkpointing to work, and avoids the model additionally caching K/V that training
                never uses).
            (b) `--batch_size` default lowered 4 -> 2, `--grad_accum_steps` default raised 4 -> 8,
                so the EFFECTIVE batch size stays 16 (unchanged) but the largest tensor materialized
                at once is half the size. Override both on the command line if you want to tune this
                yourself once the OOM is gone.
            (c) `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (only if not already set)
                before `torch` is imported, to reduce allocator fragmentation — this is the exact
                setting the original error message suggested.
            (d) `torch.cuda.empty_cache()` is called after each optimizer step and after validation,
                to release cached-but-unused blocks between steps.
            (e) The train/val loops now catch `torch.cuda.OutOfMemoryError` PER BATCH: on OOM they
                log a warning, clear gradients and the CUDA cache, skip that one batch, and continue
                — instead of the previous behaviour where one bad batch crashed the entire epoch and
                lost all progress. This is a safety net, not a fix for the root cause — if you see
                many skipped batches, (a)/(b) above need to do more (e.g. drop batch_size to 1, or
                shorten `--max_seq_len` / reduce `num_visual_tokens`).

          If it STILL OOMs after this, the next things to try, roughly in order of ease: pass
          `--batch_size 1`, reduce `max_seq_len` in `Medical3DVQADataset` from 128, or reduce
          `num_visual_tokens` from 16 in `Medical3DVLMBaseline`. All three shrink the sequence length
          the frozen 1.5B model has to backprop through, which is the actual scarce resource on a
          14.56 GiB T4.

Fixed in this version (v6), based on your last run's traceback: the OOM in v5 happened while loading
the very FIRST parameter of a brand-new model (1/338, `model.embed_tokens.weight`), with 14.24 GiB
already in use before that load even completed. A 1.5B model's embedding tensor is at most a few
hundred MB — so ~14 GiB was already occupied before this attempt started at all. FIX 8's training-loop
changes (gradient checkpointing, smaller batch, OOM-safe skipping) never even got a chance to run.

  [FIX 9] This is almost certainly leftover GPU memory from a PREVIOUS run in the SAME Kaggle
          kernel/session that was never released — a crashed cell's `model`, `base_llm`, optimizer,
          and batch tensors stay referenced (IPython's exception traceback keeps stack frames alive
          too), so PyTorch can't free that memory even though the earlier cell already errored out.
          `torch.cuda.empty_cache()` cannot fix this — it only releases PyTorch's own *unused cached*
          blocks, not memory still held by live references. This is a Kaggle session-state issue, not
          something fixable purely from inside this script.

          What this version adds: `check_gpu_memory_headroom_or_raise()` runs right before the model
          load and reports free/allocated/reserved GPU memory up front. If there isn't enough headroom
          (default: at least 5 GiB free), it raises IMMEDIATELY with a clear, actionable message —
          "restart the Kaggle kernel/session, then re-run fresh" — instead of a confusing OOM 30 stack
          frames deep inside `transformers` internals.

          ACTION NEEDED FROM YOU: in Kaggle, use the Run menu -> "Restart session" (or "Factory
          reset"), THEN run this script fresh. Re-running the same cell in place, or re-running only
          the training cell without restarting, will NOT free the leaked memory — you'll hit the same
          wall again. This isn't optional cleanup; nothing in FIX 8 (or any script-level fix) can
          reclaim GPU memory that Kaggle's kernel is still holding a live reference to.

====================================================================================================
"""

import os

# [FIX 8c] Must be set before `torch` is imported (first CUDA context init reads this env var).
# Reduces allocator fragmentation; this is the exact setting the OOM error message suggested.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import glob
import time
import json
import math
import shutil
import zipfile
import argparse
import importlib.util
import warnings
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ==================================================================================================
# 0. [FIX 6] PEFT AVAILABILITY CHECK — FAILS LOUDLY, NOT SILENTLY
# ==================================================================================================
def require_peft_or_raise(allow_no_lora: bool = False):
    spec = importlib.util.find_spec("peft")
    if spec is not None:
        import peft
        print(f"[CHECK] peft is installed (version {getattr(peft, '__version__', 'unknown')}). LoRA will be applied.")
        return True

    msg = (
        "\n" + "!" * 70 + "\n"
        "  [FATAL] 'peft' is NOT installed. LoRA will NOT be applied.\n"
        "  Your previous run silently fell back to training ONLY the projector\n"
        "  (0.29% of parameters) — that is not the experiment you planned.\n"
        "  Fix: run this in a cell BEFORE this script:\n"
        "      !pip install -q -U peft\n"
        "      !python -c \"import peft; print(peft.__version__)\"\n"
        "  Then re-run training.\n"
        + "!" * 70 + "\n"
    )
    print(msg)
    if not allow_no_lora:
        raise RuntimeError("peft not installed. Install it or pass --allow_no_lora to proceed without LoRA on purpose.")
    print("[WARN] --allow_no_lora was set: proceeding WITHOUT LoRA on purpose. This is not the planned experiment.")
    return False


# ==================================================================================================
# 0b. [FIX 7] TORCHAO DISPATCHER BYPASS — avoids peft's version-check ImportError
# ==================================================================================================
def patch_peft_torchao_check():
    """
    peft's LoRA dispatcher chain calls `is_torchao_available()` for every target module, even when
    the model has no torchao quantized layers at all. On environments with an old torchao (e.g.
    Kaggle's default 0.10.0), that check raises ImportError instead of returning False, which
    crashes get_peft_model() entirely.

    This model never uses torchao (base LLM is loaded as plain bf16/fp16), so it is safe to force
    that check to report "not available" and let peft fall through to the standard Linear-LoRA
    dispatcher.

    NOTE (learned from the v3 run): `peft.tuners.lora.torchao` does
    `from peft.import_utils import is_torchao_available` — a *value* import that creates its own
    local name bound to the original function object. Patching only `peft.import_utils` therefore
    does NOT affect the copy that `dispatch_torchao()` actually calls, so the same ImportError still
    fired. This version patches BOTH the origin module and every module we know imported a local
    copy of the name, so whichever one `dispatch_torchao` reads from is already neutralized.

    Returns True if at least one patch target succeeded, False if none did (in which case
    get_peft_model may still raise the torchao ImportError and you should upgrade torchao instead:
    `pip install -U torchao` then restart the kernel).
    """
    patched_any = False

    try:
        import peft.import_utils as _peft_import_utils
        _peft_import_utils.is_torchao_available = lambda: False
        print("[FIX 7] Patched peft.import_utils.is_torchao_available -> False")
        patched_any = True
    except Exception as e:
        print(f"[FIX 7][WARN] Could not patch peft.import_utils.is_torchao_available ({e})")

    try:
        import peft.tuners.lora.torchao as _peft_lora_torchao
        _peft_lora_torchao.is_torchao_available = lambda: False
        print("[FIX 7] Patched peft.tuners.lora.torchao.is_torchao_available -> False "
              "(this is the local copy dispatch_torchao() actually calls)")
        patched_any = True
    except Exception as e:
        print(f"[FIX 7][WARN] Could not patch peft.tuners.lora.torchao.is_torchao_available ({e})")

    # Belt-and-braces: also neutralize dispatch_torchao itself so it always returns None (no match),
    # regardless of how/whether it re-checks availability internally.
    try:
        import peft.tuners.lora.torchao as _peft_lora_torchao
        _peft_lora_torchao.dispatch_torchao = lambda target, adapter_name, config=None, **kwargs: None
        print("[FIX 7] Patched peft.tuners.lora.torchao.dispatch_torchao -> always returns None "
              "(forces peft to fall through to the standard Linear-LoRA dispatcher).")
        patched_any = True
    except Exception as e:
        print(f"[FIX 7][WARN] Could not patch dispatch_torchao ({e})")

    if not patched_any:
        print("[FIX 7][WARN] No torchao patch succeeded; get_peft_model may still raise the "
              "torchao ImportError. Fallback: `pip install -U torchao` then restart the kernel.")
    return patched_any


# ==================================================================================================
# 0c. [FIX 9] GPU MEMORY HEADROOM CHECK — catches "leftover memory from a previous run" early
# ==================================================================================================
def check_gpu_memory_headroom_or_raise(model_id: str, min_free_gib: float = 5.0):
    """
    The v5 OOM did NOT happen inside our training loop — it happened while loading the very FIRST
    parameter of a fresh model (`model.embed_tokens.weight`, 1/338), and the traceback already
    showed 14.24 GiB in use out of 14.56 GiB BEFORE that load even finished. A 1.5B model's first
    embedding tensor is at most a few hundred MB, so something else had already consumed ~14 GiB
    before this attempt even started.

    On Kaggle, this is almost always the SAME kernel/session being reused across failed runs without
    a restart: the previous attempt's `model`, `base_llm`, optimizer, and batch tensors are still
    referenced (often by the exception traceback itself, which IPython keeps around), so PyTorch
    never releases that GPU memory even though the earlier cell raised an error. This is a Kaggle
    session-state issue, not something this script can fix from the inside — `torch.cuda.empty_cache()`
    only releases PyTorch's own *unused cached* blocks, not memory still held by live references.

    This check makes that failure mode loud and actionable INSTEAD of a confusing OOM 30 stack frames
    deep inside `transformers` internals: it reports current allocated/reserved/free memory up front,
    and raises immediately with the fix (restart the kernel) if there isn't enough headroom to even
    attempt loading the base LLM.
    """
    if not torch.cuda.is_available():
        return
    torch.cuda.empty_cache()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / (1024 ** 3)
    total_gib = total_bytes / (1024 ** 3)
    allocated_gib = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved_gib = torch.cuda.memory_reserved() / (1024 ** 3)
    print(f"[FIX 9] GPU memory before model load: {free_gib:.2f} GiB free / {total_gib:.2f} GiB total "
          f"(this process: {allocated_gib:.2f} GiB allocated, {reserved_gib:.2f} GiB reserved).")
    if free_gib < min_free_gib:
        raise RuntimeError(
            "\n" + "!" * 70 + "\n"
            f"  [FIX 9][FATAL] Only {free_gib:.2f} GiB free out of {total_gib:.2f} GiB total GPU memory\n"
            f"  — not enough headroom to safely load {model_id} plus training state.\n"
            "  This is almost certainly leftover memory from a PREVIOUS run in this same Kaggle\n"
            "  kernel/session (a crashed attempt's model/optimizer/tensors are still referenced and\n"
            "  were never released) — NOT a sign that the GPU itself is too small.\n"
            "  Fix: Kaggle menu -> Run -> 'Restart session' (or 'Factory reset'), THEN re-run this\n"
            "  script fresh, without re-running any earlier cells that built a model first.\n"
            "  Do not just re-run this cell in place — that reuses the same leaked memory.\n"
            + "!" * 70 + "\n"
        )



# ==================================================================================================
# 1. [FIX 5] LONGITUDINAL STRUCTURE DIAGNOSTIC — RUN BEFORE TRUSTING ANY PAIRING
# ==================================================================================================
FILENAME_PATTERN = re.compile(r"^(?P<split>[a-zA-Z]+)_(?P<patient>\d+)_(?P<letter>[a-zA-Z]+)_(?P<recon>\d+)\.npy$")


def parse_filename(fname: str) -> Optional[Dict]:
    m = FILENAME_PATTERN.match(fname)
    if not m:
        return None
    return m.groupdict()


def diagnose_longitudinal_structure(data_dir: str, split_csv: str, n_show: int = 6):
    """
    Prints, without assuming anything:
      1. All CSV columns (to check for an explicit paired-volume reference we might be missing).
      2. A few raw longitudinal question texts (sometimes the prior scan is only referenced in text).
      3. Whether filenames follow the <split>_<patient>_<letter>_<recon>.npy pattern, and how many
         DISTINCT letters (candidate timepoints) exist per patient on average.
    Nothing here is used to silently decide anything — it is printed for you to read and confirm.
    """
    print("\n" + "=" * 70)
    print("  [FIX 5] LONGITUDINAL STRUCTURE DIAGNOSTIC")
    print("=" * 70)

    df = pd.read_csv(split_csv)
    print(f"  CSV columns found: {list(df.columns)}")

    candidate_cols = [c for c in df.columns if re.search(
        r"prior|previous|compare|second|other|pair|baseline|follow", c, re.IGNORECASE
    )]
    if candidate_cols:
        print(f"  [NOTE] Columns that MIGHT reference a second/paired volume: {candidate_cols}")
        print("         If one of these holds a second volume filename, the pairing logic below")
        print("         should use it directly instead of the filename-letter hypothesis.")
    else:
        print("  [NOTE] No column name suggests an explicit paired-volume reference.")
        print("         Falling back to the filename-letter hypothesis (see below).")

    if "TaskCategory" in df.columns:
        long_mask = df["TaskCategory"].astype(str).str.lower().str.contains(
            "longitudinal|temporal|task 5|task 6", regex=True
        )
        long_rows = df[long_mask]
        print(f"\n  Sample of {min(n_show, len(long_rows))} longitudinal question texts "
              f"(checking whether prior-scan info is embedded in TEXT, not filename):")
        for _, row in long_rows.head(n_show).iterrows():
            q = str(row.get("Question", ""))[:140]
            print(f"    - {q}")

    # Filename-letter hypothesis check
    if "NpyName" not in df.columns and "VolumeName" in df.columns:
        df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))

    parsed = df["NpyName"].dropna().unique()
    ok_count, bad_count = 0, 0
    patient_letters = defaultdict(set)
    for f in parsed:
        p = parse_filename(f)
        if p is None:
            bad_count += 1
            continue
        ok_count += 1
        patient_letters[f"{p['split']}_{p['patient']}"].add(p["letter"])

    print(f"\n  Filename pattern match: {ok_count} matched / {bad_count} did NOT match "
          f"'<split>_<patient>_<letter>_<recon>.npy'")
    if ok_count > 0:
        multi = {pid: sorted(letters) for pid, letters in patient_letters.items() if len(letters) > 1}
        pct_multi = 100.0 * len(multi) / max(len(patient_letters), 1)
        print(f"  Patients with >1 distinct study-letter (candidate multi-timepoint patients): "
              f"{len(multi)} / {len(patient_letters)} ({pct_multi:.1f}%)")
        sample_multi = list(multi.items())[:n_show]
        for pid, letters in sample_multi:
            print(f"    - {pid}: letters {letters}")
        print("\n  [INTERPRETATION]")
        if pct_multi > 30:
            print("    A meaningful fraction of patients have multiple study letters. This is")
            print("    CONSISTENT WITH the hypothesis that letters encode different scan dates")
            print("    (CT-RATE-style convention). The T0/T1 pairing below will use this.")
            print("    Still: confirm this is right for YOUR copy of 3D-RAD before trusting results —")
            print("    check the 3D-RAD documentation/paper for the exact filename convention if unsure.")
        else:
            print("    Few patients have multiple study letters. The letter-based pairing hypothesis")
            print("    may NOT hold for this data. Do not trust the T0/T1 pairing without checking")
            print("    the 3D-RAD dataset documentation for how longitudinal pairs are actually specified.")
    print("=" * 70 + "\n")
    return patient_letters


# ==================================================================================================
# 2. DYNAMIC ENVIRONMENT & PATH DISCOVERY (unchanged)
# ==================================================================================================
def resolve_dataset_paths(custom_data_dir: Optional[str] = None, custom_output_dir: Optional[str] = None):
    if custom_data_dir and os.path.exists(custom_data_dir):
        data_dir = custom_data_dir
    else:
        kaggle_img_matches = glob.glob("/kaggle/input/**/Images", recursive=True)
        if kaggle_img_matches:
            data_dir = os.path.dirname(kaggle_img_matches[0])
        else:
            kaggle_any = glob.glob("/kaggle/input/*")
            if kaggle_any:
                data_dir = kaggle_any[0]
            else:
                repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                local_candidate = os.path.join(repo_root, "3D-RAD_Dataset")
                data_dir = local_candidate if os.path.exists(local_candidate) else os.getcwd()

    if custom_output_dir:
        output_dir = custom_output_dir
    elif os.path.exists("/kaggle/working"):
        output_dir = "/kaggle/working/baseline_training_output"
    else:
        output_dir = os.path.join(os.getcwd(), "baseline_training_output")

    return data_dir, output_dir


# ==================================================================================================
# 3. PATIENT-LEVEL 80/10/10 DATASET SPLITTER — UNCHANGED, CONFIRMED CORRECT
# ==================================================================================================
def ensure_dataset_split(data_dir: str, split_dir: str, train_ratio: float = 0.80,
                         val_ratio: float = 0.10, test_ratio: float = 0.10, seed: int = 42):
    train_csv = os.path.join(split_dir, "train", "train_qa.csv")
    val_csv = os.path.join(split_dir, "val", "val_qa.csv")
    test_csv = os.path.join(split_dir, "test", "test_qa.csv")

    if os.path.exists(train_csv) and os.path.exists(val_csv) and os.path.exists(test_csv):
        print(f"[SPLIT] Verified split dataset already exists at: {split_dir}")
        return split_dir

    print("\n" + "=" * 70)
    print("  [STEP 1/6] EXECUTING PATIENT-LEVEL 80/10/10 SPLIT PIPELINE")
    print("=" * 70)
    print(f"Scanning raw dataset at: {data_dir}")

    image_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    image_map = {os.path.basename(p): p for p in image_paths}
    print(f"  -> Discovered {len(image_map):,} unique 3D CT volume files (.npy)")

    if not image_map:
        raise FileNotFoundError(f"No .npy 3D image files found under: {data_dir}")

    csv_paths = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    dfs = []
    for f in csv_paths:
        try:
            d = pd.read_csv(f)
            if "VolumeName" in d.columns and "Question" in d.columns:
                d["TaskCategory"] = os.path.basename(os.path.dirname(f))
                dfs.append(d)
        except Exception:
            continue

    if not dfs:
        raise FileNotFoundError(f"No valid QA CSV files discovered under: {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))
    df["PatientID"] = df["NpyName"].apply(lambda x: "_".join(str(x).split("_")[:2]))
    df["ImagePath"] = df["NpyName"].map(image_map)

    valid_mask = df["ImagePath"].notna()
    df = df[valid_mask].copy()
    print(f"  -> Total matched QA pairs: {len(df):,} across {df['PatientID'].nunique():,} unique patients")

    unique_patients = np.array(list(df["PatientID"].unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)

    n_tot = len(unique_patients)
    n_train = int(n_tot * train_ratio)
    n_val = int(n_tot * val_ratio)

    p_train = set(unique_patients[:n_train])
    p_val = set(unique_patients[n_train:n_train + n_val])
    p_test = set(unique_patients[n_train + n_val:])

    assert len(p_train & p_val) == 0, "Patient leakage between Train and Val!"
    assert len(p_train & p_test) == 0, "Patient leakage between Train and Test!"
    assert len(p_val & p_test) == 0, "Patient leakage between Val and Test!"
    print("  -> [AUDIT PASSED] Zero patient leakage confirmed across all sets.")

    def assign_split(pid):
        if pid in p_train:
            return "train"
        if pid in p_val:
            return "val"
        return "test"

    df["Split"] = df["PatientID"].map(assign_split)

    os.makedirs(split_dir, exist_ok=True)
    for s in ["train", "val", "test"]:
        s_dir = os.path.join(split_dir, s)
        img_out = os.path.join(s_dir, "images")
        os.makedirs(img_out, exist_ok=True)

        sub_df = df[df["Split"] == s].copy()
        linked = 0
        for src in sub_df["ImagePath"].unique():
            dst = os.path.join(img_out, os.path.basename(src))
            if not os.path.exists(dst):
                try:
                    os.symlink(src, dst)
                    linked += 1
                except OSError:
                    shutil.copy2(src, dst)
                    linked += 1

        csv_out = os.path.join(s_dir, f"{s}_qa.csv")
        sub_df.to_csv(csv_out, index=False)
        print(f"  [{s.upper():5}] {len(sub_df):,} QA pairs | {linked:,} 3D scans linked | Saved: {csv_out}")

    return split_dir


# ==================================================================================================
# 4. [FIX 5] PATIENT SCAN TIMELINE — for T0/T1 pairing under the filename-letter hypothesis
# ==================================================================================================
def build_patient_scan_timeline(data_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    """
    Returns {patient_key: [(letter, full_path), ...]} sorted alphabetically by letter (earliest first).
    patient_key = "<split>_<patient>", e.g. "test_482".
    """
    all_npy = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    timeline = defaultdict(list)
    for p in all_npy:
        info = parse_filename(os.path.basename(p))
        if info is None:
            continue
        key = f"{info['split']}_{info['patient']}"
        timeline[key].append((info["letter"], p))
    for key in timeline:
        timeline[key] = sorted(set(timeline[key]), key=lambda x: x[0])
    return timeline


# ==================================================================================================
# 5. 3D MEDICAL DATASET & AUGMENTATION PIPELINE
# ==================================================================================================
class Medical3DAugmenter:
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        if volume.ndim == 3:
            volume = np.expand_dims(volume, axis=0)
        if not self.is_train:
            return torch.from_numpy(volume).float()
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=3).copy()
        if np.random.rand() > 0.5:
            volume = np.flip(volume, axis=2).copy()
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.95, 1.05)
            volume = np.clip(volume * scale, 0.0, 1.0)
        return torch.from_numpy(volume).float()


def _resample_depth(volume_dhw: np.ndarray, target_depth: int) -> np.ndarray:
    """volume_dhw: (D, H, W). Uniformly subsamples along D to target_depth slices."""
    d = volume_dhw.shape[0]
    if d == target_depth:
        return volume_dhw
    idx = np.linspace(0, d - 1, target_depth).round().astype(int)
    return volume_dhw[idx]


class Medical3DVQADataset(Dataset):
    """
    [FIX 5] For longitudinal rows, attempts to load BOTH the referenced scan (T1) and its nearest
    earlier same-patient scan (T0, via the filename-letter timeline), resamples each to 16 depth
    slices, and concatenates them into a 32-slice volume (T0: slices 0-15, T1: slices 16-31) — so
    the existing recency-bias split (0:16 vs 16:32) becomes meaningful again.

    If no earlier scan exists for a longitudinal row (e.g. it's the patient's first/only study),
    this falls back to duplicating the single scan for both halves and marks it in
    `pairing_status` as "fallback_duplicate" — such items should be EXCLUDED or flagged separately
    in any recency-bias analysis, since they carry no real temporal signal.
    """
    def __init__(self, csv_file: str, img_dir: str, tokenizer, patient_timeline: Dict,
                data_dir: str, max_seq_len: int = 128, is_train: bool = True):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.augmenter = Medical3DAugmenter(is_train=is_train)
        self.patient_timeline = patient_timeline
        self.data_dir = data_dir

        if "NpyName" not in self.df.columns and "VolumeName" in self.df.columns:
            self.df["NpyName"] = self.df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))

        self.pairing_fallback_count = 0
        self.pairing_paired_count = 0

    def __len__(self):
        return len(self.df)

    def _load_single_volume(self, npy_name: str) -> np.ndarray:
        img_path = os.path.join(self.img_dir, npy_name)
        if not os.path.exists(img_path):
            matches = glob.glob(os.path.join(os.path.dirname(self.img_dir), "**", npy_name), recursive=True)
            if not matches:
                matches = glob.glob(os.path.join(self.data_dir, "**", npy_name), recursive=True)
            img_path = matches[0] if matches else img_path
        arr = np.load(img_path).astype(np.float32)  # expected (1, 32, 256, 256) or (32, 256, 256)
        if arr.ndim == 4:
            arr = arr[0]  # drop channel -> (32, 256, 256)
        return arr

    def _load_longitudinal_pair(self, npy_name: str) -> Tuple[np.ndarray, str]:
        info = parse_filename(npy_name)
        t1_arr = self._load_single_volume(npy_name)  # (32, 256, 256)

        if info is None:
            self.pairing_fallback_count += 1
            t0_16 = _resample_depth(t1_arr, 16)
            t1_16 = _resample_depth(t1_arr, 16)
            return np.concatenate([t0_16, t1_16], axis=0), "fallback_unparsed_filename"

        patient_key = f"{info['split']}_{info['patient']}"
        timeline = self.patient_timeline.get(patient_key, [])
        letters_before = [ltr for ltr, _ in timeline if ltr < info["letter"]]

        if not letters_before:
            self.pairing_fallback_count += 1
            t0_16 = _resample_depth(t1_arr, 16)
            t1_16 = _resample_depth(t1_arr, 16)
            return np.concatenate([t0_16, t1_16], axis=0), "fallback_duplicate_no_earlier_scan"

        t0_letter = letters_before[-1]  # nearest earlier study
        t0_path = dict(timeline)[t0_letter]
        t0_arr = np.load(t0_path).astype(np.float32)
        if t0_arr.ndim == 4:
            t0_arr = t0_arr[0]

        self.pairing_paired_count += 1
        t0_16 = _resample_depth(t0_arr, 16)
        t1_16 = _resample_depth(t1_arr, 16)
        return np.concatenate([t0_16, t1_16], axis=0), "paired_ok"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        npy_name = row["NpyName"]
        question = str(row["Question"])
        answer = str(row["Answer"])
        task_category = str(row.get("TaskCategory", "General"))
        is_longitudinal = any(k in task_category.lower() for k in ["longitudinal", "temporal", "task 5", "task 6"])

        if is_longitudinal:
            volume_dhw, pairing_status = self._load_longitudinal_pair(npy_name)
        else:
            volume_dhw = self._load_single_volume(npy_name)
            pairing_status = "not_applicable_single_scan"

        volume_arr = np.expand_dims(volume_dhw, axis=0)  # (1, 32, 256, 256)
        volume_tensor = self.augmenter(volume_arr)

        prompt_text = f"Clinical Question: {question}\nDiagnostic Answer: "
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=True)
        answer_ids = self.tokenizer.encode(answer + self.tokenizer.eos_token, add_special_tokens=False)

        min_ans_len = min(len(answer_ids), 32)
        max_prompt_len = self.max_seq_len - min_ans_len
        prompt_ids = prompt_ids[:max_prompt_len]

        combined_ids = prompt_ids + answer_ids
        combined_ids = combined_ids[:self.max_seq_len]
        seq_len = len(combined_ids)

        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((self.max_seq_len,), pad_id, dtype=torch.long)
        input_ids[:seq_len] = torch.tensor(combined_ids, dtype=torch.long)

        attention_mask = torch.zeros(self.max_seq_len, dtype=torch.long)
        attention_mask[:seq_len] = 1

        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        ans_start = len(prompt_ids)
        if ans_start < seq_len:
            labels[ans_start:seq_len] = input_ids[ans_start:seq_len]
        else:
            labels[max(0, seq_len - 1)] = input_ids[max(0, seq_len - 1)]
            ans_start = max(0, seq_len - 1)

        return {
            "image": volume_tensor,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "ans_start": ans_start,
            "is_longitudinal": is_longitudinal,
            "pairing_status": pairing_status,   # [FIX 5] audit trail per item
            "task_category": task_category,
            "question_text": question,
            "answer_text": answer
        }


# ==================================================================================================
# 6. PRETRAINED SLICE ENCODER (unchanged from previous fixed version)
# ==================================================================================================
def load_pretrained_slice_encoder(embed_dim: int = 512, freeze: bool = True) -> nn.Module:
    import torchvision.models as tv_models
    resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
    old_conv = resnet.conv1
    new_conv = nn.Conv2d(1, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                          stride=old_conv.stride, padding=old_conv.padding, bias=False)
    with torch.no_grad():
        new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
    resnet.conv1 = new_conv
    modules = list(resnet.children())[:-1]
    backbone = nn.Sequential(*modules)
    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
        backbone.eval()
    projector_head = nn.Sequential(nn.Flatten(), nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim))
    full_encoder = nn.Sequential(backbone, projector_head)
    full_encoder._is_frozen_backbone = freeze
    return full_encoder


# ==================================================================================================
# 7. 3D MEDICAL VLM ARCHITECTURE WITH XAI ATTENTION TRACKING (unchanged)
# ==================================================================================================
class SliceDepthAggregator(nn.Module):
    def __init__(self, embed_dim: int = 512, num_heads: int = 8, num_visual_tokens: int = 16):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_visual_tokens, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, slice_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = slice_features.size(0)
        queries = self.query_tokens.expand(B, -1, -1)
        attn_out, attn_weights = self.mha(query=queries, key=slice_features, value=slice_features)
        visual_tokens = self.norm(queries + attn_out)
        return visual_tokens, attn_weights


class Medical3DVLMBaseline(nn.Module):
    def __init__(self, llm_model, tokenizer, vision_embed_dim: int = 512,
                num_visual_tokens: int = 16, use_pretrained_encoder: bool = True):
        super().__init__()
        self.llm = llm_model
        self.tokenizer = tokenizer
        self.num_visual_tokens = num_visual_tokens

        if use_pretrained_encoder:
            self.slice_encoder = load_pretrained_slice_encoder(embed_dim=vision_embed_dim, freeze=True)
        else:
            self.slice_encoder = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(32), nn.GELU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(64), nn.GELU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(256), nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                nn.Linear(256, vision_embed_dim), nn.LayerNorm(vision_embed_dim)
            )

        self.depth_aggregator = SliceDepthAggregator(
            embed_dim=vision_embed_dim, num_heads=8, num_visual_tokens=num_visual_tokens
        )
        llm_hidden_dim = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_embed_dim, llm_hidden_dim), nn.GELU(),
            nn.Linear(llm_hidden_dim, llm_hidden_dim), nn.LayerNorm(llm_hidden_dim)
        )
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.latest_depth_attention = None

    def encode_3d_volume(self, volume: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, S, H, W = volume.shape
        slices = volume.permute(0, 2, 1, 3, 4).contiguous().view(B * S, C, H, W)

        if getattr(self.slice_encoder, "_is_frozen_backbone", False):
            backbone, head = self.slice_encoder[0], self.slice_encoder[1]
            backbone.eval()
            with torch.no_grad():
                feats = backbone(slices)
            slice_feats = head(feats)
        else:
            slice_feats = self.slice_encoder(slices)

        slice_feats = slice_feats.view(B, S, -1)
        visual_tokens, attn_weights = self.depth_aggregator(slice_feats)
        projected_tokens = self.projector(visual_tokens)
        self.latest_depth_attention = attn_weights.detach()
        return projected_tokens, attn_weights

    def forward(self, images: torch.Tensor, input_ids: torch.Tensor,
                attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        visual_embeds, attn_weights = self.encode_3d_volume(images)
        B, num_vis, D = visual_embeds.shape
        target_dtype = next(self.llm.parameters()).dtype
        text_embeds = self.llm.get_input_embeddings()(input_ids)
        combined_embeds = torch.cat([visual_embeds.to(target_dtype), text_embeds.to(target_dtype)], dim=1)
        vis_mask = torch.ones((B, num_vis), dtype=attention_mask.dtype, device=attention_mask.device)
        combined_mask = torch.cat([vis_mask, attention_mask], dim=1)
        combined_labels = None
        if labels is not None:
            vis_labels = torch.full((B, num_vis), -100, dtype=labels.dtype, device=labels.device)
            combined_labels = torch.cat([vis_labels, labels], dim=1)
        outputs = self.llm(
            inputs_embeds=combined_embeds, attention_mask=combined_mask,
            labels=combined_labels, output_hidden_states=False, return_dict=True
        )
        return outputs, attn_weights, combined_labels, num_vis

    @torch.no_grad()
    def generate_answer(self, image: torch.Tensor, prompt_ids: torch.Tensor,
                        prompt_mask: torch.Tensor, max_new_tokens: int = 32) -> str:
        self.eval()
        target_dtype = next(self.llm.parameters()).dtype
        visual_embeds, _ = self.encode_3d_volume(image.unsqueeze(0))
        text_embeds = self.llm.get_input_embeddings()(prompt_ids.unsqueeze(0))
        combined_embeds = torch.cat([visual_embeds.to(target_dtype), text_embeds.to(target_dtype)], dim=1)
        vis_mask = torch.ones((1, visual_embeds.size(1)), dtype=prompt_mask.dtype, device=prompt_mask.device)
        combined_mask = torch.cat([vis_mask, prompt_mask.unsqueeze(0)], dim=1)
        gen_ids = self.llm.generate(
            inputs_embeds=combined_embeds, attention_mask=combined_mask,
            max_new_tokens=max_new_tokens, do_sample=False, num_beams=1,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        decoded = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        return decoded.strip()


# ==================================================================================================
# 8. MODEL FACTORY — [FIX 6] no more silent LoRA fallback, [FIX 7] torchao check bypassed
# ==================================================================================================
def build_baseline_model(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct", lora_r: int = 16,
                         lora_alpha: int = 32, device: str = "cuda", use_pretrained_encoder: bool = True,
                         allow_no_lora: bool = False):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    has_peft = require_peft_or_raise(allow_no_lora=allow_no_lora)  # [FIX 6]
    check_gpu_memory_headroom_or_raise(model_id)  # [FIX 9] fail loud & clear, not deep in HF internals

    print(f"\n[MODEL] Loading Base LLM: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base_llm = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=device, trust_remote_code=True
    )
    for param in base_llm.parameters():
        param.requires_grad = False

    # [FIX 8a] Gradient checkpointing: recompute activations during backward instead of storing all
    # 28 layers' worth. This is the main OOM fix for a T4 (14.56 GiB) running a frozen 1.5B LLM whose
    # input (`combined_embeds`) requires grad because of the trainable visual projector upstream.
    base_llm.config.use_cache = False  # required for checkpointing; training never needs KV cache
    if hasattr(base_llm, "gradient_checkpointing_enable"):
        try:
            base_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            print("[FIX 8] Gradient checkpointing enabled on base LLM (use_cache=False).")
        except TypeError:
            # Older transformers versions don't accept gradient_checkpointing_kwargs.
            base_llm.gradient_checkpointing_enable()
            print("[FIX 8] Gradient checkpointing enabled on base LLM (legacy call signature, use_cache=False).")
    else:
        print("[FIX 8][WARN] base_llm has no gradient_checkpointing_enable(); OOM mitigation from "
              "checkpointing will not apply. Consider upgrading transformers.")

    if has_peft:
        patch_peft_torchao_check()  # [FIX 7] avoid peft's torchao version-check ImportError
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
        )
        base_llm = get_peft_model(base_llm, lora_config)
        base_llm.print_trainable_parameters()
        # [FIX 8a] Re-assert after PEFT wrapping — some versions reset use_cache when patching config.
        if hasattr(base_llm, "enable_input_require_grads"):
            base_llm.enable_input_require_grads()
        if hasattr(base_llm, "config"):
            base_llm.config.use_cache = False

    model = Medical3DVLMBaseline(
        llm_model=base_llm, tokenizer=tokenizer, vision_embed_dim=512,
        num_visual_tokens=16, use_pretrained_encoder=use_pretrained_encoder
    )
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Total Parameters: {total_params:,} | Trainable: {trainable_params:,} "
          f"({trainable_params/total_params*100:.2f}%)")
    if has_peft and trainable_params / total_params < 0.005:
        print("[WARN] Trainable fraction looks very low even with peft installed — double-check "
              "target_modules match this model's actual layer names.")

    return model, tokenizer


# ==================================================================================================
# 9. TRAINING, VALIDATION & XAI METRICS ENGINE
# ==================================================================================================
def calculate_xai_recency_bias(attn_weights: torch.Tensor) -> float:
    """Now meaningful for items with pairing_status == 'paired_ok' (real T0/T1 stacking)."""
    mean_attn = attn_weights.mean(dim=1)
    t0_weight = mean_attn[:, :16].mean().item()
    t1_weight = mean_attn[:, 16:].mean().item()
    ratio = t1_weight / (t0_weight + 1e-6)
    return ratio


def compute_per_item_loss(logits: torch.Tensor, combined_labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
    flat_losses = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    per_token_losses = flat_losses.view(shift_labels.size())
    mask = (shift_labels != -100).float()
    denom = mask.sum(dim=1).clamp(min=1.0)
    return (per_token_losses * mask).sum(dim=1) / denom


def compute_teacher_forced_accuracy(logits: torch.Tensor, combined_labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = combined_labels[..., 1:].contiguous()
    preds = shift_logits.argmax(dim=-1)
    mask = (shift_labels != -100)
    correct = (preds == shift_labels) & mask
    denom = mask.sum(dim=1).clamp(min=1)
    return correct.sum(dim=1).float() / denom.float()


def train_one_epoch(model, loader, optimizer, lr_scheduler, scaler, epoch: int,
                    total_epochs: int, grad_accum_steps: int = 4, device: str = "cuda"):
    model.train()
    total_loss = 0.0
    recency_bias_scores = []
    pairing_counts = defaultdict(int)
    start_time = time.time()
    num_batches = len(loader)

    print(f"\n>>> Epoch [{epoch+1}/{total_epochs}] Training in Progress...")
    running_window_loss = 0.0
    window_count = 0

    oom_skipped_batches = 0

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        for ps in batch["pairing_status"]:
            pairing_counts[ps] += 1

        # [FIX 8e] Catch CUDA OOM per-batch instead of letting it crash the whole epoch. This is a
        # safety net, not a substitute for FIX 8a/8b actually fitting the model in memory — if this
        # fires often, lower --batch_size further or shorten max_seq_len / num_visual_tokens.
        try:
            autocast_context = torch.amp.autocast("cuda", dtype=torch.float16) if hasattr(torch, "amp") \
                else torch.cuda.amp.autocast(dtype=torch.float16)
            with autocast_context:
                outputs, attn_weights, combined_labels, num_vis = model(images, input_ids, attention_mask, labels)
                raw_loss = outputs.loss

            if not torch.isfinite(raw_loss):
                optimizer.zero_grad()
                continue

            loss = raw_loss / grad_accum_steps
            scaler.scale(loss).backward()
        except torch.cuda.OutOfMemoryError as e:
            oom_skipped_batches += 1
            print(f"\n[FIX 8][OOM] Skipped step {step+1}/{num_batches} (batch_size={images.size(0)}): {e}")
            optimizer.zero_grad(set_to_none=True)
            del images, input_ids, attention_mask, labels
            torch.cuda.empty_cache()
            continue

        loss_val = raw_loss.item()
        total_loss += loss_val
        running_window_loss += loss_val
        window_count += 1

        if batch["is_longitudinal"].any():
            recency_bias_scores.append(calculate_xai_recency_bias(attn_weights))

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == num_batches:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad()
            if scale_before <= scale_after:
                lr_scheduler.step()
            torch.cuda.empty_cache()  # [FIX 8d] release cached-but-unused blocks between accum groups

        if (step + 1) % 20 == 0 or (step + 1) == num_batches:
            elapsed = time.time() - start_time
            rate = (step + 1) / elapsed
            eta = (num_batches - (step + 1)) / (rate + 1e-6)
            current_lr = optimizer.param_groups[0]["lr"]
            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0
            avg_window = running_window_loss / max(window_count, 1)
            print(f"  Step [{step+1:4d}/{num_batches:4d}] | Loss: {avg_window:.4f} | "
                  f"LR: {current_lr:.2e} | VRAM: {vram_mb:.0f}MB | Speed: {rate:.1f} b/s | ETA: {eta:.0f}s", end="\r")
            running_window_loss = 0.0
            window_count = 0

    print(f"\n  [PAIRING AUDIT - Epoch {epoch+1} train] {dict(pairing_counts)}")
    if oom_skipped_batches > 0:
        print(f"  [FIX 8][OOM SUMMARY] {oom_skipped_batches}/{num_batches} batches skipped due to "
              f"CUDA OOM this epoch. If this number is large, lower --batch_size further.")
    avg_train_loss = total_loss / max(num_batches - oom_skipped_batches, 1)
    avg_recency_bias = float(np.mean(recency_bias_scores)) if recency_bias_scores else 1.0
    return avg_train_loss, avg_recency_bias


def evaluate_validation(model, loader, epoch: int, tokenizer, device: str = "cuda",
                        gen_max_samples: int = 200, max_new_tokens: int = 32):
    model.eval()
    total_loss_sum, total_loss_count = 0.0, 0
    single_losses, long_losses = [], []
    single_tf_acc, long_tf_acc = [], []
    recency_bias_records = []
    pairing_counts = defaultdict(int)
    gen_exact_matches_single, gen_exact_matches_long = [], []
    gen_samples_done = 0

    print(f"\n>>> Running Validation Audit for Epoch {epoch+1}...")
    val_oom_skipped = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            for ps in batch["pairing_status"]:
                pairing_counts[ps] += 1

            try:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    outputs, attn_weights, combined_labels, num_vis = model(images, input_ids, attention_mask, labels)
            except torch.cuda.OutOfMemoryError as e:
                # [FIX 8e] Same per-batch OOM safety net as training, applied to validation.
                val_oom_skipped += 1
                print(f"\n[FIX 8][OOM] Skipped a validation batch: {e}")
                del images, input_ids, attention_mask, labels
                torch.cuda.empty_cache()
                continue

            logits = outputs.logits.float()
            per_item_loss = compute_per_item_loss(logits, combined_labels)
            per_item_tf_acc = compute_teacher_forced_accuracy(logits, combined_labels)

            total_loss_sum += per_item_loss.sum().item()
            total_loss_count += per_item_loss.size(0)

            is_long_flags = batch["is_longitudinal"]
            for i in range(per_item_loss.size(0)):
                if bool(is_long_flags[i]):
                    long_losses.append(per_item_loss[i].item())
                    long_tf_acc.append(per_item_tf_acc[i].item())
                else:
                    single_losses.append(per_item_loss[i].item())
                    single_tf_acc.append(per_item_tf_acc[i].item())

            recency_bias_records.append(calculate_xai_recency_bias(attn_weights))

            if gen_samples_done < gen_max_samples:
                for i in range(images.size(0)):
                    if gen_samples_done >= gen_max_samples:
                        break
                    ans_start = int(batch["ans_start"][i])
                    prompt_ids_i = input_ids[i, :ans_start]
                    prompt_mask_i = attention_mask[i, :ans_start]
                    generated_text = model.generate_answer(
                        images[i], prompt_ids_i, prompt_mask_i, max_new_tokens=max_new_tokens
                    )
                    gt_answer = str(batch["answer_text"][i]).strip().lower()
                    pred_answer = generated_text.strip().lower()
                    is_match = (pred_answer == gt_answer) or (gt_answer in pred_answer and len(gt_answer) > 0)
                    if bool(is_long_flags[i]):
                        gen_exact_matches_long.append(float(is_match))
                    else:
                        gen_exact_matches_single.append(float(is_match))
                    gen_samples_done += 1

    avg_val_loss = total_loss_sum / max(total_loss_count, 1)
    perplexity = math.exp(min(avg_val_loss, 20))
    avg_single_loss = float(np.mean(single_losses)) if single_losses else float("nan")
    avg_long_loss = float(np.mean(long_losses)) if long_losses else float("nan")
    avg_single_tf_acc = float(np.mean(single_tf_acc)) if single_tf_acc else float("nan")
    avg_long_tf_acc = float(np.mean(long_tf_acc)) if long_tf_acc else float("nan")
    mean_recency_ratio = float(np.mean(recency_bias_records)) if recency_bias_records else 1.0
    gen_em_single = float(np.mean(gen_exact_matches_single)) if gen_exact_matches_single else float("nan")
    gen_em_long = float(np.mean(gen_exact_matches_long)) if gen_exact_matches_long else float("nan")

    paired_ok = pairing_counts.get("paired_ok", 0)
    fallback = sum(v for k, v in pairing_counts.items() if k.startswith("fallback"))

    torch.cuda.empty_cache()  # [FIX 8d] release cached-but-unused blocks after validation

    print("=" * 70)
    print(f"  [VALIDATION REPORT - EPOCH {epoch+1}]")
    if val_oom_skipped > 0:
        print(f"  [FIX 8][OOM SUMMARY] {val_oom_skipped} validation batches skipped due to CUDA OOM.")
    print(f"  Overall Val Loss (per-item)     : {avg_val_loss:.4f} (Perplexity: {perplexity:.2f})")
    print(f"  Single-Scan Loss (per-item)     : {avg_single_loss:.4f}  | n={len(single_losses)}")
    print(f"  Longitudinal Loss (per-item)    : {avg_long_loss:.4f}  | n={len(long_losses)}")
    print(f"  Single-Scan Token Acc (TF)      : {avg_single_tf_acc:.4f}")
    print(f"  Longitudinal Token Acc (TF)     : {avg_long_tf_acc:.4f}")
    print(f"  Single-Scan Generation EM       : {gen_em_single:.4f}  | n={len(gen_exact_matches_single)}")
    print(f"  Longitudinal Generation EM      : {gen_em_long:.4f}  | n={len(gen_exact_matches_long)}")
    print(f"  XAI Recency-Bias (T1/T0)        : {mean_recency_ratio:.3f} "
          f"({'ALERT: Recency Bias Detected' if mean_recency_ratio > 1.3 else 'Balanced Attention'})")
    print(f"  [PAIRING AUDIT] paired_ok={paired_ok} | fallback(no real T0/T1)={fallback} "
          f"| detail={dict(pairing_counts)}")
    print("  NOTE: recency-bias number above is only trustworthy in proportion to how many items")
    print("        were 'paired_ok' vs fallback. If fallback dominates, treat it as unreliable.")
    print("=" * 70)

    return {
        "val_loss": avg_val_loss, "perplexity": perplexity,
        "single_scan_loss": avg_single_loss, "longitudinal_loss": avg_long_loss,
        "single_scan_token_acc_teacher_forced": avg_single_tf_acc,
        "longitudinal_token_acc_teacher_forced": avg_long_tf_acc,
        "single_scan_generation_exact_match": gen_em_single,
        "longitudinal_generation_exact_match": gen_em_long,
        "xai_recency_bias_ratio": mean_recency_ratio,
        "pairing_paired_ok": paired_ok, "pairing_fallback": fallback,
        "pairing_detail": dict(pairing_counts),
    }


# ==================================================================================================
# 10. AUTOMATED RESULT COMPRESSION
# ==================================================================================================
def create_results_zip_archive(output_dir: str):
    zip_filename = os.path.join(output_dir, "baseline_training_results.zip")
    print(f"\n[PACKAGING] Compressing results into: {zip_filename}")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, output_dir)
                zf.write(full_path, rel_path)
    zip_size_mb = os.path.getsize(zip_filename) / (1024 ** 2)
    print(f"  -> Successfully generated: baseline_training_results.zip ({zip_size_mb:.2f} MB)")
    return zip_filename


# ==================================================================================================
# 11. MAIN EXECUTION CONTROLLER
# ==================================================================================================
def main():
    default_data, default_out = resolve_dataset_paths()

    parser = argparse.ArgumentParser(description="3D-RAD Baseline Training Pipeline (Fixed v6).")
    parser.add_argument("--data_dir", type=str, default=default_data)
    parser.add_argument("--output_dir", type=str, default=default_out)
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    # [FIX 8b] Lowered 4 -> 2 (grad_accum raised 4 -> 8) so effective batch size is unchanged at 16,
    # but peak activation memory per forward pass is halved. Override on the command line as needed.
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_pretrained_encoder", action="store_true", default=True)
    parser.add_argument("--no_pretrained_encoder", dest="use_pretrained_encoder", action="store_false")
    parser.add_argument("--gen_max_samples", type=int, default=200)
    parser.add_argument("--diagnose_only", action="store_true",
                        help="[FIX 5] Run split + longitudinal-structure diagnostic only, then exit.")
    parser.add_argument("--allow_no_lora", action="store_true",
                        help="[FIX 6] Explicitly allow training WITHOUT LoRA if peft is unavailable "
                             "(not recommended — this changes the experiment).")

    if any("ipykernel" in str(arg) or str(arg).endswith(".json") for arg in sys.argv):
        args, _ = parser.parse_known_args(args=[])
    else:
        args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("  3D-RAD MULTIMODAL BASELINE TRAINING PIPELINE — FIXED v6")
    print("=" * 70)
    print(f"Device               : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Raw Dataset Path     : {args.data_dir}")
    print(f"Output Directory     : {args.output_dir}")
    print(f"Base LLM             : {args.llm_model}")
    print(f"Pretrained Encoder   : {args.use_pretrained_encoder}")
    print(f"Epochs               : {args.epochs} | Batch Size: {args.batch_size} "
          f"(Effective: {args.batch_size * args.grad_accum_steps})")
    print("=" * 70)

    # 1. AUTO-SPLIT (unchanged, confirmed correct)
    split_dir = os.path.join(args.output_dir, "split_dataset")
    ensure_dataset_split(data_dir=args.data_dir, split_dir=split_dir, seed=args.seed)

    # 2. [FIX 5] Longitudinal structure diagnostic — ALWAYS runs, read it before trusting results
    train_csv_path = os.path.join(split_dir, "train", "train_qa.csv")
    diagnose_longitudinal_structure(data_dir=args.data_dir, split_csv=train_csv_path)

    if args.diagnose_only:
        print("[INFO] --diagnose_only was set. Exiting after diagnostics.")
        return

    # Build the patient scan timeline used for T0/T1 pairing
    print("[SETUP] Building patient scan timeline for longitudinal pairing...")
    patient_timeline = build_patient_scan_timeline(args.data_dir)
    print(f"[SETUP] Timeline built for {len(patient_timeline)} patients.")

    # 3. [FIX 6 + FIX 7] MODEL & TOKENIZER INITIALIZATION
    print("\n" + "=" * 70)
    print("  [STEP 2/6] INITIALIZING MULTIMODAL ARCHITECTURE & LORA ADAPTERS")
    print("=" * 70)
    model, tokenizer = build_baseline_model(
        model_id=args.llm_model, lora_r=args.lora_r, device=device,
        use_pretrained_encoder=args.use_pretrained_encoder, allow_no_lora=args.allow_no_lora
    )

    # 4. DATALOADERS
    print("\n" + "=" * 70)
    print("  [STEP 3/6] BUILDING OPTIMIZED PYTORCH 3D DATALOADERS")
    print("=" * 70)
    train_dataset = Medical3DVQADataset(
        csv_file=os.path.join(split_dir, "train", "train_qa.csv"),
        img_dir=os.path.join(split_dir, "train", "images"),
        tokenizer=tokenizer, patient_timeline=patient_timeline, data_dir=args.data_dir, is_train=True
    )
    val_dataset = Medical3DVQADataset(
        csv_file=os.path.join(split_dir, "val", "val_qa.csv"),
        img_dir=os.path.join(split_dir, "val", "images"),
        tokenizer=tokenizer, patient_timeline=patient_timeline, data_dir=args.data_dir, is_train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"Train Batches: {len(train_loader):,} | Val Batches: {len(val_loader):,}")

    # 5. OPTIMIZER, SCHEDULER & SCALER
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01
    )
    total_steps = (len(train_loader) // args.grad_accum_steps) * args.epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))
    scaler = torch.amp.GradScaler("cuda") if hasattr(torch, "amp") else torch.cuda.amp.GradScaler()

    # 6. TRAINING LOOP
    print("\n" + "=" * 70)
    print("  [STEP 4/6] COMMENCING MULTIMODAL BASELINE TRAINING")
    print("=" * 70)
    history = []
    best_val_loss = float("inf")
    checkpoint_dir = os.path.join(args.output_dir, "best_checkpoint")
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epochs):
        train_loss, train_recency = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, lr_scheduler=lr_scheduler,
            scaler=scaler, epoch=epoch, total_epochs=args.epochs,
            grad_accum_steps=args.grad_accum_steps, device=device
        )
        val_metrics = evaluate_validation(
            model=model, loader=val_loader, epoch=epoch, tokenizer=tokenizer, device=device,
            gen_max_samples=args.gen_max_samples
        )
        epoch_record = {"epoch": epoch + 1, "train_loss": train_loss,
                        "train_recency_bias_ratio": train_recency, **val_metrics}
        history.append(epoch_record)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            torch.save({
                "epoch": epoch + 1,
                "projector_state_dict": model.projector.state_dict(),
                "slice_encoder_state_dict": model.slice_encoder.state_dict(),
                "depth_aggregator_state_dict": model.depth_aggregator.state_dict(),
                "val_loss": best_val_loss
            }, os.path.join(checkpoint_dir, "multimodal_projector_best.pt"))
            if hasattr(model.llm, "save_pretrained"):
                model.llm.save_pretrained(os.path.join(checkpoint_dir, "lora_adapters"))
            print(f"  >>> [CHECKPOINT SAVED] Best model saved with Val Loss: {best_val_loss:.4f}")

    history_file = os.path.join(args.output_dir, "training_history_metrics.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[REPORT] Complete training history saved to: {history_file}")

    print("\n" + "=" * 70)
    print("  [STEP 5/6 -> 6/6] PACKAGING RESULTS")
    print("=" * 70)
    zip_path = create_results_zip_archive(args.output_dir)
    print("\n" + "=" * 70)
    print("  TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"  All checkpoints, metrics, and logs packaged at: {zip_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
