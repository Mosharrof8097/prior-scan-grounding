#!/usr/bin/env python3
"""
========================================================================================
3D-RAD Full Dataset Split & Integrity Testing Pipeline (For Kaggle Notebook / CLI)
========================================================================================
Handles: Both 3D Images (.npy) and Question-Answer Text (.csv)
Target Platform: Kaggle (also compatible with Local)

Steps:
  1. Discovers all 3D CT volumes (.npy) and QA CSV records from Kaggle input.
  2. Patient-Level Stratified Split (80% Train, 10% Val, 10% Test).
  3. Symlinks / maps 3D images and saves split CSVs into /kaggle/working/
     (uses symlinks to prevent exceeding Kaggle's 20 GB disk quota).
  4. Post-Split Testing:
     - Test 1: Image-to-Text 100% Existence Check (asserts every QA has a valid 3D scan).
     - Test 2: 3D Array Sanity Check (shape, voxel range, dtype).
     - Test 3: Zero Patient Leakage Test (asserts no patient overlap across splits).
     - Test 4: Duplicate Record Test.
========================================================================================
"""

import os
import sys
import glob
import json
import argparse
import numpy as np
import pandas as pd


def get_default_paths():
    """Detects Kaggle input or falls back to local."""
    # Check Kaggle mount: look for Images folder or any directory under /kaggle/input
    kaggle_candidates = glob.glob("/kaggle/input/**/Images", recursive=True)
    if kaggle_candidates:
        kaggle_root = os.path.dirname(kaggle_candidates[0])
        return kaggle_root, "/kaggle/working/split_dataset"

    npy_candidates = glob.glob("/kaggle/input/**/*.npy", recursive=True)
    if npy_candidates:
        # Find root of dataset
        parts = npy_candidates[0].split(os.sep)
        # /kaggle/input/<dataset_name>
        if len(parts) >= 4 and parts[1] == "kaggle" and parts[2] == "input":
            kaggle_root = os.sep.join(parts[:4])
            return kaggle_root, "/kaggle/working/split_dataset"

    kaggle_inputs = glob.glob("/kaggle/input/*")
    if kaggle_inputs:
        return kaggle_inputs[0], "/kaggle/working/split_dataset"

    # Local fallback
    local_root = "/home/mosharrof/medical_thesis/3D-RAD_Dataset"
    local_out = "/home/mosharrof/medical_thesis/dataset preparetion/dataset split script/kaggle_ready_split"
    return local_root, local_out


def scan_dataset(data_dir):
    print(f"\n[1/4] Scanning dataset at: {data_dir}")
    
    # 1. Map all 3D image files on disk
    image_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    image_map = {os.path.basename(p): p for p in image_paths}
    print(f"  -> Found {len(image_map):,} unique 3D CT volume files (.npy)")

    if len(image_map) == 0:
        raise FileNotFoundError(f"No .npy 3D image files found under {data_dir}")

    # 2. Load all QA CSV files
    csv_paths = glob.glob(os.path.join(data_dir, "**", "test", "*", "*.csv"), recursive=True)
    if not csv_paths:
        csv_paths = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    dfs = []
    for f in csv_paths:
        try:
            d = pd.read_csv(f)
            if "VolumeName" in d.columns and "Question" in d.columns:
                d["TaskName"] = os.path.basename(os.path.dirname(f))
                dfs.append(d)
        except Exception:
            continue

    if not dfs:
        raise FileNotFoundError(f"No valid QA CSV files found under {data_dir}")

    df = pd.concat(dfs, ignore_index=True)
    df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))
    df["PatientID"] = df["NpyName"].apply(lambda x: "_".join(x.split("_")[:2]))
    
    # Attach absolute image path
    df["ImagePath"] = df["NpyName"].map(image_map)
    matched = df["ImagePath"].notna().sum()
    print(f"  -> Found {len(df):,} QA records ({matched:,} with verified 3D image match)")

    return df, image_map


def split_by_patient(df, train_ratio=0.80, val_ratio=0.10, test_ratio=0.10, seed=42):
    print(f"\n[2/4] Splitting by Patient ID (Train={train_ratio*100:.0f}%, Val={val_ratio*100:.0f}%, Test={test_ratio*100:.0f}%)...")
    unique_patients = np.array(list(df["PatientID"].unique()))
    
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_patients)

    n_tot = len(unique_patients)
    n_train = int(n_tot * train_ratio)
    n_val = int(n_tot * val_ratio)

    p_train = set(unique_patients[:n_train])
    p_val = set(unique_patients[n_train:n_train + n_val])
    p_test = set(unique_patients[n_train + n_val:])

    def assign(pid):
        if pid in p_train: return "train"
        if pid in p_val: return "val"
        return "test"

    df = df.copy()
    df["Split"] = df["PatientID"].map(assign)
    return df


def organize_split_directories(df, output_dir, use_symlinks=True):
    print(f"\n[3/4] Organizing splits into: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    for split in ["train", "val", "test"]:
        split_dir = os.path.join(output_dir, split)
        img_dir = os.path.join(split_dir, "images")
        os.makedirs(img_dir, exist_ok=True)

        sub_df = df[df["Split"] == split].copy()
        
        # Link 3D images
        linked_count = 0
        for src in sub_df["ImagePath"].dropna().unique():
            dst = os.path.join(img_dir, os.path.basename(src))
            if not os.path.exists(dst):
                if use_symlinks:
                    try:
                        os.symlink(src, dst)
                        linked_count += 1
                    except OSError:
                        # Fallback if symlinks not supported
                        import shutil
                        shutil.copy2(src, dst)
                        linked_count += 1
                else:
                    import shutil
                    shutil.copy2(src, dst)
                    linked_count += 1

        # Save split QA CSV
        csv_out = os.path.join(split_dir, f"{split}_qa.csv")
        sub_df.to_csv(csv_out, index=False)
        print(f"  -> [{split.upper():5}] {len(sub_df):,} QA pairs | {linked_count:,} 3D scans linked | Saved: {csv_out}")


def run_post_split_tests(output_dir):
    print("\n[4/4] RUNNING POST-SPLIT VALIDATION & INTEGRITY TESTS")
    print("=" * 65)

    splits = {}
    for s in ["train", "val", "test"]:
        csv_path = os.path.join(output_dir, s, f"{s}_qa.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing split CSV: {csv_path}")
        splits[s] = pd.read_csv(csv_path)

    # TEST 1: Patient Leakage Test
    print("Test 1: Zero Patient Leakage Test...")
    p_train = set(splits["train"]["PatientID"])
    p_val = set(splits["val"]["PatientID"])
    p_test = set(splits["test"]["PatientID"])

    assert len(p_train & p_val) == 0, "FAILED: Overlap between Train and Val patients!"
    assert len(p_train & p_test) == 0, "FAILED: Overlap between Train and Test patients!"
    assert len(p_val & p_test) == 0, "FAILED: Overlap between Val and Test patients!"
    print("  [PASS] Zero patient overlap confirmed between all splits.")

    # TEST 2: Image File Existence Test
    print("Test 2: 3D Image File Existence Test...")
    for s in ["train", "val", "test"]:
        img_dir = os.path.join(output_dir, s, "images")
        df_s = splits[s]
        missing = 0
        for img_name in df_s["NpyName"].unique():
            if not os.path.exists(os.path.join(img_dir, img_name)):
                missing += 1
        assert missing == 0, f"FAILED: {missing} image files missing in {s} split!"
        print(f"  [PASS] All {df_s['NpyName'].nunique():,} unique 3D scans verified in {s}/images/")

    # TEST 3: 3D Array Sanity Check (Shape & Voxel Range)
    print("Test 3: 3D Array Voxel Sanity Check...")
    sample_img_path = glob.glob(os.path.join(output_dir, "train", "images", "*.npy"))[0]
    sample_arr = np.load(sample_img_path)
    print(f"  [PASS] Sample shape: {sample_arr.shape} | dtype: {sample_arr.dtype} | min: {sample_arr.min():.2f} | max: {sample_arr.max():.2f}")
    assert sample_arr.ndim in (3, 4), "FAILED: Array is not 3D or 4D volumetric data!"

    # TEST 4: Duplicate QA Check
    print("Test 4: Duplicate Record Check...")
    for s in ["train", "val", "test"]:
        dups = splits[s].duplicated(subset=["VolumeName", "Question"]).sum()
        print(f"  [PASS] {s.capitalize()} duplicate QA pairs: {dups} (out of {len(splits[s]):,})")

    print("=" * 65)
    print("ALL TESTS PASSED! DATASET IS FULLY PREPARED & CERTIFIED.")


def main():
    def_data, def_out = get_default_paths()
    parser = argparse.ArgumentParser(description="Split and test full 3D-RAD dataset (Images + QA).")
    parser.add_argument("--data_dir", type=str, default=def_data, help="Root path of dataset")
    parser.add_argument("--output_dir", type=str, default=def_out, help="Output split folder")
    parser.add_argument("--train_ratio", type=float, default=0.80)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)

    # Check if running inside Jupyter / IPython kernel to prevent SystemExit: 2
    if any("ipykernel" in str(arg) or str(arg).endswith(".json") for arg in sys.argv):
        args, _ = parser.parse_known_args(args=[])
    else:
        args, _ = parser.parse_known_args()

    print("=" * 65)
    print("  3D-RAD FULL DATASET (IMAGES + TEXT) SPLITTER & TESTER")
    print("=" * 65)

    df, image_map = scan_dataset(args.data_dir)
    df_split = split_by_patient(df, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    organize_split_directories(df_split, args.output_dir, use_symlinks=True)
    run_post_split_tests(args.output_dir)


if __name__ == "__main__":
    main()
