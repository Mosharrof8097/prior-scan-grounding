#!/usr/bin/env python3
"""
========================================================================================
3D Medical VQA Comprehensive Duplicate & Leakage Audit Pipeline (Kaggle Ready)
========================================================================================
Algorithms Implemented:
  1. Exact VQA Duplicate Check:
     - Identifies identical (VolumeName, Question) pairs within each split.
     - Detects label conflict (same volume + question, but conflicting answers).
  2. Cross-Split Leakage Check:
     - Checks if any (VolumeName, Question) from Train appears in Val or Test.
     - Checks if any Patient ID crosses split boundaries.
  3. Normalized / Near-Duplicate Text Check:
     - Normalizes punctuation, case, and whitespace to catch near-duplicate questions.
  4. 3D Image Bitwise Hash Check (MD5):
     - Computes hashes of .npy 3D volumes to detect identical scans with different names.
     - Checks for cross-split image duplication.
  5. Global Template & QA Distribution Audit:
     - Computes frequency of question patterns and answers across partitions.
========================================================================================
"""

import os
import sys
import glob
import json
import hashlib
import re
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd


def get_default_paths():
    """Detects Kaggle split output or input folder."""
    # 1. Output from earlier split run in working directory
    if os.path.exists("/kaggle/working/split_dataset/train"):
        return "/kaggle/working/split_dataset", "/kaggle/working/duplicate_audit_results"

    # 2. Mounted notebook output or dataset input in Kaggle
    kaggle_splits = glob.glob("/kaggle/input/**/split_dataset/train", recursive=True)
    if kaggle_splits:
        split_root = os.path.dirname(kaggle_splits[0])
        return split_root, "/kaggle/working/duplicate_audit_results"

    # 3. Local fallback
    local_dir = "/home/mosharrof/medical_thesis/dataset preparetion/dataset split script/kaggle_ready_split"
    local_out = "/home/mosharrof/medical_thesis/dataset preparetion/dataset dublicate cheak/audit_output"
    return local_dir, local_out


def normalize_text(text):
    """Normalize text for near-duplicate detection."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # normalize spaces
    return text


def compute_file_hash(filepath, block_size=65536):
    """Compute MD5 hash of an image file."""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def load_splits(data_dir):
    """Loads train, val, and test QA CSVs."""
    splits = {}
    for s in ["train", "val", "test"]:
        csv_candidates = [
            os.path.join(data_dir, s, f"{s}_qa.csv"),
            os.path.join(data_dir, s, f"{s}_split.csv"),
            os.path.join(data_dir, f"{s}_split.csv")
        ]
        found = None
        for c in csv_candidates:
            if os.path.exists(c):
                found = c
                break
        
        if not found:
            raise FileNotFoundError(f"Could not find QA CSV for split '{s}' in {data_dir}")
        
        df = pd.read_csv(found)
        df["Split"] = s
        if "NpyName" not in df.columns and "VolumeName" in df.columns:
            df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))
        if "PatientID" not in df.columns and "NpyName" in df.columns:
            df["PatientID"] = df["NpyName"].apply(lambda x: "_".join(x.split("_")[:2]))
        splits[s] = df
        print(f"  Loaded {s.upper():5} split: {len(df):,} records ({found})")
    
    return splits


def algo1_exact_vqa_duplicates(splits):
    """Algorithm 1: Exact VQA Duplicate Check within each partition."""
    print("\n" + "=" * 75)
    print("ALGORITHM 1: Exact (VolumeName, Question) Duplicate Detection")
    print("=" * 75)
    
    report = {}
    for s, df in splits.items():
        total = len(df)
        dup_mask = df.duplicated(subset=["VolumeName", "Question"], keep=False)
        dup_df = df[dup_mask]
        num_dup_rows = len(dup_df)
        num_unique_cases = df.duplicated(subset=["VolumeName", "Question"]).sum()
        
        # Check for conflicting answers (same volume + question, different answer)
        conflicts = 0
        if num_unique_cases > 0:
            grouped = dup_df.groupby(["VolumeName", "Question"])["Answer"].nunique()
            conflicts = int((grouped > 1).sum())

        report[s] = {
            "total_records": total,
            "duplicate_records": int(num_unique_cases),
            "duplicate_rate_pct": round((num_unique_cases / total) * 100, 3) if total > 0 else 0,
            "conflicting_answer_cases": conflicts
        }
        
        status = "PASSED" if num_unique_cases == 0 else "FLAGGED"
        print(f"  [{s.upper():5}] Total: {total:,} | Duplicates: {num_unique_cases:,} ({report[s]['duplicate_rate_pct']}%) | Conflicts: {conflicts} | Status: {status}")
    
    return report


def algo2_cross_split_leakage(splits):
    """Algorithm 2: Cross-Split Leakage Detection (Patient & VQA Pair)."""
    print("\n" + "=" * 75)
    print("ALGORITHM 2: Cross-Split Contamination & Leakage Detection")
    print("=" * 75)

    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    report = {}

    for s1, s2 in pairs:
        # 1. Patient ID overlap
        p1 = set(splits[s1]["PatientID"].dropna())
        p2 = set(splits[s2]["PatientID"].dropna())
        patient_overlap = p1 & p2
        
        # 2. Exact (VolumeName, Question) overlap
        vqa1 = set(zip(splits[s1]["VolumeName"], splits[s1]["Question"]))
        vqa2 = set(zip(splits[s2]["VolumeName"], splits[s2]["Question"]))
        vqa_overlap = vqa1 & vqa2

        # 3. Image volume overlap
        v1 = set(splits[s1]["VolumeName"].dropna())
        v2 = set(splits[s2]["VolumeName"].dropna())
        vol_overlap = v1 & v2

        key = f"{s1}_vs_{s2}"
        report[key] = {
            "patient_leakage_count": len(patient_overlap),
            "volume_leakage_count": len(vol_overlap),
            "vqa_pair_leakage_count": len(vqa_overlap)
        }

        status = "CLEAN (PASS)" if len(patient_overlap) == 0 and len(vqa_overlap) == 0 else "LEAKAGE DETECTED"
        print(f"  [{s1.upper()} vs {s2.upper()}]:")
        print(f"    - Patient Overlap : {len(patient_overlap)} patients")
        print(f"    - Volume Overlap  : {len(vol_overlap)} volumes")
        print(f"    - VQA Pair Overlap: {len(vqa_overlap)} pairs")
        print(f"    - Integrity Result: {status}")

    return report


def algo3_normalized_text_duplicates(splits):
    """Algorithm 3: Normalized / Near-Duplicate Question Check."""
    print("\n" + "=" * 75)
    print("ALGORITHM 3: Normalized & Fuzzy Text Duplicate Detection")
    print("=" * 75)

    report = {}
    for s, df in splits.items():
        df_norm = df.copy()
        df_norm["NormQuestion"] = df_norm["Question"].apply(normalize_text)
        
        # Near-duplicate on same image
        near_dups = df_norm.duplicated(subset=["VolumeName", "NormQuestion"]).sum()
        
        report[s] = {
            "near_duplicate_questions": int(near_dups),
            "near_duplicate_pct": round((near_dups / len(df)) * 100, 3)
        }
        print(f"  [{s.upper():5}] Near-duplicate questions on same volume: {near_dups:,} ({report[s]['near_duplicate_pct']}%)")

    return report


def algo4_image_bitwise_hash_check(splits, data_dir):
    """Algorithm 4: MD5 Bitwise Hash Duplication on 3D CT Volumes."""
    print("\n" + "=" * 75)
    print("ALGORITHM 4: 3D CT Volume Image Hash (Bitwise Identity) Check")
    print("=" * 75)

    image_hashes = {}
    split_image_hashes = defaultdict(dict)
    
    # Locate images
    for s in ["train", "val", "test"]:
        img_dir = os.path.join(data_dir, s, "images")
        if not os.path.exists(img_dir):
            print(f"  [WARN] Images dir not found: {img_dir}. Checking recursive search...")
            img_dir = data_dir

        for img_name in splits[s]["NpyName"].unique():
            img_path = os.path.join(img_dir, img_name)
            if not os.path.exists(img_path):
                # Search recursively
                matches = glob.glob(os.path.join(data_dir, "**", img_name), recursive=True)
                if matches:
                    img_path = matches[0]
                else:
                    continue

            h = compute_file_hash(img_path)
            if h:
                image_hashes[img_name] = h
                split_image_hashes[s][img_name] = h

    print(f"  Hashed {len(image_hashes):,} unique 3D CT image volumes.")

    # Check for identical image files with different names
    hash_to_names = defaultdict(list)
    for name, h in image_hashes.items():
        hash_to_names[h].append(name)
    
    identical_scans = {h: names for h, names in hash_to_names.items() if len(names) > 1}
    print(f"  Bitwise identical 3D scans across different filenames: {len(identical_scans)}")

    # Cross-split hash leakage
    h_train = set(split_image_hashes["train"].values())
    h_val = set(split_image_hashes["val"].values())
    h_test = set(split_image_hashes["test"].values())

    cross_hash_leak_train_val = len(h_train & h_val)
    cross_hash_leak_train_test = len(h_train & h_test)
    cross_hash_leak_val_test = len(h_val & h_test)

    print(f"  Cross-Split Image Hash Overlap:")
    print(f"    - Train vs Val : {cross_hash_leak_train_val} shared scan hashes")
    print(f"    - Train vs Test: {cross_hash_leak_train_test} shared scan hashes")
    print(f"    - Val vs Test  : {cross_hash_leak_val_test} shared scan hashes")

    return {
        "total_hashed_volumes": len(image_hashes),
        "bitwise_identical_scan_clusters": len(identical_scans),
        "cross_split_hash_leakage": {
            "train_val": cross_hash_leak_train_val,
            "train_test": cross_hash_leak_train_test,
            "val_test": cross_hash_leak_val_test
        }
    }


def algo5_global_template_redundancy(splits):
    """Algorithm 5: Question Template and Answer Redundancy Analysis."""
    print("\n" + "=" * 75)
    print("ALGORITHM 5: Global Question Template & Answer Redundancy Audit")
    print("=" * 75)

    all_df = pd.concat(list(splits.values()), ignore_index=True)
    top_questions = all_df["Question"].value_counts().head(5)
    top_answers = all_df["Answer"].value_counts().head(5)

    print("  Top 3 Most Frequent Questions (Dataset-wide templates):")
    for q, count in top_questions.items():
        pct = (count / len(all_df)) * 100
        print(f"    - [{count:,} times | {pct:.1f}%] \"{q[:60]}...\"")

    print("\n  Top 3 Most Frequent Answers:")
    for a, count in top_answers.items():
        pct = (count / len(all_df)) * 100
        print(f"    - [{count:,} times | {pct:.1f}%] \"{str(a)[:40]}\"")

    return {
        "unique_questions_overall": int(all_df["Question"].nunique()),
        "unique_answers_overall": int(all_df["Answer"].nunique()),
        "top_question_frequency": int(top_questions.iloc[0]) if len(top_questions) > 0 else 0
    }


def main():
    def_data, def_out = get_default_paths()
    parser = argparse.ArgumentParser(description="Multi-Algorithm Duplicate & Leakage Checker for 3D-RAD Medical VQA.")
    parser.add_argument("--data_dir", type=str, default=def_data, help="Path to split_dataset folder")
    parser.add_argument("--output_dir", type=str, default=def_out, help="Output report directory")

    # Safe handling for Jupyter / IPython notebook execution
    if any("ipykernel" in str(arg) or str(arg).endswith(".json") for arg in sys.argv):
        args, _ = parser.parse_known_args(args=[])
    else:
        args, _ = parser.parse_known_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 75)
    print("  3D-RAD COMPREHENSIVE MULTI-ALGORITHM DUPLICATE & LEAKAGE AUDIT")
    print("=" * 75)
    print(f"Dataset Split Path: {args.data_dir}")
    print(f"Audit Output Path : {args.output_dir}")

    splits = load_splits(args.data_dir)

    results = {
        "algo1_exact_duplicates": algo1_exact_vqa_duplicates(splits),
        "algo2_cross_split_leakage": algo2_cross_split_leakage(splits),
        "algo3_normalized_text_duplicates": algo3_normalized_text_duplicates(splits),
        "algo4_image_bitwise_hashes": algo4_image_bitwise_hash_check(splits, args.data_dir),
        "algo5_global_templates": algo5_global_template_redundancy(splits)
    }

    report_file = os.path.join(args.output_dir, "duplicate_audit_report.json")
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(f"[SUMMARY] Comprehensive audit complete. Machine-readable report saved to:")
    print(f"  -> {report_file}")
    print("=" * 75)


if __name__ == "__main__":
    main()
