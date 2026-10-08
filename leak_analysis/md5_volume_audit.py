#!/usr/bin/env python3
"""
Algorithm 4 of the duplicate audit, re-run to completion.

The original run was interrupted by KeyboardInterrupt before hashing a single
file, so duplicate_audit_report.json reports all-zero values for this check.
Those zeros mean "not measured", not "no duplicates" -- the dataset
documentation already flags this. This script finishes it.

For every released .npy volume: MD5 of the raw bytes, cluster identical files,
then check whether any bitwise-identical cluster straddles the patient-level
80/10/10 split (which would be image leakage the filename-based audit misses).
"""
import glob, os, json, hashlib, collections, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 42


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    paths = sorted(glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/Images/**/*.npy"), recursive=True))
    print(f"[algo4] hashing {len(paths):,} volumes ...", flush=True)

    by_hash = collections.defaultdict(list)
    for i, p in enumerate(paths, 1):
        by_hash[md5(p)].append(os.path.basename(p)[:-4])
        if i % 250 == 0 or i == len(paths):
            print(f"  {i:,}/{len(paths):,}", flush=True)

    clusters = {h: v for h, v in by_hash.items() if len(v) > 1}

    # patient-level split, same grouping and seed as the training pipeline
    pats = sorted({"_".join(os.path.basename(p)[:-4].split("_")[:2]) for p in paths})
    arr = np.array(pats)
    rng = np.random.RandomState(SEED); rng.shuffle(arr)
    n_tr, n_va = int(len(arr) * .80), int(len(arr) * .10)
    split_of = {}
    for p in arr[:n_tr]:           split_of[p] = "train"
    for p in arr[n_tr:n_tr + n_va]: split_of[p] = "val"
    for p in arr[n_tr + n_va:]:     split_of[p] = "test"

    cross, same_patient, cross_patient = collections.Counter(), 0, 0
    for h, names in clusters.items():
        splits = {split_of.get("_".join(n.split("_")[:2])) for n in names}
        pset = {"_".join(n.split("_")[:2]) for n in names}
        if len(pset) == 1:
            same_patient += 1
        else:
            cross_patient += 1
        for a in ("train", "val", "test"):
            for b in ("train", "val", "test"):
                if a < b and {a, b} <= splits:
                    cross[f"{a}_{b}"] += 1

    rep = {
        "total_hashed_volumes": len(paths),
        "unique_md5": len(by_hash),
        "bitwise_identical_scan_clusters": len(clusters),
        "volumes_inside_a_duplicate_cluster": sum(len(v) for v in clusters.values()),
        "clusters_within_one_patient": same_patient,
        "clusters_spanning_different_patients": cross_patient,
        "cross_split_hash_leakage": {k: int(cross.get(k, 0)) for k in ("train_val", "train_test", "val_test")},
        "example_clusters": [v for v in list(clusters.values())[:5]],
        "note": "Completed run. Supersedes the all-zero algo4 block in duplicate_audit_report.json, "
                "which was an interrupted run, not a passed check.",
    }
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "md5_volume_audit.json")
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=2)

    print("\n" + "=" * 70)
    print("  ALGORITHM 4 - BITWISE VOLUME HASH AUDIT (COMPLETED)")
    print("=" * 70)
    for k, v in rep.items():
        if k != "example_clusters":
            print(f"  {k}: {v}")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
