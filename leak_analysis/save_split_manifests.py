#!/usr/bin/env python3
"""
Regenerate and persist the patient-level 80/10/10 split manifests locally.

The split itself is deterministic (seed 42, same grouping as the training
pipeline), but the manifests only ever existed inside a Kaggle session, so the
split could not be reproduced or inspected offline. This writes them to disk
and re-runs the integrity assertions.
"""
import glob, os, json
import pandas as pd, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = os.path.join(ROOT, "split_manifests")
SEED = 42

vols = {os.path.basename(p)[:-4]: p for p in
        glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/Images/**/*.npy"), recursive=True)}
frames = []
for f in glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/*/Task*/*.csv")):
    d = pd.read_csv(f)
    if "VolumeName" not in d.columns:
        continue
    d["TaskCategory"] = os.path.basename(os.path.dirname(f))
    d["Subtask"] = os.path.basename(f)[:-4]
    frames.append(d)
df = pd.concat(frames, ignore_index=True)
df["NpyName"] = df["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False) + ".npy"
df["ImagePath"] = df["NpyName"].str[:-4].map(vols)
df = df[df["ImagePath"].notna()].copy()
df["PatientID"] = df["NpyName"].str.split("_").str[:2].str.join("_")

pats = df["PatientID"].unique()
rng = np.random.RandomState(SEED); rng.shuffle(pats)
n_tr, n_va = int(len(pats) * .80), int(len(pats) * .10)
assign = {p: "train" for p in pats[:n_tr]}
assign.update({p: "val" for p in pats[n_tr:n_tr + n_va]})
assign.update({p: "test" for p in pats[n_tr + n_va:]})
df["Split"] = df["PatientID"].map(assign)

os.makedirs(OUT, exist_ok=True)
summary = {"seed": SEED, "total_qa": int(len(df)), "splits": {}}
sets = {}
for s in ("train", "val", "test"):
    sub = df[df.Split == s]
    sub.to_csv(os.path.join(OUT, f"{s}_qa.csv"), index=False)
    sets[s] = set(sub["PatientID"])
    summary["splits"][s] = {"qa_pairs": int(len(sub)),
                            "unique_volumes": int(sub["NpyName"].nunique()),
                            "unique_patients": int(sub["PatientID"].nunique())}

for a in ("train", "val", "test"):
    for b in ("train", "val", "test"):
        if a < b:
            assert not (sets[a] & sets[b]), f"patient leakage {a}/{b}"
summary["patient_leakage"] = "0 overlap verified across all three split pairs"

with open(os.path.join(OUT, "split_summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print("=" * 66)
print("  PATIENT-LEVEL 80/10/10 SPLIT MANIFESTS (regenerated locally)")
print("=" * 66)
for s, v in summary["splits"].items():
    print(f"  {s:<6} {v['qa_pairs']:>7,} QA | {v['unique_volumes']:>5,} volumes | "
          f"{v['unique_patients']:>5,} patients")
print(f"\n  {summary['patient_leakage']}")
print(f"  [saved] split_manifests/{{train,val,test}}_qa.csv + split_summary.json")
