#!/usr/bin/env python3
"""
ONE-FILE KAGGLE SMOKE TEST for the 3D-RAD inference audit (60 items per task, val only).

How to use on Kaggle (a GPU T4 notebook with BOTH datasets attached):
  Option A (no upload): paste this whole file into ONE notebook cell and run it.
  Option B: upload it with the audit_upload dataset, then run:   !python /kaggle/input/.../kaggle_smoke_test.py

It does, in order:
  1. pip install peft
  2. finds kaggle_inference_audit.py under /kaggle/input (must be the version with --controls)
  3. preflight (CPU, seconds)
  4. smoke test: --limit 60 --splits val --controls mean noise swap zero_prior swap_prior
  5. checks the outputs and prints PASS or FAIL with the reason
  6. zips the four files to bring back: /kaggle/working/smoke_out.zip

The accuracy numbers from 60 items mean nothing. Only PASS/FAIL matters.
"""
import argparse, glob, os, subprocess, sys, zipfile
import numpy as np
import pandas as pd

CONTROLS = ["mean", "noise", "swap", "zero_prior", "swap_prior"]
OUT = "/kaggle/working/inference_audit"

PRIOR = ("zero_prior", "swap_prior")


def check_split(d, split, controls):
    path = os.path.join(d, f"per_item_{split}.csv")
    if not os.path.exists(path):
        return [f"{split}: {path} is missing"], None
    df = pd.read_csv(path)
    errs = []
    want = ["real", "blank"] + list(controls)
    have = set(df.condition.unique())
    for c in want:
        if c not in have:
            errs.append(f"{split}: condition '{c}' has no rows")
    unexpected = have - set(want)
    if unexpected:
        errs.append(f"{split}: unexpected conditions {sorted(unexpected)} (pass --controls to match the run)")
    real = df[df.condition == "real"]
    if real.empty:
        return errs + [f"{split}: no 'real' rows"], df
    temporal = real.task.str.contains("Temporal", case=False)
    key = ["item_id"]
    for c in sorted(have - {"real"}):
        sub = df[df.condition == c]
        expected = real[temporal] if c in PRIOR else real
        if len(sub) != len(expected):
            errs.append(f"{split}/{c}: {len(sub)} rows, expected {len(expected)}")
        if set(sub.item_id) != set(expected.item_id):
            errs.append(f"{split}/{c}: item ids differ from 'real' on the same scope")
        if c in PRIOR and (~sub.task.str.contains("Temporal", case=False)).any():
            errs.append(f"{split}/{c}: applied to a non-temporal item")
    for c, sub in df.groupby("condition"):
        mcq = sub[sub.is_mcq.astype(bool)]
        opn = sub[~sub.is_mcq.astype(bool)]
        if mcq.pred_choice_norm.isna().any():
            errs.append(f"{split}/{c}: {int(mcq.pred_choice_norm.isna().sum())} MCQ rows without a prediction")
        if not mcq.mcq_correct_norm.dropna().isin([0, 1]).all():
            errs.append(f"{split}/{c}: mcq_correct_norm has values other than 0/1")
        if not np.isfinite(mcq.gold_logprob.dropna()).all():
            errs.append(f"{split}/{c}: non-finite gold_logprob")
        if opn.pred_text.isna().any():
            errs.append(f"{split}/{c}: {int(opn.pred_text.isna().sum())} open rows without generated text")
    bad_pair = real[temporal & (real.pairing != "paired_ok")]
    if len(bad_pair):
        errs.append(f"{split}: {len(bad_pair)} temporal rows are not 'paired_ok' (fallback input)")
    return errs, df


def report(df, split):
    real = df[df.condition == "real"].set_index("item_id")
    print(f"\n[{split}] rows per condition: " + ", ".join(f"{c}={n}" for c, n in df.condition.value_counts().items()))
    print(f"[{split}] items per task (real): " + ", ".join(f"{t.split('_')[0]}={n}" for t, n in real.task.value_counts().sort_index().items()))
    print(f"[{split}] share of MCQ predictions equal to the 'real' prediction, and accuracy, per condition and task:")
    for c in [x for x in df.condition.unique() if x != "real"]:
        sub = df[(df.condition == c) & df.is_mcq.astype(bool)].set_index("item_id")
        for t, g in sub.groupby("task"):
            r = real.loc[g.index]
            same = (g.pred_choice_norm == r.pred_choice_norm).mean()
            print(f"   {c:<11}{t.split('_')[0]:<7} n={len(g):<4} same-as-real={same:5.2f}  "
                  f"acc real={r.mcq_correct_norm.mean():.3f} {c}={g.mcq_correct_norm.mean():.3f}")




def find_script():
    hits = [p for p in glob.glob("/kaggle/input/**/kaggle_inference_audit.py", recursive=True)]
    good = [p for p in hits if "--controls" in open(p, errors="ignore").read()]
    print("audit scripts found:", hits)
    if not good:
        sys.exit("FAIL: no kaggle_inference_audit.py with --controls under /kaggle/input. "
                 "Upload inference_audit/kaggle_inference_audit.py (not the old retrain/ copy).")
    pick = sorted(good, key=lambda p: ("audit_upload" not in p, p))[0]
    print("using:", pick)
    return pick


def run(cmd):
    print("\n$", " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", default=None, help="only check an existing output folder (laptop use)")
    a, _ = ap.parse_known_args()
    if a.check_only:
        errs, df = check_split(a.check_only, "val", CONTROLS)
        print("PASS" if not errs else "FAIL\n  - " + "\n  - ".join(errs))
        return 0 if not errs else 1

    if run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft"]) != 0:
        sys.exit("FAIL: pip install peft")
    script = find_script()
    if run([sys.executable, script, "--preflight", "--output_dir", OUT, "--controls"] + CONTROLS) != 0:
        sys.exit("FAIL: preflight. Read the log above; do not run the smoke test.")
    rc = run([sys.executable, script, "--limit", "60", "--splits", "val", "--output_dir", OUT,
              "--controls"] + CONTROLS)
    print("\nsmoke test exit code:", rc)
    errs, df = check_split(OUT, "val", CONTROLS)
    if df is not None and not errs:
        report(df, "val")
    if rc != 0 or errs:
        print("\nFAIL")
        for e in errs:
            print("  -", e)
        print("Send me the last 30 lines of the log above (or run_log.txt).")
    else:
        print("\nPASS: all conditions ran and the outputs are consistent.")
    keep = ["per_item_val.csv", "per_task_val.csv", "inference_audit_summary.json", "run_log.txt"]
    z = "/kaggle/working/smoke_out.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in keep:
            p = os.path.join(OUT, f)
            if os.path.exists(p):
                zf.write(p, f)
    print("bring back:", z, "(Output panel -> download). Unzip into inference_audit/smoke_out/")
    return 0 if (rc == 0 and not errs) else 1


if __name__ == "__main__":
    main()
