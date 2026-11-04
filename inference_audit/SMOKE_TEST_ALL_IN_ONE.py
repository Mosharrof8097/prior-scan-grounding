#!/usr/bin/env python3
"""
ALL-IN-ONE KAGGLE SMOKE TEST for the 3D-RAD inference audit.  Paste this WHOLE file into ONE cell.

Needs (attached to the notebook): the 3D-RAD data dataset and the v7 epoch-1 checkpoint dataset.
GPU T4 x1, Internet ON (the Qwen base model is downloaded).

It does: install peft -> unpack the embedded audit script -> preflight -> smoke test
(60 items per split = 10 per task, val only, real + blank + 5 controls) -> check the outputs ->
write ONE zip:   /kaggle/working/smoke_results.zip      (download it from the Output panel)

The zip holds everything needed to judge the run: per_item_val.csv, per_task_val.csv,
inference_audit_summary.json, run_log.txt, the split manifests, console_log.txt (all screen output
including the preflight and the PASS/FAIL verdict) and the exact audit script that was run.
Accuracy numbers from 60 items mean nothing; only PASS/FAIL matters.
"""
import argparse, base64, glob, hashlib, os, subprocess, sys, zlib, zipfile
import numpy as np
import pandas as pd

AUDIT_SHA256 = "cc8c5bc5623b4087b22c0c503246695c54c432ecebed05aae323d825e2fbda62"
CONTROLS = ["mean", "noise", "swap", "zero_prior", "swap_prior"]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "inference_audit")
SCRIPT = os.path.join(WORK, "kaggle_inference_audit.py")
CONSOLE = os.path.join(WORK, "console_log.txt")

AUDIT_B64 = (
    "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"
    "2UzfUSiOnzv/F5nkivY="
)


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




def say(msg=""):
    print(msg, flush=True)
    with open(CONSOLE, "a") as fh:
        fh.write(msg + "\n")


def run(cmd):
    say("\n$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(CONSOLE, "a") as fh:
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); fh.write(line)
    return p.wait()


def unpack():
    raw = zlib.decompress(base64.b64decode("".join(AUDIT_B64)))
    assert hashlib.sha256(raw).hexdigest() == AUDIT_SHA256, "embedded audit script is corrupted"
    open(SCRIPT, "wb").write(raw)
    say(f"audit script unpacked to {SCRIPT} (sha256 {AUDIT_SHA256[:12]}...)")


def make_zip():
    z = os.path.join(WORK, "smoke_results.zip")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(glob.glob(os.path.join(OUT, "*"))):
            zf.write(p, "inference_audit/" + os.path.basename(p))
        for p in (CONSOLE, SCRIPT):
            if os.path.exists(p):
                zf.write(p, os.path.basename(p))
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", default=None, help="laptop use: only check an existing output folder")
    ap.add_argument("--only-preflight", action="store_true", help="laptop test: unpack + preflight, no GPU")
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--data_dir"); ap.add_argument("--ckpt_dir"); ap.add_argument("--split_manifests")
    a, _ = ap.parse_known_args()
    if a.check_only:
        errs, df = check_split(a.check_only, "val", CONTROLS)
        print("PASS" if not errs else "FAIL\n  - " + "\n  - ".join(errs))
        return 0 if not errs else 1

    open(CONSOLE, "w").close()
    extra = []
    for k in ("data_dir", "ckpt_dir", "split_manifests"):
        if getattr(a, k):
            extra += ["--" + k, getattr(a, k)]
    if not a.no_pip and run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft"]) != 0:
        say("FAIL: pip install peft"); make_zip(); return 1
    unpack()
    base = [sys.executable, SCRIPT, "--output_dir", OUT] + extra
    if run(base + ["--preflight", "--controls"] + CONTROLS) != 0:
        say("FAIL: preflight. The smoke test was not run."); z = make_zip(); say("zip: " + z); return 1
    if a.only_preflight:
        say("preflight OK (--only-preflight)"); return 0
    rc = run(base + ["--limit", "60", "--splits", "val", "--controls"] + CONTROLS)
    say(f"\nsmoke test exit code: {rc}")
    errs, df = check_split(OUT, "val", CONTROLS)
    if df is not None and not errs:
        report(df, "val")
    ok = rc == 0 and not errs
    say("\nPASS: every condition ran and the outputs are consistent." if ok else "\nFAIL")
    for e in errs:
        say("  - " + e)
    z = make_zip()
    say(f"\nDOWNLOAD THIS ONE FILE: {z}   (Output panel on the right -> smoke_results.zip)")
    return 0 if ok else 1


if __name__ == "__main__":
    main()
