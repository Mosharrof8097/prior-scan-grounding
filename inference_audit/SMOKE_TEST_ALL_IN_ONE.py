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

AUDIT_SHA256 = "70f3d9444f4ea16fcc0294480cac4a46a08b8394068cb03fcb0053f14da03f7e"
CONTROLS = ["mean", "noise", "swap", "zero_prior", "swap_prior"]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "inference_audit")
SCRIPT = os.path.join(WORK, "kaggle_inference_audit.py")
CONSOLE = os.path.join(WORK, "console_log.txt")

AUDIT_B64 = (
    "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"
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
