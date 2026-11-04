#!/usr/bin/env python3
"""Builds inference_audit/SMOKE_TEST_ALL_IN_ONE.py: one file to paste into one Kaggle cell.
It embeds kaggle_inference_audit.py (compressed) and the output checker, so nothing but the two
Kaggle datasets (3D-RAD data, v7 epoch-1 checkpoint) is needed. Re-run this after editing either source."""
import base64, hashlib, zlib, os
HERE = os.path.dirname(os.path.abspath(__file__))
audit = open(os.path.join(HERE, "kaggle_inference_audit.py"), "rb").read()
assert b"--controls" in audit
blob = base64.b64encode(zlib.compress(audit, 9)).decode()
sha = hashlib.sha256(audit).hexdigest()
chk = open(os.path.join(HERE, "check_smoke_output.py")).read()
chk = chk[chk.index("PRIOR = "):chk.index("def main():")]
lines = [blob[i:i + 100] for i in range(0, len(blob), 100)]
embedded = "AUDIT_B64 = (\n" + "\n".join(f'    "{l}"' for l in lines) + "\n)\n"

TEMPLATE = r'''#!/usr/bin/env python3
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

AUDIT_SHA256 = "@@SHA@@"
CONTROLS = ["mean", "noise", "swap", "zero_prior", "swap_prior"]
WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
OUT = os.path.join(WORK, "inference_audit")
SCRIPT = os.path.join(WORK, "kaggle_inference_audit.py")
CONSOLE = os.path.join(WORK, "console_log.txt")

@@EMBED@@

@@CHECK@@

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
'''
out = TEMPLATE.replace("@@SHA@@", sha).replace("@@EMBED@@", embedded).replace("@@CHECK@@", chk)
open(os.path.join(HERE, "SMOKE_TEST_ALL_IN_ONE.py"), "w").write(out)
print("written", len(out) // 1024, "KB; audit sha256", sha[:12])
