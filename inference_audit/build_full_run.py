#!/usr/bin/env python3
"""Builds inference_audit/FULL_RUN_ALL_IN_ONE.py: the full audit (val AND test, 7 conditions) for ONE epoch,
as a single file to paste into one Kaggle cell. Embeds the current kaggle_inference_audit.py, the output
checker, and the SHA-256 of every checkpoint/manifest file it expects, so a wrong or corrupted upload is
caught before any GPU time is spent.  Re-run after editing kaggle_inference_audit.py or check_smoke_output.py.

    python inference_audit/build_full_run.py
"""
import base64, hashlib, json, os, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
BK = os.path.join(ROOT, "backups", "retrain_v7_2026-11-04", "retrain_v7")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


audit = open(os.path.join(HERE, "kaggle_inference_audit.py"), "rb").read()
assert b"--controls" in audit and b"def with_item_ids" in audit and b"def majority_answer" in audit, \
    "audit script is missing one of the two bug fixes"
blob = base64.b64encode(zlib.compress(audit, 9)).decode()
embedded = "AUDIT_B64 = (\n" + "\n".join(f'    "{blob[i:i + 100]}"' for i in range(0, len(blob), 100)) + "\n)\n"
chk = open(os.path.join(HERE, "check_smoke_output.py")).read()
chk = chk[chk.index("PRIOR = "):chk.index("def main():")]

hashes = {"manifests": {s: sha(os.path.join(BK, f"split_{s}_qa.csv")) for s in ("train", "val", "test")}}
for e in ("epoch_1", "epoch_2", "epoch_3"):
    hashes[e] = {
        "projector": sha(os.path.join(BK, e, "multimodal_projector_best.pt")),
        "adapter": sha(os.path.join(BK, e, "lora_adapters", "adapter_model.safetensors")),
        "adapter_config": sha(os.path.join(BK, e, "lora_adapters", "adapter_config.json")),
    }

TEMPLATE = r'''#!/usr/bin/env python3
"""
FULL AUDIT, ONE EPOCH  --  val AND test, 7 conditions (real, blank, mean, noise, swap, zero_prior, swap_prior).
Paste this WHOLE file into ONE Kaggle cell.  Change only the EPOCH line below to audit another epoch.

Attach to the notebook: the 3D-RAD data dataset, and the checkpoint dataset that holds the epoch you chose
(epoch_1: audit_upload; epochs 2 and 3: audit_upload_epochs_2_3).  GPU T4 x1, Internet ON.

Before any GPU work it verifies the checkpoint, adapter and split-manifest files against embedded SHA-256
hashes and the epoch recorded inside the checkpoint.  It then runs val and test as two separate passes, checks
each (all 7 conditions present, row counts equal the full split, every temporal item paired), and writes
/kaggle/working/full_run_<EPOCH>.zip after EACH split, so a dropped session still leaves downloadable results.
If a split fails its checks, it stops and does not spend GPU time on the next one.

Estimated time per epoch: about 3 to 5 hours on a T4 (35,877 forward passes; measured 0.28 to 0.5 s/item).
Unattended: use Save Version -> Save & Run All (Commit) with GPU on, so the run survives a closed tab.

Nothing here writes to or replaces any existing result; outputs go to /kaggle/working/full_<EPOCH>/.
"""
EPOCH = "epoch_1"          # <-- the only line to edit: epoch_1, epoch_2 or epoch_3

import argparse, base64, glob, hashlib, json, os, subprocess, sys, time, zipfile, zlib
import numpy as np
import pandas as pd

CONTROLS = ["mean", "noise", "swap", "zero_prior", "swap_prior"]       # + real + blank = 7 conditions
CONDITIONS = ["real", "blank"] + CONTROLS
# rows the full splits must have (manifest counts) and how many of those are temporal (Task 5 + Task 6)
EXPECT = {"val": {"n": 3254, "temporal": 476}, "test": {"n": 3459, "temporal": 680}}
HASHES = @@HASHES@@
AUDIT_SHA256 = "@@SHA@@"

@@EMBED@@

@@CHECK@@

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
RUN = os.path.join(WORK, f"full_{EPOCH}")
SCRIPT = os.path.join(WORK, "kaggle_inference_audit.py")
CONSOLE = os.path.join(WORK, f"console_log_{EPOCH}.txt")


def say(msg=""):
    print(msg, flush=True)
    with open(CONSOLE, "a") as fh:
        fh.write(msg + "\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd):
    say("\n$ " + " ".join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    with open(CONSOLE, "a") as fh:
        for line in p.stdout:
            sys.stdout.write(line); sys.stdout.flush(); fh.write(line)
    return p.wait()


def unpack():
    raw = zlib.decompress(base64.b64decode("".join(AUDIT_B64)))
    if hashlib.sha256(raw).hexdigest() != AUDIT_SHA256:
        sys.exit("FAIL: embedded audit script is corrupted (paste the whole file again)")
    open(SCRIPT, "wb").write(raw)
    say(f"audit script unpacked: {SCRIPT} (sha256 {AUDIT_SHA256[:12]}...)")


def find_inputs(root):
    """Locate and verify the checkpoint dir for EPOCH and the split manifests. Exits with a clear message."""
    want = HASHES[EPOCH]
    cands = []
    for p in glob.glob(os.path.join(root, "**", EPOCH, "multimodal_projector_best.pt"), recursive=True):
        d = os.path.dirname(p)
        if os.path.isdir(os.path.join(d, "lora_adapters")):
            cands.append(d)
    if not cands:
        seen = sorted(set(os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", "epoch_*", "multimodal_projector_best.pt"), recursive=True)))
        sys.exit(f"FAIL: no checkpoint folder '{EPOCH}' (projector + lora_adapters/) under {root}.\n"
                 f"  epoch folders that ARE attached: {seen or 'none'}\n"
                 f"  Attach the dataset that holds {EPOCH} (epoch_1: audit_upload, epochs 2-3: audit_upload_epochs_2_3).")
    ckpt = sorted(cands)[0]
    if len(cands) > 1:
        say(f"note: {len(cands)} folders named {EPOCH}; using {ckpt}")
    got = {"projector": sha256(os.path.join(ckpt, "multimodal_projector_best.pt")),
           "adapter": sha256(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors")),
           "adapter_config": sha256(os.path.join(ckpt, "lora_adapters", "adapter_config.json"))}
    bad = [k for k in want if got[k] != want[k]]
    if bad:
        sys.exit(f"FAIL: {EPOCH} files do not match the expected SHA-256 for {bad}. Re-upload the dataset.")
    import torch
    meta = torch.load(os.path.join(ckpt, "multimodal_projector_best.pt"), map_location="cpu", weights_only=False)
    ep = int(EPOCH.split("_")[1])
    if int(meta.get("epoch", -1)) != ep:
        sys.exit(f"FAIL: the projector inside {EPOCH} says it is epoch {meta.get('epoch')}.")
    say(f"checkpoint OK: {ckpt}  (epoch {meta['epoch']}, val_loss {meta['val_loss']:.4f}; hashes match)")

    man = None
    for p in sorted(glob.glob(os.path.join(root, "**", "split_train_qa.csv"), recursive=True)):
        d = os.path.dirname(p)
        if all(os.path.exists(os.path.join(d, f"split_{s}_qa.csv")) for s in ("val", "test")) and \
                all(sha256(os.path.join(d, f"split_{s}_qa.csv")) == HASHES["manifests"][s] for s in ("train", "val", "test")):
            man = d
            break
    if man is None:
        sys.exit("FAIL: no split manifests (split_{train,val,test}_qa.csv) with the expected hashes under " + root)
    say(f"split manifests OK: {man}  (hashes match the trained split)")
    return ckpt, man, int(meta["epoch"]), float(meta["val_loss"])


def check_full(split, epoch_no):
    """Everything check_split checks, plus: full row counts, and the checkpoint epoch the audit itself logged."""
    out = os.path.join(RUN, split)
    errs, df = check_split(out, split, CONTROLS)
    if df is None:
        return errs, None
    real = df[df.condition == "real"]
    if len(real) != EXPECT[split]["n"]:
        errs.append(f"{split}: {len(real)} real rows, the full split has {EXPECT[split]['n']}")
    nt = int(real.task.str.contains("Temporal", case=False).sum())
    if nt != EXPECT[split]["temporal"]:
        errs.append(f"{split}: {nt} temporal rows, expected {EXPECT[split]['temporal']}")
    missing = [c for c in CONDITIONS if c not in set(df.condition)]
    if missing:
        errs.append(f"{split}: conditions missing {missing}")
    log = os.path.join(out, "run_log.txt")
    if not os.path.exists(log) or f"(epoch {epoch_no}, val_loss" not in open(log, errors="ignore").read():
        errs.append(f"{split}: run_log.txt does not show that epoch {epoch_no} was loaded")
    return errs, df


def make_zip():
    z = os.path.join(WORK, f"full_run_{EPOCH}.zip")
    keep = ("per_item_", "per_task_", "inference_audit_", "run_log")
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for split in ("val", "test"):
            for p in sorted(glob.glob(os.path.join(RUN, split, "*"))):
                if os.path.basename(p).startswith(keep):
                    zf.write(p, f"full_{EPOCH}/{split}/{os.path.basename(p)}")
        for p in (CONSOLE, SCRIPT):
            if os.path.exists(p):
                zf.write(p, os.path.basename(p))
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--dry-run", action="store_true", help="verify inputs and print the plan; run nothing")
    ap.add_argument("--only-preflight", action="store_true")
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--force", action="store_true", help="rerun a split even if its outputs already pass")
    a, _ = ap.parse_known_args()

    os.makedirs(RUN, exist_ok=True)
    open(CONSOLE, "w").close()
    say(f"FULL AUDIT  epoch={EPOCH}  splits=val,test  conditions={CONDITIONS}")
    ckpt, man, epoch_no, vloss = find_inputs(a.input_root)
    unpack()
    extra = ["--ckpt_dir", ckpt, "--split_manifests", man] + (["--data_dir", a.data_dir] if a.data_dir else [])
    plan = [sys.executable, SCRIPT, "--splits", "<val|test>", "--output_dir", RUN + "/<split>"] + extra + ["--controls"] + CONTROLS
    if a.dry_run:
        say("\nDRY RUN, nothing executed. Plan per split:\n  " + " ".join(plan))
        say(f"expected per split: {EXPECT}")
        return 0
    if not a.no_pip and run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft"]) != 0:
        say("FAIL: pip install peft"); return 1
    if run([sys.executable, SCRIPT, "--preflight", "--output_dir", os.path.join(RUN, "preflight")] + extra + ["--controls"] + CONTROLS) != 0:
        say("FAIL: preflight. No GPU time was spent."); make_zip(); return 1
    if a.only_preflight:
        say("preflight OK (--only-preflight)"); return 0

    t0 = time.time()
    for split in ("val", "test"):
        out = os.path.join(RUN, split)
        errs, _ = check_full(split, epoch_no) if (os.path.exists(os.path.join(out, f"per_item_{split}.csv")) and not a.force) else (["not run"], None)
        if not errs:
            say(f"\n[{split}] outputs from an earlier run in this session already pass every check; skipping (--force to rerun)")
            continue
        os.makedirs(out, exist_ok=True)
        s0 = time.time()
        rc = run([sys.executable, SCRIPT, "--splits", split, "--output_dir", out] + extra + ["--controls"] + CONTROLS)
        say(f"\n[{split}] exit code {rc}, {(time.time() - s0) / 60:.0f} min")
        errs, df = check_full(split, epoch_no)
        if rc != 0 or errs:
            say(f"\nFAIL on {split}:")
            for e in errs:
                say("  - " + e)
            say(f"Stopping so no more GPU time is spent. Partial results: {make_zip()}")
            say("Send me the last 30 lines of the output above, or the zip.")
            return 1
        report(df, split)
        say(f"\n[{split}] PASS: 7 conditions, {EXPECT[split]['n']} items, {EXPECT[split]['temporal']} temporal items paired")
        say("zip so far: " + make_zip())
    z = make_zip()
    say(f"\nALL DONE in {(time.time() - t0) / 3600:.1f} h.  epoch={EPOCH} (val_loss {vloss:.4f})  val: PASS  test: PASS")
    say(f"DOWNLOAD THIS ONE FILE: {z}   (Output panel -> full_run_{EPOCH}.zip)")
    return 0


if __name__ == "__main__":
    main()
'''

out = (TEMPLATE.replace("@@HASHES@@", json.dumps(hashes, indent=4))
       .replace("@@SHA@@", hashlib.sha256(audit).hexdigest())
       .replace("@@EMBED@@", embedded).replace("@@CHECK@@", chk))
path = os.path.join(HERE, "FULL_RUN_ALL_IN_ONE.py")
open(path, "w").write(out)
print(f"written {path} ({len(out) // 1024} KB); audit sha256 {hashlib.sha256(audit).hexdigest()[:12]}")
