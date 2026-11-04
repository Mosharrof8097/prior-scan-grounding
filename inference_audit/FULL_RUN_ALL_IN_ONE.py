#!/usr/bin/env python3
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
HASHES = {
    "manifests": {
        "train": "2ea865d2e9312610105eec93264b6aa2d49b06a5022b2ab15c5ac24e9c828df2",
        "val": "561fcbf4cd4384d0dccc63ecebceb21280ebdb564f861d617399db7aee77580a",
        "test": "18cedac6d5d82d56b15a24348db5d781888cdf883303ea259d3e1bf5cb167ddc"
    },
    "epoch_1": {
        "projector": "d797c9600af2213057adc777aa9b73a6743471258d5a1cfaf69df56c8262678f",
        "adapter": "8c27f97a8fb83ae9bf9ce8ae2b6ef2a0ca23aebfb8a30d896c3499588cfa7240",
        "adapter_config": "f81e0755603b71a79cf84a670fd510b6604b523f6623d8dc65bb3c0e7bbd72c4"
    },
    "epoch_2": {
        "projector": "7d7a6126caf14567fcc57088c5969b0e400a129b650409ac9e54a31cca5fb420",
        "adapter": "4a2190944f2f7db92fe897193ddcaca9ab507373688846d9fe1c8d3870e1b3ee",
        "adapter_config": "f81e0755603b71a79cf84a670fd510b6604b523f6623d8dc65bb3c0e7bbd72c4"
    },
    "epoch_3": {
        "projector": "97da7e0ad1c4508af0e4d8c7d208095c082d7cf1a60126730f57c39be339220c",
        "adapter": "f0e93867bdf72534753fe74c1265ab89917a768322fd2466a84cb7d18b914dd5",
        "adapter_config": "f81e0755603b71a79cf84a670fd510b6604b523f6623d8dc65bb3c0e7bbd72c4"
    }
}
AUDIT_SHA256 = "294fc43d6d8c932f61d947e0f756854f7a0cfa87db48c74a58aff46a8cc8e229"

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
