#!/usr/bin/env python3
"""Builds inference_audit/ALL_EPOCHS_FULL_AUDIT.py: ONE file, no flags, audits epochs 1, 2 and 3 (val AND test,
7 conditions each) on Kaggle. Embeds the current kaggle_inference_audit.py (with_item_ids + majority_answer fixes),
the output checker, and the SHA-256 of every expected checkpoint/manifest file.

    python inference_audit/build_all_epochs.py
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
ALL-EPOCHS FULL AUDIT.  Paste this WHOLE file into ONE Kaggle cell and run it.  Nothing to edit, no flags.

Attach to the notebook: the 3D-RAD data dataset and every checkpoint dataset you have (epoch_1, epoch_2, epoch_3;
the datasets that hold the split manifests are found automatically).  GPU T4 x1, Internet ON.

What it does, by itself:
  1. finds every epoch (epoch_1, epoch_2, epoch_3) among the attached datasets and verifies each file against an
     embedded SHA-256 and the epoch number stored inside the checkpoint; a missing or corrupted epoch is reported
     and skipped, the others still run
  2. preflight (CPU): split matches the trained one, no volume missing, every Task 5 / Task 6 item has an earlier scan
  3. for each epoch, for val and then test: the full audit with 7 conditions
        real, blank, mean, noise, swap, zero_prior, swap_prior      (prior controls on Task 5 and Task 6 items only)
  4. after EVERY split: checks the outputs (7 conditions, full row counts, all temporal items paired, the intended
     epoch was loaded) and rewrites ONE zip:   /kaggle/working/full_audit_results.zip
  5. stops cleanly BEFORE a split that would not finish inside the session time budget, and says how to continue

Download ONE file at the end:  full_audit_results.zip   (Output panel).
It holds: per_item_*.csv and per_task_*.csv and inference_audit_*.json and run_log.txt for every epoch and split,
summary.csv (headline accuracy by epoch, split, task, condition), status.json, run_info.json (software versions, GPU,
hashes, timings), console_log.txt (all screen output) and the exact audit script that ran.

Continuing in a new session (only if the time budget stopped the run): upload full_audit_results.zip as a Kaggle
dataset, attach it together with the other datasets, and run this same file again. Finished (epoch, split) pairs
that pass every check are reused, not rerun.

Time: about 3 to 5 hours per epoch on a T4 (35,877 forward passes per epoch, measured 0.28 to 0.5 s each).
Unattended: Save Version -> Save & Run All (Commit), GPU on.  Nothing here changes any existing result.
"""
import argparse, base64, glob, hashlib, json, os, shutil, subprocess, sys, time, zipfile, zlib
import numpy as np
import pandas as pd

EPOCHS = ["epoch_1", "epoch_2", "epoch_3"]
SPLITS = ["val", "test"]
CONTROLS = ["mean", "noise", "swap", "zero_prior", "swap_prior"]       # + real + blank = 7 conditions
CONDITIONS = ["real", "blank"] + CONTROLS
# rows each full split must have (manifest counts) and how many of those are temporal (Task 5 + Task 6)
EXPECT = {"val": {"n": 3254, "temporal": 476}, "test": {"n": 3459, "temporal": 680}}
BUDGET_HOURS = 11.0          # stop before a split that cannot finish inside this wall-clock budget
HASHES = @@HASHES@@
AUDIT_SHA256 = "@@SHA@@"

@@EMBED@@

@@CHECK@@

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
RUN = os.path.join(WORK, "full_audit")
SCRIPT = os.path.join(WORK, "kaggle_inference_audit.py")
CONSOLE = os.path.join(WORK, "console_log.txt")
ZIP = os.path.join(WORK, "full_audit_results.zip")
T_START = time.time()


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


def locate_manifests(root):
    for p in sorted(glob.glob(os.path.join(root, "**", "split_train_qa.csv"), recursive=True)):
        d = os.path.dirname(p)
        if all(os.path.exists(os.path.join(d, f"split_{s}_qa.csv")) for s in ("val", "test")) and \
                all(sha256(os.path.join(d, f"split_{s}_qa.csv")) == HASHES["manifests"][s] for s in ("train", "val", "test")):
            return d
    return None


def locate_epoch(root, epoch):
    """Returns (info dict, None) when the epoch is attached and verified, else (None, reason)."""
    cands = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "**", epoch, "multimodal_projector_best.pt"), recursive=True)
             if os.path.isdir(os.path.join(os.path.dirname(p), "lora_adapters"))]
    if not cands:
        return None, f"{epoch}: not attached (no folder '{epoch}' with projector + lora_adapters/)"
    ckpt = sorted(cands)[0]
    want = HASHES[epoch]
    got = {"projector": sha256(os.path.join(ckpt, "multimodal_projector_best.pt")),
           "adapter": sha256(os.path.join(ckpt, "lora_adapters", "adapter_model.safetensors")),
           "adapter_config": sha256(os.path.join(ckpt, "lora_adapters", "adapter_config.json"))}
    bad = [k for k in want if got[k] != want[k]]
    if bad:
        return None, f"{epoch}: files at {ckpt} do not match the expected SHA-256 for {bad} (corrupted or wrong upload)"
    import torch
    meta = torch.load(os.path.join(ckpt, "multimodal_projector_best.pt"), map_location="cpu", weights_only=False)
    no = int(epoch.split("_")[1])
    if int(meta.get("epoch", -1)) != no:
        return None, f"{epoch}: the projector inside says it is epoch {meta.get('epoch')}"
    return {"epoch": epoch, "no": no, "ckpt": ckpt, "val_loss": float(meta["val_loss"]), "sha": got}, None


def n_items(split):
    return 5 * EXPECT[split]["n"] + 2 * EXPECT[split]["temporal"]


def check_full(outdir, split, epoch_no):
    """check_split (7 conditions, same items as real, no missing predictions, temporal paired) plus full row counts
    and proof from the audit's own log that the intended epoch was loaded."""
    errs, df = check_split(outdir, split, CONTROLS)
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
    log = os.path.join(outdir, "run_log.txt")
    if not os.path.exists(log) or f"(epoch {epoch_no}, val_loss" not in open(log, errors="ignore").read():
        errs.append(f"{split}: run_log.txt does not show that epoch {epoch_no} was loaded")
    return errs, df


def find_earlier(root, epoch, split, epoch_no):
    """A finished (epoch, split) from an earlier session: in this session's folder or an attached dataset."""
    for base in [RUN] + sorted(glob.glob(os.path.join(root, "**", "full_audit"), recursive=True)):
        d = os.path.join(base, epoch, split)
        if os.path.exists(os.path.join(d, f"per_item_{split}.csv")):
            errs, _ = check_full(d, split, epoch_no)
            if not errs:
                return d
    return None


def summary_table():
    frames = []
    for e in EPOCHS:
        for s in SPLITS:
            p = os.path.join(RUN, e, s, f"per_task_{s}.csv")
            if os.path.exists(p):
                t = pd.read_csv(p)
                t.insert(0, "split", s); t.insert(0, "epoch", e)
                frames.append(t[["epoch", "split", "task", "condition", "n", "headline_acc", "text_only_floor",
                                 "gain_over_text_floor", "loss", "tf_token_acc"]])
    return pd.concat(frames, ignore_index=True) if frames else None


def write_zip(status, info):
    st = summary_table()
    if st is not None:
        st.to_csv(os.path.join(RUN, "summary.csv"), index=False)
    json.dump(status, open(os.path.join(RUN, "status.json"), "w"), indent=2)
    json.dump(info, open(os.path.join(RUN, "run_info.json"), "w"), indent=2)
    keep = ("per_item_", "per_task_", "inference_audit_", "run_log")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for e in EPOCHS:
            for s in SPLITS:
                for p in sorted(glob.glob(os.path.join(RUN, e, s, "*"))):
                    if os.path.basename(p).startswith(keep):
                        zf.write(p, f"full_audit/{e}/{s}/{os.path.basename(p)}")
        for n in ("summary.csv", "status.json", "run_info.json"):
            if os.path.exists(os.path.join(RUN, n)):
                zf.write(os.path.join(RUN, n), f"full_audit/{n}")
        for p in (CONSOLE, SCRIPT):
            if os.path.exists(p):
                zf.write(p, os.path.basename(p))
    return ZIP


def versions():
    out = {}
    try:
        from importlib import metadata
        for m in ("torch", "transformers", "peft", "numpy", "pandas"):
            out[m] = metadata.version(m)
    except Exception as ex:
        out["error"] = str(ex)
    try:
        import torch
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        pass
    out["python"] = sys.version.split()[0]
    return out


def main():
    ap = argparse.ArgumentParser()      # every flag is optional; the defaults are what you want on Kaggle
    ap.add_argument("--input-root", default="/kaggle/input")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-preflight", action="store_true")
    ap.add_argument("--no-pip", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--budget-hours", type=float, default=BUDGET_HOURS)
    a, _ = ap.parse_known_args()

    os.makedirs(RUN, exist_ok=True)
    open(CONSOLE, "w").close()
    say(f"ALL-EPOCHS FULL AUDIT   epochs={EPOCHS}   splits={SPLITS}   conditions={CONDITIONS}")
    man = locate_manifests(a.input_root)
    if man is None:
        sys.exit("FAIL: no split manifests (split_{train,val,test}_qa.csv) with the expected hashes under " + a.input_root +
                 "\n  Attach a checkpoint dataset (they contain the manifests).")
    say(f"split manifests OK: {man}  (hashes match the trained split)")
    found, problems = {}, []
    for e in EPOCHS:
        info, why = locate_epoch(a.input_root, e)
        if info:
            found[e] = info
            say(f"{e}: OK  {info['ckpt']}  (val_loss {info['val_loss']:.4f}; hashes match)")
        else:
            problems.append(why)
            say("WARNING " + why)
    if not found:
        sys.exit("FAIL: no usable epoch among the attached datasets.")
    status = {e: {s: {"state": "not reached"} for s in SPLITS} for e in EPOCHS}
    for e in EPOCHS:
        if e not in found:
            for s in SPLITS:
                status[e][s] = {"state": "skipped: epoch not available"}
    info_all = {"audit_sha256": AUDIT_SHA256, "versions": versions(), "manifests": man,
                "checkpoints": {e: {k: v for k, v in found[e].items() if k != 'sha'} | {"sha256": found[e]["sha"]} for e in found},
                "problems": problems, "expect": EXPECT, "started_unix": T_START}
    unpack()
    extra = ["--split_manifests", man] + (["--data_dir", a.data_dir] if a.data_dir else [])
    est_h = sum(n_items(s) for s in SPLITS) * len(found) * 0.4 / 3600
    say(f"\nplan: {len(found)} epoch(s) x {len(SPLITS)} splits x 7 conditions = "
        f"{sum(n_items(s) for s in SPLITS) * len(found):,} forward passes, estimated {est_h:.1f} h (0.4 s/item); budget {a.budget_hours:.0f} h")
    if a.dry_run:
        say("DRY RUN, nothing executed.")
        return 0 if not problems else 1
    if not a.no_pip and run([sys.executable, "-m", "pip", "install", "-q", "-U", "peft"]) != 0:
        say("FAIL: pip install peft"); return 1
    first = found[sorted(found)[0]]
    if run([sys.executable, SCRIPT, "--preflight", "--output_dir", os.path.join(RUN, "preflight"),
            "--ckpt_dir", first["ckpt"], "--controls"] + CONTROLS + extra) != 0:
        say("FAIL: preflight. No GPU time was spent."); write_zip(status, info_all); return 1
    if a.only_preflight:
        say("preflight OK (--only-preflight)"); return 0

    rate = 0.5                                     # seconds per item; replaced by what is measured
    stopped_for_time = None
    for e in EPOCHS:
        if e not in found:
            continue
        ep = found[e]
        for s in SPLITS:
            prev = find_earlier(a.input_root, e, s, ep["no"])
            if prev:
                if os.path.abspath(prev) != os.path.abspath(os.path.join(RUN, e, s)):
                    shutil.copytree(prev, os.path.join(RUN, e, s), dirs_exist_ok=True)
                status[e][s] = {"state": "done (reused from an earlier session)"}
                say(f"\n[{e}/{s}] finished earlier and passes every check; reused")
                write_zip(status, info_all)
                continue
            need_h = n_items(s) * rate / 3600
            elapsed_h = (time.time() - T_START) / 3600
            if elapsed_h + need_h > a.budget_hours:
                stopped_for_time = (e, s, elapsed_h, need_h)
                break
            out = os.path.join(RUN, e, s)
            os.makedirs(out, exist_ok=True)
            say(f"\n===== {e} / {s}   ({n_items(s):,} passes, ~{need_h:.1f} h, {elapsed_h:.1f} h used) =====")
            t0 = time.time()
            rc = run([sys.executable, SCRIPT, "--splits", s, "--output_dir", out, "--ckpt_dir", ep["ckpt"],
                      "--controls"] + CONTROLS + extra)
            secs = time.time() - t0
            errs, df = check_full(out, s, ep["no"])
            say(f"\n[{e}/{s}] exit code {rc}, {secs / 60:.0f} min")
            if rc != 0 or errs:
                status[e][s] = {"state": "FAILED", "errors": errs, "exit_code": rc}
                say(f"\nFAIL on {e}/{s}:")
                for x in errs:
                    say("  - " + x)
                say("Stopping so no more GPU time is spent. Partial results: " + write_zip(status, info_all))
                say("Send me the last 30 lines of the output above, or the zip.")
                return 1
            rate = max(secs / n_items(s) * 1.15, 0.2)
            report(df, s)
            status[e][s] = {"state": "PASS", "minutes": round(secs / 60, 1)}
            say(f"[{e}/{s}] PASS: 7 conditions, {EXPECT[s]['n']} items, {EXPECT[s]['temporal']} temporal items paired")
            say("zip so far: " + write_zip(status, info_all))
        if stopped_for_time:
            break

    z = write_zip(status, info_all)
    if stopped_for_time:
        e, s, eh, nh = stopped_for_time
        say(f"\nTIME BUDGET: {eh:.1f} h used; {e}/{s} needs about {nh:.1f} h more, so it was not started.")
        say("Everything finished so far is in the zip. To continue: upload " + os.path.basename(z) +
            " as a Kaggle dataset, attach it with the other datasets in a NEW session, and run this same file again.")
    done = [(e, s) for e in EPOCHS for s in SPLITS if status[e][s]["state"].startswith(("PASS", "done"))]
    say(f"\nfinished: {len(done)} of {len(EPOCHS) * len(SPLITS)} (epoch, split) pairs  in {(time.time() - T_START) / 3600:.1f} h")
    for e in EPOCHS:
        say(f"  {e}: " + ", ".join(f"{s}={status[e][s]['state']}" for s in SPLITS))
    for p in problems:
        say("  problem: " + p)
    say(f"\nDOWNLOAD THIS ONE FILE: {z}   (Output panel -> full_audit_results.zip)")
    return 0 if (len(done) == len(EPOCHS) * len(SPLITS)) else 1


if __name__ == "__main__":
    main()
'''

out = (TEMPLATE.replace("@@HASHES@@", json.dumps(hashes, indent=4))
       .replace("@@SHA@@", hashlib.sha256(audit).hexdigest())
       .replace("@@EMBED@@", embedded).replace("@@CHECK@@", chk))
path = os.path.join(HERE, "ALL_EPOCHS_FULL_AUDIT.py")
open(path, "w").write(out)
print(f"written {path} ({len(out) // 1024} KB); audit sha256 {hashlib.sha256(audit).hexdigest()[:12]}")
