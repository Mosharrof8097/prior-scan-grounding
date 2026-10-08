#!/usr/bin/env python3
"""
Compare the trained 3D VLM's reported accuracy against the text-only floor,
using the SAME val split and the SAME single-scan / longitudinal bucketing
that the training pipeline uses (TaskCategory contains 'temporal').

Reported by the checkpoint evaluation (epoch 3, val_loss 0.2853):
    single-scan  generation EM = 0.7667
    longitudinal generation EM = 0.7733
"""
import glob, os, re, json, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 42
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")
REPORTED = {"single": 0.7667, "longitudinal": 0.7733}


def load_qa():
    vols = {os.path.basename(p)[:-4] for p in
            glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/Images/**/*.npy"), recursive=True)}
    frames = []
    for f in glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/*/Task*/*.csv")):
        d = pd.read_csv(f)
        if "VolumeName" not in d.columns:
            continue
        d["TaskCategory"] = os.path.basename(os.path.dirname(f))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["NpyName"] = df["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    df = df[df["NpyName"].isin(vols)].copy()
    df["PatientID"] = df["NpyName"].str.split("_").str[:2].str.join("_")
    # exactly the bucketing used in Medical3DVQADataset.__getitem__
    df["bucket"] = np.where(
        df["TaskCategory"].str.lower().str.contains("longitudinal|temporal"),
        "longitudinal", "single")
    return df


def split_80_10_10(df):
    pats = df["PatientID"].unique()
    rng = np.random.RandomState(SEED); rng.shuffle(pats)
    n = len(pats); n_tr, n_va = int(n * .80), int(n * .10)
    tr, va = set(pats[:n_tr]), set(pats[n_tr:n_tr + n_va])
    return (df[df.PatientID.isin(tr)],
            df[df.PatientID.isin(va)],
            df[~df.PatientID.isin(tr | va)])


def text_only_pred(tr, te):
    """Best text-only rule per task: per-(task, bracket-group) majority answer."""
    def key(d):
        br = d["Question"].map(lambda q: (lambda m: any(int(x) for x in m.group(1).replace(" ", "").split(","))
                                          if m else None)(BRACKET_RE.search(str(q))))
        # NOTE: pandas>=3 turns None into NA under astype(str), which would silently
        # break the lookup key for every task that has no bracket. Map it explicitly.
        br = br.map(lambda v: "nobracket" if v is None else str(bool(v)))
        return d["TaskCategory"].astype(str) + "|" + br
    lut = tr.assign(k=key(tr)).groupby("k")["Answer"].agg(lambda s: s.value_counts().idxmax())
    glob_maj = tr["Answer"].value_counts().idxmax()
    return key(te).map(lut).fillna(glob_maj).values


def main():
    df = load_qa()
    tr, va, te = split_80_10_10(df)
    print("=" * 84)
    print("  TRAINED 3D VLM  vs  TEXT-ONLY FLOOR   (val split, patient-level, seed 42)")
    print("=" * 84)
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")
    print(f"val buckets: {dict(va['bucket'].value_counts())}")
    print("   (pipeline reported: 3254 QA = 476 longitudinal + 2778 single-scan)\n")

    pred = text_only_pred(tr, va)
    va = va.assign(text_pred=pred, text_ok=(pred == va["Answer"].values))

    print(f"{'bucket':<14}{'n':>6}{'text-only':>11}{'VLM (rep.)':>12}{'gain':>9}")
    print("-" * 52)
    rows = []
    for b in ["single", "longitudinal"]:
        sub = va[va.bucket == b]
        t = sub["text_ok"].mean(); v = REPORTED[b]
        print(f"{b:<14}{len(sub):>6}{t:>11.4f}{v:>12.4f}{v - t:>+9.4f}")
        rows.append({"bucket": b, "n": int(len(sub)), "text_only": float(t),
                     "vlm_reported_em": v, "gain_from_vision": float(v - t)})

    print(f"\n{'per-task breakdown of the val split':-<84}")
    print(f"{'Task':<38}{'n':>6}{'#ans':>6}{'text-only':>11}")
    print("-" * 61)
    per = []
    for t_ in sorted(va["TaskCategory"].unique()):
        sub = va[va.TaskCategory == t_]
        print(f"{t_:<38}{len(sub):>6}{sub['Answer'].nunique():>6}{sub['text_ok'].mean():>11.4f}")
        per.append({"task": t_, "n": int(len(sub)),
                    "n_answers_in_val": int(sub["Answer"].nunique()),
                    "text_only": float(sub["text_ok"].mean())})

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "compare_to_baseline.json"), "w") as fh:
        json.dump({"buckets": rows, "per_task": per}, fh, indent=2)
    print("\n[saved] leak_analysis/compare_to_baseline.json")


if __name__ == "__main__":
    main()
