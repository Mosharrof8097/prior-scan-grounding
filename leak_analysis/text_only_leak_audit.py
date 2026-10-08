#!/usr/bin/env python3
"""
3D-RAD text-only leakage audit  (zero GPU, CPU only)
====================================================
Question: how much of 3D-RAD can be answered WITHOUT looking at the CT volume?

Three text-only baselines per task, all on a patient-level split (no leakage):
  1. MAJORITY   - always predict the task's most frequent answer
  2. BRACKET    - Task 6 only: use the [0,1,..] past-state sequence in the question text
  3. TFIDF+LR   - logistic regression on question text alone

Any accuracy these reach is accuracy a vision-language model needs NO vision for.
Only 3D-RAD's released image subset (Images/test.zip -> 3,039 volumes) is used,
because the train-split volumes are not published on HuggingFace.
"""
import glob, os, re, json, warnings
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 42
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")


def load_qa():
    """Load every QA csv that has a matching local .npy volume (same rule the training pipeline uses)."""
    vols = {os.path.basename(p)[:-4] for p in
            glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/Images/**/*.npy"), recursive=True)}
    frames = []
    for f in glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/*/Task*/*.csv")):
        d = pd.read_csv(f)
        if "VolumeName" not in d.columns or "Question" not in d.columns:
            continue
        d["TaskCategory"] = os.path.basename(os.path.dirname(f))
        d["Subtask"] = os.path.basename(f)[:-4]
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["NpyName"] = df["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    df = df[df["NpyName"].isin(vols)].copy()
    df["PatientID"] = df["NpyName"].str.split("_").str[:2].str.join("_")
    return df


def patient_split(df, train_ratio=0.80):
    """Same patient-level grouping and seed as the training pipeline."""
    pats = df["PatientID"].unique()
    rng = np.random.RandomState(SEED)
    rng.shuffle(pats)
    n_tr = int(len(pats) * train_ratio)
    train_p = set(pats[:n_tr])
    return df[df["PatientID"].isin(train_p)], df[~df["PatientID"].isin(train_p)]


def bracket_feature(q):
    m = BRACKET_RE.search(str(q))
    if not m:
        return None
    seq = [int(x) for x in m.group(1).replace(" ", "").split(",")]
    return seq


def run_task(name, tr, te):
    out = {"task": name, "n_train": len(tr), "n_test": len(te),
           "n_unique_answers": int(tr["Answer"].nunique())}

    # --- 1. majority class
    maj = tr["Answer"].value_counts().idxmax()
    out["majority_acc"] = float((te["Answer"] == maj).mean())
    out["majority_label"] = maj

    # --- 2. bracket-only (Task 6)
    tr_b = tr["Question"].map(bracket_feature)
    if tr_b.notna().mean() > 0.99:
        # feature = (does the past sequence contain a 1?), predict per-group majority
        key_tr = tr_b.map(lambda s: any(s))
        lut = tr.groupby(key_tr)["Answer"].agg(lambda s: s.value_counts().idxmax())
        key_te = te["Question"].map(bracket_feature).map(lambda s: any(s) if s else False)
        pred = key_te.map(lut).fillna(maj)
        out["bracket_only_acc"] = float((pred.values == te["Answer"].values).mean())
        # how many of the 4 options does the bracket eliminate?
        reach = tr.groupby(key_tr)["Answer"].nunique().to_dict()
        out["answers_reachable_per_bracket_group"] = {str(k): int(v) for k, v in reach.items()}
    else:
        out["bracket_only_acc"] = None

    # --- 3. tf-idf + logistic regression on question text only
    clf = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0),
    )
    clf.fit(tr["Question"].astype(str), tr["Answer"].astype(str))
    out["text_only_acc"] = float(clf.score(te["Question"].astype(str), te["Answer"].astype(str)))
    return out


def main():
    df = load_qa()
    print("=" * 78)
    print("  3D-RAD TEXT-ONLY LEAKAGE AUDIT  (no GPU, no images read)")
    print("=" * 78)
    print(f"QA records with a local volume : {len(df):,}")
    print(f"Unique volumes                 : {df['NpyName'].nunique():,}")
    print(f"Unique patients                : {df['PatientID'].nunique():,}")

    tr_all, te_all = patient_split(df)
    assert not (set(tr_all["PatientID"]) & set(te_all["PatientID"])), "patient leakage!"
    print(f"Patient-level split (seed {SEED})  : train {len(tr_all):,} / test {len(te_all):,}  "
          f"[0 patient overlap verified]")

    results = []
    for task in sorted(df["TaskCategory"].unique()):
        tr, te = tr_all[tr_all.TaskCategory == task], te_all[te_all.TaskCategory == task]
        if len(te) < 20:
            continue
        results.append(run_task(task, tr, te))

    print("\n" + "=" * 78)
    print("  PER-TASK TEXT-ONLY ACCURACY  (how far you get without any CT scan)")
    print("=" * 78)
    hdr = f"{'Task':<34}{'n_test':>7}{'#ans':>6}{'major':>8}{'brack':>8}{'text':>8}"
    print(hdr); print("-" * len(hdr))
    for r in results:
        b = f"{r['bracket_only_acc']:.3f}" if r["bracket_only_acc"] is not None else "  -  "
        print(f"{r['task']:<34}{r['n_test']:>7}{r['n_unique_answers']:>6}"
              f"{r['majority_acc']:>8.3f}{b:>8}{r['text_only_acc']:>8.3f}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "text_only_leak_audit.json"), "w") as fh:
        json.dump({"n_qa": len(df), "n_patients": int(df["PatientID"].nunique()),
                   "seed": SEED, "results": results}, fh, indent=2)
    print("\n[saved] leak_analysis/text_only_leak_audit.json")


if __name__ == "__main__":
    main()
