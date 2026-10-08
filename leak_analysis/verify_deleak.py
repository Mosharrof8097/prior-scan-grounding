#!/usr/bin/env python3
"""
Verify that 3D-RAD-T actually removes the leak.

For Task 6, on the same patient-level split, measure the text-only floor on:
  original 3D-RAD, 3D-RAD-T/T-mask, 3D-RAD-T/T-strip
and compare against Task 5, which never had a bracket and so acts as the
"already clean" control.

If de-leaking worked, Task 6's text-only floor should fall from ~0.81 to
Task 5's level (plain majority class), with no information left in the text.
"""
import glob, os, re, json, warnings
import pandas as pd, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SEED = 42
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")


def load(root, task_glob):
    vols = {os.path.basename(p)[:-4] for p in
            glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/Images/**/*.npy"), recursive=True)}
    frames = [pd.read_csv(f) for f in glob.glob(os.path.join(root, task_glob))]
    df = pd.concat(frames, ignore_index=True)
    df["NpyName"] = df["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    df = df[df["NpyName"].isin(vols)].copy()
    df["PatientID"] = df["NpyName"].str.split("_").str[:2].str.join("_")
    return df


def split(df):
    pats = df["PatientID"].unique()
    rng = np.random.RandomState(SEED); rng.shuffle(pats)
    n_tr = int(len(pats) * 0.80)
    tr = set(pats[:n_tr])
    return df[df.PatientID.isin(tr)], df[~df.PatientID.isin(tr)]


def floors(df):
    tr, te = split(df)
    maj = tr["Answer"].value_counts().idxmax()
    res = {"n_test": int(len(te)), "majority": float((te["Answer"] == maj).mean())}

    # bracket-only rule
    def brkey(d):
        return d["Question"].map(lambda q: (lambda m: str(any(int(x) for x in m.group(1).replace(" ", "").split(",")))
                                            if m else "nobracket")(BRACKET_RE.search(str(q))))
    k_tr, k_te = brkey(tr), brkey(te)
    if (k_tr != "nobracket").mean() > 0.99:
        lut = tr.groupby(k_tr)["Answer"].agg(lambda s: s.value_counts().idxmax())
        pred = k_te.map(lut).fillna(maj).values
        res["bracket_only"] = float((pred == te["Answer"].values).mean())
    else:
        res["bracket_only"] = None

    clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
                        LogisticRegression(max_iter=2000, C=4.0))
    clf.fit(tr["Question"].astype(str), tr["Answer"].astype(str))
    res["tfidf_lr"] = float(clf.score(te["Question"].astype(str), te["Answer"].astype(str)))
    res["best_text_only"] = max(v for v in
                                [res["majority"], res["bracket_only"], res["tfidf_lr"]] if v is not None)
    return res


def main():
    cfg = [
        ("Task 5 (control, never leaked)", os.path.join(ROOT, "3D-RAD_Dataset"), "*/Task5*/*.csv"),
        ("Task 6 original",                os.path.join(ROOT, "3D-RAD_Dataset"), "*/Task6*/*.csv"),
        ("Task 6 3D-RAD-T / T-mask",       os.path.join(ROOT, "3D-RAD-T/T-mask"), "*/Task6*/*.csv"),
        ("Task 6 3D-RAD-T / T-strip",      os.path.join(ROOT, "3D-RAD-T/T-strip"), "*/Task6*/*.csv"),
    ]
    print("=" * 86)
    print("  DOES 3D-RAD-T REMOVE THE LEAK?   text-only accuracy, patient-level split, seed 42")
    print("=" * 86)
    hdr = f"{'dataset':<34}{'n_test':>8}{'majority':>10}{'bracket':>10}{'tfidf+lr':>10}{'BEST':>8}"
    print(hdr); print("-" * len(hdr))
    out = {}
    for name, root, g in cfg:
        r = floors(load(root, g))
        out[name] = r
        b = f"{r['bracket_only']:.3f}" if r["bracket_only"] is not None else "   -  "
        print(f"{name:<34}{r['n_test']:>8}{r['majority']:>10.3f}{b:>10}"
              f"{r['tfidf_lr']:>10.3f}{r['best_text_only']:>8.3f}")

    orig = out["Task 6 original"]["best_text_only"]
    ctrl = out["Task 5 (control, never leaked)"]["best_text_only"]
    print("\nInterpretation")
    print("-" * 86)
    for v in ("T-mask", "T-strip"):
        k = f"Task 6 3D-RAD-T / {v}"
        print(f"  {v:<8} text-only floor {out[k]['best_text_only']:.3f} "
              f"(was {orig:.3f}, drop of {orig - out[k]['best_text_only']:+.3f}); "
              f"Task-5 control sits at {ctrl:.3f}")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_deleak.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] leak_analysis/verify_deleak.json")


if __name__ == "__main__":
    main()
