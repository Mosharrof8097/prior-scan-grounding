#!/usr/bin/env python3
"""
Digit-aware check that 3D-RAD-T removes the Task 6 bracket leak.

`verify_deleak.py` uses TfidfVectorizer's default token pattern, which drops
single-character tokens, so the digits 0/1 inside "[0, 1]" are invisible to it. Its
Task 6 original therefore scores 0.6332 (below majority) even though the leak is real,
and its "de-leaked" scores prove nothing. This script tests the claim directly:

  1. Residual scan   : no bracket of 0/1 digits and no standalone 0/1 token is left.
  2. Phrasing check  : reports how many distinct question strings each finding has (the
                       templates vary), so step 3 must rule out phrasing as a side channel.
  3. Recoverability  : a character n-gram classifier (sees digits and brackets) predicts
                       the past bit from the question text, patient-grouped CV.
                       Positive control: original 3D-RAD must be recovered ~perfectly.
                       De-leaked variants must not beat a finding-only baseline.

The past bit is read from the ORIGINAL bracket (any 1 in it), the same definition
information_ceilings.py uses. Only the 2,873 test-directory rows (the ones with CT) are used.
"""
import glob, os, re, json, sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
T6 = "Task6_Longitudinal_Temporal_Diagnosis"
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")
DIGIT_BRACKET_RE = re.compile(r"\[\s*[01](?:\s*,\s*[01])*\s*\]")
LONE_BIT_RE = re.compile(r"(?<![\w.])[01](?![\w.])")
NOTE_RE = re.compile(r"\(Note: 0 indicates absence, and 1 indicates presence\.\)")
FINDINGS = sorted((os.path.splitext(os.path.basename(p))[0].lower() for p in
                   glob.glob(os.path.join(ROOT, "3D-RAD-T", "T-strip", "test", "Task4_Existence_Detection", "*.csv"))),
                  key=len, reverse=True)  # longest first: "coronary artery wall calcification" before shorter names
assert len(FINDINGS) == 18, FINDINGS


def finding_of(question):
    q = question.lower()
    return next((f for f in FINDINGS if f in q), "?")


def load_task6(variant_root):
    frames = []
    for f in sorted(glob.glob(os.path.join(variant_root, "test", T6, "*.csv"))):
        d = pd.read_csv(f)
        d["file"] = os.path.basename(f)
        d["row"] = np.arange(len(d))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def aligned(orig, var):
    """Variant rows, with the original's past bit attached. Alignment is asserted."""
    assert len(orig) == len(var), (len(orig), len(var))
    assert (orig["file"].values == var["file"].values).all()
    assert (orig["VolumeName"].values == var["VolumeName"].values).all(), "volumes misaligned"
    assert (orig["Answer"].values == var["Answer"].values).all(), "answers differ"
    out = var.copy()
    out["past"] = [int(any(int(x) for x in BRACKET_RE.search(q).group(1).replace(" ", "").split(",")))
                   for q in orig["Question"]]
    out["patient"] = out["VolumeName"].astype(str).str.split("_").str[:2].str.join("_")
    out["finding"] = out["Question"].map(finding_of)
    assert (out["finding"] != "?").all(), out[out["finding"] == "?"]["Question"].head().tolist()
    return out


def residual_scan(df):
    q = df["Question"].astype(str).map(lambda s: NOTE_RE.sub("", s))  # the fixed legend names 0 and 1; not a leak
    return {"digit_bracket_rows": int(q.map(lambda s: bool(DIGIT_BRACKET_RE.search(s))).sum()),
            "lone_bit_rows": int(q.map(lambda s: bool(LONE_BIT_RE.search(s))).sum())}


def questions_per_finding(df):
    return int(df.groupby("finding")["Question"].nunique().max())


def grouped_cv_acc(df, texts):
    y = df["past"].values
    pred = np.zeros(len(df), dtype=int)
    for tr, te in GroupKFold(n_splits=5).split(texts, y, df["patient"].values):
        clf = make_pipeline(TfidfVectorizer(analyzer="char", ngram_range=(1, 5), lowercase=False, min_df=2),
                            LogisticRegression(max_iter=3000, C=4.0))
        clf.fit(texts.iloc[tr], y[tr])
        pred[te] = clf.predict(texts.iloc[te])
    return float((pred == y).mean())


def finding_only_acc(df):
    """Best a model can do knowing only the finding name: per-finding majority past bit (CV)."""
    y = df["past"].values
    pred = np.zeros(len(df), dtype=int)
    for tr, te in GroupKFold(n_splits=5).split(df, y, df["patient"].values):
        lut = df.iloc[tr].groupby("finding")["past"].agg(lambda s: int(s.mean() >= 0.5))
        glob_maj = int(y[tr].mean() >= 0.5)
        pred[te] = df.iloc[te]["finding"].map(lut).fillna(glob_maj).astype(int).values
    return float((pred == y).mean())


def main():
    orig = load_task6(os.path.join(ROOT, "3D-RAD_Dataset"))
    variants = {"original": os.path.join(ROOT, "3D-RAD_Dataset"),
                "T-mask": os.path.join(ROOT, "3D-RAD-T", "T-mask"),
                "T-strip": os.path.join(ROOT, "3D-RAD-T", "T-strip")}
    res = {}
    for name, root in variants.items():
        df = aligned(orig, load_task6(root))
        res[name] = {"n": int(len(df)),
                     "past_bit_rate": float(df["past"].mean()),
                     "residual": residual_scan(df),
                     "max_distinct_questions_per_finding": questions_per_finding(df),
                     "past_bit_recovery_acc": grouped_cv_acc(df, df["Question"].astype(str)),
                     "finding_only_baseline": finding_only_acc(df)}
        r = res[name]
        print(f"{name:<9} n={r['n']}  residual={r['residual']}  "
              f"q/finding={r['max_distinct_questions_per_finding']}  "
              f"recover={r['past_bit_recovery_acc']:.3f}  finding-only={r['finding_only_baseline']:.3f}")

    # --- verdicts (asserted, so a regression fails loudly) ---
    o = res["original"]
    assert o["past_bit_recovery_acc"] > o["finding_only_baseline"] + 0.2, \
        "positive control failed: the probe cannot see the original leak"
    verdict = {}
    for v in ("T-mask", "T-strip"):
        r = res[v]
        no_bracket = r["residual"]["digit_bracket_rows"] == 0
        no_lone = r["residual"]["lone_bit_rows"] == 0
        at_baseline = r["past_bit_recovery_acc"] <= r["finding_only_baseline"] + 0.01
        verdict[v] = {"no_digit_bracket": no_bracket, "no_lone_bit": no_lone,
                      "recovery_at_finding_baseline": at_baseline,
                      "leak_removed": no_bracket and no_lone and at_baseline}
        print(f"{v}: {verdict[v]}")
    res["verdict"] = verdict
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_deleak_digits.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("[saved] leak_analysis/verify_deleak_digits.json")
    return 0 if all(v["leak_removed"] for v in verdict.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
