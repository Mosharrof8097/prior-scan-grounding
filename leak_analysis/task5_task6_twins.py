#!/usr/bin/env python3
"""
Is Task 5 a native, already-de-leaked twin of Task 6?
=====================================================
Task 5 (Static Temporal Diagnosis) and Task 6 (Longitudinal Temporal Diagnosis)
cover the SAME 943 volumes and the same 4-way lesion-category answer space.
Task 6's question hands over the past presence/absence sequence as a bracket;
Task 5's question never does.

If the two tasks agree on the answer for every (volume, finding) pair, then
3D-RAD already ships a matched leaked / un-leaked pair of the same underlying
clinical question -- which is a far better experimental control than any
synthetic de-leaking, and needs no radiologist to validate: the dataset authors
already assert the Task 5 items are answerable from the CT.
"""
import glob, os, re, json
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")
FINDINGS = sorted((os.path.basename(f)[:-4].strip().lower()
                   for f in glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/test/Task4*/*.csv"))),
                  key=len, reverse=True)


def load(task):
    rows = []
    for f in glob.glob(os.path.join(ROOT, f"3D-RAD_Dataset/*/{task}*/*.csv")):
        d = pd.read_csv(f)
        d["split_dir"] = f.split(os.sep)[-3]
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["NpyName"] = d["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    ql = d["Question"].astype(str).str.lower()
    d["finding"] = None
    for fd in FINDINGS:
        hit = d["finding"].isna() & ql.str.contains(re.escape(fd), regex=True)
        d.loc[hit, "finding"] = fd
    return d


def main():
    t5, t6 = load("Task5"), load("Task6")
    for n, d in (("Task 5", t5), ("Task 6", t6)):
        print(f"{n}: {len(d):,} rows | {d['NpyName'].nunique():,} volumes | "
              f"{d['finding'].notna().sum():,} findings matched | "
              f"{d['Answer'].nunique()} distinct answers")

    key = ["NpyName", "finding"]
    a = t5.dropna(subset=["finding"]).drop_duplicates(key)[key + ["Answer"]].rename(
        columns={"Answer": "answer_t5"})
    b = t6.dropna(subset=["finding"]).drop_duplicates(key)[key + ["Answer", "Question"]].rename(
        columns={"Answer": "answer_t6"})
    m = a.merge(b, on=key, how="inner")

    print("\n" + "=" * 78)
    print("  DO TASK 5 AND TASK 6 ANSWER THE SAME QUESTION?")
    print("=" * 78)
    print(f"  matched (volume, finding) pairs : {len(m):,}")
    ok = (m["answer_t5"] == m["answer_t6"])
    print(f"  identical answer                : {ok.sum():,} / {len(m):,}  ({100*ok.mean():.2f}%)")
    if (~ok).any():
        print("\n  disagreements:")
        print(m[~ok][key + ["answer_t5", "answer_t6"]].head(8).to_string(index=False))

    # Task 6 leaks the past bit; confirm Task 5 asks for the same thing without it
    m["past_present"] = m["Question"].map(
        lambda q: (lambda mm: any(int(x) for x in mm.group(1).replace(" ", "").split(","))
                   if mm else None)(BRACKET_RE.search(str(q))))
    print("\n" + "=" * 78)
    print("  WHAT THE LEAK BUYS, ON IDENTICAL ITEMS")
    print("=" * 78)
    ct = pd.crosstab(m["past_present"], m["answer_t5"])
    print(ct.to_string())
    floor = ct.max(axis=1).sum() / len(m)
    maj = m["answer_t5"].value_counts().iloc[0] / len(m)
    print(f"\n  majority-only floor (Task 5 phrasing) : {maj:.4f}")
    print(f"  bracket-conditioned floor (Task 6)    : {floor:.4f}")
    print(f"  information the bracket hands over    : {floor - maj:+.4f}")

    out = {"n_matched_pairs": int(len(m)),
           "answer_agreement_rate": float(ok.mean()),
           "majority_floor": float(maj), "bracket_floor": float(floor),
           "leak_value": float(floor - maj)}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "task5_task6_twins.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] leak_analysis/task5_task6_twins.json")


if __name__ == "__main__":
    main()
