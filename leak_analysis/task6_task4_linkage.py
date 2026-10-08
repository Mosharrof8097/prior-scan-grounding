#!/usr/bin/env python3
"""
Can Task 6 be reconstructed from (past bracket) + (Task 4 label)?
=================================================================
Task 6's four answers are defined over two bits:
    past-ever-present  x  currently-present
      past=1, now=1 -> Refractory Lesion
      past=1, now=0 -> Resolved Lesion
      past=0, now=1 -> New Lesion
      past=0, now=0 -> No Abnormality

The past bit is given away by the bracket in the question text (the leak).
The *current* bit is exactly what Task 4 asks about the same volume and the
same finding ("Does the CT scan demonstrate <finding>?" -> Yes/No).

If answer == f(bracket, Task4_label) holds exactly, then after de-leaking,
the only thing Task 6 still requires is the Task 4 visual decision. That
matters because it removes the need for a radiologist to establish whether
the de-leaked task is answerable: Task 4's measurable accuracy above its
majority floor is itself an empirical lower bound on Task 6 answerability.
"""
import glob, os, re, json
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")

CAT = {(True,  True):  "Refractory Lesion (Persistent or recurrent, now present)",
       (True,  False): "Resolved Lesion (Previously present or recurrent, now absent)",
       (False, True):  "New Lesion (Absent previously, now present)",
       (False, False): "No Abnormality (Always absent)"}


def task4_labels():
    """(NpyName, finding) -> currently present?  from Task 4's 18 subtask files."""
    rows = []
    for f in glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/*/Task4*/*.csv")):
        finding = os.path.basename(f)[:-4].strip().lower()
        d = pd.read_csv(f)
        d["finding"] = finding
        d["present"] = d["Answer"].astype(str).str.strip().str.lower().eq("yes")
        d["NpyName"] = d["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
        rows.append(d[["NpyName", "finding", "present"]])
    t4 = pd.concat(rows, ignore_index=True).drop_duplicates(["NpyName", "finding"])
    return t4


def task6(findings):
    rows = []
    for f in glob.glob(os.path.join(ROOT, "3D-RAD_Dataset/*/Task6*/*.csv")):
        d = pd.read_csv(f)
        d["NpyName"] = d["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
        rows.append(d)
    t6 = pd.concat(rows, ignore_index=True)
    t6["bracket"] = t6["Question"].map(
        lambda q: (lambda m: [int(x) for x in m.group(1).replace(" ", "").split(",")]
                   if m else None)(BRACKET_RE.search(str(q))))
    t6["past_present"] = t6["bracket"].map(lambda s: any(s) if s else None)
    # which of the 18 findings does the question name?
    ql = t6["Question"].astype(str).str.lower()
    t6["finding"] = None
    for fd in sorted(findings, key=len, reverse=True):
        hit = t6["finding"].isna() & ql.str.contains(re.escape(fd), regex=True)
        t6.loc[hit, "finding"] = fd
    return t6


def main():
    t4 = task4_labels()
    findings = sorted(t4["finding"].unique())
    print(f"Task 4 findings: {len(findings)}")
    print(f"Task 4 (volume, finding) labels: {len(t4):,}")

    t6 = task6(findings)
    print(f"\nTask 6 rows: {len(t6):,}")
    print(f"  bracket parsed : {t6['bracket'].notna().sum():,}")
    print(f"  finding matched: {t6['finding'].notna().sum():,}")
    unmatched = t6[t6["finding"].isna()]
    if len(unmatched):
        print(f"  [!] unmatched example: {unmatched['Question'].iloc[0][:110]}")

    m = t6.merge(t4, on=["NpyName", "finding"], how="left")
    have = m["present"].notna() & m["past_present"].notna()
    print(f"  joined to a Task 4 label: {int(have.sum()):,} / {len(m):,}")

    j = m[have].copy()
    j["reconstructed"] = [CAT[(bool(p), bool(c))] for p, c in zip(j["past_present"], j["present"])]
    ok = (j["reconstructed"] == j["Answer"].astype(str))
    print("\n" + "=" * 78)
    print("  RECONSTRUCTION TEST:  answer == f(bracket, Task-4 label) ?")
    print("=" * 78)
    print(f"  exact match: {ok.sum():,} / {len(j):,}  ({100*ok.mean():.2f}%)")
    if (~ok).any():
        bad = j[~ok]
        print(f"\n  mismatches: {len(bad):,}")
        print(bad[["NpyName", "finding", "past_present", "present", "Answer", "reconstructed"]]
              .head(8).to_string(index=False))

    # what the de-leaked task still demands: just the 'current' bit
    print("\n" + "=" * 78)
    print("  WHAT T-MASK STILL REQUIRES")
    print("=" * 78)
    dist = j["present"].value_counts(normalize=True)
    print(f"  the remaining unknown is one bit: is <finding> present in the current CT?")
    print(f"  its class balance over Task 6 items: "
          f"present {dist.get(True,0):.3f} / absent {dist.get(False,0):.3f}")
    print(f"  => majority-guessing that bit caps text-only at "
          f"{max(dist.get(True,0), dist.get(False,0)):.3f}, matching the measured floor.")

    out = {"n_task4_labels": int(len(t4)), "n_findings": len(findings),
           "n_task6": int(len(t6)), "n_joined": int(have.sum()),
           "exact_reconstruction_rate": float(ok.mean()),
           "current_bit_balance": {str(k): float(v) for k, v in dist.items()}}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "task6_task4_linkage.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] leak_analysis/task6_task4_linkage.json")


if __name__ == "__main__":
    main()
