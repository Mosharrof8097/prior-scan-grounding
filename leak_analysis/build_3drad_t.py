#!/usr/bin/env python3
"""
Build 3D-RAD-T : a de-leaked variant of 3D-RAD Task 6
=====================================================
Task 6 questions embed the past presence/absence sequence as a literal bracket,
e.g. "Past sequences of lung opacity: [1]. (Note: 0 indicates absence, ...)".
That sequence deterministically eliminates 2 of the 4 answer options, so 80.7%
of Task 6 is answerable with no CT scan at all.

Two variants are written, so the effect of removing the leak can be separated
from the effect of rewriting the prompt:

  T-mask  : the bracket contents are replaced by [MASKED]. Exactly one thing
            changes; every other character of the question is preserved.
            -> primary variant.
  T-strip : the lead-in, the bracket, and the "(Note: ...)" gloss are removed,
            keeping the trailing question only. Reads naturally but changes more.
            -> secondary / robustness variant.

Only the questions change. VolumeName, Answer, choices and the task/subtask
layout are copied verbatim, so the existing data pipeline reads it unmodified.
"""
import glob, os, re, json, shutil
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = os.path.join(ROOT, "3D-RAD_Dataset")
NOTE = "(Note: 0 indicates absence, and 1 indicates presence.)"
BRACKET_RE = re.compile(r"\[[01](?:,\s*[01])*\]")


def mask_q(q):
    """T-mask: swap only the bracket contents for a placeholder."""
    return BRACKET_RE.sub("[MASKED]", str(q))


def strip_q(q):
    """T-strip: drop everything up to and including the (Note: ...) gloss."""
    q = str(q)
    i = q.find(NOTE)
    if i == -1:
        return mask_q(q)
    tail = q[i + len(NOTE):].strip()
    # the first sentence of the tail is filler that refers to the removed sequence
    parts = [s.strip() for s in tail.split(". ") if s.strip()]
    return (parts[-1] if len(parts) > 1 else tail).strip()


def main():
    stats = {"variants": {}}
    for variant, fn in (("T-mask", mask_q), ("T-strip", strip_q)):
        out_root = os.path.join(ROOT, f"3D-RAD-T/{variant}")
        n_files = n_rows = n_changed = 0
        for f in sorted(glob.glob(os.path.join(SRC, "*/Task*/*.csv"))):
            rel = os.path.relpath(f, SRC)
            dst = os.path.join(out_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            d = pd.read_csv(f)
            if "Task6" in rel:
                before = d["Question"].astype(str).copy()
                d["Question"] = before.map(fn)
                n_changed += int((d["Question"] != before).sum())
            d.to_csv(dst, index=False)
            n_files += 1
            n_rows += len(d)
        stats["variants"][variant] = {"files": n_files, "rows": n_rows,
                                      "task6_questions_rewritten": n_changed,
                                      "path": os.path.relpath(out_root, ROOT)}
        print(f"[{variant}] {n_files} csv files, {n_rows:,} rows, "
              f"{n_changed:,} Task-6 questions rewritten -> 3D-RAD-T/{variant}/")

    # show the transformation on one example
    ex = pd.read_csv(os.path.join(SRC, "test/Task6_Longitudinal_Temporal_Diagnosis/c.csv"))
    q = ex["Question"].iloc[0]
    print("\n" + "=" * 78 + "\n  EXAMPLE TRANSFORMATION\n" + "=" * 78)
    print("original :", q)
    print("\nT-mask   :", mask_q(q))
    print("\nT-strip  :", strip_q(q))
    print("\nanswer   :", ex["Answer"].iloc[0])

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "build_3drad_t.json"), "w") as fh:
        json.dump(stats, fh, indent=2)


if __name__ == "__main__":
    main()
