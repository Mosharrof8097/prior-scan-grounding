#!/usr/bin/env python3
"""
Floors and ceilings for 3D-RAD's temporal tasks, from label structure alone.
============================================================================
No radiologist needed. The four Task 5 / Task 6 answers are a product of two
bits -- (was the finding present before?) x (is it present now?) -- so what any
model can possibly score is fixed by which bits its input actually carries.

  Task 6 question : past bit GIVEN as a bracket. Current bit must come from the CT.
  Task 5 question : neither bit given in text. 3D-RAD supplies only ONE volume
                    per row, so the past bit has no source at all unless the
                    prior scan is fed in separately.

That yields hard, assumption-free bounds:
  floor   = best achievable with text only (guess the unknown bit by its prior)
  ceiling = best achievable reading the current CT perfectly, past bit still guessed
"""
import glob, os, re, json
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")
# answer -> (past present, currently present)
BITS = {"New Lesion (Absent previously, now present)": (False, True),
        "No Abnormality (Always absent)": (False, False),
        "Refractory Lesion (Persistent or recurrent, now present)": (True, True),
        "Resolved Lesion (Previously present or recurrent, now absent)": (True, False)}


def load(task):
    rows = []
    for f in glob.glob(os.path.join(ROOT, f"3D-RAD_Dataset/*/{task}*/*.csv")):
        rows.append(pd.read_csv(f))
    d = pd.concat(rows, ignore_index=True)
    d["NpyName"] = d["VolumeName"].astype(str).str.replace(".nii.gz", "", regex=False)
    d["past"] = d["Answer"].map(lambda a: BITS[a][0])
    d["now"] = d["Answer"].map(lambda a: BITS[a][1])
    return d


def best_given(df, cols):
    """Accuracy of the optimal rule that sees only `cols`, then guesses the rest."""
    if not cols:
        return df["Answer"].value_counts().iloc[0] / len(df)
    return df.groupby(cols)["Answer"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)


def main():
    t6 = load("Task6")
    t6["bracket_past"] = t6["Question"].map(
        lambda q: (lambda m: any(int(x) for x in m.group(1).replace(" ", "").split(","))
                   if m else None)(BRACKET_RE.search(str(q))))
    assert (t6["bracket_past"] == t6["past"]).all(), "bracket does not equal the past bit"
    print(f"[check] the Task 6 bracket equals the ground-truth past bit on all "
          f"{len(t6):,} rows -- it is a label, not a hint.\n")

    t5 = load("Task5")
    n = len(t5)
    rows = [
        ("blind majority guess",              best_given(t5, [])),
        ("Task 5: text only",                 best_given(t5, [])),
        ("Task 5: + perfect current CT",      best_given(t5, ["now"])),
        ("Task 5: + prior scan as well",      1.0),
        ("Task 6: text only (bracket)",       best_given(t6, ["past"])),
        ("Task 6: bracket + perfect current", 1.0),
    ]
    print("=" * 74)
    print("  WHAT EACH INPUT COMBINATION CAN POSSIBLY SCORE  (n = %s items)" % f"{n:,}")
    print("=" * 74)
    for label, v in rows:
        print(f"  {label:<36} {v:>8.4f}")

    t5_floor, t5_ceil = best_given(t5, []), best_given(t5, ["now"])
    t6_floor = best_given(t6, ["past"])
    print("\n" + "=" * 74)
    print("  CONSEQUENCES")
    print("=" * 74)
    print(f"  Task 6 is {t6_floor:.1%} solved before any pixel is read.")
    print(f"  Task 5 given one volume caps out at {t5_ceil:.1%} even with a perfect")
    print(f"    reader, because the past bit is simply absent from the input;")
    print(f"    the headroom a single-volume model can win is only")
    print(f"    {t5_ceil:.3f} - {t5_floor:.3f} = {t5_ceil - t5_floor:.3f}.")
    print(f"  Feeding the PRIOR SCAN lifts the Task 5 ceiling to 1.000. That")
    print(f"    turns Task 5 into the genuine longitudinal task: both bits must")
    print(f"    come from images, and neither is available in the text.")

    out = {"n_items": int(n),
           "task5_text_floor": float(t5_floor),
           "task5_ceiling_single_volume": float(t5_ceil),
           "task5_ceiling_with_prior_scan": 1.0,
           "task6_text_floor_bracket": float(t6_floor),
           "bracket_equals_past_bit": True}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "information_ceilings.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] leak_analysis/information_ceilings.json")


if __name__ == "__main__":
    main()
