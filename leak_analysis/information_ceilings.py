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

The bounds below are label-structure bounds: they assume noise-free labels and
ignore any evidence of the past that is visible in the current scan, so they are
upper bounds on what each input set can support, not measured human ceilings.

Population matters. The released CSVs hold 16,031 Task 5 / Task 6 rows, but only
the 2,873 rows from the official `test/` directory (169 volumes) have CT volumes;
the 13,158 `train/` rows (774 volumes) have no images. Every number is therefore
reported for the pooled rows AND for the rows that can actually be evaluated.

"Prior scan" ceilings are reported two ways, because the bracket holds the whole
history. Its length equals the number of earlier studies, and the answer's past bit
is "present at ANY earlier study". The nearest prior scan alone only reveals the
LAST bracket element, which is what the baseline in this repo is fed.
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
T5_DIR = "Task5_Static_Temporal_Diagnosis"
T6_DIR = "Task6_Longitudinal_Temporal_Diagnosis"


def parse_bracket(question):
    m = BRACKET_RE.search(str(question))
    return [int(x) for x in m.group(1).replace(" ", "").split(",")] if m else None


def load_pairs():
    """Task 5 rows with the matching Task 6 bracket attached, aligned row by row.

    Task 5 and Task 6 ship as parallel files (same name, same row order). The
    alignment is asserted on volume and answer, not assumed.
    """
    frames = []
    for split_dir in ("test", "train"):
        for f5 in sorted(glob.glob(os.path.join(ROOT, f"3D-RAD_Dataset/{split_dir}/{T5_DIR}/*.csv"))):
            f6 = f5.replace(T5_DIR, T6_DIR)
            a, b = pd.read_csv(f5), pd.read_csv(f6)
            assert len(a) == len(b), f"row count differs: {f5}"
            assert (a["VolumeName"].values == b["VolumeName"].values).all(), f"volumes misaligned: {f5}"
            assert (a["Answer"].values == b["Answer"].values).all(), f"answers differ: {f5}"
            a = a.copy()
            a["source_dir"] = split_dir
            a["bracket"] = b["Question"].map(parse_bracket)
            frames.append(a)
    d = pd.concat(frames, ignore_index=True)
    d["past"] = d["Answer"].map(lambda x: BITS[x][0])
    d["now"] = d["Answer"].map(lambda x: BITS[x][1])
    d["bracket_any"] = d["bracket"].map(lambda b: bool(max(b)))
    d["bracket_last"] = d["bracket"].map(lambda b: bool(b[-1]))
    d["n_earlier"] = d["bracket"].map(len)
    return d


def best_given(df, cols):
    """Accuracy of the optimal rule that sees only `cols`, then guesses the rest."""
    if not cols:
        return df["Answer"].value_counts().iloc[0] / len(df)
    return df.groupby(cols)["Answer"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(df)


def summarise(df):
    return {
        "n_items": int(len(df)),
        "n_volumes": int(df["VolumeName"].nunique()),
        "p_past_present": float(df["past"].mean()),
        "task5_text_floor": float(best_given(df, [])),
        "task5_ceiling_single_volume": float(best_given(df, ["now"])),
        "task5_ceiling_with_nearest_prior_scan": float(best_given(df, ["bracket_last", "now"])),
        "task5_ceiling_with_all_prior_scans": float(best_given(df, ["bracket_any", "now"])),
        "task6_text_floor_bracket": float(best_given(df, ["bracket_any"])),
        "share_with_exactly_one_prior": float((df["n_earlier"] == 1).mean()),
    }


def main():
    d = load_pairs()
    assert (d["bracket_any"] == d["past"]).all(), "bracket does not equal the past bit"
    print(f"[check] the Task 6 bracket equals the ground-truth past bit on all "
          f"{len(d):,} rows -- it is a label, not a hint.\n")

    populations = {
        "pooled_all_rows": d,
        "evaluable_rows_with_ct_(test_dir)": d[d["source_dir"] == "test"],
        "rows_without_ct_(train_dir)": d[d["source_dir"] == "train"],
    }
    out = {"bracket_equals_past_bit": True, "populations": {}}
    cols = [("task5_text_floor", "Task 5: text only"),
            ("task5_ceiling_single_volume", "Task 5: + perfect current CT"),
            ("task5_ceiling_with_nearest_prior_scan", "Task 5: + nearest prior scan (what the baseline is fed)"),
            ("task5_ceiling_with_all_prior_scans", "Task 5: + every prior scan"),
            ("task6_text_floor_bracket", "Task 6: text only (bracket)")]
    for name, g in populations.items():
        s = summarise(g)
        out["populations"][name] = s
        print("=" * 78)
        print(f"  {name}   (n = {s['n_items']:,} items, {s['n_volumes']:,} volumes)")
        print("=" * 78)
        for key, label in cols:
            print(f"  {label:<58} {s[key]:>8.4f}")
        print(f"  {'share of items with exactly one prior study':<58} {s['share_with_exactly_one_prior']:>8.4f}")
        print()

    ev = out["populations"]["evaluable_rows_with_ct_(test_dir)"]
    print("=" * 78)
    print("  CONSEQUENCES (evaluable rows only)")
    print("=" * 78)
    print(f"  Task 6 is {ev['task6_text_floor_bracket']:.1%} solved before any pixel is read.")
    print(f"  Task 5 with one volume caps at {ev['task5_ceiling_single_volume']:.1%} for a perfect reader of that")
    print(f"    volume; with the nearest prior scan added it caps at "
          f"{ev['task5_ceiling_with_nearest_prior_scan']:.1%}.")
    print(f"  Only the full history reaches 1.000 (needs every earlier study).")

    # kept for backward compatibility with scripts that read the old keys
    out.update({
        "n_items": ev["n_items"],
        "task5_text_floor": ev["task5_text_floor"],
        "task5_ceiling_single_volume": ev["task5_ceiling_single_volume"],
        "task5_ceiling_with_prior_scan": ev["task5_ceiling_with_all_prior_scans"],
        "task6_text_floor_bracket": ev["task6_text_floor_bracket"],
        "headline_population": "evaluable_rows_with_ct_(test_dir)",
    })
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "information_ceilings.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n[saved] leak_analysis/information_ceilings.json")


if __name__ == "__main__":
    main()
