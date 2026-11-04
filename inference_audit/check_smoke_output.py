#!/usr/bin/env python3
"""
Sanity-check the outputs of kaggle_inference_audit.py (smoke test or full run) on the laptop.

    python inference_audit/check_smoke_output.py <output_dir> [--controls mean noise swap zero_prior swap_prior] [--splits val]

Reads <output_dir>/per_item_<split>.csv and fails (exit 1) if anything that would make the numbers
meaningless is wrong: a missing condition, a condition that did not cover the same items as "real",
missing predictions, non-finite log-probabilities, temporal items that were not paired with an
earlier scan, or prior controls applied to non-temporal items. It also prints, per condition, how
often the prediction equals the "real" prediction, which is information, not a pass/fail test.
"""
import argparse, os, sys
import numpy as np
import pandas as pd

PRIOR = ("zero_prior", "swap_prior")


def check_split(d, split, controls):
    path = os.path.join(d, f"per_item_{split}.csv")
    if not os.path.exists(path):
        return [f"{split}: {path} is missing"], None
    df = pd.read_csv(path)
    errs = []
    want = ["real", "blank"] + list(controls)
    have = set(df.condition.unique())
    for c in want:
        if c not in have:
            errs.append(f"{split}: condition '{c}' has no rows")
    unexpected = have - set(want)
    if unexpected:
        errs.append(f"{split}: unexpected conditions {sorted(unexpected)} (pass --controls to match the run)")
    real = df[df.condition == "real"]
    if real.empty:
        return errs + [f"{split}: no 'real' rows"], df
    temporal = real.task.str.contains("Temporal", case=False)
    key = ["item_id"]
    for c in sorted(have - {"real"}):
        sub = df[df.condition == c]
        expected = real[temporal] if c in PRIOR else real
        if len(sub) != len(expected):
            errs.append(f"{split}/{c}: {len(sub)} rows, expected {len(expected)}")
        if set(sub.item_id) != set(expected.item_id):
            errs.append(f"{split}/{c}: item ids differ from 'real' on the same scope")
        if c in PRIOR and (~sub.task.str.contains("Temporal", case=False)).any():
            errs.append(f"{split}/{c}: applied to a non-temporal item")
    for c, sub in df.groupby("condition"):
        mcq = sub[sub.is_mcq.astype(bool)]
        opn = sub[~sub.is_mcq.astype(bool)]
        if mcq.pred_choice_norm.isna().any():
            errs.append(f"{split}/{c}: {int(mcq.pred_choice_norm.isna().sum())} MCQ rows without a prediction")
        if not mcq.mcq_correct_norm.dropna().isin([0, 1]).all():
            errs.append(f"{split}/{c}: mcq_correct_norm has values other than 0/1")
        if not np.isfinite(mcq.gold_logprob.dropna()).all():
            errs.append(f"{split}/{c}: non-finite gold_logprob")
        if opn.pred_text.isna().any():
            errs.append(f"{split}/{c}: {int(opn.pred_text.isna().sum())} open rows without generated text")
    bad_pair = real[temporal & (real.pairing != "paired_ok")]
    if len(bad_pair):
        errs.append(f"{split}: {len(bad_pair)} temporal rows are not 'paired_ok' (fallback input)")
    return errs, df


def report(df, split):
    real = df[df.condition == "real"].set_index("item_id")
    print(f"\n[{split}] rows per condition: " + ", ".join(f"{c}={n}" for c, n in df.condition.value_counts().items()))
    print(f"[{split}] items per task (real): " + ", ".join(f"{t.split('_')[0]}={n}" for t, n in real.task.value_counts().sort_index().items()))
    print(f"[{split}] share of MCQ predictions equal to the 'real' prediction, and accuracy, per condition and task:")
    for c in [x for x in df.condition.unique() if x != "real"]:
        sub = df[(df.condition == c) & df.is_mcq.astype(bool)].set_index("item_id")
        for t, g in sub.groupby("task"):
            r = real.loc[g.index]
            same = (g.pred_choice_norm == r.pred_choice_norm).mean()
            print(f"   {c:<11}{t.split('_')[0]:<7} n={len(g):<4} same-as-real={same:5.2f}  "
                  f"acc real={r.mcq_correct_norm.mean():.3f} {c}={g.mcq_correct_norm.mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir")
    ap.add_argument("--controls", nargs="*", default=[])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    a = ap.parse_args()
    all_errs = []
    for s in a.splits:
        errs, df = check_split(a.output_dir, s, a.controls)
        all_errs += errs
        if df is not None and not errs:
            report(df, s)
    if all_errs:
        print("\nFAIL")
        for e in all_errs:
            print("  -", e)
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
