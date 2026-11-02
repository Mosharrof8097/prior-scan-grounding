#!/usr/bin/env python3
"""
Patient-cluster statistics for the blank-image audit (CPU, reads per_item_val.csv).

The audit's items are clustered: many questions share one volume, and volumes share a patient.
A plain binomial interval treats them as independent and is too narrow. Here the bootstrap
resamples PATIENTS. For each closed task it reports
  real accuracy, blank accuracy, and real - blank  (paired, same items),
  an exact McNemar test on real vs blank,
  and, for Task 6 vs Task 5, the paired difference on the same (volume, finding) items.
Only validation per-item data was saved; the test audit needs a rerun to produce the same.
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "leak_analysis"))
from verify_deleak_digits import finding_of  # noqa: E402

SRC = os.path.join(HERE, "interface audit result", "per_item_val.csv")
T4, T5, T6 = ("Task4_Existence_Detection", "Task5_Static_Temporal_Diagnosis",
              "Task6_Longitudinal_Temporal_Diagnosis")
B, SEED = 5000, 42


def boot(per_patient, rng):
    """per_patient: DataFrame indexed by patient, columns = [sum_x, n]. Returns (lo, hi) of sum/n."""
    s, n = per_patient["s"].values, per_patient["n"].values
    idx = rng.randint(0, len(s), size=(B, len(s)))
    stat = s[idx].sum(1) / n[idx].sum(1)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def cluster_ci(df, col, rng):
    g = df.assign(s=df[col]).groupby("patient").agg(s=("s", "sum"), n=("s", "size"))
    return float(df[col].sum() / len(df)), boot(g, rng)


def mcnemar(a, b):
    """Exact two-sided McNemar on paired 0/1 vectors."""
    n01 = int(((a == 0) & (b == 1)).sum()); n10 = int(((a == 1) & (b == 0)).sum())
    p = 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return {"only_first_right": n10, "only_second_right": n01, "p": float(p)}


def main():
    d = pd.read_csv(SRC)
    d = d[d.task.isin([T4, T5, T6])].copy()
    d["patient"] = d["npy"].str.split("_").str[:2].str.join("_")
    d["correct"] = d["mcq_correct_norm"].astype(float)
    wide = d.pivot_table(index=["task", "item_id", "npy", "patient"], columns="condition",
                         values="correct").reset_index()
    assert wide[["real", "blank"]].notna().all().all(), "real/blank not paired"
    qs = d[d.condition == "real"].set_index(["task", "item_id"])["question"]
    wide["question"] = [qs[(t, i)] for t, i in zip(wide.task, wide.item_id)]
    wide["finding"] = wide.question.map(finding_of)
    rng = np.random.RandomState(SEED)
    out = {}
    for task in (T4, T5, T6):
        w = wide[wide.task == task]
        r, lo_r = cluster_ci(w, "real", rng)
        b, lo_b = cluster_ci(w, "blank", rng)
        w = w.assign(diff=w["real"] - w["blank"])
        g = w.groupby("patient").agg(s=("diff", "sum"), n=("diff", "size"))
        dm = float(w["diff"].mean()); dci = boot(g, rng)
        out[task] = {"n_items": int(len(w)), "n_volumes": int(w.npy.nunique()),
                     "n_patients": int(w.patient.nunique()),
                     "real_acc": r, "real_ci95": lo_r, "blank_acc": b, "blank_ci95": lo_b,
                     "real_minus_blank": dm, "real_minus_blank_ci95": dci,
                     "mcnemar_real_vs_blank": mcnemar(w["real"].values, w["blank"].values)}
    # Task 6 vs Task 5, same (volume, finding) items
    a = wide[wide.task == T5][["npy", "patient", "finding", "real"]].rename(columns={"real": "t5"})
    c = wide[wide.task == T6][["npy", "finding", "real"]].rename(columns={"real": "t6"})
    m = a.merge(c, on=["npy", "finding"])
    m["diff"] = m["t6"] - m["t5"]
    g = m.groupby("patient").agg(s=("diff", "sum"), n=("diff", "size"))
    out["task6_minus_task5_real"] = {"paired_items": int(len(m)), "unmatched_t5": int(len(a) - len(m)),
                                     "t5_acc": float(m.t5.mean()), "t6_acc": float(m.t6.mean()),
                                     "diff": float(m["diff"].mean()), "diff_ci95": boot(g, rng),
                                     "mcnemar_t6_vs_t5": mcnemar(m["t6"].values, m["t5"].values)}
    with open(os.path.join(HERE, "audit_statistics_val.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    for k, v in out.items():
        print(k, json.dumps(v, indent=None))


if __name__ == "__main__":
    main()
