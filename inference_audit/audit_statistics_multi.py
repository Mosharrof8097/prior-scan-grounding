#!/usr/bin/env python3
"""
Patient-cluster statistics for the blank-image audit (CPU, reads per_item_val.csv).

The audit's items are clustered: many questions share one volume, and volumes share a patient.
A plain binomial interval treats them as independent and is too narrow. Here the bootstrap
resamples PATIENTS. For each closed task it reports
  real accuracy, blank accuracy, and real - blank  (paired, same items),
  an exact McNemar test on real vs blank,
  and, for Task 6 vs Task 5, the paired difference on the same (volume, finding) items.
Works on per-item files from the 7-condition run (val and test).
"""
import json, os, sys
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import balanced_accuracy_score, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "leak_analysis"))
from verify_deleak_digits import finding_of  # noqa: E402

SRC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "interface audit result", "per_item_val.csv")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "audit_statistics_val.json")
T4, T5, T6 = ("Task4_Existence_Detection", "Task5_Static_Temporal_Diagnosis",
              "Task6_Longitudinal_Temporal_Diagnosis")
B, SEED = 5000, 42
B_CLASS = 1000  # class metrics re-score every draw, so fewer draws


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


def class_metrics(d, rng):
    """Balanced accuracy and macro-F1 per closed task and condition, with a patient bootstrap CI on
    balanced accuracy. Plain accuracy hides a model that always says the majority class."""
    out = {}
    for task in (T4, T5, T6):
        for cond in ("real", "blank"):
            x = d[(d.task == task) & (d.condition == cond)].reset_index(drop=True)
            labels = sorted(x.gold_text.unique())

            def bal(frame):
                return balanced_accuracy_score(frame.gold_text, frame.pred_text)
            pats = x.patient.unique()
            by_p = {p: x[x.patient == p] for p in pats}
            draws = [bal(pd.concat([by_p[p] for p in rng.choice(pats, len(pats))])) for _ in range(B_CLASS)]
            out[f"{task}|{cond}"] = {
                "accuracy": float((x.gold_text == x.pred_text).mean()),
                "balanced_accuracy": float(bal(x)),
                "balanced_accuracy_ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
                "macro_f1": float(f1_score(x.gold_text, x.pred_text, labels=labels, average="macro", zero_division=0)),
                "chance_balanced_accuracy": 1.0 / len(labels),
                "gold_classes": len(labels), "classes_ever_predicted": int(x.pred_text.nunique())}
    return out


def main():
    d = pd.read_csv(SRC)
    d = d[d.task.isin([T4, T5, T6])].copy()
    d["patient"] = d["npy"].str.split("_").str[:2].str.join("_")
    d["correct"] = d["mcq_correct_norm"].astype(float)
    classes = class_metrics(d, np.random.RandomState(SEED))
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
    # extra controls: control - real... reported as real - control, paired, patient-cluster CI
    ctrls = sorted(set(d.condition) - {"real", "blank"})
    extra = {}
    for task in (T4, T5, T6):
        for cnd in ctrls:
            x = d[(d.task == task) & (d.condition.isin(["real", cnd]))]
            w = x.pivot_table(index=["item_id", "patient"], columns="condition", values="correct").reset_index()
            if cnd not in w or w[cnd].isna().all():
                continue
            w = w.dropna(subset=["real", cnd])
            w["diff"] = w["real"] - w[cnd]
            g = w.groupby("patient").agg(s=("diff", "sum"), n=("diff", "size"))
            extra[f"{task}|{cnd}"] = {"n_items": int(len(w)), "n_patients": int(w.patient.nunique()),
                "real_acc": float(w.real.mean()), "control_acc": float(w[cnd].mean()),
                "real_minus_control": float(w["diff"].mean()), "ci95": boot(g, rng),
                "mcnemar": mcnemar(w["real"].values, w[cnd].values)}
    out["controls"] = extra
    out["class_metrics"] = classes
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    for k, v in out.items():
        if k not in ("class_metrics", "controls"):
            print(k, json.dumps(v, indent=None))
    for k, v in extra.items():
        print(f"{k:<52} real-ctrl {v['real_minus_control']:+.4f} [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
              f"discordant {v['mcnemar']['only_first_right']}/{v['mcnemar']['only_second_right']}  p {v['mcnemar']['p']:.3g}")
    for k, v in classes.items():
        print(f"{k:<52} acc {v['accuracy']:.3f}  bal-acc {v['balanced_accuracy']:.3f} "
              f"[{v['balanced_accuracy_ci95'][0]:.3f},{v['balanced_accuracy_ci95'][1]:.3f}]  "
              f"macroF1 {v['macro_f1']:.3f}  chance {v['chance_balanced_accuracy']:.2f}  "
              f"predicted {v['classes_ever_predicted']}/{v['gold_classes']} classes")


if __name__ == "__main__":
    main()
