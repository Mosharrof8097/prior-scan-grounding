#!/usr/bin/env python3
"""
CPU test of the audit's input controls (mean, noise, swap, zero_prior, swap_prior) on tiny
synthetic volumes. Needs torch (CPU build is enough); no model, no GPU, no real data.

Run:  python tests/test_controls.py     (exit 1 on any failure)
"""
import importlib.util, os, sys, tempfile
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
spec = importlib.util.spec_from_file_location("audit", os.path.join(ROOT, "inference_audit", "kaggle_inference_audit.py"))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)

FAILS = []


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        FAILS.append(name)


class Tok:  # just enough of a tokenizer for EvalDataset.__getitem__
    eos_token, pad_token_id = "</s>", 0
    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 50 + 1 for c in text][:40]


def build(tmp):
    rng = np.random.RandomState(0)
    names = ["test_1_a_1", "test_1_b_1", "test_2_a_1", "test_2_b_1", "test_3_a_1", "test_4_a_1"]
    for n in names:   # each volume has its own brightness, so a swap is detectable
        v = np.clip(rng.rand(32, 8, 8) * 0.3 + (hash(n) % 7) / 10, 0, 1).astype(np.float32)
        np.save(os.path.join(tmp, n + ".npy"), v)
    rows = []
    for n in names:
        for task, qt in (("Task4_Existence_Detection", "Closed"), ("Task6_Longitudinal_Temporal_Diagnosis", "Closed")):
            rows.append({"NpyName": n + ".npy", "TaskCategory": task, "Question": "Is there a nodule?", "Answer": "No",
                         "QuestionType": qt, "Choice A": "Yes", "Choice B": "No", "Choice C": "-", "Choice D": "-",
                         "AnswerChoice": "B", "ImagePath": os.path.join(tmp, n + ".npy")})
    return pd.DataFrame(rows)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        df = build(tmp)
        timeline = audit.build_patient_timeline(tmp)
        image_map = {os.path.basename(p): p for p in __import__("glob").glob(os.path.join(tmp, "*.npy"))}
        ds = audit.EvalDataset(df, Tok(), timeline, image_map, 128, 8)
        idx = {(r.NpyName, r.TaskCategory[:5]): i for i, r in ds.df.iterrows()}

        def img(name, task, cond):
            ds.condition = cond
            return ds[idx[(name + ".npy", task)]]["image"][0].numpy()

        T4, T6 = "Task4", "Task6"
        # --- regression: real and blank behave as before
        real = img("test_1_b_1", T4, "real")
        raw = np.load(os.path.join(tmp, "test_1_b_1.npy"))
        check("real equals the stored volume", np.allclose(real, raw))
        check("blank is all zeros", float(img("test_1_b_1", T4, "blank").max()) == 0.0)

        # --- mean
        m = img("test_1_b_1", T4, "mean")
        check("mean: constant image", float(m.max() - m.min()) == 0.0)
        check("mean: keeps the volume's own mean", abs(float(m.mean()) - float(raw.mean())) < 1e-5)

        # --- noise
        n1, n2 = img("test_1_b_1", T4, "noise"), img("test_1_b_1", T4, "noise")
        check("noise: deterministic on repeat", np.array_equal(n1, n2))
        check("noise: differs from real", not np.allclose(n1, real))
        check("noise: stays in [0, 1]", n1.min() >= 0 and n1.max() <= 1)
        check("noise: mean/std match the volume", abs(float(n1.mean()) - float(raw.mean())) < 0.03
              and abs(float(n1.std()) - float(raw.std())) < 0.03)

        # --- swap (single scan): image is some OTHER patient's real volume
        s = img("test_1_b_1", T4, "swap")
        donors = [n for n in ds.donor_pool if audit.patient_of(n) != "test_1"]
        match = [d for d in donors if np.allclose(s, np.load(os.path.join(tmp, d + ".npy")) if not d.endswith(".npy")
                                                  else np.load(os.path.join(tmp, d)))]
        check("swap: equals a different patient's volume", len(match) == 1)
        check("swap: not the original", not np.allclose(s, real))

        # --- temporal: the stack is [T0 (16), T1 (16)] and test_1_b has earlier scan test_1_a
        t6 = img("test_1_b_1", T6, "real")
        check("temporal real: T0 half is the earlier scan",
              np.allclose(t6[:16], audit.resample_depth(np.load(os.path.join(tmp, "test_1_a_1.npy")), 16)))
        z = img("test_1_b_1", T6, "zero_prior")
        check("zero_prior: T0 half zeroed", float(z[:16].max()) == 0.0)
        check("zero_prior: current half untouched", np.allclose(z[16:], t6[16:]))
        sp = img("test_1_b_1", T6, "swap_prior")
        check("swap_prior: current half untouched", np.allclose(sp[16:], t6[16:]))
        check("swap_prior: T0 half replaced", not np.allclose(sp[:16], t6[:16]))
        sw6 = img("test_1_b_1", T6, "swap")
        check("swap on temporal: whole stack replaced", not np.allclose(sw6[:16], t6[:16]) and not np.allclose(sw6[16:], t6[16:]))
        check("swap on temporal keeps shape 32", sw6.shape[0] == 32)

        # --- regression: the temporal-only dataset must keep the ids of the full set
        # (a Kaggle smoke test found zero_prior/swap_prior renumbering their items 0..n-1)
        full = audit.with_item_ids(df)
        d_all = audit.EvalDataset(full, Tok(), timeline, image_map, 128, 8)
        is_t = full["TaskCategory"].str.lower().str.contains("longitudinal|temporal")
        d_tmp = audit.EvalDataset(full[is_t], Tok(), timeline, image_map, 128, 8)
        key = lambda d: {(r.NpyName, r.TaskCategory): r.item_id for r in d.df.itertuples()}
        ka, kt = key(d_all), key(d_tmp)
        check("temporal subset keeps the full set's item ids", all(ka[k] == v for k, v in kt.items()) and len(kt) < len(ka))
        check("item ids are unique over the full set", len(set(ka.values())) == len(ka))

        # --- prior controls refuse non-temporal items
        ds.condition = "zero_prior"
        try:
            ds[idx[("test_1_b_1.npy", T4)]]; refused = False
        except ValueError:
            refused = True
        check("zero_prior refuses a non-temporal item", refused)

        # --- donors
        pool = sorted(ds.donor_pool)
        check("pick_donor never returns the same patient",
              all(audit.patient_of(audit.pick_donor(n, pool)) != audit.patient_of(n) for n in pool))
        check("pick_donor is deterministic", audit.pick_donor("test_1_b_1", pool) == audit.pick_donor("test_1_b_1", pool[::-1]))
        check("pick_donor with one patient returns None", audit.pick_donor("test_1_a_1", ["test_1_a_1", "test_1_b_1"]) is None)
        check("prior_pool holds only scans with an earlier scan",
              all(n.split("_")[2] != "a" for n in ds.prior_pool) and "test_1_b_1.npy" in ds.prior_pool)

        try:
            audit.apply_control(raw, "bogus", 0); bad = False
        except ValueError:
            bad = True
        check("unknown control raises", bad)
    print("\nPASS" if not FAILS else f"\nFAIL: {len(FAILS)} check(s): {FAILS}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
