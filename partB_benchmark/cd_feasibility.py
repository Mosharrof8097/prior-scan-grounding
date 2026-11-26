#!/usr/bin/env python3
"""
partB_benchmark/cd_feasibility.py

Visual Contrastive Decoding (CD) Feasibility Analysis.
Evaluates training-free visual contrastive decoding on M3D-LaMed audit results:
Formula:
    log p_CD(y) = (1 + alpha) * log p(y | real) - alpha * log p(y | blank)
    evaluated for alpha in {0.25, 0.5, 1.0, 2.0}

Features:
1. Ingests 4-option logits for real and blank (from kaggle_m3d_full.py).
2. Computes fraction of items where argmax differs between real and blank.
3. Computes distribution of |l_real - l_blank| (mean, median, IQR, max).
4. Simulates offline accuracy under contrastive decoding for various alpha values.
5. Computes bootstrap 95% CI of accuracy gain (Acc_CD - Acc_real).
6. Emits structured verdict: "worth pursuing" only if gain CI excludes 0, else "little room".

Usage:
    python partB_benchmark/cd_feasibility.py --input_csv partB_benchmark/results_full/m3d_full_per_item.csv
    python partB_benchmark/cd_feasibility.py --synthetic --n_items 150
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy.special import log_softmax


def parse_logits(val: Any) -> Dict[str, float]:
    """Parses JSON-encoded logits dictionary."""
    if isinstance(val, dict):
        return {k: float(v) for k, v in val.items()}
    if isinstance(val, str):
        try:
            d = json.loads(val)
            return {k: float(v) for k, v in d.items()}
        except Exception:
            return {}
    return {}


def bootstrap_ci_gain(acc_cd_list: List[int], acc_real_list: List[int], 
                      n_boot: int = 2000, seed: int = 42) -> Tuple[float, float, float]:
    """Computes bootstrap 95% CI of (Acc_CD - Acc_real)."""
    rng = np.random.RandomState(seed)
    diffs = np.array(acc_cd_list) - np.array(acc_real_list)
    point_gain = float(np.mean(diffs))
    n = len(diffs)

    boot_gains = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_gains.append(float(np.mean(diffs[idx])))

    ci_low = float(np.percentile(boot_gains, 2.5))
    ci_upp = float(np.percentile(boot_gains, 97.5))
    return point_gain, ci_low, ci_upp


def analyze_cd_feasibility(df: pd.DataFrame, alphas: List[float] = [0.25, 0.5, 1.0, 2.0],
                           n_boot: int = 2000, seed: int = 42) -> Dict[str, Any]:
    n_total = len(df)
    has_logits = "logits_real" in df.columns and "logits_blank" in df.columns

    if not has_logits:
        raise ValueError(
            "Input dataframe does not contain 'logits_real' and 'logits_blank'. "
            "Full candidate logits are required for offline contrastive decoding simulation. "
            "Ensure the data was generated using kaggle_m3d_full.py or use --synthetic."
        )

    argmax_differs = 0
    logit_diff_magnitudes = []
    valid_items = 0

    acc_real_flags = []
    acc_cd_flags: Dict[float, List[int]] = {a: [] for a in alphas}

    for _, row in df.iterrows():
        l_real_dict = parse_logits(row["logits_real"])
        l_blank_dict = parse_logits(row["logits_blank"])
        gold = str(row["gold_answer"] if "gold_answer" in row else row["gold_letter"]).strip()

        common_keys = sorted(set(l_real_dict.keys()) & set(l_blank_dict.keys()))
        if len(common_keys) < 2:
            continue

        valid_items += 1
        l_r = np.array([l_real_dict[k] for k in common_keys], dtype=np.float64)
        l_b = np.array([l_blank_dict[k] for k in common_keys], dtype=np.float64)

        # Logit differences
        diff_mag = np.abs(l_r - l_b)
        logit_diff_magnitudes.extend(diff_mag.tolist())

        # Argmax check
        k_real_max = common_keys[int(np.argmax(l_r))]
        k_blank_max = common_keys[int(np.argmax(l_b))]
        if k_real_max != k_blank_max:
            argmax_differs += 1

        is_real_correct = int(k_real_max == gold)
        acc_real_flags.append(is_real_correct)

        # Log-probabilities
        logp_r = log_softmax(l_r)
        logp_b = log_softmax(l_b)

        # CD simulation: log p_CD = (1 + alpha) * log p_real - alpha * log p_blank
        for a in alphas:
            score_cd = (1.0 + a) * logp_r - a * logp_b
            k_cd = common_keys[int(np.argmax(score_cd))]
            acc_cd_flags[a].append(int(k_cd == gold))

    if valid_items == 0:
        raise ValueError("No valid rows with matching candidate logits found.")

    frac_argmax_diff = argmax_differs / valid_items
    logit_diff_arr = np.array(logit_diff_magnitudes)

    logit_diff_summary = {
        "mean": float(np.mean(logit_diff_arr)),
        "median": float(np.median(logit_diff_arr)),
        "q25": float(np.percentile(logit_diff_arr, 25)),
        "q75": float(np.percentile(logit_diff_arr, 75)),
        "max": float(np.max(logit_diff_arr))
    }

    real_acc = float(np.mean(acc_real_flags))

    simulation_results = {}
    any_promising = False

    for a in alphas:
        cd_acc = float(np.mean(acc_cd_flags[a]))
        gain, ci_low, ci_upp = bootstrap_ci_gain(acc_cd_flags[a], acc_real_flags, n_boot=n_boot, seed=seed)
        is_significant = bool(ci_low > 0.0)
        if is_significant:
            any_promising = True

        simulation_results[str(a)] = {
            "alpha": a,
            "accuracy_cd": cd_acc,
            "accuracy_real": real_acc,
            "gain": gain,
            "ci_95_gain": [ci_low, ci_upp],
            "statistically_significant": is_significant
        }

    verdict = "worth pursuing" if any_promising else "little room"
    verdict_explanation = (
        f"At least one alpha configuration achieved statistically significant visual gain (95% CI > 0). "
        f"Contrastive decoding provides a viable training-free mitigation."
        if any_promising else
        f"None of the evaluated alpha configurations achieved statistically significant gain over real CT (all 95% CIs include 0 or negative). "
        f"Real and blank logits are strongly coupled (argmax differs in only {frac_argmax_diff*100:.1f}% of items). "
        f"There is little room for standard visual contrastive decoding without visual token recalibration."
    )

    return {
        "n_evaluated": valid_items,
        "fraction_argmax_differs": frac_argmax_diff,
        "logit_diff_magnitude": logit_diff_summary,
        "baseline_real_accuracy": real_acc,
        "cd_simulations": simulation_results,
        "verdict": verdict,
        "verdict_explanation": verdict_explanation
    }


def generate_synthetic_data(n_items: int = 150, seed: int = 42) -> pd.DataFrame:
    """Generates synthetic pilot audit dataframe with candidate logits for offline testing."""
    rng = np.random.RandomState(seed)
    records = []
    letters = ["A", "B", "C", "D"]

    for i in range(n_items):
        gold = rng.choice(letters, p=[0.35, 0.35, 0.20, 0.10])
        # Model exhibits strong language prior
        base_logits = rng.randn(4)
        blank_logits = base_logits + 0.1 * rng.randn(4)
        # Small visual shift
        real_logits = blank_logits + 0.05 * rng.randn(4)

        records.append({
            "question_id": f"q_synth_{i:04d}",
            "gold_letter": gold,
            "logits_real": json.dumps({letters[j]: float(real_logits[j]) for j in range(4)}),
            "logits_blank": json.dumps({letters[j]: float(blank_logits[j]) for j in range(4)})
        })
    return pd.DataFrame(records)


def analyze_pilot_argmax(csv_path: str):
    """Analyzes real pilot CSV where predictions exist but 4-option logits were not recorded."""
    df = pd.read_csv(csv_path)
    real_col = "real_pred" if "real_pred" in df.columns else "pred_real"
    blank_col = "blank_pred" if "blank_pred" in df.columns else "pred_blank"
    if real_col in df.columns and blank_col in df.columns:
        diff_mask = df[real_col] != df[blank_col]
        diff_count = int(diff_mask.sum())
        total = len(df)
        pct = (diff_count / total) * 100
        print("\n" + "=" * 70)
        print("REAL PILOT AUDIT PREDICTION DISAGREEMENT (m3d_pilot_per_item.csv)")
        print("=" * 70)
        print(f"Total Evaluated Questions:              {total}")
        print(f"Predictions Differ (real != blank):     {diff_count}/{total} ({pct:.2f}%)")
        print(f"Predictions Identical (real == blank):  {total - diff_count}/{total} ({100 - pct:.2f}%)")
        print("Status: Offline Contrastive Decoding simulation is NOT RUN on real logits")
        print("        (pilot recorded argmax predictions and gold logp, but not full 4-option logits).")
        print("        Full candidate logits are added in kaggle_m3d_full.py for the full benchmark.")
        print("=" * 70 + "\n")
        return diff_count, total, pct
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="partB_benchmark/results_full/m3d_full_per_item.csv")
    parser.add_argument("--output_json", default="partB_benchmark/results_full/cd_feasibility_results.json")
    parser.add_argument("--synthetic", action="store_true", help="Run on synthetic data for offline code verification")
    parser.add_argument("--pilot_csv", default="partB_benchmark/pilot result original/m3d_pilot_per_item.csv", help="Analyze real pilot predictions")
    parser.add_argument("--n_items", type=int, default=150)
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # If real pilot CSV exists, report the real argmax difference first
    if os.path.isfile(args.pilot_csv):
        analyze_pilot_argmax(args.pilot_csv)

    is_synthetic_run = args.synthetic or not os.path.isfile(args.input_csv)

    if is_synthetic_run:
        print("*" * 78)
        print("*** SYNTHETIC SELF-TEST: NOT REAL AUDIT DATA - FOR CODE VERIFICATION ONLY ***")
        print("*" * 78)
        print(f"[cd_feasibility] Generating synthetic benchmark data ({args.n_items} items)...")
        df = generate_synthetic_data(n_items=args.n_items, seed=args.seed)

        # Enforce SYNTHETIC_ prefix on output file
        out_dir = os.path.dirname(args.output_json) or "."
        base_name = os.path.basename(args.output_json)
        if not base_name.startswith("SYNTHETIC_"):
            args.output_json = os.path.join(out_dir, f"SYNTHETIC_{base_name}")
    else:
        print(f"[cd_feasibility] Loading audit data from {args.input_csv}...")
        df = pd.read_csv(args.input_csv)

    results = analyze_cd_feasibility(df, n_boot=args.n_boot, seed=args.seed)
    if is_synthetic_run:
        results["status"] = "SYNTHETIC SELF-TEST (NOT REAL AUDIT DATA - FOR CODE VERIFICATION ONLY)"

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[saved] {args.output_json}")

    banner_prefix = "[SYNTHETIC TEST] " if is_synthetic_run else ""
    print("\n" + "=" * 70)
    print(f"{banner_prefix}CONTRASTIVE DECODING (CD) FEASIBILITY AUDIT")
    if is_synthetic_run:
        print("*** WARNING: NUMBERS BELOW ARE FROM SYNTHETIC TEST DATA, NOT MODEL WEIGHTS ***")
    print("=" * 70)
    print(f"Evaluated Items:                {results['n_evaluated']}")
    print(f"Argmax Difference Rate:        {results['fraction_argmax_differs']*100:.2f}%")
    print(f"Logit Diff Magnitude Median:   {results['logit_diff_magnitude']['median']:.4f}")
    print(f"Baseline Real CT Accuracy:     {results['baseline_real_accuracy']*100:.2f}%\n")

    print(f"{banner_prefix}CD Simulation Results (log p_CD = (1+alpha)*log p_real - alpha*log p_blank):")
    print(f"{'Alpha':>6} | {'CD Acc':>8} | {'Gain (pp)':>10} | {'95% CI':>20} | {'Significant':>11}")
    print("-" * 65)
    for a_str, s in results["cd_simulations"].items():
        gain_pp = s["gain"] * 100
        ci_str = f"[{s['ci_95_gain'][0]*100:+.2f}, {s['ci_95_gain'][1]*100:+.2f}]"
        sig_str = "YES" if s["statistically_significant"] else "NO"
        print(f"{s['alpha']:6.2f} | {s['accuracy_cd']*100:7.2f}% | {gain_pp:+9.2f}% | {ci_str:>20} | {sig_str:>11}")

    print("\n" + "=" * 70)
    print(f"{banner_prefix}VERDICT: {results['verdict'].upper()}")
    print(f"Details: {results['verdict_explanation']}")
    if is_synthetic_run:
        print("*** NOTE: REAL AUDIT STATUS = NOT RUN (Awaiting full GPU audit with candidate logits) ***")
    print("=" * 70)


if __name__ == "__main__":
    main()
