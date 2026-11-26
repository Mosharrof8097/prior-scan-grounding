#!/usr/bin/env python3
"""
partB_benchmark/analyze_full.py

Comprehensive Statistical Analysis Suite for TotalSegmentator v2 3D VLM Audit.
Analyzes audit results from kaggle_m3d_full.py or pilot results:
1. Condition accuracies (real, blank, swap, shuffle_slices, text_only).
2. Chance floors and majority-class baselines.
3. Answer-position bias table (options A, B, C, D).
4. Case-cluster bootstrap 95% CIs (resampling distinct CT scan case_ids with replacement).
5. McNemar exact tests with discordant counts (b, c).
6. Two One-Sided Tests (TOST) equivalence testing for real vs blank (+/- 3 percentage points).
7. Pairwise prediction agreement matrix.
8. Delta log p analysis (mean, median, 95% CI, and top 5% trimmed).
9. Per-organ error stratified by anatomical region/category.
10. Holm-Bonferroni family-wise error rate corrections.

Usage:
    python partB_benchmark/analyze_full.py --input_csv partB_benchmark/results_full/m3d_full_per_item.csv
    python partB_benchmark/analyze_full.py --input_csv "partB_benchmark/pilot result original/m3d_pilot_per_item.csv"
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Any, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

# Add benchmark directory to path for class metadata
BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
if BENCH_DIR not in sys.path:
    sys.path.insert(0, BENCH_DIR)

try:
    from class_metadata import CATEGORIES, CRANIOCAUDAL_LEVELS
except ImportError:
    CATEGORIES = {}
    CRANIOCAUDAL_LEVELS = []


def load_question_lookup() -> Dict[str, Dict[str, Any]]:
    """Loads scan_id, target_structure, and metadata lookup from benchmark manifests."""
    lookup = {}
    manifest_paths = [
        os.path.join(BENCH_DIR, "manifests", "benchmark_real.json"),
        os.path.join(BENCH_DIR, "manifests", "pilot_real.json")
    ]
    for mp in manifest_paths:
        if os.path.isfile(mp):
            try:
                with open(mp) as f:
                    items = json.load(f)
                for it in items:
                    qid = it.get("question_id")
                    if qid and qid not in lookup:
                        lookup[qid] = {
                            "scan_id": it.get("scan_id", "unknown_scan"),
                            "target_structure": it.get("target_structure", ""),
                            "structures": it.get("structures", []),
                            "family": it.get("family", "unknown"),
                            "m3d_domain": it.get("m3d_domain", "unknown"),
                            "chance_floor": it.get("chance_floor", 0.25)
                        }
            except Exception:
                pass
    return lookup


def holm_bonferroni(p_values: List[float]) -> List[float]:
    """Applies Holm-Bonferroni step-down correction to a list of p-values."""
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, (m - rank) * p)
        cum_max = max(cum_max, adj_p)
        adjusted[orig_idx] = cum_max
    return adjusted


def compute_tost_equivalence(b: int, c: int, n: int, margin: float = 0.03) -> Dict[str, Any]:
    """
    Computes Two One-Sided Tests (TOST) for paired binary proportions (real vs blank).
    Equivalence bounds: [-margin, +margin].
    d_bar = (b - c) / n
    SE = sqrt( (b + c) - (b - c)^2 / n ) / n
    """
    diff = (b - c) / n
    denom = (b + c) - ((b - c) ** 2) / n
    se = math.sqrt(max(0.0, denom)) / n

    if se < 1e-12:
        p_tost = 0.0 if abs(diff) < margin else 1.0
        return {
            "mean_diff": diff,
            "margin": margin,
            "se": se,
            "z_lower": float("inf") if diff > -margin else -float("inf"),
            "z_upper": float("inf") if diff < margin else -float("inf"),
            "p_lower": 0.0 if diff > -margin else 1.0,
            "p_upper": 0.0 if diff < margin else 1.0,
            "p_tost": p_tost,
            "equivalent": bool(p_tost < 0.05)
        }

    # Test 1: H0: diff <= -margin vs H1: diff > -margin
    z_lower = (diff - (-margin)) / se
    p_lower = 1.0 - norm.cdf(z_lower)

    # Test 2: H0: diff >= +margin vs H1: diff < +margin
    z_upper = (diff - margin) / se
    p_upper = norm.cdf(z_upper)

    p_tost = max(p_lower, p_upper)
    return {
        "mean_diff": diff,
        "margin": margin,
        "se": se,
        "z_lower": float(z_lower),
        "z_upper": float(z_upper),
        "p_lower": float(p_lower),
        "p_upper": float(p_upper),
        "p_tost": float(p_tost),
        "equivalent": bool(p_tost < 0.05)
    }


def case_cluster_bootstrap(df: pd.DataFrame, stat_func, cluster_col: str = "case_id",
                           n_boot: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """
    Non-parametric case-cluster bootstrap.
    Resamples unique cluster_col identifiers with replacement,
    aggregates all observations belonging to the sampled clusters,
    and returns (mean_stat, ci_lower_2.5, ci_upper_97.5).
    """
    rng = np.random.RandomState(seed)
    unique_clusters = df[cluster_col].unique()
    n_clusters = len(unique_clusters)

    # Pre-group cluster row indices for speed
    cluster_to_indices = df.groupby(cluster_col).indices

    point_val = stat_func(df)
    boot_vals = []

    for _ in range(n_boot):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        sampled_indices = np.concatenate([cluster_to_indices[c] for c in sampled_clusters])
        boot_df = df.iloc[sampled_indices]
        boot_vals.append(stat_func(boot_df))

    ci_lower = float(np.percentile(boot_vals, 2.5))
    ci_upper = float(np.percentile(boot_vals, 97.5))
    return float(point_val), ci_lower, ci_upper


def run_analysis(input_csv: str, output_dir: str, n_boot: int = 10000, seed: int = 42) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    lookup = load_question_lookup()

    # Standardize column names if ingesting older pilot per-item CSV
    col_mapping = {
        "real_correct": "correct_real",
        "blank_correct": "correct_blank",
        "swap_correct": "correct_swap",
        "shuff_correct": "correct_shuff",
        "real_pred": "pred_real",
        "blank_pred": "pred_blank"
    }
    df = df.rename(columns=col_mapping)

    # Ensure case_id is populated for case-cluster bootstrap
    if "case_id" not in df.columns:
        df["case_id"] = df["question_id"].apply(lambda q: lookup.get(q, {}).get("scan_id", f"case_{q}"))
    if "target_structure" not in df.columns:
        df["target_structure"] = df["question_id"].apply(lambda q: lookup.get(q, {}).get("target_structure", ""))

    n_total = len(df)
    n_cases = df["case_id"].nunique()

    # Detect available conditions
    conditions = []
    for cond in ["real", "blank", "swap", "shuff", "shuffle_slices", "shuffle_options", "textonly", "text_only"]:
        if f"correct_{cond}" in df.columns:
            conditions.append(cond)

    # 1. Overall & Subgroup Accuracy Table
    groups = {"overall": df}
    for fam in df["family"].unique():
        groups[f"family_{fam}"] = df[df["family"] == fam]
    if "m3d_domain" in df.columns:
        for dom in df["m3d_domain"].unique():
            groups[f"domain_{dom}"] = df[df["m3d_domain"] == dom]

    subgroup_records = []
    p_mcnemar_blank_list = []
    p_mcnemar_swap_list = []

    for grp_name, gdf in groups.items():
        gn = len(gdf)
        rec: Dict[str, Any] = {"group": grp_name, "n": gn, "n_cases": gdf["case_id"].nunique()}

        # Per-group chance baseline
        if "family_existence" in grp_name:
            rec["chance_baseline"] = 0.50
        elif any(f in grp_name for f in ["laterality_volume", "spatial_topography", "volume_ranking"]):
            rec["chance_baseline"] = 0.25
        else:
            rec["chance_baseline"] = float(gdf["family"].map(lambda f: 0.50 if f == "existence" else 0.25).mean())

        # Per-group majority-choice baseline
        maj_choice = gdf["gold_letter"].mode()[0]
        maj_acc = float((gdf["gold_letter"] == maj_choice).mean())
        rec["majority_choice"] = maj_choice
        rec["majority_baseline"] = maj_acc

        # Accuracies with cluster bootstrap 95% CIs
        for cond in conditions:
            col = f"correct_{cond}"
            acc, low, upp = case_cluster_bootstrap(
                gdf, lambda d: float(d[col].mean()), cluster_col="case_id", n_boot=n_boot, seed=seed
            )
            rec[f"acc_{cond}"] = acc
            rec[f"ci_{cond}"] = f"[{low:.4f}, {upp:.4f}]"

        # Delta Blank (Real - Blank)
        if "correct_blank" in gdf.columns:
            d_val, d_low, d_upp = case_cluster_bootstrap(
                gdf, lambda d: float((d["correct_real"] - d["correct_blank"]).mean()),
                cluster_col="case_id", n_boot=n_boot, seed=seed
            )
            rec["delta_blank"] = d_val
            rec["ci_delta_blank"] = f"[{d_low:.4f}, {d_upp:.4f}]"

            # McNemar discordant pairs
            b = int(((gdf["correct_real"] == 1) & (gdf["correct_blank"] == 0)).sum())
            c = int(((gdf["correct_real"] == 0) & (gdf["correct_blank"] == 1)).sum())
            rec["discordant_blank_b"] = b
            rec["discordant_blank_c"] = c
            p_val = binomtest(b, b + c, 0.5).pvalue if (b + c) > 0 else 1.0
            rec["p_mcnemar_blank"] = float(p_val)
            p_mcnemar_blank_list.append(float(p_val))

            # TOST equivalence (+/- 3 pp)
            tost_res = compute_tost_equivalence(b, c, gn, margin=0.03)
            rec["tost_p"] = tost_res["p_tost"]
            rec["tost_equivalent"] = tost_res["equivalent"]
            rec["tost_status"] = (
                f"equivalent within +/-3.0pp (p_TOST={tost_res['p_tost']:.4f})"
                if tost_res["equivalent"] else
                f"equivalence NOT established (p_TOST={tost_res['p_tost']:.4f})"
            )

        # Delta Swap (Real - Swap)
        if "correct_swap" in gdf.columns:
            d_s_val, d_s_low, d_s_upp = case_cluster_bootstrap(
                gdf, lambda d: float((d["correct_real"] - d["correct_swap"]).mean()),
                cluster_col="case_id", n_boot=n_boot, seed=seed
            )
            rec["delta_swap"] = d_s_val
            rec["ci_delta_swap"] = f"[{d_s_low:.4f}, {d_s_upp:.4f}]"

            b_sw = int(((gdf["correct_real"] == 1) & (gdf["correct_swap"] == 0)).sum())
            c_sw = int(((gdf["correct_real"] == 0) & (gdf["correct_swap"] == 1)).sum())
            rec["discordant_swap_b"] = b_sw
            rec["discordant_swap_c"] = c_sw
            p_sw = binomtest(b_sw, b_sw + c_sw, 0.5).pvalue if (b_sw + c_sw) > 0 else 1.0
            rec["p_mcnemar_swap"] = float(p_sw)
            p_mcnemar_swap_list.append(float(p_sw))

        subgroup_records.append(rec)

    # Apply Holm-Bonferroni correction
    adj_blank = holm_bonferroni(p_mcnemar_blank_list)
    adj_swap = holm_bonferroni(p_mcnemar_swap_list)
    for i, rec in enumerate(subgroup_records):
        if i < len(adj_blank):
            rec["p_mcnemar_blank_holm"] = adj_blank[i]
        if i < len(adj_swap):
            rec["p_mcnemar_swap_holm"] = adj_swap[i]

    df_subgroups = pd.DataFrame(subgroup_records)
    subgroups_csv = os.path.join(output_dir, "subgroup_metrics.csv")
    df_subgroups.to_csv(subgroups_csv, index=False)

    # 2. Answer-Position Bias Table
    bias_records = []
    gold_counts = df["gold_letter"].value_counts(normalize=True).to_dict()
    for opt in ["A", "B", "C", "D"]:
        row = {"option": opt, "gold_freq": gold_counts.get(opt, 0.0)}
        for cond in conditions:
            pred_col = f"pred_{cond}"
            if pred_col in df.columns:
                pred_counts = df[pred_col].value_counts(normalize=True).to_dict()
                row[f"pred_freq_{cond}"] = pred_counts.get(opt, 0.0)
        bias_records.append(row)
    df_bias = pd.DataFrame(bias_records)
    bias_csv = os.path.join(output_dir, "option_bias_table.csv")
    df_bias.to_csv(bias_csv, index=False)

    # Majority class baseline
    majority_letter = df["gold_letter"].mode()[0]
    majority_acc = float((df["gold_letter"] == majority_letter).mean())

    # 3. Pairwise Prediction Agreement Matrix
    agreement_matrix: Dict[str, Dict[str, float]] = {}
    for c1 in conditions:
        col1 = f"pred_{c1}"
        if col1 not in df.columns:
            continue
        agreement_matrix[c1] = {}
        for c2 in conditions:
            col2 = f"pred_{c2}"
            if col2 not in df.columns:
                continue
            agr = float((df[col1] == df[col2]).mean())
            agreement_matrix[c1][c2] = agr

    df_agree = pd.DataFrame(agreement_matrix)
    agree_csv = os.path.join(output_dir, "prediction_agreement_matrix.csv")
    df_agree.to_csv(agree_csv)

    # 4. Delta Log P Distribution Analysis
    delta_logp_stats = {}
    if "delta_logp" in df.columns:
        vals = df["delta_logp"].dropna().values
        mean_dl = float(np.mean(vals))
        median_dl = float(np.median(vals))

        # Bootstrap CI for delta log p
        dl_pt, dl_low, dl_upp = case_cluster_bootstrap(
            df, lambda d: float(d["delta_logp"].mean()), cluster_col="case_id", n_boot=n_boot, seed=seed
        )

        # 5% trimmed mean (2.5% on each tail)
        q_low, q_high = np.percentile(vals, 2.5), np.percentile(vals, 97.5)
        trimmed_vals = vals[(vals >= q_low) & (vals <= q_high)]
        trimmed_mean = float(np.mean(trimmed_vals))

        delta_logp_stats = {
            "mean": mean_dl,
            "median": median_dl,
            "ci_95": [dl_low, dl_upp],
            "trimmed_mean_5pct": trimmed_mean,
            "q25": float(np.percentile(vals, 25)),
            "q75": float(np.percentile(vals, 75))
        }

    # 5. Organ-Stratified Analysis
    organ_records = []
    # Map target structure to category
    structure_to_cat = {}
    for cat, struct_list in CATEGORIES.items():
        for s in struct_list:
            structure_to_cat[s] = cat

    df["anatomical_category"] = df["target_structure"].apply(lambda s: structure_to_cat.get(s, "other"))

    for cat_name, cat_df in df.groupby("anatomical_category"):
        if len(cat_df) >= 3:
            organ_records.append({
                "anatomical_category": cat_name,
                "n_items": len(cat_df),
                "n_cases": cat_df["case_id"].nunique(),
                "acc_real": float(cat_df["correct_real"].mean()),
                "acc_blank": float(cat_df["correct_blank"].mean()) if "correct_blank" in cat_df else None,
                "acc_swap": float(cat_df["correct_swap"].mean()) if "correct_swap" in cat_df else None
            })

    df_organ = pd.DataFrame(organ_records)
    organ_csv = os.path.join(output_dir, "organ_stratified_metrics.csv")
    df_organ.to_csv(organ_csv, index=False)

    # 6. Overall Summary JSON
    overall_rec = next(r for r in subgroup_records if r["group"] == "overall")
    summary = {
        "dataset_metrics": {
            "total_items": n_total,
            "distinct_cases": n_cases,
            "majority_choice": majority_letter,
            "majority_accuracy": majority_acc
        },
        "overall_accuracies": {cond: overall_rec[f"acc_{cond}"] for cond in conditions},
        "overall_cis": {cond: overall_rec[f"ci_{cond}"] for cond in conditions},
        "real_vs_blank": {
            "delta": overall_rec.get("delta_blank"),
            "ci_95": overall_rec.get("ci_delta_blank"),
            "discordant_b_c": [overall_rec.get("discordant_blank_b"), overall_rec.get("discordant_blank_c")],
            "p_mcnemar": overall_rec.get("p_mcnemar_blank"),
            "tost_margin": 0.03,
            "tost_p": overall_rec.get("tost_p"),
            "tost_equivalent_3pp": overall_rec.get("tost_equivalent")
        },
        "delta_logp": delta_logp_stats,
        "prediction_agreement": agreement_matrix
    }

    sum_json = os.path.join(output_dir, "summary_metrics.json")
    with open(sum_json, "w") as f:
        json.dump(summary, f, indent=2)

    # 7. Generate Full Markdown Report
    report_md = os.path.join(output_dir, "full_analysis_report.md")
    with open(report_md, "w") as f:
        f.write("# Full Audit Statistical Analysis Report\n\n")
        f.write(f"- **Evaluated Items**: {n_total} items across {n_cases} distinct cases\n")
        f.write(f"- **Bootstrap Replicates**: {n_boot} case-cluster resamples\n")
        f.write(f"- **Majority-Class Baseline**: {majority_acc:.4f} (Option {majority_letter})\n\n")

        f.write("## 1. Condition Accuracies & Hypothesis Testing\n\n")
        f.write("| Group | N | Chance | Majority | Real Acc | Blank Acc | Swapped Acc | Delta Blank [95% CI] | McNemar p (adj) | TOST (+/-3pp Margin) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |\n")
        for r in subgroup_records:
            g_lbl = r["group"]
            gn = r["n"]
            ch_b = f"{r.get('chance_baseline', 0.25):.2f}"
            maj_b = f"{r.get('majority_baseline', 0.0):.4f} ({r.get('majority_choice', '')})"
            acc_r = f"{r.get('acc_real', 0.0):.4f}"
            acc_b = f"{r.get('acc_blank', 0.0):.4f}" if "acc_blank" in r else "N/A"
            acc_s = f"{r.get('acc_swap', 0.0):.4f}" if "acc_swap" in r else "N/A"
            d_b = f"{r.get('delta_blank', 0.0):+.4f} {r.get('ci_delta_blank', '')}" if "delta_blank" in r else "N/A"
            p_m = f"{r.get('p_mcnemar_blank', 1.0):.4f} ({r.get('p_mcnemar_blank_holm', 1.0):.4f})" if "p_mcnemar_blank" in r else "N/A"
            tost_txt = r.get("tost_status", f"p_TOST={r.get('tost_p', 1.0):.4f}")
            f.write(f"| {g_lbl} | {gn} | {ch_b} | {maj_b} | {acc_r} | {acc_b} | {acc_s} | {d_b} | {p_m} | {tost_txt} |\n")

        def df_to_md(d: pd.DataFrame, include_index: bool = True) -> str:
            lines = []
            cols = ([d.index.name or ""] if include_index else []) + list(d.columns)
            lines.append("| " + " | ".join(str(c) for c in cols) + " |")
            lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
            for idx, row in d.iterrows():
                vals = ([str(idx)] if include_index else []) + [str(v) for v in row.values]
                lines.append("| " + " | ".join(vals) + " |")
            return "\n".join(lines)

        f.write("\n## 2. Pairwise Prediction Agreement Matrix\n\n")
        f.write(df_to_md(df_agree, include_index=True) + "\n\n")

        f.write("## 3. Answer Position Bias\n\n")
        f.write(df_to_md(df_bias, include_index=False) + "\n\n")

        if delta_logp_stats:
            f.write("## 4. Delta Log P Distribution\n\n")
            f.write(f"- Mean: `{delta_logp_stats['mean']:.4f}`\n")
            f.write(f"- Median: `{delta_logp_stats['median']:.4f}`\n")
            f.write(f"- 95% Case-Cluster CI: `[{delta_logp_stats['ci_95'][0]:.4f}, {delta_logp_stats['ci_95'][1]:.4f}]`\n")
            f.write(f"- Top 5% Outlier-Trimmed Mean: `{delta_logp_stats['trimmed_mean_5pct']:.4f}`\n")

    print(f"[analysis] Analysis complete! Results written to: {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default="partB_benchmark/results_full/m3d_full_per_item.csv")
    parser.add_argument("--output_dir", default="partB_benchmark/results_full")
    parser.add_argument("--n_boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Fallback to pilot results if full results do not yet exist
    if not os.path.isfile(args.input_csv):
        pilot_csv = os.path.join(BENCH_DIR, "pilot result original", "m3d_pilot_per_item.csv")
        if os.path.isfile(pilot_csv):
            print(f"[analysis] '{args.input_csv}' not found. Falling back to pilot data: {pilot_csv}")
            args.input_csv = pilot_csv

    run_analysis(args.input_csv, args.output_dir, n_boot=args.n_boot, seed=args.seed)


if __name__ == "__main__":
    main()
