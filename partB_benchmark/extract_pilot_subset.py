#!/usr/bin/env python3
"""
Extract a stratified 150-question pilot benchmark subset for M3D-LaMed evaluation on Kaggle T4.
Maintains exact family balance, seen/unseen domain ratio, and answer option balance.
"""

import json
import os
import random
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST_DIR = os.path.join(ROOT, "manifests")
SEED = 42

FAMILY_TARGETS = {
    "existence": 50,
    "laterality_volume": 40,
    "spatial_topography": 30,
    "volume_ranking": 30
}


def extract_pilot():
    rng = random.Random(SEED)
    
    with open(os.path.join(MANIFEST_DIR, "benchmark_real.json")) as f:
        real_all = json.load(f)
    with open(os.path.join(MANIFEST_DIR, "benchmark_shuffled.json")) as f:
        shuffled_all = json.load(f)
    with open(os.path.join(MANIFEST_DIR, "benchmark_swapped.json")) as f:
        swapped_all = json.load(f)

    shuffled_by_id = {q["question_id"]: q for q in shuffled_all}
    swapped_by_id = {q["question_id"]: q for q in swapped_all}

    df = pd.DataFrame(real_all)
    selected_ids = []

    # Sample per family with seen vs unseen stratification
    for fam, n_target in FAMILY_TARGETS.items():
        fam_df = df[df["family"] == fam]
        seen_df = fam_df[fam_df["m3d_domain"] == "seen_train_val"]
        unseen_df = fam_df[fam_df["m3d_domain"] == "unseen_test"]

        # ~73% seen, ~27% unseen
        n_unseen = max(1, int(n_target * 0.27))
        n_seen = n_target - n_unseen

        sampled_seen = rng.sample(seen_df["question_id"].tolist(), min(n_seen, len(seen_df)))
        sampled_unseen = rng.sample(unseen_df["question_id"].tolist(), min(n_unseen, len(unseen_df)))
        
        # Fill remaining if needed
        needed = n_target - (len(sampled_seen) + len(sampled_unseen))
        if needed > 0:
            remaining = [qid for qid in fam_df["question_id"].tolist() if qid not in sampled_seen and qid not in sampled_unseen]
            sampled_seen.extend(rng.sample(remaining, needed))

        selected_ids.extend(sampled_seen + sampled_unseen)

    print(f"Total pilot questions selected: {len(selected_ids)}")
    assert len(selected_ids) == 150, f"Expected 150 questions, got {len(selected_ids)}"

    pilot_real = [q for q in real_all if q["question_id"] in set(selected_ids)]
    pilot_shuffled = [shuffled_by_id[qid] for qid in selected_ids]
    pilot_swapped = [swapped_by_id[qid] for qid in selected_ids]

    out_real = os.path.join(MANIFEST_DIR, "pilot_real.json")
    out_shuff = os.path.join(MANIFEST_DIR, "pilot_shuffled.json")
    out_swap = os.path.join(MANIFEST_DIR, "pilot_swapped.json")

    with open(out_real, "w") as f:
        json.dump(pilot_real, f, indent=2)
    with open(out_shuff, "w") as f:
        json.dump(pilot_shuffled, f, indent=2)
    with open(out_swap, "w") as f:
        json.dump(pilot_swapped, f, indent=2)

    # Summary
    p_df = pd.DataFrame(pilot_real)
    print("\nPilot Question Families:\n", p_df["family"].value_counts().to_dict())
    print("\nPilot Domain Distribution:\n", p_df["m3d_domain"].value_counts().to_dict())
    print("\nPilot Answer Distribution:\n", p_df["answer_choice"].value_counts().to_dict())
    print(f"\n[Saved] {out_real}\n[Saved] {out_shuff}\n[Saved] {out_swap}")


if __name__ == "__main__":
    extract_pilot()
