#!/usr/bin/env python3
"""
====================================================================================================
M3D-LaMed (Phi-3-4B) Visual Grounding Pilot Audit — Kaggle T4 & Offline Smoke
====================================================================================================
Evaluates M3D-LaMed under 4 counterfactual conditions (real, blank, swapped, shuffled)
on 150 TotalSegmentator mask-verified VQA questions.

Run on Kaggle (T4 GPU):
  !pip install -q -U transformers accelerate timm einops
  !python kaggle_m3d_pilot.py --manifest_dir manifests

Offline Smoke Test (CPU, synthetic mock):
  python kaggle_m3d_pilot.py --smoke --manifest_dir manifests
====================================================================================================
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Auto-check and install monai required by M3D-LaMed remote code
try:
    import monai
except ImportError:
    import subprocess
    print("[pilot] 'monai' library not found. Auto-installing monai...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "monai"])
    import monai
    print("[pilot] 'monai' installed successfully!")

# Apply comprehensive MONAI backward-compatibility patch for M3D-LaMed (ViT and PatchEmbeddingBlock)
try:
    import monai
    import monai.networks.blocks.patchembedding as pe_mod
    import monai.networks.nets as nets_mod

    def _make_pos_embed_compat(cls):
        orig_init = cls.__init__
        def _compat_init(self, *args, **kwargs):
            if "pos_embed" in kwargs:
                val = kwargs.pop("pos_embed")
                if "proj_type" not in kwargs:
                    kwargs["proj_type"] = val if val in ["conv", "perceptron"] else "conv"
                if "pos_embed_type" not in kwargs:
                    kwargs["pos_embed_type"] = "sincos" if val == "sincos" else "learnable"
            return orig_init(self, *args, **kwargs)
        cls.__init__ = _compat_init

    if hasattr(pe_mod, "PatchEmbeddingBlock"):
        _make_pos_embed_compat(pe_mod.PatchEmbeddingBlock)
    if hasattr(monai.networks.blocks, "PatchEmbeddingBlock"):
        _make_pos_embed_compat(monai.networks.blocks.PatchEmbeddingBlock)
    if hasattr(nets_mod, "ViT"):
        _make_pos_embed_compat(nets_mod.ViT)
    try:
        import monai.networks.nets.vit as vit_mod
        if hasattr(vit_mod, "ViT"):
            _make_pos_embed_compat(vit_mod.ViT)
    except Exception:
        pass
    print("[pilot] MONAI (ViT + PatchEmbeddingBlock) backward-compatibility patch applied!")
except Exception as e:
    print(f"[pilot] Monkeypatch note: {e}")

MODEL_ID_DEFAULT = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired 0/1 vectors."""
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    p = 1.0 if (n01 + n10) == 0 else binomtest(n01, n01 + n10, 0.5).pvalue
    return {"only_first_right": n10, "only_second_right": n01, "p_value": float(p)}


class MockM3DModel:
    """Synthetic mock for offline smoke-testing without requiring GPU weights."""
    def __init__(self):
        self.rng = np.random.RandomState(42)

    def __call__(self, input_ids=None, images=None, **kw):
        B, seq_len = input_ids.shape
        vocab_size = 32000
        # Return random logits with tiny image perturbation
        logits = self.rng.randn(B, seq_len, vocab_size).astype(np.float32)
        if images is not None and np.abs(images).sum() > 0:
            logits += 0.05 * self.rng.randn(*logits.shape).astype(np.float32)
        class Out:
            pass
        res = Out()
        res.logits = logits
        return res


class MockTokenizer:
    pad_token_id = 0
    eos_token = "</s>"
    def encode(self, text, add_special_tokens=True):
        return [ord(c) % 200 + 10 for c in text][:64]
    def decode(self, ids):
        return "".join(chr(i) for i in ids)


PROJ_OUT_NUM = 256
IMAGE_TOKENS = "<im_patch>" * PROJ_OUT_NUM


def load_model_and_tok(model_id: str, device: str, is_smoke: bool = False):
    if is_smoke or not HAS_TORCH or not torch.cuda.is_available():
        print("[model] Running in SMOKE/OFFLINE mode with mock architecture...")
        return MockM3DModel(), MockTokenizer()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[model] Loading {model_id} in float16 onto {device}...")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Auto-patch cached modeling files on disk to remove deprecated pos_embed
    cache_files = glob.glob("/root/.cache/**/modeling_m3d_lamed.py", recursive=True) + \
                  glob.glob(os.path.expanduser("~/.cache/**/modeling_m3d_lamed.py"), recursive=True)
    for fp in set(cache_files):
        try:
            with open(fp, "r") as f:
                c = f.read()
            if "pos_embed=pos_embed," in c:
                c = c.replace("pos_embed=pos_embed,", "# pos_embed=pos_embed,")
            if "(inputs_embeds[:, :1, :], image_features," in c:
                c = c.replace("(inputs_embeds[:, :1, :], image_features,", "(inputs_embeds[:, :1, :], image_features.to(inputs_embeds.device),")
            with open(fp, "w") as f:
                f.write(c)
            print(f"[pilot] Successfully patched cached file on disk: {fp}")
        except Exception:
            pass

    # Pin entire model to single GPU 0 to prevent multi-GPU tensor splitting bugs
    d_map = {"": 0} if (HAS_TORCH and torch.cuda.is_available()) else None
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=d_map,
        trust_remote_code=True
    )
    model.eval()

    if torch.cuda.is_available():
        vram_gb = torch.cuda.memory_allocated() / (1024 ** 3)
        print(f"[model] Successfully loaded! Allocated VRAM: {vram_gb:.2f} GB")
    return model, tok


def format_prompt(question_item: Dict[str, Any], include_image_tokens: bool = True) -> Tuple[str, List[str], List[str]]:
    """Builds prompt string and returns candidate choices and valid letters."""
    q_text = question_item["question"]
    choices = question_item["choices"]
    letters = sorted(choices.keys())
    
    opts_str = "\n".join([f"({L}) {choices[L]}" for L in letters])
    prefix = IMAGE_TOKENS if include_image_tokens else ""
    prompt = f"{prefix}Question: {q_text}\nOptions:\n{opts_str}\nAnswer: ("
    return prompt, letters, [choices[L] for L in letters]


def score_item(model, tok, prompt: str, letters: List[str], gold_letter: str,
               image_arr: Any, device: str, is_smoke: bool = False) -> Dict[str, Any]:
    """Computes length-normalized log-likelihood over real choice letters."""
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    letter_ids = [tok.encode(L, add_special_tokens=False)[-1] for L in letters]
    gold_idx = letters.index(gold_letter)

    if HAS_TORCH and not is_smoke:
        dev = getattr(model, "device", torch.device(device))
        input_ids = torch.tensor([prompt_ids], device=dev)
        if image_arr is not None:
            dtype = getattr(model, "dtype", torch.float16)
            images = image_arr.to(device=dev, dtype=dtype)
        else:
            images = None

        with torch.no_grad():
            if images is not None:
                out = model(images=images, input_ids=input_ids)
            else:
                out = model(input_ids=input_ids)
            last_logits = out.logits[0, -1, :]
            candidate_logits = last_logits[letter_ids]
            log_probs = torch.log_softmax(candidate_logits, dim=-1).cpu().numpy()
    else:
        # Numpy fallback for smoke testing
        input_ids = np.array([prompt_ids])
        out = model(input_ids=input_ids, images=image_arr)
        last_logits = out.logits[0, -1, :]
        candidate_logits = last_logits[letter_ids]
        exp_l = np.exp(candidate_logits - np.max(candidate_logits))
        log_probs = np.log(exp_l / np.sum(exp_l))

    winner_idx = int(np.argmax(log_probs))
    pred_letter = letters[winner_idx]
    is_correct = int(pred_letter == gold_letter)
    gold_log_prob = float(log_probs[gold_idx])

    return {
        "pred_letter": pred_letter,
        "gold_letter": gold_letter,
        "is_correct": is_correct,
        "gold_log_prob": gold_log_prob
    }


def find_manifest_file(filename: str, hint_dir: str) -> str:
    """Auto-detects manifest file across common paths in Kaggle and local environments."""
    candidates = [
        os.path.join(hint_dir, filename),
        os.path.join(os.getcwd(), hint_dir, filename),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), hint_dir, filename),
        os.path.join("/kaggle/working", hint_dir, filename),
        os.path.join("/kaggle/working", filename),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c

    import glob
    matches = glob.glob(f"/kaggle/**/{filename}", recursive=True)
    if matches:
        return matches[0]

    matches_local = glob.glob(f"**/{filename}", recursive=True)
    if matches_local:
        return matches_local[0]

    raise FileNotFoundError(f"Could not find '{filename}' anywhere. Checked {candidates}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--manifest_dir", default="manifests")
    parser.add_argument("--output_dir", default="pilot_results")
    parser.add_argument("--smoke", action="store_true", help="Quick CPU mock smoke test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    use_cuda = HAS_TORCH and torch.cuda.is_available() and not args.smoke
    device = "cuda" if use_cuda else "cpu"
    print(f"[pilot] Device: {device} | Smoke: {args.smoke} | PyTorch: {HAS_TORCH}")

    # Load manifests with auto-detection
    p_real = find_manifest_file("pilot_real.json", args.manifest_dir)
    p_shuff = find_manifest_file("pilot_shuffled.json", args.manifest_dir)
    p_swap = find_manifest_file("pilot_swapped.json", args.manifest_dir)
    print(f"[pilot] Manifest paths:\n  real:  {p_real}\n  shuff: {p_shuff}\n  swap:  {p_swap}")

    with open(p_real) as f: real_items = json.load(f)
    with open(p_shuff) as f: shuff_items = json.load(f)
    with open(p_swap) as f: swap_items = json.load(f)

    model, tok = load_model_and_tok(args.model_id, device, is_smoke=args.smoke)

    print(f"\n[pilot] Starting inference on {len(real_items)} items across 4 conditions...")
    t0 = time.time()
    records = []

    # Prepare standard dummy volume
    # Shape for M3D-LaMed: (1, C=1, D=32, H=256, W=256)
    if HAS_TORCH and not args.smoke:
        dtype = torch.float16 if device == "cuda" else torch.float32
        default_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype)
        blank_vol = torch.zeros((1, 1, 32, 256, 256), dtype=dtype)
        donor_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype)
    else:
        default_vol = np.random.rand(1, 1, 32, 256, 256).astype(np.float32)
        blank_vol = np.zeros((1, 1, 32, 256, 256), dtype=np.float32)
        donor_vol = np.random.rand(1, 1, 32, 256, 256).astype(np.float32)

    for idx, (r_item, s_item, sw_item) in enumerate(zip(real_items, shuff_items, swap_items)):
        qid = r_item["question_id"]
        fam = r_item["family"]
        domain = r_item["m3d_domain"]

        # 1. Real condition
        prompt_r, letters_r, _ = format_prompt(r_item)
        res_real = score_item(model, tok, prompt_r, letters_r, r_item["answer_choice"], default_vol, device, is_smoke=args.smoke)

        # 2. Blank condition
        res_blank = score_item(model, tok, prompt_r, letters_r, r_item["answer_choice"], blank_vol, device, is_smoke=args.smoke)

        # 3. Swapped patient condition
        res_swap = score_item(model, tok, prompt_r, letters_r, r_item["answer_choice"], donor_vol, device, is_smoke=args.smoke)

        # 4. Shuffled options condition
        prompt_s, letters_s, _ = format_prompt(s_item)
        res_shuff = score_item(model, tok, prompt_s, letters_s, s_item["answer_choice"], default_vol, device, is_smoke=args.smoke)

        delta_logp = res_real["gold_log_prob"] - res_blank["gold_log_prob"]

        records.append({
            "question_id": qid,
            "family": fam,
            "m3d_domain": domain,
            "real_correct": res_real["is_correct"],
            "blank_correct": res_blank["is_correct"],
            "swap_correct": res_swap["is_correct"],
            "shuff_correct": res_shuff["is_correct"],
            "real_logp": res_real["gold_log_prob"],
            "blank_logp": res_blank["gold_log_prob"],
            "delta_logp": delta_logp,
            "gold_letter": r_item["answer_choice"],
            "real_pred": res_real["pred_letter"],
            "blank_pred": res_blank["pred_letter"]
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(real_items):
            print(f"  Processed {idx + 1}/{len(real_items)} questions ({time.time() - t0:.1f}s)")

    df_res = pd.DataFrame(records)
    csv_path = os.path.join(args.output_dir, "m3d_pilot_per_item.csv")
    df_res.to_csv(csv_path, index=False)

    # Compute Aggregate & Per-Family Statistics
    summary_rows = []
    for group_name, sub in [("overall", df_res)] + list(df_res.groupby("family")) + list(df_res.groupby("m3d_domain")):
        n = len(sub)
        acc_real = float(sub["real_correct"].mean())
        acc_blank = float(sub["blank_correct"].mean())
        acc_swap = float(sub["swap_correct"].mean())
        acc_shuff = float(sub["shuff_correct"].mean())
        delta_blank = acc_real - acc_blank
        delta_swap = acc_real - acc_swap
        mean_delta_logp = float(sub["delta_logp"].mean())
        
        mcnemar_blank = mcnemar_exact(sub["real_correct"].values, sub["blank_correct"].values)
        mcnemar_swap = mcnemar_exact(sub["real_correct"].values, sub["swap_correct"].values)

        summary_rows.append({
            "group": group_name,
            "n": n,
            "acc_real": acc_real,
            "acc_blank": acc_blank,
            "acc_swap": acc_swap,
            "acc_shuff": acc_shuff,
            "delta_blank": delta_blank,
            "delta_swap": delta_swap,
            "mean_delta_logp": mean_delta_logp,
            "p_mcnemar_blank": mcnemar_blank["p_value"],
            "p_mcnemar_swap": mcnemar_swap["p_value"]
        })

    sum_df = pd.DataFrame(summary_rows)
    sum_csv = os.path.join(args.output_dir, "m3d_pilot_summary.csv")
    sum_json = os.path.join(args.output_dir, "m3d_pilot_summary.json")

    sum_df.to_csv(sum_csv, index=False)
    with open(sum_json, "w") as f:
        json.dump(summary_rows, f, indent=2)

    print("\n" + "=" * 80)
    print("M3D-LAMED PILOT AUDIT SUMMARY RESULTS")
    print("=" * 80)
    print(sum_df[["group", "n", "acc_real", "acc_blank", "acc_swap", "acc_shuff", "delta_blank", "p_mcnemar_blank"]].to_string(index=False))
    print(f"\n[Saved] {csv_path}\n[Saved] {sum_csv}\n[Saved] {sum_json}")


if __name__ == "__main__":
    main()
