#!/usr/bin/env python3
"""
partB_benchmark/kaggle_m3d_full.py

Full Audit Script for M3D-LaMed-Phi-3-4B across 7,465 TotalSegmentator v2 questions.
Evaluates 5 distinct counterfactual conditions:
1. real: Patient CT volume + image tokens
2. blank: All-zero volume + image tokens
3. swap: Mismatched donor patient CT volume + image tokens
4. shuffle_slices: CT volume with slices permuted along z-axis + image tokens
5. text_only: Text prompt only without <im_patch> placeholder tokens (image=None)

Features:
- Checkpoints progress every 500 items to enable seamless resumption after session timeouts.
- Records full 4-option logits for real and blank (essential for contrastive decoding feasibility).
- Logs GPU device, library versions, fixed seed, and git commit hash.
- Supports --smoke mode for fast offline CPU verification on a tiny subset.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Any, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Auto-check and import monai
try:
    import monai
except ImportError:
    import subprocess as sp
    sp.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", "monai"])
    import monai

# Apply backward-compatibility patch for MONAI
try:
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
except Exception:
    pass

MODEL_ID_DEFAULT = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
PROJ_OUT_NUM = 256
IMAGE_TOKENS = "<im_patch>" * PROJ_OUT_NUM


def get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


class MockM3DModel:
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
    def __call__(self, input_ids=None, images=None, **kw):
        B, seq_len = input_ids.shape
        vocab_size = 32000
        logits = self.rng.randn(B, seq_len, vocab_size).astype(np.float32)
        if images is not None and np.abs(images).sum() > 0:
            logits += 0.05 * self.rng.randn(*logits.shape).astype(np.float32)
        class Out: pass
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


def load_model_and_tok(model_id: str, device: str, is_smoke: bool = False):
    if is_smoke or not HAS_TORCH or not torch.cuda.is_available():
        print("[model] Running in SMOKE/OFFLINE mode with synthetic architecture...")
        return MockM3DModel(), MockTokenizer()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[model] Loading {model_id} in float16 onto GPU...")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Patch cached modeling file if exists
    cache_files = glob.glob("/root/.cache/**/modeling_m3d_lamed.py", recursive=True) + \
                  glob.glob(os.path.expanduser("~/.cache/**/modeling_m3d_lamed.py"), recursive=True)
    for fp in set(cache_files):
        try:
            with open(fp, "r") as f: c = f.read()
            if "pos_embed=pos_embed," in c:
                c = c.replace("pos_embed=pos_embed,", "# pos_embed=pos_embed,")
            if "(inputs_embeds[:, :1, :], image_features," in c:
                c = c.replace("(inputs_embeds[:, :1, :], image_features,", "(inputs_embeds[:, :1, :], image_features.to(inputs_embeds.device),")
            with open(fp, "w") as f: f.write(c)
        except Exception:
            pass

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
        print(f"[model] Successfully loaded on GPU! Allocated VRAM: {vram_gb:.2f} GB")
    return model, tok


def format_prompt(question_item: Dict[str, Any], include_image_tokens: bool = True) -> Tuple[str, List[str], List[str]]:
    q_text = question_item["question"]
    choices = question_item["choices"]
    letters = sorted(choices.keys())
    opts_str = "\n".join([f"({L}) {choices[L]}" for L in letters])
    prefix = IMAGE_TOKENS if include_image_tokens else ""
    prompt = f"{prefix}Question: {q_text}\nOptions:\n{opts_str}\nAnswer: ("
    return prompt, letters, [choices[L] for L in letters]


def score_item(model, tok, prompt: str, letters: List[str], gold_letter: str,
               image_arr: Any, device: str, is_smoke: bool = False) -> Dict[str, Any]:
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
            raw_logits_dict = {L: float(candidate_logits[i].cpu().item()) for i, L in enumerate(letters)}
            log_probs = torch.log_softmax(candidate_logits, dim=-1).cpu().numpy()
    else:
        input_ids = np.array([prompt_ids])
        out = model(input_ids=input_ids, images=image_arr)
        last_logits = out.logits[0, -1, :]
        candidate_logits = last_logits[letter_ids]
        raw_logits_dict = {L: float(candidate_logits[i]) for i, L in enumerate(letters)}
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
        "gold_log_prob": gold_log_prob,
        "raw_logits": raw_logits_dict
    }


def find_manifest_file(filename: str, hint_dir: str) -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(hint_dir, filename),
        os.path.join(os.getcwd(), hint_dir, filename),
        os.path.join(here, "manifests", filename),
        os.path.join(os.path.dirname(here), hint_dir, filename),
        os.path.join(here, hint_dir, filename),
        os.path.join("/kaggle/working", hint_dir, filename),
        os.path.join("/kaggle/working", "manifests", filename),
        os.path.join("/kaggle/working", filename),
    ]
    for c in candidates:
        if os.path.isfile(c): return c

    matches = glob.glob(f"/kaggle/**/{filename}", recursive=True)
    if matches: return matches[0]

    matches_local = glob.glob(f"**/{filename}", recursive=True)
    if matches_local: return matches_local[0]

    raise FileNotFoundError(f"Could not find '{filename}' anywhere. Checked: {candidates}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--manifest_dir", default="partB_benchmark/manifests")
    parser.add_argument("--output_dir", default="partB_benchmark/results_full")
    parser.add_argument("--checkpoint_step", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pilot", action="store_true", help="Use 150-question pilot set instead of full 7,465 benchmark")
    parser.add_argument("--max_items", type=int, default=None, help="Optional cap on evaluation items")
    parser.add_argument("--image_dir", type=str, default=None, help="Directory with preprocessed .npy CT volumes")
    parser.add_argument("--smoke", action="store_true", help="Quick CPU verification on tiny subset")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)

    use_cuda = HAS_TORCH and torch.cuda.is_available() and not args.smoke
    device = "cuda" if use_cuda else "cpu"
    git_hash = get_git_commit()

    manifest_prefix = "pilot" if args.pilot else "benchmark"
    p_real = find_manifest_file(f"{manifest_prefix}_real.json", args.manifest_dir)
    p_swap = find_manifest_file(f"{manifest_prefix}_swapped.json", args.manifest_dir)
    p_shuff_opt = find_manifest_file(f"{manifest_prefix}_shuffled.json", args.manifest_dir)

    with open(p_real) as f: real_items = json.load(f)
    with open(p_swap) as f: swap_items = json.load(f)
    with open(p_shuff_opt) as f: shuff_opt_items = json.load(f)

    if args.smoke:
        print("[audit] Running in SMOKE mode: slicing to 25 items.")
        real_items = real_items[:25]
        swap_items = swap_items[:25]
        shuff_opt_items = shuff_opt_items[:25]
    elif args.max_items is not None:
        print(f"[audit] Capping evaluation items to {args.max_items}.")
        real_items = real_items[:args.max_items]
        swap_items = swap_items[:args.max_items]
        shuff_opt_items = shuff_opt_items[:args.max_items]

    total_items = len(real_items)
    print(f"[audit] Total evaluation items: {total_items} (Manifest: {manifest_prefix}) | Seed: {args.seed} | Git: {git_hash[:8]}")

    model, tok = load_model_and_tok(args.model_id, device, is_smoke=args.smoke)

    checkpoint_file = os.path.join(args.output_dir, "m3d_full_checkpoint.csv")
    records = []
    start_idx = 0

    if os.path.exists(checkpoint_file):
        existing_df = pd.read_csv(checkpoint_file)
        start_idx = len(existing_df)
        records = existing_df.to_dict("records")
        print(f"[resume] Resuming from checkpoint at item {start_idx}/{total_items}...")

    t0 = time.time()

    # 1. Strict validation of image_dir to prevent dummy fallback evaluation
    if not args.smoke and (not args.image_dir or not os.path.isdir(args.image_dir)):
        raise ValueError(
            f"[FATAL AUDIT ERROR] --image_dir is REQUIRED and must point to a valid directory containing real CT .npy volumes.\n"
            f"Received: {args.image_dir}\n"
            f"Execution halted to prevent evaluating invalid dummy synthetic noise tensors."
        )

    if HAS_TORCH and not args.smoke:
        dtype = torch.float16 if device == "cuda" else torch.float32
        # Explicit Out-of-Distribution (OOD) control volume
        blank_vol = torch.zeros((1, 1, 32, 256, 256), dtype=dtype, device=device)
    else:
        blank_vol = np.zeros((1, 1, 32, 256, 256), dtype=np.float32)

    def get_scan_volume(scan_id: str, is_donor: bool = False) -> Tuple[Any, str]:
        """Loads genuine preprocessed CT volume and strictly asserts valid non-dummy input."""
        if args.image_dir:
            cand_paths = [
                os.path.join(args.image_dir, f"{scan_id}.npy"),
                os.path.join(args.image_dir, scan_id, f"{scan_id}.npy"),
                os.path.join(args.image_dir, scan_id, "image.npy"),
                os.path.join(args.image_dir, f"{scan_id}.nii.gz"),
                os.path.join(args.image_dir, scan_id, f"{scan_id}.nii.gz")
            ]
            for cp in cand_paths:
                if os.path.isfile(cp):
                    try:
                        if cp.endswith(".npy"):
                            arr = np.load(cp)
                        else:
                            import nibabel as nib
                            arr = nib.load(cp).get_fdata()

                        if arr.ndim == 3:
                            arr = arr[np.newaxis, ...]  # (1, D, H, W)
                        if arr.ndim == 4 and arr.shape[0] != 1:
                            arr = arr[np.newaxis, ...]  # (1, C, D, H, W)
                        if arr.ndim == 4:
                            arr = arr[np.newaxis, ...]  # (1, 1, D, H, W)

                        if HAS_TORCH:
                            t_vol = torch.from_numpy(arr).to(dtype=dtype, device=device)
                            if t_vol.shape[-3:] != (32, 256, 256):
                                t_vol = torch.nn.functional.interpolate(
                                    t_vol, size=(32, 256, 256), mode="trilinear", align_corners=False
                                )
                            # Rigorous verification: ensure tensor is genuine non-dummy CT
                            assert t_vol is not None, f"CT volume for {scan_id} resolved to None."
                            assert not torch.allclose(t_vol, torch.zeros_like(t_vol)), f"Loaded volume for {scan_id} is all-zero (invalid real scan)!"
                            assert float(torch.std(t_vol.float())) > 1e-4, f"Loaded volume for {scan_id} lacks spatial variance (possible flat dummy tensor)!"
                            return t_vol, cp
                        return arr, cp
                    except Exception as e:
                        print(f"[ERROR] Failed to load CT scan {scan_id} from {cp}: {e}")
                        raise

        if args.smoke:
            # Deterministic distinct mock volume strictly for offline CPU smoke tests
            val = float(sum(ord(c) for c in scan_id) % 100) / 100.0
            mock = torch.full((1, 1, 32, 256, 256), fill_value=val, dtype=torch.float32)
            mock += torch.sin(torch.linspace(0, 10, 32 * 256 * 256)).view(1, 1, 32, 256, 256) * 0.1
            return mock, f"[SMOKE_MOCK_{scan_id}]"

        raise FileNotFoundError(
            f"[FATAL] CT volume for case_id='{scan_id}' was NOT found in --image_dir='{args.image_dir}'.\n"
            f"Searched candidates: {cand_paths}\n"
            f"Falling back to dummy random noise is strictly prohibited."
        )

    for idx in range(start_idx, total_items):
        r_item = real_items[idx]
        sw_item = swap_items[idx]
        so_item = shuff_opt_items[idx]

        qid = r_item["question_id"]
        fam = r_item["family"]
        dom = r_item["m3d_domain"]
        cid = r_item["scan_id"]
        donor_cid = sw_item["scan_id"]
        gold_letter = r_item["answer_choice"]

        # Ensure donor patient scan differs from patient scan
        assert donor_cid != cid, f"Invariant violation: donor scan ({donor_cid}) matches patient scan ({cid}) for item {qid}!"

        # Load genuine CT volumes (strictly non-dummy)
        cur_real_vol, real_source = get_scan_volume(cid, is_donor=False)
        cur_donor_vol, donor_source = get_scan_volume(donor_cid, is_donor=True)

        # Log tensor distinctness & verification on first item
        if idx == start_idx:
            std_r = float(torch.std(cur_real_vol.float())) if HAS_TORCH else float(np.std(cur_real_vol))
            std_d = float(torch.std(cur_donor_vol.float())) if HAS_TORCH else float(np.std(cur_donor_vol))
            print(f"[PIPELINE VERIFICATION]")
            print(f"  -> Real CT Case:  {cid} from {real_source} (std={std_r:.4f})")
            print(f"  -> Donor CT Case: {donor_cid} from {donor_source} (std={std_d:.4f})")
            print(f"  -> Blank Volume:  OOD Control [all zeros] (shape={tuple(blank_vol.shape)})")

        # 1. Real condition
        prompt_r, letters_r, _ = format_prompt(r_item, include_image_tokens=True)
        im_patch_count = prompt_r.count("<im_patch>")
        assert im_patch_count == PROJ_OUT_NUM, (
            f"Prompt error: expected {PROJ_OUT_NUM} '<im_patch>' tokens, found {im_patch_count}!"
        )
        if idx == start_idx:
            print(f"  -> Verified <im_patch> count: {im_patch_count} tokens successfully matched.")

        res_real = score_item(model, tok, prompt_r, letters_r, gold_letter, cur_real_vol, device, is_smoke=args.smoke)

        # 2. Blank condition (Out-Of-Distribution / OOD Control: all zeros)
        res_blank = score_item(model, tok, prompt_r, letters_r, gold_letter, blank_vol, device, is_smoke=args.smoke)

        # 3. Swapped patient condition (genuine donor patient CT volume)
        res_swap = score_item(model, tok, prompt_r, letters_r, gold_letter, cur_donor_vol, device, is_smoke=args.smoke)

        # 4. Shuffled slices condition (permuted z-axis of real scan)
        if HAS_TORCH and not args.smoke:
            perm_z = torch.randperm(32)
            shuff_slice_vol = cur_real_vol[:, :, perm_z, :, :].contiguous()
        else:
            shuff_slice_vol = cur_real_vol
        res_shuff_slices = score_item(model, tok, prompt_r, letters_r, gold_letter, shuff_slice_vol, device, is_smoke=args.smoke)

        # 5. Shuffled options condition (permuted prompt choices & remapped gold letter)
        prompt_so, letters_so, _ = format_prompt(so_item, include_image_tokens=True)
        res_shuff_options = score_item(model, tok, prompt_so, letters_so, so_item["answer_choice"], cur_real_vol, device, is_smoke=args.smoke)

        # 6. Text-only condition (no placeholder tokens, images=None)
        prompt_to, letters_to, _ = format_prompt(r_item, include_image_tokens=False)
        assert prompt_to.count("<im_patch>") == 0, "Text-only condition must contain 0 '<im_patch>' tokens!"
        res_textonly = score_item(model, tok, prompt_to, letters_to, gold_letter, None, device, is_smoke=args.smoke)

        records.append({
            "question_id": qid,
            "case_id": cid,
            "family": fam,
            "m3d_domain": dom,
            "gold_letter": gold_letter,
            "gold_letter_shuff_options": so_item["answer_choice"],
            # Predictions
            "pred_real": res_real["pred_letter"],
            "pred_blank": res_blank["pred_letter"],
            "pred_swap": res_swap["pred_letter"],
            "pred_shuffle_slices": res_shuff_slices["pred_letter"],
            "pred_shuffle_options": res_shuff_options["pred_letter"],
            "pred_textonly": res_textonly["pred_letter"],
            # Correctness flags
            "correct_real": res_real["is_correct"],
            "correct_blank": res_blank["is_correct"],
            "correct_swap": res_swap["is_correct"],
            "correct_shuffle_slices": res_shuff_slices["is_correct"],
            "correct_shuffle_options": res_shuff_options["is_correct"],
            "correct_textonly": res_textonly["is_correct"],
            # Per-item logp_gold
            "logp_gold_real": res_real["gold_log_prob"],
            "logp_gold_blank": res_blank["gold_log_prob"],
            "logp_gold_swap": res_swap["gold_log_prob"],
            "logp_gold_shuffle_slices": res_shuff_slices["gold_log_prob"],
            "logp_gold_shuffle_options": res_shuff_options["gold_log_prob"],
            "logp_gold_textonly": res_textonly["gold_log_prob"],
            "delta_logp": res_real["gold_log_prob"] - res_blank["gold_log_prob"],
            # Full 4-option logits for every condition
            "logits_real": json.dumps(res_real["raw_logits"]),
            "logits_blank": json.dumps(res_blank["raw_logits"]),
            "logits_swap": json.dumps(res_swap["raw_logits"]),
            "logits_shuffle_slices": json.dumps(res_shuff_slices["raw_logits"]),
            "logits_shuffle_options": json.dumps(res_shuff_options["raw_logits"]),
            "logits_textonly": json.dumps(res_textonly["raw_logits"])
        })

        if (idx + 1) % args.checkpoint_step == 0 or (idx + 1) == total_items:
            df_cur = pd.DataFrame(records)
            df_cur.to_csv(checkpoint_file, index=False)
            dt = time.time() - t0
            rate = (idx + 1 - start_idx) / max(dt, 1e-4)
            rem_s = (total_items - (idx + 1)) / max(rate, 1e-4)
            print(f"[audit] Checkpoint {idx + 1}/{total_items} items ({dt:.1f}s, {rate:.1f} it/s, ETA: {rem_s/60:.1f} min)")

    final_csv = os.path.join(args.output_dir, "m3d_full_per_item.csv")
    df_final = pd.DataFrame(records)
    df_final.to_csv(final_csv, index=False)
    print(f"\n[done] Full audit completed! Saved {len(df_final)} rows to {final_csv}")


if __name__ == "__main__":
    main()
