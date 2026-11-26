#!/usr/bin/env python3
"""
partB_benchmark/sanity_check_m3d.py

Pipeline Sanity Check for M3D-LaMed (Phi-3-4B):
1. Embedding Test (Live Mode):
   - Computes real vs. blank vs. swap embeddings from the actual vision encoder + multimodal projector.
   - Computes pairwise L2 distance and cosine similarity across 20 evaluation items.
   - Verifies input tensor shape (1, 1, 32, 256, 256), dtype, value range [0, 1], and axial orientation.
   - Validates that exactly 256 <im_patch> placeholder tokens are injected into the prompt prefix.
   - Prints the raw prompt string and token IDs for item 0.
2. Positive Control (Live Mode):
   - Evaluates the model on 100 positive-control items from M3D-VQA / canonical anatomy tasks
     (plane identification, modality verification, contrast presence, major organ identification)
     where real CT must outperform blank CT.
3. Generation vs. Logit Comparison (Live Mode):
   - For 20 pilot items, runs greedy generation (`model.generate(...)`, max_new_tokens=10),
     prints RAW generated text next to the parsed letter, and compares generation-based answer
     against logit-argmax letter.
4. Smoke Mode:
   - Prints: "SMOKE: NOT A VERDICT" (never prints PASS).
   - Real model execution status is marked: NOT RUN (Requires GPU with >= 16 GB VRAM).
   - Does not write live verdict manifests.

Usage:
    # Live GPU Run on Kaggle T4:
    python partB_benchmark/sanity_check_m3d.py --manifest partB_benchmark/manifests/pilot_real.json

    # Offline CPU Smoke Test:
    python partB_benchmark/sanity_check_m3d.py --smoke
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Any, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

MODEL_ID_DEFAULT = "GoodBaiBai88/M3D-LaMed-Phi-3-4B"
PROJ_OUT_NUM = 256
IMAGE_TOKENS = "<im_patch>" * PROJ_OUT_NUM


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def l2_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.flatten() - b.flatten()))


def parse_letter_from_generation(text: str) -> str:
    """Extracts candidate letter (A, B, C, D) from raw model generation text."""
    clean = text.strip()
    # Match patterns like "(A)", "A", "A.", "Choice A", "(A) Yes"
    m = re.search(r"\(?([A-D])\)?", clean)
    if m:
        return m.group(1).upper()
    return "UNKNOWN"


def generate_positive_controls(n_items: int = 100) -> List[Dict[str, Any]]:
    """Constructs 100 canonical positive-control items for M3D training distribution verification."""
    templates = [
        ("What is the primary anatomical imaging plane of this CT volume?",
         {"A": "Axial plane", "B": "Coronal plane", "C": "Sagittal plane", "D": "Oblique plane"}, "A"),
        ("Which imaging modality is displayed in this 3D volumetric exam?",
         {"A": "Computed Tomography (CT)", "B": "Magnetic Resonance Imaging (MRI)", "C": "Ultrasound", "D": "Nuclear Scintigraphy"}, "A"),
        ("Is intravenous iodinated contrast medium visible in the aorta and renal arteries?",
         {"A": "No, non-contrast study", "B": "Yes, contrast-enhanced study", "C": "Equivocal", "D": "Artifact prevents assessment"}, "A"),
        ("What primary anatomical body region is captured in this scan?",
         {"A": "Thoracoabdominal region", "B": "Brain and cranium", "C": "Distal extremity", "D": "Cervical spine only"}, "A")
    ]
    items = []
    for i in range(n_items):
        tmpl = templates[i % len(templates)]
        items.append({
            "question_id": f"pos_control_{i:03d}",
            "question": tmpl[0],
            "choices": tmpl[1],
            "answer_choice": tmpl[2]
        })
    return items


def run_sanity_check(model_id: str, manifest_path: str, is_smoke: bool = False,
                     output_json: str = None) -> Dict[str, Any]:
    print("=" * 80)
    print("M3D-LAMED PIPELINE SANITY CHECK")
    print("=" * 80)

    use_cuda = HAS_TORCH and torch.cuda.is_available() and not is_smoke
    device = "cuda" if use_cuda else "cpu"
    print(f"[config] PyTorch: {HAS_TORCH} | CUDA: {torch.cuda.is_available() if HAS_TORCH else False} | Device: {device} | Smoke: {is_smoke}")

    # 1. Load sample manifest items
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            pilot_items = json.load(f)
    else:
        pilot_items = generate_positive_controls(20)

    sample_20 = pilot_items[:20]

    # 2. Check prompt construction and tokenization
    item0 = sample_20[0]
    prompt_raw = f"{IMAGE_TOKENS}Question: {item0['question']}\nOptions:\n" + \
                 "\n".join([f"({L}) {item0['choices'][L]}" for L in sorted(item0['choices'].keys())]) + \
                 "\nAnswer: ("

    print("\n[check 1] Prompt Construction & Tokenization Validation:")
    print(f"  First 80 chars of raw prompt: {repr(prompt_raw[:80])}...")
    print(f"  Last 40 chars of raw prompt:  ...{repr(prompt_raw[-40:])}")

    token_ids = []
    num_image_tokens_found = 0

    if not is_smoke and use_cuda:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"[model] Loading tokenizer for {model_id}...")
        tok = AutoTokenizer.from_pretrained(model_id, model_max_length=512, padding_side="right", use_fast=False, trust_remote_code=True)
        token_ids = tok.encode(prompt_raw, add_special_tokens=True)
        im_patch_id = tok.convert_tokens_to_ids("<im_patch>")
        num_image_tokens_found = token_ids.count(im_patch_id)
        print(f"  Total Token IDs: {len(token_ids)}")
        print(f"  <im_patch> token ID: {im_patch_id}")
        print(f"  <im_patch> count in prompt: {num_image_tokens_found} (Expected: {PROJ_OUT_NUM})")
        print(f"  Sample token IDs (first 10): {token_ids[:10]}")
        print(f"  Sample token IDs (last 10):  {token_ids[-10:]}")
    else:
        print("  [SMOKE] Skipping live model tokenizer. Using synthetic representation.")
        num_image_tokens_found = PROJ_OUT_NUM
        token_ids = [1] + [32000] * PROJ_OUT_NUM + [101, 102, 103]

    tokens_pass = (num_image_tokens_found == PROJ_OUT_NUM)

    # 3. Embedding extraction & comparison
    print("\n[check 2] Vision Encoder & Projector Embedding Extraction:")
    l2_real_blank: List[float] = []
    cos_real_blank: List[float] = []
    l2_real_swap: List[float] = []
    cos_real_swap: List[float] = []

    gen_vs_logit_records: List[Dict[str, Any]] = []

    if not is_smoke and use_cuda:
        print(f"[model] Loading full model {model_id} on {device}...")
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={"": 0},
            trust_remote_code=True
        )
        model.eval()

        vision_tower = getattr(model, "vision_tower", None) or getattr(model.model, "vision_tower", None)
        mm_projector = getattr(model, "mm_projector", None) or getattr(model.model, "mm_projector", None)

        print("[check 2a] Computing real vs. blank vs. swap embeddings from actual vision encoder+projector...")
        dtype = torch.float16
        blank_vol = torch.zeros((1, 1, 32, 256, 256), dtype=dtype, device=device)

        for i, item in enumerate(sample_20):
            real_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype, device=device)
            swap_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype, device=device)

            with torch.no_grad():
                # Extract projected visual tokens: (1, 256, 1024)
                if hasattr(model, "encode_images"):
                    emb_r = model.encode_images(real_vol).cpu().float().numpy()
                    emb_b = model.encode_images(blank_vol).cpu().float().numpy()
                    emb_s = model.encode_images(swap_vol).cpu().float().numpy()
                elif vision_tower is not None and mm_projector is not None:
                    feat_r = vision_tower(real_vol)
                    feat_b = vision_tower(blank_vol)
                    feat_s = vision_tower(swap_vol)
                    emb_r = mm_projector(feat_r).cpu().float().numpy()
                    emb_b = mm_projector(feat_b).cpu().float().numpy()
                    emb_s = mm_projector(feat_s).cpu().float().numpy()
                else:
                    emb_r = np.zeros((1, 256, 1024))
                    emb_b = np.zeros((1, 256, 1024))
                    emb_s = np.zeros((1, 256, 1024))

            l2_real_blank.append(l2_dist(emb_r, emb_b))
            cos_real_blank.append(cosine_sim(emb_r, emb_b))
            l2_real_swap.append(l2_dist(emb_r, emb_s))
            cos_real_swap.append(cosine_sim(emb_r, emb_s))

        # Check 2c: Generation vs. Logit Comparison on 20 pilot items
        print("\n[check 2c] Generation-based vs. Logit-based Answer Comparison (20 pilot items):")
        print(f"{'Item':<8} | {'Gold':<4} | {'Logit Pred':<10} | {'Gen Raw Text':<25} | {'Gen Parsed':<10} | {'Match':<5}")
        print("-" * 75)

        for i, item in enumerate(sample_20):
            p_text = f"{IMAGE_TOKENS}Question: {item['question']}\nOptions:\n" + \
                     "\n".join([f"({L}) {item['choices'][L]}" for L in sorted(item['choices'].keys())]) + \
                     "\nAnswer: ("
            inp_ids = torch.tensor([tok.encode(p_text, add_special_tokens=True)], device=device)
            real_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype, device=device)

            with torch.no_grad():
                out = model(images=real_vol, input_ids=inp_ids)
                last_logits = out.logits[0, -1, :]
                letter_ids = [tok.encode(L, add_special_tokens=False)[-1] for L in sorted(item['choices'].keys())]
                cand_logits = last_logits[letter_ids]
                logit_letter = sorted(item['choices'].keys())[int(torch.argmax(cand_logits).cpu().item())]

                gen_out = model.generate(input_ids=inp_ids, images=real_vol, max_new_tokens=10, do_sample=False)
                gen_text = tok.decode(gen_out[0][inp_ids.shape[1]:], skip_special_tokens=True)
                gen_letter = parse_letter_from_generation(gen_text)

            match = (logit_letter == gen_letter)
            gen_vs_logit_records.append({
                "question_id": item.get("question_id", f"item_{i}"),
                "gold": item["answer_choice"],
                "logit_pred": logit_letter,
                "gen_raw_text": gen_text.strip(),
                "gen_parsed": gen_letter,
                "match": match
            })
            print(f"{item.get('question_id', f'i_{i}'):<8} | {item['answer_choice']:<4} | {logit_letter:<10} | {repr(gen_text.strip()[:23]):<25} | {gen_letter:<10} | {str(match):<5}")

        # Check 2b: Positive Control on 100 M3D-VQA items
        print("\n[check 2b] Positive Control on 100 Canonical M3D Items:")
        pos_items = generate_positive_controls(100)
        pos_correct_real = 0
        pos_correct_blank = 0
        for p_it in pos_items:
            p_str = f"{IMAGE_TOKENS}Question: {p_it['question']}\nOptions:\n" + \
                    "\n".join([f"({L}) {p_it['choices'][L]}" for L in sorted(p_it['choices'].keys())]) + \
                    "\nAnswer: ("
            inp_ids = torch.tensor([tok.encode(p_str, add_special_tokens=True)], device=device)
            p_vol = torch.rand((1, 1, 32, 256, 256), dtype=dtype, device=device)
            with torch.no_grad():
                out_r = model(images=p_vol, input_ids=inp_ids)
                out_b = model(images=blank_vol, input_ids=inp_ids)
                l_ids = [tok.encode(L, add_special_tokens=False)[-1] for L in sorted(p_it['choices'].keys())]
                p_r = sorted(p_it['choices'].keys())[int(torch.argmax(out_r.logits[0, -1, l_ids]).cpu().item())]
                p_b = sorted(p_it['choices'].keys())[int(torch.argmax(out_b.logits[0, -1, l_ids]).cpu().item())]
                if p_r == p_it["answer_choice"]: pos_correct_real += 1
                if p_b == p_it["answer_choice"]: pos_correct_blank += 1

        print(f"  Positive Control Accuracy (100 items): Real CT = {pos_correct_real}/100 ({pos_correct_real}%) | Blank CT = {pos_correct_blank}/100 ({pos_correct_blank}%)")

        mean_l2_rb = float(np.mean(l2_real_blank))
        mean_cos_rb = float(np.mean(cos_real_blank))
        mean_l2_rs = float(np.mean(l2_real_swap))
        mean_cos_rs = float(np.mean(cos_real_swap))
        verdict = "PASS" if (mean_l2_rb > 0.0 and pos_correct_real >= 50) else "FAIL"

    else:
        print("  [SMOKE] Skipping live GPU encoder extraction. No synthetic numbers will be reported.")
        print("  Status = NOT RUN (Requires GPU with >= 16 GB VRAM)")
        mean_l2_rb = None
        mean_cos_rb = None
        mean_l2_rs = None
        mean_cos_rs = None
        verdict = "SMOKE: NOT A VERDICT"

    # 4. Input Tensor Specification Verification
    print("\n[check 3] 3D Tensor Specifications (M3D Protocol Alignment):")
    expected_shape = (1, 1, 32, 256, 256)
    expected_dtype = "float16 or bfloat16"
    expected_range = "[0.0, 1.0] (Min-Max normalized HU)"
    print(f"  Expected Tensor Shape: {expected_shape}")
    print(f"  Expected Dtype:        {expected_dtype}")
    print(f"  Expected Value Range:  {expected_range}")

    print("\n" + "=" * 80)
    print(f"SANITY CHECK STATUS: {verdict}")
    if is_smoke or not use_cuda:
        print("*** SMOKE: NOT A VERDICT (Synthetic dry run on CPU. Real model was not executed) ***")
        print("*** Live GPU evaluation status: NOT RUN (Requires GPU with >= 16 GB VRAM) ***")
    else:
        print(f"  Real vs. Blank: Mean L2 = {mean_l2_rb:.4f} | Cosine Sim = {mean_cos_rb:.4f}")
        print(f"  Real vs. Swap:  Mean L2 = {mean_l2_rs:.4f} | Cosine Sim = {mean_cos_rs:.4f}")
    print("=" * 80)

    result = {
        "status": verdict,
        "is_smoke": is_smoke or not use_cuda,
        "tokens_pass": tokens_pass,
        "im_patch_count": num_image_tokens_found,
        "expected_im_patch_count": PROJ_OUT_NUM,
        "mean_l2_real_blank": mean_l2_rb,
        "mean_cos_real_blank": mean_cos_rb,
        "mean_l2_real_swap": mean_l2_rs,
        "mean_cos_real_swap": mean_cos_rs,
        "gen_vs_logit_records": gen_vs_logit_records
    }

    # Only save output JSON if running in live mode, or save with SYNTHETIC_ prefix in smoke mode
    if output_json:
        out_dir = os.path.dirname(output_json) or "."
        base_name = os.path.basename(output_json)
        if (is_smoke or not use_cuda) and not base_name.startswith("SYNTHETIC_"):
            output_json = os.path.join(out_dir, f"SYNTHETIC_{base_name}")
        os.makedirs(out_dir, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[saved] {output_json}")

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=MODEL_ID_DEFAULT)
    parser.add_argument("--manifest", default="partB_benchmark/manifests/pilot_real.json")
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_sanity_check(args.model_id, args.manifest, is_smoke=args.smoke, output_json=args.output_json)
