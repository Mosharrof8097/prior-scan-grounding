#!/usr/bin/env python3
"""
====================================================================================================
3D-RAD BASELINE RETRAIN — v7
====================================================================================================
Same model, same split, same hyperparameters as the v6 run whose checkpoint was lost, so the
resulting numbers remain comparable. What changed is everything around the training loop.

WHY THIS VERSION EXISTS
  The v6 run finished 3 epochs and reached val_loss 0.2853, but its artefacts were written only to
  /kaggle/working of a session that was never committed, so the LoRA adapters and the metrics file
  are gone and none of it can be reproduced or evaluated. Retraining is only worth doing if that
  cannot happen again.

WHAT IS FIXED
  1. SURVIVES THE SESSION. Every epoch writes projector + LoRA adapters + optimizer state +
     metrics to disk immediately, under last/ and best/ and epoch_N/. Nothing waits for the end.
  2. RESUMABLE. --resume picks up from the last completed epoch, so a 12-hour session limit or a
     crash costs one epoch, not the whole run.
  3. PER-TASK VALIDATION. Metrics are reported for Task 1..6 separately, never pooled into
     "single-scan" and "longitudinal". Tasks 5 and 6 are the SAME 16,031 clinical items with
     different wording -- Task 6 states the past presence/absence bit in the question text and
     Task 5 does not -- so pooling them averages away the one effect being studied. In the val
     split those 476 "longitudinal" items are 238 questions counted twice.
  4. MCQ ACCURACY DURING TRAINING. Closed questions are scored by length-normalised
     log-likelihood over their real options, which is the paper-comparable number. Free-text
     exact match is kept for the open tasks only.
  5. RECENCY RATIO, SPLIT BY TASK. T1/T0 is reported for Task 5 and Task 6 separately and only
     over genuinely paired items, so Task 6's text-fed past no longer contaminates Task 5.
  6. SPLIT VERIFIED, NOT ASSUMED. The patient-level split is regenerated and checked against the
     counts the original run produced. A mismatch aborts instead of quietly training on a
     different split.
  7. NO ZIP. Outputs are individual files, downloadable one click each.

AFTER TRAINING it runs the full audit (per-task accuracy on val and test, plus the blank-image
ablation) if kaggle_inference_audit.py sits next to it.

----------------------------------------------------------------------------------------------------
HOW TO RUN ON KAGGLE
----------------------------------------------------------------------------------------------------
  !pip install -q -U peft

  # 1. preflight: CPU only, seconds, confirms the split reproduces before spending any GPU
  !python kaggle_retrain_v7.py --preflight

  # 2. smoke test: 2 tiny epochs on a few hundred items, ~5 min, proves the loop and all saving works
  !python kaggle_retrain_v7.py --smoke

  # 3. the real run
  !python kaggle_retrain_v7.py

  # if the session died partway:
  !python kaggle_retrain_v7.py --resume

>>> COMMIT THE NOTEBOOK (Save Version -> Save & Run All). An interactive session's /kaggle/working
>>> is deleted when the session ends. That is exactly how the v6 artefacts were lost.
====================================================================================================
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import sys
import glob
import json
import time
import math
import random
import string
import argparse
import importlib.util
import subprocess
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
FILENAME_PATTERN = re.compile(r"^(?P<split>[a-zA-Z]+)_(?P<patient>\d+)_(?P<letter>[a-zA-Z]+)_(?P<recon>\d+)\.npy$")

# The split the lost v6 checkpoint was trained on. Regeneration must reproduce this exactly,
# otherwise the new numbers are not comparable to anything already written up.
REFERENCE = {
    "train": {"qa": 27197, "volumes": 2181, "patients": 1043},
    "val":   {"qa": 3254,  "volumes": 255,  "patients": 130},
    "test":  {"qa": 3459,  "volumes": 260,  "patients": 131},
}

TASK_ORDER = ["Task1_Image_Observation", "Task2_Anomaly_Detection", "Task3_Medical_Computation",
              "Task4_Existence_Detection", "Task5_Static_Temporal_Diagnosis",
              "Task6_Longitudinal_Temporal_Diagnosis"]


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        m = " ".join(str(x) for x in a)
        print(m, flush=True)
        self.f.write(m + "\n")


LOG = print


# ==================================================================================================
# environment guards (carried over from v6: these were real failures, not hypotheticals)
# ==================================================================================================
def require_peft():
    if importlib.util.find_spec("peft") is None:
        raise RuntimeError(
            "\n" + "!" * 88 + "\n"
            "  peft is NOT installed. Without it LoRA is never applied and you would train only\n"
            "  the projector -- a different experiment. Run this first:\n"
            "      !pip install -q -U peft\n" + "!" * 88)
    import peft
    LOG(f"[env] peft {getattr(peft, '__version__', '?')}")


def patch_peft_torchao():
    """peft's LoRA dispatcher calls is_torchao_available() per module; Kaggle's old torchao raises
    instead of returning False. Both the origin and the local copy dispatch_torchao() reads must be
    neutralised, which is why there are three targets here."""
    ok = False
    for mod, attr, val in (("peft.import_utils", "is_torchao_available", lambda: False),
                           ("peft.tuners.lora.torchao", "is_torchao_available", lambda: False),
                           ("peft.tuners.lora.torchao", "dispatch_torchao",
                            lambda target, adapter_name, config=None, **kw: None)):
        try:
            setattr(importlib.import_module(mod), attr, val)
            ok = True
        except Exception:
            pass
    LOG(f"[env] torchao dispatcher patched={ok}")


def gpu_headroom_or_raise(min_free_gib=5.0):
    if not torch.cuda.is_available():
        return
    free, total = torch.cuda.mem_get_info()
    free_g, tot_g = free / 2**30, total / 2**30
    LOG(f"[env] GPU memory: {free_g:.2f} GiB free / {tot_g:.2f} GiB total")
    if free_g < min_free_gib:
        raise RuntimeError(
            "\n" + "!" * 88 + "\n"
            f"  Only {free_g:.2f} GiB of GPU memory is free. A previous cell in this session is\n"
            "  still holding it -- empty_cache() cannot release memory that live references own.\n"
            "  Fix: Run -> Restart session, then run this script as the first GPU cell.\n"
            + "!" * 88)


# ==================================================================================================
# split
# ==================================================================================================
def autodetect_data_dir(explicit):
    if explicit:
        return explicit
    for c in sorted(glob.glob("/kaggle/input/*")):
        if glob.glob(os.path.join(c, "**", "*.npy"), recursive=True)[:1]:
            return c
    raise FileNotFoundError("No /kaggle/input/* directory contains .npy volumes. Pass --data_dir.")


def _assemble(csv_paths, image_map):
    frames = []
    for f in csv_paths:
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "VolumeName" not in d.columns or "Question" not in d.columns:
            continue
        d["TaskCategory"] = os.path.basename(os.path.dirname(f))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))
    df["PatientID"] = df["NpyName"].apply(lambda x: "_".join(str(x).split("_")[:2]))
    df["ImagePath"] = df["NpyName"].map(image_map)
    return df[df["ImagePath"].notna()].copy()


def _assign(df):
    # Must match v6 character for character: np.array(list(unique())) gives a plain numpy string
    # array. Handing a pandas extension array to rng.shuffle is not guaranteed to permute the same
    # way, which would silently produce a different split.
    pats = np.array(list(df["PatientID"].unique()))
    rng = np.random.RandomState(SEED)
    rng.shuffle(pats)
    n = len(pats)
    n_tr, n_va = int(n * 0.80), int(n * 0.10)
    a = {p: "train" for p in pats[:n_tr]}
    a.update({p: "val" for p in pats[n_tr:n_tr + n_va]})
    a.update({p: "test" for p in pats[n_tr + n_va:]})
    out = df.copy()
    out["Split"] = out["PatientID"].map(a)
    return out


def _counts(df):
    return {s: {"qa": int((df.Split == s).sum()),
                "volumes": int(df[df.Split == s]["NpyName"].nunique()),
                "patients": int(df[df.Split == s]["PatientID"].nunique())}
            for s in ("train", "val", "test")}


def build_verified_split(data_dir):
    image_map = {os.path.basename(p): p
                 for p in glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)}
    LOG(f"[split] {len(image_map):,} volumes under {data_dir}")
    csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    LOG(f"[split] {len(csvs):,} csv files")
    for name, paths in (("filesystem order", list(csvs)),
                        ("sorted by path", sorted(csvs)),
                        ("reverse sorted", sorted(csvs, reverse=True))):
        df = _assign(_assemble(paths, image_map))
        got = _counts(df)
        hit = all(got[s] == REFERENCE[s] for s in REFERENCE)
        LOG(f"[split] '{name}': " + " ".join(
            f"{s}={got[s]['qa']:,}" for s in ("train", "val", "test"))
            + ("   <== MATCHES the v6 split" if hit else ""))
        if hit:
            for a in ("train", "val", "test"):
                for b in ("train", "val", "test"):
                    if a < b:
                        assert not (set(df[df.Split == a].PatientID)
                                    & set(df[df.Split == b].PatientID)), f"leak {a}/{b}"
            LOG("[split] zero patient leakage verified")
            return df
    raise SystemExit(
        "\n" + "!" * 92 + "\n"
        "  ABORTING: could not reproduce the split the original run used\n"
        f"  (expected train/val/test = {REFERENCE['train']['qa']}/{REFERENCE['val']['qa']}/"
        f"{REFERENCE['test']['qa']} QA pairs).\n"
        "  Training on a different split would make the new numbers incomparable to the write-up\n"
        "  and could put val/test patients into training. Nothing is trained.\n"
        "  Check that --data_dir is the same dataset version used before.\n" + "!" * 92)


# ==================================================================================================
# data
# ==================================================================================================
def parse_filename(f):
    m = FILENAME_PATTERN.match(f)
    return m.groupdict() if m else None


def build_timeline(data_dir):
    tl = defaultdict(list)
    for p in glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True):
        i = parse_filename(os.path.basename(p))
        if i:
            tl[f"{i['split']}_{i['patient']}"].append((i["letter"], p))
    for k in tl:
        tl[k] = sorted(set(tl[k]), key=lambda x: x[0])
    return tl


def resample_depth(v, t):
    return v if v.shape[0] == t else v[np.linspace(0, v.shape[0] - 1, t).round().astype(int)]


class VQADataset(Dataset):
    def __init__(self, df, tokenizer, timeline, max_seq_len=128, is_train=True):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.timeline = timeline
        self.max_seq_len = max_seq_len
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def _load(self, path):
        a = np.load(path).astype(np.float32)
        return a[0] if a.ndim == 4 else a

    def _pair(self, npy, path):
        t1 = self._load(path)
        i = parse_filename(npy)
        if i is None:
            return np.concatenate([resample_depth(t1, 16)] * 2, 0), "fallback_unparsed_filename"
        tl = self.timeline.get(f"{i['split']}_{i['patient']}", [])
        earlier = [l for l, _ in tl if l < i["letter"]]
        if not earlier:
            return np.concatenate([resample_depth(t1, 16)] * 2, 0), "fallback_no_earlier_scan"
        t0 = self._load(dict(tl)[earlier[-1]])
        return np.concatenate([resample_depth(t0, 16), resample_depth(t1, 16)], 0), "paired_ok"

    def _augment(self, vol):
        if not self.is_train:
            return vol
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=2).copy()
        if np.random.rand() > 0.5:
            vol = np.flip(vol, axis=1).copy()
        if np.random.rand() > 0.5:
            vol = np.clip(vol * np.random.uniform(0.95, 1.05), 0.0, 1.0)
        return vol

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        task = str(r["TaskCategory"])
        is_long = any(k in task.lower() for k in ["longitudinal", "temporal"])
        if is_long:
            vol, pairing = self._pair(str(r["NpyName"]), r["ImagePath"])
        else:
            vol, pairing = self._load(r["ImagePath"]), "not_applicable_single_scan"
        vol = self._augment(vol)
        image = torch.from_numpy(np.expand_dims(vol, 0)).float()

        question, answer = str(r["Question"]), str(r["Answer"])
        prompt_ids = self.tok.encode(f"Clinical Question: {question}\nDiagnostic Answer: ",
                                     add_special_tokens=True)
        answer_ids = self.tok.encode(answer + self.tok.eos_token, add_special_tokens=False)
        reserve = min(len(answer_ids), 32)
        prompt_ids = prompt_ids[:max(self.max_seq_len - reserve, 8)]
        ids = (prompt_ids + answer_ids)[:self.max_seq_len]
        n = len(ids)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0

        input_ids = torch.full((self.max_seq_len,), pad, dtype=torch.long)
        input_ids[:n] = torch.tensor(ids)
        attn = torch.zeros(self.max_seq_len, dtype=torch.long)
        attn[:n] = 1
        labels = torch.full((self.max_seq_len,), -100, dtype=torch.long)
        start = len(prompt_ids)
        if start < n:
            labels[start:n] = input_ids[start:n]
        else:
            labels[max(0, n - 1)] = input_ids[max(0, n - 1)]

        # real options only: Task 4 is binary (C and D are "-"), Tasks 5/6 have four
        letters, choices = [], []
        for L in "ABCD":
            c = str(r.get(f"Choice {L}", "-")).strip()
            if c not in ("-", "", "nan", "NaN", "None"):
                letters.append(L)
                choices.append(c)
        gold = str(r.get("AnswerChoice", "")).strip()
        is_mcq = (str(r.get("QuestionType", "")).strip().lower().startswith("close")
                  and len(choices) >= 2 and gold in letters)

        return {"image": image, "input_ids": input_ids, "attention_mask": attn, "labels": labels,
                "task": task, "pairing": pairing, "is_longitudinal": is_long,
                "is_mcq": is_mcq, "choices": choices, "choice_letters": letters,
                "gold_letter": gold, "question": question, "answer": answer,
                "prompt_ids": prompt_ids}


def collate(batch):
    out = {k: torch.stack([b[k] for b in batch])
           for k in ("image", "input_ids", "attention_mask", "labels")}
    for k in ("task", "pairing", "is_longitudinal", "is_mcq", "answer"):
        out[k] = [b[k] for b in batch]
    return out


# ==================================================================================================
# model — identical to v6 so the weights and numbers stay comparable
# ==================================================================================================
class SliceDepthAggregator(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_visual_tokens=16):
        super().__init__()
        self.query_tokens = nn.Parameter(torch.randn(1, num_visual_tokens, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, sf):
        q = self.query_tokens.expand(sf.size(0), -1, -1)
        a, w = self.mha(query=q, key=sf, value=sf)
        return self.norm(q + a), w


def build_slice_encoder(embed_dim=512, pretrained=True):
    import torchvision.models as tv
    weights = tv.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    resnet = tv.resnet18(weights=weights)
    old = resnet.conv1
    new = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                    stride=old.stride, padding=old.padding, bias=False)
    with torch.no_grad():
        new.weight[:] = old.weight.mean(dim=1, keepdim=True)
    resnet.conv1 = new
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    for p in backbone.parameters():
        p.requires_grad = False
    backbone.eval()
    head = nn.Sequential(nn.Flatten(), nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim))
    enc = nn.Sequential(backbone, head)
    enc._is_frozen_backbone = True
    return enc


class Medical3DVLM(nn.Module):
    def __init__(self, llm, tokenizer, vision_embed_dim=512, num_visual_tokens=16, pretrained=True):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.num_visual_tokens = num_visual_tokens
        self.slice_encoder = build_slice_encoder(vision_embed_dim, pretrained)
        self.depth_aggregator = SliceDepthAggregator(vision_embed_dim, 8, num_visual_tokens)
        h = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h))
        for m in self.projector.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)

    def encode(self, volume):
        B, C, S, H, W = volume.shape
        slices = volume.permute(0, 2, 1, 3, 4).contiguous().view(B * S, C, H, W)
        self.slice_encoder[0].eval()
        with torch.no_grad():
            feats = self.slice_encoder[0](slices)
        sf = self.slice_encoder[1](feats).view(B, S, -1)
        tokens, attn = self.depth_aggregator(sf)
        return self.projector(tokens), attn

    def forward(self, images, input_ids, attention_mask, labels=None):
        vis, attn = self.encode(images)
        B, nv, _ = vis.shape
        dt = next(self.llm.parameters()).dtype
        txt = self.llm.get_input_embeddings()(input_ids)
        emb = torch.cat([vis.to(dt), txt.to(dt)], dim=1)
        vmask = torch.ones((B, nv), dtype=attention_mask.dtype, device=attention_mask.device)
        mask = torch.cat([vmask, attention_mask], dim=1)
        clab = None
        if labels is not None:
            vlab = torch.full((B, nv), -100, dtype=labels.dtype, device=labels.device)
            clab = torch.cat([vlab, labels], dim=1)
        out = self.llm(inputs_embeds=emb, attention_mask=mask, labels=clab, return_dict=True)
        return out, attn, clab, nv


def build_model(model_id, lora_r, lora_alpha, device, pretrained=True):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    require_peft()
    gpu_headroom_or_raise()

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    LOG(f"[model] {model_id} in {dtype}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True)
    for p in base.parameters():
        p.requires_grad = False
    base.config.use_cache = False
    try:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except TypeError:
        base.gradient_checkpointing_enable()
    LOG("[model] gradient checkpointing on, use_cache off")

    patch_peft_torchao()
    from peft import LoraConfig, get_peft_model
    base = get_peft_model(base, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    base.print_trainable_parameters()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    base.config.use_cache = False

    model = Medical3DVLM(base, tok, pretrained=pretrained).to(device)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    LOG(f"[model] params {tot:,} | trainable {tr:,} ({tr/tot*100:.2f}%)")
    if tr / tot < 0.005:
        LOG("[model][WARN] trainable fraction suspiciously low -- is LoRA really attached?")
    return model, tok


# ==================================================================================================
# metrics
# ==================================================================================================
def norm_text(s):
    s = str(s).lower().strip()
    s = "".join(c for c in s if c not in set(string.punctuation))
    return " ".join(s.split())


def recency_ratio(attn):
    """attn: (B, queries, keys=32). Keys 0-15 are T0, 16-31 are T1 -- same convention as v6."""
    m = attn.float().mean(dim=1)
    return (m[:, 16:].mean() / (m[:, :16].mean() + 1e-6)).item()


def per_item_loss(logits, labels):
    sl, sb = logits[..., :-1, :].contiguous(), labels[..., 1:].contiguous()
    f = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)(
        sl.view(-1, sl.size(-1)).float(), sb.view(-1))
    pt = f.view(sb.size())
    m = (sb != -100).float()
    return (pt * m).sum(1) / m.sum(1).clamp(min=1.0)


def tf_token_acc(logits, labels):
    sl, sb = logits[..., :-1, :], labels[..., 1:]
    pred = sl.argmax(-1)
    m = sb != -100
    return ((pred == sb) & m).sum(1).float() / m.sum(1).clamp(min=1).float()


@torch.no_grad()
def mcq_score_one(model, tok, item, device, max_seq_len, blank=False):
    """Length-normalised log-likelihood over an item's REAL options. Returns 1/0 or None."""
    img = item["image"].unsqueeze(0).to(device)
    if blank:
        img = torch.zeros_like(img)
    vis, _ = model.encode(img)
    dt = next(model.llm.parameters()).dtype
    vis = vis.to(dt)
    emb_layer = model.llm.get_input_embeddings()

    cids = [tok.encode(str(c) + tok.eos_token, add_special_tokens=False) for c in item["choices"]]
    K = len(cids)
    reserve = min(max(len(c) for c in cids), 32)
    # one shared prompt for all options, and no option text is ever clipped -- a clipped option
    # would corrupt its own likelihood and bias the argmax
    p = item["prompt_ids"][:max(max_seq_len - reserve, 8)]
    seqs = [p + c for c in cids]
    labs = [[-100] * len(p) + c for c in cids]
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id or 0
    ids = torch.full((K, L), pad, dtype=torch.long)
    lab = torch.full((K, L), -100, dtype=torch.long)
    att = torch.zeros((K, L), dtype=torch.long)
    for j, (s, l) in enumerate(zip(seqs, labs)):
        ids[j, :len(s)] = torch.tensor(s)
        lab[j, :len(l)] = torch.tensor(l)
        att[j, :len(s)] = 1
    ids, lab, att = ids.to(device), lab.to(device), att.to(device)

    txt = emb_layer(ids).to(dt)
    vK = vis.expand(K, -1, -1)
    emb = torch.cat([vK, txt], 1)
    mask = torch.cat([torch.ones((K, vK.size(1)), dtype=att.dtype, device=device), att], 1)
    clab = torch.cat([torch.full((K, vK.size(1)), -100, dtype=lab.dtype, device=device), lab], 1)

    logits = model.llm(inputs_embeds=emb, attention_mask=mask, return_dict=True).logits
    sb = clab[:, 1:]
    # cast one option at a time: all K at once would allocate ~350 MB twice on a T4
    rows = []
    for j in range(K):
        lj = logits[j:j + 1, :-1].float()
        rows.append(torch.log_softmax(lj, -1)
                    .gather(-1, sb[j:j + 1].clamp(min=0).unsqueeze(-1)).squeeze(-1))
        del lj
    lp = torch.cat(rows, 0)
    m = (sb != -100).float()
    mean = (lp * m).sum(1) / m.sum(1).clamp(min=1)
    letters = item["choice_letters"]
    return int(letters[int(mean.argmax())] == item["gold_letter"])


# ==================================================================================================
# train / validate
# ==================================================================================================
def train_one_epoch(model, loader, opt, sched, scaler, epoch, total, accum, device):
    model.train()
    model.slice_encoder[0].eval()          # frozen backbone stays in eval mode
    tot_loss, nb, skipped = 0.0, 0, 0
    rec = defaultdict(list)
    t0 = time.time()
    opt.zero_grad(set_to_none=True)
    for i, b in enumerate(loader):
        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out, attn, clab, _ = model(b["image"].to(device), b["input_ids"].to(device),
                                           b["attention_mask"].to(device), b["labels"].to(device))
                loss = out.loss / accum
            scaler.scale(loss).backward()
            if (i + 1) % accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                torch.cuda.empty_cache()
            tot_loss += out.loss.item()
            nb += 1
            for j, (task, pr) in enumerate(zip(b["task"], b["pairing"])):
                if pr == "paired_ok":
                    rec[task].append(recency_ratio(attn[j:j + 1].detach()))
        except torch.cuda.OutOfMemoryError:
            skipped += 1
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            continue
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            LOG(f"  [train e{epoch+1}/{total}] {i+1:,}/{len(loader):,} "
                f"loss {tot_loss/max(nb,1):.4f}  {el/(i+1):.2f}s/batch  "
                f"eta {(len(loader)-i-1)*el/(i+1)/60:.0f} min"
                + (f"  oom-skipped {skipped}" if skipped else ""))
    return (tot_loss / max(nb, 1),
            {t: float(np.mean(v)) for t, v in rec.items()},
            skipped)


@torch.no_grad()
def validate(model, dataset, tok, device, max_seq_len, mcq_budget, epoch):
    """Per-task validation. MCQ is scored on a stratified subsample to keep per-epoch cost sane;
    the end-of-run audit scores every item."""
    model.eval()
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2,
                        pin_memory=True, collate_fn=collate)

    agg = defaultdict(lambda: {"loss": [], "tf": [], "rec": []})
    for b in loader:
        try:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out, attn, clab, _ = model(b["image"].to(device), b["input_ids"].to(device),
                                           b["attention_mask"].to(device), b["labels"].to(device))
            pl = per_item_loss(out.logits.float(), clab)
            ta = tf_token_acc(out.logits, clab)
            for j, (task, pr) in enumerate(zip(b["task"], b["pairing"])):
                agg[task]["loss"].append(float(pl[j]))
                agg[task]["tf"].append(float(ta[j]))
                if pr == "paired_ok":
                    agg[task]["rec"].append(recency_ratio(attn[j:j + 1]))
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            continue

    # stratified MCQ subsample
    df = dataset.df
    idx_by_task = defaultdict(list)
    for i in range(len(df)):
        idx_by_task[str(df.iloc[i]["TaskCategory"])].append(i)
    per_task_budget = max(1, mcq_budget // max(len(idx_by_task), 1))
    rng = random.Random(SEED + epoch)
    mcq = defaultdict(list)
    for task, idxs in idx_by_task.items():
        pick = idxs if len(idxs) <= per_task_budget else rng.sample(idxs, per_task_budget)
        for i in pick:
            item = dataset[i]
            if not item["is_mcq"]:
                continue
            try:
                r = mcq_score_one(model, tok, item, device, max_seq_len)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue
            if r is not None:
                mcq[task].append(r)

    rows = []
    for task in sorted(agg.keys(), key=lambda t: TASK_ORDER.index(t) if t in TASK_ORDER else 99):
        a = agg[task]
        rows.append({
            "epoch": epoch + 1, "task": task, "n": len(a["loss"]),
            "loss": float(np.mean(a["loss"])) if a["loss"] else None,
            "tf_token_acc": float(np.mean(a["tf"])) if a["tf"] else None,
            "mcq_acc": float(np.mean(mcq[task])) if mcq.get(task) else None,
            "mcq_n": len(mcq.get(task, [])),
            "recency_T1_over_T0": float(np.mean(a["rec"])) if a["rec"] else None,
            "recency_n": len(a["rec"]),
        })
    overall = float(np.mean([x for t in agg for x in agg[t]["loss"]]))
    torch.cuda.empty_cache()
    return overall, rows


# ==================================================================================================
# checkpointing — the whole reason this version exists
# ==================================================================================================
def save_everything(model, opt, sched, scaler, epoch, val_loss, out_dir, tag):
    d = os.path.join(out_dir, tag)
    os.makedirs(d, exist_ok=True)
    torch.save({"epoch": epoch + 1, "val_loss": val_loss,
                "projector_state_dict": model.projector.state_dict(),
                "slice_encoder_state_dict": model.slice_encoder.state_dict(),
                "depth_aggregator_state_dict": model.depth_aggregator.state_dict()},
               os.path.join(d, "multimodal_projector_best.pt"))
    model.llm.save_pretrained(os.path.join(d, "lora_adapters"))
    torch.save({"epoch": epoch + 1, "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                "val_loss": val_loss},
               os.path.join(d, "trainer_state.pt"))
    files = sorted(os.listdir(os.path.join(d, "lora_adapters")))
    LOG(f"  [saved] {tag}/  (projector + lora_adapters{files} + trainer_state)")


def try_resume(model, opt, sched, scaler, out_dir):
    d = os.path.join(out_dir, "last")
    st = os.path.join(d, "trainer_state.pt")
    if not os.path.exists(st):
        LOG("[resume] nothing to resume from; starting at epoch 1")
        return 0
    ck = torch.load(os.path.join(d, "multimodal_projector_best.pt"),
                    map_location="cpu", weights_only=False)
    model.projector.load_state_dict(ck["projector_state_dict"])
    model.slice_encoder.load_state_dict(ck["slice_encoder_state_dict"])
    model.depth_aggregator.load_state_dict(ck["depth_aggregator_state_dict"])
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    ad = os.path.join(d, "lora_adapters")
    sf = os.path.join(ad, "adapter_model.safetensors")
    sd = load_file(sf) if os.path.exists(sf) else torch.load(
        os.path.join(ad, "adapter_model.bin"), map_location="cpu")
    set_peft_model_state_dict(model.llm, sd)
    ts = torch.load(st, map_location="cpu", weights_only=False)
    opt.load_state_dict(ts["optimizer"])
    sched.load_state_dict(ts["scheduler"])
    scaler.load_state_dict(ts["scaler"])
    LOG(f"[resume] restored through epoch {ts['epoch']} (val_loss {ts['val_loss']:.4f})")
    return int(ts["epoch"])


# ==================================================================================================
# main
# ==================================================================================================
def main():
    ap = argparse.ArgumentParser(description="3D-RAD baseline retrain v7 (resumable, per-task).")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--output_dir", default="/kaggle/working/retrain_v7")
    ap.add_argument("--llm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--val_mcq_budget", type=int, default=600,
                    help="MCQ items scored per validation (stratified). Full MCQ runs in the audit.")
    ap.add_argument("--preflight", action="store_true", help="Verify split on CPU, then exit.")
    ap.add_argument("--smoke", action="store_true",
                    help="2 short epochs on a small stratified subset. Proves the loop and saving.")
    ap.add_argument("--resume", action="store_true", help="Continue from output_dir/last/.")
    ap.add_argument("--no_audit", action="store_true", help="Skip the end-of-run audit.")
    args, _ = ap.parse_known_args(args=[] if any(
        "ipykernel" in str(a) or str(a).endswith(".json") for a in sys.argv) else None)

    os.makedirs(args.output_dir, exist_ok=True)
    global LOG
    LOG = Tee(os.path.join(args.output_dir, "train_log.txt"))

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    LOG("=" * 100)
    LOG("  3D-RAD BASELINE RETRAIN v7   (resumable, per-task metrics, saves every epoch)")
    LOG("=" * 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG(f"device     : {device} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    data_dir = autodetect_data_dir(args.data_dir)
    LOG(f"data_dir   : {data_dir}")
    LOG(f"output_dir : {args.output_dir}")

    df = build_verified_split(data_dir)
    for s in ("train", "val", "test"):
        df[df.Split == s].to_csv(os.path.join(args.output_dir, f"split_{s}_qa.csv"), index=False)
    LOG(f"[split] manifests written to {args.output_dir}/split_*_qa.csv")

    if args.preflight:
        LOG("\n" + "=" * 100)
        LOG("  PREFLIGHT  (no model, no GPU)")
        LOG("=" * 100)
        for s in ("train", "val"):
            sub = df[df.Split == s]
            LOG(f"\n  --- {s}: {len(sub):,} items ---")
            for task, g in sub.groupby("TaskCategory"):
                qt = g["QuestionType"].astype(str).str.lower()
                nreal = g[[f"Choice {L}" for L in "ABCD"]].astype(str).apply(
                    lambda c: ~c.str.strip().isin(["-", "", "nan"])).sum(axis=1)
                LOG(f"  {task:<40} n={len(g):>6}  MCQ={int((qt.str.startswith('close') & (nreal>=2)).sum()):>6}")
        LOG("\n  Split reproduces the v6 split. Safe to train. Re-run without --preflight.")
        return

    if args.smoke:
        LOG("\n*** SMOKE TEST: tiny stratified subset, 2 epochs. Not a real result. ***")
        args.epochs = 2
        args.val_mcq_budget = 24
        keep = df.groupby(["Split", "TaskCategory"], group_keys=False).head(12)
        df = keep

    timeline = build_timeline(data_dir)
    LOG(f"[setup] timeline for {len(timeline):,} patients")

    model, tok = build_model(args.llm_model, args.lora_r, args.lora_alpha, device)

    train_ds = VQADataset(df[df.Split == "train"], tok, timeline, args.max_seq_len, is_train=True)
    val_ds = VQADataset(df[df.Split == "val"], tok, timeline, args.max_seq_len, is_train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collate)
    LOG(f"[data] train {len(train_ds):,} items / {len(train_loader):,} batches | val {len(val_ds):,}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps = max((len(train_loader) // args.grad_accum_steps) * args.epochs, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = try_resume(model, opt, sched, scaler, args.output_dir) if args.resume else 0

    hist_path = os.path.join(args.output_dir, "training_history.json")
    pertask_path = os.path.join(args.output_dir, "val_per_task_by_epoch.csv")
    history = json.load(open(hist_path)) if (args.resume and os.path.exists(hist_path)) else []
    best = min([h["val_loss"] for h in history], default=float("inf"))

    LOG("\n" + "=" * 100)
    LOG(f"  TRAINING  epochs {start_epoch+1}..{args.epochs} | batch {args.batch_size} "
        f"x accum {args.grad_accum_steps} = effective {args.batch_size*args.grad_accum_steps}")
    LOG("=" * 100)

    for epoch in range(start_epoch, args.epochs):
        tr_loss, tr_rec, oom = train_one_epoch(
            model, train_loader, opt, sched, scaler, epoch, args.epochs,
            args.grad_accum_steps, device)
        val_loss, rows = validate(model, val_ds, tok, device, args.max_seq_len,
                                  args.val_mcq_budget, epoch)

        LOG("\n" + "=" * 100)
        LOG(f"  EPOCH {epoch+1}/{args.epochs}   train_loss {tr_loss:.4f}   "
            f"val_loss {val_loss:.4f} (ppl {math.exp(min(val_loss,20)):.2f})"
            + (f"   [{oom} batches OOM-skipped]" if oom else ""))
        LOG("=" * 100)
        LOG(f"  {'task':<40}{'n':>6}{'loss':>9}{'tf_acc':>9}{'mcq_acc':>9}{'mcq_n':>7}{'T1/T0':>9}")
        for r in rows:
            f = lambda v, w=9, p=4: (f"{v:>{w}.{p}f}" if v is not None else " " * (w - 1) + "-")
            LOG(f"  {r['task']:<40}{r['n']:>6}{f(r['loss'])}{f(r['tf_token_acc'])}"
                f"{f(r['mcq_acc'])}{r['mcq_n']:>7}{f(r['recency_T1_over_T0'])}")
        LOG("\n  Tasks 5 and 6 are the same items worded differently: Task 6 states the past")
        LOG("  presence/absence bit in the question, Task 5 does not. Compare those two rows.")

        history.append({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": val_loss,
                        "oom_skipped": oom, "train_recency_by_task": tr_rec,
                        "val_per_task": rows})
        # flushed every epoch, not at the end -- this is what was lost last time
        with open(hist_path, "w") as fh:
            json.dump(history, fh, indent=2)
        pd.DataFrame([r for h in history for r in h["val_per_task"]]).to_csv(
            pertask_path, index=False)

        save_everything(model, opt, sched, scaler, epoch, val_loss, args.output_dir, "last")
        save_everything(model, opt, sched, scaler, epoch, val_loss, args.output_dir,
                        f"epoch_{epoch+1}")
        if val_loss < best:
            best = val_loss
            save_everything(model, opt, sched, scaler, epoch, val_loss, args.output_dir, "best")
            LOG(f"  >>> new best val_loss {best:.4f}")

    LOG("\n" + "=" * 100)
    LOG(f"  TRAINING COMPLETE. best val_loss {best:.4f}")
    LOG("=" * 100)

    audit = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_inference_audit.py")
    if not args.no_audit and os.path.exists(audit) and not args.smoke:
        LOG("\n[audit] running the full per-task audit on val and test with the blank-image ablation")
        cmd = [sys.executable, audit, "--data_dir", data_dir,
               "--ckpt_dir", os.path.join(args.output_dir, "best"),
               "--output_dir", os.path.join(args.output_dir, "audit"),
               "--split_manifests", args.output_dir, "--splits", "val", "test"]
        LOG("[audit] " + " ".join(cmd))
        subprocess.run(cmd, check=False)
    elif not args.no_audit and not args.smoke:
        LOG(f"\n[audit] kaggle_inference_audit.py not found next to this script. Run it yourself:")
        LOG(f"  !python kaggle_inference_audit.py --ckpt_dir {args.output_dir}/best "
            f"--split_manifests {args.output_dir}")

    LOG("\n" + "=" * 100)
    LOG("  OUTPUT FILES (individual, not zipped):")
    for root, _, files in os.walk(args.output_dir):
        for f in sorted(files):
            p = os.path.join(root, f)
            LOG(f"    {os.path.relpath(p, args.output_dir):<58} "
                f"{os.path.getsize(p)/1024:>9.0f} KB")
    LOG("=" * 100)
    LOG("  >>> NOW COMMIT THE NOTEBOOK: Save Version -> Save & Run All.")
    LOG("  >>> An interactive session's /kaggle/working is deleted when the session ends.")
    LOG("  >>> That is how the previous run's LoRA adapters and metrics were lost.")
    LOG("=" * 100)


if __name__ == "__main__":
    main()
