#!/usr/bin/env python3
"""
====================================================================================================
3D-RAD CHECKPOINT INFERENCE AUDIT  —  one pass, every number the thesis is missing
====================================================================================================
This does NOT train. It loads the already-trained checkpoint and runs inference only.

What it produces, per TASK (1..6) rather than pooled:
  1. MCQ accuracy by log-likelihood over the four real choices   -> Tasks 4, 5, 6 (QuestionType=Close)
  2. Free-text generation exact-match and token-F1               -> Tasks 1, 2, 3 (QuestionType=Open)
  3. Teacher-forced token accuracy and per-item loss             -> all tasks
  4. Depth-attention recency ratio T1/T0, split per task         -> Tasks 5, 6 (paired items only)
  5. The same metrics again with the CT volume replaced by zeros -> blank-image ablation
  6. The text-only floor for each task, computed from the train split

Why per-task matters here: Tasks 5 and 6 are the SAME 16,031 clinical items with different
phrasing (Task 6 hands the past presence/absence bit to the model inside the question text,
Task 5 does not). Pooling them into one "longitudinal" bucket averages away the single effect
this thesis is about. The earlier evaluation pooled them.

Why the blank-image pass matters: it measures how much of the score survives with no CT at all.
It also doubles as the control for the recency ratio -- with a zero volume there is no visual
signal, so whatever ratio the aggregator reports is its no-information baseline.

----------------------------------------------------------------------------------------------------
HOW TO RUN ON KAGGLE
----------------------------------------------------------------------------------------------------
  Cell 1:   !pip install -q -U peft
  Cell 2:   # ALWAYS smoke-test first -- 40 items per split, finishes in ~2 minutes
            !python kaggle_inference_audit.py --limit 40
  Cell 3:   # then the real run
            !python kaggle_inference_audit.py

Outputs land as individual files in /kaggle/working/inference_audit/ (NOT zipped), so each one
is downloadable with a single click from the Output panel:

    per_task_<split>.csv          the headline table
    per_item_<split>.csv          row-level predictions -- the substrate for the taxonomy work
    inference_audit_<split>.json  every aggregate number
    split_<name>_qa.csv           the split manifests, so they are never lost again
    run_log.txt

----------------------------------------------------------------------------------------------------
SAFETY: the split must match the one the model was trained on
----------------------------------------------------------------------------------------------------
The manifests were never saved, so this script regenerates them with the same algorithm and seed.
That regeneration depends on the order the CSV files are discovered in, so the script TRIES SEVERAL
orderings and keeps the one that reproduces the known reference counts exactly. If none matches it
ABORTS rather than silently evaluating on patients the model was trained on, which would turn every
number in the report into nonsense.
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
import string
import argparse
import importlib.util
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
FILENAME_PATTERN = re.compile(r"^(?P<split>[a-zA-Z]+)_(?P<patient>\d+)_(?P<letter>[a-zA-Z]+)_(?P<recon>\d+)\.npy$")
BRACKET_RE = re.compile(r"\[([01](?:,\s*[01])*)\]")

# The split the checkpoint was trained with. Regeneration is verified against these.
REFERENCE = {
    "train": {"qa": 27197, "volumes": 2181, "patients": 1043},
    "val":   {"qa": 3254,  "volumes": 255,  "patients": 130},
    "test":  {"qa": 3459,  "volumes": 260,  "patients": 131},
}


# ==================================================================================================
# logging
# ==================================================================================================
class Tee:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        print(msg, flush=True)
        self.f.write(msg + "\n")


LOG = print  # replaced in main()


# ==================================================================================================
# peft torchao patch  (carried over verbatim in intent from the training script's FIX 7)
# ==================================================================================================
def patch_peft_torchao_check():
    ok = False
    for mod_name, attr, value in (
        ("peft.import_utils", "is_torchao_available", lambda: False),
        ("peft.tuners.lora.torchao", "is_torchao_available", lambda: False),
        ("peft.tuners.lora.torchao", "dispatch_torchao",
         lambda target, adapter_name, config=None, **kw: None),
    ):
        try:
            mod = importlib.import_module(mod_name)
            setattr(mod, attr, value)
            ok = True
        except Exception as e:
            LOG(f"[peft-patch][warn] {mod_name}.{attr}: {e}")
    LOG(f"[peft-patch] applied={ok}")
    return ok


# ==================================================================================================
# path discovery
# ==================================================================================================
SEARCH_ROOTS = ("/kaggle/input", "/kaggle/working")


def _hunt(pattern: str) -> List[str]:
    out = []
    for root in SEARCH_ROOTS:
        if os.path.isdir(root):
            out += glob.glob(os.path.join(root, "**", pattern), recursive=True)
    return out


def autodetect_data_dir(explicit: Optional[str]) -> str:
    """The dataset root, not whatever directory happens to sit above several datasets.

    Kaggle mounts each dataset under a shared parent, so "the first directory containing a .npy"
    resolves to that parent and sweeps in every sibling dataset -- including this pipeline's own
    outputs, whose split manifests then get counted as source QA data. The invariant that actually
    identifies the dataset is that the CT volumes and the Task* question files share a root, so the
    deepest directory containing both is the answer.
    """
    if explicit:
        return explicit
    npys = _hunt("*.npy")
    tasks = [f for f in _hunt(os.path.join("Task*", "*.csv"))]
    if npys and tasks:
        root = os.path.commonpath(npys + tasks)
        if os.path.isdir(root):
            return root
    if npys:
        return os.path.commonpath(npys)
    raise FileNotFoundError(
        "Found no .npy volumes under /kaggle/input or /kaggle/working. Pass --data_dir.")


def autodetect_ckpt_dir(explicit: Optional[str]) -> str:
    """A usable checkpoint directory holds BOTH the projector and lora_adapters/.

    Without the adapters the language model is the untouched base model, so a directory carrying
    only the projector is not a candidate at all.
    """
    if explicit:
        return explicit
    dirs = sorted({os.path.dirname(p) for p in _hunt("multimodal_projector_best.pt")})
    usable = [d for d in dirs if os.path.isdir(os.path.join(d, "lora_adapters"))]
    if not usable:
        extra = (f"\n  Found a projector without lora_adapters/ at: {dirs[0]}" if dirs else "")
        raise FileNotFoundError(
            "Found no directory holding both multimodal_projector_best.pt and lora_adapters/."
            + extra + "\n  Pass --ckpt_dir explicitly.")
    # Accuracy peaked at epoch 1 while val_loss kept falling, so 'best' (chosen on loss) is not
    # the most accurate checkpoint. Prefer epoch_1, and say what else was available.
    order = ["epoch_1", "best", "last"]
    pick = next((d for name in order for d in usable if os.path.basename(d) == name), usable[0])
    if len(usable) > 1:
        others = ", ".join(os.path.basename(d) for d in usable if d != pick)
        LOG(f"[paths] checkpoint candidates: {os.path.basename(pick)} (picked), {others}")
        LOG("[paths] epoch_1 is preferred over best: best was selected on validation loss, which "
            "kept improving after accuracy had already peaked. Override with --ckpt_dir.")
    return pick


def autodetect_manifest_dir(explicit: Optional[str]) -> Optional[str]:
    """The directory holding the saved split manifests, if one was mounted.

    Using them skips regeneration entirely and guarantees the evaluation split is byte-for-byte
    the one training used.
    """
    if explicit:
        return explicit
    for stem in ("split_train_qa.csv", "train_qa.csv"):
        for hit in sorted(_hunt(stem)):
            d = os.path.dirname(hit)
            mates = [os.path.exists(os.path.join(d, stem.replace("train", s)))
                     for s in ("val", "test")]
            if all(mates):
                return d
    return None


# ==================================================================================================
# split regeneration, with verification against the reference counts
# ==================================================================================================
def _assemble(csv_paths, image_map) -> pd.DataFrame:
    frames = []
    for f in csv_paths:
        # A split manifest written by an earlier run carries VolumeName and Question too, so it
        # passes the column test and its rows get counted a second time. That happens whenever the
        # retrain output is mounted under the same /kaggle/input as the dataset. The invariant that
        # actually holds for source data is that it lives in a Task* directory.
        parent = os.path.basename(os.path.dirname(f))
        if "task" not in parent.lower() or os.path.basename(f).startswith("split_"):
            continue
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if "VolumeName" not in d.columns or "Question" not in d.columns:
            continue
        if "Split" in d.columns:          # a manifest, not source data
            continue
        d["TaskCategory"] = parent
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No usable QA csv files found.")
    df = pd.concat(frames, ignore_index=True)
    df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))
    df["PatientID"] = df["NpyName"].apply(lambda x: "_".join(str(x).split("_")[:2]))
    df["ImagePath"] = df["NpyName"].map(image_map)
    return df[df["ImagePath"].notna()].copy()


def _split_by_patients(df: pd.DataFrame) -> pd.DataFrame:
    # Must match the training script character for character: it wrapped unique() in
    # np.array(list(...)), giving a plain numpy string array. Passing a pandas extension
    # array (pandas>=3 returns ArrowStringArray) to rng.shuffle is not guaranteed to
    # permute correctly, which would silently produce a DIFFERENT split.
    pats = np.array(list(df["PatientID"].unique()))   # order of first appearance
    rng = np.random.RandomState(SEED)
    rng.shuffle(pats)
    n = len(pats)
    n_tr, n_va = int(n * 0.80), int(n * 0.10)
    assign = {p: "train" for p in pats[:n_tr]}
    assign.update({p: "val" for p in pats[n_tr:n_tr + n_va]})
    assign.update({p: "test" for p in pats[n_tr + n_va:]})
    out = df.copy()
    out["Split"] = out["PatientID"].map(assign)
    return out


def _counts(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    return {s: {"qa": int((df.Split == s).sum()),
                "volumes": int(df[df.Split == s]["NpyName"].nunique()),
                "patients": int(df[df.Split == s]["PatientID"].nunique())}
            for s in ("train", "val", "test")}


def build_verified_split(data_dir: str) -> pd.DataFrame:
    img_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    image_map = {os.path.basename(p): p for p in img_paths}
    LOG(f"[split] discovered {len(image_map):,} .npy volumes under {data_dir}")
    if not image_map:
        raise FileNotFoundError(f"No .npy volumes under {data_dir}")

    raw_csvs = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    LOG(f"[split] discovered {len(raw_csvs):,} csv files")

    orderings = OrderedDict([
        ("filesystem order", list(raw_csvs)),
        ("sorted by path", sorted(raw_csvs)),
        ("reverse sorted", sorted(raw_csvs, reverse=True)),
    ])

    for name, paths in orderings.items():
        df = _assemble(paths, image_map)
        df = _split_by_patients(df)
        got = _counts(df)
        match = all(got[s] == REFERENCE[s] for s in REFERENCE)
        LOG(f"[split] ordering '{name}': "
            + " | ".join(f"{s} {got[s]['qa']:,}qa/{got[s]['volumes']}vol" for s in ("train", "val", "test"))
            + ("   <== MATCHES the trained split" if match else ""))
        if match:
            for a in ("train", "val", "test"):
                for b in ("train", "val", "test"):
                    if a < b:
                        pa = set(df[df.Split == a].PatientID)
                        pb = set(df[df.Split == b].PatientID)
                        assert not (pa & pb), f"patient leakage {a}/{b}"
            LOG("[split] zero patient leakage verified across all three pairs")
            return df

    raise SystemExit(
        "\n" + "!" * 92 + "\n"
        "  ABORTING. None of the tried CSV orderings reproduced the split the checkpoint was\n"
        f"  trained on (expected train/val/test = {REFERENCE['train']['qa']}/"
        f"{REFERENCE['val']['qa']}/{REFERENCE['test']['qa']} QA pairs).\n\n"
        "  Evaluating on the wrong split would put training patients into the test set and make\n"
        "  every number meaningless, so nothing is reported.\n\n"
        "  Fix: pass --split_manifests <dir> pointing at saved train_qa.csv / val_qa.csv /\n"
        "  test_qa.csv, or check that --data_dir holds the same dataset version used for training.\n"
        + "!" * 92)


def load_split_from_manifests(manifest_dir: str, data_dir: str) -> pd.DataFrame:
    img_paths = glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)
    image_map = {os.path.basename(p): p for p in img_paths}
    frames = []
    for s in ("train", "val", "test"):
        # the retrain writes split_<name>_qa.csv; accept either spelling
        cands = [os.path.join(manifest_dir, f"split_{s}_qa.csv"),
                 os.path.join(manifest_dir, f"{s}_qa.csv")]
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            raise FileNotFoundError(
                f"no manifest for '{s}' in {manifest_dir} "
                f"(looked for split_{s}_qa.csv and {s}_qa.csv)")
        d = pd.read_csv(p)
        d["Split"] = s
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    if "NpyName" not in df.columns:
        df["NpyName"] = df["VolumeName"].apply(lambda x: str(x).replace(".nii.gz", ".npy"))
    df["NpyName"] = df["NpyName"].astype(str).apply(lambda x: x if x.endswith(".npy") else x + ".npy")
    df["PatientID"] = df["NpyName"].apply(lambda x: "_".join(str(x).split("_")[:2]))
    df["ImagePath"] = df["NpyName"].map(image_map)
    missing = int(df["ImagePath"].isna().sum())
    if missing:
        LOG(f"[split][warn] {missing:,} manifest rows have no local volume; dropping them")
        df = df[df["ImagePath"].notna()].copy()
    LOG("[split] loaded from manifests: " + " | ".join(
        f"{s} {int((df.Split == s).sum()):,}" for s in ("train", "val", "test")))
    return df


# ==================================================================================================
# volume loading, pairing, dataset
# ==================================================================================================
def parse_filename(fname: str) -> Optional[Dict]:
    m = FILENAME_PATTERN.match(fname)
    return m.groupdict() if m else None


def build_patient_timeline(data_dir: str) -> Dict[str, List[Tuple[str, str]]]:
    timeline = defaultdict(list)
    for p in glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True):
        info = parse_filename(os.path.basename(p))
        if info is None:
            continue
        timeline[f"{info['split']}_{info['patient']}"].append((info["letter"], p))
    for k in timeline:
        timeline[k] = sorted(set(timeline[k]), key=lambda x: x[0])
    return timeline


def _preflight_pairing(npy_name: str, timeline) -> str:
    info = parse_filename(str(npy_name))
    if info is None:
        return "fallback_unparsed_filename"
    tl = timeline.get(f"{info['split']}_{info['patient']}", [])
    return "paired_ok" if [l for l, _ in tl if l < info["letter"]] else "fallback_no_earlier_scan"


def resample_depth(vol_dhw: np.ndarray, target: int) -> np.ndarray:
    d = vol_dhw.shape[0]
    if d == target:
        return vol_dhw
    idx = np.linspace(0, d - 1, target).round().astype(int)
    return vol_dhw[idx]


class VolumeCache:
    """Small LRU cache. Rows are iterated in volume order, so hit rate is near 1."""

    def __init__(self, capacity: int = 96):
        self.capacity = capacity
        self.store = OrderedDict()
        self.hits = self.misses = 0

    def get(self, path: str) -> np.ndarray:
        if path in self.store:
            self.hits += 1
            self.store.move_to_end(path)
            return self.store[path]
        self.misses += 1
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        self.store[path] = arr
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)
        return arr


class EvalDataset(Dataset):
    """
    Mirrors Medical3DVQADataset from the training script in eval mode:
    no augmentation, same prompt template, same max_seq_len, same T0/T1 pairing rule.
    """

    def __init__(self, df: pd.DataFrame, tokenizer, timeline, image_map,
                 max_seq_len: int = 128, cache_capacity: int = 96):
        # volume-sorted iteration purely for disk-cache locality; metrics are order-independent
        d = df.copy()
        d["item_id"] = np.arange(len(d))          # explicit id; never relies on a column named "index"
        self.df = d.sort_values("NpyName", kind="stable").reset_index(drop=True)
        self.blank_images = False                 # set per pass; skips disk I/O entirely when True
        self.tok = tokenizer
        self.timeline = timeline
        self.image_map = image_map
        self.max_seq_len = max_seq_len
        self.cache = VolumeCache(cache_capacity)

    def __len__(self):
        return len(self.df)

    def _single(self, npy_name: str, path: str) -> np.ndarray:
        return self.cache.get(path if isinstance(path, str) and os.path.exists(path)
                              else self.image_map[npy_name])

    def _pairing_status(self, npy_name: str) -> str:
        """Filename-only pairing verdict -- no pixels touched."""
        info = parse_filename(npy_name)
        if info is None:
            return "fallback_unparsed_filename"
        tl = self.timeline.get(f"{info['split']}_{info['patient']}", [])
        return "paired_ok" if [l for l, _ in tl if l < info["letter"]] else "fallback_no_earlier_scan"

    def _pair(self, npy_name: str, path: str) -> Tuple[np.ndarray, str]:
        t1 = self._single(npy_name, path)
        info = parse_filename(npy_name)
        if info is None:
            return np.concatenate([resample_depth(t1, 16)] * 2, axis=0), "fallback_unparsed_filename"
        key = f"{info['split']}_{info['patient']}"
        tl = self.timeline.get(key, [])
        earlier = [l for l, _ in tl if l < info["letter"]]
        if not earlier:
            return np.concatenate([resample_depth(t1, 16)] * 2, axis=0), "fallback_no_earlier_scan"
        t0 = self.cache.get(dict(tl)[earlier[-1]])
        return np.concatenate([resample_depth(t0, 16), resample_depth(t1, 16)], axis=0), "paired_ok"

    def __getitem__(self, i):
        r = self.df.iloc[i]
        task = str(r["TaskCategory"])
        is_long = any(k in task.lower() for k in ["longitudinal", "temporal", "task 5", "task 6"])

        if self.blank_images:
            # Pairing status depends only on filenames, so it is still computed exactly.
            # The pixels are never read, which halves disk I/O for the ablation pass.
            pairing = (self._pairing_status(str(r["NpyName"])) if is_long
                       else "not_applicable_single_scan")
            image = torch.zeros((1, 32, 256, 256), dtype=torch.float32)
        else:
            if is_long:
                vol, pairing = self._pair(str(r["NpyName"]), r.get("ImagePath"))
            else:
                vol = self._single(str(r["NpyName"]), r.get("ImagePath"))
                pairing = "not_applicable_single_scan"
            image = torch.from_numpy(np.expand_dims(vol, 0)).float()  # (1, 32, 256, 256)

        question = str(r["Question"])
        answer = str(r["Answer"])
        prompt_text = f"Clinical Question: {question}\nDiagnostic Answer: "
        prompt_ids = self.tok.encode(prompt_text, add_special_tokens=True)

        # The number of real options VARIES: Task 4 is binary (Choice A=Yes, B=No, C/D="-"),
        # Tasks 5/6 have four. Only the real options are scored, and the gold index is taken
        # inside that reduced list -- assuming four would make every Task 4 item unscorable.
        qtype = str(r.get("QuestionType", "")).strip().lower()
        letters, choices = [], []
        for L in "ABCD":
            c = str(r.get(f"Choice {L}", "-")).strip()
            if c not in ("-", "", "nan", "NaN", "None"):
                letters.append(L)
                choices.append(c)
        gold_letter = str(r.get("AnswerChoice", "")).strip()
        is_mcq = (qtype.startswith("close") and len(choices) >= 2 and gold_letter in letters)

        return {
            "item_id": int(r["item_id"]),
            "image": image,
            "prompt_ids": prompt_ids,
            "answer": answer,
            "question": question,
            "task": task,
            "pairing": pairing,
            "is_longitudinal": is_long,
            "is_mcq": bool(is_mcq),
            "choices": choices,
            "choice_letters": letters,
            "answer_choice": gold_letter,
            "npy": str(r["NpyName"]),
        }


def collate(batch):
    return batch  # per-item processing; batching happens across the 4 MCQ options


# ==================================================================================================
# model
# ==================================================================================================
class SliceDepthAggregator(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, num_visual_tokens=16):
        super().__init__()
        self.num_visual_tokens = num_visual_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, num_visual_tokens, embed_dim) * 0.02)
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, slice_features):
        q = self.query_tokens.expand(slice_features.size(0), -1, -1)
        attn_out, attn_w = self.mha(query=q, key=slice_features, value=slice_features)
        return self.norm(q + attn_out), attn_w


def build_slice_encoder(embed_dim=512):
    import torchvision.models as tv
    resnet = tv.resnet18(weights=None)  # weights are restored from the checkpoint
    old = resnet.conv1
    resnet.conv1 = nn.Conv2d(1, old.out_channels, kernel_size=old.kernel_size,
                             stride=old.stride, padding=old.padding, bias=False)
    backbone = nn.Sequential(*list(resnet.children())[:-1])
    head = nn.Sequential(nn.Flatten(), nn.Linear(512, embed_dim), nn.LayerNorm(embed_dim))
    enc = nn.Sequential(backbone, head)
    enc._is_frozen_backbone = True
    return enc


class Medical3DVLM(nn.Module):
    def __init__(self, llm, tokenizer, vision_embed_dim=512, num_visual_tokens=16):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.num_visual_tokens = num_visual_tokens
        self.slice_encoder = build_slice_encoder(vision_embed_dim)
        self.depth_aggregator = SliceDepthAggregator(vision_embed_dim, 8, num_visual_tokens)
        h = self.llm.config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_embed_dim, h), nn.GELU(), nn.Linear(h, h), nn.LayerNorm(h))

    @torch.no_grad()
    def encode(self, volume):
        B, C, S, H, W = volume.shape
        slices = volume.permute(0, 2, 1, 3, 4).contiguous().view(B * S, C, H, W)
        feats = self.slice_encoder[0](slices)
        sf = self.slice_encoder[1](feats).view(B, S, -1)
        tokens, attn = self.depth_aggregator(sf)
        return self.projector(tokens), attn


def load_model(model_id, ckpt_dir, device):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if importlib.util.find_spec("peft") is None:
        raise RuntimeError("peft is not installed. Run:  !pip install -q -U peft")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    LOG(f"[model] loading {model_id} in {dtype}")
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True)

    lora_dir = os.path.join(ckpt_dir, "lora_adapters")
    if not os.path.isdir(lora_dir):
        raise FileNotFoundError(
            f"\n{'!'*90}\n  LoRA adapters not found at: {lora_dir}\n"
            "  The projector alone does not reproduce the trained model -- the LLM would be the\n"
            "  untouched base model, and every number would describe a model you never trained.\n"
            "  Recover lora_adapters/ from the Kaggle output of the training run, or retrain.\n"
            f"{'!'*90}")
    patch_peft_torchao_check()
    from peft import PeftModel
    llm = PeftModel.from_pretrained(base, lora_dir)
    LOG(f"[model] LoRA adapters loaded from {lora_dir}")

    # inference settings: KV cache on, checkpointing off (training turned these the other way)
    if hasattr(llm, "gradient_checkpointing_disable"):
        try:
            llm.gradient_checkpointing_disable()
        except Exception:
            pass
    llm.config.use_cache = True

    model = Medical3DVLM(llm, tok).to(device)

    ckpt_path = os.path.join(ckpt_dir, "multimodal_projector_best.pt")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.projector.load_state_dict(ck["projector_state_dict"])
    model.slice_encoder.load_state_dict(ck["slice_encoder_state_dict"])
    model.depth_aggregator.load_state_dict(ck["depth_aggregator_state_dict"])
    LOG(f"[model] vision weights loaded from {ckpt_path} "
        f"(epoch {ck.get('epoch')}, val_loss {ck.get('val_loss'):.4f})")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model, tok, {"epoch": ck.get("epoch"), "val_loss": float(ck.get("val_loss", float("nan")))}


# ==================================================================================================
# metrics
# ==================================================================================================
def normalize_answer(s: str) -> str:
    s = str(s).lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    p, g = normalize_answer(pred).split(), normalize_answer(gold).split()
    if not p or not g:
        return float(p == g)
    common = {}
    for t in p:
        if t in g:
            common[t] = min(p.count(t), g.count(t))
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    prec, rec = overlap / len(p), overlap / len(g)
    return 2 * prec * rec / (prec + rec)


def recency_ratio(attn: torch.Tensor) -> float:
    """attn: (B, n_query, n_key=32). Keys 0-15 are T0, 16-31 are T1 -- same split as training."""
    m = attn.float().mean(dim=1)
    t0 = m[:, :16].mean().item()
    t1 = m[:, 16:].mean().item()
    return t1 / (t0 + 1e-6)


# ==================================================================================================
# scoring passes
# ==================================================================================================
@torch.no_grad()
def score_item(model, tok, item, device, blank: bool, max_new_tokens: int, max_seq_len: int):
    """Returns a dict of per-item results under either real or blanked imagery."""
    img = item["image"].unsqueeze(0).to(device)
    if blank:
        img = torch.zeros_like(img)

    vis, attn = model.encode(img)                     # (1, 16, H), (1, 16, 32)
    dtype = next(model.llm.parameters()).dtype
    vis = vis.to(dtype)
    emb_layer = model.llm.get_input_embeddings()

    prompt_ids = item["prompt_ids"]
    out = {"recency": recency_ratio(attn) if item["is_longitudinal"] else None}

    if item["is_mcq"]:
        # length-normalised log-likelihood of each choice's TEXT, which is what the model was
        # trained to produce (training targeted the free-text Answer, not the letter).
        # The four options must be scored under an IDENTICAL prompt, otherwise their
        # likelihoods are not comparable. So the prompt is truncated ONCE, reserving room
        # for the longest option (the training script reserved up to 32 answer tokens the
        # same way), and no option text is ever truncated -- a clipped option would corrupt
        # its own likelihood and silently bias the argmax.
        choice_ids = [tok.encode(str(c) + tok.eos_token, add_special_tokens=False)
                      for c in item["choices"]]
        K = len(choice_ids)
        reserve = min(max(len(c) for c in choice_ids), 32)
        p_ids = prompt_ids[:max(max_seq_len - reserve, 8)]
        seqs = [p_ids + c for c in choice_ids]
        labs = [[-100] * len(p_ids) + c for c in choice_ids]
        L = max(len(s) for s in seqs)
        pad = tok.pad_token_id or 0
        ids_t = torch.full((K, L), pad, dtype=torch.long)
        lab_t = torch.full((K, L), -100, dtype=torch.long)
        att_t = torch.zeros((K, L), dtype=torch.long)
        for j, (s, l) in enumerate(zip(seqs, labs)):
            ids_t[j, :len(s)] = torch.tensor(s)
            lab_t[j, :len(l)] = torch.tensor(l)
            att_t[j, :len(s)] = 1
        ids_t, lab_t, att_t = ids_t.to(device), lab_t.to(device), att_t.to(device)

        txt = emb_layer(ids_t).to(dtype)
        vK = vis.expand(K, -1, -1)
        combined = torch.cat([vK, txt], dim=1)
        vmask = torch.ones((K, vK.size(1)), dtype=att_t.dtype, device=device)
        cmask = torch.cat([vmask, att_t], dim=1)
        vlab = torch.full((K, vK.size(1)), -100, dtype=lab_t.dtype, device=device)
        clab = torch.cat([vlab, lab_t], dim=1)

        logits = model.llm(inputs_embeds=combined, attention_mask=cmask, return_dict=True).logits
        sb = clab[:, 1:]
        # Casting all four options' logits to fp32 at once would materialise
        # 4 x L x 152k floats (~350 MB) twice over on a 14.5 GiB T4. Doing it one option at
        # a time keeps the transient a quarter of that, at no cost to the result.
        lp_rows = []
        for j in range(logits.size(0)):
            lj = logits[j:j + 1, :-1].float()
            lp_rows.append(torch.log_softmax(lj, dim=-1)
                           .gather(-1, sb[j:j + 1].clamp(min=0).unsqueeze(-1)).squeeze(-1))
            del lj
        lp = torch.cat(lp_rows, dim=0)
        msk = (sb != -100).float()
        tot = (lp * msk).sum(1)
        mean = tot / msk.sum(1).clamp(min=1)   # length-normalised: the standard MCQ scoring rule

        letters = item["choice_letters"]
        gold = letters.index(item["answer_choice"]) if item["answer_choice"] in letters else None
        i_norm, i_sum = int(mean.argmax()), int(tot.argmax())
        out.update({
            "n_choices": K,
            "pred_choice_norm": letters[i_norm],
            "pred_choice_sum": letters[i_sum],
            "gold_choice": item["answer_choice"],
            "mcq_correct_norm": None if gold is None else int(i_norm == gold),
            "mcq_correct_sum": None if gold is None else int(i_sum == gold),
            "gold_logprob": float(mean[gold]) if gold is not None else None,
            "pred_text": item["choices"][i_norm],
        })
        # teacher-forced loss/accuracy on the gold choice
        if gold is not None:
            g_mask = msk[gold]
            out["loss"] = float(-(lp[gold] * g_mask).sum() / g_mask.sum().clamp(min=1))
            preds = logits[gold, :-1].argmax(-1)
            sel = sb[gold] != -100
            out["tf_token_acc"] = float((preds[sel] == sb[gold][sel]).float().mean())
    else:
        # open-ended: greedy generation, scored by strict EM and token F1
        a_ids_full = tok.encode(item["answer"] + tok.eos_token, add_special_tokens=False)
        reserve = min(len(a_ids_full), 32)          # same rule the training script used
        p_ids = prompt_ids[:max(max_seq_len - reserve, 8)]
        ids_t = torch.tensor([p_ids], device=device)
        txt = emb_layer(ids_t).to(dtype)
        combined = torch.cat([vis, txt], dim=1)
        cmask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)
        gen = model.llm.generate(
            inputs_embeds=combined, attention_mask=cmask, max_new_tokens=max_new_tokens,
            do_sample=False, num_beams=1, pad_token_id=tok.pad_token_id)
        pred = tok.decode(gen[0], skip_special_tokens=True).strip()
        out.update({
            "pred_text": pred,
            "gen_em": int(normalize_answer(pred) == normalize_answer(item["answer"])),
            "gen_f1": token_f1(pred, item["answer"]),
        })
        # teacher-forced loss on the gold answer
        ids2 = torch.tensor([p_ids + a_ids_full], device=device)
        lab2 = torch.full_like(ids2, -100)
        lab2[0, len(p_ids):] = ids2[0, len(p_ids):]
        txt2 = emb_layer(ids2).to(dtype)
        comb2 = torch.cat([vis, txt2], dim=1)
        m2 = torch.ones(comb2.shape[:2], dtype=torch.long, device=device)
        vlab = torch.full((1, vis.size(1)), -100, dtype=lab2.dtype, device=device)
        clab = torch.cat([vlab, lab2], dim=1)
        lg = model.llm(inputs_embeds=comb2, attention_mask=m2, return_dict=True).logits
        sl, sb = lg[:, :-1].float(), clab[:, 1:]
        lp = torch.log_softmax(sl, -1).gather(-1, sb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        msk = (sb != -100).float()
        out["loss"] = float(-(lp * msk).sum() / msk.sum().clamp(min=1))
        sel = sb[0] != -100
        out["tf_token_acc"] = float((sl[0].argmax(-1)[sel] == sb[0][sel]).float().mean())

    return out


def run_pass(model, tok, ds, device, blank, args, tag):
    ds.blank_images = blank          # when True, __getitem__ never reads a .npy from disk
    rows = []
    t0 = time.time()
    n = len(ds)
    for i in range(n):
        item = ds[i]
        try:
            r = score_item(model, tok, item, device, blank, args.max_new_tokens, args.max_seq_len)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            LOG(f"[{tag}] OOM on item {i}; skipped")
            continue
        r.update({"item_id": item["item_id"], "task": item["task"],
                  "pairing": item["pairing"], "is_mcq": item["is_mcq"],
                  "npy": item["npy"], "gold_text": item["answer"],
                  "question": item["question"], "condition": "blank" if blank else "real"})
        rows.append(r)
        if (i + 1) % 200 == 0 or i + 1 == n:
            el = time.time() - t0
            LOG(f"[{tag}] {i+1:,}/{n:,}  ({el/ max(i+1,1):.2f}s/item, eta {(n-i-1)*el/max(i+1,1)/60:.1f} min)"
                f"  cache hit {ds.cache.hits/(max(ds.cache.hits+ds.cache.misses,1)):.2%}")
        if i % 400 == 0:
            torch.cuda.empty_cache()
    return pd.DataFrame(rows)


# ==================================================================================================
# aggregation
# ==================================================================================================
def text_only_floor(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> Dict[str, float]:
    """Best accuracy from question text alone: per (task, bracket-group) majority answer."""
    def key(d):
        br = d["Question"].map(
            lambda q: (lambda m: str(any(int(x) for x in m.group(1).replace(" ", "").split(",")))
                       if m else "nobracket")(BRACKET_RE.search(str(q))))
        return d["TaskCategory"].astype(str) + "|" + br
    lut = train_df.assign(k=key(train_df)).groupby("k")["Answer"].agg(
        lambda s: s.value_counts().idxmax())
    maj = train_df["Answer"].value_counts().idxmax()
    pred = key(eval_df).map(lut).fillna(maj)
    hit = (pred.values == eval_df["Answer"].values)
    return {t: float(hit[(eval_df["TaskCategory"] == t).values].mean())
            for t in sorted(eval_df["TaskCategory"].unique())}


def aggregate(df: pd.DataFrame, floors: Dict[str, float]) -> pd.DataFrame:
    recs = []
    for (task, cond), g in df.groupby(["task", "condition"]):
        r = {"task": task, "condition": cond, "n": int(len(g)),
             "text_only_floor": floors.get(task)}
        if g["is_mcq"].any():
            mc = g[g["is_mcq"]]
            r["mcq_acc_lennorm"] = float(pd.to_numeric(mc["mcq_correct_norm"], errors="coerce").mean())
            r["mcq_acc_sumlogp"] = float(pd.to_numeric(mc["mcq_correct_sum"], errors="coerce").mean())
        if (~g["is_mcq"]).any():
            op = g[~g["is_mcq"]]
            r["gen_exact_match"] = float(pd.to_numeric(op["gen_em"], errors="coerce").mean())
            r["gen_token_f1"] = float(pd.to_numeric(op["gen_f1"], errors="coerce").mean())
        r["loss"] = float(pd.to_numeric(g["loss"], errors="coerce").mean())
        r["tf_token_acc"] = float(pd.to_numeric(g["tf_token_acc"], errors="coerce").mean())
        paired = g[g["pairing"] == "paired_ok"]
        if len(paired):
            rec = pd.to_numeric(paired["recency"], errors="coerce").dropna()
            r["recency_T1_over_T0"] = float(rec.mean())
            r["recency_n_paired"] = int(len(rec))
        recs.append(r)
    out = pd.DataFrame(recs).sort_values(["task", "condition"])
    # A split can be all-MCQ or all-open, so either column may be absent entirely.
    for col in ("mcq_acc_lennorm", "mcq_acc_sumlogp", "gen_exact_match", "gen_token_f1",
                "recency_T1_over_T0", "recency_n_paired"):
        if col not in out.columns:
            out[col] = np.nan
    # headline accuracy per task = MCQ where the task has real choices, else generation EM
    out["headline_acc"] = out["mcq_acc_lennorm"].combine_first(out["gen_exact_match"])
    out["gain_over_text_floor"] = out["headline_acc"] - out["text_only_floor"]
    cols = ["task", "condition", "n", "headline_acc", "text_only_floor", "gain_over_text_floor",
            "mcq_acc_lennorm", "mcq_acc_sumlogp", "gen_exact_match", "gen_token_f1",
            "tf_token_acc", "loss", "recency_T1_over_T0", "recency_n_paired"]
    return out[[c for c in cols if c in out.columns]]


# ==================================================================================================
# main
# ==================================================================================================
def main():
    ap = argparse.ArgumentParser(description="3D-RAD checkpoint inference audit (no training).")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--output_dir", default="/kaggle/working/inference_audit")
    ap.add_argument("--split_manifests", default=None,
                    help="Directory holding the split manifests (split_train_qa.csv or "
                         "train_qa.csv, and the val/test equivalents). Skips regeneration.")
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--llm_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--max_seq_len", type=int, default=128)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--cache_capacity", type=int, default=96)
    ap.add_argument("--limit", type=int, default=0, help="Evaluate only N items per split (smoke test).")
    ap.add_argument("--no_blank", action="store_true", help="Skip the blank-image ablation pass.")
    ap.add_argument("--preflight", action="store_true",
                    help="Validate split, pairing, MCQ coverage and file availability on CPU, "
                         "then exit WITHOUT loading the model. Takes seconds. Run this first.")
    args, _ = ap.parse_known_args(args=[] if any(
        "ipykernel" in str(a) or str(a).endswith(".json") for a in sys.argv) else None)

    os.makedirs(args.output_dir, exist_ok=True)
    global LOG
    LOG = Tee(os.path.join(args.output_dir, "run_log.txt"))

    LOG("=" * 100)
    LOG("  3D-RAD CHECKPOINT INFERENCE AUDIT   (inference only -- nothing is trained)")
    LOG("=" * 100)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOG(f"device        : {device} "
        f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'})")

    data_dir = autodetect_data_dir(args.data_dir)
    manifest_dir = autodetect_manifest_dir(args.split_manifests)
    try:
        ckpt_dir = autodetect_ckpt_dir(args.ckpt_dir)
    except FileNotFoundError:
        if not args.preflight:
            raise
        # Reporting that the checkpoint is missing IS the point of preflight, so it keeps
        # going and still validates the split, pairing and MCQ coverage.
        ckpt_dir = None
        LOG("[warn] no checkpoint found; preflight continues and reports it below")
    LOG(f"data_dir      : {data_dir}")
    LOG(f"ckpt_dir      : {ckpt_dir if ckpt_dir else '(none found)'}")
    LOG(f"manifests     : {manifest_dir if manifest_dir else '(none found, split will be regenerated)'}")
    LOG(f"output_dir    : {args.output_dir}")
    if args.limit:
        LOG(f"*** SMOKE TEST: only {args.limit} items per split ***")

    # 1. split
    if manifest_dir:
        df = load_split_from_manifests(manifest_dir, data_dir)
        got = _counts(df)
        if all(got[s] == REFERENCE[s] for s in REFERENCE):
            LOG("[split] manifest counts match the trained split exactly "
                + " ".join(f"{s}={got[s]['qa']:,}" for s in ("train", "val", "test")))
        else:
            # The manifests ARE the split training used, so this is a warning rather than a stop,
            # but a mismatch means the write-up's numbers describe a different partition.
            LOG("\n" + "!" * 92)
            LOG("  [split][WARNING] manifest counts differ from the recorded trained split.")
            for s in ("train", "val", "test"):
                LOG(f"    {s}: manifest {got[s]['qa']:,} qa / {got[s]['volumes']} vol"
                    f"   expected {REFERENCE[s]['qa']:,} qa / {REFERENCE[s]['volumes']} vol")
            LOG("  Proceeding, because a manifest is authoritative about its own run -- but any")
            LOG("  number produced here describes that partition, not the one already written up.")
            LOG("!" * 92 + "\n")
    else:
        df = build_verified_split(data_dir)
    for s in ("train", "val", "test"):
        df[df.Split == s].to_csv(os.path.join(args.output_dir, f"split_{s}_qa.csv"), index=False)
    LOG(f"[split] manifests written to {args.output_dir}/split_*_qa.csv")

    train_df = df[df.Split == "train"]
    timeline = build_patient_timeline(data_dir)
    image_map = {os.path.basename(p): p for p in
                 glob.glob(os.path.join(data_dir, "**", "*.npy"), recursive=True)}
    LOG(f"[setup] timeline for {len(timeline):,} patients")

    # 1b. preflight: everything that can be checked without a GPU
    if args.preflight:
        LOG("\n" + "=" * 100)
        LOG("  PREFLIGHT  (no model loaded, no GPU used)")
        LOG("=" * 100)
        if ckpt_dir is None:
            LOG("  multimodal_projector_best.pt : MISSING  (no checkpoint attached to this session)")
            LOG("  lora_adapters/               : MISSING")
            LOG("  -> Attach the training run's output, or pass --ckpt_dir. The data checks below")
            LOG("     still run, so you can confirm everything else is sound meanwhile.")
        else:
            lora_dir = os.path.join(ckpt_dir, "lora_adapters")
            ckpt_pt = os.path.join(ckpt_dir, "multimodal_projector_best.pt")
            LOG(f"  multimodal_projector_best.pt : {'FOUND' if os.path.exists(ckpt_pt) else 'MISSING'}")
            LOG(f"  lora_adapters/               : {'FOUND' if os.path.isdir(lora_dir) else 'MISSING'}"
                + ("" if os.path.isdir(lora_dir)
                   else "   <== without this the LLM is the untrained base model"))
            if os.path.isdir(lora_dir):
                LOG(f"     adapter files: {sorted(os.listdir(lora_dir))}")
        for split in args.splits:
            sub = df[df.Split == split]
            LOG(f"\n  --- {split}: {len(sub):,} items ---")
            LOG(f"  {'task':<40}{'n':>6}{'MCQ':>6}{'open':>6}{'paired_ok':>11}{'floor':>8}")
            floors = text_only_floor(train_df, sub)
            for task, g in sub.groupby("TaskCategory"):
                qt = g["QuestionType"].astype(str).str.lower()
                n_real = g[[f"Choice {L}" for L in "ABCD"]].astype(str).apply(
                    lambda c: ~c.str.strip().isin(["-", "", "nan"])).sum(axis=1)
                mcq = int((qt.str.startswith("close") & (n_real >= 2)).sum())
                is_long = any(k in task.lower() for k in ["longitudinal", "temporal"])
                paired = "-"
                if is_long:
                    st = g["NpyName"].map(lambda nm: _preflight_pairing(nm, timeline))
                    paired = f"{int((st == 'paired_ok').sum())}/{len(g)}"
                LOG(f"  {task:<40}{len(g):>6}{mcq:>6}{len(g)-mcq:>6}{paired:>11}"
                    f"{floors.get(task, float('nan')):>8.4f}")
            missing = int(sub["ImagePath"].isna().sum()) if "ImagePath" in sub else 0
            LOG(f"  volumes referenced but missing on disk: {missing}")
        LOG("\n  Preflight done. If the table looks right, re-run without --preflight.")
        return

    # 2. model
    model, tok, ckinfo = load_model(args.llm_model, ckpt_dir, device)

    # 3. passes
    summary = {"checkpoint": ckinfo, "data_dir": data_dir, "ckpt_dir": ckpt_dir,
                "limit": args.limit, "splits": {}}
    for split in args.splits:
        sub = df[df.Split == split]
        if args.limit:
            sub = sub.groupby("TaskCategory", group_keys=False).head(
                max(1, args.limit // sub["TaskCategory"].nunique()))
        LOG(f"\n{'='*100}\n  SPLIT = {split}   ({len(sub):,} items)\n{'='*100}")
        ds = EvalDataset(sub, tok, timeline, image_map, args.max_seq_len, args.cache_capacity)

        parts = [run_pass(model, tok, ds, device, False, args, f"{split}/real")]
        if not args.no_blank:
            parts.append(run_pass(model, tok, ds, device, True, args, f"{split}/blank"))
        per_item = pd.concat(parts, ignore_index=True)

        floors = text_only_floor(train_df, sub)
        per_task = aggregate(per_item, floors)

        per_item.to_csv(os.path.join(args.output_dir, f"per_item_{split}.csv"), index=False)
        per_task.to_csv(os.path.join(args.output_dir, f"per_task_{split}.csv"), index=False)

        LOG(f"\n  PER-TASK RESULTS ({split})")
        LOG("  " + per_task.to_string(index=False).replace("\n", "\n  "))

        real = per_task[per_task.condition == "real"].set_index("task")
        blank = per_task[per_task.condition == "blank"].set_index("task")
        if len(blank):
            LOG(f"\n  VISION CONTRIBUTION ({split}): real - blank, per task")
            for t in real.index:
                if t in blank.index:
                    LOG(f"    {t:<40} real {real.loc[t,'headline_acc']:.4f}  "
                        f"blank {blank.loc[t,'headline_acc']:.4f}  "
                        f"delta {real.loc[t,'headline_acc']-blank.loc[t,'headline_acc']:+.4f}  "
                        f"(text-only floor {real.loc[t,'text_only_floor']:.4f})")

        summary["splits"][split] = json.loads(per_task.to_json(orient="records"))
        with open(os.path.join(args.output_dir, f"inference_audit_{split}.json"), "w") as fh:
            json.dump(summary["splits"][split], fh, indent=2)

    with open(os.path.join(args.output_dir, "inference_audit_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    LOG("\n" + "=" * 100)
    LOG("  DONE. Files in " + args.output_dir + " (not zipped -- download individually):")
    for f in sorted(os.listdir(args.output_dir)):
        LOG(f"    {f}  ({os.path.getsize(os.path.join(args.output_dir, f))/1024:.0f} KB)")
    LOG("=" * 100)


if __name__ == "__main__":
    main()
