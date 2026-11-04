#!/usr/bin/env python3
"""Does the baseline really receive a genuine earlier scan (T0) for Task 5 and Task 6?
Uses the exact v7 pairing rule (retrain/kaggle_retrain_v7.py: build_timeline + VQADataset._pair).
Run: python tests/check_pairing_rate.py   (needs 3D-RAD_Dataset/ and split_manifests/)"""
import ast, glob, os, re, sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pandas as pd
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
src = open(f"{R}/retrain/kaggle_retrain_v7.py").read()
tree = ast.parse(src)
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ("build_timeline", "parse_filename"))
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "FILENAME_PATTERN" for t in n.targets))]
ns = dict(glob=glob, os=os, re=re, defaultdict=defaultdict, Dict=Dict, List=List, Tuple=Tuple, Optional=Optional)
exec(compile(ast.Module(body=keep, type_ignores=[]), "x", "exec"), ns)
tl = ns["build_timeline"](f"{R}/3D-RAD_Dataset/Images")
for split in ("train", "val", "test"):
    df = pd.read_csv(f"{R}/split_manifests/{split}_qa.csv")
    d = df[df.TaskCategory.str.contains("Temporal", case=False)]
    for t, g in d.groupby("TaskCategory"):
        ok = fb_unparsed = fb_none = 0
        for npy in g.NpyName:
            i = ns["parse_filename"](npy)
            if i is None: fb_unparsed += 1; continue
            l = tl.get(f"{i['split']}_{i['patient']}", [])
            if [x for x, _ in l if x < i["letter"]]: ok += 1
            else: fb_none += 1
        print(f"{split:5} {t:42} n={len(g):5} paired_ok={ok:5} no_earlier={fb_none:4} unparsed={fb_unparsed}")
