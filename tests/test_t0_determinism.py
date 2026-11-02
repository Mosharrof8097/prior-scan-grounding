#!/usr/bin/env python3
"""
The earlier scan (T0) fed to the model must not depend on the interpreter's hash seed.

92% of (patient, letter) scans have several reconstructions (a_1, a_2, ...). The three
pairing implementations sort by letter only and then build `dict(tl)[letter]`, so ties are
broken by iteration order of a `set` of string tuples, which changes with PYTHONHASHSEED.

Run:  python tests/test_t0_determinism.py        (exit 1 if any script is nondeterministic)
"""
import ast, json, os, subprocess, sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
IMAGES = os.path.join(ROOT, "3D-RAD_Dataset", "Images")
SCRIPTS = {"retrain/kaggle_retrain_v7.py": "build_timeline",
           "retrain/kaggle_inference_audit.py": "build_patient_timeline",
           "inference_audit/kaggle_inference_audit.py": "build_patient_timeline"}

CHILD = r'''
import ast, glob, json, os, re, sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
path, fn, images = sys.argv[1:4]
src = open(path).read()
tree = ast.parse(src)
keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in (fn, "parse_filename"))
        or (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "FILENAME_PATTERN" for t in n.targets))]
ns = dict(glob=glob, os=os, re=re, defaultdict=defaultdict, Dict=Dict, List=List, Tuple=Tuple, Optional=Optional)
exec(compile(ast.Module(body=keep, type_ignores=[]), path, "exec"), ns)
tl = ns[fn](images)
# what the dataset code actually does: dict(tl)[letter] -> one path per letter
print(json.dumps({k: dict(v) for k, v in sorted(tl.items())}, sort_keys=True))
'''


def selection(script, fn, seed):
    env = dict(os.environ, PYTHONHASHSEED=str(seed))
    out = subprocess.run([sys.executable, "-I", "-c", CHILD, os.path.join(ROOT, script), fn, IMAGES],
                         env=env, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main():
    bad = 0
    for script, fn in SCRIPTS.items():
        sels = [selection(script, fn, s) for s in (1, 2, 3, 4)]
        diff = sum(1 for k in sels[0] for s in sels[1:] if s[k] != sels[0][k])
        n_scans = sum(len(v) for v in sels[0].values())
        print(f"{script:<45} patients={len(sels[0])} scans={n_scans}  seed-dependent selections={diff}")
        bad += diff > 0
    print("PASS" if not bad else f"FAIL: {bad} script(s) pick T0 by hash order")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
