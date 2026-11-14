#!/usr/bin/env python3
"""ch1_package_final — compare a train.sh root with the shipped FcCX weights: sha256 of every weight file and, where the files
differ, equality of the tensors inside (state dicts / normalisation constants).   pf_wcmp.py ROOT OUT.json"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

S = Path(os.environ["SCR"]) / "stage11_ch1"
ROOT = Path(sys.argv[1])
PAIRS = [(ROOT / "base/frame_mlp/members", S / "e_cx/full/frame_mlp/members", "*/model.pt"),
         (ROOT / "base/temporal/full/weights", S / "e_cx/full/temporal/full/weights", "*.pt"),
         (ROOT / "base/scalars/weights/full", S / "e_cx/full/scalars/weights/full", "*.pt"),
         (ROOT / "heads", S / "f_corr_canon/full/corrector", "ck_direct2_s*.pt")]
if os.environ.get("HEADS", "v21") == "v22c10":    # profile hc10x24: the 24 v2.2 heads replace the 6 v2.1 heads
    PAIRS[-1] = (ROOT / "heads22", S / "l_corr/heads22/full", "ck_c10[abcd]_s*.pt")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def flat(o, pre=""):
    if isinstance(o, torch.Tensor):
        yield pre, o.detach().cpu().numpy()
    elif isinstance(o, np.ndarray):
        yield pre, o
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            yield from flat(o[k], f"{pre}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from flat(v, f"{pre}/{i}")


out, nsame, nteq, n = {}, 0, 0, 0
for a, b, pat in PAIRS:
    for fb in sorted(b.glob(pat)):
        rel = str(fb.relative_to(b)); fa = a / rel; n += 1
        if not fa.exists():
            out[f"{b.parent.name}/{rel}"] = {"missing_in_root": True}; continue
        r = {"bytes": [os.path.getsize(fa), os.path.getsize(fb)], "sha256_shipped": sha(fb), "sha256_equal": sha(fa) == sha(fb)}
        if not r["sha256_equal"]:
            ta, tb = dict(flat(torch.load(fa, map_location="cpu", weights_only=False))), dict(flat(torch.load(fb, map_location="cpu", weights_only=False)))
            keys = sorted(set(ta) & set(tb))
            neq = [k for k in keys if ta[k].shape != tb[k].shape or not np.array_equal(ta[k], tb[k], equal_nan=True)]
            r.update(tensors=len(keys), tensor_keys_equal=set(ta) == set(tb), tensors_differing=len(neq), first_differing=neq[:3],
                     max_abs_diff=float(max([np.max(np.abs(ta[k].astype(np.float64) - tb[k].astype(np.float64))) for k in neq if ta[k].shape == tb[k].shape and ta[k].dtype.kind == "f"] or [0.0])))
            nteq += int(len(neq) == 0 and r["tensor_keys_equal"])
        else:
            nsame += 1
        out[f"{b.parent.name}/{rel}"] = r
summ = {"files": n, "sha256_identical": nsame, "different_bytes_but_identical_tensors": nteq, "tensor_differences": n - nsame - nteq}
json.dump({"summary": summ, "files": out}, open(sys.argv[2], "w"), indent=1)
print("PF_WCMP", json.dumps(summ))
for k, v in out.items():
    if not v.get("sha256_equal"):
        print("  ", k, json.dumps({kk: vv for kk, vv in v.items() if kk != "sha256_shipped"})[:300])
