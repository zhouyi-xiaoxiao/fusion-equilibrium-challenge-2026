#!/usr/bin/env python3
"""DIII-D profile xhc — assemble the base root of the predict path from the outputs of the training stage FT, and optionally compare it
with the submitted weights (weights/MANIFEST_CH1_XHC.sha256).

  python ch1/xhc_assemble.py <BASE_ROOT> <OUT_ROOT> [--cmp OUT.json]

OUT_ROOT/xhc/frame_mlp/members/FULL_<m>/model.pt     = OUT_ROOT/ft/full/members/FULL_<m>__lab_s500EM/model.pt   (11 FM members)
OUT_ROOT/xhc/temporal/full/weights/tR_w9d_s<s>.pt    = OUT_ROOT/ft/full/tw/lab_s500E/tR_w9d_s<s>.pt            (4 TW seeds)
OUT_ROOT/xhc/scalars/weights/full/*.pt               = BASE_ROOT/scalars/weights/full/*.pt                       (17 SC members, unchanged)
--cmp: for every file of the manifest (path stage11_ch1/xhc/full/<rel>) the file OUT_ROOT/xhc/<rel>: sha256 equal, else equality of
every tensor (state dicts, normalisation constants) and the largest absolute difference.  Prints XHC_ASSEMBLED and, with --cmp,
XHC_WCMP <identical>/<n> sha256-identical, <t> tensor-identical.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

FM = [f"d30aux_s{s}" for s in range(5)] + [f"towerd30aux_s{s}" for s in range(3)] + [f"K120d30aux_s{s}" for s in range(3)]
REPO = Path(__file__).resolve().parent.parent
PREFIX = "stage11_ch1/xhc/full/"


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def flat(o, pre=""):
    import torch
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base"); ap.add_argument("out"); ap.add_argument("--cmp", default="")
    ap.add_argument("--manifest", default=str(REPO / "weights/MANIFEST_CH1_XHC.sha256"))
    ap.add_argument("--shipped-root", default="", help="work directory holding the submitted files (default $SCR) for tensor comparison")
    a = ap.parse_args()
    base, out = Path(a.base), Path(a.out)
    X = out / "xhc"
    pairs = [(out / f"ft/full/members/FULL_{m}__lab_s500EM/model.pt", X / f"frame_mlp/members/FULL_{m}/model.pt") for m in FM]
    pairs += [(out / f"ft/full/tw/lab_s500E/tR_w9d_s{s}.pt", X / f"temporal/full/weights/tR_w9d_s{s}.pt") for s in range(4)]
    sc = sorted((base / "scalars/weights/full").glob("*.pt"))
    if len(sc) != 17:
        sys.exit(f"expected 17 SC files in {base}/scalars/weights/full, found {len(sc)}")
    pairs += [(p, X / "scalars/weights/full" / p.name) for p in sc]
    for src, dst in pairs:
        if not src.exists():
            sys.exit(f"missing output of the fine-tune stage: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
    print(f"XHC_ASSEMBLED {X}: {len(pairs)} files (11 FM + 4 TW fine-tuned, 17 SC copied from {base})")
    if not a.cmp:
        return
    import os
    import torch
    shipped = Path(a.shipped_root or os.environ["SCR"])
    rows = [l.split() for l in Path(a.manifest).read_text().splitlines() if l.strip() and not l.startswith("#")]
    res, nsame, nteq = {}, 0, 0
    for nbytes, h, rel in rows:
        assert rel.startswith(PREFIX), rel
        fa = X / rel[len(PREFIX):]
        r = {"bytes": [fa.stat().st_size if fa.exists() else None, int(nbytes)], "sha256_manifest": h}
        if not fa.exists():
            r["missing"] = True
        else:
            r["sha256"] = sha(fa); r["sha256_equal"] = r["sha256"] == h
            if r["sha256_equal"]:
                nsame += 1
            else:
                fb = shipped / rel
                if fb.exists() and sha(fb) == h:
                    ta = dict(flat(torch.load(fa, map_location="cpu", weights_only=False)))
                    tb = dict(flat(torch.load(fb, map_location="cpu", weights_only=False)))
                    keys = sorted(set(ta) & set(tb))
                    neq = [k for k in keys if ta[k].shape != tb[k].shape or not np.array_equal(ta[k], tb[k], equal_nan=True)]
                    fl = [k for k in neq if ta[k].shape == tb[k].shape and ta[k].dtype.kind == "f"]
                    r.update(tensors=len(keys), tensor_keys_equal=set(ta) == set(tb), tensors_differing=len(neq), first_differing=neq[:3],
                             max_abs_diff=float(max([np.max(np.abs(ta[k].astype(np.float64) - tb[k].astype(np.float64))) for k in fl] or [0.0])))
                    nteq += int(len(neq) == 0 and r["tensor_keys_equal"])
                else:
                    r["tensor_compare"] = "submitted file not available under " + str(shipped)
        res[rel] = r
    json.dump({"n": len(rows), "sha256_identical": nsame, "tensor_identical": nteq, "files": res}, open(a.cmp, "w"), indent=1)
    print(f"XHC_WCMP {nsame}/{len(rows)} sha256-identical, {nteq} tensor-identical (others differ; see {a.cmp})")


if __name__ == "__main__":
    main()
