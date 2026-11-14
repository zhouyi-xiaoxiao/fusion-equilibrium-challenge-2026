#!/usr/bin/env python3
"""Per-scalar target gate for a given base (cross-fitted on the holdout halves).

For each of the 11 targets keep the fused target (e.g. rob) only where it beats the base's own derived value on the
CALIBRATION half; elsewhere the target is the base value itself (d_j = 0: the GN step then holds that scalar fixed
instead of pulling it toward a worse head). Gate fitted on half A is written for use on half B and vice versa.

  cc_gate.py --base fmB --tgt rob     ->  tgt_fmB_robgA.npy (gate from A), tgt_fmB_robgB.npy (gate from B), gate_fmB_rob.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import cc_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--tgt", default="rob")
ap.add_argument("--margin", type=float, default=0.0, help="keep the fused target only if R2 gain > margin")
a = ap.parse_args()
y = np.load(cc.FEAT / "holdout/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]
off = np.load(cc.FEAT / "holdout/offsets.npy"); n = len(off) - 1
fb = np.load(cc.OUT / f"f11_holdout_{a.base}.npy").astype(np.float64)
T = np.load(cc.OUT / f"tgt_{a.base}_{a.tgt}.npy").astype(np.float64)
half = {"A": np.arange(off[0], off[n // 2]), "B": np.arange(off[n // 2], off[-1])}
rec = {}
for h, idx in half.items():
    rb = np.array([cc.pooled_r2(y[idx, j], np.where(np.isfinite(fb[idx, j]), fb[idx, j], T[idx, j])) for j in range(11)])
    rt = np.array([cc.pooled_r2(y[idx, j], T[idx, j]) for j in range(11)])
    keep = rt - rb > a.margin
    out = np.where(keep[None, :] & np.isfinite(T), T, np.where(np.isfinite(fb), fb, T))
    np.save(cc.OUT / f"tgt_{a.base}_{a.tgt}g{h}.npy", out)
    rec[h] = {"keep": keep.tolist(), "r2_base": rb.tolist(), "r2_tgt": rt.tolist()}
    print(f"[gate {a.base}/{a.tgt} from {h}] drop: " + ", ".join(nm for nm, k in zip(cc.NAMES11, keep) if not k))
json.dump(rec, open(cc.OUT / f"gate_{a.base}_{a.tgt}.json", "w"), indent=1)
