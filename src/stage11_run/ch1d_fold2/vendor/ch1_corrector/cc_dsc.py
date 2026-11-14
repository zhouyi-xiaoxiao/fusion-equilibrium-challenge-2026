#!/usr/bin/env python3
"""Prior innovation RMS s_j of a model family (ridge scale R of the Gauss-Newton step, cc_apply --dsc).

s_j = sqrt(nanmean_train (f_j(psi_true) - f_j(psi_OOF))^2)  (same quantity cc_heads.py innov mode stores in stats.dsc)
  cc_dsc.py --base-train f11_train_gv21.npy --tag gv21   ->  $CC_OUT/dsc_gv21.json  ({"stats": {"dsc": [...]}})
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import cc_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("--base-train", required=True); ap.add_argument("--tag", required=True)
a = ap.parse_args()
y = np.load(cc.FEAT / "train/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]
fb = np.load(cc.OUT / a.base_train).astype(np.float64)
dsc = np.sqrt(np.nanmean((y - fb) ** 2, 0))
json.dump({"stats": {"dsc": dsc.tolist()}, "base_train": a.base_train}, open(cc.OUT / f"dsc_{a.tag}.json", "w"), indent=1)
print("dsc " + " ".join(f"{n} {v:.4g}" for n, v in zip(cc.NAMES11, dsc)))
