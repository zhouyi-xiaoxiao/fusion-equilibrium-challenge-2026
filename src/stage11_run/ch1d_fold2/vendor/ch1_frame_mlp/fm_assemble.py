#!/usr/bin/env python3
"""
Stage 11 / ch1_frame_mlp — build harness-ready holdout prediction files (stage-2 layout:
shot_XXXX_psirz / _q95 / _betaN) from member outputs.

Spec JSON: [{"tag": str, "members": [member names], "qb": "incumbent" | [member names],
             "which": "" | "_best", "smooth": [w_-1, w_0, w_+1] (optional, coefficient-time kernel)}]
Members with different PCA bases are averaged as DECODED maps (float64), written float32.
qb = "incumbent" -> stage6_H heads_mlp.npz q95_bag / betaN_bag (the heads inside the incumbent 0.9927 file),
so a tag with qb="incumbent" isolates the flux-map change.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCR = Path(os.environ["SCR"])
FEAT = Path(os.environ.get("FEAT", str(SCR / "stage11_ch1/feat_v21")))
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/frame_mlp")))
HC = SCR / "stage6_H"


def smooth_shots(C, offsets, w):
    w = np.asarray(w, np.float64)
    out = C.copy()
    for s in range(len(offsets) - 1):
        p, q = offsets[s], offsets[s + 1]
        if q - p < 3:
            continue
        c = C[p:q]
        pad = np.concatenate([c[:1], c, c[-1:]])
        out[p:q] = w[0] * pad[:-2] + w[1] * pad[1:-1] + w[2] * pad[2:]
    return out


def main():
    spec = json.load(open(sys.argv[1]))
    offsets = np.load(FEAT / "holdout/offsets.npy")
    heads = np.load(HC / "heads_mlp.npz")
    pcs = {}
    (OUT / "preds").mkdir(parents=True, exist_ok=True)
    for ent in spec:
        tag = ent["tag"]
        dst = OUT / "preds" / f"preds_d3d_{tag}.npz"
        if dst.exists():
            continue
        t0 = time.time()
        sfx = ent.get("which", "")
        psi = None
        for m in ent["members"]:
            md = OUT / "members" / m
            K = json.load(open(md / "meta.json"))["cfg"]["K"]
            if K not in pcs:
                pc = np.load(HC / f"pca_{K}.npz")
                pcs[K] = (pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64))
            f = md / f"coef_hold{sfx}.npy"
            C = np.load(f if f.exists() else md / "coef_hold.npy").astype(np.float64)
            if ent.get("smooth"):
                C = smooth_shots(C, offsets, ent["smooth"])
            p = C @ pcs[K][0] + pcs[K][1][None, :]
            psi = p if psi is None else psi + p
        psi /= len(ent["members"])
        if ent.get("qb", "incumbent") == "incumbent":
            q95, bN = heads["q95_bag"].astype(np.float64), heads["betaN_bag"].astype(np.float64)
        else:
            qb = np.mean([np.load(OUT / "members" / m / "qb_hold.npy").astype(np.float64) for m in ent["qb"]], 0)
            q95, bN = qb[:, 0], qb[:, 1]
        d = {}
        for s in range(len(offsets) - 1):
            a, b = offsets[s], offsets[s + 1]
            d[f"shot_{s:04d}_psirz"] = psi[a:b].reshape(-1, 65, 65).astype(np.float32)
            d[f"shot_{s:04d}_q95"] = q95[a:b].astype(np.float32)
            d[f"shot_{s:04d}_betaN"] = bN[a:b].astype(np.float32)
        np.savez(dst, **d)
        print(f"[assemble] {tag}: {len(ent['members'])} member(s), qb={'incumbent' if ent.get('qb','incumbent')=='incumbent' else len(ent['qb'])} "
              f"-> {dst.name} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
