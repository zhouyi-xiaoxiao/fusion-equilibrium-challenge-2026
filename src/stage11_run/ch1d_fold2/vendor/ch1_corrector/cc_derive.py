#!/usr/bin/env python3
"""Derive the 7 scorer scalars + 4 LCFS extrema from predicted maps (official prediction path).

  cc_derive.py --coef <coef.npy (N,80)> --tag T --split train|holdout|test      -> f11_<split>_<T>.npy
  cc_derive.py --pred <preds_d3d_X.npz> --tag T --split holdout|test            (harness layout)
With --split holdout and an existing hold_ref.npz it also writes der_hold_<T>.npz (haus, ssres) and
prints the emulated composite.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp

import numpy as np

import cc_common as cc
from cc_common import log

_D = {}


def work(ab):
    a, b = ab
    cc.init_scorer()
    out = np.full((b - a, 11), np.nan)
    hd = np.full(b - a, np.nan)
    ss = np.zeros(b - a)
    for i in range(a, b):
        if "coef" in _D:
            psi = (_D["mean"] + np.asarray(_D["coef"][i], np.float64) @ _D["comps"]).reshape(65, 65)
        else:
            psi = _D["psi"][i]
        f, C = cc.dpred(psi)
        out[i - a] = f
        if "Cg" in _D:
            cg = _D["Cg"][i]
            hd[i - a] = cc.haus(C, cg) if np.isfinite(cg[0, 0]) else np.nan
            r = np.asarray(_D["gt"][i], np.float64) - psi
            ss[i - a] = float((r * r).sum())
    return a, b, out, hd, ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coef", default=None)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--workers", type=int, default=70)
    ap.add_argument("--maxf", type=int, default=0)
    a = ap.parse_args()
    cc.OUT.mkdir(parents=True, exist_ok=True)
    off = np.load(cc.FEAT / a.split / "offsets.npy")
    q = b = None
    if a.coef:
        _D["coef"] = np.load(a.coef, mmap_mode="r")
        _D["mean"], _D["comps"] = cc.pca80()
        N = _D["coef"].shape[0]
    else:
        _D["psi"], q, b = cc.load_pred_npz(a.pred, off)
        N = _D["psi"].shape[0]
    assert N == off[-1], (N, off[-1])
    ref = None
    if a.split == "holdout" and (cc.OUT / "hold_ref.npz").exists():
        ref = dict(np.load(cc.OUT / "hold_ref.npz"))
        _D["Cg"] = ref["C_gt"]
        _D["gt"] = np.load(cc.HC / "cache_hold_psi.npy", mmap_mode="r")
    NT = N if not a.maxf else min(N, a.maxf)                 # smoke: derive the first maxf frames, rest stays NaN
    step = 64
    tasks = [(s, min(NT, s + step)) for s in range(0, NT, step)]
    f = np.full((N, 11), np.nan); hd = np.full(N, np.nan); ss = np.zeros(N)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for n, (s, e, o, h, r) in enumerate(pool.imap_unordered(work, tasks, chunksize=4), 1):
            f[s:e], hd[s:e], ss[s:e] = o, h, r
            if n % 4000 == 0:
                log(f"{n}/{len(tasks)} chunks")
    log(f"[{a.tag}/{a.split}] {N} frames; finite frac per target: " + " ".join(f"{v:.5f}" for v in np.isfinite(f).mean(0)))
    np.save(cc.OUT / f"f11_{a.split}_{a.tag}.npy", f)
    if a.maxf:
        return
    if ref is not None:
        if q is None:
            h = np.load(cc.HC / "heads_mlp.npz")
            q, b = h["q95_bag"].astype(np.float64), h["betaN_bag"].astype(np.float64)
        np.savez(cc.OUT / f"der_hold_{a.tag}.npz", f11=f, haus=hd, ssres=ss, q95=q, betaN=b)
        for nm, sh in (("all", None), ("A", np.arange(0, 75)), ("B", np.arange(75, 150))):
            print(cc.fmt_score(f"emul {a.tag} [{nm}]", cc.emul_score(ref, f, hd, ss, q, b, sh)), flush=True)


if __name__ == "__main__":
    main()
