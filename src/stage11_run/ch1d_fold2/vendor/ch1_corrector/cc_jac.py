#!/usr/bin/env python3
"""Step 4a — per-frame Jacobian of the 11 scalars w.r.t. the PCA80 coefficients AROUND THE PREDICTED MAP.

Central finite differences through the official derivation (prediction path), one step per coefficient:
step_k = per-coefficient OOF residual RMS of the model family (steps .npy, (80,)). The map itself need not lie
in the PCA80 span: perturbations are psi +- s_k * comp_k. NaN on one side -> one-sided secant; both -> 0.

  cc_jac.py --split holdout --tag bag20 --pred preds_d3d_X.npz --steps steps_inc.npy
  cc_jac.py --split test --tag gv21 --coef coef_test_gv21.npy --steps steps_gv21.npy
Output: $CC_OUT/jac_<split>_<tag>.npz  (f0 raw-li (N,11), f0_off official, J (N,11,80) float32, nanJ) + f11_<split>_<tag>.npy
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import numpy as np

import cc_common as cc
from cc_common import log

_D = {}


def load_maps(split, pred=None, coef=None):
    """-> dict(psi getter pieces, q, b, off). Keeps coefficient files lazy (decoded per frame)."""
    off = np.load(cc.FEAT / split / "offsets.npy")
    mean, comps = cc.pca80()
    d = {"off": off, "mean": mean, "comps": comps, "q": None, "b": None}
    if coef:
        d["coef"] = np.load(coef).astype(np.float64)
        assert d["coef"].shape[0] == off[-1]
    else:
        d["psi"], d["q"], d["b"] = cc.load_pred_npz(pred, off)
    return d


def get_psi(d, i):
    if "coef" in d:
        return (d["mean"] + d["coef"][i] @ d["comps"]).reshape(65, 65)
    return d["psi"][i]


def work(ab):
    a, b = ab
    cc.init_scorer()
    comps, st = _D["comps"], _D["steps"]
    f0 = np.full((b - a, 11), np.nan); J = np.zeros((b - a, 11, cc.K), np.float32); nn = np.zeros(b - a, np.int32)
    foff = np.full((b - a, 11), np.nan)
    t0 = time.time()
    for i in range(a, b):
        psi = get_psi(_D, i)
        fo, f, _ = cc.dpred_raw(psi)
        f0[i - a] = f; foff[i - a] = fo
        for k in range(cc.K):
            pk = comps[k].reshape(65, 65) * st[k]
            _, fp, _ = cc.dpred_raw(psi + pk)
            _, fm, _ = cc.dpred_raw(psi - pk)
            okp, okm = np.isfinite(fp), np.isfinite(fm)
            col = np.where(okp & okm, (fp - fm) / (2 * st[k]),
                           np.where(okp, (fp - f) / st[k], np.where(okm, (f - fm) / st[k], 0.0)))
            bad = ~np.isfinite(col)
            nn[i - a] += int((~(okp & okm)).sum())
            col[bad] = 0.0
            J[i - a, :, k] = col
    return a, b, f0, J, nn, time.time() - t0, foff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--pred", default=None)
    ap.add_argument("--coef", default=None)
    ap.add_argument("--steps", required=True)
    ap.add_argument("--workers", type=int, default=70)
    ap.add_argument("--maxf", type=int, default=0)
    ap.add_argument("--patch", action="store_true", help="recompute (raw-li) only the frames whose cached f0 has a nan scalar")
    a = ap.parse_args()
    t0 = time.time()
    _D.update(load_maps(a.split, a.pred, a.coef))
    _D["steps"] = np.load(a.steps).astype(np.float64)
    N = int(_D["off"][-1])
    if a.maxf:
        N = int(_D["off"][np.searchsorted(_D["off"], a.maxf, "right") - 1])       # whole shots only
    step = 8
    tasks = [(s, min(N, s + step)) for s in range(0, N, step)]
    f0 = np.full((N, 11), np.nan); J = np.zeros((N, 11, cc.K), np.float32); nn = np.zeros(N, np.int32); cs = 0.0
    foff = np.full((N, 11), np.nan)
    if a.patch:
        z = np.load(cc.OUT / f"jac_{a.split}_{a.tag}.npz")
        f0, J, nn = z["f0"].copy(), z["J"].copy(), z["nanJ"].copy()
        foff = z["f0_off"].copy() if "f0_off" in z.files else z["f0"].copy()
        cs = float(z["core_seconds"])
        bad = np.nonzero(~np.isfinite(foff[:, :7]).all(1))[0]
        tasks = [(int(i), int(i) + 1) for i in bad]
        log(f"patch: {len(bad)} frames with a nan official scalar")
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for n, (s, e, f, j, x, sec, fo) in enumerate(pool.imap_unordered(work, tasks, chunksize=1), 1):
            f0[s:e], J[s:e], nn[s:e], foff[s:e] = f, j, x, fo
            cs += sec
            if n % 1000 == 0:
                log(f"{n}/{len(tasks)} chunks, {time.time()-t0:.0f}s wall, {cs/ (n*step):.2f} core-s/frame")
    sfx = "_smoke" if a.maxf else ""
    np.savez(cc.OUT / f"jac_{a.split}_{a.tag}{sfx}.npz", f0=f0, f0_off=foff, J=J, nanJ=nn, steps=_D["steps"], core_seconds=cs)
    if not a.maxf:
        np.save(cc.OUT / f"f11_{a.split}_{a.tag}.npy", foff)          # official values = head inputs
    log(f"[jac {a.split}/{a.tag}] {N} frames, wall {time.time()-t0:.0f}s, core-h {cs/3600:.2f} ({cs/N:.2f} core-s/frame), "
        f"frames with any one-sided/zero column {float((nn > 0).mean()):.4f}, base-failed frames official {int((~np.isfinite(foff[:, :7]).all(1)).sum())} / raw-li {int((~np.isfinite(f0[:, :7]).all(1)).sum())}")


if __name__ == "__main__":
    main()
