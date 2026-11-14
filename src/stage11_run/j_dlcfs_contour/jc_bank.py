#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — step 2 (CPU): the anchor bank of one set.

For EVERY training row of the set (the rows the FM members were fitted on), on the warm-start FM-bag map
psi_bag0 = float32(mean of the 11 members' decoded maps), read as phi = -psi exactly like the scorer's prediction path:
  * the official bisection (clone of lcfs._extract_one_sign that keeps the bracket): last valid level l0 = hi, last invalid lo,
    the raw skimage polygon (grid-index coordinates, closed) and the official 512-point contour;
  * raw contour edge identities: every polygon vertex -> its grid edge (node a, node b) and phi at both nodes;
  * the critical node = the single grid node with lo < phi <= hi (the node whose crossing changes the marching-squares topology;
    the extracted level is its value + the frozen bisection offset).  No such node = mask-limited frame -> not an anchor;
  * self-check: numpy replay of the polygon from (edges, l0) + contour.resample_closed vs the official 512 points;
  * official symmetric Hausdorff of the warm-start contour to the reference contour (ch1d_liloss cache = lcfs.extract_lcfs on the
    float32 label map, 512 points).
  python jc_bank.py --set fold3 [--workers 280] [--max-shots 0]
Output $J/<set>/bank/: ok.npy why.npy nv.npy ea.npy eb.npy pa.npy pb.npy lvl0.npy crit.npy phic0.npy span0.npy hd0.npy rgeo.npy
                       reperr.npy shot.npz (off, rgeo_s, n_ok) gate_rows.npy report.json
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import jc_common as C
from jc_common import NG, VMAX, log

_G: dict = {}
N_ITER, N_POINTS = 22, 512
WHY = {0: "ok", 1: "no critical node (mask-limited)", 2: "several nodes in the bracket", 3: "vertex on a node / ambiguous edge",
       4: "more vertices than VMAX", 5: "prediction extraction failed", 6: "replay self-check failed", 7: "no reference contour"}


def init():
    if _G:
        return _G
    p = str(C.STARTER / "fusion_scoring")
    if p not in sys.path:
        sys.path.insert(0, p)
    from contour import grid_to_physical, resample_closed, symmetric_hausdorff
    from lcfs import _best_contour_at
    from o_point import find_o_point
    R, Z, MC = C.grid()
    _G.update(R=R, Z=Z, MC=MC, MF=MC.astype(np.float64), g2p=grid_to_physical, res=resample_closed, haus=symmetric_hausdorff,
              best=_best_contour_at, opt=find_o_point)
    return _G


def ext_level(phi):
    """Clone of lcfs._extract_one_sign (finite input) that also returns the bracket. -> (best_rc, hi, lo, iz, ir, axis_val) | None"""
    g = _G
    _, _, iz, ir, _ = g["opt"](phi, g["R"], g["Z"], g["MC"])
    axis_val = float(phi[iz, ir])
    border = np.concatenate([phi[0, :], phi[-1, :], phi[:, 0], phi[:, -1]])
    lo = float(np.min(border)); hi = axis_val - 1e-9
    if not (hi > lo):
        return None
    best = g["best"](phi, hi, iz, ir, g["MF"])
    if best is None:
        return None
    for _ in range(N_ITER):
        mid = 0.5 * (lo + hi)
        c = g["best"](phi, mid, iz, ir, g["MF"])
        if c is not None:
            hi = mid; best = c
        else:
            lo = mid
    return best, hi, lo, iz, ir, axis_val


def edges_of(best):
    """Raw skimage polygon (M,2) (row, col) -> (node_a (M,), node_b (M,), ok). Row-major node id = row * 65 + col."""
    r, c = best[:, 0], best[:, 1]
    ri, ci = np.floor(r), np.floor(c)
    r_int, c_int = (r == ri), (c == ci)
    hz = r_int & ~c_int                 # on a horizontal edge: (ri, ci) - (ri, ci + 1)
    vt = c_int & ~r_int                 # on a vertical edge:   (ri, ci) - (ri + 1, ci)
    if not np.all(hz | vt):
        return None, None, False
    a = (ri * NG + ci).astype(np.int64)
    b = np.where(hz, a + 1, a + NG)
    if (ci[hz] + 1 > NG - 1).any() or (ri[vt] + 1 > NG - 1).any():
        return None, None, False
    return a, b, True


def replay_np(a, b, phi_flat, lvl):
    g = _G
    pa, pb = phi_flat[a], phi_flat[b]
    u = (lvl - pa) / (pb - pa)
    Ra, Rb = g["R"][a % NG], g["R"][b % NG]
    Za, Zb = g["Z"][a // NG], g["Z"][b // NG]
    return np.column_stack([Ra + u * (Rb - Ra), Za + u * (Zb - Za)]), u


def one_frame(phi, Cref):
    """-> dict or (why code). phi (65,65) float64."""
    g = _G
    ex = ext_level(phi)
    if ex is None:
        return 5, None
    best, hi, lo, iz, ir, axv = ex
    C512 = g["res"](g["g2p"](best, g["R"], g["Z"]), N_POINTS)
    hd = float(g["haus"](C512, Cref)) if Cref is not None else np.nan
    out = dict(lvl0=hi, span0=axv - hi, hd0=hd, nv=len(best))
    if len(best) > VMAX:
        return 4, out
    a, b, ok = edges_of(best)
    if not ok:
        return 3, out
    pf = phi.ravel()
    cand = np.flatnonzero((pf > lo) & (pf <= hi))
    if len(cand) == 0:
        return 1, out
    if len(cand) > 1:
        return 2, out
    P, u = replay_np(a, b, pf, hi)
    err = float(np.abs(g["res"](P, N_POINTS) - C512).max())
    out.update(a=a, b=b, pa=pf[a], pb=pf[b], crit=int(cand[0]), phic0=float(pf[cand[0]]), reperr=err, C512=C512)
    if not (err < 1e-7) or not np.all((u >= 0) & (u <= 1)):
        return 6, out
    return 0, out


def work(i):
    init()
    a0, b0 = int(_G["off"][i]), int(_G["off"][i + 1])
    T = b0 - a0
    psi = np.zeros((T, NG * NG))
    for K, mm in _G["c0"]:
        mean, comps = _G["pca"][K]
        psi += np.asarray(mm[a0:b0], np.float64) @ comps + mean[None, :]
    psi = (psi / len(_G["c0"])).astype(np.float32)
    li = _G["li_idx"][a0:b0]
    r = dict(i=i, ok=np.zeros(T, bool), why=np.zeros(T, np.uint8), nv=np.zeros(T, np.int16), ea=np.zeros((T, VMAX), np.int16),
             eb=np.zeros((T, VMAX), np.int16), pa=np.zeros((T, VMAX), np.float32), pb=np.ones((T, VMAX), np.float32),
             lvl0=np.zeros(T), crit=np.zeros(T, np.int16), phic0=np.zeros(T, np.float32), span0=np.zeros(T, np.float32),
             hd0=np.full(T, np.nan, np.float32), rgeo=np.full(T, np.nan, np.float32), reperr=np.full(T, np.nan, np.float32))
    for k in range(T):
        refok = bool(_G["li_ok"][li[k]])
        Cref = np.asarray(_G["li_C"][li[k]], np.float64) if refok else None
        if refok:
            r["rgeo"][k] = 0.5 * (Cref[:, 0].max() + Cref[:, 0].min())
        why, o = one_frame(-psi[k].astype(np.float64).reshape(NG, NG), Cref)
        if o is not None:
            r["lvl0"][k], r["span0"][k], r["hd0"][k] = o["lvl0"], o["span0"], o["hd0"]
            r["nv"][k] = min(o["nv"], 32767)
        if why == 0 and not refok:
            why = 7
        r["why"][k] = why
        if why in (0, 7) or (why == 6 and "a" in o):
            n = o["nv"]
            for key, src in (("ea", o["a"]), ("eb", o["b"]), ("pa", o["pa"]), ("pb", o["pb"])):
                r[key][k, :n] = src
                r[key][k, n:] = src[-1]
            r["crit"][k], r["phic0"][k], r["reperr"][k] = o["crit"], o["phic0"], o["reperr"]
        r["ok"][k] = (why == 0)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NW", "280")))
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--tag", default="bank")
    a = ap.parse_args()
    t0 = time.time()
    info = C.set_info(a.set)
    off, li_idx, files = C.bank_rows(a.set)
    S = len(off) - 1 if not a.max_shots else min(a.max_shots, len(off) - 1)
    N = int(off[S])
    out = C.J / a.set / a.tag
    out.mkdir(parents=True, exist_ok=True)
    init()
    names = C.member_names(info["prefix"])
    c0 = []
    for m in names:
        mm = np.load(C.J / a.set / "c0" / f"{m}.npy", mmap_mode="r")
        assert mm.shape[0] == int(off[-1]), (m, mm.shape, int(off[-1]))
        c0.append((mm.shape[1], mm))
    ntr = np.load(C.LI / "cache/train/ok.npy").shape[0]
    li_ok = np.concatenate([np.load(C.LI / "cache/train/ok.npy"), np.load(C.LI / "cache/holdout/ok.npy")])
    Ctr = np.load(C.LI / "cache/train/C.npy", mmap_mode="r")
    Cho = np.load(C.LI / "cache/holdout/C.npy", mmap_mode="r")

    class Cat:                                             # lazy concat(train, holdout) row access
        def __getitem__(self, j):
            return Ctr[j] if j < ntr else Cho[j - ntr]
    _G.update(off=off, c0=c0, pca={K: C.pca(K) for K in (80, 120)}, li_idx=li_idx, li_ok=li_ok, li_C=Cat())
    # row identity check between the set's train rows and the li cache rows (times + first PCA coefficients)
    tv21 = np.concatenate([np.load(C.FEAT21 / "train/times.npy"), np.load(C.FEAT21 / "holdout/times.npy")])
    tset = np.load(info["feat"] / "train/times.npy")
    if a.set == "full":
        tset = np.concatenate([tset, np.load(info["feat"] / "holdout/times.npy")])
    assert np.array_equal(tset, tv21[li_idx]), "times of the set's rows != feat_v21 times[li_idx]"
    log(f"{a.set}: {S} shots, {N} rows, members {len(c0)}; row identity with the li cache OK ({time.time()-t0:.0f}s)")
    arr = {k: np.lib.format.open_memmap(out / f"{k}.npy", mode="w+", dtype=dt, shape=(N,) + sh) for k, dt, sh in
           [("ok", bool, ()), ("why", np.uint8, ()), ("nv", np.int16, ()), ("ea", np.int16, (VMAX,)), ("eb", np.int16, (VMAX,)),
            ("pa", np.float32, (VMAX,)), ("pb", np.float32, (VMAX,)), ("lvl0", np.float64, ()), ("crit", np.int16, ()),
            ("phic0", np.float32, ()), ("span0", np.float32, ()), ("hd0", np.float32, ()), ("rgeo", np.float32, ()),
            ("reperr", np.float32, ())]}
    os.environ["OMP_NUM_THREADS"] = "1"
    done = 0
    with mp.get_context("fork").Pool(a.workers) as pool:
        for r in pool.imap_unordered(work, range(S), chunksize=2):
            p, e = int(off[r["i"]]), int(off[r["i"] + 1])
            for k in arr:
                arr[k][p:e] = r[k]
            done += 1
            if done % 1000 == 0:
                log(f"  {done}/{S} shots {time.time()-t0:.0f}s")
    for k in arr:
        arr[k].flush()
    ok, why, hd0, rgeo = (np.asarray(arr[k]) for k in ("ok", "why", "hd0", "rgeo"))
    rg_s = np.full(S, np.nan); n_ok = np.zeros(S, np.int64); d_s = np.full(S, np.nan)
    for i in range(S):
        p, e = int(off[i]), int(off[i + 1])
        v = np.isfinite(rgeo[p:e])
        if v.any():
            rg_s[i] = float(rgeo[p:e][v].mean())
            h = hd0[p:e][v]
            d_s[i] = float(np.mean(np.minimum(1.0, np.where(np.isfinite(h), h, rg_s[i]) / rg_s[i])))
        n_ok[i] = int(ok[p:e].sum())
    np.savez(out / "shot.npz", off=off[:S + 1], rgeo_s=rg_s, n_ok=n_ok, d_s=d_s, files=np.array(files[:S]))
    # 256 calibration-gate rows from distinct shots (held out of the loss), fixed seed
    rng = np.random.default_rng(20261012)
    shots = rng.permutation(np.flatnonzero(n_ok > 0))[:256]
    gate = np.array([int(off[i]) + int(rng.choice(np.flatnonzero(ok[int(off[i]):int(off[i + 1])]))) for i in shots], np.int64)
    np.save(out / "gate_rows.npy", np.sort(gate))
    nvv = np.asarray(arr["nv"])[ok]
    rep = {"set": a.set, "shots": int(S), "rows": int(N), "ok_frac": float(ok.mean()),
           "why": {WHY[int(c)]: int((why == c).sum()) for c in np.unique(why)},
           "nv_ok_mean": float(nvv.mean()), "nv_ok_max": int(nvv.max()),
           "reperr_max_ok": float(np.nanmax(np.asarray(arr["reperr"])[ok])),
           "D_insample_bag": float(np.nanmean(d_s)), "hd0_mean_cm": float(np.nanmean(hd0) * 100),
           "hd0_median_cm": float(np.nanmedian(hd0) * 100), "hd0_p95_cm": float(np.nanquantile(hd0, 0.95) * 100),
           "rgeo_mean": float(np.nanmean(rg_s)), "gate_rows": int(len(gate)), "sec": time.time() - t0}
    json.dump(rep, open(out / "report.json", "w"), indent=1)
    log(json.dumps(rep))


if __name__ == "__main__":
    main()
