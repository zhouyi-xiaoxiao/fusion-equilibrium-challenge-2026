#!/usr/bin/env python3
"""
Stage 11d / ch1d_liloss — TRUE-contour cache for the exact fixed-contour li functional h_C (notes/STAGE11_CH1_PRO_FOLLOWUP1.md §2).

For every feat_v21 row of the train (6,891 shots) and holdout (150 shots) splits, from the float32 GT psi exactly as the
harness / local_score.build_reference reads it:
  C    the scorer's reference LCFS  = lcfs.extract_lcfs(psi, R, Z, "DIII-D", mask_coarse, mask_f, n_points=512)
  M    the HARD node mask           = derive._points_in_polygon(grid nodes, C)  (the 'inside' of internal_inductance)
  h    h_C(psi_true)                = derive.internal_inductance(psi_true, R, Z, "DIII-D", C)   (== y_aux cons_li, unclamped)
Rows are matched to the parquet frames by efit_times (feat_v21 times.npy, per-shot two-pointer) and cross-checked with the
first 3 PCA80 coefficients of the matched GT map against y_C80.
Out: $LI/cache/<split>/{C.npy (N,512,2) float32, Mbits.npy (N,529) uint8 (np.packbits bitorder little over the 4225 nodes,
     row-major (Z,R)), h.npy (N,) float64, ok.npy (N,) bool, C64_maxdiff.json}
Env: WS SCR LI FEAT21 D3D_TRAIN_DIR WORKERS SPLITS(train,holdout) MAXSH(0)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
SCR = Path(os.environ["SCR"])
LI = Path(os.environ.get("LI", str(SCR / "stage11_ch1/liloss")))
# development: stage11_ch1/feat_v21; its train / holdout files.txt, offsets, times, y_C80, y_aux are byte-identical to those of
# the canonical v2.2 cache e_cx/h150/feat (generated data), the package default (jc_common.py)
FEAT = Path(os.environ.get("FEAT21", str(SCR / "stage11_ch1/e_cx/h150/feat")))
FULL_DIR = Path(os.environ.get("D3D_TRAIN_DIR", str(WS / "data_full/diii_d_train")))
HC = SCR / "stage6_H"
WORKERS = int(os.environ.get("WORKERS", "280"))
SPLITS = os.environ.get("SPLITS", "holdout,train").split(",")
MAXSH = int(os.environ.get("MAXSH", "0"))
MACHINE = "DIII-D"
NB = (65 * 65 + 7) // 8
_G = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    for p in (WS / "starter", WS / "starter/fusion_scoring"):
        sys.path.insert(0, str(p))
    m = np.load(WS / "starter/fusion_scoring/masks/d3d_envelope.npz")
    _G["R"], _G["Z"] = m["grid_R"].astype(np.float64), m["grid_Z"].astype(np.float64)
    _G["mc"] = m["mask_coarse"].astype(bool)
    _G["mf"] = _G["mc"].astype(np.float64)
    pc = np.load(HC / "pca_80.npz")
    _G["pm"], _G["pc"] = pc["mean_"].astype(np.float64), pc["comps_"][:3].astype(np.float64)
    RR, ZZ = np.meshgrid(_G["R"], _G["Z"])
    _G["RR"], _G["ZZ"] = RR.ravel(), ZZ.ravel()


def work(args):
    split, i, fname, tq = args
    try:
        import pandas as pd
        from experiments import _as_psirz_stack
        from derive import _points_in_polygon, internal_inductance
        from lcfs import extract_lcfs
        t0 = time.time()
        row = pd.read_parquet(FULL_DIR / fname, columns=["efit_times", "efit_psirz"]).iloc[0]
        te = np.asarray(row["efit_times"], dtype=np.float64)
        psi = _as_psirz_stack(row["efit_psirz"])
        n = min(len(te), len(psi))
        # two-pointer match of the cached row times (a subsequence in order) to the parquet frames
        idx = np.empty(len(tq), np.int64)
        j = 0
        for k, t in enumerate(tq):
            while j < n and te[j] != t:
                j += 1
            if j >= n:
                raise RuntimeError(f"time {t} (row {k}) not found")
            idx[k] = j
            j += 1
        ps = psi[idx]
        R, Z, mc, mf = _G["R"], _G["Z"], _G["mc"], _G["mf"]
        N = len(ps)
        C = np.zeros((N, 512, 2), np.float32)
        Mb = np.zeros((N, NB), np.uint8)
        h = np.full(N, np.nan)
        ok = np.zeros(N, bool)
        c3 = ((np.nan_to_num(ps.reshape(N, -1).astype(np.float64)) - _G["pm"]) @ _G["pc"].T).astype(np.float32)
        for k in range(N):
            p = ps[k]
            if not np.isfinite(p).all():
                continue
            c = extract_lcfs(p, R, Z, MACHINE, mc, mf, n_points=512)
            if c is None or len(c) < 8:
                continue
            ins = _points_in_polygon(_G["RR"], _G["ZZ"], c)
            if ins.sum() < 4:
                continue
            h[k] = internal_inductance(p, R, Z, MACHINE, c)
            C[k] = c.astype(np.float32)
            Mb[k] = np.packbits(ins.astype(np.uint8), bitorder="little")
            ok[k] = np.isfinite(h[k])
        return {"ok": True, "split": split, "i": i, "C": C, "Mb": Mb, "h": h, "okf": ok, "c3": c3, "sec": time.time() - t0}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "split": split, "i": i, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1200:]}


def run_split(split, pool):
    t0 = time.time()
    d = FEAT / split
    files = [l.strip() for l in (d / "files.txt").read_text().split("\n") if l.strip()]
    off = np.load(d / "offsets.npy").astype(np.int64)
    tim = np.load(d / "times.npy")
    assert len(files) == len(off) - 1, (len(files), len(off))
    nsh = len(files) if not MAXSH else min(MAXSH, len(files))
    N = int(off[nsh])
    o = LI / "cache" / split
    o.mkdir(parents=True, exist_ok=True)
    C = np.lib.format.open_memmap(o / "C.npy", mode="w+", dtype=np.float32, shape=(N, 512, 2))
    Mb = np.lib.format.open_memmap(o / "Mbits.npy", mode="w+", dtype=np.uint8, shape=(N, NB))
    h = np.full(N, np.nan)
    okf = np.zeros(N, bool)
    c3 = np.zeros((N, 3), np.float32)
    tasks = [(split, i, files[i], tim[off[i]:off[i + 1]]) for i in range(nsh)]
    fails, done = [], 0
    for r in pool.imap_unordered(work, tasks, chunksize=1):
        done += 1
        if not r["ok"]:
            fails.append((r["i"], r["err"]))
            log(f"FAIL {split} {r['i']}: {r['err']}\n{r['tb']}")
            continue
        a, b = off[r["i"]], off[r["i"] + 1]
        C[a:b], Mb[a:b], h[a:b], okf[a:b], c3[a:b] = r["C"], r["Mb"], r["h"], r["okf"], r["c3"]
        if done % 500 == 0:
            log(f"  {split} {done}/{nsh} shots {time.time()-t0:.0f}s")
    C.flush(); Mb.flush()
    np.save(o / "h.npy", h)
    np.save(o / "ok.npy", okf)
    yC = np.load(d / "y_C80.npy", mmap_mode="r")[:N, :3]
    aux = np.load(d / "y_aux.npy", mmap_mode="r")[:N]
    li_aux = np.asarray(aux[:, 9], np.float64)               # cons_li (scorer-derived, clamped 0.1..5, float32)
    both = okf & np.isfinite(li_aux)
    rep = {"split": split, "shots": nsh, "rows": N, "fails": fails, "ok_frac": float(okf.mean()),
           "c3_maxabs_vs_yC80": float(np.abs(c3 - yC).max()),
           "h_vs_aux_li_max_rel": float(np.max(np.abs(h[both] - li_aux[both]) / np.abs(li_aux[both]))) if both.any() else None,
           "aux_li_finite_but_not_ok": int((np.isfinite(li_aux) & ~okf).sum()),
           "ok_but_aux_li_nan": int((okf & ~np.isfinite(li_aux)).sum()),
           "h_mean": float(np.nanmean(h[okf])), "h_var": float(np.nanvar(h[okf])), "sec": time.time() - t0}
    json.dump(rep, open(o / "prep_report.json", "w"), indent=1)
    log(f"{split}: {json.dumps(rep)}")


def main():
    ctx = mp.get_context("spawn")
    with ctx.Pool(WORKERS, initializer=_init) as pool:
        for sp in SPLITS:
            run_split(sp, pool)


if __name__ == "__main__":
    sys.path.insert(0, str(WS / "starter"))
    main()
