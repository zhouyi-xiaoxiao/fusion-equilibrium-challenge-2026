#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — step 1: finite-difference Jacobians of the scorer's seven psi-derived scalars
with respect to the PCA coefficients (stage6_H bases pca_80 / pca_120), on TRAIN frames only.

For each sampled train frame r and each basis K in {80,120}:
  base map   psi0 = mean + c_true[r] @ comps            (c_true = feat_v2/train/y_C{K}.npy)
  f0         = official derivation of psi0 (prediction path of local_score.score_shot:
               extract_lcfs_with_sign(n_iter=22, axis_sign=-1) -> derive_frame(contour=..., with_li=True))
  for each step config g (see CONFIGS) and coefficient k:
      J[g,:,k]  = (f(psi0 + d_gk comp_k) - f(psi0 - d_gk comp_k)) / (2 d_gk)        central secant
      S2[g,:,k] =  f(+) + f(-) - 2 f0                                                noise/curvature probe
      DH[g,k]   = mean_{+,-} Hausdorff(LCFS(+-), LCFS(psi0)) / d_gk / Rgeo            boundary sensitivity
  validation points (never used to build M):
      incumbent OUT-OF-FOLD error on this train frame (stage10 coef_oof_p80 - y_C80; mapped into the
      120-basis by projection) -> f(psi0 + e) ; NV synthetic errors e ~ N(0, diag(rms_k^2)) * scale
  first-order LCFS displacement operator at psi0 (optional D_LCFS surrogate):
      n(theta) = (dpsi(x_theta) - dpsi(x_X)) / |grad psi0|(x_theta)   ->  H = <B^T B>_theta / Rgeo^2

Step sizes ("delta matched to realistic model error"), from the incumbents' HOLDOUT coefficient errors
(K=80: bag20 = mean of stage6_H coef_p80_{h512x256x128,h1024x512x256}_s42..51; K=120: bag10 of
coef_p120_h1024x512x256_s42..51; targets feat_v2/holdout/y_C{K}.npy):
  A1    d_k = rms_k                    per-coefficient RMS error (literal spec)
  B0125 d_k = 0.125 * rho
  B025  d_k = 0.25 * rho               rho = sqrt(E|e|^2) total RMS error norm, same for every k
  B05   d_k = 0.5 * rho
  B10   d_k = 1.0 * rho
  B20   d_k = 2.0 * rho
Only step sizes come from the holdout; every map that is differentiated is a TRAIN frame.

Env: WS SCR OUT SHARD NSHARD NFRAMES(21600) WORKERS(70) SEED(20261011) MAXF(0) NV(4)
Output: $OUT/fd/shard_{SHARD:02d}.npz (+ deltas_{SHARD:02d}.json)
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
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/metric")))
FEAT = SCR / "stage11_ch1/feat_v2"
HC = SCR / "stage6_H"
S10 = SCR / "stage10_ch1_cons"
STARTER = WS / "starter"
sys.path.insert(0, str(STARTER / "fusion_scoring"))

SHARD = int(os.environ.get("SHARD", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
NSHARD = int(os.environ.get("NSHARD", "8"))
NFRAMES = int(os.environ.get("NFRAMES", "21600"))
WORKERS = int(os.environ.get("WORKERS", "70"))
SEED = int(os.environ.get("SEED", "20261011"))
MAXF = int(os.environ.get("MAXF", "0"))
NV = int(os.environ.get("NV", "4"))
V_SCALES = [1.0, 1.0, 0.7, 0.7][:NV] + [1.0] * max(0, NV - 4)

MACHINE = "DIII-D"
N_ITER = 22
N_POINTS = 512
KS = (80, 120)
CONFIG_NAMES = ["A1", "B0125", "B025", "B05", "B10", "B20"]
B_MULT = {"B0125": 0.125, "B025": 0.25, "B05": 0.5, "B10": 1.0, "B20": 2.0}
SEEDS = range(42, 52)

_G = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def incumbent_deltas():
    """Per-coefficient RMS error + total RMS norm of the incumbents on the standard holdout."""
    out = {}
    y80 = np.load(FEAT / "holdout/y_C80.npy").astype(np.float64)
    y120 = np.load(FEAT / "holdout/y_C120.npy").astype(np.float64)
    b512 = np.mean([np.load(HC / f"coef_p80_h512x256x128_s{s}.npy") for s in SEEDS], 0)
    b1024 = np.mean([np.load(HC / f"coef_p80_h1024x512x256_s{s}.npy") for s in SEEDS], 0)
    e80 = 0.5 * (b512 + b1024).astype(np.float64) - y80
    e120 = np.mean([np.load(HC / f"coef_p120_h1024x512x256_s{s}.npy") for s in SEEDS], 0).astype(np.float64) - y120
    for K, e, src in ((80, e80, "p80_bag20 (h512 bag10 + h1024 bag10, seeds 42..51)"),
                      (120, e120, "p120_h1024x512x256 bag10 (seeds 42..51)")):
        rms = np.sqrt((e ** 2).mean(0))
        rho = float(np.sqrt((e ** 2).sum(1).mean()))
        cfg = {"A1": rms.tolist()}
        for n, m in B_MULT.items():
            cfg[n] = [m * rho] * K
        out[K] = {"source": src, "split": "holdout sorted(scaleup_run/data/diii_d_train)[1000:1150]",
                  "n_frames": int(len(e)), "rms": rms.tolist(), "rho": rho, "configs": cfg}
    return out


def _init():
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    from common import AXIS_SIGN, CONS_SCALARS
    from contour import symmetric_hausdorff
    from derive import derive_frame
    from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius
    m = np.load(STARTER / "fusion_scoring/masks/d3d_envelope.npz")
    _G.update(R=m["grid_R"].astype(np.float64), Z=m["grid_Z"].astype(np.float64),
              MC=m["mask_coarse"].astype(bool), MF=m["mask_coarse"].astype(bool).astype(np.float64),
              AX=AXIS_SIGN[MACHINE] * 1.0, names=list(CONS_SCALARS), haus=symmetric_hausdorff,
              derive=derive_frame, ex=extract_lcfs, exs=extract_lcfs_with_sign, rmaj=major_radius)
    _G["PSI"] = np.load(S10 / "psi_train_f16.npy", mmap_mode="r")
    _G["OOF"] = np.load(S10 / "coef_oof_p80_h512x256x128.npy", mmap_mode="r")
    _G["Y"] = {K: np.load(FEAT / f"train/y_C{K}.npy", mmap_mode="r") for K in KS}
    pcs = {K: np.load(HC / f"pca_{K}.npz") for K in KS}
    _G["mean"] = {K: pcs[K]["mean_"].astype(np.float64) for K in KS}
    _G["comps"] = {K: pcs[K]["comps_"].astype(np.float64) for K in KS}
    _G["T80to120"] = _G["comps"][80] @ _G["comps"][120].T          # (80,120): e120 = e80 @ T
    d = incumbent_deltas()
    _G["rms"] = {K: np.asarray(d[K]["rms"]) for K in KS}
    _G["cfg"] = {K: [np.asarray(d[K]["configs"][n]) for n in CONFIG_NAMES] for K in KS}


def dpred(psi):
    """Official derivation of a PREDICTED map (mirrors local_score.score_shot)."""
    g = _G
    ex, _ = g["exs"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], n_iter=N_ITER, axis_sign=g["AX"])
    v = g["derive"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], contour=ex, with_li=True,
                    axis_sign=g["AX"])
    return np.array([v[n] for n in g["names"]], dtype=np.float64), ex


def dref(psi):
    """Official derivation of a GROUND-TRUTH map (mirrors local_score.build_reference)."""
    g = _G
    c = g["ex"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], n_points=N_POINTS)
    v = g["derive"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], contour=c)
    return np.array([v[n] for n in g["names"]], dtype=np.float64), c


def haus(a, b):
    if a is None or b is None:
        return np.nan
    return _G["haus"](a, np.asarray(b, dtype=np.float64))


def lcfs_operator(psi0, C0, comps):
    """First-order normal displacement of the separatrix per unit coefficient: (K, 512) metres."""
    R, Z = _G["R"], _G["Z"]
    gZ, gR = np.gradient(psi0, Z, R)
    gm = np.sqrt(gR ** 2 + gZ ** 2)
    nZ, nR = psi0.shape
    fr = np.interp(C0[:, 0], R, np.arange(nR)); fz = np.interp(C0[:, 1], Z, np.arange(nZ))
    r0 = np.clip(np.floor(fr).astype(int), 0, nR - 2); z0 = np.clip(np.floor(fz).astype(int), 0, nZ - 2)
    tr = fr - r0; tz = fz - z0
    w = [(1 - tr) * (1 - tz), tr * (1 - tz), (1 - tr) * tz, tr * tz]

    def samp(f):  # f (..., nZ, nR) -> (..., 512)
        return (f[..., z0, r0] * w[0] + f[..., z0, r0 + 1] * w[1]
                + f[..., z0 + 1, r0] * w[2] + f[..., z0 + 1, r0 + 1] * w[3])
    g_on = samp(gm)
    ix = int(np.argmin(g_on))
    cm = samp(comps.reshape(-1, nZ, nR))                      # (K,512)
    floor = np.maximum(g_on, 0.25 * np.median(g_on))
    return (cm - cm[:, ix:ix + 1]) / floor[None, :]


def work(i):
    g = _G
    r = int(g["rows"][i])
    t0 = time.time()
    try:
        out = {"i": i, "row": r, "ok": True}
        psi_gt = np.asarray(g["PSI"][r], dtype=np.float64)
        f_gt16, c_gt = dref(psi_gt)
        rgeo_gt = g["rmaj"](c_gt) if c_gt is not None else np.nan
        out["f_gt16"] = f_gt16
        out["ss_gt"] = np.array([psi_gt.sum(), (psi_gt ** 2).sum(), psi_gt.size])
        e80 = np.asarray(g["OOF"][r], np.float64) - np.asarray(g["Y"][80][r], np.float64)
        n_eval = 1
        for K in KS:
            comps, mean = g["comps"][K], g["mean"][K]
            c0 = np.asarray(g["Y"][K][r], dtype=np.float64)
            psi0 = (mean + c0 @ comps).reshape(65, 65)
            f0, C0 = dpred(psi0); n_eval += 1
            rgeo = g["rmaj"](C0) if C0 is not None else np.nan
            G = len(CONFIG_NAMES)
            J = np.full((G, 7, K), np.nan, np.float32)
            S2 = np.full((G, 7, K), np.nan, np.float32)
            DH = np.full((G, K), np.nan, np.float32)
            for gi, dl in enumerate(g["cfg"][K]):
                for k in range(K):
                    pk = comps[k].reshape(65, 65)
                    fp, Cp = dpred(psi0 + dl[k] * pk)
                    fm, Cm = dpred(psi0 - dl[k] * pk)
                    J[gi, :, k] = (fp - fm) / (2.0 * dl[k])
                    S2[gi, :, k] = fp + fm - 2.0 * f0
                    DH[gi, k] = 0.5 * (haus(Cp, C0) + haus(Cm, C0)) / dl[k] / rgeo
                    n_eval += 2
            e = e80 if K == 80 else e80 @ g["T80to120"]
            f_oof, C_oof = dpred(psi0 + (e @ comps).reshape(65, 65)); n_eval += 1
            rng = np.random.default_rng([SEED, r, K])
            ev = np.stack([rng.standard_normal(K) * g["rms"][K] * s for s in V_SCALES])
            fv = np.full((NV, 7), np.nan); dv = np.full(NV, np.nan)
            Cv = []
            for v in range(NV):
                fv[v], c_ = dpred(psi0 + (ev[v] @ comps).reshape(65, 65)); n_eval += 1
                dv[v] = haus(c_, C0) / rgeo
            lc = np.full((1 + NV, 2), np.nan)                # [oof, v...] x [mean-square, max] (normalised)
            H = np.zeros((K, K), np.float32); hok = 0
            if C0 is not None:
                B = lcfs_operator(psi0, C0, comps)
                H = (B @ B.T / B.shape[1] / rgeo ** 2).astype(np.float32); hok = 1
                for q, ee in enumerate([e] + list(ev)):
                    disp = ee @ B
                    lc[q] = [np.mean(disp ** 2) / rgeo ** 2, np.max(np.abs(disp)) / rgeo]
            out[K] = dict(f0=f0, J=J, S2=S2, DH=DH, e_oof=e.astype(np.float32), f_oof=f_oof,
                          d_oof=haus(C_oof, C0) / rgeo, d_oof_gt=haus(C_oof, c_gt) / rgeo_gt,
                          d_trunc=haus(C0, c_gt) / rgeo_gt, rgeo=rgeo, ev=ev.astype(np.float32), fv=fv, dv=dv,
                          lc=lc, H=H, hok=hok, ss_e=float(e @ e))
        out["n_eval"] = n_eval
        out["sec"] = time.time() - t0
        return out
    except Exception as ex:  # noqa: BLE001
        return {"i": i, "row": r, "ok": False, "err": f"{type(ex).__name__}: {ex}",
                "tb": traceback.format_exc()[-1500:]}


def sample_rows(n_train):
    """Frame-uniform, shot-spread systematic sample with jitter: one row per stratum of N/n rows."""
    rng = np.random.default_rng(SEED)
    u = rng.random(NFRAMES)
    rows = np.minimum(np.floor((np.arange(NFRAMES) + u) * n_train / NFRAMES).astype(np.int64), n_train - 1)
    return rows


def main():
    t0 = time.time()
    (OUT / "fd").mkdir(parents=True, exist_ok=True)
    n_train = np.load(FEAT / "train/y_C80.npy", mmap_mode="r").shape[0]
    assert np.load(S10 / "psi_train_f16.npy", mmap_mode="r").shape[0] == n_train
    assert np.load(S10 / "coef_oof_p80_h512x256x128.npy", mmap_mode="r").shape[0] == n_train
    rows_all = sample_rows(n_train)
    sel = np.arange(NFRAMES)[SHARD::NSHARD]
    if MAXF:
        sel = sel[:MAXF]
    d = incumbent_deltas()
    json.dump({"configs": CONFIG_NAMES, "seed": SEED, "nframes": NFRAMES, "nshard": NSHARD,
               **{f"K{K}": d[K] for K in KS}}, open(OUT / "fd" / f"deltas_{SHARD:02d}.json", "w"))
    shot_id = np.load(FEAT / "train/shot_id.npy", mmap_mode="r")
    log(f"shard {SHARD}/{NSHARD}: {len(sel)} frames of {NFRAMES} sampled from {n_train} train rows; "
        f"workers {WORKERS}; rho80={d[80]['rho']:.5f} rho120={d[120]['rho']:.5f}; configs {CONFIG_NAMES}")
    _G["rows"] = rows_all
    ctx = mp.get_context("fork")
    res = []
    with ctx.Pool(WORKERS, initializer=_init) as pool:
        for n, o in enumerate(pool.imap_unordered(work, [int(i) for i in sel], chunksize=1), 1):
            res.append(o)
            if not o["ok"]:
                log(f"FAILED frame i={o['i']} row={o['row']}: {o['err']}\n{o.get('tb','')}")
            if n % 100 == 0 or n == len(sel):
                secs = [x["sec"] for x in res if x["ok"]]
                log(f"{n}/{len(sel)} frames, {time.time()-t0:.0f}s wall, median {np.median(secs):.1f}s/frame "
                    f"({1e3*np.sum(secs)/max(1,sum(x['n_eval'] for x in res if x['ok'])):.2f} ms/eval)")
    ok = sorted([o for o in res if o["ok"]], key=lambda o: o["i"])
    if not ok:
        raise SystemExit("no successful frames")
    sv = {"i": np.array([o["i"] for o in ok]), "row": np.array([o["row"] for o in ok]),
          "shot": np.asarray(shot_id)[np.array([o["row"] for o in ok])],
          "f_gt16": np.stack([o["f_gt16"] for o in ok]), "ss_gt": np.stack([o["ss_gt"] for o in ok]),
          "sec": np.array([o["sec"] for o in ok]), "n_fail": np.array(len(res) - len(ok))}
    for K in KS:
        for key in ("f0", "J", "S2", "DH", "e_oof", "f_oof", "ev", "fv", "dv", "lc"):
            sv[f"{key}_{K}"] = np.stack([o[K][key] for o in ok])
        for key in ("d_oof", "d_oof_gt", "d_trunc", "rgeo", "hok", "ss_e"):
            sv[f"{key}_{K}"] = np.array([o[K][key] for o in ok])
        sv[f"Hsum_{K}"] = np.sum([o[K]["H"].astype(np.float64) for o in ok if o[K]["hok"]], axis=0)
    np.savez(OUT / "fd" / f"shard_{SHARD:02d}.npz", **sv)
    tot_eval = sum(o["n_eval"] for o in ok)
    log(f"DONE shard {SHARD}: {len(ok)} ok / {len(res)-len(ok)} failed; {tot_eval} derivations; "
        f"wall {time.time()-t0:.0f}s; cpu-core-seconds {sum(o['sec'] for o in ok):.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
