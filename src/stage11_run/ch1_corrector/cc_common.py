#!/usr/bin/env python3
"""Stage 11 / ch1_corrector — shared paths, official-scorer wrappers, emulated composite.

All derivations go through the UNMODIFIED starter/fusion_scoring code (same calls as
local_score.build_reference / score_shot). File names are prefixed cc_ so that nothing here
shadows the scorer's own modules (common.py, derive.py, ...).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
SCR = Path(os.environ["SCR"])
OUT = Path(os.environ.get("CC_OUT", str(SCR / "stage11_ch1/corrector")))
FEAT = Path(os.environ.get("CC_FEAT", str(SCR / "stage11_ch1/feat_v21")))
HC = SCR / "stage6_H"
S10 = SCR / "stage10_ch1_cons"
MET = SCR / "stage11_ch1/metric"
STARTER = WS / "starter"

MACHINE = "DIII-D"
N_ITER = 22
N_POINTS = 512
K = 80
NAMES7 = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
EXT4 = ["Rmin", "Rmax", "Zmin", "Zmax"]
NAMES11 = NAMES7 + EXT4
AUX_COLS = list(range(3, 14))        # y_aux: cons_* (3..9) + lcfs_Rmin,Rmax,Zmin,Zmax (10..13)
W_PSI, W_QB, W_LCFS, W_CONS = 0.55, 0.15, 0.10, 0.20
_G: dict = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def init_scorer():
    if "R" in _G:
        return _G
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    p = str(STARTER / "fusion_scoring")
    if p not in sys.path:
        sys.path.insert(0, p)
    from common import AXIS_SIGN
    from contour import symmetric_hausdorff
    from derive import derive_frame, internal_inductance
    from lcfs import extract_lcfs, extract_lcfs_with_sign, major_radius
    m = np.load(STARTER / "fusion_scoring/masks/d3d_envelope.npz")
    _G.update(R=m["grid_R"].astype(np.float64), Z=m["grid_Z"].astype(np.float64),
              MC=m["mask_coarse"].astype(bool), MF=m["mask_coarse"].astype(bool).astype(np.float64),
              AX=AXIS_SIGN[MACHINE] * 1.0, haus=symmetric_hausdorff, derive=derive_frame,
              li=internal_inductance, ex=extract_lcfs, exs=extract_lcfs_with_sign, rmaj=major_radius)
    return _G


def pca80():
    pc = np.load(HC / "pca_80.npz")
    return pc["mean_"].astype(np.float64), pc["comps_"].astype(np.float64)


def ext4(C):
    if C is None:
        return np.full(4, np.nan)
    return np.array([C[:, 0].min(), C[:, 0].max(), C[:, 1].min(), C[:, 1].max()])


def dpred(psi):
    """Official derivation of a PREDICTED map (mirrors local_score.score_shot) -> (f11, contour)."""
    g = _G
    C, _ = g["exs"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], n_iter=N_ITER, axis_sign=g["AX"])
    v = g["derive"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], contour=C, with_li=True, axis_sign=g["AX"])
    return np.concatenate([[v[n] for n in NAMES7], ext4(C)]), C


def dpred_raw(psi):
    """dpred + a copy whose li is the UNCLAMPED functional when the scorer's plausibility clamp (0.1..5) nulled it.
    The raw copy is used for Jacobians / target residuals only; scoring always uses the official value."""
    f, C = dpred(psi)
    fr = f
    if C is not None and not np.isfinite(f[6]):
        fr = f.copy()
        fr[6] = li_of(psi, C)
    return f, fr, C


def dref(psi):
    """Official derivation of a GROUND-TRUTH map (mirrors local_score.build_reference)."""
    g = _G
    C = g["ex"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], n_points=N_POINTS)
    v = g["derive"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], contour=C)
    return np.concatenate([[v[n] for n in NAMES7], ext4(C)]), C


def li_of(psi, C):
    g = _G
    if C is None:
        return np.nan
    return g["li"](psi, g["R"], g["Z"], MACHINE, C)


def haus(a, b):
    if a is None or b is None:
        return np.nan
    return float(_G["haus"](a, np.asarray(b, dtype=np.float64)))


def load_pred_npz(path, offsets):
    """Harness-layout npz -> (psi (N,65,65) float64, q95 (N,), betaN (N,))."""
    z = np.load(path, allow_pickle=False)
    n = len(offsets) - 1
    psi = np.concatenate([z[f"shot_{i:04d}_psirz"] for i in range(n)]).astype(np.float64)
    q = np.concatenate([z[f"shot_{i:04d}_q95"] for i in range(n)]).astype(np.float64)
    b = np.concatenate([z[f"shot_{i:04d}_betaN"] for i in range(n)]).astype(np.float64)
    assert psi.shape[0] == offsets[-1], (psi.shape, offsets[-1])
    return psi, q, b


def write_pred_npz(path, psi, q, b, offsets, shots=None):
    """Write harness layout; `shots` = iterable of shot indices to include (re-indexed from 0)."""
    d = {}
    shots = range(len(offsets) - 1) if shots is None else shots
    for j, s in enumerate(shots):
        p, e = offsets[s], offsets[s + 1]
        d[f"shot_{j:04d}_psirz"] = np.asarray(psi[p:e], dtype=np.float32).reshape(-1, 65, 65)
        d[f"shot_{j:04d}_q95"] = np.asarray(q[p:e], dtype=np.float32)
        d[f"shot_{j:04d}_betaN"] = np.asarray(b[p:e], dtype=np.float32)
    np.savez(path, **d)


def pooled_r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m].astype(np.float64), p[m].astype(np.float64)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def emul_score(ref, f7, hd, ssres, q=None, b=None, shots=None):
    """Composite S of the official harness from cached per-frame pieces.

    ref: hold_ref.npz dict (f_gt (N,11), rgeo (N,), gsum, gsq (N,), offsets, q95, betaN);
    f7 (N,>=7) scalars derived from the predicted maps (nan = failed); hd (N,) Hausdorff [m]
    (nan = predicted extraction failed); ssres (N,) per-frame flux SS_res.
    """
    off = ref["offsets"]
    S = len(off) - 1
    shots = np.arange(S) if shots is None else np.asarray(shots)
    fm = np.zeros(off[-1], bool)
    for s in shots:
        fm[off[s]:off[s + 1]] = True
    fg = ref["f_gt"][:, :7]
    n_pix = 4225.0 * fm.sum()
    tot_psi = ref["gsq"][fm].sum() - ref["gsum"][fm].sum() ** 2 / n_pix
    r2psi = 1.0 - ssres[fm].sum() / tot_psi
    r2c = np.full(7, np.nan)
    for j in range(7):
        m = fm & np.isfinite(fg[:, j])
        g = fg[m, j]
        mu = g.mean()
        p = np.where(np.isfinite(f7[m, j]), f7[m, j], mu)
        r2c[j] = 1.0 - ((g - p) ** 2).sum() / ((g - mu) ** 2).sum()
    cons = float(np.mean(np.maximum(0.0, r2c)))
    ds = []
    for s in shots:
        sl = slice(off[s], off[s + 1])
        has = np.isfinite(ref["rgeo"][sl])
        if not has.any():
            continue
        rg = ref["rgeo"][sl][has].mean()
        d = np.where(np.isfinite(hd[sl][has]), np.minimum(1.0, hd[sl][has] / rg), 1.0)
        ds.append(d.mean())
    dl = float(np.mean(ds))
    out = {"r2psi": float(r2psi), "dlcfs": dl, "cons": cons, **{n: float(r2c[j]) for j, n in enumerate(NAMES7)}}
    if q is not None:
        r2q, r2b = pooled_r2(ref["q95"][fm], q[fm]), pooled_r2(ref["betaN"][fm], b[fm])
        out["r2qb"] = 0.5 * (r2q + r2b)
        out["S"] = W_PSI * out["r2psi"] + W_QB * out["r2qb"] + W_LCFS * (1 - min(1.0, dl)) + W_CONS * cons
    return out


def fmt_score(tag, sc):
    return (f"{tag:28s} S {sc.get('S', float('nan')):.5f} psi {sc['r2psi']:.6f} 1-D {1 - sc['dlcfs']:.5f} Cons {sc['cons']:.5f} | "
            + " ".join(f"{sc[n]:.4f}" for n in NAMES7))
