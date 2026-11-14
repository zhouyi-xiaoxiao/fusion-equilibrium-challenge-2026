#!/usr/bin/env python3
"""
Stage 11 / Ch2 step 1 — label-free, INPUT-derived plasma geometry from Thomson scattering.

Physics: Te and ne are (to a good approximation) flux functions, so along any Thomson chord
  * the Te/pe maximum sits where the chord is tangent to the innermost flux surface it crosses
    (on a horizontal chord through the axis height = the magnetic axis R_axis),
  * the ne/Te "foot" (fall to a small fraction of the peak) marks the plasma edge (LCFS crossing).
MAST: core system = HORIZONTAL MIDPLANE laser, per-channel R (0.235..1.48 m incl. the edge system)
      -> R_in, R_out (=> R_geo, a) and R_axis per frame. NO labels exist or are read.
D3D:  analogue chords with labels allowed (D3D labels may tune a physics pipeline applied to MAST):
      * tangential system (thomson_edge_*): horizontal chord at Z ~ -0.06, R 1.67..2.06
        -> R_axis estimators vs efit_r_axis (outer edge is beyond its coverage -> censored),
      * core system (thomson_core_*): VERTICAL chord at R = 1.94, Z 0.03..0.76 (Z from
        thomson_chord_Z of the TS_core_* names) -> Z_top(1.94) vs the GT LCFS crossing, and the
        Te-peak Z vs the GT psi extremum along R = 1.94.
Thresholds f (edge = fall to f * peak) are SELECTED ON D3D TRAIN SHOTS ONLY and reported on the
standard 150-shot D3D holdout; MAST gets the D3D-selected values + label-free robustness stats.

Never reads: mast_envelope.npz, any MAST efit_* column, the three labelled MAST demo shots.
Reads on MAST: efit_times (the frame time base, an input), thomson_*, magnetics_{plasma,tf}_current.

Usage (sbatch, workq): python thomson_geom.py --out $OUT --procs 64 [--n-d3d-train 400]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import multiprocessing as mp
import numpy as np
import pandas as pd

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "starter"))
from data_fixes import align_mast_thomson_core  # noqa: E402

DATA = Path(os.environ.get("DATA_FULL", os.environ["SCR"] + "/data_full"))
SCALEUP_TRAIN = WS / "scaleup_run/data/diii_d_train"
MAST_DEMO_IDS = ("28348", "28350", "28351")          # never present in data_full/mast; asserted anyway
F_LIST = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40)
GAP_MS = 10.0                                        # nearest Thomson profile within 10 ms of an EFIT time
MIN_VALID = 8
LIMITER_R_CC = 0.1952                                # public MAST limiter (FAIR-MAST level-2 wall), centre column
MU0 = 4e-7 * np.pi


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ------------------------------------------------------------------ 1-D profile geometry
def _runmed3(y):
    if len(y) < 3:
        return y.copy()
    p = np.pad(y, 1, mode="edge")
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def _smooth(y):
    m = _runmed3(y)
    if len(m) < 3:
        return m
    p = np.pad(m, 1, mode="edge")
    return (p[:-2] + p[1:-1] + p[2:]) / 3.0


def _peak_refine(x, y, i):
    """Quadratic fit through up to 5 points around index i (non-uniform x)."""
    lo, hi = max(0, i - 2), min(len(x), i + 3)
    if hi - lo < 3:
        return float(x[i])
    c = np.polyfit(x[lo:hi] - x[i], y[lo:hi], 2)
    if c[0] >= 0:
        return float(x[i])
    xm = -c[1] / (2 * c[0])
    return float(x[i] + np.clip(xm, x[lo] - x[i], x[hi - 1] - x[i]))


def _crossing(x, y, thr, i0, step):
    """Walk from index i0 in direction step while y >= thr; return (position of fall below thr by
    linear interpolation, censored flag). Censored = the run reaches the end of the coverage."""
    i = i0
    n = len(x)
    while 0 <= i + step < n and y[i + step] >= thr:
        i += step
    j = i + step
    if not (0 <= j < n):
        return float(x[i]), True
    y0, y1 = y[i], y[j]
    t = (y0 - thr) / (y0 - y1) if y0 != y1 else 0.5
    return float(x[i] + t * (x[j] - x[i])), False


def profile_geometry(x, Te, ne, f_list=F_LIST):
    """x: channel positions (any order). Returns dict or None.
    Keys: n, x_tepk, x_pepk, x_sym, cov_lo, cov_hi, te_max, ne_max,
          for each f: lo_ne_f, hi_ne_f, clo_ne_f, chi_ne_f (censor flags), lo_te_f, hi_te_f, ..."""
    x = np.asarray(x, float); Te = np.asarray(Te, float); ne = np.asarray(ne, float)
    ok = np.isfinite(x) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
    if ok.sum() < MIN_VALID:
        return None
    o = np.argsort(x[ok])
    xs, te, nn = x[ok][o], Te[ok][o], ne[ok][o]
    tes, nes = _smooth(te), _smooth(nn)
    pes = _smooth(te * nn)
    it, ie, ip = int(np.argmax(tes)), int(np.argmax(nes)), int(np.argmax(pes))
    out = {"n": int(ok.sum()), "cov_lo": float(xs[0]), "cov_hi": float(xs[-1]),
           "te_max": float(tes[it]), "ne_max": float(nes[ie]),
           "x_tepk": _peak_refine(xs, tes, it), "x_pepk": _peak_refine(xs, pes, ip)}
    # symmetric-level midpoints (Te levels 0.5..0.8 of max), extrapolated to the peak level
    mids, levs = [], []
    for L in (0.5, 0.6, 0.7, 0.8):
        a, ca = _crossing(xs, tes, L * tes[it], it, -1)
        b, cb = _crossing(xs, tes, L * tes[it], it, +1)
        if not (ca or cb):
            mids.append(0.5 * (a + b)); levs.append(L)
    if len(mids) >= 2:
        c = np.polyfit(levs, mids, 1)
        out["x_sym"] = float(np.polyval(c, 1.0))
    elif len(mids) == 1:
        out["x_sym"] = float(mids[0])
    else:
        out["x_sym"] = np.nan
    for f in f_list:
        for key, y, i0 in (("ne", nes, ie), ("te", tes, it)):
            a, ca = _crossing(xs, y, f * y[i0], i0, -1)
            b, cb = _crossing(xs, y, f * y[i0], i0, +1)
            out[f"lo_{key}_{f:g}"], out[f"clo_{key}_{f:g}"] = a, ca
            out[f"hi_{key}_{f:g}"], out[f"chi_{key}_{f:g}"] = b, cb
    return out


def thomson_valid(t):
    """Organiser Thomson validity contract (starter issue #12, 2026-10-11): keep sample i only if
    t_i > max(t_<i) (drops out-of-order / reset / constant-padded samples). Returns a bool mask."""
    t = np.asarray(t, float)
    if len(t) == 0:
        return np.zeros(0, bool)
    prev = np.concatenate([[-np.inf], np.maximum.accumulate(np.where(np.isfinite(t), t, -np.inf))[:-1]])
    return np.isfinite(t) & (t > prev)


def nearest_rows(t_src, t_out, gap=GAP_MS):
    """Index of the nearest VALID Thomson sample within `gap` ms of each output time (-1 if none).
    2026-10-11 fix: invalid samples are dropped (validity contract) instead of being sorted in."""
    t_src = np.asarray(t_src, float)
    t_out = np.asarray(t_out, float)
    o = np.where(thomson_valid(t_src))[0]              # strictly increasing by construction
    if len(o) == 0:
        return np.full(len(t_out), -1)
    ts = t_src[o]
    idx = np.clip(np.searchsorted(ts, t_out), 0, len(ts) - 1)
    lo = np.clip(idx - 1, 0, len(ts) - 1)
    pick = np.where(np.abs(ts[lo] - t_out) < np.abs(ts[idx] - t_out), lo, idx)
    res = o[pick]
    res[np.abs(ts[pick] - t_out) > gap] = -1
    return res


def _stack(col):
    return np.stack([np.asarray(p, dtype=np.float64) for p in col]) if len(col) else np.zeros((0, 0))


# ------------------------------------------------------------------ D3D GT helpers (labels allowed on D3D)
def lcfs_crossings_vertical(lr, lz, R0):
    """Z of the LCFS polygon crossings with the vertical line R = R0."""
    zs = []
    n = len(lr)
    for k in range(n):
        r1, z1, r2, z2 = lr[k], lz[k], lr[(k + 1) % n], lz[(k + 1) % n]
        if (r1 - R0) * (r2 - R0) < 0:
            t = (R0 - r1) / (r2 - r1)
            zs.append(z1 + t * (z2 - z1))
    return zs


def lcfs_crossings_horizontal(lr, lz, Z0):
    rs = []
    n = len(lr)
    for k in range(n):
        r1, z1, r2, z2 = lr[k], lz[k], lr[(k + 1) % n], lz[(k + 1) % n]
        if (z1 - Z0) * (z2 - Z0) < 0:
            t = (Z0 - z1) / (z2 - z1)
            rs.append(r1 + t * (r2 - r1))
    return rs


def d3d_shot(path):
    cols = ["efit_times", "efit_psirz", "efit_grid_R", "efit_grid_Z", "efit_r_axis", "efit_z_axis",
            "efit_lcfs_n", "efit_lcfs_r", "efit_lcfs_z", "thomson_core_times", "thomson_core_Te",
            "thomson_core_ne", "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne",
            "thomson_chord_name", "thomson_chord_R", "thomson_chord_Z"]
    row = pd.read_parquet(path, columns=cols).iloc[0]
    t = np.asarray(row["efit_times"], float)
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    cR = np.asarray(row["thomson_chord_R"], float); cZ = np.asarray(row["thomson_chord_Z"], float)
    core = np.array([n.startswith("TS_core") for n in names])
    tang = np.array([n.startswith("TS_tangential") for n in names])
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    psi = row["efit_psirz"]
    rax = np.asarray(row["efit_r_axis"], float).ravel(); zax = np.asarray(row["efit_z_axis"], float).ravel()
    lr_all, lz_all = row["efit_lcfs_r"], row["efit_lcfs_z"]
    ln = np.asarray(row["efit_lcfs_n"]).ravel() if row["efit_lcfs_n"] is not None else None
    recs = []
    # core (vertical) system
    cTe, cne = _stack(row["thomson_core_Te"]), _stack(row["thomson_core_ne"])
    zc = cZ[core]
    if cTe.shape[1] != len(zc):
        n = min(cTe.shape[1], len(zc)); cTe, cne, zc = cTe[:, :n], cne[:, :n], zc[:n]
    ic = nearest_rows(row["thomson_core_times"], t)
    eTe, ene = _stack(row["thomson_edge_Te"]), _stack(row["thomson_edge_ne"])
    rt = cR[tang]; zt = cZ[tang]
    if eTe.size and eTe.shape[1] != len(rt):
        n = min(eTe.shape[1], len(rt)); eTe, ene, rt, zt = eTe[:, :n], ene[:, :n], rt[:n], zt[:n]
    ie = nearest_rows(row["thomson_edge_times"], t) if eTe.size else np.full(len(t), -1)
    z_chord = float(np.nanmean(zt)) if len(zt) else -0.06
    i194 = None
    for k in range(len(t)):
        rec = {"file": Path(path).name, "k": k, "t": float(t[k])}
        # GT
        try:
            lr = np.asarray(lr_all[k], float); lz = np.asarray(lz_all[k], float)
            if ln is not None and np.isfinite(ln[k]):
                lr, lz = lr[:int(ln[k])], lz[:int(ln[k])]
            m = np.isfinite(lr) & np.isfinite(lz) & (lr > 0)
            lr, lz = lr[m], lz[m]
            zs = lcfs_crossings_vertical(lr, lz, 1.94)
            rec["gt_ztop194"] = max(zs) if zs else np.nan
            rec["gt_zbot194"] = min(zs) if zs else np.nan
            rs = lcfs_crossings_horizontal(lr, lz, z_chord)
            rec["gt_rout_tan"] = max(rs) if rs else np.nan
            rec["gt_rin_tan"] = min(rs) if rs else np.nan
            rec["gt_rgeo"] = 0.5 * (lr.max() + lr.min()); rec["gt_a"] = 0.5 * (lr.max() - lr.min())
        except Exception:
            pass
        rec["gt_raxis"] = float(rax[k]) if k < len(rax) else np.nan
        rec["gt_zaxis"] = float(zax[k]) if k < len(zax) else np.nan
        try:
            P = np.asarray(psi[k], float) if np.ndim(psi[k]) == 2 else np.stack([np.asarray(r, float) for r in psi[k]])
            if i194 is None:
                i194 = int(np.argmin(np.abs(gR - 1.94)))
            w = (1.94 - gR[i194 - 1]) / (gR[i194] - gR[i194 - 1]) if gR[i194] > 1.94 else None
            colv = P[:, i194] if w is None else (1 - w) * P[:, i194 - 1] + w * P[:, i194]
            zin = (gZ > -0.6) & (gZ < 0.9)
            jj = np.where(zin)[0]
            j = jj[int(np.argmin(colv[zin]))]                     # D3D EFIT axis = MIN of psi
            rec["gt_zext194"] = _peak_refine(gZ, -colv, j)
        except Exception:
            rec["gt_zext194"] = np.nan
        # Thomson
        if ic[k] >= 0:
            g = profile_geometry(zc, cTe[ic[k]], cne[ic[k]])
            if g:
                rec.update({f"core_{a}": b for a, b in g.items()})
        if ie[k] >= 0:
            g = profile_geometry(rt, eTe[ie[k]], ene[ie[k]])
            if g:
                rec.update({f"tan_{a}": b for a, b in g.items()})
        rec["z_chord_tan"] = z_chord
        recs.append(rec)
    return recs


def mast_shot(path):
    name = Path(path).name
    assert not any(d in name for d in MAST_DEMO_IDS)
    cols = ["efit_times", "thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R",
            "thomson_edge_times", "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_spatial",
            "magnetics_time", "magnetics_plasma_current", "magnetics_tf_current"]
    row = pd.read_parquet(path, columns=cols).iloc[0]
    assert not any(c.startswith("efit_") and c != "efit_times" for c in row.index)
    t = np.asarray(row["efit_times"], float)
    R, Te, ne = align_mast_thomson_core(row)
    tt = np.asarray(row["thomson_core_times"], float)
    eR = np.asarray(row["thomson_edge_spatial"], float)
    eTe, ene = _stack(row["thomson_edge_Te"]), _stack(row["thomson_edge_ne"])
    et = np.asarray(row["thomson_edge_times"], float)
    mt = np.asarray(row["magnetics_time"], float)
    ip = np.asarray(row["magnetics_plasma_current"], float); itf = np.asarray(row["magnetics_tf_current"], float)
    m = np.isfinite(mt) & np.isfinite(ip); ip_t = np.interp(t, mt[m], ip[m]) if m.sum() > 2 else np.full(len(t), np.nan)
    m = np.isfinite(mt) & np.isfinite(itf); tf_t = np.interp(t, mt[m], itf[m]) if m.sum() > 2 else np.full(len(t), np.nan)
    ic = nearest_rows(tt, t)
    ie = nearest_rows(et, t) if eTe.size else np.full(len(t), -1)
    recs = []
    for k in range(len(t)):
        rec = {"file": Path(path).name, "k": k, "t": float(t[k]), "ip_kA": float(ip_t[k]), "itf_kA": float(tf_t[k]),
               "dt_th": float(abs(tt[ic[k]] - t[k])) if ic[k] >= 0 else np.nan}
        if ic[k] >= 0:
            g = profile_geometry(R, Te[ic[k]], ne[ic[k]])
            if g:
                rec.update({f"core_{a}": b for a, b in g.items()})
            # core + edge merged (edge system extends the outboard coverage to 1.48 m)
            if ie[k] >= 0 and eTe.shape[1] == len(eR):
                x = np.concatenate([R, eR]); y1 = np.concatenate([Te[ic[k]], eTe[ie[k]]])
                y2 = np.concatenate([ne[ic[k]], ene[ie[k]]])
                g = profile_geometry(x, y1, y2)
                if g:
                    rec.update({f"mrg_{a}": b for a, b in g.items()})
        recs.append(rec)
    # raw time-base diagnostics (label-free)
    diag = {"file": name, "n_core_times": int(len(tt)), "core_t_min": float(np.nanmin(tt)) if len(tt) else np.nan,
            "core_t_max": float(np.nanmax(tt)) if len(tt) else np.nan,
            "core_dt_med": float(np.nanmedian(np.diff(np.sort(tt)))) if len(tt) > 1 else np.nan,
            "n_core_t_lt_1ms": int((tt < 1.0).sum()), "efit_t_min": float(t.min()), "efit_t_max": float(t.max()),
            "frac_efit_with_th": float((ic >= 0).mean())}
    return recs, diag


def _task(args):
    kind, path = args
    try:
        if kind == "d3d":
            return kind, d3d_shot(path), None
        recs, diag = mast_shot(path)
        return kind, recs, diag
    except Exception as e:  # noqa: BLE001
        return kind, [], {"file": Path(path).name, "error": f"{type(e).__name__}: {e}",
                          "tb": traceback.format_exc()[-800:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    ap.add_argument("--n-d3d-train", type=int, default=400)
    ap.add_argument("--max-mast", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    hold_names = sorted(p.name for p in SCALEUP_TRAIN.glob("*.parquet"))[1000:1150]
    assert len(hold_names) == 150, len(hold_names)
    all_d3d = sorted(p.name for p in (DATA / "diii_d_train").glob("*.parquet"))
    hs = set(hold_names)
    train_names = [n for n in all_d3d if n not in hs][: args.n_d3d_train]
    mast_files = sorted((DATA / "mast").glob("*.parquet"))
    if args.max_mast:
        mast_files = mast_files[: args.max_mast]
    tasks = ([("d3d", str(DATA / "diii_d_train" / n)) for n in train_names + hold_names]
             + [("mast", str(p)) for p in mast_files])
    log(f"{len(train_names)} D3D train (selection) + {len(hold_names)} D3D holdout + {len(mast_files)} MAST shots, "
        f"{args.procs} procs")
    d3d, mast, diags = [], [], []
    t0 = time.time()
    with mp.get_context("fork").Pool(args.procs) as pool:
        for i, (kind, recs, diag) in enumerate(pool.imap_unordered(_task, tasks, chunksize=2)):
            (d3d if kind == "d3d" else mast).extend(recs)
            if diag:
                diags.append(diag)
            if (i + 1) % 200 == 0:
                log(f"  {i + 1}/{len(tasks)} ({time.time() - t0:.0f}s)")
    D = pd.DataFrame(d3d); M = pd.DataFrame(mast)
    D["split"] = np.where(D["file"].isin(hs), "hold", "train")
    D.to_parquet(out / "d3d_thomson_geom.parquet"); M.to_parquet(out / "mast_thomson_geom.parquet")
    pd.DataFrame(diags).to_csv(out / "mast_time_diag.csv", index=False)
    log(f"wrote {len(D)} D3D frames, {len(M)} MAST frames, {len(diags)} diag rows in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
