#!/usr/bin/env python3
"""
G1 data layer: guarded shot loaders and Thomson-derived per-frame inputs.

MAST rows are read ONLY through rule3_guard.read_mast_parquet (input-column allow-list; the guard also refuses
to open the scorer envelope, demo files and equilibrium stores). D3D rows may carry labels; they are returned
separately (`labels`) and never enter a solver input.

Thomson validity contract (organiser issue #12): per system a sample is used only if t_i > max(t_<i)
(thomson_geom.thomson_valid) BEFORE any resampling; D3D zeros and MAST NaNs mean "no measurement".
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path(os.environ["WS"])
for p in (WS / "starter", WS / "stage11_run/ch2_geometry"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rule3_guard  # noqa: E402
import thomson_geom as TG  # noqa: E402
from data_fixes import align_mast_thomson_core, fix_d3d_ip_times  # noqa: E402

from g1_core import MU0, N_TF, current_columns  # noqa: E402

E_CH = 1.602176634e-19
GEOM_KEYS = ("input_column", "R", "Z", "width", "height", "angle1", "angle2")
MAST_COLS = (["efit_times", "efit_grid_R", "efit_grid_Z", "magnetics_time", "magnetics_plasma_current",
              "magnetics_tf_current"] + [f"magnetics_{c}" for c in current_columns("mast")]
             + ["thomson_core_times", "thomson_core_Te", "thomson_core_ne", "thomson_core_R", "thomson_edge_times",
                "thomson_edge_Te", "thomson_edge_ne", "thomson_edge_spatial", "thomson_chord_Z"]
             + [f"coil_{k}" for k in GEOM_KEYS])
LABEL_COLS = ("efit_psirz", "efit_q95", "efit_beta_n", "efit_r_axis", "efit_z_axis", "efit_li")


def interp(t, v, tout):
    t = np.asarray(t, float); v = np.asarray(v, float)
    n = min(len(t), len(v))
    t, v = t[:n], v[:n]
    m = np.isfinite(t) & np.isfinite(v)
    if m.sum() < 2:
        return np.zeros(len(tout))
    return np.interp(np.asarray(tout, float), t[m], v[m], left=v[m][0], right=v[m][-1])


def load_row(path, machine):
    if machine == "mast":
        return rule3_guard.read_mast_parquet(path, MAST_COLS).iloc[0]
    return pd.read_parquet(path).iloc[0]


def geom_of(row):
    g = {k: np.asarray(row[f"coil_{k}"]) for k in GEOM_KEYS}
    h = hashlib.sha1()
    for k in GEOM_KEYS:
        a = g[k]
        h.update(("|".join(map(str, a)) if a.dtype.kind in "OUS" else np.ascontiguousarray(a, dtype=np.float64).tobytes().hex()).encode())
    return g, h.hexdigest()[:16]


def labels_of(row):
    out = {}
    if "efit_psirz" not in row.index or row["efit_psirz"] is None:
        return out
    try:
        psi = np.asarray(row["efit_psirz"], dtype=np.float32)
        if psi.ndim != 3:
            raise ValueError
    except (ValueError, TypeError):
        psi = np.stack([np.stack([np.asarray(r, dtype=np.float32) for r in g]) for g in row["efit_psirz"]])
    out["psi"] = psi
    try:                                    # GT LCFS extents (diagnostics only)
        ln = np.asarray(row["efit_lcfs_n"]).ravel()
        ext = np.full((len(psi), 8), np.nan)
        for k in range(len(psi)):
            lr = np.asarray(row["efit_lcfs_r"][k], float); lz = np.asarray(row["efit_lcfs_z"][k], float)
            n = int(ln[k]) if np.isfinite(ln[k]) else len(lr)
            lr, lz = lr[:n], lz[:n]
            m = np.isfinite(lr) & np.isfinite(lz) & (lr > 0)
            if m.sum() > 8:
                lr, lz = lr[m], lz[m]
                ext[k, :6] = [lr.min(), lr.max(), lz.max(), lz.min(), lr[np.argmin(lz)], lr[np.argmax(lz)]]
                zz = np.append(lz, lz[0]); rr = np.append(lr, lr[0])
                cr = [rr[i] + (0.0 - zz[i]) * (rr[i + 1] - rr[i]) / (zz[i + 1] - zz[i]) for i in range(len(lr))
                      if (zz[i] <= 0.0 < zz[i + 1]) or (zz[i + 1] <= 0.0 < zz[i])]
                if len(cr) >= 2:
                    ext[k, 6:8] = [min(cr), max(cr)]          # LCFS crossings of the midplane Z = 0
        out["lcfs_ext"] = ext
    except Exception:
        pass
    for name in LABEL_COLS[1:]:
        if name in row.index and row[name] is not None:
            try:
                out[name] = np.asarray(row[name], dtype=np.float64).ravel()
            except Exception:
                pass
    return out


def frame_inputs(row, machine, t):
    """Shipped-unit inputs resampled onto the frame times: ip_A (signed), cur {col: A}, fvac = mu0 N I_TF / 2pi."""
    mt = np.asarray(row["magnetics_time"], float)
    ip = np.asarray(row["magnetics_plasma_current"], float)
    tip = np.asarray(fix_d3d_ip_times(row), float) if machine == "d3d" else mt
    ip_kA = interp(tip, ip, t)
    cur = {}
    for c in current_columns(machine):
        col = f"magnetics_{c}"
        cur[c] = interp(mt, row[col], t) * 1e3 if col in row.index else np.zeros(len(t))
    tf_kA = interp(mt, row["magnetics_bcoil" if machine == "d3d" else "magnetics_tf_current"], t)
    return {"ip_A": ip_kA * 1e3, "cur": cur, "tf_kA": tf_kA, "fvac": MU0 * N_TF[machine] * tf_kA * 1e3 / (2 * np.pi)}


# ----------------------------------------------------------------------------------------------- Thomson
def _stack(col):
    try:
        if col is None or len(col) == 0:
            return np.zeros((0, 0))
        return np.stack([np.asarray(p, dtype=np.float64) for p in col])
    except Exception:
        return np.zeros((0, 0))


def _pe_peak(x, Te, ne):
    m = np.isfinite(x) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
    if m.sum() < 3:
        return np.nan
    o = np.argsort(x[m])
    return float(TG._runmed3(E_CH * Te[m][o] * ne[m][o]).max())


def _pe_series(times, x, Te, ne, t, max_gap=50.0):
    """Peak electron pressure per VALID Thomson sample -> frame times (linear between valid samples; NaN if the
    nearest valid sample is further than max_gap ms)."""
    times = np.asarray(times, float)
    n = min(len(times), Te.shape[0]) if Te.ndim == 2 else 0
    out = np.full(len(t), np.nan)
    if n == 0 or Te.shape[1] != len(x):
        return out
    v = TG.thomson_valid(times[:n])
    tv = times[:n][v]
    pv = np.array([_pe_peak(x, Te[i], ne[i]) for i in np.nonzero(v)[0]])
    ok = np.isfinite(pv)
    if ok.sum() == 0:
        return out
    tv, pv = tv[ok], pv[ok]
    out = np.interp(t, tv, pv)
    idx = np.clip(np.searchsorted(tv, t), 0, len(tv) - 1)
    lo = np.clip(idx - 1, 0, len(tv) - 1)
    d = np.minimum(np.abs(tv[idx] - t), np.abs(tv[lo] - t))
    out[d > max_gap] = np.nan
    return out


def thomson_frames(row, machine, t, cfg):
    """Per frame: pe_max [Pa], edge (MAST: R_out on the midplane laser; D3D: Z_top on the vertical core chord),
    cens (True = the foot is beyond the coverage or no profile), chord = ("h", Z_laser) | ("v", R_chord)."""
    T = len(t)
    f, gap = cfg["f_edge"], cfg["th_gap_ms"]
    out = {"pe_max": np.full(T, np.nan), "edge": np.full(T, np.nan), "cens": np.ones(T, bool),
           "rin": np.full(T, np.nan), "cin": np.ones(T, bool), "rsym": np.full(T, np.nan), "chord": None,
           "covlo": np.full(T, np.nan), "covhi": np.full(T, np.nan), "has": np.zeros(T, bool)}
    out["prof"] = [None] * T            # per frame: (R_pts, Z_pts, Te, ne) of the nearest valid sample(s)
    kh, kc = f"hi_ne_{f:g}", f"chi_ne_{f:g}"
    kl, kcl = f"lo_ne_{f:g}", f"clo_ne_{f:g}"
    try:
        if machine == "mast":
            zl = 0.0
            try:
                zz = np.asarray(row["thomson_chord_Z"], float)
                if zz.size and np.isfinite(zz).any() and abs(np.nanmedian(zz)) < 0.2:
                    zl = float(np.nanmedian(zz))
            except Exception:
                pass
            out["chord"] = ("h", zl)
            R, Te, ne = align_mast_thomson_core(row)
            Te = np.asarray(Te, float); ne = np.asarray(ne, float)
            tt = np.asarray(row["thomson_core_times"], float)
            out["pe_max"] = _pe_series(tt, R, Te, ne, t)
            ic = TG.nearest_rows(tt[:Te.shape[0]], t, gap)
            eR = np.asarray(row["thomson_edge_spatial"], float)
            eTe, ene = _stack(row["thomson_edge_Te"]), _stack(row["thomson_edge_ne"])
            et = np.asarray(row["thomson_edge_times"], float)
            use_e = eTe.ndim == 2 and eTe.size and eTe.shape[1] == len(eR)
            ie = TG.nearest_rows(et[:eTe.shape[0]], t, gap) if use_e else np.full(T, -1)
            for k in range(T):
                if ic[k] < 0:
                    continue
                if ie[k] >= 0:
                    xr, xt, xn = np.concatenate([R, eR]), np.concatenate([Te[ic[k]], eTe[ie[k]]]), np.concatenate([ne[ic[k]], ene[ie[k]]])
                else:
                    xr, xt, xn = R, Te[ic[k]], ne[ic[k]]
                g = TG.profile_geometry(xr, xt, xn, f_list=(f,))
                out["prof"][k] = (xr, np.full(len(xr), zl), xt, xn)
                if g:
                    out["edge"][k], out["cens"][k] = g[kh], g[kc]
                    out["rin"][k], out["cin"][k], out["rsym"][k] = g[kl], g[kcl], g["x_sym"]
                    out["covlo"][k], out["covhi"][k], out["has"][k] = g["cov_lo"], g["cov_hi"], True
        else:
            names = np.asarray(row["thomson_chord_name"]).astype(str)
            cR = np.asarray(row["thomson_chord_R"], float); cZ = np.asarray(row["thomson_chord_Z"], float)
            core = np.array([n.startswith("TS_core") for n in names])
            tang = np.array([n.startswith("TS_tangential") for n in names])
            zc = cZ[core]
            out["chord"] = ("v", float(np.nanmedian(cR[core])) if core.any() else 1.94)
            cTe, cne = _stack(row["thomson_core_Te"]), _stack(row["thomson_core_ne"])
            if cTe.ndim == 2 and cTe.size:
                n = min(cTe.shape[1], len(zc)); cTe, cne, zc = cTe[:, :n], cne[:, :n], zc[:n]
                tt = np.asarray(row["thomson_core_times"], float)
                pe_c = _pe_series(tt, zc, cTe, cne, t)
                ic = TG.nearest_rows(tt[:cTe.shape[0]], t, gap)
                rcore = float(np.nanmedian(cR[core]))
                for k in range(T):
                    if ic[k] >= 0:
                        out["prof"][k] = (np.full(len(zc), rcore), zc, cTe[ic[k]], cne[ic[k]])
                        g = TG.profile_geometry(zc, cTe[ic[k]], cne[ic[k]], f_list=(f,))
                        if g:
                            out["edge"][k], out["cens"][k] = g[kh], g[kc]
                            out["covhi"][k], out["has"][k] = g["cov_hi"], True
            else:
                pe_c = np.full(T, np.nan)
            eTe, ene = _stack(row["thomson_edge_Te"]), _stack(row["thomson_edge_ne"])
            rt = cR[tang]
            if eTe.ndim == 2 and eTe.size and len(rt):
                n = min(eTe.shape[1], len(rt)); eTe, ene, rt = eTe[:, :n], ene[:, :n], rt[:n]
                pe_t = _pe_series(np.asarray(row["thomson_edge_times"], float), rt, eTe, ene, t)
                it_ = TG.nearest_rows(np.asarray(row["thomson_edge_times"], float)[:eTe.shape[0]], t, gap)
                ztan = cZ[tang][:len(rt)]
                for k in range(T):
                    if it_[k] >= 0:
                        add = (rt, ztan, eTe[it_[k]], ene[it_[k]])
                        out["prof"][k] = add if out["prof"][k] is None else tuple(np.concatenate([a, b]) for a, b in zip(out["prof"][k], add))
            else:
                pe_t = np.full(T, np.nan)
            with np.errstate(all="ignore"):
                out["pe_max"] = np.fmax(pe_c, pe_t)
    except Exception as e:  # noqa: BLE001
        out["error"] = f"{type(e).__name__}: {e}"
    return out
