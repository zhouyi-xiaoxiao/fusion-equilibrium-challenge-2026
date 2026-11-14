#!/usr/bin/env python3
"""
Stage-2 feature engineering for the Fusion Equilibrium Challenge.

Two feature sets:
  full_features_d3d  — DIII-D-specialized: 21 magnetics + coil lags (t-25, t-50 ms)
                       + Thomson profile summaries + physics proxies (~91 dims).
  agnostic_features  — machine-agnostic (works on DIII-D and MAST rows): no raw
                       coil identities; Ip/B/solenoid/shaping-sum/up-down asymmetry
                       + q*/betaN/stored-energy proxies + Thomson summaries + lags.
                       Meant to be per-machine StandardScaler'ed (scaler fit on that
                       machine's *inputs* only — legal, inputs are released for MAST).

Also: MAST shot loader (starter's load_shot_from_hf_row is DIII-D-only) and a
chunked (optionally pixel-weighted) FluxPCA.

Predict path reads ONLY input columns (magnetics_*, thomson_*); never efit_* /
magnetics_dsep. fix_d3d_ip_times is applied by the starter loader for DIII-D.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

from data_fixes import align_mast_thomson_core
from experiments import D3D_MAGNETICS_SIGNALS, _as_psirz_stack, load_shot_from_hf_row

MU0 = 4e-7 * np.pi

MAST_SIGNALS = [
    "efps_current", "p2l_current", "p2u_current", "p3l_current", "p3u_current",
    "p4l_current", "p4u_current", "p5l_current", "p5u_current", "p6l_current",
    "p6u_current", "plasma_current", "sol_current", "tf_current",
]


# ---------------------------------------------------------------- loaders

def load_d3d_shot(path) -> dict:
    row = pd.read_parquet(path).iloc[0]
    shot = load_shot_from_hf_row(row)  # applies fix_d3d_ip_times
    shot["machine"] = "d3d"
    return shot


def load_mast_shot(path) -> dict:
    row = pd.read_parquet(path).iloc[0]
    shot: dict = {"machine": "mast", "source": "MAST"}
    shot["efit_times"] = np.asarray(row["efit_times"], dtype=np.float64)
    if "efit_psirz" in row.index and row["efit_psirz"] is not None:
        try:
            shot["efit_psirz"] = _as_psirz_stack(row["efit_psirz"])
        except Exception:
            pass
    mt = np.asarray(row["magnetics_time"], dtype=np.float64)
    mag = {}
    for sig in MAST_SIGNALS:
        col = f"magnetics_{sig}"
        if col in row.index:
            mag[sig] = {"values": np.asarray(row[col], dtype=np.float64), "times": mt}
    shot["magnetics"] = mag
    try:  # MAST erratum: drop ghost Thomson core channel 0
        _R, Te, ne = align_mast_thomson_core(row)
        shot["thomson_core"] = {
            "times": np.asarray(row["thomson_core_times"], dtype=np.float64),
            "Te": Te.astype(np.float32), "ne": ne.astype(np.float32)}
    except Exception:
        pass
    try:
        shot["thomson_edge"] = {
            "times": np.asarray(row["thomson_edge_times"], dtype=np.float64),
            "Te": np.stack([np.asarray(p, dtype=np.float32) for p in row["thomson_edge_Te"]]),
            "ne": np.stack([np.asarray(p, dtype=np.float32) for p in row["thomson_edge_ne"]])}
    except Exception:
        pass
    scal = {}
    for name in ("efit_q95", "efit_beta_n"):
        if name in row.index and row[name] is not None:
            try:
                scal[name] = np.asarray(row[name], dtype=np.float64).ravel()
            except Exception:
                pass
    shot["scalars"] = scal
    return shot


# ---------------------------------------------------------------- helpers

def interp_sig(times, values, t_out, lag: float = 0.0) -> np.ndarray:
    """NaN-safe linear interpolation onto t_out - lag (ms)."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(v)
    if m.sum() < 2:
        return np.zeros(len(t_out), dtype=np.float64)
    f = interp1d(t[m], v[m], kind="linear", bounds_error=False, fill_value="extrapolate")
    out = f(np.asarray(t_out, dtype=np.float64) - lag)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _sig(shot, name, efit_times, lag=0.0):
    s = shot["magnetics"].get(name)
    if s is None:
        return np.zeros(len(efit_times), dtype=np.float64)
    return interp_sig(s["times"], s["values"], efit_times, lag)


def thomson_summary(shot, efit_times) -> np.ndarray:
    """(T, 20) fixed-length, chord-layout-independent Thomson summaries.
    Per system (core, edge): Te [mean,max,med], ne [mean,max,med],
    pe=Te*ne [mean,max,med], finite-fraction. Nearest-in-time profile,
    gap > 100 ms -> zeros."""
    T = len(efit_times)
    blocks = []
    for sys_key in ("thomson_core", "thomson_edge"):
        cols = np.zeros((T, 10), dtype=np.float64)
        sysd = shot.get(sys_key)
        if sysd is not None:
            tt = np.asarray(sysd["times"], dtype=np.float64)
            Te = np.asarray(sysd["Te"], dtype=np.float64)
            ne = np.asarray(sysd["ne"], dtype=np.float64)
            if Te.ndim == 1:
                Te = Te[:, None]
            if ne.ndim == 1:
                ne = ne[:, None]
            n = min(len(tt), len(Te), len(ne))
            if n >= 1:
                tt, Te, ne = tt[:n], Te[:n], ne[:n]
                pe = Te * ne
                idx = np.clip(np.searchsorted(tt, efit_times), 0, n - 1)
                idx_lo = np.clip(idx - 1, 0, n - 1)
                pick = np.where(
                    np.abs(tt[idx_lo] - efit_times) < np.abs(tt[idx] - efit_times),
                    idx_lo, idx)
                for ti in range(T):
                    j = pick[ti]
                    if abs(tt[j] - efit_times[ti]) > 100.0:
                        continue
                    row = []
                    for arr in (Te[j], ne[j], pe[j]):
                        fin = np.isfinite(arr)
                        if fin.any():
                            a = arr[fin]
                            row += [float(a.mean()), float(a.max()), float(np.median(a))]
                        else:
                            row += [0.0, 0.0, 0.0]
                    row.append(float(np.isfinite(Te[j]).mean()))
                    cols[ti] = row
        blocks.append(cols)
    return np.nan_to_num(np.hstack(blocks), nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- feature sets

def full_features_d3d(shot, thomson: np.ndarray | None = None) -> np.ndarray:
    """DIII-D-specialized features (T, ~91)."""
    t = shot["efit_times"]
    base = [_sig(shot, s, t) for s in D3D_MAGNETICS_SIGNALS]                 # 21
    lags = [_sig(shot, s, t, lag) for lag in (25.0, 50.0)
            for s in D3D_MAGNETICS_SIGNALS]                                  # 42
    th = thomson if thomson is not None else thomson_summary(shot, t)        # 20
    ip = base[D3D_MAGNETICS_SIGNALS.index("plasma_current")]
    bt = base[D3D_MAGNETICS_SIGNALS.index("bcoil")]
    up = sum(base[D3D_MAGNETICS_SIGNALS.index(f"F{i}A")] for i in range(1, 10))
    lo = sum(base[D3D_MAGNETICS_SIGNALS.index(f"F{i}B")] for i in range(1, 10))
    pe_core_mean = th[:, 6]  # pe mean of core block
    ip_MA = ip / 1e6
    proxies = [
        np.abs(bt) / (np.abs(ip_MA) + 1e-3),                    # q* proxy shape
        pe_core_mean,                                           # stored-energy proxy
        pe_core_mean * np.abs(bt) / (ip_MA ** 2 + 1e-3),        # betaN proxy shape
        up + lo, up - lo, np.abs(up) + np.abs(lo),              # shaping / asymmetry
        np.sign(ip),
    ]                                                            # 7
    X = np.column_stack(base + lags + [th] + proxies)
    return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def agnostic_features(shot, thomson: np.ndarray | None = None) -> np.ndarray:
    """Machine-agnostic features (T, 36). Same columns/meaning on both machines;
    intended for per-machine standardization."""
    t = shot["efit_times"]
    machine = shot["machine"]
    th = thomson if thomson is not None else thomson_summary(shot, t)        # 20
    if machine == "d3d":
        ip_A = _sig(shot, "plasma_current", t)          # amps
        bt = _sig(shot, "bcoil", t)
        sol = _sig(shot, "ECOILA", t)
        up = sum(_sig(shot, f"F{i}A", t) for i in range(1, 10))
        lo = sum(_sig(shot, f"F{i}B", t) for i in range(1, 10))
        ip_l25 = _sig(shot, "plasma_current", t, 25.0)
        ip_l50 = _sig(shot, "plasma_current", t, 50.0)
        bt_l25 = _sig(shot, "bcoil", t, 25.0)
        bt_l50 = _sig(shot, "bcoil", t, 50.0)
    else:
        ip_A = _sig(shot, "plasma_current", t) * 1e3    # kA -> A
        bt = _sig(shot, "tf_current", t)
        sol = _sig(shot, "sol_current", t)
        up = sum(_sig(shot, f"p{i}u_current", t) for i in range(2, 7))
        lo = sum(_sig(shot, f"p{i}l_current", t) for i in range(2, 7))
        ip_l25 = _sig(shot, "plasma_current", t, 25.0) * 1e3
        ip_l50 = _sig(shot, "plasma_current", t, 50.0) * 1e3
        bt_l25 = _sig(shot, "tf_current", t, 25.0)
        bt_l50 = _sig(shot, "tf_current", t, 50.0)
    ip_MA = ip_A / 1e6
    pe_core_mean = th[:, 6]
    cols = [
        np.abs(ip_A), np.sign(ip_A), bt, np.abs(bt), sol,
        up, lo, up - lo, np.abs(up) + np.abs(lo),
        np.abs(bt) / (np.abs(ip_MA) + 1e-3),                    # q* proxy
        pe_core_mean,                                           # W proxy
        pe_core_mean * np.abs(bt) / (ip_MA ** 2 + 1e-3),        # betaN proxy
        ip_l25, ip_l50, bt_l25, bt_l50,
    ]                                                            # 16
    X = np.column_stack(cols + [th])                             # 36
    return np.nan_to_num(X.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def psi_norm_scale(shot) -> np.ndarray:
    """(T,) physics amplitude scale mu0 * |Ip[A]| * R0 for psi normalization."""
    t = shot["efit_times"]
    if shot["machine"] == "d3d":
        ip_A = np.abs(_sig(shot, "plasma_current", t))
        R0 = 1.69  # DIII-D grid-center major radius (mask grid_R mean)
    else:
        ip_A = np.abs(_sig(shot, "plasma_current", t)) * 1e3
        R0 = 1.03  # MAST grid-center major radius
    return MU0 * np.maximum(ip_A, 5e4) * R0  # floor 50 kA avoids divide-blowup


# ---------------------------------------------------------------- flux PCA

class FluxPCA:
    """PCA over flattened 65x65 flux maps with chunked transforms and an
    optional per-pixel weight (weight applied before fit/transform; divided
    out on inverse_transform => L2 in coefficient space approximates a
    w^2-weighted pixel loss)."""

    def __init__(self, n_components: int = 80, weight: np.ndarray | None = None):
        self.n = n_components
        self.w = None if weight is None else weight.reshape(-1).astype(np.float32)

    def fit(self, Y: np.ndarray, max_fit_rows: int = 80000, seed: int = 0) -> "FluxPCA":
        F = Y.reshape(len(Y), -1)
        if len(F) > max_fit_rows:
            rng = np.random.default_rng(seed)
            F = F[rng.choice(len(F), max_fit_rows, replace=False)]
        F = F.astype(np.float32)
        if self.w is not None:
            F = F * self.w[None, :]
        p = PCA(n_components=self.n, svd_solver="randomized", random_state=seed)
        p.fit(F)
        self.mean_ = p.mean_.astype(np.float32)
        self.comps_ = p.components_.astype(np.float32)
        self.evr_ = p.explained_variance_ratio_
        return self

    def transform(self, Y: np.ndarray, chunk: int = 20000) -> np.ndarray:
        F = Y.reshape(len(Y), -1)
        out = np.empty((len(F), self.n), dtype=np.float32)
        for i in range(0, len(F), chunk):
            b = F[i:i + chunk].astype(np.float32)
            if self.w is not None:
                b = b * self.w[None, :]
            out[i:i + chunk] = (b - self.mean_[None, :]) @ self.comps_.T
        return out

    def inverse_transform(self, C: np.ndarray) -> np.ndarray:
        F = C.astype(np.float32) @ self.comps_ + self.mean_[None, :]
        if self.w is not None:
            F = F / self.w[None, :]
        return F.reshape(len(C), 65, 65)
