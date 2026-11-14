#!/usr/bin/env python3
"""
Stage 11 / Ch1 FEATURE-V2 for DIII-D (inputs only) — per-shot feature builder.

Reads ONLY input columns (magnetics_* except magnetics_dsep*, thomson_*, coil_*, efit_times,
efit_grid_R/Z, source). Targets / EFIT-derived columns (efit_* other than times/grid,
magnetics_dsep) are never touched here; prep_v2.py reads targets in a separate call for
train/holdout only.

Blocks (column order is fixed; names from `feature_names()`):
  v1    (90)  bit-compatible with stage2_features.full_features_d3d on the starter loader
              (incl. its Thomson bug: the starter's `np.asarray(row['thomson_core_Te'], float32)`
              raises on parquet object arrays -> Thomson silently absent -> the 20 Thomson-summary
              columns and 2 pe proxies are ALL ZERO in every v1 cache).  Kept identical so the
              first 90 standardised columns equal stage6_H Xfs.npy / L Xt.npy.
  th1   (20)  the same thomson_summary() but on correctly loaded Thomson (np.stack).
  mag  (294)  21 currents (ECOILA, bcoil, F1A..F9B, Ip[fixed time axis]) after a centred 1-ms
              boxcar: value at t, differences I(t+o)-I(t) for o in OFFSETS_MS\{0}, and central
              dI/dt at h=1 ms and h=5 ms (kA[/turn] per ms).
  ipn   (20)  F1A..F9B, ECOILA, bcoil divided by max(|Ip(t)|, 100 kA).
  time  (14)  t, frame fraction, time since breakdown / to end, Ip fraction of shot max,
              flat-top / ramp-up / ramp-down flags + distances, Ip max, sign(Ip), sign(bcoil).
  tsc  (156)  CORE (vertical, R=1.94 m) Thomson on a fixed Z grid ZGRID (39 pts, 0..0.95 m):
              Te [keV], ne [1e19 m^-3], pe [kPa], mask.  Profiles are time-interpolated per
              channel to the EFIT frame (bracketing profiles within GAP_CORE_MS, else the
              nearest one within the gap, else masked); a channel sample is valid iff finite
              and Te>0 and ne>0 (0 is the invalid flag in this dataset); channel Z comes from
              thomson_chord_Z of the TS_core_* chords (per shot); grid points farther than
              MAXD_CORE from a valid channel are masked (NaN raw -> 0 after standardising).
  tscm  (32)  core-profile moments: valid fraction, nearest-profile |dt|, top valid Z,
              core refs (median over Z<0.25) of Te/ne/pe, maxima, Z where Te/ne/pe first
              drop below 10/20/50 % of the ref (scan upward, two-consecutive rule), Z where Te
              drops below 100/300 eV, chord integrals, line-averaged ne, <pe>, centroids,
              peaking factors, li proxy (integral of (Te/Te_ref)^1.5).
  tse   (56)  EDGE/tangential Thomson (Z~-0.05) on a fixed R grid RGRID (14 pts 1.65..2.30 m),
              same four quantities, chord R from thomson_chord_R of TS_tangential_* chords.
  tsem   (8)  edge moments: valid fraction, nearest |dt|, mean Te/ne/pe, dTe/dR, dne/dR, max valid R.
  phys   (3)  betaN proxy <pe>/(|bcoil||Ip|), W proxy <pe>*Z_ne20, q-like |bcoil|*Z_ne20/|Ip|.
Raw features may contain NaN only in tsc/tscm/tse/tsem/phys (masked / undefined); prep_v2.py
standardises with NaN-aware train statistics and maps NaN -> 0 (= train mean).
"""
from __future__ import annotations

import numpy as np

D3D_SIGNALS = [
    "ECOILA", "bcoil",
    "F1A", "F1B", "F2A", "F2B", "F3A", "F3B",
    "F4A", "F4B", "F5A", "F5B", "F6A", "F6B",
    "F7A", "F7B", "F8A", "F8B", "F9A", "F9B",
    "plasma_current",
]  # == starter experiments.D3D_MAGNETICS_SIGNALS (asserted in prep)
FCOILS = [s for s in D3D_SIGNALS if s.startswith("F")]
OFFSETS_MS = [-200, -100, -50, -20, -10, -5, -2, 0, 2, 5, 10, 20]
SMOOTH_MS = 1.0
GAP_CORE_MS = 30.0
GAP_EDGE_MS = 40.0
ZGRID = np.round(np.arange(0.0, 0.951, 0.025), 3)          # 39
RGRID = np.round(np.arange(1.65, 2.301, 0.05), 3)          # 14
MAXD_CORE = 0.04
MAXD_EDGE = 0.05
TE_CLIP_EV = 15000.0
NE_CLIP = 2.0e20
IP_FLOOR_KA = 100.0
ALLOWED_EFIT = {"efit_times", "efit_grid_R", "efit_grid_Z"}

# thomson_summary order: per system Te[mean,max,med], ne[...], pe[...], finite-frac
V1_TH_NAMES = []
for _sysn in ("core", "edge"):
    for _q in ("Te", "ne", "pe"):
        for _st in ("mean", "max", "med"):
            V1_TH_NAMES.append(f"{_sysn}_{_q}_{_st}")
    V1_TH_NAMES.append(f"{_sysn}_finite_frac")

TSCM_NAMES = [
    "valid_frac", "dt_near_ms", "z_top_valid",
    "Te_ref_keV", "ne_ref_1e19", "pe_ref_kPa", "Te_max_keV", "ne_max_1e19", "pe_max_kPa",
    "z_Te10", "z_Te20", "z_Te50", "z_ne10", "z_ne20", "z_ne50", "z_pe10", "z_pe20", "z_pe50",
    "z_Te100eV", "z_Te300eV",
    "int_ne", "int_Te", "int_pe", "lineavg_ne", "avg_pe",
    "zc_ne", "zc_pe", "peak_Te", "peak_ne", "peak_pe", "li_proxy", "n_valid",
]
TSEM_NAMES = ["valid_frac", "dt_near_ms", "Te_mean_keV", "ne_mean_1e19", "pe_mean_kPa",
              "dTe_dR", "dne_dR", "r_max_valid"]
TIME_NAMES = ["t_ms", "frame_frac", "t_since_bd_ms", "t_to_end_ms", "ip_frac", "flat_flag",
              "rampup_flag", "rampdown_flag", "t_since_flat_start_ms", "t_to_flat_end_ms",
              "ip_max_kA", "sign_ip_t", "sign_bcoil_t", "sign_ip_shot"]
PHYS_NAMES = ["betaN_proxy", "W_proxy", "qlike_proxy"]


def v1_names():
    from experiments import D3D_MAGNETICS_SIGNALS as S
    n = [f"v1_{s}" for s in S]
    n += [f"v1_{s}_lag{lag}" for lag in (25, 50) for s in S]
    n += [f"v1_th_{x}" for x in V1_TH_NAMES]
    n += ["v1_qstar", "v1_pe_core_mean", "v1_betaN_proxy", "v1_up_plus_lo", "v1_up_minus_lo",
          "v1_abs_up_plus_abs_lo", "v1_sign_ip"]
    return n


def feature_names():
    """Ordered list of (name, block)."""
    out = [(n, "v1") for n in v1_names()]
    out += [(f"th1_{x}", "th1") for x in V1_TH_NAMES]
    for s in D3D_SIGNALS:
        out.append((f"mag_{s}_t0", "mag"))
        for o in OFFSETS_MS:
            if o != 0:
                out.append((f"mag_{s}_d{o:+d}ms", "mag"))
        out.append((f"mag_{s}_dIdt_1ms", "mag"))
        out.append((f"mag_{s}_dIdt_5ms", "mag"))
    out += [(f"ipn_{s}_over_Ip", "ipn") for s in FCOILS + ["ECOILA", "bcoil"]]
    out += [(f"time_{x}", "time") for x in TIME_NAMES]
    for q in ("Te", "ne", "pe", "mask"):
        out += [(f"tsc_{q}_z{z:.3f}", "tsc") for z in ZGRID]
    out += [(f"tscm_{x}", "tscm") for x in TSCM_NAMES]
    for q in ("Te", "ne", "pe", "mask"):
        out += [(f"tse_{q}_r{r:.3f}", "tse") for r in RGRID]
    out += [(f"tsem_{x}", "tsem") for x in TSEM_NAMES]
    out += [(f"phys_{x}", "phys") for x in PHYS_NAMES]
    return out


# ------------------------------------------------------------------ reading (inputs only)

def input_columns(path):
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    keep = [c for c in names
            if not (c.startswith("efit_") and c not in ALLOWED_EFIT)
            and not c.startswith("magnetics_dsep")]
    return keep


def read_inputs(path):
    import pandas as pd
    cols = input_columns(path)
    assert not any((c.startswith("efit_") and c not in ALLOWED_EFIT) or c.startswith("magnetics_dsep")
                   for c in cols)
    return pd.read_parquet(path, columns=cols).iloc[0]


def shot_v1_from_row(row):
    """Bug-compatible copy of starter experiments.load_shot_from_hf_row minus targets
    (identical to L prep_test.load_d3d_inputs)."""
    from data_fixes import fix_d3d_ip_times
    from experiments import D3D_MAGNETICS_SIGNALS
    shot = {"source": row.get("source", "DIII-D"), "machine": "d3d",
            "efit_times": np.asarray(row["efit_times"], dtype=np.float64)}
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = fix_d3d_ip_times(row)
    mag = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            continue
        mag[sig] = {"values": np.asarray(row[col], dtype=np.float32),
                    "times": ip_times if sig == "plasma_current" else shared}
    shot["magnetics"] = mag
    for sk in ("thomson_core", "thomson_edge"):
        if f"{sk}_times" in row:
            try:   # starter behaviour: raises on parquet object arrays -> Thomson absent (v1 bug)
                shot[sk] = {"times": np.asarray(row[f"{sk}_times"], dtype=np.float64),
                            "Te": np.asarray(row[f"{sk}_Te"], dtype=np.float32),
                            "ne": np.asarray(row[f"{sk}_ne"], dtype=np.float32)}
            except Exception:
                pass
    shot["scalars"] = {}
    return shot


def _stack(obj):
    try:
        a = np.asarray(obj, dtype=np.float64)
        if a.ndim == 2:
            return a
    except (ValueError, TypeError):
        pass
    if len(obj) == 0:
        return np.zeros((0, 0))
    return np.stack([np.asarray(p, dtype=np.float64) for p in obj])


def thomson_systems(row):
    """Correctly loaded Thomson: {'core': (t, Te, ne, Z per channel), 'edge': (t, Te, ne, R per channel)}
    plus layout diagnostics."""
    names = np.array([str(x) for x in row["thomson_chord_name"]]) if "thomson_chord_name" in row else np.array([])
    cR = np.asarray(row["thomson_chord_R"], dtype=np.float64) if "thomson_chord_R" in row else np.zeros(0)
    cZ = np.asarray(row["thomson_chord_Z"], dtype=np.float64) if "thomson_chord_Z" in row else np.zeros(0)
    out, diag = {}, {}
    for sk, prefix, coord in (("core", "TS_core", cZ), ("edge", "TS_tangential", cR)):
        key = f"thomson_{sk}"
        if f"{key}_times" not in row:
            diag[sk] = "missing"
            continue
        try:
            tt = np.asarray(row[f"{key}_times"], dtype=np.float64)
            Te = _stack(row[f"{key}_Te"])
            ne = _stack(row[f"{key}_ne"])
        except Exception as e:  # noqa: BLE001
            diag[sk] = f"load_error:{type(e).__name__}"
            continue
        n = min(len(tt), len(Te), len(ne))
        tt, Te, ne = tt[:n], Te[:n], ne[:n]
        sel = np.array([nm.startswith(prefix) for nm in names], dtype=bool) if len(names) else np.zeros(0, bool)
        C = Te.shape[1] if Te.ndim == 2 else 0
        if C == 0 or n == 0:
            diag[sk] = "empty"
            continue
        if sel.sum() != C or Te.shape != ne.shape:
            diag[sk] = f"chord_mismatch:{int(sel.sum())}vs{C}"
            continue
        pos = coord[sel]
        if not np.all(np.diff(tt) >= 0):
            o = np.argsort(tt, kind="stable")
            tt, Te, ne = tt[o], Te[o], ne[o]
        out[sk] = (tt, Te, ne, pos)
        diag[sk] = f"ok:{C}"
    return out, diag


# ------------------------------------------------------------------ helpers

def _smooth_fn(times, values, win_ms=SMOOTH_MS):
    """Return (ts, vs): centred boxcar of width ~win_ms on a strictly increasing time base."""
    t = np.asarray(times, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    n = min(len(t), len(v))
    t, v = t[:n], v[:n]
    m = np.isfinite(t) & np.isfinite(v)
    t, v = t[m], v[m]
    if len(t) < 2:
        return None
    if not np.all(np.diff(t) > 0):
        o = np.argsort(t, kind="stable")
        t, v = t[o], v[o]
        keep = np.r_[True, np.diff(t) > 0]
        t, v = t[keep], v[keep]
        if len(t) < 2:
            return None
    dt = float(np.median(np.diff(t)))
    w = max(1, int(round(win_ms / max(dt, 1e-9))))
    if w > 1 and len(t) > w:
        cs = np.cumsum(np.r_[0.0, v])
        vs = (cs[w:] - cs[:-w]) / w
        ts = 0.5 * (t[w - 1:] + t[:len(t) - w + 1])
        return ts, vs
    return t, v


def _ev(fn, tq):
    if fn is None:
        return np.zeros(len(tq))
    return np.interp(tq, fn[0], fn[1])


def _tinterp(tt, A, valid, tq, gap):
    """Per-channel time interpolation of profiles A (N,C) onto tq (T,).  Returns (out (T,C) NaN where
    invalid, dnear (T,) distance to the nearest profile in ms)."""
    N, C = A.shape
    T = len(tq)
    if N == 0:
        return np.full((T, C), np.nan), np.full(T, np.inf)
    i = np.searchsorted(tt, tq)
    lo = np.clip(i - 1, 0, N - 1)
    hi = np.clip(i, 0, N - 1)
    dlo = tq - tt[lo]
    dhi = tt[hi] - tq
    okl = (dlo >= 0) & (dlo <= gap)
    okh = (dhi >= 0) & (dhi <= gap)
    vl = valid[lo] & okl[:, None]
    vh = valid[hi] & okh[:, None]
    den = dlo + dhi
    w = np.where((den > 0) & okl & okh, dlo / np.where(den > 0, den, 1.0), 0.0)[:, None]
    Al, Ah = A[lo], A[hi]
    out = np.full((T, C), np.nan)
    both = vl & vh
    out = np.where(both, (1.0 - w) * Al + w * Ah, out)
    out = np.where(vl & ~vh, Al, out)
    out = np.where(vh & ~vl, Ah, out)
    dl = np.where(dlo >= 0, dlo, np.inf)
    dh = np.where(dhi >= 0, dhi, np.inf)
    return out, np.minimum(dl, dh)


def _grid(pos_sorted, V, grid, maxd):
    """V (T,C) channel values (NaN invalid), pos sorted ascending -> (T,G) on grid, NaN where farther
    than maxd from a valid channel."""
    T = V.shape[0]
    G = len(grid)
    out = np.full((T, G), np.nan)
    for k in range(T):
        m = np.isfinite(V[k])
        if not m.any():
            continue
        z = pos_sorted[m]
        v = V[k, m]
        g = np.interp(grid, z, v)            # constant beyond ends (masked by distance below)
        j = np.searchsorted(z, grid)
        d = np.minimum(np.abs(grid - z[np.clip(j - 1, 0, len(z) - 1)]),
                       np.abs(z[np.clip(j, 0, len(z) - 1)] - grid))
        g[d > maxd] = np.nan
        out[k] = g
    return out


def _drop_pos(z, v, thr):
    """First position (scanning upward in z) where v drops below thr on two consecutive valid
    channels; linear interpolation from the previous channel. z ascending, v finite."""
    n = len(z)
    if n == 0 or not np.isfinite(thr):
        return np.nan
    below = v < thr
    cond = below & np.r_[below[1:], True]
    if not cond.any():
        return float(z[-1])
    k = int(np.argmax(cond))
    if k == 0:
        return float(z[0])
    v0, v1 = v[k - 1], v[k]
    z0, z1 = z[k - 1], z[k]
    if v0 > v1:
        f = np.clip((v0 - thr) / (v0 - v1), 0.0, 1.0)
        return float(z0 + f * (z1 - z0))
    return float(z1)


def _trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


# ------------------------------------------------------------------ main builder

def features_v1(row):
    from stage2_features import full_features_d3d, thomson_summary
    shot = shot_v1_from_row(row)
    t = shot["efit_times"]
    th = thomson_summary(shot, t)
    return full_features_d3d(shot, thomson=th)


def features_v2(row, return_diag=False):
    """Full feature-v2 row block (T, D) float32 = [v1 | th1 | mag | ipn | time | tsc | tscm | tse | tsem | phys]."""
    from data_fixes import fix_d3d_ip_times
    from stage2_features import thomson_summary
    t = np.asarray(row["efit_times"], dtype=np.float64)
    T = len(t)
    diag = {}
    X1 = features_v1(row)                                           # (T, 90)

    # --- th1: thomson_summary on correctly loaded Thomson
    shot_ok = {"efit_times": t}
    for sk in ("thomson_core", "thomson_edge"):
        if f"{sk}_times" in row:
            try:
                shot_ok[sk] = {"times": np.asarray(row[f"{sk}_times"], dtype=np.float64),
                               "Te": _stack(row[f"{sk}_Te"]).astype(np.float32),
                               "ne": _stack(row[f"{sk}_ne"]).astype(np.float32)}
            except Exception:
                pass
    TH1 = thomson_summary(shot_ok, t)                              # (T, 20)

    # --- magnetics
    mag_t = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = np.asarray(fix_d3d_ip_times(row), dtype=np.float64)
    fns = {}
    for s in D3D_SIGNALS:
        col = f"magnetics_{s}"
        if col not in row:
            fns[s] = None
            continue
        tb = ip_times if s == "plasma_current" else mag_t
        fns[s] = _smooth_fn(tb, np.asarray(row[col], dtype=np.float64))
    diag["missing_signals"] = [s for s in D3D_SIGNALS if fns[s] is None]
    MAG = []
    vals0 = {}
    for s in D3D_SIGNALS:
        f = fns[s]
        v0 = _ev(f, t)
        vals0[s] = v0
        MAG.append(v0)
        for o in OFFSETS_MS:
            if o != 0:
                MAG.append(_ev(f, t + o) - v0)
        MAG.append((_ev(f, t + 1.0) - _ev(f, t - 1.0)) / 2.0)
        MAG.append((_ev(f, t + 5.0) - _ev(f, t - 5.0)) / 10.0)
    MAG = np.column_stack(MAG)

    ip = vals0["plasma_current"]
    ipa = np.maximum(np.abs(ip), IP_FLOOR_KA)
    IPN = np.column_stack([vals0[s] / ipa for s in FCOILS + ["ECOILA", "bcoil"]])

    # --- time / Ip-phase block (whole-shot Ip trace on the fixed axis)
    ipv = np.abs(np.asarray(row["magnetics_plasma_current"], dtype=np.float64))
    n_ip = min(len(ipv), len(ip_times))
    ipv, ipt = ipv[:n_ip], ip_times[:n_ip]
    fin = np.isfinite(ipv) & np.isfinite(ipt)
    ipv, ipt = ipv[fin], ipt[fin]
    if len(ipv) > 10:
        ipmax = float(np.nanpercentile(ipv, 99.5))
        on = ipv > 0.10 * ipmax
        t_bd = float(ipt[np.argmax(on)]) if on.any() else float(t[0])
        t_end = float(ipt[len(on) - 1 - np.argmax(on[::-1])]) if on.any() else float(t[-1])
        fl = ipv >= 0.9 * ipmax
        t_fs = float(ipt[np.argmax(fl)]) if fl.any() else t_bd
        t_fe = float(ipt[len(fl) - 1 - np.argmax(fl[::-1])]) if fl.any() else t_end
        sgn_shot = float(np.sign(np.median(np.asarray(row["magnetics_plasma_current"], dtype=np.float64)[:n_ip][fin][on]))) if on.any() else 0.0
    else:
        ipmax, t_bd, t_end, t_fs, t_fe, sgn_shot = 1.0, float(t[0]), float(t[-1]), float(t[0]), float(t[-1]), 0.0
    ipmax = max(ipmax, 1.0)
    f_ip = fns["plasma_current"]
    ip_frac = np.abs(ip) / ipmax
    slope50 = np.abs(_ev(f_ip, t + 50.0) - _ev(f_ip, t - 50.0)) / ipmax
    flat = ((ip_frac > 0.85) & (slope50 < 0.05)).astype(np.float64)
    span = max(float(t[-1] - t[0]), 1e-6) if T > 1 else 1.0
    TIME = np.column_stack([
        t, (t - t[0]) / span, t - t_bd, t_end - t, ip_frac, flat,
        (t < t_fs).astype(np.float64), (t > t_fe).astype(np.float64), t - t_fs, t_fe - t,
        np.full(T, ipmax), np.sign(ip), np.sign(vals0["bcoil"]), np.full(T, sgn_shot)])

    # --- Thomson profiles
    systems, tdiag = thomson_systems(row)
    diag["thomson"] = tdiag
    G1, G2 = len(ZGRID), len(RGRID)
    TSC = np.full((T, 4 * G1), np.nan)
    TSCM = np.full((T, len(TSCM_NAMES)), np.nan)
    TSE = np.full((T, 4 * G2), np.nan)
    TSEM = np.full((T, len(TSEM_NAMES)), np.nan)
    TSC[:, 3 * G1:] = 0.0
    TSE[:, 3 * G2:] = 0.0
    TSCM[:, 0] = 0.0
    TSCM[:, -1] = 0.0
    TSEM[:, 0] = 0.0
    zne20 = np.full(T, np.nan)
    avg_pe = np.full(T, np.nan)
    if "core" in systems:
        tt, Te, ne, pos = systems["core"]
        valid = np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
        Te = np.clip(np.nan_to_num(Te), 0, TE_CLIP_EV) / 1e3        # keV
        ne = np.clip(np.nan_to_num(ne), 0, NE_CLIP) / 1e19          # 1e19 m^-3
        Tei, dnear = _tinterp(tt, Te, valid, t, GAP_CORE_MS)
        nei, _ = _tinterp(tt, ne, valid, t, GAP_CORE_MS)
        o = np.argsort(pos, kind="stable")
        z = pos[o]
        Tei, nei = Tei[:, o], nei[:, o]
        pei = 1.602 * Tei * nei                                     # kPa
        gT = _grid(z, Tei, ZGRID, MAXD_CORE)
        gn = _grid(z, nei, ZGRID, MAXD_CORE)
        gp = _grid(z, pei, ZGRID, MAXD_CORE)
        gm = np.isfinite(gT).astype(np.float64)
        TSC = np.hstack([gT, gn, gp, gm])
        C = len(z)
        for k in range(T):
            m = np.isfinite(Tei[k])
            nv = int(m.sum())
            row_m = np.full(len(TSCM_NAMES), np.nan)
            row_m[0] = nv / C
            row_m[1] = min(float(dnear[k]), 1000.0)
            row_m[-1] = nv
            if nv >= 2:
                zz, Tv, nv_, pv = z[m], Tei[k, m], nei[k, m], pei[k, m]
                core = zz < 0.25
                refT = float(np.median(Tv[core])) if core.any() else float(np.max(Tv[:3]))
                refn = float(np.median(nv_[core])) if core.any() else float(np.max(nv_[:3]))
                refp = float(np.median(pv[core])) if core.any() else float(np.max(pv[:3]))
                row_m[2] = zz[-1]
                row_m[3:9] = [refT, refn, refp, Tv.max(), nv_.max(), pv.max()]
                c = 9
                for vv, ref in ((Tv, refT), (nv_, refn), (pv, refp)):
                    for fr in (0.1, 0.2, 0.5):
                        row_m[c] = _drop_pos(zz, vv, fr * ref)
                        c += 1
                row_m[18] = _drop_pos(zz, Tv, 0.1)
                row_m[19] = _drop_pos(zz, Tv, 0.3)
                gTk = np.nan_to_num(gT[k]); gnk = np.nan_to_num(gn[k]); gpk = np.nan_to_num(gp[k])
                iT, i_n, ip_ = _trapz(gTk, ZGRID), _trapz(gnk, ZGRID), _trapz(gpk, ZGRID)
                L = max(row_m[13], 0.05)                       # z_ne20 extent
                row_m[20:25] = [i_n, iT, ip_, i_n / L, ip_ / L]
                row_m[25] = _trapz(ZGRID * gnk, ZGRID) / i_n if i_n > 0 else np.nan
                row_m[26] = _trapz(ZGRID * gpk, ZGRID) / ip_ if ip_ > 0 else np.nan
                row_m[27] = refT / (iT / L) if iT > 0 else np.nan
                row_m[28] = refn / (i_n / L) if i_n > 0 else np.nan
                row_m[29] = refp / (ip_ / L) if ip_ > 0 else np.nan
                if refT > 0:
                    row_m[30] = _trapz(np.clip(gTk / refT, 0, 3) ** 1.5, ZGRID) / L
                zne20[k] = row_m[13]
                avg_pe[k] = row_m[24]
            TSCM[k] = row_m
    if "edge" in systems:
        tt, Te, ne, pos = systems["edge"]
        valid = np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
        Te = np.clip(np.nan_to_num(Te), 0, TE_CLIP_EV) / 1e3
        ne = np.clip(np.nan_to_num(ne), 0, NE_CLIP) / 1e19
        Tei, dnear = _tinterp(tt, Te, valid, t, GAP_EDGE_MS)
        nei, _ = _tinterp(tt, ne, valid, t, GAP_EDGE_MS)
        o = np.argsort(pos, kind="stable")
        r = pos[o]
        Tei, nei = Tei[:, o], nei[:, o]
        pei = 1.602 * Tei * nei
        gT = _grid(r, Tei, RGRID, MAXD_EDGE)
        gn = _grid(r, nei, RGRID, MAXD_EDGE)
        gp = _grid(r, pei, RGRID, MAXD_EDGE)
        TSE = np.hstack([gT, gn, gp, np.isfinite(gT).astype(np.float64)])
        C = len(r)
        for k in range(T):
            m = np.isfinite(Tei[k])
            nv = int(m.sum())
            row_e = np.full(len(TSEM_NAMES), np.nan)
            row_e[0] = nv / C
            row_e[1] = min(float(dnear[k]), 1000.0)
            if nv >= 1:
                row_e[2:5] = [Tei[k, m].mean(), nei[k, m].mean(), pei[k, m].mean()]
                row_e[7] = r[m].max()
            if nv >= 3:
                rr = r[m]
                if np.ptp(rr) > 0.05:
                    row_e[5] = np.polyfit(rr, Tei[k, m], 1)[0]
                    row_e[6] = np.polyfit(rr, nei[k, m], 1)[0]
            TSEM[k] = row_e

    bt = np.abs(vals0["bcoil"])
    ipa_k = np.maximum(np.abs(ip), IP_FLOOR_KA)
    PHYS = np.column_stack([avg_pe / (np.maximum(bt, 1.0) * ipa_k) * 1e3,
                            avg_pe * zne20,
                            bt * zne20 / ipa_k])

    X = np.hstack([X1.astype(np.float64), TH1, MAG, IPN, TIME, TSC, TSCM, TSE, TSEM, PHYS]).astype(np.float32)
    X[np.isinf(X)] = np.nan
    if return_diag:
        diag["T"] = T
        return X, diag
    return X
