#!/usr/bin/env python3
"""
Stage 11 / Ch1 FEATURE-V2.1 for DIII-D (inputs only) — per-shot feature builder.

Superset of FEATURE-V2 (feat_v2.py): columns 0..692 keep the v2 names/positions, three blocks are
appended (D = 789).  `row` may be a pandas row or a plain dict (Hugging Face row) — only INPUT columns are
read (magnetics_* except magnetics_dsep*, thomson_*, efit_times); no efit_* target, no dsep.

What changed against v2
  * THOMSON VALIDITY CONTRACT (organiser issue #12, 2026-10-11) is enforced in the reader
    (`thomson_systems_v21`):
      1. per system keep sample i only if t_i > max(t_<i)  (`thomson_valid`; v2 SORTED non-monotone
         arrays instead of dropping, and kept duplicated times);
      2. a channel sample is a measurement iff finite and Te > 0 and ne > 0 (DIII-D zero = "no
         measurement", MAST-style NaN handled too) -> per-sample/per-channel mask `valid`;
      3. resampling to the EFIT frame uses, PER CHANNEL, the previous / next VALID sample only
         (linear in time if both are within the gap, else the nearer valid one within the gap, else
         masked).  v2 bracketed with the raw neighbouring profiles, so an all-zero (or partly zero)
         neighbour demoted the frame to one-sided / masked even if a valid profile was one sample
         further away.
    Affected v2 blocks (same names, values now contract-compliant): th1, tsc, tscm, tse, tsem, phys.
    `th1` = the v1 Thomson summary but on the nearest VALID profile (<= 100 ms), statistics over
    valid channels only, last column = valid-channel fraction (v2: zeros entered mean/max/median and
    the "finite fraction" was 1 for all-zero rows).
  * v1 block (first 90 columns): unchanged values.  Its 20 Thomson-summary columns and
    v1_pe_core_mean / v1_betaN_proxy were ALL ZERO in every v1 cache (starter loader raised on the
    parquet object arrays); v2 reproduced that through the same exception, v2.1 hard-wires the zeros
    so the block cannot depend on any Thomson array (robust on unseen shots).
  * unchanged code path, bit-identical values: mag, ipn, time.
New blocks
  tsv  (12)  Thomson validity, per system (core, edge): dt to previous / next valid profile [ms,
             clip 1000], valid-channel fraction of the nearest valid profile, fraction and count of
             valid profiles within +-100 ms, valid-profile fraction over the EFIT window (shot const).
  drv  (80)  long-timescale "driving" history bank from the full-rate magnetics record:
             for ECOILA (ohmic coil), plasma_current, bcoil:
               causal exponential moving averages, tau = 50/100/250/500/1000/2500 ms (6),
               value - EMA (6)  [= tau x leaky integral of dI/dt],
               centred dI/dt over 20/50/200 ms (3);
             ohmic flux-swing proxies (ECOILA change since record start / since breakdown, total
             variation and time integral since breakdown, swing per unit Ip, loop-voltage-per-Ip
             proxies), Ip-normalised Ip history, Ip-seconds since breakdown, Ip-phase timing
             (time since / to the 50 % crossings, ramp-down progress, shot-phase fraction, final
             decile flag, time to the end of the Ip record) and EFIT-frame gaps / frame counters.
             Uses the whole input record of the shot (acausal where stated; inputs only).
  rec   (4)  acquisition-record metadata of the shot (constants per shot): magnetics record start /
             sampling step / end, start of the (fixed) Ip time axis.  They index the DAQ configuration
             (20 kHz vs 2 kHz magnetics, 69 % / 31 % Ip-axis classes) and hence the campaign era.
             OPTIONAL block — kept separate so a model can drop it (`blocks["rec"]` in spec.json).
Raw features may contain NaN only in tsc/tscm/tse/tsem/phys; `standardise` maps NaN -> 0 (= train
mean) for every non-v1 column and clips to +-clip.
"""
from __future__ import annotations

import numpy as np

from feat_v2 import (D3D_SIGNALS, FCOILS, OFFSETS_MS, SMOOTH_MS, GAP_CORE_MS, GAP_EDGE_MS, ZGRID, RGRID,  # noqa: F401
                     MAXD_CORE, MAXD_EDGE, TE_CLIP_EV, NE_CLIP, IP_FLOOR_KA, TSCM_NAMES, TSEM_NAMES,
                     _stack, _smooth_fn, _ev, _grid, _drop_pos, _trapz, read_inputs, input_columns)
from feat_v2 import feature_names as _feature_names_v2

D_V2 = 693
TH1_GAP_MS = 100.0
TSV_CLIP_MS = 1000.0
TSV_WIN_MS = 100.0
TSV_NAMES = ["dt_prev_valid_ms", "dt_next_valid_ms", "nearest_valid_chfrac", "valid_prof_frac_pm100",
             "n_valid_prof_pm100", "shot_valid_prof_frac"]
DRIVE_SIGNALS = ["ECOILA", "plasma_current", "bcoil"]
DRV_TAUS_MS = [50, 100, 250, 500, 1000, 2500]
DRV_DWIN_MS = [20, 50, 200]
DRV_GRID_MS = 1.0
DRV_GAP_CLIP_MS = 500.0
DRV_EXTRA_NAMES = (
    ["E_minus_start", "E_minus_bd", "E_tv_since_bd", "E_int_since_bd_kAs", "E_swing_over_Ip"]
    + ["E_dIdt_c50ms_over_Ip", "E_dIdt_c200ms_over_Ip"]
    + [f"E_minus_ema{tau}ms_over_Ip" for tau in DRV_TAUS_MS]
    + [f"Ip_minus_ema{tau}ms_over_Ipmax" for tau in DRV_TAUS_MS]
    + ["Ip_int_since_bd_kAs", "Ip_int_since_bd_over_Ipmax_s"]
    + ["t_since_ip50_ms", "t_to_ip50_last_ms", "t_to_iprec_end_ms", "rampdown_frac", "phase_frac",
       "final_decile_flag", "efit_end_minus_ip_end_ms"]
    + ["gap_prev_ms", "gap_next_ms", "is_first_frame", "is_last_frame", "frames_from_start",
       "frames_to_end", "t_to_last_frame_ms"])
V2_COMPAT = {"v1": "identical", "mag": "identical", "ipn": "identical", "time": "identical",
             "th1": "contract-fixed", "tsc": "contract-fixed", "tscm": "contract-fixed",
             "tse": "contract-fixed", "tsem": "contract-fixed", "phys": "contract-fixed",
             "tsv": "new", "drv": "new", "rec": "new"}
REC_NAMES = ["mag_t0_ms", "mag_dt_ms", "mag_t_end_ms", "ip_t0_ms"]


def feature_names():
    """Ordered list of (name, block); the first 693 entries are exactly feat_v2.feature_names()."""
    out = list(_feature_names_v2())
    assert len(out) == D_V2
    for sk in ("core", "edge"):
        out += [(f"tsv_{sk}_{x}", "tsv") for x in TSV_NAMES]
    for s in DRIVE_SIGNALS:
        out += [(f"drv_{s}_ema{tau}ms", "drv") for tau in DRV_TAUS_MS]
        out += [(f"drv_{s}_minus_ema{tau}ms", "drv") for tau in DRV_TAUS_MS]
        out += [(f"drv_{s}_dIdt_c{w}ms", "drv") for w in DRV_DWIN_MS]
    out += [(f"drv_{x}", "drv") for x in DRV_EXTRA_NAMES]
    out += [(f"rec_{x}", "rec") for x in REC_NAMES]
    return out


# ------------------------------------------------------------------ v1 block

def features_v1(row):
    """v1 block (T, 90), values identical to stage2_features.full_features_d3d on the starter loader.
    The Thomson summary is the constant zero block (see module docstring) — no Thomson array is read."""
    from data_fixes import fix_d3d_ip_times
    from experiments import D3D_MAGNETICS_SIGNALS
    from stage2_features import full_features_d3d
    t = np.asarray(row["efit_times"], dtype=np.float64)
    shared = np.asarray(row["magnetics_time"], dtype=np.float64)
    ip_times = fix_d3d_ip_times(row)
    mag = {}
    for sig in D3D_MAGNETICS_SIGNALS:
        col = f"magnetics_{sig}"
        if col not in row:
            continue
        mag[sig] = {"values": np.asarray(row[col], dtype=np.float32),
                    "times": ip_times if sig == "plasma_current" else shared}
    shot = {"machine": "d3d", "efit_times": t, "magnetics": mag, "scalars": {}}
    return full_features_d3d(shot, thomson=np.zeros((len(t), 20), dtype=np.float64))


# ------------------------------------------------------------------ Thomson reader (validity contract)

def thomson_valid(times):
    """Organiser rule (issue #12): True where a sample's time is later than every earlier sample's
    time.  Non-finite times are never valid and do not poison the running maximum."""
    t = np.asarray(times, dtype=np.float64)
    if t.size == 0:
        return np.zeros(0, dtype=bool)
    fin = np.isfinite(t)
    tf = np.where(fin, t, -np.inf)
    prev_max = np.concatenate(([-np.inf], np.maximum.accumulate(tf)[:-1]))
    return fin & (t > prev_max)


def thomson_systems_v21(row):
    """Contract-compliant Thomson.  Returns (systems, diag, stats):
    systems[sk] = dict(t (N,) strictly increasing, Te (N,C) eV, ne (N,C) m^-3, pos (C,), valid (N,C) bool)
    for sk in {'core' (pos = Z of TS_core_* chords), 'edge' (pos = R of TS_tangential_* chords)}."""
    names = np.array([str(x) for x in row["thomson_chord_name"]]) if "thomson_chord_name" in row else np.array([])
    cR = np.asarray(row["thomson_chord_R"], dtype=np.float64) if "thomson_chord_R" in row else np.zeros(0)
    cZ = np.asarray(row["thomson_chord_Z"], dtype=np.float64) if "thomson_chord_Z" in row else np.zeros(0)
    out, diag, stats = {}, {}, {}
    for sk, prefix, coord in (("core", "TS_core", cZ), ("edge", "TS_tangential", cR)):
        key = f"thomson_{sk}"
        if f"{key}_times" not in row or f"{key}_Te" not in row or f"{key}_ne" not in row:
            diag[sk] = "missing"
            continue
        try:
            tt = np.asarray(row[f"{key}_times"], dtype=np.float64).ravel()
            Te = _stack(row[f"{key}_Te"])
            ne = _stack(row[f"{key}_ne"])
        except Exception as e:  # noqa: BLE001
            diag[sk] = f"load_error:{type(e).__name__}"
            continue
        n = min(len(tt), len(Te), len(ne))
        C = Te.shape[1] if (Te.ndim == 2 and n > 0) else 0
        if C == 0 or n == 0:
            diag[sk] = "empty"
            continue
        tt, Te, ne = tt[:n], Te[:n], ne[:n]
        sel = np.array([nm.startswith(prefix) for nm in names], dtype=bool) if len(names) else np.zeros(0, bool)
        if sel.sum() != C or Te.shape != ne.shape:
            diag[sk] = f"chord_mismatch:{int(sel.sum())}vs{C}"
            continue
        keep = thomson_valid(tt)                                   # (1) time validity, BEFORE resampling
        st = {"n_raw": int(n), "n_dropped": int((~keep).sum()),
              "n_dropped_with_data": int((((Te != 0) & np.isfinite(Te)).any(1) & ~keep).sum())}
        tt, Te, ne = tt[keep], Te[keep], ne[keep]
        if len(tt) == 0:
            diag[sk] = "empty"
            stats[sk] = st
            continue
        vT = np.isfinite(Te) & (Te > 0)                            # (2) value validity, element by element
        vn = np.isfinite(ne) & (ne > 0)
        valid = vT & vn
        st.update(n_kept=int(len(tt)), n_prof_valid=int(valid.any(1).sum()),
                  n_elem=int(valid.size), n_elem_valid=int(valid.sum()), n_elem_xor=int((vT ^ vn).sum()))
        out[sk] = {"t": tt, "Te": Te, "ne": ne, "pos": coord[sel], "valid": valid}
        diag[sk] = f"ok:{C}"
        stats[sk] = st
    return out, diag, stats


def _tinterp_valid(tt, A, valid, tq, gap):
    """Per-channel time resampling of A (N,C) onto tq (T,) using ONLY valid samples of that channel:
    linear between the previous and next valid sample when both are within `gap` ms, otherwise the
    one that is within the gap, otherwise NaN."""
    N, C = A.shape
    T = len(tq)
    out = np.full((T, C), np.nan)
    for c in range(C):
        m = valid[:, c]
        if not m.any():
            continue
        tv, av = tt[m], A[m, c]
        n = len(tv)
        i = np.searchsorted(tv, tq)                    # tv[i-1] < tq <= tv[i]
        lo = np.clip(i - 1, 0, n - 1)
        hi = np.clip(i, 0, n - 1)
        dlo = tq - tv[lo]
        dhi = tv[hi] - tq
        okl = (i >= 1) & (dlo <= gap)
        okh = (i <= n - 1) & (dhi <= gap)
        den = dlo + dhi
        both = okl & okh & (den > 0)
        w = np.where(both, dlo / np.where(den > 0, den, 1.0), 0.0)
        o = np.where(both, (1.0 - w) * av[lo] + w * av[hi],
                     np.where(okl, av[lo], np.where(okh, av[hi], np.nan)))
        out[:, c] = o
    return out


def _tsv(sysd, t):
    """(T, 6) validity features of one system + distance to the nearest VALID profile (T,)."""
    T = len(t)
    out = np.zeros((T, len(TSV_NAMES)))
    out[:, 0] = out[:, 1] = TSV_CLIP_MS
    dnear = np.full(T, np.inf)
    if sysd is None or T == 0:
        return out, dnear
    tt, valid = sysd["t"], sysd["valid"]
    pv = valid.any(1)
    if pv.any():
        tv = tt[pv]
        cf = valid[pv].mean(1)
        n = len(tv)
        i = np.searchsorted(tv, t)
        lo = np.clip(i - 1, 0, n - 1)
        hi = np.clip(i, 0, n - 1)
        dprev = np.where(i >= 1, t - tv[lo], np.inf)
        dnext = np.where(i <= n - 1, tv[hi] - t, np.inf)
        dnear = np.minimum(dprev, dnext)
        near = np.where(dprev < dnext, lo, hi)
        out[:, 0] = np.minimum(dprev, TSV_CLIP_MS)
        out[:, 1] = np.minimum(dnext, TSV_CLIP_MS)
        out[:, 2] = np.where(dnear <= TSV_CLIP_MS, cf[near], 0.0)
        nv = np.searchsorted(tv, t + TSV_WIN_MS, side="right") - np.searchsorted(tv, t - TSV_WIN_MS, side="left")
        na = np.searchsorted(tt, t + TSV_WIN_MS, side="right") - np.searchsorted(tt, t - TSV_WIN_MS, side="left")
        out[:, 3] = np.where(na > 0, nv / np.maximum(na, 1), 0.0)
        out[:, 4] = nv
        inw = (tt >= t[0] - 50.0) & (tt <= t[-1] + 50.0)
        out[:, 5] = float(pv[inw].mean()) if inw.any() else 0.0
    return out, dnear


def thomson_summary_v21(systems, t):
    """(T, 20) v1-style Thomson summary on contract-compliant data: per system Te[mean,max,med],
    ne[...], pe=Te*ne[...] over the VALID channels of the nearest VALID profile (<= 100 ms), and the
    valid-channel fraction of that profile.  No valid profile within the gap -> zeros (v1 convention)."""
    T = len(t)
    blocks = []
    for sk in ("core", "edge"):
        cols = np.zeros((T, 10), dtype=np.float64)
        sysd = systems.get(sk)
        if sysd is not None and T > 0:
            valid = sysd["valid"]
            pv = valid.any(1)
            if pv.any():
                tv, Te, ne, vv = sysd["t"][pv], sysd["Te"][pv], sysd["ne"][pv], valid[pv]
                n = len(tv)
                idx = np.clip(np.searchsorted(tv, t), 0, n - 1)
                idx_lo = np.clip(idx - 1, 0, n - 1)
                pick = np.where(np.abs(tv[idx_lo] - t) < np.abs(tv[idx] - t), idx_lo, idx)
                for ti in range(T):
                    j = pick[ti]
                    if abs(tv[j] - t[ti]) > TH1_GAP_MS:
                        continue
                    m = vv[j]
                    a, b = Te[j][m], ne[j][m]
                    p = a * b
                    cols[ti] = [a.mean(), a.max(), np.median(a), b.mean(), b.max(), np.median(b),
                                p.mean(), p.max(), np.median(p), m.mean()]
        blocks.append(cols)
    return np.nan_to_num(np.hstack(blocks), nan=0.0, posinf=0.0, neginf=0.0)


# ------------------------------------------------------------------ driving history bank

def _uniform(fn, dt=DRV_GRID_MS):
    """Resample a smoothed signal (ts, vs) onto a uniform dt grid spanning its record."""
    ts, vs = fn
    n = int(np.floor((ts[-1] - ts[0]) / dt)) + 1
    if n < 2:
        return None
    tg = ts[0] + dt * np.arange(n)
    return tg, np.interp(tg, ts, vs)


def _ema_bank(xg, taus, dt=DRV_GRID_MS):
    """Causal exponential moving averages y' = (x - y)/tau on a uniform grid (exact for piecewise-
    constant x); initial state = mean of the first 20 ms (signal assumed constant before the record)."""
    from scipy.signal import lfilter
    x0 = float(np.mean(xg[:20]))
    out = []
    for tau in taus:
        a = 1.0 - np.exp(-dt / float(tau))
        y, _ = lfilter([a], [1.0, -(1.0 - a)], xg, zi=[(1.0 - a) * x0])
        out.append(y)
    return out


def _boxcar_edge(x, k):
    xp = np.pad(x, (k // 2, k - 1 - k // 2), mode="edge")
    cs = np.cumsum(np.r_[0.0, xp])
    return (cs[k:] - cs[:-k]) / k


def drive_bank(t, fns, vals0, ph):
    """(T, 80) driving-history block.  fns: smoothed full-rate signals {name: (ts, vs) or None};
    vals0: their values at t; ph: Ip-phase quantities from the time block (ipmax, t_bd, t_end, t_fe,
    sgn_shot, ipv (|Ip| samples), ipt (their times))."""
    T = len(t)
    Z = np.zeros(T)
    cols = []
    grid, ema_t = {}, {}
    for s in DRIVE_SIGNALS:
        f = fns.get(s)
        g = _uniform(f) if f is not None else None
        grid[s] = g
        if g is None:
            ema_t[s] = [Z] * len(DRV_TAUS_MS)
            cols += [Z] * (2 * len(DRV_TAUS_MS) + len(DRV_DWIN_MS))
            continue
        tg, xg = g
        e_t = [np.interp(t, tg, y) for y in _ema_bank(xg, DRV_TAUS_MS)]
        ema_t[s] = e_t
        cols += e_t
        cols += [vals0[s] - e for e in e_t]
        cols += [(_ev(f, t + 0.5 * w) - _ev(f, t - 0.5 * w)) / w for w in DRV_DWIN_MS]

    ipmax, t_bd, t_end, t_fe = ph["ipmax"], ph["t_bd"], ph["t_end"], ph["t_fe"]
    sgn = ph["sgn_shot"] if ph["sgn_shot"] != 0 else 1.0
    ip = vals0["plasma_current"]
    ip_sf = sgn * np.maximum(np.abs(ip), IP_FLOOR_KA)

    # --- ohmic-coil flux-swing proxies
    gE = grid["ECOILA"]
    if gE is not None:
        tg, xg = gE
        E_start = float(np.mean(xg[:20]))
        E_bd = float(np.interp(t_bd, tg, xg))
        after = tg >= t_bd
        xs = _boxcar_edge(xg, 20)
        tv_c = np.cumsum(np.where(after, np.abs(np.diff(xs, prepend=xs[0])), 0.0))
        int_c = np.cumsum(np.where(after, xg - E_bd, 0.0)) * (DRV_GRID_MS / 1e3)
        E = vals0["ECOILA"]
        fE = fns["ECOILA"]
        cols += [E - E_start, E - E_bd, np.interp(t, tg, tv_c), np.interp(t, tg, int_c), (E - E_bd) / ip_sf]
        cols += [(_ev(fE, t + 0.5 * w) - _ev(fE, t - 0.5 * w)) / w / ip_sf for w in (50, 200)]
        cols += [(E - e) / ip_sf for e in ema_t["ECOILA"]]
    else:
        cols += [Z] * (5 + 2 + len(DRV_TAUS_MS))

    # --- Ip history normalised by the shot's Ip scale, Ip-seconds since breakdown
    cols += [sgn * (ip - e) / ipmax for e in ema_t["plasma_current"]]
    gI = grid["plasma_current"]
    if gI is not None:
        tg, xg = gI
        ic = np.cumsum(np.where(tg >= t_bd, np.abs(xg), 0.0)) * (DRV_GRID_MS / 1e3)
        ii = np.interp(t, tg, ic)
        cols += [ii, ii / ipmax]
    else:
        cols += [Z, Z]

    # --- Ip-phase timing
    ipv, ipt = ph["ipv"], ph["ipt"]
    if len(ipv) > 10:
        on = ipv >= 0.5 * ipmax
        t50a = float(ipt[np.argmax(on)]) if on.any() else t_bd
        t50b = float(ipt[len(on) - 1 - np.argmax(on[::-1])]) if on.any() else t_end
        t_rec_end = float(ipt[-1])
    else:
        t50a, t50b, t_rec_end = t_bd, t_end, (float(t[-1]) if T else 0.0)
    phase = np.clip((t - t_bd) / max(t_end - t_bd, 1.0), -0.5, 1.5)
    cols += [t - t50a, t50b - t, t_rec_end - t,
             np.clip((t - t_fe) / max(t_end - t_fe, 1.0), 0.0, 1.5), phase,
             (phase > 0.9).astype(np.float64), np.full(T, (float(t[-1]) if T else 0.0) - t_end)]

    # --- EFIT-frame gaps / counters (efit_times is an input for every split)
    if T >= 2:
        g = np.clip(np.diff(t), 0.0, DRV_GAP_CLIP_MS)
        gp, gn = np.r_[g[0], g], np.r_[g, g[-1]]
    else:
        gp = gn = np.full(T, 20.0)
    k = np.arange(T, dtype=np.float64)
    first = (k == 0).astype(np.float64)
    last = (k == T - 1).astype(np.float64)
    cols += [gp, gn, first, last, k, (T - 1) - k, (t[-1] - t) if T else Z]
    D = np.column_stack(cols) if T else np.zeros((0, len(cols)))
    return D


# ------------------------------------------------------------------ main builder

def n_features():
    return len(feature_names())


def features_v21(row, return_diag=False):
    """Feature-v2.1 block (T, 789) float32 =
    [v1 | th1 | mag | ipn | time | tsc | tscm | tse | tsem | phys | tsv | drv | rec]."""
    from data_fixes import fix_d3d_ip_times
    t = np.asarray(row["efit_times"], dtype=np.float64)
    T = len(t)
    diag = {}
    if T == 0:
        X0 = np.zeros((0, n_features()), dtype=np.float32)
        return (X0, {"T": 0, "thomson": {}, "thomson_stats": {}, "missing_signals": []}) if return_diag else X0
    X1 = features_v1(row)                                           # (T, 90)

    systems, tdiag, tstats = thomson_systems_v21(row)
    diag["thomson"] = tdiag
    diag["thomson_stats"] = tstats
    TH1 = thomson_summary_v21(systems, t)                           # (T, 20)

    # --- magnetics (code identical to feat_v2.features_v2)
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

    # --- time / Ip-phase block (code identical to feat_v2)
    ip_raw = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)
    ipv = np.abs(ip_raw)
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
        sgn_shot = float(np.sign(np.median(ip_raw[:n_ip][fin][on]))) if on.any() else 0.0
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

    # --- Thomson profiles (contract-compliant reader, valid-only time resampling)
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
    TSVc, dnear_c = _tsv(systems.get("core"), t)
    TSVe, dnear_e = _tsv(systems.get("edge"), t)
    if "core" in systems:
        sd = systems["core"]
        tt, valid, pos = sd["t"], sd["valid"], sd["pos"]
        Te = np.clip(np.nan_to_num(sd["Te"]), 0, TE_CLIP_EV) / 1e3        # keV
        ne = np.clip(np.nan_to_num(sd["ne"]), 0, NE_CLIP) / 1e19          # 1e19 m^-3
        Tei = _tinterp_valid(tt, Te, valid, t, GAP_CORE_MS)
        nei = _tinterp_valid(tt, ne, valid, t, GAP_CORE_MS)
        dnear = dnear_c
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
        sd = systems["edge"]
        tt, valid, pos = sd["t"], sd["valid"], sd["pos"]
        Te = np.clip(np.nan_to_num(sd["Te"]), 0, TE_CLIP_EV) / 1e3
        ne = np.clip(np.nan_to_num(sd["ne"]), 0, NE_CLIP) / 1e19
        Tei = _tinterp_valid(tt, Te, valid, t, GAP_EDGE_MS)
        nei = _tinterp_valid(tt, ne, valid, t, GAP_EDGE_MS)
        dnear = dnear_e
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

    DRV = drive_bank(t, fns, vals0, {"ipmax": ipmax, "t_bd": t_bd, "t_end": t_end, "t_fe": t_fe,
                                     "sgn_shot": sgn_shot, "ipv": ipv, "ipt": ipt})

    mt_f = mag_t[np.isfinite(mag_t)]
    it_f = ip_times[np.isfinite(ip_times)]
    REC = np.tile([float(mt_f[0]) if len(mt_f) else 0.0,
                   float(np.median(np.diff(mt_f))) if len(mt_f) > 1 else 0.0,
                   float(mt_f[-1]) if len(mt_f) else 0.0,
                   float(it_f[0]) if len(it_f) else 0.0], (T, 1))

    X = np.hstack([X1.astype(np.float64), TH1, MAG, IPN, TIME, TSC, TSCM, TSE, TSEM, PHYS,
                   TSVc, TSVe, DRV, REC]).astype(np.float32)
    X[np.isinf(X)] = np.nan
    if return_diag:
        diag["T"] = T
        return X, diag
    return X


# ------------------------------------------------------------------ standardiser (shared by prep + predict path)

def standardise(Xraw, mean, scale, is_v1, clip):
    """z = (x - mean)/scale in float64 -> float32.  v1 columns untouched otherwise (H convention);
    every other column: NaN -> 0 (= train mean), +-inf -> +-clip, clip to +-clip."""
    b = np.asarray(Xraw, dtype=np.float64)
    z = (b - mean[None, :]) / scale[None, :]
    nv = ~is_v1
    zz = np.nan_to_num(z[:, nv], nan=0.0, posinf=clip, neginf=-clip)
    z[:, nv] = np.clip(zz, -clip, clip)
    return z.astype(np.float32)


def load_scaler(path):
    s = np.load(path, allow_pickle=False)
    return s["mean"].astype(np.float64), s["scale"].astype(np.float64), s["is_v1"].astype(bool), float(s["clip"])


def predict_features(path_or_row, scaler=None):
    """Inputs-only predict path for ONE shot (parquet path, pandas row or HF dict row).
    Returns (X_raw (T,789) float32, efit_times (T,)) or, with scaler=(mean, scale, is_v1, clip) /
    a scaler_v21.npz path, (X_std, X_raw, efit_times)."""
    row = read_inputs(path_or_row) if isinstance(path_or_row, (str, bytes)) or hasattr(path_or_row, "__fspath__") \
        else path_or_row
    X = features_v21(row)
    t = np.asarray(row["efit_times"], dtype=np.float64)
    if scaler is None:
        return X, t
    if isinstance(scaler, (str, bytes)) or hasattr(scaler, "__fspath__"):
        scaler = load_scaler(scaler)
    return standardise(X, *scaler), X, t
