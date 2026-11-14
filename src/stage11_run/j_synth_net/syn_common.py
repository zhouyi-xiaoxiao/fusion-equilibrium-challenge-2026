#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — shared helpers (CPU side, .venv_gs).

  ts_feats      ONE feature function for real and synthetic frames: Thomson edge observations (outer / inner, measured or
                censored at the coverage limit), isotherm-pair crossings (inv_core.chord_crossings = the inverse's own code),
                Te-peak / symmetric-axis positions, electron beta. Positions are (R - R0) / a with the PUBLIC R0, a.
  render        synthetic Te / ne on given channel positions (MAST: the frame's REAL channels + validity mask; DIII-D: the
                pseudo-horizontal laser of the harness) from a solved synthetic equilibrium: flux functions inside the LCFS,
                exponential SOL outside, multiplicative noise, optional inboard calibration error.
  syn_edges     edge observations of a synthetic equilibrium = its LCFS midplane crossings + noise, with the coverage /
                censoring logic of run_g1.build_obs (MAST) / run_inv7.pseudo_frame (DIII-D harness).
  coil_x        network coil inputs (shipped current columns per unit Ip, solver orientation; + P2 split currents; log Ip, log F).
Nothing here reads a file. Rule 3: callers install rule3_guard for MAST and pass input columns only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
RUN = WS / "stage11_run"
for p in (RUN / "g_profile", RUN / "g_restrict", RUN / "e_mast_inv6", RUN / "ch2d_inverse", RUN / "ch2_G1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
if str(RUN / "h_p2chan") not in sys.path:
    sys.path.append(str(RUN / "h_p2chan"))                 # LAST: only p2split / p2wiring are taken from there

import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402
import inv_core as K  # noqa: E402
import thomson_geom as TG  # noqa: E402

E_CH = 1.602176634e-19
MU0 = C.MU0
LEVELS = (0.2, 0.35, 0.5, 0.65, 0.8)
NF = 8 + 4 * len(LEVELS) + 5
FEAT_NAMES = (["out_eq", "out_eq_x", "out_ge", "out_ge_x", "in_eq", "in_eq_x", "in_ge", "in_ge_x"]
              + [f"{s}{f:g}{k}" for f in LEVELS for s in ("lo", "hi") for k in ("_has", "_x")]
              + ["prof_has", "tepk_x", "sym_has", "sym_x", "log_beta_pe"])
# noise realisations of the synthetic diagnostics: (sd Te, sd ne, sd outer edge [m], sd inner edge [m], inboard Te calibration)
NOISE = ((0.03, 0.03, 0.005, 0.010, 0.00),          # 0 low
         (0.07, 0.06, 0.015, 0.030, 0.00),          # 1 nominal (MAST-like: 7 % / 6 % channel noise; edge jitter ~ 1-3 cm)
         (0.10, 0.10, 0.030, 0.050, 0.05))          # 2 stress-like (harness: 10 % noise, 5 % inboard calibration, 3 cm edges)
NREAL = len(NOISE)
SCAL_KEYS = ("q95", "betaN", "li", "kappa", "a", "Rgeo", "volume", "tri_top", "tri_bot", "betap", "Rin", "Rout", "Ztop", "Zbot",
             "edge_gs", "edge_gs_in", "f95_ratio", "pav")
DIAG_KEYS = ("Rax", "Zax", "zt", "dIz_frac", "f_press", "kind", "n_it", "bps", "fvac", "ip", "t", "k", "gev", "dr", "cp", "b1", "b2",
             "be", "P0", "s", "a0", "Rxl", "Zxl", "Rxu", "Zxu", "dsep_psi", "d_tgt", "secs", "gev_g1")
KIND = {"xpt": 0.0, "wall": 1.0, "open": 2.0}


def ts_feats(prof, edges, bps, machine, levels=LEVELS):
    """prof = (R, Z, Te, ne) of the frame's laser profile or None; edges = list of {name out|in, kind eq|ge, P}."""
    R0, a = C.R0_PUBLIC[machine], C.A_PUBLIC[machine]
    v = np.zeros(NF)
    for o in edges or []:
        j = (0 if o["name"] == "out" else 4) + (0 if o["kind"] == "eq" else 2)
        v[j], v[j + 1] = 1.0, (float(o["P"][0]) - R0) / a
    if prof is None:
        return v
    xr, Te, ne = np.asarray(prof[0], float), np.asarray(prof[2], float), np.asarray(prof[3], float)
    try:
        _, cr = K.chord_crossings(xr, Te, list(levels))
    except Exception:  # noqa: BLE001
        cr = {}
    j = 8
    for f in levels:
        rec = cr.get(f, {})
        for side in ("lo", "hi"):
            if side in rec and np.isfinite(rec[side][0]):
                v[j], v[j + 1] = 1.0, (rec[side][0] - R0) / a
            j += 2
    try:
        g = TG.profile_geometry(xr, Te, ne, f_list=())
    except Exception:  # noqa: BLE001
        g = None
    if g:
        v[j] = 1.0
        v[j + 1] = (g["x_tepk"] - R0) / a if np.isfinite(g["x_tepk"]) else 0.0
        if np.isfinite(g["x_sym"]):
            v[j + 2], v[j + 3] = 1.0, (g["x_sym"] - R0) / a
        m = np.isfinite(xr) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
        o = np.argsort(xr[m])
        pe = E_CH * float(TG._smooth(Te[m][o] * ne[m][o]).max())
        v[j + 4] = float(np.clip(np.log10(max(2 * MU0 * pe / max(bps, 1e-9) ** 2, 1e-6)), -4.0, 1.5))
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def sample_ts_spec(rng, P0, te_ref, amp_sd=0.15):
    """Thomson profile shapes of one synthetic equilibrium (independent of the GS source shapes: the labels are a magnetics-only
    reconstruction, the Thomson profile is a geometry likelihood + a soft pressure-amplitude predictor)."""
    lu = lambda lo, hi: float(np.exp(rng.uniform(np.log(lo), np.log(hi))))  # noqa: E731
    s = {"aT": lu(1.0, 4.0), "bT": lu(0.8, 3.0), "fT": lu(0.01, 0.10), "lamT": lu(0.01, 0.10),
         "aN": lu(1.5, 10.0), "bN": lu(0.2, 1.5), "fN": float(rng.uniform(0.08, 0.40)), "lamN": lu(0.01, 0.10)}
    te0 = float(te_ref * np.exp(0.10 * rng.standard_normal()))
    pe0 = float(P0 * np.exp(amp_sd * rng.standard_normal()))
    s["Te0"], s["ne0"] = te0, pe0 / (E_CH * te0)
    return s


FIT_BOUNDS = {"T": ((0.7, 6.0), (0.5, 6.0), (0.003, 0.3), (0.005, 0.3)),      # a, b, f_sep, lambda
              "N": ((1.2, 16.0), (0.1, 2.5), (0.02, 0.8), (0.005, 0.3))}


def fit_shape(u, y, kind):
    """Robust log-space LSQ fit of y(u) = y0 [fs + (1 - fs)(1 - u^a)^b] (u <= 1), y0 fs exp(-(u - 1) / lam) (u > 1) to a REAL laser
    profile mapped on one of our own solutions (label-free). Returns dict(a, b, fs, lam, has_sol) or None."""
    from scipy.optimize import least_squares
    m = np.isfinite(u) & np.isfinite(y) & (y > 0)
    u, y = u[m], y[m]
    core = u <= 1.0
    if core.sum() < 8 or (u[core].max() - u[core].min()) < 0.4:
        return None
    bd = FIT_BOUNDS[kind]
    lo = [np.log(y.max()) - 1.5] + [np.log(b[0]) for b in bd]
    hi = [np.log(y.max()) + 1.5] + [np.log(b[1]) for b in bd]
    x0 = [np.log(np.percentile(y[core], 95))] + [0.5 * (np.log(b[0]) + np.log(b[1])) for b in bd]
    has_sol = (~core).sum() >= 2

    def model(x, uu):
        y0, a, b_, fs, lam = np.exp(x)
        uc = np.clip(uu, 0, 1)
        v = y0 * (fs + (1 - fs) * (1 - uc ** a) ** b_)
        return np.where(uu > 1, y0 * fs * np.exp(-(uu - 1) / lam), v)

    def f(x):
        r = np.log(np.maximum(model(x, u), 1e-30)) - np.log(y)
        return r if has_sol else r[core]
    try:
        r = least_squares(f, np.clip(x0, lo, hi), bounds=(lo, hi), loss="soft_l1", f_scale=0.15, max_nfev=200)
    except Exception:  # noqa: BLE001
        return None
    if not r.success and r.status <= 0:
        return None
    _, a, b_, fs, lam = np.exp(r.x)
    return {"a": float(a), "b": float(b_), "fs": float(fs), "lam": float(lam), "has_sol": bool(has_sol)}


def fit_real_shapes(prof, sol, zl, rin, rout):
    """Te / ne shape fits of the frame's REAL laser profile in the flux coordinate of one of OUR solutions (label-free)."""
    if prof is None:
        return None
    b = sol["b"]
    xr, Te, ne = np.asarray(prof[0], float), np.asarray(prof[2], float), np.asarray(prof[3], float)
    m = np.isfinite(xr) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
    if m.sum() < 10:
        return None
    u = (b["psi_axis"] - b["spl"].ev(xr[m], np.full(int(m.sum()), zl))) / (b["psi_axis"] - b["psi_b"])
    inside = (xr[m] >= rin) & (xr[m] <= rout)
    u = np.where(inside, np.clip(u, 0.0, 1.0), np.clip(u, 1.0 + 1e-3, 3.0))
    ft, fn = fit_shape(u, Te[m], "T"), fit_shape(u, ne[m], "N")
    if ft is None or fn is None:
        return None
    return {"T": ft, "N": fn}


def sample_ts_spec_fit(rng, P0, te_ref, fit, amp_sd=0.15, jit=0.25, p_broad=0.2):
    """Thomson shapes drawn around the frame's own real-profile fit (log-normal jitter), or from the broad prior with p_broad."""
    if fit is None or rng.uniform() < p_broad:
        return sample_ts_spec(rng, P0, te_ref, amp_sd)
    s = {}
    for kind, keys in (("T", ("aT", "bT", "fT", "lamT")), ("N", ("aN", "bN", "fN", "lamN"))):
        f, bd = fit[kind], FIT_BOUNDS[kind]
        for key, val, (lo, hi), sd in zip(keys, (f["a"], f["b"], f["fs"], f["lam"]), bd, (jit, jit, jit, jit if f["has_sol"] else 2 * jit)):
            s[key] = float(np.clip(val * np.exp(sd * rng.standard_normal()), lo, hi))
    te0 = float(te_ref * np.exp(0.10 * rng.standard_normal()))
    pe0 = float(P0 * np.exp(amp_sd * rng.standard_normal()))
    s["Te0"], s["ne0"] = te0, pe0 / (E_CH * te0)
    return s


def te_ne(u, s):
    u = np.asarray(u, float)
    uc = np.clip(u, 0.0, 1.0)
    Ts, ns = s["Te0"] * s["fT"], s["ne0"] * s["fN"]
    Te = Ts + (s["Te0"] - Ts) * (1.0 - uc ** s["aT"]) ** s["bT"]
    ne = ns + (s["ne0"] - ns) * (1.0 - uc ** s["aN"]) ** s["bN"]
    so = u > 1.0
    if np.any(so):
        Te = np.where(so, Ts * np.exp(-(u - 1.0) / s["lamT"]), Te)
        ne = np.where(so, ns * np.exp(-(u - 1.0) / s["lamN"]), ne)
    return Te, ne


def render(sol, xr, valid, zl, rin, rout, spec, rng, sT, sN, branch=0.0):
    """(R, Z, Te, ne) on the channels xr (NaN where not valid)."""
    b = sol["b"]
    D_ = b["psi_axis"] - b["psi_b"]
    xr = np.asarray(xr, float)
    fin = np.isfinite(xr)
    u = np.full(len(xr), 2.0)
    u[fin] = (b["psi_axis"] - b["spl"].ev(xr[fin], np.full(int(fin.sum()), zl))) / D_
    inside = fin & (xr >= rin) & (xr <= rout)
    u = np.where(inside, np.clip(u, 0.0, 1.0), np.clip(u, 1.0 + 1e-3, 3.0))
    Te, ne = te_ne(u, spec)
    Te = Te * np.exp(sT * rng.standard_normal(len(xr)))
    ne = ne * np.exp(sN * rng.standard_normal(len(xr)))
    if branch:
        Te = np.where(xr < b["axis"][0], Te * (1.0 + branch * (1.0 if rng.uniform() < 0.5 else -1.0)), Te)
    valid = np.asarray(valid, bool) & fin
    return (xr, np.full(len(xr), zl), np.where(valid, Te, np.nan), np.where(valid, ne, np.nan))


def syn_edges(rout, rin, zl, cov_lo, cov_hi, e_out, e_in, margin=0.0):
    ro, ri = rout + e_out, rin + e_in
    edges = []
    if np.isfinite(cov_hi) and ro >= cov_hi - margin:
        edges.append({"name": "out", "P": (float(cov_hi), zl), "e": (1.0, 0.0), "kind": "ge"})
    else:
        edges.append({"name": "out", "P": (float(ro), zl), "e": (1.0, 0.0), "kind": "eq"})
    if np.isfinite(cov_lo) and ri <= cov_lo + margin:
        edges.append({"name": "in", "P": (float(cov_lo), zl), "e": (-1.0, 0.0), "kind": "ge"})
    else:
        edges.append({"name": "in", "P": (float(ri), zl), "e": (-1.0, 0.0), "kind": "eq"})
    return edges


def coil_x(machine, inp, sgn, Ip, k, p2d=None):
    cols = C.current_columns(machine)
    ip = max(float(Ip[k]), 1.0)
    v = [sgn[k] * inp["cur"][c][k] / ip for c in cols]
    d = [sgn[k] * p2d["U"][k] / ip, sgn[k] * p2d["L"][k] / ip] if p2d is not None else [0.0, 0.0]
    return np.array(v + d + [np.log(ip / 1e6), np.log(max(abs(float(inp["fvac"][k])), 1e-6))], float)


def xpoints(sol, machine):
    b = sol["b"]
    Ra, Za = b["axis"]
    dz = 0.5 * C.A_PUBLIC[machine]
    sad = [x for x in b["saddles"] if x[2] < b["psi_axis"]]
    lo = [x for x in sad if x[1] < Za - dz]
    up = [x for x in sad if x[1] > Za + dz]
    xl = max(lo, key=lambda x: x[2]) if lo else None
    xu = max(up, key=lambda x: x[2]) if up else None
    span = b["psi_axis"] - b["psi_b"]
    return (float(xl[0]) if xl else np.nan, float(xl[1]) if xl else np.nan, float(xu[0]) if xu else np.nan,
            float(xu[1]) if xu else np.nan, float((xu[2] - xl[2]) / span) if (xl and xu) else np.nan)


# ----------------------------------------------------------------------------------------------- DIII-D REAL chords (real-chord leg)
# Layout of the real-chord features (NF_RC = NF + 4 x levels): slots 0-32 as FEAT_NAMES but on the VERTICAL core chord at R = rc
# (edge = upper LCFS crossing Z_top, positions Z / a; 'lo' / 'hi' = lower / upper isotherm crossing in Z; inner-edge and symmetric-axis
# slots stay 0), then the TANGENTIAL system (near-midplane points, positions (R - R0) / a): lo / hi isotherm crossings per level.
NF_RC = NF + 4 * len(LEVELS)


def split_rc(prof, rc):
    R, Z, Te, ne = (np.asarray(x_, float) for x_ in prof)
    Z = np.broadcast_to(Z, R.shape)
    core = np.abs(R - rc) < 1e-6
    return (Z[core], Te[core], ne[core]), (R[~core], Te[~core], ne[~core]), core


def ts_feats_rc(prof, edges, bps, rc, levels=LEVELS):
    a = C.A_PUBLIC["d3d"]; R0 = C.R0_PUBLIC["d3d"]
    v = np.zeros(NF_RC)
    for o in edges or []:
        if o["name"] != "out":
            continue
        j = 0 if o["kind"] == "eq" else 2
        v[j], v[j + 1] = 1.0, float(o["P"][1]) / a
    if prof is None:
        return v
    (zc, tc, nc), (rt, tt, nt), _ = split_rc(prof, rc)
    for blk, (x, te, sc_, off) in enumerate(((zc, tc, a, 0.0), (rt, tt, a, R0))):
        try:
            _, cr = K.chord_crossings(x, te, list(levels), min_pts=8 if blk == 0 else 5)
        except Exception:  # noqa: BLE001
            cr = {}
        j = 8 if blk == 0 else NF
        for f in levels:
            rec = cr.get(f, {})
            for side in ("lo", "hi"):
                if side in rec and np.isfinite(rec[side][0]):
                    v[j], v[j + 1] = 1.0, (rec[side][0] - off) / sc_
                j += 2
    m = np.isfinite(zc) & np.isfinite(tc) & np.isfinite(nc) & (tc > 0) & (nc > 0)
    if m.sum() >= 5:
        o = np.argsort(zc[m])
        ts_ = K._smooth(tc[m][o])
        v[28], v[29] = 1.0, float(zc[m][o][int(np.argmax(ts_))]) / a
        pe = E_CH * float(TG._smooth(tc[m][o] * nc[m][o]).max())
        v[32] = float(np.clip(np.log10(max(2 * MU0 * pe / max(bps, 1e-9) ** 2, 1e-6)), -4.0, 1.5))
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


def render_rc(sol, Rp, Zp, valid, Cc, spec, rng, sT, sN, core, branch=0.0):
    """(R, Z, Te, ne) of a synthetic equilibrium on arbitrary channel positions (inside = within the 99.5 % flux contour and u <= 1);
    branch = relative calibration error of the tangential system against the core chord."""
    b = sol["b"]
    D_ = b["psi_axis"] - b["psi_b"]
    Rp, Zp = np.asarray(Rp, float), np.broadcast_to(np.asarray(Zp, float), np.shape(Rp))
    fin = np.isfinite(Rp) & np.isfinite(Zp)
    u = np.full(len(Rp), 2.0)
    u[fin] = (b["psi_axis"] - b["spl"].ev(Rp[fin], Zp[fin])) / D_
    ins = np.zeros(len(Rp), bool)
    if Cc is not None and fin.any():
        ins[fin] = C.points_in_poly(Rp[fin], Zp[fin], Cc[:, 0], Cc[:, 1])
    inside = fin & ins & (u <= 1.0 + 1e-6)
    u = np.where(inside, np.clip(u, 0.0, 1.0), np.clip(u, 1.0 + 1e-3, 3.0))
    Te, ne = te_ne(u, spec)
    Te = Te * np.exp(sT * rng.standard_normal(len(Rp)))
    ne = ne * np.exp(sN * rng.standard_normal(len(Rp)))
    if branch:
        Te = np.where(~core, Te * (1.0 + branch * (1.0 if rng.uniform() < 0.5 else -1.0)), Te)
    valid = np.asarray(valid, bool) & fin
    return (Rp, Zp, np.where(valid, Te, np.nan), np.where(valid, ne, np.nan))


def syn_edges_rc(ztop, cov_hi, e_out, rc, margin=0.0):
    zo = ztop + e_out
    if np.isfinite(cov_hi) and zo >= cov_hi - margin:
        return [{"name": "out", "P": (float(rc), float(cov_hi)), "e": (0.0, 1.0), "kind": "ge"}]
    return [{"name": "out", "P": (float(rc), float(zo)), "e": (0.0, 1.0), "kind": "eq"}]


def fit_real_shapes_rc(prof, sol, Cc):
    """real-chord leg, generator v2: Te / ne shape fits of the frame's REAL DIII-D channels (any (R, Z)) in the flux coordinate of one
    of OUR solutions (label-free in the same sense as fit_real_shapes; inside = within our own 99.5 % contour)."""
    if prof is None or Cc is None:
        return None
    b = sol["b"]
    R = np.asarray(prof[0], float); Z = np.broadcast_to(np.asarray(prof[1], float), R.shape)
    Te, ne = np.asarray(prof[2], float), np.asarray(prof[3], float)
    m = np.isfinite(R) & np.isfinite(Z) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0)
    if m.sum() < 10:
        return None
    u = (b["psi_axis"] - b["spl"].ev(R[m], Z[m])) / (b["psi_axis"] - b["psi_b"])
    ins = C.points_in_poly(R[m], Z[m], Cc[:, 0], Cc[:, 1]) & (u <= 1.0 + 1e-6)
    u = np.where(ins, np.clip(u, 0.0, 1.0), np.clip(u, 1.0 + 1e-3, 3.0))
    ft, fn = fit_shape(u, Te[m], "T"), fit_shape(u, ne[m], "N")
    if ft is None or fn is None:
        return None
    return {"T": ft, "N": fn}
