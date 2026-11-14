#!/usr/bin/env python3
"""Stage 11h / p2chan — P2 winding currents from the shipped MAST INPUT channels (documented optional module).

Channel facts (an internal development note §1-§2; public FAIR-MAST level-1 `amc` signal list + label-free input statistics):
  magnetics_p2{u,l}_current = AMC_P2{U,L} CURRENT: ONE Rogowski total per P2 coil can [kA.turn] = 12 I_in + 8 I_out + I_case.
  magnetics_efps_current    = feed current [kA] of the EFPS supply.  On 94 % of the shots all 20 turns of both P2 coils are in
                              series on EFPS:  T_U = T_L = 20 I_EFPS  (measured: T_L / (20 I_EFPS) = 1.000 +- 0.003).
  On the remaining shots the 8-turn OUTER windings are in series with the solenoid feed (T = 8 I_sol + 12 I_in), the LOWER
  inner winding is fed by EFPS and the UPPER inner winding carries no supply current.
Winding currents returned (ampere-turns per winding; they sum to the measured Rogowski total, so the case current stays where
G1 always had it, on the turns):
  wiring "efps" :  AT_in = 0.6 T,  AT_out = 0.4 T          (per-turn currents equal: the G1 assumption, exact here)
  wiring "sol"  :  AT_out = 8 I_sol,  AT_in = T - 8 I_sol   (per-turn difference d = I_in - I_out = (T - 20 I_sol) / 12)
No constant is fitted to equilibrium data or to any score: 8, 12, 20 are the public turn counts; the wiring class is decided per
shot by which of the two public-turn-count models reproduces the two Rogowski totals from the two feed currents.
"""
from __future__ import annotations

import numpy as np

N_IN, N_OUT = 12, 8
COLS = ["efit_times", "magnetics_time", "magnetics_p2u_current", "magnetics_p2l_current", "magnetics_efps_current",
        "magnetics_sol_current", "magnetics_plasma_current"]


def load_p2(path):
    import rule3_guard
    row = rule3_guard.read_mast_parquet(path, COLS).iloc[0]
    return p2_from_row(row)


def p2_from_row(row):
    t = np.asarray(row["magnetics_time"], float)
    v = {k: np.asarray(row[f"magnetics_{k}"], float) for k in ("p2u_current", "p2l_current", "efps_current", "sol_current", "plasma_current")}
    n = min([len(t)] + [len(x) for x in v.values()])
    t = t[:n]
    m = np.isfinite(t)
    for x in v.values():
        m &= np.isfinite(x[:n])
    return {"t": t[m], "TU": v["p2u_current"][:n][m], "TL": v["p2l_current"][:n][m], "E": v["efps_current"][:n][m],
            "S": v["sol_current"][:n][m], "ip": v["plasma_current"][:n][m], "et": np.asarray(row["efit_times"], float), "early": bool(n < 20000)}


def _rms(x):
    return float(np.sqrt(np.mean(x * x))) if len(x) else np.nan


def classify_p2(d, full=False):
    """Wiring class of one shot: 'efps' (all 20 turns in series on EFPS), 'sol' (outer windings in series with the solenoid),
    'off' (P2 never energised).  Decision = smaller rms of the two fixed-turn-count models over the whole energised record."""
    t, TU, TL, E, S = d["t"], d["TU"], d["TL"], d["E"], d["S"]
    act = (np.abs(TU) > 2.0) | (np.abs(TL) > 2.0) | (np.abs(E) > 0.2) | (np.abs(S) > 0.5)
    w = act & (t > t[0] + 5.0)
    tmax = float(max(np.abs(TU).max(), np.abs(TL).max()))
    if w.sum() < 100 or tmax < 5.0:
        out = {"cfg": "off", "margin": np.inf, "tmax": tmax}
        if not full:
            return out
        w = np.ones(len(t), bool)
    # fixed-turn-count models (no free parameter)
    rA_U, rA_L = _rms(TU[w] - 20.0 * E[w]), _rms(TL[w] - 20.0 * E[w])
    rB_U0, rB_U12 = _rms(TU[w] - 8.0 * S[w]), _rms(TU[w] - 8.0 * S[w] - 12.0 * E[w])
    rB_L0, rB_L12 = _rms(TL[w] - 8.0 * S[w]), _rms(TL[w] - 8.0 * S[w] - 12.0 * E[w])
    rA = float(np.hypot(rA_U, rA_L))
    rB = float(np.hypot(min(rB_U0, rB_U12), min(rB_L0, rB_L12)))
    if tmax < 5.0:
        cfg = "off"
    else:
        cfg = "efps" if rA <= rB else "sol"
    out = {"cfg": cfg, "margin": float(max(rA, rB) / max(min(rA, rB), 1e-9)), "tmax": tmax,
           "inner_u_on_efps": bool(rB_U12 < rB_U0), "inner_l_on_efps": bool(rB_L12 < rB_L0)}
    if not full:
        return out
    A = np.stack([E[w], S[w], np.ones(w.sum())], 1)
    cu, *_ = np.linalg.lstsq(A, TU[w], rcond=None)
    cl, *_ = np.linalg.lstsq(A, TL[w], rcond=None)
    out.update(aU=float(cu[0]), bU=float(cu[1]), aL=float(cl[0]), bL=float(cl[1]), rmsU=_rms(TU[w] - A @ cu), rmsL=_rms(TL[w] - A @ cl),
               rms_modU=rA_U if cfg != "sol" else min(rB_U0, rB_U12), rms_modL=rA_L if cfg != "sol" else min(rB_L0, rB_L12),
               rms_altU=min(rB_U0, rB_U12) if cfg != "sol" else rA_U, rms_altL=min(rB_L0, rB_L12) if cfg != "sol" else rA_L, early=d["early"])
    # U / L gain on the pre-breakdown ramp (no plasma: an up-down symmetric drive) and gains against the EFPS feed
    on = np.nonzero(np.abs(d["ip"]) > 30.0)[0]
    tb = t[on[0]] if len(on) else t[-1]
    wp = (t < tb - 2.0) & (np.abs(TL) > 20.0)
    out["g_ul"] = float(np.sum(TU[wp] * TL[wp]) / np.sum(TL[wp] ** 2)) if wp.sum() > 20 else np.nan
    we = np.abs(E) > 2.0
    out["gU"] = float(np.sum(TU[we] * E[we]) / np.sum(20.0 * E[we] ** 2)) if (cfg == "efps" and we.sum() > 20) else np.nan
    out["gL"] = float(np.sum(TL[we] * E[we]) / np.sum(20.0 * E[we] ** 2)) if (cfg == "efps" and we.sum() > 20) else np.nan
    et = d["et"]
    wf = (t >= et[0]) & (t <= et[-1])
    wd = winding_currents(d, cfg)
    for s in ("U", "L"):
        T = d["T" + s][wf]
        dd = (wd[f"in{s}"][wf] / N_IN - wd[f"out{s}"][wf] / N_OUT) if wf.any() else np.array([np.nan])
        out[f"d{s}_med"] = float(np.median(dd))
        big = np.abs(T) > 10.0
        out[f"f{s}_med"] = float(np.median(wd[f"in{s}"][wf][big] / T[big])) if big.sum() > 5 else np.nan
        out[f"T{s}_med"] = float(np.median(T)) if wf.any() else np.nan
    out["S_med"], out["E_med"] = float(np.median(S[wf])), float(np.median(E[wf]))
    return out


def winding_currents(d, cfg):
    """Ampere-turns of the four P2 windings on the shot's own magnetics samples: dict inU, outU, inL, outL [kA.turn]."""
    out = {}
    for s in ("U", "L"):
        T = d["T" + s]
        if cfg == "sol":
            o = N_OUT * d["S"]
            out["out" + s], out["in" + s] = o, T - o
        else:
            out["out" + s], out["in" + s] = (N_OUT / (N_IN + N_OUT)) * T, (N_IN / (N_IN + N_OUT)) * T
    return out


def split_d(d, cfg):
    """Per-turn current difference d = I_in - I_out [kA] per side (review follow-up A, Q-C notation)."""
    w = winding_currents(d, cfg)
    return {s: w["in" + s] / N_IN - w["out" + s] / N_OUT for s in ("U", "L")}
