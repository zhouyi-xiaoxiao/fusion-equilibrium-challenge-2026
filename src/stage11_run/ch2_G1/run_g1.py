#!/usr/bin/env python3
"""
G1 driver (v1) — Thomson-constrained free-boundary GS for every EFIT frame of a set of shots.
CPU pool, ONE TASK PER SHOT (full-shot continuation: forward pass in time order with warm starts, then a reverse
pass that retries failed / unconstrained-early frames from their later neighbour). Deterministic: no random numbers.

  python run_g1.py --machine mast --data-dir <dir of MAST parquet> --wall <mast_limiter_mask.npz> --cfg cfg.json --tag T --out <out>
  python run_g1.py --machine d3d --data-dir <d3d train dir> --holdout-names <dir>:1000:1150 --wall <d3d_limiter_mask.npz> ...

MAST processes install rule3_guard (no envelope / demo / equilibrium file can be opened; input columns only).
D3D processes may read labels; they are used ONLY for diagnostic columns of the frame table, never as a solver input.
Outputs: <out>/parts/<tag>_<machine>_<idx>.npz (psi float32 [T,65,65] EFIT layout, raw scalars, ok flags),
         <out>/frames_<tag>_<machine>.parquet, <out>/summary_<tag>_<machine>.json, optional harness-layout preds npz.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.interpolate import RectBivariateSpline

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402

OUT_SIGN = {"d3d": -1.0, "mast": 1.0}       # EFIT layout sign (D3D: axis = min, fitted on D3D; MAST: the scorer is
#                                             sign-invariant per machine and Ip > 0 on every shot -> +1 is a convention)
D3D_GAUGE = {"a_IE_kA": 0.008446511664630217, "d": 0.00041299442486922056}   # D3D-fitted E-coil gauge offset (variant J)
_S = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def get_solver(machine, row, cfg, wall_npz):
    geom, gh = D.geom_of(row)
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    key = (machine, gh, gR.tobytes(), gZ.tobytes(), float(cfg.get("sol_scale_mast", 1.0)))
    if key not in _S:
        ns = int({**C.DEFAULT_CFG, **cfg}["nsub"])
        if ns > 1:                                   # solve on a refined grid, sample back to the 65x65 output grid
            gR = np.linspace(gR[0], gR[-1], (len(gR) - 1) * ns + 1); gZ = np.linspace(gZ[0], gZ[-1], (len(gZ) - 1) * ns + 1)
        grid = C.Grid(gR, gZ)
        w = np.load(wall_npz)
        wall = C.Wall(grid, w["limiter_R"], w["limiter_Z"], up=2 if ns == 1 else 1)
        coils = C.CoilSet(geom, machine, grid,
                          sol_scale=float({**C.DEFAULT_CFG, **cfg}["sol_scale_mast"]) if machine == "mast" else 1.0)
        _S.clear()
        _S[key] = C.Solver(machine, grid, wall, coils, cfg)
    return _S[key]


def gauge(machine, inp, k):
    if machine != "d3d":
        return 0.0                                   # MAST: true flux of the coil model incl. solenoid, no offset
    ie = inp["cur"]["ECOILA"][k] / 1e3
    return D3D_GAUGE["a_IE_kA"] * (-np.sign(inp["ip_A"][k]) * ie) + D3D_GAUGE["d"]


def pe_fallback(machine, Ip, cfg):
    """Dimensionless fallback when a shot has no usable Thomson pressure at all: beta_pe0 * Bpa^2 / 2mu0."""
    a, kap = C.A_PUBLIC[machine], 1.8
    bpa = C.MU0 * Ip / (2 * np.pi * a * np.sqrt(0.5 * (1 + kap ** 2)))
    return cfg.get("beta_pe0", 0.87) * bpa ** 2 / (2 * C.MU0)


def run_shot(task):
    machine, path, idx, cfg, wall_npz, out_dir, tag, nsel = task
    part = Path(out_dir) / "parts" / f"{tag}_{machine}_{idx:05d}.npz"
    try:
        if part.exists():
            return json.loads(str(np.load(part, allow_pickle=False)["info"]))
        return _run_shot(machine, path, idx, cfg, wall_npz, part, nsel)
    except Exception as e:  # noqa: BLE001
        return [{"file": Path(path).name, "idx": idx, "fatal": f"{type(e).__name__}: {e}",
                 "tb": traceback.format_exc()[-800:]}]


def pseudo_obs(lab, idx, k, cfg, wall_rin):
    """DIII-D VALIDATION EXPERIMENT ONLY (never used for a shipped file): midplane-laser observations synthesised from
    the labelled EFIT LCFS (crossings of Z = 0) + deterministic noise, with MAST-like inner censoring, to test the
    two-edge / shaping machinery that MAST's horizontal Thomson chord uses and DIII-D's real chords cannot exercise."""
    e = lab.get("lcfs_ext")
    if e is None or k >= len(e) or not np.isfinite(e[k, 6:8]).all():
        return []
    rng = np.random.default_rng(1000003 * idx + k)
    sig, ns = cfg["sigma_s"], cfg.get("pseudo_sigma", 0.02)
    rin, rout = e[k, 6] + ns * rng.standard_normal(), e[k, 7] + ns * rng.standard_normal()
    obs = [{"name": "out", "P": (rout, 0.0), "e": (1.0, 0.0), "sig": sig, "kind": "eq"}]
    if cfg["use_inner"]:
        cov = wall_rin + 0.04                            # MAST: coverage limit 0.235 m vs centre column 0.195 m
        if rin > cov:
            obs.append({"name": "in", "P": (rin, 0.0), "e": (-1.0, 0.0), "sig": sig, "kind": "eq"})
        else:
            obs.append({"name": "in", "P": (cov, 0.0), "e": (-1.0, 0.0), "sig": sig, "kind": "ge"})
    return obs


def build_obs(machine, th, k, cfg):
    """Thomson edge observations of frame k (see Solver.solve). MAST: midplane laser (horizontal chord);
    D3D: vertical core chord at R = 1.94 m (upper LCFS crossing)."""
    chord = th["chord"]
    if chord is None or not th["has"][k]:
        return []
    sig, bias = cfg["sigma_s"], cfg["edge_bias_m"].get(machine, 0.0)
    obs = []
    if chord[0] == "h":
        zl = chord[1]
        if np.isfinite(th["edge"][k]) and not th["cens"][k]:
            obs.append({"name": "out", "P": (th["edge"][k] + bias, zl), "e": (1.0, 0.0), "sig": sig, "kind": "eq"})
        elif np.isfinite(th["covhi"][k]):
            obs.append({"name": "out", "P": (th["covhi"][k], zl), "e": (1.0, 0.0), "sig": sig, "kind": "ge"})
        if cfg["use_inner"]:
            if np.isfinite(th["rin"][k]) and not th["cin"][k]:
                obs.append({"name": "in", "P": (th["rin"][k] - bias, zl), "e": (-1.0, 0.0), "sig": sig, "kind": "eq"})
            elif np.isfinite(th["covlo"][k]):       # censored: the plasma reaches inside the coverage limit (inequality)
                obs.append({"name": "in", "P": (th["covlo"][k], zl), "e": (-1.0, 0.0), "sig": sig, "kind": "ge"})
    else:
        rc = chord[1]
        if np.isfinite(th["edge"][k]) and not th["cens"][k]:
            obs.append({"name": "out", "P": (rc, th["edge"][k] + bias), "e": (0.0, 1.0), "sig": sig, "kind": "eq"})
        elif np.isfinite(th["covhi"][k]):
            obs.append({"name": "out", "P": (rc, th["covhi"][k]), "e": (0.0, 1.0), "sig": sig, "kind": "ge"})
    return obs


def _run_shot(machine, path, idx, cfg, wall_npz, part, nsel):
    t_start = time.time()
    row = D.load_row(path, machine)
    S = get_solver(machine, row, cfg, wall_npz)
    cfg = S.cfg
    g, cs, ns = S.grid, S.coils, int(S.cfg["nsub"])
    t = np.asarray(row["efit_times"], float)
    T = len(t)
    inp = D.frame_inputs(row, machine, t)
    th = D.thomson_frames(row, machine, t, cfg)
    lab = D.labels_of(row) if machine == "d3d" else {}
    sgn = np.where(inp["ip_A"] >= 0, 1.0, -1.0)
    Ip = np.abs(inp["ip_A"])
    ks = list(range(T))
    if nsel:                                           # evenly over the WHOLE shot (ramps and TS-missing intervals included)
        ks = sorted(set(np.linspace(0, T - 1, min(nsel, T)).round().astype(int).tolist()))
    pe_ok = th["pe_max"][np.isfinite(th["pe_max"]) & (th["pe_max"] > 0)]
    pe_med = float(np.median(pe_ok)) if len(pe_ok) else None
    chord = th["chord"]

    def base_vec(k):
        return cs.vec({c: sgn[k] * inp["cur"][c][k] for c in C.current_columns(machine)})

    def frame_setup(k):
        pe, src = th["pe_max"][k], "ts"
        if not (np.isfinite(pe) and pe > 0):
            pe, src = (pe_med, "shot_median") if pe_med else (pe_fallback(machine, Ip[k], cfg), "beta_pe0")
        paxis = float(np.clip(cfg["k_paxis"] * pe, cfg["paxis_min"], cfg["paxis_max"]))
        anchor = (S.Rn, 0.0)
        if machine == "mast" and np.isfinite(th["edge"][k]) and np.isfinite(th["rin"][k]):
            anchor = (float(np.clip(0.5 * (th["edge"][k] + th["rin"][k]), 0.6, 1.1)), 0.0)
        if cfg.get("pseudo_ts") and machine == "d3d":
            return pe, src, paxis, pseudo_obs(lab, idx, k, cfg, S.wall.R_in_wall), anchor
        return pe, src, paxis, build_obs(machine, th, k, cfg), anchor

    def attempt(k, warm, rho, fast=False, zt=0.0):
        pe, src, paxis, obs, anchor = frame_setup(k)
        I = base_vec(k)
        has_eq = any(o["kind"] == "eq" for o in obs)
        theta0 = None if (has_eq and warm is not None) else (None if rho is None else rho * Ip[k])
        if fast:                                       # >= 2 consecutive failures in this shot: one cheap cold attempt
            ladder = [dict(warm=None, blend=0.5, maxits=60)]
        else:
            ladder = [dict(warm=warm)]
            if warm is not None:
                ladder.append(dict(warm=warm, blend=0.6, maxits=160))
            ladder.append(dict(warm=None, blend=0.5, maxits=160))
            if obs and cfg["edge_mode"] != "none":
                ladder.append(dict(warm=None, blend=0.5, maxits=160, edge_mode="none"))
        res, secs = None, 0.0
        for lvl, kw in enumerate(ladder):
            ok, res = S.solve(I, Ip[k], paxis, obs=obs, anchor=anchor, theta0=theta0, zt=zt, **kw)
            secs += res["secs"]
            if ok:
                if kw.get("edge_mode") == "none":
                    res["iso"] = False
                break
        res["lvl"], res["secs"] = (lvl if res["ok"] else -1), secs
        res["has_eq"] = bool(has_eq and res.get("iso"))
        if res["ok"] and (cfg["z_shrink"] > 0 or cfg["z_eta"] > 0) and abs(res.get("fz1", 0.0)) > 1e-6:
            # vertical position from the integrated force balance of the measured currents, ridge-regularised toward
            # Z = 0:  min Z^2/sZ^2 + F(Z)^2/sF^2,  F(Z) = fz0 + fz1 (Z - zt),  sF/sZ = z_eta * Bp* / a  (dimensionless z_eta)
            f0, f1 = res["fz0"], res["fz1"]
            if cfg["z_eta"] > 0:
                lam = cfg["z_eta"] * res["bps"] / C.A_PUBLIC[machine]
                z1 = f1 * (f1 * zt - f0) / (f1 ** 2 + lam ** 2)
            else:
                z1 = zt - cfg["z_shrink"] * f0 / f1
            z1 = float(np.clip(z1, -S.zt_max, S.zt_max))
            w2 = {"plasma": res["plasma"], "dIz": res["dIz"], "theta": res["theta"], "axis": res["b"]["axis"]}
            ok2, r2_ = S.solve(I, Ip[k], paxis, obs=obs, anchor=anchor, theta0=None if has_eq else res["theta"], zt=z1, warm=w2)
            if ok2:
                r2_["lvl"], r2_["secs"], r2_["has_eq"] = res["lvl"], secs + r2_["secs"], res["has_eq"]
                res = r2_
        # ---- profile closures (outer secant loops, warm re-solves):
        #   axis_mode: per-frame pressure multiplier exp(dlc) so that R_axis matches the Thomson symmetric-isotherm axis
        #              (MAST midplane laser; DIII-D: pseudo-observation in validation experiments only), prior N(0, cp_sigma^2)
        #   li_mode:   alpha_n so that the solution's li matches a target (DIII-D oracle experiments / a D3D-trained rule)
        dlc, dla = pstate.get("dlc", 0.0), pstate.get("dla", 0.0)
        res["dlc"], res["dla"] = 0.0, 0.0
        if res["ok"] and (cfg.get("axis_mode") or cfg.get("li_mode") or cfg.get("lam_mode")):
            zt_now = res.get("zt", zt)
            if cfg.get("li_mode") == "ts_rule":
                try:
                    nu_now[k] = ts_exponents(k, res["b"])
                except Exception:
                    nu_now[k] = None
            rt = axis_target(k) if cfg.get("axis_mode") else np.nan
            lt = li_target(k) if cfg.get("li_mode") else np.nan

            def resolve(dlc_, dla_, wres):
                w2 = {"plasma": wres["plasma"], "dIz": wres["dIz"], "theta": wres["theta"], "axis": wres["b"]["axis"]}
                pax = float(np.clip(paxis * np.exp(dlc_), cfg["paxis_min"], cfg["paxis_max"]))
                ok2, r2_ = S.solve(I, Ip[k], pax, obs=obs, anchor=anchor, theta0=None if has_eq else wres["theta"], zt=zt_now,
                                   warm=w2, an=float(cfg["alpha_n"] * np.exp(dla_)))
                xs[0] += r2_["secs"]
                return ok2, r2_

            xs = [0.0]

            def li_of(r):
                return C.scalars(S.grid, S.wall, r["psi"], r["b"], r["prof"], inp["fvac"][k], Ip[k]).get("li", np.nan)

            cur, n_extra = res, 0
            if dlc or dla:                               # start from the previous frame's profile state
                ok2, r2_ = resolve(dlc, dla, cur)
                n_extra += 1
                if ok2:
                    cur = r2_
                else:
                    dlc = dla = 0.0
            sc_, sa_ = pstate.get("sc", cfg.get("axis_s0", 0.03)), pstate.get("sa", cfg.get("li_s0", 0.35))
            sl_ = pstate.get("sl", cfg.get("lam_s0", 0.4))
            lam = cfg.get("lam_mode") if has_eq else None
            for _ in range(int(cfg.get("prof_its", 2))):
                moved = False
                if lam and np.isfinite(cur.get("delta", {}).get("out", np.nan)):
                    # radial force balance closure: with the MEASURED coil currents (virtual field prior kept tight) the outer
                    # edge position fixes Lambda = beta_p + li/2; attribute it to the pressure multiplier ('cp') or alpha_n ('an')
                    r0 = cur["delta"]["out"]
                    x0, sx, lo_, hi_ = (dlc, cfg["cp_sigma"], -cfg["cp_clip"], cfg["cp_clip"]) if lam == "cp" else (dla, cfg.get("an_sigma", 0.5), -1.0, 1.4)
                    w = min(1.0, cfg["huber"] * cfg["sigma_s"] / max(abs(r0), 1e-9))
                    d = -(w * sl_ * r0 / cfg["sigma_s"] ** 2 + x0 / sx ** 2) / (w * sl_ ** 2 / cfg["sigma_s"] ** 2 + 1 / sx ** 2)
                    d = float(np.clip(d, -0.35, 0.35)); new = float(np.clip(x0 + d, lo_, hi_)); d = new - x0
                    if abs(d) > 0.02:
                        ok2, r2_ = resolve(new, dla, cur) if lam == "cp" else resolve(dlc, new, cur)
                        n_extra += 1
                        if ok2 and np.isfinite(r2_.get("delta", {}).get("out", np.nan)):
                            s_new = (r2_["delta"]["out"] - r0) / d
                            if 0.02 < s_new < 5.0:
                                sl_ = s_new
                            cur, moved = r2_, True
                            if lam == "cp":
                                dlc = new
                            else:
                                dla = new
                if np.isfinite(rt):
                    r0 = cur["b"]["axis"][0] - rt
                    w = min(1.0, cfg["huber"] * cfg["axis_sigma"] / max(abs(r0), 1e-9))
                    d = -(w * sc_ * r0 / cfg["axis_sigma"] ** 2 + dlc / cfg["cp_sigma"] ** 2) / (w * sc_ ** 2 / cfg["axis_sigma"] ** 2 + 1 / cfg["cp_sigma"] ** 2)
                    d = float(np.clip(d, -0.5, 0.5)); new = float(np.clip(dlc + d, -cfg["cp_clip"], cfg["cp_clip"])); d = new - dlc
                    if abs(d) > 0.03:
                        ok2, r2_ = resolve(new, dla, cur); n_extra += 1
                        if ok2:
                            s_new = (r2_["b"]["axis"][0] - rt - r0) / d
                            if 0.003 < s_new < 0.3:
                                sc_ = s_new
                            cur, dlc, moved = r2_, new, True
                if np.isfinite(lt):
                    l0 = li_of(cur)
                    if np.isfinite(l0):
                        d = float(np.clip((lt - l0) / sa_, -0.6, 0.6)); new = float(np.clip(dla + d, -1.0, 1.4)); d = new - dla
                        if abs(d) > 0.03:
                            ok2, r2_ = resolve(dlc, new, cur); n_extra += 1
                            if ok2:
                                l1 = li_of(r2_)
                                if np.isfinite(l1) and 0.05 < (l1 - l0) / d < 2.0:
                                    sa_ = (l1 - l0) / d
                                cur, dla, moved = r2_, new, True
                if not moved:
                    break
            if cur is not res:
                cur["lvl"], cur["has_eq"] = res["lvl"], res["has_eq"]
                cur["secs"] = secs + xs[0]
                res = cur
            res["dlc"], res["dla"], res["n_extra"] = dlc, dla, n_extra
            pstate.update({"dlc": dlc, "dla": dla, "sc": sc_, "sa": sa_, "sl": sl_})
            paxis = float(np.clip(paxis * np.exp(dlc), cfg["paxis_min"], cfg["paxis_max"]))
        return res, dict(pe=pe, pe_src=src, paxis=paxis, I=I, obs=obs)

    pstate = {}

    def axis_target(k):
        if machine == "mast":
            return th["rsym"][k] if (np.isfinite(th["rsym"][k]) and th["has"][k]) else np.nan
        if cfg.get("pseudo_ts") and "efit_r_axis" in lab and k < len(lab["efit_r_axis"]):       # D3D validation only
            return lab["efit_r_axis"][k] + cfg.get("pseudo_axis_sigma", 0.015) * np.random.default_rng(7000003 * idx + k).standard_normal()
        return np.nan

    def li_target(k):
        if cfg.get("li_mode") == "oracle" and machine == "d3d" and "efit_li" in lab and k < len(lab["efit_li"]):   # D3D upper bound only
            return lab["efit_li"][k] * cfg.get("li_scale", 1.0)
        if cfg.get("li_mode") == "ts_rule":             # D3D-train fit: li ~ c0 + c1 nuT + c2 nuP (Thomson exponents in psiN)
            nu = nu_now.get(k)
            if nu is not None:
                c = cfg["li_rule"]
                return float(np.clip(c[0] + c[1] * np.clip(nu[0], 0, 6) + c[2] * np.clip(nu[1], 0, 6), cfg.get("li_lo", 0.6), cfg.get("li_hi", 1.6)))
        return np.nan

    nu_now = {}

    def ts_exponents(k, b):
        pr = th["prof"][k]
        if pr is None:
            return None
        u = (b["spl"].ev(pr[0], pr[1]) - b["psi_axis"]) / (b["psi_b"] - b["psi_axis"])
        m = np.isfinite(u) & np.isfinite(pr[2]) & np.isfinite(pr[3]) & (pr[2] > 0) & (pr[3] > 0) & (u > 0.03) & (u < 0.9)
        if m.sum() < 8 or u[m].max() - u[m].min() <= 0.4:
            return None
        x = np.log(1.0 - u[m])
        return float(np.polyfit(x, 1.5 * np.log(pr[2][m]), 1)[0]), float(np.polyfit(x, np.log(pr[2][m] * pr[3][m]), 1)[0])

    def warm_of(res):
        return {"plasma": res["plasma"], "dIz": res["dIz"], "theta": res["theta"], "axis": res["b"]["axis"]}

    psi_x = np.full((T, g.nx, g.ny), np.nan, np.float32 if ns > 1 else np.float64)   # psi_total - vacuum(I_base), ok frames
    q95 = np.full(T, np.nan); betaN = np.full(T, np.nan)
    okf = np.zeros(T, bool)
    axR = np.full(T, np.nan); axZ = np.full(T, np.nan)
    recs, state = {}, {}
    for k in ks:
        recs[k] = {"file": Path(path).name, "idx": idx, "k": int(k), "t": float(t[k]), "ip_kA": float(inp["ip_A"][k] / 1e3),
                   "tf_kA": float(inp["tf_kA"][k]), "ts_edge": float(th["edge"][k]), "ts_cens": bool(th["cens"][k]),
                   "ts_rin": float(th["rin"][k]), "ts_cin": bool(th["cin"][k]), "ts_rsym": float(th["rsym"][k]),
                   "ts_has": bool(th["has"][k]), "ok": False, "err": "", "lvl": -9, "pass": 0}

    def store(k, res, su, pas):
        _store(S, machine, k, res, su, recs[k], psi_x, q95, betaN, okf, inp, chord, lab, th["prof"])
        recs[k]["pass"] = pas
        if res["ok"]:
            axR[k], axZ[k] = res["b"]["axis"]
            state[k] = (warm_of(res), (np.array(res["theta"]) / Ip[k]) if res["has_eq"] and not res["clip"] else None)

    # ---- forward pass
    warm, rho, nfail = None, None, 0
    for k in ks:
        if Ip[k] < cfg["ip_min"]:
            recs[k].update({"err": "Ip<min", "lvl": -2})
            continue
        res, su = attempt(k, warm, rho, fast=nfail >= 2)
        nfail = 0 if res["ok"] else nfail + 1
        store(k, res, su, 1)
        if res["ok"]:
            warm = state[k][0]
            rho = state[k][1] if state[k][1] is not None else rho
        else:
            warm = None
    # ---- reverse pass: failed frames, and frames solved without any radial information before the first constrained one
    nxt, rho_n = None, None
    for k in reversed(ks):
        if Ip[k] < cfg["ip_min"]:
            continue
        r = recs[k]
        redo = (not r["ok"]) or (r.get("rho_src") == "none" and rho_n is not None and cfg["edge_mode"] != "none")
        if redo and nxt is not None:
            res, su = attempt(k, state[nxt][0], rho_n)
            if res["ok"]:
                store(k, res, su, 2)
        if recs[k]["ok"]:
            nxt = k
            if state[k][1] is not None:
                rho_n = state[k][1]
    # ---- assemble every frame: exact vacuum of the frame + (converged | time-interpolated plasma + virtual part / Ip)
    psi_out = np.zeros((T, (g.ny - 1) // ns + 1, (g.nx - 1) // ns + 1), np.float32)
    good = np.where(okf)[0]
    for k in range(T):
        if nsel and k not in recs:
            continue
        vac = S.vacuum(base_vec(k))
        if okf[k]:
            px = psi_x[k].astype(float)
        elif len(good):
            # fallback: exact vacuum of THIS frame + plasma/control flux of the converged neighbours (interpolated in
            # time per unit Ip), re-centred with two vacuum modes so that the O-point sits at the neighbours' axis
            j1 = int(np.searchsorted(good, k))
            lo = good[j1 - 1] if j1 > 0 else None
            hi = good[j1] if j1 < len(good) else None
            jn = lo if hi is None else (hi if lo is None else (lo if t[k] - t[lo] <= t[hi] - t[k] else hi))
            if Ip[k] < cfg["ip_min"]:                    # no usable Ip: neighbour's plasma flux unscaled
                px, Rt, Zt = psi_x[jn].astype(float), axR[jn], axZ[jn]
            elif lo is not None and hi is not None:
                w = (t[k] - t[lo]) / max(t[hi] - t[lo], 1e-9)
                px = Ip[k] * ((1 - w) * psi_x[lo].astype(float) / Ip[lo] + w * psi_x[hi].astype(float) / Ip[hi])
                Rt, Zt = (1 - w) * axR[lo] + w * axR[hi], (1 - w) * axZ[lo] + w * axZ[hi]
            else:
                px, Rt, Zt = psi_x[jn].astype(float) * (Ip[k] / Ip[jn]), axR[jn], axZ[jn]
            recs[k]["fill"] = "interp"
            try:
                spl = RectBivariateSpline(g.gR, g.gZ, vac + px)
                mV, mR = 0.5 * g.R ** 2, g.R ** 2 * (g.Z - Zt)
                corr = (-float(spl.ev(Rt, Zt, dx=1)) / Rt) * (mV - mV.mean()) + (-float(spl.ev(Rt, Zt, dy=1)) / Rt ** 2) * (mR - mR.mean())
                bb = C.analyse(g, S.wall, vac + px + corr, (Rt, Zt), 0.15)
                if bb is not None and np.hypot(bb["axis"][0] - Rt, bb["axis"][1] - Zt) < 0.05:
                    px, recs[k]["fill"] = px + corr, "interp_rc"
                else:                                    # last resort: total flux of the nearest converged frame
                    px, recs[k]["fill"] = psi_x[jn].astype(float) + S.vacuum(base_vec(jn)) - vac, "stale"
            except Exception:
                pass
        else:
            px = S.cold_plasma(max(Ip[k], cfg["ip_min"]), S.Rn, 0.0)
            recs[k]["fill"] = "cold"
        full = OUT_SIGN[machine] * (vac + px)
        psi_out[k] = full[::ns, ::ns].T + gauge(machine, inp, k)
        if "psi" in lab and k < len(lab["psi"]) and np.isfinite(lab["psi"][k]).all():      # D3D diagnostics only
            gt = lab["psi"][k].astype(np.float64)
            d = psi_out[k].astype(np.float64) - gt
            dv = (OUT_SIGN[machine] * vac)[::ns, ::ns].T + gauge(machine, inp, k) - gt
            recs[k].update({"r2_frame": float(1 - (d ** 2).sum() / ((gt - gt.mean()) ** 2).sum()), "npx": int(gt.size),
                            "gt_s1": float(gt.sum()), "gt_s2": float((gt ** 2).sum()), "d_mean": float(d.mean()),
                            "d_sse": float((d ** 2).sum()), "dv_mean": float(dv.mean()), "dv_sse": float((dv ** 2).sum())})
    info = [recs[k] for k in sorted(recs)]
    for r in info:
        r["shot_secs"] = time.time() - t_start
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, psi=psi_out, q95=q95, betaN=betaN, ok=okf, t=t, ip_A=inp["ip_A"], ks=np.array(sorted(recs)),
                        info=json.dumps(info, default=float))
    os.replace(tmp, part)
    return info


def _store(S, machine, k, res, su, rec, psi_x, q95, betaN, okf, inp, chord, lab, th_prof=None):
    th = res.get("theta", [])
    rec.update({"ok": bool(res["ok"]), "err": res["err"], "lvl": res["lvl"], "n_it": res["n_it"], "secs": res["secs"],
                "iso": bool(res.get("iso")), "has_eq": bool(res.get("has_eq")), "bps": res.get("bps", np.nan),
                "bv": th[0] if len(th) else 0.0, "bq": th[1] if len(th) > 1 else 0.0, "clip": bool(res.get("clip", False)),
                "gam": res.get("gam", 1.0), "dIz_frac": res.get("dIz", 0.0) / max(abs(inp["ip_A"][k]), 1.0),
                "zt": res.get("zt", 0.0), "dlc": res.get("dlc", 0.0), "dla": res.get("dla", 0.0), "pe": su["pe"], "pe_src": su["pe_src"], "paxis": su["paxis"],
                "n_obs": len(su["obs"]), "rho_src": "eq" if res.get("has_eq") else ("carried" if any(abs(x) > 0 for x in th) else "none"),
                **{f"d_{n}": float(v) for n, v in res.get("delta", {}).items()},
                **{n: res.get(n, np.nan) for n in ("fz0", "fz1", "fzfb")}})
    for o in su["obs"]:
        rec[f"obs_{o['name']}"] = o["kind"]
    if not res["ok"]:
        return
    b, prof = res["b"], res["prof"]
    psi_x[k] = res["psi"] - S.vacuum(su["I"])
    okf[k] = True
    sc = C.scalars(S.grid, S.wall, res["psi"], b, prof, inp["fvac"][k], abs(inp["ip_A"][k]), chord=chord)
    q95[k], betaN[k] = sc.get("q95", np.nan), sc.get("betaN", np.nan)
    pr = th_prof[k] if th_prof is not None else None
    if pr is not None:                                   # Thomson profiles mapped onto the solution's flux coordinate
        try:
            u = (b["spl"].ev(pr[0], pr[1]) - b["psi_axis"]) / (b["psi_b"] - b["psi_axis"])
            m = np.isfinite(u) & np.isfinite(pr[2]) & np.isfinite(pr[3]) & (pr[2] > 0) & (pr[3] > 0) & (u > 0.03) & (u < 0.9)
            if m.sum() >= 8 and u[m].max() - u[m].min() > 0.4:
                x = np.log(1.0 - u[m])
                rec["ts_nuT"] = float(np.polyfit(x, 1.5 * np.log(pr[2][m]), 1)[0])      # Te^1.5 ~ (1 - psiN)^nuT (Spitzer J shape)
                rec["ts_nuP"] = float(np.polyfit(x, np.log(pr[2][m] * pr[3][m]), 1)[0])  # p_e ~ (1 - psiN)^nuP
                rec["ts_umin"], rec["ts_npt"] = float(u[m].min()), int(m.sum())
        except Exception:
            pass
    rec.update({"gs_Rax": b["axis"][0], "gs_Zax": b["axis"][1], "gs_limited": b["limited"], "gs_kind": b["kind"],
                "gs_nx": len(b["xact"]), "psi_axis": b["psi_axis"], "psi_b": b["psi_b"], "Beta0": prof.Beta0,
                "paxis_eff": prof.paxis_eff, "fvac": float(inp["fvac"][k]), **{f"gs_{a}": v for a, v in sc.items()}})
    if b["bpt"] is not None:
        rec.update({"gs_bR": b["bpt"][0], "gs_bZ": b["bpt"][1]})
    if b["xact"]:
        xl = min(b["xact"], key=lambda x: x[1])
        rec.update({"gs_Rx": xl[0], "gs_Zx": xl[1]})
    if "lcfs_ext" in lab and k < len(lab["lcfs_ext"]):
        e = lab["lcfs_ext"][k]
        rec.update({"gt_rin": e[0], "gt_rout": e[1], "gt_ztop": e[2], "gt_zbot": e[3], "gt_r_zbot": e[4], "gt_r_ztop": e[5]})
    for name, key in (("efit_q95", "gt_q95"), ("efit_beta_n", "gt_betaN"), ("efit_r_axis", "gt_raxis"),
                      ("efit_z_axis", "gt_zaxis"), ("efit_li", "gt_li")):
        if name in lab and k < len(lab[name]):
            rec[key] = float(lab[name][k])


def select_shots(args):
    ddir = Path(args.data_dir)
    if args.holdout_names:
        d, a, b = args.holdout_names.rsplit(":", 2)
        names = sorted(p.name for p in Path(d).glob("*.parquet"))[int(a):int(b)]
        files = [ddir / n for n in names]
    elif args.exclude_holdout:
        d, a, b = args.exclude_holdout.rsplit(":", 2)
        hs = set(sorted(p.name for p in Path(d).glob("*.parquet"))[int(a):int(b)])
        files = [p for p in sorted(ddir.glob("*.parquet")) if p.name not in hs]
    else:
        files = sorted(ddir.glob("*.parquet"))
    if args.shots:
        a, b = (int(x) for x in args.shots.split(":"))
        files = files[a:b]
    if args.stride > 1:
        files = files[::args.stride]
    return list(enumerate(files))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["d3d", "mast"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--wall", required=True, help="npz with limiter_R / limiter_Z (public first wall)")
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--set", action="append", default=[], help="override cfg entries: key=value (json value)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default=None, help="a:b slice of the selected (sorted) file list")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--holdout-names", default=None, help="<dir>:a:b -> sorted names of <dir>[a:b] taken from --data-dir")
    ap.add_argument("--exclude-holdout", default=None, help="<dir>:a:b -> all files of --data-dir except those names")
    ap.add_argument("--frames-per-shot", type=int, default=0)
    ap.add_argument("--preds", action="store_true", help="also write the harness-layout preds npz (D3D)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
    if args.machine == "mast":
        import rule3_guard
        rule3_guard.install()
    cfg = json.load(open(args.cfg)) if args.cfg else {}
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    items = select_shots(args)
    (out / f"files_{args.tag}_{args.machine}.txt").write_text("\n".join(p.name for _, p in items) + "\n")
    json.dump({**C.DEFAULT_CFG, **cfg}, open(out / f"cfg_{args.tag}_{args.machine}.json", "w"), indent=1)
    log(f"G1 {args.machine} tag={args.tag} shots={len(items)} procs={args.procs} cfg={json.dumps(cfg)}")
    tasks = [(args.machine, str(p), i, cfg, args.wall, str(out), args.tag, args.frames_per_shot) for i, p in items]
    t0 = time.time()
    recs = []
    with get_context("fork").Pool(args.procs) as pool:
        for n, r in enumerate(pool.imap_unordered(run_shot, tasks, chunksize=1)):
            recs.extend(r)
            if (n + 1) % max(1, len(tasks) // 10) == 0 or n == len(tasks) - 1:
                log(f"  {n + 1}/{len(tasks)} shots, {len(recs)} frames, {time.time() - t0:.0f}s")
    import pandas as pd
    df = pd.DataFrame(recs)
    for c in df.columns:
        if df[c].dtype == object and c not in ("file", "err", "pe_src", "fill", "fatal", "tb", "gs_kind", "rho_src",
                                               "obs_out", "obs_in"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(out / f"frames_{args.tag}_{args.machine}.parquet")
    wall_s = time.time() - t0
    summ = {"tag": args.tag, "machine": args.machine, "n_shots": len(items), "n_frames": int(len(df)),
            "wall_s": wall_s, "procs": args.procs, "cpu_core_hours": wall_s * args.procs / 3600.0}
    if "fatal" in df:
        summ["n_fatal"] = int(df["fatal"].notna().sum())
    if "ok" in df:
        d = df[df["ok"].notna()]
        pl = d[d["err"] != "Ip<min"]
        okd = pl[pl["ok"].astype(bool)]
        summ.update({"plasma_frames": int(len(pl)), "converged": float(len(okd) / max(len(pl), 1))})
        for c in ("gs_limited", "has_eq", "clip"):
            if c in okd:
                summ[f"frac_{c}"] = float(okd[c].astype(float).mean())
        if "fill" in d:
            summ["fill_counts"] = {str(a): int(b) for a, b in d["fill"].value_counts().items()}
        if "pass" in okd:
            summ["pass_counts"] = {str(int(a)): int(b) for a, b in okd["pass"].value_counts().items()}
    json.dump(summ, open(out / f"summary_{args.tag}_{args.machine}.json", "w"), indent=1)
    log("SUMMARY " + json.dumps(summ))
    if args.preds:
        preds = {}
        for j, (i, p) in enumerate(items):
            c = np.load(out / "parts" / f"{args.tag}_{args.machine}_{i:05d}.npz", allow_pickle=False)
            q, b = c["q95"].astype(float), c["betaN"].astype(float)
            tt = np.arange(len(q), dtype=float)
            for arr, dflt in ((q, 4.0), (b, 1.5)):
                m = np.isfinite(arr)
                arr[~m] = np.interp(tt[~m], tt[m], arr[m]) if m.any() else dflt
            preds[f"shot_{j:04d}_psirz"] = c["psi"].astype(np.float32)
            preds[f"shot_{j:04d}_q95"] = q.astype(np.float32)
            preds[f"shot_{j:04d}_betaN"] = b.astype(np.float32)
        np.savez(out / f"preds_{args.machine}_{args.tag}_f32.npz", **preds)
        for k in list(preds):
            if k.endswith("_psirz"):
                preds[k] = preds[k].astype(np.float16).astype(np.float32)          # as serialised in a submission
        np.savez(out / f"preds_{args.machine}_{args.tag}.npz", **preds)
        log(f"wrote preds_{args.machine}_{args.tag}.npz (+ _f32)")
    log(f"done in {wall_s:.0f}s wall ({summ['cpu_core_hours']:.2f} core-h)")


if __name__ == "__main__":
    main()
