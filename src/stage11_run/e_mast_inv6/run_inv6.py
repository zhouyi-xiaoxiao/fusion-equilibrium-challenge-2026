#!/usr/bin/env python3
"""
Stage 11E / Ch2 INV6 — driver of the kinetic inverse on top of G1 (k = 0.5) = ch2d_inverse/run_inv.py (INV_P5) plus
  * density pairs with correlated errors (inv6_core.residuals6; cfg ne_levels / ne_floor / ne_rho / ne_ell),
  * multi-start + bidirectional continuation for AMBIGUOUS frames (non-X-point-bounded, G1 fallback, failed, elongation
    spike vs the +-2 neighbours): candidates = forward solution, reverse continuation from the later neighbour, cold G1 start
    at the prior, cold G1 start at the forward theta; selection = lowest objective, and among candidates within ms_dcost of
    it (statistically indistinguishable) prefer X-point-bounded (limited frames were removed from the corpus: organiser
    FAQ + README) and then shape continuity with the neighbours (equilibria evolve smoothly over the 5 ms frame spacing),
  * labelfit with the MAST-like pseudo-horizontal p_e (cfg lf_pe = 'pseudo') + coverage records (coverage-aware c_p prior),
  * per-frame p_e,max (measured) / p_e,axis (inferred) / <p_e>_V (fitted, volume average) records.
With ne_levels = [] and ms = false INV6 reproduces INV_P5 (same code path; observation cache only).
Original header of run_inv.py:
Stage 11D / Ch2 — driver of the small regularised kinetic inverse (inv_core.py) on top of G1 (k = 0.5).
CPU pool, one task per shot: forward pass in time order (warm start of solution AND parameters along the shot), reverse
pass for failed frames and for frames solved before the first data-constrained frame, then G1's frame assembly (exact
vacuum of every frame + converged plasma/control flux, or the time-interpolated per-Ip fallback).

  --mode real     MAST: midplane laser isotherm pairs + soft outer edge + inner edge (soft when uncensored, inequality
                  when censored) | D3D: real chords (vertical-chord upper edge + vertical/tangential isotherm pairs)
  --mode pseudo   D3D VALIDATION ONLY: MAST-like midplane laser synthesised from the EFIT surfaces (+ stresses)
  --mode labelfit D3D ONLY: theta fitted to the EFIT flux shape (learned priors for the kinetic family)

MAST processes install rule3_guard (no envelope / demo / equilibrium store can be opened; input columns only).
D3D labels are read only in D3D processes: as diagnostics, for the pseudo-data synthesis and for labelfit.
Outputs (G1 layout, so eval_frames.py / package_g1.py / lcfs_check.py / dev_summary.py work unchanged):
  <out>/parts/<tag>_<machine>_<idx>.npz, <out>/frames_<tag>_<machine>.parquet, <out>/summary_..., preds npz (--preds).
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
G1DIR = HERE.parent / "ch2_G1"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "ch2d_inverse"))
sys.path.insert(0, str(G1DIR))
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402
import run_g1 as RG  # noqa: E402
import inv_core as K  # noqa: E402
import inv6_core as K6  # noqa: E402

INV6_DEFAULT = {"ne_levels": [], "ne_floor": 0.03, "ne_rho": 0.5, "ne_ell": 0.02, "lf_pe": "real",
                "ms": False, "ms_dcost": 2.0, "ms_kspike": 0.15, "ms_cost_per_obs": 0.0, "ms_prefer_xpt": True,
                "ms_prefer_cont": True}

E_CH = 1.602176634e-19


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


# ----------------------------------------------------------------------------------------------- Thomson helpers
def d3d_chords(row, t, gap):
    """DIII-D real chords per frame: vertical core chord (R, Z, Te, ne) and tangential chord (R, Z, Te, ne)."""
    import thomson_geom as TG
    T = len(t)
    vert, tang = [None] * T, [None] * T
    names = np.asarray(row["thomson_chord_name"]).astype(str)
    cR = np.asarray(row["thomson_chord_R"], float); cZ = np.asarray(row["thomson_chord_Z"], float)
    core = np.array([n.startswith("TS_core") for n in names]); tg = np.array([n.startswith("TS_tangential") for n in names])
    cTe, cne = D._stack(row["thomson_core_Te"]), D._stack(row["thomson_core_ne"])
    if cTe.ndim == 2 and cTe.size:
        zc = cZ[core]; n = min(cTe.shape[1], len(zc)); cTe, cne, zc = cTe[:, :n], cne[:, :n], zc[:n]
        rc = np.full(n, float(np.nanmedian(cR[core])))
        ic = TG.nearest_rows(np.asarray(row["thomson_core_times"], float)[:cTe.shape[0]], t, gap)
        for k in range(T):
            if ic[k] >= 0:
                vert[k] = (rc, zc, cTe[ic[k]], cne[ic[k]])
    eTe, ene = D._stack(row["thomson_edge_Te"]), D._stack(row["thomson_edge_ne"])
    if eTe.ndim == 2 and eTe.size:
        rt = cR[tg]; n = min(eTe.shape[1], len(rt)); eTe, ene, rt = eTe[:, :n], ene[:, :n], rt[:n]
        zt = cZ[tg][:n]
        it = TG.nearest_rows(np.asarray(row["thomson_edge_times"], float)[:eTe.shape[0]], t, gap)
        for k in range(T):
            if it[k] >= 0:
                tang[k] = (rt, zt, eTe[it[k]], ene[it[k]])
    return vert, tang


def pe_points(*profs, drop_bands=None):
    """Thomson (R, Z, p_e) points; drop_bands = Te/Te_max levels whose +-0.05 bands are removed (held-out validation)."""
    R, Z, P = [], [], []
    tmax = None
    if drop_bands:
        tm = [np.nanmax(K._smooth(np.asarray(p[2], float)[np.isfinite(p[2]) & (np.asarray(p[2], float) > 0)]))
              for p in profs if p is not None and (np.isfinite(p[2]) & (np.asarray(p[2], float) > 0)).sum() >= 3]
        tmax = max(tm) if tm else None
    for p in profs:
        if p is None:
            continue
        r, z, te, ne = (np.asarray(a, float) for a in p)
        z = np.broadcast_to(z, r.shape)
        m = np.isfinite(r) & np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        if tmax:
            for f in drop_bands:
                m &= ~(np.abs(te / tmax - f) < 0.05)
        R.append(r[m]); Z.append(z[m]); P.append(E_CH * te[m] * ne[m])
    if not R:
        return None
    R, Z, P = np.concatenate(R), np.concatenate(Z), np.concatenate(P)
    return (R, Z, P) if len(R) >= 8 else None


# ----------------------------------------------------------------------------------------------- D3D pseudo midplane laser
def efit_frame(row, lab, k):
    """EFIT flux of frame k in the solver convention (axis = MAX, (R, Z) layout) + psi_a, psi_b."""
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    psi = -lab["psi"][k].astype(np.float64).T                  # D3D EFIT axis = min -> flip
    spl = RectBivariateSpline(gR, gZ, psi)
    ra, za = float(lab["efit_r_axis"][k]), float(lab["efit_z_axis"][k])
    lr = np.asarray(row["efit_lcfs_r"][k], float); lz = np.asarray(row["efit_lcfs_z"][k], float)
    ln = np.asarray(row["efit_lcfs_n"]).ravel()
    if np.isfinite(ln[k]):
        lr, lz = lr[:int(ln[k])], lz[:int(ln[k])]
    m = np.isfinite(lr) & np.isfinite(lz) & (lr > 0)
    if m.sum() < 8:
        return None
    pa = float(spl.ev(ra, za)); pb = float(np.median(spl.ev(lr[m], lz[m])))
    if not pa > pb:
        return None
    return spl, pa, pb, ra, za


def _fit_level(u, y):
    """smooth monotone fit y(u) on [0, 1] -> function of u (any u >= 0; SOL: exponential decay)."""
    t = K.PeTable.fit(u, y, umax=1.0, taper=1e-6, min_pts=6, min_span=0.3)
    if t is None:
        return None
    edge = max(t.edge, 1e-3 * float(t.vals[0]))
    return t, edge


def pseudo_frame(row, lab, k, t, vert, tang, stress, idx, cov_lo, rng):
    """Synthesise a MAST-like midplane laser (Z = 0, 1.05 cm channels from cov_lo to 2.45 m) on DIII-D frame k from the
    EFIT surfaces and the frame's real Thomson profiles mapped to EFIT psi_N. Returns (prof_k, edges, info) or None."""
    ef = efit_frame(row, lab, k)
    if ef is None:
        return None
    spl, pa, pb, ra, za = ef
    kk = k
    if stress.get("timing", 0):                                    # +-10 ms profile timing error (alternating per shot)
        dt = stress["timing"] * (1 if idx % 2 == 0 else -1)
        kk = int(np.argmin(np.abs(t - (t[k] + dt))))
    pts = pe_points(vert[kk], tang[kk])
    if pts is None:
        return None
    R_, Z_ = [], []
    te_, ne_ = [], []
    for p in (vert[kk], tang[kk]):
        if p is None:
            continue
        r, z, te, ne = (np.asarray(a, float) for a in p)
        m = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
        R_.append(r[m]); Z_.append(np.broadcast_to(z, r.shape)[m]); te_.append(te[m]); ne_.append(ne[m])
    R_, Z_, te_, ne_ = map(np.concatenate, (R_, Z_, te_, ne_))
    u = (pa - spl.ev(R_, Z_)) / (pa - pb)
    ft, fn = _fit_level(u, te_), _fit_level(u, ne_)
    if ft is None or fn is None:
        return None
    xr = np.arange(cov_lo, 2.45, 0.0105)
    uc = (pa - spl.ev(xr, np.zeros_like(xr))) / (pa - pb)

    def ev(f, lam):
        tab, edge = f
        ui = np.clip(uc, 0, 1)
        y = tab.pe(ui) + edge
        out = np.where(uc > 1, edge * np.exp(-(uc - 1) / lam), y)
        return out

    Te, ne = ev(ft, 0.04), ev(fn, 0.08)
    if stress.get("noise", 0):
        Te = Te * (1 + stress["noise"] * rng.standard_normal(len(xr)))
        ne = ne * (1 + stress["noise"] * rng.standard_normal(len(xr)))
    if stress.get("branch", 0):
        s = 1 if (idx // 2) % 2 == 0 else -1
        inb = xr < ra
        Te = np.where(inb, Te * (1 + s * stress["branch"]), Te)
    Te = np.clip(Te, 1e-3, None); ne = np.clip(ne, 1e15, None)
    e = lab.get("lcfs_ext")
    if e is None or not np.isfinite(e[k, 6:8]).all():
        return None
    rin, rout = float(e[k, 6]), float(e[k, 7])
    so = stress.get("sep", 0.0) * (1 if (idx // 4) % 2 == 0 else -1)
    en = stress.get("edge_noise", 0.0)
    rout_o = rout + so + en * rng.standard_normal()
    rin_o = rin - so + en * rng.standard_normal()
    edges = [{"name": "out", "P": (rout_o, 0.0), "e": (1.0, 0.0), "sig": None, "kind": "eq"}]
    if rin_o > cov_lo:
        edges.append({"name": "in", "P": (rin_o, 0.0), "e": (-1.0, 0.0), "sig": None, "kind": "eq"})
    else:
        edges.append({"name": "in", "P": (cov_lo, 0.0), "e": (-1.0, 0.0), "sig": None, "kind": "ge"})
    return (xr, np.zeros_like(xr), Te, ne), edges, {"efit_rin": rin, "efit_rout": rout}


# ----------------------------------------------------------------------------------------------- one shot
def run_shot(task):
    machine, path, idx, cfg, wall_npz, out_dir, tag, nsel, mode = task
    part = Path(out_dir) / "parts" / f"{tag}_{machine}_{idx:05d}.npz"
    try:
        if part.exists():
            return json.loads(str(np.load(part, allow_pickle=False)["info"]))
        return _run_shot(machine, path, idx, cfg, wall_npz, part, nsel, mode)
    except Exception as e:  # noqa: BLE001
        return [{"file": Path(path).name, "idx": idx, "fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1200:]}]


def _run_shot(machine, path, idx, ucfg, wall_npz, part, nsel, mode):
    t_start = time.time()
    row = D.load_row(path, machine)
    S = RG.get_solver(machine, row, ucfg, wall_npz)
    cfg = {**K.INV_DEFAULT, **INV6_DEFAULT, **S.cfg}
    shape_info = K.install_shape_mode(S, machine) if cfg.get("shape_mode") == "coil" else None
    g, cs = S.grid, S.coils
    t = np.asarray(row["efit_times"], float)
    T = len(t)
    inp = D.frame_inputs(row, machine, t)
    th = D.thomson_frames(row, machine, t, cfg)
    lab = D.labels_of(row) if machine == "d3d" else {}
    if machine == "d3d":
        vert, tang = d3d_chords(row, t, cfg["th_gap_ms"])
    sgn = np.where(inp["ip_A"] >= 0, 1.0, -1.0)
    Ip = np.abs(inp["ip_A"])
    ks = list(range(T))
    if nsel:
        ks = sorted(set(np.linspace(0, T - 1, min(nsel, T)).round().astype(int).tolist()))
    chord = th["chord"]
    pe_ok = th["pe_max"][np.isfinite(th["pe_max"]) & (th["pe_max"] > 0)]
    pe_med = float(np.median(pe_ok)) if len(pe_ok) else None
    iq = cs.index["VQ"]
    levels = list(cfg["fit_levels"] if cfg.get("fit_levels") else cfg["levels"])
    val_levels = list(cfg.get("val_levels") or [])
    stress = cfg.get("stress", {})
    rng = np.random.default_rng(7919 * idx + 17)
    cov_lo = S.wall.R_in_wall + 0.04
    a_pub, zt_max, z_eta = C.A_PUBLIC[machine], S.zt_max, cfg["z_eta"]

    def base_vec(k):
        return cs.vec({c: sgn[k] * inp["cur"][c][k] for c in C.current_columns(machine)})

    def edge_filter(edges):
        """outer edge (eq: hard-closure target + soft residual; ge: one-sided), censored inner edge (ge, one-sided);
        the uncensored inner edge (eq) only with use_inner_eq."""
        if not cfg.get("use_edges", True):
            return []
        out = []
        for o in edges:
            o["sig"] = cfg["sigma_s"] if o["name"] == "out" else cfg["sigma_in"]
            if o["name"] == "in" and (not cfg.get("use_inner", True) or (o["kind"] == "eq" and not cfg["use_inner_eq"])):
                continue
            out.append(o)
        return out

    # ---- per-frame observations (cached: every candidate of a frame sees the SAME observation set, incl. pseudo noise)
    ne_levels = list(cfg.get("ne_levels") or [])
    obs_cache = {}

    def frame_obs(k):
        if k not in obs_cache:
            obs_cache[k] = _frame_obs(k)
        return obs_cache[k]

    def _frame_obs(k):
        """(iso, iso_val, edges, ts_pts, pe_src, extra-info); extra-info['iso_n'] = density pairs"""
        info = {}
        if mode == "pseudo" or (mode == "labelfit" and cfg.get("lf_pe") == "pseudo"):
            pf = pseudo_frame(row, lab, k, t, vert, tang, stress, idx, cov_lo, rng)
            if pf is None:
                return [], [], [], None, "none", {"iso_n": []}
            prof_k, edges, info = pf
            edges = edge_filter(edges)
            iso = K.mast_iso_obs(prof_k, levels, 0.0)
            iso_v = K.mast_iso_obs(prof_k, val_levels, 0.0) if val_levels else []
            info["iso_n"] = K6.mast_ne_obs(prof_k, ne_levels, 0.0) if cfg.get("use_iso", True) else []
            pts = pe_points(prof_k, drop_bands=val_levels)
            return (iso if cfg.get("use_iso", True) else []), iso_v, edges, pts, "ts", info
        edges = edge_filter(RG.build_obs(machine, th, k, {**cfg, "use_inner": True}))
        if machine == "mast":
            pk = th["prof"][k]
            iso = K.mast_iso_obs(pk, levels, chord[1]) if pk is not None else []
            iso_v = K.mast_iso_obs(pk, val_levels, chord[1]) if (pk is not None and val_levels) else []
            info["iso_n"] = K6.mast_ne_obs(pk, ne_levels, chord[1]) if (pk is not None and cfg.get("use_iso", True)) else []
            pts = pe_points(pk, drop_bands=val_levels) if pk is not None else None
        else:
            iso = K.d3d_iso_obs(vert[k], tang[k], levels)
            iso_v = K.d3d_iso_obs(vert[k], tang[k], val_levels) if val_levels else []
            info["iso_n"] = K6.d3d_ne_obs(vert[k], tang[k], ne_levels) if cfg.get("use_iso", True) else []
            pts = pe_points(vert[k], tang[k], drop_bands=val_levels)
        return (iso if cfg.get("use_iso", True) else []), iso_v, edges, pts, ("ts" if pts is not None else "none"), info

    def coverage(k):
        """D3D only (labels allowed): rho_min of the frame's REAL-chord Thomson points on the EFIT map (core coverage)."""
        if machine != "d3d" or "psi" not in lab:
            return {}
        pts = pe_points(vert[k], tang[k])
        if pts is None:
            return {}
        ef = efit_frame(row, lab, k)
        if ef is None:
            return {}
        spl, pa, pb, _, _ = ef
        u = (pa - spl.ev(pts[0], pts[1])) / (pa - pb)
        u = u[np.isfinite(u)]
        return {"rho_min_efit": float(np.sqrt(max(float(u.min()), 0.0)))} if len(u) else {}

    # ---- label-fit residual (D3D only): EFIT flux shape on the official D3D scoring mask, free offset
    if mode == "labelfit":
        assert machine == "d3d"
        Mk = np.load(D.WS / "starter/fusion_scoring/masks/d3d_envelope.npz")["mask_coarse"].astype(bool)   # (Z, R) layout

    def lf_resid(k):
        gt = lab["psi"][k].astype(np.float64)

        def f(sol):
            pr = (RG.OUT_SIGN[machine] * sol["psi"]).T
            d = (pr - gt)[Mk]
            d = d - d.mean()
            return d / cfg.get("lf_sigma", 2e-3), np.full(d.size, 1e9)
        return f

    recs, sols, xs_, zts, pets = {}, {}, {}, {}, {}
    psi_x = np.full((T, g.nx, g.ny), np.nan)
    q95 = np.full(T, np.nan); betaN = np.full(T, np.nan)
    okf = np.zeros(T, bool)
    axR = np.full(T, np.nan); axZ = np.full(T, np.nan)
    for k in ks:
        recs[k] = {"file": Path(path).name, "idx": idx, "k": int(k), "t": float(t[k]), "ip_kA": float(inp["ip_A"][k] / 1e3),
                   "tf_kA": float(inp["tf_kA"][k]), "ts_edge": float(th["edge"][k]), "ts_cens": bool(th["cens"][k]),
                   "ts_rin": float(th["rin"][k]), "ts_cin": bool(th["cin"][k]), "ts_rsym": float(th["rsym"][k]),
                   "ts_has": bool(th["has"][k]), "ok": False, "err": "", "lvl": -9, "pass": 0}

    def pet_for(k, pts, warm_sol):
        """initial p_e table: Thomson points mapped with the warm solution, else the previous table, else a shape."""
        if pts is not None and warm_sol is not None:
            b = warm_sol["b"]
            u = (b["psi_axis"] - b["spl"].ev(pts[0], pts[1])) / (b["psi_axis"] - b["psi_b"])
            tt = K.pe_model(cfg).fit(u, pts[2])
            if tt is not None:
                return tt
        pe = th["pe_max"][k]
        if pets:
            kp = min(pets, key=lambda j: abs(j - k))
            ref = pets[kp]
            if np.isfinite(pe) and pe > 0 and ref.vals[0] > 0:
                return ref.scaled(pe / ref.vals[0])
            return ref
        if not (np.isfinite(pe) and pe > 0):
            pe = pe_med if pe_med else RG.pe_fallback(machine, Ip[k], cfg)
        return K.pe_model(cfg).from_shape(pe)

    def g1_start(k, obs_edges):
        """G1 soft solve (its own profile family) used only to obtain a converged starting equilibrium."""
        pe = th["pe_max"][k]
        if not (np.isfinite(pe) and pe > 0):
            pe = pe_med if pe_med else RG.pe_fallback(machine, Ip[k], cfg)
        paxis = float(np.clip(cfg["k_paxis"] * pe, cfg["paxis_min"], cfg["paxis_max"]))
        anchor = (S.Rn, 0.0)
        if machine == "mast" and np.isfinite(th["edge"][k]) and np.isfinite(th["rin"][k]):
            anchor = (float(np.clip(0.5 * (th["edge"][k] + th["rin"][k]), 0.6, 1.1)), 0.0)
        g1obs = [o for o in obs_edges if o["name"] == "out" and o["kind"] == "eq"]
        for kw in (dict(), dict(blend=0.5, maxits=160), dict(blend=0.5, maxits=160, edge_mode="none")):
            ok, r = S.solve(base_vec(k), Ip[k], paxis, obs=g1obs, anchor=anchor, **kw)
            if ok:
                return r, anchor
        return None, anchor

    def solve_frame(k, warm_pack, allow_g1=True, x0_override=None):
        """warm_pack = (sol, x, zt) of a neighbour or None. Returns (sol, info, obs-pack, x0).
        x0_override: start the LM at this whitened theta (cold G1 start at a given theta = one of the multi-starts)."""
        iso, iso_v, edges, pts, pe_src, oinfo = frame_obs(k)
        oinfo = dict(oinfo)
        iso_n = oinfo.pop("iso_n", [])
        I = base_vec(k)
        bps = S.bp_star(Ip[k])
        anchor = (S.Rn, 0.0)
        if machine == "mast" and np.isfinite(th["edge"][k]) and np.isfinite(th["rin"][k]):
            anchor = (float(np.clip(0.5 * (th["edge"][k] + th["rin"][k]), 0.6, 1.1)), 0.0)
        params = list(cfg["params"])
        edge_out = next(({"P": o["P"], "e": o["e"]} for o in edges if o["name"] == "out" and o["kind"] == "eq"), None)
        zt0 = 0.0
        if mode == "labelfit":
            zt0 = float(lab["efit_z_axis"][k])
            e = lab.get("lcfs_ext")
            edge_out = {"P": (float(e[k, 7]), 0.0), "e": (1.0, 0.0)} if (e is not None and np.isfinite(e[k, 7])) else None
        warm, x0, src, gev_fixed, x_prior = None, {}, "prior", cfg["gev0"], None
        if warm_pack is not None:
            wsol, x0, zt0w, tnb = warm_pack
            if cfg.get("tau_t", 0) > 0:                  # OU temporal prior N(rho x_nb, 1 - rho^2) on the regularised params
                rho = float(np.exp(-abs(t[k] - tnb) / cfg["tau_t"]))
                x_prior = ({p_: rho * x0.get(p_, 0.0) for p_ in x0}, {p_: max(1 - rho ** 2, 0.05) for p_ in x0})
            warm = K.warm_of(wsol)
            gev_fixed = wsol["gev"]
            if mode != "labelfit":
                zt0 = zt0w
            src = "carried"
            pet0 = pet_for(k, pts, wsol)
        else:
            r1, anchor = g1_start(k, edges)
            if r1 is not None:
                warm = {"plasma": r1["plasma"], "dIz": r1["dIz"], "theta": r1["theta"], "axis": r1["b"]["axis"]}
                if r1.get("iso"):
                    gev_fixed = float(r1["theta"][0] / bps)
                pet0 = pet_for(k, pts, {"b": r1["b"]})
            else:
                pet0 = pet_for(k, pts, None)
            if x0_override is not None:
                x0, src = dict(x0_override), "ms_cold"
        fm = K.FrameModel(S, I, Ip[k], bps, pet0, pts, cfg, iq, edge_out=edge_out, gev_fixed=gev_fixed, anchor=anchor)
        has_data = bool(iso) or bool(edges) or bool(iso_n) or mode == "labelfit"
        if mode == "labelfit":
            rfn = lf_resid(k)
        elif iso_n:
            rfn = lambda s_, iso=iso, iso_n=iso_n, edges=edges: K6.residuals6(s_, iso, iso_n, edges, cfg)
        else:
            rfn = None                                   # INV_P5 path (inv_core.residuals)
        sol, info = K.invert_frame(fm, x0, zt0, warm, iso, edges, 0.0 if mode == "labelfit" else z_eta, a_pub, zt_max,
                                   params=params if has_data else [], resid_fn=rfn, x_prior=x_prior)
        if sol is not None and sol.get("ok"):
            sol["zt"] = info.get("zt", zt0)
        elif allow_g1 and cfg.get("g1_fallback", True) and mode != "labelfit":
            # the inverse found no valid branch: ship G1's own soft solution for this frame (flagged x_src = 'g1')
            r1b, _ = g1_start(k, edges)
            if r1b is not None:
                r1b["g1"] = True
                sol = r1b
                info["err"] = (info.get("err", "") + " -> G1 fallback").strip()
                info["g1_fallback"] = True
        info.update({"n_solves": fm.n_solves, "solve_secs": fm.secs, "pe_src": pe_src if pts is not None else ("table" if pets else "shape"),
                     "x_src": "g1" if info.get("g1_fallback") else ("fit" if (has_data and params) else src),
                     "closure": cfg["closure_mode"] if edge_out is not None else "fixed", **oinfo})
        return sol, info, (iso, iso_v, edges, iso_n), x0

    def store(k, sol, info, obs, pas):
        rec = recs[k]
        iso, iso_v, edges, iso_n = obs
        rec.update({"n_ne": int(sum(len(l["pairs"]) for l in iso_n)), "ms_choice": info.get("ms_choice", ""),
                    "ms_ncand": info.get("ms_ncand", 0), "ms_dcost": info.get("ms_dcost", np.nan), **coverage(k)})
        rec.update({"pass": pas, "err": info.get("err", ""), "lm_it": info.get("lm_it", 0), "n_rej": info.get("n_rej", 0),
                    "n_frozen": info.get("n_frozen", 0), "n_solves": info.get("n_solves", 0), "secs": info.get("solve_secs", 0.0),
                    "cost0": info.get("cost0", np.nan), "cost": info.get("cost", np.nan), "pe_src": info.get("pe_src"),
                    "x_src": info.get("x_src"), "n_iso": int(sum(len(l["pairs"]) for l in iso)), "n_edges": len(edges),
                    **{f"sv{j}": v for j, v in enumerate(info.get("sv", []))},
                    **{f"obs_{o['name']}": o["kind"] for o in edges}})
        for kk_ in ("efit_rin", "efit_rout"):
            if kk_ in info:
                rec[kk_] = info[kk_]
        if sol is None or not sol.get("ok"):
            rec["ok"] = False
            return
        b, prof = sol["b"], sol["prof"]
        if sol.get("g1"):
            rec.update({"ok": True, "lvl": 1, "x_src": "g1", "zt": float(sol.get("zt", 0.0)), "gev": float(sol["theta"][0] / S.bp_star(Ip[k])),
                        "bv": float(sol["theta"][0]), "bps": float(S.bp_star(Ip[k])), "dIz_frac": sol["dIz"] / max(Ip[k], 1.0),
                        "fz0": sol.get("fz0", np.nan), "fz1": sol.get("fz1", np.nan), "n_it": sol["n_it"], "clip": bool(sol.get("clip", False)),
                        "Beta0": prof.Beta0, "bq": 0.0, "gam": 1.0, "iso": bool(edges), "has_eq": any(o["kind"] == "eq" for o in edges)})
            for n_, v in K.edge_deltas(sol, edges).items():
                rec[f"d_{n_}"] = v
            psi_x[k] = sol["psi"] - S.vacuum(base_vec(k))
            okf[k] = True
            sc = C.scalars(g, S.wall, sol["psi"], b, prof, inp["fvac"][k], Ip[k], chord=chord)
            q95[k], betaN[k] = sc.get("q95", np.nan), sc.get("betaN", np.nan)
            axR[k], axZ[k] = b["axis"]
            rec.update({"gs_Rax": b["axis"][0], "gs_Zax": b["axis"][1], "gs_limited": b["limited"], "gs_kind": b["kind"],
                        "gs_nx": len(b["xact"]), "psi_axis": b["psi_axis"], "psi_b": b["psi_b"], "fvac": float(inp["fvac"][k]),
                        **{f"gs_{a}": v for a, v in sc.items()}})
            for name, key in (("efit_q95", "gt_q95"), ("efit_beta_n", "gt_betaN"), ("efit_r_axis", "gt_raxis"),
                              ("efit_z_axis", "gt_zaxis"), ("efit_li", "gt_li")):
                if name in lab and k < len(lab[name]):
                    rec[key] = float(lab[name][k])
            return                                         # not used as a warm start for the inverse
        th_ = sol["par"]
        rec.update({"ok": True, "lvl": 0, "zt": float(sol.get("zt", 0.0)), "cp": th_["cp"], "b1": th_["b"][0], "b2": th_["b"][1],
                    "gev": sol["gev"], "gs": th_["gs"], "dr": th_["dr"], "a0": float(prof.a0), "f_press": float(prof.f_press),
                    "pe0": float(prof.pet.vals[0]), "pe_edge": float(prof.pet.edge), "pet_src": prof.pet.src, "n_refit": info.get("n_refit", 0),
                    "closure": info.get("closure"), "bps": float(S.bp_star(Ip[k])), "bv": sol["gev"] * float(S.bp_star(Ip[k])),
                    "dIz_frac": sol["dIz"] / max(Ip[k], 1.0), "fz0": sol.get("fz0", np.nan), "fz1": sol.get("fz1", np.nan), "n_it": sol["n_it"],
                    "iso": bool(edges), "has_eq": any(o["kind"] == "eq" for o in edges), "clip": bool(sol.get("clip", False)),
                    "Beta0": np.nan, "bq": th_["gs"] * float(S.bp_star(Ip[k])), "gam": 1.0})
        for n_, v in K.edge_deltas(sol, edges).items():
            rec[f"d_{n_}"] = v
        if iso:
            r_, _, _ = K.residuals(sol, iso, [], cfg)
            mm = K.iso_mismatch(b["spl"], iso)
            rec.update({"iso_rms": float(np.sqrt(np.mean(r_ ** 2))), "iso_mae_cm": float(np.mean([abs(v) for _, v in mm]) * 100)})
        if iso_v:
            r_, _, _ = K.residuals(sol, iso_v, [], cfg)
            mm = K.iso_mismatch(b["spl"], iso_v)
            rec.update({"val_rms": float(np.sqrt(np.mean(r_ ** 2))), "val_mae_cm": float(np.mean([abs(v) for _, v in mm]) * 100),
                        "val_n": len(mm)})
        psi_x[k] = sol["psi"] - S.vacuum(base_vec(k))
        okf[k] = True
        sc = K.scalars_kin(g, S.wall, sol["psi"], b, prof, inp["fvac"][k], Ip[k], chord=chord)
        q95[k], betaN[k] = sc.get("q95", np.nan), sc.get("betaN", np.nan)
        axR[k], axZ[k] = b["axis"]
        rec.update({"gs_Rax": b["axis"][0], "gs_Zax": b["axis"][1], "gs_limited": b["limited"], "gs_kind": b["kind"],
                    "gs_nx": len(b["xact"]), "psi_axis": b["psi_axis"], "psi_b": b["psi_b"], "paxis_eff": float(th_["cp"] * prof.pet.vals[0]),
                    "fvac": float(inp["fvac"][k]), **{f"gs_{a}": v for a, v in sc.items()},
                    "pe_max_meas": float(th["pe_max"][k]), "pe_axis_fit": float(prof.pet.vals[0]),
                    "pe_vol_fit": float(sc["pav"] / th_["cp"]) if "pav" in sc else np.nan})
        if iso_n:
            mm = K.iso_mismatch(b["spl"], iso_n)
            rec["ne_mae_cm"] = float(np.mean([abs(v) for _, v in mm]) * 100)
        if b["xact"]:
            xl = min(b["xact"], key=lambda x: x[1])
            rec.update({"gs_Rx": xl[0], "gs_Zx": xl[1]})
        if "lcfs_ext" in lab and k < len(lab["lcfs_ext"]):
            e = lab["lcfs_ext"][k]
            rec.update({"gt_rin": e[0], "gt_rout": e[1], "gt_ztop": e[2], "gt_zbot": e[3], "gt_rmid_in": e[6], "gt_rmid_out": e[7]})
        for name, key in (("efit_q95", "gt_q95"), ("efit_beta_n", "gt_betaN"), ("efit_r_axis", "gt_raxis"),
                          ("efit_z_axis", "gt_zaxis"), ("efit_li", "gt_li")):
            if name in lab and k < len(lab[name]):
                rec[key] = float(lab[name][k])
        sols[k] = sol
        xs_[k] = dict(sol["x"])
        zts[k] = float(sol.get("zt", 0.0))
        pets[k] = prof.pet

    # ---- forward pass
    prev, fitted_yet = None, False
    fwd_info, fwd_obs = {}, {}
    for k in ks:
        if Ip[k] < cfg["ip_min"]:
            recs[k].update({"err": "Ip<min", "lvl": -2})
            continue
        sol, info, obs, _ = solve_frame(k, prev, allow_g1=prev is None)
        if (sol is None or not sol.get("ok")) and prev is not None:
            sol, info, obs, _ = solve_frame(k, None)                  # cold retry (G1 start; G1 solution if the inverse fails)
        if info.get("x_src") == "carried" and not fitted_yet:
            info["x_src"] = "prior"
        store(k, sol, info, obs, 1)
        fwd_info[k], fwd_obs[k] = info, obs
        if k in sols:
            fitted_yet = fitted_yet or info.get("x_src") == "fit"
            prev = (sols[k], xs_[k], zts[k], t[k])
        else:
            prev = None
    # ---- reverse pass: failed frames and frames that only carried / used the prior before the first fitted frame (INV_P5),
    #      + INV6 multi-start / bidirectional continuation for ambiguous frames (cfg ms)
    use_ms = bool(cfg.get("ms")) and mode != "labelfit" and bool(cfg["params"])

    def kap_nb(k):
        """median elongation of the +-2 neighbours (current choices), NaN if < 2 available"""
        j = ks.index(k)
        v = [recs[ks[i]].get("gs_kappa", np.nan) for i in (j - 2, j - 1, j + 1, j + 2) if 0 <= i < len(ks) and recs[ks[i]].get("ok")]
        v = [x for x in v if np.isfinite(x)]
        return float(np.median(v)) if len(v) >= 2 else np.nan

    def ambiguous(k):
        r = recs[k]
        if not r["ok"] or r.get("x_src") == "g1":
            return "fail"
        if r.get("gs_kind") != "xpt":
            return "nonx"
        kn = kap_nb(k)
        if np.isfinite(kn) and np.isfinite(r.get("gs_kappa", np.nan)) and abs(r["gs_kappa"] - kn) > cfg["ms_kspike"]:
            return "kspike"
        if cfg["ms_cost_per_obs"] > 0 and r.get("n_iso", 0) + r.get("n_edges", 0) > 0 and \
                r.get("cost", 0.0) / max(r.get("n_iso", 0) + r.get("n_edges", 0) + r.get("n_ne", 0), 1) > cfg["ms_cost_per_obs"]:
            return "poorfit"
        return ""

    nxt = None
    n_ms = {"frames": 0, "changed": 0}
    for k in reversed(ks):
        if Ip[k] < cfg["ip_min"]:
            continue
        r = recs[k]
        redo = (not r["ok"]) or r.get("x_src") == "g1" or (r.get("x_src") == "prior" and mode != "labelfit" and cfg["params"])
        why = ambiguous(k) if use_ms else ""
        if why:
            n_ms["frames"] += 1
            cands = []
            if r["ok"] and k in sols and r.get("x_src") != "g1":
                cands.append(("fwd", sols[k], fwd_info.get(k, {}), fwd_obs.get(k)))
            starts = []
            if nxt is not None:
                starts.append(("rev", dict(warm_pack=(sols[nxt], xs_[nxt], zts[nxt], t[nxt]))))
            starts.append(("cold", dict(warm_pack=None)))
            if k in xs_:
                starts.append(("cold_x", dict(warm_pack=None, x0_override=xs_[k])))
            for lab_, kw in starts:
                sol_c, info_c, obs_c, _ = solve_frame(k, allow_g1=False, **kw)
                if sol_c is not None and sol_c.get("ok") and not sol_c.get("g1"):
                    if info_c.get("x_src") == "carried":
                        info_c["x_src"] = "carried_rev"
                    cands.append((lab_, sol_c, info_c, obs_c))
            if cands:
                kn = kap_nb(k)
                ev = []
                for lab_, sol_c, info_c, obs_c in cands:
                    kap = K6.lcfs_shape(sol_c, S.wall)[0]
                    ev.append((lab_, sol_c, info_c, obs_c, float(info_c.get("cost", np.inf)), sol_c["b"]["kind"] == "xpt", kap))
                cbest = min(e[4] for e in ev)
                elig = [e for e in ev if e[4] <= cbest + cfg["ms_dcost"]]
                key = lambda e: ((0 if e[5] else 1) if cfg["ms_prefer_xpt"] else 0,
                                 (abs(e[6] - kn) if (cfg["ms_prefer_cont"] and np.isfinite(kn) and np.isfinite(e[6])) else 0.0), e[4])
                ch = min(elig, key=key)
                ch[2].update({"ms_choice": f"{why}:{ch[0]}", "ms_ncand": len(ev), "ms_dcost": ch[4] - cbest})
                if ch[0] != "fwd":
                    n_ms["changed"] += 1
                    store(k, ch[1], ch[2], ch[3], 3)
                else:
                    recs[k].update({"ms_choice": f"{why}:fwd", "ms_ncand": len(ev), "ms_dcost": 0.0})
        elif redo and nxt is not None:
            sol, info, obs, _ = solve_frame(k, (sols[nxt], xs_[nxt], zts[nxt], t[nxt]), allow_g1=False)
            if sol is not None and sol.get("ok"):
                if info.get("x_src") == "carried":
                    info["x_src"] = "carried_rev"
                store(k, sol, info, obs, 2)
        if recs[k]["ok"] and k in sols and (recs[k].get("x_src") in ("fit", "carried", "carried_rev", "ms_cold") or nxt is None):
            nxt = k
    # ---- assemble (G1 layout)
    psi_out = np.zeros((T, g.ny, g.nx), np.float32)
    good = np.where(okf)[0]
    for k in range(T):
        if nsel and k not in recs:
            continue
        vac = S.vacuum(base_vec(k))
        if okf[k]:
            px = psi_x[k]
        elif len(good):
            j1 = int(np.searchsorted(good, k))
            lo = good[j1 - 1] if j1 > 0 else None
            hi = good[j1] if j1 < len(good) else None
            jn = lo if hi is None else (hi if lo is None else (lo if t[k] - t[lo] <= t[hi] - t[k] else hi))
            if Ip[k] < cfg["ip_min"]:
                px, Rt, Zt = psi_x[jn], axR[jn], axZ[jn]
            elif lo is not None and hi is not None:
                w = (t[k] - t[lo]) / max(t[hi] - t[lo], 1e-9)
                px = Ip[k] * ((1 - w) * psi_x[lo] / Ip[lo] + w * psi_x[hi] / Ip[hi])
                Rt, Zt = (1 - w) * axR[lo] + w * axR[hi], (1 - w) * axZ[lo] + w * axZ[hi]
            else:
                px, Rt, Zt = psi_x[jn] * (Ip[k] / Ip[jn]), axR[jn], axZ[jn]
            if k in recs:
                recs[k]["fill"] = "interp"
            try:
                spl = RectBivariateSpline(g.gR, g.gZ, vac + px)
                mV, mR = 0.5 * g.R ** 2, g.R ** 2 * (g.Z - Zt)
                corr = (-float(spl.ev(Rt, Zt, dx=1)) / Rt) * (mV - mV.mean()) + (-float(spl.ev(Rt, Zt, dy=1)) / Rt ** 2) * (mR - mR.mean())
                bb = C.analyse(g, S.wall, vac + px + corr, (Rt, Zt), 0.15)
                if bb is not None and np.hypot(bb["axis"][0] - Rt, bb["axis"][1] - Zt) < 0.05:
                    px = px + corr
                    if k in recs:
                        recs[k]["fill"] = "interp_rc"
                else:
                    px = psi_x[jn] + S.vacuum(base_vec(jn)) - vac
                    if k in recs:
                        recs[k]["fill"] = "stale"
            except Exception:
                pass
        else:
            px = S.cold_plasma(max(Ip[k], cfg["ip_min"]), S.Rn, 0.0)
            if k in recs:
                recs[k]["fill"] = "cold"
        full = RG.OUT_SIGN[machine] * (vac + px)
        psi_out[k] = full.T + RG.gauge(machine, inp, k)
        if k in recs and "psi" in lab and k < len(lab["psi"]) and np.isfinite(lab["psi"][k]).all():
            gt = lab["psi"][k].astype(np.float64)
            d = psi_out[k].astype(np.float64) - gt
            dv = (RG.OUT_SIGN[machine] * vac).T + RG.gauge(machine, inp, k) - gt
            recs[k].update({"r2_frame": float(1 - (d ** 2).sum() / ((gt - gt.mean()) ** 2).sum()), "npx": int(gt.size),
                            "gt_s1": float(gt.sum()), "gt_s2": float((gt ** 2).sum()), "d_mean": float(d.mean()),
                            "d_sse": float((d ** 2).sum()), "dv_mean": float(dv.mean()), "dv_sse": float((dv ** 2).sum())})
    info = [recs[k] for k in sorted(recs)]
    for r in info:
        r["shot_secs"] = time.time() - t_start
        r["shot_ms_frames"], r["shot_ms_changed"] = n_ms["frames"], n_ms["changed"]
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, psi=psi_out, q95=q95, betaN=betaN, ok=okf, t=t, ip_A=inp["ip_A"], ks=np.array(sorted(recs)),
                        info=json.dumps(info, default=float))
    os.replace(tmp, part)
    return info


# ----------------------------------------------------------------------------------------------- main
def select_shots(args):
    items = RG.select_shots(args)
    if args.offset:
        items = items[args.offset:]
    if args.disjoint_from:
        ban = set()
        for f in args.disjoint_from.split(","):
            ban |= {x for x in Path(f).read_text().split("\n") if x}
        items = [(i, p) for i, p in items if p.name not in ban]
    if args.max_shots:
        items = items[:args.max_shots]
    return [(j, p) for j, (_, p) in enumerate(items)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["d3d", "mast"], required=True)
    ap.add_argument("--mode", choices=["real", "pseudo", "labelfit"], default="real")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--wall", required=True)
    ap.add_argument("--cfg", action="append", default=[], help="json config(s), merged left to right")
    ap.add_argument("--set", action="append", default=[], help="key=json value")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--offset", type=int, default=0, help="drop the first N selected shots (after stride)")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--disjoint-from", default=None, help="comma list of files_*.txt whose shot names are excluded")
    ap.add_argument("--holdout-names", default=None)
    ap.add_argument("--exclude-holdout", default=None)
    ap.add_argument("--frames-per-shot", type=int, default=0)
    ap.add_argument("--preds", action="store_true")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
    if args.machine == "mast":
        import rule3_guard
        rule3_guard.install()
        assert args.mode == "real", "MAST runs are label-free: mode real only"
    cfg = {}
    for f in args.cfg:
        cfg.update(json.load(open(f)))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    items = select_shots(args)
    (out / f"files_{args.tag}_{args.machine}.txt").write_text("\n".join(p.name for _, p in items) + "\n")
    json.dump({**K.INV_DEFAULT, **C.DEFAULT_CFG, **cfg, "_mode": args.mode}, open(out / f"cfg_{args.tag}_{args.machine}.json", "w"), indent=1)
    log(f"INV {args.machine} mode={args.mode} tag={args.tag} shots={len(items)} procs={args.procs} cfg={json.dumps(cfg)}")
    tasks = [(args.machine, str(p), i, cfg, args.wall, str(out), args.tag, args.frames_per_shot, args.mode) for i, p in items]
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
        if df[c].dtype == object and c not in ("file", "err", "pe_src", "fill", "fatal", "tb", "gs_kind", "x_src", "obs_out", "obs_in",
                                               "closure", "pet_src", "ms_choice"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(out / f"frames_{args.tag}_{args.machine}.parquet")
    wall_s = time.time() - t0
    summ = {"tag": args.tag, "machine": args.machine, "mode": args.mode, "n_shots": len(items), "n_frames": int(len(df)),
            "wall_s": wall_s, "procs": args.procs, "cpu_core_hours": wall_s * args.procs / 3600.0}
    if "fatal" in df:
        summ["n_fatal"] = int(df["fatal"].notna().sum())
        if summ["n_fatal"]:
            summ["fatal_example"] = str(df.loc[df["fatal"].notna(), "tb"].iloc[0])[-600:]
    if "ok" in df:
        d = df[df["ok"].notna()]
        pl = d[d["err"] != "Ip<min"]
        okd = pl[pl["ok"].astype(bool)]
        summ.update({"plasma_frames": int(len(pl)), "converged": float(len(okd) / max(len(pl), 1))})
        for c in ("gs_limited", "has_eq"):
            if c in okd:
                summ[f"frac_{c}"] = float(okd[c].astype(float).mean())
        for c in ("n_solves", "lm_it", "n_frozen", "secs", "iso_mae_cm", "val_mae_cm", "ne_mae_cm", "n_ne", "cp", "b1", "b2", "gev", "dr",
                  "f_press", "gs_kappa"):
            if c in okd and okd[c].notna().any():
                summ[f"{c}_p5_50_95"] = [float(x) for x in np.nanpercentile(okd[c].astype(float), [5, 50, 95])]
        if "fill" in d:
            summ["fill_counts"] = {str(a): int(b) for a, b in d["fill"].value_counts().items()}
        if "x_src" in okd:
            summ["x_src"] = {str(a): int(b) for a, b in okd["x_src"].value_counts().items()}
        if "gs_kind" in okd:
            summ["gs_kind"] = {str(a): int(b) for a, b in okd["gs_kind"].value_counts().items()}
        if "gs_kappa" in okd:
            summ["frac_kappa_gt_2.2"] = float((okd["gs_kappa"] > 2.2).mean())
        if "ms_choice" in okd:
            summ["ms_choice"] = {str(a): int(b) for a, b in okd["ms_choice"].value_counts().items() if a}
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
                preds[k] = preds[k].astype(np.float16).astype(np.float32)
        np.savez(out / f"preds_{args.machine}_{args.tag}.npz", **preds)
        log(f"wrote preds_{args.machine}_{args.tag}.npz (+ _f32)")
    log(f"done in {wall_s:.0f}s wall ({summ['cpu_core_hours']:.2f} core-h)")


if __name__ == "__main__":
    main()
