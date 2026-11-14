#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — (a) synthetic free-boundary equilibria from the forward model of the gamma_shape-free
kinetic inverse driven by real frames' INPUTS, (b) the real frames' network inputs. CPU pool, one task per shot (.venv_gs).

  --mode synth   per anchor frame (Ip >= ip_min, outer Thomson edge measured): G1 soft start, centre solve at the prior mean,
                 then --nsamp draws of the latents from the inverse's own priors (DIII-D label fits; cfg chain = the inverse's):
                   log c_p ~ N(lcp0, sig_lcp^2),  FF' shape (b1, b2[, be]) ~ N(b0, sig_b^2),  dr ~ N(0, sig_dr^2),
                   p' shape s ~ N(0.5, s_sd^2) (or fixed by pe_s_fix),  p_e amplitude = measured p_e x logN(0, pe_sd^2)
                 forward solve = prof7.FrameModel7 (INV_P7a model for family fe / src native; INV_P7c for the cubic family):
                 measured coils (k = 0.5; P2 wiring repair), measured Ip, stiff radial closure at the measured outer Thomson
                 edge + dr (the uniform vertical field is derived and recorded), ridge-Z rule iterated.
                 Each accepted equilibrium is rendered into NREAL noisy Thomson realisations on the frame's real channels
                 (MAST) / the harness's pseudo-horizontal laser (DIII-D) and reduced by syn_common.ts_feats.
  --mode real    every (or --frames-per-shot) frame of every shot: vacuum flux of the measured coils, Ip, F_vac, gauge, coil
                 inputs and the Thomson features of the REAL frame (MAST: real midplane laser; DIII-D: the harness's
                 pseudo-horizontal laser, clean or stress, identical to run_inv7 --mode pseudo incl. the noise stream).
MAST: rule3_guard installed before the fork pool; rows through the guarded loader only. DIII-D labels are read only in DIII-D
processes: pseudo-laser synthesis (as the harness does) and, in real mode, q95 / betaN labels for the train-set calibration.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import syn_common as SC  # noqa: E402

C, D, K, TG = SC.C, SC.D, SC.K, SC.TG
import run_g1 as RG  # noqa: E402
import run_inv7 as R7  # noqa: E402  (pseudo_frame, d3d_chords, pe_points: the harness's own functions)
import prof7 as P7  # noqa: E402
import p2split as P2S  # noqa: E402

P7.install()
MODE = "synth"
GEN_DEFAULT = {"nsamp": 12, "frame_stride": 3, "n_anchor": 0, "x_wid": 1.0, "dr_sd": 0.03, "s_mean": 0.5, "s_sd": 0.5, "pe_sd": 0.30,
               "amp_sd": 0.15, "ridge_its": 3, "gen_rtol": 1e-3, "ts_fit": False, "ts_jit": 0.25, "ts_p_broad": 0.2}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


class Shot:
    """Everything run_inv7._run_shot sets up for a shot (same calls, same order), + the P2 wiring repair."""

    def __init__(self, machine, path, idx, ucfg, wall_npz, stress=None):
        self.machine, self.path, self.idx = machine, path, idx
        row = D.load_row(path, machine)
        S = RG.get_solver(machine, row, ucfg, wall_npz)
        self.t = t = np.asarray(row["efit_times"], float)
        self.T = len(t)
        self.inp = inp = D.frame_inputs(row, machine, t)
        p2 = P2S.setup(machine, path, row, ucfg, wall_npz, inp)
        self.p2d, self.p2cfg = None, (p2 or {}).get("cfg", "")
        if p2 is not None and p2["solver"] is not None:
            S, self.p2d = p2["solver"], p2["d"]
        self.S, self.row = S, row
        self.cfg = cfg = {**K.INV_DEFAULT, **R7.INV6_DEFAULT, **P7.P7_DEFAULT, **GEN_DEFAULT, **S.cfg}
        self.th = D.thomson_frames(row, machine, t, cfg)
        self.lab = D.labels_of(row) if machine == "d3d" else {}
        if machine == "d3d":
            self.vert, self.tang = R7.d3d_chords(row, t, cfg["th_gap_ms"])
        self.sgn = np.where(inp["ip_A"] >= 0, 1.0, -1.0)
        self.Ip = np.abs(inp["ip_A"])
        self.rc = bool(cfg.get("rc", False)) and machine == "d3d"          # real-chord leg: DIII-D's own vertical core + tangential Thomson
        self.chord = self.th["chord"] if (machine == "mast" or self.rc) else ("h", 0.0)
        if self.chord is None:
            self.chord = ("h", 0.0) if not self.rc else ("v", 1.94)
        self.rcR = float(self.chord[1]) if self.rc else np.nan
        self.zl = float(self.chord[1]) if not self.rc else 0.0
        pe_ok = self.th["pe_max"][np.isfinite(self.th["pe_max"]) & (self.th["pe_max"] > 0)]
        self.pe_med = float(np.median(pe_ok)) if len(pe_ok) else None
        self.cov_lo_d3d = S.wall.R_in_wall + 0.04
        self.stress = stress or {}
        self.rng_ps = np.random.default_rng(7919 * idx + 17)           # run_inv7's pseudo-noise stream (same seed, same order)
        self.levels = list(cfg["levels"])

    def base_vec(self, k):
        cs, inp, sgn = self.S.coils, self.inp, self.sgn
        cur = {c: sgn[k] * inp["cur"][c][k] for c in C.current_columns(self.machine)}
        if self.p2d is not None:
            cur["P2DU"], cur["P2DL"] = sgn[k] * self.p2d["U"][k], sgn[k] * self.p2d["L"][k]
        return cs.vec(cur)

    def frame_obs(self, k):
        """(profile (R, Z, Te, ne) | None, edge observations [all: outer + inner, measured or censored], pe points | None, cov_lo, cov_hi)"""
        cfg = self.cfg
        if self.machine == "d3d" and self.rc:
            th = self.th
            edges = RG.build_obs("d3d", th, k, cfg)
            prof = th["prof"][k]
            pts = R7.pe_points(prof) if prof is not None else None
            cl, ch = np.nan, float(th["covhi"][k])
        elif self.machine == "d3d":
            pf = R7.pseudo_frame(self.row, self.lab, k, self.t, self.vert, self.tang, self.stress, self.idx, self.cov_lo_d3d, self.rng_ps)
            if pf is None:
                return None, [], None, self.cov_lo_d3d, np.inf
            prof, edges, _ = pf
            pts = R7.pe_points(prof)
            cl, ch = self.cov_lo_d3d, np.inf
        else:
            th = self.th
            edges = RG.build_obs(self.machine, th, k, {**cfg, "use_inner": True})
            prof = th["prof"][k]
            pts = R7.pe_points(prof) if prof is not None else None
            cl, ch = float(th["covlo"][k]), float(th["covhi"][k])
        for o in edges:
            o["sig"] = cfg["sigma_s"] if o["name"] == "out" else cfg["sigma_in"]
        return prof, edges, pts, cl, ch

    def anchor(self, k):
        th = self.th
        if self.machine == "mast" and np.isfinite(th["edge"][k]) and np.isfinite(th["rin"][k]):
            return (float(np.clip(0.5 * (th["edge"][k] + th["rin"][k]), 0.6, 1.1)), 0.0)
        return (self.S.Rn, 0.0)

    def g1_start(self, k, edges, anchor):
        """run_inv7.g1_start: G1 soft solve (its own profile family) = converged starting equilibrium."""
        cfg, S, th = self.cfg, self.S, self.th
        pe = th["pe_max"][k]
        if not (np.isfinite(pe) and pe > 0):
            pe = self.pe_med if self.pe_med else RG.pe_fallback(self.machine, self.Ip[k], cfg)
        paxis = float(np.clip(cfg["k_paxis"] * pe, cfg["paxis_min"], cfg["paxis_max"]))
        g1obs = [o for o in edges if o["name"] == "out" and o["kind"] == "eq"]
        for kw in (dict(), dict(blend=0.5, maxits=160), dict(blend=0.5, maxits=160, edge_mode="none")):
            ok, r = S.solve(self.base_vec(k), self.Ip[k], paxis, obs=g1obs, anchor=anchor, **kw)
            if ok:
                return r, pe
        return None, pe


# ----------------------------------------------------------------------------------------------- real frames
def run_real(task):
    machine, path, idx, ucfg, wall, out, tag, seed, gcfg = task
    part = Path(out) / "real" / tag / f"{idx:05d}.npz"
    if part.exists():
        return {"idx": idx, "cached": True}
    sh = Shot(machine, path, idx, ucfg, wall, stress=gcfg.get("stress"))
    S, T, t, Ip, inp = sh.S, sh.T, sh.t, sh.Ip, sh.inp
    nsel = int(gcfg.get("frames_per_shot", 0))
    ks = list(range(T))
    if nsel:
        ks = sorted(set(np.linspace(0, T - 1, min(nsel, T)).round().astype(int).tolist()))
    nk = len(ks)
    g = S.grid
    X = np.zeros((nk, len(C.current_columns(machine)) + 4))
    F = np.zeros((nk, SC.NF_RC if sh.rc else SC.NF))
    vac = np.zeros((nk, g.nx, g.ny), np.float32)
    bps = np.zeros(nk); gauge = np.zeros(nk)
    for j, k in enumerate(ks):
        I = sh.base_vec(k)
        vac[j] = S.vacuum(I)
        bps[j] = S.bp_star(max(Ip[k], 1.0))
        gauge[j] = RG.gauge(machine, inp, k)
        X[j] = SC.coil_x(machine, inp, sh.sgn, Ip, k, sh.p2d)
        if Ip[k] < sh.cfg["ip_min"]:
            continue                                        # run_inv7 never builds observations for these frames
        prof, edges, _, _, _ = sh.frame_obs(k)
        F[j] = SC.ts_feats_rc(prof, edges, bps[j], sh.rcR, sh.levels) if sh.rc else SC.ts_feats(prof, edges, bps[j], machine, sh.levels)
    extra = {}
    if machine == "d3d":                                    # D3D labels: scalar calibration on TRAIN sets / diagnostics only
        for nm, key in (("efit_q95", "gt_q95"), ("efit_beta_n", "gt_betaN"), ("efit_r_axis", "gt_raxis"), ("efit_z_axis", "gt_zaxis"),
                        ("efit_li", "gt_li")):
            if nm in sh.lab:
                extra[key] = np.asarray(sh.lab[nm], float)[ks]
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, t=t, ks=np.array(ks), T=T, Ip=Ip, sgn=sh.sgn, fvac=np.asarray(inp["fvac"], float), gauge=gauge, X=X,
                        F=F.astype(np.float32), vac=vac, bps=bps, fname=np.array(Path(path).name), p2cfg=np.array(sh.p2cfg),
                        gR=np.asarray(g.gR, float), gZ=np.asarray(g.gZ, float), **extra)
    os.replace(tmp, part)
    return {"idx": idx, "T": T, "nk": nk, "has_ts": float(np.mean(F[:, SC.FEAT_NAMES.index("prof_has")] > 0)) if nk else 0.0, "p2": sh.p2cfg}


# ----------------------------------------------------------------------------------------------- synthetic equilibria
def run_synth(task):
    machine, path, idx, ucfg, wall, out, tag, seed, gcfg = task
    part = Path(out) / "synth" / tag / f"{idx:05d}.npz"
    if part.exists():
        return {"idx": idx, "cached": True}
    t0 = time.time()
    sh = Shot(machine, path, idx, ucfg, wall)
    S, T, t, Ip, inp, cfg, th = sh.S, sh.T, sh.t, sh.Ip, sh.inp, sh.cfg, sh.th
    g, cs = S.grid, S.coils
    iq = cs.index["VQ"]
    a_pub, zt_max, z_eta = C.A_PUBLIC[machine], S.zt_max, cfg["z_eta"]
    rng = np.random.default_rng(1_000_003 * int(seed) + 7919 * int(idx) + 17)
    PEM = K.pe_model(cfg)
    s_fix = cfg.get("pe_s_fix")
    reg = [p for p in cfg["params"] if p in ("lcp", "b1", "b2", "be")]
    cand = [k for k in range(T) if Ip[k] >= cfg["ip_min"]]
    if cfg["n_anchor"] and len(cand) > cfg["n_anchor"]:
        cand = sorted(rng.choice(cand, int(cfg["n_anchor"]), replace=False).tolist())
    elif cfg["frame_stride"] > 1:
        cand = cand[int(rng.integers(cfg["frame_stride"]))::int(cfg["frame_stride"])]
    rec = {"X": [], "F": [], "psi": [], "scal": [], "diag": []}
    errs, cnt = {}, {"anchors": 0, "noedge": 0, "g1fail": 0, "try": 0, "ok": 0, "centre_ok": 0}
    te_all = []
    for k in cand:
        pk = th["prof"][k]
        if pk is not None:
            m = np.isfinite(pk[2]) & (np.asarray(pk[2], float) > 0)
            if m.sum() >= 5:
                te_all.append(float(np.nanmax(K._smooth(np.asarray(pk[2], float)[m]))))
    te_med = float(np.median(te_all)) if te_all else (700.0 if machine == "mast" else 2500.0)

    def err(e):
        e = (e or "?")[:40]
        errs[e] = errs.get(e, 0) + 1

    for k in cand:
        prof_k, edges, pts, cov_lo, cov_hi = sh.frame_obs(k)
        edge_out = next(({"P": o["P"], "e": o["e"]} for o in edges if o["name"] == "out" and o["kind"] == "eq"), None)
        if edge_out is None:
            cnt["noedge"] += 1
            continue
        cnt["anchors"] += 1
        I = sh.base_vec(k)
        bps = S.bp_star(Ip[k])
        fvac = float(inp["fvac"][k])
        anchor = sh.anchor(k)
        saved_rtol = S.cfg["rtol"]
        r1, pe_meas = sh.g1_start(k, edges, anchor)
        if r1 is None:
            cnt["g1fail"] += 1
            continue
        warm1 = {"plasma": r1["plasma"], "dIz": r1["dIz"], "theta": r1["theta"], "axis": r1["b"]["axis"]}
        gev1 = float(r1["theta"][0] / bps) if r1.get("theta") else 0.0
        pe_ref = float(pe_meas)
        if pts is not None:                                 # measured p_e mapped on the starting equilibrium -> axis amplitude
            b1_ = r1["b"]
            u = (b1_["psi_axis"] - b1_["spl"].ev(pts[0], pts[1])) / (b1_["psi_axis"] - b1_["psi_b"])
            tt = K.PePoly.fit(u, pts[2])
            if tt is not None and tt.P0 > 0:
                pe_ref = float(tt.P0)
        if sh.rc and prof_k is None:
            cnt["noedge"] += 1
            continue
        if prof_k is not None:
            xr_t = np.asarray(prof_k[0], float)
            valid_t = np.isfinite(prof_k[2]) & np.isfinite(prof_k[3]) & (np.asarray(prof_k[2], float) > 0) & (np.asarray(prof_k[3], float) > 0)
            te_ref = float(np.nanmax(K._smooth(np.asarray(prof_k[2], float)[valid_t]))) if valid_t.sum() >= 5 else te_med
        else:
            xr_t, valid_t, te_ref = None, None, te_med
        S.cfg["rtol"] = float(cfg["gen_rtol"])              # tighter Picard tolerance for the synthetic truth (restored below)
        try:
            fm = P7.FrameModel7(S, I, Ip[k], bps, PEM(pe_ref, cfg["s_mean"] if s_fix is None else float(s_fix), 0.0, "syn"), None, cfg, iq,
                                edge_out=edge_out, gev_fixed=gev1, anchor=anchor)
            fm.fvac = fvac
            warmc, ztc = warm1, 0.0
            sc0 = fm.robust_solve({}, 0.0, warm1)
            if sc0["ok"]:
                cnt["centre_ok"] += 1
                z1 = K.ridge_z(sc0, 0.0, z_eta, a_pub, zt_max)
                if abs(z1) > 0.002:
                    s2 = fm.solve({}, z1, warm=K.warm_of(sc0))
                    if s2["ok"]:
                        sc0, ztc = s2, z1
                warmc = K.warm_of(sc0)
            tsfit = None
            if cfg["ts_fit"] and sh.rc:                        # real-chord leg v2: the same idea on the real DIII-D channels
                try:
                    sref = sc0 if sc0["ok"] else r1
                    bb_ = sref["b"]
                    Cc0 = C.closed_contour(S.wall, bb_["fine"], bb_["psi_axis"] + 0.995 * (bb_["psi_b"] - bb_["psi_axis"]), bb_["axis"][0], bb_["axis"][1])
                    tsfit = SC.fit_real_shapes_rc(prof_k, sref, Cc0)
                    cnt["tsfit_ok"] = cnt.get("tsfit_ok", 0) + int(tsfit is not None)
                except Exception:  # noqa: BLE001
                    tsfit = None
            elif cfg["ts_fit"]:                                # Thomson shape prior = the frame's own real profile on our own solution
                try:
                    sref = sc0 if sc0["ok"] else r1
                    scr = K.scalars_kin(g, S.wall, sref["psi"], sref["b"], sref["prof"], fvac, Ip[k], chord=("h", sh.zl)) if sc0["ok"] else {}
                    rin0, rout0 = scr.get("edge_gs_in", np.nan), scr.get("edge_gs", np.nan)
                    if not (np.isfinite(rin0) and np.isfinite(rout0)):
                        rin0 = next((o["P"][0] for o in edges if o["name"] == "in"), -np.inf); rout0 = edge_out["P"][0]
                    tsfit = SC.fit_real_shapes(prof_k, sref, sh.zl, rin0, rout0)
                    cnt["tsfit_ok"] = cnt.get("tsfit_ok", 0) + int(tsfit is not None)
                except Exception:  # noqa: BLE001
                    tsfit = None
            for _ in range(int(cfg["nsamp"])):
                cnt["try"] += 1
                try:
                    n_ok_draw = _draw(sh, fm, rng, cfg, reg, s_fix, PEM, pe_ref, te_ref, xr_t, valid_t, ztc, warmc, k, I, bps, fvac, cov_lo, cov_hi,
                                      gev1, rec, err, tsfit)
                    cnt["ok"] += n_ok_draw
                except Exception as e:  # noqa: BLE001
                    err("EXC " + type(e).__name__ + ": " + str(e))
        finally:
            S.cfg["rtol"] = saved_rtol
    part.parent.mkdir(parents=True, exist_ok=True)
    tmp = part.with_suffix(".tmp.npz")
    arr = {n: (np.array(v, dtype=np.float32 if n in ("psi", "F") else np.float64) if len(v) else np.zeros((0,))) for n, v in rec.items()}
    np.savez_compressed(tmp, **arr, fname=np.array(Path(path).name), cnt=np.array(json.dumps(cnt)), errs=np.array(json.dumps(errs)),
                        p2cfg=np.array(sh.p2cfg))
    os.replace(tmp, part)
    return {"idx": idx, **cnt, "errs": errs, "secs": time.time() - t0, "p2": sh.p2cfg}


def _draw(sh, fm, rng, cfg, reg, s_fix, PEM, pe_ref, te_ref, xr_t, valid_t, ztc, warmc, k, I, bps, fvac, cov_lo, cov_hi, gev1, rec, err, tsfit=None):
    """one synthetic equilibrium of anchor frame k (returns 1 if accepted)"""
    S, machine, Ip, inp, t = sh.S, sh.machine, sh.Ip, sh.inp, sh.t
    g = S.grid
    a_pub, zt_max, z_eta = C.A_PUBLIC[machine], S.zt_max, cfg["z_eta"]
    x = {p: float(cfg["x_wid"] * rng.standard_normal()) for p in reg}
    x["dr"] = float(cfg["dr_sd"] / cfg["sig_dr"] * rng.standard_normal())
    x = K.clip_x(x, cfg)
    P0 = float(pe_ref * np.exp(cfg["pe_sd"] * rng.standard_normal()))
    s_ = float(s_fix) if s_fix is not None else float(np.clip(cfg["s_mean"] + cfg["s_sd"] * rng.standard_normal(), -0.8, 3.0))
    pet = PEM(P0, s_, 0.0, "syn")
    sol = fm.robust_solve(x, ztc, warmc, pet=pet)
    if not sol["ok"]:
        err(sol.get("err", ""))
        return 0
    zt, secs = ztc, sol["secs"]
    for _i in range(int(cfg["ridge_its"])):
        z1 = K.ridge_z(sol, zt, z_eta, a_pub, zt_max)
        if abs(z1 - zt) <= 0.002:
            break
        s2 = fm.solve(x, z1, warm=K.warm_of(sol), pet=pet)
        secs += s2["secs"]
        if not s2["ok"]:
            break
        sol, zt = s2, z1
    b, prof = sol["b"], sol["prof"]
    sc = K.scalars_kin(g, S.wall, sol["psi"], b, prof, fvac, Ip[k], chord=("v", sh.rcR) if sh.rc else ("h", sh.zl))
    need = ("q95", "betaN", "edge_gs", "kappa", "li") if sh.rc else ("q95", "betaN", "edge_gs", "edge_gs_in", "kappa", "li")
    if not all(np.isfinite(sc.get(n, np.nan)) for n in need):
        err("scalars")
        return 0
    if sh.rc:
        return _draw_rc(sh, sol, sc, rng, cfg, P0, te_ref, bps, cov_hi, k, I, fvac, gev1, rec, zt, secs, x, s_, tsfit)
    rout, rin = float(sc["edge_gs"]), float(sc["edge_gs_in"])
    if xr_t is None:
        xr_t = np.arange(sh.cov_lo_d3d, 2.45, 0.0105); valid_t = np.ones(len(xr_t), bool)
    spec = (SC.sample_ts_spec_fit(rng, P0, te_ref, tsfit, cfg["amp_sd"], cfg["ts_jit"], cfg["ts_p_broad"]) if cfg["ts_fit"]
            else SC.sample_ts_spec(rng, P0, te_ref, cfg["amp_sd"]))
    feats = []
    for (sT, sN, so, si, br) in SC.NOISE:
        pr = SC.render(sol, xr_t, valid_t, sh.zl, rin, rout, spec, rng, sT, sN, br)
        ed = SC.syn_edges(rout, rin, sh.zl, cov_lo, cov_hi, so * rng.standard_normal(), si * rng.standard_normal(),
                          margin=0.01 if machine == "mast" else 0.0)
        feats.append(SC.ts_feats(pr, ed, bps, machine, sh.levels))
    th_ = sol["par"]
    rxl, zxl, rxu, zxu, dsep = SC.xpoints(sol, machine)
    rec["X"].append(SC.coil_x(machine, inp, sh.sgn, Ip, k, sh.p2d))
    rec["F"].append(np.stack(feats))
    rec["psi"].append((sol["psi"] - S.vacuum(I)).astype(np.float32))
    rec["scal"].append([sc.get(n, np.nan) for n in SC.SCAL_KEYS])
    rec["diag"].append([b["axis"][0], b["axis"][1], zt, sol["dIz"] / max(Ip[k], 1.0), float(prof.f_press), SC.KIND.get(b["kind"], 3.0),
                        sol["n_it"], bps, fvac, Ip[k], t[k], k, sol["gev"], th_["dr"], th_["cp"], th_["b"][0], th_["b"][1],
                        th_.get("be", 0.0), P0, s_, float(prof.a0), rxl, zxl, rxu, zxu, dsep,
                        float(sol.get("delta", {}).get("tgt", np.nan)), secs, gev1])
    return 1


def _draw_rc(sh, sol, sc, rng, cfg, P0, te_ref, bps, cov_hi, k, I, fvac, gev1, rec, zt, secs, x, s_, tsfit=None):
    """real-chord leg: render the synthetic equilibrium on the frame's REAL DIII-D channels (vertical core chord + tangential)."""
    S, machine, Ip, inp, t = sh.S, sh.machine, sh.Ip, sh.inp, sh.t
    b, prof = sol["b"], sol["prof"]
    pk = sh.th["prof"][k]
    Rp, Zp = np.asarray(pk[0], float), np.broadcast_to(np.asarray(pk[1], float), np.shape(pk[0]))
    valid = np.isfinite(pk[2]) & np.isfinite(pk[3]) & (np.asarray(pk[2], float) > 0) & (np.asarray(pk[3], float) > 0)
    core = np.abs(Rp - sh.rcR) < 1e-6
    span = b["psi_b"] - b["psi_axis"]
    Cc = C.closed_contour(S.wall, b["fine"], b["psi_axis"] + 0.995 * span, b["axis"][0], b["axis"][1])
    if Cc is None:
        return 0
    ztop = float(sc["edge_gs"])
    spec = (SC.sample_ts_spec_fit(rng, P0, te_ref, tsfit, cfg["amp_sd"], cfg["ts_jit"], cfg["ts_p_broad"]) if cfg["ts_fit"]
            else SC.sample_ts_spec(rng, P0, te_ref, cfg["amp_sd"]))
    feats = []
    for (sT, sN, so, si, br) in SC.NOISE:
        pr = SC.render_rc(sol, Rp, Zp, valid, Cc, spec, rng, sT, sN, core, br)
        ed = SC.syn_edges_rc(ztop, cov_hi, so * rng.standard_normal(), sh.rcR)
        feats.append(SC.ts_feats_rc(pr, ed, bps, sh.rcR, sh.levels))
    th_ = sol["par"]
    rxl, zxl, rxu, zxu, dsep = SC.xpoints(sol, machine)
    rec["X"].append(SC.coil_x(machine, inp, sh.sgn, Ip, k, sh.p2d))
    rec["F"].append(np.stack(feats))
    rec["psi"].append((sol["psi"] - S.vacuum(I)).astype(np.float32))
    rec["scal"].append([sc.get(n, np.nan) for n in SC.SCAL_KEYS])
    rec["diag"].append([b["axis"][0], b["axis"][1], zt, sol["dIz"] / max(Ip[k], 1.0), float(prof.f_press), SC.KIND.get(b["kind"], 3.0),
                        sol["n_it"], bps, fvac, Ip[k], t[k], k, sol["gev"], th_["dr"], th_["cp"], th_["b"][0], th_["b"][1],
                        th_.get("be", 0.0), P0, s_, float(prof.a0), rxl, zxl, rxu, zxu, dsep,
                        float(sol.get("delta", {}).get("tgt", np.nan)), secs, gev1])
    return 1


def run_task(task):
    try:
        return run_synth(task) if MODE == "synth" else run_real(task)
    except Exception as e:  # noqa: BLE001
        return {"idx": task[2], "fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def main():
    global MODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "real"], required=True)
    ap.add_argument("--machine", choices=["d3d", "mast"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--wall", required=True)
    ap.add_argument("--cfg", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", default=None)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--holdout-names", default=None)
    ap.add_argument("--exclude-holdout", default=None)
    ap.add_argument("--disjoint-from", default=None, help="comma list of files with shot names to exclude")
    ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--shot-list", default=None, help="file with shot file names: keep only these (order of the sorted directory)")
    ap.add_argument("--shard", default=None, help="i:N -> shots i::N of the selected list (global indices kept)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = ap.parse_args()
    MODE = args.mode
    if args.machine == "mast":
        import rule3_guard
        rule3_guard.install()
    cfg = {}
    for f in args.cfg:
        cfg.update(json.load(open(f)))
    for kv in args.set:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    cfg.setdefault("p2_wiring", "auto")
    gcfg = {k: cfg[k] for k in ("stress", "frames_per_shot") if k in cfg}
    out = Path(args.out); (out / args.mode / args.tag).mkdir(parents=True, exist_ok=True)
    items = RG.select_shots(args)
    if args.disjoint_from:
        ban = set()
        for f in args.disjoint_from.split(","):
            ban |= {x for x in Path(f).read_text().split("\n") if x}
        items = [(i, p) for i, p in items if p.name not in ban]
    if args.shot_list:
        keep = {x for x in Path(args.shot_list).read_text().split("\n") if x}
        items = [(i, p) for i, p in items if p.name in keep]
        assert len(items) == len(keep), f"shot list: {len(keep)} names, {len(items)} found"
    if args.max_shots:
        items = items[:args.max_shots]
    items = [(j, p) for j, (_, p) in enumerate(items)]
    lst = out / args.mode / f"files_{args.tag}.txt"
    txt = "\n".join(p.name for _, p in items) + "\n"
    if not lst.exists() or lst.read_text() != txt:
        tmpf = lst.with_suffix(f".{os.getpid()}.tmp")
        tmpf.write_text(txt)
        os.replace(tmpf, lst)
    sfx = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split(":"))
        items = items[i::n]
        sfx = f"_s{i:02d}"
    json.dump({"cfg": {**GEN_DEFAULT, **cfg}, "noise": SC.NOISE, "feat_names": SC.FEAT_NAMES, "args": vars(args)},
              open(out / args.mode / f"cfg_{args.tag}{sfx}.json", "w"), indent=1, default=str)
    log(f"synth_net {args.mode} {args.machine} tag={args.tag}{sfx} shots={len(items)} procs={args.procs} cfg={json.dumps(cfg)}")
    tasks = [(args.machine, str(p), i, cfg, args.wall, str(out), args.tag, args.seed, gcfg) for i, p in items]
    t0 = time.time()
    recs = []
    with get_context("fork").Pool(args.procs) as pool:
        for n, r in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
            recs.append(r)
            if (n + 1) % max(1, len(tasks) // 10) == 0 or n == len(tasks) - 1:
                log(f"  {n + 1}/{len(tasks)} shots {time.time() - t0:.0f}s")
    wall_s = time.time() - t0
    fat = [r for r in recs if "fatal" in r]
    summ = {"mode": args.mode, "machine": args.machine, "tag": args.tag + sfx, "n_shots": len(items), "n_fatal": len(fat),
            "wall_s": wall_s, "procs": args.procs, "cpu_core_hours": wall_s * args.procs / 3600.0}
    if args.mode == "synth":
        for kk in ("anchors", "noedge", "g1fail", "try", "ok", "centre_ok", "tsfit_ok"):
            summ[kk] = int(sum(r.get(kk, 0) for r in recs))
        summ["ok_frac"] = summ["ok"] / max(summ["try"], 1)
        ee = {}
        for r in recs:
            for e, c in r.get("errs", {}).items():
                ee[e] = ee.get(e, 0) + c
        summ["errs"] = dict(sorted(ee.items(), key=lambda kv: -kv[1])[:12])
        summ["p2"] = {c: int(sum(r.get("p2") == c for r in recs)) for c in set(r.get("p2") for r in recs if "p2" in r)}
        ss = [r["secs"] for r in recs if "secs" in r]
        summ["shot_secs_p50_max"] = [float(np.median(ss)), float(np.max(ss))] if ss else []
    if fat:
        summ["fatal_examples"] = fat[:3]
    json.dump(summ, open(out / args.mode / f"summary_{args.tag}{sfx}.json", "w"), indent=1, default=str)
    log("SUMMARY " + json.dumps({k: v for k, v in summ.items() if k != "fatal_examples"}))
    for f in fat[:3]:
        log(f"FATAL {f}")


if __name__ == "__main__":
    main()
