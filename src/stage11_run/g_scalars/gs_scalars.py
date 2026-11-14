#!/usr/bin/env python3
"""
Stage 11G / Ch2 TRACK scalars — THE scalar module: q95 and beta_N as functions of a SOLVED FRAME, with frozen constants.

A solved frame = the exported flux map (65 x 65) + the frame's inputs (Ip, TF current, Thomson profile, edge flags) + the
solver's own frame record (its contour q95 incl. F95, kinetic pressure scale c_p and <p_e>_V of its fit, the recorded
information on c_p, convergence / observation flags). Three layers:

  frame_record(grid, wall, phi, ref, fvac, Ip, prof, machine)     map-derived quantities (geometry on the 99.5 % contour,
        contour q95 with the vacuum F, scorer-independent li, Thomson p_e mapped onto rho = sqrt(psiN) of THIS map ->
        beta_N,e (the scal2 estimator), beta_p,e, core coverage rho_min)                      [no constant, no label]
  q95_frame(rec, K) / betaN_frame(rec, K)                          the estimators (constants K, see fit_constants)
  shot_series(t, ks, q, b, K, full)                                validity window, in-shot interpolation, centred
        3-frame running median (full-shot runs only), clips        = the INV_P6 packager convention, unchanged

q95 (review round 2, Q3):   q = C_q95 * q_GS * exp(w * yhat),   q_GS = F95/(2pi) * closed-integral dl/(R |grad psi|) on the
        solver's own 95 % surface;  C_q95 = the pipeline's DIII-D pseudo-horizontal TRAIN calibration (median label/ours;
        the convention already on the board);  yhat = guarded DIII-D residual model: ridge of y = log(q95_label / (C_q95
        q_GS)) on standardised [log(mu0 Ip / (a B_T)), eps, kappa, delta_t, delta_b, li, beta_p,e, kinetic residual,
        censoring flags];  w = w0 * support gate,  w0 = clip((M_gg - M_gm)/(M_gg + M_mm - 2 M_gm), 0, 0.35) from
        out-of-fold log-error second moments,  gate = exp(-1/2 sum_j (e_j / tau_j)^2),  e_j = distance of the standardised
        feature outside the DIII-D training range [p0.5, p99.5] in training sd (tau = 1, eps: 0.5).
beta_N:  'mix' (FROZEN DEFAULT, accepted on DIII-D labels):  beta = beta_vol^(1 - w) * (C_kin beta_kin)^w,  w = w_mix * gate,
        beta_vol = c(coverage) * beta_N,e (Thomson p_e on the map's own surfaces x the DIII-D total / electron factor),
        beta_kin = the inverse's own force-balance value 200 mu0 c_p <p_e,fit>_V a / (B0 Ip), C_kin its DIII-D calibration,
        w_mix = clip((M_vv - M_vk) / (M_vv + M_kk - 2 M_vk), 0, 1) from DIII-D TRAIN log-error second moments, gate = the
        support gate on z = log(C_kin beta_kin / beta_vol) (training range p0.5..p99.5, tau = 1 sd). Frames without a kinetic
        solution (G1 fallback) -> beta_vol; frames without a Thomson mapping -> in-shot time interpolation (packager rule);
        a shot without any Thomson -> C_kin x the inverse's kinetic value.
        Other modes (evaluated, NOT accepted, kept for the record):
        prior mean = coverage-corrected DIII-D volume-pressure model  beta_vol = c(coverage) * beta_N,e ;
        'hybrid' (review default, linearised prior swap of the solved frame):  beta = beta_kin * exp((m1 - m0) / (1 + s)),
        beta_kin = 200 mu0 c_p <p_e,fit>_V a / (B0 Ip) = the inverse's own force-balance value, m0 = the prior mean of
        log c_p the frame was solved with, m1 = log(c <p_e>_V,scal2 / <p_e,fit>_V) = the volume-pressure model expressed as
        a prior mean of log c_p, s = recorded information on log c_p after profiling out the FF' and missing-field
        parameters (Schur complement, in units of the prior precision 1 / 0.35^2). s = 0 -> beta_vol; s -> inf -> beta_kin.
        A frame solved WITH the volume-pressure prior (lcp0_table) has m0 = m1 and the hybrid is the solver's own value.
Constants come ONLY from DIII-D labels (fit_constants on the pipeline's DIII-D pseudo-horizontal TRAIN run) and public
machine data (N_TF); nothing here reads a MAST label, the envelope or a board number. MAST entry points install rule3_guard.

CLI:  gs_scalars.py table --machine M --run DIR --tag T --data-dir D --wall W --out PARQUET [--procs N]
      gs_scalars.py fit   --train PARQUET [PARQUET ...] --out const.json [--beta-mode vol|hybrid|kin]
      gs_scalars.py apply --table PARQUET --const const.json --run DIR --tag T --machine M --out-scal NPZ [--parts-out DIR]
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

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "stage11_run/ch2_G1"))
sys.path.insert(0, str(WS / "stage11_run/ch2d_scalars"))
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402
from scal_frames import fit_profile  # noqa: E402  (the scal2 profile fit, unchanged)

MU0 = C.MU0
E_CH = 1.602176634e-19
FEATS = ["lIaB", "eps", "kappa", "tri_top", "tri_bot", "li", "betap_e", "kin_res", "c_in", "c_out", "no_iso", "no_fit"]
FEATS_OV = ["kappa", "tri_top", "tri_bot", "li", "betap_e", "kin_res", "no_iso", "no_fit"]     # no machine-size / aspect-ratio terms
TAU = {"eps": 0.5}
LAMBDAS = (0.1, 1.0, 10.0, 100.0)
W0_MAX = 0.35
SIG_LCP = 0.35
A = None
_G = {}


# =============================================================================================== layer 1: the frame record
def frame_record(grid, wall, phi, ref, fvac, Ip, prof, machine):
    """Map-derived quantities of one frame. phi: canonical flux (axis = maximum), (nR, nZ); ref: (R, Z) axis guess or None;
    fvac = mu0 N_TF I_TF / 2pi [T m]; Ip [A]; prof = (R, Z, Te[eV], ne[m^-3]) Thomson points or None."""
    out = {"ok_ana": False}
    b = C.analyse(grid, wall, phi, ref, 0.25 if ref is not None else None)
    if b is None:
        return out
    Ra, Za = b["axis"]; pa, pb = b["psi_axis"], b["psi_b"]
    span = pb - pa
    spl, fine = b["spl"], b["fine"]
    out.update({"ok_ana": True, "Rax": Ra, "Zax": Za, "kind": b["kind"]})
    C95 = C.closed_contour(wall, fine, pa + 0.95 * span, Ra, Za)
    if C95 is not None:
        out["q95v"] = C.q_on_contour(spl, C95, fvac)
    Cb = C.closed_contour(wall, fine, pa + 0.995 * span, Ra, Za)
    if Cb is None:
        return out
    R_, Z_ = Cb[:-1, 0], Cb[:-1, 1]
    a = 0.5 * (R_.max() - R_.min()); Rgeo = 0.5 * (R_.max() + R_.min())
    R2 = np.append(R_, R_[0]); Z2 = np.append(Z_, Z_[0])
    cr = R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]
    vol = 2 * np.pi * abs(np.sum((R2[:-1] + R2[1:]) * cr) / 6.0)
    lp = float(np.sum(np.hypot(np.diff(R2), np.diff(Z2))))
    out.update({"a": a, "Rgeo": Rgeo, "eps": a / Rgeo, "kappa": (Z_.max() - Z_.min()) / (2 * a),
                "tri_top": (Rgeo - R_[np.argmax(Z_)]) / a, "tri_bot": (Rgeo - R_[np.argmin(Z_)]) / a,
                "vol": vol, "lp": lp, "Rin": R_.min(), "Rout": R_.max()})
    m = b["mask"]
    u = np.clip((phi - pa) / span, 0.0, 1.0)
    rho = np.sqrt(u)
    wv = grid.R * m
    Bpa = MU0 * abs(Ip) / lp
    dpr = spl(grid.gR, grid.gZ, dx=1); dpz = spl(grid.gR, grid.gZ, dy=1)
    bp2 = (dpr ** 2 + dpz ** 2) / grid.R ** 2
    out["li"] = float((bp2 * wv).sum() / max(wv.sum(), 1e-12) / Bpa ** 2) if Bpa > 0 else np.nan
    B0 = abs(fvac) / Rgeo
    out["B0"] = B0
    if np.isfinite(Ip) and abs(Ip) > 1e4 and B0 > 0:
        out["lIaB"] = float(np.log(MU0 * abs(Ip) / (a * B0)))
    if prof is None or not np.isfinite(Ip) or abs(Ip) < 1e4:
        return out
    xr, xz, te, ne = (np.asarray(v, float) for v in prof)
    ok = np.isfinite(xr) & np.isfinite(xz) & np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    if ok.sum() < 6:
        return out
    xr, xz, te, ne = xr[ok], xz[ok], te[ok], ne[ok]
    ut = (spl.ev(xr, xz) - pa) / span
    inside_box = (xr > grid.gR[0]) & (xr < grid.gR[-1]) & (xz > grid.gZ[0]) & (xz < grid.gZ[-1])
    rt = np.sqrt(np.clip(ut, 0.0, None)); rt[~inside_box] = np.nan
    pe = E_CH * te * ne
    fp = fit_profile(rt, pe)
    if fp is None:
        return out
    out["ts_rho_min"] = fp[2]
    pe_g = np.interp(rho, fp[0], fp[1]) * m
    W = max(wv.sum(), 1e-12)
    pe_v = float((pe_g * grid.R).sum() / W)
    ipm = abs(Ip) / 1e6
    out.update({"pe_vol": pe_v, "pe0": float(fp[1][0]), "pe_b": float(fp[1][-1]),
                "betaN_e": float(200 * MU0 * pe_v * a / (B0 * ipm)), "betap_e": float(2 * MU0 * pe_v / Bpa ** 2)})
    return out


# =============================================================================================== layer 2: the estimators
def feature_vector(rec, names=FEATS):
    """Raw feature values of one merged record (map-derived + solver record); NaN where unavailable."""
    g = lambda k: float(rec[k]) if (k in rec and rec[k] is not None and np.isfinite(rec[k])) else np.nan  # noqa: E731
    a = g("a")
    iso = g("iso_mae_cm")
    v = {"lIaB": g("lIaB"), "eps": g("eps"), "kappa": g("kappa"), "tri_top": g("tri_top"), "tri_bot": g("tri_bot"), "li": g("li"),
         "betap_e": g("betap_e"), "kin_res": (iso / 100.0 / a) if (np.isfinite(iso) and np.isfinite(a) and a > 0) else 0.0,
         "c_in": float(bool(rec.get("c_in", False))), "c_out": float(bool(rec.get("c_out", False))),
         "no_iso": float(not (np.isfinite(iso))), "no_fit": float(str(rec.get("x_src", "fit")) != "fit")}
    return np.array([v[n] for n in names], float)


def standardise(X, K):
    mu, sd = np.asarray(K["mu"], float), np.asarray(K["sd"], float)
    Xf = np.where(np.isfinite(X), X, mu)                       # missing -> training mean (zero contribution)
    return (Xf - mu) / sd


def support_gate(Z, K):
    """exp(-1/2 sum_j (e_j / tau_j)^2), e_j = excursion of the standardised feature outside the training [p0.5, p99.5]."""
    lo, hi = np.asarray(K["z_lo"], float), np.asarray(K["z_hi"], float)
    tau = np.array([TAU.get(n, 1.0) for n in K["feats"]])
    e = np.maximum(0.0, np.maximum(lo - Z, Z - hi))
    e = np.where(np.asarray(K["sd_const"], bool), 0.0, e)       # (near-)constant training features carry no gate / no weight
    return np.exp(-0.5 * np.sum((e / tau) ** 2, axis=-1)), e


def q95_frame(rec, K):
    """(q95, diagnostics). K = constants['q95']."""
    qg = rec.get("q_gs", np.nan)
    if not (qg is not None and np.isfinite(qg) and qg > 0):
        qg = rec.get("q95v", np.nan)                           # vacuum-F contour on the exported map (F95/F_vac - 1 ~ 1e-3)
    qb = K["C_q95"] * qg
    if not K.get("w0", 0.0) > 0 or not np.isfinite(qb):
        return qb, {"q_base": qb, "yhat": 0.0, "gate": 0.0, "w": 0.0}
    Z = standardise(feature_vector(rec, K["feats"]), K)
    yh = float(K["b0"] + Z @ np.asarray(K["beta"], float))
    g, _ = support_gate(Z, K)
    w = float(K["w0"] * g)
    return float(qb * np.exp(w * yh)), {"q_base": qb, "yhat": yh, "gate": float(g), "w": w}


def c_cov(rho_min, K):
    """coverage-corrected total / electron factor of the volume-pressure model (DIII-D label classes)."""
    if not np.isfinite(rho_min) or rho_min < 0.15:
        return K["c_core"]
    return K["c_mid"] if rho_min < 0.35 else K["c_far"]


def betaN_frame(rec, K):
    """(beta_N, diagnostics). K = constants['betaN']; mode vol | hybrid | kin."""
    be = rec.get("betaN_e", np.nan)
    c = c_cov(rec.get("ts_rho_min", np.nan), K)
    bvol = c * be if (be is not None and np.isfinite(be)) else np.nan
    bkin = rec.get("b_kin", np.nan)                            # solver's force-balance value 200 mu0 c_p <p_e,fit>_V a / (B0 Ip)
    d = {"b_vol": bvol, "b_kin": bkin, "wp": 1.0}
    mode = K.get("mode", "mix")
    if mode == "kin":
        return (K["C_kin"] * bkin if (bkin is not None and np.isfinite(bkin)) else np.nan), d
    if not np.isfinite(bvol):
        return np.nan, d                                         # no Thomson mapping on this frame -> in-shot time interpolation
    if mode == "vol" or not (bkin is not None and np.isfinite(bkin) and bkin > 0) or str(rec.get("x_src", "")) == "g1":
        return bvol, d
    if mode == "mix":
        z = float(np.log(K["C_kin"] * bkin / bvol)) if bvol > 0 else np.nan
        if not np.isfinite(z):
            return bvol, d
        e = max(0.0, K["mix_z_lo"] - z, z - K["mix_z_hi"]) / K["mix_z_sd"]
        g = float(np.exp(-0.5 * e * e))
        w = K["w_mix"] * g
        d["wp"] = 1.0 - w; d["mix_gate"] = g; d["mix_z"] = z
        return float(bvol * np.exp(w * z)), d
    pv, pf, m0 = rec.get("pe_vol", np.nan), rec.get("pe_vol_fit", np.nan), rec.get("lcp0", np.nan)
    if not (np.isfinite(pv) and np.isfinite(pf) and pv > 0 and pf > 0 and np.isfinite(m0)):
        return bvol, d
    s = rec.get(K.get("s_key", "lcp_s_pri"), np.nan)
    s = float(s) if (s is not None and np.isfinite(s) and s > 0) else 0.0
    wp = 1.0 / (1.0 + K.get("s_scale", 1.0) * s)
    m1 = float(np.log(c * pv / pf))
    d["wp"] = wp; d["m1_m0"] = m1 - m0
    return float(bkin * np.exp((m1 - m0) * wp)), d


# =============================================================================================== layer 3: the shot series
def rmed(x, w):
    import pandas as pd
    return x if w <= 1 else pd.Series(x).rolling(w, center=True, min_periods=1).median().values


def shot_series(t, ks, q, b, full, smooth=3, q_dflt=5.0, b_dflt=1.5, ip=None, fvac=None, fill="lin"):
    """INV_P6 packager convention: values outside (1.2, 25) / (0, 8) -> in-shot time interpolation; centred running median
    (full-shot runs only); clips. q, b: length-T arrays (NaN where not available); ks: indices of the solved frames.
    fill = 'norm' (needs ip): missing frames interpolated in q Ip and beta_N Ip instead of q and beta_N. TESTED AND REJECTED on
    DIII-D labels (pooled q95 R2 0.674 vs 0.795 with 'lin' on the pseudo-horizontal holdout: the unsolved frames are the
    Ip < 50 kA start-up frames); kept only so that an_q95.py part C stays reproducible. apply always uses 'lin'."""
    q = np.asarray(q, float).copy(); b = np.asarray(b, float).copy(); t = np.asarray(t, float)
    q[~((q > 1.2) & (q < 25.0))] = np.nan
    b[~((b > 0) & (b < 8.0))] = np.nan
    mk = np.zeros(len(t), bool); mk[np.asarray(ks, int)] = True
    nq = nb = None
    if fill == "norm" and ip is not None:
        ipa = np.abs(np.asarray(ip, float))
        nq = nb = np.where(np.isfinite(ipa) & (ipa > 1e4), ipa, np.nan)
    for arr, dflt, nrm in ((q, q_dflt, nq), (b, b_dflt, nb)):
        m = np.isfinite(arr) & mk
        bad = ~np.isfinite(arr)
        if nrm is not None and (m & np.isfinite(nrm)).sum() >= 1:
            mm = m & np.isfinite(nrm)
            g = np.interp(t[bad], t[mm], arr[mm] * nrm[mm])
            val = g / nrm[bad]
            lin = np.interp(t[bad], t[m], arr[m])
            arr[bad] = np.where(np.isfinite(val), val, lin)
            continue
        arr[bad] = np.interp(t[bad], t[m], arr[m]) if m.any() else dflt
    if full:
        q, b = rmed(q, smooth), rmed(b, smooth)
    return np.clip(q, 1.2, 25.0), np.clip(b, 0.0, 8.0)


# =============================================================================================== constants (DIII-D labels only)
def r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ridge_fit(Z, y, lam):
    """min sum (y - b0 - Z beta)^2 + lam |beta|^2 (intercept unpenalised)."""
    zm, ym = Z.mean(0), y.mean()
    Zc = Z - zm
    beta = np.linalg.solve(Zc.T @ Zc + lam * np.eye(Z.shape[1]), Zc.T @ (y - ym))
    return float(ym - zm @ beta), beta


def fit_q95(d, feats=FEATS, lambdas=LAMBDAS, nfold=5, seed=0, w0_max=W0_MAX, C_q95=None):
    """d: merged training table (DIII-D pseudo-horizontal TRAIN frames, labels gt_q95). Returns the q95 constants."""
    v = d[(d.gt_q95 > 1) & (d.gt_q95 < 20) & (d.q_gs > 0) & np.isfinite(d.q_gs)].copy()
    if C_q95 is None:
        C_q95 = float(np.median(v.gt_q95 / v.q_gs))
    v = v[(C_q95 * v.q_gs > 1.2) & (C_q95 * v.q_gs < 25)]
    X = np.array([feature_vector(r, feats) for r in v.to_dict("records")])
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0)
    sd_const = sd < 1e-6 * np.maximum(1.0, np.abs(mu))
    isflag = np.array([n in ("c_in", "c_out", "no_iso", "no_fit") for n in feats])
    sd_const |= isflag & (np.minimum(np.nanmean(X, 0), 1 - np.nanmean(X, 0)) < 0.005)   # flags that (almost) never vary
    sd = np.where(sd_const, 1.0, sd)
    K = {"feats": list(feats), "mu": mu.tolist(), "sd": sd.tolist(), "sd_const": sd_const.tolist(), "C_q95": C_q95}
    Z = standardise(X, K)
    Z[:, sd_const] = 0.0
    K["z_lo"] = np.percentile(Z, 0.5, axis=0).tolist(); K["z_hi"] = np.percentile(Z, 99.5, axis=0).tolist()
    y = np.log(v.gt_q95.values / (C_q95 * v.q_gs.values))
    shots = v["file"].values
    us = np.unique(shots)
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(us, rng.permutation(len(us)) % nfold))
    fold = np.array([fold_of[s] for s in shots])
    cv = {}
    oof = {}
    for lam in lambdas:
        p = np.zeros(len(y))
        for f in range(nfold):
            tr = fold != f
            b0, be = ridge_fit(Z[tr], y[tr], lam)
            p[~tr] = b0 + Z[~tr] @ be
        oof[lam] = p
        cv[lam] = float(np.mean((y - p) ** 2))
    lam = min(cv, key=cv.get)
    p = oof[lam]
    eg, em = -y, p - y                                         # log errors of q_GS(calibrated) and of the model
    Mgg, Mmm, Mgm = float(np.mean(eg * eg)), float(np.mean(em * em)), float(np.mean(eg * em))
    den = Mgg + Mmm - 2 * Mgm
    w_opt = (Mgg - Mgm) / den if den > 0 else 0.0
    b0, be = ridge_fit(Z, y, lam)
    K.update({"lambda": lam, "cv_mse": {str(k): x for k, x in cv.items()}, "b0": b0, "beta": be.tolist(),
              "M_gg": Mgg, "M_mm": Mmm, "M_gm": Mgm, "w_opt_unclipped": float(w_opt), "w0": float(np.clip(w_opt, 0.0, w0_max)),
              "n_train_frames": int(len(y)), "n_train_shots": int(len(us)), "oof_r2_log": r2(y, p)})
    return K


def fit_betaN(d, mode="mix", s_key="lcp_s_pri", s_scale=1.0, d_mix=None):
    """c of the volume-pressure model by Thomson core coverage (least squares through the origin, Ip > 200 kA: the scal2
    definition, scal_c.py) + the kinetic calibration C_kin (median label / kinetic)."""
    v = d[np.isfinite(d.betaN_e) & np.isfinite(d.gt_betaN) & (d.ip_A > 2e5) & np.isfinite(d.ts_rho_min)]
    out = {"mode": mode, "s_key": s_key, "s_scale": s_scale}
    for key, m in (("c_core", v.ts_rho_min < 0.15), ("c_mid", (v.ts_rho_min >= 0.15) & (v.ts_rho_min < 0.35)), ("c_far", v.ts_rho_min >= 0.35),
                   ("c_all", v.ts_rho_min >= 0)):
        x = v[m]
        out[key] = float((x.gt_betaN * x.betaN_e).sum() / (x.betaN_e ** 2).sum()) if len(x) > 50 else np.nan
        out["n_" + key] = int(len(x))
    for key in ("c_mid", "c_far"):
        if not np.isfinite(out[key]):
            out[key] = out["c_all"]
    k = d[d.ok & np.isfinite(d.b_kin) & np.isfinite(d.gt_betaN) & (d.gt_betaN > 0.05) & (d.gt_betaN < 8) & (d.b_kin > 0)]
    out["C_kin"] = float(np.median(k.gt_betaN / k.b_kin)) if len(k) > 50 else 1.0
    # mixing weight of the kinetic (force-balance) value: log-error second moments on the train frames (d_mix: clean + stress)
    dm = d if d_mix is None else d_mix
    m = dm[dm.ok & (dm.x_src != "g1") & np.isfinite(dm.b_kin) & (dm.b_kin > 0) & np.isfinite(dm.betaN_e) & (dm.betaN_e > 0)
           & np.isfinite(dm.gt_betaN) & (dm.gt_betaN > 0.05) & (dm.gt_betaN <= 6.0)]
    if len(m) > 100:
        bv = np.array([c_cov(r, out) for r in m.ts_rho_min.values]) * m.betaN_e.values
        ev = np.log(bv / m.gt_betaN.values); ek = np.log(out["C_kin"] * m.b_kin.values / m.gt_betaN.values)
        Mvv, Mkk, Mvk = float(np.mean(ev * ev)), float(np.mean(ek * ek)), float(np.mean(ev * ek))
        z = ek - ev
        out.update({"M_vv": Mvv, "M_kk": Mkk, "M_vk": Mvk, "w_mix": float(np.clip((Mvv - Mvk) / (Mvv + Mkk - 2 * Mvk), 0.0, 1.0)),
                    "mix_z_lo": float(np.percentile(z, 0.5)), "mix_z_hi": float(np.percentile(z, 99.5)), "mix_z_sd": float(np.std(z)),
                    "mix_z_median": float(np.median(z)), "n_mix": int(len(m))})
    else:
        out.update({"w_mix": 0.0, "mix_z_lo": -1e9, "mix_z_hi": 1e9, "mix_z_sd": 1.0})
    return out


# =============================================================================================== tables
def grid_wall(row, wall_npz):
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    key = (gR.tobytes(), gZ.tobytes())
    if key not in _G:
        g = C.Grid(gR, gZ)
        w = np.load(wall_npz)
        _G.clear()
        _G[key] = (g, C.Wall(g, w["limiter_R"], w["limiter_Z"], up=2))
    return _G[key]


SOLVER_COLS = ["ok", "x_src", "gs_q95", "gs_q95_vac", "gs_f95_ratio", "gs_betaN", "gs_pav", "gs_a", "gs_Rgeo", "gs_B0", "gs_kappa", "gs_li",
               "gs_kind", "cp", "b1", "b2", "gs", "gev", "dr", "lcp0", "lcp_s_raw", "lcp_s_pri", "lcp_s_pure", "lcp_nrow", "pe_vol_fit",
               "pe_axis_fit", "pe_edge", "pe_max_meas", "iso_mae_cm", "n_iso", "n_edges", "ts_cens", "ts_cin", "ts_has", "ts_rin", "ts_edge",
               "efit_rin", "efit_rout", "cost", "sv0", "sv1", "sv2", "sv3", "sv4", "n_frozen", "f_press", "a0", "obs_out", "obs_in",
               "gt_q95", "gt_betaN", "gt_li", "gt_raxis", "gt_rin", "gt_rout", "gt_ztop", "gt_zbot"]


def _table_shot(task):
    idx, path = task
    try:
        machine = A.machine
        if machine == "mast":
            import rule3_guard
            rule3_guard.install()
        row = D.load_row(path, machine)
        t = np.asarray(row["efit_times"], float)
        grid, wall = grid_wall(row, A.wall)
        c = np.load(Path(A.run) / "parts" / f"{A.tag}_{machine}_{idx:05d}.npz", allow_pickle=False)
        P = c["psi"]; qp = c["q95"].astype(float)
        sg = 1.0 if machine == "mast" else -1.0
        inp = D.frame_inputs(row, machine, t)
        th = D.thomson_frames(row, machine, t, C.DEFAULT_CFG)
        refs = A.refs.get(idx, {})
        lab = D.labels_of(row) if machine == "d3d" else {}
        recs = []
        for k in c["ks"].astype(int):
            rec = {"idx": idx, "file": Path(path).name, "k": int(k), "t": float(t[k]), "T": len(t), "n_ks": int(len(c["ks"])),
                   "ip_A": float(abs(inp["ip_A"][k])), "fvac": float(abs(inp["fvac"][k])), "q_part": float(qp[k])}
            for nm, key in (("efit_q95", "lab_q95"), ("efit_beta_n", "lab_betaN")):       # D3D only: labels of every frame
                if nm in lab and k < len(lab[nm]):
                    rec[key] = float(lab[nm][k])
            if np.isfinite(P[k]).all() and np.abs(P[k]).max() > 0:
                try:
                    rec.update(frame_record(grid, wall, sg * P[k].astype(np.float64).T, refs.get(int(k)), abs(inp["fvac"][k]),
                                            abs(inp["ip_A"][k]), th["prof"][k], machine))
                except Exception as e:  # noqa: BLE001
                    rec["err_rec"] = f"{type(e).__name__}: {e}"
            recs.append(rec)
        return recs
    except Exception as e:  # noqa: BLE001
        return [{"idx": idx, "file": Path(path).name, "fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}]


def merge_solver(df, fr, machine, wall_rin=None):
    """Attach the solver's frame record and derive the model inputs q_gs, b_kin, flags."""
    import pandas as pd
    cols = ["file", "k"] + [c for c in SOLVER_COLS if c in fr.columns]
    d = df.merge(fr[cols], on=["file", "k"], how="left")
    d["ok"] = d["ok"].fillna(False).astype(bool)
    d["q_gs"] = np.where(d["ok"], d["gs_q95"], np.nan) if "gs_q95" in d else np.nan
    d["b_kin"] = np.where(d["ok"], d["gs_betaN"], np.nan) if "gs_betaN" in d else np.nan
    if machine == "mast":
        d["c_in"] = d["ts_cin"].fillna(True).astype(bool); d["c_out"] = d["ts_cens"].fillna(True).astype(bool)
    else:                                                         # DIII-D pseudo-horizontal laser: coverage from wall + 4 cm to 2.45 m
        cov_lo = (wall_rin if wall_rin is not None else 1.016) + 0.04
        d["c_in"] = (d["efit_rin"] <= cov_lo) if "efit_rin" in d else False
        d["c_out"] = False
    for c in ("iso_mae_cm", "lcp0", "lcp_s_raw", "lcp_s_pri", "lcp_s_pure", "pe_vol_fit", "cp"):
        if c not in d:
            d[c] = np.nan
    d["x_src"] = d["x_src"].fillna("none") if "x_src" in d else "none"
    for key, lk in (("gt_q95", "lab_q95"), ("gt_betaN", "lab_betaN")):                  # D3D: labels of ALL frames (also unsolved ones)
        if lk in d:
            d[key] = d[lk] if key not in d else d[lk].where(np.isfinite(d[lk]), d[key])
    return pd.DataFrame(d)


def cmd_table(args):
    global A
    A = args
    if A.machine == "mast":
        import rule3_guard
        rule3_guard.install()
    import pandas as pd
    names = [x for x in (Path(A.run) / f"files_{A.tag}_{A.machine}.txt").read_text().split("\n") if x]
    paths = [str(next(Path(dd) / n for dd in A.data_dir if (Path(dd) / n).exists())) for n in names]
    fr = pd.read_parquet(Path(A.run) / f"frames_{A.tag}_{A.machine}.parquet")
    A.refs = {}
    fo = fr[fr.ok == True]  # noqa: E712
    for i, k, ra, za in zip(fo.idx.values, fo.k.values, fo.gs_Rax.values, fo.gs_Zax.values):
        A.refs.setdefault(int(i), {})[int(k)] = (float(ra), float(za))
    t0 = time.time()
    recs = []
    with get_context("fork").Pool(A.procs) as pool:
        for r in pool.imap_unordered(_table_shot, list(enumerate(paths)), chunksize=1):
            recs += r
    df = pd.DataFrame(recs)
    w = np.load(A.wall)
    d = merge_solver(df, fr, A.machine, float(np.min(w["limiter_R"])))
    for c in d.columns:
        if d[c].dtype == object and c not in ("file", "kind", "err_rec", "fatal", "tb", "x_src", "gs_kind", "obs_out", "obs_in"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    d.to_parquet(A.out)
    fat = int(d["fatal"].notna().sum()) if "fatal" in d else 0
    print("GSTABLE", json.dumps({"out": A.out, "shots": len(paths), "frames": len(d), "fatal": fat, "ok": float(d.ok.mean()),
                                 "betaN_e": float(d.betaN_e.notna().mean()) if "betaN_e" in d else 0.0, "secs": time.time() - t0}))
    if fat:
        print(d[d["fatal"].notna()][["file", "fatal", "tb"]].head(2).to_string())


def cmd_fit(args):
    import pandas as pd
    d = pd.concat([pd.read_parquet(p) for p in args.train], ignore_index=True)
    d = d[d.ok]
    feats = FEATS_OV if args.feats == "ov" else FEATS
    dall = pd.read_parquet(args.train[0]); dq = dall[dall.ok]      # C_q95: the pipeline convention = the CLEAN train run (first table)
    v = dq[(dq.gt_q95 > 1) & (dq.gt_q95 < 20) & (dq.q_gs > 0)]
    Kq = fit_q95(d, feats=feats, C_q95=float(np.median(v.gt_q95 / v.q_gs)))
    if args.no_resid:
        Kq["w0"] = 0.0
    Kb = fit_betaN(dall, mode=args.beta_mode, s_key=args.s_key,   # c: every frame with a map (the scal_c.py / INV_P6 definition)
                   d_mix=pd.concat([pd.read_parquet(p) for p in args.train], ignore_index=True))
    if args.single_c:                                             # the keepCq convention: one c (core-covered) for every frame
        Kb["c_mid"] = Kb["c_far"] = Kb["c_core"]
        Kb["raw_kin_fallback"] = True
    K = {"q95": Kq, "betaN": Kb, "smooth": 3, "sig_lcp": SIG_LCP,
         "provenance": {"train_tables": args.train, "labels": "DIII-D TRAIN shots only (holdout excluded); no MAST quantity",
                        "fit": "gs_scalars.py fit (deterministic: shot-grouped 5-fold CV with seed 0)"}}
    json.dump(K, open(args.out, "w"), indent=1)
    print("GSFIT", json.dumps({"C_q95": Kq["C_q95"], "lambda": Kq["lambda"], "w0": Kq["w0"], "w_opt": Kq["w_opt_unclipped"],
                               "oof_r2_log": Kq["oof_r2_log"], "c_core": Kb["c_core"], "c_mid": Kb["c_mid"], "c_far": Kb["c_far"], "C_kin": Kb["C_kin"]}))


def apply_table(d, K):
    """Per-frame estimates for a merged table -> adds q_new, b_new and diagnostics (no shot-level post-processing)."""
    qs, bs, gate, wq, yh, wp = [], [], [], [], [], []
    for r in d.to_dict("records"):
        if not r.get("ok", False):                              # unsolved frame (its map is a time-interpolated fill): q95 -> in-shot
            be = r.get("betaN_e", np.nan)                        # interpolation; beta_N -> the volume model on the filled map (packager rule)
            bv = c_cov(r.get("ts_rho_min", np.nan), K["betaN"]) * be if (be is not None and np.isfinite(be)) else np.nan
            qs.append(np.nan); bs.append(bv if K["betaN"].get("mode", "mix") != "kin" else np.nan)
            gate.append(np.nan); wq.append(np.nan); yh.append(np.nan); wp.append(np.nan)
            continue
        q, dq = q95_frame(r, K["q95"]); b, db = betaN_frame(r, K["betaN"])
        qs.append(q); bs.append(b); gate.append(dq["gate"]); wq.append(dq["w"]); yh.append(dq["yhat"]); wp.append(db.get("wp", np.nan))
    d = d.copy()
    d["q_new"], d["b_new"], d["q_gate"], d["q_w"], d["q_yhat"], d["b_wp"] = qs, bs, gate, wq, yh, wp
    return d


def cmd_apply(args):
    if args.machine == "mast":
        import rule3_guard
        rule3_guard.install()
    import pandas as pd
    K = json.load(open(args.const))
    d = apply_table(pd.read_parquet(args.table), K)
    run = Path(args.run)
    files = [x for x in (run / f"files_{args.tag}_{args.machine}.txt").read_text().split("\n") if x]
    g = {i: s for i, s in d.groupby("idx")}
    out = {}
    st = {"shots": len(files), "frames": 0, "q_nan": 0, "b_nan": 0}
    for i, f in enumerate(files):
        cz = np.load(run / "parts" / f"{args.tag}_{args.machine}_{i:05d}.npz", allow_pickle=False)
        T = cz["psi"].shape[0]; ks = cz["ks"].astype(int); tt = cz["t"].astype(float)
        q = np.full(T, np.nan); b = np.full(T, np.nan)
        s = g.get(i)
        if s is not None:
            kk = s.k.values.astype(int)
            q[kk] = s.q_new.values; b[kk] = s.b_new.values
        b[~((b > 0) & (b < 8.0))] = np.nan
        if not np.isfinite(b[ks]).any():                         # a shot without any valid Thomson value: the inverse's kinetic beta_N
            b = (1.0 if K["betaN"].get("raw_kin_fallback") else K["betaN"]["C_kin"]) * cz["betaN"].astype(float)
        st["q_nan"] += int((~np.isfinite(q[ks])).sum()); st["b_nan"] += int((~np.isfinite(b[ks])).sum()); st["frames"] += T
        q, b = shot_series(tt, ks, q, b, full=len(ks) == T, smooth=K.get("smooth", 3))
        out[f"shot_{i:04d}_q95"] = q.astype(np.float32); out[f"shot_{i:04d}_betaN"] = b.astype(np.float32)
        if args.parts_out:                                        # copies of the run's parts with the module scalars (eval_frames.py)
            po = Path(args.parts_out); (po / "parts").mkdir(parents=True, exist_ok=True)
            cd = dict(cz); cd["q95"], cd["betaN"] = q, b
            np.savez_compressed(po / "parts" / f"{args.tag}_{args.machine}_{i:05d}.npz", **cd)
    if args.parts_out:
        import shutil
        for f in (f"files_{args.tag}_{args.machine}.txt", f"frames_{args.tag}_{args.machine}.parquet"):
            if (run / f).exists():
                shutil.copy(run / f, Path(args.parts_out) / f)
    np.savez_compressed(args.out_scal, **out)
    qa = np.concatenate([out[f"shot_{i:04d}_q95"] for i in range(len(files))]); ba = np.concatenate([out[f"shot_{i:04d}_betaN"] for i in range(len(files))])
    print("GSAPPLY", json.dumps({**st, "out": args.out_scal, "q95_p1_50_99": [float(x) for x in np.percentile(qa, [1, 50, 99])],
                                 "betaN_p1_50_99": [float(x) for x in np.percentile(ba, [1, 50, 99])]}))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("table")
    p.add_argument("--machine", required=True, choices=["mast", "d3d"]); p.add_argument("--run", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--data-dir", required=True, nargs="+"); p.add_argument("--wall", required=True); p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=8)
    p = sub.add_parser("fit")
    p.add_argument("--train", required=True, nargs="+"); p.add_argument("--out", required=True)
    p.add_argument("--beta-mode", default="mix", choices=["mix", "vol", "hybrid", "kin"]); p.add_argument("--s-key", default="lcp_s_pri")
    p.add_argument("--feats", default="full", choices=["full", "ov"]); p.add_argument("--no-resid", action="store_true")
    p.add_argument("--single-c", action="store_true")
    p = sub.add_parser("apply")
    p.add_argument("--table", required=True); p.add_argument("--const", required=True); p.add_argument("--run", required=True)
    p.add_argument("--tag", required=True); p.add_argument("--machine", required=True); p.add_argument("--out-scal", required=True)
    p.add_argument("--parts-out", default=None)
    args = ap.parse_args()
    {"table": cmd_table, "fit": cmd_fit, "apply": cmd_apply}[args.cmd](args)


if __name__ == "__main__":
    main()
