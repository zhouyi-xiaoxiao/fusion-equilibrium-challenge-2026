#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK li_startup — li_core.py

(A) li from INPUTS.  Dimensionless ramp-history features of the plasma-current record in units of a resistive time
    tau_R(t) built from the Thomson core T_e and the public minor radius (Spitzer, Z_eff 2, ln Lambda 17, trapped-particle
    factor at mid-radius; public physics, no fitted constant), plus two Thomson profile exponents of the horizontal laser.
    A ridge model fitted on DIII-D TRAIN labels against the SCORER-derived li of the label maps (li_fit.py) predicts
    li and its uncertainty per frame; outside the DIII-D feature support the uncertainty is inflated.
    The prediction enters the kinetic inverse as a Gaussian pseudo-observation of the solution's li (= a prior on the
    FF' shape direction, expressed in the quantity the model predicts), never as a hard value.
(B) start-up frames.  A one-sided soft wall-clearance datum (the X-point separatrix must not lie behind the limiter
    surface at the wall contact point; G1's own definition of the clearance, g1_core.Solver.solve 'gap') evaluated INSIDE
    the LM objective of the inverse (so the profile / radial parameters trade it against the Thomson data; no field push),
    and the audit's fallback validity gate F1.

Nothing here reads a label on MAST.  DIII-D labels are used only by li_table.py / li_fit.py (training) and by the
DIII-D validation harness.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
for p in (WS / "starter", WS / "stage11_run/ch2_geometry", WS / "stage11_run/ch2_G1"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import thomson_geom as TG  # noqa: E402

R0_PUBLIC = {"d3d": 1.70, "mast": 0.85}      # = g1_core.R0_PUBLIC (public nominal major radius)
A_PUBLIC = {"d3d": 0.60, "mast": 0.55}       # = g1_core.A_PUBLIC  (public nominal minor radius)

RAMP_C = (0.03, 0.1, 0.3, 1.0)               # memory lengths in units of tau_R
AGE_C = (0.1, 0.3, 1.0)
FEATS_RAMP0 = ["r0"] + [f"m{c:g}" for c in RAMP_C] + [f"age{c:g}" for c in AGE_C] + ["lfrac"]     # models v1 - v3
FEATS_RAMP = FEATS_RAMP0 + ["lq"]            # + ln of the cylindrical safety factor q_c = 2 pi a^2 B_T / (mu0 R0 Ip)  (models v4+)
MU0 = 4e-7 * np.pi
FEATS_TS = ["aT", "an"]
FEATS_FULL = FEATS_RAMP + FEATS_TS
LI_DEFAULT = {
    "li_model": None,          # path of the model json (li_fit.py); None = module off
    "li_sig_scale": 1.0,       # multiplies the predicted sd of the li pseudo-observation
    "li_sig_add": 0.0,         # added in quadrature to the sd (machine-transfer term, e.g. the tau_R bracket)
    "li_sig_min": 0.05,        # floor of the sd
    "li_b1_var": 1.0,          # whitened prior variance of b1 when the li datum is active (> 1: the static b1 prior is relaxed)
    "li_b2_var": 1.0,
    "li_huber": 1e9,
    "prior_b_var": 1.0,        # whitened variance of the static b1 / b2 priors WITHOUT a li datum (diagnostic: data-driven li level)
    "li_offset": 0.0,          # added to the predicted datum (level of the prediction; 0 = the DIII-D-trained level)
    "li_use": "auto",          # auto: full model where the exponents exist, else ramp | ramp | full
    "li_tau_scale": 1.0,       # sensitivity knob: multiplies tau_R (transfer-risk bracket)
    "gap_w": 0.0,              # sd [m] of the one-sided wall-clearance datum inside the LM objective; 0 = off
    "gap_min_lm": 0.01,        # required clearance [m] (= g1_core DEFAULT_CFG gap_min)
    "fb_gate": 0,              # audit fix F1: ship a G1 fallback only if it passes the inverse's own validity test
}


# ----------------------------------------------------------------------------------------------- resistive time
def tau_ref_s(te_keV, machine, scale=1.0):
    """Fundamental current-diffusion time [s]: mu0 a^2 / (5.78 eta_Spitzer(Z_eff 2, ln Lambda 17)) x (1 - sqrt(eps_half))^2.
    eta_Sp = 1.65e-9 ln Lambda Z_eff / T_e[keV]^1.5 Ohm m (Wesson); eps_half = a / (2 R0): trapped-particle reduction of the
    conductivity at mid-radius. Public formulae; machine enters only through its public a and R0."""
    a, R0 = A_PUBLIC[machine], R0_PUBLIC[machine]
    fneo = (1.0 - np.sqrt(0.5 * a / R0)) ** 2
    return scale * 3.88 * a * a * np.clip(np.asarray(te_keV, float), 0.02, None) ** 1.5 * fneo


def te_top(prof):
    """Robust core T_e [eV] of one Thomson profile (R, Z, Te, ne): median of the three largest valid values."""
    if prof is None:
        return np.nan
    te = np.asarray(prof[2], float); ne = np.asarray(prof[3], float)
    m = np.isfinite(te) & np.isfinite(ne) & (te > 0) & (ne > 0)
    if m.sum() < 5:
        return np.nan
    v = np.sort(te[m])
    return float(np.median(v[-3:]))


def _fill(v):
    v = np.asarray(v, float).copy()
    m = np.isfinite(v)
    if not m.any():
        return None
    x = np.arange(len(v), dtype=float)
    v[~m] = np.interp(x[~m], x[m], v[m])
    p = np.pad(v, 2, mode="edge")
    return np.median(np.stack([p[i:i + len(v)] for i in range(5)]), axis=0)


def ramp_features(tip_ms, ip, t_frames, te_frames_eV, machine, tau_scale=1.0, fvac=None, ip_unit_A=1e3):
    """Ramp-history features on the frame times.  tip_ms / ip: the Ip record (only ratios enter, except for lq, which needs
    ip_unit_A = amperes per unit of ip); te_frames_eV: core T_e per frame (NaN allowed); fvac: R B_T per frame [T m] (for lq).
    lq = ln q_c, q_c = 2 pi a^2 B_T(R0) / (mu0 R0 Ip): the on-axis identity j0 / <j> = q_c (1 + kappa0^2) kappa / (2 kappa0 q0) holds at
    any aspect ratio, so q_c is the dimensionless measure of how peaked a relaxed (q0 ~ 1) current profile must be.
    Returns dict name -> array(T) + diagnostics (tau_ms, theta)."""
    T = len(t_frames)
    out = {n: np.full(T, np.nan) for n in FEATS_RAMP}
    out["tau_ms"] = np.full(T, np.nan); out["theta"] = np.full(T, np.nan); out["te_keV"] = np.full(T, np.nan)
    tip = np.asarray(tip_ms, float); ip = np.asarray(ip, float)
    n = min(len(tip), len(ip)); tip, ip = tip[:n], ip[:n]
    m = np.isfinite(tip) & np.isfinite(ip)
    if m.sum() < 20:
        return out
    tip, ip = tip[m], np.abs(ip[m])
    o = np.argsort(tip, kind="stable"); tip, ip = tip[o], ip[o]
    keep = np.concatenate([[True], np.diff(tip) > 0]); tip, ip = tip[keep], ip[keep]
    t0, t1 = float(tip[0]), float(max(tip[-1], t_frames[-1]))
    tg = np.arange(t0, t1 + 1.0, 1.0)
    ipa = np.interp(tg, tip, ip)
    k5 = np.ones(5) / 5.0
    ipa = np.convolve(np.pad(ipa, 2, mode="edge"), k5, mode="valid")
    imax = float(ipa.max())
    if not imax > 0:
        return out
    imin = 0.1 * imax
    g = np.log(np.maximum(ipa, imin))
    te = _fill(te_frames_eV)
    if te is None:
        te = np.full(T, 500.0)                                   # no Thomson at all: neutral default (flagged by te_keV = NaN)
        have_te = False
    else:
        have_te = True
    te_g = np.interp(tg, np.asarray(t_frames, float), te)
    tau = tau_ref_s(te_g / 1e3, machine, tau_scale) * 1e3          # ms
    above = ipa > imin
    i0 = int(np.argmax(above)) if above.any() else 0
    dth = 1.0 / tau
    dth[:i0] = 0.0
    theta = np.cumsum(dth)
    dg = np.diff(g, prepend=g[0])
    mem = {}
    for c in RAMP_C:
        dec = np.exp(-dth / c)
        mm = np.zeros(len(tg)); acc = 0.0
        for i in range(len(tg)):
            acc = acc * dec[i] + dg[i]
            mm[i] = acc
        mem[c] = mm
    k10 = np.ones(11) / 11.0
    dgs = np.convolve(np.pad(dg, 5, mode="edge"), k10, mode="valid")
    r0 = np.clip(tau * dgs, -5.0, 5.0)
    tf = np.asarray(t_frames, float)
    out["r0"] = np.interp(tf, tg, r0)
    for c in RAMP_C:
        out[f"m{c:g}"] = np.interp(tf, tg, mem[c])
    for c in AGE_C:
        out[f"age{c:g}"] = np.interp(tf, tg, np.exp(-theta / c))
    out["lfrac"] = np.interp(tf, tg, np.log(np.maximum(ipa, imin) / imax))
    if fvac is not None:
        a_, R0_ = A_PUBLIC[machine], R0_PUBLIC[machine]
        fv = np.abs(np.asarray(fvac, float))
        ipf = np.interp(tf, tg, np.maximum(ipa, imin)) * ip_unit_A
        with np.errstate(all="ignore"):
            out["lq"] = np.log(np.maximum(2 * np.pi * a_ * a_ * fv / (MU0 * R0_ * R0_ * ipf), 1e-3))
    out["tau_ms"] = np.interp(tf, tg, tau); out["theta"] = np.interp(tf, tg, theta)
    out["te_keV"] = te / 1e3 if have_te else np.full(T, np.nan)
    return out


def exponents(prof, f_edge=0.3):
    """Thomson profile exponents on a HORIZONTAL laser chord, outboard side, in the geometric normalised radius
    x = (R - R_c) / (R_out - R_c):  ln T_e ~ aT ln(1 - x^2),  ln n_e ~ an ln(1 - x^2)  (0.05 < x < 0.85).
    R_c = symmetric-isotherm centre (else the T_e peak), R_out = n_e foot (thomson_geom.profile_geometry, the pipeline's own
    edge rule).  None when the outer foot is censored or the profile is too sparse."""
    if prof is None:
        return None
    R, _, Te, ne = (np.asarray(a, float) for a in prof)
    g = TG.profile_geometry(R, Te, ne, f_list=(f_edge,))
    if not g:
        return None
    rc = g.get("x_sym", np.nan)
    if not np.isfinite(rc):
        rc = g.get("x_tepk", np.nan)
    rout, cens = g.get(f"hi_ne_{f_edge:g}", np.nan), g.get(f"chi_ne_{f_edge:g}", True)
    if cens or not (np.isfinite(rc) and np.isfinite(rout) and rout - rc > 0.15):
        return None
    x = (R - rc) / (rout - rc)
    m = np.isfinite(x) & np.isfinite(Te) & np.isfinite(ne) & (Te > 0) & (ne > 0) & (x > 0.05) & (x < 0.85)
    if m.sum() < 8 or x[m].max() - x[m].min() < 0.4:
        return None
    X = np.log(1.0 - x[m] ** 2)
    aT = float(np.polyfit(X, np.log(Te[m]), 1)[0]); an = float(np.polyfit(X, np.log(ne[m]), 1)[0])
    if not (np.isfinite(aT) and np.isfinite(an)):
        return None
    return float(np.clip(aT, -1.0, 6.0)), float(np.clip(an, -1.0, 6.0))


def ip_record(row, machine):
    """(time [ms], Ip) of the shipped plasma-current record (DIII-D: organiser time-axis fix; MAST: magnetics_time)."""
    ip = np.asarray(row["magnetics_plasma_current"], float)
    if machine == "d3d":
        from data_fixes import fix_d3d_ip_times
        tip = np.asarray(fix_d3d_ip_times(row), float)
    else:
        tip = np.asarray(row["magnetics_time"], float)
    return tip, ip


# ----------------------------------------------------------------------------------------------- model
ZCLIP = 3.5


def expand(z, quad):
    """design row(s) of the ridge: standardised features, plus (quad) all pairwise products of the features clipped at
    +-ZCLIP sd.  z: (p,) or (n, p)."""
    z = np.asarray(z, float)
    if not quad:
        return z
    zc = np.clip(z, -ZCLIP, ZCLIP)
    iu = np.triu_indices(z.shape[-1])
    if z.ndim == 1:
        return np.concatenate([zc, np.outer(zc, zc)[iu]])
    return np.hstack([zc, (zc[:, :, None] * zc[:, None, :])[:, iu[0], iu[1]]])


class LiModel:
    """Ridge models 'full' (ramp + exponents) and 'ramp' (ramp only) with support statistics (li_fit.py json)."""

    def __init__(self, path):
        self.j = json.load(open(path))
        self.path = str(path)
        self.sub = {}
        for name in ("full", "ramp"):
            s = self.j[name]
            self.sub[name] = {k: (np.array(v, float) if (isinstance(v, list) and k != "feats") else v) for k, v in s.items()}
        self.li_lo, self.li_hi = float(self.j["li_lo"]), float(self.j["li_hi"])
        self.cdef = float(self.j.get("cdef", 1.044))
        self.datum = "lic" if self.j.get("target") == "gt_lic" else "li"
        if self.datum == "lic":
            self.cdef = 1.0

    def predict(self, feat: dict, use="auto"):
        """feat: name -> value (NaN = missing). Returns dict(li, sig, d, insup, model) or None."""
        name = "full" if use in ("auto", "full") else "ramp"
        if name == "full" and not all(np.isfinite(feat.get(f, np.nan)) for f in FEATS_TS):
            if use == "full":
                return None
            name = "ramp"
        s = self.sub[name]
        names = s["feats"]
        x = np.array([feat.get(f, np.nan) for f in names], float)
        if not np.isfinite(x).all():
            return None
        nclip = int(((x < s["lo"]) | (x > s["hi"])).sum())
        xc = np.clip(x, s["lo"], s["hi"])                       # winsorise at the training range: no extrapolation
        z = (xc - s["mu"]) / s["sd"]
        li = float(s["b"] + expand(z, bool(s.get("quad", False))) @ s["coef"])
        zr = (x - s["mu"]) / s["sd"]                            # support distance uses the UNCLIPPED features
        d = float(np.sqrt(zr @ s["Sinv"] @ zr))
        infl = 1.0 if d <= s["d95"] else (d / s["d95"]) ** 2
        return {"li": float(np.clip(li, self.li_lo, self.li_hi)), "li_raw": li, "sig": float(s["sig"] * infl), "d": d,
                "insup": bool(d <= s["d99"]), "nclip": nclip, "model": name, "infl": float(infl)}


class ShotLi:
    """Per-shot li prior: ramp features of every frame + lazily evaluated exponents of a frame's laser profile."""

    def __init__(self, model: LiModel, row, machine, t, th, cfg, fvac=None):
        self.model, self.machine, self.cfg = model, machine, cfg
        te = np.array([te_top(p) for p in th["prof"]], float)
        tip, ip = ip_record(row, machine)
        self.ramp = ramp_features(tip, ip, t, te, machine, tau_scale=float(cfg.get("li_tau_scale", 1.0)), fvac=fvac)
        self.f_edge = float(cfg.get("f_edge", 0.3))
        self.cache = {}

    def frame(self, k, prof):
        """prediction for frame k; prof = the frame's horizontal-laser profile (R, Z, Te, ne) or None."""
        if k in self.cache:
            return self.cache[k]
        feat = {n: float(self.ramp[n][k]) for n in FEATS_RAMP}
        ex = None
        if prof is not None and self.cfg.get("li_use", "auto") != "ramp":
            try:
                ex = exponents(prof, self.f_edge)
            except Exception:  # noqa: BLE001
                ex = None
        feat["aT"], feat["an"] = ex if ex is not None else (np.nan, np.nan)
        pr = self.model.predict(feat, self.cfg.get("li_use", "auto"))
        res = {"feat": feat, "pred": pr}
        self.cache[k] = res
        return res


# ----------------------------------------------------------------------------------------------- in-loop quantities
def solver_li(S, sol, Ip):
    """li of a converged solution in the solver's convention (= inv_core.scalars_kin: 99.5 % surface, perimeter-normalised)."""
    import g1_core as C
    b = sol["b"]
    g, wall = S.grid, S.wall
    span = b["psi_b"] - b["psi_axis"]
    try:
        Cc = C.closed_contour(wall, b["fine"], b["psi_axis"] + 0.995 * span, *b["axis"])
    except Exception:  # noqa: BLE001
        Cc = None
    if Cc is None:
        return np.nan
    R_, Z_ = Cc[:-1, 0], Cc[:-1, 1]
    R2 = np.append(R_, R_[0]); Z2 = np.append(Z_, Z_[0])
    lp = float(np.sum(np.hypot(np.diff(R2), np.diff(Z2))))
    m = b["mask"]
    wv = g.R * m
    spl = b["spl"]
    dpr = spl(g.gR, g.gZ, dx=1); dpz = spl(g.gR, g.gZ, dy=1)
    bp2 = (dpr ** 2 + dpz ** 2) / g.R ** 2
    Bpa = C.MU0 * abs(Ip) / lp
    return float((bp2 * wv).sum() / max(wv.sum(), 1e-12) / Bpa ** 2)


LIC_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.995)


def li_cyl(fR, fZ, fine, spl, psi_axis, psi_b, axis, levels=LIC_LEVELS):
    """Geometry-independent current-peaking functional ("cylinder-equivalent li"): the internal inductance of the circular
    cylinder that carries the same enclosed-current profile I(rho), rho = sqrt(A(u) / A_edge):
        li_c = 2 int_0^1 (I(rho) / I_edge)^2 d rho / rho          (uniform current 0.5, parabolic current 11/12)
    I(u) = (1/mu0) oint_{psi_N = u} |grad psi| / R dl (Ampere on the flux surface), A(u) = area enclosed by the surface,
    edge = the 99.5 % surface.  Works on any flux map (EFIT label or own solution): fine = psi on the grid (fR, fZ)
    (layout [R, Z]), spl = spline of psi, axis = (R, Z) of the magnetic axis; orientation / sign of psi is irrelevant."""
    import contourpy
    cg = contourpy.contour_generator(x=fR, y=fZ, z=fine.T)
    Ra, Za = axis
    rho2, cur = [], []
    for u in levels:
        best = None
        for ln in cg.lines(psi_axis + u * (psi_b - psi_axis)):
            if len(ln) < 8 or np.hypot(*(ln[0] - ln[-1])) > 1e-9:
                continue
            x, y = ln[:-1, 0], ln[:-1, 1]
            # point-in-polygon (ray casting) for the axis
            j = np.arange(len(x)) - 1
            c = ((y > Za) != (y[j] > Za)) & (Ra < (x[j] - x) * (Za - y) / (y[j] - y + 1e-300) + x)
            if c.sum() % 2 == 1 and (best is None or len(ln) > len(best)):
                best = ln
        if best is None:
            return np.nan
        mid = 0.5 * (best[1:] + best[:-1])
        dl = np.hypot(*(best[1:] - best[:-1]).T)
        g = np.hypot(spl.ev(mid[:, 0], mid[:, 1], dx=1), spl.ev(mid[:, 0], mid[:, 1], dy=1))
        cur.append(float(np.sum(g / mid[:, 0] * dl)))
        x, y = best[:, 0], best[:, 1]
        rho2.append(0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))))
    rho = np.sqrt(np.array(rho2) / rho2[-1]); f = np.array(cur) / cur[-1]
    if not (np.all(np.diff(rho) > 0) and np.isfinite(f).all()):
        return np.nan
    y = f ** 2 / rho
    return float(2.0 * (f[0] ** 2 / 4.0 + np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(rho))))


def solver_lic(S, sol):
    b = sol["b"]
    try:
        return li_cyl(S.wall.fR, S.wall.fZ, b["fine"], b["spl"], b["psi_axis"], b["psi_b"], b["axis"])
    except Exception:  # noqa: BLE001
        return np.nan


def wall_gap(S, sol, gap_min):
    """G1's wall-clearance residual dw [m] (g1_core.Solver.solve 'gap'): (psi(wall contact) - psi(X-point)) / |grad psi| + gap_min.
    dw <= 0: the X-point separatrix clears the wall by at least gap_min; dw > 0: clearance violated (limited solution when
    dw > gap_min). None when no wall point / no X-point candidate exists."""
    import g1_core as C
    b = sol["b"]
    if b.get("wpt") is None:
        return None
    RC, ZC, pC = b["wpt"]
    if b["kind"] == "xpt":
        xr = (b["bpt"][0], b["bpt"][1], b["psi_b"])
    else:
        cand = [x for x in b["saddles"] if x[2] < b["psi_b"] - b["tol"] and abs(x[1] - b["axis"][1]) > 0.3 * C.A_PUBLIC[S.machine]]
        xr = max(cand, key=lambda x: x[2]) if cand else None
    if xr is None:
        return None
    spl = b["spl"]
    gn = max(float(np.hypot(spl.ev(RC, ZC, dx=1), spl.ev(RC, ZC, dy=1))), 1e-9)
    return float((pC - xr[2]) / gn + gap_min)


def extra_residuals(S, sol, Ip, li_t, li_sig, cdef, gap_w, gap_min, li_huber=1e9, datum="li"):
    """Fixed-layout extra rows of the LM residual vector: [li pseudo-observation][wall clearance].
    datum 'li': the solution's li (solver convention x cdef); 'lic': its cylinder-equivalent li (li_cyl, cdef = 1)."""
    r, h = [], []
    if li_t is not None:
        v = solver_lic(S, sol) if datum == "lic" else solver_li(S, sol, Ip)
        r.append((cdef * v - li_t) / li_sig if np.isfinite(v) else 0.0); h.append(li_huber)
    if gap_w > 0:
        dw = wall_gap(S, sol, gap_min)
        r.append(max(dw, 0.0) / gap_w if dw is not None else 0.0); h.append(1e9)
    return r, h
