#!/usr/bin/env python3
"""
Stage 11D / Ch2 — small regularised KINETIC INVERSE Grad-Shafranov reconstruction on top of G1 (k = 0.5).
(an internal development note §P items 1-2; note an internal development note)

Per frame, theta = (log c_p, b1, b2, gamma_even[, gamma_q | p2 split]) is fitted by a damped Gauss-Newton /
Levenberg-Marquardt loop (4-8 iterations, forward-difference Jacobian of converged warm-started free-boundary solves,
SVD of the whitened Jacobian -> weak directions frozen at the prior mean). gamma_odd enters through G1's ridge-
regularised vertical force-balance closure (Lackner pin at Z_t, z_eta), i.e. the strongly regularised odd field.

Forward model (one GS solve at fixed theta, measured coil currents, measured Ip):
    J_phi = R p'(psi) + FF'(psi) / (mu0 R)
    p(u)   = c_p * p_e(u),     u = (psi_a - psi) / (psi_a - psi_b),     p'(psi) = -(c_p / Delta) dp_e/du
    p_e(u) = smooth, positive, monotone fit of the Thomson electron pressure mapped into the CURRENT flux coordinate
             (quadratic I-spline basis on knots 0, .2, ..., 1 + a constant; non-negative LS) - re-mapped every Picard step
    FF'(u) = a0 * s_F(u; b1, b2),  'ev': (1-u)(1 + b1 u + b2 u^2)  |  'fe': 1 + b1 u + b2 u^2
             a0 eliminated so that Ip is enforced exactly:  a0 = mu0 (Ip - int R p' dA) / int s_F / R dA
    virtual vacuum fields from g1_core.CoilSet: VBZ (uniform B_z; gamma_even = B_z / Bp*), VQ (even quadrupole), and an
    optional MAST P2 inner/outer split mode  A_P2 * df * (G_in - G_out)  (A_P2 = shipped kA.turn column).
Observations (whitened residuals, Huber):
    isotherm pairs  r = [psi(P_i) - psi(P_j)] / sqrt((g_i s_i)^2 + (g_j s_j)^2 + (0.02 Delta)^2), Huber 2 sigma
    soft edges      r = (psi(P) - psi_b) / slope / sqrt(sig_s^2 + (sig_u Delta / slope)^2), Huber 1.5 sigma ('eq')
                    censored inner edge: the coverage limit must be INSIDE the plasma (one-sided, 'ge')
    priors          (theta - theta0) / sigma_theta  (log c_p ~ N(log c_p0, 0.35^2), gamma ~ N(0, 0.03^2) ...)
Nothing here reads a file. Rule 3: the caller (run_inv.py) installs rule3_guard for MAST and passes inputs only.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.integrate import cumulative_trapezoid, romb
from scipy.interpolate import BSpline, RectBivariateSpline
from scipy.optimize import nnls

import g1_core as C

MU0 = C.MU0

# ----------------------------------------------------------------------------------------------- monotone p_e(u) basis
UG = np.linspace(0.0, 1.0, 401)
_trapz = getattr(np, "trapezoid", None) or np.trapz


def _ispline_basis(knots=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0), k=2):
    t = np.r_[[knots[0]] * k, knots, [knots[-1]] * k]
    n = len(t) - k - 1
    M = np.zeros((n, len(UG)))
    for j in range(n):
        c = np.zeros(n); c[j] = 1.0
        M[j] = np.nan_to_num(BSpline(t, c, k, extrapolate=False)(UG))
    M[:, -1] = M[:, -2] + (M[:, -2] - M[:, -3])            # right end point (BSpline returns nan / 0 there)
    M = np.clip(M, 0.0, None)[:-1]                         # drop the basis with non-zero density at u = 1 -> p'(1) = 0
    M /= _trapz(M, UG)[:, None]
    I = cumulative_trapezoid(M, UG, initial=0.0, axis=1)
    I /= I[:, -1:]
    return M, 1.0 - I                                       # density (dI/du), decreasing basis Phi = 1 - I


MB, PHI = _ispline_basis()
NB = MB.shape[0]


class PeTable:
    """p_e(u) on UG: p_e = sum_j c_j Phi_j(u) (>= 0, monotone non-increasing, p_e(1) = 0); dp_e/du = -sum c_j M_j."""

    def __init__(self, coef, edge=0.0, src="ts"):
        self.coef = np.clip(np.asarray(coef, float), 0.0, None)
        self.vals = self.coef @ PHI
        self.dvals = -(self.coef @ MB)
        self.edge, self.src = float(edge), src
        self.cum = cumulative_trapezoid(self.vals, UG, initial=0.0)    # int_0^u p_e

    @classmethod
    def from_shape(cls, pe0, am=1.38, an=1.08, src="shape"):
        """Fallback: least-squares projection of pe0 (1 - u^am)^an onto the basis."""
        y = (1.0 - UG ** am) ** an
        coef, _ = nnls(PHI.T, y)
        return cls(coef * pe0, 0.0, src)

    @classmethod
    def fit(cls, u, pe, ridge=0.003, umax=1.02, taper=0.07, min_pts=8, min_span=0.4):
        u = np.asarray(u, float); pe = np.asarray(pe, float)
        m = np.isfinite(u) & np.isfinite(pe) & (pe > 0) & (u >= -0.05) & (u <= umax)
        if m.sum() < min_pts:
            return None
        u, pe = np.clip(u[m], 0.0, 1.0), pe[m]
        if u.max() - u.min() < min_span:
            return None
        w = np.clip((umax - u) / taper, 0.0, 1.0) ** 0.5        # points near / beyond the LCFS fade out smoothly
        s = float(np.percentile(pe, 95))
        if not s > 0:
            return None
        A = np.column_stack([np.ones_like(u)] + [np.interp(u, UG, PHI[j]) for j in range(NB)]) * w[:, None]
        b = pe / s * w
        R = np.sqrt(ridge * len(u)) * np.eye(NB + 1)[1:]          # ridge on the shape coefficients (not the edge level)
        coef, _ = nnls(np.vstack([A, R]), np.concatenate([b, np.zeros(NB)]))
        if not coef[1:].sum() > 0:
            return None
        return cls(coef[1:] * s, coef[0] * s, "ts")

    def pe(self, u):
        return np.interp(u, UG, self.vals)

    def dpe(self, u):
        return np.interp(u, UG, self.dvals)

    def scaled(self, f):
        return PeTable(self.coef * f, self.edge * f, self.src)


class PePoly:
    """EFIT-like smooth pressure shape fitted to Thomson: p_e(u) = P0 * phi(u; s) (+ an edge level c0 that is dropped),
    phi(u) = 1 - F(u)/F(1),  F(u) = int_0^u (1 - v)(1 + s v) dv  ->  p_e' = -P0 (1 - u)(1 + s u) / F(1)
    i.e. p'(u) is a low-order edge-vanishing polynomial with p'(0) > 0 and p'(1) = 0 (s >= -0.9 keeps it monotone).
    (The 6-knot I-spline PeTable reproduced the sawtooth-flattened core as p'(0) = 0, which together with a diamagnetic
    FF' (Ip elimination, a0 < 0) gave hollow / negative axis current and Picard failures on DIII-D - see the note.)"""
    S_GRID = np.linspace(-0.9, 4.0, 50)

    def __init__(self, P0, s, edge=0.0, src="ts"):
        self.P0, self.s, self.edge, self.src = float(P0), float(s), float(edge), src
        F1 = 0.5 + self.s / 6.0
        self.vals = self.P0 * (1.0 - self._F(UG, self.s) / F1)
        self.dvals = -self.P0 * (1.0 - UG) * (1.0 + self.s * UG) / F1
        self.coef = np.array([self.P0, self.s])

    @staticmethod
    def _F(u, s):
        return u - u * u / 2 + s * (u * u / 2 - u ** 3 / 3)

    @classmethod
    def from_shape(cls, pe0, s=0.5, src="shape"):
        return cls(pe0, s, 0.0, src)

    @classmethod
    def fit(cls, u, pe, umax=1.02, taper=0.07, min_pts=8, min_span=0.4, **_):
        u = np.asarray(u, float); pe = np.asarray(pe, float)
        m = np.isfinite(u) & np.isfinite(pe) & (pe > 0) & (u >= -0.05) & (u <= umax)
        if m.sum() < min_pts:
            return None
        u, pe = np.clip(u[m], 0.0, 1.0), pe[m]
        if u.max() - u.min() < min_span:
            return None
        w = np.clip((umax - u) / taper, 0.0, 1.0) ** 0.5
        sc = float(np.percentile(pe, 95))
        if not sc > 0:
            return None
        y = pe / sc * w
        best = None
        for s_ in cls.S_GRID:
            phi = 1.0 - cls._F(u, s_) / (0.5 + s_ / 6.0)
            A = np.column_stack([np.ones_like(u), phi]) * w[:, None]
            c, rn = nnls(A, y)
            if c[1] > 0 and (best is None or rn < best[0]):
                best = (rn, s_, c)
        if best is None:
            return None
        _, s_, c = best
        return cls(c[1] * sc, s_, c[0] * sc, "ts")

    def pe(self, u):
        return np.interp(u, UG, self.vals)

    def dpe(self, u):
        return np.interp(u, UG, self.dvals)

    def scaled(self, f):
        return PePoly(self.P0 * f, self.s, self.edge * f, self.src)


def ff_basis(u, basis):
    """FF' basis (g0, g1, g2); g0 is the coefficient eliminated by the Ip constraint."""
    if basis == "fe":
        return 1.0 + 0.0 * u, u, u * u
    return 1.0 - u, u * (1.0 - u), u * u * (1.0 - u)


def ff_basis_int(u0, basis):
    """int_{u0}^1 g_k(u) du for k = 0, 1, 2 (analytic)."""
    if basis == "fe":
        G = (lambda x: x, lambda x: x ** 2 / 2, lambda x: x ** 3 / 3)
    else:
        G = (lambda x: x - x ** 2 / 2, lambda x: x ** 2 / 2 - x ** 3 / 3, lambda x: x ** 3 / 3 - x ** 4 / 4)
    return np.array([f(1.0) - f(u0) for f in G])


class KinProfile:
    """Kinetic profile family (duck-types g1_core.Profile inside G1 Solver.solve: jtor(grid, psi, b))."""

    def __init__(self, pet: PeTable, cp, b, basis, Ip):
        self.pet, self.cp, self.b, self.basis, self.Ip = pet, float(cp), (float(b[0]), float(b[1])), basis, float(Ip)
        self.a0 = np.nan
        self.delta = np.nan
        self.f_press = np.nan             # fraction of Ip carried by the p' term
        self.L, self.Beta0 = np.nan, np.nan   # G1-compatibility attributes (unused)

    def jtor(self, grid: C.Grid, psi, b):
        """J = R p' + FF' / (mu0 R),  FF' = a0 g0 + a1 g1 + a2 g2,  a_k = b_k A_ref (k = 1, 2),  A_ref = mu0 Ip / int g0/R dA,
        a0 eliminated so that int J dA = Ip exactly (audit P1: Ip enforced by eliminating a0; a0 < 0 = diamagnetic allowed)."""
        D = b["psi_axis"] - b["psi_b"]
        if not D > 0:
            raise ValueError("non-positive flux span")
        u = np.clip((b["psi_axis"] - psi) / D, 0.0, 1.0)
        m = b["mask"]
        pprime = -self.cp * self.pet.dpe(u) / D
        Jp = grid.R * pprime * m
        w = grid.dR * grid.dZ
        Ipp = romb(romb(Jp)) * w
        g = ff_basis(u, self.basis)
        Jk = [romb(romb(gk * m / grid.R)) * w for gk in g]
        if not Jk[0] > 1e-12:
            raise ValueError("FF' g0 integral <= 0")
        Aref = MU0 * self.Ip / Jk[0]
        a1, a2 = self.b[0] * Aref, self.b[1] * Aref
        a0 = (MU0 * (self.Ip - Ipp) - a1 * Jk[1] - a2 * Jk[2]) / Jk[0]
        self.a = np.array([a0, a1, a2])
        self.a0, self.Aref = float(a0), float(Aref)
        self.delta, self.f_press = float(D), float(Ipp / self.Ip)
        ffp = (a0 * g[0] + a1 * g[1] + a2 * g[2]) * m
        return Jp + ffp / (MU0 * grid.R)

    def pressure(self, u):
        return self.cp * self.pet.pe(np.clip(u, 0.0, 1.0))

    def ff_int(self, u0):
        """int_{u0}^1 FF'(u) du."""
        return float(np.dot(self.a, ff_basis_int(u0, self.basis)))

    def state(self):
        return {"coef": self.pet.coef.copy(), "edge": self.pet.edge, "src": self.pet.src}


# ----------------------------------------------------------------------------------------------- gamma_shape (coil-based)
def even_pairs(cs, machine):
    """up/down-symmetric circuit pairs of the shipped coil set (unit = 1 A in each member, shipped unit convention)."""
    if machine == "d3d":
        pr = [(f"F{i}A", f"F{i}B") for i in range(1, 10)]
    else:
        pr = [(f"p{i}u_current", f"p{i}l_current") for i in range(2, 7)]
    return [(a, b) for a, b in pr if a in cs.index and b in cs.index]


def install_shape_mode(S, machine):
    """Replace the CoilSet's analytic 'VQ' circuit by gamma_shape: an EVEN, coil-realisable shaping field = the MINIMUM-NORM
    combination of the symmetric PF pairs (norm = each pair's RMS flux over the region inside the wall, so no pair is driven
    to huge cancelling currents) with  B_Z(R0, 0) = 0  (orthogonal to the radial mode at the axis; B_R = 0 there by symmetry,
    i.e. orthogonal to the odd mode) and  B_Z(R0 - a, 0) - B_Z(R0 + a, 0) = 2 T  (field-gradient / decay-index change, same
    normalisation as the analytic quadrupole VQ). Measured currents are untouched: this is a model-discrepancy direction
    (external review, follow-up 1, 'gamma_shape'). An unregularised least-squares match to the quadrupole inside the plasma region
    was tried first and gave cancelling 1e7-1e8 A combinations with a wild flux outside the plasma (pseudo test R2psi 0.43)."""
    cs, g, wall = S.coils, S.grid, S.wall
    if getattr(cs, "_shape_installed", False):
        return cs._shape_info
    R0, a = C.R0_PUBLIC[machine], C.A_PUBLIC[machine]
    pairs = even_pairs(cs, machine)
    E = np.array([cs.G[cs.index[p]] + cs.G[cs.index[q]] for p, q in pairs])          # (n, nx, ny)
    bzu = cs.bz_units
    u0, um, up = bzu(R0, 0.0), bzu(R0 - a, 0.0), bzu(R0 + a, 0.0)
    A = np.array([[u0[cs.index[p]] + u0[cs.index[q]] for p, q in pairs],
                  [(um[cs.index[p]] + um[cs.index[q]]) - (up[cs.index[p]] + up[cs.index[q]]) for p, q in pairs]])
    bvec = np.array([0.0, 2.0])
    Ei = E[:, wall.inside]
    wn = np.sqrt(((Ei - Ei.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))          # RMS flux per unit current over the wall interior
    Winv2 = 1.0 / np.maximum(wn, 1e-30) ** 2
    c = Winv2 * (A.T @ np.linalg.solve((A * Winv2) @ A.T, bvec))                      # min sum (w_j c_j)^2  s.t.  A c = b
    coef = np.zeros(cs.nc)
    for (p, q), cj in zip(pairs, c):
        coef[cs.index[p]] += cj; coef[cs.index[q]] += cj
    smap = np.tensordot(coef, cs.G, axes=1)
    _, vq = cs._modes_psi(g.R, g.Z)
    reg = wall.inside & (((g.R - R0) / (1.1 * a)) ** 2 + (g.Z / (2.0 * a)) ** 2 < 1.0)
    d = (smap[reg] - smap[reg].mean()) - (vq[reg] - vq[reg].mean())
    r2 = 1.0 - (d ** 2).sum() / ((vq[reg] - vq[reg].mean()) ** 2).sum()
    rms_s = float(np.std(smap[wall.inside])); rms_q = float(np.std(vq[wall.inside]))
    cs.G[cs.iq] = smap
    orig = (cs.psi_units, cs.br_units, cs.bz_units)

    def wrap(f):
        def h(R, Z):
            u = f(R, Z)
            u[cs.iq] = float(np.dot(coef, u))
            return u
        return h
    cs.psi_units, cs.br_units, cs.bz_units = (wrap(f) for f in orig)
    cs._shape_installed = True
    cs._shape_info = {"pairs": [p for p, _ in pairs], "coef_A_per_T": [float(coef[cs.index[p]]) for p, _ in pairs],
                      "r2_vs_quadrupole_in_plasma": float(r2), "rms_flux_in_wall_vs_VQ": rms_s / max(rms_q, 1e-30),
                      "bz_axis": float(np.dot(coef, bzu(R0, 0.0)))}
    return cs._shape_info


# ----------------------------------------------------------------------------------------------- observations
def _smooth(y):
    m = y.copy()
    if len(y) >= 3:
        p = np.pad(y, 1, mode="edge")
        m = np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)
        p = np.pad(m, 1, mode="edge")
        m = (p[:-2] + p[1:-1] + p[2:]) / 3.0
    return m


def _walk(x, y, L, i0, step):
    """From the peak index i0 walk in direction step while y >= L; returns (x_cross, idx_lo, idx_hi) or None if censored."""
    i, n = i0, len(x)
    while 0 <= i + step < n and y[i + step] >= L:
        i += step
    j = i + step
    if not (0 <= j < n):
        return None
    t = (y[i] - L) / (y[i] - y[j]) if y[i] != y[j] else 0.5
    return float(x[i] + t * (x[j] - x[i])), i, j


def _n_cross(y, L, band=0.03):
    """number of crossings of level L with hysteresis +-band*L (robust count)."""
    hi, lo = L * (1 + band), L * (1 - band)
    st, n = None, 0
    for v in y:
        s = 1 if v > hi else (-1 if v < lo else None)
        if s is None:
            continue
        if st is not None and s != st:
            n += 1
        st = s
    return n


def chord_crossings(x, Te, levels, rel_noise=0.07, min_pts=8, max_sig=0.03):
    """Isotherm crossings on one chord (positions x along it). Returns (Tmax, {f: {'lo': (x, sig), 'hi': (x, sig)}})
    with a side present only if it is UNAMBIGUOUS (exactly one robust crossing on that side of the peak) and uncensored."""
    x = np.asarray(x, float); Te = np.asarray(Te, float)
    ok = np.isfinite(x) & np.isfinite(Te) & (Te > 0)
    if ok.sum() < min_pts:
        return np.nan, {}
    o = np.argsort(x[ok]); xs, te = x[ok][o], Te[ok][o]
    ts = _smooth(te)
    ip = int(np.argmax(ts)); Tm = float(ts[ip])
    rms = float(np.sqrt(np.median((te - ts) ** 2)))
    dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 0.01
    out = {}
    for f in levels:
        L = f * Tm
        rec = {}
        for side, step, seg in (("lo", -1, ts[:ip + 1]), ("hi", +1, ts[ip:])):
            if _n_cross(seg, L) != 1:
                continue
            w = _walk(xs, ts, L, ip, step)
            if w is None:
                continue
            xc, i, j = w
            slope = abs(ts[i] - ts[j]) / max(abs(xs[i] - xs[j]), 1e-6)
            sT = max(rel_noise * L, rms)
            sig = float(np.hypot(sT / max(slope, 1e-9), 0.3 * dx))
            if sig > max_sig:                              # tiny gradient: ambiguous position -> reject
                continue
            rec[side] = (xc, max(sig, 0.003))
        if rec:
            out[f] = rec
    return Tm, out


def mast_iso_obs(prof_k, levels, zl):
    """MAST midplane laser: inboard/outboard crossing pairs (horizontal chord at Z = zl)."""
    if prof_k is None:
        return []
    xr, _, Te, _ = prof_k
    Tm, cr = chord_crossings(xr, Te, levels)
    obs = []
    for f, rec in cr.items():
        if "lo" in rec and "hi" in rec:
            obs.append({"f": f, "pts": [((rec["lo"][0], zl), (-1.0, 0.0), rec["lo"][1]), ((rec["hi"][0], zl), (1.0, 0.0), rec["hi"][1])],
                        "pairs": [(0, 1)]})
    return obs


def d3d_iso_obs(vert, tang, levels):
    """DIII-D real chords: vertical core chord (R = Rc, Z up) and tangential chord (Z ~ -0.06, R). Level crossings of the
    common Te maximum; pairs (vertical upper crossing <-> tangential outboard) and (tangential inboard <-> outboard)."""
    Tv = Tt = np.nan
    if vert is not None:
        Rc, zc, te_v, _ = vert
        okv = np.isfinite(te_v) & (te_v > 0)
        Tv = float(np.max(_smooth(te_v[okv][np.argsort(zc[okv])]))) if okv.sum() >= 8 else np.nan
    if tang is not None:
        rt, zt_, te_t, _ = tang
        okt = np.isfinite(te_t) & (te_t > 0)
        Tt = float(np.max(_smooth(te_t[okt][np.argsort(rt[okt])]))) if okt.sum() >= 4 else np.nan
    Tm = np.nanmax([Tv, Tt]) if np.isfinite([Tv, Tt]).any() else np.nan
    if not np.isfinite(Tm):
        return []
    lv = {}
    if vert is not None and np.isfinite(Tv):
        # absolute levels f*Tm on the vertical chord: rescale fractions to the chord's own maximum
        fl = [f * Tm / Tv for f in levels]
        _, cr = chord_crossings(zc, te_v, [x for x in fl if x < 0.98])
        for f, x in zip(levels, fl):
            if x in cr and "hi" in cr[x]:
                lv.setdefault(f, []).append(((float(np.nanmedian(Rc)), cr[x]["hi"][0]), (0.0, 1.0), cr[x]["hi"][1]))
    if tang is not None and np.isfinite(Tt):
        fl = [f * Tm / Tt for f in levels]
        _, cr = chord_crossings(rt, te_t, [x for x in fl if x < 0.98], min_pts=4)
        zt0 = float(np.nanmedian(zt_))
        for f, x in zip(levels, fl):
            if x in cr:
                for side, e in (("hi", 1.0), ("lo", -1.0)):
                    if side in cr[x]:
                        lv.setdefault(f, []).append(((cr[x][side][0], zt0), (e, 0.0), cr[x][side][1], side))
    obs = []
    for f, pts in lv.items():
        P = [(p[0], p[1], p[2]) for p in pts]
        tags = [("v" if len(p) == 3 else p[3]) for p in pts]
        pairs = []
        if "hi" in tags:
            j = tags.index("hi")
            for i, tg in enumerate(tags):
                if i != j:
                    pairs.append((i, j))
        if pairs:
            obs.append({"f": f, "pts": P, "pairs": pairs})
    return obs


def residuals(sol, iso, edges, cfg):
    """Whitened residual vector (fixed layout for a fixed observation list) + Huber thresholds + names."""
    b = sol["b"]
    spl = b["spl"]
    D = b["psi_axis"] - b["psi_b"]
    r, h, nm = [], [], []
    for lev in iso:
        vals = []
        for (P, e, sg) in lev["pts"]:
            p = float(spl.ev(*P)); pr, pz = float(spl.ev(*P, dx=1)), float(spl.ev(*P, dy=1))
            slope = max(abs(e[0] * pr + e[1] * pz), 0.3 * np.hypot(pr, pz), 1e-12)
            vals.append((p, slope * sg))
        for i, j in lev["pairs"]:
            s = np.sqrt(vals[i][1] ** 2 + vals[j][1] ** 2 + (cfg["iso_floor"] * D) ** 2)
            r.append((vals[i][0] - vals[j][0]) / s); h.append(cfg["iso_huber"]); nm.append(f"iso{lev['f']:g}_{i}{j}")
    for o in edges:
        P, e = o["P"], o["e"]
        pr, pz = float(spl.ev(*P, dx=1)), float(spl.ev(*P, dy=1))
        slope = max(-(e[0] * pr + e[1] * pz), 0.3 * np.hypot(pr, pz), 1e-12)
        d = (float(spl.ev(*P)) - b["psi_b"]) / slope
        se = np.hypot(o["sig"], cfg["sigma_u"] * D / slope)
        if o["kind"] == "ge":
            d = min(d, 0.0)
        r.append(d / se); h.append(cfg["edge_huber"]); nm.append(f"edge_{o['name']}_{o['kind']}")
    return np.array(r, float), np.array(h, float), nm


def huber_cost(r, h):
    a = np.abs(r)
    return float(np.sum(np.where(a <= h, a * a, 2 * h * a - h * h)))


def edge_deltas(sol, edges):
    b = sol["b"]; spl = b["spl"]; out = {}
    for o in edges:
        P, e = o["P"], o["e"]
        pr, pz = float(spl.ev(*P, dx=1)), float(spl.ev(*P, dy=1))
        slope = max(-(e[0] * pr + e[1] * pz), 0.3 * np.hypot(pr, pz), 1e-12)
        out[o["name"]] = (float(spl.ev(*P)) - b["psi_b"]) / slope
    return out


def iso_mismatch(spl, iso):
    """Label-free check: for every pair, the displacement [m] of point i along its chord that would put it on the
    flux surface of point j:  (psi_i - psi_j) / |e_i . grad psi(P_i)|  (signed)."""
    out = []
    for lev in iso:
        for i, j in lev["pairs"]:
            (Pi, ei, _), (Pj, _, _) = lev["pts"][i], lev["pts"][j]
            pr, pz = float(spl.ev(*Pi, dx=1)), float(spl.ev(*Pi, dy=1))
            sl = max(abs(ei[0] * pr + ei[1] * pz), 0.3 * np.hypot(pr, pz), 1e-12)
            out.append((lev["f"], (float(spl.ev(*Pi)) - float(spl.ev(*Pj))) / sl))
    return out


# ----------------------------------------------------------------------------------------------- scalars
def scalars_kin(grid, wall, psi, b, prof: KinProfile, fvac, Ip, chord=None):
    """G1 scalars with the kinetic profile: F(psi95)^2 = F_b^2 + 2 Delta int_0.95^1 FF' du; <p>_V from c_p p_e(u)."""
    out = {}
    Ra, Za = b["axis"]
    span = b["psi_b"] - b["psi_axis"]
    spl, fine = b["spl"], b["fine"]
    D = -span
    try:
        C95 = C.closed_contour(wall, fine, b["psi_axis"] + 0.95 * span, Ra, Za)
        if C95 is not None:
            f95 = np.sqrt(max(fvac ** 2 + 2.0 * D * prof.ff_int(0.95), 1e-12))
            out["q95_vac"] = C.q_on_contour(spl, C95, fvac)
            out["f95_ratio"] = float(f95 / max(abs(fvac), 1e-12))
            out["q95"] = out["q95_vac"] * out["f95_ratio"]
    except Exception:
        pass
    try:
        Cc = C.closed_contour(wall, fine, b["psi_axis"] + 0.995 * span, Ra, Za)
        if Cc is not None:
            R_, Z_ = Cc[:-1, 0], Cc[:-1, 1]
            Rmax, Rmin, Zmax, Zmin = R_.max(), R_.min(), Z_.max(), Z_.min()
            a = 0.5 * (Rmax - Rmin); Rgeo = 0.5 * (Rmax + Rmin)
            R2 = np.append(R_, R_[0]); Z2 = np.append(Z_, Z_[0])
            cross = R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]
            vol = 2 * np.pi * abs(np.sum((R2[:-1] + R2[1:]) * cross) / 6.0)
            lp = float(np.sum(np.hypot(np.diff(R2), np.diff(Z2))))
            out.update({"a": float(a), "Rgeo": float(Rgeo), "kappa": float((Zmax - Zmin) / (2 * a)),
                        "tri_top": float((Rgeo - R_[np.argmax(Z_)]) / a), "tri_bot": float((Rgeo - R_[np.argmin(Z_)]) / a),
                        "volume": float(vol), "lp": lp, "Rin": float(Rmin), "Rout": float(Rmax), "Ztop": float(Zmax), "Zbot": float(Zmin)})
            if chord is not None:
                if chord[0] == "h":
                    rr = C.crossings_h(Cc, chord[1])
                    if rr:
                        out["edge_gs"], out["edge_gs_in"] = float(max(rr)), float(min(rr))
                else:
                    zz = C.crossings_v(Cc, chord[1])
                    if zz:
                        out["edge_gs"] = float(max(zz))
            m = b["mask"]
            u = np.clip((psi - b["psi_axis"]) / span, 0, 1)
            p = prof.pressure(u) * m
            wv = grid.R * m
            pav = float((grid.R * p).sum() / max(wv.sum(), 1e-12))
            B0 = abs(fvac) / Rgeo
            out["pav"] = pav
            out["betaN"] = float(100 * 2 * MU0 * pav * a / (B0 * (abs(Ip) / 1e6)))
            Bpa = MU0 * abs(Ip) / lp
            out["betap"] = float(2 * MU0 * pav / Bpa ** 2)
            dpr = spl(grid.gR, grid.gZ, dx=1); dpz = spl(grid.gR, grid.gZ, dy=1)
            bp2 = (dpr ** 2 + dpz ** 2) / grid.R ** 2
            out["li"] = float((bp2 * wv).sum() / max(wv.sum(), 1e-12) / Bpa ** 2)
            out["B0"] = float(B0)
    except Exception:
        pass
    return out


# ----------------------------------------------------------------------------------------------- the frame inverse
INV_DEFAULT = {
    "basis": "ev", "params": ["lcp", "b1", "b2", "dr"],
    "lcp0": float(np.log(2.0)), "sig_lcp": 0.35, "lcp_lo": float(np.log(1.2)), "lcp_hi": float(np.log(6.0)),
    "b0": [0.0, 0.0], "sig_b": [0.5, 0.5], "b_lo": -3.0, "b_hi": 3.0,
    "gev0": 0.0, "sig_gev": 0.03, "sig_gs": 0.03, "g_clip": 0.30, "sig_dr": 0.03, "dr_clip": 0.15,
    "fd_step": {"lcp": 0.08, "b1": 0.1, "b2": 0.1, "dr": 0.01, "gs": 0.01}, "pe_model": "poly",
    "iso_huber": 2.0, "edge_huber": 1.5, "iso_floor": 0.02, "sigma_u": 0.015, "sigma_in": 0.03, "use_inner_eq": True,
    "lm_its": 6, "lm_mu0": 1e-2, "svd_min": 0.3, "trust": 1.5, "freeze": True, "remap": True,
    "levels": [0.20, 0.35, 0.50, 0.65, 0.80], "max_sig_R": 0.03, "tau_t": 0.0, "shape_mode": "coil",
    # G1 solver settings used by the inner solves (tighter than G1's 3e-3 so that FD Jacobians are clean)
    "rtol": 2e-4, "hard_tol": 1e-4,
    # radial closure inside the Picard loop: stiff soft iso-flux (5 mm, G1's 0.03 Bp* prior keeps the field bounded) + branch checks
    "closure_mode": "soft", "closure_sig": 0.005, "closure_tol": 0.03,
    "axis_range": {"mast": [0.45, 1.25], "d3d": [1.40, 2.05]},
}
REG = ("lcp", "b1", "b2", "gs")                     # parameters with a Gaussian prior (dr is penalised by the edge datum)


def theta_of(x, cfg):
    """whitened x (dict name -> value) -> physical parameters."""
    return {"cp": float(np.exp(np.clip(cfg["lcp0"] + cfg["sig_lcp"] * x.get("lcp", 0.0), cfg["lcp_lo"], cfg["lcp_hi"]))),
            "b": (float(np.clip(cfg["b0"][0] + cfg["sig_b"][0] * x.get("b1", 0.0), cfg["b_lo"], cfg["b_hi"])),
                  float(np.clip(cfg["b0"][1] + cfg["sig_b"][1] * x.get("b2", 0.0), cfg["b_lo"], cfg["b_hi"]))),
            "gs": float(np.clip(cfg["sig_gs"] * x.get("gs", 0.0), -cfg["g_clip"], cfg["g_clip"])),
            "dr": float(np.clip(cfg["sig_dr"] * x.get("dr", 0.0), -cfg["dr_clip"], cfg["dr_clip"]))}


def clip_x(x, cfg):
    y = dict(x)
    if "lcp" in y:
        y["lcp"] = float(np.clip(y["lcp"], (cfg["lcp_lo"] - cfg["lcp0"]) / cfg["sig_lcp"], (cfg["lcp_hi"] - cfg["lcp0"]) / cfg["sig_lcp"]))
    for j, n in enumerate(("b1", "b2")):
        if n in y:
            y[n] = float(np.clip(y[n], (cfg["b_lo"] - cfg["b0"][j]) / cfg["sig_b"][j], (cfg["b_hi"] - cfg["b0"][j]) / cfg["sig_b"][j]))
    if "gs" in y:
        y["gs"] = float(np.clip(y["gs"], -cfg["g_clip"] / cfg["sig_gs"], cfg["g_clip"] / cfg["sig_gs"]))
    if "dr" in y:
        y["dr"] = float(np.clip(y["dr"], -cfg["dr_clip"] / cfg["sig_dr"], cfg["dr_clip"] / cfg["sig_dr"]))
    return y


def _sig(pn, cfg):
    return {"lcp": cfg["sig_lcp"], "b1": cfg["sig_b"][0], "b2": cfg["sig_b"][1], "gs": cfg["sig_gs"], "dr": cfg["sig_dr"]}[pn]


def warm_of(sol):
    return {"plasma": sol["plasma"], "dIz": sol["dIz"], "theta": sol["theta"], "axis": sol["b"]["axis"]}


class FrameModel:
    """Forward model of one frame at a whitened parameter vector x.
    Radial closure (G1 Solver.solve, hard mode): the uniform vertical field gamma_even is solved inside the Picard loop so
    that the LCFS passes through the measured outer Thomson edge shifted by dr along the chord; gamma_even is therefore a
    DERIVED quantity that carries its own prior N(gev0, sig_gev^2) in the objective (exact reparametrisation of the joint
    MAP over (gamma_even, profile) that keeps the radial position pinned and the forward solve robust).
    Without an outer edge: the field is fixed at gev_fixed * Bp* (carried from the neighbours / prior)."""

    def __init__(self, S, I_base, Ip, bps, pet0, ts_pts, cfg, iq, edge_out=None, gev_fixed=0.0, anchor=None):
        self.S, self.I, self.Ip, self.bps, self.pet0, self.ts_pts, self.cfg = S, I_base, Ip, bps, pet0, ts_pts, cfg
        self.iq, self.edge_out, self.gev_fixed, self.anchor = iq, edge_out, float(gev_fixed), anchor
        self.n_solves, self.secs = 0, 0.0

    def solve(self, x, zt, warm=None, pet=None, maxits=None, blend=None):
        cfg, S = self.cfg, self.S
        th = theta_of(x, cfg)
        prof = KinProfile(pet or self.pet0, th["cp"], th["b"], cfg["basis"], self.Ip)
        I = self.I.copy()
        I[self.iq] += th["gs"] * self.bps
        if self.edge_out is not None:
            P, e = self.edge_out["P"], self.edge_out["e"]
            obs = [{"name": "tgt", "P": (P[0] + th["dr"] * e[0], P[1] + th["dr"] * e[1]), "e": e, "sig": cfg["closure_sig"], "kind": "eq"}]
            mode, theta0 = cfg["closure_mode"], None
        else:
            obs, mode, theta0 = [], "soft", [self.gev_fixed * self.bps]
        saved = C.Profile
        C.Profile = lambda *a, **k: prof                    # inject the kinetic profile into G1's Picard loop
        try:
            ok, r = S.solve(I, self.Ip, 1.0, obs=obs, warm=warm, anchor=self.anchor, theta0=theta0, edge_mode=mode, zt=zt,
                            maxits=maxits, blend=blend)
        finally:
            C.Profile = saved
        self.n_solves += 1
        self.secs += r["secs"]
        r["x"], r["par"], r["kin"] = dict(x), th, prof
        r["gev"] = float(r["theta"][0] / self.bps) if r.get("theta") else self.gev_fixed
        if r["ok"]:                                          # reject spurious branches (far O-point, saturated closure)
            lo, hi = cfg["axis_range"][S.machine]
            Ra = r["b"]["axis"][0]
            dt = abs(r.get("delta", {}).get("tgt", 0.0)) if self.edge_out is not None else 0.0
            if not (lo <= Ra <= hi) or r.get("clip") or dt > cfg["closure_tol"]:
                r["ok"] = False
                r["err"] = f"invalid branch: R_axis {Ra:.3f}, clip {r.get('clip')}, |d_tgt| {dt:.3f}"
        return r

    def robust_solve(self, x, zt, warm, pet=None):
        r = self.solve(x, zt, warm=warm, pet=pet)
        if r["ok"]:
            return r
        if warm is not None:
            r = self.solve(x, zt, warm=warm, pet=pet, maxits=160, blend=0.6)
            if r["ok"]:
                return r
        return self.solve(x, zt, warm=None, pet=pet, maxits=160, blend=0.5)


def ridge_z(sol, zt, z_eta, a_pub, zt_max):
    f0, f1 = sol.get("fz0"), sol.get("fz1")
    if f0 is None or f1 is None or abs(f1) < 1e-6 or z_eta <= 0:
        return zt
    lam = z_eta * sol["bps"] / a_pub
    return float(np.clip(f1 * (f1 * zt - f0) / (f1 ** 2 + lam ** 2), -zt_max, zt_max))


def pe_model(cfg):
    return PeTable if cfg.get("pe_model", "poly") == "ispline" else PePoly


def refit_pet(sol, ts_pts, cfg):
    if ts_pts is None:
        return None
    b = sol["b"]
    u = (b["psi_axis"] - b["spl"].ev(ts_pts[0], ts_pts[1])) / (b["psi_axis"] - b["psi_b"])
    return pe_model(cfg).fit(u, ts_pts[2])


def invert_frame(fm: FrameModel, x0, zt0, warm, iso, edges, z_eta, a_pub, zt_max, params=None, resid_fn=None, x_prior=None):
    """Damped Gauss-Newton / LM on the whitened parameters with an FD Jacobian of converged warm-started solves.
    Objective = sum Huber(iso pairs) + Huber(outer edge) + one-sided(censored inner edge) + (gamma_even prior)^2
              + |x_reg|^2.  The p_e(u) table is re-fitted from the Thomson points in the flux coordinate of every new base
    solution (alternating update; the table is held fixed inside one Jacobian / step evaluation)."""
    cfg = fm.cfg
    params = list(params if params is not None else cfg["params"])
    if fm.edge_out is None:
        params = [p for p in params if p != "dr"]
    x = clip_x({n: float(x0.get(n, 0.0)) for n in params}, cfg)
    info = {"lm_it": 0, "n_rej": 0, "sv": [], "n_frozen": 0, "cost0": np.nan, "cost": np.nan, "err": "", "n_refit": 0}
    zt = zt0
    sol = fm.robust_solve(x, zt, warm)
    if not sol["ok"]:
        info["err"] = "base: " + sol["err"]
        return None, info

    def rebase(s, xx, zz):
        """ridge-Z follow-up + p_e table re-fit in the new flux coordinate (each followed by one warm re-solve)."""
        z1 = ridge_z(s, zz, z_eta, a_pub, zt_max)
        pet = refit_pet(s, fm.ts_pts, cfg) if cfg["remap"] else None
        if abs(z1 - zz) > 0.002 or pet is not None:
            s2 = fm.solve(xx, z1, warm=warm_of(s), pet=pet or s["prof"].pet)
            if s2["ok"]:
                info["n_refit"] += int(pet is not None)
                return s2, z1
        return s, zz

    sol, zt = rebase(sol, x, zt)
    # Gaussian prior on the regularised parameters (whitened): static N(0, 1), or the OU / Kalman-like temporal prior
    # N(rho x_prev, 1 - rho^2) when x_prior = (mean dict, var dict) is given (weak temporal smoothing along the shot)
    pmean = np.array([float((x_prior or ({}, {}))[0].get(p, 0.0)) if p in REG else 0.0 for p in params])
    pvar = np.array([float((x_prior or ({}, {}))[1].get(p, 1.0)) if p in REG else np.inf for p in params])
    pm = np.where(np.isfinite(pvar), 1.0 / pvar, 0.0)

    def rfun(s):
        if resid_fn is not None:
            r_, h_ = resid_fn(s)
        else:
            r_, h_, _ = residuals(s, iso, edges, cfg)
        if np.isfinite(cfg["sig_gev"]) and fm.edge_out is not None:
            r_ = np.append(r_, (s["gev"] - cfg["gev0"]) / cfg["sig_gev"]); h_ = np.append(h_, 1e9)
        return r_, h_

    def total(s, xx):
        r_, h_ = rfun(s)
        xv_ = np.array([xx[p] for p in params])
        return huber_cost(r_, h_) + float(np.sum(pm * (xv_ - pmean) ** 2)), r_, h_

    cost, r, h = total(sol, x)
    info["cost0"] = cost
    has_data = bool(iso) or bool(edges) or resid_fn is not None
    if not has_data or not params:
        info["cost"], info["zt"] = cost, zt
        return sol, info
    mu, tr = cfg["lm_mu0"], cfg["trust"]
    n = len(params)
    for it in range(cfg["lm_its"]):
        info["lm_it"] = it + 1
        w = np.minimum(1.0, h / np.maximum(np.abs(r), 1e-12))
        Jm = np.zeros((len(r), n))
        wb, pet = warm_of(sol), sol["prof"].pet
        for j, pn in enumerate(params):
            hx = cfg["fd_step"][pn] / _sig(pn, cfg)
            xj = dict(x); xj[pn] = x[pn] + hx
            xj = clip_x(xj, cfg)
            if abs(xj[pn] - x[pn]) < 0.5 * hx:
                xj[pn] = x[pn] - hx
            sj = fm.solve(xj, zt, warm=wb, pet=pet)
            if not sj["ok"]:
                xj = dict(x); xj[pn] = x[pn] - hx
                xj = clip_x(xj, cfg)
                sj = fm.solve(xj, zt, warm=wb, pet=pet) if abs(xj[pn] - x[pn]) > 0.5 * hx else sj
            if sj["ok"]:
                rj, _ = rfun(sj)
                Jm[:, j] = (rj - r) / (xj[pn] - x[pn])
        sw = np.sqrt(w)
        Jw, rw = Jm * sw[:, None], r * sw
        xv = np.array([x[p] for p in params])
        A = Jw.T @ Jw + np.diag(pm + 1e-6)
        g = Jw.T @ rw + pm * (xv - pmean)
        _, sv, Vt = np.linalg.svd(Jw, full_matrices=False)
        info["sv"] = [float(v) for v in sv]
        strong = sv >= cfg["svd_min"] if cfg["freeze"] else np.ones(len(sv), bool)
        Vs, Vw = Vt[strong].T, Vt[~strong].T
        info["n_frozen"] = int((~strong).sum())
        accepted = False
        for _ in range(4):
            d = np.zeros(n)
            if Vs.shape[1]:
                As = Vs.T @ (A + mu * np.diag(np.diag(A))) @ Vs
                d = -Vs @ np.linalg.solve(As, Vs.T @ g)
            if Vw.shape[1]:
                d += -Vw @ (Vw.T @ (np.minimum(pm, 1.0) * (xv - pmean)))   # weak directions -> prior mean
            nd = float(np.linalg.norm(d))
            if nd > tr:
                d *= tr / nd
            xn = clip_x({p: x[p] + float(d[k]) for k, p in enumerate(params)}, cfg)
            sn = fm.solve(xn, zt, warm=wb, pet=pet)
            if sn["ok"]:
                cn, rn, hn = total(sn, xn)
                if cn < cost - 1e-9:
                    accepted = True
                    break
            info["n_rej"] += 1
            mu *= 5.0
            tr *= 0.5
        if not accepted:
            break
        dc = cost - cn
        x, sol = xn, sn
        mu = max(mu / 3.0, 1e-4)
        tr = min(cfg["trust"], tr * 1.5)
        sol, zt = rebase(sol, x, zt)
        cost, r, h = total(sol, x)
        if nd < 0.02 or dc < 1e-3 * max(cost, 1.0):
            break
    info["cost"], info["zt"] = cost, zt
    return sol, info
