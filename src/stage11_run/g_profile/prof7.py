#!/usr/bin/env python3
"""
Stage 11G / Ch2 TRACK profile (external review, round 2, fix #2) — SOURCE TREATMENT and CURRENT-PROFILE FAMILIES for the
kinetic inverse (INV_P7).  Drop-in replacement of inv_core.KinProfile / FrameModel; nothing in ch2_G1, ch2d_inverse or
e_mast_inv6 is edited (their modules are imported unchanged; `install()` rebinds five names of inv_core in THIS process).

1. Source integration (cfg src)
   native   INV_P5/P6: J at the grid nodes x the binary node mask; Ip normalisation by scipy `romb` on the masked
            (discontinuous) integrand; the GS solver itself sees  sum_ij J_ij dR dZ  (uniform weights).
   cc       conservative cut-cell quadrature: every grid cell [R_i +- dR/2] x [Z_j +- dZ/2] is sub-sampled (quad_n = 4 x 8
            midpoints, bicubic spline of the current psi); the plasma region on the sub-samples = {psi > psi_b} inside the
            wall, cut at the active X-point(s) (same rule as g1_core.analyse), connected component of the axis; the nodal
            source is the CELL AVERAGE of J x indicator, and the SAME sum (uniform weights = what the finite-difference
            solver integrates) normalises Ip, so  sum_ij J_ij dR dZ = Ip  exactly.
   edge_w   flux width of the edge regularisation (0 | 0.01 | 0.02): for u in (1, 1 + edge_w) the family's edge value
            J(u = 1) is continued with a linear ramp to zero (controlled exterior support inside the X-point cut; nothing is
            painted "wherever clipped u = 1").  Irrelevant for a family with J(1) = 0.
2. Families (cfg family), FF'(u) on the normalised flux u = (psi_a - psi)/(psi_a - psi_b):
   fe       INV_P5/P6:  A_ref [a0' + b1 u + b2 u^2],  A_ref = mu0 Ip / int 1/R dA  (finite edge value a0' + b1 + b2)
   cub      review round 2: mu0 R_geo J* [a0 (1-u) + b1 u(1-u) + b2 u(1-u)(2u-1) + e u^2],  J* = Ip / (pi a^2);
            R_geo, a = midplane (Z = Z_axis) half-sum / half-width of the current LCFS; FF'(1) = e mu0 R_geo J*
   lao      = cub with b2 = e = 0 (params without 'b2', 'be'): FF' = (1-u)(a0 + b1 u) - the public EFIT++ magnetics-only
            family (Lao polynomials, n_F = 1, FF'(1) = 0; INTEL §4.5).  p' is already of that form in INV (PePoly:
            p' ~ (1-u)(1 + s u), p'(1) = 0); pe_s_fix freezes s at a DIII-D-fitted value instead of the per-frame fit.
   a0 is always eliminated with the SAME quadrature as the GS source so that the plasma current is Ip.
3. Pressure edge (cfg pedge): offset = INV (the fitted Thomson edge level is dropped: p(1) = 0 by subtraction) | taper = the
   edge level is kept as a pedestal and brought to zero linearly over the last ped_w of normalised flux.
Nothing here reads a file.  Rule 3: the caller installs rule3_guard for MAST and passes inputs only.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.integrate import romb

import g1_core as C
import inv_core as K

MU0 = C.MU0
UG = K.UG

P7_DEFAULT = {"family": "fe", "src": "native", "edge_w": 0.0, "quad_n": [4, 8], "be0": 0.0, "sig_be": 0.15, "be_lo": -1.5, "be_hi": 1.5,
              "pe_s_fix": None, "pedge": "offset", "ped_w": 0.05, "f2_min": 0.02, "lf_target": "flux", "lf_li_sigma": 0.01}
FD7 = {"be": 0.03}
REG7 = ("lcp", "b1", "b2", "be", "gs")


# ----------------------------------------------------------------------------------------------- parameters (+ 'be')
_theta_of0, _clip_x0, _sig0, _pe_model0 = K.theta_of, K.clip_x, K._sig, K.pe_model


def theta_of7(x, cfg):
    th = _theta_of0(x, cfg)
    th["be"] = float(np.clip(cfg.get("be0", 0.0) + cfg.get("sig_be", 0.15) * x.get("be", 0.0), cfg.get("be_lo", -1.5), cfg.get("be_hi", 1.5)))
    return th


def clip_x7(x, cfg):
    y = _clip_x0(x, cfg)
    if "be" in y:
        s, m = cfg.get("sig_be", 0.15), cfg.get("be0", 0.0)
        y["be"] = float(np.clip(y["be"], (cfg.get("be_lo", -1.5) - m) / s, (cfg.get("be_hi", 1.5) - m) / s))
    return y


def _sig7(pn, cfg):
    return cfg.get("sig_be", 0.15) if pn == "be" else _sig0(pn, cfg)


# ----------------------------------------------------------------------------------------------- pressure shape options
class PePoly7(K.PePoly):
    """PePoly with (a) an optional FIXED shape s (class attribute S_GRID of length 1) and (b) the 'taper' edge treatment:
    the fitted edge level is kept as a pedestal  edge * tau(u),  tau = 1 for u < 1 - PED_W, linear to 0 at u = 1."""
    PEDGE, PED_W = "offset", 0.05

    def __init__(self, P0, s, edge=0.0, src="ts"):
        super().__init__(P0, s, edge, src)
        if self.PEDGE == "taper" and self.edge > 0:
            e = min(self.edge, self.P0)                        # a pedestal above the core amplitude is a failed fit
            ramp = UG > 1.0 - self.PED_W
            self.vals = self.vals + e * np.clip((1.0 - UG) / self.PED_W, 0.0, 1.0)
            self.dvals = self.dvals - np.where(ramp, e / self.PED_W, 0.0)

    @classmethod
    def from_shape(cls, pe0, s=0.5, src="shape"):
        return cls(pe0, float(cls.S_GRID[0]) if len(cls.S_GRID) == 1 else s, 0.0, src)

    def scaled(self, f):
        return type(self)(self.P0 * f, self.s, self.edge * f, self.src)


_PE_CACHE = {}


def pe_model7(cfg):
    if cfg.get("pe_model", "poly") == "ispline":
        return K.PeTable
    key = (cfg.get("pe_s_fix"), cfg.get("pedge", "offset"), float(cfg.get("ped_w", 0.05)))
    if key[0] is None and key[1] == "offset":
        return K.PePoly                                        # INV path, unchanged class
    if key not in _PE_CACHE:
        _PE_CACHE[key] = type("PePoly7x", (PePoly7,), {"S_GRID": (np.array([float(key[0])]) if key[0] is not None else K.PePoly.S_GRID),
                                                       "PEDGE": key[1], "PED_W": key[2]})
    return _PE_CACHE[key]


def install():
    """Rebind inv_core's parameter helpers / pressure-model factory in THIS process (adds 'be', s-fix, pedestal taper)."""
    K.theta_of, K.clip_x, K._sig, K.REG, K.pe_model = theta_of7, clip_x7, _sig7, REG7, pe_model7


# ----------------------------------------------------------------------------------------------- cut-cell quadrature
class FineQuad:
    """Conservative cut-cell quadrature of the plasma source on the solver grid (see the module docstring)."""

    def __init__(self, grid: C.Grid, wall: C.Wall, nr=4, nz=8):
        self.grid, self.wall, self.nr, self.nz = grid, wall, int(nr), int(nz)
        offr = ((np.arange(self.nr) + 0.5) / self.nr - 0.5) * grid.dR
        offz = ((np.arange(self.nz) + 0.5) / self.nz - 0.5) * grid.dZ
        self.fR = (grid.gR[:, None] + offr[None, :]).ravel()
        self.fZ = (grid.gZ[:, None] + offz[None, :]).ravel()
        FR, FZ = np.meshgrid(self.fR, self.fZ, indexing="ij")
        self.inside = C.points_in_poly(FR, FZ, wall.pr, wall.pz)
        self.hf = wall.fR[1] - wall.fR[0]

    def cuts(self, b, w=0.0):
        """X-point cuts (Z_x, +1 keep above | -1 keep below): the rule of g1_core.analyse (saddles within 2 tol of the
        boundary flux and within 10 cm of the core region), plus - for an exterior ramp - saddles inside its support."""
        wall = self.wall
        Ra, Za = b["axis"]
        pa, pb, tol = b["psi_axis"], b["psi_b"], b["tol"]
        lab, _ = ndimage.label(b["fine"] > b["c_star"] + 0.25 * tol)
        k = lab[wall.fidx(Ra, Za)]
        near = ndimage.binary_dilation(lab == k, iterations=max(3, int(np.ceil(0.10 / self.hf))))
        out = []
        for x in b["saddles"]:
            if x[2] < pa and near[wall.fidx(x[0], x[1])] and \
                    (abs(x[2] - pb) <= 2 * tol or (w > 0 and x[2] > pb - (w + 0.01) * (pa - pb))):
                out.append((x[1], 1.0 if x[1] < Za else -1.0))
        return out

    def region(self, b, w=0.0):
        """-> box (i0, i1, j0, j1), u_raw (sub-samples), weight (indicator x exterior ramp), R of the sub-sample rows."""
        g, nr, nz = self.grid, self.nr, self.nz
        m = b["mask"]
        ii = np.nonzero(m.any(axis=1))[0]; jj = np.nonzero(m.any(axis=0))[0]
        pad = 2 + (int(np.ceil(150.0 * w)) if w > 0 else 0)
        i0, i1 = max(ii[0] - pad, 0), min(ii[-1] + pad + 1, g.nx)
        j0, j1 = max(jj[0] - pad, 0), min(jj[-1] + pad + 1, g.ny)
        fr, fz = self.fR[i0 * nr:i1 * nr], self.fZ[j0 * nz:j1 * nz]
        pa, pb = b["psi_axis"], b["psi_b"]
        u = (pa - b["spl"](fr, fz)) / (pa - pb)
        A = (u < 1.0 + w) & self.inside[i0 * nr:i1 * nr, j0 * nz:j1 * nz]
        FZ = fz[None, :]
        for zx, s in self.cuts(b, w):
            A &= (FZ > zx) if s > 0 else (FZ < zx)
        lab, _ = ndimage.label(A)
        Ra, Za = b["axis"]
        ia = int(np.clip(np.searchsorted(fr, Ra), 1, len(fr) - 1)); ja = int(np.clip(np.searchsorted(fz, Za), 1, len(fz) - 1))
        sub = lab[max(ia - 2, 0):ia + 2, max(ja - 2, 0):ja + 2]
        nzl = sub[sub > 0]
        if len(nzl) == 0:
            raise ValueError("cut-cell region: axis outside the plasma region")
        wt = (lab == int(np.bincount(nzl).argmax())).astype(float)
        if w > 0:
            wt *= np.clip((1.0 + w - u) / w, 0.0, 1.0)
        return (i0, i1, j0, j1), u, wt, fr

    def cellavg(self, f, box):
        i0, i1, j0, j1 = box
        out = np.zeros((self.grid.nx, self.grid.ny))
        out[i0:i1, j0:j1] = f.reshape(i1 - i0, self.nr, j1 - j0, self.nz).mean(axis=(1, 3))
        return out


def quad_for(S, cfg):
    n = tuple(cfg.get("quad_n", (4, 8)))
    q = getattr(S, "_quad7", None)
    if q is None or (q.nr, q.nz) != n:
        q = FineQuad(S.grid, S.wall, *n)
        S._quad7 = q
    return q


def midplane_extent(b, grid, wall):
    """(R_in, R_out) of the plasma boundary on the horizontal line through the axis (sub-grid, from the spline)."""
    Ra, Za = b["axis"]
    R1 = np.linspace(grid.gR[0], grid.gR[-1], 4 * (grid.nx - 1) + 1)
    p = b["spl"].ev(R1, np.full_like(R1, Za)) - b["psi_b"]
    ia = int(np.argmin(np.abs(R1 - Ra)))
    if not p[ia] > 0:
        return None
    j = ia
    while j + 1 < len(R1) and p[j + 1] > 0:
        j += 1
    Rout = R1[j] + (R1[j + 1] - R1[j]) * p[j] / (p[j] - p[j + 1]) if j + 1 < len(R1) else R1[-1]
    i = ia
    while i - 1 >= 0 and p[i - 1] > 0:
        i -= 1
    Rin = R1[i] - (R1[i] - R1[i - 1]) * p[i] / (p[i] - p[i - 1]) if i - 1 >= 0 else R1[0]
    return float(max(Rin, wall.R_in_wall)), float(min(Rout, float(wall.pr.max())))


# ----------------------------------------------------------------------------------------------- families
def basis7(u, family):
    if family == "fe":
        return [1.0 + 0.0 * u, u, u * u]
    return [1.0 - u, u * (1.0 - u), u * (1.0 - u) * (2.0 * u - 1.0), u * u]


def basis7_int(u0, family):
    """int_{u0}^1 g_k(u) du (analytic)."""
    if family == "fe":
        G = (lambda x: x, lambda x: x ** 2 / 2, lambda x: x ** 3 / 3)
    else:
        G = (lambda x: x - x ** 2 / 2, lambda x: x ** 2 / 2 - x ** 3 / 3, lambda x: -x ** 2 / 2 + x ** 3 - x ** 4 / 2, lambda x: x ** 3 / 3)
    return np.array([f(1.0) - f(u0) for f in G])


class Prof7(K.KinProfile):
    """Kinetic profile with selectable FF' family and source quadrature (duck-types g1_core.Profile in Solver.solve)."""

    def __init__(self, pet, th, cfg, Ip, S):
        super().__init__(pet, th["cp"], th["b"], "fe", Ip)
        self.family, self.src, self.edge_w = cfg.get("family", "fe"), cfg.get("src", "native"), float(cfg.get("edge_w", 0.0))
        self.be = float(th.get("be", 0.0))
        self.S, self.cfg7 = S, cfg
        self.acc_uni = self.acc_romb = self.Rgeo = self.amin = np.nan
        self.plain = self.family == "fe" and self.src == "native"

    def jtor(self, grid: C.Grid, psi, b):
        wc = grid.dR * grid.dZ
        if self.plain:                                         # INV_P5/P6 code path, bit for bit
            J = super().jtor(grid, psi, b)
            self.acc_uni, self.acc_romb = float(J.sum() * wc / self.Ip), float(romb(romb(J)) * wc / self.Ip)
            return J
        D = b["psi_axis"] - b["psi_b"]
        if not D > 0:
            raise ValueError("non-positive flux span")
        if self.src == "native":
            u = np.clip((b["psi_axis"] - psi) / D, 0.0, 1.0)
            m = b["mask"]
            Jp = grid.R * (-self.cp * self.pet.dpe(u) / D) * m
            G = [gk * m / grid.R for gk in basis7(u, self.family)]
            Ipp = romb(romb(Jp)) * wc
            Jk = [romb(romb(Gk)) * wc for Gk in G]
        else:
            q = quad_for(self.S, self.cfg7)
            box, ur, wt, fr = q.region(b, self.edge_w)
            u = np.clip(ur, 0.0, 1.0)
            Rf = fr[:, None]
            pp = np.where(ur <= 1.0, -self.cp * self.pet.dpe(u) / D, 0.0)          # no pressure-gradient current outside
            Jp = q.cellavg(Rf * pp * wt, box)
            G = [q.cellavg(gk / Rf * wt, box) for gk in basis7(u, self.family)]
            Ipp = Jp.sum() * wc
            Jk = [Gk.sum() * wc for Gk in G]
        if not Jk[0] > 1e-12:
            raise ValueError("FF' g0 integral <= 0")
        if self.family == "fe":
            Aref = MU0 * self.Ip / Jk[0]
            ak = [self.b[0] * Aref, self.b[1] * Aref]
        else:
            ext = midplane_extent(b, grid, self.S.wall)
            if ext is None or not ext[1] - ext[0] > 0.05:
                raise ValueError("no midplane extent")
            self.Rgeo, self.amin = 0.5 * (ext[0] + ext[1]), 0.5 * (ext[1] - ext[0])
            Aref = MU0 * self.Rgeo * self.Ip / (np.pi * self.amin ** 2)
            ak = [self.b[0] * Aref, self.b[1] * Aref, self.be * Aref]
        a0 = (MU0 * (self.Ip - Ipp) - sum(a * Jk[k + 1] for k, a in enumerate(ak))) / Jk[0]
        self.a = np.array([a0] + ak)
        self.a0, self.Aref = float(a0), float(Aref)
        self.delta, self.f_press = float(D), float(Ipp / self.Ip)
        J = Jp + sum(a * Gk for a, Gk in zip(self.a, G)) / MU0
        self.acc_uni, self.acc_romb = float(J.sum() * wc / self.Ip), float(romb(romb(J)) * wc / self.Ip)
        return J

    def ffp(self, u):
        """FF'(u) on 0 <= u <= 1 (profile only)."""
        fam = "fe" if self.plain else self.family
        return sum(a * g for a, g in zip(self.a, basis7(np.asarray(u, float), fam)))

    def ff_int(self, u0):
        """int_{u0}^{1 (+ edge_w)} FF' du  (incl. the exterior ramp: FF'(1) edge_w / 2)."""
        fam = "fe" if self.plain else self.family
        v = float(np.dot(self.a, basis7_int(u0, fam)))
        if self.src == "cc" and self.edge_w > 0:
            v += float(self.ffp(1.0)) * self.edge_w / 2.0
        return v

    def f2_min(self, fvac):
        """min over the plasma of F^2 / F_vac^2,  F^2(u) = F_vac^2 + 2 Delta int_u^1 FF' du."""
        uu = np.linspace(0.0, 1.0, 21)
        return float(min(fvac ** 2 + 2.0 * self.delta * self.ff_int(x) for x in uu) / max(fvac ** 2, 1e-30))


# ----------------------------------------------------------------------------------------------- frame model
class FrameModel7(K.FrameModel):
    """inv_core.FrameModel with the Prof7 profile (solve() = the INV code with one changed line + the F^2 > 0 check)."""
    fvac = None

    def solve(self, x, zt, warm=None, pet=None, maxits=None, blend=None):
        cfg, S = self.cfg, self.S
        th = K.theta_of(x, cfg)
        prof = Prof7(pet or self.pet0, th, cfg, self.Ip, S)
        I = self.I.copy()
        I[self.iq] += th["gs"] * self.bps
        if self.edge_out is not None:
            P, e = self.edge_out["P"], self.edge_out["e"]
            obs = [{"name": "tgt", "P": (P[0] + th["dr"] * e[0], P[1] + th["dr"] * e[1]), "e": e, "sig": cfg["closure_sig"], "kind": "eq"}]
            mode, theta0 = cfg["closure_mode"], None
        else:
            obs, mode, theta0 = [], "soft", [self.gev_fixed * self.bps]
        saved = C.Profile
        C.Profile = lambda *a, **k: prof                    # inject the profile into G1's Picard loop
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
            elif self.fvac is not None and not prof.plain:
                f2 = prof.f2_min(self.fvac)
                if not f2 > cfg.get("f2_min", 0.02):
                    r["ok"] = False
                    r["err"] = f"invalid branch: F^2 <= 0 (min F^2/Fvac^2 {f2:.3f})"
        return r


# ----------------------------------------------------------------------------------------------- current accounting
def accounting(S, sol, Ip):
    """Per-frame current accounting of a converged solution (all relative to the measured Ip):
      src_uni   sum_ij J_ij dR dZ          = the current the finite-difference GS solve integrates
      src_romb  romb(romb(J)) dR dZ        = the INV_P5/P6 normalisation quadrature
      circ_pl   (1/mu0) closed-integral B_p . dl of the PLASMA flux alone on a rectangle 2 cells outside the plasma region
                (Ampere on the discrete solution: what the exported map actually carries)
      circ995   (1/mu0) closed-integral |grad psi|/R dl on the 99.5 % surface of the total flux (bicubic spline)
      enc995    the family's continuous current inside u < 0.995 on the solution's own surfaces (cut-cell quadrature)
      circ_ng   as circ995 with np.gradient fields sampled bilinearly (the scorer's field representation)"""
    g, b, prof = S.grid, sol["b"], sol["prof"]
    out = {"acc_uni": float(getattr(prof, "acc_uni", np.nan)), "acc_romb": float(getattr(prof, "acc_romb", np.nan))}
    try:
        from scipy.interpolate import RectBivariateSpline
        m = b["mask"]
        ii = np.nonzero(m.any(axis=1))[0]; jj = np.nonzero(m.any(axis=0))[0]
        i0, i1 = max(ii[0] - 3, 2), min(ii[-1] + 3, g.nx - 3)
        j0, j1 = max(jj[0] - 3, 2), min(jj[-1] + 3, g.ny - 3)
        sp = RectBivariateSpline(g.gR, g.gZ, sol["plasma"])
        r0, r1, z0, z1 = g.gR[i0], g.gR[i1], g.gZ[j0], g.gZ[j1]
        n = 400
        tr = np.linspace(r0, r1, n); tz = np.linspace(z0, z1, n)
        _tz = getattr(np, "trapezoid", None) or np.trapz
        # B_R = -(1/R) dpsi/dZ, B_Z = (1/R) dpsi/dR ; counter-clockwise loop in the (R, Z) plane
        bot = _tz(-sp.ev(tr, np.full(n, z0), dy=1) / tr, tr)                 # +R direction: B_R
        rig = _tz(sp.ev(np.full(n, r1), tz, dx=1) / r1, tz)                  # +Z direction: B_Z
        top = -_tz(-sp.ev(tr, np.full(n, z1), dy=1) / tr, tr)                # -R direction
        lef = -_tz(sp.ev(np.full(n, r0), tz, dx=1) / r0, tz)                 # -Z direction
        out["acc_circ_pl"] = float(abs(bot + rig + top + lef) / MU0 / Ip)
    except Exception:
        pass
    try:
        Ra, Za = b["axis"]
        span = b["psi_b"] - b["psi_axis"]
        Cn = C.closed_contour(S.wall, b["fine"], b["psi_axis"] + 0.995 * span, Ra, Za)
        if Cn is not None:
            mid = 0.5 * (Cn[1:] + Cn[:-1]); dl = np.hypot(*(Cn[1:] - Cn[:-1]).T)
            spl = b["spl"]
            gp = np.hypot(spl.ev(mid[:, 0], mid[:, 1], dx=1), spl.ev(mid[:, 0], mid[:, 1], dy=1))
            out["acc_circ995"] = float(np.sum(gp / mid[:, 0] * dl) / MU0 / Ip)
            psi = sol["psi"]
            dR_, dZ_ = np.gradient(psi, g.dR, axis=0), np.gradient(psi, g.dZ, axis=1)
            bp = np.hypot(dR_, dZ_) / g.R
            fi = (mid[:, 0] - g.gR[0]) / g.dR; fj = (mid[:, 1] - g.gZ[0]) / g.dZ
            out["acc_circ_ng"] = float(np.sum(ndimage.map_coordinates(bp, [fi, fj], order=1) * dl) / MU0 / Ip)
            if hasattr(prof, "a") and np.all(np.isfinite(prof.a)):
                q = quad_for(S, getattr(prof, "cfg7", {}))
                box, ur, wt, fr = q.region(b, 0.0)
                u = np.clip(ur, 0.0, 1.0)
                D = b["psi_axis"] - b["psi_b"]
                Rf = fr[:, None]
                Jf = (Rf * (-prof.cp * prof.pet.dpe(u) / D) + prof.ffp(u) / (MU0 * Rf)) * wt
                wsub = g.dR * g.dZ / (q.nr * q.nz)
                out["acc_enc995"] = float((Jf * (ur < 0.995)).sum() * wsub / Ip)
                out["acc_enc1"] = float(Jf.sum() * wsub / Ip)
    except Exception:
        pass
    return out
