#!/usr/bin/env python3
"""Stage 11K / Ch2 corrector — map-space realisation of control-point corrections.

warp(psi, R, Z, P_old, d, a, b): psi_new(x) = psi_old(x - u(x)); u = normalised-Gaussian-RBF blend of the control displacements d
  (exact at the NEW control positions P_old + d, reproduces a rigid shift exactly), multiplied by a taper that is 1 on the
  bounding box of the control points and 0 beyond `margin * a` outside it  =>  the vacuum far field / coil regions are untouched.
li_remap(...): inside the last closed contour, phi -> lev + (phi - lev)(1 + c s), s = normalised flux (0 at the boundary, 1 on axis);
  value and gradient continuous at the boundary; c solved so that <Bp^2>_V changes by the requested ratio.
psi arrays are [Z, R] (shipped orientation)."""
import numpy as np
from scipy.interpolate import RectBivariateSpline


def disp_field(RR, ZZ, cen, d, sR, sZ, margin):
    n = len(cen)
    Phi = np.exp(-0.5 * (((cen[:, None, 0] - cen[None, :, 0]) / sR) ** 2 + ((cen[:, None, 1] - cen[None, :, 1]) / sZ) ** 2))
    Phi = Phi / Phi.sum(1, keepdims=True)
    W = np.linalg.solve(Phi + 1e-9 * np.eye(n), d)
    B = np.exp(-0.5 * (((RR[..., None] - cen[:, 0]) / sR) ** 2 + ((ZZ[..., None] - cen[:, 1]) / sZ) ** 2))
    B = B / np.maximum(B.sum(-1, keepdims=True), 1e-300)
    u = B @ W
    r0, r1, z0, z1 = cen[:, 0].min(), cen[:, 0].max(), cen[:, 1].min(), cen[:, 1].max()
    dx = np.maximum(np.maximum(r0 - RR, RR - r1), 0.0)
    dz = np.maximum(np.maximum(z0 - ZZ, ZZ - z1), 0.0)
    q = np.clip(np.hypot(dx, dz) / margin, 0.0, 1.0)
    return u * (0.5 * (1.0 + np.cos(np.pi * q)))[..., None]


def warp(psi, R, Z, P_old, d, a, b, s=0.45, margin=0.5, cap=0.3):
    """P_old (n,2) [R,Z] control points of the old map, d (n,2) displacements [m]; a, b = half width / half height of the contour."""
    d = np.asarray(d, float).copy()
    nrm = np.hypot(d[:, 0], d[:, 1])
    f = np.minimum(1.0, cap * a / np.maximum(nrm, 1e-12))
    d *= f[:, None]
    if not np.isfinite(d).all() or not np.isfinite(P_old).all() or np.abs(d).max() < 1e-5:
        return psi
    RR, ZZ = np.meshgrid(R, Z)
    u = disp_field(RR, ZZ, np.asarray(P_old, float) + d, d, s * a, s * b, margin * a)
    spl = RectBivariateSpline(Z, R, psi, kx=3, ky=3)
    zz = np.clip(ZZ - u[..., 1], Z[0], Z[-1]); rr = np.clip(RR - u[..., 0], R[0], R[-1])
    return spl.ev(zz, rr)


def li_coef(psi, R, Z, C, sign, derive):
    """(mask, s, lev, m1, m2) for the interior flux re-map on contour C."""
    phi = sign * psi
    RR, ZZ = np.meshgrid(R, Z)
    ins = derive._points_in_polygon(RR.ravel(), ZZ.ravel(), C).reshape(RR.shape)
    lev = float(np.mean(derive._bilinear(phi, R, Z, C[:, 0], C[:, 1])))
    m = ins & (phi > lev)
    if m.sum() < 12:
        return None
    phia = float(phi[m].max())
    s = np.clip((phi - lev) / max(phia - lev, 1e-12), 0.0, 1.0)
    gz, gr = np.gradient(phi, Z, R)
    bp2 = (gz ** 2 + gr ** 2) / RR ** 2
    w = bp2[m]
    m1 = float((s[m] * w).sum() / w.sum()); m2 = float((s[m] ** 2 * w).sum() / w.sum())
    return m, s, lev, m1, m2


def li_remap(psi, R, Z, C, sign, ratio, derive, clo=-0.25, chi=0.6):
    """scale <Bp^2>_V inside contour C by `ratio` (boundary value and gradient unchanged)."""
    if C is None or not np.isfinite(ratio) or abs(ratio - 1.0) < 1e-4:
        return psi
    o = li_coef(psi, R, Z, C, sign, derive)
    if o is None:
        return psi
    m, s, lev, m1, m2 = o
    disc = m1 * m1 + m2 * (ratio - 1.0)
    c = (-m1 + np.sqrt(max(disc, 0.0))) / (2.0 * max(m2, 1e-12))
    c = float(np.clip(c, clo, chi))
    phi = sign * psi
    out = np.where(m, lev + (phi - lev) * (1.0 + c * s), phi)
    return sign * out
