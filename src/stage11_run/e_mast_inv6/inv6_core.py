#!/usr/bin/env python3
"""
Stage 11E / Ch2 INV6 — additions to the INV_P5 kinetic inverse (ch2d_inverse/inv_core.py, imported unchanged):

  (a) DENSITY PAIRS as supplementary flux-function constraints: inboard/outboard crossings of n_e/n_e,max levels on the same
      chord(s) as the T_e isotherms, same unambiguity / slope rules (exactly one robust crossing per side, sigma_R <= 3 cm),
      their own model-discrepancy floor (ne_floor x Delta psi, larger than the T_e floor: n_e is a flux function only to the
      extent that poloidal density asymmetries and the edge-recycling region can be neglected).
  (b) CORRELATED ERRORS between crossings that share Thomson channels: crossing-position errors of two points on the same
      chord side (same chord direction) are correlated with rho(d) = ne_rho * exp(-d / ne_ell) whenever at least one of them
      is a density crossing (same scattering volumes, same pulse, same smoothing kernel). T_e-T_e pairs keep the INV_P5
      independent model, so INV6 with ne_levels = [] reproduces INV_P5 exactly. The pair residual vector is whitened with the
      Cholesky factor of its correlation matrix (re-evaluated with the current flux gradients).
  (c) helpers for the multi-start / bidirectional-continuation branch selection (LCFS elongation of a candidate).
Nothing here reads a file.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky, solve_triangular

import g1_core as C
import inv_core as K


# ----------------------------------------------------------------------------------------------- density observations
def mast_ne_obs(prof_k, levels, zl, max_sig=0.03):
    """MAST midplane laser: inboard/outboard crossing pairs of n_e / n_e,max (horizontal chord at Z = zl)."""
    if prof_k is None or not levels:
        return []
    xr, _, _, ne = prof_k
    _, cr = K.chord_crossings(xr, ne, levels, max_sig=max_sig)
    obs = []
    for f, rec in cr.items():
        if "lo" in rec and "hi" in rec:
            obs.append({"f": f, "q": "ne", "pts": [((rec["lo"][0], zl), (-1.0, 0.0), rec["lo"][1]), ((rec["hi"][0], zl), (1.0, 0.0), rec["hi"][1])],
                        "pairs": [(0, 1)]})
    return obs


def d3d_ne_obs(vert, tang, levels):
    """DIII-D real chords, density analogue of inv_core.d3d_iso_obs (n_e arrays instead of T_e)."""
    sw = lambda p: None if p is None else (p[0], p[1], p[3], p[3])       # put n_e in the T_e slot
    return [{**o, "q": "ne"} for o in K.d3d_iso_obs(sw(vert), sw(tang), levels)] if levels else []


# ----------------------------------------------------------------------------------------------- residuals
def _vals(spl, lev):
    out = []
    for pt in lev["pts"]:
        P, e, sg = pt[0], pt[1], pt[2]
        p = float(spl.ev(*P)); pr, pz = float(spl.ev(*P, dx=1)), float(spl.ev(*P, dy=1))
        slope = max(abs(e[0] * pr + e[1] * pz), 0.3 * np.hypot(pr, pz), 1e-12)
        out.append((p, slope * sg, P, e))
    return out


def residuals6(sol, iso, iso_n, edges, cfg):
    """Whitened residuals: T_e pairs (INV_P5 model), n_e pairs (own floor), correlated through shared channels, + edges."""
    b = sol["b"]
    spl = b["spl"]
    D = b["psi_axis"] - b["psi_b"]
    r, h, meta = [], [], []
    for q, levs, floor in (("te", iso, cfg["iso_floor"]), ("ne", iso_n, cfg.get("ne_floor", 0.03))):
        for lev in levs:
            v = _vals(spl, lev)
            for i, j in lev["pairs"]:
                s = np.sqrt(v[i][1] ** 2 + v[j][1] ** 2 + (floor * D) ** 2)
                r.append((v[i][0] - v[j][0]) / s); h.append(cfg["iso_huber"])
                meta.append((s, q, ((v[i][2], v[i][3], v[i][1], 1.0), (v[j][2], v[j][3], v[j][1], -1.0))))
    n = len(r)
    r = np.array(r, float)
    rho0, ell = float(cfg.get("ne_rho", 0.0)), float(cfg.get("ne_ell", 0.02))
    if n > 1 and rho0 > 0 and any(m[1] == "ne" for m in meta):
        Rm = np.eye(n)
        for p in range(n):
            for qq in range(p + 1, n):
                if meta[p][1] == "te" and meta[qq][1] == "te":
                    continue
                c = 0.0
                for (Pa, ea, ga, sa) in meta[p][2]:
                    for (Pb, eb, gb, sb) in meta[qq][2]:
                        if ea[0] * eb[0] + ea[1] * eb[1] < 0.9:          # different chord side / direction
                            continue
                        d = float(np.hypot(Pa[0] - Pb[0], Pa[1] - Pb[1]))
                        c += sa * sb * ga * gb * rho0 * np.exp(-d / ell)
                Rm[p, qq] = Rm[qq, p] = c / (meta[p][0] * meta[qq][0])
        try:
            L = cholesky(Rm + 1e-9 * np.eye(n), lower=True)
            r = solve_triangular(L, r, lower=True)
        except np.linalg.LinAlgError:
            pass
    r, h = list(r), list(h)
    for o in edges:
        P, e = o["P"], o["e"]
        pr, pz = float(spl.ev(*P, dx=1)), float(spl.ev(*P, dy=1))
        slope = max(-(e[0] * pr + e[1] * pz), 0.3 * np.hypot(pr, pz), 1e-12)
        d = (float(spl.ev(*P)) - b["psi_b"]) / slope
        se = np.hypot(o["sig"], cfg["sigma_u"] * D / slope)
        if o["kind"] == "ge":
            d = min(d, 0.0)
        r.append(d / se); h.append(cfg["edge_huber"])
    return np.array(r, float), np.array(h, float)


# ----------------------------------------------------------------------------------------------- candidate geometry
def lcfs_shape(sol, wall):
    """(kappa, R_in, R_out, Z_top, Z_bot) of the 99.5 % surface of a converged solution (NaN if no closed contour)."""
    b = sol["b"]
    try:
        Cn = C.closed_contour(wall, b["fine"], b["psi_axis"] + 0.995 * (b["psi_b"] - b["psi_axis"]), *b["axis"])
    except Exception:
        Cn = None
    if Cn is None:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    R_, Z_ = Cn[:-1, 0], Cn[:-1, 1]
    a = 0.5 * (R_.max() - R_.min())
    return float((Z_.max() - Z_.min()) / (2 * a)), float(R_.min()), float(R_.max()), float(Z_.max()), float(Z_.min())
