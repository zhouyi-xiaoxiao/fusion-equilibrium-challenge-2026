#!/usr/bin/env python3
"""
Stage 11G / Ch2 TRACK restrict — additions to the INV_P6 kinetic inverse (inv_core.py / inv6_core.py are imported UNCHANGED).
external review, round 2 (internal note, fix #1 "restrict the discrepancy field"):

  closure fields      the radial-closure knob 'VBZ' of g1_core.CoilSet is the analytic uniform vertical field psi = B R^2/2
                      (that is ALREADY the review's "delta_psi = 1/2 dBz R^2"; no P4+P5 circuit exists in the code). Variants:
                        'npres'  decay-index-preserving response  psi = B [R^2/2 + q psi_Q],  q = n a / R0,
                                 n = -(R/Bz) dBz/dR of the frame's MEASURED coil field at (R0, 0)  (dBz' = -(n/R) dBz)
                        'coil'   coil-realisable combination of the symmetric PF pairs with Bz(R0, 0) = 1 T that minimises
                                 the unwanted field gradients (Bz - 1, B_R) over the plasma region (ridge on the pair norm)
  slow prior          information-weighted temporal pooling of the frame linearisations: the prior mean of the regularised
                      parameters evolves slowly through the shot (kernel exp(-|dt|/tau)); with svd_min = 1 no frame updates
                      along modes with singular value < 1 (they sit on the slow prior mean)
  invert_frame7       = inv_core.invert_frame + the linearisation record needed by the slow prior (same numerics otherwise)
  h3_experiment       Jacobian-SVD weak-direction experiment (review Q1, rank 1): prior-scaled Jacobian of the DATA residuals,
                      +-1 prior sd along every weak singular direction, refit of the well-observed directions, movement of
                      held-out isotherms vs X-points / kappa / li
Nothing here reads a file. Rule 3: the caller installs rule3_guard for MAST and passes inputs only.
"""
from __future__ import annotations

import numpy as np

import g1_core as C
import inv_core as K

MU0 = C.MU0


# ----------------------------------------------------------------------------------------------- closure fields
def install_closure(S, machine, mode, ridge=1e-2):
    """Replace the response of the radial-closure knob (circuit 'VBZ'). The knob amplitude stays B_Z(R0, 0) in tesla, so
    the prior (sigma_b Bp*), the bound (b_clip Bp*) and the response gain keep their meaning."""
    cs, g, wall = S.coils, S.grid, S.wall
    if getattr(cs, "_clos", None) is not None:
        assert cs._clos["mode"] == mode, "one closure mode per process"
        return cs._clos["info"]
    R0, a = C.R0_PUBLIC[machine], C.A_PUBLIC[machine]
    psiV, psiQ = cs._modes_psi(g.R, g.Z)                       # analytic uniform field / quadrupole about (R0, 0)
    st = {"mode": mode, "q": 0.0, "GV": np.array(psiV, float), "GQ": np.array(psiQ, float), "coef": None, "info": {"mode": mode}}
    base = (cs.psi_units, cs.br_units, cs.bz_units)
    st["base"] = base
    iv = cs.iv
    if mode == "coil":
        pairs = K.even_pairs(cs, machine)
        E = np.array([cs.G[cs.index[p]] + cs.G[cs.index[q]] for p, q in pairs])          # flux per unit pair current
        bz = np.gradient(E, g.dR, axis=1) / g.R[None]
        br = -np.gradient(E, g.dZ, axis=2) / g.R[None]
        reg = wall.inside & (((g.R - R0) / (1.1 * a)) ** 2 + (g.Z / (2.0 * a)) ** 2 < 1.0)
        A = np.concatenate([bz[:, reg], br[:, reg]], axis=1).T                             # (2 n_reg, n_pairs)
        b = np.concatenate([np.ones(int(reg.sum())), np.zeros(int(reg.sum()))])
        u0 = base[2](R0, 0.0)
        a0 = np.array([u0[cs.index[p]] + u0[cs.index[q]] for p, q in pairs])               # Bz(R0, 0) per unit pair current
        Ei = E[:, wall.inside]
        wn = np.sqrt(((Ei - Ei.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))           # pair norm = RMS flux inside the wall
        W = np.diag((wn / wn.mean()) ** 2)
        H = A.T @ A
        H = H + ridge * (np.trace(H) / len(pairs)) * W / np.trace(W) * len(pairs)
        # min c'Hc - 2 c'A'b  s.t.  a0'c = 1   (KKT)
        n = len(pairs)
        KKT = np.zeros((n + 1, n + 1)); KKT[:n, :n] = H; KKT[:n, n] = a0; KKT[n, :n] = a0
        sol = np.linalg.solve(KKT, np.concatenate([A.T @ b, [1.0]]))
        c = sol[:n]
        coef = np.zeros(cs.nc)
        for (p, q), cj in zip(pairs, c):
            coef[cs.index[p]] += cj; coef[cs.index[q]] += cj
        smap = np.tensordot(coef, cs.G, axes=1)
        res = A @ c - b
        d = (smap[reg] - smap[reg].mean()) - (psiV[reg] - psiV[reg].mean())
        st["coef"] = coef
        cs.G[iv] = smap
        st["info"].update({"pairs": [p for p, _ in pairs], "coef_A_per_T": [float(x) for x in c],
                           "rms_bz_minus_1_in_plasma": float(np.sqrt((res[:int(reg.sum())] ** 2).mean())),
                           "rms_br_in_plasma": float(np.sqrt((res[int(reg.sum()):] ** 2).mean())),
                           "r2_flux_vs_uniform_in_plasma": float(1.0 - (d ** 2).sum() / ((psiV[reg] - psiV[reg].mean()) ** 2).sum()),
                           "rms_flux_in_wall_vs_uniform": float(np.std(smap[wall.inside]) / np.std(psiV[wall.inside]))})
    elif mode != "npres":
        raise ValueError(f"unknown closure mode {mode}")

    def wrap(f, kind):
        def h(R, Z):
            u = f(R, Z)
            if st["mode"] == "npres":
                q = st["q"]
                if kind == "psi":
                    u[iv] = 0.5 * R ** 2 + q * cs.cq * (R ** 2 * Z ** 2 - 0.25 * R ** 4 + 0.5 * cs.R0 ** 2 * R ** 2)
                elif kind == "br":
                    u[iv] = q * (-cs.cq * 2.0 * R * Z)
                else:
                    u[iv] = 1.0 + q * cs.cq * (2.0 * Z ** 2 - R ** 2 + cs.R0 ** 2)
            else:
                u[iv] = float(np.dot(st["coef"], u))          # coef is zero on the virtual circuits
            return u
        return h
    cs.psi_units, cs.br_units, cs.bz_units = wrap(cs.psi_units, "psi"), wrap(cs.br_units, "br"), wrap(cs.bz_units, "bz")
    cs._clos = st
    return st["info"]


def decay_index(S, I_base, h=0.02, n_clip=2.0, bz_min=1e-4):
    """n = -(R / Bz) dBz/dR of the measured coil field at (R0, 0) (virtual circuits excluded: their entries of I_base are 0)."""
    cs = S.coils
    bzf = cs._clos["base"][2] if getattr(cs, "_clos", None) is not None else cs.bz_units
    R0 = cs.R0
    Iv = np.array(I_base, float).copy()
    Iv[cs.iv] = 0.0; Iv[cs.iq] = 0.0
    b0 = float(np.dot(Iv, bzf(R0, 0.0)))
    bp = float(np.dot(Iv, bzf(R0 + h, 0.0))); bm = float(np.dot(Iv, bzf(R0 - h, 0.0)))
    if abs(b0) < bz_min:
        return 0.0, b0
    return float(np.clip(-R0 * (bp - bm) / (2 * h) / b0, -n_clip, n_clip)), b0


def set_frame_closure(S, machine, I_base):
    """per-frame set-up of the 'npres' closure response (no-op for the other modes). Returns the decay index used."""
    cs = S.coils
    st = getattr(cs, "_clos", None)
    if st is None or st["mode"] != "npres":
        return np.nan
    n, _ = decay_index(S, I_base)
    st["q"] = n * C.A_PUBLIC[machine] / C.R0_PUBLIC[machine]
    cs.G[cs.iv] = st["GV"] + st["q"] * st["GQ"]
    return n


# ----------------------------------------------------------------------------------------------- restricted gamma_shape
XREG = {"mast": (0.45, 0.80, 0.95, 1.40), "d3d": (1.15, 1.60, 0.95, 1.40)}   # (Rlo, Rhi, |Z|lo, |Z|hi) null (X-point) zones


def install_shape_xnull(S, machine, ridge=1e-2):
    """RESTRICTED gamma_shape (replaces inv_core.install_shape_mode): the even, coil-realisable shaping field with the same
    two midplane conditions as INV_P5/P6 (B_Z(R0, 0) = 0;  B_Z(R0 - a, 0) - B_Z(R0 + a, 0) = 2 T, so the prior 0.03 Bp* keeps
    its meaning) but, instead of the minimum-norm combination (dominated by the divertor coils P2/P3 on MAST, i.e. a virtual
    current next to the X-points), the combination that MINIMISES ITS OWN POLOIDAL FIELD IN THE NULL ZONES above and below
    the plasma (|B|^2 integrated over XREG; a small ridge on the pair norm keeps the currents finite). The discrepancy
    field then acts where the midplane laser constrains it (width / field gradient across the midplane) and leaves the
    X-points to the measured coils and the plasma current distribution (review round 2: 'restrict the discrepancy field').
    XREG: MAST = central range of the X-points of our OWN label-free INV_P6 solutions (R p5-p95 0.51-0.72 m, |Z| 1.01-1.33 m)
    padded to a box; DIII-D = the divertor null zone of the public machine geometry (labels are allowed there anyway)."""
    cs, g, wall = S.coils, S.grid, S.wall
    if getattr(cs, "_shape_installed", False):
        return cs._shape_info
    R0, a = C.R0_PUBLIC[machine], C.A_PUBLIC[machine]
    pairs = K.even_pairs(cs, machine)
    E = np.array([cs.G[cs.index[p]] + cs.G[cs.index[q]] for p, q in pairs])
    bzu = cs.bz_units
    u0, um, up = bzu(R0, 0.0), bzu(R0 - a, 0.0), bzu(R0 + a, 0.0)
    A = np.array([[u0[cs.index[p]] + u0[cs.index[q]] for p, q in pairs],
                  [(um[cs.index[p]] + um[cs.index[q]]) - (up[cs.index[p]] + up[cs.index[q]]) for p, q in pairs]])
    bvec = np.array([0.0, 2.0])
    Ei = E[:, wall.inside]
    wn = np.sqrt(((Ei - Ei.mean(axis=1, keepdims=True)) ** 2).mean(axis=1))
    Winv2 = 1.0 / np.maximum(wn, 1e-30) ** 2
    c_mn = Winv2 * (A.T @ np.linalg.solve((A * Winv2) @ A.T, bvec))                   # INV_P6 minimum-norm mode (reference)
    bz = np.gradient(E, g.dR, axis=1) / g.R[None]
    br = -np.gradient(E, g.dZ, axis=2) / g.R[None]
    rl, rh, zl, zh = XREG[machine]
    reg = (g.R >= rl) & (g.R <= rh) & (np.abs(g.Z) >= zl) & (np.abs(g.Z) <= zh)
    B = np.concatenate([bz[:, reg], br[:, reg]], axis=1).T                             # (2 n_reg, n_pairs)
    H = B.T @ B / reg.sum()
    W = np.diag((wn / wn.mean()) ** 2)
    H = H + ridge * (np.trace(H) / np.trace(W)) * W
    n = len(pairs)
    KKT = np.zeros((n + 2, n + 2)); KKT[:n, :n] = H; KKT[:n, n:] = A.T; KKT[n:, :n] = A
    c = np.linalg.solve(KKT, np.concatenate([np.zeros(n), bvec]))[:n]
    coef = np.zeros(cs.nc)
    for (p, q), cj in zip(pairs, c):
        coef[cs.index[p]] += cj; coef[cs.index[q]] += cj
    smap = np.tensordot(coef, cs.G, axes=1)
    coef_mn = np.zeros(cs.nc)
    for (p, q), cj in zip(pairs, c_mn):
        coef_mn[cs.index[p]] += cj; coef_mn[cs.index[q]] += cj
    smap_mn = np.tensordot(coef_mn, cs.G, axes=1)
    rms = lambda cc: float(np.sqrt(((B @ cc) ** 2).sum() / reg.sum()))               # rms |B| in the null zones per 2 T
    mid = (np.abs(g.Z) < 0.5 * g.dZ + 1e-9) & (np.abs(g.R - R0) <= a)
    bz_mid = lambda cc: np.tensordot(cc, bz, axes=1)[mid]
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
    cs._shape_info = {"mode": "coil_xnull", "pairs": [p for p, _ in pairs], "coef_A_per_T": [float(x) for x in c],
                      "coef_minnorm_A_per_T": [float(x) for x in c_mn], "xreg": list(XREG[machine]),
                      "rms_B_nullzone": rms(c), "rms_B_nullzone_minnorm": rms(c_mn),
                      "norm_ratio_vs_minnorm": float(np.sqrt(((wn * c) ** 2).sum() / ((wn * c_mn) ** 2).sum())),
                      "bz_midplane_profile": [float(x) for x in bz_mid(c)], "bz_midplane_profile_minnorm": [float(x) for x in bz_mid(c_mn)],
                      "rms_flux_in_wall_ratio_vs_minnorm": float(np.std(smap[wall.inside]) / max(np.std(smap_mn[wall.inside]), 1e-30))}
    return cs._shape_info


# ----------------------------------------------------------------------------------------------- slow prior mean
class SlowPrior:
    """Information-weighted temporal pooling of the frame linearisations (whitened parameters, prior N(0, 1) on the regularised
    ones). Frame j contributes D_j = J'J and h_j = J'(J x_j - r_j) (Huber-weighted DATA rows, linearised at x_j). The prior
    mean for frame k is  m_k = (L0 + sum_{j != k} w_jk D_j)^-1 sum_{j != k} w_jk h_j,  w_jk = exp(-|t_k - t_j| / tau):
    directions that single frames cannot resolve (singular value < 1) follow the pooled evidence of the neighbouring frames
    and otherwise relax to the static prior mean 0."""

    def __init__(self, tau, wmax=1.0):
        self.tau, self.wmax, self.rec = float(tau), float(wmax), {}

    def set(self, k, t, lin):
        if lin is None:
            self.rec.pop(k, None)
            return
        params, J, r, xb = lin
        if J.size == 0:
            self.rec.pop(k, None)
            return
        D = J.T @ J
        self.rec[k] = (float(t), list(params), D, J.T @ (J @ xb - r))

    def prior(self, k, t, params):
        """(mean dict, var dict) for inv_core's x_prior interface (unit variance, shifted mean) or None without neighbours."""
        n = len(params)
        D = np.zeros((n, n)); h = np.zeros(n); used = 0
        for j, (tj, pj, Dj, hj) in self.rec.items():
            if j == k:
                continue
            w = self.wmax * float(np.exp(-abs(t - tj) / self.tau))
            if w < 1e-3:
                continue
            ix = [params.index(p) if p in params else -1 for p in pj]
            for a_, ia in enumerate(ix):
                if ia < 0:
                    continue
                h[ia] += w * hj[a_]
                for b_, ib in enumerate(ix):
                    if ib >= 0:
                        D[ia, ib] += w * Dj[a_, b_]
            used += 1
        if not used:
            return None
        L0 = np.array([1.0 if p in K.REG else 1e-6 for p in params])
        m = np.linalg.solve(D + np.diag(L0), h)
        return ({p: float(m[i]) for i, p in enumerate(params) if p in K.REG}, {p: 1.0 for p in params if p in K.REG})


def invert_frame7(fm, x0, zt0, warm, iso, edges, z_eta, a_pub, zt_max, params=None, resid_fn=None, x_prior=None):
    """inv_core.invert_frame (same statements, same order) + info['lin'] = (params, Jw_data, rw_data, x_base) of the LAST
    Jacobian evaluation (Huber-weighted data rows only; the gamma_radial prior row is excluded)."""
    cfg = fm.cfg
    params = list(params if params is not None else cfg["params"])
    if fm.edge_out is None:
        params = [p for p in params if p != "dr"]
    x = K.clip_x({n: float(x0.get(n, 0.0)) for n in params}, cfg)
    info = {"lm_it": 0, "n_rej": 0, "sv": [], "n_frozen": 0, "cost0": np.nan, "cost": np.nan, "err": "", "n_refit": 0, "lin": None}
    zt = zt0
    sol = fm.robust_solve(x, zt, warm)
    if not sol["ok"]:
        info["err"] = "base: " + sol["err"]
        return None, info

    def rebase(s, xx, zz):
        z1 = K.ridge_z(s, zz, z_eta, a_pub, zt_max)
        pet = K.refit_pet(s, fm.ts_pts, cfg) if cfg["remap"] else None
        if abs(z1 - zz) > 0.002 or pet is not None:
            s2 = fm.solve(xx, z1, warm=K.warm_of(s), pet=pet or s["prof"].pet)
            if s2["ok"]:
                info["n_refit"] += int(pet is not None)
                return s2, z1
        return s, zz

    sol, zt = rebase(sol, x, zt)
    pmean = np.array([float((x_prior or ({}, {}))[0].get(p, 0.0)) if p in K.REG else 0.0 for p in params])
    pvar = np.array([float((x_prior or ({}, {}))[1].get(p, 1.0)) if p in K.REG else np.inf for p in params])
    pm = np.where(np.isfinite(pvar), 1.0 / pvar, 0.0)
    has_gev = bool(np.isfinite(cfg["sig_gev"]) and fm.edge_out is not None)

    def rfun(s):
        if resid_fn is not None:
            r_, h_ = resid_fn(s)
        else:
            r_, h_, _ = K.residuals(s, iso, edges, cfg)
        if has_gev:
            r_ = np.append(r_, (s["gev"] - cfg["gev0"]) / cfg["sig_gev"]); h_ = np.append(h_, 1e9)
        return r_, h_

    def total(s, xx):
        r_, h_ = rfun(s)
        xv_ = np.array([xx[p] for p in params])
        return K.huber_cost(r_, h_) + float(np.sum(pm * (xv_ - pmean) ** 2)), r_, h_

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
        wb, pet = K.warm_of(sol), sol["prof"].pet
        for j, pn in enumerate(params):
            hx = cfg["fd_step"][pn] / K._sig(pn, cfg)
            xj = dict(x); xj[pn] = x[pn] + hx
            xj = K.clip_x(xj, cfg)
            if abs(xj[pn] - x[pn]) < 0.5 * hx:
                xj[pn] = x[pn] - hx
            sj = fm.solve(xj, zt, warm=wb, pet=pet)
            if not sj["ok"]:
                xj = dict(x); xj[pn] = x[pn] - hx
                xj = K.clip_x(xj, cfg)
                sj = fm.solve(xj, zt, warm=wb, pet=pet) if abs(xj[pn] - x[pn]) > 0.5 * hx else sj
            if sj["ok"]:
                rj, _ = rfun(sj)
                Jm[:, j] = (rj - r) / (xj[pn] - x[pn])
        sw = np.sqrt(w)
        Jw, rw = Jm * sw[:, None], r * sw
        xv = np.array([x[p] for p in params])
        nd_ = len(r) - (1 if has_gev else 0)
        info["lin"] = (list(params), Jw[:nd_].copy(), rw[:nd_].copy(), xv.copy())
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
                d += -Vw @ (Vw.T @ (np.minimum(pm, 1.0) * (xv - pmean)))   # weak directions -> (slow) prior mean
            nd = float(np.linalg.norm(d))
            if nd > tr:
                d *= tr / nd
            xn = K.clip_x({p: x[p] + float(d[k]) for k, p in enumerate(params)}, cfg)
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


# ----------------------------------------------------------------------------------------------- H3 experiment
def geom_record(sol, S, fvac, Ip, chord):
    """Solver-side geometry of a converged solution (label-free): LCFS shape scalars, both X-points, axis."""
    b = sol["b"]
    sc = K.scalars_kin(S.grid, S.wall, sol["psi"], b, sol["prof"], fvac, Ip, chord=chord)
    Ra, Za = b["axis"]
    dz = 0.5 * C.A_PUBLIC[S.machine]
    sad = [x for x in b["saddles"] if x[2] < b["psi_axis"]]
    lo = [x for x in sad if x[1] < Za - dz]; up = [x for x in sad if x[1] > Za + dz]
    xl = max(lo, key=lambda x: x[2]) if lo else None
    xu = max(up, key=lambda x: x[2]) if up else None
    out = {k: float(sc.get(k, np.nan)) for k in ("kappa", "li", "Ztop", "Zbot", "Rin", "Rout", "volume", "q95", "tri_top", "tri_bot",
                                                 "a", "Rgeo", "betaN")}
    span = b["psi_axis"] - b["psi_b"]
    out.update({"Rax": float(Ra), "Zax": float(Za), "kind": b["kind"], "span": float(span),
                "Rxl": float(xl[0]) if xl else np.nan, "Zxl": float(xl[1]) if xl else np.nan,
                "Rxu": float(xu[0]) if xu else np.nan, "Zxu": float(xu[1]) if xu else np.nan,
                "dsep_psi": float((xu[2] - xl[2]) / span) if (xl and xu) else np.nan})
    return out


def h3_experiment(fm, sol, zt, iso, iso_v, edges, S, fvac, Ip, chord, sig_zt=0.015, fd_zt=0.005, sv_weak=1.0, max_it=4):
    """review round 2, Q1 rank 1. Parameters y = (theta whitened by the prior sd ..., Z_t / sig_zt). J~ = d r_data / d y (Huber-
    weighted isotherm + edge rows only) = U S V'. For every weak direction (S < sv_weak): y +- v, then the well-observed
    directions are refitted (Gauss-Newton in the strong subspace on the FULL objective: data + gamma_radial prior + parameter
    priors, base Jacobian, true residuals) and the movement of the held-out isotherms / fitted isotherms / edges and of the
    X-points, kappa, li ... is recorded. Returns a dict (JSON-able)."""
    cfg = fm.cfg
    params = [p for p in cfg["params"] if not (p == "dr" and fm.edge_out is None)]
    npar = len(params)
    names = params + ["zt"]
    x = {p: float(sol["x"].get(p, 0.0)) for p in params}
    y0 = np.array([x[p] for p in params] + [zt / sig_zt])
    wb, pet = K.warm_of(sol), sol["prof"].pet
    has_gev = bool(np.isfinite(cfg["sig_gev"]) and fm.edge_out is not None)
    pm = np.array([1.0 if p in K.REG else 0.0 for p in params] + [0.0])
    out = {"names": names, "n_solves": 0, "err": ""}

    def solve_y(y):
        xx = K.clip_x({p: float(y[i]) for i, p in enumerate(params)}, cfg)
        s = fm.solve(xx, float(y[npar] * sig_zt), warm=wb, pet=pet)
        out["n_solves"] += 1
        return s

    def rdata(s):
        r_, h_, _ = K.residuals(s, iso, edges, cfg)
        return r_, h_

    def rall(s, y):
        r_, h_ = rdata(s)
        w_ = np.sqrt(np.minimum(1.0, h_ / np.maximum(np.abs(r_), 1e-12)))
        parts = [r_ * w_]
        if has_gev:
            parts.append([(s["gev"] - cfg["gev0"]) / cfg["sig_gev"]])
        parts.append(np.sqrt(pm) * y)
        cost = K.huber_cost(r_, h_) + (parts[1][0] ** 2 if has_gev else 0.0) + float(np.sum(pm * y ** 2))
        return np.concatenate([np.atleast_1d(p_) for p_ in parts]), cost

    def meas(s):
        b = s["b"]
        rec = geom_record(s, S, fvac, Ip, chord)
        rec["mv"] = [v for _, v in K.iso_mismatch(b["spl"], iso_v)] if iso_v else []
        rec["mf"] = [v for _, v in K.iso_mismatch(b["spl"], iso)] if iso else []
        rec["ed"] = K.edge_deltas(s, edges)
        rec["gev"] = float(s["gev"])
        return rec

    r0, h0 = rdata(sol)
    if len(r0) == 0:
        out["err"] = "no data rows"
        return out
    R0v, cost0 = rall(sol, y0)
    w0 = np.sqrt(np.minimum(1.0, h0 / np.maximum(np.abs(r0), 1e-12)))
    nd = len(r0)
    Jf = np.zeros((len(R0v), npar + 1))
    steps = [cfg["fd_step"][p] / K._sig(p, cfg) for p in params] + [fd_zt / sig_zt]
    for j in range(npar + 1):
        for sgn in (1.0, -1.0):
            yj = y0.copy(); yj[j] += sgn * steps[j]
            sj = solve_y(yj)
            if sj["ok"]:
                if j < npar:
                    yj[j] = sj["x"][params[j]]               # after clipping
                if abs(yj[j] - y0[j]) < 0.25 * steps[j]:
                    continue
                Rj, _ = rall(sj, yj)
                # data rows with the BASE Huber weights (fixed weighting inside one linearisation)
                rj, _ = rdata(sj)
                Rj[:nd] = rj * w0
                Jf[:, j] = (Rj - R0v) / (yj[j] - y0[j])
                break
    Jd = Jf[:nd]
    U, sv, Vt = np.linalg.svd(Jd, full_matrices=True)
    svf = np.zeros(npar + 1); svf[:len(sv)] = sv
    out.update({"sv": [float(v) for v in svf], "V": [[float(c) for c in row] for row in Vt], "n_data": int(nd),
                "cost0": float(cost0), "base": meas(sol)})
    weak = svf < sv_weak
    Vs = Vt[~weak].T
    out["n_weak"] = int(weak.sum())
    base = out["base"]
    res = []
    for iw in np.nonzero(weak)[0]:
        v = Vt[iw]
        ent = {"i": int(iw), "sv": float(svf[iw]), "v": [float(c) for c in v], "pm": []}
        for sgn in (1.0, -1.0):
            y = y0 + sgn * v
            s = solve_y(y)
            if not s["ok"]:
                s = None
            else:
                Rv, cst = rall(s, y)
                mu = 1e-2
                for _ in range(max_it):
                    if Vs.shape[1] == 0:
                        break
                    Js = Jf @ Vs
                    step_ok = False
                    for _try in range(3):
                        A = Js.T @ Js
                        dc = -np.linalg.solve(A + mu * np.diag(np.diag(A)) + 1e-9 * np.eye(A.shape[0]), Js.T @ Rv)
                        if np.linalg.norm(dc) > 1.5:
                            dc *= 1.5 / np.linalg.norm(dc)
                        yn = y + Vs @ dc
                        sn = solve_y(yn)
                        if sn["ok"]:
                            Rn, cn = rall(sn, yn)
                            if cn < cst - 1e-9:
                                step_ok = True
                                break
                        mu *= 5.0
                    if not step_ok:
                        break
                    gain = cst - cn
                    y, s, Rv, cst = yn, sn, Rn, cn
                    mu = max(mu / 3.0, 1e-4)
                    if np.linalg.norm(dc) < 0.02 or gain < 1e-3 * max(cst, 1.0):
                        break
            if s is None:
                ent["pm"].append(None)
                continue
            m = meas(s)
            d = {"cost": float(cst), "dcost": float(cst - cost0), "moved_strong": float(np.linalg.norm(Vs.T @ (y - y0 - sgn * v))) if Vs.shape[1] else 0.0}
            for kk in ("kappa", "li", "Ztop", "Zbot", "Rin", "Rout", "volume", "q95", "tri_top", "tri_bot", "Rax", "Zax", "Rxl", "Zxl", "Rxu",
                       "Zxu", "dsep_psi", "betaN", "gev"):
                d["d_" + kk] = float(m[kk] - base[kk]) if (np.isfinite(m[kk]) and np.isfinite(base[kk])) else np.nan
            d["kind"] = m["kind"]
            d["d_iso_val"] = [float(a_ - b_) for a_, b_ in zip(m["mv"], base["mv"])] if len(m["mv"]) == len(base["mv"]) else []
            d["d_iso_fit"] = [float(a_ - b_) for a_, b_ in zip(m["mf"], base["mf"])] if len(m["mf"]) == len(base["mf"]) else []
            d["d_edge"] = {n_: float(m["ed"][n_] - base["ed"][n_]) for n_ in m["ed"] if n_ in base["ed"]}
            ent["pm"].append(d)
        res.append(ent)
    out["weak"] = res
    return out
