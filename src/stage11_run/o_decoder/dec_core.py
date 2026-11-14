#!/usr/bin/env python3
"""
Stage 11o / Ch2 decoder — JOINT FUNCTIONAL DECODER core (external review, round 3 §E.ii / §E2).

One frame:  psi = psi_vac(I_meas) + psi_p.  The decoder keeps psi_vac exactly and moves the plasma part of a START map along a
small set of smooth response coordinates until the published scorer functionals f_j of the map (starter/fusion_scoring, unmodified,
extraction with the PUBLIC first-wall mask only) match per-functional targets mu_j, at minimal pooled flux change:

    L(c) = lam * 0.55 |psi(c) - mu_psi|^2 / (Npix v_psi) + sum_j W_j ((mu_j - f_j(psi(c))) / s_j)^2 + kp |c|^2        (c in prior-sd units)
    dc   = (kp I + lam A + J' W J)^-1 (J' W r - lam g)            = C J'(J C J' + R)^-1 r  for A = 0, C = I / kp, R = 1 / W
    2-3 Gauss-Newton iterations, central finite-difference Jacobian through the organisers' own derivation code, per-coordinate trust
    region (0.25 prior sd per iteration), back-tracking on the true L, rejection of new axes / invalid contours / current oscillation.

Response coordinates (16; fields in the stored [Z, R] orientation, flux units of the file):
  displacement generators of the plasma part, B = -T xi.grad(psi_p), with x = (R - Rgeo) / a, z = (Z - Zc) / b of the START contour:
    sR (1,0)  sZ (0,1)  dR (x,0)  dZ (0,z)  shr (z,0)  tri (z^2,0)  z2 (0,z^2)  sq (x z^2,0)  [T = smooth box taper]
    cR, cZ   = the same rigid shifts under a core-localised Gaussian bump (axis relative to the boundary)
  li   interior flux re-map (phi - lev) s inside the last closed contour (value and gradient continuous at the boundary, total current kept)
  D    current-neutral net-minus-inverse difference of the plasma parts
  m1..m4 curl-free (vacuum) multipoles R^2, R^2 Z, R^2 (R^2 - 4 Z^2), R^2 Z (3 R^2 - 4 Z^2), mean inside the contour removed
Every field is projected to carry no net toroidal current (Ampere circulation around the grid border), so the measured Ip closure of the
start map is preserved by construction.  No label, no envelope: MAST uses the public wall polygon only.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
OUT_SIGN = {"d3d": -1.0, "mast": 1.0}
AXS = {"d3d": -1.0, "mast": 1.0}
NAME = {"d3d": "DIII-D", "mast": "MAST"}
FN = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
CPN = ["Rt", "Zt", "Rb", "Zb", "Ri", "Ro"]
TG = FN + CPN
COORDS = ["sR", "sZ", "dR", "dZ", "shr", "tri", "z2", "sq", "cR", "cZ", "li", "D", "m1", "m2", "m3", "m4"]
# a-priori prior sds (prior-sd units are what the trust region and kp refer to); the displacement coordinates take theirs from the DIII-D
# TRAIN pool (robust sd of the start map's control-point errors in units of the minor radius, dw.py fit -> "prior")
APRIORI = {"z2": None, "sq": None, "D": 1.0, "m1": 0.02, "m2": 0.02, "m3": 0.02, "m4": 0.02}


_trap = getattr(np, "trapezoid", None) or np.trapz


class Geo:
    def __init__(self, machine, wall):
        if machine == "mast":
            sys.path.insert(0, str(WS / "stage11_run/ch2_geometry"))
            import rule3_guard
            rule3_guard.install()
        sys.path.insert(0, str(WS / "starter/fusion_scoring"))
        import derive, lcfs, o_point
        self.derive, self.lcfs, self.op = derive, lcfs, o_point
        w = np.load(wall)
        if machine == "d3d":                          # grid only (as k_corrector/ktab.py); the envelope mask itself is NOT used
            m = np.load(WS / "starter/fusion_scoring/masks/d3d_envelope.npz")
            self.R, self.Z = m["grid_R"].astype(float), m["grid_Z"].astype(float)
        else:
            self.R, self.Z = w["grid_R"].astype(float), w["grid_Z"].astype(float)
        self.machine, self.name, self.sign = machine, NAME[machine], AXS[machine]
        P = np.column_stack([w["limiter_R"], w["limiter_Z"]]).astype(float)
        self.RR, self.ZZ = np.meshgrid(self.R, self.Z)
        self.WM = derive._points_in_polygon(self.RR.ravel(), self.ZZ.ravel(), P).reshape(self.RR.shape)
        self.WMf = self.WM.astype(float)
        self.dR, self.dZ = float(np.mean(np.diff(self.R))), float(np.mean(np.diff(self.Z)))

    # ---- published functionals with the public wall mask (identical call sequence to k_corrector/ktab.geom)
    def F(self, p):
        d, LC = self.derive, self.lcfs
        try:
            Ra, Za, iz, ir = d.magnetic_axis(p, self.R, self.Z, self.name, self.WM, self.sign)
        except Exception:  # noqa: BLE001
            return None
        C, _ = LC.extract_lcfs_with_sign(p, self.R, self.Z, self.name, self.WM, self.WMf, n_iter=22, axis_sign=self.sign)
        if C is None or len(C) < 8:
            return None
        r, z = C[:, 0], C[:, 1]
        it, ib, ii, io = int(np.argmax(z)), int(np.argmin(z)), int(np.argmin(r)), int(np.argmax(r))
        s = dict(d._shape_from_contour(C))
        try:
            s["li"] = d.internal_inductance(p, self.R, self.Z, self.name, C)
        except Exception:  # noqa: BLE001
            s["li"] = np.nan
        s = d._clamp_to_nan(s)
        phi = self.sign * p
        v = np.array([Ra, Za, s.get("kappa", np.nan), s.get("tri_top", np.nan), s.get("tri_bot", np.nan), s.get("volume", np.nan), s.get("li", np.nan),
                      r[it], z[it], r[ib], z[ib], r[ii], r[io]], float)
        return {"v": v, "C": C, "iz": iz, "ir": ir, "lev": float(np.mean(d._bilinear(phi, self.R, self.Z, r, z))), "phia": float(phi[iz, ir])}

    def gamma(self, f):
        """Ampere circulation of the field f around the grid border (∮ (dψ/dn) / R dl; arbitrary constant factor)."""
        gz, gr = np.gradient(f, self.Z, self.R, edge_order=2)
        R, Z = self.R, self.Z
        tz = lambda y: float(_trap(y, Z))  # noqa: E731
        tr = lambda y: float(_trap(y / R, R))  # noqa: E731
        return tz(gr[:, -1]) / R[-1] - tz(gr[:, 0]) / R[0] + tr(gz[-1, :]) - tr(gz[0, :])

    def jphi(self, p):
        """toroidal current density proxy  -Delta* psi / R  (arbitrary units) on the grid."""
        gz, gr = np.gradient(p, self.Z, self.R)
        gzz = np.gradient(gz, self.Z, axis=0)
        grr = np.gradient(gr, self.R, axis=1)
        return -(grr - gr / self.RR + gzz) / self.RR

    def inside(self, C):
        return self.derive._points_in_polygon(self.RR.ravel(), self.ZZ.ravel(), C).reshape(self.RR.shape)

    def n_axes(self, p, C, lev):
        """number of interior local maxima of phi inside the contour C (an added one = a new axis)."""
        phi = self.sign * p
        lm = self.op.local_maxima(phi, self.WM, 2)
        return int((lm & self.inside(C) & (phi > lev)).sum())


def _taper(u, u0=1.25, u1=2.0):
    q = np.clip((np.abs(u) - u0) / (u1 - u0), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * q))


def osc_metrics(G, p, ins):
    """(negative-current fraction, roughness) of the current proxy inside the mask `ins` (eroded by one cell)."""
    m = G.op._erode(ins, 1)
    if m.sum() < 12:
        m = ins
    if m.sum() < 6:
        return 0.0, 0.0
    j = G.jphi(p)
    jm = j[m]
    sg = np.sign(jm.sum()) or 1.0
    neg = float((jm * sg < 0).mean())
    dz = np.abs(np.diff(j, axis=0))[m[1:, :] & m[:-1, :]]
    dr = np.abs(np.diff(j, axis=1))[m[:, 1:] & m[:, :-1]]
    tv = float((dz.sum() + dr.sum()) / max(np.abs(jm).sum(), 1e-30))
    return neg, tv


class FrameDecoder:
    """decoder of one frame. All maps are in the stored orientation [Z, R] and the file's flux convention."""

    def __init__(self, G: Geo, cfg: dict, prior: dict):
        self.G, self.cfg = G, cfg
        self.coords = [c for c in COORDS if c in cfg.get("coords", COORDS)]
        self.prior = prior                                  # dimensionless sds (units of a; li: log; D: 1; multipoles: fraction of the flux span)

    def sigmas(self, a):
        p = self.prior
        s = {"sR": p["sR"] * a, "sZ": p["sZ"] * a, "dR": p["dR"] * a, "dZ": p["dZ"] * a, "shr": p["shr"] * a, "tri": p["tri"] * a,
             "z2": p["cZ"] * a, "sq": 0.5 * p["dR"] * a, "cR": p["cR"] * a, "cZ": p["cZ"] * a, "li": p["li"], "D": APRIORI["D"],
             "m1": APRIORI["m1"], "m2": APRIORI["m2"], "m3": APRIORI["m3"], "m4": APRIORI["m4"]}
        sc = self.cfg.get("sd_scale", 1.0)
        return np.array([s[c] * (1.0 if c == "D" else sc) for c in self.coords])

    def basis(self, psi, vac, f, g0, Dn, sig):
        """fields per unit prior sd of each coordinate, for the CURRENT map psi (geometry frame g0 of the start contour)."""
        G = self.G
        pp = psi - vac
        gz, gr = np.gradient(pp, G.Z, G.R)
        x = (G.RR - g0["Rg"]) / g0["a"]; z = (G.ZZ - g0["Zc"]) / g0["b"]
        T = _taper(x) * _taper(z)
        rho2 = ((G.RR - g0["Ra"]) / g0["a"]) ** 2 + ((G.ZZ - g0["Za"]) / g0["b"]) ** 2
        Gc = np.exp(-rho2 / 0.25)
        phi = G.sign * psi
        ins = G.inside(f["C"]) & (phi > f["lev"])
        span = max(f["phia"] - f["lev"], 1e-9)
        sfl = np.clip((phi - f["lev"]) / span, 0.0, 1.0)
        u = G.RR / g0["Rg"]; v = (G.ZZ - g0["Zc"]) / g0["Rg"]
        mp = {"m1": u ** 2, "m2": u ** 2 * v, "m3": u ** 2 * (u ** 2 - 4 * v ** 2), "m4": u ** 2 * v * (3 * u ** 2 - 4 * v ** 2)}
        gpp = G.gamma(pp)
        out = []
        for c in self.coords:
            if c == "sR": B = -T * gr
            elif c == "sZ": B = -T * gz
            elif c == "dR": B = -T * x * gr
            elif c == "dZ": B = -T * z * gz
            elif c == "shr": B = -T * z * gr
            elif c == "tri": B = -T * z ** 2 * gr
            elif c == "z2": B = -T * z ** 2 * gz
            elif c == "sq": B = -T * x * z ** 2 * gr
            elif c == "cR": B = -Gc * gr
            elif c == "cZ": B = -Gc * gz
            elif c == "li": B = G.sign * np.where(ins, (phi - f["lev"]) * sfl, 0.0)
            elif c == "D": B = Dn
            else:
                m = mp[c]
                mi = m[ins] if ins.sum() > 3 else m.ravel()
                m = m - mi.mean()
                rms = float(np.sqrt(np.mean((mi - mi.mean()) ** 2))) or 1.0
                B = G.sign * span * m / rms
            if c not in ("m1", "m2", "m3", "m4") and abs(gpp) > 0:             # current-neutral projection (multipoles are curl-free)
                B = B - (G.gamma(B) / gpp) * pp
            out.append(B)
        return np.stack(out) * sig[:, None, None]

    def decode(self, psi0, psiI, psiN, vac, mu, s, wj, mu_psi, v_psi, rn=None):
        """returns (psi, info). mu / s / wj: targets, corpus sds and loss weights of the TG entries (NaN target -> ignored)."""
        G, cfg = self.G, self.cfg
        info = {"status": "start", "n_it": 0, "L0": np.nan, "L1": np.nan}
        f0 = G.F(psi0)
        if f0 is None:
            info["status"] = "start_invalid"
            return psi0, info
        v0 = f0["v"]
        Rt, Zt, Rb, Zb, Ri, Ro = v0[7:13]
        a, b = 0.5 * (Ro - Ri), 0.5 * (Zt - Zb)
        if not (np.isfinite([a, b]).all() and a > 0.05 and b > 0.05):
            info["status"] = "start_degenerate"
            return psi0, info
        g0 = {"a": a, "b": b, "Rg": 0.5 * (Ro + Ri), "Zc": 0.5 * (Zt + Zb), "Ra": v0[0], "Za": v0[1]}
        sig = self.sigmas(a)
        nc = len(self.coords)
        # current-neutral net-minus-inverse direction
        pI, pN = psiI - vac, psiN - vac
        gI, gN = G.gamma(pI), G.gamma(pN)
        Dn = (pN * (gI / gN) - pI) if (abs(gN) > 0.5 * abs(gI) > 0) else (pN - pI)
        mode = cfg.get("mode", "X")
        lam = cfg.get("lam", 1.0) * 0.55 / (psi0.size * v_psi)
        kp = cfg.get("kp", 1e-3 if mode == "X" else 1.0)
        W = np.where(np.isfinite(mu) & np.isfinite(s) & (s > 0), wj, 0.0)
        if mode == "K" and rn is not None:
            W = np.where(W > 0, 1.0 / np.maximum(rn, 1e-3), 0.0) * np.where(np.arange(len(TG)) < 7, 1.0, cfg.get("cp_rel", 0.5))
        mu0 = np.where(W > 0, mu, 0.0)
        ins0 = G.inside(f0["C"])
        neg0, tv0 = osc_metrics(G, psi0, ins0)
        nax0 = G.n_axes(psi0, f0["C"], f0["lev"])
        gam0 = G.gamma(psi0 - vac)

        def loss(p, f, ctot):
            r = np.where(W > 0, (mu0 - np.nan_to_num(f["v"], nan=1e3)) / np.where(W > 0, s, 1.0), 0.0)
            bad = (W > 0) & ~np.isfinite(f["v"])
            return float(lam * ((p - mu_psi) ** 2).sum() + (W * r ** 2).sum() + kp * (ctot ** 2).sum() + 1e3 * bad.sum()), r

        def valid(p, f):
            """rejection gates: invalid contour, new axis, current oscillation, large jumps of the control points (topology switch)."""
            if f is None:
                return "contour"
            v = f["v"]
            if not np.isfinite(v[:7][np.isfinite(v0[:7])]).all():
                return "scalar_nan"
            if (np.abs(v[[8, 10]] - v0[[8, 10]]) > cfg.get("jump", 0.3) * b).any() or (np.abs(v[[11, 12]] - v0[[11, 12]]) > cfg.get("jump", 0.3) * a).any():
                return "jump"
            if np.hypot((v[0] - v0[0]) / a, (v[1] - v0[1]) / b) > cfg.get("axis_jump", 0.25):
                return "axis_jump"
            if G.n_axes(p, f["C"], f["lev"]) > nax0:
                return "new_axis"
            neg, tv = osc_metrics(G, p, ins0)
            if neg - neg0 > cfg.get("osc_neg", 0.02) or tv > cfg.get("osc_tv", 1.3) * tv0 + 1e-12:
                return "current_osc"
            return ""

        psi, f, ctot = psi0.copy(), f0, np.zeros(nc)
        L, _ = loss(psi, f, ctot)
        info["L0"] = L
        h = cfg.get("fd", 0.25); trust = cfg.get("trust", 0.25)
        rej = []
        for it in range(cfg.get("n_it", 3)):
            B = self.basis(psi, vac, f, g0, Dn, sig)
            J = np.zeros((len(TG), nc))
            for i in range(nc):
                fp, fm = G.F(psi + h * B[i]), G.F(psi - h * B[i])
                if fp is not None and fm is not None:
                    dv = (fp["v"] - fm["v"]) / (2 * h)
                elif fp is not None:
                    dv = (fp["v"] - f["v"]) / h
                elif fm is not None:
                    dv = (f["v"] - fm["v"]) / h
                else:
                    dv = np.zeros(len(TG))
                J[:, i] = np.where(np.isfinite(dv) & (W > 0), dv / np.where(W > 0, s, 1.0), 0.0)
            Bm = B.reshape(nc, -1)
            A = lam * (Bm @ Bm.T)
            gfl = lam * (Bm @ (psi - mu_psi).ravel())
            _, r = loss(psi, f, ctot)
            H = kp * np.eye(nc) + A + J.T @ (W[:, None] * J)
            g = J.T @ (W * r) - gfl - kp * ctot
            try:
                dc = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                break
            mx = float(np.abs(dc).max())
            if mx > trust:
                dc *= trust / mx
            ok_step = False
            for _bt in range(4):
                pt = psi + np.tensordot(dc, B, axes=1)
                ft = G.F(pt)
                why = valid(pt, ft)
                if not why:
                    Lt, _ = loss(pt, ft, ctot + dc)
                    if Lt < L - 1e-12:
                        ok_step = True
                        break
                else:
                    rej.append(why)
                dc *= 0.5
            if not ok_step:
                break
            psi, f, ctot, L = pt, ft, ctot + dc, Lt
            info["n_it"] = it + 1
        info.update({"status": "decoded" if info["n_it"] else "kept", "L1": L, "c": ctot, "v0": v0, "v1": f["v"], "rej": ",".join(sorted(set(rej))),
                     "closure": float((G.gamma(psi - vac) - gam0) / gam0) if gam0 else 0.0, "a": a, "b": b,
                     "dpsi_rms": float(np.sqrt(((psi - psi0) ** 2).mean()))})
        neg1, tv1 = osc_metrics(G, psi, ins0)
        info.update({"neg0": neg0, "neg1": neg1, "tv0": tv0, "tv1": tv1})
        return psi, info
