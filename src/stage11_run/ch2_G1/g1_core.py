#!/usr/bin/env python3
"""
G1 core — rule-3-compliant, Thomson-constrained free-boundary Grad-Shafranov solver (CPU, deterministic).

Nothing here reads a file: the caller supplies the 65x65 grid (the shipped INPUT columns efit_grid_R/Z), the
shipped coil table, a PUBLIC first-wall polygon and per-frame inputs. The scorer envelope masks are never used.

Pieces
  Grid      linear Delta* solve (FreeGS 0.8.2 4th-order sparse direct solver) + von Hagenow free boundary with a
            pre-computed boundary Green matrix (identical numbers to freegs.boundary.freeBoundaryHagenow; self-test).
  CoilSet   shipped coil rectangles -> filaments (README unit conventions) + unit flux maps per circuit, a virtual
            uniform-vertical-field pair (radial closure knob) and the up/down antisymmetric set (Lackner feedback).
  Wall      public limiter polygon -> inside masks on the solver grid and on a 2x finer grid.
  analyse   O/X points (vectorised Newton on a bicubic spline) and the plasma boundary by a CONNECTIVITY test:
            psi_b = lowest level whose {psi > c} component around the axis does not reach outside the wall, snapped
            to an adjacent saddle (diverted) or to the max of psi on the adjacent wall points (limited).
  Solver    Picard iteration with three closures solved every step:
              vertical  Lackner feedback current so that B_r(R_axis, Z_t) = 0 (Z_t = 0 or the field-null rule),
              radial    iso-flux: psi(P_Thomson) = psi_b through the virtual vertical-field current,
              profile   J_phi = L [b0 R/Rn + (1-b0) Rn/R] (1 - psiN^am)^an with p_axis and Ip given.
All solves use Ip > 0 (caller multiplies every current by sign(Ip)); the axis is a MAXIMUM of psi.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import ndimage
from scipy.integrate import romb
from scipy.interpolate import RectBivariateSpline
from scipy.special import beta as beta_fn

import freegs
from freegs.boundary import freeBoundaryHagenow
from freegs.gradshafranov import Greens, GreensBr, GreensBz
from freegs.machine import Machine

MU0 = 4e-7 * np.pi
N_TF = {"d3d": 144, "mast": 24}          # TF turns: F = mu0 N I / 2pi (see note for the citation status)
R0_PUBLIC = {"d3d": 1.70, "mast": 0.85}  # nominal major radius (profile normalisation radius / cold-start anchor)
A_PUBLIC = {"d3d": 0.60, "mast": 0.55}   # nominal minor radius (only used to scale the Thomson edge bias)


# ----------------------------------------------------------------------------------------------- grid
def _romb_weights(n):
    w = np.zeros(n)
    e = np.zeros(n)
    for i in range(n):
        e[:] = 0.0
        e[i] = 1.0
        w[i] = romb(e)
    return w


class Grid:
    def __init__(self, gR, gZ):
        gR, gZ = np.asarray(gR, float), np.asarray(gZ, float)
        self.nx, self.ny = len(gR), len(gZ)
        self.eq = freegs.Equilibrium(tokamak=Machine([]), Rmin=gR[0], Rmax=gR[-1], Zmin=gZ[0], Zmax=gZ[-1],
                                     nx=self.nx, ny=self.ny, boundary=freeBoundaryHagenow)
        self.R, self.Z = self.eq.R, self.eq.Z            # exactly uniform (the shipped grid is float32-rounded)
        self.gR, self.gZ = np.array(self.R[:, 0]), np.array(self.Z[0, :])
        assert np.abs(self.gR - gR).max() < 1e-5 and np.abs(self.gZ - gZ).max() < 1e-5, "grid is not uniform"
        self.dR = float(self.gR[1] - self.gR[0])
        self.dZ = float(self.gZ[1] - self.gZ[0])
        self._build_hagenow()

    def _build_hagenow(self):
        R, Z, nx, ny = self.R, self.Z, self.nx, self.ny
        eps = 1e-2
        wz = _romb_weights(ny) * self.dZ
        wr = _romb_weights(nx) * self.dR
        segs = [[(x, 0, 0.0, -eps) for x in range(nx)], [(x, ny - 1, 0.0, eps) for x in range(nx)],
                [(0, y, -eps, 0.0) for y in range(ny)], [(nx - 1, y, eps, 0.0) for y in range(ny)]]
        self._hag = []
        for seg in segs:
            B = np.zeros((len(seg), 2 * ny + 2 * nx))
            for n, (x, y, Re, Ze) in enumerate(seg):
                Rp, Zp = R[x, y] + Re, Z[x, y] + Ze
                B[n, 0:ny] = Greens(R[0, :], Z[0, :], Rp, Zp) / R[0, :] * wz
                B[n, ny:2 * ny] = Greens(R[-1, :], Z[-1, :], Rp, Zp) / R[-1, :] * wz
                B[n, 2 * ny:2 * ny + nx] = Greens(R[:, 0], Z[:, 0], Rp, Zp) / R[:, 0] * wr
                B[n, 2 * ny + nx:] = Greens(R[:, -1], Z[:, -1], Rp, Zp) / R[:, -1] * wr
            self._hag.append((np.array([s[0] for s in seg]), np.array([s[1] for s in seg]), B))

    def hagenow(self, Jtor, psi):
        """In-place boundary values of the plasma flux (same algebra as freegs.boundary.freeBoundaryHagenow)."""
        dR, dZ = self.dR, self.dZ
        rhs = self.R * Jtor
        rhs[0, :] = 0.0
        rhs[:, 0] = 0.0
        rhs[-1, :] = 0.0
        rhs[:, -1] = 0.0
        pf = self.eq.callSolver(psi, rhs)
        co = [(0, 25.0 / 12), (1, -4.0), (2, 3.0), (3, -16.0 / 12), (4, 1.0 / 4)]
        dL = sum(w * pf[i, :] for i, w in co) / dR
        dRt = sum(w * pf[-(1 + i), :] for i, w in co) / dR
        dD = sum(w * pf[:, i] for i, w in co) / dZ
        dU = sum(w * pf[:, -(1 + i)] for i, w in co) / dZ
        dd = np.sqrt(dR ** 2 + dZ ** 2)
        dL[0] = dD[0] = sum(w * pf[i, i] for i, w in co) / dd
        dL[-1] = dU[0] = sum(w * pf[i, -(1 + i)] for i, w in co) / dd
        dRt[0] = dD[-1] = sum(w * pf[-(1 + i), i] for i, w in co) / dd
        dRt[-1] = dU[-1] = sum(w * pf[-(1 + i), -(1 + i)] for i, w in co) / dd
        v = np.concatenate([dL, dRt, dD, dU])
        for xs, ys, B in self._hag:
            psi[xs, ys] = B @ v

    def solve_plasma(self, Jtor, guess):
        """Plasma flux of a given J_phi with a free boundary (Equilibrium.solve minus the O/X-point search)."""
        psi = np.array(guess, dtype=float, copy=True)
        self.hagenow(Jtor, psi)
        rhs = -MU0 * self.R * Jtor
        rhs[0, :] = psi[0, :]
        rhs[:, 0] = psi[:, 0]
        rhs[-1, :] = psi[-1, :]
        rhs[:, -1] = psi[:, -1]
        return self.eq.callSolver(psi, rhs)


# ----------------------------------------------------------------------------------------------- coils
def rect_filaments(Rc, Zc, w, h, a1=0.0, a2=0.0, step=0.03):
    """Sub-filament grid of a rectangle (EFIT-style: a1 = rotation, a2 = parallelogram angle)."""
    w, h = abs(float(w)), abs(float(h))
    nr = max(1, min(int(np.ceil(w / step)), 12))
    nz = max(1, min(int(np.ceil(h / step)), 120))
    dr = (np.arange(nr) - (nr - 1) / 2.0) * (w / nr if nr > 1 else 0.0)
    dz = (np.arange(nz) - (nz - 1) / 2.0) * (h / nz if nz > 1 else 0.0)
    DR, DZ = np.meshgrid(dr, dz)
    DR, DZ = DR.ravel(), DZ.ravel()
    if a2 and abs(a2) > 1e-6:
        DR = DR + DZ / np.tan(np.deg2rad(a2))
    if a1 and abs(a1) > 1e-6:
        c, s = np.cos(np.deg2rad(a1)), np.sin(np.deg2rad(a1))
        DR, DZ = c * DR - s * DZ, s * DR + c * DZ
    return Rc + DR, Zc + DZ


def current_columns(machine):
    if machine == "d3d":
        return [f"F{i}{s}" for i in range(1, 10) for s in "AB"] + ["ECOILA"]
    return [f"p{i}{s}_current" for i in range(2, 7) for s in "lu"] + ["sol_current"]


def zfb_weights(machine):
    if machine == "d3d":
        return {**{f"F{i}A": 1.0 for i in range(1, 10)}, **{f"F{i}B": -1.0 for i in range(1, 10)}}
    return {**{f"p{i}u_current": 1.0 for i in range(2, 7)}, **{f"p{i}l_current": -1.0 for i in range(2, 7)}}


class CoilSet:
    """Circuits = shipped magnetics columns (+ 'VBZ' virtual pair). Unit conventions (starter README):
       D3D F*: kA.turn per rectangle; D3D ECOILA: no resolved shape -> 0 turns here, its flux is the fitted gauge
       offset; MAST p*: kA.turn shared by the N elements of the circuit; MAST sol: kA in each of the 656 elements.
       Circuit current unit = A of the shipped unit (column value in kA x 1e3)."""

    def __init__(self, geom, machine, grid: Grid, sol_scale: float = 1.0):
        # sol_scale (MAST only): effective solenoid turns / 656 shipped elements (Ch2 forensic, an internal development note)
        names = [str(c).replace("magnetics_", "") for c in geom["input_column"]]
        self.machine = machine
        self.names = sorted(set(names)) + ["VBZ", "VQ"]
        self.index = {n: i for i, n in enumerate(self.names)}
        fR, fZ, fw, fc = [], [], [], []
        for key in self.names[:-2]:
            idx = [i for i, n in enumerate(names) if n == key]
            for i in idx:
                Rs, Zs = rect_filaments(geom["R"][i], geom["Z"][i], geom["width"][i], geom["height"][i],
                                        geom["angle1"][i], geom["angle2"][i])
                if machine == "mast":
                    mult = float(sol_scale) if key == "sol_current" else 1.0 / len(idx)
                else:
                    mult = 0.0 if key == "ECOILA" else 1.0
                if mult == 0.0:
                    continue
                fR += list(Rs); fZ += list(Zs)
                fw += [mult / len(Rs)] * len(Rs); fc += [self.index[key]] * len(Rs)
        # analytic virtual vacuum modes (exact solutions of Delta* psi = 0), unit = 1 T:
        #   VBZ  uniform vertical field        psi = R^2 / 2                       (even radial-force mode)
        #   VQ   quadrupole about (R0, 0)       psi = cq (R^2 Z^2 - R^4/4 + R0^2 R^2/2), Bz(R0 -+ a, 0) ~ +-1 T (even shaping mode)
        self.iv, self.iq = self.index["VBZ"], self.index["VQ"]
        self.R0, self.cq = R0_PUBLIC[machine], 1.0 / (2.0 * A_PUBLIC[machine] * R0_PUBLIC[machine])
        self.fR, self.fZ, self.fw = np.array(fR, float), np.array(fZ, float), np.array(fw, float)
        self.fc = np.array(fc, int)
        self.nc = len(self.names)
        self.G = np.zeros((self.nc, grid.nx, grid.ny))
        for k in range(len(self.fR)):
            self.G[self.fc[k]] += self.fw[k] * Greens(self.fR[k], self.fZ[k], grid.R, grid.Z)
        self.G[self.iv], self.G[self.iq] = self._modes_psi(grid.R, grid.Z)
        self.wz = np.zeros(self.nc)
        for n, w in zfb_weights(machine).items():
            if n in self.index:
                self.wz[self.index[n]] = w
        self.bz_v0 = 1.0

    def _modes_psi(self, R, Z):
        return 0.5 * R ** 2, self.cq * (R ** 2 * Z ** 2 - 0.25 * R ** 4 + 0.5 * self.R0 ** 2 * R ** 2)

    def vec(self, cur: dict):
        v = np.zeros(self.nc)
        for n, x in cur.items():
            if n in self.index:
                v[self.index[n]] = x
        return v

    def psi_grid(self, I):
        return np.tensordot(I, self.G, axes=1)

    def _units(self, fn, R, Z):
        return np.bincount(self.fc, weights=self.fw * fn(self.fR, self.fZ, R, Z), minlength=self.nc)

    def psi_units(self, R, Z):
        u = self._units(Greens, R, Z)
        u[self.iv], u[self.iq] = self._modes_psi(R, Z)
        return u

    def br_units(self, R, Z):                            # B_R = -(1/R) dpsi/dZ
        u = self._units(GreensBr, R, Z)
        u[self.iv], u[self.iq] = 0.0, -self.cq * 2.0 * R * Z
        return u

    def bz_units(self, R, Z):                            # B_Z = (1/R) dpsi/dR
        u = self._units(GreensBz, R, Z)
        u[self.iv], u[self.iq] = 1.0, self.cq * (2.0 * Z ** 2 - R ** 2 + self.R0 ** 2)
        return u


# ----------------------------------------------------------------------------------------------- wall
def points_in_poly(r, z, pr, pz):
    """Even-odd ray cast; r, z arrays of any shape; polygon given open or closed."""
    r = np.asarray(r, float); z = np.asarray(z, float)
    inside = np.zeros(r.shape, bool)
    n = len(pr)
    for e in range(n):
        r1, z1, r2, z2 = pr[e], pz[e], pr[(e + 1) % n], pz[(e + 1) % n]
        if z1 == z2:
            continue
        cross = (z1 > z) != (z2 > z)
        xint = r1 + (z - z1) * (r2 - r1) / (z2 - z1)
        inside ^= cross & (r < xint)
    return inside


class Wall:
    def __init__(self, grid: Grid, lim_R, lim_Z, up=2):
        pr, pz = np.asarray(lim_R, float), np.asarray(lim_Z, float)
        if np.hypot(pr[0] - pr[-1], pz[0] - pz[-1]) < 1e-9:
            pr, pz = pr[:-1], pz[:-1]
        self.pr, self.pz = pr, pz
        self.inside = points_in_poly(grid.R, grid.Z, pr, pz)
        self.crit = ndimage.binary_dilation(self.inside, iterations=1)
        self.up = up
        self.fR = np.linspace(grid.gR[0], grid.gR[-1], (grid.nx - 1) * up + 1)
        self.fZ = np.linspace(grid.gZ[0], grid.gZ[-1], (grid.ny - 1) * up + 1)
        FR, FZ = np.meshgrid(self.fR, self.fZ, indexing="ij")
        self.inside_f = points_in_poly(FR, FZ, pr, pz)
        border = np.zeros_like(self.inside_f)
        border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
        ring = ndimage.binary_dilation(self.inside_f, iterations=1) & ~self.inside_f
        self.leak_idx = np.nonzero(ring | (self.inside_f & border))
        P = np.column_stack([np.append(pr, pr[0]), np.append(pz, pz[0])])
        dense = []
        for a, b in zip(P[:-1], P[1:]):
            m = max(1, int(np.ceil(np.hypot(*(b - a)) / 0.01)))
            t = np.arange(m) / m
            dense.append(a[None] + t[:, None] * (b - a)[None])
        self.dense = np.vstack(dense)
        self.dense_ij = (np.clip(np.rint((self.dense[:, 0] - self.fR[0]) / (self.fR[1] - self.fR[0])).astype(int), 0,
                                 len(self.fR) - 1),
                         np.clip(np.rint((self.dense[:, 1] - self.fZ[0]) / (self.fZ[1] - self.fZ[0])).astype(int), 0,
                                 len(self.fZ) - 1))
        self.R_in_wall = float(pr.min())

    def fidx(self, r, z):
        return (int(np.clip(round((r - self.fR[0]) / (self.fR[1] - self.fR[0])), 0, len(self.fR) - 1)),
                int(np.clip(round((z - self.fZ[0]) / (self.fZ[1] - self.fZ[0])), 0, len(self.fZ) - 1)))


# ----------------------------------------------------------------------------------------------- boundary analysis
def critical_points(grid: Grid, wall: Wall, spl):
    """Stationary points of psi inside the (dilated) wall: returns (maxima, saddles) as lists of (R, Z, psi)."""
    gR, gZ = grid.gR, grid.gZ
    pr = spl(gR, gZ, dx=1)
    pz = spl(gR, gZ, dy=1)
    bp2 = (pr * pr + pz * pz) / grid.R ** 2
    c = bp2[1:-1, 1:-1]
    ismin = np.ones_like(c, bool)
    n0, n1 = bp2.shape
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                ismin &= c < bp2[1 + di:n0 - 1 + di, 1 + dj:n1 - 1 + dj]
    full = np.zeros_like(bp2, bool)
    full[1:-1, 1:-1] = ismin
    full &= wall.crit
    full[:2, :] = full[-2:, :] = False
    full[:, :2] = full[:, -2:] = False
    ii, jj = np.nonzero(full)
    if len(ii) == 0:
        return [], []
    R1, Z1 = gR[ii].astype(float), gZ[jj].astype(float)
    R0, Z0 = R1.copy(), Z1.copy()
    step = np.full(len(R1), np.inf)
    for _ in range(30):
        a = spl.ev(R1, Z1, dx=1); b = spl.ev(R1, Z1, dy=1)
        arr = spl.ev(R1, Z1, dx=2); azz = spl.ev(R1, Z1, dy=2); arz = spl.ev(R1, Z1, dx=1, dy=1)
        det = arr * azz - arz * arz
        det = np.where(np.abs(det) < 1e-14, np.nan, det)
        sR = -(azz * a - arz * b) / det
        sZ = -(-arz * a + arr * b) / det
        sR = np.clip(np.nan_to_num(sR), -grid.dR, grid.dR)
        sZ = np.clip(np.nan_to_num(sZ), -grid.dZ, grid.dZ)
        R1 = np.clip(R1 + sR, gR[0], gR[-1]); Z1 = np.clip(Z1 + sZ, gZ[0], gZ[-1])
        step = np.hypot(sR, sZ)
        if step.max() < 1e-7:
            break
    arr = spl.ev(R1, Z1, dx=2); azz = spl.ev(R1, Z1, dy=2); arz = spl.ev(R1, Z1, dx=1, dy=1)
    det = arr * azz - arz * arz
    ok = (step < 1e-4) & (np.abs(R1 - R0) < 3 * grid.dR) & (np.abs(Z1 - Z0) < 3 * grid.dZ) & np.isfinite(det)
    val = spl.ev(R1, Z1)
    maxima, saddles, seen = [], [], []
    for k in np.nonzero(ok)[0]:
        if any((R1[k] - s[0]) ** 2 + (Z1[k] - s[1]) ** 2 < 1e-5 for s in seen):
            continue
        seen.append((R1[k], Z1[k]))
        if det[k] < 0:
            saddles.append((float(R1[k]), float(Z1[k]), float(val[k])))
        elif arr[k] < 0:
            maxima.append((float(R1[k]), float(Z1[k]), float(val[k])))
    return maxima, saddles


def analyse(grid: Grid, wall: Wall, psi, ref=None, dmax=None, n_bis=18):
    """Axis + connectivity-tested boundary + core mask. Returns dict or None."""
    spl = RectBivariateSpline(grid.gR, grid.gZ, psi)
    maxima, saddles = critical_points(grid, wall, spl)
    cands = [o for o in maxima if points_in_poly(np.array(o[0]), np.array(o[1]), wall.pr, wall.pz)]
    if not cands:
        return None
    if ref is not None and dmax is not None:
        near = [o for o in cands if np.hypot(o[0] - ref[0], o[1] - ref[1]) < dmax]
        if near:
            cands = near
    axis = max(cands, key=lambda o: o[2])
    Ra, Za, pa = axis
    fine = spl(wall.fR, wall.fZ)
    ia = wall.fidx(Ra, Za)
    lo = float(fine[wall.leak_idx].min()) - 1e-9
    hi = float(pa)
    if not hi > lo:
        return None
    struct = None

    def comp(c):
        lab, _ = ndimage.label(fine > c, structure=struct)
        k = lab[ia]
        return lab, k

    def leaks(c):
        lab, k = comp(c)
        return k != 0 and bool((lab[wall.leak_idx] == k).any())

    if leaks(lo):
        for _ in range(n_bis):
            mid = 0.5 * (lo + hi)
            if leaks(mid):
                lo = mid
            else:
                hi = mid
    else:
        hi = lo + 1e-9
    c_star, prec = hi, hi - lo                       # first wall-connection (maximin) level on the fine grid
    span0 = pa - c_star
    if span0 <= 0:
        return None
    tol = max(0.01 * span0, 8 * prec)
    lab, k = comp(c_star + 0.25 * tol)
    if k == 0:
        return None
    core_f = lab == k
    hf = wall.fR[1] - wall.fR[0]
    near = ndimage.binary_dilation(core_f, iterations=max(3, int(np.ceil(0.10 / hf))))     # 10 cm neighbourhood
    near_w = ndimage.binary_dilation(core_f, iterations=3)
    xact = [x for x in saddles if x[2] < pa and abs(x[2] - c_star) <= tol and near[wall.fidx(x[0], x[1])]]
    pw = spl.ev(wall.dense[:, 0], wall.dense[:, 1])
    nw = near_w[wall.dense_ij]
    psi_b = float(c_star)                             # no snapping: the flood level IS the boundary flux
    bpt, wpt, kind = None, None, "open"
    if nw.any():
        kk = int(np.argmax(np.where(nw, pw, -np.inf)))
        wpt = (float(wall.dense[kk, 0]), float(wall.dense[kk, 1]), float(pw[kk]))
    if xact:
        xb = min(xact, key=lambda x: abs(x[2] - c_star))
        bpt, kind = (xb[0], xb[1]), "xpt"
    elif wpt is not None and abs(wpt[2] - c_star) <= 2 * tol:
        bpt, kind = (wpt[0], wpt[1]), "wall"
    m = (psi > psi_b) & wall.inside
    for x in saddles:
        if x[2] < pa and abs(x[2] - psi_b) <= 2 * tol and near[wall.fidx(x[0], x[1])]:
            m &= (grid.Z > x[1]) if x[1] < Za else (grid.Z < x[1])
    labc, _ = ndimage.label(m)
    i0 = int(np.clip(round((Ra - grid.gR[0]) / grid.dR), 0, grid.nx - 1))
    j0 = int(np.clip(round((Za - grid.gZ[0]) / grid.dZ), 0, grid.ny - 1))
    kc = labc[i0, j0]
    if kc == 0:
        sub = labc[max(i0 - 1, 0):i0 + 2, max(j0 - 1, 0):j0 + 2]
        nz = sub[sub > 0]
        if len(nz) == 0:
            return None
        kc = int(np.bincount(nz).argmax())
    mask = labc == kc
    if mask.sum() < 6:
        return None
    return {"axis": (Ra, Za), "psi_axis": pa, "psi_b": psi_b, "limited": kind != "xpt", "kind": kind, "xact": xact,
            "saddles": saddles, "bpt": bpt, "wpt": wpt, "mask": mask, "spl": spl, "fine": fine, "c_star": float(c_star),
            "tol": float(tol)}


# ----------------------------------------------------------------------------------------------- profile
def shape_integral(am, an):
    return beta_fn(1.0 / am, an + 1.0) / am          # int_0^1 (1 - x^am)^an dx


class Profile:
    """J_phi = L [b0 R/Rn + (1 - b0) Rn/R] s(psiN),  s(u) = (1 - eps)(1 - u^am)^an + eps  inside the core (eps = edge / axis
    current-density ratio; eps = 0 is the edge-vanishing FreeGS family). p' and FF' share the shape s."""

    def __init__(self, paxis, Ip, am, an, Rn, b0max=0.98, eps=0.0):
        self.paxis, self.Ip, self.am, self.an, self.Rn, self.b0max, self.eps = float(paxis), float(Ip), am, an, Rn, b0max, float(eps)
        self.L = self.Beta0 = np.nan
        self.paxis_eff = self.paxis
        x = np.linspace(0.0, 1.0, 401)
        s = self.shape(x)
        self._x = x
        self._cum = np.concatenate([[0.0], np.cumsum(0.5 * (s[1:] + s[:-1]) * np.diff(x))])

    def shape(self, pn):
        return (1.0 - self.eps) * (1.0 - pn ** self.am) ** self.an + self.eps

    def jtor(self, grid: Grid, psi, b):
        span = b["psi_b"] - b["psi_axis"]
        pn = np.clip((psi - b["psi_axis"]) / span, 0.0, 1.0)
        shape = self.shape(pn) * b["mask"]
        si = ((1.0 - self.eps) * shape_integral(self.am, self.an) + self.eps) * span
        IR = romb(romb(shape * grid.R / self.Rn)) * grid.dR * grid.dZ
        I_R = romb(romb(shape * self.Rn / grid.R)) * grid.dR * grid.dZ
        if not (I_R > 0 and IR > 0):
            raise ValueError("empty core mask")
        LB = -self.paxis * self.Rn / si
        L = self.Ip / I_R - LB * (IR / I_R - 1.0)
        B0 = LB / L if L != 0 else self.b0max
        if not (0.0 <= B0 <= self.b0max):
            B0 = float(np.clip(B0, 0.0, self.b0max)) if np.isfinite(B0) and L > 0 else self.b0max
            L = self.Ip / (B0 * IR + (1.0 - B0) * I_R)
            LB = L * B0
        self.L, self.Beta0 = float(L), float(B0)
        self.paxis_eff = float(-LB * si / self.Rn)
        return L * (B0 * grid.R / self.Rn + (1.0 - B0) * self.Rn / grid.R) * shape

    def pressure(self, pn):
        pn = np.clip(pn, 0.0, 1.0)
        return self.paxis_eff * (1.0 - np.interp(pn, self._x, self._cum) / self._cum[-1])


# ----------------------------------------------------------------------------------------------- contours / scalars
def closed_contour(wall: Wall, fine, level, Ra, Za):
    import contourpy
    cg = contourpy.contour_generator(x=wall.fR, y=wall.fZ, z=fine.T)
    best = None
    for ln in cg.lines(level):
        if len(ln) < 8 or np.hypot(*(ln[0] - ln[-1])) > 1e-9:
            continue
        if points_in_poly(np.array(Ra), np.array(Za), ln[:-1, 0], ln[:-1, 1]):
            if best is None or len(ln) > len(best):
                best = ln
    return best


def q_on_contour(spl, C, fvac):
    mid = 0.5 * (C[1:] + C[:-1])
    dl = np.hypot(*(C[1:] - C[:-1]).T)
    g = np.hypot(spl.ev(mid[:, 0], mid[:, 1], dx=1), spl.ev(mid[:, 0], mid[:, 1], dy=1))
    return float(abs(fvac) / (2 * np.pi) * np.sum(dl / (mid[:, 0] * np.maximum(g, 1e-12))))


def crossings_h(C, Z0):
    r = []
    for k in range(len(C) - 1):
        z1, z2 = C[k, 1], C[k + 1, 1]
        if (z1 <= Z0 < z2) or (z2 <= Z0 < z1):
            t = (Z0 - z1) / (z2 - z1)
            r.append(C[k, 0] + t * (C[k + 1, 0] - C[k, 0]))
    return r


def crossings_v(C, R0):
    z = []
    for k in range(len(C) - 1):
        r1, r2 = C[k, 0], C[k + 1, 0]
        if (r1 <= R0 < r2) or (r2 <= R0 < r1):
            t = (R0 - r1) / (r2 - r1)
            z.append(C[k, 1] + t * (C[k + 1, 1] - C[k, 1]))
    return z


def scalars(grid: Grid, wall: Wall, psi, b, prof: Profile, fvac, Ip, chord=None):
    """Solution-derived scalars (all from our own psi + profile; fvac = mu0 N_TF I_TF / 2pi)."""
    out = {}
    Ra, Za = b["axis"]
    span = b["psi_b"] - b["psi_axis"]
    spl, fine = b["spl"], b["fine"]
    try:
        C95 = closed_contour(wall, fine, b["psi_axis"] + 0.95 * span, Ra, Za)
        if C95 is not None:
            # F(psi95)^2 = F_vac^2 + 2 (psi_a - psi_b) int_0.95^1 FF' du   (FF' = mu0 L (1 - b0) Rn shape)
            ff_int = MU0 * prof.L * (1.0 - prof.Beta0) * prof.Rn * (prof._cum[-1] - np.interp(0.95, prof._x, prof._cum))
            f95 = np.sqrt(max(fvac ** 2 + 2.0 * (b["psi_axis"] - b["psi_b"]) * ff_int, 1e-12))
            out["q95_vac"] = q_on_contour(spl, C95, fvac)
            out["f95_ratio"] = float(f95 / max(abs(fvac), 1e-12))
            out["q95"] = out["q95_vac"] * out["f95_ratio"]
    except Exception:
        pass
    try:
        C = closed_contour(wall, fine, b["psi_axis"] + 0.995 * span, Ra, Za)
        if C is not None:
            R_, Z_ = C[:-1, 0], C[:-1, 1]
            Rmax, Rmin, Zmax, Zmin = R_.max(), R_.min(), Z_.max(), Z_.min()
            a = 0.5 * (Rmax - Rmin); Rgeo = 0.5 * (Rmax + Rmin)
            R2 = np.append(R_, R_[0]); Z2 = np.append(Z_, Z_[0])
            cross = R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]
            vol = 2 * np.pi * abs(np.sum((R2[:-1] + R2[1:]) * cross) / 6.0)
            area = abs(0.5 * np.sum(cross))
            lp = float(np.sum(np.hypot(np.diff(R2), np.diff(Z2))))
            out.update({"a": float(a), "Rgeo": float(Rgeo), "kappa": float((Zmax - Zmin) / (2 * a)),
                        "tri_top": float((Rgeo - R_[np.argmax(Z_)]) / a), "tri_bot": float((Rgeo - R_[np.argmin(Z_)]) / a),
                        "volume": float(vol), "area": float(area), "lp": lp, "Rin": float(Rmin), "Rout": float(Rmax),
                        "Ztop": float(Zmax), "Zbot": float(Zmin)})
            rh = crossings_h(C, Za)
            if rh:
                out["Rin_ax"], out["Rout_ax"] = float(min(rh)), float(max(rh))
            if chord is not None:                       # LCFS crossing with the Thomson chord
                if chord[0] == "h":
                    rr = crossings_h(C, chord[1])
                    if rr:
                        out["edge_gs"], out["edge_gs_in"] = float(max(rr)), float(min(rr))
                else:
                    zz = crossings_v(C, chord[1])
                    if zz:
                        out["edge_gs"] = float(max(zz))
            m = b["mask"]
            pn = np.clip((psi - b["psi_axis"]) / span, 0, 1)
            p = prof.pressure(pn) * m
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


# ----------------------------------------------------------------------------------------------- solver
DEFAULT_CFG = {
    # profile closure (constants fitted on DIII-D only)
    "k_paxis": 2.2364, "alpha_m": 1.0, "alpha_n": 1.0, "j_edge": 0.0, "paxis_min": 300.0, "paxis_max": 4e5,
    # Picard iteration
    "rtol": 3e-3, "maxits": 80, "blend": 0.3, "relax_z": 0.6, "relax_r": 0.5,
    # Thomson edge closure: none | soft (MAP with a field prior, Huber) | hard (exact iso-flux, no prior)
    "edge_mode": "soft", "knobs": ["VBZ"], "sigma_s": 0.03, "huber": 1.5, "sigma_b": 0.03, "b_clip": 0.30,
    "use_inner": False, "gap_sigma": 0.0, "gap_min": 0.01, "hard_tol": 0.005,
    "resp_gain": {"d3d": 4.3, "mast": 5.6},          # d delta / d theta = resp_gain x vacuum Jacobian (fd_check.py, label-free)
    "edge_bias_m": {"d3d": 0.011, "mast": 0.0},      # D3D vertical-chord ne-foot bias vs EFIT (train); NOT transferred to MAST
    "f_edge": 0.3, "th_gap_ms": 10.0,
    # vertical: Lackner feedback pins the axis at zt (numerical stabilisation); zt = z_shrink * zero-force height
    "z_shrink": 0.0, "z_eta": 0.0, "zt_max_frac": 0.12, "push_frac": 0.01,
    "ip_min": 5e4, "dmax_frac": 0.45, "nsub": 1,
    # MAST solenoid effective-turn factor (1.0 = 656 turns x feed current; G1_soft_fix conventions in configs/)
    "sol_scale_mast": 1.0,
    # profile closures (run_g1.attempt): axis radius <-> pressure multiplier; li target <-> alpha_n
    "axis_mode": False, "axis_sigma": 0.025, "cp_sigma": 0.35, "cp_clip": 1.1, "li_mode": None, "lam_mode": None, "prof_its": 2,
}


class Solver:
    def __init__(self, machine, grid: Grid, wall: Wall, coils: CoilSet, cfg: dict):
        self.machine, self.grid, self.wall, self.coils = machine, grid, wall, coils
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.Rn = R0_PUBLIC[machine]
        self.zt_max = self.cfg["zt_max_frac"] * 0.5 * (grid.gZ[-1] - grid.gZ[0])
        self.dmax = self.cfg["dmax_frac"] * (grid.gR[-1] - grid.gR[0])
        self.knobs = [coils.index[n] for n in self.cfg["knobs"]]
        wzG = np.tensordot(coils.wz, coils.G, axes=1)
        self._br_fb = -np.gradient(wzG, grid.dZ, axis=1) / grid.R        # B_R of the unit feedback combination

    def cold_plasma(self, Ip, Rc, Zc):
        g = self.grid
        sig = 0.5 * A_PUBLIC[self.machine]
        w = np.exp(-((g.R - Rc) ** 2 + (g.Z - Zc) ** 2) / (2 * sig ** 2)) * self.wall.inside
        w *= Ip / (romb(romb(w)) * g.dR * g.dZ)
        return g.solve_plasma(w, np.zeros_like(g.R))

    def vacuum(self, I):
        return self.coils.psi_grid(I)

    def bp_star(self, Ip):
        return MU0 * Ip / (2 * np.pi * A_PUBLIC[self.machine])

    def solve(self, I_base, Ip, paxis, obs=None, warm=None, anchor=None, theta0=None, blend=None, maxits=None,
              am=None, an=None, edge_mode=None, zt=0.0):
        """One frame. I_base: circuit vector (A; Ip > 0 convention; virtual-mode entries 0).
        obs: list of Thomson edge observations {name, P=(R,Z), e=(eR,eZ) chord direction pointing out of the plasma,
             sig [m], kind 'eq' | 'ge' (P must be inside the plasma)}.
        Unknowns adjusted every Picard step: Lackner feedback current (axis pinned at zt) and theta = amplitudes [T] of
        the virtual vacuum modes cfg['knobs'], from  min sum_i huber(delta_i / sig_i) + sum_k (theta_k / sigma_k)^2,
        delta_i = (psi(P_i) - psi_b) / |dpsi/ds|  [m],  d delta_i / d theta_k = (G_k(P_i) - G_k(B)) / |dpsi/ds|
        (B = active X-point or wall contact: psi_b moves with theta) times a secant gain for the plasma response."""
        cfg, g, cs = self.cfg, self.grid, self.coils
        blend = cfg["blend"] if blend is None else blend
        maxits = cfg["maxits"] if maxits is None else maxits
        am = cfg["alpha_m"] if am is None else am
        an = cfg["alpha_n"] if an is None else an
        mode = cfg["edge_mode"] if edge_mode is None else edge_mode
        obs = list(obs or []) if mode != "none" else []
        use_gap = mode != "none" and cfg["gap_sigma"] > 0
        knobs, nk = self.knobs, len(self.knobs)
        bps = self.bp_star(Ip)
        sig_k = np.full(nk, cfg["sigma_b"] * bps)
        g0 = float(cfg["resp_gain"][self.machine])
        clip = cfg["b_clip"] * bps
        active = bool(obs) or use_gap
        t0 = time.time()
        res = {"ok": False, "err": "", "n_it": 0, "iso": bool(obs), "bps": float(bps), "zt": float(zt)}
        anchor = anchor or (self.Rn, 0.0)
        th = np.zeros(nk) if theta0 is None else np.array(theta0, float).copy()
        if warm is not None:
            plasma, dIz, ref = warm["plasma"].copy(), warm["dIz"], warm["axis"]
            if active and theta0 is None:
                th = np.array(warm["theta"], float).copy()
        else:
            plasma, dIz, ref = self.cold_plasma(Ip, anchor[0], zt), 0.0, (anchor[0], zt)
        if mode == "none":
            th[:] = 0.0
        prof = Profile(paxis, Ip, am, an, self.Rn, eps=cfg["j_edge"])
        st = {"relax": cfg["relax_r"], "g_prev": None, "d_prev": None, "pred_prev": None, "delta": {}, "gam": 1.0}
        ctrl_change, change = np.inf, np.nan

        def cvec():
            v = I_base + dIz * cs.wz
            v[knobs] += th
            return v

        def controls(b, plasma):
            nonlocal dIz, th
            Rax = b["axis"][0]
            spl_p = RectBivariateSpline(g.gR, g.gZ, plasma)
            br_pl = -float(spl_p.ev(Rax, zt, dy=1)) / Rax
            bru = cs.br_units(Rax, zt)
            i0 = I_base.copy(); i0[knobs] += th
            br0 = float(np.dot(i0, bru)); bunit = float(np.dot(cs.wz, bru))
            d_old, th_old = dIz, th.copy()
            if abs(bunit) > 1e-30:
                d_new = float(np.clip(-(br_pl + br0) / bunit, -0.5 * Ip, 0.5 * Ip))
                dIz = (1 - cfg["relax_z"]) * dIz + cfg["relax_z"] * d_new
            if active:
                spl = b["spl"]
                if b["bpt"] is not None:
                    RB, ZB = b["bpt"]
                elif b["wpt"] is not None:
                    RB, ZB = b["wpt"][0], b["wpt"][1]
                else:
                    RB, ZB = self.wall.R_in_wall, 0.0
                GB = cs.psi_units(RB, ZB)[knobs]
                names, D, Jm, S = [], [], [], []
                st["delta"] = {}
                for o in obs:
                    RP, ZP = o["P"]
                    pr, pz = float(spl.ev(RP, ZP, dx=1)), float(spl.ev(RP, ZP, dy=1))
                    slope = max(-(o["e"][0] * pr + o["e"][1] * pz), 0.3 * np.hypot(pr, pz), 1e-9)
                    d = (float(spl.ev(RP, ZP)) - b["psi_b"]) / slope
                    st["delta"][o["name"]] = d
                    if o["kind"] == "ge" and d >= 0:
                        continue
                    names.append(o["name"]); D.append(d); S.append(o["sig"])
                    Jm.append((cs.psi_units(RP, ZP)[knobs] - GB) / slope)
                if use_gap and b["wpt"] is not None:        # wall clearance: one-sided "X-point bounded" constraint
                    RC, ZC, pC = b["wpt"]
                    if b["kind"] == "xpt":
                        xr = (b["bpt"][0], b["bpt"][1], b["psi_b"])
                    else:
                        cand = [x for x in b["saddles"] if x[2] < b["psi_b"] - b["tol"] and abs(x[1] - b["axis"][1]) > 0.3 * A_PUBLIC[self.machine]]
                        xr = max(cand, key=lambda x: x[2]) if cand else None
                    if xr is not None:
                        gn = max(float(np.hypot(spl.ev(RC, ZC, dx=1), spl.ev(RC, ZC, dy=1))), 1e-9)
                        dw = (pC - xr[2]) / gn + cfg["gap_min"]
                        st["delta"]["gap"] = dw
                        if dw > 0:
                            names.append("gap"); D.append(dw); S.append(cfg["gap_sigma"])
                            Jm.append((cs.psi_units(RC, ZC)[knobs] - cs.psi_units(xr[0], xr[1])[knobs]) / gn)
                outer = False
                if b["kind"] == "wall" and obs and obs[0]["kind"] == "eq":
                    P0, e0 = obs[0]["P"], obs[0]["e"]         # contact point beyond the Thomson edge along the chord
                    outer = (RB - P0[0]) * e0[0] + (ZB - P0[1]) * e0[1] > -0.05
                if outer and mode != "probe" and nk:        # limited on the outboard side: unphysical branch for this
                    th = th.copy()                          # corpus (all frames diverted) -> push inward, skip the fit
                    th[0] = float(np.clip(th[0] - cfg["push_frac"] * bps, -clip, clip))
                    st["d_prev"] = st["pred_prev"] = st["g_prev"] = None
                    names = []
                if names and mode == "probe":               # diagnostics only: residuals + vacuum Jacobian, no update
                    st["jac"] = {n: [float(x) for x in np.atleast_1d(j)] for n, j in zip(names, Jm)}
                    st["jac_frozen"] = {o["name"]: float(cs.psi_units(*o["P"])[knobs][0]) for o in obs if o["name"] in names}
                elif names:
                    D, Jm, S = np.array(D), np.array(Jm).reshape(len(D), nk), np.array(S)
                    if mode == "hard":
                        w, P = np.ones(len(D)), np.zeros(nk)
                    else:
                        w = np.minimum(1.0, cfg["huber"] * S / np.maximum(np.abs(D), 1e-12))
                        P = 1.0 / sig_k ** 2
                    if st["d_prev"] is not None:            # secant gain: actual / predicted change of the residuals
                        common = [n for n in names if n in st["d_prev"] and n in st["pred_prev"]]
                        num = sum((D[names.index(n)] - st["d_prev"][n]) * st["pred_prev"][n] for n in common)
                        den = sum(st["pred_prev"][n] ** 2 for n in common)
                        if den > 1e-12:
                            st["gam"] = float(np.clip(num / den, 1.0, 8.0))
                    # true sensitivity = g0 x vacuum Jacobian (plasma response; fd_check.py: 4.3 D3D / 5.6 MAST, label-free)
                    A = g0 * max(g0, st["gam"]) * (Jm.T * (w / S ** 2)) @ Jm + np.diag(P)
                    gv = g0 * (Jm.T @ (w * D / S ** 2)) + P * th
                    if st["g_prev"] is not None and float(np.dot(gv, st["g_prev"])) < 0:
                        st["relax"] = max(0.15, 0.7 * st["relax"])
                    step = np.linalg.lstsq(A, -gv, rcond=None)[0]
                    th_new = np.clip(th + st["relax"] * step, -clip, clip)
                    st["g_prev"] = gv
                    st["d_prev"] = dict(zip(names, D))
                    st["pred_prev"] = dict(zip(names, Jm @ (th_new - th)))
                    th = th_new
                else:                                       # no active datum: keep the carried value
                    st["d_prev"] = st["pred_prev"] = st["g_prev"] = None
            return abs(dIz - d_old) / max(Ip, 1.0) + float(np.max(np.abs(th - th_old), initial=0.0)) / (0.4 * bps)

        try:
            b = analyse(g, self.wall, plasma + cs.psi_grid(cvec()), ref, self.dmax)
            if b is not None:
                controls(b, plasma)
            psi = plasma + cs.psi_grid(cvec())
            conv, bfin = False, None
            for it in range(maxits):
                psi_last = psi
                b = analyse(g, self.wall, psi, ref, self.dmax)
                if b is None:
                    res["err"] = "no O-point / boundary inside the wall"
                    break
                ref = b["axis"]
                J = prof.jtor(g, psi, b)
                plasma = g.solve_plasma(J, plasma)
                psi_new = plasma + cs.psi_grid(cvec())
                change = float(np.max(np.abs(psi_new - psi_last)) / (b["psi_axis"] - b["psi_b"]))
                res["n_it"] = it + 1
                if cfg.get("trace"):
                    res.setdefault("trace", []).append((it, round(b["axis"][0], 3), round(b["axis"][1], 3), round(b["psi_axis"], 4),
                                                        round(b["psi_b"], 4), b["kind"], len(b["xact"]), int(b["mask"].sum()),
                                                        round(prof.Beta0, 3), round(change, 4), round(dIz / Ip, 4),
                                                        [round(float(x), 4) for x in th], round(st["gam"], 2),
                                                        {k: round(v, 3) for k, v in st["delta"].items()}))
                b2 = analyse(g, self.wall, psi_new, ref, self.dmax)
                if b2 is None:
                    res["err"] = "no O-point / boundary inside the wall (post)"
                    break
                if not np.isfinite(change):
                    res["err"] = "non-finite"
                    break
                hard_ok = mode != "hard" or not obs or abs(st["delta"].get(obs[0]["name"], 0.0)) < cfg["hard_tol"] \
                    or np.any(np.abs(th) >= 0.999 * clip)
                if change < cfg["rtol"] and ctrl_change < cfg["rtol"] and hard_ok and it >= 2:
                    conv, bfin, psi = True, b2, psi_new
                    break
                ctrl_change = controls(b2, plasma)
                psi = (1.0 - blend) * (plasma + cs.psi_grid(cvec())) + blend * psi_last
            else:
                res["err"] = "Picard maxits"
            res.update({"dIz": float(dIz), "theta": [float(x) for x in th], "change": float(change), "gam": st["gam"],
                        "delta": dict(st["delta"]), "jac": st.get("jac"), "jac_frozen": st.get("jac_frozen"), "clip": bool(np.any(np.abs(th) >= 0.999 * clip)) if nk else False})
            if conv:
                res.update({"ok": True, "psi": psi, "plasma": plasma, "b": bfin, "prof": prof})
                # integrated vertical-force diagnostics of the MEASURED currents (+ virtual modes), feedback excluded:
                #   fz0 = <B_R^ext>_J, fz1 = <dB_R^ext/dZ>_J (weights R J_phi); zero-force height ~ Z_axis - fz0 / fz1
                Jf = prof.jtor(g, psi, bfin)
                i0 = I_base.copy(); i0[knobs] += th
                br = -np.gradient(cs.psi_grid(i0), g.dZ, axis=1) / g.R
                wj = g.R * Jf
                sw = float(wj.sum())
                if sw > 0:
                    res.update({"fz0": float((wj * br).sum() / sw), "fz1": float((wj * np.gradient(br, g.dZ, axis=1)).sum() / sw),
                                "fzfb": float(dIz * (wj * self._br_fb).sum() / sw)})
        except Exception as e:  # noqa: BLE001
            res["err"] = f"{type(e).__name__}: {e}"
        res["secs"] = time.time() - t0
        return res["ok"], res
