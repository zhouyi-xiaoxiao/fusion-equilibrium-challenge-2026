#!/usr/bin/env python3
"""Stage 11h / p2chan — P2 inner/outer split as two extra circuits of the G1 coil model (documented optional module).

Notation of external review follow-up A (Q-C): per P2 coil, T = 12 I_in + 8 I_out (shipped Rogowski total, A.turn),
d = I_in - I_out (A)  =>  I_in = (T + 8 d) / 20,  I_out = (T - 12 d) / 20.
The G1 CoilSet spreads T equally over the 20 shipped elements (d = 0).  P2SplitCoilSet keeps that circuit untouched and adds
one circuit per coil, 'P2DU' / 'P2DL', whose unit current is d = 1 A:  +0.4 A in each of the 12 inner elements, -0.6 A in each
of the 8 outer elements (+4.8 / -4.8 A.turn; zero net current), i.e. delta psi = 4.8 d (G_in - G_out).
With d = 0 the model is the G1 model.  The class is only instantiated for shots that need it, so every other shot runs the
unmodified code path bit for bit.

Sources of d (cfg key p2_wiring):
  "auto"  : per-shot wiring class from the shipped channels (p2wiring.classify_p2).  'efps' -> d = 0 (standard solver);
            'sol' -> d_s = (T_s - 20 I_sol) / 12  (outer winding carries the solenoid feed current).
  "f"     : experiment / prior module: inner ampere-turn fraction f_s = 12 I_in / T  ->  d_s = (T_s / 4.8) (f_s - 0.6);
            cfg p2_f = [f_U, f_L].  Near T = 0 use p2_d = [d_U, d_L] (A) directly.
Prior of an external review for the case that the split is NOT observed (kept for the integrator; not needed on this corpus because
the wiring is observed on all 1,206 shots): f ~ N(0.6, 0.12^2) on [0, 1]; |T|/20 < I_ref: d ~ N(0, (0.5 I_ref)^2),
I_ref = max(median_plasma |T| / 20, 1 kA); shot-level offset + 50 ms-correlated variation; common (U = L) mode with twice the
prior width of the differential mode -> helper split_prior().
"""
from __future__ import annotations

import numpy as np

import g1_core as C
import run_g1 as RG
import g1_data as D

N_IN, N_OUT = 12, 8
_SP = {}


def p2_elements(geom):
    """Indices of the inner (12) and outer (8) elements of each P2 coil in the shipped coil table (the two radial columns of
    smaller R are the inner winding: public FAIR-MAST level-2 names p2_inner_* / p2_outer_*, EFUND groups P2I* / P2O*)."""
    names = [str(c).replace("magnetics_", "") for c in geom["input_column"]]
    R = np.asarray(geom["R"], float)
    out = {}
    for side in "ul":
        idx = [i for i, n in enumerate(names) if n == f"p2{side}_current"]
        cols = sorted(set(np.round(R[idx], 4).tolist()))
        assert len(idx) == N_IN + N_OUT and len(cols) == 4, f"unexpected P2 element table: {len(idx)} elements, {len(cols)} radial columns"
        inner = [i for i in idx if round(float(R[i]), 4) in cols[:2]]
        outer = [i for i in idx if round(float(R[i]), 4) in cols[2:]]
        assert len(inner) == N_IN and len(outer) == N_OUT, (len(inner), len(outer))
        out[side] = (inner, outer)
    return out


class P2SplitCoilSet(C.CoilSet):
    def __init__(self, geom, machine, grid, sol_scale=1.0):
        super().__init__(geom, machine, grid, sol_scale)
        assert machine == "mast"
        el = p2_elements(geom)
        fR, fZ, fw, fc, rows = list(self.fR), list(self.fZ), list(self.fw), list(self.fc), []
        for side in "ul":
            name = f"P2D{side.upper()}"
            ci = len(self.names)
            self.names.append(name); self.index[name] = ci
            Gm = np.zeros((grid.nx, grid.ny))
            inner, outer = el[side]
            for i, w_el in [(i, +N_OUT / (N_IN + N_OUT)) for i in inner] + [(i, -N_IN / (N_IN + N_OUT)) for i in outer]:
                Rs, Zs = C.rect_filaments(geom["R"][i], geom["Z"][i], geom["width"][i], geom["height"][i], geom["angle1"][i], geom["angle2"][i])
                for r_, z_ in zip(Rs, Zs):
                    w = w_el / len(Rs)
                    fR.append(r_); fZ.append(z_); fw.append(w); fc.append(ci)
                    Gm += w * C.Greens(r_, z_, grid.R, grid.Z)
            rows.append(Gm)
        self.fR, self.fZ, self.fw = np.array(fR, float), np.array(fZ, float), np.array(fw, float)
        self.fc = np.array(fc, int)
        self.nc = len(self.names)
        self.G = np.concatenate([self.G, np.array(rows)], axis=0)
        self.wz = np.concatenate([self.wz, np.zeros(2)])          # the split modes are not part of the vertical feedback combination


def get_split_solver(machine, row, cfg, wall_npz):
    """Solver with the two split circuits (cached per geometry / grid / solenoid factor, like run_g1.get_solver)."""
    geom, gh = D.geom_of(row)
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    full = {**C.DEFAULT_CFG, **cfg}
    key = (machine, gh, gR.tobytes(), gZ.tobytes(), float(full["sol_scale_mast"]), int(full["nsub"]))
    if key not in _SP:
        ns = int(full["nsub"])
        if ns > 1:
            gR = np.linspace(gR[0], gR[-1], (len(gR) - 1) * ns + 1); gZ = np.linspace(gZ[0], gZ[-1], (len(gZ) - 1) * ns + 1)
        grid = C.Grid(gR, gZ)
        w = np.load(wall_npz)
        wall = C.Wall(grid, w["limiter_R"], w["limiter_Z"], up=2 if ns == 1 else 1)
        coils = P2SplitCoilSet(geom, machine, grid, sol_scale=float(full["sol_scale_mast"]))
        _SP.clear()
        _SP[key] = C.Solver(machine, grid, wall, coils, cfg)
    return _SP[key]


def setup(machine, path, row, cfg, wall_npz, inp):
    """Returns None (standard path) or {'solver': S, 'd': {'U': (T,), 'L': (T,)} [A], 'cfg': class}.
    inp = g1_data.frame_inputs(row, machine, efit_times): currents already on the frame times, A / A.turn."""
    mode = cfg.get("p2_wiring", "off")
    if machine != "mast" or mode == "off":
        return None
    TU, TL, S = inp["cur"]["p2u_current"], inp["cur"]["p2l_current"], inp["cur"]["sol_current"]
    if mode == "auto":
        import p2wiring as PW
        cls = PW.classify_p2(PW.load_p2(path))
        if cls["cfg"] != "sol":
            return {"solver": None, "d": None, "cfg": cls["cfg"], "margin": cls["margin"]}
        d = {"U": (TU - (N_IN + N_OUT) * S) / N_IN, "L": (TL - (N_IN + N_OUT) * S) / N_IN}
        return {"solver": get_split_solver(machine, row, cfg, wall_npz), "d": d, "cfg": "sol", "margin": cls["margin"]}
    if mode == "f":
        fU, fL = cfg.get("p2_f", [0.6, 0.6])
        d0 = cfg.get("p2_d", [0.0, 0.0])
        d = {"U": TU / 4.8 * (fU - 0.6) + d0[0], "L": TL / 4.8 * (fL - 0.6) + d0[1]}
        return {"solver": get_split_solver(machine, row, cfg, wall_npz), "d": d, "cfg": f"f{fU:g}_{fL:g}", "margin": np.nan}
    raise ValueError(f"p2_wiring = {mode!r}")


def split_prior(T_abs_med_plasma_A, sigma_f=0.12, common_factor=2.0):
    """Prior widths of an external review for an UNOBSERVED split.  Returns dict with I_ref [A], sigma_d_near0 [A] and the
    (common, differential) standard deviations of f: the common up/down mode gets twice the width of the differential one,
    with the per-coil marginal width kept at sigma_f:  f_U, f_L = 0.6 + c +- e,  sd(c) = 2 sd(e),  sd(c)^2 + sd(e)^2 = sigma_f^2."""
    i_ref = max(T_abs_med_plasma_A / 20.0, 1e3)
    sd_e = sigma_f / np.sqrt(1.0 + common_factor ** 2)
    return {"I_ref": i_ref, "sigma_d_near0": 0.5 * i_ref, "sd_f_common": common_factor * sd_e, "sd_f_diff": sd_e, "tau_ms": 50.0}
