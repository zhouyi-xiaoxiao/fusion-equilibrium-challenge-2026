#!/usr/bin/env python3
"""
Ch2-D scalar track: per-frame physics quantities from a flux-map source + the shot's Thomson profiles.

Sources (one per run):
  g1     : our own G1 maps (<run>/parts/<tag>_<machine>_<idx>.npz), MAST or D3D
  efit   : DIII-D EFIT label maps (D3D ONLY; labels are allowed on DIII-D)
Per frame (canonical orientation phi = sign*psi, magnetic axis = maximum):
  * axis / connectivity boundary (g1_core.analyse, public wall), LCFS geometry (a, R_geo, kappa, V, perimeter), li(1)
  * q95 with the vacuum F = mu0 N_TF I_TF / 2pi (contour integral on the 95 % surface)
  * Thomson T_e, n_e, p_e mapped onto rho = sqrt(psiN) of THIS map -> monotone binned profiles -> volume averages
    -> beta_N,e = 200 mu0 <p_e>_V a / (B0 Ip[MA]), B0 = F_vac / R_geo (and at a fixed R0)
  * flux-surface quantities on 12 surfaces (eps, q, trapped fraction from |B| on the contour) -> neoclassical
    (Sauter 1999) and Spitzer conductivity on the grid -> plasma resistance R_p for Zeff in {1, 1.5, 2, 2.5, 3}
  * simple bootstrap-current estimate (Sauter-type local coefficients not used; global C sqrt(eps) beta_p,e kept raw)
MAST: rule3_guard installed before anything is opened; shots read via g1_data.load_row (input columns only). No MAST
label, demo shot or scorer envelope is read. D3D: labels only for the efit source + diagnostic columns.

Usage: python scal_frames.py --machine mast --src g1 --run <G1 out> --tag softk50 --data-dir <mast dir> --wall <npz>
           --out <parquet> --procs N
       python scal_frames.py --machine d3d --src efit --names <txt or dir:a:b> --data-dir <d3d dir> --wall ... --out ...
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
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402

MU0 = C.MU0
E_CH = 1.602176634e-19
ZEFFS = (1.0, 1.5, 2.0, 2.5, 3.0)
LEVELS = np.array([0.02, 0.06, 0.12, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.97])     # psiN of the surfaces
R0_FIX = {"d3d": 1.6955, "mast": 0.85}
_G = {}
A = None


def grid_wall(row, wall_npz):
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    key = (gR.tobytes(), gZ.tobytes())
    if key not in _G:
        g = C.Grid(gR, gZ)
        w = np.load(wall_npz)
        _G.clear()
        _G[key] = (g, C.Wall(g, w["limiter_R"], w["limiter_Z"], up=2))
    return _G[key]


# ------------------------------------------------------------------------------------------- profile fitting
def pav(y, w):
    """Weighted pool-adjacent-violators: non-increasing fit."""
    y = list(map(float, y)); w = list(map(float, w)); blocks = []
    for yi, wi in zip(y, w):
        blocks.append([yi, wi, 1])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            y2, w2, n2 = blocks.pop(); y1, w1, n1 = blocks.pop()
            blocks.append([(y1 * w1 + y2 * w2) / (w1 + w2), w1 + w2, n1 + n2])
    out = []
    for yb, wb, nb in blocks:
        out += [yb] * nb
    return np.array(out)


def fit_profile(rho, val, nb=12):
    """Monotone (non-increasing) binned profile on rho in [0, 1]; returns (rho_knots, values) or None."""
    m = np.isfinite(rho) & np.isfinite(val) & (val > 0) & (rho <= 1.08)
    if m.sum() < 6:
        return None
    rho, val = rho[m], val[m]
    edges = np.linspace(0.0, 1.0, nb + 1)
    c, v, n = [], [], []
    for i in range(nb):
        s = (rho >= edges[i]) & (rho < edges[i + 1])
        if s.sum():
            c.append(float(np.median(rho[s]))); v.append(float(np.median(val[s]))); n.append(int(s.sum()))
    if len(c) < 3:
        return None
    v = pav(v, n)
    e = (rho >= 0.95) & (rho <= 1.08)
    if e.sum() >= 2:
        vb = min(float(np.median(val[e])), v[-1])
    elif len(c) >= 2 and c[-1] < 1.0:
        sl = (v[-1] - v[-2]) / max(c[-1] - c[-2], 1e-3)
        vb = float(np.clip(v[-1] + sl * (1.0 - c[-1]), 0.0, v[-1]))
    else:
        vb = v[-1]
    if c[-1] < 0.999:
        c.append(1.0); v = np.append(v, vb)
    if c[0] > 0.0:
        c.insert(0, 0.0); v = np.insert(v, 0, v[0])
    return np.array(c), np.asarray(v, float), float(rho.min())


# ------------------------------------------------------------------------------------------- flux surfaces
def surface(spl, wall, fine, level, Ra, Za, F):
    Cn = C.closed_contour(wall, fine, level, Ra, Za)
    if Cn is None or len(Cn) < 12:
        return None
    mid = 0.5 * (Cn[1:] + Cn[:-1]); dl = np.hypot(*(Cn[1:] - Cn[:-1]).T)
    R = mid[:, 0]
    g = np.hypot(spl.ev(mid[:, 0], mid[:, 1], dx=1), spl.ev(mid[:, 0], mid[:, 1], dy=1))
    Bp = np.maximum(g, 1e-12) / R
    B = np.sqrt(Bp ** 2 + (F / R) ** 2)
    w = dl / Bp
    W = w.sum()
    h = B / B.max()
    h2 = float((h ** 2 * w).sum() / W)
    lam = (np.arange(48) + 0.5) / 48
    sq = np.array([(np.sqrt(np.maximum(1.0 - l * h, 0.0)) * w).sum() / W for l in lam])
    ft = float(1.0 - 0.75 * h2 * np.sum(lam / np.maximum(sq, 1e-6)) / 48)
    Rmax, Rmin = Cn[:, 0].max(), Cn[:, 0].min()
    return {"eps": float((Rmax - Rmin) / (Rmax + Rmin)), "Rc": float(0.5 * (Rmax + Rmin)),
            "q": float(F / (2 * np.pi) * np.sum(w / R ** 2)), "ft": float(np.clip(ft, 0.0, 1.0))}


def sauter_sigma(Te, ne, Z, ft, q, Rc, eps):
    """Spitzer and neoclassical parallel conductivity (Sauter, Angioni & Lin-Liu, PoP 6 (1999) 2834). Te eV, ne m^-3."""
    Te = np.maximum(Te, 5.0); ne = np.maximum(ne, 1e17)
    lnL = 31.3 - np.log(np.sqrt(ne) / Te)
    NZ = 0.58 + 0.74 / (0.76 + Z)
    s_sp = 1.9012e4 * Te ** 1.5 / (Z * NZ * lnL)
    nus = 6.921e-18 * q * Rc * ne * Z * lnL / (Te ** 2 * np.maximum(eps, 1e-3) ** 1.5)
    X = ft / (1.0 + (0.55 - 0.1 * ft) * np.sqrt(nus) + 0.45 * (1.0 - ft) * nus / Z ** 1.5)
    fneo = 1.0 - (1.0 + 0.36 / Z) * X + (0.59 / Z) * X ** 2 - (0.23 / Z) * X ** 3
    return s_sp, s_sp * np.clip(fneo, 0.02, 1.0), nus


# ------------------------------------------------------------------------------------------- per frame
def frame(grid, wall, phi, ref, F, Ip, prof, machine):
    out = {"ok_ana": False}
    b = C.analyse(grid, wall, phi, ref, 0.25 if ref is not None else None)
    if b is None:
        return out
    Ra, Za = b["axis"]; pa, pb = b["psi_axis"], b["psi_b"]
    span = pb - pa
    spl, fine = b["spl"], b["fine"]
    out.update({"ok_ana": True, "Rax": Ra, "Zax": Za, "psi_a": pa, "psi_b": pb, "kind": b["kind"]})
    C95 = C.closed_contour(wall, fine, pa + 0.95 * span, Ra, Za)
    if C95 is not None:
        out["q95v"] = C.q_on_contour(spl, C95, F)
    Cb = C.closed_contour(wall, fine, pa + 0.995 * span, Ra, Za)
    if Cb is None:
        return out
    R_, Z_ = Cb[:-1, 0], Cb[:-1, 1]
    a = 0.5 * (R_.max() - R_.min()); Rgeo = 0.5 * (R_.max() + R_.min())
    R2 = np.append(R_, R_[0]); Z2 = np.append(Z_, Z_[0])
    cr = R2[:-1] * Z2[1:] - R2[1:] * Z2[:-1]
    vol = 2 * np.pi * abs(np.sum((R2[:-1] + R2[1:]) * cr) / 6.0)
    lp = float(np.sum(np.hypot(np.diff(R2), np.diff(Z2))))
    out.update({"a": a, "Rgeo": Rgeo, "kappa": (Z_.max() - Z_.min()) / (2 * a), "vol": vol, "lp": lp,
                "Rin": R_.min(), "Rout": R_.max()})
    m = b["mask"]
    u = np.clip((phi - pa) / span, 0.0, 1.0)
    rho = np.sqrt(u)
    wv = grid.R * m
    Bpa = MU0 * abs(Ip) / lp
    dpr = spl(grid.gR, grid.gZ, dx=1); dpz = spl(grid.gR, grid.gZ, dy=1)
    bp2 = (dpr ** 2 + dpz ** 2) / grid.R ** 2
    out["li_map"] = float((bp2 * wv).sum() / max(wv.sum(), 1e-12) / Bpa ** 2) if Bpa > 0 else np.nan
    # flux surfaces
    S = []
    for lv in LEVELS:
        s = surface(spl, wall, fine, pa + lv * span, Ra, Za, F)
        if s is not None:
            S.append((np.sqrt(lv), s))
    if len(S) >= 4:
        rs = np.array([x[0] for x in S])
        ft_r = np.interp(rho, np.r_[0.0, rs], np.r_[0.0, [x[1]["ft"] for x in S]])
        q_r = np.interp(rho, rs, [x[1]["q"] for x in S])
        eps_r = np.interp(rho, np.r_[0.0, rs], np.r_[0.0, [x[1]["eps"] for x in S]])
        Rc_r = np.interp(rho, rs, [x[1]["Rc"] for x in S])
        out.update({"q0s": S[0][1]["q"], "ft_mid": float(np.interp(0.7, rs, [x[1]["ft"] for x in S])),
                    "eps_a": float(S[-1][1]["eps"])})
    else:
        ft_r = q_r = eps_r = Rc_r = None
    # Thomson profiles on this map
    B0 = abs(F) / Rgeo
    out["B0"] = B0
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
    fp, ft_, fn = fit_profile(rt, pe), fit_profile(rt, te), fit_profile(rt, ne)
    out["ts_n"] = int(((rt <= 1.0) & inside_box).sum())
    if fp is None or ft_ is None or fn is None:
        return out
    out["ts_rho_min"] = fp[2]
    pe_g = np.interp(rho, fp[0], fp[1]) * m
    te_g = np.interp(rho, ft_[0], ft_[1]); ne_g = np.interp(rho, fn[0], fn[1])
    W = max(wv.sum(), 1e-12)
    pe_v = float((pe_g * grid.R).sum() / W)
    out.update({"pe_vol": pe_v, "ne_vol": float((ne_g * wv).sum() / W), "te_vol_n": float((te_g * ne_g * wv).sum() / max((ne_g * wv).sum(), 1e-12)),
                "pe0": float(fp[1][0]), "te0": float(ft_[1][0]), "ne0": float(fn[1][0]),
                "We": 1.5 * pe_v * vol})
    sel = (rt < 0.9) & inside_box
    if sel.sum() >= 4:
        out["ts_rms_logte"] = float(np.sqrt(np.mean(np.log(te[sel] / np.interp(rt[sel], ft_[0], ft_[1])) ** 2)))
    ipm = abs(Ip) / 1e6
    out["betaN_e"] = float(200 * MU0 * pe_v * a / (B0 * ipm))
    out["betaN_e_R0"] = float(200 * MU0 * pe_v * a / ((abs(F) / R0_FIX[machine]) * ipm))
    out["betap_e"] = float(2 * MU0 * pe_v / Bpa ** 2)
    # conductivity -> plasma resistance
    if ft_r is not None:
        dA = grid.dR * grid.dZ
        for Z in ZEFFS:
            s_sp, s_neo, nus = sauter_sigma(te_g, ne_g, Z, ft_r, q_r, Rc_r, eps_r)
            Gneo = float((s_neo * m / (2 * np.pi * grid.R)).sum() * dA)
            out[f"Rp_neo_Z{Z:g}"] = 1.0 / Gneo if Gneo > 0 else np.nan
            if Z in (1.0, 2.0):
                Gsp = float((s_sp * m / (2 * np.pi * grid.R)).sum() * dA)
                out[f"Rp_sp_Z{Z:g}"] = 1.0 / Gsp if Gsp > 0 else np.nan
            if Z == 2.0:
                jw = s_neo * m / grid.R
                out["nus_j"] = float((nus * jw).sum() / max(jw.sum(), 1e-30))
                out["fneo_j"] = float(((s_neo / s_sp) * (s_sp * m / grid.R)).sum() / max((s_sp * m / grid.R).sum(), 1e-30))
    return out


def load_maps(src, machine, idx, path, row):
    """Returns (T x Z x R maps, sign to canonical, frames-to-process mask)."""
    t = np.asarray(row["efit_times"], float)
    if src == "efit":
        lab = D.labels_of(row)
        return lab["psi"].astype(np.float64), -1.0, np.ones(len(t), bool), lab
    c = np.load(Path(A.run) / "parts" / f"{A.tag}_{machine}_{idx:05d}.npz", allow_pickle=False)
    P = c["psi"].astype(np.float64)
    sel = np.zeros(len(P), bool)
    sel[c["ks"].astype(int)] = True
    lab = D.labels_of(row) if machine == "d3d" else {}
    return P, (1.0 if machine == "mast" else -1.0), sel, lab


def run_shot(task):
    idx, path = task
    try:
        return _run_shot(idx, path)
    except Exception as e:  # noqa: BLE001
        return [{"idx": idx, "file": Path(path).name, "fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}]


def _run_shot(idx, path):
    machine = A.machine
    if machine == "mast":
        import rule3_guard
        rule3_guard.install()
    row = D.load_row(path, machine)
    t = np.asarray(row["efit_times"], float)
    grid, wall = grid_wall(row, A.wall)
    P, sg, sel, lab = load_maps(A.src, machine, idx, path, row)
    inp = D.frame_inputs(row, machine, t)
    th = D.thomson_frames(row, machine, t, C.DEFAULT_CFG)
    refs = A.refs.get(idx, {}) if A.refs is not None else {}
    recs = []
    for k in range(len(t)):
        if not sel[k] or not np.isfinite(P[k]).all() or np.abs(P[k]).max() == 0:
            continue
        phi = sg * P[k].T
        ref = refs.get(k)
        if ref is None and A.src == "efit" and "efit_r_axis" in lab:
            ref = (float(lab["efit_r_axis"][k]), float(lab["efit_z_axis"][k]))
        rec = {"idx": idx, "file": Path(path).name, "k": k, "t": float(t[k]), "ip_A": float(abs(inp["ip_A"][k])),
               "fvac": float(abs(inp["fvac"][k])), "tf_kA": float(inp["tf_kA"][k])}
        try:
            rec.update(frame(grid, wall, phi, ref, abs(inp["fvac"][k]), abs(inp["ip_A"][k]), th["prof"][k], machine))
        except Exception as e:  # noqa: BLE001
            rec["err"] = f"{type(e).__name__}: {e}"
        for name, key in (("efit_q95", "gt_q95"), ("efit_beta_n", "gt_betaN"), ("efit_li", "gt_li"),
                          ("efit_r_axis", "gt_raxis")):
            if name in lab and k < len(lab[name]):
                rec[key] = float(lab[name][k])
        recs.append(rec)
    return recs


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["mast", "d3d"])
    ap.add_argument("--src", required=True, choices=["g1", "efit"])
    ap.add_argument("--run", default=None); ap.add_argument("--tag", default=None)
    ap.add_argument("--names", default=None, help="file list txt, or dir:a:b slice of sorted names")
    ap.add_argument("--data-dir", required=True, nargs="+")
    ap.add_argument("--wall", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=8); ap.add_argument("--limit", type=int, default=0)
    A = ap.parse_args()
    if A.machine == "mast":
        assert A.src == "g1", "MAST: own maps only"
        import rule3_guard
        rule3_guard.install()
    if A.src == "g1":
        names = [x for x in (Path(A.run) / f"files_{A.tag}_{A.machine}.txt").read_text().split("\n") if x]
    elif A.names and Path(A.names).is_file():
        names = [x for x in Path(A.names).read_text().split("\n") if x]
    else:
        d, a, b = A.names.rsplit(":", 2)
        names = sorted(p.name for p in Path(d).glob("*.parquet"))[int(a):int(b)]
    paths = []
    for n in names:
        for dd in A.data_dir:
            p = Path(dd) / n
            if p.exists():
                paths.append(str(p)); break
        else:
            raise FileNotFoundError(n)
    A.refs = None
    if A.src == "g1":
        import pandas as pd
        fr = pd.read_parquet(Path(A.run) / f"frames_{A.tag}_{A.machine}.parquet", columns=["idx", "k", "ok", "gs_Rax", "gs_Zax"])
        fr = fr[fr.ok == True]  # noqa: E712
        A.refs = {}
        for r in fr.itertuples():
            A.refs.setdefault(int(r.idx), {})[int(r.k)] = (float(r.gs_Rax), float(r.gs_Zax))
    tasks = list(enumerate(paths))
    if A.limit:
        tasks = tasks[:A.limit]
    t0 = time.time()
    recs = []
    with get_context("fork").Pool(A.procs) as pool:
        for i, r in enumerate(pool.imap_unordered(run_shot, tasks, chunksize=1)):
            recs += r
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(tasks)} shots {len(recs)} frames {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(recs)
    for c in df.columns:
        if df[c].dtype == object and c not in ("file", "kind", "err", "fatal", "tb"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_parquet(A.out)
    fat = df["fatal"].notna().sum() if "fatal" in df else 0
    print("SUMMARY", json.dumps({"out": A.out, "shots": len(tasks), "frames": len(df), "fatal": int(fat),
                                 "ok_ana": float(df.get("ok_ana", pd.Series([0])).fillna(False).astype(bool).mean()),
                                 "betaN_e": float(df["betaN_e"].notna().mean()) if "betaN_e" in df else 0.0,
                                 "secs": time.time() - t0}))
    if fat:
        print(df[df["fatal"].notna()][["file", "fatal", "tb"]].head(3).to_string())


if __name__ == "__main__":
    main()
