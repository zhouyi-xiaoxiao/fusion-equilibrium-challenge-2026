#!/usr/bin/env python3
"""
Stage 11K / Ch2 corrector — per-frame geometry table of a "parts" run (G1 / INV layout) with the scorer's PUBLISHED functionals
(starter/fusion_scoring, unmodified code).

  w_*  OUR map, last closed contour extracted with the PUBLIC first-wall mask (the only extraction available label-free on MAST)
  p_*  OUR map, official extraction with the scorer envelope            (DIII-D only)
  g_*  LABEL map, official extraction with the scorer envelope          (DIII-D only)
Per set: axis (Ra, Za), control points of the contour: top (Rt, Zt), bottom (Rb, Zb), inboard (Ri, Zi), outboard (Ro, Zo),
kappa, tri_top, tri_bot, volume, li, boundary level `lev` and axis value `phia` of phi = sign * psi.
MAST: rule3_guard installed; only our parts + the public wall polygon are opened (no label, no envelope).
Usage: ktab.py --machine d3d|mast --run DIR --tag T --wall WALL.npz [--data-dir D3D_TRAIN] --out X.parquet [--procs N]
"""
import argparse, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(os.environ["WS"])
A = None
G = {}
PTS = ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Zi", "Ro", "Zo")
SC = ("kappa", "tri_top", "tri_bot", "volume", "li")


def setup(machine, wall):
    if machine == "mast":
        sys.path.insert(0, str(WS / "stage11_run/ch2_geometry"))
        import rule3_guard
        rule3_guard.install()
    sys.path.insert(0, str(WS / "starter/fusion_scoring"))
    import derive, lcfs
    G["derive"], G["lcfs"] = derive, lcfs
    w = np.load(wall)
    if machine == "d3d":
        m = np.load(WS / "starter/fusion_scoring/masks/d3d_envelope.npz")
        G["R"], G["Z"] = m["grid_R"].astype(float), m["grid_Z"].astype(float)
        G["MC"] = m["mask_coarse"].astype(bool)
        G["name"], G["sign"] = "DIII-D", -1.0
    else:
        G["R"], G["Z"] = w["grid_R"].astype(float), w["grid_Z"].astype(float)
        G["MC"] = None
        G["name"], G["sign"] = "MAST", 1.0
    P = np.column_stack([w["limiter_R"], w["limiter_Z"]]).astype(float)
    RR, ZZ = np.meshgrid(G["R"], G["Z"])
    G["WM"] = derive._points_in_polygon(RR.ravel(), ZZ.ravel(), P).reshape(RR.shape)
    return G


def geom(p, mc, sign, G=G):
    """axis, contour control points, shape scalars, li, levels of map p with extraction mask mc; returns (dict, contour)."""
    derive, LC, R, Z, nm = G["derive"], G["lcfs"], G["R"], G["Z"], G["name"]
    out = {k: np.nan for k in PTS + SC + ("lev", "phia")}
    try:
        Ra, Za, iz, ir = derive.magnetic_axis(p, R, Z, nm, mc, sign)
    except Exception:  # noqa: BLE001
        return out, None
    out["Ra"], out["Za"] = float(Ra), float(Za)
    C, _ = LC.extract_lcfs_with_sign(p, R, Z, nm, mc, None if mc is None else mc.astype(float), n_iter=22, axis_sign=sign)
    if C is None or len(C) < 8:
        return out, None
    r, z = C[:, 0], C[:, 1]
    it, ib, ii, io = int(np.argmax(z)), int(np.argmin(z)), int(np.argmin(r)), int(np.argmax(r))
    out.update(Rt=float(r[it]), Zt=float(z[it]), Rb=float(r[ib]), Zb=float(z[ib]), Ri=float(r[ii]), Zi=float(z[ii]), Ro=float(r[io]), Zo=float(z[io]))
    s = dict(derive._shape_from_contour(C))
    try:
        s["li"] = derive.internal_inductance(p, R, Z, nm, C)
    except Exception:  # noqa: BLE001
        s["li"] = np.nan
    s = derive._clamp_to_nan(s)
    for k in SC:
        out[k] = float(s.get(k, np.nan))
    phi = sign * p
    out["lev"] = float(np.mean(derive._bilinear(phi, R, Z, r, z)))
    out["phia"] = float(phi[iz, ir])
    return out, C


def stack_psi(col):
    try:
        a = np.asarray(col, dtype=np.float32)
        if a.ndim == 3:
            return a
    except (ValueError, TypeError):
        pass
    return np.stack([np.stack([np.asarray(r, dtype=np.float32) for r in g]) for g in col])


def one(job):
    i, name = job
    f = Path(A.run) / "parts" / f"{A.tag}_{A.machine}_{i:05d}.npz"
    if not f.exists():
        return []
    c = np.load(f, allow_pickle=False)
    ks = c["ks"].astype(int)
    if A.every > 1:
        ks = ks[::A.every]
    d3 = A.machine == "d3d"
    if d3:
        row = pd.read_parquet(Path(A.data_dir) / name, columns=["efit_psirz", "efit_q95", "efit_beta_n"]).iloc[0]
        gt = stack_psi(row["efit_psirz"])
        gq = np.asarray(row["efit_q95"], float).ravel(); gb = np.asarray(row["efit_beta_n"], float).ravel()
    rows = []
    T = len(c["t"])
    for k in ks:
        p = c["psi"][k].astype(np.float32).astype(np.float64)
        row = {"file": name, "idx": i, "k": int(k), "nT": T, "t": float(c["t"][k]), "ip_A": float(c["ip_A"][k]), "ok": bool(c["ok"][k]),
               "q95": float(c["q95"][k]), "betaN": float(c["betaN"][k])}
        o, _ = geom(p, G["WM"], G["sign"])
        row.update({f"w_{a}": v for a, v in o.items()})
        if d3:
            o, _ = geom(p, G["MC"], G["sign"])
            row.update({f"p_{a}": v for a, v in o.items()})
            g = gt[k].astype(np.float64)
            o, _ = geom(g, G["MC"], G["sign"])
            row.update({f"g_{a}": v for a, v in o.items()})
            row.update({"gt_q95": float(gq[k]), "gt_betaN": float(gb[k]), "psi_sse": float(((p - g) ** 2).sum()), "psi_s1": float(g.sum()),
                        "psi_s2": float((g ** 2).sum()), "psi_n": int(g.size), "psi_dmean": float((p - g).mean())})
        rows.append(row)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["d3d", "mast"])
    ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--wall", required=True)
    ap.add_argument("--data-dir", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--procs", type=int, default=int(os.environ.get("NPROC", "8")))
    A = ap.parse_args()
    setup(A.machine, A.wall)
    names = [x for x in (Path(A.run) / f"files_{A.tag}_{A.machine}.txt").read_text().split("\n") if x]
    with Pool(A.procs) as pool:
        rows = [r for rr in pool.imap_unordered(one, list(enumerate(names)), chunksize=2) for r in rr]
    d = pd.DataFrame(rows).sort_values(["idx", "k"]).reset_index(drop=True)
    fr = Path(A.run) / f"frames_{A.tag}_{A.machine}.parquet"
    if fr.exists():
        f = pd.read_parquet(fr)
        drop = [c_ for c_ in f.columns if c_ in d.columns and c_ not in ("file", "k")]
        d = d.merge(f.drop(columns=drop), on=["file", "k"], how="left")
    d.to_parquet(A.out)
    print(f"KTAB {A.machine} {A.tag}: {len(d)} frames, {d['file'].nunique()} shots -> {A.out}; w contour fail {float(d['w_kappa'].isna().mean()):.4f}")
