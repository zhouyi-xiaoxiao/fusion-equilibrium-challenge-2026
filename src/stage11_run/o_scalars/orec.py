#!/usr/bin/env python3
"""
Stage 11o / Ch2 scalars — per-frame scalar record of ANY parts run (G1 layout: <run>/parts/<tag>_<machine>_<idx>.npz + files_<tag>_<machine>.txt),
a pure function of (the run's own map, shipped inputs).  One parquet per run; no per-frame files.

  base   g_scalars/gs_scalars.frame_record (code unchanged, imported): contour q95 with F_vac (`q95v`), LCFS geometry on the 99.5 % contour
         (a, Rgeo, eps, kappa, tri, vol, lp, Rin, Rout), li, B0, Thomson-volume betaN_e / betap_e of THIS map (ref = None: no solver record needed)
  extra  q on psiN 0.80 / 0.90 / 0.95 (`q80v`, `q90v`, `q95x`), map-implied toroidal current inside the 95 % and 99.5 % contours (`I95`, `Ipb`),
         midplane crossings of the 99.5 % contour at the laser height (`Mi`, `Mo`), psiN of the map at the Thomson ne-foot edges (`u_Ti`, `u_To`),
         q95 of the map after a horizontal registration to the measured midplane edges:
           sA  affine R' = al + s R : LCFS crossings -> (inner edge, outer edge); a censored inner edge moves the LCFS only if it lies outside the
               coverage (to the first channel); a censored outer edge is not used
           sP  inboard-only stretch about the axis: inboard crossing of the map surface through the OUTER edge -> the measured inner edge
         (`q95_sA`, `Ipb_sA`, `sA_s`, `q95_sP`, `Ipb_sP`, `sP_s`); current-normalised values are formed downstream (q x I_map / Ip_measured)
  parts  the run's own part scalars (`q_part`, `b_part`), `ok`, and the net's auxiliary heads when present (`aux_<name>`)
  --src label (DIII-D ONLY): the same record on the EFIT label map (the label family's own functional: implied F95, G, c = betaN_label / betaN_e)

DIII-D: labels of every frame are attached (`lab_*`); edge targets = the pseudo-laser edges of the inverse runs (frames table efit_rin / efit_rout,
        + the stress offsets of ch2d_inverse/run_inv.pseudo_frame when --stress is given: sep 3 cm alternating per 4 shots, 1 cm noise).
MAST:   rule3_guard installed first; reads only our parts + shipped input columns (g1_data.load_row). No label, demo shot, envelope, efm product.

Usage: orec.py --machine d3d|mast --run DIR --tag T --data-dir D [--frames F.parquet] [--tar TAR] [--src parts|label] [--stress] --wall W --out X.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tarfile
import time
import traceback
from multiprocessing import get_context
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "stage11_run/g_scalars"))
sys.path.insert(0, str(WS / "stage11_run/ch2_geometry"))
import gs_scalars as GS  # noqa: E402  (read-only module of the 11g scalars track)

C, D = GS.C, GS.D
MU0 = C.MU0
A = None
LEVELS = ((0.80, "80"), (0.90, "90"), (0.95, "95"))


def cint(spl, Cn, f=None, fp=None):
    """(H, I) on the closed polyline Cn: H = sum dl / (R |grad psi|), I = sum |grad psi| / R dl; optionally under R' = f(R), fp = dR'/dR."""
    mid = 0.5 * (Cn[1:] + Cn[:-1])
    gR = spl.ev(mid[:, 0], mid[:, 1], dx=1); gZ = spl.ev(mid[:, 0], mid[:, 1], dy=1)
    if f is None:
        dl = np.hypot(*(Cn[1:] - Cn[:-1]).T); Rm = mid[:, 0]; g = np.hypot(gR, gZ)
    else:
        Rp = f(Cn[:, 0]); dl = np.hypot(Rp[1:] - Rp[:-1], Cn[1:, 1] - Cn[:-1, 1]); Rm = f(mid[:, 0]); g = np.hypot(gR / fp(mid[:, 0]), gZ)
    g = np.maximum(g, 1e-12)
    return float(np.sum(dl / (Rm * g))), float(np.sum(g / Rm * dl))


def extras(grid, wall, phi, fvac, zc, Ti, To, cin, cout, covlo):
    out = {}
    b = C.analyse(grid, wall, phi, None, None)
    if b is None:
        return out
    Ra, Za = b["axis"]; pa, pb = b["psi_axis"], b["psi_b"]; span = pb - pa
    spl, fine = b["spl"], b["fine"]
    out.update({"x_Ra": float(Ra), "x_Za": float(Za), "x_kind": b["kind"], "x_span": float(span)})
    k2 = abs(fvac) / (2 * np.pi)
    cont = {}
    for u, nm in LEVELS:
        cont[u] = C.closed_contour(wall, fine, pa + u * span, Ra, Za)
        if cont[u] is not None:
            H, I = cint(spl, cont[u])
            out[f"q{nm}x"] = k2 * H; out[f"I{nm}"] = I / MU0
    Cb = C.closed_contour(wall, fine, pa + 0.995 * span, Ra, Za)
    if Cb is None:
        return out
    _, Ib = cint(spl, Cb)
    out["Ipb"] = Ib / MU0
    cr = C.crossings_h(Cb, zc)
    if len(cr) < 2:
        return out
    Mi, Mo = float(min(cr)), float(max(cr))
    out["Mi"], out["Mo"] = Mi, Mo
    u_at = lambda R: float((spl.ev(R, zc) - pa) / span)  # noqa: E731
    inbox = lambda R: np.isfinite(R) and grid.gR[0] < R < grid.gR[-1]  # noqa: E731
    if inbox(To):
        out["u_To"] = u_at(To)
    if inbox(Ti):
        out["u_Ti"] = u_at(Ti)
    C95 = cont[0.95]
    if C95 is None:
        return out
    # ---- sA: affine two-sided registration of the LCFS midplane crossings
    ti, to, use = Mi, Mo, False
    if np.isfinite(To) and not cout:
        to, use = float(To), True
    if np.isfinite(Ti) and not cin:
        ti, use = float(Ti), True
    elif cin and np.isfinite(covlo) and Mi > covlo:
        ti, use = float(covlo), True
    if use and (to - ti) > 0.05 and (Mo - Mi) > 0.05:
        s = (to - ti) / (Mo - Mi); al = ti - s * Mi
        if 0.5 < s < 2.0:
            f = lambda R: al + s * R  # noqa: E731
            fp = lambda R: np.full(np.shape(R), s)  # noqa: E731
            H, _ = cint(spl, C95, f, fp); _, I = cint(spl, Cb, f, fp)
            out.update({"sA_s": float(s), "sA_al": float(al), "q95_sA": k2 * H, "Ipb_sA": I / MU0})
    # ---- sP: inboard-only stretch about the axis from the same-flux pair (outer edge, inner edge)
    if np.isfinite(To) and not cout and np.isfinite(Ti) and not cin and Ti < Ra < To and "u_To" in out:
        uo = out["u_To"]
        if uo > 0.3:
            Cu = Cb if uo >= 0.995 else C.closed_contour(wall, fine, pa + uo * span, Ra, Za)
            if Cu is not None:
                cu = C.crossings_h(Cu, zc)
                if len(cu) >= 2:
                    Miu = float(min(cu)); out["Mi_uo"] = Miu
                    if Ra - Miu > 0.03:
                        sn = (Ra - Ti) / (Ra - Miu)
                        if 0.5 < sn < 2.0:
                            f = lambda R: np.where(R < Ra, Ra - sn * (Ra - R), R)  # noqa: E731
                            fp = lambda R: np.where(R < Ra, sn, 1.0)  # noqa: E731
                            H, _ = cint(spl, C95, f, fp); _, I = cint(spl, Cb, f, fp)
                            out.update({"sP_s": float(sn), "q95_sP": k2 * H, "Ipb_sP": I / MU0})
    return out


def load_part(i):
    name = f"{A.tag}_{A.machine}_{i:05d}.npz"
    p = Path(A.run) / "parts" / name
    if p.exists():
        return np.load(p, allow_pickle=False)
    if A.tar_index is not None:
        off, size = A.tar_index.get(name, (None, None))
        if off is None:
            return None
        with open(A.tar, "rb") as f:
            f.seek(off); buf = f.read(size)
        return np.load(io.BytesIO(buf), allow_pickle=False)
    return None


def _shot(task):
    idx, path = task
    try:
        machine = A.machine
        if machine == "mast":
            import rule3_guard
            rule3_guard.install()
        row = D.load_row(path, machine)
        t = np.asarray(row["efit_times"], float)
        grid, wall = GS.grid_wall(row, A.wall)
        inp = D.frame_inputs(row, machine, t)
        th = D.thomson_frames(row, machine, t, C.DEFAULT_CFG)
        lab = D.labels_of(row) if machine == "d3d" else {}
        sg = 1.0 if machine == "mast" else -1.0
        name = Path(path).name
        fr = A.fr.get(name, {})
        if A.src == "label":
            P = lab["psi"]; ks = np.asarray(sorted(x for x in fr if isinstance(x, int)) if (fr and not A.all_frames) else range(len(t)), int)
            qp = bp = okp = aux = None; auxn = []
        else:
            c = load_part(idx)
            if c is None:
                return [{"idx": idx, "file": name, "fatal": "no part"}]
            P = c["psi"]; ks = c["ks"].astype(int); qp = c["q95"].astype(float); bp = c["betaN"].astype(float)
            okp = c["ok"] if "ok" in c.files else None
            aux = c["aux"] if "aux" in c.files else None
            auxn = [str(x) for x in c["aux_names"]] if aux is not None else []
        if machine == "mast":
            zc = float(th["chord"][1]) if th.get("chord") else 0.0
        else:
            zc = 0.0
        rng = np.random.default_rng(7_000_003 + idx)
        so = 0.03 * (1 if (int(fr.get("_idx", idx)) // 4) % 2 == 0 else -1) if A.stress else 0.0
        recs = []
        for k in ks:
            k = int(k)
            rec = {"idx": idx, "file": name, "k": k, "t": float(t[k]), "T": len(t), "n_ks": int(len(ks)),
                   "ip_A": float(abs(inp["ip_A"][k])), "ip_sgn": float(np.sign(inp["ip_A"][k])), "fvac": float(abs(inp["fvac"][k])), "zc": zc}
            if qp is not None:
                rec["q_part"], rec["b_part"] = float(qp[k]), float(bp[k])
                if okp is not None:
                    rec["ok_part"] = bool(okp[k])
                for j, n in enumerate(auxn):
                    rec[f"aux_{n}"] = float(aux[k, j])
            for nm in ("efit_q95", "efit_beta_n", "efit_li", "efit_r_axis", "efit_z_axis"):
                if nm in lab and k < len(lab[nm]):
                    rec["lab_" + nm[5:]] = float(lab[nm][k])
            # midplane edge targets
            if machine == "mast":
                Ti, cin, To, cout, covlo = float(th["rin"][k]), bool(th["cin"][k]), float(th["edge"][k]), bool(th["cens"][k]), float(th["covlo"][k])
                rec.update({"ts_has": bool(th["has"][k]), "ts_rin": Ti, "ts_cin": cin, "ts_edge": To, "ts_cens": cout, "ts_covlo": covlo,
                            "ts_rsym": float(th["rsym"][k]), "pe_max": float(th["pe_max"][k])})
            else:
                f = fr.get(k, {})
                rin, rout = f.get("efit_rin", np.nan), f.get("efit_rout", np.nan)
                covlo = A.wall_rin + 0.04
                e1, e2 = (0.01 * rng.standard_normal(2)) if A.stress else (0.0, 0.0)
                To = rout + so + e1; Ti = rin - so + e2
                cout = not np.isfinite(To); cin = (not np.isfinite(Ti)) or (Ti <= covlo)
                rec.update({"efit_rin": rin, "efit_rout": rout, "ts_rin": Ti, "ts_cin": cin, "ts_edge": To, "ts_cens": cout, "ts_covlo": covlo})
            Pk = P[k]
            if np.isfinite(Pk).all() and np.abs(Pk).max() > 0:
                phi = sg * Pk.astype(np.float64).T
                try:
                    rec.update(GS.frame_record(grid, wall, phi, None, abs(inp["fvac"][k]), abs(inp["ip_A"][k]), th["prof"][k], machine))
                except Exception as e:  # noqa: BLE001
                    rec["err_rec"] = f"{type(e).__name__}: {e}"
                try:
                    rec.update(extras(grid, wall, phi, abs(inp["fvac"][k]), zc, Ti, To, cin, cout, covlo))
                except Exception as e:  # noqa: BLE001
                    rec["err_x"] = f"{type(e).__name__}: {e}"
            recs.append(rec)
        return recs
    except Exception as e:  # noqa: BLE001
        return [{"idx": idx, "file": Path(path).name, "fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-600:]}]


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["d3d", "mast"]); ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", nargs="+", required=True); ap.add_argument("--wall", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default=None, help="frames table of the inverse member on the same shots (DIII-D: pseudo-laser edges efit_rin / efit_rout)")
    ap.add_argument("--tar", default=None, help="tar that holds the parts (members .../parts/<tag>_<machine>_<idx>.npz), read in place")
    ap.add_argument("--src", default="parts", choices=["parts", "label"]); ap.add_argument("--stress", action="store_true")
    ap.add_argument("--all-frames", action="store_true"); ap.add_argument("--max-shots", type=int, default=0); ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("NPROC", "8")))
    A = ap.parse_args()
    if A.machine == "mast":
        import rule3_guard
        rule3_guard.install()
        assert A.src == "parts", "label source is DIII-D only"
    import pandas as pd
    names = [x for x in (Path(A.run) / f"files_{A.tag}_{A.machine}.txt").read_text().split("\n") if x]
    paths = [str(next(Path(dd) / n for dd in A.data_dir if (Path(dd) / n).exists())) for n in names]
    w = np.load(A.wall)
    A.wall_rin = float(np.min(w["limiter_R"]))
    A.fr = {}
    if A.frames:
        cols = ["file", "idx", "k"] + (["efit_rin", "efit_rout"] if A.machine == "d3d" else [])
        fr = pd.read_parquet(A.frames)
        fr = fr[[c for c in cols if c in fr.columns]]
        for f, g in fr.groupby("file"):
            d = {int(k): {c: float(v) for c, v in zip(g.columns[3:], r)} for k, r in zip(g.k.values, g.iloc[:, 3:].values)}
            d["_idx"] = int(g.idx.values[0])
            A.fr[f] = d
    A.tar_index = None
    if A.tar:
        pre = f"{A.tag}_{A.machine}_"
        A.tar_index = {}
        with tarfile.open(A.tar) as tf:
            for m in tf:
                b = os.path.basename(m.name)
                if b.startswith(pre) and b.endswith(".npz") and b[len(pre):-4].isdigit():
                    A.tar_index[b] = (m.offset_data, m.size)
        print(f"tar index: {len(A.tar_index)} members of {A.tar} for tag {A.tag}", flush=True)
    tasks = list(enumerate(paths))[::A.stride]
    if A.max_shots:
        tasks = tasks[:A.max_shots]
    t0 = time.time(); recs = []
    with get_context("fork").Pool(A.procs) as pool:
        for r in pool.imap_unordered(_shot, tasks, chunksize=1):
            recs += r
    d = pd.DataFrame(recs).sort_values(["idx", "k"] if "k" in recs[0] else ["idx"]).reset_index(drop=True)
    for c in d.columns:
        if d[c].dtype == object and c not in ("file", "kind", "x_kind", "err_rec", "err_x", "fatal", "tb"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    tmp = A.out + ".tmp"
    d.to_parquet(tmp); os.replace(tmp, A.out)
    fat = int(d["fatal"].notna().sum()) if "fatal" in d else 0
    st = {"out": A.out, "shots": len(tasks), "frames": int(len(d)), "fatal": fat, "secs": round(time.time() - t0, 1)}
    for c in ("q95v", "q95x", "Ipb", "betaN_e", "q95_sA", "q95_sP", "err_rec", "err_x"):
        if c in d:
            st["n_" + c] = int(d[c].notna().sum())
    print("OREC " + json.dumps(st))
    if fat:
        print(d[d["fatal"].notna()][[c for c in ("file", "fatal", "tb") if c in d]].head(3).to_string())


if __name__ == "__main__":
    main()
