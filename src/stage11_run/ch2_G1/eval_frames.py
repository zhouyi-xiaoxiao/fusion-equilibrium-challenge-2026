#!/usr/bin/env python3
"""
DIII-D acceptance metrics for a G1 run on a frame subset: pooled score COMPONENTS with the official functionals
(starter/local_score.py build_reference / psi_residuals / score_shot + fusion_scoring finalize_machine, unmodified), on
exactly the frames the run solved (e.g. 150 holdout shots x 8 frames), with a shot bootstrap, all seven derived-scalar R2,
and the gauge-vs-shape decomposition of the flux error over the same pooled SST. DIII-D only (labels allowed there).
Usage: python eval_frames.py --run <dir> --tag T --data-dir <d3d train dir> [--calib calib.json] [--f32] [--boot 200]
"""
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd

WS = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(WS / "starter"))
import matplotlib
matplotlib.use("Agg")
import local_score as LS  # noqa: E402
from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS  # noqa: E402
from experiments import _as_psirz_stack  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

LS.MACHINE = "DIII-D"
_M = np.load(WS / "starter/fusion_scoring/masks/d3d_envelope.npz")      # official D3D scoring mask (D3D scoring only)
R, Z, MC = _M["grid_R"], _M["grid_Z"], _M["mask_coarse"].astype(bool)
MF = MC.astype(np.float64)
A = None


def fill(arr, tt, dflt):
    arr = arr.astype(float).copy(); m = np.isfinite(arr)
    arr[~m] = np.interp(tt[~m], tt[m], arr[m]) if m.any() else dflt
    return arr


def phase1(job):
    i, name = job
    c = np.load(Path(A.run) / "parts" / f"{A.tag}_d3d_{i:05d}.npz", allow_pickle=False)
    ks = c["ks"]
    row = pd.read_parquet(Path(A.data_dir) / name, columns=["efit_psirz", "efit_q95", "efit_beta_n"]).iloc[0]
    gt = {"psi": _as_psirz_stack(row["efit_psirz"])[ks], "q95": np.asarray(row["efit_q95"], float).ravel()[ks],
          "betaN": np.asarray(row["efit_beta_n"], float).ravel()[ks]}
    psi = c["psi"][ks].astype(np.float32)
    if not A.f32:
        psi = psi.astype(np.float16).astype(np.float32)
    tt = c["t"][ks].astype(float)
    pred = {"psirz": psi.astype(np.float64), "q95": np.clip(fill(A.cq * c["q95"][ks], tt, 4.0), 1.2, 25.0),
            "betaN": np.clip(fill(A.cb * c["betaN"][ks], tt, 1.5), 0.0, 8.0)}
    ref = LS.build_reference(gt["psi"], R, Z, MC, MF)
    g = gt["psi"].astype(np.float64); fin = np.isfinite(g)
    st = {"psi_sum": float(g[fin].sum()), "psi_sumsq": float((g[fin] ** 2).sum()), "psi_n": int(fin.sum()),
          "scal_sum": np.zeros(N_SCALARS), "scal_sumsq": np.zeros(N_SCALARS), "scal_n": np.zeros(N_SCALARS),
          "cons_sum": np.zeros(N_CONS), "cons_sumsq": np.zeros(N_CONS), "cons_n": np.zeros(N_CONS)}
    for j, nm in enumerate(SCALARS):
        v = gt[nm][np.isfinite(gt[nm])]
        st["scal_sum"][j], st["scal_sumsq"][j], st["scal_n"][j] = v.sum(), (v ** 2).sum(), v.size
    for j in range(N_CONS):
        v = ref[1][ref[2][:, j], j]
        st["cons_sum"][j], st["cons_sumsq"][j], st["cons_n"][j] = v.sum(), (v ** 2).sum(), v.size
    return gt, ref, pred, st


def phase2(a):
    gt, ref, pred, sign, means = a
    return LS.score_shot(gt, ref, pred, R, Z, MC, MF, sign, means)


def pool_stats(sts):
    return {k: sum(s[k] for s in sts) for k in sts[0]}


def finalize(parts, sts, sign):
    acc = Accum(); acc.psi_sign = sign
    for p in parts:
        acc.add(p)
    return finalize_machine(acc, pool_stats(sts))


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--data-dir", required=True)
    ap.add_argument("--calib", default=None); ap.add_argument("--f32", action="store_true")
    ap.add_argument("--self-calib", action="store_true", help="fit C_q95 / C_betaN on THESE frames (use on train subsets only)")
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--procs", type=int, default=32)
    A = ap.parse_args()
    cal = json.load(open(A.calib)) if A.calib else {}
    A.cq, A.cb = cal.get("C_q95", 1.0), cal.get("C_betaN", 1.0)
    if A.self_calib:
        d0 = pd.read_parquet(Path(A.run) / f"frames_{A.tag}_d3d.parquet")
        d0 = d0[d0["ok"].astype(bool)]
        v = d0[["gs_q95", "gt_q95"]].dropna(); v = v[(v.gt_q95 > 1) & (v.gt_q95 < 20) & (v.gs_q95 > 0)]
        A.cq = float(np.median(v.gt_q95 / v.gs_q95))
        v = d0[["gs_betaN", "gt_betaN"]].dropna(); v = v[(v.gt_betaN > 0.05) & (v.gt_betaN < 8) & (v.gs_betaN > 0)]
        A.cb = float(np.median(v.gt_betaN / v.gs_betaN))
        zz = d0[["fz0", "fz1", "gs_Zax", "gt_zaxis", "zt"]].dropna()
        zs = (zz.gs_Zax - zz.fz0 / zz.fz1).clip(-0.3, 0.3)
        json.dump({"C_q95": A.cq, "C_betaN": A.cb, "n_ok": int(len(d0)), "source": f"{A.run}/frames_{A.tag}_d3d.parquet",
                   "z_shrink_ls": float(np.clip((zs * zz.gt_zaxis).sum() / (zs ** 2).sum(), 0, 1)),
                   "z_corr": float(np.corrcoef(zs, zz.gt_zaxis)[0, 1])}, open(Path(A.run) / f"calib_{A.tag}.json", "w"), indent=1)
    names = [x for x in (Path(A.run) / f"files_{A.tag}_d3d.txt").read_text().split("\n") if x]
    with Pool(A.procs) as pool:
        p1 = pool.map(phase1, list(enumerate(names)), chunksize=2)
        sts = [x[3] for x in p1]
        tot = pool_stats(sts)
        mean_psi = tot["psi_sum"] / tot["psi_n"]
        means = (mean_psi, np.where(tot["scal_n"] > 0, tot["scal_sum"] / np.maximum(tot["scal_n"], 1), 0.0),
                 np.where(tot["cons_n"] > 0, tot["cons_sum"] / np.maximum(tot["cons_n"], 1), 0.0))
        totals = {sg: 0.0 for sg in PSI_SIGNS}
        for gt, ref, pred, _ in p1:
            rr, _, _ = LS.psi_residuals(gt["psi"], pred["psirz"], mean_psi)
            for sg in PSI_SIGNS:
                totals[sg] += rr[sg]
        sign = min(PSI_SIGNS, key=lambda sg: totals[sg])
        parts = pool.map(phase2, [(gt, ref, pred, sign, means) for gt, ref, pred, _ in p1], chunksize=2)
    res = finalize(parts, sts, sign)
    rng = np.random.default_rng(0)
    keys = ("S", "r2_psi", "r2_qb", "dlcfs", "consistency")
    bs = {k: [] for k in keys}
    for _ in range(A.boot):
        ii = rng.integers(0, len(parts), len(parts))
        r = finalize([parts[i] for i in ii], [sts[i] for i in ii], sign)
        for k in keys:
            bs[k].append(r[k])
    sd = {k: float(np.std(v)) for k, v in bs.items()}
    out = {"tag": A.tag, "n_shots": len(names), "n_frames": int(sum(len(x[0]["q95"]) for x in p1)), "f32": A.f32,
           "calib": {"C_q95": A.cq, "C_betaN": A.cb}, "score": res, "boot_sd": sd}
    fr = Path(A.run) / f"frames_{A.tag}_d3d.parquet"
    if fr.exists():                                   # gauge-vs-shape decomposition over the same pooled SST
        d = pd.read_parquet(fr)
        d = d[d.get("npx").notna()] if "npx" in d else d.iloc[0:0]
        if len(d):
            N = d["npx"].sum(); sst = d["gt_s2"].sum() - d["gt_s1"].sum() ** 2 / N
            P = d["npx"]
            dec = {}
            for lab, m, sse in (("vacuum_only", "dv_mean", "dv_sse"), ("full", "d_mean", "d_sse")):
                gauge = float((P * d[m] ** 2).sum()); tot_ = float(d[sse].sum())
                dec[lab] = {"sse_over_sst": tot_ / sst, "gauge_part": gauge / sst, "shape_part": (tot_ - gauge) / sst,
                            "r2": 1 - tot_ / sst, "r2_if_gauge_perfect": 1 - (tot_ - gauge) / sst}
            pl = d[d["err"] != "Ip<min"]
            out["gauge_decomposition"] = dec
            out["converged"] = float(pl["ok"].astype(bool).mean())
    json.dump(out, open(Path(A.run) / f"eval_{A.tag}{'_f32' if A.f32 else ''}.json", "w"), indent=1, default=float)
    c = res["r2_cons_each"]; e = res["r2_each"]
    print(f"EVAL {A.tag}{' f32' if A.f32 else ''}: S {res['S']:.4f}+-{sd['S']:.4f} | R2psi {res['r2_psi']:.4f}+-{sd['r2_psi']:.4f} "
          f"| qb {res['r2_qb']:.3f}+-{sd['r2_qb']:.3f} (q95 {e['q95']:.3f} betaN {e['betaN']:.3f}) | 1-D {1 - min(1, res['dlcfs']):.4f}+-{sd['dlcfs']:.4f} "
          f"| Cons {res['consistency']:.3f}+-{sd['consistency']:.3f} [" + " ".join(f"{k} {('%.3f' % v) if v is not None else 'n/a'}" for k, v in c.items())
          + f"] | lcfs_fail {res['lcfs_fail_frac']:.3f} conv {out.get('converged', float('nan')):.3f} | {out['n_shots']} shots {out['n_frames']} frames")
    if "gauge_decomposition" in out:
        for lab, v in out["gauge_decomposition"].items():
            print(f"   gauge/shape [{lab}]: SSE/SST {v['sse_over_sst']:.5f} = gauge {v['gauge_part']:.5f} + shape {v['shape_part']:.5f}  (R2 {v['r2']:.4f}; gauge-perfect R2 {v['r2_if_gauge_perfect']:.4f})")


if __name__ == "__main__":
    main()
