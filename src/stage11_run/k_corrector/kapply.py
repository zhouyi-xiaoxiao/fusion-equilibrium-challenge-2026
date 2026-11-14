#!/usr/bin/env python3
"""Stage 11K / Ch2 corrector — apply the control-point corrector to a parts run and write corrected parts (same layout).
  --mode pred    corrections = model prediction x gate x alpha (optionally debiased: zero corpus mean)
  --mode oracle  corrections = LABEL displacements (DIII-D only; upper bound of the map-space realisation, never shipped)
MAST: rule3_guard installed; inputs = our own parts + own table (ktab.py) + the public wall polygon.  No label is read.
Usage: kapply.py --machine M --run DIR --tag T --table TAB.parquet --model model.pkl --out DIR2 --newtag T2 [...]"""
import argparse, json, os, pickle, shutil, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
import kcommon as KC
import ktab as KT
import kwarp as KW

A = None
JOB = {}


def gate_weight(model, X, hard=None):
    g = model["gate"]
    Zs = np.clip(np.nan_to_num((X.to_numpy(float) - g["med"]) / g["sc"]), -8, 8)
    d2 = np.einsum("ij,jk,ik->i", Zs, g["icov"], Zs)
    q99, q999 = g["d2_q"][0.99], g["d2_q"][0.999]
    w = np.clip((2.0 * q999 - d2) / (2.0 * q999 - q99), 0.0, 1.0)          # 1 inside the DIII-D 99 % ellipsoid, 0 beyond twice the 99.9 % radius^2
    xn = X.to_numpy(float)
    outside = ((xn < g["lo"]) | (xn > g["hi"])) & np.isfinite(xn)
    return w, d2, outside


def one(job):
    i, name = job
    src = Path(A.run) / "parts" / f"{A.tag}_{A.machine}_{i:05d}.npz"
    dst = Path(A.out) / "parts" / f"{A.newtag}_{A.machine}_{i:05d}.npz"
    if not src.exists():
        return 0
    c = dict(np.load(src, allow_pickle=False))
    psi = c["psi"].copy()
    rows = JOB.get(name)
    n = 0
    if rows is not None:
        R, Z, sign = KT.G["R"], KT.G["Z"], KT.G["sign"]
        for r in rows:
            k = int(r["k"])
            p = psi[k].astype(np.float32).astype(np.float64)
            P = np.array([[r["w_Ra"], r["w_Za"]], [r["w_Rt"], r["w_Zt"]], [r["w_Rb"], r["w_Zb"]], [r["w_Ri"], r["w_Zi"]], [r["w_Ro"], r["w_Zo"]]])
            dd = np.array([[r["dRa"], r["dZa"]], [r["dRt"], r["dZt"]], [r["dRb"], r["dZb"]], [r["dRi"], 0.0], [r["dRo"], 0.0]]) * r["a"]
            if not (np.isfinite(P).all() and np.isfinite(dd).all()):
                continue
            q = KW.warp(p, R, Z, P, dd, r["a"], r["b"], s=A.s, margin=A.margin)
            if A.li and np.isfinite(r["dlli"]) and abs(r["dlli"]) > 1e-4:
                o, C = KT.geom(q, KT.G["WM"], sign)
                if C is not None and np.isfinite(o["li"]) and np.isfinite(r["w_li"]):
                    q = KW.li_remap(q, R, Z, C, sign, float(r["w_li"] * np.exp(r["dlli"]) / o["li"]), KT.G["derive"])
            psi[k] = q.astype(psi.dtype)
            n += 1
    c["psi"] = psi
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **c)
    os.replace(tmp, dst)
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["d3d", "mast"])
    ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--table", required=True)
    ap.add_argument("--model", default=None); ap.add_argument("--out", required=True); ap.add_argument("--newtag", required=True)
    ap.add_argument("--wall", required=True)
    ap.add_argument("--mode", default="pred", choices=["pred", "oracle"])
    ap.add_argument("--outs", default="dRa,dZa,dRt,dZt,dRb,dZb,dRi,dRo"); ap.add_argument("--li", action="store_true")
    ap.add_argument("--alphas", default=None, help="per-output strengths, e.g. dRa=1,dZa=1,dlli=1,dRt=0.5 (multiplies --alpha)")
    ap.add_argument("--alpha", type=float, default=1.0); ap.add_argument("--debias", action="store_true"); ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--ref", default="auto", help="auto: d3d -> the model's DIII-D train medians, mast -> this corpus' own medians (label-free)")
    ap.add_argument("--s", type=float, default=0.45); ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("NPROC", "8")))
    A = ap.parse_args()
    KT.A = argparse.Namespace(machine=A.machine)
    KT.setup(A.machine, A.wall)
    d = pd.read_parquet(A.table)
    model = pickle.load(open(A.model, "rb")) if A.model else None
    ref = model["ref_d3d"] if (model and A.machine == "d3d" and A.ref == "auto") else KC.corpus_ref(d)
    F, B = KC.feats(d, ref)
    outs = A.outs.split(",") + (["dlli"] if A.li else [])
    Y = pd.DataFrame(0.0, index=d.index, columns=KC.OUTS)
    info = {"machine": A.machine, "mode": A.mode, "outs": outs, "alpha": A.alpha, "debias": A.debias, "ref": ref, "n": int(len(d))}
    valid = np.isfinite(d[[f"w_{p}" for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro")]].to_numpy(float)).all(1) & (B["a"] > 0.1) & (B["b"] > 0.1)
    if A.mode == "oracle":
        T = KC.targets(d, B)
        for o in outs:
            Y[o] = T[o].clip(-0.6, 0.6).fillna(0.0)
        gw = np.ones(len(d))
    else:
        X = F[model["cols"]]
        for o in outs:
            Y[o] = model["models"][o].predict(X)
        gw, d2, outside = gate_weight(model, X)
        info["gate"] = {"mean": float(gw[valid].mean()), "frac_full": float((gw[valid] >= 0.999).mean()), "frac_zero": float((gw[valid] <= 0.001).mean()),
                        "q": {str(q): float(np.quantile(gw[valid], q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75)},
                        "d2_q": {str(q): float(np.quantile(d2[valid], q)) for q in (0.05, 0.5, 0.95, 0.99)}, "d2_train_q": model["gate"]["d2_q"],
                        "feat_outside_frac": {c: float(outside[valid][:, j].mean()) for j, c in enumerate(model["cols"]) if outside[valid][:, j].mean() > 0.02}}
        if A.no_gate:
            gw = np.ones(len(d))
        if A.debias:
            Y = Y - Y[valid].mean()
            info["removed_mean"] = "corpus mean of the predicted corrections"
        Y = Y.mul(A.alpha * gw, axis=0)
        if A.alphas:
            al = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in A.alphas.split(",")}
            for o in KC.OUTS:
                Y[o] = Y[o] * al.get(o, 1.0)
            info["alphas"] = al
        d["gate"], d["d2"] = gw, d2
    Y[~valid] = 0.0
    info["corr_abs_med_cm"] = {o: float(np.median(np.abs(Y[o][valid] * B["a"][valid])) * 100) for o in outs if o != "dlli"}
    info["corr_mean_cm"] = {o: float(np.mean(Y[o][valid] * B["a"][valid]) * 100) for o in outs if o != "dlli"}
    out = Path(A.out); out.mkdir(parents=True, exist_ok=True)
    keepc = ["file", "idx", "k"] + [c for c in ("gate", "d2") if c in d]
    pd.concat([d[keepc], Y], axis=1).to_parquet(out / f"pred_{A.newtag}_{A.machine}.parquet")
    if A.machine == "d3d" and "g_Ra" in d:
        import kfit as KF
        info["scalar_space"] = KF.sc_report(d, B, Y, valid & np.isfinite(d["g_Ra"].to_numpy(float)), f"APPLY {A.newtag}")
    t = pd.concat([d[["file", "k", "w_Ra", "w_Za", "w_Rt", "w_Zt", "w_Rb", "w_Zb", "w_Ri", "w_Zi", "w_Ro", "w_Zo", "w_li"]], Y], axis=1)
    t["a"], t["b"] = B["a"], B["b"]
    t = t[valid & (Y[outs].abs().sum(1) > 0)]
    JOB.update({n: g.to_dict("records") for n, g in t.groupby("file")})
    names = [x for x in (Path(A.run) / f"files_{A.tag}_{A.machine}.txt").read_text().split("\n") if x]
    (out / f"files_{A.newtag}_{A.machine}.txt").write_text("\n".join(names) + "\n")
    with Pool(A.procs) as pool:
        nn = pool.map(one, list(enumerate(names)), chunksize=2)
    info["frames_changed"] = int(sum(nn))
    json.dump(info, open(out / f"apply_{A.newtag}_{A.machine}.json", "w"), indent=1, default=float)
    print("KAPPLY " + json.dumps({k: v for k, v in info.items() if k not in ("ref", "scalar_space")}, default=float)[:3000])
