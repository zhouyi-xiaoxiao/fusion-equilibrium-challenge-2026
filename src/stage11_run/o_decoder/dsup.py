#!/usr/bin/env python3
"""Stage 11o decoder — SUPPORT gate of the source (DIII-D) weights (external review, round 3 §E2): support = 1 below the source leave-one-shot-out 95th
percentile of the whitened 32-NN distance, 0 above the 99.5th, linear in between.
Feature space (dimensionless, own maps only => label-free on MAST): net-minus-inverse disagreement of the seven functionals
(axis in units of the minor radius, kappa, triangularities, log volume ratio, log li ratio) + mean kappa, triangularities, log li.
Absolute size / aspect ratio are NOT features (the machines differ by construction; cf. the corrector track's transfer set).
Usage: dsup.py --source est1.parquet [est2.parquet ...] --query name=est.parquet ... --out DIR"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

K = 32


def feats(d):
    a = 0.5 * (d["I_Ro"] - d["I_Ri"]).to_numpy(float)
    g = lambda c: d[c].to_numpy(float)  # noqa: E731
    with np.errstate(all="ignore"):
        F = np.stack([(g("N_Ra") - g("I_Ra")) / a, (g("N_Za") - g("I_Za")) / a, g("N_kappa") - g("I_kappa"), g("N_tri_top") - g("I_tri_top"),
                      g("N_tri_bot") - g("I_tri_bot"), np.log(g("N_volume") / g("I_volume")), np.log(g("N_li") / g("I_li")),
                      0.5 * (g("N_kappa") + g("I_kappa")), 0.5 * (g("N_tri_top") + g("I_tri_top")), 0.5 * (g("N_tri_bot") + g("I_tri_bot")),
                      np.log(0.5 * (g("N_li") + g("I_li")))], 1)
    return F


ap = argparse.ArgumentParser()
ap.add_argument("--source", nargs="+", required=True); ap.add_argument("--query", nargs="+", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
S = pd.concat([pd.read_parquet(f) for f in a.source], ignore_index=True)
S = S[S["ok"].astype(bool)].reset_index(drop=True)
FS = feats(S)
m = np.isfinite(FS).all(1)
S, FS = S[m].reset_index(drop=True), FS[m]
mu = FS.mean(0); C = np.cov((FS - mu).T)
ev, U = np.linalg.eigh(C)
Wh = U / np.sqrt(np.maximum(ev, 1e-9 * ev.max()))
ZS = (FS - mu) @ Wh
tree = cKDTree(ZS)
sid = pd.factorize(S["file"])[0]
dd, ii = tree.query(ZS, k=K + 30, workers=-1)
other = sid[ii] != sid[:, None]
rank = np.cumsum(other, 1)
pos = (rank >= K).argmax(1)
dloo = dd[np.arange(len(dd)), pos]
q95, q995 = np.percentile(dloo, [95, 99.5])
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
rep = {"n_source": int(len(S)), "q95": float(q95), "q995": float(q995), "features": 11, "k": K, "sets": {}}
for spec in a.query:
    nm, f = spec.split("=", 1)
    Q = pd.read_parquet(f)
    FQ = feats(Q)
    fin = np.isfinite(FQ).all(1)
    d32 = np.full(len(Q), np.inf)
    if fin.any():
        dq, _ = tree.query((FQ[fin] - mu) @ Wh, k=K, workers=-1)
        d32[fin] = dq[:, -1]
    sup = np.clip((q995 - d32) / (q995 - q95), 0.0, 1.0)
    pd.DataFrame({"file": Q["file"], "k": Q["k"], "support": sup, "d32": d32}).to_parquet(out / f"sup_{nm}.parquet")
    ok = Q["ok"].astype(bool).to_numpy()
    zq = np.abs((FQ[fin] - mu) / np.sqrt(np.diag(C)))
    rep["sets"][nm] = {"n": int(len(Q)), "support_mean": float(sup[ok].mean()), "full": float((sup[ok] >= 1).mean()), "zero": float((sup[ok] <= 0).mean()),
                       "d32_p50_p95": [float(np.percentile(d32[ok & fin], 50)), float(np.percentile(d32[ok & fin], 95))],
                       "feature_absz_p95": [float(x) for x in np.percentile(zq, 95, axis=0)]}
    print(f"DSUP {nm}: {len(Q)} frames; support mean {rep['sets'][nm]['support_mean']:.3f}, full {rep['sets'][nm]['full']:.3f}, zero {rep['sets'][nm]['zero']:.3f}; "
          f"d32 p50/p95 {rep['sets'][nm]['d32_p50_p95'][0]:.2f}/{rep['sets'][nm]['d32_p50_p95'][1]:.2f} (source LOO q95 {q95:.2f}, q99.5 {q995:.2f}); "
          f"|z| p95 per feature " + " ".join(f"{x:.1f}" for x in rep["sets"][nm]["feature_absz_p95"]), flush=True)
json.dump(rep, open(out / "support_report.json", "w"), indent=1)
