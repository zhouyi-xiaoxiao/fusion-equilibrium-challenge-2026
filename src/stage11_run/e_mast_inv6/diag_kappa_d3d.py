#!/usr/bin/env python3
"""DIII-D (labels allowed) diagnostics of inverse runs: kappa error vs gamma_shape / FF' shape / c_p, and the q95 gap
decomposition (is a q95 factor on THIS pipeline anything but the kappa deficit?). Usage: diag_kappa_d3d.py <dir> <tag> [<tag> ...]"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

out = Path(sys.argv[1])


def pct(x, q=(5, 25, 50, 75, 95)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [round(float(v), 4) for v in np.percentile(x, q)] if len(x) else []


def r2(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()) if len(y) > 5 else np.nan


res = {}
for tag in sys.argv[2:]:
    p = out / f"frames_{tag}_d3d.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    d = d[d["ok"].astype(bool) & (d["err"] != "Ip<min")].copy()
    if "gt_ztop" not in d:
        continue
    d["kap_gt"] = (d["gt_ztop"] - d["gt_zbot"]) / (d["gt_rout"] - d["gt_rin"])
    d["a_gt"] = 0.5 * (d["gt_rout"] - d["gt_rin"]); d["R_gt"] = 0.5 * (d["gt_rout"] + d["gt_rin"])
    d["dk"] = d["gs_kappa"] - d["kap_gt"]
    r = {"n": int(len(d)), "dk_p": pct(d["dk"]), "kappa_r2": r2(d["kap_gt"].values, d["gs_kappa"].values),
         "kappa_gs_p": pct(d["gs_kappa"]), "kappa_gt_p": pct(d["kap_gt"])}
    for c in ("gs", "b1", "b2", "cp", "gev", "dr", "n_frozen", "iso_mae_cm"):
        if c in d and d[c].notna().sum() > 10:
            v = d[[c, "dk"]].replace([np.inf, -np.inf], np.nan).dropna()
            r[f"corr_dk_{c}"] = float(np.corrcoef(v[c], v["dk"])[0, 1])
    if "gs" in d and d["gs"].notna().sum() > 10:
        qs = d["gs"].quantile([0, .1, .25, .5, .75, .9, 1.0]).values
        r["dk_by_gs_bin"] = {f"{qs[j]:.3f}..{qs[j + 1]:.3f}": [round(float(d[(d.gs >= qs[j]) & (d.gs <= qs[j + 1])]["dk"].median()), 4),
                                                              round(float((d[(d.gs >= qs[j]) & (d.gs <= qs[j + 1])]["gs_kappa"] > d[(d.gs >= qs[j]) & (d.gs <= qs[j + 1])]["kap_gt"] + 0.2).mean()), 4)]
                             for j in range(len(qs) - 1)}
        hi = d[d["gs_kappa"] > d["kap_gt"].quantile(0.99)]
        r["frames_kappa_above_gt_p99"] = {"n": int(len(hi)), "frac": float(len(hi) / len(d)), "gs_p": pct(hi["gs"]), "dk_p": pct(hi["dk"])}
    for bc in ("b1", "b2", "gs_li"):
        if bc in d and d[bc].notna().sum() > 50:
            qs = d[bc].quantile([0, .1, .25, .5, .75, .9, 1.0]).values
            r[f"dk_by_{bc}_bin"] = {f"{qs[j]:.2f}..{qs[j + 1]:.2f}": [round(float(d[(d[bc] >= qs[j]) & (d[bc] <= qs[j + 1])]["dk"].median()), 4),
                                                                   round(float(d[(d[bc] >= qs[j]) & (d[bc] <= qs[j + 1])]["gs_kappa"].median()), 4)]
                                    for j in range(len(qs) - 1)}
    v = d[["gs_kappa", "kap_gt"]].dropna()
    r["kappa_r2_decomp"] = {"rho": float(np.corrcoef(v.gs_kappa, v.kap_gt)[0, 1]), "sd_ratio": float(v.gs_kappa.std() / v.kap_gt.std()),
                            "bias_over_sd": float((v.gs_kappa.mean() - v.kap_gt.mean()) / v.kap_gt.std()),
                            "frac_pred_gt_true_p99": float((v.gs_kappa > v.kap_gt.quantile(0.99)).mean())}
    # q95: label / ours and the attribution regression (as the scal2 analysis did for G1)
    v = d[["gs_q95", "gt_q95", "gs_kappa", "kap_gt", "gs_a", "a_gt", "gs_Rgeo", "R_gt"]].replace([np.inf, -np.inf], np.nan).dropna()
    v = v[(v.gt_q95 > 1) & (v.gt_q95 < 20) & (v.gs_q95 > 0)]
    if len(v) > 20:
        y = np.log(v.gt_q95 / v.gs_q95)
        X = np.column_stack([np.ones(len(v)), v.kap_gt - v.gs_kappa, np.log(v.a_gt / v.gs_a), np.log(v.R_gt / v.gs_Rgeo)])
        co, *_ = np.linalg.lstsq(X, y, rcond=None)
        pred = X @ co
        r["q95"] = {"median_label_over_ours": float(np.exp(np.median(y))), "p10_p90": [float(np.exp(np.percentile(y, 10))), float(np.exp(np.percentile(y, 90)))],
                    "r2_raw": r2(v.gt_q95.values, v.gs_q95.values), "r2_median_scaled": r2(v.gt_q95.values, v.gs_q95.values * np.exp(np.median(y))),
                    "attribution_coef[c, dkappa, dlog_a, dlog_R]": [float(x) for x in co], "attribution_r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                    "explained_by_kappa_term_median": float(np.median(co[1] * (v.kap_gt - v.gs_kappa))),
                    "median_dkappa": float(np.median(v.kap_gt - v.gs_kappa)), "median_dlog_a": float(np.median(np.log(v.a_gt / v.gs_a))),
                    "median_dlog_R": float(np.median(np.log(v.R_gt / v.gs_Rgeo)))}
        # kappa-only correction: if we knew kappa exactly, what would the residual factor be?
        yk = y - co[1] * (v.kap_gt - v.gs_kappa)
        r["q95"]["median_factor_after_kappa_term"] = float(np.exp(np.median(yk)))
    vb = d[["gs_betaN", "gt_betaN"]].replace([np.inf, -np.inf], np.nan).dropna()
    vb = vb[(vb.gt_betaN > 0.05) & (vb.gs_betaN > 0)]
    if len(vb) > 20:
        r["betaN"] = {"median_label_over_ours": float(np.median(vb.gt_betaN / vb.gs_betaN)), "r2_raw": r2(vb.gt_betaN.values, vb.gs_betaN.values),
                      "r2_median_scaled": r2(vb.gt_betaN.values, vb.gs_betaN.values * np.median(vb.gt_betaN / vb.gs_betaN))}
    res[tag] = r
print(json.dumps(res, indent=1))
import os
json.dump(res, open(Path(os.environ.get("OUT6", ".")) / "dev" / ("diag_kappa_" + "_".join(sys.argv[2:]) + ".json"), "w"), indent=1)
