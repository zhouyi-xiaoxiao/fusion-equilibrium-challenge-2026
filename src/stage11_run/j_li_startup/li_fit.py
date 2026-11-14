#!/usr/bin/env python3
"""
Stage 11J / li_startup — li_fit.py: ridge model li = f(dimensionless ramp history, Thomson exponents) fitted on DIII-D TRAIN
labels against the SCORER-derived li of the label maps (table of li_table.py), shot-grouped CV, support statistics, and the
label-free transfer report on the MAST feature table (support, predicted range, tau_R bracket).

Usage: li_fit.py --train T.parquet [--check C.parquet] [--mast M.parquet] --out model.json --report report.json
No MAST label is read: the MAST table holds input-derived features only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import li_core as LI  # noqa: E402

LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def r2(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def ridge(Z, y, lam):
    n, p = Z.shape
    b = float(y.mean())
    w = np.linalg.solve(Z.T @ Z + lam * n * np.eye(p), Z.T @ (y - b))
    return w, b


def folds_of(files, k=5):
    u = np.array(sorted(set(files)))
    rng = np.random.default_rng(20261012)
    perm = rng.permutation(len(u))
    f = {u[j]: i % k for i, j in enumerate(perm)}
    return np.array([f[x] for x in files])


def prep(df, feats):
    X = df[feats].to_numpy(float)
    lo, hi = np.nanpercentile(X, 0.5, axis=0), np.nanpercentile(X, 99.5, axis=0)
    Xc = np.clip(X, lo, hi)
    mu, sd = Xc.mean(0), Xc.std(0) + 1e-9
    return X, lo, hi, mu, sd


def cv_fit(df, feats, target, lam=None, quad=False):
    y = df[target].to_numpy(float)
    files = df["file"].to_numpy()
    fo = folds_of(files)
    X, lo, hi, mu, sd = prep(df, feats)
    best = None
    for lm in ([lam] if lam is not None else LAMS):
        p = np.full(len(y), np.nan)
        for i in range(5):
            tr, te = fo != i, fo == i
            lo_i, hi_i = np.nanpercentile(X[tr], 0.5, axis=0), np.nanpercentile(X[tr], 99.5, axis=0)
            Xi = np.clip(X, lo_i, hi_i)
            mu_i, sd_i = Xi[tr].mean(0), Xi[tr].std(0) + 1e-9
            Zi = LI.expand((Xi - mu_i) / sd_i, quad)
            w, b = ridge(Zi[tr], y[tr], lm)
            p[te] = b + Zi[te] @ w
        sc = r2(y, p)
        if best is None or sc > best[0]:
            best = (sc, lm, p)
    sc, lm, p = best
    Z = LI.expand((np.clip(X, lo, hi) - mu) / sd, quad)
    w, b = ridge(Z, y, lm)
    res = y - p
    g = pd.DataFrame({"f": files, "y": y, "p": p})
    ym, pm = g.groupby("f")["y"].transform("mean").to_numpy(), g.groupby("f")["p"].transform("mean").to_numpy()
    within = float(1 - (((y - ym) - (p - pm)) ** 2).sum() / ((y - ym) ** 2).sum())
    between = r2(ym, pm)
    Zr = (X - mu) / sd
    S = np.cov(Zr.T) + 1e-3 * np.eye(len(feats))
    Sinv = np.linalg.inv(S)
    d = np.sqrt(np.einsum("ij,jk,ik->i", Zr, Sinv, Zr))
    out = {"feats": list(feats), "lo": lo.tolist(), "hi": hi.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "coef": w.tolist(), "b": b,
           "lam": lm, "sig": float(np.sqrt(np.mean(res ** 2))), "cv_r2": sc, "cv_r2_within": within, "cv_r2_between": between,
           "quad": bool(quad), "Sinv": Sinv.tolist(), "d95": float(np.percentile(d, 95)), "d99": float(np.percentile(d, 99)), "n": int(len(y)),
           "n_shots": int(len(set(files)))}
    return out, p, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--check", default=None)
    ap.add_argument("--mast", default=None)
    ap.add_argument("--mast-alt", action="append", default=[], help="label=path of a MAST table built with another tau scale")
    ap.add_argument("--target", default="gt_li")
    ap.add_argument("--cdef", type=float, default=1.044)
    ap.add_argument("--quad", action="store_true", help="ridge on the features + their pairwise products (clipped at +-3.5 sd)")
    ap.add_argument("--name", default="li_model_v1")
    ap.add_argument("--lq", action="store_true", help="include the cylindrical-q feature lq")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    a = ap.parse_args()
    df = pd.read_parquet(a.train)
    if "fatal" in df:
        df = df[df["fatal"].isna()]
    FR = LI.FEATS_RAMP if a.lq else LI.FEATS_RAMP0
    FF = FR + LI.FEATS_TS
    df = df[np.isfinite(df[a.target]) & np.isfinite(df[FR]).all(axis=1)].reset_index(drop=True)
    rep = {"train_rows": int(len(df)), "train_shots": int(df["file"].nunique()), "target": a.target}
    y_all = df.loc[df.cond == "clean", a.target].to_numpy(float)
    rep["target_stats"] = {"mean": float(y_all.mean()), "sd": float(y_all.std()), "p1_5_50_95_99": np.percentile(y_all, [1, 5, 50, 95, 99]).round(4).tolist()}
    if "efit_li" in df:
        cl = df[df.cond == "clean"]
        m = np.isfinite(cl["efit_li"])
        rep["scorer_li_vs_efit_li"] = {"corr": float(np.corrcoef(cl.loc[m, a.target], cl.loc[m, "efit_li"])[0, 1]),
                                       "ratio_median": float(np.median(cl.loc[m, a.target] / cl.loc[m, "efit_li"]))}
    full_df = df[np.isfinite(df[LI.FEATS_TS]).all(axis=1)].reset_index(drop=True)
    ramp_df = df[df.cond == "clean"].reset_index(drop=True)
    rep["exponent_availability"] = {c: float(np.isfinite(df.loc[df.cond == c, LI.FEATS_TS]).all(axis=1).mean()) for c in ("clean", "stress")}
    full, p_full, d_full = cv_fit(full_df, FF, a.target, quad=a.quad)
    ramp, p_ramp, d_ramp = cv_fit(ramp_df, FR, a.target, quad=a.quad)
    # diagnostics: feature-subset CV R2 (clean rows), also against the EFIT li label
    sub = {}
    cl = full_df[full_df.cond == "clean"].reset_index(drop=True)
    subsets = [("ramp", FR), ("exponents", LI.FEATS_TS), ("full", FF), ("ramp_no_tau(lfrac only)", ["lfrac"]), ("r0+m", ["r0"] + [f"m{c:g}" for c in LI.RAMP_C])]
    if a.lq:
        subsets += [("full without lq", LI.FEATS_RAMP0 + LI.FEATS_TS), ("lq only", ["lq"]), ("lq + exponents", ["lq"] + LI.FEATS_TS)]
    for nm, fs in subsets:
        for tg in (a.target, "efit_li"):
            if tg not in cl:
                continue
            c2 = cl[np.isfinite(cl[tg])].reset_index(drop=True)
            o, _, _ = cv_fit(c2, fs, tg, quad=a.quad)
            sub[f"{nm}|{tg}"] = {"cv_r2": round(o["cv_r2"], 4), "within": round(o["cv_r2_within"], 4), "between": round(o["cv_r2_between"], 4), "lam": o["lam"]}
    rep["subset_cv_clean"] = sub
    for cond in ("clean", "stress"):
        m = (full_df.cond == cond).to_numpy()
        rep[f"full_cv_r2_{cond}"] = r2(full_df.loc[m, a.target], p_full[m])
    # phase split of the CV error (full model, clean rows)
    m = (full_df.cond == "clean").to_numpy()
    ph = np.where(full_df["r0"].to_numpy() > 0.02, "ramp_up", np.where(full_df["r0"].to_numpy() < -0.02, "ramp_down", "flat"))
    rep["full_cv_by_phase"] = {p_: {"n": int((m & (ph == p_)).sum()), "label_mean": float(full_df.loc[m & (ph == p_), a.target].mean()),
                                    "label_sd": float(full_df.loc[m & (ph == p_), a.target].std()),
                                    "rmse": float(np.sqrt(np.mean((full_df.loc[m & (ph == p_), a.target].to_numpy() - p_full[m & (ph == p_)]) ** 2)))}
                               for p_ in ("ramp_up", "flat", "ramp_down") if (m & (ph == p_)).sum() > 20}
    # residual sd vs support distance (calibration of the inflation rule)
    res = full_df[a.target].to_numpy() - p_full
    qs = np.percentile(d_full, [0, 50, 80, 90, 95, 99, 100])
    rep["full_resid_sd_by_d"] = [{"d_lo": float(qs[i]), "d_hi": float(qs[i + 1]), "sd": float(np.sqrt(np.mean(res[(d_full >= qs[i]) & (d_full <= qs[i + 1])] ** 2)))}
                                 for i in range(len(qs) - 1)]
    model = {"version": a.name, "quad": bool(a.quad), "target": a.target, "full": full, "ramp": ramp, "cdef": a.cdef,
             "li_lo": float(np.percentile(y_all, 1)), "li_hi": float(np.percentile(y_all, 99)),
             "ramp_c": list(LI.RAMP_C), "age_c": list(LI.AGE_C), "train_table": str(a.train)}
    json.dump(model, open(a.out, "w"), indent=1)
    rep["full"] = {k: v for k, v in full.items() if k not in ("Sinv",)}
    rep["ramp"] = {k: v for k, v in ramp.items() if k not in ("Sinv",)}
    M = LI.LiModel(a.out)

    def apply(tab, use="auto"):
        rows = []
        for r in tab.to_dict("records"):
            pr = M.predict(r, use)
            rows.append(pr if pr is not None else {"li": np.nan, "li_raw": np.nan, "sig": np.nan, "d": np.nan, "insup": False, "nclip": -1, "model": "none", "infl": np.nan})
        return pd.DataFrame(rows)

    if a.check:
        ck = pd.read_parquet(a.check)
        if "fatal" in ck:
            ck = ck[ck["fatal"].isna()]
        ck = ck[np.isfinite(ck[a.target])].reset_index(drop=True)
        out = {}
        for cond in ("clean", "stress"):
            c2 = ck[ck.cond == cond].reset_index(drop=True)
            for use in ("auto", "ramp"):
                pr = apply(c2, use)
                m = np.isfinite(pr["li"]).to_numpy()
                y, p = c2.loc[m, a.target].to_numpy(), pr.loc[m, "li"].to_numpy()
                g = pd.DataFrame({"f": c2.loc[m, "file"].to_numpy(), "y": y, "p": p})
                ym, pm = g.groupby("f")["y"].transform("mean").to_numpy(), g.groupby("f")["p"].transform("mean").to_numpy()
                z = (y - p) / pr.loc[m, "sig"].to_numpy()
                out[f"{cond}|{use}"] = {"n": int(m.sum()), "shots": int(c2["file"].nunique()), "r2": r2(y, p), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
                                        "within": float(1 - (((y - ym) - (p - pm)) ** 2).sum() / ((y - ym) ** 2).sum()), "between": r2(ym, pm),
                                        "corr": float(np.corrcoef(y, p)[0, 1]), "r_sd": float(p.std() / y.std()), "bias": float((p - y).mean()),
                                        "insup": float(pr.loc[m, "insup"].mean()), "full_frac": float((pr.loc[m, "model"] == "full").mean()),
                                        "z_sd": float(z.std()), "z_abs_gt2": float((np.abs(z) > 2).mean())}
        rep["check_set"] = out
    if a.mast:
        def mast_report(path):
            mt = pd.read_parquet(path)
            if "fatal" in mt:
                mt = mt[mt["fatal"].isna()]
            mt = mt.reset_index(drop=True)
            pr = apply(mt)
            m = np.isfinite(pr["li"]).to_numpy()
            o = {"frames": int(len(mt)), "shots": int(mt["file"].nunique()), "pred_frac": float(m.mean()),
                 "full_model_frac": float((pr["model"] == "full").mean()), "insup_d99_frac": float(pr.loc[m, "insup"].mean()),
                 "d_le_d95_frac": float((pr.loc[m, "infl"] <= 1.0).mean()),
                 "li_raw_in_d3d_range_frac": float(((pr.loc[m, "li_raw"] >= M.li_lo) & (pr.loc[m, "li_raw"] <= M.li_hi)).mean()),
                 "li_pred_p1_5_25_50_75_95_99": np.nanpercentile(pr.loc[m, "li"], [1, 5, 25, 50, 75, 95, 99]).round(4).tolist(),
                 "li_raw_p1_5_50_95_99": np.nanpercentile(pr.loc[m, "li_raw"], [1, 5, 50, 95, 99]).round(4).tolist(),
                 "sig_p50_95": np.nanpercentile(pr.loc[m, "sig"], [50, 95]).round(4).tolist(),
                 "nclip_ge1_frac": float((pr.loc[m, "nclip"] >= 1).mean()),
                 "tau_ms_p5_50_95": np.nanpercentile(mt["tau_ms"], [5, 50, 95]).round(1).tolist(),
                 "theta_end_p5_50_95": np.nanpercentile(mt.groupby("file")["theta"].max(), [5, 50, 95]).round(3).tolist(),
                 "te_keV_p5_50_95": np.nanpercentile(mt["te_keV"], [5, 50, 95]).round(3).tolist()}
            fq = (mt["frac_t"].to_numpy() * 5).clip(0, 4.999).astype(int)
            o["li_pred_median_by_shot_fifth"] = [round(float(np.nanmedian(pr.loc[m & (fq == i), "li"])), 4) for i in range(5)]
            o["insup_by_shot_fifth"] = [round(float(pr.loc[m & (fq == i), "insup"].mean()), 4) for i in range(5)]
            ff = {}
            s = M.sub["full"]
            for j, f in enumerate(s["feats"]):
                v = mt[f].to_numpy(float); v = v[np.isfinite(v)]
                ff[f] = {"mast_p1_50_99": np.percentile(v, [1, 50, 99]).round(4).tolist(), "d3d_lo_hi": [round(float(s["lo"][j]), 4), round(float(s["hi"][j]), 4)],
                         "d3d_mu_sd": [round(float(s["mu"][j]), 4), round(float(s["sd"][j]), 4)],
                         "mast_outside_frac": float(((v < s["lo"][j]) | (v > s["hi"][j])).mean()), "coef": round(float(s["coef"][j]), 4)}
            o["features"] = ff
            return o, pr, mt
        o, pr, mt = mast_report(a.mast)
        rep["mast"] = o
        pd.concat([mt[["file", "k", "t", "frac_t"]], pr[["li", "li_raw", "sig", "d", "insup", "nclip", "model"]]], axis=1).to_parquet(Path(a.out).with_suffix(".mast_pred.parquet"))
        for la in a.mast_alt:
            lab_, pth = la.split("=", 1)
            o2, pr2, mt2 = mast_report(pth)
            j = mt[["file", "k"]].assign(li0=pr["li"].to_numpy()).merge(mt2[["file", "k"]].assign(li1=pr2["li"].to_numpy()), on=["file", "k"])
            dd = (j["li1"] - j["li0"]).to_numpy(); dd = dd[np.isfinite(dd)]
            rep[f"mast_tau_{lab_}"] = {"insup_d99_frac": o2["insup_d99_frac"], "li_pred_p5_50_95": o2["li_pred_p1_5_25_50_75_95_99"][1::2][:3],
                                       "delta_li_vs_base_mean_sd_p95abs": [float(dd.mean()), float(dd.std()), float(np.percentile(np.abs(dd), 95))]}
    json.dump(rep, open(a.report, "w"), indent=1)
    print("LIFIT " + json.dumps({k: rep[k] for k in rep if k not in ("mast", "full", "ramp")})[:6000])
    if "mast" in rep:
        print("LIMAST " + json.dumps(rep["mast"])[:6000])


if __name__ == "__main__":
    main()
