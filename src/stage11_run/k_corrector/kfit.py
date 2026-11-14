#!/usr/bin/env python3
"""Stage 11K / Ch2 corrector — fit E[label control-point displacement | our map's geometry, inverse diagnostics, inputs] on DIII-D TRAIN
tables (ktab.py), shot-grouped CV, scalar-space report (R2 = 2 rho r - r^2 - b^2 of the scorer scalars implied by the corrected
control points), leave-configuration-out transfer tests, support statistics for the gate.  DIII-D labels only; no MAST file is read.
Usage: kfit.py --train T1.parquet [T2.parquet ...] --fset core --out model.pkl [--lco] [--folds 5]"""
import argparse, json, os, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import kcommon as KC

NJ = int(os.environ.get("NPROC", os.environ.get("SLURM_CPUS_PER_TASK", "16")))
LGB = dict(objective="regression", learning_rate=0.05, n_estimators=300, num_leaves=15, min_child_samples=200, subsample=0.7, subsample_freq=1,
           colsample_bytree=0.7, reg_lambda=20.0, n_jobs=NJ, verbose=-1)
SCN = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "li"]
EXCL = set()


def load(paths):
    """tables 'path' or 'path@group'; tables of one group share the corpus medians (label-free normalisation per corpus)."""
    ds = []
    for v, p in enumerate(paths):
        p, _, g = p.partition("@")
        d = pd.read_parquet(p); d["view"] = v; d["grp"] = g or "d3d"
        if EXCL:
            d = d[~d["file"].isin(EXCL)].reset_index(drop=True)
        d["file"] = d["grp"] + ":" + d["file"].astype(str)
        ds.append(d)
    return pd.concat(ds, ignore_index=True)


def prep(d, refs, fset, clipy=0.6):
    Fs, Bs = [], []
    for g, dg in d.groupby("grp", sort=False):
        F, B = KC.feats(dg, refs[g])
        Fs.append(F); Bs.append(pd.DataFrame(B, index=dg.index))
    F = pd.concat(Fs).loc[d.index]; Bd = pd.concat(Bs).loc[d.index]
    B = {c: Bd[c].to_numpy(float) for c in Bd.columns}
    Y = KC.targets(d, B)
    need = [f"w_{p}" for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro", "li")] + [f"g_{p}" for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro", "li")]
    ok = np.isfinite(d[need].to_numpy(float)).all(1) & (B["a"] > 0.1) & (B["b"] > 0.1)
    Yc = Y.clip(-clipy, clipy)
    return F[KC.FSETS[fset]], B, Y, Yc, ok


def fit_models(X, Y, w):
    return {o: lgb.LGBMRegressor(**LGB).fit(X, Y[o], sample_weight=w) for o in KC.OUTS}


def predict(models, X):
    return pd.DataFrame({o: models[o].predict(X) for o in KC.OUTS}, index=X.index)


def sc_report(d, B, P, m, title):
    """scalar-space R2 decomposition of base (w), official extraction (p) and corrected (w + P) against the label (g)."""
    lab = KC.scalars_from(d, B, None, "g_"); bas = KC.scalars_from(d, B, None, "w_"); off = KC.scalars_from(d, B, None, "p_")
    cor = KC.scalars_from(d, B, P, "w_")
    rows = {}
    for n in SCN:
        a, b_, c = KC.r2dec(lab[n][m], bas[n][m]), KC.r2dec(lab[n][m], off[n][m]), KC.r2dec(lab[n][m], cor[n][m])
        rows[n] = {"base": a, "official": b_, "corr": c}
    cb = np.mean([max(0, rows[n]["base"]["r2"]) for n in SCN]); cc = np.mean([max(0, rows[n]["corr"]["r2"]) for n in SCN])
    print(f"SC {title}: n={int(m.sum())} cons6 base {cb:.3f} -> corr {cc:.3f} | " + " ".join(
        f"{n} {rows[n]['base']['r2']:+.2f}({rows[n]['official']['r2']:+.2f})->{rows[n]['corr']['r2']:+.2f}[rho {rows[n]['base']['rho']:.2f}->{rows[n]['corr']['rho']:.2f} r {rows[n]['base']['r']:.2f}->{rows[n]['corr']['r']:.2f}]" for n in SCN), flush=True)
    return {"cons6_base": float(cb), "cons6_corr": float(cc), "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True); ap.add_argument("--fset", default="core"); ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5); ap.add_argument("--lco", action="store_true"); ap.add_argument("--max-shots", type=int, default=0)
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--exclude", default=None, help="txt file of shot file names never used for training (held-out test shots)")
    A = ap.parse_args()
    if A.exclude:
        EXCL |= {x for x in open(A.exclude).read().split("\n") if x}
    d = load(A.train)
    if A.max_shots:
        keep = sorted(d["file"].unique())[:A.max_shots]
        d = d[d["file"].isin(keep)].reset_index(drop=True)
    refs = {g: KC.corpus_ref(dg[dg["view"] == dg["view"].min()]) for g, dg in d.groupby("grp", sort=False)}
    ref = refs.get("d3d", next(iter(refs.values())))
    X, B, Y, Yc, ok = prep(d, refs, A.fset)
    k = d["k"].to_numpy(); w = np.where((k == 0) | (k == d["nT"].to_numpy() - 1), 0.5, 1.0)
    if A.balance:                                   # equal total weight per training corpus (DIII-D labels vs own synthetic equilibria)
        gcol = d["grp"].to_numpy(); ng = len(set(gcol))
        for g in set(gcol):
            w[gcol == g] *= len(d) / (ng * max(int(((gcol == g) & ok).sum()), 1))
    print(f"KFIT rows {len(d)} valid {int(ok.sum())} shots {d['file'].nunique()} fset {A.fset} ({X.shape[1]} features); corpus ref {json.dumps({k_: round(v, 4) for k_, v in ref.items()})}", flush=True)
    grp = d["file"].to_numpy()
    oof = pd.DataFrame(np.nan, index=d.index, columns=KC.OUTS)
    for tr, te in GroupKFold(A.folds).split(X, groups=grp):
        tr = tr[ok[tr]]
        ms = fit_models(X.iloc[tr], Yc.iloc[tr], w[tr])
        oof.iloc[te] = predict(ms, X.iloc[te]).to_numpy()
    rep = {"targets": {}}
    for o in KC.OUTS:
        y, p = Y[o].to_numpy()[ok], oof[o].to_numpy()[ok]
        rep["targets"][o] = {"sd": float(np.std(y)), "mean": float(np.mean(y)), "r2_oof": float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
                             "r2_oof_about0": float(1 - ((y - p) ** 2).sum() / (y ** 2).sum())}
    print("TARGETS " + " ".join(f"{o}: sd {v['sd']:.3f} mean {v['mean']:+.3f} R2oof {v['r2_oof']:.2f} (vs 0: {v['r2_oof_about0']:.2f})" for o, v in rep["targets"].items()), flush=True)
    for v in sorted(d["view"].unique()):
        rep[f"sc_view{v}"] = sc_report(d, B, oof, ok & (d["view"].to_numpy() == v), f"OOF view{v}")
    # shrink-only variant: zero-mean corrections (the machine mean of our maps is not moved)
    oo = oof - oof[ok].mean()
    rep["sc_debias"] = sc_report(d, B, oo, ok, "OOF debiased")
    if A.lco:
        Fd = pd.concat([KC.feats(dg, refs[g])[0] for g, dg in d.groupby("grp", sort=False)]).loc[d.index]
        sm = lambda n: Fd["s_" + n].to_numpy()
        tri = np.maximum(Fd["s_dt"].to_numpy(), Fd["s_db"].to_numpy())
        dn = (np.abs(Fd["s_ud"].to_numpy() + ref["ud"]) < np.nanquantile(np.abs(Fd["s_ud"].to_numpy() + ref["ud"]), 0.3)) & (np.minimum(Fd["s_dt"].to_numpy() + ref["dt"], Fd["s_db"].to_numpy() + ref["db"]) > 0.3)
        groups = {"kappa_hi": sm("kap") > np.nanquantile(sm("kap"), 0.85), "tri_hi": tri > np.nanquantile(tri, 0.85),
                  "eps_hi(lowA)": sm("eps") > np.nanquantile(sm("eps"), 0.85), "dn_like": dn}
        rep["lco"] = {}
        for gname, gm in groups.items():
            gm = gm & np.isfinite(sm("kap"))
            te = np.where(gm)[0]; tr = np.where(~gm & ok)[0]
            ms = fit_models(X.iloc[tr], Yc.iloc[tr], w[tr])
            P = pd.DataFrame(0.0, index=d.index, columns=KC.OUTS); P.iloc[te] = predict(ms, X.iloc[te]).to_numpy()
            a_ = sc_report(d, B, P, ok & gm, f"LCO {gname} (trained WITHOUT the group; {d['file'][gm].nunique()} shots)")
            b_ = sc_report(d, B, oof, ok & gm, f"    {gname} in-distribution CV reference")
            rep["lco"][gname] = {"lco": a_, "cv": b_, "n_shots": int(d["file"][gm].nunique())}
    models = fit_models(X[ok], Yc[ok], w[ok])
    # support statistics (robust standardisation + Mahalanobis on the numeric features of the set)
    Xn = X[ok].to_numpy(float)
    med = np.nanmedian(Xn, 0); q1, q3 = np.nanquantile(Xn, 0.25, 0), np.nanquantile(Xn, 0.75, 0)
    sc = np.where((q3 - q1) > 1e-9, (q3 - q1) / 1.349, np.nanstd(Xn, 0) + 1e-9)
    Zs = np.clip(np.nan_to_num((Xn - med) / sc), -8, 8)
    cov = np.cov(Zs, rowvar=False); cov = 0.9 * cov + 0.1 * np.diag(np.diag(cov)) + 1e-3 * np.eye(cov.shape[0])
    icov = np.linalg.inv(cov)
    d2 = np.einsum("ij,jk,ik->i", Zs, icov, Zs)
    gate = {"med": med, "sc": sc, "icov": icov, "d2_q": {q: float(np.quantile(d2, q)) for q in (0.5, 0.9, 0.95, 0.99, 0.999)},
            "lo": np.nanquantile(Xn, 0.005, 0), "hi": np.nanquantile(Xn, 0.995, 0)}
    imp = {o: dict(sorted(zip(X.columns, models[o].feature_importances_.tolist()), key=lambda t: -t[1])[:12]) for o in KC.OUTS}
    pickle.dump({"models": models, "fset": A.fset, "cols": list(X.columns), "ref_d3d": ref, "refs": refs, "gate": gate, "report": rep, "oof_mean": oof[ok].mean().to_dict(),
                 "importance": imp, "train": A.train}, open(A.out, "wb"))
    json.dump({"report": rep, "importance": imp, "d2_q": gate["d2_q"]}, open(A.out.replace(".pkl", ".json"), "w"), indent=1, default=float)
    print("IMPORTANCE " + json.dumps({o: list(v)[:6] for o, v in imp.items()}))
    print(f"KFIT wrote {A.out}")
