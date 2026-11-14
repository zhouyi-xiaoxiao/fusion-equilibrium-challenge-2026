#!/usr/bin/env python3
"""Stage 11K / Ch2 corrector — features, targets, scalar-space algebra shared by kfit / kapply.
All features are dimensionless. Corpus / shot medians are taken on 12 evenly spaced frames per shot on BOTH machines
(the DIII-D training runs have exactly these frames), from OUR OWN maps + inputs only (label-free normalisation)."""
import numpy as np
import pandas as pd

OUTS = ["dRa", "dZa", "dRt", "dZt", "dRb", "dZb", "dRi", "dRo", "dlli"]
PT = {"dRa": "Ra", "dZa": "Za", "dRt": "Rt", "dZt": "Zt", "dRb": "Rb", "dZb": "Zb", "dRi": "Ri", "dRo": "Ro"}
GEO = ["kap", "dt", "db", "sh", "za", "zc", "li", "eps", "lna", "lnrg", "ud"]


def base(d, pre="w_"):
    g = lambda n: d[pre + n].to_numpy(float)
    a = 0.5 * (g("Ro") - g("Ri")); rg = 0.5 * (g("Ro") + g("Ri")); b = 0.5 * (g("Zt") - g("Zb")); zc = 0.5 * (g("Zt") + g("Zb"))
    with np.errstate(all="ignore"):
        o = {"a": a, "rg": rg, "b": b, "zcen": zc, "kap": b / a, "dt": (rg - g("Rt")) / a, "db": (rg - g("Rb")) / a,
             "sh": (g("Ra") - rg) / a, "za": (g("Za") - zc) / a, "zc": zc / a, "li": g("li"), "eps": a / rg, "lna": np.log(a), "lnrg": np.log(rg),
             "ud": (g("Zt") + g("Zb") - 2 * g("Za")) / a, "Ra": g("Ra"), "Za": g("Za"), "vol": g("volume")}
    return o


def sub12(d):
    """boolean mask: the 12 evenly spaced frames of each shot (frames of the sparse training runs)."""
    k = d["k"].to_numpy(int); nT = d["nT"].to_numpy(int)
    m = np.zeros(len(d), bool)
    for T in np.unique(nT):
        sel = set(np.linspace(0, T - 1, min(12, T)).round().astype(int).tolist())
        j = nT == T
        m[j] = np.isin(k[j], list(sel))
    return m


def corpus_ref(d):
    """machine-level medians of OUR geometry (label-free), on the 12-frame sub-sampling."""
    B = base(d); m = sub12(d)
    if m.sum() < 0.3 * len(d) / max(1.0, float(np.median(d["nT"])) / 12.0):          # anchors-only tables (synthetic test): use every row
        m = np.ones(len(d), bool)
    return {n: float(np.nanmedian(B[n][m])) for n in GEO}


def feats(d, ref):
    B = base(d)
    F = pd.DataFrame(index=d.index)
    m12 = sub12(d)
    fi = d["file"].to_numpy()
    for n in GEO:
        x = pd.Series(B[n], index=d.index)
        sm = x.where(m12).groupby(fi).transform("median")
        sm = sm.fillna(x.groupby(fi).transform("median"))
        F["h_" + n] = x - sm
        F["s_" + n] = sm - ref[n]
    a = B["a"]
    k = d["k"].to_numpy(float); nT = d["nT"].to_numpy(float)
    F["tf"] = k / np.maximum(nT - 1, 1)
    F["first"] = (k == 0).astype(float); F["last"] = (k == nT - 1).astype(float)
    ip = np.abs(d["ip_A"].to_numpy(float))
    F["ipf"] = ip / np.maximum(pd.Series(ip).groupby(fi).transform("max").to_numpy(), 1.0)
    col = lambda n: d[n] if n in d else pd.Series(np.nan, index=d.index)
    F["ok"] = col("ok").astype(float)
    F["lvl0"] = (pd.to_numeric(col("lvl"), errors="coerce") == 0).astype(float)
    F["pass1"] = (pd.to_numeric(col("pass"), errors="coerce") == 1).astype(float)
    F["xpt"] = (col("gs_kind").astype(str) == "xpt").astype(float); F["wall"] = (col("gs_kind").astype(str) == "wall").astype(float)
    F["xfit"] = (col("x_src").astype(str) == "fit").astype(float)
    F["cens_o"] = pd.to_numeric(col("ts_cens"), errors="coerce").fillna(1.0).astype(float)
    F["cens_i"] = pd.to_numeric(col("ts_cin"), errors="coerce").fillna(1.0).astype(float)
    num = lambda n: pd.to_numeric(col(n), errors="coerce").to_numpy(float)
    F["iso"] = num("iso_mae_cm") / 100.0 / a
    F["dr"] = num("dr") / a
    F["gev"] = num("gev"); F["zt"] = num("zt") / a
    with np.errstate(all="ignore"):
        F["lcp"] = np.log(num("cp"))
    F["b1"] = num("b1"); F["b2"] = num("b2"); F["fpress"] = num("f_press")
    F["e_out"] = (num("ts_edge") - d["w_Ro"].to_numpy(float)) / a
    F["e_in"] = np.where(F["cens_i"].to_numpy() > 0.5, np.nan, (num("ts_rin") - d["w_Ri"].to_numpy(float)) / a)
    F["dli"] = num("gs_li") - B["li"]; F["dkap"] = num("gs_kappa") - B["kap"]
    F["betap"] = num("gs_betap"); F["nx"] = num("gs_nx")
    for n in ("sv0", "sv1", "sv2", "sv3"):
        with np.errstate(all="ignore"):
            F["l" + n] = np.log10(np.maximum(num(n), 1e-6))
    for n in GEO:                       # absolute (machine-revealing) versions: only for the DIII-D-only upper bound
        F["abs_" + n] = B[n]
    return F, B


CORE = (["h_" + n for n in GEO] + ["s_" + n for n in GEO] +
        ["tf", "first", "last", "ipf", "ok", "lvl0", "pass1", "xpt", "wall", "xfit", "cens_o", "cens_i", "iso", "dr", "gev", "zt", "lcp", "b1", "b2",
         "fpress", "e_out", "e_in", "dli", "dkap", "betap", "nx", "lsv0", "lsv1", "lsv2", "lsv3"])
SHRINK = ["h_" + n for n in GEO] + ["s_" + n for n in GEO] + ["tf", "first", "last", "ipf", "ok", "lvl0", "xpt", "wall", "cens_o", "cens_i"]
FULL = CORE + ["abs_" + n for n in GEO]
# transfer set: only features whose MAST distribution (own maps + inputs, label-free) lies inside the DIII-D support and whose
# definition is the same on both machines (kdiag.py: dropped = R_geo / aspect-ratio deviations, laser-edge residuals and censoring
# flags [different laser geometry], solver-vs-scorer kappa difference, number of X-points, first singular value)
_DROP = {"h_lnrg", "s_lnrg", "h_eps", "s_eps", "e_out", "e_in", "cens_o", "cens_i", "dkap", "lsv0", "nx"}
TR = [c for c in CORE if c not in _DROP]
TRS = [c for c in SHRINK if c not in _DROP]
FSETS = {"core": CORE, "shrink": SHRINK, "full": FULL, "tr": TR, "trs": TRS}


def targets(d, B):
    a = B["a"]
    Y = pd.DataFrame(index=d.index)
    for o, p in PT.items():
        Y[o] = (d["g_" + p].to_numpy(float) - d["w_" + p].to_numpy(float)) / a
    with np.errstate(all="ignore"):
        Y["dlli"] = np.log(d["g_li"].to_numpy(float) / d["w_li"].to_numpy(float))
    return Y


def scalars_from(d, B, Y=None, pre="w_"):
    """the scorer's shape scalars implied by control points (+ optional normalised corrections Y); li via the log ratio."""
    a = B["a"]
    P = {p: d[pre + p].to_numpy(float).copy() for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro")}
    li = d[pre + "li"].to_numpy(float).copy()
    if Y is not None:
        for o, p in PT.items():
            if o in Y:
                P[p] = P[p] + np.nan_to_num(np.asarray(Y[o], float)) * a
        if "dlli" in Y:
            li = li * np.exp(np.nan_to_num(np.asarray(Y["dlli"], float)))
    aa = 0.5 * (P["Ro"] - P["Ri"]); rg = 0.5 * (P["Ro"] + P["Ri"])
    with np.errstate(all="ignore"):
        return {"R_axis": P["Ra"], "Z_axis": P["Za"], "kappa": (P["Zt"] - P["Zb"]) / (2 * aa), "tri_top": (rg - P["Rt"]) / aa,
                "tri_bot": (rg - P["Rb"]) / aa, "li": li}


def r2dec(g, p):
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    if len(g) < 10 or g.std() == 0:
        return dict(r2=np.nan, rho=np.nan, r=np.nan, b=np.nan)
    sg = g.std()
    return dict(r2=float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum()), rho=float(np.corrcoef(g, p)[0, 1]), r=float(p.std() / sg),
                b=float((p.mean() - g.mean()) / sg))
