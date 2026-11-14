#!/usr/bin/env python3
"""
Stage 11o / Ch2 decoder — per-functional estimator tables and SOURCE weights (DIII-D TRAIN pool; labels allowed there).

Estimators of the seven scorer functionals f_j (starter/fusion_scoring, public-wall extraction `w_*` of the corrector-track tables):
  I  functional of the INVERSE map          (table of the inverse run)
  N  functional of the NET map              (table of the net run)
  H  the net's auxiliary HEAD               (aux array of the net parts; Rax, Zax, kappa, volume, li only; temporal sigma = --hsig frames)
Truth (DIII-D only): g_* = the same published functional on the LABEL map (official envelope extraction).

  est    merge the tables + heads of one set into one estimator table (parquet)        [any machine; MAST: no truth columns]
  fit    uncentred out-of-fold error moments on the pool (shot-grouped K folds), convex weights per functional and regime
         (pairwise formula of external review, round 3 §E2 for two sources; its simplex generalisation min w'Mw for three), scalar-space report
  apply  targets mu_j for a set from a weights json (+ support gate, + synthetic weight if present) -> parquet with mu_* columns
  score  scalar-space R2 of estimators / targets against g_* on a DIII-D set (diagnostic)
"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

FN = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
WCOL = {"R_axis": "Ra", "Z_axis": "Za", "kappa": "kappa", "tri_top": "tri_top", "tri_bot": "tri_bot", "volume": "volume", "li": "li"}
CPN = ["Rt", "Zt", "Rb", "Zb", "Ri", "Ro"]                 # boundary control coordinates (proxy targets of the D_LCFS term)
WCOL.update({c: c for c in CPN})
ALLT = FN + CPN
HEAD = {"R_axis": "Rax", "Z_axis": "Zax", "kappa": "kappa", "volume": "volume", "li": "li"}
REG = ["ordinary", "startup", "inner_cens", "near_dn"]


def gsm(x, ok, sig):
    """Gaussian smoothing along the shot over the ok frames (kernel shrinks at the ends), as syn_predict --smooth."""
    x = np.asarray(x, float).copy()
    gi = np.where(ok & np.isfinite(x))[0]
    if sig <= 0 or len(gi) < 3:
        return x
    y = x[gi]; H = int(np.ceil(3 * sig)); y2 = y.copy()
    for n in range(len(gi)):
        hh = min(H, n, len(gi) - 1 - n)
        wk = np.exp(-0.5 * (np.arange(-hh, hh + 1) / sig) ** 2); wk /= wk.sum()
        y2[n] = float(wk @ y[n - hh:n + hh + 1])
    x[gi] = y2
    return x


def _head_rows(z, name, hsig):
    an = [str(a) for a in z["aux_names"]]
    aux = z["aux"].astype(float)
    okall = np.isfinite(aux).all(1)
    ks = z["ks"].astype(int)
    d = {"file": name, "k": ks}
    for j, h in HEAD.items():
        col = aux[:, an.index(h)]
        d[f"H0_{j}"] = col[ks]
        d[f"H_{j}"] = gsm(col, okall, hsig)[ks]
    return pd.DataFrame(d)


def heads_of(run, tag, machine, names, hsig, tar=None):
    rows = []
    if tar:                                           # parts packed by their owner: one sequential pass over the archive
        import io, tarfile
        pre = f"{tag}_{machine}_"
        with tarfile.open(tar, "r") as tf:
            for m in tf:
                b = os.path.basename(m.name)
                if not (b.startswith(pre) and b.endswith(".npz")):
                    continue
                rest = b[len(pre):-4]
                if not rest.isdigit():
                    continue
                z = np.load(io.BytesIO(tf.extractfile(m).read()), allow_pickle=False)
                rows.append(_head_rows(z, names[int(rest)], hsig))
        return pd.concat(rows, ignore_index=True)
    for i, name in enumerate(names):
        f = Path(run) / "parts" / f"{tag}_{machine}_{i:05d}.npz"
        if not f.exists():
            continue
        rows.append(_head_rows(np.load(f, allow_pickle=False), name, hsig))
    return pd.concat(rows, ignore_index=True)


def regimes(d):
    """<= 4 regimes from inputs + OUR OWN inverse solution (same definition on both machines; no label)."""
    fi = d["file"].to_numpy()
    ip = np.abs(d["ip_A"].to_numpy(float))
    ipf = ip / np.maximum(pd.Series(ip).groupby(fi).transform("max").to_numpy(), 1.0)
    tf = d["k"].to_numpy(float) / np.maximum(d["nT"].to_numpy(float) - 1, 1)
    kind = d["gs_kind"].astype(str).to_numpy() if "gs_kind" in d else np.full(len(d), "xpt")
    startup = (kind != "xpt") | ((ipf < 0.6) & (tf < 0.35))
    cin = pd.to_numeric(d["ts_cin"], errors="coerce").fillna(1.0).to_numpy() > 0.5 if "ts_cin" in d else np.zeros(len(d), bool)
    zt, zb, za = d["I_Zt"].to_numpy(float), d["I_Zb"].to_numpy(float), d["I_Za"].to_numpy(float)
    with np.errstate(all="ignore"):
        ud = np.abs(zt + zb - 2 * za) / np.maximum(zt - zb, 1e-6)
    near = np.isfinite(ud) & (ud < 0.05)
    r = np.zeros(len(d), int)                       # ordinary
    r[near] = 3
    r[cin & ~near] = 2
    r[startup] = 1
    return r, {"ipf": ipf, "tf": tf, "ud": ud}


def cmd_est(a):
    ti = pd.read_parquet(a.inv_tab); tn = pd.read_parquet(a.net_tab)
    d3 = "g_Ra" in ti.columns and not a.truth_tab      # synthetic sets: the truth comes from --truth-tab, never from the real shot's labels
    keep = ["file", "idx", "k", "nT", "t", "ip_A", "ok"] + [c for c in ("gs_kind", "gs_nx", "ts_cin", "ts_cens", "ts_has", "lvl", "pass", "gs_li", "gs_kappa", "li_t", "li_insup", "fill") if c in ti.columns]
    d = ti[keep].copy()
    for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro", "kappa", "tri_top", "tri_bot", "volume", "li"):
        d["I_" + p] = ti["w_" + p].to_numpy(float)
        if d3:
            d["Ip_" + p] = ti["p_" + p].to_numpy(float); d["g_" + p] = ti["g_" + p].to_numpy(float)
    d["I_q95"], d["I_betaN"] = ti["q95"].to_numpy(float), ti["betaN"].to_numpy(float)
    if d3:
        d["gt_q95"], d["gt_betaN"] = ti["gt_q95"].to_numpy(float), ti["gt_betaN"].to_numpy(float)
        d["I_psi_sse"] = ti["psi_sse"].to_numpy(float); d["psi_s1"] = ti["psi_s1"].to_numpy(float); d["psi_s2"] = ti["psi_s2"].to_numpy(float); d["psi_n"] = ti["psi_n"].to_numpy(float)
        for p in ("Rt", "Zt", "Rb", "Zb", "Ri", "Ro"):
            pass
    n2 = tn[["file", "k"] + ["w_" + p for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro", "kappa", "tri_top", "tri_bot", "volume", "li")] + ["q95", "betaN"]
            + (["p_" + p for p in ("Ra", "Za", "kappa", "tri_top", "tri_bot", "volume", "li")] + ["psi_sse"] if d3 else [])].copy()
    n2.columns = ["file", "k"] + ["N_" + p for p in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro", "kappa", "tri_top", "tri_bot", "volume", "li")] + ["N_q95", "N_betaN"] \
        + (["Np_" + p for p in ("Ra", "Za", "kappa", "tri_top", "tri_bot", "volume", "li")] + ["N_psi_sse"] if d3 else [])
    d = d.merge(n2, on=["file", "k"], how="inner")
    if a.blend_tab:
        tb = pd.read_parquet(a.blend_tab)
        CP = ["Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro", "kappa", "tri_top", "tri_bot", "volume", "li"]
        b2 = tb[["file", "k"] + ["w_" + c for c in CP] + (["p_" + WCOL[j] for j in FN] if d3 else []) + (["psi_sse"] if d3 else [])].copy()
        b2.columns = ["file", "k"] + ["B_" + c for c in CP] + (["Bp_" + WCOL[j] for j in FN] if d3 else []) + (["B_psi_sse"] if d3 else [])
        d = d.merge(b2, on=["file", "k"], how="left")
    if a.truth_tab:                                   # synthetic populations: truth functionals / flux moments of OUR OWN synthetic equilibria
        tt = pd.read_parquet(a.truth_tab)
        d = d.merge(tt, on=["file", "k"], how="inner")
    names = [x for x in (Path(a.net_run) / f"files_{a.net_tag}_{a.machine}.txt").read_text().split("\n") if x]
    h = heads_of(a.net_run, a.net_tag, a.machine, names, a.hsig, a.net_tar)
    d = d.merge(h, on=["file", "k"], how="left")
    r, ex = regimes(d)
    d["regime"] = r
    for k_, v_ in ex.items():
        d[k_] = v_
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out)
    print(f"DWEST {a.out}: {len(d)} frames, {d['file'].nunique()} shots; regimes " + " ".join(f"{REG[i]} {float((r == i).mean()):.3f}" for i in range(4))
          + f"; head NaN {float(d['H_R_axis'].isna().mean()):.4f}", flush=True)


def colmap(d, j, src):
    c = {"I": "I_", "N": "N_", "B": "B_"}.get(src)
    return d[c + WCOL[j]].to_numpy(float) if c else (d[f"{src}_{j}"].to_numpy(float) if f"{src}_{j}" in d else np.full(len(d), np.nan))


def simplex_w(M, cap=None):
    """argmin w'Mw on the simplex (w >= 0, sum 1) by active-set enumeration (<= 3 sources)."""
    n = len(M); best = (np.inf, None)
    from itertools import combinations
    for m in range(1, n + 1):
        for S in combinations(range(n), m):
            S = list(S)
            A = M[np.ix_(S, S)]
            try:
                x = np.linalg.solve(A + 1e-12 * np.trace(A) * np.eye(m), np.ones(m))
            except np.linalg.LinAlgError:
                continue
            if x.sum() <= 0:
                continue
            x = x / x.sum()
            if (x < -1e-9).any():
                continue
            w = np.zeros(n); w[S] = x
            v = float(w @ M @ w)
            if v < best[0]:
                best = (v, w)
    return best[1], best[0]


def moments(E):
    """uncentred second-moment matrix of the error columns E (n, m) on rows where all are finite."""
    m = np.isfinite(E).all(1)
    X = E[m]
    return (X.T @ X) / max(len(X), 1), int(m.sum())


def fit_weights(d, srcs_of, min_shots=30):
    """weights[j][regime] (dict src -> w) with pooling of regimes that have < min_shots source shots."""
    W = {}
    info = {}
    for j in ALLT:
        srcs = srcs_of.get(j, ["I", "N"])
        g = d["g_" + WCOL[j]].to_numpy(float)
        E = np.stack([colmap(d, j, s) - g for s in srcs], 1)
        Mall, nall = moments(E)
        wall, vall = simplex_w(Mall)
        W[j] = {"srcs": srcs, "pooled": wall.tolist()}
        info[j] = {"pooled": {"M": Mall.tolist(), "n": nall, "mse_w": vall, "mse_src": np.diag(Mall).tolist()}}
        for ri, rn in enumerate(REG):
            sel = d["regime"].to_numpy() == ri
            ns = d.loc[sel & np.isfinite(E).all(1), "file"].nunique()
            if ns < min_shots:
                W[j][rn] = wall.tolist(); info[j][rn] = {"n_shots": int(ns), "pooled": True}
                continue
            M, n = moments(E[sel])
            w, v = simplex_w(M)
            W[j][rn] = w.tolist(); info[j][rn] = {"n_shots": int(ns), "n": n, "mse_w": v, "mse_src": np.diag(M).tolist(), "pooled": False}
    return W, info


def targets(d, W, shrink=None, per_regime=True):
    """mu_j = sum_s w_s est_s (renormalised over the finite sources). shrink: None or g_D array in [0, 1] -> w0 = (1-g) wS + g wD with wS = W['_wS'][j] or uniform."""
    out = {}
    reg = d["regime"].to_numpy()
    for j in [t for t in ALLT if t in W]:
        srcs = W[j]["srcs"]
        X = np.stack([colmap(d, j, s) for s in srcs], 1)
        w = np.zeros_like(X)
        for ri, rn in enumerate(REG):
            sel = reg == ri
            w[sel] = np.asarray(W[j][rn] if per_regime else W[j]["pooled"], float)[None, :]
        if shrink is not None:
            gD = np.asarray(shrink, float)[:, None]
            ws = W[j].get("wS")
            if isinstance(ws, dict) and len(srcs) == 2:              # dsyn.py ws: per regime {w_net, lo, hi}; srcs = [I, N]
                wn, lo, hi = np.zeros(len(d)), np.zeros(len(d)), np.ones(len(d))
                for ri, rn in enumerate(REG):
                    e = ws.get(rn) or ws.get("pooled") or {"w_net": 0.5, "lo": 0.0, "hi": 1.0}
                    sel = reg == ri
                    wn[sel], lo[sel], hi[sel] = e["w_net"], e["lo"], e["hi"]
                w0 = (1 - gD[:, 0]) * wn + gD[:, 0] * w[:, 1]
                w0 = np.clip(w0, lo, hi)                             # projection: every L_jd(w) <= L_jd(w_S) + tol
                w = np.stack([1 - w0, w0], 1)
            else:
                wS = np.asarray(ws if ws is not None else [1.0 / len(srcs)] * len(srcs), float)
                w = (1 - gD) * wS[None, :] + gD * w
        fin = np.isfinite(X)
        w = np.where(fin, w, 0.0)
        s = w.sum(1)
        # all weighted sources missing -> fall back to the mean of whatever is finite
        fb = (s <= 1e-9)
        w[fb] = fin[fb].astype(float)
        s = w.sum(1)
        with np.errstate(all="ignore"):
            out[j] = np.where(s > 0, (np.where(fin, X, 0.0) * w).sum(1) / np.maximum(s, 1e-12), np.nan)
    return out


def r2(g, p):
    m = np.isfinite(g) & np.isfinite(p)
    g, p = g[m], p[m]
    return float(1 - ((g - p) ** 2).sum() / ((g - g.mean()) ** 2).sum()) if len(g) > 10 else np.nan


def report(d, cols, tag=""):
    """scalar-space R2 against g_* per functional + Cons-like mean of max(0, R2)."""
    lines = []
    for name, getter in cols.items():
        v = [r2(d["g_" + WCOL[j]].to_numpy(float), getter(j)) for j in FN]
        lines.append((name, v, float(np.mean([max(0.0, x) for x in v if np.isfinite(x)]))))
    print(f"-- scalar-space R2 vs label functionals {tag} ({len(d)} frames, {d['file'].nunique()} shots)")
    print(f"   {'estimator':28s} " + " ".join(f"{j[:7]:>8s}" for j in FN) + "     cons")
    for name, v, c in lines:
        print(f"   {name:28s} " + " ".join(f"{x:8.3f}" for x in v) + f"   {c:.4f}")
    return {name: {"r2": dict(zip(FN, v)), "cons": c} for name, v, c in lines}


SRCSETS = {
    "IN": {j: ["I", "N"] for j in FN},
    "INH": {j: (["I", "N", "H"] if j in HEAD else ["I", "N"]) for j in FN},
    "INH0": {j: (["I", "N", "H0"] if j in HEAD else ["I", "N"]) for j in FN},
}


def cmd_fit(a):
    d = pd.concat([pd.read_parquet(f).assign(cond=i) for i, f in enumerate(a.est)], ignore_index=True)
    d = d[d["ok"].astype(bool)].reset_index(drop=True)
    if a.exclude:
        ex = {x for x in Path(a.exclude).read_text().split("\n") if x}
        d = d[~d["file"].isin(ex)].reset_index(drop=True)
    shots = np.array(sorted(d["file"].unique()))
    rng = np.random.default_rng(0)
    fold = dict(zip(shots, rng.permutation(len(shots)) % a.folds))
    fo = d["file"].map(fold).to_numpy()
    res = {"n_frames": len(d), "n_shots": len(shots), "regime_frac": {REG[i]: float((d["regime"] == i).mean()) for i in range(4)},
           "regime_shots": {REG[i]: int(d.loc[d["regime"] == i, "file"].nunique()) for i in range(4)}}
    out = {}
    for sname, srcs_of in SRCSETS.items():
        W, info = fit_weights(d, srcs_of)
        # out-of-fold targets
        mu_oof = {j: np.full(len(d), np.nan) for j in ALLT}; mu_oofp = {j: np.full(len(d), np.nan) for j in ALLT}
        for k in range(a.folds):
            tr, te = fo != k, fo == k
            Wk, _ = fit_weights(d[tr], srcs_of)
            t = targets(d[te], Wk); tp = targets(d[te], Wk, per_regime=False)
            for j in ALLT:
                mu_oof[j][te] = t[j]; mu_oofp[j][te] = tp[j]
        half = np.full(len(d), 0.5)
        mu_half = targets(d, W, shrink=half)
        gcol = lambda j: d["g_" + WCOL[j]].to_numpy(float)  # noqa: E731
        for j in ALLT:
            W[j]["r2_oof"] = r2(gcol(j), mu_oof[j]); W[j]["r2_oof_pooled"] = r2(gcol(j), mu_oofp[j])
            m_ = np.isfinite(gcol(j)) & np.isfinite(mu_oof[j])
            W[j]["rmse_oof"] = float(np.sqrt(np.mean((gcol(j)[m_] - mu_oof[j][m_]) ** 2))); W[j]["sd_truth"] = float(np.std(gcol(j)[m_]))
        out[sname] = {"W": W, "info": info}
        cols = {"I (inverse map, wall)": lambda j: colmap(d, j, "I"), "N (net map, wall)": lambda j: colmap(d, j, "N"),
                "H (head, sigma)": lambda j: colmap(d, j, "H"), "H0 (head, raw)": lambda j: colmap(d, j, "H0"),
                "B (map blend 0.5, wall)": lambda j: colmap(d, j, "B"), "mean(I, N)": lambda j: 0.5 * (colmap(d, j, "I") + colmap(d, j, "N")),
                f"mu OOF per-regime [{sname}]": lambda j: mu_oof[j], f"mu OOF pooled [{sname}]": lambda j: mu_oofp[j],
                f"mu gD=0.5 shrink [{sname}]": lambda j: mu_half[j]}
        if "Ip_Ra" in d:
            cols["I (official envelope)"] = lambda j: d["Ip_" + WCOL[j]].to_numpy(float)
            cols["N (official envelope)"] = lambda j: d["Np_" + WCOL[j]].to_numpy(float)
            if "Bp_Ra" in d:
                cols["B (official envelope)"] = lambda j: d["Bp_" + WCOL[j]].to_numpy(float)
        out[sname]["report"] = report(d, cols, f"[{sname}] pool")
        for c in sorted(d["cond"].unique()):
            dc = d[d["cond"] == c]
            m = (d["cond"] == c).to_numpy()
            report(dc, {"B (map blend 0.5, wall)": lambda j: colmap(dc, j, "B"), f"mu OOF per-regime [{sname}]": lambda j: mu_oof[j][m],
                        f"mu OOF pooled [{sname}]": lambda j: mu_oofp[j][m]}, f"[{sname}] cond {c}")
        print(f"   weights [{sname}] (pooled | " + " | ".join(REG) + ")")
        for j in FN:
            print(f"   {j:8s} srcs {W[j]['srcs']}: " + " | ".join("[" + " ".join(f"{x:.2f}" for x in W[j][r_]) + "]" for r_ in ["pooled"] + REG)
                  + "   rmse src " + " ".join(f"{np.sqrt(x):.4f}" for x in info[j]["pooled"]["mse_src"]) + f" -> {np.sqrt(info[j]['pooled']['mse_w']):.4f}")
    res["sets"] = out
    # ---- flux weight of the net (uncentred moments of the map errors; cross moment from the 0.5 / 0.5 blend's SSE)
    if "B_psi_sse" in d:
        fl = {}
        for rn, sel in [("pooled", np.ones(len(d), bool))] + [(REG[i], (d["regime"] == i).to_numpy()) for i in range(4)]:
            m_ = sel & np.isfinite(d["I_psi_sse"]) & np.isfinite(d["N_psi_sse"]) & np.isfinite(d["B_psi_sse"])
            if d.loc[m_, "file"].nunique() < 30:
                fl[rn] = fl["pooled"]; continue
            mii, mnn, mbb = float(d.loc[m_, "I_psi_sse"].sum()), float(d.loc[m_, "N_psi_sse"].sum()), float(d.loc[m_, "B_psi_sse"].sum())
            mi_n = 2.0 * mbb - 0.5 * (mii + mnn)
            den = mnn + mii - 2 * mi_n
            fl[rn] = {"w_net": float(np.clip((mii - mi_n) / den, 0, 1)) if den > 0 else 0.5, "M_II": mii, "M_NN": mnn, "M_IN": mi_n, "n": int(m_.sum())}
        res["flux"] = fl
        tot = float(d["psi_s2"].sum() - d["psi_s1"].sum() ** 2 / d["psi_n"].sum())
        res["flux_r2"] = {"I": 1 - float(d["I_psi_sse"].sum()) / tot, "N": 1 - float(d["N_psi_sse"].sum()) / tot, "B": 1 - float(d["B_psi_sse"].sum()) / tot}
        print("   flux weight of the net:", json.dumps({k: (v["w_net"] if isinstance(v, dict) else v) for k, v in fl.items()}), "R2psi", json.dumps(res["flux_r2"]))
    # ---- prior sd of the decoder's response coordinates = robust sd (1.4826 x median |.|) of the START map's control-point errors, in units of
    #      the label minor radius (dimensionless => transferable); li: log ratio
    pri = {}
    ag = 0.5 * (d["g_Ro"] - d["g_Ri"]).to_numpy(float)
    for st in ("I", "N", "B"):
        e = {c: (d["g_" + c].to_numpy(float) - d[f"{st}_" + c].to_numpy(float)) / ag for c in ("Ra", "Za", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro")}
        rs = lambda v: float(1.4826 * np.nanmedian(np.abs(v)))  # noqa: E731
        with np.errstate(all="ignore"):
            lli = np.log(d["g_li"].to_numpy(float) / d[f"{st}_li"].to_numpy(float))
        pri[st] = {"sR": rs(0.5 * (e["Ri"] + e["Ro"])), "sZ": rs(0.5 * (e["Zt"] + e["Zb"])), "dR": rs(0.5 * (e["Ro"] - e["Ri"])), "dZ": rs(0.5 * (e["Zt"] - e["Zb"])),
                   "shr": rs(0.5 * (e["Rt"] - e["Rb"])), "tri": rs(0.5 * (e["Rt"] + e["Rb"]) - 0.5 * (e["Ri"] + e["Ro"])),
                   "cR": rs(e["Ra"] - 0.5 * (e["Ri"] + e["Ro"])), "cZ": rs(e["Za"] - 0.5 * (e["Zt"] + e["Zb"])), "li": rs(lli)}
        print(f"   coordinate prior sd [{st}] (units of a): " + " ".join(f"{k} {v:.4f}" for k, v in pri[st].items()))
    res["prior"] = pri
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, default=float)
    print("DWFIT", a.out, json.dumps({k: res[k] for k in ("n_frames", "n_shots", "regime_frac", "regime_shots")}), flush=True)


def cmd_apply(a):
    """targets mu_* (7 functionals + 6 control coordinates) for one set; --gd = share of the SOURCE weight (g_D; 1 = pure DIII-D weights)."""
    d = pd.read_parquet(a.est)
    R = json.load(open(a.weights))
    W = R["sets"][a.set]["W"]
    if a.ws:
        WS_ = json.load(open(a.ws))
        for j in ALLT:
            if j in WS_:
                W[j]["wS"] = WS_[j]
    gd = np.full(len(d), a.gd)
    if a.support and Path(a.support).exists():
        sp = pd.read_parquet(a.support)
        gd = a.gd * d[["file", "k"]].merge(sp, on=["file", "k"], how="left")["support"].fillna(0.0).to_numpy(float)
    mu = targets(d, W, shrink=None if a.gd >= 1.0 and not a.support and not a.ws else gd, per_regime=not a.pooled)
    o = d[["file", "k", "regime", "ok"]].copy()
    for j in ALLT:
        o["mu_" + j] = mu[j]
    o["gd"] = gd
    fl = R.get("flux", {})
    wn = np.array([fl.get("pooled" if a.pooled else REG[r_], fl.get("pooled", {"w_net": 0.5}))["w_net"] for r_ in d["regime"].to_numpy()]) if fl else np.full(len(d), 0.5)
    o["w_psi_net"] = (1 - gd) * 0.5 + gd * wn
    if a.ws and a.psi_ws:                             # robust synthetic flux weight per regime (dsyn.py ws -> "psi") + projection interval
        ps = json.load(open(a.ws)).get("psi") or {}
        if ps.get("pooled"):
            reg_ = d["regime"].to_numpy()
            wS, lo, hi = np.full(len(d), 0.5), np.zeros(len(d)), np.ones(len(d))
            for ri, rn in enumerate(REG):
                e = ps.get(rn) or ps["pooled"]
                sel = reg_ == ri
                wS[sel], lo[sel], hi[sel] = e["w_net"], e["lo"], e["hi"]
            o["w_psi_net"] = np.clip((1 - gd) * wS + gd * wn, lo, hi)
            if a.psi_ws_reg:                          # regime-aware flux mean ONLY in the listed regimes; the plain 0.5 / 0.5 blend elsewhere
                keepr = [REG.index(x) for x in a.psi_ws_reg.split(",")]
                o["w_psi_net"] = np.where(np.isin(reg_, keepr), o["w_psi_net"], 0.5)
    for j in ALLT:
        o["rn_" + j] = max(1e-3, 1.0 - float(W[j].get("r2_oof", 0.0)))          # relative target noise (K-mode R_f / v_j)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(a.out)
    okm = o["ok"].astype(bool).to_numpy()
    st = {"sd": {j: float(np.nanstd(o.loc[okm, "mu_" + j])) for j in ALLT}, "n": int(okm.sum())}
    json.dump(st, open(str(a.out).replace(".parquet", "_stats.json"), "w"), indent=1)
    print(f"DWAPPLY {a.out}: {len(o)} frames; gd mean {float(gd.mean()):.3f}; w_psi_net mean {float(o['w_psi_net'].mean()):.3f}; sd " + " ".join(f"{j} {v:.4f}" for j, v in st["sd"].items()), flush=True)


def cmd_score(a):
    d = pd.read_parquet(a.est)
    d = d[d["ok"].astype(bool)].reset_index(drop=True)
    R = json.load(open(a.weights))
    rep = {}
    for sname in a.sets.split(","):
        W = R["sets"][sname]["W"]
        mu = targets(d, W); mup = targets(d, W, per_regime=False); muh = targets(d, W, shrink=np.full(len(d), 0.5)); muhp = targets(d, W, shrink=np.full(len(d), 0.5), per_regime=False)
        cols = {"I (inverse map, wall)": lambda j: colmap(d, j, "I"), "N (net map, wall)": lambda j: colmap(d, j, "N"), "H (head, sigma)": lambda j: colmap(d, j, "H"),
                "B (map blend 0.5, wall)": lambda j: colmap(d, j, "B"), "mean(I, N)": lambda j: 0.5 * (colmap(d, j, "I") + colmap(d, j, "N")),
                f"mu per-regime [{sname}]": lambda j: mu[j], f"mu pooled [{sname}]": lambda j: mup[j], f"mu gD=.5 per-regime [{sname}]": lambda j: muh[j],
                f"mu gD=.5 pooled [{sname}]": lambda j: muhp[j]}
        if "Bp_Ra" in d:
            cols["B (official envelope)"] = lambda j: d["Bp_" + WCOL[j]].to_numpy(float)
            cols["I (official envelope)"] = lambda j: d["Ip_" + WCOL[j]].to_numpy(float)
        rep[sname] = report(d, cols, f"[{sname}] {a.est}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("est")
    p.add_argument("--machine", required=True); p.add_argument("--inv-tab", required=True); p.add_argument("--net-tab", required=True); p.add_argument("--blend-tab", default=None)
    p.add_argument("--net-run", required=True); p.add_argument("--net-tag", required=True); p.add_argument("--net-tar", default=None); p.add_argument("--truth-tab", default=None); p.add_argument("--hsig", type=float, default=1.0); p.add_argument("--out", required=True)
    p = sp.add_parser("fit")
    p.add_argument("--est", nargs="+", required=True); p.add_argument("--folds", type=int, default=5); p.add_argument("--out", required=True)
    p.add_argument("--exclude", default=None, help="txt list of shot files left out of the fit (dev shots)")
    p = sp.add_parser("score")
    p.add_argument("--est", required=True); p.add_argument("--weights", required=True); p.add_argument("--sets", default="IN,INH"); p.add_argument("--out", default=None)
    p = sp.add_parser("apply")
    p.add_argument("--est", required=True); p.add_argument("--weights", required=True); p.add_argument("--set", default="IN"); p.add_argument("--gd", type=float, default=1.0)
    p.add_argument("--ws", default=None); p.add_argument("--psi-ws", action="store_true", help="also blend the flux weight with the synthetic one"); p.add_argument("--psi-ws-reg", default=None, help="comma list of regimes where --psi-ws applies (others: 0.5)"); p.add_argument("--support", default=None); p.add_argument("--pooled", action="store_true"); p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"est": cmd_est, "fit": cmd_fit, "score": cmd_score, "apply": cmd_apply}[a.cmd](a)
