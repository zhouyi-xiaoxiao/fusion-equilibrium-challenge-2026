#!/usr/bin/env python3
"""Step 1 — diagnostics of a holdout prediction file (official derivations on GT and prediction).

Writes  $CC_OUT/hold_ref.npz            GT reference cache (f_gt11, contours, rgeo, flux sums, q95/betaN)
        $CC_OUT/der_hold_<tag>.npz      prediction-side pieces (f11, haus, ssres, li four-way)
        $CC_OUT/diag_<tag>.json/.txt    (a) kappa height/width decomposition, (b) li four-way,
                                        (c) shot-bias vs within-shot, (d) loss by phase
Usage: cc_diag.py --pred <preds_d3d_X.npz> --tag X [--workers 70] [--maxf 0]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp

import numpy as np

import cc_common as cc
from cc_common import log

_D = {}


def work(i):
    g = cc.init_scorer()
    pg = np.asarray(_D["gt"][i], dtype=np.float64)
    pp = _D["pr"][i]
    fg, Cg = cc.dref(pg)
    fp, Cp = cc.dpred(pp)
    li4 = np.array([cc.li_of(pg, Cg), cc.li_of(pp, Cg), cc.li_of(pg, Cp), cc.li_of(pp, Cp)])
    hd = cc.haus(Cp, Cg)
    rg = g["rmaj"](Cg) if Cg is not None else np.nan
    r = pg - pp
    return (i, fg, fp, li4, hd, rg, float((r * r).sum()), float(pg.sum()), float((pg * pg).sum()),
            (Cg if Cg is not None else np.full((cc.N_POINTS, 2), np.nan)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=70)
    ap.add_argument("--maxf", type=int, default=0)
    a = ap.parse_args()
    cc.OUT.mkdir(parents=True, exist_ok=True)
    hold = np.load(cc.HC / "cache_hold.npz")
    off = hold["offsets"]
    N = int(off[-1])
    psi_pr, q, b = cc.load_pred_npz(a.pred, off)
    _D["gt"] = np.load(cc.HC / "cache_hold_psi.npy", mmap_mode="r")
    _D["pr"] = psi_pr
    idx = np.arange(N) if not a.maxf else np.linspace(0, N - 1, a.maxf).astype(int)
    fg = np.full((N, 11), np.nan); fp = np.full((N, 11), np.nan); li4 = np.full((N, 4), np.nan)
    hd = np.full(N, np.nan); rg = np.full(N, np.nan); ssres = np.zeros(N); gsum = np.zeros(N); gsq = np.zeros(N)
    Cg = np.full((N, cc.N_POINTS, 2), np.nan)
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for n, o in enumerate(pool.imap_unordered(work, [int(i) for i in idx], chunksize=32), 1):
            i = o[0]
            fg[i], fp[i], li4[i], hd[i], rg[i], ssres[i], gsum[i], gsq[i], Cg[i] = o[1:]
            if n % 5000 == 0:
                log(f"{n}/{len(idx)}")
    if a.maxf:
        log(f"smoke ok: {np.isfinite(fp[idx]).mean():.4f} finite; kappa gt/pred {fg[idx[:3], 2]} {fp[idx[:3], 2]}")
        return
    ref = dict(f_gt=fg, C_gt=Cg, rgeo=rg, gsum=gsum, gsq=gsq, offsets=off, q95=hold["q95"].astype(np.float64),
               betaN=hold["betaN"].astype(np.float64))
    np.savez(cc.OUT / "hold_ref.npz", **ref)
    np.savez(cc.OUT / f"der_hold_{a.tag}.npz", f11=fp, haus=hd, ssres=ssres, li4=li4, q95=q, betaN=b)
    np.save(cc.OUT / f"f11_holdout_{a.tag}.npy", fp)

    S = len(off) - 1
    res = {"tag": a.tag, "frames": N, "shots": S}
    lines = []
    for nm, sh in (("all", None), ("A", np.arange(0, 75)), ("B", np.arange(75, 150))):
        sc = cc.emul_score(ref, fp, hd, ssres, q, b, sh)
        res[f"emul_{nm}"] = sc
        lines.append(cc.fmt_score(f"emul {a.tag} [{nm}]", sc))
    # aux vs own derivation (target identity check for the heads)
    aux = np.load(cc.FEAT / "holdout/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]
    res["aux_vs_ref_maxabs"] = [float(np.nanmax(np.abs(aux[:, j] - fg[:, j]))) for j in range(11)]
    lines.append("y_aux vs hold_ref max|diff| per target: " + " ".join(f"{v:.2e}" for v in res["aux_vs_ref_maxabs"]))

    ok = np.isfinite(fg).all(1) & np.isfinite(fp).all(1)
    e = fp - fg                                            # (N,11) errors
    var = np.array([np.nanvar(fg[:, j]) for j in range(11)])
    res["r2_map_derived"] = {n: cc.pooled_r2(fg[:, j], fp[:, j]) for j, n in enumerate(cc.NAMES11)}
    res["rmse_map_derived"] = {n: float(np.sqrt(np.nanmean(e[:, j] ** 2))) for j, n in enumerate(cc.NAMES11)}
    lines.append("map-derived pooled R2 (11): " + " ".join(f"{n} {res['r2_map_derived'][n]:.4f}" for n in cc.NAMES11))

    # (a) kappa = H/W: exact identity dk = dH/W_hat - kappa*dW/W_hat
    Rmin, Rmax, Zmin, Zmax = (fg[ok, 7 + k] for k in range(4))
    dRmin, dRmax, dZmin, dZmax = (e[ok, 7 + k] for k in range(4))
    What = fp[ok, 8] - fp[ok, 7]
    kap = fg[ok, 2]
    h = (dZmax - dZmin) / What
    w = -kap * (dRmax - dRmin) / What
    dk = e[ok, 2]
    vk = var[2]
    top, bot = dZmax / What, -dZmin / What
    inn, out = kap * dRmin / What, -kap * dRmax / What
    A = {"identity_max_abs": float(np.max(np.abs(dk - h - w))),
         "kappa_mse_over_var": float(np.mean(dk ** 2) / vk),
         "height_term": float(np.mean(h ** 2) / vk), "width_term": float(np.mean(w ** 2) / vk),
         "cov_term_2hw": float(2 * np.mean(h * w) / vk),
         "top_Zmax": float(np.mean(top ** 2) / vk), "bottom_Zmin": float(np.mean(bot ** 2) / vk),
         "cov_top_bottom_2": float(2 * np.mean(top * bot) / vk),
         "inner_Rmin": float(np.mean(inn ** 2) / vk), "outer_Rmax": float(np.mean(out ** 2) / vk),
         "cov_inner_outer_2": float(2 * np.mean(inn * out) / vk),
         "rmse_mm": {"top_Zmax": float(1e3 * np.sqrt(np.mean(dZmax ** 2))), "bottom_Zmin": float(1e3 * np.sqrt(np.mean(dZmin ** 2))),
                     "inner_Rmin": float(1e3 * np.sqrt(np.mean(dRmin ** 2))), "outer_Rmax": float(1e3 * np.sqrt(np.mean(dRmax ** 2))),
                     "height": float(1e3 * np.sqrt(np.mean((dZmax - dZmin) ** 2))), "width": float(1e3 * np.sqrt(np.mean((dRmax - dRmin) ** 2)))},
         "median_abs_mm": {"top_Zmax": float(1e3 * np.median(np.abs(dZmax))), "bottom_Zmin": float(1e3 * np.median(np.abs(dZmin))),
                           "inner_Rmin": float(1e3 * np.median(np.abs(dRmin))), "outer_Rmax": float(1e3 * np.median(np.abs(dRmax)))},
         }
    s2 = np.sort(dk ** 2)[::-1]
    A["share_of_kappa_sse_top_1pct_frames"] = float(s2[: max(1, len(s2) // 100)].sum() / s2.sum())
    A["share_of_kappa_sse_top_5pct_frames"] = float(s2[: max(1, len(s2) // 20)].sum() / s2.sum())
    res["a_kappa"] = A
    lines.append("(a) kappa: 1-R2 %.4f = height %.4f + width %.4f + cov %.4f | top %.4f bottom %.4f cov_tb %.4f | inner %.4f outer %.4f cov_io %.4f"
                 % (A["kappa_mse_over_var"], A["height_term"], A["width_term"], A["cov_term_2hw"], A["top_Zmax"], A["bottom_Zmin"],
                    A["cov_top_bottom_2"], A["inner_Rmin"], A["outer_Rmax"], A["cov_inner_outer_2"]))
    lines.append("    RMSE mm: " + " ".join(f"{k} {v:.2f}" for k, v in A["rmse_mm"].items())
                 + " | median|err| mm: " + " ".join(f"{k} {v:.2f}" for k, v in A["median_abs_mm"].items())
                 + f" | top1% frames carry {A['share_of_kappa_sse_top_1pct_frames']:.2f}, top5% {A['share_of_kappa_sse_top_5pct_frames']:.2f} of SSE")

    # (b) li four-way
    m = np.isfinite(li4).all(1)
    tt, pt, tp, pp_ = li4[m].T
    F, Cn, T = pt - tt, tp - tt, pp_ - tt
    I = T - F - Cn
    vl = np.var(tt)
    B = {"n": int(m.sum()), "total_over_var": float(np.mean(T ** 2) / vl), "field_only": float(np.mean(F ** 2) / vl),
         "contour_only": float(np.mean(Cn ** 2) / vl), "interaction": float(np.mean(I ** 2) / vl),
         "cov_2FC": float(2 * np.mean(F * Cn) / vl), "cov_2FI": float(2 * np.mean(F * I) / vl), "cov_2CI": float(2 * np.mean(Cn * I) / vl),
         "corr_F_C": float(np.corrcoef(F, Cn)[0, 1]), "li_scorer_vs_li4_maxabs": float(np.nanmax(np.abs(fp[m, 6] - pp_)))}
    res["b_li"] = B
    lines.append("(b) li: 1-R2 %.4f ; field(psi_pred,C_true) %.4f ; contour(psi_true,C_pred) %.4f ; interaction %.4f ; 2cov(F,C) %.4f 2cov(F,I) %.4f 2cov(C,I) %.4f ; corr(F,C) %.2f"
                 % (B["total_over_var"], B["field_only"], B["contour_only"], B["interaction"], B["cov_2FC"], B["cov_2FI"], B["cov_2CI"], B["corr_F_C"]))

    # (c) persistent shot bias vs within-shot
    Cc = {}
    for j, n in enumerate(cc.NAMES11):
        sb = sw = 0.0
        for s in range(S):
            x = e[off[s]:off[s + 1], j]
            x = x[np.isfinite(x)]
            if len(x) == 0:
                continue
            sb += len(x) * x.mean() ** 2
            sw += ((x - x.mean()) ** 2).sum()
        Cc[n] = {"shot_bias_share": sb / (sb + sw), "within_share": sw / (sb + sw)}
    res["c_shot_bias"] = Cc
    lines.append("(c) shot-bias share of SSE: " + " ".join(f"{n} {Cc[n]['shot_bias_share']:.2f}" for n in cc.NAMES11))

    # (d) by phase
    names = (cc.FEAT / "names.txt").read_text().split("\n")
    Xr = np.load(cc.FEAT / "holdout/X_raw.npy", mmap_mode="r")
    col = lambda nm: np.asarray(Xr[:, names.index(nm)])  # noqa: E731
    up, flat, down = col("time_rampup_flag") > 0.5, col("time_flat_flag") > 0.5, col("time_rampdown_flag") > 0.5
    last = col("drv_final_decile_flag") > 0.5
    ph = {"rampup": up & ~flat, "flat": flat, "rampdown": down & ~flat & ~up}
    ph["other"] = ~(ph["rampup"] | ph["flat"] | ph["rampdown"])
    ph["last_decile(overlaps)"] = last
    fg7 = fg[:, :7]
    tot = np.array([np.nansum((fg7[:, j] - np.nanmean(fg7[:, j])) ** 2) for j in range(7)])
    sq = np.where(np.isfinite(e[:, :7]), e[:, :7] ** 2, 0.0)
    tot_psi = gsq.sum() - gsum.sum() ** 2 / (4225.0 * N)
    Dd = {}
    for k, mk in ph.items():
        per = sq[mk].sum(0) / tot
        Dd[k] = {"frame_share": float(mk.mean()), "cons_loss_S_units": float(cc.W_CONS / 7 * per.sum()),
                 "per_scalar_1mR2_contrib": {n: float(per[j]) for j, n in enumerate(cc.NAMES7)},
                 "psi_loss_S_units": float(cc.W_PSI * ssres[mk].sum() / tot_psi),
                 "mean_haus_over_R": float(np.nanmean(hd[mk] / rg[mk])) if mk.any() else float("nan")}
        lines.append(f"(d) {k:22s} frames {Dd[k]['frame_share']:.3f} cons-loss {Dd[k]['cons_loss_S_units']:.5f} psi-loss {Dd[k]['psi_loss_S_units']:.5f} "
                     f"mean d/R {Dd[k]['mean_haus_over_R']:.5f} | " + " ".join(f"{n} {per[j]:.4f}" for j, n in enumerate(cc.NAMES7)))
    res["d_phase"] = Dd
    json.dump(res, open(cc.OUT / f"diag_{a.tag}.json", "w"), indent=1)
    (cc.OUT / f"diag_{a.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
