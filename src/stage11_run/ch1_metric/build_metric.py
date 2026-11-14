#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — step 2: build the Jacobian ("metric") loss matrices from the FD shards.

    loss(e) ~ e^T M e,   e = c_pred - c_true  (PCA coefficients, stage6_H pca_80 / pca_120, orthonormal)
    M = (W_PSI / SS_tot_psi_per_frame) I  +  sum_j (W_CONS/7) alpha_j E_frames[J_j^T J_j] / Var(f_j)

Everything here uses TRAIN frames only:
  * J_j           central secants from fd_worker.py (official derivation, MACHINE=DIII-D)
  * Var(f_j)      pooled exactly as the scorer pools SS_tot (sumsq - sum^2/n over valid frames) / n,
                  f_j(psi_gt) of ALL 1,525,285 train frames (feat_v2/train/y_aux.npy cons_* columns)
  * SS_tot_psi    pooled over every pixel of every train frame (psi_train_f16.npy), divided by frames
  * step config   chosen per scalar by how well the split-half (shot-disjoint) quadratic form predicts the
                  ACTUAL scalar error of the incumbent's out-of-fold predictions on train frames
  * alpha_j       pooled calibration on the same train OOF errors (alpha_j = actual / predicted);
                  the uncalibrated (alpha=1) matrix is saved too.
Two families are built and validated side by side:
  G  (global)                 loss_i = e_i^T M e_i                          -> plain MSE on z = c @ L
  N  (amplitude-normalised)   loss_i = a_psi |e_i|^2 + |e_i @ Ln|^2 / A_i^2 -> per-sample weight 1/A_i^2
     A_i = |psi_axis - psi_boundary| of the frame's GT map (y_aux). All seven scalars are invariant under
     psi -> lambda psi + const, so J_i scales as 1/A_i exactly; E[A^2 J^T J] is the shape part.
Outputs ($OUT): M_{K}.npy L_{K}.npy Linv_{K}.npy E_{K}.npy Einv_{K}.npy M_{K}_uncal.npy M_parts_{K}.npz
                Mn_cons_{K}.npy Ln_{K}.npy amp_train.npy amp_holdout.npy
                M_lcfs_{K}.npy L_{K}_lcfs.npy Linv_{K}_lcfs.npy metric_{K}.json candidates_{K}.json
Env: WS SCR OUT NPROC(24) SKIP_PSI(0)
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

SCR = Path(os.environ["SCR"])
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/metric")))
FEAT = SCR / "stage11_ch1/feat_v2"
S10 = SCR / "stage10_ch1_cons"
NPROC = int(os.environ.get("NPROC", "24"))
W_PSI, W_QB, W_LCFS, W_CONS = 0.55, 0.15, 0.10, 0.20
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
KS = (80, 120)
AVG_SETS = {"B025+B05": ["B025", "B05"], "B05+B10": ["B05", "B10"],
            "B025+B05+B10": ["B025", "B05", "B10"], "Ball": ["B0125", "B025", "B05", "B10", "B20"]}
CROSS = {"X:B025*B05": ("B025", "B05"), "X:B05*B10": ("B05", "B10"), "X:B025*B10": ("B025", "B10")}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _psi_chunk(ab):
    a, b = ab
    p = np.asarray(np.load(S10 / "psi_train_f16.npy", mmap_mode="r")[a:b], dtype=np.float64)
    return float(p.sum()), float((p * p).sum()), float(p.size)


def pooled_psi_sstot():
    n = np.load(S10 / "psi_train_f16.npy", mmap_mode="r").shape[0]
    chunks = [(a, min(n, a + 8000)) for a in range(0, n, 8000)]
    with mp.get_context("fork").Pool(NPROC) as pool:
        r = np.array(pool.map(_psi_chunk, chunks))
    s, sq, m = r.sum(0)
    return float(sq - s * s / m), int(n), float(s / m)


def psd_clip(M, floor=0.0):
    M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    return (V * np.maximum(w, floor)) @ V.T


def cosine(absdf, q):
    m = np.isfinite(absdf) & np.isfinite(q)
    return float(np.sum(absdf[m] * np.sqrt(np.maximum(q[m], 0))) / np.sqrt(np.sum(absdf[m] ** 2) * np.sum(np.maximum(q[m], 0)) + 1e-300))


def main():
    t0 = time.time()
    shards = sorted((OUT / "fd").glob("shard_*.npz"))
    dj = [json.load(open(f)) for f in sorted((OUT / "fd").glob("deltas_*.json"))]
    assert all(d == dj[0] for d in dj), "delta files differ between shards"
    cfg_names = dj[0]["configs"]
    Z = [np.load(f) for f in shards]
    order = np.argsort(np.concatenate([z["i"] for z in Z]))

    def cat(key):
        return np.concatenate([z[key] for z in Z])[order]
    rows, shot = cat("row"), cat("shot")
    N = len(rows)
    log(f"{len(shards)} shards, {N} frames, {len(np.unique(shot))} shots; configs {cfg_names}; "
        f"core-hours {sum(float(z['sec'].sum()) for z in Z)/3600:.1f}; failed frames {sum(int(z['n_fail']) for z in Z)}")
    half = (shot % 2).astype(int)

    # ---------------- pooled statistics over the whole train population
    an = json.load(open(FEAT / "spec.json"))["aux_names"]
    cols = [an.index("cons_" + c) for c in NAMES]
    aux = np.load(FEAT / "train/y_aux.npy", mmap_mode="r")
    fg_all = np.asarray(aux[:, cols], dtype=np.float64)
    fin = np.isfinite(fg_all)
    n_j = fin.sum(0)
    s_j = np.where(fin, fg_all, 0).sum(0); sq_j = np.where(fin, fg_all ** 2, 0).sum(0)
    var_j = (sq_j - s_j ** 2 / n_j) / n_j                      # SS_tot_j / n_j, scorer pooling
    mean_j = s_j / n_j
    f_gt = fg_all[rows]
    var_sample = np.nanvar(f_gt, 0)
    amp_all = np.abs(np.asarray(aux[:, an.index("psi_axis")], np.float64) - np.asarray(aux[:, an.index("psi_boundary")], np.float64))
    np.save(OUT / "amp_train.npy", amp_all.astype(np.float32))
    auxh = np.load(FEAT / "holdout/y_aux.npy")
    np.save(OUT / "amp_holdout.npy", np.abs(auxh[:, an.index("psi_axis")].astype(np.float64) - auxh[:, an.index("psi_boundary")].astype(np.float64)).astype(np.float32))
    amp = amp_all[rows]
    aok = np.isfinite(amp) & (amp > 1e-3)
    names_x = (FEAT / "names.txt").read_text().split()
    try:
        xr = np.load(FEAT / "train/X_raw.npy", mmap_mode="r")
        ipabs = np.asarray(xr[rows, names_x.index("time_ip_frac")], np.float64) * np.asarray(xr[rows, names_x.index("time_ip_max_kA")], np.float64)
        mm_ = aok & np.isfinite(ipabs) & (ipabs > 0)
        ip_info = {"corr_log_amp_log_ip": float(np.corrcoef(np.log(amp[mm_]), np.log(ipabs[mm_]))[0, 1]),
                   "sd_log_amp_over_ip": float(np.std(np.log(amp[mm_] / ipabs[mm_]))), "sd_log_amp": float(np.std(np.log(amp[mm_])))}
    except Exception as ex:  # noqa: BLE001
        ip_info = {"error": str(ex)}
    log(f"flux amplitude A=|psi_axis-psi_b|: train finite {np.isfinite(amp_all).mean():.5f}, quantiles(1,10,50,90,99%) "
        + " ".join(f"{v:.4f}" for v in np.nanquantile(amp_all, [.01, .1, .5, .9, .99])) + f"; vs |Ip|: {json.dumps(ip_info)}")
    if os.environ.get("SKIP_PSI", "0") == "1":
        ss = cat("ss_gt"); s, sq, m = ss.sum(0)
        ss_tot, n_fr, mean_psi = float(sq - s * s / m) * (fg_all.shape[0] / N), fg_all.shape[0], float(s / m)
    else:
        ss_tot, n_fr, mean_psi = pooled_psi_sstot()
    ss_frame = ss_tot / n_fr
    ss = cat("ss_gt"); s, sq, m = ss.sum(0)
    ss_frame_sample = float(sq - s * s / m) / N
    a_psi = W_PSI / ss_frame
    log(f"pooled train stats: SS_tot_psi/frame = {ss_frame:.4f} (sample {ss_frame_sample:.4f}), mean_psi {mean_psi:.5f}, "
        f"a_psi = {a_psi:.6e}")
    log("Var(f_j) pooled train: " + ", ".join(f"{n}={v:.6g}" for n, v in zip(NAMES, var_j)))
    log("Var(f_j) sample      : " + ", ".join(f"{n}={v:.6g}" for n, v in zip(NAMES, var_sample)))
    w_j = (W_CONS / 7.0) / var_j

    for K in KS:
        log(f"================ K = {K}")
        J = cat(f"J_{K}").astype(np.float64)                   # (N,G,7,K)
        f0 = cat(f"f0_{K}")
        valid = np.isfinite(f0) & np.isfinite(f_gt) & aok[:, None]   # (N,7)
        nanJ = {c: float(np.isnan(J[:, g][valid]).mean()) for g, c in enumerate(cfg_names)}
        Jz = np.where(np.isfinite(J), J, 0.0) * valid[:, None, :, None]
        e = cat(f"e_oof_{K}").astype(np.float64)
        df = cat(f"f_oof_{K}") - f0
        ev = cat(f"ev_{K}").astype(np.float64); dfv = cat(f"fv_{K}") - f0[:, None, :]
        vok = valid & np.isfinite(df)
        trunc = np.nanmean((f0 - f_gt) ** 2, 0) / var_j
        log("truncation floor (f(dec c_true) - f(psi_gt))^2/Var: " + " ".join(f"{v:.4f}" for v in trunc))
        log(f"train OOF incumbent: rho = {np.sqrt((e**2).sum(1).mean()):.5f}; f_oof failure rate "
            + " ".join(f"{v:.4f}" for v in (valid & ~np.isfinite(df)).sum(0) / valid.sum(0)))
        act = np.array([np.mean(df[vok[:, j], j] ** 2) for j in range(7)])
        log("ACTUAL OOF model-error loss E[(f(c+e)-f(c))^2]/Var: " + " ".join(f"{v:.4f}" for v in act / var_j))

        cands = {}
        for g, c in enumerate(cfg_names):
            cands[c] = Jz[:, g]
        for nm, mem in AVG_SETS.items():
            if all(m_ in cfg_names for m_ in mem):
                cands[nm] = np.mean([Jz[:, cfg_names.index(m_)] for m_ in mem], axis=0)

        ampz = np.where(aok, amp, 1.0)

        def mats(Jc, sel, p=0):                                # (7,K,K) mean_i A^p J^T J over frames in sel
            out = []
            for j in range(7):
                s_ = sel & valid[:, j]
                Js = Jc[s_, j] * (ampz[s_, None] ** (p / 2.0))
                out.append(Js.T @ Js / max(1, s_.sum()))
            return np.stack(out)

        def xmats(Ja, Jb, sel, p=0):
            out = []
            for j in range(7):
                s_ = sel & valid[:, j]
                out.append(psd_clip((Ja[s_, j] * ampz[s_, None] ** p).T @ Jb[s_, j] / max(1, s_.sum())))
            return np.stack(out)

        allsel = np.ones(N, bool)
        FAM = {"": 0, "N:": 2}
        Mfull, Mhalf, PEXP, JREF = {}, {}, {}, {}
        for pre, p in FAM.items():
            for nm, Jc in cands.items():
                if pre and nm in ("A1", "B0125", "B20", "Ball"):
                    continue
                Mfull[pre + nm] = mats(Jc, allsel, p); Mhalf[pre + nm] = [mats(Jc, half == h, p) for h in (0, 1)]
                PEXP[pre + nm] = p; JREF[pre + nm] = nm
            for nm, (ca, cb) in CROSS.items():
                if ca in cfg_names and cb in cfg_names and not (pre and nm != "X:B025*B05"):
                    Ja, Jb = Jz[:, cfg_names.index(ca)], Jz[:, cfg_names.index(cb)]
                    Mfull[pre + nm] = xmats(Ja, Jb, allsel, p); Mhalf[pre + nm] = [xmats(Ja, Jb, half == h, p) for h in (0, 1)]
                    PEXP[pre + nm] = p
        if "B05" in cands:                                     # exponent diagnostic: p = 1
            Mfull["P1:B05"] = mats(cands["B05"], allsel, 1); Mhalf["P1:B05"] = [mats(cands["B05"], half == h, 1) for h in (0, 1)]
            PEXP["P1:B05"] = 1; JREF["P1:B05"] = "B05"

        def qform(Mh, E, p=0):                                 # split-half: frame in half h uses M of half 1-h
            q = np.empty((N, 7))
            for h in (0, 1):
                s_ = half == h
                q[s_] = np.einsum("nk,jkl,nl->nj", E[s_], Mh[1 - h], E[s_], optimize=True)
            return q / (ampz[:, None] ** p)

        e2 = (e ** 2).sum(1)
        table = {}
        for nm in Mfull:
            q = qform(Mhalf[nm], e, PEXP[nm])
            rec = {"avg_over_var": [], "ratio_pred_over_actual": [], "cos": [], "spearman": [],
                   "lin_over_var": [], "r2_signed": [], "corr_signed": [],
                   "synth_ratio_s1.0": [], "synth_ratio_s0.7": [], "synth_cos_s1.0": []}
            lin = np.einsum("njk,nk->nj", cands[nm], e) if nm in cands else None
            qv = np.stack([qform(Mhalf[nm], ev[:, v], PEXP[nm]) for v in range(ev.shape[1])], 1)   # (N,NV,7)
            for j in range(7):
                m_ = vok[:, j]
                rec["avg_over_var"].append(float(q[m_, j].mean() / var_j[j]))
                rec["ratio_pred_over_actual"].append(float(q[m_, j].mean() / act[j]))
                rec["cos"].append(cosine(np.abs(df[m_, j]), q[m_, j]))
                rec["spearman"].append(float(spearmanr(df[m_, j] ** 2, q[m_, j]).statistic))
                if lin is not None:
                    rec["lin_over_var"].append(float(np.mean(lin[m_, j] ** 2) / var_j[j]))
                    rec["r2_signed"].append(float(1 - np.sum((df[m_, j] - lin[m_, j]) ** 2) / np.sum(df[m_, j] ** 2)))
                    rec["corr_signed"].append(float(np.corrcoef(df[m_, j], lin[m_, j])[0, 1]))
                for tag, vs in (("s1.0", [0, 1]), ("s0.7", [2, 3])):
                    vs = [v for v in vs if v < ev.shape[1]]
                    if not vs:
                        continue
                    mm = valid[:, j][:, None] & np.isfinite(dfv[:, vs, j])
                    rec[f"synth_ratio_{tag}"].append(float(qv[:, vs, j][mm].mean() / np.mean(dfv[:, vs, j][mm] ** 2)))
                    if tag == "s1.0":
                        rec["synth_cos_s1.0"].append(cosine(np.abs(dfv[:, vs, j][mm]), qv[:, vs, j][mm]))
            table[nm] = rec
        base = {"cos": [cosine(np.abs(df[vok[:, j], j]), e2[vok[:, j]]) for j in range(7)],
                "spearman": [float(spearmanr(df[vok[:, j], j] ** 2, e2[vok[:, j]]).statistic) for j in range(7)]}
        log("candidate table (train OOF errors, split-half shot-disjoint M). columns = " + " ".join(NAMES))
        for key in ("ratio_pred_over_actual", "cos", "spearman", "r2_signed", "synth_ratio_s1.0", "synth_ratio_s0.7"):
            log(f"  -- {key}")
            for nm, rec in table.items():
                if rec[key]:
                    log(f"     {nm:14s} " + " ".join(f"{v:7.3f}" for v in rec[key]))
            if key in base:
                log(f"     {'identity |e|^2':14s} " + " ".join(f"{v:7.3f}" for v in base[key]))
        log("  nan fraction of J entries per config: " + json.dumps({k: round(v, 5) for k, v in nanJ.items()}))

        # ---------------- selection (train only): best split-half cosine per scalar, within each family
        # A1 / B0125 are excluded on principle: their secants over tiny steps are dominated by the non-smooth
        # (contour-quantisation) part of kappa/tri and inflate the trailing-coefficient weights (confirmed by the
        # holdout intervention I_removeRand42). Default = ONE fixed config for all scalars (SEL_G / SEL_N);
        # "auto" = best split-half cosine per scalar.
        famG = [nm for nm in table if PEXP[nm] == 0 and nm not in ("A1", "B0125")]
        famN = [nm for nm in table if PEXP[nm] == 2]
        SEL_G = os.environ.get("SEL_G", "B05+B10"); SEL_N = os.environ.get("SEL_N", "N:B05+B10")
        sel = [SEL_G] * 7 if SEL_G != "auto" else [max(famG, key=lambda nm: table[nm]["cos"][j]) for j in range(7)]
        auto_g = [max(famG, key=lambda nm: table[nm]["cos"][j]) for j in range(7)]
        log("G best-by-cos per scalar (info): " + ", ".join(f"{n}:{s_}({table[s_]['cos'][j]:.3f})" for j, (n, s_) in enumerate(zip(NAMES, auto_g))))
        alpha = np.array([1.0 / table[sel[j]]["ratio_pred_over_actual"][j] for j in range(7)])
        seln = [SEL_N] * 7 if SEL_N != "auto" else [max(famN, key=lambda nm: table[nm]["cos"][j]) for j in range(7)]
        alphan = np.array([1.0 / table[seln[j]]["ratio_pred_over_actual"][j] for j in range(7)])
        log("G selected config per scalar: " + ", ".join(f"{n}:{s_}" for n, s_ in zip(NAMES, sel)))
        log("G alpha_j (actual/predicted pooled, train OOF): " + " ".join(f"{v:.3f}" for v in alpha))
        log("G cos " + " ".join(f"{table[sel[j]]['cos'][j]:.3f}" for j in range(7)) + " | spearman " + " ".join(f"{table[sel[j]]['spearman'][j]:.3f}" for j in range(7)))
        log("N selected config per scalar: " + ", ".join(f"{n}:{s_}" for n, s_ in zip(NAMES, seln)))
        log("N alpha_j (actual/predicted pooled, train OOF): " + " ".join(f"{v:.3f}" for v in alphan))
        log("N cos " + " ".join(f"{table[seln[j]]['cos'][j]:.3f}" for j in range(7)) + " | spearman " + " ".join(f"{table[seln[j]]['spearman'][j]:.3f}" for j in range(7)))
        Mj = np.stack([Mfull[sel[j]][j] for j in range(7)])                    # E[J^T J] (7,K,K)
        parts = Mj * w_j[:, None, None]                                         # weighted, uncalibrated
        I = np.eye(K)
        M_uncal = a_psi * I + parts.sum(0)
        M = a_psi * I + (parts * alpha[:, None, None]).sum(0)
        M = 0.5 * (M + M.T); M_uncal = 0.5 * (M_uncal + M_uncal.T)

        def factor(Mx):
            w, V = np.linalg.eigh(Mx)
            assert w.min() > 0, w.min()
            L = np.linalg.cholesky(Mx)                                          # M = L L^T ; z = c @ L
            Linv = np.linalg.inv(L)
            o = np.argsort(w)[::-1]
            E = V[:, o] * np.sqrt(w[o]); Einv = (V[:, o] / np.sqrt(w[o])).T      # z = c @ E ; c = z @ Einv
            return w[o], L, Linv, E, Einv
        w, L, Linv, E, Einv = factor(M)
        rng = np.random.default_rng(0)
        x = rng.standard_normal((5, K))
        chk1 = float(np.max(np.abs(((x @ L) ** 2).sum(1) - np.einsum("nk,kl,nl->n", x, M, x)) / np.einsum("nk,kl,nl->n", x, M, x)))
        chk2 = float(np.max(np.abs((x @ L) @ Linv - x)))
        chk3 = float(np.max(np.abs(((x @ E) ** 2).sum(1) - np.einsum("nk,kl,nl->n", x, M, x)) / np.einsum("nk,kl,nl->n", x, M, x)))
        log(f"M: min eig {w.min():.4e} max eig {w.max():.4e} cond {w.max()/w.min():.3e} (psi floor a={a_psi:.4e}); "
            f"|z|^2 == e^T M e rel err {chk1:.2e} (chol) {chk3:.2e} (eig); inverse round trip {chk2:.2e}")

        # predicted loss decomposition on the train OOF errors (full-sample M; in-sample but M has << dof than data)
        qj_cal = np.einsum("nk,jkl,nl->nj", e, parts * alpha[:, None, None], e, optimize=True).mean(0)
        qj_unc = np.einsum("nk,jkl,nl->nj", e, parts, e, optimize=True).mean(0)
        psi_term = a_psi * e2.mean()
        act_w = (W_CONS / 7.0) * act / var_j
        log(f"train-OOF predicted score loss: psi {psi_term:.6f} (=0.55*(1-R2psi), 1-R2psi={e2.mean()/ss_frame:.6f}); "
            f"consistency predicted cal {qj_cal.sum():.6f} uncal {qj_unc.sum():.6f} actual {act_w.sum():.6f}")
        # eigen-direction contributions on OOF (for the holdout intervention test)
        wc, Vc = np.linalg.eigh(0.5 * ((parts * alpha[:, None, None]).sum(0) + (parts * alpha[:, None, None]).sum(0).T))
        proj2 = ((e @ Vc) ** 2).mean(0)
        contrib = wc * proj2
        o = np.argsort(contrib)[::-1]
        cum = np.cumsum(contrib[o]) / contrib.sum()
        cum_e = np.cumsum(proj2[o]) / proj2.sum()
        m80 = int(np.searchsorted(cum, 0.80) + 1)
        log(f"consistency-metric eigen-directions ranked by OOF loss contribution: top {m80} carry 80% of the predicted "
            f"consistency loss and {cum_e[m80-1]:.3f} of the error energy; top5 {cum[4]:.3f} top10 {cum[9]:.3f} top20 {cum[19]:.3f}")

        # ---------------- family N: amplitude-normalised consistency metric
        Mnj = np.stack([Mfull[seln[j]][j] for j in range(7)])                   # E[A^2 J^T J] (7,K,K)
        partsn = Mnj * (w_j * alphan)[:, None, None]
        Mn_cons = 0.5 * (partsn.sum(0) + partsn.sum(0).T)
        wn, Vn = np.linalg.eigh(Mn_cons)
        wn = np.maximum(wn, 0.0)
        on = np.argsort(wn)[::-1]
        Ln = Vn[:, on] * np.sqrt(wn[on])                                        # |e @ Ln|^2 = e^T Mn_cons e
        en = e / ampz[:, None]
        qn = np.einsum("nk,jkl,nl->nj", en, partsn, en, optimize=True)
        qn_mean = np.array([qn[valid[:, j], j].mean() for j in range(7)])
        projn = ((en[aok] @ Vn[:, on]) ** 2).mean(0)
        contribn = wn[on] * projn
        o2 = np.argsort(contribn)[::-1]
        cumn = np.cumsum(contribn[o2]) / contribn.sum()
        m80n = int(np.searchsorted(cumn, 0.80) + 1)
        tot_g = np.einsum("nk,jkl,nl->n", e, parts * alpha[:, None, None], e, optimize=True)
        tot_n = qn.sum(1)
        act_tot = np.nansum(np.where(vok, df ** 2, np.nan) * w_j[None, :], axis=1)
        mt = vok.all(1)
        fam_cmp = {"G_cos_total": cosine(np.sqrt(act_tot[mt]), tot_g[mt]), "N_cos_total": cosine(np.sqrt(act_tot[mt]), tot_n[mt]),
                   "I_cos_total": cosine(np.sqrt(act_tot[mt]), e2[mt]),
                   "G_spearman_total": float(spearmanr(act_tot[mt], tot_g[mt]).statistic), "N_spearman_total": float(spearmanr(act_tot[mt], tot_n[mt]).statistic),
                   "I_spearman_total": float(spearmanr(act_tot[mt], e2[mt]).statistic),
                   "G_pred_mean": float(tot_g[mt].mean()), "N_pred_mean": float(tot_n[mt].mean()), "actual_mean": float(act_tot[mt].mean())}
        log("per-frame TOTAL weighted consistency loss, predicted vs actual on train OOF (full-sample M): " + json.dumps({k: round(v, 5) for k, v in fam_cmp.items()}))
        log(f"N: top {m80n} eigen-directions of Mn_cons carry 80% of the predicted loss; eig max {wn.max():.4e}, "
            f"#eig > 1e-6*max: {(wn > 1e-6 * wn.max()).sum()}")
        np.save(OUT / f"Mn_cons_{K}.npy", Mn_cons); np.save(OUT / f"Ln_{K}.npy", Ln)

        # ---------------- optional D_LCFS surrogate (first-order separatrix displacement)
        H = np.sum([z[f"Hsum_{K}"] for z in Z], 0) / sum(float(z[f"hok_{K}"].sum()) for z in Z)
        H = 0.5 * (H + H.T)
        d_oof = cat(f"d_oof_{K}"); d_gt = cat(f"d_oof_gt_{K}"); d_tr = cat(f"d_trunc_{K}")
        qH = np.einsum("nk,kl,nl->n", e, H, e, optimize=True)
        mH = np.isfinite(d_oof)
        gamma2 = float(np.mean(d_oof[mH] ** 2) / np.mean(qH[mH]))
        dbar = float(np.mean(d_oof[mH]))
        M_lcfs = (W_LCFS * gamma2 / (2.0 * dbar)) * H
        lc = cat(f"lc_{K}")
        lcfs_info = {"gamma2": gamma2, "d_bar_model_error_vs_truncated_truth": dbar,
                     "d_bar_vs_gt": float(np.nanmean(d_gt)), "d_truncation_floor": float(np.nanmean(d_tr)),
                     "cos_d_vs_sqrt_eHe": cosine(d_oof[mH], qH[mH]), "spearman_d_vs_eHe": float(spearmanr(d_oof[mH], qH[mH]).statistic),
                     "cos_d_vs_identity": cosine(d_oof[mH], e2[mH]), "spearman_d_vs_identity": float(spearmanr(d_oof[mH], e2[mH]).statistic),
                     "spearman_d_vs_perframe_linmax": float(spearmanr(d_oof[mH], lc[mH, 0, 1]).statistic),
                     "weight_formula": "M_lcfs = W_LCFS * gamma2 / (2 d_bar) * H ; 0.10*D ~ 0.10*(d^2/(2 d_bar) + d_bar/2), d^2 ~ gamma2 e^T H e",
                     "predicted_lcfs_loss_on_oof": float(np.mean(np.einsum("nk,kl,nl->n", e, M_lcfs, e)))}
        log("LCFS surrogate: " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in lcfs_info.items() if k != "weight_formula"}))
        Mx = 0.5 * ((M + M_lcfs) + (M + M_lcfs).T)
        wx, Lx, Lxinv, _, _ = factor(Mx)

        # z-space target statistics (train subsample) for training-normalisation advice
        yc = np.asarray(np.load(FEAT / f"train/y_C{K}.npy", mmap_mode="r")[::25], dtype=np.float64)
        zt = yc @ L
        zsd = zt.std(0)
        ze = e @ L
        log(f"z = c @ L targets: total var {zt.var(0).sum():.4f}; OOF error |dz|^2 mean {np.mean((ze**2).sum(1)):.6f} "
            f"(= predicted loss); R2 in z-space of incumbent OOF {1 - (ze**2).sum(1).mean()/zt.var(0).sum():.5f}")

        np.save(OUT / f"M_{K}.npy", M); np.save(OUT / f"L_{K}.npy", L); np.save(OUT / f"Linv_{K}.npy", Linv)
        np.save(OUT / f"E_{K}.npy", E); np.save(OUT / f"Einv_{K}.npy", Einv)
        np.save(OUT / f"M_{K}_uncal.npy", M_uncal)
        np.save(OUT / f"M_lcfs_{K}.npy", M_lcfs); np.save(OUT / f"L_{K}_lcfs.npy", Lx); np.save(OUT / f"Linv_{K}_lcfs.npy", Lxinv)
        np.savez(OUT / f"M_parts_{K}.npz", EJtJ=Mj, w_j=w_j, alpha=alpha, a_psi=a_psi, names=np.array(NAMES),
                 selected=np.array(sel), H_lcfs=H, eig_cons_vecs=Vc[:, o], eig_cons_contrib=contrib[o],
                 EJtJn=Mnj, alphan=alphan, selectedn=np.array(seln), eign_cons_vecs=Vn[:, on][:, o2], eign_cons_contrib=contribn[o2],
                 **{f"EJtJ_{nm.replace(':', '_').replace('*', 'x').replace('+', '_')}": Mfull[nm] for nm in Mfull})
        meta = {
            "K": K, "basis": f"stage6_H/pca_{K}.npz (mean_, comps_ orthonormal rows; psi = mean_ + c @ comps_)",
            "definition": "loss(e) = e^T M e per frame, e = c_pred - c_true; M = a_psi I + sum_j (W_CONS/7) alpha_j E[J_j^T J_j] / Var_j",
            "weights": {"W_PSI": W_PSI, "W_CONS": W_CONS, "W_LCFS": W_LCFS},
            "scalars": NAMES, "var_pooled_train": var_j.tolist(), "mean_pooled_train": mean_j.tolist(),
            "n_valid_train_frames_per_scalar": n_j.tolist(), "var_fd_sample": var_sample.tolist(),
            "ss_tot_psi_per_frame_train": ss_frame, "ss_tot_psi_per_frame_fd_sample": ss_frame_sample,
            "mean_psi_train": mean_psi, "n_train_frames": n_fr, "a_psi": a_psi,
            "fd": {"n_frames": int(N), "n_shots": int(len(np.unique(shot))), "seed": dj[0]["seed"],
                   "rows_sha1": hashlib.sha1(rows.astype(np.int64).tobytes()).hexdigest(),
                   "configs": cfg_names, "delta_source": dj[0][f"K{K}"]["source"], "delta_split": dj[0][f"K{K}"]["split"],
                   "rho": dj[0][f"K{K}"]["rho"], "rms_k": dj[0][f"K{K}"]["rms"],
                   "delta_by_config": {c: (dj[0][f"K{K}"]["configs"][c][0] if c != "A1" else "rms_k") for c in cfg_names},
                   "nan_fraction_J": nanJ},
            "selected_config_per_scalar": dict(zip(NAMES, sel)), "alpha": dict(zip(NAMES, alpha.tolist())),
            "truncation_floor_over_var": dict(zip(NAMES, trunc.tolist())),
            "train_oof": {"rho": float(np.sqrt(e2.mean())), "actual_over_var": dict(zip(NAMES, (act / var_j).tolist())),
                          "pred_cal_weighted": dict(zip(NAMES, qj_cal.tolist())), "pred_uncal_weighted": dict(zip(NAMES, qj_unc.tolist())),
                          "actual_weighted": dict(zip(NAMES, act_w.tolist())), "psi_term": float(psi_term),
                          "identity_baseline": base},
            "eig": {"min": float(w.min()), "max": float(w.max()), "cond": float(w.max() / w.min()),
                    "quantiles_0_10_50_90_100": np.quantile(w, [0, .1, .5, .9, 1]).tolist(),
                    "n_dirs_for_80pct_cons_loss": m80, "energy_frac_in_those_dirs": float(cum_e[m80 - 1])},
            "checks": {"chol_quadratic_rel_err": chk1, "eig_quadratic_rel_err": chk3, "inverse_roundtrip_abs_err": chk2},
            "z_space": {"target_total_var": float(zt.var(0).sum()), "target_sd_min_med_max": [float(zsd.min()), float(np.median(zsd)), float(zsd.max())]},
            "lcfs_addon": lcfs_info,
            "family_N": {"definition": "loss_i = a_psi |e_i|^2 + |e_i @ Ln|^2 / A_i^2 ; Ln Ln^T = Mn_cons = sum_j (W_CONS/7) alphan_j E[A^2 J_j^T J_j] / Var_j ; A_i = |psi_axis - psi_boundary| (amp_train.npy / amp_holdout.npy, from y_aux)",
                         "selected_config_per_scalar": dict(zip(NAMES, seln)), "alpha": dict(zip(NAMES, alphan.tolist())),
                         "pred_cal_weighted_train_oof": dict(zip(NAMES, qn_mean.tolist())), "n_dirs_for_80pct_cons_loss": m80n,
                         "eig_max": float(wn.max()), "amp_vs_ip": ip_info,
                         "amp_quantiles_1_10_50_90_99": np.nanquantile(amp_all, [.01, .1, .5, .9, .99]).tolist()},
            "family_comparison_train_oof": fam_cmp,
            "files": {"M": f"M_{K}.npy", "L": f"L_{K}.npy (lower Cholesky, M = L L^T, z = c @ L)", "Linv": f"Linv_{K}.npy (c = z @ Linv)",
                      "E": f"E_{K}.npy (eig factor, z = c @ E, columns by decreasing eigenvalue)", "Einv": f"Einv_{K}.npy"},
        }
        json.dump(meta, open(OUT / f"metric_{K}.json", "w"), indent=1)
        json.dump({"names": NAMES, "table": table, "identity": base}, open(OUT / f"candidates_{K}.json", "w"), indent=1)
        del J, Jz, cands
    log(f"build_metric done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
