#!/usr/bin/env python3
"""Stage 11d / ch1d_specialists — Gauss-Newton map correction (fork of ch1_corrector/cc_apply.py) with an extra
METRIC-COORDINATE mode (external review, follow-up 1, internal note):

  --prior oof    (cc_apply behaviour) Sigma = shrunk OOF coefficient-error covariance, R = tau * diag(s_j^2)
  --prior metric Sigma = M0^-1 (M0 = stage11 metric M_80, score units per frame) and, with --ridge trace,
                 K = W^1/2 J M0^-1/2, r = W^1/2 (d - f), z = eta K^T (K K^T + lam I)^-1 r, dc = M0^-1/2 z,
                 W_jj = 0.20 / (7 sigma_j^2) (train variances; extrema get --wext times that), lam = tr(K K^T) / ndiv
                 (per frame, ndiv = 7), which is the same as  dc = eta M0^-1 J^T (J M0^-1 J^T + lam W^-1)^-1 d.
                 Trust region (--rs): ||M0^1/2 dc|| <= r * sqrt(tr(M0 Sigma_oof)), i.e. r in units of the train-OOF RMS
                 coefficient error measured in the M0 norm (r = 0.25 = the review's cap; 1e9 = off).
Everything else (re-extraction with the official code, guard/fallback, frozen-J corrective steps, re-linearisation,
emulated official composite on all / A / B, file writing) is unchanged from cc_apply.py.  Extra: --save-f stores the
official scalars of every written config (f11_holdout_<stem>.npy) for the innovation diagnostic.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time

import numpy as np

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ch1_corrector"))
import cc_common as cc  # noqa: E402
from cc_common import log  # noqa: E402
from cc_jac import get_psi, load_maps  # noqa: E402

_D = {}


def gain(Ja, act):
    """G = Sigma J^T (J Sigma J^T + R)^-1 ; R = tau diag(s^2) (ridge tau) or lam W^-1, lam = tr(W J Sigma J^T)/ndiv (ridge trace)."""
    D = _D
    SJt = D["Sig"] @ Ja.T
    JSJ = Ja @ SJt
    if D["ridge"] == "trace":
        w = D["W"][act]
        lam = float((w * np.diag(JSJ)).sum() / D["ndiv"])
        if not lam > 0:
            raise np.linalg.LinAlgError("zero Jacobian")
        Rv = lam / w
    else:
        Rv = D["Rd"][act]
    return SJt @ np.linalg.inv(JSJ + np.diag(Rv))


def work(ab):
    a, b = ab
    cc.init_scorer()
    D = _D
    comps, Sig, Sinv, Rd, sd, cfgs, act_all = D["comps"], D["Sig"], D["Sinv"], D["Rd"], D["sd"], D["cfgs"], D["act"]
    out = []
    for i in range(a, b):
        psi = get_psi(D, i)
        f0 = D["f0"][i]; d = D["dhat"][i]            # f0: raw-li version (targets / residuals); official one decides scoring
        act = [j for j in act_all if np.isfinite(d[j]) and np.isfinite(f0[j])]
        if not np.isfinite(f0[:7]).all() or not act:
            out.append((i, None, 0.0)); continue
        base_ok = bool(np.isfinite(D["f0o"][i][:7]).all())
        Ja = D["J"][i].astype(np.float64)[act]
        try:
            Gm = gain(Ja, act)
        except np.linalg.LinAlgError:
            out.append((i, None, 0.0)); continue
        da = d[act]; sj = sd[act]
        df = Gm @ da
        mn = float(np.sqrt(max(df @ D["Nm"] @ df, 0.0) / D["Nden"]))
        cache, res = {}, {}
        gt = np.asarray(D["gt"][i], np.float64) if "gt" in D else None

        def ev(dl, tgt):
            pn = psi + (dl @ comps).reshape(65, 65)
            fo, fr, Cn = cc.dpred_raw(pn)
            e = np.sum(((tgt - fr[act]) / sj) ** 2) if np.isfinite(fr[act]).all() else np.inf
            return [dl, pn, fo, fr, Cn, e]

        def iterate(st, G, tgt, n, cap):                      # corrective steps toward the same target with a fixed gain G
            for _ in range(n):
                if not np.isfinite(st[5]):
                    break
                d2 = G @ (tgt - st[3][act])
                n2 = np.linalg.norm(d2)
                if n2 > cap > 0:
                    d2 *= cap / n2
                s2 = ev(st[0] + d2, tgt)
                if s2[5] < st[5]:
                    st = s2
                else:
                    break
            return st

        def jac_at(pn):                                        # re-linearise around the current map (160 derivations)
            Jn = np.zeros((len(act), cc.K))
            _, f, _ = cc.dpred_raw(pn)
            for k in range(cc.K):
                h = D["steps"][k]
                pk = comps[k].reshape(65, 65) * h
                p_, m_ = cc.dpred_raw(pn + pk)[1][act], cc.dpred_raw(pn - pk)[1][act]
                op, om = np.isfinite(p_), np.isfinite(m_)
                col = np.where(op & om, (p_ - m_) / (2 * h), np.where(op, (p_ - f[act]) / h, np.where(om, (f[act] - m_) / h, 0.0)))
                col[~np.isfinite(col)] = 0.0
                Jn[:, k] = col
            return Jn

        for ci, (r, eta, it) in enumerate(cfgs):
            sc = eta * min(1.0, r / mn) if mn > 0 else 0.0
            key = (round(sc, 9), it)
            if key not in cache:
                tgt = f0[act] + sc * da
                st = ev(sc * df, tgt)
                n1 = np.linalg.norm(st[0])
                st = iterate(st, Gm, tgt, it % 10 - 1, n1)
                nre = 0
                for _ in range(it // 10):                     # relinearise on demand: failed or still far from the target
                    if np.isfinite(st[5]) and st[5] <= len(act) * D["relin_thr"] ** 2:
                        break
                    if not np.isfinite(st[5]):
                        for h in (0.5, 0.25):
                            s_ = ev(h * sc * df, tgt)
                            if np.isfinite(s_[5]):
                                st = s_
                                break
                        if not np.isfinite(st[5]):
                            break
                    J2 = jac_at(st[1]); nre += 1
                    try:
                        G2 = gain(J2, act)
                    except np.linalg.LinAlgError:
                        break
                    st = iterate(st, G2, tgt, 3, n1)
                dl, pn, fo, fr, Cn, e1 = st
                ok = bool(np.isfinite(e1)) and (bool(np.isfinite(fo[:7]).all()) or not base_ok)
                if ok and D["guard"] >= 0:      # farther from the target than before the step (beyond a noise floor)
                    ok = e1 <= max(np.sum((sc * da / sj) ** 2), len(act) * D["guard"] ** 2)
                if not ok:
                    cache[key] = (None, nre)
                else:
                    hd = ss = np.nan
                    if gt is not None:
                        cg = D["Cg"][i]
                        hd = cc.haus(Cn, cg) if np.isfinite(cg[0, 0]) else np.nan
                        rr = gt - pn
                        ss = float((rr * rr).sum())
                    cache[key] = ((fo, hd, ss, dl.astype(np.float32)), nre)
            res[ci] = cache[key]
        out.append((i, res, mn))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="holdout")
    ap.add_argument("--tag", required=True, help="base tag (jac_<split>_<tag>.npz, der_hold_<tag>.npz)")
    ap.add_argument("--pred", default=None); ap.add_argument("--coef", default=None)
    ap.add_argument("--dhat", required=True, help="npy (N,11) innovation estimate (a+b = mean of files), or 'oracle'")
    ap.add_argument("--direct", action="store_true", help="--dhat file holds direct estimates T; dhat = lam * (T - f0)")
    ap.add_argument("--lam", default=None, help="11 comma-separated multipliers on dhat")
    ap.add_argument("--name", required=True)
    ap.add_argument("--cov", required=True); ap.add_argument("--dsc", required=True)
    ap.add_argument("--targets", default="11")
    ap.add_argument("--rs", default="0.1,0.25,0.5"); ap.add_argument("--etas", default="0.25,0.5,1")
    ap.add_argument("--iters", default="1")
    ap.add_argument("--tau", type=float, default=0.05); ap.add_argument("--shrink", type=float, default=0.1)
    ap.add_argument("--guard", type=float, default=0.5, help="<0 off; else fallback if target residual > max(before, n_act*guard^2) in prior-RMS units")
    ap.add_argument("--relin-thr", type=float, default=0.3, help="iters >= 10: re-linearise when the target residual RMS exceeds this (prior-RMS units)")
    ap.add_argument("--qb", default=None, help="harness-layout npz to take q95/betaN from (when --coef is used)")
    ap.add_argument("--write", default="", help="'sel' = configs chosen on A, on B and on all; or comma list of config keys")
    ap.add_argument("--workers", type=int, default=70); ap.add_argument("--maxf", type=int, default=0)
    ap.add_argument("--prior", default="oof", choices=["oof", "metric"])
    ap.add_argument("--ridge", default="tau", choices=["tau", "trace"])
    ap.add_argument("--metric", default=str(cc.MET / "M_80.npy"))
    ap.add_argument("--wext", type=float, default=0.0, help="trace ridge: weight of the 4 LCFS extrema relative to 0.20/(7 sigma^2)")
    ap.add_argument("--ndiv", type=float, default=7.0)
    ap.add_argument("--save-f", action="store_true")
    ap.add_argument("--prefix", default="", help="extra name part for written files")
    a = ap.parse_args()
    t0 = time.time()
    D = _D
    D.update(load_maps(a.split, a.pred, a.coef))
    jz = np.load(cc.OUT / f"jac_{a.split}_{a.tag}{'_smoke' if a.maxf else ''}.npz")
    D["f0"], D["J"] = jz["f0"], jz["J"]
    D["f0o"] = jz["f0_off"] if "f0_off" in jz.files else jz["f0"]
    D["steps"] = jz["steps"].astype(np.float64); D["relin_thr"] = a.relin_thr
    if a.qb:
        zq = np.load(a.qb)
        nsh = len(D["off"]) - 1
        D["q"] = np.concatenate([zq[f"shot_{i:04d}_q95"] for i in range(nsh)]).astype(np.float64)
        D["b"] = np.concatenate([zq[f"shot_{i:04d}_betaN"] for i in range(nsh)]).astype(np.float64)
        assert len(D["q"]) == D["off"][-1]
    N = D["f0"].shape[0]
    ref = None
    if a.split == "holdout":
        ref = dict(np.load(cc.OUT / "hold_ref.npz"))
        D["Cg"] = ref["C_gt"]; D["gt"] = np.load(cc.HC / "cache_hold_psi.npy", mmap_mode="r")
    if ref is not None and N < ref["f_gt"].shape[0]:                  # smoke: jac covers the first S' whole shots
        S1 = int(np.searchsorted(ref["offsets"], N, "right") - 1)
        assert ref["offsets"][S1] == N, (N, S1)
        ref = {k: (v[:S1 + 1] if k == "offsets" else v[:N]) for k, v in ref.items()}
    if a.dhat == "oracle":
        dh = ref["f_gt"][:N] - D["f0"]
    else:
        dh = np.mean([np.load(x if "/" in x else cc.OUT / x).astype(np.float64)[:N] for x in a.dhat.split("+")], 0)
        if a.direct:
            dh = dh - D["f0"]
    if a.lam:
        dh = dh * np.array([float(x) for x in a.lam.split(",")])[None, :]
    D["dhat"] = dh
    C = np.load(a.cov if "/" in a.cov else cc.OUT / a.cov).astype(np.float64)
    D["Sig"] = (1 - a.shrink) * C + a.shrink * np.diag(np.diag(C))
    D["Sinv"] = np.linalg.inv(D["Sig"])
    D["Nm"], D["Nden"] = D["Sinv"], float(cc.K)
    if a.prior == "metric":
        M0 = np.load(a.metric).astype(np.float64)
        D["Sig"] = np.linalg.inv(M0)
        D["Sinv"] = M0
        D["Nm"], D["Nden"] = M0, float(np.trace(M0 @ C))          # trust norm in units of the OOF RMS (M0 norm)
        log(f"metric prior: M0 {a.metric}; OOF RMS in M0 norm sqrt(tr(M0 C)) = {np.sqrt(D['Nden']):.5f}")
    D["ridge"], D["ndiv"] = a.ridge, a.ndiv
    vj = cc.FEAT / "train/y_aux_var.json"            # frozen statistics: the 11 training variances (exact, hex floats)
    if vj.exists():
        var = np.array([float.fromhex(x) for x in json.load(open(vj))["var_hex"]], np.float64)
    else:
        ytr = np.load(cc.FEAT / "train/y_aux.npy", mmap_mode="r")[:, cc.AUX_COLS].astype(np.float64)
        var = np.nanvar(ytr, 0)
    D["W"] = (0.20 / 7.0) / var * np.r_[np.ones(7), np.full(4, a.wext)]
    if a.ridge == "trace":
        log("trace ridge: sigma^2 " + " ".join(f"{v:.4g}" for v in var) + f"; wext {a.wext}; ndiv {a.ndiv}")
    dj = json.load(open(a.dsc if "/" in a.dsc else cc.OUT / a.dsc))
    D["sd"] = np.asarray(dj["stats"]["dsc"], np.float64)
    D["Rd"] = a.tau * D["sd"] ** 2
    D["act"] = list(range(7)) if a.targets == "7" else list(range(11)) if a.targets == "11" else [cc.NAMES11.index(x) for x in a.targets.split(",")]
    assert not (a.ridge == "trace" and a.wext == 0 and max(D["act"]) >= 7), "trace ridge with LCFS extrema needs --wext > 0"
    D["guard"] = a.guard
    cfgs = [(float(r), float(e), int(i)) for i in a.iters.split(",") for r in a.rs.split(",") for e in a.etas.split(",")]
    keys = [f"r{r:g}_e{e:g}_i{i}" for r, e, i in cfgs]
    D["cfgs"] = cfgs
    step = 32
    tasks = [(s, min(N, s + step)) for s in range(0, N, step)]
    nc = len(cfgs)
    F = np.repeat(D["f0o"][None], nc, 0).copy()                        # (nc,N,11) default = base
    DL = np.zeros((nc, N, cc.K), np.float32)
    FL = np.ones((nc, N), np.int8)                                     # 1 = skipped, 2 = fallback, 0 = corrected
    MN = np.zeros(N)
    RL = np.zeros((nc, N), np.int8)                                    # re-linearisations used
    if ref is not None:
        bz = np.load(cc.OUT / f"der_hold_{a.tag}.npz")
        HD = np.repeat(bz["haus"][None, :N], nc, 0).copy(); SS = np.repeat(bz["ssres"][None, :N], nc, 0).copy()
        q, b = bz["q95"][:N], bz["betaN"][:N]
    ctx = mp.get_context("fork")
    with ctx.Pool(a.workers) as pool:
        for chunk in pool.imap_unordered(work, tasks, chunksize=2):
            for i, res, mn in chunk:
                MN[i] = mn
                if res is None:
                    continue
                for ci, (v, nre) in res.items():
                    RL[ci, i] = nre
                    if v is None:
                        FL[ci, i] = 2
                    else:
                        F[ci, i] = v[0]; DL[ci, i] = v[3]; FL[ci, i] = 0
                        if ref is not None:
                            HD[ci, i], SS[ci, i] = v[1], v[2]
    log(f"[apply {a.tag}/{a.name}] {N} frames x {nc} configs in {time.time()-t0:.0f}s; targets {[cc.NAMES11[j] for j in D['act']]}; "
        f"trust norm m quantiles 50/90/99: {np.quantile(MN, [0.5, 0.9, 0.99]).round(3).tolist()}")
    if ref is None:
        if a.write:
            for kk in a.write.split(","):
                ci = keys.index(kk)
                psi = np.stack([get_psi(D, i) for i in range(N)]) + (DL[ci].astype(np.float64) @ D["comps"]).reshape(N, 65, 65)
                cc.write_pred_npz(cc.OUT / f"preds_{a.split}_{a.tag}_{a.name}{a.prefix}_{kk}.npz", psi, D["q"], D["b"], D["off"])
                log(f"wrote preds_{a.split}_{a.tag}_{a.name}_{kk}.npz  (fallback {float((FL[ci] == 2).mean()):.4f}, skipped {float((FL[ci] == 1).mean()):.4f}, "
                    f"relinearised {float((RL[ci] > 0).mean()):.4f}, mean |delta| {float(np.linalg.norm(DL[ci], axis=1).mean()):.4f})")
        return
    S = len(ref["offsets"]) - 1
    off = ref["offsets"]
    shA, shB = np.arange(0, S // 2), np.arange(S // 2, S)
    base = {nm: cc.emul_score(ref, D["f0o"], bz["haus"][:N], bz["ssres"][:N], q, b, sh) for nm, sh in (("all", None), ("A", shA), ("B", shB))}
    rows = []
    for ci, kk in enumerate(keys):
        sc = {nm: cc.emul_score(ref, F[ci], HD[ci], SS[ci], q, b, sh) for nm, sh in (("all", None), ("A", shA), ("B", shB))}
        cor = FL[ci] == 0
        tg = dh[:, D["act"]] * np.minimum(1.0, cfgs[ci][0] / np.maximum(MN, 1e-12))[:, None] * cfgs[ci][1]
        mv = (F[ci] - D["f0o"])[:, D["act"]]
        ok = cor[:, None] & np.isfinite(tg) & np.isfinite(mv)
        slope = [float((mv[ok[:, k], k] * tg[ok[:, k], k]).sum() / max((tg[ok[:, k], k] ** 2).sum(), 1e-30)) for k in range(len(D["act"]))]
        rows.append({"key": kk, "scores": sc, "fallback": float((FL[ci] == 2).mean()), "skipped": float((FL[ci] == 1).mean()), "achieved_slope": slope,
                     "relin": float((RL[ci] > 0).mean())})
    print(cc.fmt_score(f"BASE {a.tag} [all]", base["all"]))
    print(f"{'config':15s} {'dS_all':>8s} {'dS_A':>8s} {'dS_B':>8s} | dpsi e-6 d(1-D) dCons e-4 | dR2 e-4: " + " ".join(cc.NAMES7) + " | fb%")
    for r in rows:
        s_, b_ = r["scores"]["all"], base["all"]
        print(f"{r['key']:15s} {s_['S']-b_['S']:+8.5f} {r['scores']['A']['S']-base['A']['S']:+8.5f} {r['scores']['B']['S']-base['B']['S']:+8.5f} | "
              f"{1e6*(s_['r2psi']-b_['r2psi']):+7.1f} {1e4*(b_['dlcfs']-s_['dlcfs']):+6.2f} {1e4*(s_['cons']-b_['cons']):+7.1f} | "
              + " ".join(f"{1e4*(s_[n]-b_[n]):+6.0f}" for n in cc.NAMES7) + f" | {100*r['fallback']:.1f} rl {100*r['relin']:.1f}")
    iA = int(np.argmax([r["scores"]["A"]["S"] for r in rows])); iB = int(np.argmax([r["scores"]["B"]["S"] for r in rows]))
    iAll = int(np.argmax([r["scores"]["all"]["S"] for r in rows]))
    sel = {"chosen_on_A": keys[iA], "B_gain_of_A_choice": rows[iA]["scores"]["B"]["S"] - base["B"]["S"],
           "chosen_on_B": keys[iB], "A_gain_of_B_choice": rows[iB]["scores"]["A"]["S"] - base["A"]["S"],
           "best_all": keys[iAll], "all_gain_best": rows[iAll]["scores"]["all"]["S"] - base["all"]["S"],
           "crossfit_gain_mean": 0.5 * (rows[iA]["scores"]["B"]["S"] - base["B"]["S"] + rows[iB]["scores"]["A"]["S"] - base["A"]["S"])}
    print(f"SELECT [{a.tag}/{a.name}] " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in sel.items()}))
    print("achieved/intended slope (best_all): " + " ".join(f"{cc.NAMES11[j]} {v:.2f}" for j, v in zip(D["act"], rows[iAll]["achieved_slope"])))
    tgt = D["f0"] + dh * (cfgs[iAll][1] * np.minimum(1.0, cfgs[iAll][0] / np.maximum(MN, 1e-12)))[:, None]
    fbm = FL[iAll] == 2
    parts = []
    for j in D["act"]:
        if j >= 7:
            continue
        g = ref["f_gt"][:, j]
        m = np.isfinite(tgt[:, j]) & np.isfinite(F[iAll][:, j]) & np.isfinite(D["f0"][:, j])
        den = ((D["f0"][m, j] - tgt[m, j]) ** 2).sum()
        parts.append(f"{cc.NAMES11[j]} tgt {cc.pooled_r2(g, tgt[:, j]):.4f} new {rows[iAll]['scores']['all'][cc.NAMES11[j]]:.4f} "
                     f"unreal {((F[iAll][m, j] - tgt[m, j]) ** 2).sum() / den:.3f} (fb {((D['f0'][m & fbm, j] - tgt[m & fbm, j]) ** 2).sum() / den:.3f})")
    print("REALIZE (best_all; target R2 / achieved R2 / unrealised share of intended SSE, of which fallback frames): " + " | ".join(parts))
    json.dump({"tag": a.tag, "name": a.name, "args": vars(a), "base": base, "rows": rows, "select": sel, "seconds": time.time() - t0},
              open(cc.OUT / f"apply_{a.tag}_{a.name}{a.prefix}.json", "w"), indent=1)
    if a.write:
        wk = sorted({keys[iA], keys[iB], keys[iAll]}) if a.write == "sel" else a.write.split(",")
        base_psi = np.stack([get_psi(D, i) for i in range(N)])
        for kk in wk:
            ci = keys.index(kk)
            psi = base_psi + (DL[ci].astype(np.float64) @ D["comps"]).reshape(N, 65, 65)
            stem = f"preds_d3d_cc_{a.tag}_{a.name}{a.prefix}_{kk}"
            if a.save_f:
                np.save(cc.OUT / f"f11_holdout_{stem[10:]}.npy", F[ci]); np.save(cc.OUT / f"fl_holdout_{stem[10:]}.npy", FL[ci])
                np.savez(cc.OUT / f"der_{stem[10:]}.npz", f11=F[ci], haus=HD[ci], ssres=SS[ci], q95=q, betaN=b)
            cc.write_pred_npz(cc.OUT / f"{stem}.npz", psi, q, b, off)
            cc.write_pred_npz(cc.OUT / f"{stem}_A.npz", psi, q, b, off, shA)
            cc.write_pred_npz(cc.OUT / f"{stem}_B.npz", psi, q, b, off, shB)
            print(f"WROTE {stem}", flush=True)


if __name__ == "__main__":
    main()
