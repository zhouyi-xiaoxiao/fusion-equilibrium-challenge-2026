#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — step 3: sanity of the metric on the standard 150-shot HOLDOUT (diagnostic only;
nothing here feeds back into M).

  prep    for every coefficient-space prediction set with an official-harness log (single seeds, bag5,
          bag10, bag20, PCA oracles) compute the PREDICTED per-scalar Consistency loss
              pred_j = mean_i e_i^T (alpha_j E[J_j^T J_j]) e_i / Var_holdout(f_j),   e = c_pred - c_true
          and write new prediction files for (a) the two half-data bags of stage 10 and (b) oracle
          INTERVENTIONS on the incumbent bag20 (error removed / kept only in the top consistency-metric
          eigen-directions, vs the first PCA coefficients, vs random directions) -> scored by the
          official harness in run_post.sh.
  report  parse the official logs and print predicted vs official 1-R2 per scalar + correlations.
Env: WS SCR OUT
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
SCR = Path(os.environ["SCR"])
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/metric")))
FEAT = SCR / "stage11_ch1/feat_v2"
HC, S8, S10 = SCR / "stage6_H", SCR / "stage8_L_ch1_finish_package", SCR / "stage10_ch1_cons"
HV = WS / "stage6_run/variants/H_ch1_fulldata_bag"
HOLD = OUT / "hold"
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
W_PSI, W_CONS = 0.55, 0.20


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def variants():
    v = {}
    for tag, d, logdir, lf in (("p80_h512x256x128", HC, HV, "score_{n}.log"), ("p80_h1024x512x256", HC, HV, "score_{n}.log"),
                               ("p120_h1024x512x256", HC, HV, "score_{n}.log"),
                               ("p120_h512x256x128", S8, S8 / "scores", "score_{n}_skip1000_n150.log")):
        K = int(tag.split("_")[0][1:])
        for nm, seeds in (("s42", [42]), ("bag5", range(42, 47)), ("bag10", range(42, 52))):
            n = f"{tag}_{nm}"
            v[n] = dict(K=K, coefs=[str(d / f"coef_{tag}_s{s}.npy") for s in seeds], log=str(logdir / lf.format(n=n)), pca_dir=str(d))
    v["p80_bag20"] = dict(K=80, coefs=[str(HC / f"coef_p80_{h}_s{s}.npy") for h in ("h512x256x128", "h1024x512x256") for s in range(42, 52)],
                          log=str(S10 / "scores/score_p80_bag20_hbag_skip1000_n150.log"), pca_dir=str(HC))
    for K in (80, 120):
        v[f"oracle_pca{K}"] = dict(K=K, coefs=[str(S10 / f"coef_oracle{K}.npy")],
                                   log=str(S10 / f"scores/score_oracle_pca{K}_hbag_skip1000_n150.log"), pca_dir=str(HC))
    return v


def parse_log(path):
    try:
        t = Path(path).read_text()
    except Exception:
        return None
    if "COMPOSITE" not in t:
        return None
    out = {"S": float(re.search(r"COMPOSITE S = ([\d.]+)", t).group(1))}
    for key, pat in (("r2_psi", r"R2_psi\s+([\d.-]+)"), ("r2_qb", r"R2_\{q95,betaN\}\s+([\d.-]+)"),
                     ("one_minus_d", r"1 - D_LCFS\s+([\d.-]+)"), ("cons", r"Consistency\s+([\d.-]+)")):
        out[key] = float(re.search(pat, t).group(1))
    for n in NAMES:
        out[n] = float(re.search(rf"^\s+{n}\s+([\d.-]+)\s*$", t, re.M).group(1))
    return out


def load_metric(K):
    p = np.load(OUT / f"M_parts_{K}.npz", allow_pickle=False)
    meta = json.load(open(OUT / f"metric_{K}.json"))
    return p, meta


def hold_stats():
    an = json.load(open(FEAT / "spec.json"))["aux_names"]
    a = np.asarray(np.load(FEAT / "holdout/y_aux.npy")[:, [an.index("cons_" + c) for c in NAMES]], np.float64)
    fin = np.isfinite(a)
    n = fin.sum(0); s = np.where(fin, a, 0).sum(0); sq = np.where(fin, a * a, 0).sum(0)
    var = (sq - s * s / n) / n
    psi = np.load(HC / "cache_hold_psi.npy", mmap_mode="r")
    S = SQ = 0.0; cnt = 0; fr_norm2 = []
    mean_map = np.load(HC / "pca_80.npz")["mean_"].astype(np.float64)
    for a0 in range(0, psi.shape[0], 4096):
        p = np.asarray(psi[a0:a0 + 4096], np.float64).reshape(-1, 4225)
        S += p.sum(); SQ += (p * p).sum(); cnt += p.size
        fr_norm2.append(((p - mean_map) ** 2).sum(1))
    ss_frame = (SQ - S * S / cnt) / psi.shape[0]
    return var, ss_frame, np.concatenate(fr_norm2)


def predict(e, parts, var_hold, ss_frame, amp):
    EJ, alpha, w_tr = parts["EJtJ"], parts["alpha"], parts["w_j"]
    q = np.einsum("nk,jkl,nl->j", e, EJ, e, optimize=True) / len(e)          # mean e^T E[J^T J] e per scalar
    en = e / amp[:, None]
    qn = np.einsum("nk,jkl,nl->j", en, parts["EJtJn"], en, optimize=True) / len(e)
    return {"pred_uncal": (q / var_hold).tolist(), "pred_cal": (alpha * q / var_hold).tolist(),
            "pred_n": (parts["alphan"] * qn / var_hold).tolist(),
            "psi_model": float((e ** 2).sum(1).mean() / ss_frame),
            "metric_loss_train_norm": float(parts["a_psi"] * (e ** 2).sum(1).mean() + (alpha * w_tr * q).sum())}


def prep():
    HOLD.mkdir(parents=True, exist_ok=True); (HOLD / "scores").mkdir(exist_ok=True)
    var_hold, ss_frame, fr_norm2 = hold_stats()
    log(f"holdout pooled: SS_tot_psi/frame {ss_frame:.4f}; Var(f_j) " + " ".join(f"{v:.5g}" for v in var_hold))
    Y = {K: np.load(FEAT / f"holdout/y_C{K}.npy").astype(np.float64) for K in (80, 120)}
    P = {K: load_metric(K) for K in (80, 120)}
    pcs = {K: np.load(HC / f"pca_{K}.npz") for K in (80, 120)}
    trunc_psi = {K: float((fr_norm2 - (Y[K] ** 2).sum(1)).mean() / ss_frame) for K in (80, 120)}
    rec = {"var_hold": var_hold.tolist(), "ss_frame_hold": ss_frame, "trunc_psi": trunc_psi, "variants": {}}
    amp = np.load(OUT / "amp_holdout.npy").astype(np.float64)
    amp = np.where(np.isfinite(amp) & (amp > 1e-3), amp, np.nanmedian(amp))
    cache = np.load(HC / "cache_hold.npz"); offsets = cache["offsets"]
    heads = np.load(HC / "heads_mlp.npz"); q95, betaN = heads["q95_bag"], heads["betaN_bag"]

    def write_preds(name, C, K):
        psi = (C @ pcs[K]["comps_"].astype(np.float64) + pcs[K]["mean_"].astype(np.float64)[None, :])
        out = {}
        for i in range(len(offsets) - 1):
            a, b = offsets[i], offsets[i + 1]
            out[f"shot_{i:04d}_psirz"] = psi[a:b].reshape(-1, 65, 65).astype(np.float32)
            out[f"shot_{i:04d}_q95"] = q95[a:b].astype(np.float32)
            out[f"shot_{i:04d}_betaN"] = betaN[a:b].astype(np.float32)
        np.savez(HOLD / f"preds_d3d_{name}.npz", **out)

    for name, v in variants().items():
        K = v["K"]
        try:
            cs = [np.load(c) for c in v["coefs"]]
            pd = np.load(Path(v["pca_dir"]) / f"pca_{K}.npz")
            assert np.array_equal(pd["comps_"], pcs[K]["comps_"]), "basis differs"
            C = np.mean(cs, 0).astype(np.float64)
            assert C.shape == Y[K].shape, (C.shape, Y[K].shape)
        except Exception as ex:  # noqa: BLE001
            log(f"skip {name}: {type(ex).__name__}: {ex}"); continue
        r = predict(C - Y[K], P[K][0], var_hold, ss_frame, amp)
        r.update(K=K, log=v["log"], kind="existing")
        rec["variants"][name] = r

    # ---- new prediction sets -> official harness. Once written they are a FIXED test set: later runs
    # (after M is rebuilt) reload the same coefficients so predictions and official logs stay paired.
    K = 80
    parts, meta = P[K]
    b = variants()["p80_bag20"]
    Cb = np.mean([np.load(c) for c in b["coefs"]], 0).astype(np.float64)
    e = Cb - Y[K]
    new = {}
    if (HOLD / "new_names.txt").exists():
        cm, mm = pcs[K]["comps_"].astype(np.float64), pcs[K]["mean_"].astype(np.float64)
        for name in (HOLD / "new_names.txt").read_text().split():
            f = HOLD / f"coef_{name}.npy"
            if not f.exists():
                z = np.load(HOLD / f"preds_d3d_{name}.npz")
                psi = np.concatenate([z[f"shot_{i:04d}_psirz"].reshape(-1, 4225) for i in range(len(offsets) - 1)]).astype(np.float64)
                np.save(f, ((psi - mm[None, :]) @ cm.T).astype(np.float32))
            new[name] = (K, np.load(f).astype(np.float64))
        m = None
    else:
        for k in (0, 1):
            new[f"halfbag_k{k}"] = (80, np.load(S10 / f"coef_halfbag_k{k}.npy").astype(np.float64))
        m = int(meta["eig"]["n_dirs_for_80pct_cons_loss"])
        V = parts["eig_cons_vecs"][:, :m]                      # ranked on TRAIN OOF contributions
        etop = (e @ V) @ V.T
        rng = np.random.default_rng(20261011)
        Q = np.linalg.qr(rng.standard_normal((K, m)))[0]
        new[f"I_removeTop{m}"] = (K, Cb - etop)                # error removed in the top-m metric directions
        new[f"I_keepTop{m}"] = (K, Y[K] + etop)                # ONLY the top-m metric-direction error kept
        c_pca = Cb.copy(); c_pca[:, :m] = Y[K][:, :m]
        new[f"I_pcaFirst{m}"] = (K, c_pca)                     # first m PCA coefficients set to truth
        new[f"I_removeRand{m}"] = (K, Cb - (e @ Q) @ Q.T)      # control: m random orthonormal directions
        mn = int(meta["family_N"]["n_dirs_for_80pct_cons_loss"])
        Vn = parts["eign_cons_vecs"][:, :mn]                   # family N directions, ranked on TRAIN OOF
        etn = (e @ Vn) @ Vn.T
        new[f"In_removeTop{mn}"] = (K, Cb - etn)
        new[f"In_keepTop{mn}"] = (K, Y[K] + etn)
    for name, (K_, C) in new.items():
        r = predict(C - Y[K_], P[K_][0], var_hold, ss_frame, amp)
        r.update(K=K_, log=str(HOLD / f"scores/score_{name}.log"), kind="new")
        rec["variants"][name] = r
        if not (HOLD / f"preds_d3d_{name}.npz").exists():
            write_preds(name, C, K_)
            np.save(HOLD / f"coef_{name}.npy", C.astype(np.float32))
        r["err_energy_ratio_to_bag20"] = float(((C - Y[K_]) ** 2).sum() / (e ** 2).sum())
        log(f"new variant {name}: |e|^2/bag20 {r['err_energy_ratio_to_bag20']:.3f} psi_model {r['psi_model']:.6f} pred cons loss G {np.mean(r['pred_cal']):.5f} N {np.mean(r['pred_n']):.5f}")
    rec["new_names"] = list(new.keys()); rec["m_dirs"] = m
    json.dump(rec, open(HOLD / "predicted.json", "w"), indent=1)
    (HOLD / "new_names.txt").write_text("\n".join(new.keys()) + "\n")
    log(f"prep done: {len(rec['variants'])} variants; new to score: {list(new.keys())}")


def report():
    rec = json.load(open(HOLD / "predicted.json"))
    floors = {}
    for K in (80, 120):
        o = parse_log(rec["variants"][f"oracle_pca{K}"]["log"])
        floors[K] = np.array([1 - o[n] for n in NAMES])
    rows = []
    log("columns per scalar: official 1-R2 | predicted (cal) + oracle floor | predicted (uncal) + floor")
    for name, r in rec["variants"].items():
        o = parse_log(r["log"])
        if o is None:
            log(f"{name}: no official log"); continue
        off = np.array([1 - o[n] for n in NAMES]); fl = floors[r["K"]]
        pc = np.array(r["pred_cal"]) + fl; pu = np.array(r["pred_uncal"]) + fl; pn = np.array(r["pred_n"]) + fl
        if name.startswith("oracle"):
            pc, pu, pn = fl.copy(), fl.copy(), fl.copy()
        rows.append(dict(name=name, K=r["K"], kind=r["kind"], S=o["S"], official=off.tolist(), pred_cal=pc.tolist(), pred_uncal=pu.tolist(),
                         pred_n=pn.tolist(), cons_loss_pred_n=float(pn.mean()),
                         cons_loss_official=1 - o["cons"], cons_loss_pred_cal=float(pc.mean()), cons_loss_pred_uncal=float(pu.mean()),
                         r2psi_official=o["r2_psi"], psi_loss_pred=r["psi_model"] + rec["trunc_psi"][str(r["K"])],
                         one_minus_d=o["one_minus_d"], metric_loss=r["metric_loss_train_norm"]))
    hdr = f"{'variant':30s} {'S':>6s} {'1-Cons off':>10s} {'pred G':>8s} {'pred N':>8s} {'G uncal':>8s} {'1-R2psi off':>11s} {'pred':>8s} | per-scalar official/G/N: " + " ".join(NAMES)
    print(hdr)
    for x in rows:
        print(f"{x['name']:30s} {x['S']:6.4f} {x['cons_loss_official']:10.4f} {x['cons_loss_pred_cal']:8.4f} {x['cons_loss_pred_n']:8.4f} {x['cons_loss_pred_uncal']:8.4f} "
              f"{1-x['r2psi_official']:11.4f} {x['psi_loss_pred']:8.5f} | " + " ".join(f"{a:.4f}/{b:.4f}/{c:.4f}" for a, b, c in zip(x["official"], x["pred_cal"], x["pred_n"])))
    summ = {}
    for label, sel in (("all", rows), ("existing_models", [x for x in rows if x["kind"] == "existing" and not x["name"].startswith("oracle")]),
                       ("models_incl_halfbags", [x for x in rows if not x["name"].startswith(("oracle", "I_", "In_"))]),
                       ("interventions+bag20", [x for x in rows if x["name"].startswith(("I_", "In_")) or x["name"] == "p80_bag20"])):
        if len(sel) < 3:
            continue
        a = np.array([x["cons_loss_official"] for x in sel]); b = np.array([x["cons_loss_pred_cal"] for x in sel]); c = np.array([x["cons_loss_pred_uncal"] for x in sel])
        A = np.array([x["official"] for x in sel]); B = np.array([x["pred_cal"] for x in sel])
        d = np.array([x["cons_loss_pred_n"] for x in sel]); D = np.array([x["pred_n"] for x in sel])
        summ[label] = {"n": len(sel), "pearson_cons_cal": float(np.corrcoef(a, b)[0, 1]), "pearson_cons_uncal": float(np.corrcoef(a, c)[0, 1]),
                       "pearson_cons_n": float(np.corrcoef(a, d)[0, 1]), "mean_ratio_pred_n_over_official": float(np.mean(d / a)),
                       "max_abs_err_cons_n": float(np.max(np.abs(a - d))),
                       "per_scalar_ratio_pred_n_over_official_mean": dict(zip(NAMES, np.nanmean(np.where(A > 1e-4, D / np.maximum(A, 1e-12), np.nan), 0).round(3).tolist())),
                       "per_scalar_pearson_n": dict(zip(NAMES, [float(np.corrcoef(A[:, j], D[:, j])[0, 1]) for j in range(7)])),
                       "mean_ratio_pred_cal_over_official": float(np.mean(b / a)), "mean_ratio_pred_uncal_over_official": float(np.mean(c / a)),
                       "max_abs_err_cons_cal": float(np.max(np.abs(a - b))),
                       "per_scalar_ratio_pred_cal_over_official_mean": dict(zip(NAMES, np.nanmean(np.where(A > 1e-4, B / np.maximum(A, 1e-12), np.nan), 0).round(3).tolist())),
                       "per_scalar_pearson": dict(zip(NAMES, [float(np.corrcoef(A[:, j], B[:, j])[0, 1]) for j in range(7)]))}
        print(f"\n[{label}] n={len(sel)} Pearson(official 1-Cons, predicted cal) = {summ[label]['pearson_cons_cal']:.4f} (uncal {summ[label]['pearson_cons_uncal']:.4f}); "
              f"mean pred/official cal {summ[label]['mean_ratio_pred_cal_over_official']:.3f} uncal {summ[label]['mean_ratio_pred_uncal_over_official']:.3f}; "
              f"max |err| {summ[label]['max_abs_err_cons_cal']:.4f}")
        print(f"   family N: Pearson {summ[label]['pearson_cons_n']:.4f}; mean pred/official {summ[label]['mean_ratio_pred_n_over_official']:.3f}; max |err| {summ[label]['max_abs_err_cons_n']:.4f}")
        print("   per-scalar mean pred/official G: " + json.dumps(summ[label]["per_scalar_ratio_pred_cal_over_official_mean"]))
        print("   per-scalar mean pred/official N: " + json.dumps(summ[label]["per_scalar_ratio_pred_n_over_official_mean"]))
        print("   per-scalar Pearson across variants: " + " ".join(f"{n}={v:.3f}" for n, v in summ[label]["per_scalar_pearson"].items()))
    json.dump({"rows": rows, "summary": summ, "floors": {str(k): v.tolist() for k, v in floors.items()}, "m_dirs": rec.get("m_dirs")},
              open(OUT / "sanity_holdout.json", "w"), indent=1)


if __name__ == "__main__":
    {"prep": prep, "report": report}[sys.argv[1]]()
