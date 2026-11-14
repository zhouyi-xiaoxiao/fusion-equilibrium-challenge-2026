#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — step 5: merge the per-frame Jacobians and check them on the holdout.

  merge   jfull shards -> JW_{split}_{K}.npy (n,7,K) float32 = sqrt((W_CONS/7)/Var_j) * J  (NaN -> 0)
          so that   loss_i = a_psi |e_i|^2 + |JW_i e_i|^2   is the Gauss-Newton form of the score loss.
  prep    holdout: predicted per-scalar loss sum_i (J_i e_i)^2 for every scored coefficient-space variant,
          plus two oracle interventions on the incumbent bag20 for the official harness:
          PF_remove7 (error projected out of span(J_i), 7 of 80 dims per frame) / PF_keep7 (only that part).
  report  official vs predicted (per-frame J, global G, amplitude-normalised N).
Env: WS SCR OUT K(80)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout_check as HCK  # noqa: E402

OUT = HCK.OUT
K = int(os.environ.get("K", "80"))
NAMES = HCK.NAMES
W_CONS = 0.20


def merge():
    meta = json.load(open(OUT / f"metric_{K}.json"))
    var = np.array(meta["var_pooled_train"])
    sw = np.sqrt((W_CONS / 7.0) / var)
    info = {"K": K, "var_pooled_train": var.tolist(), "a_psi": meta["a_psi"],
            "definition": "JW = sqrt((W_CONS/7)/Var_j) * J_ij ; loss_i = a_psi*|e_i|^2 + |JW_i @ e_i|^2 ; J by central secant, step = rho for every k"}
    for split in ("holdout", "train"):
        fs = sorted((OUT / "jfull").glob(f"J_{split}_{K}_*.npy"))
        ms = [json.load(open(str(f).replace("/J_", "/meta_").replace(".npy", ".json"))) for f in fs]
        if not fs or len(fs) != ms[0]["nsh"]:
            print(f"{split}: {len(fs)} shards present, expected {ms[0]['nsh'] if ms else '?'} -> skipped"); continue
        n = sum(m["rows"][1] - m["rows"][0] for m in ms)
        n_ref = np.load(HCK.FEAT / f"{split}/y_C{K}.npy", mmap_mode="r").shape[0]
        assert n == n_ref and all(ms[i]["rows"][1] == ms[i + 1]["rows"][0] for i in range(len(ms) - 1)), (n, n_ref)
        JW = np.lib.format.open_memmap(OUT / f"JW_{split}_{K}.npy", mode="w+", dtype=np.float32, shape=(n, 7, K))
        F0 = np.empty((n, 7), np.float32)
        nan_rows = np.zeros(7); rn = []
        for f, m in zip(fs, ms):
            a, b = m["rows"]
            J = np.load(f)
            nan_rows += np.isnan(J).any(2).sum(0)
            Jw = np.nan_to_num(J, nan=0.0, posinf=0.0, neginf=0.0) * sw[None, :, None].astype(np.float32)
            JW[a:b] = Jw
            F0[a:b] = np.load(str(f).replace("/J_", "/f0_"))
            rn.append(np.sqrt((Jw.astype(np.float64) ** 2).sum(2)))
        JW.flush()
        np.save(OUT / f"f0_{split}_{K}.npy", F0)
        rn = np.concatenate(rn)
        info[split] = {"n": int(n), "shards": len(fs), "step": ms[0]["step"], "core_hours": sum(m["core_seconds"] for m in ms) / 3600,
                       "wall_seconds_max": max(m["wall_seconds"] for m in ms), "frames_with_nan_J_per_scalar": dict(zip(NAMES, nan_rows.tolist())),
                       "rownorm_quantiles_50_90_99_99.9_100": {nm: np.quantile(rn[:, j], [.5, .9, .99, .999, 1.0]).tolist() for j, nm in enumerate(NAMES)}}
        print(split, json.dumps(info[split])[:900])
    json.dump(info, open(OUT / f"JW_{K}.json", "w"), indent=1)


def _holdout():
    var_hold, ss_frame, _ = HCK.hold_stats()
    Y = np.load(HCK.FEAT / f"holdout/y_C{K}.npy").astype(np.float64)
    meta = json.load(open(OUT / f"metric_{K}.json"))
    sw = np.sqrt((W_CONS / 7.0) / np.array(meta["var_pooled_train"]))
    J = np.load(OUT / f"JW_holdout_{K}.npy").astype(np.float64) / sw[None, :, None]
    return var_hold, ss_frame, Y, J


def prep():
    var_hold, ss_frame, Y, J = _holdout()
    rec = json.load(open(HCK.HOLD / "predicted.json"))
    out = {}
    for name, r in rec["variants"].items():
        if r["K"] != K:
            continue
        v = HCK.variants().get(name)
        if v is not None:
            C = np.mean([np.load(c) for c in v["coefs"]], 0).astype(np.float64)
        else:
            z = np.load(HCK.HOLD / f"coef_{name}.npy") if (HCK.HOLD / f"coef_{name}.npy").exists() else None
            if z is None:
                continue
            C = z.astype(np.float64)
        e = C - Y
        out[name] = (np.einsum("njk,nk->nj", J, e) ** 2).mean(0) / var_hold
    # oracle interventions on bag20 with the per-frame Jacobians
    b = HCK.variants()["p80_bag20"]
    Cb = np.mean([np.load(c) for c in b["coefs"]], 0).astype(np.float64)
    e = Cb - Y
    ep = np.empty_like(e)
    for i in range(len(e)):                                  # projection of e_i on span(rows of J_i)
        A = J[i]                                             # (7,K)
        ep[i] = np.linalg.lstsq(A.T, e[i], rcond=1e-10)[0] @ A if np.any(A) else 0.0
    new = {"PF_remove7": Cb - ep, "PF_keep7": Y + ep}
    cache = np.load(HCK.HC / "cache_hold.npz"); offsets = cache["offsets"]
    heads = np.load(HCK.HC / "heads_mlp.npz"); pc = np.load(HCK.HC / f"pca_{K}.npz")
    for name, C in new.items():
        ee = C - Y
        out[name] = (np.einsum("njk,nk->nj", J, ee) ** 2).mean(0) / var_hold
        np.save(HCK.HOLD / f"coef_{name}.npy", C.astype(np.float32))
        psi = C @ pc["comps_"].astype(np.float64) + pc["mean_"].astype(np.float64)[None, :]
        d = {}
        for s in range(len(offsets) - 1):
            a, b_ = offsets[s], offsets[s + 1]
            d[f"shot_{s:04d}_psirz"] = psi[a:b_].reshape(-1, 65, 65).astype(np.float32)
            d[f"shot_{s:04d}_q95"] = heads["q95_bag"][a:b_].astype(np.float32)
            d[f"shot_{s:04d}_betaN"] = heads["betaN_bag"][a:b_].astype(np.float32)
        np.savez(HCK.HOLD / f"preds_d3d_{name}.npz", **d)
        print(f"{name}: |e|^2 ratio to bag20 {np.mean((ee**2).sum(1))/np.mean((e**2).sum(1)):.4f}; predicted per-frame cons loss {out[name].mean():.5f}")
    json.dump({k: v.tolist() for k, v in out.items()}, open(HCK.HOLD / f"predicted_perframe_{K}.json", "w"), indent=1)
    (HCK.HOLD / "new_names_pf.txt").write_text("\n".join(new.keys()) + "\n")


def report():
    pf = json.load(open(HCK.HOLD / f"predicted_perframe_{K}.json"))
    base = json.load(open(OUT / "sanity_holdout.json"))
    floors = np.array(base["floors"][str(K)])
    grows = {r["name"]: r for r in base["rows"]}
    rows = []
    for name, p in pf.items():
        if name in grows:
            off = np.array(grows[name]["official"]); g = grows[name]["cons_loss_pred_cal"]; nn = grows[name]["cons_loss_pred_n"]; S = grows[name]["S"]
        else:
            o = HCK.parse_log(HCK.HOLD / f"scores/score_{name}.log")
            if o is None:
                print(f"{name}: no official log"); continue
            off = np.array([1 - o[n] for n in NAMES]); g = nn = float("nan"); S = o["S"]
        pp = np.array(p) + floors
        if name.startswith("oracle"):
            pp = floors.copy()
        rows.append(dict(name=name, S=S, official=off.tolist(), pred_pf=pp.tolist(), cons_off=float(off.mean()), cons_pf=float(pp.mean()), cons_G=g, cons_N=nn))
    print(f"{'variant':28s} {'S':>6s} {'1-Cons off':>10s} {'per-frame':>9s} {'global G':>8s} {'norm N':>8s} | per-scalar official/per-frame: " + " ".join(NAMES))
    for x in rows:
        print(f"{x['name']:28s} {x['S']:6.4f} {x['cons_off']:10.4f} {x['cons_pf']:9.4f} {x['cons_G']:8.4f} {x['cons_N']:8.4f} | "
              + " ".join(f"{a:.4f}/{b:.4f}" for a, b in zip(x["official"], x["pred_pf"])))
    summ = {}
    for label, sel in (("all", rows), ("models", [x for x in rows if not x["name"].startswith(("oracle", "I_", "In_", "PF_"))]),
                       ("interventions", [x for x in rows if x["name"].startswith(("I_", "In_", "PF_")) or x["name"] == "p80_bag20"])):
        if len(sel) < 3:
            continue
        a = np.array([x["cons_off"] for x in sel]); b = np.array([x["cons_pf"] for x in sel])
        A = np.array([x["official"] for x in sel]); B = np.array([x["pred_pf"] for x in sel])
        summ[label] = {"n": len(sel), "pearson": float(np.corrcoef(a, b)[0, 1]), "mean_ratio": float(np.mean(b / a)), "max_abs_err": float(np.max(np.abs(a - b))),
                       "per_scalar_mean_ratio": dict(zip(NAMES, np.mean(B / A, 0).round(3).tolist())),
                       "per_scalar_pearson": dict(zip(NAMES, [round(float(np.corrcoef(A[:, j], B[:, j])[0, 1]), 4) for j in range(7)]))}
        print(f"[{label}] n={len(sel)} per-frame-J prediction vs official 1-Consistency: Pearson {summ[label]['pearson']:.4f}, mean pred/official {summ[label]['mean_ratio']:.3f}, max |err| {summ[label]['max_abs_err']:.4f}")
        print("   per-scalar mean pred/official: " + json.dumps(summ[label]["per_scalar_mean_ratio"]) + "\n   per-scalar Pearson: " + json.dumps(summ[label]["per_scalar_pearson"]))
    json.dump({"rows": rows, "summary": summ}, open(OUT / f"sanity_perframe_{K}.json", "w"), indent=1)


if __name__ == "__main__":
    {"merge": merge, "prep": prep, "report": report}[sys.argv[1]]()
