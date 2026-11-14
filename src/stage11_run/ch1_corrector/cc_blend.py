#!/usr/bin/env python3
"""Target blending — combine the map-derived scalars f_b with head estimates into one target vector per frame.

Weights are fitted per scalar on the 5 % shot-disjoint validation slice of TRAIN (base = OUT-OF-FOLD map there),
never on the holdout:
  dlam : T = f_b + lam_j (T_direct - f_b)                         lam_j = <(y-f_b)(T_d-f_b)> / <(T_d-f_b)^2>, clipped to [0,1]
  ls   : T = f_b + a_j (T_direct - f_b) + b_j (T_stack - f_b)     least squares (a, b), no intercept
  rob  : T = T_direct + lam_j * clip(f_b - T_direct, +-c_j s_j)   s_j = 1.4826 MAD(f_b - T_direct); (c_j, lam_j) by val-slice SSE
         (map-derived scalars have heavy-tailed errors: when map and head disagree wildly the head wins)
Where the base value is nan (failed derivation) the target falls back to the direct head.
  cc_blend.py --base gv21 --base-train f11_train_gv21.npy --direct direct2 --stack stack_gv21 --stack-app gv21
Outputs: tgt_<base>_{dlam,rob,ls}.npy (holdout, (N,11) absolute targets; use with cc_apply --direct), blend_<base>.json
"""
import argparse
import json

import numpy as np

import cc_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--base-train", required=True)
ap.add_argument("--direct", required=True); ap.add_argument("--stack", default=None); ap.add_argument("--stack-app", default=None)
ap.add_argument("--split", default="holdout")
ap.add_argument("--direct-pred", default=None, help="override: direct-head predictions for this split (npy (N,11))")
ap.add_argument("--frozen", default=None, help="blend json of a previous run: reuse its constants instead of fitting "
                "(final refit on all shots has no honest validation slice)")
ap.add_argument("--keep-base", default="", help="comma list of scalar names whose target is the base's own value "
                "(d_j = 0: the step holds them fixed), e.g. volume,Rmin  (per-scalar gate, cc_gate.py)")
a = ap.parse_args()
CG = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 1e9]
ts = None
v = np.zeros(0, bool)
lam, ab = np.zeros(11), np.zeros((11, 2))
rob = np.zeros((11, 3))
if a.frozen:
    fz = json.load(open(a.frozen if "/" in a.frozen else cc.OUT / a.frozen))
    lam, ab, rob = np.asarray(fz["lam"]), np.asarray(fz["ab"]), np.asarray(fz["rob_clip_lam_c"])
    print(f"[blend] FROZEN constants from {a.frozen}")
else:
    y = np.load(cc.FEAT / "train/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]
    off = np.load(cc.FEAT / "train/offsets.npy"); S = len(off) - 1
    perm = np.random.default_rng(0).permutation(S)
    isv = np.zeros(S, bool); isv[perm[: max(1, int(0.05 * S))]] = True
    v = np.repeat(isv, np.diff(off))
    fb = np.load(cc.OUT / a.base_train)[v].astype(np.float64)
    td = np.load(cc.OUT / f"hp_{a.direct}_mlp_train.npy")[v].astype(np.float64)
    ts = np.load(cc.OUT / f"hp_{a.stack}_mlp_train.npy")[v].astype(np.float64) if a.stack else None
    yv = y[v]
    lam, ab = np.zeros(11), np.zeros((11, 2))
    rob = np.zeros((11, 3))                                 # (c_j * s_j, lam_j, c_j)
    for j in range(11):
        m = np.isfinite(yv[:, j]) & np.isfinite(fb[:, j])
        u, w = td[m, j] - fb[m, j], yv[m, j] - fb[m, j]
        lam[j] = np.clip((u * w).sum() / (u * u).sum(), 0.0, 1.0)
        g, e = fb[m, j] - td[m, j], yv[m, j] - td[m, j]
        sj = 1.4826 * np.median(np.abs(g - np.median(g)))
        best = (np.inf, 0, 0, 0)
        for c in CG:
            z = np.clip(g, -c * sj, c * sj)
            l = float(np.clip((z * e).sum() / (z * z).sum(), 0.0, 1.0))
            sse = float(((e - l * z) ** 2).sum())
            if sse < best[0]:
                best = (sse, c * sj, l, c)
        rob[j] = best[1:]
        if ts is not None:
            A = np.stack([u, ts[m, j] - fb[m, j]], 1)
            ab[j] = np.linalg.lstsq(A, w, rcond=None)[0]
Fb = np.load(cc.OUT / f"f11_{a.split}_{a.base}.npy").astype(np.float64)
Td = np.load(a.direct_pred or cc.OUT / f"hp_{a.direct}_mlp_{a.split}.npy").astype(np.float64)
ok = np.isfinite(Fb)
out = {"dlam": np.where(ok, Fb + lam * (Td - Fb), Td),
       "rob": np.where(ok, Td + rob[:, 1] * np.clip(np.nan_to_num(Fb - Td), -rob[:, 0], rob[:, 0]), Td)}
if a.keep_base:
    for nm in a.keep_base.split(","):
        j = cc.NAMES11.index(nm)
        for k in out:
            out[k][:, j] = np.where(ok[:, j], Fb[:, j], out[k][:, j])
    print(f"[blend] keep-base (gated out): {a.keep_base}")
cands = {"base": Fb, "direct": Td, "dlam": out["dlam"], "rob": out["rob"]}
if a.stack:
    Ts = np.load(cc.OUT / f"hp_{a.stack}_mlp_{a.stack_app or a.base}.npy").astype(np.float64)
    out["ls"] = np.where(ok, Fb + ab[:, 0] * (Td - Fb) + ab[:, 1] * (Ts - Fb), Td)
    cands.update(stack=Ts, avg_direct_stack=0.5 * (Td + Ts), ls=out["ls"])
pre = f"tgt_{a.base}" if a.split == "holdout" else f"tgt_{a.split}_{a.base}"
for k, t in out.items():
    np.save(cc.OUT / f"{pre}_{k}.npy", t)
print(f"[blend {a.base}/{a.split}] val-slice frames {int(v.sum())}; lam: " + " ".join(f"{x:.2f}" for x in lam))
print("   rob (c, lam): " + " ".join(f"({c:g},{l:.2f})" for _, l, c in rob))
if ts is not None:
    print("   ls (a_direct, b_stack): " + " ".join(f"({p:.2f},{q:.2f})" for p, q in ab))
rec = {"lam": lam.tolist(), "ab": ab.tolist(), "rob_clip_lam_c": rob.tolist(), "r2": {}}
if a.split == "holdout":
    yh = np.load(cc.FEAT / "holdout/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]
    for k, t in cands.items():
        r = [cc.pooled_r2(yh[:, j], t[:, j]) for j in range(11)]
        rec["r2"][k] = r
        print(f"   holdout R2 {k:17s} " + " ".join(f"{x:.4f}" for x in r) + f" | mean7 {np.mean(r[:7]):.4f}")
json.dump(rec, open(cc.OUT / f"blend_{a.split}_{a.base}.json", "w"), indent=1)
