#!/usr/bin/env python3
"""Stage 11d / ch1d_specialists — label-free per-frame failure fallback between two corrected files of the same split.
Take the PRIMARY corrected map; where the scorer's own prediction-path derivation of the primary map still fails on any of
the 7 consistency scalars (nan, e.g. the li plausibility clamp 0.1..5) and the FALLBACK map's derivation is finite, use
the fallback frame. q95/betaN are taken from the primary (they are identical in both).
  sp_merge.py --primary P.npz --fallback F.npz --f11-primary fP.npy --f11-fallback fF.npy --feat-dir DIR --out O.npz"""
import argparse

import numpy as np

ap = argparse.ArgumentParser()
for k in ("primary", "fallback", "f11-primary", "f11-fallback", "feat-dir", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
off = np.load(f"{a.feat_dir}/offsets.npy"); n = len(off) - 1
fp, ff = np.load(a.f11_primary)[:, :7], np.load(a.f11_fallback)[:, :7]
assert fp.shape[0] == ff.shape[0] == off[-1]
use = (~np.isfinite(fp).all(1)) & np.isfinite(ff).all(1)
P, F = np.load(a.primary), np.load(a.fallback)
assert sorted(P.files) == sorted(F.files) and len(P.files) == 3 * n
d, nrep = {}, 0
for s in range(n):
    k = f"shot_{s:04d}_"
    x = P[k + "psirz"].copy()
    u = use[off[s]:off[s + 1]]
    if u.any():
        x[u] = F[k + "psirz"][u]; nrep += int(u.sum())
        print(f"shot {s}: frames {np.nonzero(u)[0].tolist()} <- fallback")
    d[k + "psirz"] = x.astype(np.float32)
    d[k + "q95"] = P[k + "q95"]; d[k + "betaN"] = P[k + "betaN"]
np.savez(a.out, **d)
print(f"[merge] {n} shots, {off[-1]} frames; primary failing frames {int((~np.isfinite(fp).all(1)).sum())}, replaced {nrep}; wrote {a.out}")
