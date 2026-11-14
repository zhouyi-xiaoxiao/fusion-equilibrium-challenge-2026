#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — step 4 (extension): PER-FRAME Jacobians for every frame of a split.

Why: build_metric.py shows that one global M over-predicts the axis terms on real model errors (the
sensitive direction of each frame is frame-specific), whereas the per-frame linearisation J_i e_i predicts
the actual scalar error frame by frame. This writes J_i for ALL frames so a model can train on
    loss_i = a_psi |e_i|^2 + sum_j (W_CONS/7)/Var_j (J_ij . e_i)^2 .

Per frame: psi0 = mean + c_true @ comps (stage6_H pca_K), f0 = official derivation (prediction path,
MACHINE=DIII-D), J[:,k] = (f(psi0 + d comp_k) - f(psi0 - d comp_k)) / (2 d) with ONE step d = DMULT * rho
for every k (rho = incumbent total RMS coefficient-error norm on the holdout; DMULT=1.0 -> config "B10").

Env: WS SCR OUT SPLIT(train|holdout) K(80) DMULT(1.0) NSH(20) SHARD WORKERS(70) MAXF(0)
Output: $OUT/jfull/J_{SPLIT}_{K}_{SHARD:03d}.npy (n,7,K) float32, f0_{...}.npy (n,7) float32, meta json
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fd_worker as W  # noqa: E402  (reuses dpred / incumbent_deltas / paths; official scorer imports)

SPLIT = os.environ.get("SPLIT", "train")
K = int(os.environ.get("K", "80"))
DMULT = float(os.environ.get("DMULT", "1.0"))
NSH = int(os.environ.get("NSH", "20"))
SHARD = int(os.environ.get("SHARD", os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
WORKERS = int(os.environ.get("WORKERS", "70"))
MAXF = int(os.environ.get("MAXF", "0"))
CHUNK = 16
_L = {}


def _init():
    W._init()
    _L["Y"] = np.load(W.FEAT / f"{SPLIT}/y_C{K}.npy", mmap_mode="r")
    _L["comps"] = W._G["comps"][K]; _L["mean"] = W._G["mean"][K]
    _L["d"] = DMULT * W.incumbent_deltas()[K]["rho"]


def work(ab):
    a, b = ab
    comps, mean, d = _L["comps"], _L["mean"], _L["d"]
    J = np.full((b - a, 7, K), np.nan, np.float32); F0 = np.full((b - a, 7), np.nan, np.float32)
    t0 = time.time()
    for n, r in enumerate(range(a, b)):
        psi0 = (mean + np.asarray(_L["Y"][r], np.float64) @ comps).reshape(65, 65)
        try:
            f0, _ = W.dpred(psi0)
            F0[n] = f0
            for k in range(K):
                pk = comps[k].reshape(65, 65)
                fp, _ = W.dpred(psi0 + d * pk); fm, _ = W.dpred(psi0 - d * pk)
                J[n, :, k] = (fp - fm) / (2.0 * d)
        except Exception as ex:  # noqa: BLE001
            print(f"row {r}: {type(ex).__name__}: {ex}", flush=True)
    return a, b, J, F0, time.time() - t0


def main():
    t0 = time.time()
    out = W.OUT / "jfull"; out.mkdir(parents=True, exist_ok=True)
    n_all = np.load(W.FEAT / f"{SPLIT}/y_C{K}.npy", mmap_mode="r").shape[0]
    edges = np.linspace(0, n_all, NSH + 1).astype(np.int64)
    a0, b0 = int(edges[SHARD]), int(edges[SHARD + 1])
    if MAXF:
        b0 = min(b0, a0 + MAXF)
    tag = f"{SPLIT}_{K}_{SHARD:03d}"
    if (out / f"J_{tag}.npy").exists() and not MAXF:
        W.log(f"exists: J_{tag}.npy"); return 0
    d = DMULT * W.incumbent_deltas()[K]["rho"]
    W.log(f"{tag}: rows [{a0},{b0}) of {n_all}; K={K}; step d={d:.6f} (= {DMULT} * rho); workers {WORKERS}")
    J = np.empty((b0 - a0, 7, K), np.float32); F0 = np.empty((b0 - a0, 7), np.float32)
    chunks = [(a, min(b0, a + CHUNK)) for a in range(a0, b0, CHUNK)]
    cpu = 0.0
    with mp.get_context("fork").Pool(WORKERS, initializer=_init) as pool:
        for n, (a, b, j, f, sec) in enumerate(pool.imap_unordered(work, chunks, chunksize=1), 1):
            J[a - a0:b - a0] = j; F0[a - a0:b - a0] = f; cpu += sec
            if n % 500 == 0 or n == len(chunks):
                W.log(f"{n}/{len(chunks)} chunks, {time.time()-t0:.0f}s wall, {cpu:.0f} core-s")
    np.save(out / f"J_{tag}.npy", J); np.save(out / f"f0_{tag}.npy", F0)
    json.dump({"split": SPLIT, "K": K, "shard": SHARD, "nsh": NSH, "rows": [a0, b0], "step": d, "dmult": DMULT,
               "core_seconds": cpu, "wall_seconds": time.time() - t0, "nan_frac_J": float(np.isnan(J).mean()),
               "nan_frac_f0": float(np.isnan(F0).mean())}, open(out / f"meta_{tag}.json", "w"))
    W.log(f"DONE {tag}: {b0-a0} frames, nan J {np.isnan(J).mean():.5f}, wall {time.time()-t0:.0f}s, core-s {cpu:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
