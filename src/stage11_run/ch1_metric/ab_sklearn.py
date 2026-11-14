#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — A/B with the INCUMBENT recipe itself: sklearn MLPRegressor(512,256,128),
early_stopping, seed 42, v1 features (stage6_H Xfs.npy) — identical to coef_p80_h512x256x128_s42 except the
TARGET is z = C80 @ L (global metric G) instead of C80; holdout coefficients = predict(Xhs) @ Linv.
Baseline (official log, same seed/recipe/features, raw targets): S 0.9885, Consistency 0.9571.
Env: SCR OUT THREADS(32) SEED(42)
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_loss import Metric
SCR = Path(os.environ["SCR"])
ROOT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/metric"))); AB = ROOT / "ab"; HC = SCR / "stage6_H"
SEED = int(os.environ.get("SEED", "42"))
from threadpoolctl import threadpool_limits
from sklearn.neural_network import MLPRegressor
threadpool_limits(int(os.environ.get("THREADS", "32")))
AB.mkdir(parents=True, exist_ok=True)
m = Metric(80, ROOT); t = time.time()
Xfs = np.ascontiguousarray(np.load(HC / "Xfs.npy")); C = np.load(HC / "C80.npy").astype(np.float64)
Z = (C @ m.L).astype(np.float32)
hold = np.load(HC / "cache_hold.npz"); Xhs = hold["Xhs"]; offsets = hold["offsets"]
net = MLPRegressor(hidden_layer_sizes=(512, 256, 128), max_iter=500, early_stopping=True, random_state=SEED)
net.fit(Xfs, Z)
Ch = net.predict(Xhs).astype(np.float64) @ m.Linv
name = f"ab_skG_s{SEED}"
np.save(AB / f"coef_{name}.npy", Ch.astype(np.float32))
pc = np.load(HC / "pca_80.npz"); heads = np.load(HC / "heads_mlp.npz")
psi = Ch @ pc["comps_"].astype(np.float64) + pc["mean_"].astype(np.float64)[None, :]
d = {}
for s in range(len(offsets) - 1):
    p, q = offsets[s], offsets[s + 1]
    d[f"shot_{s:04d}_psirz"] = psi[p:q].reshape(-1, 65, 65).astype(np.float32)
    d[f"shot_{s:04d}_q95"] = heads["q95_bag"][p:q].astype(np.float32); d[f"shot_{s:04d}_betaN"] = heads["betaN_bag"][p:q].astype(np.float32)
np.savez(AB / f"preds_d3d_{name}.npz", **d)
json.dump({"n_iter": int(net.n_iter_), "fit_min": (time.time() - t) / 60, "best_val": float(net.best_validation_score_)}, open(AB / f"{name}.json", "w"))
print(f"sklearn G-target fit done: n_iter {net.n_iter_}, {(time.time()-t)/60:.1f} min", flush=True)
