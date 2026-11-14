#!/usr/bin/env python3
"""ch1_package_final — FM psi bag inference (the 11 FULL frame-MLP members; no q95/betaN nets: FcCX takes q95/betaN from SC).
Same code path as ch1_frame_mlp/fm_predict.py (load_member / expand / fm_train.predict), member order = pattern order.
  FEAT=<canonical cache with spec.json> pf_fm.py --feat-dir DIR --members-root R --psi 'A*+B*+C*' --out fm.npz [--save-raw D] [--device cuda|cpu]"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "stage11_run/ch1_frame_mlp"))
import fm_predict as P  # noqa: E402

T = P.T
ap = argparse.ArgumentParser()
for k in ("feat-dir", "members-root", "psi", "out"):
    ap.add_argument("--" + k, required=True)
ap.add_argument("--save-raw", default=None); ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()
fd = Path(a.feat_dir)
X = np.nan_to_num(np.load(fd / "X.npy")).astype(np.float32)
off = np.load(fd / "offsets.npy")
spec = json.load(open(P.FEAT / "spec.json"))
assert X.shape[1] == max(v[1] for v in spec["blocks"].values()), X.shape
dev = torch.device(a.device)
if dev.type == "cpu":
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "32")))
Xd = torch.from_numpy(X).to(dev)
root = Path(a.members_root)
raw = Path(a.save_raw) if a.save_raw else None
if raw:
    raw.mkdir(parents=True, exist_ok=True)
pcs, psi, n = {}, None, 0
for m in P.expand(a.psi, root):
    net, cfg, cidx, nm = P.load_member(root / m / "model.pt", spec, X.shape[1], dev)
    assert cfg["target"] == "coef", m
    o = T.predict(net, Xd[:, torch.from_numpy(cidx).to(dev)] if len(cidx) != X.shape[1] else Xd)[0]
    C = (o.cpu() * nm["sd"] + nm["mu"]).numpy().astype(np.float32)
    if raw:
        np.save(raw / f"coef_{m}.npy", C)
    K = cfg["K"]
    if K not in pcs:
        pc = np.load(P.HC / f"pca_{K}.npz")
        pcs[K] = (pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64))
    p = C.astype(np.float64) @ pcs[K][0] + pcs[K][1][None, :]
    psi = p if psi is None else psi + p
    n += 1
    print(f"psi member {m} (K={K}, {cfg['arch']}) on {dev}", flush=True)
psi /= n
np.savez(a.out, **{f"shot_{s:04d}_psirz": psi[off[s]:off[s + 1]].reshape(-1, 65, 65).astype(np.float32) for s in range(len(off) - 1)})
print(f"PF_FM wrote {a.out}: {len(off) - 1} shots, {len(X)} frames, {n} members, device {dev}", flush=True)
