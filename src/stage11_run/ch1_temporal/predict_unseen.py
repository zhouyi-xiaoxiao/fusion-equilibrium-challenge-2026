#!/usr/bin/env python3
"""
Stage 11 / ch1_temporal — inputs-only inference for window-model checkpoints written by win_train.py, on ANY feature-v2.1
directory written by ch1_features/build_unseen.py (or an existing feat_v21/{holdout,test} cache).  Reads no labels.

  python predict_unseen.py --feat DIR --weights a.pt [b.pt ...] --out preds_d3d_X.npz
         [--members-out bag_X.npz] [--basis $SCR/stage6_H/pca_80.npz] [--device cuda|cpu] [--chunk 16384]

  DIR must hold X.npy (standardised with scaler_v21), offsets.npy, times.npy (EFIT frame times, ms) and files.txt.
  Output (harness / Codabench layout): shot_{i:04d}_psirz (n,65,65), shot_{i:04d}_q95 (n,), shot_{i:04d}_betaN (n,),
  i = row of DIR/files.txt, frames in DIR order, ALL arrays float32.  The members are averaged in coefficient space
  (float64), psi = mean coef @ comps_ + mean_.  --members-out also stores the bagged coef/q95/betaN (float32, DIR frame order).
  The window construction (offsets clamped at the shot ends + validity flag, dt = (t_j - t_i)/100 ms clipped to +-5) and the
  decode are the same code path as win_train.py, so on the training machine the result matches the members written at
  training time (checked in run_check_unseen.sh).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from win_train import WinModel  # noqa: E402  (pure nn.Module definitions; nothing is trained on import)

SCR = Path(os.environ["SCR"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--members-out", default=None)
    ap.add_argument("--basis", default=None, help="PCA basis npz (comps_, mean_); default $SCR/stage6_H/pca_<K>.npz")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--chunk", type=int, default=16384)
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.device)
    fd = Path(a.feat)
    off = np.load(fd / "offsets.npy").astype(np.int64)
    files = [l for l in (fd / "files.txt").read_text().split("\n") if l.strip()]
    assert len(files) == len(off) - 1, (len(files), len(off))
    Xall = np.nan_to_num(np.load(fd / "X.npy"))
    n = Xall.shape[0]
    assert off[-1] == n
    fs, fe = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for s in range(len(off) - 1):
        fs[off[s]:off[s + 1]] = off[s]
        fe[off[s]:off[s + 1]] = off[s + 1]
    fs_t, fe_t = torch.from_numpy(fs).to(dev), torch.from_numpy(fe).to(dev)
    t_t = torch.from_numpy(np.asarray(np.load(fd / "times.npy"), np.float32)).to(dev)
    Cs, qs, bs, K = [], [], [], None
    for wp in a.weights:
        ck = torch.load(wp, map_location=dev, weights_only=False)
        cfg, wcfg, offs = ck["cfg"], ck["wcfg"], list(ck["offs"])
        n_out = ck["net"]["trunk.head.weight"].shape[0]
        net = WinModel(cfg["DX"], n_out, cfg, offs, wcfg).to(dev)
        net.load_state_dict(ck["net"])
        net.eval()
        nrm = {k: v.to(dev) for k, v in ck["nrm"].items()}
        kk = int(nrm["muC"].shape[0])
        assert K in (None, kk), (K, kk)
        K = kk
        X = torch.from_numpy(np.ascontiguousarray(Xall[:, :cfg["DX"]])).to(dev)
        o = torch.tensor(offs, device=dev)[None, :]
        outs = []
        with torch.no_grad():
            for s in range(0, n, a.chunk):
                i = torch.arange(s, min(s + a.chunk, n), device=dev)
                j = i[:, None] + o
                valid = (j >= fs_t[i][:, None]) & (j < fe_t[i][:, None])
                j = torch.where(valid, j, i[:, None])
                dt = ((t_t[j] - t_t[i][:, None]) / 100.0).clamp(-5.0, 5.0)
                outs.append(net(X[j], dt, valid.to(torch.float32)))
        out = torch.cat(outs)
        Cs.append((out[:, :K] * nrm["sdC"] + nrm["muC"]).double().cpu().numpy())
        QB = (out[:, K:K + 2] * nrm["sdQ"] + nrm["muQ"]).double().cpu().numpy()
        qs.append(QB[:, 0])
        bs.append(QB[:, 1])
        print(f"member {os.path.basename(wp)}: seed {ck.get('seed')} offs {offs} frames {n}", flush=True)
        del X, net, out
    C, q, b = np.mean(Cs, 0), np.mean(qs, 0), np.mean(bs, 0)
    pc = np.load(a.basis or SCR / "stage6_H" / f"pca_{K}.npz")
    comps, pmean = pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64)
    if a.members_out:
        np.savez(a.members_out, coef=C.astype(np.float32), q95=q.astype(np.float32), betaN=b.astype(np.float32),
                 basis=str(a.basis or f"stage6_H/pca_{K}.npz"), weights=np.array([os.path.basename(w) for w in a.weights]),
                 files=np.array(files))
    d = {}
    for s in range(len(off) - 1):
        p, e = off[s], off[s + 1]
        d[f"shot_{s:04d}_psirz"] = (C[p:e] @ comps + pmean[None, :]).reshape(-1, 65, 65).astype(np.float32)
        d[f"shot_{s:04d}_q95"] = q[p:e].astype(np.float32)
        d[f"shot_{s:04d}_betaN"] = b[p:e].astype(np.float32)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **d)
    print(f"wrote {a.out}: {len(off) - 1} shots, {n} frames, {len(a.weights)} members, device {dev}, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
