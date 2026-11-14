#!/usr/bin/env python3
"""Step 3a — fast GPU PCA80 coefficient model on feature-v2.1 used as the OOF teacher.

Recipe = the metric track's A/B winner (MLP D-1024-1024-512-80 SiLU, AdamW 1e-3, OneCycle, 30 epochs,
batch 4096, global metric loss |e @ L|^2 with L = stage11_ch1/metric/L_80.npy; LOSS=raw gives |e|^2).
  * K shot-grouped folds -> out-of-fold coefficients for every train shot (mean over OOF_SEEDS)
  * full-train fits (FULL_SEEDS) -> holdout and public-test coefficients (seed bag)
Outputs ($CC_OUT): net_<T>_s<seed>.pt (full-train nets; cc_base_predict.py applies them to any feature dir), coef_oof_<T>.npy, coef_hold_<T>.npy (+ per seed), coef_test_<T>.npy, folds_<T>.npy,
preds_d3d_<T>.npz (holdout, harness layout, incumbent q95/betaN head bag), base_<T>.json
Env: TAG(gv21) KF(4) OOF_SEEDS(0,1) FULL_SEEDS(0,1,2,3) EPOCHS(30) LOSS(G) SMOKE(0)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import cc_common as cc
from cc_common import log

TAG = os.environ.get("TAG", "gv21")
KF = int(os.environ.get("KF", "4"))
OOF_SEEDS = [int(s) for s in os.environ.get("OOF_SEEDS", "0,1").split(",")]
FULL_SEEDS = [int(s) for s in os.environ.get("FULL_SEEDS", "0,1,2,3").split(",")]
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LOSS = os.environ.get("LOSS", "G")
SMOKE = int(os.environ.get("SMOKE", "0"))
BATCH, LR = 4096, 1e-3


def main():
    cc.OUT.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()
    X = torch.from_numpy(np.nan_to_num(np.load(cc.FEAT / "train/X.npy"))).to(dev)
    Xh = torch.from_numpy(np.nan_to_num(np.load(cc.FEAT / "holdout/X.npy"))).to(dev)
    Xt = torch.from_numpy(np.nan_to_num(np.load(cc.FEAT / "test/X.npy"))).to(dev)
    Y = torch.from_numpy(np.load(cc.FEAT / "train/y_C80.npy")).to(dev)
    Yh = np.load(cc.FEAT / "holdout/y_C80.npy").astype(np.float64)
    off = np.load(cc.FEAT / "train/offsets.npy")
    S, N, D = len(off) - 1, X.shape[0], X.shape[1]
    L = torch.tensor(np.load(cc.MET / "L_80.npy"), dtype=torch.float32, device=dev)
    mu, sd = Y.mean(0), Y.std(0)
    U = (Y - mu) / sd
    rng = np.random.default_rng(0)
    fold_of_shot = np.empty(S, np.int64)
    fold_of_shot[rng.permutation(S)] = np.arange(S) % KF
    frame_fold = torch.from_numpy(np.repeat(fold_of_shot, np.diff(off))).to(dev)
    np.save(cc.OUT / f"folds_{TAG}.npy", fold_of_shot)
    epochs = 2 if SMOKE else EPOCHS
    log(f"loaded X {tuple(X.shape)} in {time.time()-t0:.0f}s; KF={KF} loss={LOSS} epochs={epochs}")

    def fit(tr_idx, seed):
        torch.manual_seed(seed)
        net = torch.nn.Sequential(torch.nn.Linear(D, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 1024), torch.nn.SiLU(),
                                  torch.nn.Linear(1024, 512), torch.nn.SiLU(), torch.nn.Linear(512, cc.K)).to(dev)
        n = tr_idx.numel()
        steps = epochs * (n // BATCH)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-6)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
        g = torch.Generator(device=dev); g.manual_seed(seed)
        for _ in range(epochs):
            perm = tr_idx[torch.randperm(n, device=dev, generator=g)]
            for b in range(n // BATCH):
                idx = perm[b * BATCH:(b + 1) * BATCH]
                e = (net(X[idx]) - U[idx]) * sd
                loss = ((e @ L) ** 2).sum(1).mean() if LOSS == "G" else (e ** 2).sum(1).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        net.eval()
        return net

    @torch.no_grad()
    def pred(net, Z):
        return torch.cat([net(Z[i:i + 65536]) * sd + mu for i in range(0, Z.shape[0], 65536)])

    oof = torch.zeros((N, cc.K), device=dev)
    allidx = torch.arange(N, device=dev)
    for k in range(KF):
        te = allidx[frame_fold == k]
        tr = allidx[frame_fold != k]
        for s in OOF_SEEDS:
            t1 = time.time()
            net = fit(tr, 1000 * (k + 1) + s)
            oof[te] += pred(net, X[te]) / len(OOF_SEEDS)
            log(f"fold {k} seed {s}: {time.time()-t1:.0f}s")
    e_oof = (oof - Y).double().cpu().numpy()
    np.save(cc.OUT / f"coef_oof_{TAG}.npy", oof.cpu().numpy().astype(np.float32))
    ch, ct = [], []
    for s in FULL_SEEDS:
        net = fit(allidx, s)
        torch.save({"state": net.state_dict(), "mu": mu.cpu(), "sd": sd.cpu(), "D": D, "K": cc.K}, cc.OUT / f"net_{TAG}_s{s}.pt")
        ch.append(pred(net, Xh).double().cpu().numpy()); ct.append(pred(net, Xt).double().cpu().numpy())
        np.save(cc.OUT / f"coef_hold_{TAG}_s{s}.npy", ch[-1].astype(np.float32))
    Ch, Ct = np.mean(ch, 0), np.mean(ct, 0)
    np.save(cc.OUT / f"coef_hold_{TAG}.npy", Ch.astype(np.float32))
    np.save(cc.OUT / f"coef_test_{TAG}.npy", Ct.astype(np.float32))
    eh = Ch - Yh
    rec = {"tag": TAG, "KF": KF, "oof_seeds": OOF_SEEDS, "full_seeds": FULL_SEEDS, "epochs": epochs, "loss": LOSS, "D": D,
           "oof_e2_train": float((e_oof ** 2).sum(1).mean()), "full_bag_e2_holdout": float((eh ** 2).sum(1).mean()),
           "full_seed_e2_holdout": [float(((c - Yh) ** 2).sum(1).mean()) for c in ch],
           "oof_rms_per_coef": np.sqrt((e_oof ** 2).mean(0)).tolist(), "gpu_seconds": time.time() - t0}
    np.save(cc.OUT / f"cov_oof_{TAG}.npy", (e_oof.T @ e_oof) / len(e_oof))
    json.dump(rec, open(cc.OUT / f"base_{TAG}.json", "w"), indent=1)
    mean, comps = cc.pca80()
    hoff = np.load(cc.FEAT / "holdout/offsets.npy")
    h = np.load(cc.HC / "heads_mlp.npz")
    if len(h["q95_bag"]) != int(hoff[-1]):          # fold2 vendor patch: no qb source for this split
        log(f"skip preds_d3d_{TAG}.npz (incumbent q95/betaN only exist for holdout150)")
    else:
        cc.write_pred_npz(cc.OUT / f"preds_d3d_{TAG}.npz", Ch @ comps + mean[None, :], h["q95_bag"], h["betaN_bag"], hoff)
    log(f"done {TAG}: OOF e2 {rec['oof_e2_train']:.5f} | holdout full-bag e2 {rec['full_bag_e2_holdout']:.5f} "
        f"(seeds {[round(v, 5) for v in rec['full_seed_e2_holdout']]}) | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
