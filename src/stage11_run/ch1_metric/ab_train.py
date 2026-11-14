#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — step 6: A/B of the training LOSS only (same features, net, optimiser, seeds).
Frame-level MLP on feature-v2 (D=693) -> standardised PCA80 coefficients; loss variants:
  raw  |e|^2                      (= psi MSE, the incumbent objective)
  G    |e @ L|^2                  (global metric)
  N    a|e|^2 + |e @ Ln|^2/A^2    (amplitude-normalised metric)
  PF   a|e|^2 + |JW_i e|^2        (per-frame Gauss-Newton)
  HYB  a|e|^2 + PF rows (R_axis,Z_axis,kappa,volume,li) + N-metric (tri_top,tri_bot)
  Glcfs |e @ L_lcfs|^2            (G + the optional D_LCFS surrogate)
  GPF  0.5 G + 0.5 PF             (per-frame rows clipped at their 99.9 % row-norm quantile)
  PFlr PF at 0.3 x LR             (is the PF failure an optimisation instability?)
No early stopping, no holdout-based choice: fixed epochs; the holdout is only scored at the end.
Env: SCR OUT EPOCHS(30) SEEDS(0,1) BATCH(4096) LR(1e-3) VARIANTS(raw,G,N,PF,HYB)
Output: $OUT/ab/coef_ab_<variant>_s<seed>.npy, preds_d3d_ab_<variant>_{s0,bag}.npz, ab_train.json, names.txt
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metric_loss import Metric, N_W_HYBRID, PF_W_HYBRID  # noqa: E402

SCR = Path(os.environ["SCR"])
ROOT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/metric")))
AB = ROOT / "ab"
FEAT, HC = SCR / "stage11_ch1/feat_v2", SCR / "stage6_H"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
SEEDS = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",")]
BATCH = int(os.environ.get("BATCH", "4096"))
LR = float(os.environ.get("LR", "1e-3"))
VARIANTS = os.environ.get("VARIANTS", "raw,G,N,PF,HYB").split(",")
K = 80


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    AB.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    t0 = time.time()
    m = Metric(K, ROOT)
    X = torch.from_numpy(np.nan_to_num(np.load(FEAT / "train/X.npy"))).to(dev)
    Xh = torch.from_numpy(np.nan_to_num(np.load(FEAT / "holdout/X.npy"))).to(dev)
    Y = torch.from_numpy(np.load(FEAT / f"train/y_C{K}.npy")).to(dev)
    Yh = np.load(FEAT / f"holdout/y_C{K}.npy").astype(np.float64)
    mu, sd = Y.mean(0), Y.std(0)
    U = (Y - mu) / sd
    amp = torch.from_numpy(m.amp("train").astype(np.float32)).to(dev)
    JW = torch.from_numpy(np.load(ROOT / f"JW_train_{K}.npy")).to(dev)
    f32 = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)  # noqa: E731
    L, Ln, Ln_sub = f32(m.L), f32(m.Ln), f32(m.factor_N(N_W_HYBRID))
    pfw = f32(PF_W_HYBRID); a = float(m.a_psi)
    Llc = f32(Metric(K, ROOT, lcfs=True).L)
    q999 = json.load(open(ROOT / f"JW_{K}.json"))["train"]["rownorm_quantiles_50_90_99_99.9_100"]
    clipv = f32([q999[n][3] for n in ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]])
    N, D = X.shape
    log(f"loaded: X {tuple(X.shape)} JW {tuple(JW.shape)} in {time.time()-t0:.0f}s; gpu mem {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def loss_fn(v, e, idx):
        if v == "raw":
            return (e ** 2).sum(1)
        if v == "G":
            return ((e @ L) ** 2).sum(1)
        if v == "N":
            return a * (e ** 2).sum(1) + ((e @ Ln) ** 2).sum(1) / amp[idx] ** 2
        if v == "Glcfs":
            return ((e @ Llc) ** 2).sum(1)
        if v == "GPF":
            Jb = JW[idx]
            rn = Jb.norm(dim=2, keepdim=True).clamp_min(1e-12)
            Jb = Jb * torch.clamp(clipv[None, :, None] / rn, max=1.0)
            s = torch.einsum("bjk,bk->bj", Jb, e) ** 2
            return 0.5 * ((e @ L) ** 2).sum(1) + 0.5 * (a * (e ** 2).sum(1) + s.sum(1))
        s = torch.einsum("bjk,bk->bj", JW[idx], e) ** 2
        if v in ("PF", "PFlr"):
            return a * (e ** 2).sum(1) + s.sum(1)
        if v == "HYB":
            return a * (e ** 2).sum(1) + (s * pfw[None, :]).sum(1) + ((e @ Ln_sub) ** 2).sum(1) / amp[idx] ** 2
        raise ValueError(v)

    JWh = np.load(ROOT / f"JW_holdout_{K}.npy"); amph = m.amp("holdout")
    cache = np.load(HC / "cache_hold.npz"); offsets = cache["offsets"]
    heads = np.load(HC / "heads_mlp.npz"); pc = np.load(HC / f"pca_{K}.npz")
    comps, mean = pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64)
    psi_h = np.load(HC / "cache_hold_psi.npy", mmap_mode="r")
    ss_tot = None

    def write_preds(name, C):
        psi = C @ comps + mean[None, :]
        d = {}
        for s in range(len(offsets) - 1):
            p, q = offsets[s], offsets[s + 1]
            d[f"shot_{s:04d}_psirz"] = psi[p:q].reshape(-1, 65, 65).astype(np.float32)
            d[f"shot_{s:04d}_q95"] = heads["q95_bag"][p:q].astype(np.float32)
            d[f"shot_{s:04d}_betaN"] = heads["betaN_bag"][p:q].astype(np.float32)
        np.savez(AB / f"preds_d3d_{name}.npz", **d)

    rec, names = {}, []
    steps = EPOCHS * (N // BATCH)
    for v in VARIANTS:
        cs = []
        for seed in SEEDS:
            torch.manual_seed(seed); np.random.seed(seed)
            net = torch.nn.Sequential(torch.nn.Linear(D, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 1024), torch.nn.SiLU(),
                                      torch.nn.Linear(1024, 512), torch.nn.SiLU(), torch.nn.Linear(512, K)).to(dev)
            lr = LR * (0.3 if v == "PFlr" else 1.0)
            opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-6)
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
            g = torch.Generator(device=dev); g.manual_seed(seed)
            t1 = time.time(); last = 0.0
            for ep in range(EPOCHS):
                perm = torch.randperm(N, device=dev, generator=g)
                tot = 0.0
                for b in range(N // BATCH):
                    idx = perm[b * BATCH:(b + 1) * BATCH]
                    e = (net(X[idx]) - U[idx]) * sd
                    loss = loss_fn(v, e, idx).mean()
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
                    tot += float(loss.detach())
                last = tot / (N // BATCH)
            with torch.no_grad():
                Ch = (net(Xh) * sd + mu).double().cpu().numpy()
            np.save(AB / f"coef_ab_{v}_s{seed}.npy", Ch.astype(np.float32))
            cs.append(Ch)
            eh = Ch - Yh
            r = {"train_loss_last_epoch": last, "sec": time.time() - t1, "hold_e2": float((eh ** 2).sum(1).mean()),
                 "hold_G": float(m.loss_G(Ch, Yh).mean()), "hold_N": float(m.loss_N(Ch, Yh, amph).mean()),
                 "hold_PF": float(m.loss_PF(Ch, Yh, JWh).mean())}
            rec[f"{v}_s{seed}"] = r
            log(f"{v} seed {seed}: {json.dumps({k: round(x, 6) for k, x in r.items()})}")
        write_preds(f"ab_{v}_s{SEEDS[0]}", cs[0]); names.append(f"ab_{v}_s{SEEDS[0]}")
        if len(cs) > 1:
            Cb = np.mean(cs, 0); eh = Cb - Yh
            rec[f"{v}_bag"] = {"hold_e2": float((eh ** 2).sum(1).mean()), "hold_G": float(m.loss_G(Cb, Yh).mean()),
                               "hold_N": float(m.loss_N(Cb, Yh, amph).mean()), "hold_PF": float(m.loss_PF(Cb, Yh, JWh).mean())}
            write_preds(f"ab_{v}_bag", Cb); names.append(f"ab_{v}_bag")
            log(f"{v} bag{len(cs)}: {json.dumps({k: round(x, 6) for k, x in rec[f'{v}_bag'].items()})}")
    tagj = os.environ.get("AB_TAG", "")
    json.dump({"epochs": EPOCHS, "seeds": SEEDS, "batch": BATCH, "lr": LR, "net": "MLP 693-1024-1024-512-80 SiLU, AdamW wd 1e-6, OneCycle",
               "results": rec, "gpu_seconds_total": time.time() - t0}, open(AB / f"ab_train{tagj}.json", "w"), indent=1)
    (AB / "names.txt").write_text("\n".join(names) + "\n")
    log(f"done in {time.time()-t0:.0f}s (GPU wall incl. load)")


if __name__ == "__main__":
    main()
