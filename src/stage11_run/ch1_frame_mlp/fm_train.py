#!/usr/bin/env python3
"""
Stage 11 / ch1_frame_mlp — per-frame residual MLP on FEATURE-V2.1 (one GPU, fp32).

One process loads the feature cache once (train + holdout + test on the GPU) and trains a list of
configs from a JSON file ({"defaults": {...}, "runs": [{"name": ..., ...}, ...]}).

Targets
  target=coef : standardised PCA-K coefficients (stage6_H bases).  loss=metric -> |e @ L_K|^2 (score units,
                e in raw coefficient units), loss=mse -> a_psi |e|^2, loss=metric_lcfs -> L_K_lcfs.
                Optional heads on the same trunk: aux (17 scorer-derived scalars of y_aux, never efit_*),
                qb (q95, betaN).
  target=qb   : a separate net for (q95, betaN); loss = 0.15 * mean standardised squared error
                (= the R2_{q95,betaN} term of the score in original units).
Train sets
  t95  : the 6,891 train shots minus a fixed 5 % shot-disjoint validation slice (seed 20261011); the EMA
         model is evaluated on that slice after every epoch (epoch selection / early-stop diagnostics).
  t100 : all 6,891 train shots, fixed epochs, no validation.
  full : train + holdout = all 7,041 shots, fixed epochs (FINAL recipe; holdout outputs are in-sample).
The 150-shot holdout is never used for selection here: a coefficient-space proxy is only logged.
Outputs per run: $OUT/members/<name>/{coef_hold,coef_test[,coef_val]}.npy, qb_*.npy, aux_hold.npy, model.pt,
meta.json (config, val curve, timings, proxies).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SCR = Path(os.environ["SCR"])
FEAT = Path(os.environ.get("FEAT", str(SCR / "stage11_ch1/feat_v21")))
MET = Path(os.environ.get("METRIC", str(SCR / "stage11_ch1/metric")))
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/frame_mlp")))
LOSS_SCALE = 100.0
AUX_FROM = 3            # y_aux[:, :3] = efit_li, efit_r_axis, efit_z_axis -> never supervised
VAL_SEED, VAL_FRAC = 20261011, 0.05
TH_BLOCKS = ("th1", "tsc", "tscm", "tse", "tsem", "phys", "tsv")

DEFAULTS = dict(cols="all", K=80, loss="metric", target="coef", arch="res", width=1024, blocks=4, hidden=[512, 512],
                tower=[768, 384], dropout=0.05, epochs=40, lr=1e-3, lr_min=1e-5, wd=1e-4, batch=4096, warm=2,
                ema=0.999, clip=1.0, seed=0, aux_w=0.0, qb_w=0.0, trainset="t95", skip=True, in_noise=0.0, drop="", ctx="")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def block_cols(cols, blocks):
    if cols == "all":
        names = list(blocks)
    elif cols == "v2":
        names = [b for b in blocks if b not in ("tsv", "drv", "rec")]
    elif cols == "norec":
        names = [b for b in blocks if b != "rec"]
    else:
        names = cols.split("+")
    return names


class Data:
    def __init__(self, dev):
        t0 = time.time()
        self.dev = dev
        spec = json.load(open(FEAT / "spec.json"))
        self.blocks = spec["blocks"]
        self.aux_names = spec["aux_names"][AUX_FROM:]
        ld = lambda sp, n: np.load(FEAT / sp / n)  # noqa: E731
        xtr, xh = np.nan_to_num(ld("train", "X.npy")), np.nan_to_num(ld("holdout", "X.npy"))
        self.ntr, self.nh = len(xtr), len(xh)
        self.X = torch.from_numpy(np.concatenate([xtr, xh])).to(dev)          # rows [0,ntr) train, [ntr,ntr+nh) holdout
        self.Xt = torch.from_numpy(np.nan_to_num(ld("test", "X.npy"))).to(dev)
        del xtr, xh
        cat = lambda n: np.concatenate([ld("train", n), ld("holdout", n)])  # noqa: E731
        self.qb = torch.from_numpy(np.stack([cat("y_q95.npy"), cat("y_betaN.npy")], 1).astype(np.float32)).to(dev)
        self.aux = torch.from_numpy(cat("y_aux.npy")[:, AUX_FROM:].astype(np.float32)).to(dev)
        self._Y = {}
        sid = ld("train", "shot_id.npy").astype(np.int64)
        nshot = int(sid.max()) + 1
        perm = np.random.default_rng(VAL_SEED).permutation(nshot)
        isval = np.zeros(nshot, bool)
        isval[perm[:int(round(VAL_FRAC * nshot))]] = True
        vm = isval[sid]
        self.val_rows = torch.from_numpy(np.flatnonzero(vm)).to(dev)
        self.t95_rows = torch.from_numpy(np.flatnonzero(~vm)).to(dev)
        self.n_val_shots = int(isval.sum())
        (OUT / "members").mkdir(parents=True, exist_ok=True)
        if not (OUT / "val_rows.npy").exists():
            np.save(OUT / "val_rows.npy", np.flatnonzero(vm))
        log(f"data: X {tuple(self.X.shape)} test {tuple(self.Xt.shape)} val shots {self.n_val_shots} "
            f"({len(self.val_rows)} frames) in {time.time()-t0:.0f}s; gpu {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def ctx(self):
        """Per-shot mean / std of the standardised features (inputs only), broadcast to every frame of the shot."""
        if not hasattr(self, "_ctx"):
            def agg(X, off):
                off = torch.as_tensor(np.asarray(off), device=X.device)
                n = (off[1:] - off[:-1])
                sid = torch.repeat_interleave(torch.arange(len(n), device=X.device), n)
                m = torch.zeros(len(n), X.shape[1], device=X.device).index_add_(0, sid, X) / n[:, None]
                v = torch.zeros(len(n), X.shape[1], device=X.device).index_add_(0, sid, X ** 2) / n[:, None] - m ** 2
                return m[sid], v.clamp_min(0).sqrt()[sid]
            otr, oh = np.load(FEAT / "train/offsets.npy"), np.load(FEAT / "holdout/offsets.npy")
            off = np.concatenate([otr, oh[1:] + otr[-1]])
            self._ctx = (agg(self.X, off), agg(self.Xt, np.load(FEAT / "test/offsets.npy")))
        return self._ctx

    def Y(self, K):
        if K not in self._Y:
            y = np.concatenate([np.load(FEAT / "train" / f"y_C{K}.npy"), np.load(FEAT / "holdout" / f"y_C{K}.npy")])
            self._Y[K] = torch.from_numpy(y.astype(np.float32)).to(self.dev)
        return self._Y[K]


class Block(nn.Module):
    def __init__(self, w, p):
        super().__init__()
        self.n, self.a, self.b, self.d = nn.LayerNorm(w), nn.Linear(w, w), nn.Linear(w, w), nn.Dropout(p)
        nn.init.zeros_(self.b.weight); nn.init.zeros_(self.b.bias)

    def forward(self, x):
        return x + self.b(self.d(F.silu(self.a(self.n(x)))))


class Net(nn.Module):
    def __init__(self, din, cfg, nout, naux, nqb, tower_idx=None):
        super().__init__()
        self.arch, w, p = cfg["arch"], cfg["width"], cfg["dropout"]
        if self.arch == "res":
            self.inp = nn.Linear(din, w)
        elif self.arch == "tower":
            ic, it = tower_idx
            self.register_buffer("ic", ic); self.register_buffer("it", it)
            wc, wt = cfg["tower"]
            self.in_c = nn.Sequential(nn.Linear(len(ic), wc), Block(wc, p))
            self.in_t = nn.Sequential(nn.Linear(len(it), wt), Block(wt, p))
            self.merge = nn.Linear(wc + wt, w)
        elif self.arch == "mlp":
            dims, layers = [din] + list(cfg["hidden"]), []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.SiLU()] + ([nn.Dropout(p)] if p > 0 else [])
            self.mlp = nn.Sequential(*layers)
            w = dims[-1]
        else:
            raise ValueError(self.arch)
        if self.arch in ("res", "tower"):
            self.body = nn.Sequential(*[Block(w, p) for _ in range(cfg["blocks"])])
            self.norm = nn.LayerNorm(w)
        self.head = nn.Linear(w, nout)
        self.skip = nn.Linear(din, nout, bias=False) if cfg["skip"] else None
        self.aux = nn.Linear(w, naux) if naux else None
        self.qbh = nn.Linear(w, nqb) if nqb else None

    def trunk(self, x):
        if self.arch == "mlp":
            return self.mlp(x)
        if self.arch == "res":
            h = self.inp(x)
        else:
            h = self.merge(torch.cat([self.in_c(x[:, self.ic]), self.in_t(x[:, self.it])], 1))
        return self.norm(self.body(h))

    def forward(self, x):
        h = self.trunk(x)
        y = self.head(h)
        if self.skip is not None:
            y = y + self.skip(x)
        return y, (self.aux(h) if self.aux is not None else None), (self.qbh(h) if self.qbh is not None else None)


@torch.no_grad()
def predict(model, X, bs=65536):
    model.eval()
    outs = [[], [], []]
    for i in range(0, len(X), bs):
        o = model(X[i:i + bs])
        for k in range(3):
            if o[k] is not None:
                outs[k].append(o[k].float())
    return [torch.cat(o) if o else None for o in outs]


def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run(cfg, D):
    dev = D.dev
    name = cfg["name"]
    odir = OUT / "members" / name
    if (odir / "meta.json").exists():
        log(f"{name}: exists, skip")
        return
    odir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    names = block_cols(cfg["cols"], D.blocks)
    cidx = np.concatenate([np.arange(*D.blocks[b]) for b in names])
    if cfg.get("drop"):                                         # regex on column names (feature ablations)
        import re
        cn = (FEAT / "names.txt").read_text().split()
        cidx = np.array([i for i in cidx if not re.search(cfg["drop"], cn[i])])
    full_cols = len(cidx) == D.X.shape[1]
    if full_cols:
        X, Xt = D.X, D.Xt
    else:
        ci = torch.from_numpy(cidx).to(dev)
        X, Xt = D.X[:, ci].contiguous(), D.Xt[:, ci].contiguous()
    if cfg.get("ctx"):
        (m, sdv), (mt, sdt) = D.ctx()
        ci = torch.from_numpy(cidx).to(dev)
        ex = [m[:, ci]] + ([sdv[:, ci]] if cfg["ctx"] == "meanstd" else [])
        ext = [mt[:, ci]] + ([sdt[:, ci]] if cfg["ctx"] == "meanstd" else [])
        X, Xt = torch.cat([X] + ex, 1), torch.cat([Xt] + ext, 1)
    din = X.shape[1]
    tower_idx = None
    if cfg["arch"] == "tower":
        isth = np.zeros(D.X.shape[1], bool)
        for b in TH_BLOCKS:
            isth[D.blocks[b][0]:D.blocks[b][1]] = True
        ic, it = np.flatnonzero(~isth[cidx]).tolist(), np.flatnonzero(isth[cidx]).tolist()
        tower_idx = (torch.tensor(ic, device=dev), torch.tensor(it, device=dev))
    ts = cfg["trainset"]
    rows = {"t95": D.t95_rows, "t100": torch.arange(D.ntr, device=dev), "full": torch.arange(D.ntr + D.nh, device=dev)}[ts]
    N = len(rows)
    hold = slice(D.ntr, D.ntr + D.nh)
    K, target = cfg["K"], cfg["target"]
    qmu, qsd = D.qb[rows].mean(0), D.qb[rows].std(0)
    QB = (D.qb - qmu) / qsd
    ar = D.aux[rows]
    afin = torch.isfinite(ar)
    amu = torch.where(afin, ar, torch.zeros_like(ar)).sum(0) / afin.sum(0)
    asd = (torch.where(afin, (ar - amu) ** 2, torch.zeros_like(ar)).sum(0) / afin.sum(0)).sqrt()
    AMASK = torch.isfinite(D.aux).float()                       # failed scorer derivations are NaN -> masked
    AUX = torch.nan_to_num((D.aux - amu) / asd).clamp(-10, 10)
    del ar, afin
    if target == "coef":
        Y = D.Y(K)
        mu, sd = Y[rows].mean(0), Y[rows].std(0)
        U = (Y - mu) / sd
        meta_m = json.load(open(MET / f"metric_{K}.json"))
        a_psi = float(meta_m["a_psi"])
        Lnp = np.load(MET / (f"L_{K}_lcfs.npy" if cfg["loss"] == "metric_lcfs" else f"L_{K}.npy"))
        L = torch.tensor(Lnp, dtype=torch.float32, device=dev)
        Lg = torch.tensor(np.load(MET / f"L_{K}.npy"), dtype=torch.float32, device=dev)
        nout, nqb = K, (2 if cfg["qb_w"] > 0 else 0)
    else:
        nout, nqb = 2, 0
    naux = AUX.shape[1] if cfg["aux_w"] > 0 else 0
    net = Net(din, cfg, nout, naux, nqb, tower_idx).to(dev)
    nparam = sum(p.numel() for p in net.parameters())
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"]))
    decay = [p for n_, p in net.named_parameters() if p.ndim >= 2]
    nodecay = [p for n_, p in net.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=cfg["lr"], eps=1e-10)
    B, E = cfg["batch"], cfg["epochs"]
    spe = N // B
    total, warm = E * spe, cfg["warm"] * spe

    def lr_at(t):
        if t < warm:
            return cfg["lr"] * (t + 1) / warm
        return cfg["lr_min"] + 0.5 * (cfg["lr"] - cfg["lr_min"]) * (1 + math.cos(math.pi * (t - warm) / max(1, total - warm)))

    def main_loss(o, b):
        if target == "coef":
            e = (o - U[b]) * sd
            if cfg["loss"] == "mse":
                return a_psi * (e ** 2).sum(1).mean()
            return ((e @ L) ** 2).sum(1).mean()
        return 0.15 * ((o - QB[b]) ** 2).mean()

    @torch.no_grad()
    def val_metrics(model):
        vr = D.val_rows
        o, _, q = predict(model, X[vr])
        if target == "coef":
            e = (o - U[vr]) * sd
            d = {"G": float(((e @ Lg) ** 2).sum(1).mean()), "e2": float((e ** 2).sum(1).mean())}
            if q is not None:
                d["qb"] = float(0.15 * ((q - QB[vr]) ** 2).mean() * (qsd ** 2 / D.qb[vr].var(0)).mean())
            d["sel"] = d["G"] + cfg["qb_w"] * d.get("qb", 0.0) if cfg["qb_w"] > 0 else d["G"]
            return d
        mse = ((o - QB[vr]) ** 2).mean(0) * qsd ** 2 / D.qb[vr].var(0)
        return {"1-R2_q95": float(mse[0]), "1-R2_betaN": float(mse[1]), "sel": float(mse.mean())}

    g = torch.Generator(device=dev); g.manual_seed(cfg["seed"])
    curve, best, step = [], None, 0
    t_train = time.time()
    for ep in range(E):
        net.train()
        perm = rows[torch.randperm(N, device=dev, generator=g)]
        tot = 0.0
        for i in range(spe):
            b = perm[i * B:(i + 1) * B]
            for gq in opt.param_groups:
                gq["lr"] = lr_at(step)
            xb = X[b]
            if cfg["in_noise"] > 0:
                xb = xb + cfg["in_noise"] * torch.randn_like(xb)
            o, a, q = net(xb)
            loss = main_loss(o, b)
            if a is not None:
                mb = AMASK[b]
                loss = loss + cfg["aux_w"] * (((a - AUX[b]) ** 2) * mb).sum() / mb.sum()
            if q is not None:
                loss = loss + cfg["qb_w"] * 0.15 * ((q - QB[b]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            (LOSS_SCALE * loss).backward()
            if cfg["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
            opt.step(); ema.update_parameters(net); step += 1
            tot += float(loss.detach())
        rec = {"ep": ep + 1, "train": tot / spe}
        if ts == "t95":
            rec.update(val_metrics(ema))
            if best is None or rec["sel"] < best[0]:
                best = (rec["sel"], ep + 1, copy.deepcopy(ema.state_dict()))
        curve.append(rec)
        if ep == 0 or (ep + 1) % 10 == 0 or ep + 1 == E:
            log(f"{name} ep {ep+1}/{E} " + " ".join(f"{k} {v:.6f}" for k, v in rec.items() if k != "ep")
                + f" | {(time.time()-t_train)/(ep+1):.1f}s/ep")
    sec_train = time.time() - t_train

    meta = {"cfg": cfg, "din": din, "n_params": nparam, "n_train_frames": N, "steps": total, "sec_train": sec_train,
            "sec_per_epoch": sec_train / E, "curve": curve, "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}

    def dump(model, sfx):
        oh, ah, qh = predict(model, X[hold]); ot, _, qt = predict(model, Xt)
        d = {}
        if target == "coef":
            ch = (oh * sd + mu).cpu().numpy().astype(np.float32)
            np.save(odir / f"coef_hold{sfx}.npy", ch)
            np.save(odir / f"coef_test{sfx}.npy", (ot * sd + mu).cpu().numpy().astype(np.float32))
            e = torch.from_numpy(ch).to(dev) - D.Y(K)[hold]
            d["hold_G"] = float(((e @ Lg) ** 2).sum(1).mean()); d["hold_e2"] = float((e ** 2).sum(1).mean())
            if ts == "t95" and sfx == "":
                ov, _, qv = predict(model, X[D.val_rows])
                np.save(odir / "coef_val.npy", (ov * sd + mu).cpu().numpy().astype(np.float32))
                if qv is not None:
                    np.save(odir / "qb_val.npy", (qv * qsd + qmu).cpu().numpy().astype(np.float32))
        else:
            qh, qt = oh, ot
            if ts == "t95" and sfx == "":
                ov, _, _ = predict(model, X[D.val_rows])
                np.save(odir / "qb_val.npy", (ov * qsd + qmu).cpu().numpy().astype(np.float32))
        if qh is not None:
            qhn = (qh * qsd + qmu).cpu().numpy().astype(np.float32)
            np.save(odir / f"qb_hold{sfx}.npy", qhn)
            np.save(odir / f"qb_test{sfx}.npy", (qt * qsd + qmu).cpu().numpy().astype(np.float32))
            yq = D.qb[hold].cpu().numpy()
            d["hold_r2_q95"], d["hold_r2_betaN"] = r2(yq[:, 0], qhn[:, 0]), r2(yq[:, 1], qhn[:, 1])
        if ah is not None and sfx == "":
            an = (ah * asd + amu).cpu().numpy().astype(np.float32)
            np.save(odir / "aux_hold.npy", an)
            ya = D.aux[hold].cpu().numpy()
            fin = np.isfinite(ya)
            d["hold_r2_aux"] = {n_: round(r2(ya[fin[:, j], j], an[fin[:, j], j]), 5) for j, n_ in enumerate(D.aux_names)}
        return d

    meta["last"] = dump(ema, "")
    if ts == "t95":
        meta["val_last"], meta["val_best"], meta["best_epoch"] = curve[-1]["sel"], best[0], best[1]
        if best[1] != E:
            bm = copy.deepcopy(ema); bm.load_state_dict(best[2])
            meta["best"] = dump(bm, "_best")
    torch.save({"state": ema.module.state_dict(), "cfg": cfg, "cols": names,
                "norm": {"qmu": qmu.cpu(), "qsd": qsd.cpu(), "amu": amu.cpu(), "asd": asd.cpu(),
                         **({"mu": mu.cpu(), "sd": sd.cpu()} if target == "coef" else {})}}, odir / "model.pt")
    meta["sec_total"] = time.time() - t0
    json.dump(meta, open(odir / "meta.json", "w"), indent=1)
    log(f"{name} DONE params {nparam/1e6:.2f}M train {sec_train:.0f}s total {meta['sec_total']:.0f}s "
        f"last {json.dumps({k: (round(v, 6) if isinstance(v, float) else '..') for k, v in meta['last'].items()})}"
        + (f" best_ep {meta['best_epoch']}/{E} val_last {meta['val_last']:.6f} val_best {meta['val_best']:.6f}" if ts == "t95" else ""))
    del net, ema, opt, X, Xt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    spec = json.load(open(args.configs))
    si, sn = [int(x) for x in args.shard.split("/")]
    runs = []
    for r in spec["runs"]:
        seeds = r.pop("seeds", None)
        base = {**DEFAULTS, **spec.get("defaults", {}), **r}
        if seeds is None:
            runs.append(base)
        else:
            runs += [{**base, "seed": s, "name": f"{base['name']}_s{s}"} for s in seeds]
    if args.only:
        runs = [r for r in runs if r["name"] in args.only.split(",")]
    runs = runs[si::sn]
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda")
    log(f"shard {si}/{sn}: {len(runs)} runs on {torch.cuda.get_device_name(0)}: {[r['name'] for r in runs]}")
    t0 = time.time()
    D = Data(dev)
    for cfg in runs:
        try:
            run(cfg, D)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"{cfg['name']} FAILED: {ex}")
            torch.cuda.empty_cache()
    log(f"shard {si}/{sn} finished in {time.time()-t0:.0f}s (GPU wall incl. data load)")


if __name__ == "__main__":
    sys.exit(main())
