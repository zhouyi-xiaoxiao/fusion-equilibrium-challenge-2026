#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — the same warm-start fine-tune for the TEMPORAL window members (tR_w9d), modes r0 | lab.

Port of ch1d_liloss/li_ft_tw.py: warm start = the member's saved weights (window model, no EMA), same model rows (dev: the set's
train shots minus the member's own 5 % validation shots, split seed 1000 + seed; full: all 7,041 shots), same loss
(metric |e @ L_80|^2 + W_QB q95/betaN + W_AUX aux, window dropout ON) + lambda_eff * L_aux on a dropout-OFF pass over the first
quarter of each batch.  L_aux (mode lab) = the label-polygon normal-displacement smooth-max of jc_ft.py (LabBank.dn), equal-shot
weights.  Schedule / calibration / cap exactly as jc_ft.py (FT dict, --share --cap --lr --epochs).
Out: $J/<set>/tw/<mode><suffix>/hold_tR_w9d_s<seed>.npz (coef of the evaluation frames; full: test_tR_w9d_s<seed>.npz) + .pt + .json
  SET=fold3 python jc_ft_tw.py --modes lab --share 1 --cap 4 [--suffix _s100] [--seeds 0,1] [--shard i/n]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

import jc_common as C
from jc_common import log

SET = os.environ["SET"]
MODE = "full" if SET == "full" else "dev"
_info = C.set_info(SET)
os.environ.update(MET=str(_info["metric"]), FULL="1" if MODE == "full" else "0", TTAG="tR", T_ARCH="res", WTAG="w9d", W_DROP="0.1",
                  EP_W="40", STAGES="win")
import jc_ft as F  # noqa: E402  (sets FEAT / METRIC / OUT, imports torch and fm_train)
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch1d_fold2/vendor/ch1_temporal"))
import seq_train as S  # noqa: E402
import win_train as W  # noqa: E402

FT = F.FT
TWD = (C.XFT_BASE / "temporal" / MODE) if SET == "full" else (C.CX / SET / "temporal" / MODE)


def prep(d_np, dev):
    D = S.to_gpu(d_np)
    off = d_np["off"]
    n = len(d_np["t"])
    fs, fe = np.zeros(n, np.int64), np.zeros(n, np.int64)
    for s in range(len(off) - 1):
        fs[off[s]:off[s + 1]] = off[s]
        fe[off[s]:off[s + 1]] = off[s + 1]
    D["fs"], D["fe"] = torch.from_numpy(fs).to(dev), torch.from_numpy(fe).to(dev)
    D["t"] = torch.from_numpy(np.asarray(d_np["t"], np.float32)).to(dev)
    return D


def window(D, i, offs):
    o = torch.tensor(offs, device=i.device)[None, :]
    j = i[:, None] + o
    valid = (j >= D["fs"][i][:, None]) & (j < D["fe"][i][:, None])
    j = torch.where(valid, j, i[:, None])
    dt = ((D["t"][j] - D["t"][i][:, None]) / 100.0).clamp(-5.0, 5.0)
    return D["X"][j], dt, valid.to(torch.float32)


def run(seed, mode, Dm, Dtgt, BK, LB, args, dev):
    arm = f"{mode}{args.suffix}"
    odir = C.J / SET / "tw" / arm
    odir.mkdir(parents=True, exist_ok=True)
    name = f"tR_w9d_s{seed}"
    if (odir / f"{name}.json").exists():
        log(f"{arm}/{name}: exists, skip")
        return
    t0 = time.time()
    ck = torch.load(TWD / "weights" / f"{name}.pt", map_location="cpu", weights_only=False)
    cfg, wcfg, offs = ck["cfg"], ck["wcfg"], ck["offs"]
    K = cfg["K"]
    nrm = {k: v.to(dev) for k, v in ck["nrm"].items()}
    n_aux = Dm["A"].shape[1]
    n_out = K + 2 + n_aux
    net = W.WinModel(cfg["DX"], n_out, cfg, offs, wcfg).to(dev)
    net.load_state_dict(ck["net"])
    n_shots = Dm["n_shots"]
    if MODE == "full":
        tr_sh = torch.arange(n_shots, device=dev)
    else:
        perm = np.random.RandomState(1000 + seed).permutation(n_shots)
        n_val = int(round(cfg["VAL_FRAC"] * n_shots))
        tr_sh = torch.from_numpy(np.sort(perm[n_val:])).to(dev)
    fr = S.frames_of(Dm, tr_sh)
    N = len(fr)
    L = torch.tensor(np.load(S.MET / f"L_{K}.npy"), dtype=torch.float32, device=dev)
    Linv_T = torch.linalg.inv(L.double()).T.contiguous()
    wq = (0.15 / 2) / nrm["sdQ"] ** 2
    sdC, muC = nrm["sdC"], nrm["muC"]
    comps = BK.comps[K]

    def znorm(g):
        return float((((g[:, :K].double() / sdC.double()) @ Linv_T) ** 2).sum().sqrt())

    def losses(out, fidx):
        e = (out[..., :K] - (Dm["C"][fidx] - muC) / sdC) * sdC
        lm = ((e @ L) ** 2).sum(-1)
        eq = (out[..., K:K + 2] - (Dm["QB"][fidx] - nrm["muQ"]) / nrm["sdQ"]) * nrm["sdQ"]
        lq = (eq ** 2 * wq).sum(-1)
        a = Dm["A"][fidx]
        ma = torch.isfinite(a)
        ua = torch.where(ma, (a - nrm["muA"]) / nrm["sdA"], torch.zeros_like(a))
        la = (((out[..., K + 2:] - ua) ** 2) * ma).mean(-1)
        return lm, lq, la

    def lab_loss(out_eval, fidx, tau):
        v = LB.ok[fidx]
        st = {"n": int(v.sum()), "D_hard": 0.0}
        if st["n"] == 0:
            return None, st
        bv = fidx[v]
        phi_hat = -((out_eval[v, :K] * sdC + muC).double() @ comps + LB.mean[K])
        dn, wv = LB.dn(phi_hat, bv, FT["gfloor"])
        f = torch.sqrt(dn * dn + FT["eps_d"] ** 2) - FT["eps_d"]
        soft = tau * torch.logsumexp(f / tau + torch.log(wv / wv.sum(1, keepdim=True)), 1)
        rg = BK.rg[bv]
        st["D_hard"] = float(((dn.abs() * (wv > 0)).max(1).values / rg).mean().detach())
        return (BK.w[bv] * FT["w_lcfs"] * soft / rg).mean(), st

    B = wcfg["BATCH_W"]
    Q = int(round(FT["aux_frac"] * B))
    E = FT["epochs"]
    spe = N // B
    total = E * spe
    lam = g0 = ga = 0.0
    if mode != "r0":
        gc = torch.Generator(device=dev); gc.manual_seed(FT["calib_seed"])
        g0s, gas = [], []
        for _ in range(FT["calib_batches"]):
            idx = fr[torch.randperm(N, device=dev, generator=gc)[:B]]
            net.train()
            out = net(*window(Dm, idx, offs))
            lm, _, _ = losses(out, idx)
            g0s.append(znorm(torch.autograd.grad(lm.mean(), out)[0]) ** 2)
            net.eval()
            oq = net(*window(Dm, idx[:Q], offs))
            ll, _ = lab_loss(oq, idx[:Q], FT["tau0"])
            gas.append(znorm(torch.autograd.grad(ll, oq)[0]) ** 2)
        g0, ga = math.sqrt(np.mean(g0s)), math.sqrt(np.mean(gas))
        # heavy-tailed auxiliary gradients (low-current frames): the RMS over 16 batches is dominated by one batch, so the
        # multiplier is set from the MEDIAN per-batch ratio
        lam = args.share * float(np.median(np.sqrt(np.array(g0s) / np.maximum(np.array(gas), 1e-300))))
    log(f"{arm}/{name}: N {N} B {B} Q {Q} | calib g0 {g0:.4e} g_aux {ga:.4e} -> lambda {lam:.4e} (share {args.share} cap {args.cap} lr {FT['lr']})")

    gd = torch.Generator(device=dev); gd.manual_seed(4242)
    diag_rows = fr[torch.randperm(N, device=dev, generator=gd)[:16384]]

    @torch.no_grad()
    def diag(model):
        model.eval()
        tot = cnt = 0.0
        for i in range(0, len(diag_rows), 2048):
            bq = diag_rows[i:i + 2048]
            _, st = lab_loss(model(*window(Dm, bq, offs)), bq, FT["tau1"])
            tot += st["D_hard"] * st["n"]; cnt += st["n"]
        return tot / max(cnt, 1)

    d_start = diag(net)
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(FT["ema"]))
    opt = torch.optim.AdamW(net.parameters(), lr=FT["lr"], weight_decay=cfg["WD_T"])

    def lr_at(t):
        if t < FT["warm_steps"]:
            return FT["lr"] * (t + 1) / FT["warm_steps"]
        return FT["lr_min"] + 0.5 * (FT["lr"] - FT["lr_min"]) * (1 + math.cos(math.pi * (t - FT["warm_steps"]) / max(1, total - FT["warm_steps"])))

    g = torch.Generator(device=dev); g.manual_seed(seed + 2026)
    torch.manual_seed(seed + 3000)
    ramp = FT["ramp_epochs"] * spe
    step, curve = 0, []
    t1 = time.time()
    for ep in range(E):
        pm = fr[torch.randperm(N, device=dev, generator=g)]
        acc = dict(main=0.0, D=0.0, n=0, share=0.0, capped=0)
        for bi in range(spe):
            idx = pm[bi * B:(bi + 1) * B]
            for gp in opt.param_groups:
                gp["lr"] = lr_at(step)
            net.train()
            out = net(*window(Dm, idx, offs))
            lm, lq, la = losses(out, idx)
            loss = (lm + cfg["W_QB"] * lq + cfg["W_AUX"] * la).mean()
            if mode != "r0":
                tau = FT["tau0"] * (FT["tau1"] / FT["tau0"]) ** (step / max(1, total - 1))
                lam_t = lam * min(1.0, (step + 1) / ramp)
                g0b = znorm(torch.autograd.grad(lm.mean(), out, retain_graph=True)[0])
                net.eval()
                oq = net(*window(Dm, idx[:Q], offs))
                ll, st = lab_loss(oq, idx[:Q], tau)
                net.train()
                if ll is not None:
                    gaq = torch.autograd.grad(ll, oq)[0]
                    gab = znorm(gaq)
                    if math.isfinite(gab) and gab > 0:
                        lam_eff = min(lam_t, args.cap * g0b / gab)
                        acc["capped"] += int(lam_eff < lam_t); acc["share"] += lam_eff * gab / g0b
                        loss = loss + (oq * (lam_eff * gaq).detach().to(oq.dtype)).sum()
                    acc["D"] += st["D_hard"] * st["n"]; acc["n"] += st["n"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); ema.update_parameters(net); step += 1
            acc["main"] += float(lm.mean().detach())
        rec = {"ep": ep + 1, "train_metric": acc["main"] / spe, "aux_D_lab": acc["D"] / max(acc["n"], 1), "share": acc["share"] / spe,
               "capped": acc["capped"] / spe}
        curve.append(rec)
        log(f"{arm}/{name} ep {ep+1}/{E} " + " ".join(f"{k} {v:.6g}" for k, v in rec.items() if k != "ep") + f" | {(time.time()-t1)/(ep+1):.1f}s/ep")

    @torch.no_grad()
    def predict(model, D):
        model.eval()
        fidx = torch.arange(D["X"].shape[0], device=dev)
        return torch.cat([model(*window(D, fidx[i:i + 16384], offs)) for i in range(0, len(fidx), 16384)])

    d_end = diag(ema.module)
    out = predict(ema, Dtgt)
    Cf = (out[:, :K] * sdC + muC).double().cpu().numpy().astype(np.float32)
    tag = "hold" if MODE == "dev" else "test"
    np.savez(odir / f"{tag}_{name}.npz", coef=Cf)
    info = {"arm": arm, "name": name, "set": SET, "mode": mode, "share": args.share, "cap": args.cap, "lam": lam, "g0": g0, "g_aux": ga,
            "N": N, "ft": dict(FT), "curve": curve, "train_D_lab": [d_start, d_end], "sec_train": time.time() - t1, "sec_total": time.time() - t0}
    torch.save({"net": ema.module.state_dict(), "cfg": cfg, "wcfg": wcfg, "offs": offs, "seed": seed,
                "nrm": {k: v.cpu() for k, v in nrm.items()}, "info": info}, odir / f"{name}.pt")
    json.dump(info, open(odir / f"{name}.json", "w"), indent=1)
    log(f"{arm}/{name} DONE {info['sec_train']:.0f}s | train label-poly D {d_start:.6f} -> {d_end:.6f}")
    del net, ema, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="lab")
    ap.add_argument("--seeds", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--share", type=float, default=FT["share"])
    ap.add_argument("--cap", type=float, default=FT["cap"])
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    if a.lr > 0:
        FT["lr"], FT["lr_min"] = a.lr, a.lr / 10.0
    if a.epochs > 0:
        FT["epochs"] = a.epochs
    seeds = [int(s) for s in re.split(r"[,+]", a.seeds) if s] or ([0, 1] if SET in ("fold3", "fold2") else [0, 1, 2, 3])
    jobs = [(s, md) for md in re.split(r"[,+]", a.modes) if md for s in seeds]
    si, sn = [int(x) for x in a.shard.split("/")]
    jobs = jobs[si::sn]
    dev = S.dev
    tr, ho = S.load_split("train"), S.load_split("holdout")
    if MODE == "full":
        Dm = prep(S.merge(tr, ho), dev)
        Dtgt = prep(S.load_split("test", need_y=False), dev)
    else:
        Dm, Dtgt = prep(tr, dev), prep(ho, dev)
    del tr, ho
    N = Dm["X"].shape[0]
    BK = F.Bank(dev, N)
    LB = F.LabBank(dev, N)
    log(f"TW ft {SET} shard {si}/{sn}: {jobs}")
    for seed, md in jobs:
        try:
            run(seed, md, Dm, Dtgt, BK, LB, a, dev)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"seed {seed} {md} FAILED: {ex}")
            torch.cuda.empty_cache()
    log("shard finished")


if __name__ == "__main__":
    sys.exit(main())
