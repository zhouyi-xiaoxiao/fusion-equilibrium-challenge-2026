#!/usr/bin/env python3
"""
Stage 11 / ch1_temporal — JOINT local-window sequence model (dilated temporal context, i.i.d. frame sampling).

Motivation (measured in this stage): a sequence module stacked on a converged frame trunk gains only 1-4 % of the metric
loss, and joint training on whole-shot batches is handicapped by the correlated batches (16 shots/step). This model keeps
the i.i.d. FRAME sampling of the frame trunk but gives every sampled frame a dilated window of neighbouring EFIT frames of
the same shot:

    offsets O (frames), e.g. -8,-4,-2,-1,0,1,2,4,8      (clamped at the shot ends, with a validity flag)
    emb_j = trunk_body(x[i + o_j])                       shared per-frame trunk
    out_i = head(emb_0) + head2( SiLU( W [proj(emb_j), dt_j, valid_j]_j ) )      head2 ZERO-initialised
    dt_j  = (t[i + o_j] - t[i]) / 100 ms  (frame-gap / time encoding)

so at initialisation the model equals the frame model, and trunk + temporal mixing are trained jointly from scratch with
the same loss as seq_train.py. The frame-only CONTROL is the same code with O = (0,) (same batches, optimiser, epochs).

Env: everything of seq_train.py (trunk CFG, K, DX, FULL, VAL_FRAC, SEEDS, TTAG) plus
     WIN_OFFS ("-8,-4,-2,-1,0,1,2,4,8"), WTAG ("w9"), W_PW (128), W_HW (512), BATCH_W (2048), EP_W (30), STAGES ("ctrl,win").
Outputs like seq_train.py: names <TTAG>_w1 (control) and <TTAG>_<WTAG>.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from seq_train import CFG, SEEDS, TTAG, OUT, MODE, K, DX, FULL, HC, MET, E, dev, log, Metric, Trunk, load_split, merge, \
    to_gpu, pad, frames_of

OFFS = [int(o) for o in E("WIN_OFFS", "-8,-4,-2,-1,0,1,2,4,8").split(",")]
WTAG = E("WTAG", "w9")
WCFG = dict(W_PW=int(E("W_PW", "128")), W_HW=int(E("W_HW", "512")), BATCH_W=int(E("BATCH_W", "2048")), EP_W=int(E("EP_W", "30")),
            LR_W=float(E("LR_W", "1e-3")), W_DROP=float(E("W_DROP", "0")))
STAGES = E("STAGES", "ctrl,win").split(",")
# training-loss spike guard (label-free): after an epoch whose mean TRAIN loss exceeds SPIKE_GUARD x the best epoch so far,
# restore net + optimiser from the older of the last two good epochs and halve the learning-rate schedule from there on.
# A run without a spike is bitwise unaffected (the guard only copies state). 0 = off (all runs before 2026-10-11 23:20 UTC).
SPIKE_GUARD = float(E("SPIKE_GUARD", "1.5"))
WRITE_PRED = E("WRITE_PRED", "1") == "1"
RUN = E("RUN_NAME", f"{TTAG}_{WTAG}")


class WinModel(nn.Module):
    def __init__(self, d_in, n_out, cfg, offs, wc):
        super().__init__()
        self.trunk = Trunk(d_in, n_out, cfg)
        self.offs = list(offs)
        self.c = self.offs.index(0)
        self.ctx = len(self.offs) > 1
        if self.ctx:
            self.proj = nn.Linear(cfg["T_E"], wc["W_PW"])
            self.mix = nn.Linear(len(self.offs) * (wc["W_PW"] + 2), wc["W_HW"])
            self.drop = nn.Dropout(wc["W_DROP"])
            self.head2 = nn.Linear(wc["W_HW"], n_out)
            nn.init.zeros_(self.head2.weight)
            nn.init.zeros_(self.head2.bias)

    def forward(self, x, dt, valid):
        emb = self.trunk.body(x)                                  # (B, nO, E)
        out = self.trunk.head(emb[:, self.c])
        if not self.ctx:
            return out
        v = valid[..., None]
        z = torch.cat([F.silu(self.proj(emb)) * v, dt[..., None] * v, v], -1).flatten(1)
        return out + self.head2(self.drop(F.silu(self.mix(z))))


def main():
    t00 = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for sub in ("weights", "members", "preds", "logs"):
        (OUT / MODE / sub).mkdir(parents=True, exist_ok=True)
    m = Metric(K, MET)
    tr, ho = load_split("train"), load_split("holdout")
    te_np = load_split("test", need_y=False) if FULL else None
    model_np = merge(tr, ho) if FULL else tr
    del tr

    def prep(d_np):
        D = to_gpu(d_np)
        off = d_np["off"]
        n = len(d_np["t"])
        fs, fe = np.zeros(n, np.int64), np.zeros(n, np.int64)
        for s in range(len(off) - 1):
            fs[off[s]:off[s + 1]] = off[s]
            fe[off[s]:off[s + 1]] = off[s + 1]
        D["fs"], D["fe"] = torch.from_numpy(fs).to(dev), torch.from_numpy(fe).to(dev)
        D["t"] = torch.from_numpy(np.asarray(d_np["t"], np.float32)).to(dev)
        return D

    Dm, Dh = prep(model_np), prep(ho)
    te = prep(te_np) if FULL else None
    del model_np
    n_shots, n_aux = Dm["n_shots"], Dm["A"].shape[1]
    n_out = K + 2 + n_aux
    L = torch.tensor(m.L, dtype=torch.float32, device=dev)
    pc = np.load(HC / f"pca_{K}.npz")
    comps, pmean = pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64)
    off_h, Yh = ho["off"], ho["C"].astype(np.float64)
    JWh = np.load(MET / f"JW_holdout_{K}.npy") if (K == 80 and not FULL) else None
    log(f"WIN mode {MODE} K {K} DX {DX} shots {n_shots} | run {RUN} seeds {SEEDS} stages {STAGES} offs {OFFS} | load {time.time()-t00:.0f}s")
    log("CFG " + json.dumps(CFG) + " WCFG " + json.dumps(WCFG))
    names_of = {"ctrl": f"{TTAG}_w1", "win": f"{TTAG}_{WTAG}"}
    offs_of = {"ctrl": [0], "win": OFFS}
    rec, members = {}, {"ctrl": [], "win": []}

    def window(D, i, offs):
        o = torch.tensor(offs, device=dev)[None, :]
        j = i[:, None] + o
        valid = (j >= D["fs"][i][:, None]) & (j < D["fe"][i][:, None])
        j = torch.where(valid, j, i[:, None])
        dt = ((D["t"][j] - D["t"][i][:, None]) / 100.0).clamp(-5.0, 5.0)
        return D["X"][j], dt, valid.to(torch.float32)

    def write_preds(name, C, q, b):
        psi = C @ comps + pmean[None, :]
        d = {}
        for s in range(len(off_h) - 1):
            p, e = off_h[s], off_h[s + 1]
            d[f"shot_{s:04d}_psirz"] = psi[p:e].reshape(-1, 65, 65).astype(np.float32)
            d[f"shot_{s:04d}_q95"] = q[p:e].astype(np.float32)
            d[f"shot_{s:04d}_betaN"] = b[p:e].astype(np.float32)
        np.savez(OUT / MODE / "preds" / f"preds_d3d_{name}.npz", **d)

    for seed in SEEDS:
        rs = np.random.RandomState(1000 + seed)
        perm = rs.permutation(n_shots)
        n_val = 0 if FULL else int(round(CFG["VAL_FRAC"] * n_shots))
        va_sh = torch.from_numpy(np.sort(perm[:n_val])).to(dev)
        tr_sh = torch.from_numpy(np.sort(perm[n_val:])).to(dev)
        fr = frames_of(Dm, tr_sh)
        fr_va = frames_of(Dm, va_sh) if n_val else None
        nrm = {"muC": Dm["C"][fr].mean(0), "sdC": Dm["C"][fr].std(0), "muQ": Dm["QB"][fr].mean(0), "sdQ": Dm["QB"][fr].std(0)}
        A = Dm["A"][fr]
        fin = torch.isfinite(A)
        A0 = torch.where(fin, A, torch.zeros_like(A))
        nrm["muA"] = A0.sum(0) / fin.sum(0)
        nrm["sdA"] = (((A0 - nrm["muA"]) ** 2 * fin).sum(0) / fin.sum(0)).sqrt().clamp_min(1e-6)
        del A, A0, fin
        wq = (0.15 / 2) / nrm["sdQ"] ** 2

        def losses(out, D, fidx):
            e = (out[..., :K] - (D["C"][fidx] - nrm["muC"]) / nrm["sdC"]) * nrm["sdC"]
            lm = ((e @ L) ** 2).sum(-1)
            eq = (out[..., K:K + 2] - (D["QB"][fidx] - nrm["muQ"]) / nrm["sdQ"]) * nrm["sdQ"]
            lq = (eq ** 2 * wq).sum(-1)
            a = D["A"][fidx]
            ma = torch.isfinite(a)
            ua = torch.where(ma, (a - nrm["muA"]) / nrm["sdA"], torch.zeros_like(a))
            la = (((out[..., K + 2:] - ua) ** 2) * ma).mean(-1)
            return lm, lq, la

        @torch.no_grad()
        def predict(net, D, fidx):
            net.eval()
            return torch.cat([net(*window(D, fidx[i:i + 16384], net.offs)) for i in range(0, len(fidx), 16384)])

        def decode(out):
            C = (out[:, :K] * nrm["sdC"] + nrm["muC"]).double().cpu().numpy()
            QB = (out[:, K:K + 2] * nrm["sdQ"] + nrm["muQ"]).double().cpu().numpy()
            return C, QB[:, 0], QB[:, 1]

        for stage in ("ctrl", "win"):
            if stage not in STAGES:
                continue
            name = names_of[stage]
            t1 = time.time()
            torch.manual_seed(seed)
            net = WinModel(DX, n_out, CFG, offs_of[stage], WCFG).to(dev)
            N = len(fr)
            B = WCFG["BATCH_W"]
            spe = N // B
            opt = torch.optim.AdamW(net.parameters(), lr=WCFG["LR_W"], weight_decay=CFG["WD_T"])
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=WCFG["LR_W"], total_steps=WCFG["EP_W"] * spe, pct_start=0.05)
            g = torch.Generator(device=dev)
            g.manual_seed(seed)
            best, best_ep, best_sd, curve = float("inf"), -1, None, []
            lr_fac, snaps, best_tr, n_rb = 1.0, [], float("inf"), 0
            for ep in range(WCFG["EP_W"]):
                net.train()
                pm = fr[torch.randperm(N, device=dev, generator=g)]
                tot = 0.0
                for bi in range(spe):
                    idx = pm[bi * B:(bi + 1) * B]
                    lm, lq, la = losses(net(*window(Dm, idx, net.offs)), Dm, idx)
                    loss = (lm + CFG["W_QB"] * lq + CFG["W_AUX"] * la).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    if lr_fac != 1.0:
                        for gp in opt.param_groups:
                            gp["lr"] *= lr_fac
                    tot += float(loss.detach())
                trl = tot / spe
                if SPIKE_GUARD > 0 and snaps and trl > SPIKE_GUARD * best_tr:
                    net.load_state_dict(snaps[0][0])
                    opt.load_state_dict(snaps[0][1])
                    lr_fac *= 0.5
                    n_rb += 1
                    log(f"spike guard: epoch {ep + 1} train {trl:.5f} > {SPIKE_GUARD} x {best_tr:.5f} -> restored epoch "
                        f"{snaps[0][2]} state, lr factor {lr_fac}")
                    snaps = snaps[:1]
                elif SPIKE_GUARD > 0:
                    best_tr = min(best_tr, trl)
                    snaps = (snaps + [(copy.deepcopy(net.state_dict()), copy.deepcopy(opt.state_dict()), ep + 1)])[-2:]
                if n_val:
                    lm, lq, _ = losses(predict(net, Dm, fr_va), Dm, fr_va)
                    vm, vq = float(lm.mean()), float(lq.mean())
                    curve.append([ep + 1, tot / spe, vm, vq])
                    if vm + vq < best:
                        best, best_ep, best_sd = vm + vq, ep + 1, copy.deepcopy(net.state_dict())
                else:
                    curve.append([ep + 1, tot / spe])
            if best_sd is not None:
                net.load_state_dict(best_sd)
            info = {"epochs": WCFG["EP_W"], "best_ep": best_ep, "val_best": best if n_val else None,
                    "val_last": (curve[-1][2] + curve[-1][3]) if n_val else None,
                    "val_metric_best": min(c[2] for c in curve) if n_val else None, "train_last": curve[-1][1],
                    "sec": time.time() - t1, "rollbacks": n_rb, "spike_guard": SPIKE_GUARD, "curve": curve}
            torch.save({"net": net.state_dict(), "cfg": CFG, "wcfg": WCFG, "offs": offs_of[stage], "seed": seed,
                        "nrm": {k: v.cpu() for k, v in nrm.items()}, "info": info}, OUT / MODE / "weights" / f"{name}_s{seed}.pt")
            if FULL:
                C, q, b = decode(predict(net, te, torch.arange(te["X"].shape[0], device=dev)))
                tag = "test"
            else:
                fh = torch.arange(Dh["X"].shape[0], device=dev)
                out = predict(net, Dh, fh)
                lm, lq, _ = losses(out, Dh, fh)
                info.update(hold_metric=float(lm.mean()), hold_qb=float(lq.mean()))
                C, q, b = decode(out)
                if JWh is not None:
                    info["hold_PF"] = float(m.loss_PF(C, Yh, JWh).mean())
                tag = "hold"
            np.savez(OUT / MODE / "members" / f"{tag}_{name}_s{seed}.npz", coef=C.astype(np.float32), q95=q.astype(np.float32),
                     betaN=b.astype(np.float32), basis=f"stage6_H/pca_{K}.npz", offs=np.array(offs_of[stage]))
            members[stage].append((C, q, b))
            rec[f"{name}_s{seed}"] = info
            log(f"== {name} seed {seed}: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items() if k != 'curve'}))
        del fr

    names = []
    for stage, ms in members.items():
        if not ms:
            continue
        name = names_of[stage]
        Cb, qb, bb = (np.mean([x[i] for x in ms], 0) for i in range(3))
        tag = "test" if FULL else "hold"
        np.savez(OUT / MODE / "members" / f"{tag}_{name}_bag{len(ms)}.npz", coef=Cb.astype(np.float32), q95=qb.astype(np.float32),
                 betaN=bb.astype(np.float32), basis=f"stage6_H/pca_{K}.npz", seeds=np.array(SEEDS))
        if FULL:
            continue
        rec[f"{name}_bag{len(ms)}"] = {"hold_metric": float(m.loss_G(Cb, Yh).mean())}
        log(f"== {name} bag{len(ms)}: {rec[f'{name}_bag{len(ms)}']}")
        if WRITE_PRED:
            write_preds(f"{name}_s{SEEDS[0]}", *ms[0])
            names.append(f"{name}_s{SEEDS[0]}")
            if len(ms) > 1:
                write_preds(f"{name}_bag{len(ms)}", Cb, qb, bb)
                names.append(f"{name}_bag{len(ms)}")
    gsec = time.time() - t00
    json.dump({"run": RUN, "mode": MODE, "cfg": CFG, "wcfg": WCFG, "offs": OFFS, "seeds": SEEDS, "results": rec,
               "gpu_seconds_total": gsec}, open(OUT / MODE / "logs" / f"{RUN}.json", "w"), indent=1)
    if names:
        with open(OUT / MODE / "preds" / f"names_{RUN}.txt", "w") as f:
            f.write("\n".join(names) + "\n")
    log(f"done {RUN} in {gsec:.0f}s GPU wall (incl. load)")


if __name__ == "__main__":
    main()
