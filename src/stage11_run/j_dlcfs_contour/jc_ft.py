#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — step 3 (GPU): warm-start fine-tune of the FM bag members with the scorer-aligned contour-replay
loss (mode ctr), the 50 mm Gaussian boundary-band pixel loss (mode band, cheap control) or nothing (mode r0, matched control).

    loss = metric |e @ L_K|^2 + aux_w * aux            (the member's own recipe: same rows, batch, dropout ON)
         + lambda_eff * L_aux                           (second, dropout-OFF pass over the first quarter of every batch)
  ctr : L_aux = mean over valid anchors of  w_shot * (0.10 / R_s) * smoothmax_tau( directed NN distances, both ways, 1024 values )
        of the REPLAYED polygon of the virtual bag map  phi = phi_bag0 + decode(c_member - c_member0)  against the official 512-point
        reference contour (jc_torch.Replay; frozen grid edges and critical node from jc_bank.py).  eps_d 0.25 mm, tau 2 -> 0.5 mm
        (geometric in the step), equal-shot weights w_shot = (N_ok / H) / n_s.  An anchor is skipped in a step when a replayed vertex
        leaves its grid edge by more than OVER_TOL (2 mm) or an edge denominator falls below 1e-6 of the axis-to-boundary span.
  band: L_aux = mean over frames of  sum_x exp(-d(x)^2 / (2 * (50 mm)^2)) * (psi_hat - P_K psi_true)(x)^2 ,  d = distance of grid node
        x to the reference contour (256 of its 512 points); no shot weights (pixel loss).
  lambda = share * g0 / g_aux from 16 fixed batches (Frobenius norms of the output gradients in z = c L_K), share = 0.05, ramped
  linearly over the first 2 epochs; per batch lambda_eff = min(lambda_t, cap * g0_batch / g_aux_batch), cap = 0.20.
  Optimiser: fresh AdamW (member's wd, eps 1e-10), 8 epochs, 100-step linear warm-up then cosine 5e-5 -> 5e-6, grad-clip as the
  member, EMA 0.999 initialised at the warm start.  Outputs = EMA weights.
Out: $J/<set>/members/<member>__<mode>/{coef_hold,coef_test}.npy, model.pt (fm_predict-compatible), meta.json
  SET=fold3 python jc_ft.py --modes ctr [--members a+b] [--shard i/n] [--share 0.05]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np

import jc_common as C
from jc_common import log

SET = os.environ["SET"]
info = C.set_info(SET)
os.environ["FEAT"] = str(info["feat"])
os.environ["METRIC"] = str(info["metric"])
os.environ["OUT"] = str(C.J / SET / "fm_out")
import torch  # noqa: E402

import jc_torch as JT  # noqa: E402
import fm_train as T  # noqa: E402

FT = dict(epochs=8, lr=5e-5, lr_min=5e-6, warm_steps=100, aux_frac=0.25, ramp_epochs=2, calib_batches=16, ema=0.999,
          calib_seed=777, share=0.05, cap=0.20, tau0=2e-3, tau1=0.5e-3, eps_d=0.25e-3, over_tol=2e-3, den_rel=1e-6,
          band_sigma=0.05, w_lcfs=0.10, gfloor=0.1)


class Bank:
    def __init__(self, dev, N):
        t0 = time.time()
        b = C.J / SET / "bank"
        ld = lambda n: np.load(b / f"{n}.npy")  # noqa: E731
        ok = ld("ok").copy()
        assert len(ok) == N, (len(ok), N)
        gate = np.load(b / "gate_rows.npy")
        ok[gate] = False                                          # calibration-gate anchors never enter the loss
        sh = np.load(b / "shot.npz")
        off = sh["off"].astype(np.int64)
        nshot = len(off) - 1
        sid = np.repeat(np.arange(nshot), np.diff(off))
        n_s = np.bincount(sid[ok], minlength=nshot)
        H = int((n_s > 0).sum())
        w = (ok.sum() / H) / np.maximum(n_s, 1)[sid]
        rg = sh["rgeo_s"][sid]
        t = lambda a, dt=None: torch.as_tensor(a, device=dev) if dt is None else torch.as_tensor(a, dtype=dt, device=dev)  # noqa: E731
        self.N, self.H = N, H
        self.ok = t(ok)
        self.ea, self.eb = t(ld("ea")), t(ld("eb"))
        self.pa, self.pb = t(ld("pa")), t(ld("pb"))
        lvl0, phic0 = ld("lvl0"), ld("phic0")
        self.crit = t(ld("crit").astype(np.int64))
        self.phic0 = t(phic0.astype(np.float64))
        self.dl0 = t(lvl0 - phic0.astype(np.float64))
        self.span0 = t(ld("span0").astype(np.float64))
        self.w = t(w, torch.float64)
        self.rg = t(np.nan_to_num(rg, nan=1.7), torch.float64)
        self.hd0 = t(np.nan_to_num(ld("hd0"), nan=0.0), torch.float64)
        _, li_idx, _ = C.bank_rows(SET)
        ntr = np.load(C.LI / "cache/train/ok.npy").shape[0]
        Cc = np.load(C.LI / "cache/train/C.npy", mmap_mode="r")
        if SET in ("h150", "full"):
            Cr = np.asarray(Cc)
            if SET == "full":
                Cr = np.concatenate([Cr, np.load(C.LI / "cache/holdout/C.npy")])
        else:
            Cr = np.asarray(Cc)[li_idx]
        assert Cr.shape[0] == N
        self.Cref = torch.from_numpy(np.ascontiguousarray(Cr)).to(dev)
        refok = np.concatenate([np.load(C.LI / "cache/train/ok.npy"), np.load(C.LI / "cache/holdout/ok.npy")])[li_idx]
        self.refok = t(refok)
        del Cr
        R, Z, _ = C.grid()
        RR, ZZ = np.meshgrid(R, Z)
        self.nodes = t(np.stack([RR.ravel(), ZZ.ravel()], 1), torch.float32)
        self.comps = {K: t(C.pca(K)[1], torch.float64) for K in (80, 120)}
        self.rp = JT.Replay(R, Z, dev)
        log(f"bank {SET}: N {N} ok {ok.mean():.4f} (gate rows excluded {len(gate)}) shots {H} "
            f"| in-sample bag D (frame mean hd0/R) {float((self.hd0[self.ok] / self.rg[self.ok]).mean()):.5f} "
            f"| {time.time()-t0:.0f}s gpu {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def band_w(self, rows, sigma):
        """Gaussian boundary-band weights (q,4225) float64 for bank rows (reference contour, every 2nd point)."""
        out = []
        with torch.no_grad():
            for i in range(0, len(rows), 256):
                Cq = self.Cref[rows[i:i + 256]][:, ::2, :]                              # (q,256,2) float32
                d2 = ((self.nodes[None, :, None, :] - Cq[:, None, :, :]) ** 2).sum(-1).min(2).values
                out.append(torch.exp(-d2.double() / (2.0 * sigma * sigma)))
        return torch.cat(out)


class LabBank:
    """Label-topology bank (jc_bank_label.py) for the rows of this set: raw reference polygon edges, label phi at the edge nodes,
    critical node, per-vertex normal factor and arc-length weight."""

    def __init__(self, dev, N):
        t0 = time.time()
        _, li_idx, _ = C.bank_rows(SET)
        b = C.J / "banklab"
        ntr = np.load(b / "train/ok.npy").shape[0]
        need_h = bool((li_idx >= ntr).any())

        def ld(n):
            x = np.load(b / "train" / f"{n}.npy", mmap_mode="r")
            if need_h:
                return np.concatenate([np.asarray(x), np.load(b / "holdout" / f"{n}.npy")])[li_idx]
            return np.asarray(x)[li_idx] if len(li_idx) != ntr else np.asarray(x)
        t = lambda a_: torch.as_tensor(np.ascontiguousarray(a_), device=dev)  # noqa: E731
        ok = ld("ok").copy()
        assert len(ok) == N
        ok[np.load(C.J / SET / "bank/gate_rows.npy")] = False
        self.ok = t(ok)
        self.ea, self.eb = t(ld("ea")), t(ld("eb"))
        self.pa, self.pb = t(ld("pa")), t(ld("pb"))
        self.cosl, self.wseg = t(ld("cosl")), t(ld("wseg"))
        self.crit = t(ld("crit").astype(np.int64))
        self.phic0 = t(ld("phic0").astype(np.float64))
        self.lvl0 = t(ld("lvl0"))
        self.mean = {K: t(C.pca(K)[0]) for K in (80, 120)}
        log(f"label bank {SET}: N {N} ok {ok.mean():.4f} | {time.time()-t0:.0f}s gpu {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def dn(self, phi_hat, rows, gfloor):
        """phi_hat (q,4225) float64 predicted phi; rows (q,) bank rows (all ok). -> signed first-order normal displacement (q,V) [m],
        arc-length weights (q,V)."""
        ea, eb = self.ea[rows].long(), self.eb[rows].long()
        pa, pb = self.pa[rows].double(), self.pb[rows].double()
        e_a = phi_hat.gather(1, ea) - pa
        e_b = phi_hat.gather(1, eb) - pb
        e_l = phi_hat.gather(1, self.crit[rows][:, None])[:, 0] - self.phic0[rows]
        den = pb - pa
        u = (self.lvl0[rows][:, None] - pa) / den
        wv = self.wseg[rows].double()
        g = den.abs() / self.cosl[rows].double().clamp_min(1e-6)
        gm = torch.nanmedian(torch.where(wv > 0, g, torch.full_like(g, float("nan"))), dim=1).values
        q = 1.0 / torch.maximum(g, gfloor * gm[:, None])
        return q * (e_l[:, None] - (1.0 - u) * e_a - u * e_b), wv


def r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def run(member, mode, D, BK, LB, args):
    dev = D.dev
    name = f"{member}__{mode}{args.suffix}"
    odir = C.J / SET / "members" / name
    if (odir / "meta.json").exists():
        log(f"{name}: exists, skip")
        return
    odir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    M = JT.load_member(T, D, info["members"] / member / "model.pt")
    net, cfg, K, X, Xt, mu, sd, nm = M["net"], M["cfg"], M["K"], M["X"], M["Xt"], M["mu"], M["sd"], M["nm"]
    ts = info["trainset"]
    assert cfg["trainset"] == ts, (cfg["trainset"], ts)
    rows = torch.arange(D.ntr + (D.nh if ts == "full" else 0), device=dev)
    N = len(rows)
    assert N == BK.N
    hold = slice(D.ntr, D.ntr + D.nh)
    Y = D.Y(K)
    U = (Y - mu) / sd
    QB = (D.qb - nm["qmu"]) / nm["qsd"]
    AMASK = torch.isfinite(D.aux).float()
    AUX = torch.nan_to_num((D.aux - nm["amu"]) / nm["asd"]).clamp(-10, 10)
    L = torch.tensor(np.load(T.MET / f"L_{K}.npy"), dtype=torch.float32, device=dev)
    Linv_T = torch.linalg.inv(L.double()).T.contiguous()
    comps = BK.comps[K]
    rp = BK.rp
    cm0 = (T.predict(net, X[rows])[0] * sd + mu)                     # member's own warm-start outputs (eval mode), (N,K) float32
    B = cfg["batch"]
    Q = int(round(FT["aux_frac"] * B))
    E = FT["epochs"]
    spe = N // B
    total = E * spe
    share = args.share

    def main_loss(o, b):
        e = (o - U[b]) * sd
        return ((e @ L) ** 2).sum(1).mean()

    def znorm(g):
        return float((((g.double() / sd.double()) @ Linv_T) ** 2).sum().sqrt())

    def ctr_loss(o_eval, bq, tau):
        """-> (loss, stats). stats: n valid anchors, n replayable, mean hard D (hd/R), mean soft."""
        v = BK.ok[bq]
        st = {"n_anchor": int(v.sum()), "n_valid": 0, "D_hard": 0.0, "D_soft": 0.0}
        if st["n_anchor"] == 0:
            return None, st
        bv = bq[v]
        dc = ((o_eval[v] * sd + mu) - cm0[bv]).double()
        dphi = -(dc @ comps)                                          # phi = -psi
        ea, eb = BK.ea[bv].long(), BK.eb[bv].long()
        pa = BK.pa[bv].double() + dphi.gather(1, ea)
        pb = BK.pb[bv].double() + dphi.gather(1, eb)
        lvl = BK.phic0[bv] + dphi.gather(1, BK.crit[bv][:, None])[:, 0] + BK.dl0[bv]
        with torch.no_grad():
            _, over, dmin = rp.edge_state(ea, eb, pa, pb, lvl)
            good = (over <= FT["over_tol"]) & (dmin > FT["den_rel"] * BK.span0[bv])
        ng = int(good.sum())
        st["n_valid"] = ng
        if ng == 0:
            return None, st
        g = torch.nonzero(good)[:, 0]
        bg = bv[g]
        soft, hard, _ = rp.forward(ea[g], eb[g], pa[g], pb[g], lvl[g], BK.Cref[bg].double(), tau, FT["eps_d"])
        rg = BK.rg[bg]
        loss = (BK.w[bg] * FT["w_lcfs"] * soft / rg).sum() / ng
        st["D_hard"] = float((hard / rg).mean().detach())
        st["D_soft"] = float((soft / rg).mean().detach())
        return loss, st

    def band_loss(o_eval, bq):
        v = BK.refok[bq]
        st = {"n_anchor": int(v.sum()), "n_valid": int(v.sum()), "D_hard": 0.0, "D_soft": 0.0}
        if st["n_anchor"] == 0:
            return None, st
        bv = bq[v]
        e = ((o_eval[v] * sd + mu) - Y[bv]).double() @ comps
        w = BK.band_w(bv, FT["band_sigma"])
        loss = (w * e * e).sum(1).mean()
        st["D_soft"] = float(loss.detach())
        return loss, st

    def lab_loss(o_eval, bq, tau):
        """First-order normal displacement of the predicted LCFS on the label polygon (member's own decoded map vs the label)."""
        v = LB.ok[bq]
        st = {"n_anchor": int(v.sum()), "n_valid": int(v.sum()), "D_hard": 0.0, "D_soft": 0.0}
        if st["n_anchor"] == 0:
            return None, st
        bv = bq[v]
        phi_hat = -((o_eval[v] * sd + mu).double() @ comps + LB.mean[K])
        dn, wv = LB.dn(phi_hat, bv, FT["gfloor"])
        f = torch.sqrt(dn * dn + FT["eps_d"] ** 2) - FT["eps_d"]
        logw = torch.log(wv / wv.sum(1, keepdim=True))
        soft = tau * torch.logsumexp(f / tau + logw, 1)
        hard = (dn.abs() * (wv > 0)).max(1).values
        rg = BK.rg[bv]
        loss = (BK.w[bv] * FT["w_lcfs"] * soft / rg).mean()
        st["D_hard"] = float((hard / rg).mean().detach())
        st["D_soft"] = float((soft / rg).mean().detach())
        return loss, st

    def aux_loss(o_eval, bq, tau):
        return {"ctr": ctr_loss, "lab": lab_loss}[mode](o_eval, bq, tau) if mode in ("ctr", "lab") else band_loss(o_eval, bq)

    # ---------------- lambda calibration (16 fixed batches at the warm-start weights, tau = tau0)
    lam, g0, ga = 0.0, 0.0, 0.0
    if mode != "r0":
        gc = torch.Generator(device=dev); gc.manual_seed(FT["calib_seed"])
        g0s, gas = [], []
        for _ in range(FT["calib_batches"]):
            b = rows[torch.randperm(N, device=dev, generator=gc)[:B]]
            net.train()
            o, _, _ = net(X[b])
            g0s.append(znorm(torch.autograd.grad(main_loss(o, b), o)[0]) ** 2)
            net.eval()
            oq, _, _ = net(X[b[:Q]])
            la, _ = aux_loss(oq, b[:Q], FT["tau0"])
            gas.append(znorm(torch.autograd.grad(la, oq)[0]) ** 2 if la is not None else 0.0)
        g0, ga = math.sqrt(np.mean(g0s)), math.sqrt(np.mean(gas))
        lam = share * g0 / ga
        if args.calib == "median":      # robust to one heavy-tailed batch (low-current frames): median per-batch ratio
            lam = share * float(np.median(np.sqrt(np.array(g0s) / np.maximum(np.array(gas), 1e-300))))
    log(f"{name}: K {K} arch {cfg['arch']} N {N} B {B} Q {Q} | calib g0 {g0:.4e} g_aux {ga:.4e} -> lambda {lam:.4e} (share {share} cap {args.cap} lr {FT['lr']})")

    # ---------------- fixed diagnostic anchors: surrogate D of the virtual bag map before / after
    gd = torch.Generator(device=dev); gd.manual_seed(4242)
    diag_rows = rows[torch.randperm(N, device=dev, generator=gd)[:16384]]

    @torch.no_grad()
    def diag(model):
        model.eval()
        tot = cnt = nan = 0.0
        e2 = 0.0
        for i in range(0, len(diag_rows), 2048):
            bq = diag_rows[i:i + 2048]
            oq, _, _ = model(X[bq])
            _, st = ctr_loss(oq, bq, FT["tau1"])
            tot += st["D_hard"] * st["n_valid"]; cnt += st["n_valid"]; nan += st["n_anchor"]
            e2 += float((((oq - U[bq]) * sd) @ L).pow(2).sum())
        out = {"train_D_replay": tot / max(cnt, 1), "train_replayable": cnt / max(nan, 1), "train_G": e2 / len(diag_rows)}
        if LB is not None:
            tl = nl = 0.0
            for i in range(0, len(diag_rows), 2048):
                bq = diag_rows[i:i + 2048]
                oq, _, _ = model(X[bq])
                _, st = lab_loss(oq, bq, FT["tau1"])
                tl += st["D_hard"] * st["n_valid"]; nl += st["n_valid"]
            out["train_D_lab"] = tl / max(nl, 1)
        return out

    d_start = diag(net)

    # ---------------- fine-tune
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(FT["ema"]))
    decay = [p for _, p in net.named_parameters() if p.ndim >= 2]
    nodecay = [p for _, p in net.named_parameters() if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["wd"]}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=FT["lr"], eps=1e-10)

    def lr_at(t):
        if t < FT["warm_steps"]:
            return FT["lr"] * (t + 1) / FT["warm_steps"]
        return FT["lr_min"] + 0.5 * (FT["lr"] - FT["lr_min"]) * (1 + math.cos(math.pi * (t - FT["warm_steps"]) / max(1, total - FT["warm_steps"])))

    g = torch.Generator(device=dev); g.manual_seed(cfg["seed"] + 2026)
    torch.manual_seed(cfg["seed"] + 3000)        # same dropout stream for every mode (the calibration above consumed RNG)
    curve, step = [], 0
    ramp = FT["ramp_epochs"] * spe
    t_train = time.time()
    for ep in range(E):
        acc = dict(main=0.0, D_hard=0.0, D_soft=0.0, n_anchor=0, n_valid=0, share=0.0, capped=0, skipped=0)
        perm = rows[torch.randperm(N, device=dev, generator=g)]
        for i in range(spe):
            b = perm[i * B:(i + 1) * B]
            for gq in opt.param_groups:
                gq["lr"] = lr_at(step)
            xb = X[b]
            net.train()
            o, a, q = net(xb)
            lm = main_loss(o, b)
            loss = lm
            if a is not None:
                mb = AMASK[b]
                loss = loss + cfg["aux_w"] * (((a - AUX[b]) ** 2) * mb).sum() / mb.sum()
            if q is not None:
                loss = loss + cfg["qb_w"] * 0.15 * ((q - QB[b]) ** 2).mean()
            if mode != "r0":
                tau = FT["tau0"] * (FT["tau1"] / FT["tau0"]) ** (step / max(1, total - 1))
                lam_t = lam * min(1.0, (step + 1) / ramp)
                g0b = znorm(torch.autograd.grad(lm, o, retain_graph=True)[0])
                net.eval()
                oq, _, _ = net(xb[:Q])
                la, st = aux_loss(oq, b[:Q], tau)
                net.train()
                if la is not None:
                    gaq = torch.autograd.grad(la, oq)[0]
                    gab = znorm(gaq)
                    if math.isfinite(gab) and gab > 0:
                        lam_eff = min(lam_t, args.cap * g0b / gab)
                        acc["capped"] += int(lam_eff < lam_t)
                        acc["share"] += lam_eff * gab / g0b
                        loss = loss + (oq * (lam_eff * gaq).detach().to(oq.dtype)).sum()   # injects lam_eff * dL_aux/d(oq)
                    else:
                        acc["skipped"] += 1
                else:
                    acc["skipped"] += 1
                for k in ("n_anchor", "n_valid"):
                    acc[k] += st[k]
                acc["D_hard"] += st["D_hard"] * st["n_valid"]; acc["D_soft"] += st["D_soft"] * st["n_valid"]
            opt.zero_grad(set_to_none=True)
            (T.LOSS_SCALE * loss).backward()
            if cfg["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["clip"])
            opt.step(); ema.update_parameters(net); step += 1
            acc["main"] += float(lm.detach())
        nv = max(acc["n_valid"], 1)
        rec = {"ep": ep + 1, "train_main": acc["main"] / spe, "aux_D_hard": acc["D_hard"] / nv, "aux_soft": acc["D_soft"] / nv,
               "replayable": acc["n_valid"] / max(acc["n_anchor"], 1), "share": acc["share"] / spe, "capped": acc["capped"] / spe,
               "skipped": acc["skipped"], "lr_end": lr_at(step - 1)}
        curve.append(rec)
        log(f"{name} ep {ep+1}/{E} " + " ".join(f"{k} {v:.6g}" for k, v in rec.items() if k != "ep")
            + f" | {(time.time()-t_train)/(ep+1):.1f}s/ep")
    sec_train = time.time() - t_train

    # ---------------- outputs
    d_end = diag(ema.module)
    oh = T.predict(ema, X[hold])[0]
    ch = oh * sd + mu
    eh = ch - Y[hold]
    d_end["hold_G"] = float(((eh @ L) ** 2).sum(1).mean())
    net0 = JT.load_member(T, D, info["members"] / member / "model.pt")["net"]
    e0 = (T.predict(net0, X[hold])[0] * sd + mu) - Y[hold]
    d_start["hold_G"] = float(((e0 @ L) ** 2).sum(1).mean())
    del net0
    np.save(odir / "coef_hold.npy", ch.cpu().numpy().astype(np.float32))
    ot = T.predict(ema, Xt)[0]
    np.save(odir / "coef_test.npy", (ot * sd + mu).cpu().numpy().astype(np.float32))
    cfg_out = dict(cfg, name=name, ft=dict(FT, src=member, mode=mode, share=share, cap=args.cap, lam=lam, g0=g0, g_aux=ga, set=SET))
    torch.save({"state": ema.module.state_dict(), "cfg": cfg_out, "cols": M["names"], "norm": M["ck"]["norm"]}, odir / "model.pt")
    meta = {"cfg": cfg_out, "n_train_frames": N, "steps": total, "sec_train": sec_train, "curve": curve, "start": d_start,
            "end": d_end, "sec_total": time.time() - t0, "gpu": torch.cuda.get_device_name(0)}
    json.dump(meta, open(odir / "meta.json", "w"), indent=1)
    log(f"{name} DONE {sec_train:.0f}s | start {json.dumps({k: round(v, 6) for k, v in d_start.items()})} -> "
        f"end {json.dumps({k: round(v, 6) for k, v in d_end.items()})}")
    del net, ema, opt, cm0
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="ctr")
    ap.add_argument("--members", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--share", type=float, default=FT["share"])
    ap.add_argument("--suffix", default="")
    ap.add_argument("--cap", type=float, default=FT["cap"])
    ap.add_argument("--epochs", type=int, default=0, help="override the number of fine-tune epochs")
    ap.add_argument("--calib", default="rms", choices=["rms", "median"])
    ap.add_argument("--labdiag", action="store_true", help="load the label bank for the diagnostics of non-lab modes too")
    ap.add_argument("--lr", type=float, default=0.0, help="override the peak lr (lr_min = lr / 10)")
    args = ap.parse_args()
    if args.lr > 0:
        FT["lr"], FT["lr_min"] = args.lr, args.lr / 10.0
    if args.epochs > 0:
        FT["epochs"] = args.epochs
    members = [m for m in re.split(r"[,+]", args.members) if m] or C.member_names(info["prefix"])
    jobs = [(m, md) for md in re.split(r"[,+]", args.modes) if md for m in members]
    si, sn = [int(x) for x in args.shard.split("/")]
    jobs = jobs[si::sn]
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    log(f"{SET} shard {si}/{sn}: {len(jobs)} runs: {jobs}")
    D = T.Data(dev)
    BK = Bank(dev, D.ntr + (D.nh if info["trainset"] == "full" else 0))
    LB = LabBank(dev, BK.N) if (any(md == "lab" for _, md in jobs) or args.labdiag) else None
    for m, md in jobs:
        try:
            run(m, md, D, BK, LB, args)
        except Exception as ex:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            log(f"{m} {md} FAILED: {ex}")
            torch.cuda.empty_cache()
    log("shard finished")


if __name__ == "__main__":
    sys.exit(main())
