#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — train the amortised network on SYNTHETIC equilibria only (torch, .venv, GPU).

Inputs  x = [coil currents / Ip (+ P2 split currents), log Ip, log F_vac | Thomson features (syn_common.ts_feats)].
Outputs y = [K PCA coefficients of psi_res / Ip[MA] (basis fitted on the synthetic TRAIN maps of this set only),
             q95, betaN (inv_core.scalars_kin of the synthetic solutions = the inverse's own conventions),
             auxiliaries li, kappa, R_axis, Z_axis and the latents (log c_p, b1, b2, gamma_ev, dr)].
Each equilibrium has NREAL noise realisations of its diagnostics; every mini-batch draws one of the allowed realisations per
sample (--reals) + feature dropout (whole Thomson block with p --p-ts, single isotherm level with p --p-lvl), so the same net
serves frames with and without a laser profile. Loss: Ip^2-weighted pixel MSE of psi_res (orthonormal basis) + w_s x MSE of
standardised scalars. Shot-split validation (--val-frac of the driver shots). Ensemble of --seeds members.

  python syn_train.py --synth DIR [DIR ...] --out OUTDIR --name NAME [--reals 1,2 --no-inner --xpt-only --gev-sig 0.03]
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from syn_gate import gate
from syn_shift import zshift

SCAL_KEYS = ("q95", "betaN", "li", "kappa", "a", "Rgeo", "volume", "tri_top", "tri_bot", "betap", "Rin", "Rout", "Ztop", "Zbot",
             "edge_gs", "edge_gs_in", "f95_ratio", "pav")
DIAG_KEYS = ("Rax", "Zax", "zt", "dIz_frac", "f_press", "kind", "n_it", "bps", "fvac", "ip", "t", "k", "gev", "dr", "cp", "b1", "b2",
             "be", "P0", "s", "a0", "Rxl", "Zxl", "Rxu", "Zxu", "dsep_psi", "d_tgt", "secs", "gev_g1")
OUT_NAMES = ("q95", "betaN", "li", "kappa", "Rax", "Zax", "volume", "lcp", "b1", "b2", "gev", "dr")
N_TGT = 2                                                   # q95, betaN = scored scalars; the rest are auxiliaries
NF_EDGE, NF_LVL0, N_LVL = 8, 8, 5


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_synth(dirs, max_per_shot=0, seed=0, ban=()):
    X, F, P, Sc, Dg, shot = [], [], [], [], [], []
    ban = set(ban)
    rng = np.random.default_rng(seed)
    cnt = {}
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f)
            for k_, v_ in json.loads(str(z["cnt"])).items():
                cnt[k_] = cnt.get(k_, 0) + v_
            n = len(z["X"])
            if n == 0 or z["X"].ndim != 2 or str(z["fname"]) in ban:
                continue
            sel = np.arange(n)
            if max_per_shot and n > max_per_shot:
                sel = np.sort(rng.choice(n, max_per_shot, replace=False))
            X.append(z["X"][sel]); F.append(z["F"][sel]); P.append(z["psi"][sel]); Sc.append(z["scal"][sel]); Dg.append(z["diag"][sel])
            shot.append(np.full(len(sel), zlib.crc32(str(z["fname"]).encode()) & 0x7FFFFFFF))
    return (np.concatenate(X), np.concatenate(F), np.concatenate(P), np.concatenate(Sc), np.concatenate(Dg), np.concatenate(shot), cnt)


class Net(nn.Module):
    def __init__(self, nin, nout, width=512, depth=4, drop=0.05):
        super().__init__()
        self.inp = nn.Linear(nin, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Dropout(drop),
                                                   nn.Linear(width, width)) for _ in range(depth)])
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, nout))

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def feat_mask(nf, no_inner=False, no_iso=False, no_prof=False, edges_only=False):
    m = np.ones(nf, np.float32)
    if no_inner:
        m[4:8] = 0.0
    if no_iso or edges_only:
        m[NF_LVL0:NF_LVL0 + 4 * N_LVL] = 0.0
    if no_prof or edges_only:
        m[NF_LVL0 + 4 * N_LVL:NF_LVL0 + 4 * N_LVL + 4] = 0.0          # Te peak / symmetric axis (beta_pe kept)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--ncomp", type=int, default=48)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed0", type=int, default=0, help="first member seed (members seed0 .. seed0+seeds-1; the data split / PCA basis are seed-independent)")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--bs", type=int, default=2048)
    ap.add_argument("--w-scal", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--drop", type=float, default=0.05)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--reals", default="1,2", help="noise realisations used for training (0 low, 1 nominal, 2 stress-like)")
    ap.add_argument("--val-real", type=int, default=1)
    ap.add_argument("--p-ts", type=float, default=0.10)
    ap.add_argument("--p-lvl", type=float, default=0.20)
    ap.add_argument("--p-prof", type=float, default=0.20, help="drop every profile-derived feature, keep the edges")
    ap.add_argument("--p-in", type=float, default=0.08, help="drop the inner-edge observation")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--ban", default=None, help="comma list of files with driver-shot names to exclude")
    ap.add_argument("--no-inner", action="store_true", help="inner-edge features removed (the INV_P7a inverse does not use them)")
    ap.add_argument("--no-iso", action="store_true")
    ap.add_argument("--no-prof", action="store_true")
    ap.add_argument("--edges-only", action="store_true")
    ap.add_argument("--xpt-only", action="store_true", help="train on X-point-bounded synthetic equilibria only")
    ap.add_argument("--gev-sig", type=float, default=0.0, help="> 0: importance weight exp(-gev^2 / 2 sig^2) (the inverse's field prior)")
    ap.add_argument("--max-per-shot", type=int, default=0)
    ap.add_argument("--no-ipw", action="store_true", help="no Ip^2 weighting of the flux loss")
    ap.add_argument("--zcanon", type=float, default=0.0, help="> 0 (= grid dZ [m]): Z-canonical flux targets = psi_res shifted vertically so "
                    "that the axis sits at Z = 0 (the map is shifted back by the Zax head at prediction time)")
    a = ap.parse_args()
    t_start = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ban = []
    for f in (a.ban.split(",") if a.ban else []):
        ban += [x for x in Path(f).read_text().split("\n") if x]
    X, F, P, Sc, Dg, shot, cnt = load_synth(a.synth, a.max_per_shot, ban=ban)
    gstat = {}
    if not a.no_gate:
        F, gstat = gate(F)
        log(f"feature gate on the synthetic set: {gstat}")
    sc = {n: Sc[:, i] for i, n in enumerate(SCAL_KEYS)}; dg = {n: Dg[:, i] for i, n in enumerate(DIAG_KEYS)}
    ip_ma = dg["ip"] / 1e6
    nR, nF = F.shape[1], F.shape[2]
    good = (np.isfinite(X).all(1) & np.isfinite(F.reshape(len(F), -1)).all(1) & np.isfinite(P.reshape(len(P), -1)).all(1)
            & (sc["q95"] > 1.0) & (sc["q95"] < 30.0) & (sc["betaN"] > 0.0) & (sc["betaN"] < 10.0) & (ip_ma > 0.02))
    for n in ("li", "kappa", "volume"):
        good &= np.isfinite(sc[n])
    n_all = int(good.sum())
    kind_frac = {k_: float(np.mean(dg["kind"][good] == v_)) for k_, v_ in (("xpt", 0.0), ("wall", 1.0), ("open", 2.0))}
    if a.xpt_only:
        good &= dg["kind"] == 0.0
    log(f"loaded {len(X)} synthetic samples from {len(np.unique(shot))} driver shots (generator counts {cnt}); usable {n_all}, "
        f"used {int(good.sum())} ({'X-point-bounded only' if a.xpt_only else 'all topologies'}; kinds {kind_frac}) on {dev}")
    X, F, P, shot, ip_ma = X[good], F[good], P[good], shot[good], ip_ma[good]
    sc = {n: v[good] for n, v in sc.items()}; dg = {n: v[good] for n, v in dg.items()}
    npx = P.shape[1] * P.shape[2]
    if a.zcanon > 0:
        P = zshift(P, dg["Zax"] / a.zcanon)
    Y = (P.reshape(len(P), npx) / ip_ma[:, None]).astype(np.float32)
    del P
    rng = np.random.default_rng(0)
    us = np.unique(shot)
    val_shots = set(rng.choice(us, max(1, int(round(a.val_frac * len(us)))), replace=False).tolist())
    is_val = np.array([s in val_shots for s in shot])
    mu = Y[~is_val].mean(0)
    sub = rng.choice(np.where(~is_val)[0], min(int((~is_val).sum()), 60000), replace=False)
    Yt = torch.tensor(Y[sub] - mu, device=dev)
    _, Sv, Vh = torch.linalg.svd(Yt, full_matrices=False)
    K = a.ncomp
    comps = Vh[:K].float()
    ev = (Sv ** 2).cpu().numpy(); evr = ev / ev.sum()
    del Yt
    Yc = torch.cat([torch.tensor(Y[i:i + 65536] - mu, device=dev) @ comps.T for i in range(0, len(Y), 65536)])
    rec = (Yc @ comps).cpu().numpy() + mu
    w_ip = (ip_ma ** 2).astype(np.float64)
    pca_r2 = 1 - ((rec - Y) ** 2 * w_ip[:, None]).sum() / (((Y - Y.mean(0)) ** 2) * w_ip[:, None]).sum()
    del rec
    log(f"PCA: {K} comps explain {evr[:K].sum():.6f} (8: {evr[:8].sum():.5f}, 16: {evr[:16].sum():.5f}, 32: {evr[:32].sum():.6f}); "
        f"reconstruction R2 of Ip*psi_res {pca_r2:.6f}")
    tot_var = float(Yc.var(0).sum())
    ys = np.stack([sc["q95"], sc["betaN"], sc["li"], sc["kappa"], dg["Rax"], dg["Zax"], sc["volume"], np.log(dg["cp"]), dg["b1"], dg["b2"],
                   dg["gev"], dg["dr"]], 1).astype(np.float32)
    ys_mu, ys_sd = ys[~is_val].mean(0), ys[~is_val].std(0) + 1e-6
    ysn = (ys - ys_mu) / ys_sd
    reals = [int(r) for r in a.reals.split(",")]
    fm = feat_mask(nF, a.no_inner, a.no_iso, a.no_prof, a.edges_only)
    F = F * fm[None, None, :]
    xc_mu, xc_sd = X[~is_val].mean(0), X[~is_val].std(0)
    xc_sd = np.where(xc_sd > 1e-8, xc_sd, 1.0)
    Ftr = F[~is_val][:, reals].reshape(-1, nF)
    f_mu, f_sd = Ftr.mean(0), Ftr.std(0)
    f_sd = np.where(f_sd > 1e-8, f_sd, 1.0)
    x_mu, x_sd = np.concatenate([xc_mu, f_mu]).astype(np.float32), np.concatenate([xc_sd, f_sd]).astype(np.float32)
    xall = np.concatenate([np.repeat(((X[~is_val] - xc_mu) / xc_sd)[:, None, :], len(reals), 1).reshape(-1, X.shape[1]), (Ftr - f_mu) / f_sd], 1)
    x_lo, x_hi = np.percentile(xall, 0.1, axis=0).astype(np.float32), np.percentile(xall, 99.9, axis=0).astype(np.float32)
    del xall, Ftr
    nXc = X.shape[1]
    w = np.ones(len(X), np.float32) if a.no_ipw else (ip_ma ** 2 / np.mean(ip_ma[~is_val] ** 2)).astype(np.float32)
    if a.gev_sig > 0:
        wg = np.exp(-0.5 * (dg["gev"] / a.gev_sig) ** 2).astype(np.float32)
        w = w * wg / wg[~is_val].mean()
        log(f"gev importance weights: effective sample fraction {(wg.sum() ** 2 / (wg ** 2).sum()) / len(wg):.3f}")
    nA = ys.shape[1] - N_TGT
    swt = torch.tensor([1.0] * N_TGT + [0.3] * nA, device=dev)
    T = lambda v: torch.tensor(v, device=dev)  # noqa: E731
    tr, va = T(~is_val), T(is_val)
    Xc = T(((X - xc_mu) / xc_sd).astype(np.float32)); Fn = T(((F - f_mu) / f_sd).astype(np.float32))
    z0 = T(((0.0 - f_mu) / f_sd).astype(np.float32))                    # normalised "no feature" vector
    Xc_tr, Fn_tr, C_tr, S_tr, W_tr = Xc[tr], Fn[tr], Yc[tr], T(ysn)[tr], T(w)[tr]
    Xc_va, Fn_va, C_va, S_va, W_va = Xc[va], Fn[va], Yc[va], T(ysn)[va], T(w)[va]
    reals_t = T(np.array(reals))
    nin, nout = nXc + nF, K + ys.shape[1]
    ntr = len(Xc_tr)
    members, hist = [], []

    def loss_fn(o, c, s, ww):
        lp = ((o[:, :K] - c) ** 2).sum(1) / tot_var
        ls = (((o[:, K:] - s) ** 2) * swt).sum(1) / swt.sum()
        return (ww * lp).mean() + a.w_scal * ls.mean(), (ww * lp).mean(), ls.mean()

    def xval(r, drop_ts=False):
        f = Fn_va[:, r]
        if drop_ts:
            f = z0.expand_as(f)
        return torch.cat([Xc_va, f], 1)

    for seed in range(a.seed0, a.seed0 + a.seeds):
        torch.manual_seed(1000 + seed)
        net = Net(nin, nout, a.width, a.depth, a.drop).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
        steps = a.epochs * int(np.ceil(ntr / a.bs))
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=max(0.05, 3.0 / steps))
        g = torch.Generator(device=dev).manual_seed(seed)
        best, best_state = np.inf, None
        for ep in range(a.epochs):
            net.train()
            perm = torch.randperm(ntr, generator=g, device=dev)
            for i in range(0, ntr, a.bs):
                j = perm[i:i + a.bs]
                r = reals_t[torch.randint(len(reals), (len(j),), generator=g, device=dev)]
                f = Fn_tr[j, r]
                if a.p_lvl > 0:                                  # drop single isotherm levels (flag + position, both sides)
                    dl = (torch.rand((len(j), N_LVL), generator=g, device=dev) < a.p_lvl).repeat_interleave(4, dim=1)
                    f = f.clone()
                    seg = f[:, NF_LVL0:NF_LVL0 + 4 * N_LVL]
                    f[:, NF_LVL0:NF_LVL0 + 4 * N_LVL] = torch.where(dl, z0[NF_LVL0:NF_LVL0 + 4 * N_LVL].expand_as(seg), seg)
                if a.p_prof > 0:                                 # edges only (profile-derived features failed the gate / missing)
                    dp_ = torch.rand((len(j), 1), generator=g, device=dev) < a.p_prof
                    f = f.clone()
                    f[:, NF_LVL0:] = torch.where(dp_, z0[NF_LVL0:].expand_as(f[:, NF_LVL0:]), f[:, NF_LVL0:])
                if a.p_in > 0:
                    di_ = torch.rand((len(j), 1), generator=g, device=dev) < a.p_in
                    f = f.clone()
                    f[:, 4:8] = torch.where(di_, z0[4:8].expand_as(f[:, 4:8]), f[:, 4:8])
                if a.p_ts > 0:                                   # frames without any laser profile
                    dt_ = torch.rand((len(j), 1), generator=g, device=dev) < a.p_ts
                    f = torch.where(dt_, z0.expand_as(f), f)
                l, _, _ = loss_fn(net(torch.cat([Xc_tr[j], f], 1)), C_tr[j], S_tr[j], W_tr[j])
                opt.zero_grad(set_to_none=True)
                l.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step(); sch.step()
            if ep % 5 == 4 or ep == a.epochs - 1:
                net.eval()
                with torch.no_grad():
                    xv = xval(a.val_real)
                    o = torch.cat([net(xv[i:i + 16384]) for i in range(0, len(xv), 16384)])
                    lv, lpv, lsv = loss_fn(o, C_va, S_va, W_va)
                hist.append({"seed": seed, "ep": ep + 1, "val": float(lv), "val_psi": float(lpv), "val_scal": float(lsv)})
                log(f"  seed {seed} ep {ep + 1}: val {float(lv):.5f} (psi {float(lpv):.5f}, scal {float(lsv):.4f})")
                if float(lv) < best:
                    best, best_state = float(lv), {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        members.append(best_state)
    # ---- ensemble validation diagnostics (synthetic, shot-held-out), per realisation and without Thomson
    nets = []
    for st in members:
        n_ = Net(nin, nout, a.width, a.depth, a.drop).to(dev); n_.load_state_dict(st); n_.eval(); nets.append(n_)
    Yv = Y[is_val]; ipv = ip_ma[is_val]; yv = ys[is_val]
    psi_true = Yv * ipv[:, None]
    sst = ((psi_true - psi_true.mean(0)) ** 2).sum()
    val = {}
    for lab, r, dts in [(f"real{r}", r, False) for r in range(nR)] + [("no_ts", a.val_real, True)]:
        with torch.no_grad():
            xv = xval(r, dts)
            o = torch.stack([torch.cat([m(xv[i:i + 16384]) for i in range(0, len(xv), 16384)]) for m in nets]).mean(0)
        psi_pred = ((o[:, :K] @ comps).cpu().numpy() + mu) * ipv[:, None]
        os_ = o[:, K:].cpu().numpy() * ys_sd + ys_mu
        val[lab] = {"r2_psi_res_abs": float(1 - ((psi_pred - psi_true) ** 2).sum() / sst),
                    **{n: float(1 - ((os_[:, i] - yv[:, i]) ** 2).sum() / ((yv[:, i] - yv[:, i].mean()) ** 2).sum()) for i, n in enumerate(OUT_NAMES)}}
        log(f"VAL {lab}: R2 Ip*psi_res {val[lab]['r2_psi_res_abs']:.5f}; " + " ".join(f"{n} {val[lab][n]:.3f}" for n in OUT_NAMES))
    diagj = {"n_train": int((~is_val).sum()), "n_val": int(is_val.sum()), "n_shots": int(len(us)), "pca_evr_K": float(evr[:K].sum()),
             "pca_r2": float(pca_r2), "val": val, "hist": hist, "args": vars(a), "gen_counts": cnt, "kind_frac": kind_frac, "gate": gstat,
             "gpu_seconds": time.time() - t_start, "device": str(dev),
             "latent_stats": {n: [float(np.percentile(dg[n], q)) for q in (5, 50, 95)] for n in ("gev", "dr", "cp", "b1", "b2", "f_press", "Zax", "d_tgt")},
             "scal_stats": {n: [float(np.percentile(sc[n], q)) for q in (5, 50, 95)] for n in ("q95", "betaN", "li", "kappa", "volume")}}
    torch.save({"members": members, "x_mu": x_mu, "x_sd": x_sd, "x_lo": x_lo, "x_hi": x_hi, "mu": mu, "comps": comps.cpu().numpy(),
                "ys_mu": ys_mu, "ys_sd": ys_sd, "names": list(OUT_NAMES), "K": K, "width": a.width, "depth": a.depth, "drop": a.drop,
                "nin": nin, "nXc": nXc, "nF": nF, "feat_mask": fm, "gate": not a.no_gate, "zcanon": float(a.zcanon), "diag": diagj}, out / f"model_{a.name}.pt")
    json.dump(diagj, open(out / f"train_{a.name}.json", "w"), indent=1, default=float)
    log(f"saved {out / f'model_{a.name}.pt'}; wall {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
