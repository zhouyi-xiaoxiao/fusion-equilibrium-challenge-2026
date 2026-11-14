#!/usr/bin/env python3
"""
Stage 11 / ch1_temporal — shot-sequence model for Ch1 (DIII-D), feature-v2.1 inputs, PCA-K coefficient outputs.

Stages (per seed):
  trunk : per-frame MLP  x(D) -> emb -> [K standardised PCA coefficients | q95 | betaN | aux]; random FRAME batches.
  seq   : trunk checkpoint + sequence module over ALL EFIT frames of a shot (bidirectional GRU or dilated TCN) with a
          ZERO-INITIALISED residual head (step 0 == the frame model); whole shots in padded batches; trunk frozen for
          FREEZE epochs, then fine-tuned at a lower learning rate.
  ctrl  : the same shot-batch fine-tuning schedule WITHOUT the sequence module (so "seq - ctrl" is the sequence gain,
          not the gain of extra training).

Loss per frame, in official score units:
  |e @ L_K|^2                      metric loss, e = raw-unit coefficient error (0.55(1-R2psi) + 0.20(1-Consistency))
  + W_QB * sum_j (0.15/2)/Var_j e_j^2   q95, betaN in original units, pooled train variances
  + W_AUX * mean standardised aux MSE   (20 scorer-/EFIT-derived geometry scalars, auxiliary only)

Modes:
  dev  (FULL=0): model data = the 6,891 train shots; VAL_FRAC of them (shot-disjoint, split seed 1000+seed) is held out
                 for early stopping = best-validation-epoch weights are restored. Holdout (150 shots) is only predicted
                 at the end (never used for a choice inside this script).
  full (FULL=1): model data = 6,891 + 150 = 7,041 shots, no validation, FIXED epochs, last weights; predicts the 874
                 public-test shots (order = feat/test/files.txt).

Everything is configured by environment variables (see CFG below); the run_*.sh scripts hold the exact recipes.
"""
from __future__ import annotations

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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch1_metric"))
from metric_loss import Metric  # noqa: E402

E = os.environ.get
SCR = Path(os.environ["SCR"])
FEAT = Path(E("FEAT", str(SCR / "stage11_ch1/feat_v21")))
MET = Path(E("MET", str(SCR / "stage11_ch1/metric")))
HC = SCR / "stage6_H"
OUT = Path(E("OUT", str(SCR / "stage11_ch1/temporal")))

CFG = dict(
    K=int(E("K", "80")), DX=int(E("DX", "785")),           # DX=785 drops the optional rec block (DAQ-era metadata)
    FULL=int(E("FULL", "0")), VAL_FRAC=float(E("VAL_FRAC", "0.05")),
    W_QB=float(E("W_QB", "1")), W_AUX=float(E("W_AUX", "0.01")),
    # trunk
    T_ARCH=E("T_ARCH", "mlp"), T_W=int(E("T_W", "1024")), T_E=int(E("T_E", "512")), T_NB=int(E("T_NB", "3")),
    T_DROP=float(E("T_DROP", "0")), EP_T=int(E("EP_T", "30")), LR_T=float(E("LR_T", "1e-3")),
    BATCH=int(E("BATCH", "4096")), WD_T=float(E("WD_T", "1e-6")),
)
SCFG = dict(
    S_ARCH=E("S_ARCH", "gru"), S_H=int(E("S_H", "192")), S_L=int(E("S_L", "2")), S_P=int(E("S_P", "256")),
    S_DROP=float(E("S_DROP", "0.1")), S_RAWX=int(E("S_RAWX", "1")), EP_S=int(E("EP_S", "20")),
    FREEZE=int(E("FREEZE", "2")), LR_S=float(E("LR_S", "5e-4")), LR_TFT=float(E("LR_TFT", "1e-4")),
    SEQ_BATCH=int(E("SEQ_BATCH", "16")), WD_S=float(E("WD_S", "1e-4")), S_CROP=float(E("S_CROP", "0")),
    S_SCRATCH=int(E("S_SCRATCH", "0")),                      # 1 = joint training from a randomly initialised trunk
)
SEEDS = [int(s) for s in E("SEEDS", "0").split(",")]
STAGES = E("STAGES", "trunk,seq,ctrl").split(",")
TTAG, STAG = E("TTAG", "t0"), E("STAG", "s0")
WRITE_PRED = E("WRITE_PRED", "1") == "1"
RUN = E("RUN_NAME", f"{TTAG}__{STAG}")
K, DX, FULL = CFG["K"], CFG["DX"], bool(CFG["FULL"])
MODE = "full" if FULL else "dev"
N_EX = 6
dev = torch.device("cuda")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ------------------------------------------------------------------ data
def time_extras(t, off):
    """Per-frame time encodings computed from the EFIT frame times only (ms)."""
    n = len(t)
    gp, gn = np.zeros(n), np.zeros(n)
    first, last, frac, ln = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for s in range(len(off) - 1):
        p, q = int(off[s]), int(off[s + 1])
        d = np.clip(np.diff(np.asarray(t[p:q], np.float64)), 0.0, None)
        gp[p + 1:q] = d
        gn[p:q - 1] = d
        first[p] = 1.0
        last[q - 1] = 1.0
        frac[p:q] = (np.arange(q - p) + 0.5) / (q - p)
        ln[p:q] = math.log(q - p) / 6.0
    ex = np.stack([np.log1p(gp / 20.0), np.log1p(gn / 20.0), first, last, frac, ln], 1)
    return np.nan_to_num(ex).astype(np.float32)


def load_split(name, need_y=True):
    d = FEAT / name
    o = {"X": np.nan_to_num(np.load(d / "X.npy"))[:, :DX], "off": np.load(d / "offsets.npy").astype(np.int64),
         "t": np.load(d / "times.npy")}
    if need_y:
        o["C"] = np.load(d / f"y_C{K}.npy").astype(np.float32)
        o["QB"] = np.stack([np.load(d / "y_q95.npy"), np.load(d / "y_betaN.npy")], 1).astype(np.float32)
        o["A"] = np.load(d / "y_aux.npy").astype(np.float32)
    return o


def merge(a, b):
    o = {k: np.concatenate([a[k], b[k]]) for k in a if k != "off"}
    o["off"] = np.concatenate([a["off"], a["off"][-1] + b["off"][1:]])
    return o


def to_gpu(d):
    g = {"X": torch.from_numpy(d["X"]).to(dev), "off": torch.from_numpy(d["off"]).to(dev),
         "EX": torch.from_numpy(time_extras(d["t"], d["off"])).to(dev), "n_shots": len(d["off"]) - 1}
    for k in ("C", "QB", "A"):
        if k in d:
            g[k] = torch.from_numpy(d[k]).to(dev)
    return g


def pad(D, sh):
    p = D["off"][sh]
    lens = D["off"][sh + 1] - p
    ar = torch.arange(int(lens.max()), device=dev)[None, :]
    mask = ar < lens[:, None]
    return torch.where(mask, p[:, None] + ar, p[:, None]), mask, lens


def frames_of(D, sh):
    out = []
    for i in range(0, len(sh), 256):
        idx, mask, _ = pad(D, sh[i:i + 256])
        out.append(idx[mask])
    return torch.cat(out)


# ------------------------------------------------------------------ models
class ResBlock(nn.Module):
    def __init__(self, w, drop):
        super().__init__()
        self.n, self.l1, self.l2, self.d = nn.LayerNorm(w), nn.Linear(w, w), nn.Linear(w, w), nn.Dropout(drop)

    def forward(self, h):
        return h + self.l2(self.d(F.silu(self.l1(F.silu(self.n(h))))))


class Trunk(nn.Module):
    def __init__(self, d_in, n_out, c):
        super().__init__()
        w, e = c["T_W"], c["T_E"]
        if c["T_ARCH"] == "mlp":
            self.body = nn.Sequential(nn.Linear(d_in, w), nn.SiLU(), nn.Linear(w, w), nn.SiLU(), nn.Linear(w, e), nn.SiLU())
        else:
            self.body = nn.Sequential(nn.Linear(d_in, w), *[ResBlock(w, c["T_DROP"]) for _ in range(c["T_NB"])],
                                      nn.LayerNorm(w), nn.SiLU(), nn.Linear(w, e), nn.SiLU())
        self.head = nn.Linear(e, n_out)

    def forward(self, x):
        h = self.body(x)
        return self.head(h), h


class TBlock(nn.Module):
    def __init__(self, p, k, dil, drop):
        super().__init__()
        self.n = nn.LayerNorm(p)
        self.c1 = nn.Conv1d(p, p, k, padding=dil * (k - 1) // 2, dilation=dil)
        self.c2 = nn.Conv1d(p, p, 1)
        self.d = nn.Dropout(drop)

    def forward(self, z, m):
        h = (self.n(z) * m).transpose(1, 2)
        h = self.c2(F.silu(self.c1(h))).transpose(1, 2)
        return z + self.d(h) * m


class SeqMod(nn.Module):
    """Sequence module; its output is ADDED to the trunk output. head is zero-initialised."""

    def __init__(self, e, d_in, n_out, c):
        super().__init__()
        p, h = c["S_P"], c["S_H"]
        self.arch, self.rawx = c["S_ARCH"], bool(c["S_RAWX"])
        self.pe = nn.Linear(e, p)
        self.px = nn.Linear(d_in, p // 2) if self.rawx else None
        self.inp = nn.Linear(p + (p // 2 if self.rawx else 0) + N_EX, p)
        self.drop = nn.Dropout(c["S_DROP"])
        if self.arch == "gru":
            self.rnn = nn.GRU(p, h, c["S_L"], batch_first=True, bidirectional=True,
                              dropout=c["S_DROP"] if c["S_L"] > 1 else 0.0)
            d_out = 2 * h
        else:
            self.blocks = nn.ModuleList([TBlock(p, 5, 2 ** i, c["S_DROP"]) for i in range(c["S_L"])])
            self.fn = nn.LayerNorm(p)
            d_out = p
        self.head = nn.Linear(d_out, n_out)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, emb, x, ex, mask, lens):
        z = [self.pe(emb), ex]
        if self.rawx:
            z.insert(1, self.px(x))
        z = self.drop(F.silu(self.inp(torch.cat(z, -1))))
        if self.arch == "gru":
            pk = pack_padded_sequence(z, lens.cpu(), batch_first=True, enforce_sorted=False)
            o, _ = self.rnn(pk)
            o, _ = pad_packed_sequence(o, batch_first=True, total_length=z.shape[1])
        else:
            m = mask[..., None].to(z.dtype)
            o = z * m
            for b in self.blocks:
                o = b(o, m)
            o = F.silu(self.fn(o))
        return self.head(o)


def fwd(trunk, seqm, D, idx, mask, lens, grad_trunk):
    x = D["X"][idx]
    if grad_trunk:
        out0, emb = trunk(x)
    else:
        with torch.no_grad():
            out0, emb = trunk(x)
    if seqm is None:
        return out0
    return out0 + seqm(emb, x, D["EX"][idx], mask, lens)


# ------------------------------------------------------------------ main
def main():
    t00 = time.time()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    for sub in ("weights", "members", "preds", "logs"):
        (OUT / MODE / sub).mkdir(parents=True, exist_ok=True)
    m = Metric(K, MET)
    tr = load_split("train")
    ho = load_split("holdout")
    n_tr_shots = len(tr["off"]) - 1
    if FULL:
        te = to_gpu(load_split("test", need_y=False))
        model_np = merge(tr, ho)
    else:
        model_np = tr
    Dm = to_gpu(model_np)
    Dh = to_gpu(ho)
    del tr, model_np
    n_shots = Dm["n_shots"]
    n_aux = Dm["A"].shape[1]
    n_out = K + 2 + n_aux
    L = torch.tensor(m.L, dtype=torch.float32, device=dev)
    log(f"mode {MODE} K {K} DX {DX} shots {n_shots} frames {Dm['X'].shape[0]} | run {RUN} seeds {SEEDS} stages {STAGES} "
        f"| load {time.time()-t00:.0f}s gpu {torch.cuda.memory_allocated()/1e9:.1f} GB | torch {torch.__version__}")
    log("CFG " + json.dumps(CFG) + " SCFG " + json.dumps(SCFG))

    pc = np.load(HC / f"pca_{K}.npz")
    comps, pmean = pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64)
    off_h = ho["off"]
    Yh = ho["C"].astype(np.float64)
    JWh = np.load(MET / f"JW_holdout_{K}.npy") if (K == 80 and not FULL) else None

    def write_preds(name, C, q, b):
        psi = C @ comps + pmean[None, :]
        d = {}
        for s in range(len(off_h) - 1):
            p, e = off_h[s], off_h[s + 1]
            d[f"shot_{s:04d}_psirz"] = psi[p:e].reshape(-1, 65, 65).astype(np.float32)
            d[f"shot_{s:04d}_q95"] = q[p:e].astype(np.float32)
            d[f"shot_{s:04d}_betaN"] = b[p:e].astype(np.float32)
        np.savez(OUT / MODE / "preds" / f"preds_d3d_{name}.npz", **d)

    rec, names = {}, []
    members = {st: [] for st in ("trunk", "seq", "ctrl")}
    stage_name = {"trunk": TTAG, "seq": f"{TTAG}__{STAG}", "ctrl": f"{TTAG}__{STAG}c"}

    for seed in SEEDS:
        # ---------------- split (by shot) + normalisation constants from the training rows only
        rs = np.random.RandomState(1000 + seed)
        perm = rs.permutation(n_shots)
        n_val = 0 if FULL else int(round(CFG["VAL_FRAC"] * n_shots))
        va_sh = torch.from_numpy(np.sort(perm[:n_val])).to(dev)
        tr_sh = torch.from_numpy(np.sort(perm[n_val:])).to(dev)
        fr = frames_of(Dm, tr_sh)
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
        def run_eval(trunk, seqm, D, sh, want_out=False, has_y=True):
            trunk.eval()
            if seqm is not None:
                seqm.eval()
            s_m = s_q = 0.0
            n = 0
            outs = []
            for i in range(0, len(sh), 64):
                idx, mask, lens = pad(D, sh[i:i + 64])
                out = fwd(trunk, seqm, D, idx, mask, lens, False)[mask]
                if has_y:
                    lm, lq, _ = losses(out, D, idx[mask])
                    s_m += float(lm.sum())
                    s_q += float(lq.sum())
                n += out.shape[0]
                if want_out:
                    outs.append(out)
            return (s_m / max(n, 1), s_q / max(n, 1), torch.cat(outs) if want_out else None)

        def decode(out):
            C = (out[:, :K] * nrm["sdC"] + nrm["muC"]).double().cpu().numpy()
            QB = (out[:, K:K + 2] * nrm["sdQ"] + nrm["muQ"]).double().cpu().numpy()
            return C, QB[:, 0], QB[:, 1]

        def finish(stage, trunk, seqm, info):
            """Predict holdout (dev) or test (full), save member outputs + weights, log diagnostics."""
            name = stage_name[stage]
            sd_ = {"trunk": trunk.state_dict(), "seq": None if seqm is None else seqm.state_dict(),
                   "nrm": {k: v.cpu() for k, v in nrm.items()}, "cfg": CFG, "scfg": SCFG if stage != "trunk" else None,
                   "seed": seed, "info": info}
            torch.save(sd_, OUT / MODE / "weights" / f"{name}_s{seed}.pt")
            if FULL:
                _, _, out = run_eval(trunk, seqm, te, torch.arange(te["n_shots"], device=dev), want_out=True, has_y=False)
                C, q, b = decode(out)
                np.savez(OUT / MODE / "members" / f"test_{name}_s{seed}.npz", coef=C.astype(np.float32), q95=q.astype(np.float32),
                         betaN=b.astype(np.float32), basis=f"stage6_H/pca_{K}.npz", order="feat_v21/test/files.txt")
            else:
                hm, hq, out = run_eval(trunk, seqm, Dh, torch.arange(Dh["n_shots"], device=dev), want_out=True)
                C, q, b = decode(out)
                np.savez(OUT / MODE / "members" / f"hold_{name}_s{seed}.npz", coef=C.astype(np.float32), q95=q.astype(np.float32),
                         betaN=b.astype(np.float32), basis=f"stage6_H/pca_{K}.npz", order="standard 150-shot holdout frame order")
                info.update(hold_metric=hm, hold_qb=hq)
                if JWh is not None:
                    info["hold_PF"] = float(m.loss_PF(C, Yh, JWh).mean())
            members[stage].append((C, q, b))
            rec[f"{name}_s{seed}"] = info
            log(f"== {name} seed {seed}: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in info.items() if k != 'curve'}))

        # ---------------- stage 1: frame trunk
        torch.manual_seed(seed)
        np.random.seed(seed)
        trunk = Trunk(DX, n_out, CFG).to(dev)
        ck = OUT / MODE / "weights" / f"{TTAG}_s{seed}.pt"
        if ck.exists():
            sd_ = torch.load(ck, map_location=dev)
            assert sd_["cfg"] == CFG, f"trunk checkpoint {ck} was trained with a different CFG"
            trunk.load_state_dict(sd_["trunk"])
            log(f"trunk {TTAG} seed {seed}: loaded checkpoint (val {sd_['info'].get('val_best')})")
            if "trunk" in STAGES:
                finish("trunk", trunk, None, dict(sd_["info"], reloaded=True))
        else:
            t1 = time.time()
            N = len(fr)
            spe = N // CFG["BATCH"]
            opt = torch.optim.AdamW(trunk.parameters(), lr=CFG["LR_T"], weight_decay=CFG["WD_T"])
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["LR_T"], total_steps=CFG["EP_T"] * spe, pct_start=0.05)
            g = torch.Generator(device=dev)
            g.manual_seed(seed)
            best, best_ep, best_sd, curve = float("inf"), -1, None, []
            for ep in range(CFG["EP_T"]):
                trunk.train()
                pm = fr[torch.randperm(N, device=dev, generator=g)]
                tot = 0.0
                for bi in range(spe):
                    idx = pm[bi * CFG["BATCH"]:(bi + 1) * CFG["BATCH"]]
                    out, _ = trunk(Dm["X"][idx])
                    lm, lq, la = losses(out, Dm, idx)
                    loss = (lm + CFG["W_QB"] * lq + CFG["W_AUX"] * la).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    tot += float(loss.detach())
                if n_val:
                    vm, vq, _ = run_eval(trunk, None, Dm, va_sh)
                    curve.append([ep + 1, tot / spe, vm, vq])
                    if vm + vq < best:
                        best, best_ep, best_sd = vm + vq, ep + 1, copy.deepcopy(trunk.state_dict())
                else:
                    curve.append([ep + 1, tot / spe])
            if best_sd is not None:
                trunk.load_state_dict(best_sd)
            info = {"epochs": CFG["EP_T"], "best_ep": best_ep, "val_best": best if n_val else None,
                    "val_last": (curve[-1][2] + curve[-1][3]) if n_val else None, "val_metric_best": min(c[2] for c in curve) if n_val else None,
                    "train_last": curve[-1][1], "sec": time.time() - t1, "curve": curve}
            finish("trunk", trunk, None, info)
        trunk_sd = copy.deepcopy(trunk.state_dict())

        # ---------------- stage 2: sequence fine-tune (and the no-sequence control)
        for stage in ("seq", "ctrl"):
            if stage not in STAGES:
                continue
            t1 = time.time()
            torch.manual_seed(10_000 + seed)
            trunk = Trunk(DX, n_out, CFG).to(dev)
            if not SCFG["S_SCRATCH"]:
                trunk.load_state_dict(trunk_sd)
            seqm = SeqMod(CFG["T_E"], DX, n_out, SCFG).to(dev) if stage == "seq" else None
            B = SCFG["SEQ_BATCH"]
            spe = len(tr_sh) // B
            total, fz = SCFG["EP_S"] * spe, SCFG["FREEZE"] * spe
            warm = max(1, int(0.03 * total))
            groups = [{"params": trunk.parameters(), "lr": SCFG["LR_TFT"], "weight_decay": CFG["WD_T"]}]
            lams = [lambda st: 0.0 if st < fz else min(1.0, (st - fz + 1) / warm) * 0.5 * (1 + math.cos(math.pi * (st - fz) / max(1, total - fz)))]
            if seqm is not None:
                groups.append({"params": seqm.parameters(), "lr": SCFG["LR_S"], "weight_decay": SCFG["WD_S"]})
                lams.append(lambda st: min(1.0, (st + 1) / warm) * 0.5 * (1 + math.cos(math.pi * st / total)))
            opt = torch.optim.AdamW(groups)
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lams)
            g = torch.Generator(device=dev)
            g.manual_seed(10_000 + seed)
            v0 = run_eval(trunk, seqm, Dm, va_sh)[:2] if n_val else (0.0, 0.0)
            best, best_ep, curve = (v0[0] + v0[1]) if n_val else float("inf"), 0, []
            best_sd = (copy.deepcopy(trunk.state_dict()), None if seqm is None else copy.deepcopy(seqm.state_dict()))
            for ep in range(SCFG["EP_S"]):
                frozen = ep < SCFG["FREEZE"]
                if frozen and seqm is None:
                    for _ in range(spe):
                        sched.step()
                    continue
                trunk.train(not frozen)
                if seqm is not None:
                    seqm.train()
                pm = tr_sh[torch.randperm(len(tr_sh), device=dev, generator=g)]
                tot = 0.0
                for bi in range(spe):
                    idx, mask, lens = pad(Dm, pm[bi * B:(bi + 1) * B])
                    if SCFG["S_CROP"] > 0 and float(torch.rand(1, device=dev, generator=g)) < SCFG["S_CROP"]:
                        T = idx.shape[1]
                        w = int(torch.randint(max(8, T // 4), T + 1, (1,), device=dev, generator=g))
                        s0 = int(torch.randint(0, T - w + 1, (1,), device=dev, generator=g))
                        idx, mask = idx[:, s0:s0 + w], mask[:, s0:s0 + w]
                        keep = mask.any(1)
                        idx, mask = idx[keep], mask[keep]
                        lens = mask.sum(1)
                    out = fwd(trunk, seqm, Dm, idx, mask, lens, not frozen)[mask]
                    lm, lq, la = losses(out, Dm, idx[mask])
                    loss = (lm + CFG["W_QB"] * lq + CFG["W_AUX"] * la).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    sched.step()
                    tot += float(loss.detach())
                if n_val:
                    vm, vq, _ = run_eval(trunk, seqm, Dm, va_sh)
                    curve.append([ep + 1, tot / spe, vm, vq])
                    if vm + vq < best:
                        best, best_ep = vm + vq, ep + 1
                        best_sd = (copy.deepcopy(trunk.state_dict()), None if seqm is None else copy.deepcopy(seqm.state_dict()))
                else:
                    curve.append([ep + 1, tot / spe])
            if n_val:
                trunk.load_state_dict(best_sd[0])
                if seqm is not None:
                    seqm.load_state_dict(best_sd[1])
            info = {"epochs": SCFG["EP_S"], "best_ep": best_ep, "val_start": (v0[0] + v0[1]) if n_val else None,
                    "val_best": best if n_val else None, "val_last": (curve[-1][2] + curve[-1][3]) if n_val else None,
                    "val_metric_start": v0[0] if n_val else None, "val_metric_best": min(c[2] for c in curve) if n_val else None,
                    "train_last": curve[-1][1], "sec": time.time() - t1, "curve": curve}
            finish(stage, trunk, seqm, info)
        del fr

    # ---------------- bags + harness-layout prediction files (dev only)
    for stage in ("trunk", "seq", "ctrl"):
        ms = members[stage]
        if not ms:
            continue
        name = stage_name[stage]
        Cb, qb, bb = (np.mean([x[i] for x in ms], 0) for i in range(3))
        tag = "test" if FULL else "hold"
        np.savez(OUT / MODE / "members" / f"{tag}_{name}_bag{len(ms)}.npz", coef=Cb.astype(np.float32), q95=qb.astype(np.float32),
                 betaN=bb.astype(np.float32), basis=f"stage6_H/pca_{K}.npz", seeds=np.array(SEEDS))
        if FULL:
            continue
        r = {"hold_metric": float(m.loss_G(Cb, Yh).mean())}
        if JWh is not None:
            r["hold_PF"] = float(m.loss_PF(Cb, Yh, JWh).mean())
        rec[f"{name}_bag{len(ms)}"] = r
        log(f"== {name} bag{len(ms)}: " + json.dumps({k: round(v, 6) for k, v in r.items()}))
        if WRITE_PRED:
            write_preds(f"{name}_s{SEEDS[0]}", *ms[0])
            names.append(f"{name}_s{SEEDS[0]}")
            if len(ms) > 1:
                write_preds(f"{name}_bag{len(ms)}", Cb, qb, bb)
                names.append(f"{name}_bag{len(ms)}")
    gsec = time.time() - t00
    json.dump({"run": RUN, "mode": MODE, "cfg": CFG, "scfg": SCFG, "seeds": SEEDS, "stages": STAGES, "results": rec,
               "gpu_seconds_total": gsec, "torch": torch.__version__}, open(OUT / MODE / "logs" / f"{RUN}.json", "w"), indent=1)
    if names:
        with open(OUT / MODE / "preds" / f"names_{RUN}.txt", "w") as f:
            f.write("\n".join(names) + "\n")
    log(f"done {RUN} in {gsec:.0f}s GPU wall (incl. load)")


if __name__ == "__main__":
    main()
