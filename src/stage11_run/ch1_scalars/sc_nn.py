#!/usr/bin/env python3
"""
Stage 11 / ch1_scalars — GPU heads for q95 and betaN on FEATURE-V2.1 (D = 789).

Families
  mlp_*  frame-level residual MLP (input linear -> 3 pre-norm residual blocks of width 1024 -> LayerNorm -> SiLU -> linear head,
         plus a zero-initialised linear skip from the inputs). Outputs: q95 (raw or log parametrisation), betaN, and 36 auxiliary
         regression targets (20 EFIT / scorer-derived geometry scalars + the first 16 PCA80 flux coefficients) that share the trunk.
  gru_*  the trained MLP of the same seed + a zero-initialised residual bidirectional GRU over the EFIT frames of each shot
         (trunk frozen for the first epochs, then fine-tuned at a low learning rate).
Loss: Huber (quadratic below 0.5 sigma) on standardised targets, so it is the variance-normalised squared error the score uses.
No early stopping and no holdout-based choice inside a run: fixed epochs, fixed seeds; the holdout is only read after training.

Env: TASKS="cfg:seed,cfg:seed,..."  WORKER(0) NWORK(1) -> this process runs TASKS[WORKER::NWORK]
     FULL=1 trains on all 7,041 shots (train + holdout) and predicts the public test (row order = test/files.txt)
     EPOCHS / GEPOCHS override the fixed epoch counts (smoke tests only)
Output: members/<cfg>_s<seed>.npz {q95, betaN} in holdout frame order (FULL: full/<cfg>_s<seed>.npz in test row order),
        weights/<mode>/<cfg>_s<seed>.pt, nn_log.jsonl
"""
from __future__ import annotations

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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sc_common import FEAT, FULL, MEM, MODE, WTS, log, phys_extra, phys_raw, pooled_r2  # noqa: E402

NAUX, NPCMAX = 20, 80   # EFIT / scorer-derived scalars, PCA80 flux coefficients available as auxiliary targets
LOGQMAX = math.log(60.0)
BASE = dict(kind="mlp", qpar="raw", qloss="t", aux_w=0.1, width=1024, blocks=3, drop=0.0, lr=1e-3, wd=1e-5, batch=4096,
            epochs=30, huber=0.5, cols="all", npc=16, innoise=0.0, ctx=0, phys=0)
GRU = dict(emb=256, hid=192, layers=2, gdrop=0.05, sbatch=16, gepochs=15, freeze=2, lr_trunk=1e-4, lr_gru=5e-4)
CFG = {
    "mlp_raw_noaux": dict(aux_w=0.0),
    "mlp_raw": dict(),
    "mlp_raw_aux1": dict(aux_w=1.0),
    "mlp_log": dict(qpar="log"),
    "mlp_logr": dict(qpar="log", qloss="raw"),
    "mlp_raw_norec": dict(cols="norec"),
    "mlp_log_norec": dict(qpar="log", cols="norec"),
    "mlp_log_e60": dict(qpar="log", epochs=60),
    "mlp_log_w2k": dict(qpar="log", width=2048),
    "mlp_log_aux1": dict(qpar="log", aux_w=1.0),
    # round 2: regularisation / capacity / context around mlp_log_aux1 ("la1")
    "la1_e15": dict(qpar="log", aux_w=1.0, epochs=15),
    "la1_e20": dict(qpar="log", aux_w=1.0, epochs=20),
    "la1_d10": dict(qpar="log", aux_w=1.0, drop=0.10),
    "la1_d20": dict(qpar="log", aux_w=1.0, drop=0.20),
    "la1_wd2": dict(qpar="log", aux_w=1.0, wd=1e-2),
    "la1_in05": dict(qpar="log", aux_w=1.0, innoise=0.05),
    "la1_aux3": dict(qpar="log", aux_w=3.0),
    "la1_pc80": dict(qpar="log", aux_w=1.0, npc=80),
    "la1_ctx": dict(qpar="log", aux_w=1.0, ctx=1),
    "la1_b2k": dict(qpar="log", aux_w=1.0, batch=2048),
    "la1_w512": dict(qpar="log", aux_w=1.0, width=512),
    "la1_nb5": dict(qpar="log", aux_w=1.0, blocks=5),
    "la1_lr3": dict(qpar="log", aux_w=1.0, lr=3e-4),
    "gru_la1_fz": dict(kind="gru", base="mlp_log_aux1", freeze=15),
    # round 3: around la1_d10 (dropout 0.10), the round-2 winner
    "la1_d10wd": dict(qpar="log", aux_w=1.0, drop=0.10, wd=1e-2),
    "la1_d10_e45": dict(qpar="log", aux_w=1.0, drop=0.10, epochs=45),
    "la1_d10_w2k": dict(qpar="log", aux_w=1.0, drop=0.10, width=2048),
    "la1_d10_phys": dict(qpar="log", aux_w=1.0, drop=0.10, phys=1),
    "ra1_d10": dict(qpar="raw", aux_w=1.0, drop=0.10),
    "gru_d10_fz": dict(kind="gru", base="la1_d10", freeze=15),
    # round 4 (2026-10-12): the pooled-R2 score is MSE, and |e| > 0.5 sd frames (0.05 %) carry 23 % (q95) / 9 % (betaN) of
    # the bl_d10 holdout SSE -> loosen / remove the Huber cap on the two scalar residuals (aux loss unchanged)
    "la1_d10_h2": dict(qpar="log", aux_w=1.0, drop=0.10, huber=2.0),
    "la1_d10_mse": dict(qpar="log", aux_w=1.0, drop=0.10, huber=1e6),
    "ra1_d10_mse": dict(qpar="raw", aux_w=1.0, drop=0.10, huber=1e6),
    "lr1_d10_mse": dict(qpar="log", qloss="raw", aux_w=1.0, drop=0.10, huber=1e6),
    # round 5: the opposite direction (round 4: every looser cap lost) -> tighter caps on the scalars and on the aux targets
    "la1_d10_h025": dict(qpar="log", aux_w=1.0, drop=0.10, huber=0.25),
    "la1_d10_h01": dict(qpar="log", aux_w=1.0, drop=0.10, huber=0.1),
    "la1_d10_h025a": dict(qpar="log", aux_w=1.0, drop=0.10, huber=0.25, ahuber=0.25),
    "la1_d10_a025": dict(qpar="log", aux_w=1.0, drop=0.10, ahuber=0.25),
    "gru_raw": dict(kind="gru", base="mlp_raw"),
    "gru_log": dict(kind="gru", base="mlp_log"),
    "gru_log_aux1": dict(kind="gru", base="mlp_log_aux1"),
    "gru_log_norec": dict(kind="gru", base="mlp_log_norec"),
    "gru_log_h384": dict(kind="gru", base="mlp_log", hid=384, emb=384),
    "gru_log_e30": dict(kind="gru", base="mlp_log", gepochs=30),
}


def get_cfg(name):
    over = CFG[name]
    if over.get("kind") == "gru":
        c = get_cfg(over["base"]); c.update(GRU); c.update(over)
    else:
        c = dict(BASE); c.update(over)
    c["name"] = name
    if "EPOCHS" in os.environ:
        c["epochs"] = int(os.environ["EPOCHS"])
    if "GEPOCHS" in os.environ and "gepochs" in c:
        c["gepochs"] = int(os.environ["GEPOCHS"]); c["freeze"] = min(c["freeze"], c["gepochs"] // 2)
    return c


class Data:
    """Everything on the GPU once per process. X carries one extra all-zero row (index N) used as sequence padding."""

    def __init__(self, dev):
        t0 = time.time()

        def ld(split):
            d = FEAT / split
            return [np.nan_to_num(np.load(d / "X.npy")), np.load(d / "y_q95.npy"), np.load(d / "y_betaN.npy"),
                    np.load(d / "y_aux.npy"), np.load(d / "y_C80.npy")[:, :NPCMAX], np.load(d / "offsets.npy").astype(np.int64)]
        X, q, b, aux, pc, off = ld("train")
        if FULL:
            h = ld("holdout")
            X = np.concatenate([X, h[0]]); q = np.concatenate([q, h[1]]); b = np.concatenate([b, h[2]])
            aux = np.concatenate([aux, h[3]]); pc = np.concatenate([pc, h[4]]); off = np.concatenate([off, off[-1] + h[5][1:]])
            Xe = np.nan_to_num(np.load(FEAT / "test/X.npy")); offe = np.load(FEAT / "test/offsets.npy").astype(np.int64)
            self.yq_e = self.yb_e = None
        else:
            Xe, qe, be, _, _, offe = ld("holdout")
            self.yq_e, self.yb_e = qe.astype(np.float64), be.astype(np.float64)
        self.N, self.D = X.shape
        self.names = (FEAT / "names.txt").read_text().split()
        T = np.column_stack([q, b, aux, pc]).astype(np.float64)
        tq = np.log(np.maximum(q.astype(np.float64), 0.05))
        self.mu, self.sd = np.nanmean(T, 0), np.maximum(np.nanstd(T, 0), 1e-12)   # auxiliary targets can be NaN (derivation failed)
        self.mu_t, self.sd_t = float(tq.mean()), float(tq.std())
        U = (T - self.mu) / self.sd
        assert np.isfinite(U[:, :2]).all(), "non-finite q95 / betaN label"
        self.aux_ok = torch.from_numpy(np.isfinite(U[:, 2:])).to(dev)
        self.aux_nan_frac = float(1.0 - np.isfinite(U[:, 2:]).mean())
        U = np.clip(np.nan_to_num(U), -50.0, 50.0)
        self.U = torch.from_numpy(U.astype(np.float32)).to(dev)                                  # standardised targets, q raw
        self.Uq_log = torch.from_numpy(((tq - self.mu_t) / self.sd_t).astype(np.float32)).to(dev)
        self.q_raw = torch.from_numpy(q.astype(np.float32)).to(dev)
        self.X = torch.from_numpy(np.concatenate([X, np.zeros((1, self.D), np.float32)])).to(dev)
        del X
        self.start = torch.from_numpy(off[:-1]).to(dev); self.len = torch.from_numpy(np.diff(off)).to(dev)
        self.Xe = torch.from_numpy(np.concatenate([Xe, np.zeros((1, self.D), np.float32)])).to(dev)
        self.Ne = len(Xe)
        self.start_e = torch.from_numpy(offe[:-1]).to(dev); self.len_e = torch.from_numpy(np.diff(offe)).to(dev)
        self._ctx = None
        self._phys = None
        self.full = FULL
        log(f"data: mode {MODE} X {tuple(self.X.shape)} eval {tuple(self.Xe.shape)} shots {len(off) - 1}/{len(offe) - 1} "
            f"targets {T.shape[1]} aux NaN frac {self.aux_nan_frac:.2e} in {time.time() - t0:.0f}s; gpu {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    def stats(self):
        return {"mu": self.mu.tolist(), "sd": self.sd.tolist(), "mu_t": self.mu_t, "sd_t": self.sd_t}

    def inputs(self, c):
        """(train inputs, eval inputs). ctx=1 appends the per-shot mean of every feature (shot context) to each frame."""
        if c.get("phys", 0):
            assert not c["ctx"]
            E, Ee, _ = self.phys(self.X.device)
            return torch.cat([self.X, E], 1), torch.cat([self.Xe, Ee], 1)
        if not c["ctx"]:
            return self.X, self.Xe
        if self._ctx is None:
            def add(X, start, lens):
                n = len(X) - 1
                sof = torch.repeat_interleave(torch.arange(len(start), device=X.device), lens)
                m = torch.zeros((len(start), X.shape[1]), device=X.device).index_add_(0, sof, X[:n]) / lens[:, None]
                return torch.cat([X, torch.cat([m[sof], torch.zeros((1, X.shape[1]), device=X.device)])], 1)
            self._ctx = (add(self.X, self.start, self.len), add(self.Xe, self.start_e, self.len_e))
        return self._ctx

    def phys(self, dev):
        """(train extra, eval extra, stats): physics log features from X_raw, train-row standardised (stats saved with weights)."""
        if self._phys is None:
            R = phys_raw(FEAT / "train")
            if self.full:
                R = np.concatenate([R, phys_raw(FEAT / "holdout")])
            E, st = phys_extra(R)
            Ee, _ = phys_extra(phys_raw(FEAT / ("test" if self.full else "holdout")), st)
            z = lambda A: torch.from_numpy(np.concatenate([A, np.zeros((1, A.shape[1]), np.float32)])).to(dev)
            self._phys = (z(E), z(Ee), st)
        return self._phys

    def colmask(self, c, dev):
        cols = c["cols"]
        m = torch.ones(in_dim(c, self), device=dev)
        if cols == "norec":
            for i, n in enumerate(self.names):
                if n.startswith("rec_"):
                    m[i] = 0.0
        return m


class ResBlock(nn.Module):
    def __init__(self, w, drop):
        super().__init__()
        self.n = nn.LayerNorm(w); self.a = nn.Linear(w, w); self.b = nn.Linear(w, w); self.d = nn.Dropout(drop)

    def forward(self, x):
        return x + self.b(self.d(F.silu(self.a(self.n(x)))))


class FrameNet(nn.Module):
    def __init__(self, D, w, nb, n_out, drop, colmask):
        super().__init__()
        self.register_buffer("cm", colmask)
        self.inp = nn.Linear(D, w)
        self.blocks = nn.ModuleList([ResBlock(w, drop) for _ in range(nb)])
        self.norm = nn.LayerNorm(w)
        self.head = nn.Linear(w, n_out)
        self.skip = nn.Linear(D, n_out, bias=False)
        nn.init.zeros_(self.skip.weight)

    def feats(self, x):
        h = self.inp(x * self.cm)
        for blk in self.blocks:
            h = blk(h)
        return F.silu(self.norm(h))

    def base(self, x, h):
        return self.head(h) + self.skip(x * self.cm)

    def forward(self, x):
        return self.base(x, self.feats(x))


class SeqNet(nn.Module):
    def __init__(self, frame, w, n_out, emb, hid, layers, gdrop):
        super().__init__()
        self.frame = frame
        self.proj = nn.Linear(w, emb)
        self.gru = nn.GRU(emb, hid, num_layers=layers, batch_first=True, bidirectional=True, dropout=gdrop if layers > 1 else 0.0)
        self.head = nn.Linear(2 * hid, n_out)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x, lens):
        h = self.frame.feats(x)
        z = F.silu(self.proj(h))
        pk = pack_padded_sequence(z, lens.cpu(), batch_first=True, enforce_sorted=False)
        g, _ = self.gru(pk)
        g, _ = pad_packed_sequence(g, batch_first=True, total_length=x.shape[1])
        return self.frame.base(x, h) + self.head(g)


def huber2(r, d):
    a = r.abs()
    return torch.where(a < d, r * r, d * (2.0 * a - d))


def loss_fn(c, D, out, rows):
    d = c["huber"]; na = out.shape[1]
    if c["qpar"] == "log" and c["qloss"] == "raw":
        qh = torch.exp((out[:, 0] * D.sd_t + D.mu_t).clamp(max=LOGQMAX))
        rq = (qh - D.q_raw[rows]) / float(D.sd[0])
    elif c["qpar"] == "log":
        rq = out[:, 0] - D.Uq_log[rows]
    else:
        rq = out[:, 0] - D.U[rows, 0]
    loss = huber2(rq, d).mean() + huber2(out[:, 1] - D.U[rows, 1], d).mean()
    if c["aux_w"] > 0:
        ok = D.aux_ok[rows, :na - 2]
        loss = loss + c["aux_w"] * (huber2(out[:, 2:] - D.U[rows, 2:na], c.get("ahuber", 1.0)) * ok).sum() / ok.sum().clamp_min(1)
    return loss


def decode(c, D, out):
    o = out.double().cpu().numpy()
    q = np.exp(o[:, 0] * D.sd_t + D.mu_t) if c["qpar"] == "log" else o[:, 0] * D.sd[0] + D.mu[0]
    return q, o[:, 1] * D.sd[1] + D.mu[1]


def shot_batch(start, lens_all, sel, pad, dev):
    lens = lens_all[sel]
    ar = torch.arange(int(lens.max()), device=dev)[None, :]
    mask = ar < lens[:, None]
    idx = torch.where(mask, start[sel][:, None] + ar, torch.full_like(ar, pad))
    return idx, mask, lens


def n_out(c):
    return 2 + NAUX + c["npc"]


def in_dim(c, D):
    return D.D * (2 if c["ctx"] else 1) + (4 if c.get("phys", 0) else 0)


def make_frame(c, D, dev):
    return FrameNet(in_dim(c, D), c["width"], c["blocks"], n_out(c), c["drop"], D.colmask(c, dev)).to(dev)


def train_frame(c, seed, D, dev):
    torch.manual_seed(seed); np.random.seed(seed)
    net = make_frame(c, D, dev)
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["wd"])
    spe = D.N // c["batch"]; steps = c["epochs"] * spe
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=c["lr"], total_steps=steps, pct_start=0.05)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    last = 0.0
    X, _ = D.inputs(c)
    for ep in range(c["epochs"]):
        perm = torch.randperm(D.N, device=dev, generator=g)
        tot = torch.zeros((), device=dev)
        for b in range(spe):
            rows = perm[b * c["batch"]:(b + 1) * c["batch"]]
            x = X[rows]
            if c["innoise"] > 0:
                x = x + c["innoise"] * torch.randn(x.shape, device=dev, generator=g)
            loss = loss_fn(c, D, net(x), rows)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.detach()
        last = float(tot) / spe
    torch.backends.cuda.matmul.allow_tf32 = False
    return net, last


def train_seq(c, seed, D, dev, frame):
    torch.manual_seed(seed + 100000); np.random.seed(seed + 100000)
    net = SeqNet(frame, c["width"], n_out(c), c["emb"], c["hid"], c["layers"], c["gdrop"]).to(dev)
    X, _ = D.inputs(c)
    seqp = list(net.proj.parameters()) + list(net.gru.parameters()) + list(net.head.parameters())
    opt = torch.optim.AdamW([{"params": list(frame.parameters()), "lr": 0.0}, {"params": seqp, "lr": c["lr_gru"]}], weight_decay=c["wd"])
    S = len(D.start); sb = c["sbatch"]; spe = S // sb; total = c["gepochs"] * spe; fr = c["freeze"] * spe; warm = max(1, spe // 2)
    g = torch.Generator(device=dev); g.manual_seed(seed + 100000)
    torch.backends.cuda.matmul.allow_tf32 = True
    step, last = 0, 0.0
    for ep in range(c["gepochs"]):
        perm = torch.randperm(S, device=dev, generator=g)
        tot = torch.zeros((), device=dev)
        for b in range(spe):
            idx, mask, lens = shot_batch(D.start, D.len, perm[b * sb:(b + 1) * sb], D.N, dev)
            m = min(1.0, (step + 1) / warm) * 0.5 * (1.0 + math.cos(math.pi * step / total))
            opt.param_groups[0]["lr"] = 0.0 if step < fr else c["lr_trunk"] * m
            opt.param_groups[1]["lr"] = c["lr_gru"] * m
            out = net(X[idx], lens)[mask]
            loss = loss_fn(c, D, out, idx[mask])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); step += 1
            tot += loss.detach()
        last = float(tot) / spe
    torch.backends.cuda.matmul.allow_tf32 = False
    return net, last


@torch.no_grad()
def predict(c, D, dev, net):
    net.eval()
    _, Xe = D.inputs(c)
    out = torch.zeros((D.Ne, n_out(c)), device=dev)
    if isinstance(net, SeqNet):
        S = len(D.start_e)
        for a in range(0, S, 64):
            sel = torch.arange(a, min(S, a + 64), device=dev)
            idx, mask, lens = shot_batch(D.start_e, D.len_e, sel, D.Ne, dev)
            out[idx[mask]] = net(Xe[idx], lens)[mask]
    else:
        for a in range(0, D.Ne, 65536):
            out[a:a + 65536] = net(Xe[a:min(D.Ne, a + 65536)])
    return decode(c, D, out)


def run_task(name, seed, D, dev):
    c = get_cfg(name)
    tag = f"{name}_s{seed}"
    if (MEM / f"{tag}.npz").exists() and (WTS / f"{tag}.pt").exists():
        log(f"{tag}: exists, skip"); return
    t0 = time.time()
    rec = {"name": name, "seed": seed, "mode": MODE, "cfg": c}
    if c["kind"] == "mlp":
        net, last = train_frame(c, seed, D, dev)
    else:
        base = WTS / f"{c['base']}_s{seed}.pt"
        if not base.exists():
            run_task(c["base"], seed, D, dev)
        frame = make_frame(c, D, dev)
        frame.load_state_dict(torch.load(base, map_location=dev, weights_only=False)["state"])
        t0 = time.time()
        net, last = train_seq(c, seed, D, dev, frame)
    rec["train_loss_last_epoch"] = last; rec["sec"] = time.time() - t0
    q, b = predict(c, D, dev, net)
    np.savez(MEM / f"{tag}.npz", q95=q.astype(np.float32), betaN=b.astype(np.float32))
    st = D.stats()
    if c.get("phys", 0):
        st["phys"] = D.phys(dev)[2]
    torch.save({"state": net.state_dict(), "cfg": c, "stats": st, "seed": seed, "D": D.D, "n_out": n_out(c)}, WTS / f"{tag}.pt")
    if D.yq_e is not None:
        rec["hold_r2_q95"] = pooled_r2(D.yq_e, q); rec["hold_r2_betaN"] = pooled_r2(D.yb_e, b)
        rec["hold_r2_qb"] = 0.5 * (rec["hold_r2_q95"] + rec["hold_r2_betaN"])
    with open(MEM.parent / "nn_log.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"{tag}: " + json.dumps({k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items() if k not in ("cfg", "mode")}))


def main():
    MEM.mkdir(parents=True, exist_ok=True); WTS.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in os.environ["TASKS"].split(",") if t]
    tasks = tasks[int(os.environ.get("WORKER", "0"))::int(os.environ.get("NWORK", "1"))]
    dev = torch.device("cuda")
    t0 = time.time()
    D = Data(dev)
    for t in tasks:
        name, seed = t.split(":")
        run_task(name, int(seed), D, dev)
    log(f"worker done: {len(tasks)} tasks in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
