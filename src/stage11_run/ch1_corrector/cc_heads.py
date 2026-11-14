#!/usr/bin/env python3
"""Step 2/3 — scalar heads on feature-v2.1 (GPU train, CPU/GPU predict).

Targets (11): the 7 scorer-derived scalars f_j(psi_true) + the 4 LCFS extrema (y_aux cols 3..13; never EFIT columns).
  direct mode : x -> f_true
  innov mode  : [x, standardised f(psi_base), fail flag] -> d = f_true - f(psi_base); on TRAIN psi_base is the
                OUT-OF-FOLD base map, at application it is the map being corrected. Head/skip are zero-initialised
                (start = "no correction"); targets clipped at +-6 RMS.
Models: residual MLP (768 x 4 blocks, LayerNorm -> Linear -> SiLU -> Dropout 0.03 -> Linear + skip, linear input
skip, AdamW 5e-4, wd 1e-4, 2 warm-up epochs + cosine, 30 epochs, batch 4096, grad-clip 1, EMA 0.995, bf16 forward)
then a zero-initialised bi-GRU residual over whole shots on the MLP embedding (256-d projection + log-gap inputs,
192/direction, trunk frozen 2 epochs then lr 1e-4, GRU lr 3e-4, EMA 0.99).
Early stopping on a 5 % shot-disjoint slice of TRAIN (fixed, seed 0). The holdout is only scored.
--full (final refit, MLP only): all shots of $CC_FEAT/train, exactly --epochs epochs (cosine over them), final EMA kept.

  cc_heads.py train --tag direct --seeds 0,1,2
  cc_heads.py train --tag innov_gv21 --seeds 0,1,2 --base-train f11_train_gv21.npy --apply gv21:holdout:f11_holdout_gv21.npy
  cc_heads.py train --tag stack_gv21 --stack --no-gru --base-train f11_train_gv21.npy --apply gv21:holdout:f11_holdout_gv21.npy
        (stack: x + f(psi_base) as inputs, target = f_true itself, no residual parametrisation / clipping)
  cc_heads.py predict --tag innov_gv21 --seeds 0,1,2 --feat-dir <dir with X.npy offsets.npy times.npy> --base f11.npy --out d.npy
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import cc_common as cc
from cc_common import log

B_MLP, E_MLP, LR_MLP, EMA_MLP, PAT_MLP = 4096, 30, 5e-4, 0.995, 6
B_SEQ, E_SEQ, EMA_SEQ, PAT_SEQ = 48, 16, 0.99, 4
W = 768


class Block(nn.Module):
    def __init__(self, w, p=0.03):
        super().__init__()
        self.n = nn.LayerNorm(w); self.a = nn.Linear(w, w); self.d = nn.Dropout(p); self.b = nn.Linear(w, w)

    def forward(self, h):
        return h + self.b(self.d(nn.functional.silu(self.a(self.n(h)))))


class MLP(nn.Module):
    def __init__(self, din, nout, w=W, nb=4):
        super().__init__()
        self.inp = nn.Linear(din, w)
        self.blocks = nn.ModuleList([Block(w) for _ in range(nb)])
        self.norm = nn.LayerNorm(w)
        self.head = nn.Linear(w, nout)
        self.skip = nn.Linear(din, nout)

    def forward(self, x):
        h = self.inp(x)
        for b in self.blocks:
            h = b(h)
        z = self.norm(h)
        return self.head(z) + self.skip(x), z


class Seq(nn.Module):
    def __init__(self, trunk, nout, w=W, demb=256, hid=192):
        super().__init__()
        self.trunk = trunk
        self.proj = nn.Linear(w, demb)
        self.gru = nn.GRU(demb + 2, hid, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(0.05)
        self.out = nn.Linear(2 * hid, nout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, lens, gaps):
        y0, z = self.trunk(x)
        u = torch.cat([self.proj(z), gaps], -1)
        pk = nn.utils.rnn.pack_padded_sequence(u, lens.cpu(), batch_first=True, enforce_sorted=False)
        o, _ = self.gru(pk)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True, total_length=x.shape[1])
        return y0 + self.out(self.drop(o))


def gaps_of(times, off):
    g = np.zeros((len(times), 2), np.float32)
    for s in range(len(off) - 1):
        a, b = off[s], off[s + 1]
        if b - a > 1:
            dt = np.diff(times[a:b].astype(np.float64))
            g[a + 1:b, 0] = np.log1p(np.clip(dt, 0, None) / 20.0)
            g[a:b - 1, 1] = np.log1p(np.clip(dt, 0, None) / 20.0)
    return g


def load_feat(d, base=None, stats=None):
    d = Path(d)
    X = np.nan_to_num(np.load(d / "X.npy")).astype(np.float32)
    off = np.load(d / "offsets.npy")
    out = {"off": off, "gaps": gaps_of(np.load(d / "times.npy"), off), "fb": None}
    if base is not None:
        fb = np.load(base).astype(np.float64)
        assert fb.shape == (X.shape[0], 11), (fb.shape, X.shape)
        out["fb"] = fb
        if stats is not None and stats.get("extra", stats.get("innov", False)):
            X = np.concatenate([X, extra_of(fb, stats)], 1)
    out["X"] = X
    return out


def extra_of(fb, stats):
    e = (fb - np.asarray(stats["ymu"])) / np.asarray(stats["ysd"])
    bad = ~np.isfinite(e)
    e = np.where(bad, 0.0, np.clip(e, -10, 10))
    return np.concatenate([e, bad.any(1, keepdims=True)], 1).astype(np.float32)


def mse(out, t, m):
    return (((out - t) ** 2) * m).sum() / m.sum().clamp_min(1.0)


class Runner:
    def __init__(self, dev):
        self.dev = dev

    # ------------------------------------------------------------------ frame MLP
    def train_mlp(self, Xg, Tg, Mg, tr, va, nout, seed, zero_init, epochs):
        dev = self.dev
        torch.manual_seed(seed)
        net = MLP(Xg.shape[1], nout).to(dev)
        if zero_init:
            for l in (net.head, net.skip):
                nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        ema = copy.deepcopy(net)
        for p in ema.parameters():
            p.requires_grad_(False)
        opt = torch.optim.AdamW(net.parameters(), lr=LR_MLP, weight_decay=1e-4)
        n = tr.numel(); spe = n // B_MLP; total = epochs * spe; warm = min(2 * spe, total // 2)
        g = torch.Generator(device=dev); g.manual_seed(seed)
        best = (float("inf"), None, -1, None); step = 0
        for ep in range(epochs):
            net.train()
            perm = tr[torch.randperm(n, device=dev, generator=g)]
            for b in range(spe):
                idx = perm[b * B_MLP:(b + 1) * B_MLP]
                lr = LR_MLP * (step + 1) / warm if step < warm else 1e-5 + 0.5 * (LR_MLP - 1e-5) * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
                for gr in opt.param_groups:
                    gr["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out, _ = net(Xg[idx])
                loss = mse(out.float(), Tg[idx], Mg[idx])
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), net.parameters()):
                        pe.lerp_(p, 1 - EMA_MLP)
            if va.numel() == 0:                           # --full: fixed epoch count, keep the final EMA
                best = (float("nan"), None, ep, np.full(nout, np.nan))
                continue
            v = self.eval_mlp(ema, Xg, Tg, Mg, va)
            if float(v.mean()) < best[0]:
                best = (float(v.mean()), copy.deepcopy(ema.state_dict()), ep, v.cpu().numpy())
            elif ep - best[2] >= PAT_MLP:
                break
        if best[1] is not None:
            ema.load_state_dict(best[1])
        return ema, best

    @torch.no_grad()
    def eval_mlp(self, net, Xg, Tg, Mg, idx):
        net.eval()
        se = torch.zeros(Tg.shape[1], device=self.dev); cnt = torch.zeros_like(se)
        for i in range(0, idx.numel(), 65536):
            j = idx[i:i + 65536]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out, _ = net(Xg[j])
            se += (((out.float() - Tg[j]) ** 2) * Mg[j]).sum(0); cnt += Mg[j].sum(0)
        return se / cnt.clamp_min(1.0)

    @torch.no_grad()
    def pred_mlp(self, net, X):
        net.eval()
        outs = []
        for i in range(0, X.shape[0], 65536):
            if self.dev.type == "cuda":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    o, _ = net(X[i:i + 65536])
            else:
                o, _ = net(X[i:i + 65536])
            outs.append(o.float())
        return torch.cat(outs)

    # ------------------------------------------------------------------ shot bi-GRU
    def batch(self, Xg, Gg, off_t, sh, Tg=None, Mg=None):
        st = off_t[sh]; lens = off_t[sh + 1] - st
        ar = torch.arange(int(lens.max()), device=self.dev)
        valid = ar[None, :] < lens[:, None]
        idx = torch.where(valid, st[:, None] + ar[None, :], torch.zeros_like(ar)[None, :])
        o = [Xg[idx], lens, Gg[idx], valid, idx]
        if Tg is not None:
            o += [Tg[idx], Mg[idx] * valid[..., None]]
        return o

    def train_seq(self, mlp, Xg, Tg, Mg, Gg, off_t, sh_tr, sh_va, nout, seed, epochs):
        dev = self.dev
        torch.manual_seed(10_000 + seed)
        seq = Seq(copy.deepcopy(mlp), nout).to(dev)
        for p in seq.parameters():
            p.requires_grad_(True)
        ema = copy.deepcopy(seq)
        for p in ema.parameters():
            p.requires_grad_(False)
        new = [p for n, p in seq.named_parameters() if not n.startswith("trunk.")]
        opt = torch.optim.AdamW([{"params": list(seq.trunk.parameters()), "lr": 0.0}, {"params": new, "lr": 3e-4}], weight_decay=1e-4)
        ns = sh_tr.numel(); spe = math.ceil(ns / B_SEQ); total = epochs * spe; step = 0
        g = torch.Generator(device=dev); g.manual_seed(seed)
        best = (float("inf"), None, -1, None)
        for ep in range(epochs):
            seq.train()
            order = sh_tr[torch.randperm(ns, device=dev, generator=g)]
            for b in range(spe):
                sh = order[b * B_SEQ:(b + 1) * B_SEQ]
                x, lens, gaps, _, _, t, m = self.batch(Xg, Gg, off_t, sh, Tg, Mg)
                cf = 0.5 * (1 + math.cos(math.pi * step / total))
                opt.param_groups[0]["lr"] = 0.0 if ep < 2 else 1e-5 + (1e-4 - 1e-5) * cf
                opt.param_groups[1]["lr"] = 1e-5 + (3e-4 - 1e-5) * cf
                loss = mse(seq(x, lens, gaps), t, m)
                opt.zero_grad(set_to_none=True); loss.backward()
                nn.utils.clip_grad_norm_(seq.parameters(), 1.0); opt.step(); step += 1
                with torch.no_grad():
                    for pe, p in zip(ema.parameters(), seq.parameters()):
                        pe.lerp_(p, 1 - EMA_SEQ)
            v = self.eval_seq(ema, Xg, Tg, Mg, Gg, off_t, sh_va)
            if float(v.mean()) < best[0]:
                best = (float(v.mean()), copy.deepcopy(ema.state_dict()), ep, v.cpu().numpy())
            elif ep - best[2] >= PAT_SEQ:
                break
        ema.load_state_dict(best[1])
        return ema, best

    @torch.no_grad()
    def eval_seq(self, net, Xg, Tg, Mg, Gg, off_t, shots):
        net.eval()
        se = torch.zeros(Tg.shape[1], device=self.dev); cnt = torch.zeros_like(se)
        for i in range(0, shots.numel(), 64):
            x, lens, gaps, _, _, t, m = self.batch(Xg, Gg, off_t, shots[i:i + 64], Tg, Mg)
            o = net(x, lens, gaps)
            se += (((o - t) ** 2) * m).sum((0, 1)); cnt += m.sum((0, 1))
        return se / cnt.clamp_min(1.0)

    @torch.no_grad()
    def pred_seq(self, net, X, G, off):
        net.eval()
        off_t = torch.as_tensor(np.asarray(off), device=self.dev, dtype=torch.long)
        out = torch.zeros((X.shape[0], net.out.out_features), device=self.dev)
        S = len(off) - 1
        for i in range(0, S, 64):
            sh = torch.arange(i, min(S, i + 64), device=self.dev)
            x, lens, gaps, valid, idx = self.batch(X, G, off_t, sh)
            o = net(x, lens, gaps)
            out[idx[valid]] = o[valid]
        return out


def r2_table(y, p, base=None):
    """Pooled R2 per target; with `base` also the innovation R2 = 1 - sum(d-dhat)^2/sum d^2."""
    r = []
    for j in range(y.shape[1]):
        m = np.isfinite(y[:, j]) & np.isfinite(p[:, j])
        if base is not None:
            m &= np.isfinite(base[:, j])
        r.append(1.0 - ((y[m, j] - p[m, j]) ** 2).sum() / ((y[m, j] - y[m, j].mean()) ** 2).sum())
    out = {"r2": r}
    if base is not None:
        iv, rb = [], []
        for j in range(y.shape[1]):
            m = np.isfinite(y[:, j]) & np.isfinite(p[:, j]) & np.isfinite(base[:, j])
            d = y[m, j] - base[m, j]
            iv.append(1.0 - ((y[m, j] - p[m, j]) ** 2).sum() / (d ** 2).sum())
            rb.append(1.0 - (d ** 2).sum() / ((y[m, j] - y[m, j].mean()) ** 2).sum())
        out.update(innov_r2=iv, r2_base=rb)
    return out


def cmd_train(a):
    dev = torch.device("cuda")
    run = Runner(dev)
    t0 = time.time()
    extra = a.base_train is not None
    innov = extra and not a.stack                      # stack: f_base only as extra inputs, target = f_true itself
    seeds = [int(s) for s in a.seeds.split(",")]
    models = ["mlp"] if a.no_gru else ["mlp", "gru"]
    y = np.load(cc.FEAT / "train/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]
    off = np.load(cc.FEAT / "train/offsets.npy")
    S = len(off) - 1
    perm = np.random.default_rng(0).permutation(S)
    va_s = np.sort(perm[: max(1, int(0.05 * S))]); tr_s = np.sort(perm[max(1, int(0.05 * S)):])
    if a.full:                                          # final refit: every train shot, fixed --epochs, no early stop
        tr_s, va_s = np.arange(S), np.zeros(0, np.int64)
    if a.smoke:
        tr_s = tr_s[:300]; va_s = va_s[:40]
    is_tr = np.zeros(S, bool); is_tr[tr_s] = True
    fr_tr = np.repeat(is_tr, np.diff(off))
    is_va = np.zeros(S, bool); is_va[va_s] = True
    fr_va = np.repeat(is_va, np.diff(off))
    stats = {"ymu": np.nanmean(y[fr_tr], 0).tolist(), "ysd": np.nanstd(y[fr_tr], 0).tolist(), "innov": innov, "extra": extra}
    data = load_feat(cc.FEAT / "train", cc.OUT / a.base_train if extra else None, stats)
    if innov:
        d = y - data["fb"]
        dsc = np.sqrt(np.nanmean(d[fr_tr] ** 2, 0))
        stats["dsc"] = dsc.tolist()
        T = np.clip(d / dsc, -6, 6)
        scale, shift = dsc, np.zeros(11)
    else:
        T = (y - np.asarray(stats["ymu"])) / np.asarray(stats["ysd"])
        scale, shift = np.asarray(stats["ysd"]), np.asarray(stats["ymu"])
    M = np.isfinite(T)
    Xg = torch.from_numpy(data["X"]).to(dev)
    Tg = torch.from_numpy(np.where(M, T, 0.0).astype(np.float32)).to(dev)
    Mg = torch.from_numpy(M.astype(np.float32)).to(dev)
    Gg = torch.from_numpy(data["gaps"]).to(dev)
    off_t = torch.as_tensor(off, device=dev, dtype=torch.long)
    tr = torch.from_numpy(np.nonzero(fr_tr)[0]).to(dev); va = torch.from_numpy(np.nonzero(fr_va)[0]).to(dev)
    sh_tr = torch.from_numpy(tr_s).to(dev); sh_va = torch.from_numpy(va_s).to(dev)
    log(f"[{a.tag}] innov={innov} extra={extra} X {tuple(Xg.shape)} train shots {len(tr_s)} val shots {len(va_s)} load {time.time()-t0:.0f}s")

    apps = {}                                           # name -> (split, feature dict)
    if extra:
        for spec in (a.apply or []):
            nm, split, path = spec.split(":")
            apps[nm] = (split, load_feat(cc.FEAT / split, cc.OUT / path, stats))
    else:
        for split in ("holdout", "test"):
            apps[split] = (split, load_feat(cc.FEAT / split))
    appt = {nm: (torch.from_numpy(dd["X"]).to(dev), torch.from_numpy(dd["gaps"]).to(dev), dd["off"]) for nm, (_, dd) in apps.items()}
    apps["train"] = ("train", data); appt["train"] = (Xg, Gg, off)        # only the 5 % val-slice shots are honest
    yh = np.load(cc.FEAT / "holdout/y_aux.npy").astype(np.float64)[:, cc.AUX_COLS]

    preds = {nm: {m: [] for m in models} for nm in apps}
    rec = {"tag": a.tag, "stats": stats, "seeds": seeds, "per_seed": {}, "val_shots": va_s.tolist()}
    e_mlp, e_seq = (2, 2) if a.smoke else (a.epochs, E_SEQ)
    for s in seeds:
        t1 = time.time()
        mlp, bm = run.train_mlp(Xg, Tg, Mg, tr, va, 11, s, innov, e_mlp)
        t2 = time.time()
        ck = {"mlp": mlp.state_dict(), "stats": stats, "din": Xg.shape[1]}
        ps = {"mlp_best_epoch": bm[2], "mlp_val": bm[3].tolist(), "sec_mlp": t2 - t1}
        msg = f"[{a.tag}] seed {s}: mlp val {bm[0]:.5f} @ep{bm[2]} ({t2-t1:.0f}s)"
        if "gru" in models:
            seq, bs = run.train_seq(mlp, Xg, Tg, Mg, Gg, off_t, sh_tr, sh_va, 11, s, e_seq)
            ck["seq"] = seq.state_dict()
            ps.update(seq_best_epoch=bs[2], seq_val=bs[3].tolist(), sec_seq=time.time() - t2)
            msg += f" | gru val {bs[0]:.5f} @ep{bs[2]} ({time.time()-t2:.0f}s)"
        torch.save(ck, cc.OUT / f"ck_{a.tag}_s{s}.pt")
        rec["per_seed"][s] = ps
        log(msg)
        for nm, (Xa, Ga, offa) in appt.items():
            preds[nm]["mlp"].append((run.pred_mlp(mlp, Xa).double().cpu().numpy()) * scale + shift)
            if "gru" in models:
                preds[nm]["gru"].append((run.pred_seq(seq, Xa, Ga, offa).double().cpu().numpy()) * scale + shift)
    rec["val_mse_std_units"] = {"mlp": np.mean([rec["per_seed"][s]["mlp_val"] for s in seeds], 0).tolist()}
    rec["holdout"] = {}
    for nm, (split, dd) in apps.items():
        for model in models:
            bag = np.mean(preds[nm][model], 0)
            np.save(cc.OUT / f"hp_{a.tag}_{model}_{nm}.npy", bag.astype(np.float32) if split == "train" else bag)
            if split != "holdout":
                continue
            fb = dd["fb"]
            full = (fb + bag) if innov else bag
            tb = r2_table(yh, full, fb)
            tb["r2_seed0"] = r2_table(yh, (fb + preds[nm][model][0]) if innov else preds[nm][model][0], fb)["r2"]
            rec["holdout"][f"{nm}/{model}"] = tb
            line = f"[{a.tag}] HOLDOUT {nm}/{model} bag{len(seeds)} R2: " + " ".join(f"{n} {v:.4f}" for n, v in zip(cc.NAMES11, tb["r2"]))
            line += "\n      seed0 R2: " + " ".join(f"{v:.4f}" for v in tb["r2_seed0"])
            if fb is not None:
                line += "\n      base R2: " + " ".join(f"{v:.4f}" for v in tb["r2_base"]) + "\n      innovation R2: " + " ".join(f"{v:+.3f}" for v in tb["innov_r2"])
            print(line, flush=True)
    rec["gpu_seconds"] = time.time() - t0
    json.dump(rec, open(cc.OUT / f"heads_{a.tag}.json", "w"), indent=1)
    log(f"[{a.tag}] done in {time.time()-t0:.0f}s")


def cmd_predict(a):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = Runner(dev)
    seeds = [int(s) for s in a.seeds.split(",")]
    outs = []
    data = None
    for s in seeds:
        ck = torch.load(cc.OUT / f"ck_{a.tag}_s{s}.pt", map_location=dev)
        st = ck["stats"]
        if data is None:
            data = load_feat(a.feat_dir, a.base, st)
            Xa = torch.from_numpy(data["X"]).to(dev); Ga = torch.from_numpy(data["gaps"]).to(dev)
        mlp = MLP(ck["din"], 11).to(dev)
        if a.model == "mlp":
            mlp.load_state_dict(ck["mlp"]); o = run.pred_mlp(mlp, Xa)
        else:
            seq = Seq(mlp, 11).to(dev); seq.load_state_dict(ck["seq"]); o = run.pred_seq(seq, Xa, Ga, data["off"])
        o = o.double().cpu().numpy()
        outs.append(o * np.asarray(st["dsc"]) if st["innov"] else o * np.asarray(st["ysd"]) + np.asarray(st["ymu"]))
    np.save(a.out, np.mean(outs, 0))
    log(f"wrote {a.out} ({a.tag}/{a.model}, {len(seeds)} seeds, device {dev})")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--tag", required=True); t.add_argument("--seeds", default="0,1,2")
    t.add_argument("--base-train", default=None); t.add_argument("--apply", action="append")
    t.add_argument("--smoke", action="store_true"); t.add_argument("--stack", action="store_true")
    t.add_argument("--no-gru", action="store_true"); t.add_argument("--epochs", type=int, default=E_MLP)
    t.add_argument("--full", action="store_true", help="train on ALL shots of $CC_FEAT/train for exactly --epochs (MLP only)")
    p = sub.add_parser("predict")
    p.add_argument("--tag", required=True); p.add_argument("--seeds", default="0,1,2"); p.add_argument("--model", default="gru")
    p.add_argument("--feat-dir", required=True); p.add_argument("--base", default=None); p.add_argument("--out", required=True)
    a = ap.parse_args()
    (cmd_train if a.cmd == "train" else cmd_predict)(a)


if __name__ == "__main__":
    main()
