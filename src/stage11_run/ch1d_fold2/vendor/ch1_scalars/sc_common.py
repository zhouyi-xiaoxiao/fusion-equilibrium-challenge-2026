#!/usr/bin/env python3
"""Stage 11 / ch1_scalars — shared paths and helpers (q95 / betaN heads on FEATURE-V2.1)."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

SCR = Path(os.environ["SCR"])
FEAT = Path(os.environ.get("FEAT", str(SCR / "stage11_ch1/feat_v21")))
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/scalars")))
INCUMBENT = Path(os.environ.get("INCUMBENT", str(SCR / "stage10_ch1_cons/preds_d3d_p80_bag20_hbag.npz")))
FULL = os.environ.get("FULL", "0") == "1"          # 1 = train on all 7,041 shots (train + holdout), predict the public test
MODE = "full" if FULL else "hold"
MEM = OUT / ("full" if FULL else "members")        # per-member outputs (holdout frame order, or test row order when FULL)
WTS = OUT / "weights" / MODE
REGIMES = ["ramp-up", "flat-top", "ramp-down"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def names():
    return (FEAT / "names.txt").read_text().split()


def pooled_r2(y, p):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def regime_index(split="holdout"):
    """0 ramp-up (t before |Ip| first reaches 90 % of its shot maximum), 2 ramp-down (t after it last holds 90 %), 1 flat-top."""
    n = names()
    Xr = np.load(FEAT / split / "X_raw.npy", mmap_mode="r")
    up = np.asarray(Xr[:, n.index("time_rampup_flag")]) > 0.5
    dn = np.asarray(Xr[:, n.index("time_rampdown_flag")]) > 0.5
    r = np.ones(len(up), dtype=np.int64)
    r[up] = 0
    r[dn & ~up] = 2
    return r


PHYS_COLS = ("v1_plasma_current", "v1_bcoil", "phys_qlike_proxy")


def phys_raw(split_dir):
    """The three raw input columns the phys extra features are built from (mmap read of X_raw.npy)."""
    n = names()
    Xr = np.load(Path(split_dir) / "X_raw.npy", mmap_mode="r")
    return np.stack([np.asarray(Xr[:, n.index(c)], np.float64) for c in PHYS_COLS], 1)


def phys_extra(R, st=None):
    """log|Ip|, log|B_T coil|, log(|B_T|/|Ip|) (the q_cyl ~ a^2 B_T/(R Ip) scaling without geometry), log(qlike proxy).
    Floors and standardisation stats come from the training rows (st=None) and are stored with the weights."""
    ip, bt, ql = np.abs(R[:, 0]), np.abs(R[:, 1]), R[:, 2]
    if st is None:
        st = {"fi": float(0.05 * np.nanpercentile(ip, 99)), "fb": float(0.05 * np.nanpercentile(bt, 99))}
    li, lb = np.log(np.maximum(np.nan_to_num(ip), st["fi"])), np.log(np.maximum(np.nan_to_num(bt), st["fb"]))
    lq = np.where(np.isfinite(ql) & (ql > 0), np.log(np.maximum(np.where(np.isfinite(ql), ql, 1.0), 1e-3)), np.nan)
    E = np.column_stack([li, lb, lb - li, lq])
    if "mu" not in st:
        st["mu"] = np.nanmean(E, 0).tolist(); st["sd"] = np.maximum(np.nanstd(E, 0), 1e-9).tolist()
    E = np.clip(np.nan_to_num((E - np.asarray(st["mu"])) / np.asarray(st["sd"])), -10, 10).astype(np.float32)
    return E, st


def shot_of_frame(offsets):
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def diag(y, p, reg, sof):
    """Pooled R2 plus where the squared error sits: by regime, and shot-mean bias vs within-shot."""
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    e = p - y
    sst = ((y - y.mean()) ** 2).sum()
    out = {"r2": 1.0 - (e ** 2).sum() / sst, "rmse": float(np.sqrt((e ** 2).mean())), "regimes": {}}
    for k, nm in enumerate(REGIMES):
        m = reg == k
        out["regimes"][nm] = {"n": int(m.sum()), "rmse": float(np.sqrt((e[m] ** 2).mean())), "bias": float(e[m].mean()),
                              "r2_loss": float((e[m] ** 2).sum() / sst), "share": float((e[m] ** 2).sum() / (e ** 2).sum()),
                              "r2_within_regime": float(1.0 - (e[m] ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum())}
    ns = np.bincount(sof)
    mb = np.bincount(sof, weights=e) / np.maximum(ns, 1)
    out["shot_bias_share"] = float((ns * mb ** 2).sum() / (e ** 2).sum())
    return out
