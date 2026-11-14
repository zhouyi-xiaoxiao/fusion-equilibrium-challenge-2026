#!/usr/bin/env python3
"""
Stage 11 / ch1_metric — loader + loss functions for the Jacobian ("metric") loss on PCA coefficients.

Coefficients c are in the stage6_H bases (pca_80.npz / pca_120.npz: psi = mean_ + c @ comps_), exactly the
y_C80 / y_C120 targets of feat_v2. e = c_pred - c_true. All losses are PER FRAME in official score units:
mean over frames ~ 0.55*(1-R2psi) + 0.20*(1-Consistency) attributable to the model error (truncation floor excluded).

  G  global, fixed target transform (works with any plain-MSE trainer, e.g. sklearn):
        z = c @ L            train plain SUM-over-dims squared error on z (no per-dimension re-standardising;
        c = z @ Linv         one global scale factor is fine)                       loss = |z_pred - z_true|^2
  N  amplitude-normalised (needs per-sample weights, train-time only):
        loss = a_psi |e|^2 + |e @ Ln|^2 / A^2,   A = |psi_axis - psi_boundary| of the TRUE map (amp_train.npy)
  PF per-frame Gauss-Newton (K=80 only; needs JW_train_80.npy, 3.4 GB):
        loss = a_psi |e|^2 + sum_j rw_j (JW[i,j,:] . e)^2        rw = optional per-scalar weights (default 1)
  hybrid (recommended starting point): PF rows for R_axis, Z_axis, kappa, volume, li  +  N-metric for
        tri_top, tri_bot (their per-frame linearisation is unreliable):
        loss = a_psi |e|^2 + sum_j pf_w[j] (JW_ij . e)^2 + |e @ Ln_sub|^2 / A^2,  Ln_sub = Metric.factor_N(n_w)

Usage (torch):
    from metric_loss import Metric, loss_g, loss_n, loss_pf
    m = Metric(80); L = torch.tensor(m.L, dtype=torch.float32, device=dev)
    loss = loss_g(c_hat, c_true, L)
"""
from __future__ import annotations
import os

import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_DIR = Path(os.environ["SCR"] + "/stage11_ch1/metric")
NAMES = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
PF_W_HYBRID = np.array([1, 1, 1, 0, 0, 1, 1], dtype=np.float64)
N_W_HYBRID = np.array([0, 0, 0, 1, 1, 0, 0], dtype=np.float64)


class Metric:
    def __init__(self, K=80, root=DEFAULT_DIR, lcfs=False):
        self.K, self.root = K, Path(root)
        self.meta = json.load(open(self.root / f"metric_{K}.json"))
        self.a_psi = float(self.meta["a_psi"])
        sfx = "_lcfs" if lcfs else ""
        self.M = np.load(self.root / f"M_{K}.npy")
        self.L = np.load(self.root / f"L_{K}{sfx}.npy")              # z = c @ L
        self.Linv = np.load(self.root / f"Linv_{K}{sfx}.npy")        # c = z @ Linv
        self.Ln = np.load(self.root / f"Ln_{K}.npy")                 # |e @ Ln|^2 = e^T Mn_cons e
        self._parts = np.load(self.root / f"M_parts_{K}.npz")

    # ---- family G
    def to_z(self, c):
        return np.asarray(c, np.float64) @ self.L

    def from_z(self, z):
        return np.asarray(z, np.float64) @ self.Linv

    # ---- family N with a subset / re-weighting of the seven scalars
    def factor_N(self, scalar_w=None):
        p = self._parts
        w = np.ones(7) if scalar_w is None else np.asarray(scalar_w, np.float64)
        Mn = np.einsum("j,jkl->kl", w * p["w_j"] * p["alphan"], p["EJtJn"])
        Mn = 0.5 * (Mn + Mn.T)
        ev, V = np.linalg.eigh(Mn)
        o = np.argsort(ev)[::-1]
        return V[:, o] * np.sqrt(np.maximum(ev[o], 0.0))

    def amp(self, split="train"):
        a = np.load(self.root / f"amp_{split}.npy").astype(np.float64)
        return np.where(np.isfinite(a) & (a > 1e-3), a, np.nanmedian(a))

    def jw(self, split="train", mmap=True):
        return np.load(self.root / f"JW_{split}_{self.K}.npy", mmap_mode="r" if mmap else None)

    # ---- numpy evaluators (per-frame loss vectors)
    def loss_G(self, c_pred, c_true):
        e = np.asarray(c_pred, np.float64) - np.asarray(c_true, np.float64)
        return ((e @ self.L) ** 2).sum(1)

    def loss_N(self, c_pred, c_true, amp, Ln=None):
        e = np.asarray(c_pred, np.float64) - np.asarray(c_true, np.float64)
        Ln = self.Ln if Ln is None else Ln
        return self.a_psi * (e ** 2).sum(1) + ((e @ Ln) ** 2).sum(1) / np.asarray(amp, np.float64) ** 2

    def loss_PF(self, c_pred, c_true, JW, scalar_w=None):
        e = np.asarray(c_pred, np.float64) - np.asarray(c_true, np.float64)
        s = np.einsum("njk,nk->nj", np.asarray(JW, np.float64), e) ** 2
        if scalar_w is not None:
            s = s * np.asarray(scalar_w, np.float64)[None, :]
        return self.a_psi * (e ** 2).sum(1) + s.sum(1)


# ---- torch losses (batch means). Tensors: c_hat, c (B,K); L (K,K); Ln (K,r); amp (B,); JW (B,7,K); rw (7,)
def loss_g(c_hat, c, L):
    return (((c_hat - c) @ L) ** 2).sum(1).mean()


def loss_n(c_hat, c, amp, Ln, a_psi):
    e = c_hat - c
    return (a_psi * (e ** 2).sum(1) + ((e @ Ln) ** 2).sum(1) / amp ** 2).mean()


def loss_pf(c_hat, c, JW, a_psi, rw=None):
    import torch
    e = c_hat - c
    s = torch.einsum("bjk,bk->bj", JW, e) ** 2
    if rw is not None:
        s = s * rw[None, :]
    return (a_psi * (e ** 2).sum(1) + s.sum(1)).mean()


def loss_hybrid(c_hat, c, JW, amp, Ln_sub, a_psi, pf_w):
    import torch
    e = c_hat - c
    s = (torch.einsum("bjk,bk->bj", JW, e) ** 2 * pf_w[None, :]).sum(1)
    return (a_psi * (e ** 2).sum(1) + s + ((e @ Ln_sub) ** 2).sum(1) / amp ** 2).mean()


def selftest(root=DEFAULT_DIR):
    """Numeric self-check on the incumbent bag20 holdout errors (prints per-frame-mean losses)."""
    root = Path(root)
    scr = root.parents[1]
    feat, hc = scr / "stage11_ch1/feat_v2", scr / "stage6_H"
    for K in (80, 120):
        m = Metric(K, root)
        y = np.load(feat / f"holdout/y_C{K}.npy").astype(np.float64)
        tag = ["p80_h512x256x128", "p80_h1024x512x256"] if K == 80 else ["p120_h1024x512x256"]
        cp = np.mean([np.load(hc / f"coef_{t}_s{s}.npy") for t in tag for s in range(42, 52)], 0).astype(np.float64)
        e = cp - y
        q = np.einsum("nk,kl,nl->n", e, m.M, e)
        assert np.allclose(m.loss_G(cp, y), q, rtol=1e-9), "L factor mismatch"
        assert np.allclose(m.from_z(m.to_z(y[:100])), y[:100], atol=1e-8), "inverse mismatch"
        w = np.linalg.eigvalsh(m.M)
        amp = m.amp("holdout")
        out = {"K": K, "min_eig": float(w.min()), "cond": float(w.max() / w.min()), "psi_term": float(m.a_psi * (e ** 2).sum(1).mean()),
               "loss_G": float(q.mean()), "loss_N": float(m.loss_N(cp, y, amp).mean())}
        if K == 80 and (root / "JW_holdout_80.npy").exists():
            JW = m.jw("holdout")
            out["loss_PF"] = float(m.loss_PF(cp, y, JW).mean())
            out["loss_hybrid"] = float((m.loss_PF(cp, y, JW, PF_W_HYBRID) + m.loss_N(cp, y, amp, m.factor_N(N_W_HYBRID)) - m.a_psi * (e ** 2).sum(1)).mean())
            try:
                import torch
                t = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64)  # noqa: E731
                n = 4096
                a1 = float(loss_pf(t(cp[:n]), t(y[:n]), t(JW[:n]), m.a_psi)); a2 = float(m.loss_PF(cp[:n], y[:n], JW[:n]).mean())
                b1 = float(loss_g(t(cp[:n]), t(y[:n]), t(m.L))); b2 = float(m.loss_G(cp[:n], y[:n]).mean())
                c1 = float(loss_n(t(cp[:n]), t(y[:n]), t(amp[:n]), t(m.Ln), m.a_psi)); c2 = float(m.loss_N(cp[:n], y[:n], amp[:n]).mean())
                out["torch_vs_numpy_max_rel"] = float(max(abs(a1 - a2) / a2, abs(b1 - b2) / b2, abs(c1 - c2) / c2))
            except Exception as ex:  # noqa: BLE001
                out["torch"] = f"skipped: {ex}"
        print("SELFTEST", json.dumps(out))
    print("SELFTEST OK (official bag20 holdout: 0.55*(1-R2psi)+0.20*(1-Cons) = 0.55*0.0002 + 0.20*0.0269 = 0.00549 incl. truncation floor 0.20*0.0017)")


if __name__ == "__main__":
    selftest(Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DIR)
