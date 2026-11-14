#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — torch pieces: member loading (as ch1d_liloss/li_ft.py) and the contour-replay surrogate.

Replay (notes/STAGE11F_CH1_PRO_FOLLOWUP2.md §B 1(b)), everything in phi = -psi and float64:
  * the predicted LCFS polygon of the scorer is the skimage marching-squares contour at the last valid bisection level; every
    polygon vertex sits on one grid edge (a, b):  u = (l - phi_a) / (phi_b - phi_a),  p = (1 - u) p_a + u p_b  (grid_to_physical
    is linear inside a cell, so this is exact);
  * the extracted level is the value of ONE grid node (the critical node whose crossing changes the marching-squares topology)
    plus the frozen bisection offset:  l(c) = phi_crit(c) + (l0 - phi_crit(c0));
  * the polygon is resampled to 512 points uniformly in arc length exactly as contour.resample_closed (start = first skimage
    vertex, same orientation, np.linspace(0, total, 512, endpoint=False) + linear interpolation);
  * distance = the two directed nearest-neighbour distance sets between the 512 predicted and the 512 reference points
    (hard assignments recomputed at every call), smooth-max
        L = tau * log( (1/1024) * sum_k exp( (sqrt(d_k^2 + eps^2) - eps) / tau ) ).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch1d_fold2/vendor/ch1_frame_mlp"))


def load_member(T, D, path):
    """-> dict(net, cfg, names, K, X, Xt, mu, sd, norm, tower_idx, naux, nqb, ck). Mirrors li_ft.run."""
    dev = D.dev
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = dict(ck["cfg"])
    assert cfg["target"] == "coef" and cfg["loss"] == "metric" and not cfg.get("ctx") and not cfg.get("drop"), path
    names = ck["cols"]
    cidx = np.concatenate([np.arange(*D.blocks[b]) for b in names])
    if len(cidx) == D.X.shape[1]:
        X, Xt = D.X, D.Xt
    else:
        ci = torch.from_numpy(cidx).to(dev)
        X, Xt = D.X[:, ci].contiguous(), D.Xt[:, ci].contiguous()
    tower_idx = None
    if cfg["arch"] == "tower":
        isth = np.zeros(D.X.shape[1], bool)
        for b in T.TH_BLOCKS:
            isth[D.blocks[b][0]:D.blocks[b][1]] = True
        tower_idx = (torch.tensor(np.flatnonzero(~isth[cidx]), device=dev), torch.tensor(np.flatnonzero(isth[cidx]), device=dev))
    K = cfg["K"]
    nm = {k: v.to(dev) for k, v in ck["norm"].items()}
    naux = D.aux.shape[1] if cfg["aux_w"] > 0 else 0
    nqb = 2 if cfg["qb_w"] > 0 else 0
    torch.manual_seed(cfg["seed"] + 1000)
    net = T.Net(X.shape[1], cfg, K, naux, nqb, tower_idx).to(dev)
    net.load_state_dict(ck["state"])
    return dict(net=net, cfg=cfg, names=names, K=K, X=X, Xt=Xt, mu=nm["mu"], sd=nm["sd"], nm=nm, tower_idx=tower_idx,
                naux=naux, nqb=nqb, ck=ck)


class Replay:
    def __init__(self, R, Z, dev, n=512):
        self.dev = torch.device(dev)
        RR, ZZ = np.meshgrid(np.asarray(R, np.float64), np.asarray(Z, np.float64))
        t = lambda a: torch.as_tensor(a, dtype=torch.float64, device=self.dev)  # noqa: E731
        self.Rn, self.Zn = t(RR.ravel()), t(ZZ.ravel())
        self.n = n
        self.frac = torch.arange(n, dtype=torch.float64, device=self.dev) / n

    def edge_state(self, ea, eb, pa, pb, lvl):
        """ea, eb (Q,V) long node ids; pa, pb (Q,V) float64 phi at the nodes; lvl (Q,). -> u (Q,V), overshoot [m] (Q,), min|den| (Q,)"""
        den = pb - pa
        u = (lvl[:, None] - pa) / den
        el = torch.sqrt((self.Rn[eb] - self.Rn[ea]) ** 2 + (self.Zn[eb] - self.Zn[ea]) ** 2)
        over = (torch.clamp(torch.maximum(-u, u - 1.0), min=0.0) * el)
        over = torch.nan_to_num(over, nan=1e9, posinf=1e9)
        return u, over.max(1).values, den.abs().min(1).values

    def polygon(self, ea, eb, u):
        Ra, Rb, Za, Zb = self.Rn[ea], self.Rn[eb], self.Zn[ea], self.Zn[eb]
        return torch.stack([Ra + u * (Rb - Ra), Za + u * (Zb - Za)], -1)

    def resample(self, P):
        """P (Q,V,2) closed polygon (first == last vertex, tail padded with copies of the last vertex) -> (Q,n,2)."""
        d = P[:, 1:] - P[:, :-1]
        seg = torch.sqrt((d * d).sum(-1) + 1e-30)
        s = torch.cat([torch.zeros_like(seg[:, :1]), torch.cumsum(seg, 1)], 1)
        tot = s[:, -1]
        t = tot[:, None] * self.frac[None, :]
        idx = (torch.searchsorted(s.detach().contiguous(), t.detach().contiguous(), right=True) - 1).clamp(0, P.shape[1] - 2)
        s0 = s.gather(1, idx)
        sg = seg.gather(1, idx)
        w = (t - s0) / sg
        i2 = idx[..., None].expand(-1, -1, 2)
        P0 = P.gather(1, i2)
        P1 = P.gather(1, i2 + 1)
        return P0 + w[..., None] * (P1 - P0)

    @staticmethod
    def nn_sets(A, B):
        """A (Q,n,2) predicted, B (Q,n,2) reference -> squared directed NN distances (dA2 (Q,n): pred->ref, dB2 (Q,n): ref->pred),
        differentiable w.r.t. A; hard assignments from the current positions."""
        with torch.no_grad():
            a2 = (A * A).sum(-1)
            b2 = (B * B).sum(-1)
            d2 = a2[:, :, None] + b2[:, None, :] - 2.0 * torch.bmm(A, B.transpose(1, 2))
            iA = d2.argmin(2)
            iB = d2.argmin(1)
        Bn = B.gather(1, iA[..., None].expand(-1, -1, 2))
        An = A.gather(1, iB[..., None].expand(-1, -1, 2))
        return ((A - Bn) ** 2).sum(-1), ((An - B) ** 2).sum(-1)

    @staticmethod
    def smooth_hd(dA2, dB2, tau, eps):
        d2 = torch.cat([dA2, dB2], 1)
        f = torch.sqrt(d2 + eps * eps) - eps
        soft = tau * (torch.logsumexp(f / tau, 1) - math.log(d2.shape[1]))
        hard = torch.sqrt(d2.max(1).values)
        return soft, hard

    def forward(self, ea, eb, pa, pb, lvl, Bref, tau, eps):
        """Convenience: full replay for rows already known to be valid. -> soft (Q,), hard (Q,), A (Q,n,2)."""
        den = pb - pa
        u = (lvl[:, None] - pa) / den
        A = self.resample(self.polygon(ea, eb, u))
        dA2, dB2 = self.nn_sets(A, Bref)
        soft, hard = self.smooth_hd(dA2, dB2, tau, eps)
        return soft, hard, A
