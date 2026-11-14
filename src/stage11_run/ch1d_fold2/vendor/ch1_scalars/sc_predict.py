#!/usr/bin/env python3
"""
Stage 11 / ch1_scalars — inputs-only predict path for q95 / betaN on an arbitrary new shot directory.

  1. python ../ch1_features/build_unseen.py --in-dir <parquet shot dir> --out <featdir>     (X.npy, X_raw.npy, offsets.npy, files.txt)
  2. python sc_predict.py --feat-dir <featdir> --weights <w1.pt> <w2.pt> ... --out <out.npz>

Every weight file carries its config, target standardisation and (phys configs) the physics-feature stats, so nothing else is
read from the training data. Output: q95 / betaN (float32, feature-dir row order; mean over families of the per-family seed means = the sc_report
BLENDS rule, or plain member mean with --member-mean),
per-member arrays q95_<tag> / betaN_<tag>, offsets. Works on the GPU if present, else CPU.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sc_nn as S  # noqa: E402
from sc_common import PHYS_COLS, phys_extra  # noqa: E402


class Shim:
    """The subset of sc_nn.Data that predict() / decode() read."""

    def __init__(self, fd: Path, dev, st):
        X = np.nan_to_num(np.load(fd / "X.npy"))
        off = np.load(fd / "offsets.npy").astype(np.int64)
        self.Ne, self.D = X.shape
        self.Xe = torch.from_numpy(np.concatenate([X, np.zeros((1, self.D), np.float32)])).to(dev)
        self.start_e = torch.from_numpy(off[:-1]).to(dev); self.len_e = torch.from_numpy(np.diff(off)).to(dev)
        self.fd, self.dev, self.off = fd, dev, off
        self.set_stats(st)

    def set_stats(self, st):
        self.mu, self.sd = np.asarray(st["mu"]), np.asarray(st["sd"]); self.mu_t, self.sd_t = st["mu_t"], st["sd_t"]
        self.pst = st.get("phys")

    def inputs(self, c):
        if c.get("phys", 0):
            names = (self.fd / "names.txt").read_text().split() if (self.fd / "names.txt").exists() else S.FEAT.joinpath("names.txt").read_text().split()
            Xr = np.load(self.fd / "X_raw.npy", mmap_mode="r")
            R = np.stack([np.asarray(Xr[:, names.index(k)], np.float64) for k in PHYS_COLS], 1)
            E, _ = phys_extra(R, dict(self.pst))
            E = torch.from_numpy(np.concatenate([E, np.zeros((1, 4), np.float32)])).to(self.dev)
            return None, torch.cat([self.Xe, E], 1)
        assert not c["ctx"], "ctx configs are not part of the predict path"
        return None, self.Xe


def load_net(ck, D, dev):
    c = ck["cfg"]
    din = D.D + (4 if c.get("phys", 0) else 0)
    frame = S.FrameNet(din, c["width"], c["blocks"], ck["n_out"], c["drop"], torch.ones(din)).to(dev)
    net = frame if c["kind"] == "mlp" else S.SeqNet(frame, c["width"], ck["n_out"], c["emb"], c["hid"], c["layers"], c["gdrop"]).to(dev)
    net.load_state_dict(ck["state"])
    return c, net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--member-mean", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    D = None; res = {}
    for w in a.weights:
        ck = torch.load(w, map_location=dev, weights_only=False)
        if D is None:
            D = Shim(Path(a.feat_dir), dev, ck["stats"])
        assert ck["D"] == D.D, f"{w}: trained on D={ck['D']}, feature dir has D={D.D}"
        D.set_stats(ck["stats"])
        c, net = load_net(ck, D, dev)
        q, b = S.predict(c, D, dev, net)
        res[Path(w).stem] = (q, b)
        print(f"{Path(w).stem}: q95 mean {q.mean():.4f} betaN mean {b.mean():.4f}", flush=True)
    fam = {}
    for k in res:
        fam.setdefault(k if a.member_mean else re.sub(r"_s\d+$", "", k), []).append(k)
    fq = [np.mean([res[k][0] for k in v], 0) for v in fam.values()]; fb = [np.mean([res[k][1] for k in v], 0) for v in fam.values()]
    q = np.mean(fq, 0); b = np.mean(fb, 0)
    print("families: " + ", ".join(f"{f}x{len(v)}" for f, v in fam.items()), flush=True)
    out = {"q95": q.astype(np.float32), "betaN": b.astype(np.float32), "offsets": D.off}
    for k, (qq, bb) in res.items():
        out[f"q95_{k}"] = qq.astype(np.float32); out[f"betaN_{k}"] = bb.astype(np.float32)
    np.savez(a.out, **out)
    print(f"wrote {a.out}: {len(q)} frames, {len(D.off) - 1} shots, {len(res)} members", flush=True)


if __name__ == "__main__":
    main()
