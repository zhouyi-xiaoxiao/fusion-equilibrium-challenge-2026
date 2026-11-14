#!/usr/bin/env python3
"""
Stage 11 / ch1_frame_mlp — inference on an ARBITRARY directory of DIII-D parquet shots (inputs only).

  1. features : build_unseen.py (FEATURE-V2.1, inputs only, scaler_v21.npz fit on TRAIN rows)  [skipped with --feat-dir]
  2. members  : every <members-root>/<name>/model.pt listed in --psi / --qb (glob patterns allowed)
  3. decode   : coefficients -> 65x65 psi with the PCA basis of each member (stage6_H pca_<K>.npz); psi members are
                averaged as decoded maps (float64), (q95, betaN) members averaged in physical units
  4. write    : <out>.npz in the submission / harness layout  shot_XXXX_psirz (T,65,65) float32, shot_XXXX_q95,
                shot_XXXX_betaN (T,) float32, XXXX = index in sorted file order; <out>.files.txt = that order.

  python fm_predict.py --in-dir DIR --out preds.npz --psi 'FULL_d30aux_s*+FULL_towerd30aux_s*+FULL_K120d30aux_s*' \
        --qb 'FULL_Qd30aux10_s*' [--feat-dir EXISTING_FEATURE_CACHE] [--members-root .../frame_mlp/members]
No label column is read anywhere on this path.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fm_train as T  # noqa: E402  (Net / Block / block_cols / constants only; no side effects)

SCR = T.SCR
FEAT = T.FEAT
HC = Path(os.environ.get("PCA_DIR", str(SCR / "stage6_H")))


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def expand(pats, root):
    names = sorted(p.name for p in root.iterdir() if (p / "model.pt").exists())
    out = []
    for pat in [p for p in re.split(r"[,+ ]", pats) if p]:  # "+" is safe inside sbatch --export
        hit = [n for n in names if fnmatch.fnmatch(n, pat)]
        if not hit:
            raise SystemExit(f"no member matches {pat!r} under {root}")
        out += [h for h in hit if h not in out]
    return out


def load_member(path, spec, din_all, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    if cfg.get("ctx") or cfg.get("drop"):
        raise SystemExit(f"{path}: ctx/drop members are not supported by the predict path")
    blocks = spec["blocks"]
    cidx = np.concatenate([np.arange(*blocks[b]) for b in ck["cols"]])
    tower_idx = None
    if cfg["arch"] == "tower":
        isth = np.zeros(din_all, bool)
        for b in T.TH_BLOCKS:
            isth[blocks[b][0]:blocks[b][1]] = True
        tower_idx = (torch.tensor(np.flatnonzero(~isth[cidx])), torch.tensor(np.flatnonzero(isth[cidx])))
    naux = len(spec["aux_names"]) - T.AUX_FROM if cfg["aux_w"] > 0 else 0
    if cfg["target"] == "coef":
        nout, nqb = cfg["K"], (2 if cfg["qb_w"] > 0 else 0)
    else:
        nout, nqb = 2, 0
    net = T.Net(len(cidx), cfg, nout, naux, nqb, tower_idx)
    net.load_state_dict(ck["state"])
    return net.to(dev).eval(), cfg, cidx, ck["norm"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default=None, help="directory of d3d_shot_*.parquet (inputs only)")
    ap.add_argument("--feat-dir", default=None, help="existing FEATURE-V2.1 cache (X.npy, offsets.npy, files.txt)")
    ap.add_argument("--work", default=None, help="where build_unseen writes features (default <out>_feat)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--psi", required=True)
    ap.add_argument("--qb", required=True)
    ap.add_argument("--members-root", default=str(T.OUT / "members"))
    ap.add_argument("--save-raw", default=None, help="dir: per-member raw outputs (coef_<name>.npy / qb_<name>.npy)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "70")))
    a = ap.parse_args()
    out = Path(a.out)
    if a.feat_dir:
        fd = Path(a.feat_dir)
    else:
        fd = Path(a.work or str(out.with_suffix("")) + "_feat")
        if not (fd / "X.npy").exists():
            cmd = [sys.executable, str(HERE.parent / "ch1_features" / "build_unseen.py"), "--in-dir", a.in_dir,
                   "--out", str(fd), "--workers", str(a.workers)]
            log("features: " + " ".join(cmd))
            subprocess.run(cmd, check=True)
    X = np.nan_to_num(np.load(fd / "X.npy")).astype(np.float32)
    off = np.load(fd / "offsets.npy")
    files = (fd / "files.txt").read_text().split()
    spec = json.load(open(FEAT / "spec.json"))
    assert X.shape[1] == max(v[1] for v in spec["blocks"].values()), X.shape
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cpu":
        torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "32")))
    Xd = torch.from_numpy(X).to(dev)
    root = Path(a.members_root)
    raw = Path(a.save_raw) if a.save_raw else None
    if raw:
        raw.mkdir(parents=True, exist_ok=True)
    pcs, psi, npsi = {}, None, 0
    for m in expand(a.psi, root):
        net, cfg, cidx, nm = load_member(root / m / "model.pt", spec, X.shape[1], dev)
        assert cfg["target"] == "coef", m
        o = T.predict(net, Xd[:, torch.from_numpy(cidx).to(dev)] if len(cidx) != X.shape[1] else Xd)[0]
        C = (o.cpu() * nm["sd"] + nm["mu"]).numpy().astype(np.float32)
        if raw:
            np.save(raw / f"coef_{m}.npy", C)
        K = cfg["K"]
        if K not in pcs:
            pc = np.load(HC / f"pca_{K}.npz")
            pcs[K] = (pc["comps_"].astype(np.float64), pc["mean_"].astype(np.float64))
        p = C.astype(np.float64) @ pcs[K][0] + pcs[K][1][None, :]
        psi = p if psi is None else psi + p
        npsi += 1
        log(f"psi member {m} (K={K}, {cfg['arch']}) done")
    psi /= npsi
    qb, nq = None, 0
    for m in expand(a.qb, root):
        net, cfg, cidx, nm = load_member(root / m / "model.pt", spec, X.shape[1], dev)
        assert cfg["target"] == "qb", m
        o = T.predict(net, Xd[:, torch.from_numpy(cidx).to(dev)] if len(cidx) != X.shape[1] else Xd)[0]
        q = (o.cpu() * nm["qsd"] + nm["qmu"]).numpy().astype(np.float64)
        if raw:
            np.save(raw / f"qb_{m}.npy", q.astype(np.float32))
        qb = q if qb is None else qb + q
        nq += 1
        log(f"qb member {m} done")
    qb /= nq
    d = {}
    for s in range(len(off) - 1):
        lo, hi = off[s], off[s + 1]
        d[f"shot_{s:04d}_psirz"] = psi[lo:hi].reshape(-1, 65, 65).astype(np.float32)
        d[f"shot_{s:04d}_q95"] = qb[lo:hi, 0].astype(np.float32)
        d[f"shot_{s:04d}_betaN"] = qb[lo:hi, 1].astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **d)
    Path(str(out) + ".files.txt").write_text("\n".join(files) + "\n")
    log(f"wrote {out} ({len(off)-1} shots, {len(X)} frames; {npsi} psi members, {nq} qb members; float32)")


if __name__ == "__main__":
    main()
