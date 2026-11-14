#!/usr/bin/env python3
"""Apply the saved full-train PCA80 base nets (cc_base.py: net_<TAG>_s<seed>.pt) to ANY feature-v2.1 directory.

  cc_base_predict.py --tag gv21 --seeds 0,1,2,3 --feat-dir <dir with X.npy> --out coef.npy
Seed-bag mean of the decoded coefficients (N, 80), float32. GPU if available, else CPU.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

import cc_common as cc
from cc_common import log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gv21"); ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--feat-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--net-dir", default=None, help="directory holding net_<tag>_s<seed>.pt (default $CC_OUT)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(np.nan_to_num(np.load(f"{a.feat_dir}/X.npy"))).to(dev)
    nd = a.net_dir or str(cc.OUT)
    outs = []
    for s in [int(x) for x in a.seeds.split(",")]:
        ck = torch.load(f"{nd}/net_{a.tag}_s{s}.pt", map_location=dev)
        D, K = int(ck["D"]), int(ck["K"])
        net = torch.nn.Sequential(torch.nn.Linear(D, 1024), torch.nn.SiLU(), torch.nn.Linear(1024, 1024), torch.nn.SiLU(),
                                  torch.nn.Linear(1024, 512), torch.nn.SiLU(), torch.nn.Linear(512, K)).to(dev)
        net.load_state_dict(ck["state"]); net.eval()
        mu, sd = ck["mu"].to(dev), ck["sd"].to(dev)
        with torch.no_grad():
            outs.append(torch.cat([net(X[i:i + 65536]) * sd + mu for i in range(0, X.shape[0], 65536)]).double().cpu().numpy())
    np.save(a.out, np.mean(outs, 0).astype(np.float32))
    log(f"wrote {a.out} {np.shape(outs[0])} ({len(outs)} seeds, {dev})")


if __name__ == "__main__":
    main()
