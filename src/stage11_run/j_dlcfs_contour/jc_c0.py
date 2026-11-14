#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — step 1 (GPU): eval-mode (dropout off) coefficient predictions of the 11 warm-start FM members on
the rows they were trained on (the bank rows of the set).  In-sample by construction: the contour loss is a function of the
member's own residual on its training rows.
  SET=<h150|fold3|fold2|full> python jc_c0.py     ->  $J/<set>/c0/<member>.npy  float32 (N, K)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import jc_common as C

SET = os.environ["SET"]
info = C.set_info(SET)
os.environ["FEAT"] = str(info["feat"])
os.environ["METRIC"] = str(info["metric"])
os.environ["OUT"] = str(C.J / SET / "fm_out")
import torch  # noqa: E402

import jc_torch as JT  # noqa: E402
import fm_train as T  # noqa: E402


def main():
    dev = torch.device("cuda")
    D = T.Data(dev)
    n = D.ntr + (D.nh if info["trainset"] == "full" else 0)
    rows = torch.arange(n, device=dev)
    off, idx, _ = C.bank_rows(SET)
    assert int(off[-1]) == n, (int(off[-1]), n)
    out = C.J / SET / "c0"
    out.mkdir(parents=True, exist_ok=True)
    for m in C.member_names(info["prefix"]):
        M = JT.load_member(T, D, info["members"] / m / "model.pt")
        o = T.predict(M["net"], M["X"][rows])[0]
        c = (o * M["sd"] + M["mu"]).cpu().numpy().astype(np.float32)
        np.save(out / f"{m}.npy", c)
        Y = D.Y(M["K"])[rows]
        e = torch.from_numpy(c).to(dev) - Y
        C.log(f"{SET} {m}: K {M['K']} rows {n} in-sample coef rms err {float((e ** 2).sum(1).mean().sqrt()):.5f}")
        del M
        torch.cuda.empty_cache()
    C.log("c0 done")


if __name__ == "__main__":
    sys.exit(main())
