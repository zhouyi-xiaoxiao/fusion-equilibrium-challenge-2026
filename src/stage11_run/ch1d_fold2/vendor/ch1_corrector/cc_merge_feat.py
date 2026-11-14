#!/usr/bin/env python3
"""Build a feature-v2.1 directory whose `train` split is the concatenation of several splits (final refit on all
7,041 labelled DIII-D shots = feat_v21/train (6,891) + feat_v21/holdout (150)).

  cc_merge_feat.py --src $SCR/stage11_ch1/feat_v21 --dst $SCR/stage11_ch1/feat_v21_all --splits train,holdout
Writes <dst>/train/{X,y_C80,y_C120,y_q95,y_betaN,y_aux,shot_id,times}.npy + offsets.npy + files.txt (memmap copy,
row order = split order then the split's own order) and symlinks <dst>/{holdout,test} and the scaler/spec files to
<src> so every other script runs unchanged with CC_FEAT=<dst>. NB: with this dir the "holdout" is IN-SAMPLE.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

ARR = ["X", "y_C80", "y_C120", "y_q95", "y_betaN", "y_aux", "shot_id", "times"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--dst", required=True)
    ap.add_argument("--splits", default="train,holdout")
    a = ap.parse_args()
    src, dst = Path(a.src), Path(a.dst)
    sp = a.splits.split(",")
    (dst / "train").mkdir(parents=True, exist_ok=True)
    for nm in ARR:
        parts = [np.load(src / s / f"{nm}.npy", mmap_mode="r") for s in sp]
        n = sum(p.shape[0] for p in parts)
        out = np.lib.format.open_memmap(dst / "train" / f"{nm}.npy", mode="w+", dtype=parts[0].dtype,
                                        shape=(n,) + parts[0].shape[1:])
        i = 0
        for p in parts:
            for j in range(0, p.shape[0], 200_000):
                blk = np.asarray(p[j:j + 200_000]); out[i:i + len(blk)] = blk; i += len(blk)
        out.flush(); del out
        print(f"{nm}: {n} rows")
    offs, files, base = [np.zeros(1, np.int64)], [], 0
    for s in sp:
        o = np.load(src / s / "offsets.npy").astype(np.int64)
        offs.append(o[1:] + base); base += int(o[-1])
        files += [l for l in (src / s / "files.txt").read_text().splitlines() if l.strip()]
    off = np.concatenate(offs)
    np.save(dst / "train" / "offsets.npy", off)
    (dst / "train" / "files.txt").write_text("\n".join(files) + "\n")
    assert len(files) == len(off) - 1, (len(files), len(off))
    for s in ("holdout", "test"):
        if not (dst / s).exists():
            os.symlink(src / s, dst / s)
    for f in os.listdir(src):
        if (src / f).is_file() and not (dst / f).exists():
            os.symlink(src / f, dst / f)
    print(f"merged {sp}: {len(files)} shots, {int(off[-1])} frames -> {dst}/train")


if __name__ == "__main__":
    main()
