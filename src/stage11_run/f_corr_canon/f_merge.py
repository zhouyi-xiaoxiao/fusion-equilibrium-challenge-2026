#!/usr/bin/env python3
"""Stage 11F / corr_canon — FULL (7,041-shot) mirror-canonical head-training cache.

train = canonical h150 train (6,891) + canonical h150 holdout (150), i.e. exactly the row order of feat_v21_all/train
(cc_merge_feat.py), standardised with the canonical h150 scaler ($E/h150_cfl/feat/scaler_v21.npz = the scaler of the
E_cfl FULL FM/TW/SC models).  Only X.npy is written; every other array is a symlink to feat_v21_all/train (identical
rows: the canonical cache symlinks its labels/offsets to feat_v21).  holdout/test/scaler -> canonical h150 cache.
  python f_merge.py     (env E, S, F)
"""
import os
from pathlib import Path
import numpy as np

S, E, F = (Path(os.environ[k]) for k in ("S", "E", "F"))
src, ref, dst = E / "h150_cfl/feat", S / "feat_v21_all/train", F / "feat_all"
(dst / "train").mkdir(parents=True, exist_ok=True)
files = []
for sp in ("train", "holdout"):
    files += [l for l in (src / sp / "files.txt").read_text().splitlines() if l.strip()]
assert files == [l for l in (ref / "files.txt").read_text().splitlines() if l.strip()], "row order != feat_v21_all/train"
o = [np.load(src / sp / "offsets.npy").astype(np.int64) for sp in ("train", "holdout")]
off = np.concatenate([o[0], o[1][1:] + o[0][-1]])
assert np.array_equal(off, np.load(ref / "offsets.npy")), "offsets != feat_v21_all/train"
parts = [np.load(src / sp / "X.npy", mmap_mode="r") for sp in ("train", "holdout")]
n = sum(p.shape[0] for p in parts)
assert n == off[-1]
out = np.lib.format.open_memmap(dst / "train/X.npy", mode="w+", dtype=parts[0].dtype, shape=(n,) + parts[0].shape[1:])
i = 0
for p in parts:
    for j in range(0, p.shape[0], 200_000):
        blk = np.asarray(p[j:j + 200_000]); out[i:i + len(blk)] = blk; i += len(blk)
out.flush(); del out
for f in os.listdir(ref):
    if f != "X.npy" and not (dst / "train" / f).exists():
        os.symlink((ref / f).resolve(), dst / "train" / f)
for sp in ("holdout", "test"):
    if not (dst / sp).exists():
        os.symlink(src / sp, dst / sp)
for f in os.listdir(src):
    if (src / f).is_file() and not (dst / f).exists():
        os.symlink(src / f, dst / f)
sc = np.load(dst / "scaler_v21.npz")
print(f"feat_all/train: {len(files)} shots {n} frames, canonical variant {sc['variant']}, odd cols {int(sc['odd'].sum())}; "
      f"holdout/test/scaler -> {src}")
