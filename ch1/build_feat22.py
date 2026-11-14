#!/usr/bin/env python3
"""Training cache of the "hc10x24" corrector heads: mirror-canonical FEATURE-V2.2 rows of all 7,041 labelled DIII-D shots.

  build_feat22.py [<out dir>] [--expect-sha256 <hex>]        (default out dir: $SCR/stage11_ch1/l_corr/feat22_all)

  train/X.npy   = the rows of e_cx/h150/feat/train (6,891 shots) followed by those of e_cx/h150/feat/holdout (150 shots); both are
                  already standardised with the canonical v2.2 scaler of the base models. This is the row order of the existing
                  7,041-shot head cache f_corr_canon/feat_all/train (asserted below: offsets, times and labels are identical).
  train/{offsets,times,y_aux}.npy -> links to f_corr_canon/feat_all/train ;  holdout/, test/ -> links to e_cx/h150/feat.
Everything it reads is listed in weights/GENERATED.sha256. The result is byte-identical to the cache of the development run
(sha256 of train/X.npy in ch1/profiles/hc10x24.sh, checked with --expect-sha256).
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np

S = Path(os.environ["SCR"]) / "stage11_ch1"
args = [a for a in sys.argv[1:] if not a.startswith("--")]
exp = sys.argv[sys.argv.index("--expect-sha256") + 1] if "--expect-sha256" in sys.argv else None
if exp in args:
    args.remove(exp)
dst = Path(args[0]) if args else S / "l_corr/feat22_all"
src, ref = S / "e_cx/h150/feat", S / "f_corr_canon/feat_all/train"
(dst / "train").mkdir(parents=True, exist_ok=True)

o = [np.load(src / sp / "offsets.npy").astype(np.int64) for sp in ("train", "holdout")]
off = np.concatenate([o[0], o[1][1:] + o[0][-1]])
assert np.array_equal(off, np.load(ref / "offsets.npy")), "offsets differ from f_corr_canon/feat_all/train"
ya = np.concatenate([np.load(src / sp / "y_aux.npy") for sp in ("train", "holdout")])
assert np.array_equal(ya, np.load(ref / "y_aux.npy"), equal_nan=True), "labels differ from f_corr_canon/feat_all/train"
tm = np.concatenate([np.load(src / sp / "times.npy") for sp in ("train", "holdout")])
assert np.array_equal(tm, np.load(ref / "times.npy")), "times differ from f_corr_canon/feat_all/train"
parts = [np.load(src / sp / "X.npy", mmap_mode="r") for sp in ("train", "holdout")]
n = sum(p.shape[0] for p in parts)
assert n == off[-1]
if not (dst / "train/X.npy").exists():
    out = np.lib.format.open_memmap(dst / "train/X.npy.tmp", mode="w+", dtype=parts[0].dtype, shape=(n,) + parts[0].shape[1:])
    i = 0
    for p in parts:
        for j in range(0, p.shape[0], 200_000):
            blk = np.asarray(p[j:j + 200_000]); out[i:i + len(blk)] = blk; i += len(blk)
    out.flush(); del out
    os.rename(dst / "train/X.npy.tmp", dst / "train/X.npy")
for f in ("offsets.npy", "times.npy", "y_aux.npy"):
    if not (dst / "train" / f).exists():
        os.symlink(ref / f, dst / "train" / f)
for sp in ("holdout", "test"):
    if not (dst / sp).exists():
        os.symlink(src / sp, dst / sp)
h = hashlib.sha256()
with open(dst / "train/X.npy", "rb") as fh:
    for b in iter(lambda: fh.read(1 << 24), b""):
        h.update(b)
print(f"feat22_all/train: {len(off) - 1} shots, {n} frames, {parts[0].shape[1]} columns; X.npy {os.path.getsize(dst / 'train/X.npy')} bytes sha256 {h.hexdigest()}")
if exp:
    print("FEAT22 sha256", "IDENTICAL to the development cache" if h.hexdigest() == exp else f"DIFFERENT from the recorded {exp}")
    sys.exit(0 if h.hexdigest() == exp else 1)
