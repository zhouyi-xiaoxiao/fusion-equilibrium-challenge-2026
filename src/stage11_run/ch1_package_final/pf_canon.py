#!/usr/bin/env python3
"""ch1_package_final — mirror-canonical step with an EXPLICIT per-shot sign, then the canonical TRAIN scaler.

  pf_canon.py --feat OUT_OF_pf_features --raw raw22|raw21 --scaler <canonical cache>/scaler_v21.npz --out DIR

Arithmetic = e_cx/cx_canon.canon + f2_build.standardise:  X_c = X_raw (float64); X_c[:, odd] *= s  (430 odd columns; raw
v1_sign_ip kept as a flag: variant cfl, mask stored in the scaler file);  X = clip((X_c - mean) / scale).
The only difference to cx_canon: s is the per-shot sign decided by pf_features.py (classes 1-3), not the feature column, so
a zero / contradictory flag cannot abort the run.  For class 1-2 shots s == the feature column and the result is bit-identical.
DIR gets X.npy plus links to X_raw.npy (the RAW rows: the SC physics members read them), offsets / times / shot_id / files.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "stage11_run/ch1d_fold2"))
from f2_build import standardise  # noqa: E402

ap = argparse.ArgumentParser()
for k in ("feat", "raw", "scaler", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
fd, out = Path(a.feat), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
sc = np.load(a.scaler)
assert str(sc["variant"]) == "cfl", sc["variant"]
odd, scol, is_v1, clip = sc["odd"].astype(bool), int(sc["sign_col"]), sc["is_v1"].astype(bool), float(sc["clip"])
off = np.load(fd / "offsets.npy")
s_shot = np.load(fd / "sign.npy")
assert len(s_shot) == len(off) - 1 and np.isin(s_shot, (-1.0, 1.0)).all()
s = np.repeat(s_shot, np.diff(off))
Xr = np.load(fd / a.raw / "X_raw.npy", mmap_mode="r")
assert Xr.shape == (off[-1], len(odd)), (Xr.shape, off[-1], len(odd))
X = np.lib.format.open_memmap(out / "X.npy", mode="w+", dtype=np.float32, shape=Xr.shape)
nmis = 0
for i in range(0, Xr.shape[0], 200_000):
    b = np.asarray(Xr[i:i + 200_000], np.float64)
    nmis += int((b[:, scol] != s[i:i + 200_000]).sum())
    b[:, odd] *= s[i:i + 200_000, None]
    X[i:i + 200_000] = standardise(b, sc["mean"], sc["scale"], is_v1, clip)
X.flush()
for f, src in (("X_raw.npy", fd / a.raw / "X_raw.npy"), ("offsets.npy", fd / "offsets.npy"), ("times.npy", fd / "times.npy"),
               ("shot_id.npy", fd / "shot_id.npy"), ("files.txt", fd / "files.txt")):
    if (out / f).is_symlink() or (out / f).exists():
        (out / f).unlink()
    os.symlink(src.resolve(), out / f)
nm = Path(a.scaler).parent / "names.txt"
if nm.exists():
    shutil.copy(nm, out / "names.txt")
print(f"PF_CANON {a.raw}: {Xr.shape[0]} frames, {int(odd.sum())} odd columns, Ip<0 shots {int((s_shot < 0).sum())}/{len(s_shot)}, "
      f"frames where the explicit sign != feature column {nmis}, scaler {a.scaler}", flush=True)
