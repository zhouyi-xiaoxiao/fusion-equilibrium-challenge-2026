#!/usr/bin/env python3
"""Stage 11D / ch1d_fold2 — step 1: the SECOND validation fold ("fold2") and its feature cache.

Fold definition (inputs only; documented in an internal development note):
  The shot files are hashed (no shot number, no date). The closest available proxy for "contiguous shot-number
  blocks" is the Thomson-scattering CHORD-GEOMETRY signature (chord names + surveyed R,Z; f2_meta.py `sig_th`): it
  changes whenever the diagnostic is re-surveyed or re-configured, i.e. it labels calibration epochs (contiguous
  shot ranges). 38 signatures exist in the 7,041 train shots (3 giant ones of 1.2-1.4k shots, the rest 1-335).
  fold2 = SIX whole signature groups, holdout150 shots removed, chosen so that every acquisition era / Thomson layout
  stays represented in the remaining training data and the era mix is close to the train/holdout mix:
     d55edb5fce 155 (0.5 ms magnetics DAQ era, edge layout 47b0)   afc0c9086b 170 (44-ch core, edge 12f5)
     f4798816db  96 (44-ch core, edge 5576)                          0ffd935936  53 (44-ch core, edge 12f5)
     7f2ceda557  16 (42-ch core, edge 1790)                          780c0ce72c  10 (42-ch core, edge 1790)
  = 500 shots; training set = the other 6,391 train shots (holdout150 is never touched).

Feature cache  $F2/feat/{train,holdout,test}  (same layout as feat_v21, so every track runs with FEAT=$F2/feat):
  train   = 6,391 shots (original train order),          labels y_*
  holdout = fold2, 500 shots, sorted by file name         labels y_*   (symlink `fold2` -> holdout for the corrector)
  test    = the standard holdout150 (sorted order)        labels y_*   (bonus: fold2-trained models also predict it)
  X = re-standardised with a scaler refit on the 6,391 training shots (v1 block: StandardScaler semantics as stage6_H
  scaler_f; other blocks: NaN-aware mean/std(ddof=1), NaN->0, clip +-10 = feat_v21.standardise).
Self-tests: (a) the scaler code reproduces scaler_v21.npz when fit on all 6,891 train shots; (b) standardising the
holdout150 X_raw with the ORIGINAL scaler reproduces feat_v21/holdout/X.npy bit-for-bit.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SCR = Path(os.environ["SCR"])
SRC = SCR / "stage11_ch1/feat_v21"
DATA = SCR / "data_full/diii_d_train"
F2 = Path(os.environ.get("F2", str(SCR / "stage11_ch1/fold2")))
FEAT = F2 / "feat"
BLOCKS = ["d55edb5fce", "afc0c9086b", "f4798816db", "0ffd935936", "7f2ceda557", "780c0ce72c"]
ARR = ["X_raw", "y_C80", "y_C120", "y_q95", "y_betaN", "y_aux", "times"]
CLIP = 10.0
FOLD = os.environ.get("FOLD", "blocked")          # blocked = fold2 (campaign blocks) | random = fold3 (uniform random shots)
SEED_RANDOM = 20261012


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def standardise(Xraw, mean, scale, is_v1, clip):          # == feat_v21.standardise (copied; verified bitwise below)
    b = np.asarray(Xraw, dtype=np.float64)
    z = (b - mean[None, :]) / scale[None, :]
    nv = ~is_v1
    zz = np.nan_to_num(z[:, nv], nan=0.0, posinf=clip, neginf=-clip)
    z[:, nv] = np.clip(zz, -clip, clip)
    return z.astype(np.float32)


def fit_scaler(Xr, rows, is_v1, chunk=200_000):
    """v1: nan-aware mean / population std (sklearn StandardScaler, zero std -> 1);
    others: nan-aware mean / sample std (prep_v2.nan_stats), std <= 1e-8 -> 1."""
    D = Xr.shape[1]
    s = np.zeros(D); c = np.zeros(D)
    for a in range(0, len(rows), chunk):
        b = np.asarray(Xr[rows[a:a + chunk]], np.float64)
        f = np.isfinite(b); s += np.where(f, b, 0).sum(0); c += f.sum(0)
    mean = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    v = np.zeros(D)
    for a in range(0, len(rows), chunk):
        b = np.asarray(Xr[rows[a:a + chunk]], np.float64)
        f = np.isfinite(b); v += np.where(f, (b - mean) ** 2, 0).sum(0)
    sd1 = np.sqrt(np.where(c > 1, v / np.maximum(c - 1, 1), 0.0))
    sd0 = np.sqrt(np.where(c > 0, v / np.maximum(c, 1), 0.0))
    scale = np.where(sd1 > 1e-8, sd1, 1.0)
    scale[is_v1] = np.where(sd0[is_v1] > 0, sd0[is_v1], 1.0)   # sklearn: _handle_zeros_in_scale
    return mean, scale


def rows_of(off, shots):
    return np.concatenate([np.arange(off[s], off[s + 1]) for s in shots]).astype(np.int64)


def write_split(name, src_split, shots, files, mean, scale, is_v1):
    d = FEAT / name
    d.mkdir(parents=True, exist_ok=True)
    off_src = np.load(SRC / src_split / "offsets.npy").astype(np.int64)
    rows = rows_of(off_src, shots)
    n = np.diff(off_src)[shots]
    off = np.concatenate([[0], np.cumsum(n)]).astype(off_src.dtype)
    for a in ARR:
        m = np.load(SRC / src_split / f"{a}.npy", mmap_mode="r")
        out = np.lib.format.open_memmap(d / f"{a}.npy", mode="w+", dtype=m.dtype, shape=(len(rows),) + m.shape[1:])
        for i in range(0, len(rows), 200_000):
            out[i:i + 200_000] = m[rows[i:i + 200_000]]
        out.flush(); del out
    Xr = np.load(d / "X_raw.npy", mmap_mode="r")
    X = np.lib.format.open_memmap(d / "X.npy", mode="w+", dtype=np.float32, shape=Xr.shape)
    for i in range(0, len(rows), 100_000):
        X[i:i + 100_000] = standardise(Xr[i:i + 100_000], mean, scale, is_v1, CLIP)
    X.flush(); del X
    np.save(d / "offsets.npy", off)
    np.save(d / "shot_id.npy", np.repeat(np.arange(len(shots), dtype=np.int32), n))
    (d / "files.txt").write_text("\n".join(files) + "\n")
    log(f"  {name}: {len(shots)} shots, {len(rows)} frames (from {src_split})")
    return rows


def main():
    t0 = time.time()
    FEAT.mkdir(parents=True, exist_ok=True)
    z = np.load(F2 / "meta_all.npz")
    tr_files = [l for l in (SRC / "train/files.txt").read_text().split() if l]
    ho_files = [l for l in (SRC / "holdout/files.txt").read_text().split() if l]
    ntr = len(tr_files)
    assert list(z["fname"][:ntr]) == tr_files and list(z["fname"][ntr:]) == ho_files
    sig = z["sig_th"][:ntr]
    in_f2 = np.isin(sig, BLOCKS)
    blk_names = BLOCKS
    if FOLD == "random":                                            # fold3: 500 shots uniformly at random (seed 20261012)
        in_f2 = np.zeros(ntr, bool)
        in_f2[np.random.default_rng(SEED_RANDOM).choice(ntr, 500, replace=False)] = True
        era = np.where(z["mag_dt"][:ntr] > 0.1, "0.5ms", np.where(np.isin(z["n_core_ch"][:ntr], [40, 42]), "40/42ch",
                       np.where(z["n_core_ch"][:ntr] == 54, "54ch", np.where(z["sig_edge"][:ntr] == "557603556f", "44ch-edge5576", "44ch-edge12f5"))))
        blk_names = ["0.5ms", "40/42ch", "44ch-edge12f5", "44ch-edge5576", "54ch"]
        sig = era                                                   # pseudo-blocks = acquisition era / Thomson layout
    keep = np.flatnonzero(~in_f2)                                   # original train order
    f2_idx = np.flatnonzero(in_f2)
    f2_idx = f2_idx[np.argsort([tr_files[i] for i in f2_idx])]     # fold2 order = sorted file name
    f2_files = [tr_files[i] for i in f2_idx]
    blk = np.array([blk_names.index(sig[i]) for i in f2_idx], np.int64)
    assert len(f2_idx) == 500 and len(keep) == 6391, (len(f2_idx), len(keep))
    assert not set(f2_files) & set(ho_files)
    np.save(F2 / "fold2_trainidx.npy", f2_idx); np.save(F2 / "fold2_block.npy", blk)
    np.save(F2 / "train_keep_idx.npy", keep)
    (F2 / "fold2_files.txt").write_text("\n".join(f2_files) + "\n")
    (F2 / "gt_list_fold2.txt").write_text("\n".join(str(DATA / f) for f in f2_files) + "\n")
    (F2 / "gt_list_holdout150.txt").write_text("\n".join(str(DATA / f) for f in ho_files) + "\n")
    off_tr = np.load(SRC / "train/offsets.npy").astype(np.int64)
    nfr = np.diff(off_tr)
    info = {"fold": FOLD, "blocks": blk_names, "block_shots": [int((blk == b).sum()) for b in range(len(blk_names))],
            "block_frames": [int(nfr[f2_idx[blk == b]].sum()) for b in range(len(blk_names))],
            "fold2_shots": 500, "fold2_frames": int(nfr[f2_idx].sum()), "train_shots": int(len(keep)),
            "train_frames": int(nfr[keep].sum()), "order": "fold2 = sorted by file name; block id per shot in fold2_block.npy"}
    log("fold2: " + json.dumps(info))

    # ---------------- scaler: self-test on all 6,891 train shots, then refit on the 6,391
    Xr = np.load(SRC / "train/X_raw.npy", mmap_mode="r")
    s0 = np.load(SRC / "scaler_v21.npz")
    is_v1, mean0, scale0 = s0["is_v1"].astype(bool), s0["mean"].astype(np.float64), s0["scale"].astype(np.float64)
    all_rows = np.arange(Xr.shape[0])
    mA, sA = fit_scaler(Xr, all_rows, is_v1)
    tst = {"mean_maxabs_rel": float(np.max(np.abs(mA - mean0) / np.maximum(np.abs(mean0), 1e-6))),
           "scale_maxabs_rel": float(np.max(np.abs(sA - scale0) / scale0))}
    log(f"scaler self-test (all 6,891 vs scaler_v21.npz): {tst}")
    Xh_raw = np.load(SRC / "holdout/X_raw.npy")
    bit = bool(np.array_equal(standardise(Xh_raw, mean0, scale0, is_v1, float(s0["clip"])), np.load(SRC / "holdout/X.npy")))
    log(f"standardise self-test (holdout150, original scaler) bit-identical: {bit}")
    assert bit and tst["scale_maxabs_rel"] < 1e-5, "scaler self-test failed"
    keep_rows = rows_of(off_tr, keep)
    mean, scale = fit_scaler(Xr, keep_rows, is_v1)
    np.savez(FEAT / "scaler_v21.npz", mean=mean, scale=scale, is_v1=is_v1, clip=CLIP, names=s0["names"])
    log(f"fold2 scaler (6,391 shots): max rel change vs scaler_v21 mean {np.max(np.abs(mean-mean0)/np.maximum(np.abs(mean0),1e-6)):.3e} "
        f"scale {np.max(np.abs(scale-scale0)/scale0):.3e}")

    # ---------------- splits
    write_split("train", "train", keep, [tr_files[i] for i in keep], mean, scale, is_v1)
    write_split("holdout", "train", f2_idx, f2_files, mean, scale, is_v1)
    write_split("test", "holdout", np.arange(len(ho_files)), ho_files, mean, scale, is_v1)
    if not (FEAT / "fold2").exists():
        os.symlink("holdout", FEAT / "fold2")
    if not (FEAT / "holdout150").exists():
        os.symlink("test", FEAT / "holdout150")
    shutil.copy(SRC / "names.txt", FEAT / "names.txt")
    spec = json.load(open(SRC / "spec.json"))
    spec["version"] = "feat_v21 fold2 (stage11D) 2026-10-12"
    spec["splits"] = {"train": {"shots": info["train_shots"], "frames": info["train_frames"]},
                      "holdout": {"shots": 500, "frames": info["fold2_frames"], "is": "fold2"},
                      "test": {"shots": len(ho_files), "frames": int(np.load(SRC / "holdout/offsets.npy")[-1]), "is": "holdout150"}}
    spec["fold2"] = info
    spec["scaler"] = "refit on the 6,391 fold2-training shots (fit_scaler in ch1d_fold2/f2_build.py)"
    json.dump(spec, open(FEAT / "spec.json", "w"), indent=1)
    json.dump({**info, "scaler_selftest": tst, "standardise_bit_identical": bit}, open(F2 / "fold2_info.json", "w"), indent=1)
    log(f"done in {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
