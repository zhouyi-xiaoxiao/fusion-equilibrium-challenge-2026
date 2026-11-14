#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — shared paths and set definitions (no torch import here).

Sets: h150 | fold3 | fold2 (members trained on the set's train split, trainset t100) and full (FULL members, trainset full =
train + holdout rows of the h150 cache).  Parent family = e_cx (mirror-canonical x feature v2.2).

Package (DIII-D profile xhc, training stage FT = ch1/train_xhc_ft.sh): only the set 'full' is used. Paths under the work directory SCR:
  XFT_BASE  root of the warm-start FULL base (frame_mlp/members/FULL_*, temporal/full/weights/tR_w9d_s*.pt)  [SCR/stage11_ch1/e_cx/full]
  J         output root of the stage (c0, bank, banklab, fine-tuned members)                                    [SCR/stage11_ch1/j_dlcfs_contour]
  LI        reference-contour cache of the training labels (ch1d_liloss/li_prep.py)                              [SCR/stage11_ch1/liloss]
  FEAT21    split files / offsets / times / y_C80 / y_aux of the labelled rows. Development read stage11_ch1/feat_v21; for the
            train and holdout splits these five files are byte-identical to those of the canonical v2.2 cache e_cx/h150/feat
            (generated data of weights/GENERATED.sha256), which is the package default.                         [SCR/stage11_ch1/e_cx/h150/feat]
  D3D_TRAIN_DIR  the public diii_d_train parquet files (labels of the training shots)                            [WS/data_full/diii_d_train]
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
SCR = Path(os.environ["SCR"])
S = SCR / "stage11_ch1"
CX = S / "e_cx"
J = Path(os.environ.get("J", str(S / "j_dlcfs_contour")))
LI = Path(os.environ.get("LI", str(S / "liloss")))
XFT_BASE = Path(os.environ.get("XFT_BASE", str(CX / "full")))
FEAT21 = Path(os.environ.get("FEAT21", str(CX / "h150/feat")))
D3D_TRAIN_DIR = Path(os.environ.get("D3D_TRAIN_DIR", str(WS / "data_full/diii_d_train")))
HC = SCR / "stage6_H"
STARTER = WS / "starter"
VMAX = 320
NG = 65


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def set_info(name):
    """-> dict(feat, members_root, prefix, trainset, metric)."""
    if name == "full":
        return dict(feat=CX / "h150/feat", members=XFT_BASE / "frame_mlp/members", prefix="FULL_", trainset="full",
                    metric=S / "metric")
    return dict(feat=CX / name / "feat", members=CX / name / "frame_mlp/members", prefix="F_", trainset="t100",
                metric=CX / name / "metric")


def member_names(prefix):
    return ([f"{prefix}d30aux_s{s}" for s in range(5)] + [f"{prefix}towerd30aux_s{s}" for s in range(3)]
            + [f"{prefix}K120d30aux_s{s}" for s in range(3)])


def bank_rows(name):
    """Row offsets of the set's bank (= the rows the members train on) and their index into concat(li train, li holdout).
    -> (off (S+1,), li_idx (N,), files list)"""
    info = set_info(name)
    off_v21 = np.load(FEAT21 / "train/offsets.npy").astype(np.int64)
    ntr_v21 = int(off_v21[-1])
    off_tr = np.load(info["feat"] / "train/offsets.npy").astype(np.int64)
    files = [l for l in (info["feat"] / "train/files.txt").read_text().split() if l]
    if name in ("h150", "full"):
        assert np.array_equal(off_tr, off_v21)
        idx = np.arange(ntr_v21, dtype=np.int64)
        off = off_tr
        if name == "full":
            off_h = np.load(info["feat"] / "holdout/offsets.npy").astype(np.int64)
            idx = np.concatenate([idx, ntr_v21 + np.arange(int(off_h[-1]), dtype=np.int64)])
            off = np.concatenate([off_tr, off_tr[-1] + off_h[1:]])
            files = files + [l for l in (info["feat"] / "holdout/files.txt").read_text().split() if l]
    else:
        keep = np.load(S / name / "train_keep_idx.npy").astype(np.int64)
        idx = np.concatenate([np.arange(off_v21[i], off_v21[i + 1], dtype=np.int64) for i in keep])
        off = off_tr
        v21_files = [l for l in (FEAT21 / "train/files.txt").read_text().split() if l]
        assert [v21_files[i] for i in keep] == files, "fold train files != feat_v21 train files[keep]"
    assert len(idx) == int(off[-1]), (len(idx), int(off[-1]))
    return off, idx, files


def grid():
    m = np.load(STARTER / "fusion_scoring/masks/d3d_envelope.npz")
    return m["grid_R"].astype(np.float64), m["grid_Z"].astype(np.float64), m["mask_coarse"].astype(bool)


def pca(K):
    pc = np.load(HC / f"pca_{K}.npz")
    return pc["mean_"].astype(np.float64), pc["comps_"].astype(np.float64)
