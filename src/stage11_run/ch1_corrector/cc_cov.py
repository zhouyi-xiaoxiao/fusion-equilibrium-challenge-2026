#!/usr/bin/env python3
"""OOF coefficient-error covariance + per-coefficient RMS (FD steps) of a model family.
  cc_cov.py --coef <coef_oof.npy (N_train,80)> --tag T  ->  cov_oof_<T>.npy, steps_<T>.npy"""
import argparse

import numpy as np

import cc_common as cc

ap = argparse.ArgumentParser()
ap.add_argument("--coef", required=True); ap.add_argument("--tag", required=True)
a = ap.parse_args()
e = np.asarray(np.load(a.coef, mmap_mode="r"), np.float64) - np.asarray(np.load(cc.FEAT / "train/y_C80.npy", mmap_mode="r"), np.float64)
cov = e.T @ e / len(e)
np.save(cc.OUT / f"cov_oof_{a.tag}.npy", cov)
np.save(cc.OUT / f"steps_{a.tag}.npy", np.sqrt(np.diag(cov)))
w = np.linalg.eigvalsh(cov)
print(f"[cov {a.tag}] rho = sqrt(E|e|^2) = {np.sqrt(np.trace(cov)):.5f}; rms_k min/med/max {np.sqrt(np.diag(cov)).min():.5f}/{np.median(np.sqrt(np.diag(cov))):.5f}/{np.sqrt(np.diag(cov)).max():.5f}; cond {w.max()/w.min():.1f}", flush=True)
