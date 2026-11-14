#!/usr/bin/env python3
"""Stage 11e / Ch1 FEATURE-V2.2 = FEATURE-V2.1 with the corrected DIII-D Ip time axis (ip_axis.py).

Identical code path to feat_v21.features_v21 (D = 789, same column names/order); the only change is that every Ip-derived
quantity (mag Ip history, ipn coil/Ip ratios, time/phase block, phys proxies, drive bank Ip/ECOILA-at-breakdown terms,
v1 Ip columns) is computed on the v2.2 Ip axis.  The `rec` block is kept at its v2.1 values (rec_ip_t0_ms = start of the
v2.1 axis), so the era index the v2.1 models saw is unchanged.  Re-origined (new-era) shots are bit-identical to v2.1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = Path(os.environ["WS"])
for p in (WS / "starter", WS / "starter/fusion_scoring", WS / "stage2_run", WS / "stage11_run/ch1_features", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import ip_axis  # noqa: E402
import feat_v21 as F21  # noqa: E402
from feat_v21 import feature_names, standardise, load_scaler, read_inputs, n_features  # noqa: E402,F401

_FEATURES_V21 = F21.features_v21
NAMES = [n for n, _ in feature_names()]
REC_IPT0 = NAMES.index("rec_ip_t0_ms")


def features_v22(row, return_diag=False):
    ip_axis.install()
    info = ip_axis.ip_axis_info(row)
    out = _FEATURES_V21(row, return_diag=return_diag)
    X = out[0] if return_diag else out
    a21 = np.asarray(ip_axis.original()(row), dtype=np.float64)
    f = a21[np.isfinite(a21)]
    if len(X):
        X[:, REC_IPT0] = np.float32(f[0]) if len(f) else np.float32(0.0)
    if return_diag:
        out[1]["ip_axis"] = {"branch": info["branch"], "shift_ms": info["shift_ms"], "t_bd_v21": info["t_bd_v21"],
                             "t_bd_v22": info["t_bd_v22"], "mag_t0": info["mag_t0"]}
        return X, out[1]
    return X


def patch_feat_v21():
    """Make `from feat_v21 import features_v21` return the v2.2 builder (for build_unseen-style callers)."""
    F21.features_v21 = features_v22
    ip_axis.install()


def predict_features(path_or_row, scaler=None):
    row = read_inputs(path_or_row) if isinstance(path_or_row, (str, bytes)) or hasattr(path_or_row, "__fspath__") \
        else path_or_row
    X = features_v22(row)
    t = np.asarray(row["efit_times"], dtype=np.float64)
    if scaler is None:
        return X, t
    if isinstance(scaler, (str, bytes)) or hasattr(scaler, "__fspath__"):
        scaler = load_scaler(scaler)
    return standardise(X, *scaler), X, t
