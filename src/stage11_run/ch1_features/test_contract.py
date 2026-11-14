#!/usr/bin/env python3
"""
Robustness tests of the FEATURE-V2.1 Thomson reader against the 2026-10-11 validity contract and the
two defects announced for the Final inputs (one DIII-D edge array padded with a constant non-zero
time; one core array with an out-of-order sample).  Real shots are read (inputs only), their Thomson
arrays are corrupted in memory, and features_v21 must be EXACTLY equal to the uncorrupted features.
The same corruptions are applied to features_v2 to document v2's exposure (informational).

Also: dict row == pandas row; Hugging-Face style nested-list row == pandas row; NaN instead of 0 as the
"no measurement" flag gives identical features; reader output times strictly increasing.
Writes $OUT/contract_tests.json; exit code 1 if any v2.1 test fails.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
HERE = Path(__file__).resolve().parent
for p in (WS / "starter", WS / "starter/fusion_scoring", WS / "stage2_run", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SCR = Path(os.environ["SCR"])
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/feat_v21")))
FULL_DIR = Path(os.environ.get("FULL_DIR", str(SCR / "data_full/diii_d_train")))
TEST_DIR = Path(os.environ.get("TEST_DIR", str(SCR / "data_full/diii_d_public_test")))


def obj(rows):
    a = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        a[i] = np.asarray(r, dtype=np.float64)
    return a


def pad_edge(d, c, K=40):
    """Append K padding samples (time c, Te = ne = 0) to the edge system."""
    d = dict(d)
    C = len(d["thomson_edge_Te"][0])
    d["thomson_edge_times"] = np.r_[np.asarray(d["thomson_edge_times"], np.float64), np.full(K, c)]
    d["thomson_edge_Te"] = obj(list(d["thomson_edge_Te"]) + [np.zeros(C)] * K)
    d["thomson_edge_ne"] = obj(list(d["thomson_edge_ne"]) + [np.zeros(C)] * K)
    return d


def ooo_core(d, t_mid, time_value=None, seed=0):
    """Insert ONE out-of-order sample carrying non-zero garbage into the core system near t_mid."""
    d = dict(d)
    rng = np.random.default_rng(seed)
    tt = np.asarray(d["thomson_core_times"], np.float64)
    k = int(np.clip(np.searchsorted(tt, t_mid), 2, len(tt) - 2))
    C = len(d["thomson_core_Te"][0])
    tv = tt[k - 1] - 37.0 if time_value is None else time_value
    Te, ne = list(d["thomson_core_Te"]), list(d["thomson_core_ne"])
    Te.insert(k, rng.uniform(3000, 9000, C))
    ne.insert(k, rng.uniform(5e19, 1.5e20, C))
    d["thomson_core_times"] = np.insert(tt, k, tv)
    d["thomson_core_Te"], d["thomson_core_ne"] = obj(Te), obj(ne)
    return d


def zeros_to_nan(d):
    d = dict(d)
    for sk in ("core", "edge"):
        for q in ("Te", "ne"):
            rows = [np.where(np.asarray(p, np.float64) == 0, np.nan, np.asarray(p, np.float64))
                    for p in d[f"thomson_{sk}_{q}"]]
            d[f"thomson_{sk}_{q}"] = obj(rows)
    return d


def to_lists(d):
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out[k] = [np.asarray(p).tolist() for p in v] if v.dtype == object else v.tolist()
        else:
            out[k] = v
    return out


def nchg(A, B):
    nA, nB = np.isnan(A), np.isnan(B)
    d = (nA != nB) | (~nA & ~nB & (A != B))
    return int(d.any(1).sum())


def main():
    from feat_v2 import features_v2, read_inputs
    from feat_v21 import features_v21, thomson_systems_v21, thomson_valid
    assert thomson_valid([0.0, 1.0, 1.0, 0.5, 2.0, np.nan, 3.0, 2.5]).tolist() == \
        [True, True, False, False, True, False, True, False]
    fs = sorted(FULL_DIR.glob("*.parquet"))
    shots = [FULL_DIR / "d3d_shot_8076a5ac36.parquet", FULL_DIR / "d3d_shot_461ef46821.parquet",
             fs[0], fs[1234], fs[4000], sorted(TEST_DIR.glob("*.parquet"))[0]]
    res, fail = [], 0
    for p in shots:
        if not p.exists():
            continue
        row = read_inputs(str(p))
        d = {k: row[k] for k in row.index}
        t = np.asarray(d["efit_times"], np.float64)
        X0 = features_v21(row)
        V0 = features_v2(row)
        sysd, diag, st = thomson_systems_v21(d)
        mono = all(bool(np.all(np.diff(s["t"]) > 0)) for s in sysd.values())
        te = np.asarray(d["thomson_edge_times"], np.float64)
        tmid = float(np.median(t))
        cases = {
            "dict_row": d,
            "nested_list_row": to_lists(d),
            "edge_pad_const_inside": pad_edge(d, c=float(np.round(tmid)) + 0.123),
            "edge_pad_const_after_end": pad_edge(d, c=float(np.nanmax(te)) + 5000.0),
            "core_out_of_order_garbage": ooo_core(d, tmid),
            "core_nan_time_garbage": ooo_core(d, tmid, time_value=np.nan),
            "zeros_as_nan": zeros_to_nan(d),
        }
        r = {"file": p.name, "T": int(len(t)), "reader_times_strictly_increasing": mono,
             "thomson_stats": st, "v21": {}, "v2_frames_changed": {}}
        fail += (not mono)
        for name, dd in cases.items():
            n21 = nchg(features_v21(dd), X0)
            r["v21"][name] = n21
            fail += (n21 != 0)
            try:
                r["v2_frames_changed"][name] = nchg(features_v2(dd), V0)
            except Exception as e:  # noqa: BLE001
                r["v2_frames_changed"][name] = f"error:{type(e).__name__}"
        res.append(r)
        print(json.dumps(r), flush=True)
    out = {"n_shots": len(res), "v21_failures": int(fail), "results": res}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "contract_tests.json").write_text(json.dumps(out, indent=1))
    print(f"CONTRACT_TESTS {'PASS' if fail == 0 else 'FAIL'} shots={len(res)} v21_failures={fail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
