#!/usr/bin/env python3
"""
Inputs-only predict path: FEATURE-V2.1 for any directory of DIII-D parquet shots (e.g. the Final
`diii_d_private_test`).  Reads no target column and needs no stage6 cache — only scaler_v21.npz
(written by prep_v21.py from TRAIN rows).

  python build_unseen.py --in-dir DIR --out OUT [--scaler .../feat_v21/scaler_v21.npz]
        [--files list.txt] [--workers 70] [--limit N] [--verify-against .../feat_v21/test]

Row order = sorted file names (or the order given in --files); frames in efit_times order (ALL frames).
Outputs in OUT: X.npy (standardised float32), X_raw.npy, offsets.npy, shot_id.npy, times.npy, files.txt,
build_unseen.json (per-shot Thomson reader diagnostics incl. samples dropped by the validity rule).
--verify-against: assert bit-equality of X_raw / X with an existing cache for the shared files.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import multiprocessing as mp
import numpy as np

WS = Path(os.environ["WS"])
HERE = Path(__file__).resolve().parent
for p in (WS / "starter", WS / "starter/fusion_scoring", WS / "stage2_run", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SCR = Path(os.environ["SCR"])


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def work(args):
    i, path = args
    try:
        from feat_v21 import features_v21, read_inputs
        row = read_inputs(path)
        X, diag = features_v21(row, return_diag=True)
        return {"i": i, "ok": True, "X": X, "t": np.asarray(row["efit_times"], dtype=np.float64), "diag": diag}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scaler", default=str(SCR / "stage11_ch1/feat_v21/scaler_v21.npz"))
    ap.add_argument("--files", default=None)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "70")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--verify-against", default=None)
    a = ap.parse_args()
    from feat_v21 import feature_names, load_scaler, standardise
    t0 = time.time()
    ind, out = Path(a.in_dir), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.files:
        files = [ind / l.strip() for l in Path(a.files).read_text().split("\n") if l.strip()]
    else:
        files = sorted(ind.glob("*.parquet"))
    if a.limit:
        files = files[:a.limit]
    assert files, f"no parquet files in {ind}"
    mean, scale, is_v1, clip = load_scaler(a.scaler)
    D = len(feature_names())
    assert len(mean) == D, (len(mean), D)
    res = [None] * len(files)
    with mp.get_context("spawn").Pool(min(a.workers, len(files)), initializer=_init) as pool:
        for r in pool.imap_unordered(work, [(i, str(f)) for i, f in enumerate(files)], chunksize=1):
            if not r["ok"]:
                raise SystemExit(f"FAILED {files[r['i']].name}: {r['err']}\n{r['tb']}")
            res[r["i"]] = r
    Xraw = np.concatenate([r["X"] for r in res]).astype(np.float32)
    off = np.concatenate([[0], np.cumsum([len(r["X"]) for r in res])]).astype(np.int64)
    X = np.concatenate([standardise(Xraw[s:s + 100_000], mean, scale, is_v1, clip)
                        for s in range(0, len(Xraw), 100_000)]) if len(Xraw) else np.zeros((0, D), np.float32)
    np.save(out / "X_raw.npy", Xraw)
    np.save(out / "X.npy", X)
    np.save(out / "offsets.npy", off)
    np.save(out / "shot_id.npy", np.repeat(np.arange(len(res), dtype=np.int32), np.diff(off)))
    np.save(out / "times.npy", np.concatenate([r["t"] for r in res]))
    (out / "files.txt").write_text("\n".join(f.name for f in files) + "\n")
    info = {"in_dir": str(ind), "shots": len(files), "frames": int(off[-1]), "D": D, "scaler": a.scaler,
            "seconds": round(time.time() - t0, 1),
            "thomson": {f.name: {"layout": r["diag"]["thomson"], "stats": r["diag"]["thomson_stats"]}
                        for f, r in zip(files, res)
                        if any(s.get("n_dropped", 0) for s in r["diag"]["thomson_stats"].values())
                        or any(not str(v).startswith("ok") for v in r["diag"]["thomson"].values())
                        or len(r["diag"]["thomson"]) < 2},
            "missing_signals": {f.name: r["diag"]["missing_signals"] for f, r in zip(files, res)
                                if r["diag"]["missing_signals"]}}
    if a.verify_against:
        V = Path(a.verify_against)
        vf = V.joinpath("files.txt").read_text().split()
        voff = np.load(V / "offsets.npy")
        vraw = np.load(V / "X_raw.npy", mmap_mode="r")
        vstd = np.load(V / "X.npy", mmap_mode="r")
        pos = {n: i for i, n in enumerate(vf)}
        nchk, ok_raw, ok_std = 0, True, True
        for k, f in enumerate(files):
            if f.name not in pos:
                continue
            j = pos[f.name]
            nchk += 1
            ok_raw &= bool(np.array_equal(Xraw[off[k]:off[k + 1]], vraw[voff[j]:voff[j + 1]], equal_nan=True))
            ok_std &= bool(np.array_equal(X[off[k]:off[k + 1]], vstd[voff[j]:voff[j + 1]]))
        info["verify"] = {"against": str(V), "shots_checked": nchk, "X_raw_equal": ok_raw, "X_std_equal": ok_std}
        print("VERIFY", json.dumps(info["verify"]), flush=True)
        if not (nchk and ok_raw and ok_std):
            (out / "build_unseen.json").write_text(json.dumps(info, indent=1))
            raise SystemExit("VERIFY FAILED")
    (out / "build_unseen.json").write_text(json.dumps(info, indent=1))
    print(f"BUILD_UNSEEN DONE shots={len(files)} frames={int(off[-1])} D={D} sec={info['seconds']}", flush=True)


if __name__ == "__main__":
    main()
