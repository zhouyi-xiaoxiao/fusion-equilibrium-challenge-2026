#!/usr/bin/env python3
"""Stage 11e / ip_axis — FEATURE-V2.2 cache, row-aligned to the feat_v21 cache (inputs only).

  OUT=$SCR/stage11_ch1/feat_v22 python prep_v22.py

For every split (train 6,891 / holdout 150 / public test 874) the files, offsets, times and targets are those of feat_v21
(symlinked: y_*, offsets, shot_id, times, files.txt); X_raw is rebuilt with feat_v22.features_v22 from the raw parquet
(inputs only) and the frames are matched to the v2.1 rows by exact efit time (train rows were filtered by prep_v21).
Scaler (scaler_v21.npz name kept so every track loads it unchanged): v1 block = the v2.1 / stage6_H scaler_f (no clip),
other columns = NaN-aware TRAIN mean / std (ddof 1), refit on the v2.2 train X_raw.
Checks: (a) re-origined (new-era) shots: X_raw bit-identical to v2.1 in every column; (b) per-block fraction of changed
frames on the corrected shots; (c) scaler of every unchanged column equals scaler_v21 (rel 1e-9).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

import multiprocessing as mp
import numpy as np

HERE = Path(__file__).resolve().parent
WS = Path(os.environ["WS"])
for p in (WS / "starter", WS / "starter/fusion_scoring", WS / "stage2_run", WS / "stage11_run/ch1_features", HERE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
SCR = Path(os.environ["SCR"])
SRC = Path(os.environ.get("SRC", str(SCR / "stage11_ch1/feat_v21")))
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/feat_v22")))
FULL_DIR = Path(os.environ.get("FULL_DIR", str(SCR / "data_full/diii_d_train")))
TEST_DIR = Path(os.environ.get("TEST_DIR", str(SCR / "data_full/diii_d_public_test")))
WORKERS = int(os.environ.get("WORKERS", "70"))
assert OUT.resolve() != SRC.resolve()
LINK = ["y_C80.npy", "y_C120.npy", "y_q95.npy", "y_betaN.npy", "y_aux.npy", "offsets.npy", "shot_id.npy", "times.npy", "files.txt"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    import feat_v22  # noqa: F401
    import ip_axis
    ip_axis.install()


def work(args):
    split, i, path = args
    try:
        from feat_v22 import features_v22, read_inputs
        row = read_inputs(path)
        X, diag = features_v22(row, return_diag=True)
        return {"split": split, "i": i, "ok": True, "X": X, "t": np.asarray(row["efit_times"], np.float64), "ip": diag["ip_axis"]}
    except Exception as e:  # noqa: BLE001
        return {"split": split, "i": i, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def nan_stats(Xmm, cols, chunk=100_000):          # == prep_v2.nan_stats
    D = len(cols)
    s = np.zeros(D); c = np.zeros(D)
    for a in range(0, len(Xmm), chunk):
        b = np.asarray(Xmm[a:a + chunk][:, cols], dtype=np.float64)
        f = np.isfinite(b); s += np.where(f, b, 0).sum(0); c += f.sum(0)
    mean = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    v = np.zeros(D)
    for a in range(0, len(Xmm), chunk):
        b = np.asarray(Xmm[a:a + chunk][:, cols], dtype=np.float64)
        f = np.isfinite(b); v += np.where(f, (b - mean) ** 2, 0).sum(0)
    std = np.sqrt(np.where(c > 1, v / np.maximum(c - 1, 1), 0.0))
    return mean, std, c


def main():
    t0 = time.time()
    from feat_v21 import feature_names, standardise
    fn = feature_names()
    blocks = [b for _, b in fn]
    blk = {}
    for j, b in enumerate(blocks):
        blk.setdefault(b, [j, j + 1]); blk[b][1] = j + 1
    D = len(fn)
    splits = {}
    for sp in ("train", "holdout", "test"):
        files = [l for l in (SRC / sp / "files.txt").read_text().split() if l]
        off = np.load(SRC / sp / "offsets.npy").astype(np.int64)
        tim = np.load(SRC / sp / "times.npy")
        d = TEST_DIR if sp == "test" else FULL_DIR
        splits[sp] = dict(files=files, off=off, times=tim, dir=d)
        (OUT / sp).mkdir(parents=True, exist_ok=True)
        log(f"{sp}: {len(files)} shots, {int(off[-1])} frames")
    Xm = {sp: np.lib.format.open_memmap(OUT / sp / "X_raw.npy", mode="w+", dtype=np.float32, shape=(int(s["off"][-1]), D))
          for sp, s in splits.items()}
    ipinfo = {sp: [None] * len(s["files"]) for sp, s in splits.items()}
    tasks = [(sp, i, str(splits[sp]["dir"] / f)) for sp in ("holdout", "test", "train") for i, f in enumerate(splits[sp]["files"])]
    failed, mism = [], []
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(WORKERS, initializer=_init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=2):
            done += 1
            if not r["ok"]:
                failed.append(r); log(f"FAILED {r['split']} {r['i']}: {r['err']}\n{r['tb']}"); continue
            s = splits[r["split"]]; i = r["i"]
            a, b = s["off"][i], s["off"][i + 1]
            tv = s["times"][a:b]
            if len(r["t"]) == b - a and np.array_equal(r["t"], tv):
                sel = np.arange(len(tv))
            else:
                pos = {float(x): k for k, x in enumerate(r["t"])}
                sel = np.array([pos.get(float(x), -1) for x in tv])
                if (sel < 0).any():
                    mism.append((r["split"], i, int((sel < 0).sum()))); continue
            Xm[r["split"]][a:b] = r["X"][sel]
            ipinfo[r["split"]][i] = r["ip"]
            if done % 1000 == 0:
                log(f"  {done}/{len(tasks)} ({len(failed)} failed, {len(mism)} mismatched) {time.time()-t0:.0f}s")
    for x in Xm.values():
        x.flush()
    if failed or mism:
        (OUT / "prep_failed.json").write_text(json.dumps({"failed": [(r["split"], r["i"], r["err"]) for r in failed], "mism": mism}))
        raise SystemExit(f"ABORT {len(failed)} failed {len(mism)} mismatched")
    log(f"feature pass done {time.time()-t0:.0f}s")
    br_code = {"reorigined": 0, "kept_unchanged": 1, "old_era_8192": 2, "fallback_tbd": 3}
    rep = {"splits": {}}
    for sp, s in splits.items():
        br = np.array([br_code[x["branch"]] for x in ipinfo[sp]])
        sh = np.array([x["shift_ms"] for x in ipinfo[sp]])
        np.savez(OUT / sp / "ip_axis.npz", branch=br, shift_ms=sh, t_bd_v21=np.array([x["t_bd_v21"] for x in ipinfo[sp]]),
                 t_bd_v22=np.array([x["t_bd_v22"] for x in ipinfo[sp]]), mag_t0=np.array([x["mag_t0"] for x in ipinfo[sp]]))
        # ---- diff vs v2.1 per block, split by branch
        Xo = np.load(SRC / sp / "X_raw.npy", mmap_mode="r")
        Xn = Xm[sp]
        fr_br = np.repeat(br, np.diff(s["off"]))
        ch_blk = {b: np.zeros(len(fr_br), bool) for b in blk}
        for a in range(0, len(fr_br), 100_000):
            e = min(len(fr_br), a + 100_000)
            A = np.asarray(Xn[a:e], np.float64); B = np.asarray(Xo[a:e], np.float64)
            nA, nB = np.isnan(A), np.isnan(B)
            dd = (nA != nB) | (~nA & ~nB & (A != B))
            for b, (c0, c1) in blk.items():
                ch_blk[b][a:e] = dd[:, c0:c1].any(1)
        anych = np.zeros(len(fr_br), bool)
        for v in ch_blk.values():
            anych |= v
        r = {"shots": len(br), "branches": {k: int((br == v).sum()) for k, v in br_code.items()},
             "shift_ms_pct_5_50_95": [float(x) for x in np.percentile(sh[br >= 2], [5, 50, 95])] if (br >= 2).any() else None,
             "reorigined_frames_changed_any_column": int(anych[fr_br == 0].sum()),
             "corrected_frames": int((fr_br >= 2).sum()),
             "corrected_frames_changed_by_block": {b: float(v[fr_br >= 2].mean()) for b, v in ch_blk.items()} if (fr_br >= 2).any() else {}}
        rep["splits"][sp] = r
        log(f"{sp}: {json.dumps(r)}")
        assert r["reorigined_frames_changed_any_column"] == 0, "re-origined shots must be bit-identical to v2.1"
        for f in LINK:
            dst = OUT / sp / f
            if dst.is_symlink() or dst.exists():
                dst.unlink()
            os.symlink(SRC / sp / f, dst)
    # ---- scaler (train only)
    s0 = np.load(SRC / "scaler_v21.npz")
    is_v1 = s0["is_v1"].astype(bool)
    mean = s0["mean"].astype(np.float64).copy(); scale = s0["scale"].astype(np.float64).copy()
    nv = np.flatnonzero(~is_v1)
    mu, sd, _ = nan_stats(Xm["train"], nv)
    mean[nv] = mu; scale[nv] = np.where(sd > 1e-8, sd, 1.0)
    np.savez(OUT / "scaler_v21.npz", mean=mean, scale=scale, is_v1=is_v1, clip=float(s0["clip"]), names=s0["names"])
    m0, sc0 = s0["mean"].astype(np.float64), s0["scale"].astype(np.float64)
    relm = np.abs(mean - m0) / np.maximum(np.abs(m0), 1e-6); rels = np.abs(scale - sc0) / sc0
    changed_cols = np.flatnonzero((relm > 1e-9) | (rels > 1e-9))
    names = [n for n, _ in fn]
    rep["scaler"] = {"changed_columns": len(changed_cols), "changed_names_head": [names[j] for j in changed_cols[:200]],
                     "changed_blocks": sorted({blocks[j] for j in changed_cols})}
    log(f"scaler: {len(changed_cols)} columns differ from scaler_v21 (blocks {rep['scaler']['changed_blocks']})")
    for sp in splits:
        Xr = np.load(OUT / sp / "X_raw.npy", mmap_mode="r")
        mm = np.lib.format.open_memmap(OUT / sp / "X.npy", mode="w+", dtype=np.float32, shape=Xr.shape)
        for a in range(0, Xr.shape[0], 100_000):
            mm[a:a + 100_000] = standardise(Xr[a:a + 100_000], mean, scale, is_v1, float(s0["clip"]))
        mm.flush(); del mm
        # re-origined rows x columns with an unchanged scaler must reproduce v2.1 X bit-for-bit
        Xo = np.load(SRC / sp / "X.npy", mmap_mode="r")
        Xn = np.load(OUT / sp / "X.npy", mmap_mode="r")
        same_cols = np.setdiff1d(np.arange(D), changed_cols)
        frb = np.repeat(np.load(OUT / sp / "ip_axis.npz")["branch"], np.diff(splits[sp]["off"])) == 0
        eq = True
        for a in range(0, Xr.shape[0], 200_000):
            m = frb[a:a + 200_000]
            if m.any():
                eq &= bool(np.array_equal(np.asarray(Xn[a:a + 200_000])[m][:, same_cols], np.asarray(Xo[a:a + 200_000])[m][:, same_cols]))
        rep["splits"][sp]["X_std_unchanged_columns_bit_identical"] = bool(eq)
        log(f"{sp}: standardised; re-origined rows x unchanged-scaler columns bit-identical to v2.1 X: {eq}")
    import shutil
    shutil.copy(SRC / "names.txt", OUT / "names.txt")
    spec = json.load(open(SRC / "spec.json"))
    spec["version"] = "feat_v22 (stage11e ip_axis) 2026-10-12: feat_v21 + corrected DIII-D Ip time axis (old-era +8192 ms rule)"
    spec["ip_axis"] = rep
    json.dump(spec, open(OUT / "spec.json", "w"), indent=1)
    json.dump(rep, open(OUT / "prep_v22_report.json", "w"), indent=1)
    log(f"PREP_V22 DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
