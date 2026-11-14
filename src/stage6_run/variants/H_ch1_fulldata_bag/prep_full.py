#!/usr/bin/env python3
"""
Variant H_ch1_fulldata_bag — step 0: feature/PCA cache over the FULL DIII-D train
download (data_full/diii_d_train, 7,041 shots) minus the standard 150-shot holdout.

Holdout (unchanged from stage 2 / round 1, so scores stay comparable):
    sorted(S6_SHOT_DIR)[S6_N_TRAIN : S6_N_TRAIN+S6_N_HOLD]   (S6_SHOT_DIR = the original
    1,150-file dir; default [1000:1150]).  The same filenames exist in data_full.
Train = every parquet in S6_FULL_DIR whose name is NOT in the holdout (split BY SHOT).
Optional S6_MAX_TRAIN=n keeps only the first n train shots (smoke tests).

Same protocol as variant D's prep_cache.py: full_features_d3d (~90 dims) -> StandardScaler
(train only); FluxPCA (randomized SVD, seed 42) for n_components in S6_PCA_SIZES (80,120)
fitted on a random frame subsample (S6_PCA_FIT_FRAMES, default 120k); train psi projected
shot by shot in a second parallel pass. Reads run in a multiprocessing pool (S6_WORKERS).

Outputs (S6_OUT_DIR):
  Xfs.npy (F,90) f32 | C80.npy C120.npy (F,k) f32 | q95.npy betaN.npy shot_id.npy
  cache_train.npz  (same keys, for D-compatible tools)   [written only if S6_WRITE_NPZ=1]
  cache_hold.npz   Xhs, offsets, times, q95, betaN, files     (assemble.py format)
  cache_hold_psi.npy   holdout GT psi (F_h,65,65) f32 (local R2 only; official = harness)
  pca_<k>.npz  mean_, comps_, evr_
  prep_diag.json, train_files.txt, holdout_files.txt, failed_files.txt
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

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "starter"))
sys.path.insert(0, str(WS / "stage2_run"))

SHOT_DIR = Path(os.environ.get("S6_SHOT_DIR", str(WS / "scaleup_run/data/diii_d_train")))
FULL_DIR = Path(os.environ.get("S6_FULL_DIR", str(WS / "data_full/diii_d_train")))
N_TRAIN = int(os.environ.get("S6_N_TRAIN", "1000"))
N_HOLD = int(os.environ.get("S6_N_HOLD", "150"))
MAX_TRAIN = int(os.environ.get("S6_MAX_TRAIN", "0"))
OUT = Path(os.environ.get("S6_OUT_DIR", os.getcwd()))
PCA_SIZES = [int(x) for x in os.environ.get("S6_PCA_SIZES", "80,120").split(",")]
PCA_FIT_FRAMES = int(os.environ.get("S6_PCA_FIT_FRAMES", "120000"))
WORKERS = int(os.environ.get("S6_WORKERS", "48"))
SEED = 42


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- workers
_G = {}


def _init_worker(comps=None):
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _G["comps"] = comps


def _load(path):
    import matplotlib
    matplotlib.use("Agg")
    from stage2_features import full_features_d3d, load_d3d_shot, thomson_summary
    shot = load_d3d_shot(path)
    t = shot["efit_times"]
    th = thomson_summary(shot, t)
    Xf = full_features_d3d(shot, thomson=th)
    psi = shot["efit_psirz"].astype(np.float32)
    times = np.asarray(t, dtype=np.float64)
    q95 = np.asarray(shot["scalars"].get("efit_q95", np.full(len(t), np.nan)), dtype=np.float64)
    betaN = np.asarray(shot["scalars"].get("efit_beta_n", np.full(len(t), np.nan)), dtype=np.float64)
    n = min(len(Xf), len(psi))
    Xf, psi, times, q95, betaN = Xf[:n], psi[:n], times[:n], q95[:n], betaN[:n]
    valid = np.isfinite(Xf).all(1) & np.isfinite(psi.reshape(n, -1)).all(1)
    return Xf, psi, times, q95, betaN, valid


def pass1(args):
    """(idx, path, sub_frac, is_hold) -> dict (no psi for train except a random subsample)."""
    idx, path, sub_frac, is_hold = args
    try:
        Xf, psi, times, q95, betaN, valid = _load(path)
        out = {"idx": idx, "file": Path(path).name, "ok": True, "n": len(psi),
               "n_valid": int(valid.sum())}
        if is_hold:
            out.update(Xf=Xf, psi=psi, times=times, q95=q95, betaN=betaN)
        else:
            rng = np.random.default_rng(SEED * 1_000_003 + idx)
            pick = np.flatnonzero(valid & (rng.random(len(psi)) < sub_frac))
            out.update(Xf=Xf[valid], q95=q95[valid], betaN=betaN[valid], sub_psi=psi[pick])
        return out
    except Exception as e:  # noqa: BLE001
        return {"idx": idx, "file": Path(path).name, "ok": False,
                "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}


def pass2(args):
    """(idx, path) -> (idx, {k: coeffs (n_valid,k) f32}) using the fitted PCA bases."""
    idx, path = args
    Xf, psi, *_r, valid = _load(path)
    psi = psi[valid].reshape(int(valid.sum()), -1)
    res = {}
    for k, (mean_, comps_) in _G["comps"].items():
        res[k] = ((psi - mean_[None, :]) @ comps_.T).astype(np.float32)
    return idx, res


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    orig = sorted(SHOT_DIR.glob("*.parquet"))
    if len(orig) < N_TRAIN + N_HOLD:
        raise SystemExit(f"need {N_TRAIN+N_HOLD} parquets in {SHOT_DIR}, found {len(orig)}")
    hold_names = [f.name for f in orig[N_TRAIN:N_TRAIN + N_HOLD]]
    hold_set = set(hold_names)
    full = sorted(FULL_DIR.glob("*.parquet"))
    full_names = {f.name for f in full}
    missing = [n for n in hold_names if n not in full_names]
    if missing:
        raise SystemExit(f"{len(missing)} holdout files missing from {FULL_DIR}: {missing[:5]}")
    train_files = [f for f in full if f.name not in hold_set]
    if MAX_TRAIN:
        train_files = train_files[:MAX_TRAIN]
    hold_files = [FULL_DIR / n for n in hold_names]  # same order as sorted(SHOT_DIR)[N_TRAIN:]
    (OUT / "train_files.txt").write_text("\n".join(f.name for f in train_files) + "\n")
    (OUT / "holdout_files.txt").write_text("\n".join(hold_names) + "\n")
    log(f"full dir {FULL_DIR}: {len(full)} parquets; holdout {len(hold_names)} = sorted({SHOT_DIR.name})[{N_TRAIN}:{N_TRAIN+N_HOLD}]; "
        f"train = {len(train_files)} shots (full minus holdout{' , capped' if MAX_TRAIN else ''})")

    # frames per shot ~220 -> subsample fraction for the PCA fit
    est_frames = 220 * len(train_files)
    sub_frac = min(1.0, PCA_FIT_FRAMES / max(est_frames, 1))
    tasks = [(i, str(f), sub_frac, False) for i, f in enumerate(train_files)] + \
            [(i, str(f), 0.0, True) for i, f in enumerate(hold_files)]
    os.environ["OMP_NUM_THREADS"] = "1"; os.environ["OPENBLAS_NUM_THREADS"] = "1"; os.environ["MKL_NUM_THREADS"] = "1"
    ctx = mp.get_context("spawn")
    log(f"pass 1: {len(tasks)} shots on {WORKERS} workers (pca sub_frac={sub_frac:.4f})")
    tr = [None] * len(train_files)
    hold = [None] * len(hold_files)
    failed = []
    done = 0
    with ctx.Pool(WORKERS, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(pass1, tasks, chunksize=4):
            done += 1
            if not r["ok"]:
                failed.append(r)
            elif "psi" in r:
                hold[r["idx"]] = r
            else:
                tr[r["idx"]] = r
            if done % 500 == 0:
                log(f"  {done}/{len(tasks)} ({len(failed)} failed) {time.time()-t0:.0f}s")
    if failed:
        (OUT / "failed_files.txt").write_text("\n".join(f"{r['file']}\t{r['err']}" for r in failed) + "\n")
        log(f"WARNING: {len(failed)} shots failed to load (listed in failed_files.txt): "
            + "; ".join(f"{r['file']}: {r['err']}" for r in failed[:5]))
    if any(h is None for h in hold):
        raise SystemExit("a HOLDOUT shot failed to load -> abort (holdout must stay exactly the standard 150)")
    keep = [i for i, r in enumerate(tr) if r is not None]
    train_files = [train_files[i] for i in keep]
    tr = [tr[i] for i in keep]
    (OUT / "train_files.txt").write_text("\n".join(f.name for f in train_files) + "\n")

    Xf = np.concatenate([r["Xf"] for r in tr])
    q95 = np.concatenate([r["q95"] for r in tr]); betaN = np.concatenate([r["betaN"] for r in tr])
    shot_id = np.concatenate([np.full(len(r["Xf"]), i, dtype=np.int32) for i, r in enumerate(tr)])
    sub_psi = np.concatenate([r["sub_psi"] for r in tr if len(r["sub_psi"])])
    n_dropped = sum(r["n"] - r["n_valid"] for r in tr)
    for r in tr:
        r.pop("Xf"); r.pop("sub_psi")
    log(f"train matrix: Xf={Xf.shape} from {len(tr)} shots (dropped {n_dropped} non-finite rows); PCA subsample={sub_psi.shape}")

    from sklearn.preprocessing import StandardScaler
    scaler_f = StandardScaler().fit(Xf)
    Xfs = scaler_f.transform(Xf).astype(np.float32); del Xf
    np.save(OUT / "Xfs.npy", Xfs); np.save(OUT / "q95.npy", q95); np.save(OUT / "betaN.npy", betaN)
    np.save(OUT / "shot_id.npy", shot_id)
    np.savez(OUT / "scaler_f.npz", mean_=scaler_f.mean_, scale_=scaler_f.scale_)

    Xh = [scaler_f.transform(np.nan_to_num(h["Xf"])).astype(np.float32) for h in hold]
    offsets = np.cumsum([0] + [len(x) for x in Xh]).astype(np.int64)
    Xhs = np.concatenate(Xh)
    np.savez_compressed(OUT / "cache_hold.npz", Xhs=Xhs, offsets=offsets,
                        times=np.concatenate([h["times"] for h in hold]),
                        q95=np.concatenate([h["q95"] for h in hold]),
                        betaN=np.concatenate([h["betaN"] for h in hold]),
                        files=np.array([h["file"] for h in hold]))
    np.save(OUT / "cache_hold_psi.npy", np.concatenate([h["psi"] for h in hold]).astype(np.float32))
    log(f"hold matrix: Xhs={Xhs.shape} over {len(hold)} shots -> cache_hold.npz (+cache_hold_psi.npy)")
    del hold, Xh

    diag = {"n_train_shots": len(tr), "n_train_frames": int(len(Xfs)), "n_hold_frames": int(len(Xhs)),
            "n_hold_shots": int(len(offsets) - 1), "n_failed_shots": len(failed), "pca": {},
            "pca_fit_frames": int(len(sub_psi)), "full_dir_files": len(full), "holdout_first": hold_names[0],
            "holdout_last": hold_names[-1]}

    from stage2_features import FluxPCA
    comps = {}
    for k in PCA_SIZES:
        t = time.time()
        p = FluxPCA(k).fit(sub_psi, max_fit_rows=len(sub_psi), seed=SEED)
        comps[k] = (p.mean_, p.comps_)
        np.savez(OUT / f"pca_{k}.npz", mean_=p.mean_, comps_=p.comps_, evr_=p.evr_)
        diag["pca"][str(k)] = {"evr_cum": float(np.cumsum(p.evr_)[-1]), "fit_s": round(time.time() - t, 1)}
        log(f"  PCA{k}: cum evr={np.cumsum(p.evr_)[-1]:.6f} ({time.time()-t:.0f}s)")
    del sub_psi

    log(f"pass 2: projecting {len(train_files)} train shots onto the PCA bases")
    Cs = {k: np.empty((len(Xfs), k), dtype=np.float32) for k in PCA_SIZES}
    starts = np.concatenate([[0], np.cumsum([r["n_valid"] for r in tr])])
    done = 0
    with ctx.Pool(WORKERS, initializer=_init_worker, initargs=(comps,)) as pool:
        for idx, res in pool.imap_unordered(pass2, [(i, str(f)) for i, f in enumerate(train_files)], chunksize=4):
            a, b = starts[idx], starts[idx + 1]
            for k in PCA_SIZES:
                assert len(res[k]) == b - a, (idx, len(res[k]), b - a)
                Cs[k][a:b] = res[k]
            done += 1
            if done % 500 == 0:
                log(f"  {done}/{len(train_files)} {time.time()-t0:.0f}s")
    for k in PCA_SIZES:
        np.save(OUT / f"C{k}.npy", Cs[k])
    if os.environ.get("S6_WRITE_NPZ", "0") == "1":
        np.savez(OUT / "cache_train.npz", Xfs=Xfs, q95=q95, betaN=betaN, shot_id=shot_id,
                 **{f"C{k}": Cs[k] for k in PCA_SIZES})
    diag["minutes"] = round((time.time() - t0) / 60, 2)
    (OUT / "prep_diag.json").write_text(json.dumps(diag, indent=1))
    log(f"PREP DONE in {diag['minutes']} min: {diag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
