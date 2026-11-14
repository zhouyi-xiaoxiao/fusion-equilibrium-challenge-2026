#!/usr/bin/env python3
"""
Stage 11 / Ch1 FEATURE-V2.1 prep — same splits / row order / targets as prep_v2.py (see its
docstring), features from feat_v21.features_v21 (D=789, superset of the 693 v2 columns).

Additional outputs / checks against the v2 cache ($V2DIR, read-only):
  * targets (y_C80, y_C120, y_q95, y_betaN, y_aux), times, offsets: array-equal to feat_v2;
  * blocks v1/mag/ipn/time of X_raw: array-equal to feat_v2 (code path unchanged);
  * blocks th1/tsc/tscm/tse/tsem/phys: per-block count of frames / elements whose value changed under
    the Thomson validity contract (NaN-aware, rel. tol 1e-5), coverage (mask) gains;
  * per-split Thomson contract statistics (samples dropped by the monotone rule, shots affected,
    all-invalid profile fraction, Te-xor-ne elements).
Standardiser `scaler_v21.npz`: v1 block = H scaler_f (no clip); other columns NaN-aware train
mean/std, NaN -> 0, clip +-CLIP (feat_v21.standardise, the same function the predict path uses).

Env: WS OUT(=.../feat_v21) V2DIR(=.../feat_v2) HCACHE FULL_DIR TEST_DIR WORKERS(70) MAX_TRAIN MAX_HOLD
     MAX_TEST CLIP(10)
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import multiprocessing as mp
import numpy as np

import prep_v2 as P2          # helpers only (_init, frame_aux, _scal, nan_stats, log); sets sys.path
from prep_v2 import WS, SCR, HCACHE, FULL_DIR, TEST_DIR, WORKERS, MAX_TRAIN, MAX_HOLD, MAX_TEST, CLIP, \
    TARGET_COLS, AUX_NAMES, log, nan_stats

OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/feat_v21")))
V2DIR = Path(os.environ.get("V2DIR", str(SCR / "stage11_ch1/feat_v2")))
assert OUT.resolve() != V2DIR.resolve(), "OUT must not be the v2 cache"
TH_BLOCKS = ("th1", "tsc", "tscm", "tse", "tsem", "phys")
SAME_BLOCKS = ("v1", "mag", "ipn", "time")


def work(args):
    split, idx, path = args
    try:
        import pandas as pd
        from feat_v21 import features_v21, read_inputs
        from experiments import _as_psirz_stack
        t0 = time.time()
        row = read_inputs(path)
        X, diag = features_v21(row, return_diag=True)
        t = np.asarray(row["efit_times"], dtype=np.float64)
        res = {"split": split, "idx": idx, "file": Path(path).name, "ok": True, "diag": diag}
        if split == "test":
            res.update(X=X, t=t, n=len(X))
            res["sec"] = time.time() - t0
            return res
        trow = pd.read_parquet(path, columns=TARGET_COLS).iloc[0]
        psi = _as_psirz_stack(trow["efit_psirz"])                     # float32, as the harness
        n = min(len(X), len(psi))
        X, psi, t = X[:n], psi[:n], t[:n]
        if split == "train":
            sel = np.isfinite(X[:, :90]).all(1) & np.isfinite(psi.reshape(n, -1)).all(1)
        else:
            sel = np.ones(n, bool)
        q95 = P2._scal(trow, "efit_q95", n)[sel]
        bN = P2._scal(trow, "efit_beta_n", n)[sel]
        ef = np.column_stack([P2._scal(trow, c, n)[sel] for c in ("efit_li", "efit_r_axis", "efit_z_axis")])
        ps = psi[sel]
        flat = ps.reshape(len(ps), -1)
        Cs = {}
        for k, (mean_, comps_) in P2._G["pca"].items():
            Cs[k] = ((np.nan_to_num(flat) - mean_[None, :]) @ comps_.T).astype(np.float32)
        aux = np.empty((len(ps), len(AUX_NAMES)), np.float32)
        aux[:, :3] = ef
        for k in range(len(ps)):
            aux[k, 3:] = P2.frame_aux(ps[k])
        res.update(X=X[sel], t=t[sel], n=int(sel.sum()), n_all=n, q95=q95.astype(np.float32),
                   betaN=bN.astype(np.float32), C80=Cs[80], C120=Cs[120], aux=aux)
        res["sec"] = time.time() - t0
        return res
    except Exception as e:  # noqa: BLE001
        return {"split": split, "idx": idx, "file": Path(path).name, "ok": False,
                "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def write_std(Xraw, path, mean, scale, is_v1, chunk=100_000):
    from feat_v21 import standardise
    F, D = Xraw.shape
    mm = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(F, D))
    for a in range(0, F, chunk):
        mm[a:a + chunk] = standardise(Xraw[a:a + chunk], mean, scale, is_v1, CLIP)
    mm.flush()
    del mm


def diff_vs_v2(Xnew, v2_path, blk, rows_flag=None, chunk=100_000, exact=True):
    """Per-block change statistics of Xnew[:, :693] against the v2 X_raw (same rows).  rows_flag:
    optional bool (F,) marking frames of shots in which the monotone rule dropped samples."""
    if not Path(v2_path).exists():
        return {"available": False}
    Xo = np.load(v2_path, mmap_mode="r")
    F = Xnew.shape[0]
    if Xo.shape[0] < F or (exact and Xo.shape[0] != F):
        return {"available": False, "reason": f"row mismatch {Xo.shape[0]} vs {F}"}
    res = {"available": True, "frames": int(F), "blocks": {}}
    acc = {b: [0, 0, 0] for b in blk if b in SAME_BLOCKS + TH_BLOCKS}       # frames, elems, frames in flagged shots
    any_th = 0
    cov = {"tsc": [0.0, 0.0, 0, 0], "tse": [0.0, 0.0, 0, 0]}                  # old mask sum, new mask sum, 0->1, 1->0
    for a in range(0, F, chunk):
        e = min(F, a + chunk)
        A = np.asarray(Xnew[a:e, :Xo.shape[1]], np.float64)
        B = np.asarray(Xo[a:e], np.float64)
        nA, nB = np.isnan(A), np.isnan(B)
        d = (nA != nB) | (~nA & ~nB & (np.abs(A - B) > 1e-5 * np.maximum(1.0, np.abs(B))))
        th_any = np.zeros(e - a, bool)
        for b in acc:
            c0, c1 = blk[b]
            db = d[:, c0:c1]
            fr = db.any(1)
            acc[b][0] += int(fr.sum())
            acc[b][1] += int(db.sum())
            if rows_flag is not None:
                acc[b][2] += int((fr & rows_flag[a:e]).sum())
            if b in TH_BLOCKS:
                th_any |= fr
        any_th += int(th_any.sum())
        for b, ng in (("tsc", 39), ("tse", 14)):
            c0, c1 = blk[b]
            mo, mn = B[:, c1 - ng:c1], A[:, c1 - ng:c1]
            cov[b][0] += float(mo.sum()); cov[b][1] += float(mn.sum())
            cov[b][2] += int(((mo == 0) & (mn == 1)).sum()); cov[b][3] += int(((mo == 1) & (mn == 0)).sum())
    for b, (nf, ne, nfl) in acc.items():
        c0, c1 = blk[b]
        res["blocks"][b] = {"frames_changed": nf, "frac_frames": nf / max(F, 1),
                            "frac_elems": ne / max(F * (c1 - c0), 1), "frames_changed_in_drop_shots": nfl}
    res["frames_any_thomson_block_changed"] = any_th
    res["frac_frames_any_thomson_block_changed"] = any_th / max(F, 1)
    for b, ng in (("tsc", 39), ("tse", 14)):
        o, n, g, l = cov[b]
        res[f"{b}_mask_coverage"] = {"v2": o / (F * ng), "v21": n / (F * ng), "gained_points": g, "lost_points": l}
    if rows_flag is not None:
        res["frames_in_drop_shots"] = int(rows_flag.sum())
    return res


def th_stats(diags, files):
    agg = {}
    for sk in ("core", "edge"):
        tot = Counter()
        shots = []
        for d, f in zip(diags, files):
            st = (d or {}).get("thomson_stats", {}).get(sk)
            if not st:
                continue
            for k, v in st.items():
                tot[k] += v
            if st["n_dropped"]:
                shots.append([f, st["n_dropped"], st["n_dropped_with_data"]])
        agg[sk] = {**{k: int(v) for k, v in tot.items()}, "shots_with_dropped": len(shots),
                   "dropped_shots": shots[:40]}
        if tot.get("n_kept"):
            agg[sk]["frac_profiles_all_invalid"] = 1.0 - tot["n_prof_valid"] / tot["n_kept"]
            agg[sk]["frac_elems_valid"] = tot["n_elem_valid"] / max(tot["n_elem"], 1)
            agg[sk]["frac_elems_te_xor_ne"] = tot["n_elem_xor"] / max(tot["n_elem"], 1)
    return agg


def main():
    t0 = time.time()
    from feat_v21 import (feature_names, D3D_SIGNALS, ZGRID, RGRID, OFFSETS_MS, GAP_CORE_MS, GAP_EDGE_MS,
                          MAXD_CORE, MAXD_EDGE, SMOOTH_MS, DRV_TAUS_MS, DRV_DWIN_MS, DRIVE_SIGNALS, V2_COMPAT,
                          TH1_GAP_MS, D_V2)
    from experiments import D3D_MAGNETICS_SIGNALS
    assert list(D3D_MAGNETICS_SIGNALS) == list(D3D_SIGNALS)
    fn = feature_names()
    names = [n for n, _ in fn]
    blocks = [b for _, b in fn]
    D = len(names)
    assert len(set(names)) == D
    is_v1 = np.array([b == "v1" for b in blocks])
    assert is_v1.sum() == 90 and is_v1[:90].all()
    v2names = (V2DIR / "names.txt").read_text().split()
    assert names[:D_V2] == v2names, "first 693 names must equal feat_v2 names"
    blk = {}
    for j, b in enumerate(blocks):
        blk.setdefault(b, [j, j + 1])
        blk[b][1] = j + 1
    for sp in ("train", "holdout", "test"):
        (OUT / sp).mkdir(parents=True, exist_ok=True)
    tr_names = [l.strip() for l in (HCACHE / "train_files.txt").read_text().split("\n") if l.strip()]
    ho_names = [l.strip() for l in (HCACHE / "holdout_files.txt").read_text().split("\n") if l.strip()]
    te_files = sorted(TEST_DIR.glob("*.parquet"))
    H_sid = np.load(HCACHE / "shot_id.npy")
    H_counts = np.bincount(H_sid, minlength=len(tr_names))
    assert len(H_counts) == len(tr_names) == 6891 and len(ho_names) == 150, (len(H_counts), len(tr_names), len(ho_names))
    assert not (set(tr_names) & set(ho_names))
    full = not (MAX_TRAIN or MAX_HOLD or MAX_TEST)
    if MAX_TRAIN:
        tr_names = tr_names[:MAX_TRAIN]
    if MAX_HOLD:
        ho_names = ho_names[:MAX_HOLD]
    if MAX_TEST:
        te_files = te_files[:MAX_TEST]
    counts = H_counts[:len(tr_names)]
    tr_off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    F_tr = int(tr_off[-1])
    log(f"D={D} features; train {len(tr_names)} shots / {F_tr} frames (H rows); holdout {len(ho_names)}; "
        f"test {len(te_files)}; workers {WORKERS}; OUT={OUT}")

    Xtr = np.lib.format.open_memmap(OUT / "train/X_raw.npy", mode="w+", dtype=np.float32, shape=(F_tr, D))
    ytr = {"times": np.empty(F_tr, np.float64), "q95": np.empty(F_tr, np.float32),
           "betaN": np.empty(F_tr, np.float32), "C80": np.empty((F_tr, 80), np.float32),
           "C120": np.empty((F_tr, 120), np.float32), "aux": np.empty((F_tr, len(AUX_NAMES)), np.float32)}
    hold = [None] * len(ho_names)
    test = [None] * len(te_files)
    diags = {"train": [None] * len(tr_names), "holdout": [None] * len(ho_names), "test": [None] * len(te_files)}
    failed, mism = [], []
    tasks = ([("holdout", i, str(FULL_DIR / n)) for i, n in enumerate(ho_names)]
             + [("test", i, str(f)) for i, f in enumerate(te_files)]
             + [("train", i, str(FULL_DIR / n)) for i, n in enumerate(tr_names)])
    secs = []
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(WORKERS, initializer=P2._init) as pool:
        for r in pool.imap_unordered(work, tasks, chunksize=1):
            done += 1
            if not r["ok"]:
                failed.append(r)
                log(f"FAILED {r['split']} {r['file']}: {r['err']}\n{r['tb']}")
                continue
            secs.append(r["sec"])
            sp, i = r["split"], r["idx"]
            diags[sp][i] = r["diag"]
            if sp == "train":
                a, b = tr_off[i], tr_off[i + 1]
                if r["n"] != b - a:
                    mism.append((r["file"], r["n"], int(b - a)))
                    continue
                Xtr[a:b] = r["X"]
                ytr["times"][a:b] = r["t"]
                for k in ("q95", "betaN", "C80", "C120", "aux"):
                    ytr[k][a:b] = r[k]
            elif sp == "holdout":
                hold[i] = r
            else:
                test[i] = r
            if done % 1000 == 0:
                log(f"  {done}/{len(tasks)} ({len(failed)} failed, {len(mism)} row-mismatch) {time.time()-t0:.0f}s "
                    f"median {np.median(secs):.2f}s/shot")
    Xtr.flush()
    if failed or mism or any(h is None for h in hold) or any(x is None for x in test):
        (OUT / "prep_failed.json").write_text(json.dumps(
            {"failed": [(r["split"], r["file"], r["err"]) for r in failed], "row_mismatch": mism}, indent=1))
        raise SystemExit(f"ABORT: {len(failed)} failed, {len(mism)} row mismatches (see prep_failed.json)")
    log(f"pass done in {time.time()-t0:.0f}s; median {np.median(secs):.2f}s/shot, max {np.max(secs):.1f}s")

    # ---------------- alignment checks against H (train) / cache_hold (holdout) / L Xt (test)
    sc = np.load(HCACHE / "scaler_f.npz")
    m1, s1 = sc["mean_"].astype(np.float64), sc["scale_"].astype(np.float64)
    checks = {}
    Hx = np.load(HCACHE / "Xfs.npy", mmap_mode="r")
    HC80 = np.load(HCACHE / "C80.npy", mmap_mode="r")
    HC120 = np.load(HCACHE / "C120.npy", mmap_mode="r")
    Hq = np.load(HCACHE / "q95.npy", mmap_mode="r")
    Hb = np.load(HCACHE / "betaN.npy", mmap_mode="r")
    dx = dc80 = dc120 = dq = db = 0.0
    for a in range(0, F_tr, 200_000):
        b = min(F_tr, a + 200_000)
        z = ((np.asarray(Xtr[a:b, :90], np.float64) - m1) / s1).astype(np.float32)
        dx = max(dx, float(np.abs(z - Hx[a:b]).max()))
        dc80 = max(dc80, float(np.abs(ytr["C80"][a:b] - HC80[a:b]).max()))
        dc120 = max(dc120, float(np.abs(ytr["C120"][a:b] - HC120[a:b]).max()))
        dq = max(dq, float(np.nanmax(np.abs(ytr["q95"][a:b] - Hq[a:b]))))
        db = max(db, float(np.nanmax(np.abs(ytr["betaN"][a:b] - Hb[a:b]))))
    checks["train_vs_H"] = {"rows": F_tr, "H_rows": int(Hx.shape[0]) if full else None,
                            "max_abs_diff_v1_std": dx, "max_abs_diff_C80": dc80,
                            "max_abs_diff_C120": dc120, "C80_max_abs": float(np.abs(np.asarray(HC80[::97])).max()),
                            "max_abs_diff_q95": dq, "max_abs_diff_betaN": db,
                            "shot_counts_match": True}
    log(f"train vs H: {checks['train_vs_H']}")
    ch = np.load(HCACHE / "cache_hold.npz")
    if not MAX_HOLD:
        assert list(map(str, ch["files"])) == ho_names
    Xh_raw = np.concatenate([h["X"] for h in hold]).astype(np.float32)
    h_off = np.concatenate([[0], np.cumsum([len(h["X"]) for h in hold])]).astype(np.int64)
    zh = ((Xh_raw[:, :90].astype(np.float64) - m1) / s1).astype(np.float32)
    nH = int(h_off[-1])
    yh = {k: np.concatenate([h[k] for h in hold]) for k in ("q95", "betaN", "C80", "C120", "aux")}
    th_ = np.concatenate([h["t"] for h in hold])
    checks["holdout_vs_cache_hold"] = {
        "frames": nH, "cache_hold_frames": int(ch["Xhs"].shape[0]),
        "offsets_equal": bool(np.array_equal(h_off, ch["offsets"][:len(h_off)])),
        "max_abs_diff_v1_std": float(np.abs(zh - ch["Xhs"][:nH]).max()),
        "max_abs_diff_q95": float(np.nanmax(np.abs(yh["q95"] - ch["q95"][:nH]))),
        "max_abs_diff_times": float(np.abs(th_ - ch["times"][:nH]).max())}
    log(f"holdout vs cache_hold: {checks['holdout_vs_cache_hold']}")
    Xt_raw = np.concatenate([x["X"] for x in test]).astype(np.float32)
    t_off = np.concatenate([[0], np.cumsum([len(x["X"]) for x in test])]).astype(np.int64)
    tt_ = np.concatenate([x["t"] for x in test])
    Lx = SCR / "stage8_L_ch1_finish_package/Xt.npy"
    if Lx.exists():
        Lxt = np.load(Lx, mmap_mode="r")
        Lfiles = (SCR / "stage8_L_ch1_finish_package/test_files.txt").read_text().split()
        nT = int(t_off[-1])
        zt = ((Xt_raw[:, :90].astype(np.float64) - m1) / s1).astype(np.float32)
        checks["test_vs_L_Xt"] = {"frames": nT, "L_frames": int(Lxt.shape[0]),
                                  "files_equal_prefix": Lfiles[:len(test)] == [x["file"] for x in test],
                                  "max_abs_diff_v1_std": float(np.abs(zt - Lxt[:nT]).max())}
        log(f"test vs L Xt: {checks['test_vs_L_Xt']}")
    hold_psi = np.load(HCACHE / "cache_hold_psi.npy", mmap_mode="r")
    pc80 = np.load(HCACHE / "pca_80.npz")
    sub = np.arange(0, nH, 37)
    ref = (np.asarray(hold_psi[sub]).reshape(len(sub), -1) - pc80["mean_"][None]) @ pc80["comps_"].T
    checks["holdout_C80_vs_cache_psi"] = float(np.abs(yh["C80"][sub] - ref).max())

    # ---------------- equality of targets / unchanged blocks against the v2 cache, Thomson change statistics
    v2 = {"dir": str(V2DIR)}
    if full and (V2DIR / "train/y_C80.npy").exists():
        eq = {}
        for k in ("q95", "betaN", "C80", "C120", "aux"):
            eq[f"train_y_{k}"] = bool(np.array_equal(ytr[k], np.load(V2DIR / f"train/y_{k}.npy", mmap_mode="r"), equal_nan=True))
            eq[f"holdout_y_{k}"] = bool(np.array_equal(yh[k], np.load(V2DIR / f"holdout/y_{k}.npy"), equal_nan=True))
        eq["train_times"] = bool(np.array_equal(ytr["times"], np.load(V2DIR / "train/times.npy")))
        eq["holdout_times"] = bool(np.array_equal(th_, np.load(V2DIR / "holdout/times.npy")))
        eq["test_times"] = bool(np.array_equal(tt_, np.load(V2DIR / "test/times.npy")))
        eq["train_offsets"] = bool(np.array_equal(tr_off, np.load(V2DIR / "train/offsets.npy")))
        eq["holdout_offsets"] = bool(np.array_equal(h_off, np.load(V2DIR / "holdout/offsets.npy")))
        eq["test_offsets"] = bool(np.array_equal(t_off, np.load(V2DIR / "test/offsets.npy")))
        eq["test_files"] = (V2DIR / "test/files.txt").read_text().split() == [x["file"] for x in test]
        v2["targets_and_index_equal"] = eq
        v2["all_equal"] = bool(all(eq.values()))
        log(f"targets/index vs v2: all_equal={v2['all_equal']}")

    def drop_rows(dg, off):
        fl = np.zeros(int(off[-1]), bool)
        for i, d in enumerate(dg):
            st = (d or {}).get("thomson_stats", {})
            if any(st.get(sk, {}).get("n_dropped", 0) for sk in ("core", "edge")):
                fl[off[i]:off[i + 1]] = True
        return fl
    if True:   # smoke runs compare against the row prefix of the v2 cache (same file order)
        v2["diff"] = {"train": diff_vs_v2(Xtr, V2DIR / "train/X_raw.npy", blk, drop_rows(diags["train"], tr_off), exact=full),
                      "holdout": diff_vs_v2(Xh_raw, V2DIR / "holdout/X_raw.npy", blk, drop_rows(diags["holdout"], h_off), exact=full),
                      "test": diff_vs_v2(Xt_raw, V2DIR / "test/X_raw.npy", blk, drop_rows(diags["test"], t_off), exact=full)}
        for sp, r in v2["diff"].items():
            if r.get("available"):
                log(f"v2 diff {sp}: any-Thomson-block frames changed {r['frames_any_thomson_block_changed']} "
                    f"({100*r['frac_frames_any_thomson_block_changed']:.2f} %); "
                    + "; ".join(f"{b} {100*v['frac_frames']:.2f}%" for b, v in r["blocks"].items()))
    thc = {"train": th_stats(diags["train"], tr_names), "holdout": th_stats(diags["holdout"], ho_names),
           "test": th_stats(diags["test"], [x["file"] for x in test])}
    for sp, a in thc.items():
        log(f"thomson contract {sp}: " + "; ".join(
            f"{sk} dropped {a[sk].get('n_dropped', 0)} samples in {a[sk]['shots_with_dropped']} shots "
            f"(with data {a[sk].get('n_dropped_with_data', 0)}), all-invalid profiles "
            f"{100*a[sk].get('frac_profiles_all_invalid', float('nan')):.1f}%" for sk in ("core", "edge")))

    # ---------------- standardiser (train only)
    log("NaN-aware train statistics ...")
    nv_cols = np.flatnonzero(~is_v1)
    mean = np.zeros(D); scale = np.ones(D); cnt = np.zeros(D)
    mean[:90], scale[:90] = m1, s1
    mu, sd, c = nan_stats(Xtr, nv_cols)
    mean[nv_cols] = mu
    scale[nv_cols] = np.where(sd > 1e-8, sd, 1.0)
    cnt[nv_cols] = c
    cnt[:90] = F_tr
    np.savez(OUT / "scaler_v21.npz", mean=mean, scale=scale, is_v1=is_v1, clip=CLIP, names=np.array(names))
    if full and (V2DIR / "scaler_v2.npz").exists():
        s2 = np.load(V2DIR / "scaler_v2.npz")
        same_cols = np.concatenate([np.arange(*blk[b]) for b in SAME_BLOCKS])
        v2["scaler_same_blocks_max_abs_diff"] = {
            "mean": float(np.abs(mean[same_cols] - s2["mean"][same_cols]).max()),
            "scale": float(np.abs(scale[same_cols] - s2["scale"][same_cols]).max())}
    log("writing standardised matrices ...")
    write_std(Xtr, OUT / "train/X.npy", mean, scale, is_v1)
    write_std(Xh_raw, OUT / "holdout/X.npy", mean, scale, is_v1)
    write_std(Xt_raw, OUT / "test/X.npy", mean, scale, is_v1)
    np.save(OUT / "holdout/X_raw.npy", Xh_raw)
    np.save(OUT / "test/X_raw.npy", Xt_raw)
    if full and (V2DIR / "holdout/X.npy").exists():
        same_cols = np.concatenate([np.arange(*blk[b]) for b in SAME_BLOCKS])
        a = np.load(OUT / "holdout/X.npy", mmap_mode="r")[:, same_cols]
        b = np.load(V2DIR / "holdout/X.npy", mmap_mode="r")[:, same_cols]
        v2["holdout_X_std_same_blocks_equal"] = bool(np.array_equal(a, b))
        a = np.load(OUT / "test/X.npy", mmap_mode="r")[:, same_cols]
        b = np.load(V2DIR / "test/X.npy", mmap_mode="r")[:, same_cols]
        v2["test_X_std_same_blocks_equal"] = bool(np.array_equal(a, b))

    # ---------------- targets / indices
    np.save(OUT / "train/offsets.npy", tr_off)
    np.save(OUT / "train/shot_id.npy", np.repeat(np.arange(len(tr_names), dtype=np.int32), counts))
    np.save(OUT / "train/times.npy", ytr["times"])
    (OUT / "train/files.txt").write_text("\n".join(tr_names) + "\n")
    for k in ("q95", "betaN", "C80", "C120", "aux"):
        np.save(OUT / f"train/y_{k}.npy", ytr[k])
        np.save(OUT / f"holdout/y_{k}.npy", yh[k])
    np.save(OUT / "holdout/offsets.npy", h_off)
    np.save(OUT / "holdout/shot_id.npy", np.repeat(np.arange(len(hold), dtype=np.int32), np.diff(h_off)))
    np.save(OUT / "holdout/times.npy", th_)
    (OUT / "holdout/files.txt").write_text("\n".join(h["file"] for h in hold) + "\n")
    np.save(OUT / "test/offsets.npy", t_off)
    np.save(OUT / "test/shot_id.npy", np.repeat(np.arange(len(test), dtype=np.int32), np.diff(t_off)))
    np.save(OUT / "test/times.npy", tt_)
    (OUT / "test/files.txt").write_text("\n".join(x["file"] for x in test) + "\n")

    # ---------------- spec + diag
    th_layout = {sp: Counter(f"{d['thomson'].get('core','-')}|{d['thomson'].get('edge','-')}" for d in diags[sp] if d)
                 for sp in diags}
    nanfrac_tr = 1.0 - cnt / max(F_tr, 1)
    cols = []
    for j, (n, b) in enumerate(fn):
        cols.append({"i": j, "name": n, "block": b, "v2_compat": V2_COMPAT[b], "mean": float(mean[j]),
                     "scale": float(scale[j]), "nan_frac_train": float(nanfrac_tr[j]),
                     "nan_frac_holdout": float(np.isnan(Xh_raw[:, j]).mean()),
                     "nan_frac_test": float(np.isnan(Xt_raw[:, j]).mean())})
    spec = {
        "version": "feat_v21 2026-10-11",
        "D": D, "D_v2": D_V2, "blocks": blk, "v2_compat": V2_COMPAT, "aux_names": AUX_NAMES,
        "splits": {"train": {"shots": len(tr_names), "frames": F_tr},
                   "holdout": {"shots": len(hold), "frames": nH},
                   "test": {"shots": len(test), "frames": int(t_off[-1])}},
        "params": {"OFFSETS_MS": OFFSETS_MS, "SMOOTH_MS": SMOOTH_MS, "GAP_CORE_MS": GAP_CORE_MS,
                   "GAP_EDGE_MS": GAP_EDGE_MS, "ZGRID": ZGRID.tolist(), "RGRID": RGRID.tolist(),
                   "MAXD_CORE": MAXD_CORE, "MAXD_EDGE": MAXD_EDGE, "CLIP": CLIP, "TH1_GAP_MS": TH1_GAP_MS,
                   "DRIVE_SIGNALS": DRIVE_SIGNALS, "DRV_TAUS_MS": DRV_TAUS_MS, "DRV_DWIN_MS": DRV_DWIN_MS},
        "standardiser": "v1 block: H scaler_f (no clip); others: NaN-aware train mean/std, NaN->0, clip +-CLIP",
        "thomson_contract": "issue #12: keep t_i > max(t_<i) per system before resampling; valid iff finite & Te>0 & ne>0; "
                            "per-channel resampling between VALID samples only",
        "checks": checks, "v2": v2, "thomson_stats": thc,
        "thomson_layouts": {k: dict(v) for k, v in th_layout.items()},
        "columns": cols,
    }
    (OUT / "spec.json").write_text(json.dumps(spec, indent=1))
    (OUT / "names.txt").write_text("\n".join(names) + "\n")
    diag = {"minutes": round((time.time() - t0) / 60, 2), "D": D, "checks": checks, "v2": v2,
            "thomson_stats": thc, "sec_per_shot_median": float(np.median(secs)), "workers": WORKERS,
            "cpu_core_hours": round(WORKERS * (time.time() - t0) / 3600, 2),
            "thomson_layouts": spec["thomson_layouts"], "splits": spec["splits"]}
    (OUT / "prep_v21_diag.json").write_text(json.dumps(diag, indent=1))
    log(f"PREP_V21 DONE minutes={diag['minutes']} D={D} splits={json.dumps(spec['splits'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
