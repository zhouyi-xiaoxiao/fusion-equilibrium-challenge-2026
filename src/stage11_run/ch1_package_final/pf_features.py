#!/usr/bin/env python3
"""ch1_package_final — inputs-only features of the FcCX recipe for ANY directory of DIII-D parquet shots.

  pf_features.py --in-dir DIR --out OUT [--files order.txt] [--workers 70]

Three passes, each in its own process pool (the v2.2 builder monkeypatches data_fixes.fix_d3d_ip_times process-wide, so the
v2.1 and v2.2 passes must not share worker processes):
  1. FEATURE-V2.1 raw rows  (feat_v21.features_v21: organiser Ip-axis fix only)          -> OUT/raw21/X_raw.npy
  2. FEATURE-V2.2 raw rows  (feat_v22.features_v22: + the guarded 8192 ms old-DAQ rule)  -> OUT/raw22/X_raw.npy
  3. per-shot SIGN and CLOCK record from four input columns (same functions as the pre-submission gate
     f_ch1_harden/final_gate.py: sign_stats, clock_stats)                                -> OUT/shots.json, OUT/sign.npy
Sign rule (pre-registered in an internal development note §2; the pipeline NEVER aborts on an ambiguous sign):
  class 1  whole-record agreement of the high-current samples >= 99.5 % and flag == majority != 0     -> s = flag
  class 2  else: agreement inside the EFIT window (first - 200 ms .. last + 20 ms, v2.2 Ip axis) >= 99.5 % and
           flag == majority != 0                                                                       -> s = flag   (listed for review)
  class 3  else (reversal inside the scored window, flag != majority, zero flag): s = majority sign of the high-current
           samples inside the EFIT window; if there is none: majority of the record, else the flag, else +1   (gate G7 FAILS: a person decides)
Clock rule: ip_axis.ip_axis_info (8192 ms rule only when the organiser fix kept the shipped axis AND the magnetics record
starts before -6 s AND the corrected breakdown lands in [-100, +150] ms; otherwise its own fallback / the v2.1 axis).  The schema
guard (OLD_SCHEMA) is recorded per shot here and enforced by the gate (G8); nothing aborts.
No label column is read: feat_v21.read_inputs reads the input columns only (efit_times / efit_grid_R / efit_grid_Z are query inputs).
Row order = sorted file names, or the order of --files.  Frames = all efit_times of the shot.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
for p in (WS / "starter", WS / "starter/fusion_scoring", WS / "stage2_run", WS / "stage11_run/ch1_features", WS / "stage11_run/e_ip_axis",
          WS / "stage11_run/f_ch1_harden"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
AGREE_MIN = 0.995


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def work21(args):
    i, path = args
    try:
        assert "ip_axis" not in sys.modules, "v2.1 pass must run in a process that never imported ip_axis"
        from feat_v21 import features_v21, read_inputs
        row = read_inputs(path)
        X = features_v21(row)
        return {"i": i, "ok": True, "X": np.asarray(X, np.float32), "t": np.asarray(row["efit_times"], dtype=np.float64)}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def work22(args):
    i, path = args
    try:
        from feat_v22 import features_v22, read_inputs
        row = read_inputs(path)
        X, diag = features_v22(row, return_diag=True)
        return {"i": i, "ok": True, "X": np.asarray(X, np.float32), "t": np.asarray(row["efit_times"], dtype=np.float64), "ip_axis": diag["ip_axis"]}
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def work_sign(args):
    i, path = args
    try:
        import pyarrow.parquet as pq
        import final_gate as G
        import ip_axis
        tb = pq.read_table(path, columns=["efit_times", "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times"])
        row = {c: np.asarray(tb[c][0].as_py()) for c in tb.column_names}
        t = np.asarray(row["efit_times"], np.float64)
        rec = {"i": i, "ok": True, "file": Path(path).name, "T": int(len(t))}
        rec.update(G.sign_stats(row["magnetics_plasma_current"]))
        rec.update(G.clock_stats(row, t))
        ax = np.asarray(ip_axis.ip_axis_info(row)["axis"], np.float64)
        ip = np.asarray(row["magnetics_plasma_current"], np.float64)
        n_ = min(len(ax), len(ip))
        s_win = 0.0
        if n_ >= 10 and len(t):
            a_, v_ = ax[:n_], ip[:n_]
            m_ = np.isfinite(a_) & np.isfinite(v_) & (a_ >= t[0] - 200.0) & (a_ <= t[-1] + 20.0)
            fin = np.abs(ip[np.isfinite(ip)])
            thr = max(G.SIGN_MIN_KA, G.SIGN_FRAC_P95 * float(np.percentile(fin, 95))) if len(fin) else G.SIGN_MIN_KA
            h_ = m_ & (np.abs(v_) >= thr)
            if h_.any():
                s_win = float(np.sign(int((v_[h_] > 0).sum()) - int((v_[h_] < 0).sum())))
        rec["s_win"] = s_win
        flag_ok = rec["s_feat"] != 0 and rec["s_conf"] == rec["s_feat"]
        if flag_ok and rec["agree"] >= AGREE_MIN:
            cls, s = 1, rec["s_feat"]
        elif flag_ok and rec["win_agree22"] >= AGREE_MIN:
            cls, s = 2, rec["s_feat"]
        else:
            cls = 3
            s = s_win if s_win != 0 else (rec["s_conf"] if rec["s_conf"] != 0 else (rec["s_feat"] if rec["s_feat"] != 0 else 1.0))
        rec["sign_class"], rec["s"] = cls, float(s)
        return rec
    except Exception as e:  # noqa: BLE001
        return {"i": i, "ok": False, "err": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-1500:]}


def run(fn, files, workers):
    res = [None] * len(files)
    with mp.get_context("spawn").Pool(min(workers, len(files)), initializer=_init) as pool:
        for r in pool.imap_unordered(fn, [(i, str(f)) for i, f in enumerate(files)], chunksize=1):
            if not r["ok"]:
                raise SystemExit(f"FAILED {fn.__name__} {files[r['i']].name}: {r['err']}\n{r['tb']}")
            res[r["i"]] = r
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--files", default=None); ap.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "70")))
    a = ap.parse_args()
    t0 = time.time()
    ind, out = Path(a.in_dir), Path(a.out)
    files = [ind / l.strip() for l in Path(a.files).read_text().split("\n") if l.strip()] if a.files else sorted(ind.glob("*.parquet"))
    assert files, f"no parquet files in {ind}"
    (out / "raw21").mkdir(parents=True, exist_ok=True); (out / "raw22").mkdir(parents=True, exist_ok=True)
    r21 = run(work21, files, a.workers); t1 = time.time()
    r22 = run(work22, files, a.workers); t2 = time.time()
    rs = run(work_sign, files, a.workers); t3 = time.time()
    n21 = [len(r["X"]) for r in r21]
    assert n21 == [len(r["X"]) for r in r22] == [r["T"] for r in rs], "frame counts differ between the passes"
    assert all(np.array_equal(a_["t"], b_["t"]) for a_, b_ in zip(r21, r22))
    off = np.concatenate([[0], np.cumsum(n21)]).astype(np.int64)
    X21 = np.concatenate([r["X"] for r in r21]).astype(np.float32); X22 = np.concatenate([r["X"] for r in r22]).astype(np.float32)
    np.save(out / "raw21/X_raw.npy", X21); np.save(out / "raw22/X_raw.npy", X22)
    np.save(out / "offsets.npy", off)
    np.save(out / "shot_id.npy", np.repeat(np.arange(len(files), dtype=np.int32), np.diff(off)))
    np.save(out / "times.npy", np.concatenate([r["t"] for r in r21]))
    (out / "files.txt").write_text("\n".join(f.name for f in files) + "\n")
    from feat_v21 import feature_names
    names = [n for n, _ in feature_names()]
    sc = names.index("time_sign_ip_shot")
    shots = []
    for k, (r, q) in enumerate(zip(rs, r22)):
        d = {kk: v for kk, v in r.items() if kk not in ("i", "ok")}
        d["ip_axis"] = q["ip_axis"]
        c21, c22 = X21[off[k]:off[k + 1], sc], X22[off[k]:off[k + 1], sc]
        d["feature_sign_col"] = [float(np.unique(c21)[0]) if len(np.unique(c21)) == 1 else None, float(np.unique(c22)[0]) if len(np.unique(c22)) == 1 else None]
        shots.append(d)
    np.save(out / "sign.npy", np.array([d["s"] for d in shots], np.float64))
    cls = np.array([d["sign_class"] for d in shots]); br = {}
    for d in shots:
        br[d["branch"]] = br.get(d["branch"], 0) + 1
    summ = {"in_dir": str(ind), "shots": len(files), "frames": int(off[-1]), "D": int(X21.shape[1]),
            "sign_class_counts": {str(c): int((cls == c).sum()) for c in (1, 2, 3)}, "ip_neg_shots": int(sum(d["s"] < 0 for d in shots)),
            "review_class2": [(d["file"], round(d["agree"], 4), round(d["win_agree22"], 4)) for d in shots if d["sign_class"] == 2],
            "FAIL_class3": [(d["file"], d["agree"], d["win_agree22"], d["s_feat"], d["s_conf"], d["s_win"], d["s"]) for d in shots if d["sign_class"] == 3],
            "sign_col_ne_s": [d["file"] for d in shots if d["feature_sign_col"] != [d["s"], d["s"]]],
            "clock_branches": br, "old_daq_shots": int(sum(d["old_like"] for d in shots)),
            "clock_guard_fallbacks": [d["file"] for d in shots if d["branch"] in ("fallback_tbd", "kept_unchanged")],
            "rule_off_validated_schema": [(d["file"], d["schema_bad"]) for d in shots if d["branch"] == "old_era_8192" and not d["schema_ok"]],
            "old_daq_without_rule": [d["file"] for d in shots if d["old_like"] and d["branch"] != "old_era_8192"],
            "seconds": {"v21": round(t1 - t0, 1), "v22": round(t2 - t1, 1), "sign_clock": round(t3 - t2, 1)}}
    json.dump({"summary": summ, "shots": shots}, open(out / "shots.json", "w"), indent=1, default=float)
    print("PF_FEATURES " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
