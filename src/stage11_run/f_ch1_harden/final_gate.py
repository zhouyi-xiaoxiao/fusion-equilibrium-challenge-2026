#!/usr/bin/env python3
"""Stage 11F / f_ch1_harden — PRE-SUBMISSION GATE for a DIII-D submission file (Challenge 1).  Label-free: reads the
submission .npz and the INPUT columns of the parquet shots only (efit_times, magnetics_time, magnetics_plasma_current[_times]);
never an efit_* target, never magnetics_dsep.

  python final_gate.py --sub SUB.npz --in-dir PARQUET_DIR [--order stream_order.txt|sorted]
        [--base BLEND.npz --robmi ROBMI.npz --incgn CORRECTED.npz --premerge ROBMI_OR_GUARD_INPUT.npz]   chain intermediates
        [--guard-json X.guard.json] [--expect-md5 HEX] [--report OUT.json] [--workers 280]

Gates (PASS / FAIL; REVIEW = allowed to ship, but look at it first)
  G1  file        readable without pickle, md5 (== --expect-md5 if given), exactly 3 arrays per shot, no extra keys
  G2  coverage    every parquet shot has shot_JJJJ_{psirz,q95,betaN}; psirz (T,65,65), scalars (T,), T = len(efit_times): 100 % shots + frames
  G3  dtype       every array float32 (never float16 / float64)
  G4  finite      no NaN / Inf anywhere
  G5  order       JJJJ = index in the HF STREAM order given by --order (a permutation of the parquet names; checked against
                  ch1_package/check_order.py's order_check.json when it sits next to the order file); 'sorted' = REVIEW
  G6  derivation  OFFICIAL prediction-path derivation of every frame (extract_lcfs_with_sign n_iter=22 + derive_frame with li):
                  zero LCFS extraction failures, zero non-finite consistency scalars; shot-weighted failure rate reported
  G7  sign        per shot input-only Ip sign: agreement among samples with |Ip| >= max(50 kA, 0.2 P95|Ip|) >= 99.5 %, majority sign
                  == the feature flag formula (time_sign_ip_shot) and != 0; ambiguous shots listed (fallback = REVIEW, see note)
  G8  clock       Ip clock per shot (--clock v21 = organiser fix, v22 = + 8192 ms rule): branch census; no old-DAQ record re-origined
                  by the organiser fix; (v22) every old-DAQ shot takes the rule branch (guard fallbacks = 0) on the validated schema;
                  axis strictly increasing, corrected breakdown inside the observed window, EFIT-time coverage >= 99.5 %,
                  extrapolation <= 10 ms (REVIEW if only coverage / extrapolation miss)
  G9  chain       (needs --base/--robmi) robmi fallback rate overall <= 1.0 % and no stratum (sign, era, length bin) more than
                  0.5 pp above the overall rate (REVIEW otherwise); frames replaced by the merge fallback / li guard counted
  G10 values      q95 / betaN / psi ranges inside loose physical windows; no fixed-decimal rounding (validator's test)
Exit code 0 only if no gate FAILs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
STARTER = WS / "starter"
for p in (STARTER, STARTER / "fusion_scoring", WS / "stage11_run/e_ip_axis", WS / "stage11_run/ch1_features", WS / "stage2_run"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MACHINE = "DIII-D"
NAMES7 = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li"]
SIGN_MIN_KA, SIGN_FRAC_P95, SIGN_AGREE_MIN = 50.0, 0.2, 0.995
COV_MIN, EXTRAP_MAX_MS = 0.995, 10.0
# validated on the 2,111 train + 288 public-test old-DAQ shots (an internal development note §3): record start -9371.8..-8998.5 ms,
# record length 24,575.5 / 25,085.5..25,088.0 ms, 0.5 ms (2 shots 0.05 ms) sampling, generic shipped Ip axis on every DIII-D file
OLD_SCHEMA = {"ship_n": 30719, "ship_t0": -858.187, "ship_dt": 0.5, "mag_dt": (0.5, 0.05), "mag_t0": (-9400.0, -8950.0), "mag_len": (24570.0, 25095.0)}
NEW_T0 = (-4350.0, -3880.0)             # new-DAQ record start (train + public test: -4301.2 .. -3926.5 ms)
BD_OBS = (-130.0, 140.0)                # corrected breakdown time seen on 7,915 shots: -118.3 .. +130.7 ms
FB_MAX, FB_STRATUM_PP = 0.010, 0.005
Q95_WIN, BETAN_WIN, PSI_ABS_MAX = (0.3, 80.0), (-8.0, 10.0), 5.0
_G: dict = {}


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def _init(sub, chain):
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    from common import AXIS_SIGN, CONS_SCALARS
    from derive import derive_frame
    from lcfs import extract_lcfs_with_sign
    m = np.load(STARTER / "fusion_scoring/masks/d3d_envelope.npz")
    _G.update(R=m["grid_R"].astype(np.float64), Z=m["grid_Z"].astype(np.float64), MC=m["mask_coarse"].astype(bool),
              MF=m["mask_coarse"].astype(bool).astype(np.float64), AX=AXIS_SIGN[MACHINE] * 1.0, names=list(CONS_SCALARS),
              derive=derive_frame, exs=extract_lcfs_with_sign, sub=np.load(sub, allow_pickle=False),
              chain={k: np.load(v, allow_pickle=False) for k, v in chain.items() if v})
    assert _G["names"] == NAMES7, _G["names"]


def sign_stats(ip):
    ip = np.asarray(ip, np.float64)
    v = ip[np.isfinite(ip)]
    if len(v) < 10:
        return {"agree": float("nan"), "s_conf": 0.0, "s_feat": 0.0, "n_hi": 0, "ip_p95": float("nan")}
    a = np.abs(v)
    p95 = float(np.percentile(a, 95))
    hi = a >= max(SIGN_MIN_KA, SIGN_FRAC_P95 * p95)
    npos, nneg, nhi = int((v[hi] > 0).sum()), int((v[hi] < 0).sum()), int(hi.sum())
    on = a > 0.10 * float(np.nanpercentile(a, 99.5))
    return {"agree": (max(npos, nneg) / nhi) if nhi else float("nan"), "s_conf": float(np.sign(npos - nneg)),
            "s_feat": float(np.sign(np.median(v[on]))) if on.any() else 0.0, "n_hi": nhi, "ip_p95": p95}


def clock_stats(row, t):
    import ip_axis
    info = ip_axis.ip_axis_info(row)
    shipped = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    a22 = np.asarray(info["axis"], dtype=np.float64)
    a21 = np.asarray(ip_axis.original()(row), dtype=np.float64)
    mt = np.asarray(row["magnetics_time"], dtype=np.float64)
    mtf = mt[np.isfinite(mt)]

    def cov(ax):
        f = ax[np.isfinite(ax)]
        if len(f) < 2 or len(t) == 0:
            return float("nan"), float("nan")
        return float(((t >= f[0]) & (t <= f[-1])).mean()), float(max(0.0, f[0] - t[0], t[-1] - f[-1]))
    c21, c22 = cov(a21), cov(a22)
    dsh = np.diff(shipped) if len(shipped) > 1 else np.zeros(1)
    mag_dt = float(np.median(np.diff(mtf))) if len(mtf) > 1 else float("nan")
    old_like = bool(len(mtf) and mtf[0] < -6000.0)
    new_ok = bool(len(mtf)) and NEW_T0[0] <= float(mtf[0]) <= NEW_T0[1]
    sch = {"ship_n": len(shipped) == OLD_SCHEMA["ship_n"], "ship_t0": abs(float(shipped[0]) - OLD_SCHEMA["ship_t0"]) < 0.01 if len(shipped) else False,
           "ship_dt": abs(float(np.median(dsh)) - OLD_SCHEMA["ship_dt"]) < 1e-3 and float(np.max(np.abs(dsh - np.median(dsh)))) < 0.01,
           "mag_dt": any(abs(mag_dt - x) < 2e-3 for x in OLD_SCHEMA["mag_dt"]),
           "mag_t0": bool(len(mtf)) and OLD_SCHEMA["mag_t0"][0] <= float(mtf[0]) <= OLD_SCHEMA["mag_t0"][1],
           "mag_len": bool(len(mtf)) and OLD_SCHEMA["mag_len"][0] <= float(mtf[-1] - mtf[0]) <= OLD_SCHEMA["mag_len"][1]}
    ip = np.asarray(row["magnetics_plasma_current"], dtype=np.float64)

    def win_agree(ax):                      # sign agreement of the high-current samples inside the EFIT window (+ feature look-back/ahead)
        n_ = min(len(ax), len(ip))
        if n_ < 10 or len(t) == 0:
            return float("nan")
        a_, v_ = ax[:n_], ip[:n_]
        m_ = np.isfinite(a_) & np.isfinite(v_) & (a_ >= t[0] - 200.0) & (a_ <= t[-1] + 20.0)
        fin = np.abs(ip[np.isfinite(ip)])
        thr = max(SIGN_MIN_KA, SIGN_FRAC_P95 * float(np.percentile(fin, 95))) if len(fin) else SIGN_MIN_KA
        h_ = m_ & (np.abs(v_) >= thr)
        if not h_.any():
            return float("nan")
        return float(max((v_[h_] > 0).sum(), (v_[h_] < 0).sum()) / h_.sum())
    return {"branch": info["branch"], "shift_ms": float(info["shift_ms"]), "t_bd": float(info["t_bd_v22"]), "t_bd_v21": float(info["t_bd_v21"]),
            "axis_inc": bool(np.all(np.diff(a22) > 0)), "axis21_inc": bool(np.all(np.diff(a21) > 0)),
            "cov": c22[0], "extrap_ms": c22[1], "cov21": c21[0], "extrap21_ms": c21[1], "old_like": old_like, "new_t0_ok": new_ok, "schema_ok": bool(all(sch.values())),
            "win_agree21": win_agree(a21), "win_agree22": win_agree(a22),
            "schema_bad": [k for k, v in sch.items() if not v], "mag_dt": mag_dt, "mag_t0": float(mtf[0]) if len(mtf) else float("nan")}


def work(args):
    j, path, src = args                      # j = submission index (stream order); src = row in sorted order (chain files)
    import pyarrow.parquet as pq
    g = _G
    rec = {"j": j, "file": Path(path).name, "err": []}
    tb = pq.read_table(path, columns=["efit_times", "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times"])
    row = {c: np.asarray(tb[c][0].as_py()) for c in tb.column_names}
    t = np.asarray(row["efit_times"], np.float64)
    T = len(t)
    rec["T"] = T
    sub = g["sub"]
    k = f"shot_{j:04d}_"
    arrs = {}
    for nm, shp in (("psirz", (T, 65, 65)), ("q95", (T,)), ("betaN", (T,))):
        if k + nm not in sub.files:
            rec["err"].append(f"missing {k+nm}"); continue
        a = sub[k + nm]
        arrs[nm] = a
        if a.shape != shp:
            rec["err"].append(f"shape {k+nm} {a.shape} != {shp}")
        if a.dtype != np.float32:
            rec["err"].append(f"dtype {k+nm} {a.dtype}")
        if not np.isfinite(a).all():
            rec["err"].append(f"nonfinite {k+nm} {int((~np.isfinite(a)).sum())}")
    rec.update(sign_stats(row["magnetics_plasma_current"]))
    rec.update(clock_stats(row, t))
    rec["efit_inc"] = bool(np.all(np.diff(t) > 0)) if T > 1 else True
    if "psirz" in arrs and arrs["psirz"].shape == (T, 65, 65):
        P = arrs["psirz"]
        lf = np.zeros(T, bool); cf = np.zeros((T, 7), bool); li = np.full(T, np.nan)
        for f in range(T):
            psi = P[f].astype(np.float64)
            ex, _ = g["exs"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], n_iter=22, axis_sign=g["AX"])
            v = g["derive"](psi, g["R"], g["Z"], MACHINE, g["MC"], g["MF"], contour=ex, with_li=True, axis_sign=g["AX"])
            lf[f] = ex is None
            vv = np.array([v[n] for n in NAMES7], np.float64)
            cf[f] = ~np.isfinite(vv)
            li[f] = vv[6]
        rec.update(lcfs_fail=int(lf.sum()), cons_fail=cf.sum(0).tolist(), fail_frames=np.flatnonzero(lf | cf.any(1)).tolist()[:50],
                   li_max=float(np.nanmax(li)) if np.isfinite(li).any() else float("nan"),
                   li_min=float(np.nanmin(li)) if np.isfinite(li).any() else float("nan"), n_li_near=int(((li > 4.9) | (li < 0.12)).sum()),
                   psi_absmax=float(np.max(np.abs(P))), q_min=float(arrs["q95"].min()) if "q95" in arrs else float("nan"),
                   q_max=float(arrs["q95"].max()) if "q95" in arrs else float("nan"),
                   b_min=float(arrs["betaN"].min()) if "betaN" in arrs else float("nan"), b_max=float(arrs["betaN"].max()) if "betaN" in arrs else float("nan"))
        a = P.astype(np.float64).ravel()
        scale = float(np.max(np.abs(a))) if a.size else 0.0
        rnd = 0
        for dec in range(1, 6):
            step = 10.0 ** (-dec)
            if scale <= 0 or step < scale * 1e-4:
                break
            if np.allclose(a / step, np.round(a / step), atol=1e-6, rtol=0):
                rnd = dec; break
        rec["rounded_decimals"] = rnd
        ch = g["chain"]
        kk = f"shot_{src:04d}_psirz"
        if "base" in ch and "robmi" in ch:
            b, r = ch["base"][kk], ch["robmi"][kk]
            same = np.all(b.reshape(T, -1) == r.reshape(T, -1), axis=1)
            rec["n_fb"] = int(same.sum())
            rec["fb_first10pct"] = int(same[:max(1, T // 10)].sum()); rec["fb_last10pct"] = int(same[T - max(1, T // 10):].sum())
            if "incgn" in ch:
                ig = ch["incgn"][kk]
                rec["n_final_is_incgn"] = int((np.all(P.reshape(T, -1) == ig.reshape(T, -1), axis=1) & ~np.all(P.reshape(T, -1) == r.reshape(T, -1), axis=1)).sum())
            rec["n_final_ne_robmi"] = int((~np.all(P.reshape(T, -1) == r.reshape(T, -1), axis=1)).sum())
            rec["n_final_is_base"] = int(np.all(P.reshape(T, -1) == b.reshape(T, -1), axis=1).sum())
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", required=True); ap.add_argument("--in-dir", required=True)
    ap.add_argument("--order", default="sorted")
    for k in ("base", "robmi", "incgn"):
        ap.add_argument("--" + k, default="")
    ap.add_argument("--guard-json", default=""); ap.add_argument("--expect-md5", default="")
    ap.add_argument("--report", default=""); ap.add_argument("--workers", type=int, default=int(os.environ.get("NW", "280")))
    ap.add_argument("--clock", default="v22", choices=["v21", "v22"], help="Ip axis the pipeline's features use (v21 = organiser fix only; v22 = + 8192 ms rule)")
    ap.add_argument("--limit", type=int, default=0, help="debug: only the first N shots (coverage gates then FAIL by design)")
    a = ap.parse_args()
    t0 = time.time()
    sub_p, ind = Path(a.sub), Path(a.in_dir)
    G = {}                                     # gate -> (status, message)

    def gate(name, ok, msg, review=False):
        G[name] = ("PASS" if ok else ("REVIEW" if review else "FAIL"), msg)
        print(f"[{G[name][0]:6s}] {name:14s} {msg}", flush=True)

    files = sorted(p.name for p in ind.glob("*.parquet"))
    n = len(files)
    # ---------------- G1
    try:
        z = np.load(sub_p, allow_pickle=False)
        keys = list(z.files)
        h = md5(sub_p)
        exp = {f"shot_{j:04d}_{s}" for j in range(n) for s in ("psirz", "q95", "betaN")}
        extra = [k for k in keys if k not in exp]; miss = [k for k in exp if k not in set(keys)]
        ok = (not extra) and (not miss) and (not a.expect_md5 or a.expect_md5 == h)
        gate("G1 file", ok, f"{sub_p.name}: {sub_p.stat().st_size} bytes, md5 {h}" + (f" (expected {a.expect_md5})" if a.expect_md5 else "")
             + f", {len(keys)} arrays for {n} parquet shots, extra keys {len(extra)}, missing keys {len(miss)}")
        del z
    except Exception as e:  # noqa: BLE001
        gate("G1 file", False, f"cannot read {sub_p}: {type(e).__name__}: {e}")
        return 1
    # ---------------- G5 order
    if a.order == "sorted":
        order = list(files)
        gate("G5 order", False, "order = sorted file names (NOT verified against the HF stream order)", review=True)
    else:
        order = [x.strip() for x in Path(a.order).read_text().split("\n") if x.strip()]
        perm = len(order) == n and sorted(order) == files
        oc = Path(a.order).parent / "order_check.json"
        extra_msg = ""
        if oc.exists():
            try:
                j = json.load(open(oc))
                extra_msg = "; order_check.json: " + json.dumps({k: j[k] for k in list(j)[:8] if not isinstance(j[k], (list, dict))})[:300]
            except Exception:  # noqa: BLE001
                extra_msg = "; order_check.json unreadable"
        gate("G5 order", perm, f"{a.order}: {len(order)} names, permutation of the parquet dir: {perm}, identity with sorted: {order == files}{extra_msg}")
        if not perm:
            order = list(files)
    pos = {f: i for i, f in enumerate(files)}
    tasks = [(j, str(ind / f), pos.get(f, j)) for j, f in enumerate(order)]
    if a.limit:
        tasks = tasks[:a.limit]
    chain = {"base": a.base, "robmi": a.robmi, "incgn": a.incgn}
    import common  # noqa: F401  (import the scorer modules before forking)
    import derive  # noqa: F401
    import lcfs  # noqa: F401
    import ip_axis  # noqa: F401
    import data_fixes  # noqa: F401
    import pyarrow.parquet  # noqa: F401
    with mp.get_context("fork").Pool(min(a.workers, len(tasks)), initializer=_init, initargs=(str(sub_p), chain)) as pool:
        recs = pool.map(work, tasks, chunksize=1)
    recs.sort(key=lambda r: r["j"])
    S = len(recs)
    Tn = np.array([r["T"] for r in recs])
    nfr = int(Tn.sum())
    errs = [e for r in recs for e in r["err"]]
    # ---------------- G2-G4
    cov_e = [e for e in errs if e.startswith(("missing", "shape"))]
    gate("G2 coverage", not cov_e and S == n, f"{S}/{n} shots, {nfr} frames; coverage/shape problems {len(cov_e)}" + (f": {cov_e[:3]}" if cov_e else ""))
    dt_e = [e for e in errs if e.startswith("dtype")]
    gate("G3 dtype", not dt_e, f"float32 everywhere: {not dt_e}" + (f"; {dt_e[:3]}" if dt_e else ""))
    nf_e = [e for e in errs if e.startswith("nonfinite")]
    gate("G4 finite", not nf_e, f"non-finite arrays {len(nf_e)}" + (f": {nf_e[:3]}" if nf_e else ""))
    # ---------------- G6 derivation
    have = [r for r in recs if "lcfs_fail" in r]
    lf = np.array([r["lcfs_fail"] for r in have]); cf = np.array([r["cons_fail"] for r in have]).reshape(len(have), 7)
    Th = np.array([r["T"] for r in have])
    sw_l = float(np.mean(lf / np.maximum(Th, 1))) if len(have) else float("nan")
    sw_c = float(np.mean(cf.max(1) / np.maximum(Th, 1))) if len(have) else float("nan")
    gate("G6 derivation", len(have) == S and lf.sum() == 0 and cf.sum() == 0,
         f"LCFS extraction failures {int(lf.sum())}/{int(Th.sum())} frames in {int((lf > 0).sum())} shots (shot-weighted rate {sw_l:.2e}); "
         f"non-finite scalars per name {dict(zip(NAMES7, cf.sum(0).tolist()))} (shot-weighted max-scalar rate {sw_c:.2e}); "
         f"frames with li outside 0.12..4.9: {sum(r['n_li_near'] for r in have)}, li range [{min(r['li_min'] for r in have):.3f}, {max(r['li_max'] for r in have):.3f}]")
    # ---------------- G7 sign
    ag = np.array([r["agree"] for r in recs]); sc_ = np.array([r["s_conf"] for r in recs]); sf = np.array([r["s_feat"] for r in recs])
    wk = "win_agree22" if a.clock == "v22" else "win_agree21"
    amb = [r["file"] for r in recs if not (r["agree"] >= SIGN_AGREE_MIN)]
    amb_win = [r["file"] for r in recs if not (r["agree"] >= SIGN_AGREE_MIN) and not (r[wk] >= SIGN_AGREE_MIN)]
    hard = [r["file"] for r in recs if r["s_feat"] == 0 or r["s_conf"] != r["s_feat"]]
    gate("G7 sign", not amb and not hard, f"Ip<0 shots {int((sc_ < 0).sum())}/{S} ({(sc_ < 0).mean():.3%}); whole-record agreement min {np.nanmin(ag):.5f}, "
         f"< {SIGN_AGREE_MIN}: {len(amb)} shots, of which also < {SIGN_AGREE_MIN} inside the EFIT window (fallback: no mirror, s = +1): {len(amb_win)}; "
         f"flag formula != majority sign or zero sign (canonicalisation would be wrong / abort): {len(hard)}"
         + (f"; LOW-AGREEMENT {[(r['file'], round(r['agree'], 4), round(r[wk], 4)) for r in recs if r['file'] in set(amb)][:10]}" if amb else "")
         + (f"; HARD {hard[:10]}" if hard else ""), review=not hard and not amb_win)
    # ---------------- G8 clock
    br = {}
    for r in recs:
        br[r["branch"]] = br.get(r["branch"], 0) + 1
    rule = [r for r in recs if r["branch"] == "old_era_8192"]
    oldlike_norule = [r["file"] for r in recs if r["old_like"] and r["branch"] != "old_era_8192"]
    fb = [r["file"] for r in recs if r["branch"] in ("fallback_tbd", "kept_unchanged")]
    ck, ce, ci = ("cov", "extrap_ms", "axis_inc") if a.clock == "v22" else ("cov21", "extrap21_ms", "axis21_inc")
    bad_axis = [r["file"] for r in recs if not r[ci]]
    bad_cov = [r["file"] for r in recs if not (r[ck] >= COV_MIN)]
    bad_ext = [r["file"] for r in recs if not (r[ce] <= EXTRAP_MAX_MS)]
    bad_sch = [(r["file"], r["schema_bad"]) for r in rule if not r["schema_ok"]]
    tb = np.array([r["t_bd"] for r in rule]) if rule else np.zeros(0)
    bad_new = [r["file"] for r in recs if not r["old_like"] and not r["new_t0_ok"]]
    # corrected breakdown inside the window seen on 7,915 shots (v21 pipeline: only the re-origined shots have a corrected axis)
    bad_bd = [r["file"] for r in recs if (a.clock == "v22" or r["branch"] == "reorigined") and not (BD_OBS[0] <= r["t_bd"] <= BD_OBS[1])]
    # an old-DAQ record that the organiser fix RE-ORIGINED would be 8.2 s off on either pipeline -> FAIL in both modes
    ok8 = not (bad_axis or oldlike_norule or bad_new or bad_bd) and (a.clock == "v21" or not (fb or bad_sch))
    gate("G8 clock", ok8 and not bad_cov and not bad_ext,
         f"pipeline axis {a.clock}; rule branches {br}; old-DAQ share {len(rule)/max(S,1):.3%}; guard fallbacks {len(fb)}; old-DAQ shots without the rule {len(oldlike_norule)}; "
         f"non-increasing axes {len(bad_axis)}; rule shots off the validated schema {len(bad_sch)}; EFIT coverage < {COV_MIN}: {len(bad_cov)} shots "
         f"(min {min(r[ck] for r in recs):.4f}); extrapolation > {EXTRAP_MAX_MS} ms: {len(bad_ext)} shots (max {max(r[ce] for r in recs):.1f} ms)"
         + (f"; corrected breakdown 5/50/95 % {np.percentile(tb, [5, 50, 95]).round(1).tolist()} ms" if len(tb) else "")
         + f"; new-DAQ shots with a record start outside {NEW_T0}: {len(bad_new)}; corrected breakdown outside {BD_OBS} ms: {len(bad_bd)}"
         + (f"; schema issues {bad_sch[:3]}" if bad_sch else "") + (f"; fallback shots {fb[:5]}" if fb else ""), review=ok8)
    # ---------------- G9 chain
    if a.base and a.robmi:
        nfb = np.array([r.get("n_fb", 0) for r in recs]); tot = nfb.sum() / nfr
        neg = sc_ < 0; old = np.array([r["old_like"] for r in recs])
        strata = {"Ip<0": neg, "Ip>0": ~neg, "old-DAQ": old, "new-DAQ": ~old, "len<=50": Tn <= 50, "len51-150": (Tn > 50) & (Tn <= 150), "len>150": Tn > 150}
        st = {k: (float(nfb[m].sum() / max(Tn[m].sum(), 1)), float(np.mean(nfb[m] / Tn[m])) if m.any() else float("nan"), int(m.sum())) for k, m in strata.items()}
        worst = max((v[0] for v in st.values() if v[2] > 0), default=0.0)
        nmerge = sum(r.get("n_final_ne_robmi", 0) for r in recs)
        ninc = sum(r.get("n_final_is_incgn", 0) for r in recs)
        gmsg = ""
        if a.guard_json and Path(a.guard_json).exists():
            gj = json.load(open(a.guard_json))
            gmsg = f"; li guard: guarded {gj.get('guarded')}, failed {gj.get('failed')} of {gj.get('frames')} frames"
        gate("G9 chain", tot <= FB_MAX and worst <= tot + FB_STRATUM_PP,
             f"robmi fallback (map == base) {int(nfb.sum())}/{nfr} frames = {tot:.3%} (shot-weighted {np.mean(nfb / Tn):.3%}); by stratum frame-rate/shot-rate/shots: "
             + ", ".join(f"{k} {v[0]:.3%}/{v[1]:.3%}/{v[2]}" for k, v in st.items())
             + f"; in the first / last 10 % of a shot: {sum(r.get('fb_first10pct', 0) for r in recs)} / {sum(r.get('fb_last10pct', 0) for r in recs)}"
             + f"; final != robmi on {nmerge} frames (of which == incumbent-GN frame {ninc}); final == base on {sum(r.get('n_final_is_base', 0) for r in recs)}{gmsg}",
             review=True)
    else:
        gate("G9 chain", False, "chain intermediates not given (--base/--robmi): fallback rates NOT checked", review=True)
    # ---------------- G10 values
    qmn, qmx = min(r.get("q_min", np.nan) for r in have), max(r.get("q_max", np.nan) for r in have)
    bmn, bmx = min(r.get("b_min", np.nan) for r in have), max(r.get("b_max", np.nan) for r in have)
    pam = max(r.get("psi_absmax", np.nan) for r in have)
    rnd = [r["file"] for r in have if r.get("rounded_decimals", 0)]
    okv = Q95_WIN[0] <= qmn and qmx <= Q95_WIN[1] and BETAN_WIN[0] <= bmn and bmx <= BETAN_WIN[1] and pam <= PSI_ABS_MAX
    gate("G10 values", okv and not rnd, f"q95 [{qmn:.3f}, {qmx:.3f}] betaN [{bmn:.3f}, {bmx:.3f}] max|psi| {pam:.3f}; fixed-decimal rounding detected on {len(rnd)} shots",
         review=not rnd)
    nfail = sum(1 for v in G.values() if v[0] == "FAIL"); nrev = sum(1 for v in G.values() if v[0] == "REVIEW")
    print(f"\n=== GATE {'PASS' if nfail == 0 else 'FAIL'}: {len(G) - nfail - nrev} PASS, {nrev} REVIEW, {nfail} FAIL  ({S} shots, {nfr} frames, {time.time()-t0:.0f}s)", flush=True)
    if a.report:
        json.dump({"sub": str(sub_p), "in_dir": str(ind), "gates": {k: {"status": v[0], "msg": v[1]} for k, v in G.items()},
                   "overall": "PASS" if nfail == 0 else "FAIL", "shots": recs}, open(a.report, "w"), indent=1, default=float)
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
