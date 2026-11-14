#!/usr/bin/env python3
"""
Stage 11o / Ch2 scalars — APPLY: rewrite the q95 / betaN arrays of a candidate's parts dir (G1 layout) with a scalar rule; psi and every other
array are copied unchanged (psi equality is re-read and asserted per shot).  The output parts carry CALIBRATED scalars (pack with a unit
calibration json, as the blend parts of synth_net / integrate: syn_pack then applies the validity window, in-shot interpolation, the centred
3-frame median and the clips).

Rules (per scalar; `keep` leaves the candidate's array as it is):
  q95    keep | blend | diff
  betaN  keep | blend | diff | b3
    blend : recompute the two-source combination from the component runs given by --net RUN:TAG:CALIB and --inv RUN:TAG:CALIB with the weight
            of the net --wq / --wb (q95: log-mean of the calibrated values, betaN: arithmetic mean — exactly syn_avg.py at 0.5 / 0.5), or with
            one weight per label-free Ip phase (--wq-phase w0,w1,w2,w3,w4: ramp-up < 0.5 max, ramp-up 0.5-0.9, flat, ramp-down >= 0.5, ramp-down < 0.5);
            --gate-pop train_<net>.json multiplies the net's q95 weight by the training-support gate of rule Q11a (excursion of the RAW q95 head outside
            the [p5, p95] of the net's synthetic training targets, sd = (ln p95 - ln p5) / 3.29); --gate-aux adds the kappa and li heads (Q11b)
    diff  : value_base x F[final map] / F[base map], F = contour q95 (`q95v`) resp. Thomson-volume betaN_e of the orec.py record tables
            --rec-final / --rec-base (rule Q10 / B10); value_base where a term is missing or the ratio is outside [0.5, 2]
    b3    : integrate's 3-estimator mean (2 betaN + c betaN_e) / 3 with betaN_e from --rec-base (or --rec-final with --b3-final) and c from --c-json
  --prepack-q W / --prepack-b W (default 0 = off; added 2026-10-14): apply the packer's own scalar steps (validity window -> in-shot time
            interpolation -> centred W-frame median with edge padding when the shot has >= 3 frames; all-invalid shots stay NaN for the packer's base
            fallback; NO clip) to the float32-cast values, so that syn_pack.py --smooth 1 reproduces a per-scalar window (rule B7w5: q 3 / betaN 5)
MAST: rule3_guard installed first; reads only our own parts, our own record tables (map functionals + shipped input columns) and DIII-D-derived
constants.  Nothing here reads a label on either machine.
Usage: oapply.py --machine mast --run DIR --tag T --out DIR2 --newtag T2 --q95 diff --betaN keep --rec-final F.parquet --rec-base B.parquet
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def phase_id(ip, t):
    """label-free phase of one shot's Ip trace (same definition as oev.phase_id)."""
    ip = np.abs(np.asarray(ip, float)); t = np.asarray(t, float)
    ipn = ip / max(float(np.nanmax(ip)), 1.0)
    hi = ipn >= 0.9
    first = t[hi].min() if hi.any() else np.inf; last = t[hi].max() if hi.any() else -np.inf
    ph = np.full(len(t), 2)
    ph[(t < first) & (ipn < 0.5)] = 0; ph[(t < first) & (ipn >= 0.5)] = 1; ph[(t > last) & (ipn >= 0.5)] = 3; ph[(t > last) & (ipn < 0.5)] = 4
    return ph


def gate1(x, p, log):
    """exp(-e^2 / 2), e = excursion of x outside the training-target [p5, p95] in units of (p95 - p5) / 3.29 (log scale for q95)."""
    lo, hi = (np.log(p[0]), np.log(p[2])) if log else (p[0], p[2])
    sd = (hi - lo) / 3.29
    with np.errstate(all="ignore"):
        v = np.log(np.where(x > 0, x, np.nan)) if log else np.asarray(x, float)
    e = np.maximum(0.0, np.maximum(lo - v, v - hi)) / sd
    return np.exp(-0.5 * np.where(np.isfinite(e), e, 0.0) ** 2)


def rec_map(path, names, cols):
    """record table -> {shot name: {k: (cols...)}} as arrays indexed by frame."""
    import pandas as pd
    d = pd.read_parquet(path, columns=["file", "k"] + cols)
    out = {}
    for f, g in d.groupby("file"):
        out[f] = (g["k"].to_numpy(int), {c: g[c].to_numpy(float) for c in cols})
    return out


def col_of(rm, name, T, c):
    v = np.full(T, np.nan)
    if name in rm:
        kk, cc = rm[name]; m = (kk >= 0) & (kk < T)
        v[kk[m]] = cc[c][m]
    return v


def prepack(v, tt, lo, hi, w):
    """syn_pack.py's scalar steps without the clip: validity window -> interpolation in time -> centred w-frame median (edge padding, len >= 3)."""
    v = np.asarray(v, float).copy(); tt = np.asarray(tt, float)
    v[~((v > lo) & (v < hi))] = np.nan
    m = np.isfinite(v)
    if not m.any():
        return v                                     # all invalid: the packer falls back to the base file
    if not m.all():
        v[~m] = np.interp(tt[~m], tt[m], v[m])
    if w <= 1 or len(v) < 3:
        return v
    p = np.pad(v, w // 2, mode="edge")
    return np.median(np.stack([p[i:i + len(v)] for i in range(w)]), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["d3d", "mast"]); ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--newtag", required=True)
    ap.add_argument("--q95", default="keep", choices=["keep", "blend", "diff"]); ap.add_argument("--betaN", default="keep", choices=["keep", "blend", "diff", "b3"])
    ap.add_argument("--net", default=None); ap.add_argument("--inv", default=None)
    ap.add_argument("--wq", type=float, default=0.5); ap.add_argument("--wb", type=float, default=0.5); ap.add_argument("--wq-phase", default=None)
    ap.add_argument("--gate-pop", default=None); ap.add_argument("--gate-aux", action="store_true")
    ap.add_argument("--rec-final", default=None); ap.add_argument("--rec-base", default=None)
    ap.add_argument("--c-json", default=None); ap.add_argument("--b3-final", action="store_true")
    ap.add_argument("--prepack-q", type=int, default=0); ap.add_argument("--prepack-b", type=int, default=0)
    A = ap.parse_args()
    M = A.machine
    if M == "mast":
        sys.path.insert(0, str(WS / "stage11_run/ch2_geometry"))
        import rule3_guard
        rule3_guard.install()
    run, out = Path(A.run), Path(A.out); (out / "parts").mkdir(parents=True, exist_ok=True)
    names = [x for x in (run / f"files_{A.tag}_{M}.txt").read_text().split("\n") if x]
    prov = {"rule_q95": A.q95, "rule_betaN": A.betaN, "src": f"{A.run}:{A.tag}", "inputs": {}, "prepack_q": A.prepack_q, "prepack_b": A.prepack_b}
    comp = {}
    if "blend" in (A.q95, A.betaN):
        for key, spec in (("net", A.net), ("inv", A.inv)):
            r, t, cal = spec.split(":")
            lst = [x for x in (Path(r) / f"files_{t}_{M}.txt").read_text().split("\n") if x]
            comp[key] = (Path(r), t, json.load(open(cal)), {n: j for j, n in enumerate(lst)})
            prov["inputs"][f"calib_{key}"] = {"path": cal, "md5": md5(cal)}
        prov["wq"] = [float(x) for x in A.wq_phase.split(",")] if A.wq_phase else A.wq
        prov["wb"] = A.wb
    wqp = np.array([float(x) for x in A.wq_phase.split(",")]) if A.wq_phase else None
    pop = None
    if A.gate_pop:
        pop = json.load(open(A.gate_pop))["scal_stats"]; prov["inputs"]["gate_pop"] = {"path": A.gate_pop, "md5": md5(A.gate_pop)}
        prov["gate"] = {"q95": pop["q95"], "kappa": pop["kappa"], "li": pop["li"], "aux": bool(A.gate_aux)}
    gstat = []
    rf = rb = None
    need = {"q95v"} if A.q95 == "diff" else set()
    if A.betaN in ("diff", "b3"):
        need.add("betaN_e")
    if need:
        if A.rec_base:
            rb = rec_map(A.rec_base, names, sorted(need)); prov["inputs"]["rec_base"] = {"path": A.rec_base, "md5": md5(A.rec_base)}
        if A.rec_final:
            rf = rec_map(A.rec_final, names, sorted(need)); prov["inputs"]["rec_final"] = {"path": A.rec_final, "md5": md5(A.rec_final)}
    c3 = None
    if A.betaN == "b3":
        c3 = float(json.load(open(A.c_json))["c_used"]); prov["c_b3"] = c3; prov["inputs"]["c_json"] = {"path": A.c_json, "md5": md5(A.c_json)}
    st = {"n_shots": 0, "frames": 0, "q_changed": 0, "b_changed": 0, "q_fallback": 0, "b_fallback": 0}
    rq, rbb = [], []
    for i, name in enumerate(names):
        src = run / "parts" / f"{A.tag}_{M}_{i:05d}.npz"
        if not src.exists():
            continue
        z = dict(np.load(src, allow_pickle=False))
        T = len(z["t"]); ks = z["ks"].astype(int)
        q0 = z["q95"].astype(float); b0 = z["betaN"].astype(float); q, b = q0.copy(), b0.copy()
        cs = {}
        for key, (r, t, cal, pos) in comp.items():
            cs[key] = (np.load(r / "parts" / f"{t}_{M}_{pos[name]:05d}.npz", allow_pickle=False), cal)
            assert np.array_equal(cs[key][0]["ks"], z["ks"]), f"{name}: component frames differ"
        if A.q95 == "blend":
            w = wqp[phase_id(z["ip_A"], z["t"])] if wqp is not None else np.full(T, A.wq)
            if pop is not None:
                cn = cs["net"][0]
                g = gate1(cn["q95"].astype(float), pop["q95"], True)
                if A.gate_aux:
                    an = [str(x) for x in cn["aux_names"]]; ax = cn["aux"].astype(float)
                    g = g * gate1(ax[:, an.index("kappa")], pop["kappa"], False) * gate1(ax[:, an.index("li")], pop["li"], False)
                w = w * g; gstat.append(g[ks])
            lN = np.log(np.clip(cs["net"][1]["C_q95"] * cs["net"][0]["q95"].astype(float), 1e-3, None))
            lL = np.log(np.clip(cs["inv"][1]["C_q95"] * cs["inv"][0]["q95"].astype(float), 1e-3, None))
            q = np.exp(w * lN + (1 - w) * lL)
        elif A.q95 == "diff":
            hf, hb = col_of(rf, name, T, "q95v"), col_of(rb, name, T, "q95v")
            with np.errstate(all="ignore"):
                r = hf / hb
            ok = np.isfinite(r) & (r > 0.5) & (r < 2.0) & np.isfinite(q0)
            q = np.where(ok, q0 * np.where(ok, r, 1.0), q0); st["q_fallback"] += int((~ok[ks]).sum())
        if A.betaN == "blend":
            b = A.wb * cs["net"][1]["C_betaN"] * cs["net"][0]["betaN"].astype(float) + (1 - A.wb) * cs["inv"][1]["C_betaN"] * cs["inv"][0]["betaN"].astype(float)
        elif A.betaN == "diff":
            ef, eb = col_of(rf, name, T, "betaN_e"), col_of(rb, name, T, "betaN_e")
            with np.errstate(all="ignore"):
                r = ef / eb
            ok = np.isfinite(r) & (r > 0.5) & (r < 2.0) & np.isfinite(b0) & (np.where(np.isfinite(eb), eb, -1) > 0)
            b = np.where(ok, b0 * np.where(ok, r, 1.0), b0); st["b_fallback"] += int((~ok[ks]).sum())
        elif A.betaN == "b3":
            be = col_of(rf if A.b3_final else rb, name, T, "betaN_e"); bV = c3 * be
            ok = np.isfinite(bV) & (np.where(np.isfinite(bV), bV, -1) > 0.0) & (np.where(np.isfinite(bV), bV, 99) < 8.0) & np.isfinite(b0)
            b = np.where(ok, (2.0 * b0 + np.where(ok, bV, 0.0)) / 3.0, b0); st["b_fallback"] += int((~ok[ks]).sum())
        if A.prepack_q:
            q = prepack(q.astype(z["q95"].dtype).astype(float), z["t"], 1.2, 25.0, A.prepack_q)
        if A.prepack_b:
            b = prepack(b.astype(z["betaN"].dtype).astype(float), z["t"], 0.0, 8.0, A.prepack_b)
        with np.errstate(all="ignore"):
            rq.append((q / q0)[ks]); rbb.append((b / b0)[ks])
        st["n_shots"] += 1; st["frames"] += len(ks)
        st["q_changed"] += int((~np.isclose(q[ks], q0[ks], rtol=1e-12, atol=0, equal_nan=True)).sum())
        st["b_changed"] += int((~np.isclose(b[ks], b0[ks], rtol=1e-12, atol=0, equal_nan=True)).sum())
        psi0 = z["psi"]
        z["q95"] = q.astype(z["q95"].dtype); z["betaN"] = b.astype(z["betaN"].dtype)
        dst = out / "parts" / f"{A.newtag}_{M}_{i:05d}.npz"; tmp = out / "parts" / f"{A.newtag}_{M}_{i:05d}.tmp.npz"
        np.savez_compressed(tmp, **z); os.replace(tmp, dst)
        chk = np.load(dst, allow_pickle=False)
        assert chk["psi"].dtype == psi0.dtype and np.array_equal(chk["psi"], psi0, equal_nan=True), f"{name}: psi changed"
    (out / f"files_{A.newtag}_{M}.txt").write_text("\n".join(names) + "\n")
    pc = lambda v: [float(x) for x in np.nanpercentile(np.concatenate(v), [1, 5, 50, 95, 99])] if v else []  # noqa: E731
    if gstat:
        gg = np.concatenate(gstat); st["gate_mean"] = float(gg.mean()); st["gate_lt_0.9"] = float((gg < 0.9).mean()); st["gate_lt_0.5"] = float((gg < 0.5).mean())
    st.update({"q_ratio_p1_5_50_95_99": pc(rq), "b_ratio_p1_5_50_95_99": pc(rbb), **prov})
    json.dump(st, open(out / f"cfg_{A.newtag}_{M}.json", "w"), indent=1)
    json.dump({"C_q95": 1.0, "C_betaN": 1.0, "source": f"oapply: calibrated scalars, q95 {A.q95}, betaN {A.betaN}"}, open(out / f"calib_{A.newtag}.json", "w"))
    print("OAPPLY " + json.dumps(st))


if __name__ == "__main__":
    main()
