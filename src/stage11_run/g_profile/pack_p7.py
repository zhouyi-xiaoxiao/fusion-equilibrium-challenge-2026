#!/usr/bin/env python3
"""
[Stage 11G profile track: verbatim copy of e_mast_inv6/pack_inv6.py (md5 3a39ae86...), used with --q-factor = the pipeline's own
 DIII-D pseudo calibration C_q95 (the frozen INV_P6_k50_keepCq scalar convention); only this header was added.]
INV6 packaging with the scalars computed CONSISTENTLY ON THE INV6 MAPS:
  q95   = the raw contour integral F(psi95)/(2 pi) * closed-integral dl/(R |grad psi|) on the 95 % surface of the inverse's own
          solution (F95 from the fitted FF'; F95/F_vac - 1 ~ 1e-3) -- no calibration factor (the DIII-D q95 gap of this
          pipeline is the model's elongation deficit, see diag_kappa_d3d.py: review rule 'never patch the DIII-D kappa offset')
  betaN = c * beta_N,e  (scal2 method: Thomson p_e mapped onto rho = sqrt(psiN) of THIS map, monotone binned, volume averaged
          inside the LCFS; c = coverage-matched DIII-D factor from scal_c.py on this pipeline's DIII-D maps).
          Frames without Thomson: time interpolation within the shot; a shot without any: the inverse's own kinetic beta_N.
  both  : centred 3-frame running median along the shot (full-shot runs only), then the packager clips (q 1.2-25, beta_N 0-8)
  psi   : float32 with the mantissa truncated to --psi-bits bits (binary truncation, as INV_P5_k50) or plain float32 (0)
Outputs: --out <npz> (submission layout shot_XXXX_{psirz,q95,betaN}, HF stream order = sorted file names, asserted against
--order-file) and/or --parts-out <dir> (copies of the run's parts with q95/betaN replaced, + frame table / file list links,
for eval_frames.py on DIII-D subsets). MAST: rule3_guard installed, reads only our own parts / tables / JSON.
Usage: pack_inv6.py --run DIR --tag T --machine mast|d3d --scal <parquet> --c-json <json> [--out npz] [--parts-out DIR]
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ch2_geometry"))
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--machine", required=True)
ap.add_argument("--scal", required=True); ap.add_argument("--c-json", required=True)
ap.add_argument("--out", default=None); ap.add_argument("--parts-out", default=None); ap.add_argument("--order-file", default=None)
ap.add_argument("--psi-bits", type=int, default=24); ap.add_argument("--smooth", type=int, default=3)
ap.add_argument("--q-factor", type=float, default=1.0)
a = ap.parse_args()
if a.machine == "mast":
    import rule3_guard
    rule3_guard.install()
run = Path(a.run)
files = [x for x in (run / f"files_{a.tag}_{a.machine}.txt").read_text().split("\n") if x]
assert files == sorted(files), "file list must be sorted (= HF stream order)"
if a.order_file:
    order = [x for x in Path(a.order_file).read_text().split("\n") if x]
    assert order == files, "file list differs from the recorded HF stream order"
cj = json.load(open(a.c_json))
c = float(cj["c_used"])
sc = pd.read_parquet(a.scal, columns=["idx", "k", "betaN_e"])
g = {i: s for i, s in sc.groupby("idx")}


def rmed(x, w):
    return x if w <= 1 else pd.Series(x).rolling(w, center=True, min_periods=1).median().values


st = {"n_shots": len(files), "frames": 0, "ok": 0, "b_thomson": 0, "b_interp": 0, "b_kin_shots": 0, "q_nan": 0}
sub, qs, bs = {}, [], []
if a.parts_out:
    po = Path(a.parts_out); (po / "parts").mkdir(parents=True, exist_ok=True)
    for f in (f"files_{a.tag}_{a.machine}.txt", f"frames_{a.tag}_{a.machine}.parquet", f"cfg_{a.tag}_{a.machine}.json"):
        if (run / f).exists():
            shutil.copy(run / f, po / f)
t0 = time.time()
for i, f in enumerate(files):
    pth = run / "parts" / f"{a.tag}_{a.machine}_{i:05d}.npz"
    cz = dict(np.load(pth, allow_pickle=False))
    psi = cz["psi"].astype(np.float32)
    T = psi.shape[0]
    ks = cz["ks"].astype(int)
    full = len(ks) == T
    tt = cz["t"].astype(float)
    q = a.q_factor * cz["q95"].astype(float)
    q[~((q > 1.2) & (q < 25.0))] = np.nan
    st["q_nan"] += int(np.isnan(q[ks]).sum())
    be = np.full(T, np.nan)
    s = g.get(i)
    if s is not None:
        kk = s.k.values.astype(int); m = (kk >= 0) & (kk < T)
        be[kk[m]] = s.betaN_e.values[m]
    b = c * be
    b[~((b > 0) & (b < 8.0))] = np.nan
    mk = np.zeros(T, bool); mk[ks] = True
    st["b_thomson"] += int(np.isfinite(b[ks]).sum())
    if np.isfinite(b[ks]).any():
        st["b_interp"] += int((~np.isfinite(b[ks])).sum())
    else:
        st["b_kin_shots"] += 1
        b = cz["betaN"].astype(float).copy()
        b[~((b > 0) & (b < 8.0))] = np.nan
    for arr, dflt in ((q, 5.0), (b, 1.5)):
        m = np.isfinite(arr) & mk
        bad = ~np.isfinite(arr)
        arr[bad] = np.interp(tt[bad], tt[m], arr[m]) if m.any() else dflt
    if full:
        q, b = rmed(q, a.smooth), rmed(b, a.smooth)
    q, b = np.clip(q, 1.2, 25.0), np.clip(b, 0.0, 8.0)
    st["frames"] += T; st["ok"] += int(cz["ok"].sum())
    qs.append(q[ks]); bs.append(b[ks])
    if a.out:
        if a.psi_bits:
            mm, ee = np.frexp(psi.astype(np.float64))
            p_ = np.ldexp(np.round(mm * 2.0 ** a.psi_bits) / 2.0 ** a.psi_bits, ee).astype(np.float32)
        else:
            p_ = psi
        assert p_.shape[1:] == (65, 65) and np.isfinite(p_).all(), f"bad psi in {f}"
        sub[f"shot_{i:04d}_psirz"] = p_
        sub[f"shot_{i:04d}_q95"] = q.astype(np.float32)
        sub[f"shot_{i:04d}_betaN"] = b.astype(np.float32)
    if a.parts_out:
        cz["q95"], cz["betaN"] = q, b
        np.savez_compressed(Path(a.parts_out) / "parts" / pth.name, **cz)
qs, bs = np.concatenate(qs), np.concatenate(bs)
man = {"run": str(run), "tag": a.tag, "machine": a.machine, "c_betaN": c, "c_json": a.c_json, "c_source": cj.get("src"), "q_factor": a.q_factor,
       "smooth": a.smooth, "psi_bits": a.psi_bits, **st, "q95_p1_50_99": [float(x) for x in np.percentile(qs, [1, 50, 99])],
       "betaN_p1_50_99": [float(x) for x in np.percentile(bs, [1, 50, 99])], "secs": time.time() - t0,
       "rule3": "MAST: own parts + own scal table + DIII-D-derived c only; no MAST label, no envelope, no demo" if a.machine == "mast" else "D3D"}
if a.out:
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **sub)
    man["out"] = a.out
    man["md5"] = hashlib.md5(Path(a.out).read_bytes()).hexdigest()
    man["bytes"] = Path(a.out).stat().st_size
    if a.machine == "mast":
        man["cfg"] = json.load(open(run / f"cfg_{a.tag}_mast.json"))
    json.dump(man, open(Path(a.out).parent / "manifest.json", "w"), indent=1)
print("PACK " + json.dumps({k: v for k, v in man.items() if k != "cfg"}))
