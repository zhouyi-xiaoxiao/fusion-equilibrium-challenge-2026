#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — packaging of a MAST candidate (submission layout shot_XXXX_{psirz,q95,betaN}).

  psi    from --parts RUN:TAG (G1-layout parts: the net's own maps, or inverse maps after syn_pin.py), float32 with the mantissa
         truncated to --psi-bits bits (binary truncation, the INV_P5..P7 convention)
  q95    'net:RUN:TAG:calib.json'  = C_q95 x q95 head of a syn_predict run (C_q95 = that pipeline's own DIII-D train calibration)
         'base'                    = copied from --base-sub (e.g. the audited INV_P7a file)
  betaN  'net:RUN:TAG:calib.json'  = C_betaN x betaN head            | 'base'
  Scalars from the net get the inverse's conventions: validity window (q 1.2-25, betaN 0-8; invalid -> time interpolation
  within the shot, then the base file's value), centred 3-frame running median, clip.
HF stream order = sorted file names, asserted against --order-file and against the base file's shot count / frame counts.
Label-free: rule3_guard installed; reads only our own parts, our own base submission file and DIII-D-derived calibration JSON.
Usage: syn_pack.py --parts RUN:TAG --q95 net:RUN:TAG:calib.json --betaN base --base-sub <npz> --name NAME --out-dir DIR [--order-file F]
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ws = Path(os.environ["WS"])
sys.path.insert(0, str(ws / "stage11_run/ch2_geometry"))
import rule3_guard  # noqa: E402

rule3_guard.install()
ap = argparse.ArgumentParser()
ap.add_argument("--parts", required=True); ap.add_argument("--q95", required=True); ap.add_argument("--betaN", required=True)
ap.add_argument("--base-sub", default=None); ap.add_argument("--name", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--order-file", default=None); ap.add_argument("--psi-bits", type=int, default=24); ap.add_argument("--smooth", type=int, default=3)
a = ap.parse_args()
run, tag = a.parts.rsplit(":", 1)
run = Path(run)
files = [x for x in (run / f"files_{tag}_mast.txt").read_text().split("\n") if x]
assert files == sorted(files), "file list must be sorted (= HF stream order)"
if a.order_file:
    assert [x for x in Path(a.order_file).read_text().split("\n") if x] == files, "file list differs from the recorded HF stream order"
base = np.load(a.base_sub) if a.base_sub else None


def src(spec):
    if spec == "base":
        assert base is not None, "--base-sub needed"
        return None
    kind, r, t, cal = spec.split(":", 3)
    assert kind == "net"
    fl = [x for x in (Path(r) / f"files_{t}_mast.txt").read_text().split("\n") if x]
    assert fl == files, "scalar source lists different shots"
    return Path(r), t, json.load(open(cal))


def rmed(x, w):
    if w <= 1 or len(x) < 3:
        return x
    p = np.pad(x, w // 2, mode="edge")
    return np.median(np.stack([p[i:i + len(x)] for i in range(w)]), axis=0)


qs_, bs_ = src(a.q95), src(a.betaN)
st = {"n_shots": len(files), "frames": 0, "q_from_net": 0, "q_fallback_base": 0, "b_from_net": 0, "b_fallback_base": 0}
sub, qall, ball = {}, [], []
t0 = time.time()
for i, f in enumerate(files):
    c = np.load(run / "parts" / f"{tag}_mast_{i:05d}.npz", allow_pickle=False)
    psi = c["psi"].astype(np.float32)
    T = psi.shape[0]
    assert psi.shape[1:] == (65, 65) and np.isfinite(psi).all(), f"bad psi in {f}"
    tt = c["t"].astype(float)
    out = {}
    for nm, s_, lo, hi, dflt, key in (("q95", qs_, 1.2, 25.0, 5.0, "C_q95"), ("betaN", bs_, 0.0, 8.0, 1.5, "C_betaN")):
        bv = base[f"shot_{i:04d}_{nm}"].astype(float) if base is not None else None
        if bv is not None:
            assert len(bv) == T, f"frame count differs from the base file in {f}"
        if s_ is None:
            out[nm] = bv
            continue
        z = np.load(s_[0] / "parts" / f"{s_[1]}_mast_{i:05d}.npz", allow_pickle=False)
        v = s_[2][key] * z[nm].astype(float)
        v[~((v > lo) & (v < hi))] = np.nan
        m = np.isfinite(v)
        st[("q" if nm == "q95" else "b") + "_from_net"] += int(m.sum())
        if m.any() and not m.all():
            v[~m] = np.interp(tt[~m], tt[m], v[m])
        elif not m.any():
            v = bv.copy() if bv is not None else np.full(T, dflt)
            st[("q" if nm == "q95" else "b") + "_fallback_base"] += T
        out[nm] = np.clip(rmed(v, a.smooth), lo, hi)
    if a.psi_bits:
        mm, ee = np.frexp(psi.astype(np.float64))
        psi = np.ldexp(np.round(mm * 2.0 ** a.psi_bits) / 2.0 ** a.psi_bits, ee).astype(np.float32)
    sub[f"shot_{i:04d}_psirz"] = psi
    sub[f"shot_{i:04d}_q95"] = out["q95"].astype(np.float32)
    sub[f"shot_{i:04d}_betaN"] = out["betaN"].astype(np.float32)
    st["frames"] += T
    qall.append(out["q95"]); ball.append(out["betaN"])
od = Path(a.out_dir) / "submission" / a.name
od.mkdir(parents=True, exist_ok=True)
path = od / "mast_public_test.npz"
assert not path.exists(), f"refusing to overwrite {path}"
np.savez_compressed(path, **sub)
qall, ball = np.concatenate(qall), np.concatenate(ball)
man = {"name": a.name, "parts": a.parts, "q95": a.q95, "betaN": a.betaN, "base_sub": a.base_sub, "psi_bits": a.psi_bits, "smooth": a.smooth, **st,
       "q95_p1_50_99": [float(x) for x in np.percentile(qall, [1, 50, 99])], "betaN_p1_50_99": [float(x) for x in np.percentile(ball, [1, 50, 99])],
       "md5": hashlib.md5(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size, "secs": time.time() - t0,
       "parts_cfg": json.load(open(run / f"cfg_{tag}_mast.json")) if (run / f"cfg_{tag}_mast.json").exists() else None,
       "rule3": "MAST input columns + public geometry + our own synthetic equilibria / own inverse maps + DIII-D-derived calibration only; "
                "no MAST label, no envelope, no demo shot, no board number"}
json.dump(man, open(od / "manifest_build.json", "w"), indent=1, default=float)
print("PACK " + json.dumps({k: v for k, v in man.items() if k != "parts_cfg"}))
