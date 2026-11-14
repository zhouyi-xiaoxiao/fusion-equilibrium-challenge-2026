#!/usr/bin/env python3
"""ch1_package_final — base blend, arithmetic of e_cx/cx_full_assemble.py (= ch1_blend/cb_test.py convention):
  psi   = float32( 0.5 * (float32(FM bag mean map) + float32(TW bag map)) )
  q95 / betaN = SC bl_d10: float64 mean over the 7 families of the per-family seed means of the float32 member outputs.
  pf_blend.py --fm fm.npz --tw tw.npz --sc sc.npz --feat-dir DIR --out blend.npz"""
import argparse
from pathlib import Path

import numpy as np

SC_FAMS = {"gru_d10_fz": [0, 1], "la1_d10_e45": [0, 1], "la1_d10wd": [0, 1], "la1_d10": [0, 1, 2, 3, 4], "la1_d20": [0, 1],
           "la1_d10_phys": [0, 1], "ra1_d10": [0, 1]}
ap = argparse.ArgumentParser()
for k in ("fm", "tw", "sc", "feat-dir", "out"):
    ap.add_argument("--" + k, required=True)
a = ap.parse_args()
off = np.load(f"{a.feat_dir}/offsets.npy"); n = len(off) - 1; N = int(off[-1])
files = Path(f"{a.feat_dir}/files.txt").read_text()
fm, tw, sc = np.load(a.fm), np.load(a.tw), np.load(a.sc)
assert (sc["offsets"] == off).all(), "SC offsets differ from the feature offsets"
fam = []
for f, seeds in SC_FAMS.items():
    q = [sc[f"q95_{f}_s{s}"] for s in seeds]; b = [sc[f"betaN_{f}_s{s}"] for s in seeds]
    assert all(x.dtype == np.float32 and len(x) == N for x in q + b)
    fam.append((np.mean([x.astype(np.float64) for x in q], 0), np.mean([x.astype(np.float64) for x in b], 0)))
q = np.mean([x for x, _ in fam], 0).astype(np.float32); b = np.mean([y for _, y in fam], 0).astype(np.float32)
assert np.isfinite(q).all() and np.isfinite(b).all()
d, rel = {}, []
for s in range(n):
    p, e = off[s], off[s + 1]
    pf, pt = fm[f"shot_{s:04d}_psirz"], tw[f"shot_{s:04d}_psirz"]
    assert pf.dtype == pt.dtype == np.float32 and pf.shape == pt.shape == (e - p, 65, 65), (s, pf.shape, pt.shape)
    pf, pt = pf.astype(np.float64), pt.astype(np.float64)
    rel.append(np.sqrt(((pf - pt) ** 2).mean()) / (pf.std() + 1e-12))
    psi = (0.5 * (pf + pt)).astype(np.float32)
    assert np.isfinite(psi).all()
    d[f"shot_{s:04d}_psirz"] = psi; d[f"shot_{s:04d}_q95"] = q[p:e]; d[f"shot_{s:04d}_betaN"] = b[p:e]
np.savez(a.out, **d)
Path(a.out + ".files.txt").write_text(files)
rel = np.array(rel)
print(f"PF_BLEND {n} shots {N} frames; FM-vs-TW rel RMS / psi sd: median {np.median(rel):.4f} p99 {np.quantile(rel, .99):.4f} max {rel.max():.4f} (shot {rel.argmax()}); "
      f"q95 [{q.min():.3f}, {q.max():.3f}] betaN [{b.min():.3f}, {b.max():.3f}]", flush=True)
