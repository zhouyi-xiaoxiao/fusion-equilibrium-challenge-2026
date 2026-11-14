#!/usr/bin/env python3
"""Stage 11 / ch1_blend — integrity check of a corrected unseen-split file against its uncorrected base blend.
Checks: identical key set, float32, (n_t,65,65) shapes, finite; frame counts vs <feat-dir>/offsets.npy; q95/betaN bit-equal to
the base (corrector passes them through); per-shot psi change |corr-base| RMS / base psi sd (should be small, never ~1).
Usage: cb_verify.py --corr C.npz --base B.npz --feat-dir DIR"""
import argparse, hashlib
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--corr", required=True); ap.add_argument("--base", required=True); ap.add_argument("--feat-dir", required=True)
a = ap.parse_args()
off = np.load(f"{a.feat_dir}/offsets.npy"); n = len(off) - 1
c, b = np.load(a.corr), np.load(a.base)
assert sorted(c.files) == sorted(b.files) and len(c.files) == 3 * n, (len(c.files), len(b.files), n)
rel, qd, bd, nch = [], 0.0, 0.0, 0
for s in range(n):
    k = f"shot_{s:04d}_"
    pc, pb = c[k + "psirz"], b[k + "psirz"]
    assert pc.dtype == np.float32 and pc.shape == (off[s + 1] - off[s], 65, 65) and np.isfinite(pc).all(), (s, pc.dtype, pc.shape)
    for f in ("q95", "betaN"):
        assert c[k + f].dtype == np.float32 and c[k + f].shape == (off[s + 1] - off[s],)
    qd = max(qd, float(np.abs(c[k + "q95"] - b[k + "q95"]).max())); bd = max(bd, float(np.abs(c[k + "betaN"] - b[k + "betaN"]).max()))
    d = pc.astype(np.float64) - pb
    nch += int((np.abs(d).reshape(len(d), -1).max(1) > 0).sum())
    rel.append(np.sqrt((d ** 2).mean()) / (pb.std() + 1e-12))
rel = np.array(rel)
h = hashlib.md5()
with open(a.corr, "rb") as f:
    for blk in iter(lambda: f.read(1 << 24), b""): h.update(blk)
print(f"[verify] {n} shots {off[-1]} frames OK (float32, shapes, finite); frames changed {nch}/{off[-1]} ({nch/off[-1]:.4f}); "
      f"max|dq95| {qd:.3g} max|dbetaN| {bd:.3g}; per-shot rel psi change median {np.median(rel):.4f} p99 {np.quantile(rel,.99):.4f} "
      f"max {rel.max():.4f} (shot {rel.argmax()}); md5 {h.hexdigest()}", flush=True)
