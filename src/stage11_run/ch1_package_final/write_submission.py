#!/usr/bin/env python3
"""
Stage 11 / ch1_package — write the OFFICIAL DIII-D submission file from a prediction file in feature-row order.

  python write_submission.py --pred P.npz --feat-dir FEAT --order stream_order.txt|sorted --out diii_d_public_test.npz

--pred     harness layout shot_XXXX_{psirz,q95,betaN}, XXXX = row of FEAT/files.txt (sorted parquet names; this is what
           fm_predict / predict_unseen / cb_test / the corrector all write). A sidecar P.npz.files.txt, if present,
           must equal FEAT/files.txt.
--feat-dir the inputs-only FEATURE-V2.1 directory of the same shots (files.txt, offsets.npy, times.npy).
--order    file with the local parquet names in HF STREAM order (check_order.py), or 'sorted' (= FEAT/files.txt order).
--out      shot_JJJJ_psirz (T,65,65) FLOAT32, shot_JJJJ_q95 / shot_JJJJ_betaN (T,) float32, JJJJ = stream index.
           psi is copied value-for-value (float32 source: bit-identical; never np.round, never float16).
           <out>.manifest.json: md5, bytes, n_shots, n_frames, input md5, per-key shot map; <out>.files.txt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def rd(p) -> list[str]:
    return [x.strip() for x in Path(p).read_text().split("\n") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--feat-dir", required=True)
    ap.add_argument("--order", default="sorted")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compress", action="store_true", help="np.savez_compressed (default: np.savez, as the scored f32 files)")
    a = ap.parse_args()
    t0 = time.time()
    feat, pred, out = Path(a.feat_dir), Path(a.pred), Path(a.out)
    files = rd(feat / "files.txt")
    off = np.load(feat / "offsets.npy")
    times = np.load(feat / "times.npy")
    n = len(files)
    assert len(off) == n + 1 and off[-1] == len(times), (len(off), n, off[-1], len(times))
    side = Path(str(pred) + ".files.txt")
    if side.exists():
        assert rd(side) == files, "pred sidecar files.txt differs from FEAT/files.txt"
    order = list(files) if a.order == "sorted" else rd(a.order)
    assert len(order) == n and len(set(order)) == n and sorted(order) == sorted(files), "order must be a permutation of FEAT files"
    pos = {f: i for i, f in enumerate(files)}
    P = np.load(pred)
    assert len(P.files) == 3 * n, (len(P.files), n)
    sub, fmap = {}, []
    qmin, qmax, bmin, bmax = np.inf, -np.inf, np.inf, -np.inf
    src_dtypes = set()
    for j, fname in enumerate(order):
        i = pos[fname]
        T = int(off[i + 1] - off[i])
        psi = P[f"shot_{i:04d}_psirz"]
        q = P[f"shot_{i:04d}_q95"]
        b = P[f"shot_{i:04d}_betaN"]
        src_dtypes.update({str(psi.dtype), str(q.dtype), str(b.dtype)})
        assert psi.shape == (T, 65, 65) and q.shape == (T,) and b.shape == (T,), (fname, psi.shape, q.shape, b.shape, T)
        psi32 = psi.astype(np.float32, copy=False)
        assert np.isfinite(psi32).all() and np.isfinite(q).all() and np.isfinite(b).all(), f"non-finite in {fname}"
        sub[f"shot_{j:04d}_psirz"] = psi32
        sub[f"shot_{j:04d}_q95"] = q.astype(np.float32, copy=False)
        sub[f"shot_{j:04d}_betaN"] = b.astype(np.float32, copy=False)
        qmin, qmax = min(qmin, float(q.min())), max(qmax, float(q.max()))
        bmin, bmax = min(bmin, float(b.min())), max(bmax, float(b.max()))
        fmap.append({"key": f"shot_{j:04d}", "file": fname, "src_row": i, "T": T})
    t1 = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    (np.savez_compressed if a.compress else np.savez)(out, **sub)
    t2 = time.time()
    Path(str(out) + ".files.txt").write_text("\n".join(order) + "\n")
    m = {"file": str(out), "md5": md5(out), "bytes": out.stat().st_size, "n_shots": n, "n_frames": int(off[-1]),
         "psirz_dtype": "float32", "scalar_dtype": "float32", "source_dtypes": sorted(src_dtypes),
         "order": a.order, "order_is_identity": order == files, "pred": str(pred), "pred_md5": md5(pred),
         "feat_dir": str(feat), "q95_range": [qmin, qmax], "betaN_range": [bmin, bmax],
         "seconds": {"read_check": round(t1 - t0, 1), "write": round(t2 - t1, 1)}, "shot_map": fmap}
    Path(str(out) + ".manifest.json").write_text(json.dumps(m, indent=1))
    print(f"[write_submission] {out} {m['bytes']} bytes md5 {m['md5']} | {n} shots {off[-1]} frames | order "
          f"{'identity' if m['order_is_identity'] else a.order} | src dtypes {m['source_dtypes']} | q95 [{qmin:.3f},{qmax:.3f}] "
          f"betaN [{bmin:.3f},{bmax:.3f}] | {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
