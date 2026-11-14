#!/usr/bin/env python3
"""
Stage 11 / ch1_package — HF stream-order check (generalised copy of stage6 L_ch1_finish_package/check_order.py).
The scorer / validator index shots by STREAM ORDER of load_dataset(REPO, CONFIG, split=SPLIT, streaming=True)
(shot_0000 = first streamed row). HF rows carry no shot id, so each streamed row is matched to a local parquet by its
efit_times array (exact float64), ties broken by magnetics_time length + the first 64 ECOILA samples.

  python check_order.py --test-dir DIR --out-dir OUT [--repo Sophelio/fusion-equilibrium-challenge]
                        [--config diii_d_public_test] [--split public_test]
Writes OUT/stream_order.txt (local file name per stream index) and OUT/order_check.json. Exit 0 only if every streamed
row maps to exactly one distinct local file. Needs compute-node egress; set HF_HOME on scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [order] {m}", flush=True)


def sig_times(t):
    return hashlib.sha1(np.ascontiguousarray(np.asarray(t, dtype=np.float64)).tobytes()).hexdigest()


def sig_full(t, mt, ec):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(t, dtype=np.float64)).tobytes())
    h.update(np.ascontiguousarray(np.asarray(mt, dtype=np.float64)[:64]).tobytes())
    h.update(np.ascontiguousarray(np.asarray(ec, dtype=np.float32)[:64]).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", default="Sophelio/fusion-equilibrium-challenge")
    ap.add_argument("--config", default="diii_d_public_test")
    ap.add_argument("--split", default="public_test")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from datasets import load_dataset
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(Path(a.test_dir).glob("*.parquet"))
    local = {}
    for f in files:
        tb = pq.read_table(f, columns=["efit_times", "magnetics_time", "magnetics_ECOILA"])
        t = np.asarray(tb["efit_times"][0].as_py(), dtype=np.float64)
        mt = np.asarray(tb["magnetics_time"][0].as_py(), dtype=np.float64)
        ec = np.asarray(tb["magnetics_ECOILA"][0].as_py(), dtype=np.float32)
        local.setdefault(sig_times(t), []).append((f.name, sig_full(t, mt, ec)))
    log(f"{len(files)} local files, {len(local)} distinct efit_times signatures ({time.time()-t0:.0f}s)")
    ds = load_dataset(a.repo, a.config, split=a.split, streaming=True)
    order, unmatched, ambiguous = [], [], []
    for i, row in enumerate(ds):
        t = np.asarray(row["efit_times"], dtype=np.float64)
        cands = local.get(sig_times(t), [])
        if len(cands) == 1:
            order.append(cands[0][0])
        elif len(cands) > 1:
            sf = sig_full(t, row["magnetics_time"], row["magnetics_ECOILA"])
            hit = [c[0] for c in cands if c[1] == sf]
            if len(hit) == 1:
                order.append(hit[0]); ambiguous.append({"i": i, "resolved": hit[0], "n_cands": len(cands)})
            else:
                order.append(None); unmatched.append({"i": i, "T": int(len(t)), "n_cands": len(cands), "n_hit": len(hit)})
        else:
            order.append(None); unmatched.append({"i": i, "T": int(len(t)), "n_cands": 0})
        if (i + 1) % 200 == 0:
            log(f"  streamed {i+1} rows ({time.time()-t0:.0f}s)")
    names = [f.name for f in files]
    res = {"repo": a.repo, "config": a.config, "split": a.split, "n_stream": len(order), "n_local": len(files),
           "n_matched": sum(o is not None for o in order), "identical_to_sorted": order == names,
           "n_unique": len(set(o for o in order if o)), "n_unmatched": len(unmatched), "unmatched": unmatched[:20],
           "n_ambiguous_resolved": len(ambiguous), "ambiguous": ambiguous[:20], "seconds": round(time.time() - t0, 1)}
    (out / "order_check.json").write_text(json.dumps(res, indent=1))
    ok = res["n_matched"] == res["n_stream"] == res["n_local"] == res["n_unique"]
    if ok:
        (out / "stream_order.txt").write_text("\n".join(order) + "\n")
    log(f"ORDER_CHECK {'OK' if ok else 'FAILED'}: " + json.dumps({k: v for k, v in res.items() if k not in ('unmatched', 'ambiguous')}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
