#!/usr/bin/env python3
"""Derive the scorer's shot order (= stream order of the Hugging Face configuration) for a local directory of shot files.

  stream_order.py --shot-dir DIR --repo Sophelio/fusion-equilibrium-challenge-final --config diii_d_private_test --out-dir OUT [--split SPLIT]

The scorer and the validator index shots by the stream order of load_dataset(REPO, CONFIG, split=SPLIT, streaming=True)
(shot_0000 = first streamed row). Streamed rows carry no file name, so each row is matched to a local parquet file by a hash of
its efit_times array, ties broken by the first 64 samples of magnetics_time and of every other magnetics_* column.
Writes OUT/stream_order.txt (local file name per stream index) and OUT/order_check.json. Exit code 0 only if every streamed
row maps to exactly one distinct local file. If no --split is given the only split of the configuration is used."""
from __future__ import annotations

import argparse, hashlib, json, time
from pathlib import Path

import numpy as np


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] [order] {m}", flush=True)


def sig(row, cols, full):
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(row["efit_times"], dtype=np.float64)).tobytes())
    if full:
        for c in cols:
            v = row[c]
            if v is None:
                continue
            h.update(c.encode()); h.update(np.ascontiguousarray(np.asarray(v, dtype=np.float64).ravel()[:64]).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--repo", required=True); ap.add_argument("--config", required=True); ap.add_argument("--split", default=None)
    a = ap.parse_args()
    import pyarrow.parquet as pq
    from datasets import get_dataset_split_names, load_dataset
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    files = sorted(Path(a.shot_dir).glob("*.parquet"))
    cols = sorted(c for c in pq.read_schema(files[0]).names if c.startswith("magnetics_") and not c.startswith("magnetics_dsep"))
    local = {}
    for f in files:
        tb = pq.read_table(f, columns=["efit_times"] + cols)
        row = {c: tb[c][0].as_py() for c in ["efit_times"] + cols}
        local.setdefault(sig(row, cols, False), []).append((f.name, sig(row, cols, True)))
    log(f"{len(files)} local files, {len(local)} distinct efit_times signatures, {len(cols)} tie-break columns ({time.time() - t0:.0f}s)")
    split = a.split
    if split is None:
        splits = get_dataset_split_names(a.repo, a.config)
        if len(splits) != 1:
            raise SystemExit(f"give --split, one of {splits}")
        split = splits[0]
    ds = load_dataset(a.repo, a.config, split=split, streaming=True)
    order, unmatched, ambiguous = [], [], []
    for i, row in enumerate(ds):
        cands = local.get(sig(row, cols, False), [])
        if len(cands) == 1:
            order.append(cands[0][0])
        else:
            sf = sig(row, cols, True) if cands else None
            hit = [c[0] for c in cands if c[1] == sf]
            if len(hit) == 1:
                order.append(hit[0]); ambiguous.append({"i": i, "resolved": hit[0], "n_cands": len(cands)})
            else:
                order.append(None); unmatched.append({"i": i, "n_cands": len(cands), "n_hit": len(hit)})
        if (i + 1) % 200 == 0:
            log(f"  streamed {i + 1} rows ({time.time() - t0:.0f}s)")
    names = [f.name for f in files]
    res = {"repo": a.repo, "config": a.config, "split": split, "n_stream": len(order), "n_local": len(files),
           "n_matched": sum(o is not None for o in order), "identical_to_sorted": order == names,
           "n_unique": len(set(o for o in order if o)), "n_unmatched": len(unmatched), "unmatched": unmatched[:20],
           "n_ambiguous_resolved": len(ambiguous), "seconds": round(time.time() - t0, 1)}
    (out / "order_check.json").write_text(json.dumps(res, indent=1))
    ok = res["n_matched"] == res["n_stream"] == res["n_local"] == res["n_unique"]
    if ok:
        (out / "stream_order.txt").write_text("\n".join(order) + "\n")
    log(f"ORDER_CHECK {'OK' if ok else 'FAILED'}: " + json.dumps({k: v for k, v in res.items() if k != 'unmatched'}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
