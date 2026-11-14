#!/usr/bin/env python3
"""
Offline mirror of starter/validate_submission.py: the SAME structural checks (keys, (T,65,65)
float psirz, (T,) float q95/betaN, no extra keys, decimal-rounding warning) but T comes from the
LOCAL parquet files instead of streaming the inputs from the Hub. Shot i of the file (keys shot_{i:04d}_*) is
checked against the local file named on line i of --order <stream_order.txt> (tools/stream_order.py: the
configuration's stream order, which the scorer uses); without --order, against the i-th file in sorted
file-name order (for the public-test configurations sorted order == Hugging Face stream order, verified
in an internal development note). Use it when the Hub is unreachable; the official validator remains the record.

    python validate_local.py submission/mast_public_test.npz --shot-dir <data>/mast [--order <stream_order.txt>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "starter"))
from validate_submission import _quantization_warning  # noqa: E402

SCALARS = ["q95", "betaN"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path)
    ap.add_argument("--shot-dir", type=Path, required=True)
    ap.add_argument("--machine", default="MAST", choices=["MAST", "DIII-D"])
    ap.add_argument("--order", type=Path, default=None,
                    help="stream order file (one parquet file name per line; line i = shot_{i:04d}); default: sorted file names")
    args = ap.parse_args()
    H = W = 65
    files = sorted(args.shot_dir.glob("*.parquet"))
    order_note = "sorted file names"
    if args.order is not None:
        names = [x.strip() for x in args.order.read_text().split("\n") if x.strip()]
        local = {f.name: f for f in files}
        bad = sorted(set(names) - set(local)) + sorted(set(local) - set(names))
        if bad or len(set(names)) != len(names):
            print(f"FAIL the order file {args.order} is not a permutation of the {len(files)} local parquet files "
                  f"({len(names)} lines, {len(set(names))} distinct, e.g. {bad[:3]})")
            return 1
        order_note = (f"stream order {args.order.name} ({len(names)} names, "
                      f"{'identical to' if names == [f.name for f in files] else 'differs from'} sorted)")
        files = [local[n] for n in names]
    sub = np.load(args.npz, allow_pickle=False)          # arrays are read one at a time
    keys = set(sub.files)
    print(f"Validating {args.npz.name} against {len(files)} local shots ({args.machine}, grid {(H, W)}, order: {order_note}); {len(keys)} arrays")
    errors, warnings = [], []
    for i, f in enumerate(files):
        T = len(np.asarray(pq.read_table(f, columns=["efit_times"]).column("efit_times")[0].as_py()))
        prefix = f"shot_{i:04d}"
        k = f"{prefix}_psirz"
        if k not in keys:
            errors.append(f"{k}: MISSING (expected ({T}, {H}, {W}))")
        else:
            arr = sub[k]
            if arr.shape != (T, H, W):
                errors.append(f"{k}: shape {arr.shape}, expected ({T}, {H}, {W})")
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"{k}: dtype {arr.dtype}, expected float")
            if args.machine == "DIII-D" and not np.isfinite(arr).all():
                errors.append(f"{k}: contains NaN/Inf")
            w = _quantization_warning(arr, k)
            if w:
                warnings.append(w)
        for name in SCALARS:
            k = f"{prefix}_{name}"
            if k not in keys:
                errors.append(f"{k}: MISSING (expected ({T},))")
                continue
            arr = sub[k]
            if arr.shape != (T,):
                errors.append(f"{k}: shape {arr.shape}, expected ({T},)")
            if not np.issubdtype(arr.dtype, np.floating):
                errors.append(f"{k}: dtype {arr.dtype}, expected float")
    expected = {f"shot_{i:04d}_{s}" for i in range(len(files)) for s in ("psirz", *SCALARS)}
    extra = sorted(k for k in keys if k not in expected)
    if extra:
        errors.append(f"unexpected keys: {extra[:5]}")
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for w in warnings[:10]:
            print("    -", w)
    if errors:
        print(f"\nFAIL {len(errors)} problem(s):")
        for e in errors[:40]:
            print("    -", e)
        return 1
    print(f"\nOK -- {len(files)} shots, all shapes/keys/dtype valid (local mirror of the official validator).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
