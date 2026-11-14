#!/usr/bin/env python3
"""Put a submission file written in sorted file-name order into the stream order of a configuration.

  reorder_npz.py <in.npz> <sorted_names.txt> <stream_order.txt> <out.npz>

<sorted_names.txt>: the shot file names in the order the file was written (shot_0000 = first line); <stream_order.txt>: the
file name of every stream index (tools/stream_order.py). Key shot_<i>_* of the input becomes shot_<j>_*, j = stream index of
file i; arrays are copied unchanged, keys are written in stream order, compressed as the MAST packers write. If the two orders
are identical nothing is rewritten: the output is a byte copy (same md5)."""
import shutil, sys
from pathlib import Path
import numpy as np


def main():
    src, names_f, order_f, out = sys.argv[1:5]
    names = [x for x in Path(names_f).read_text().split("\n") if x]
    order = [x for x in Path(order_f).read_text().split("\n") if x]
    assert sorted(names) == sorted(order) and len(set(order)) == len(order), "the stream order is not a permutation of the file list"
    if names == order:
        shutil.copyfile(src, out); print(f"REORDER identical order ({len(order)} shots): byte copy"); return 0
    z = np.load(src, allow_pickle=False)
    pos = {n: i for i, n in enumerate(names)}
    sub = {}
    for j, n in enumerate(order):
        i = pos[n]
        for suf in ("psirz", "q95", "betaN"):
            sub[f"shot_{j:04d}_{suf}"] = z[f"shot_{i:04d}_{suf}"]
    assert len(sub) == len(z.files), "unexpected keys in the input file"
    np.savez_compressed(out, **sub)
    moved = sum(1 for j, n in enumerate(order) if pos[n] != j)
    print(f"REORDER {len(order)} shots, {moved} moved to their stream index"); return 0


if __name__ == "__main__":
    sys.exit(main())
