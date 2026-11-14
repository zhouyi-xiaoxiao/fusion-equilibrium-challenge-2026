#!/usr/bin/env python3
"""Split a 150-shot harness prediction file into halves A (shots 0-74) and B (75-149), re-indexed from 0."""
import sys

import numpy as np

for f in sys.argv[1:]:
    z = np.load(f)
    n = len([k for k in z.files if k.endswith("_psirz")])
    for nm, sh in (("A", range(0, n // 2)), ("B", range(n // 2, n))):
        d = {}
        for j, s in enumerate(sh):
            for k in ("psirz", "q95", "betaN"):
                d[f"shot_{j:04d}_{k}"] = z[f"shot_{s:04d}_{k}"]
        np.savez(f.replace(".npz", f"_{nm}.npz"), **d)
