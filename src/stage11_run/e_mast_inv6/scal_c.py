#!/usr/bin/env python3
"""DIII-D ONLY (labels allowed): the scal2 total/electron pressure factor c for beta_N = c * beta_N,e on the maps of ONE
pipeline (scal_frames.py table of that pipeline's DIII-D maps): least squares through the origin of beta_N(label) on
beta_N,e, by Thomson core coverage (rho_min of the frame's real-chord Thomson on THIS map), frames with Ip > 200 kA
(exactly the an earlier check script definition used for G1_soft_fix_k50_scal2).
Usage: scal_c.py <scal parquet> <out json> [--use core|all]"""
import json
import sys

import numpy as np
import pandas as pd

src, outp = sys.argv[1], sys.argv[2]
use = sys.argv[sys.argv.index("--use") + 1] if "--use" in sys.argv else "core"


def q(x, p=(10, 25, 50, 75, 90)):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return [round(float(v), 4) for v in np.percentile(x, p)] if len(x) else None


d = pd.read_parquet(src)
d = d[np.isfinite(d.betaN_e) & np.isfinite(d.gt_betaN) & (d.ip_A > 2e5) & np.isfinite(d.ts_rho_min)]
r = {"src": src}
for lab, m in (("all", d.ts_rho_min >= 0), ("core_covered_rho_min<0.15", d.ts_rho_min < 0.15),
               ("rho_min_0.15-0.35", (d.ts_rho_min >= 0.15) & (d.ts_rho_min < 0.35)), ("rho_min>=0.35", d.ts_rho_min >= 0.35)):
    x = d[m]
    if len(x) > 50:
        c = float((x.gt_betaN * x.betaN_e).sum() / (x.betaN_e ** 2).sum())
        p = c * x.betaN_e
        r[lab] = {"n": int(len(x)), "c_lsq": c, "c_med": float(np.median(x.gt_betaN / x.betaN_e)),
                  "ratio_p25_75": q(x.gt_betaN / x.betaN_e, (25, 75)),
                  "r2_betaN_with_c": float(1 - ((x.gt_betaN - p) ** 2).sum() / ((x.gt_betaN - x.gt_betaN.mean()) ** 2).sum())}
r["c_used"] = r["core_covered_rho_min<0.15" if use == "core" else "all"]["c_lsq"]
r["use"] = use
r["ts_rho_min_p"] = q(d.ts_rho_min)
json.dump(r, open(outp, "w"), indent=1)
print("SCALC " + json.dumps(r))
