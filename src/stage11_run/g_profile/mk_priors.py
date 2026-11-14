#!/usr/bin/env python3
"""Stage 11G profile track — DIII-D-fitted priors of a profile family from its label fit (run_p7 --mode labelfit, TRAIN
shots, holdout and pilot shots excluded): prior means = medians over the converged frames (the INV_P5 convention), written
as config layers for the pilots.  sigma: review round 2 base values (cub / lao: sig_b 0.5, sig_e 0.15) with half / double
variants; fe keeps the INV sigma (1.5, 1.0).  DIII-D labels only.
Usage: mk_priors.py <dev dir> <lf tag> <family name: fe|cub|lao> <out cfg dir>"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

dev, tag, fam, outd = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
outd.mkdir(parents=True, exist_ok=True)
d = pd.read_parquet(dev / f"frames_{tag}_d3d.parquet")
d = d[d.ok.astype(bool) & (d.x_src == "fit")]
med = lambda c: float(np.nanmedian(d[c].astype(float)))
iqs = lambda c: float(np.subtract(*np.nanpercentile(d[c].astype(float), [75, 25])) / 1.35)
st = {"lf_tag": tag, "n_frames": int(len(d)), "n_shots": int(d.file.nunique()),
      **{c: {"median": med(c), "iqr_sd": iqs(c), "p5_25_75_95": [float(x) for x in np.nanpercentile(d[c].astype(float), [5, 25, 75, 95])]}
         for c in ("cp", "b1", "b2", "be", "pe_s", "dr", "gev", "f_press", "a0", "ffp_edge", "f2min") if c in d and d[c].notna().any()}}
src = f"D3D label fit {tag} ({st['n_shots']} TRAIN shots x 8 frames, medians over {st['n_frames']} converged frames)"
lcp0 = float(np.log(st["cp"]["median"]))
if fam == "fe":
    base = {"lcp0": lcp0, "b0": [st["b1"]["median"], st["b2"]["median"]], "_prior_source": src}
    json.dump({**base, "_what": "fe family, priors re-fitted with this source quadrature (sigma unchanged from INV)"}, open(outd / f"fam_{tag}.json", "w"), indent=1)
else:
    full = fam == "cub"
    for suf, f in (("", 1.0), ("H", 0.5), ("D", 2.0)):
        c = {"family": "cub", "params": ["dr", "lcp", "b1"] + (["b2", "be"] if full else []), "lcp0": lcp0,
             "b0": [st["b1"]["median"], st["b2"]["median"] if full else 0.0], "sig_b": [0.5 * f, 0.5 * f],
             "be0": st["be"]["median"] if full else 0.0, "sig_be": 0.15 * f, "b_lo": -3.0, "b_hi": 3.0, "be_lo": -0.5, "be_hi": 1.5,
             "_prior_source": src, "_what": f"{fam} family, DIII-D-fitted prior means, sigma = {f} x review base (sig_b 0.5, sig_e 0.15)"}
        json.dump(c, open(outd / f"fam_{tag}{suf}.json", "w"), indent=1)
    if "pe_s" in st:
        c = json.load(open(outd / f"fam_{tag}.json"))
        c.update({"pe_s_fix": st["pe_s"]["median"], "_what": c["_what"] + "; p' shape s frozen at the DIII-D median of the per-frame Thomson fits"})
        json.dump(c, open(outd / f"fam_{tag}S.json", "w"), indent=1)
json.dump(st, open(outd / f"prior_stats_{tag}.json", "w"), indent=1)
print("PRIORS " + json.dumps(st))
