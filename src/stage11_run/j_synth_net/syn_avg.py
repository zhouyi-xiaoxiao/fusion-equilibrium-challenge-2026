#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — output-level ensemble of several syn_predict runs on the SAME real set (G1-layout parts):
psi = mean of the maps (all runs share the exact vacuum flux + gauge, so this is the mean of the plasma parts), q95 = geometric
mean and betaN = arithmetic mean of the CALIBRATED scalars (each run x its own DIII-D train calibration), aux = mean.
The output tag carries calibrated scalars; its calibration file (C_q95 = C_betaN = 1) is written next to it.
Usage: syn_avg.py --runs RUN:TAG:calib.json [RUN:TAG:calib.json ...] --out DIR --tag NEWTAG --machine d3d|mast
Label-free on MAST (rule3_guard installed): reads only our own parts and DIII-D-derived calibration JSON.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ws = Path(os.environ["WS"])
sys.path.insert(0, str(ws / "stage11_run/ch2_geometry"))
import rule3_guard  # noqa: E402

rule3_guard.install()
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--machine", required=True, choices=["d3d", "mast"])
ap.add_argument("--weights", default=None, help="comma list of member weights (normalised; default equal)")
ap.add_argument("--skip-missing", action="store_true", help="skip shots for which a member part is missing (unfinished reference run)")
a = ap.parse_args()
M = a.machine
specs = []
for s in a.runs:
    run, tag, cal = s.split(":")
    specs.append((Path(run), tag, json.load(open(cal))))
lists = [[x for x in (r / f"files_{t}_{M}.txt").read_text().split("\n") if x] for r, t, _ in specs]
assert all(set(l) == set(lists[0]) and len(l) == len(lists[0]) for l in lists), "runs list different shots"
pos = [{n: j for j, n in enumerate(l)} for l in lists]                   # member index of each shot (output order = first run's)
out = Path(a.out); (out / "parts").mkdir(parents=True, exist_ok=True)
(out / f"files_{a.tag}_{M}.txt").write_text("\n".join(lists[0]) + "\n")
json.dump({"syn_avg": a.runs, "weights": a.weights}, open(out / f"cfg_{a.tag}_{M}.json", "w"), indent=1)
json.dump({"C_q95": 1.0, "C_betaN": 1.0, "source": "syn_avg: calibrated member scalars"}, open(out / f"calib_{a.tag}.json", "w"), indent=1)
n = len(specs)
W = np.array([float(x) for x in a.weights.split(",")], float) if a.weights else np.ones(n)
assert len(W) == n and (W >= 0).all() and W.sum() > 0
W = W / W.sum()
for i in range(len(lists[0])):
    name = lists[0][i]
    if a.skip_missing and not all((r / "parts" / f"{t}_{M}_{pos[m][name]:05d}.npz").exists() for m, (r, t, _) in enumerate(specs)):
        continue
    cs = [np.load(r / "parts" / f"{t}_{M}_{pos[m][name]:05d}.npz", allow_pickle=False) for m, (r, t, _) in enumerate(specs)]
    ks = cs[0]["ks"]
    for c in cs[1:]:
        assert np.array_equal(c["ks"], ks) and c["psi"].shape == cs[0]["psi"].shape
    psi = sum(w * c["psi"].astype(np.float64) for w, c in zip(W, cs)).astype(np.float32)
    lq = sum(w * np.log(np.clip(cal["C_q95"] * c["q95"].astype(float), 1e-3, None)) for w, c, (_, _, cal) in zip(W, cs, specs))
    bn = sum(w * cal["C_betaN"] * c["betaN"].astype(float) for w, c, (_, _, cal) in zip(W, cs, specs))
    ok = np.logical_and.reduce([c["ok"] for c in cs])
    aux = np.nanmean(np.stack([c["aux"].astype(float) for c in cs if "aux" in c]), 0).astype(np.float32) if any("aux" in c for c in cs) else None
    an = next((c["aux_names"] for c in cs if "aux_names" in c), None)
    extra = {"aux": aux, "aux_names": an} if (aux is not None and an is not None) else {}
    tmp = out / "parts" / f"{a.tag}_{M}_{i:05d}.tmp.npz"
    np.savez_compressed(tmp, psi=psi, q95=np.exp(lq), betaN=bn, ok=ok, t=cs[0]["t"], ip_A=cs[0]["ip_A"], ks=ks, info=cs[0]["info"], **extra)
    os.replace(tmp, out / "parts" / f"{a.tag}_{M}_{i:05d}.npz")
print(f"AVG {a.tag}: {n} runs, {len(lists[0])} shots -> {out}")
