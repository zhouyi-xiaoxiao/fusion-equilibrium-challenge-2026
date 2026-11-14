#!/usr/bin/env python3
"""
Exported-map current check (external review, follow-up 1): toroidal current carried by a SUBMITTED MAST flux map,
    I_map = -(1/mu0) * int_{inside the public wall} Delta* psi / R dR dZ,   Delta* = R d/dR (1/R d/dR) + d2/dZ2,
compared with the measured Ip of the frame (catches 2pi / kA-A / sign / mask / serialisation errors). Label-free:
reads the submission npz (our own output), MAST input columns through rule3_guard (efit_times, Ip) and the public wall.
Usage: python map_current_check.py <mast_public_test.npz> --data-dir <mast dir> --wall <mast_limiter_mask.npz> [--every 30]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ch2_G1")); sys.path.insert(0, str(HERE.parent / "ch2_geometry"))
import rule3_guard  # noqa: E402
rule3_guard.install()
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("npz"); ap.add_argument("--data-dir", required=True); ap.add_argument("--wall", required=True)
ap.add_argument("--every", type=int, default=30); ap.add_argument("--order-file", default=None)
a = ap.parse_args()
z = np.load(a.npz)
names = sorted(p.name for p in Path(a.data_dir).glob("*.parquet"))
w = np.load(a.wall)
ratios, sgn_ok = [], []
for i in range(0, len(names), a.every):
    row = D.load_row(Path(a.data_dir) / names[i], "mast")
    gR = np.asarray(row["efit_grid_R"], float); gZ = np.asarray(row["efit_grid_Z"], float)
    R, Z = np.meshgrid(gR, gZ, indexing="ij")
    inside = C.points_in_poly(R, Z, w["limiter_R"], w["limiter_Z"])
    t = np.asarray(row["efit_times"], float)
    ip = np.abs(D.frame_inputs(row, "mast", t)["ip_A"])
    psi = z[f"shot_{i:04d}_psirz"].astype(np.float64)              # (T, Z, R) EFIT layout
    dR, dZ = gR[1] - gR[0], gZ[1] - gZ[0]
    for k in range(0, len(t), max(1, len(t) // 6)):
        P = psi[k].T                                                # (R, Z)
        dPdR = np.gradient(P, dR, axis=0)
        ds = R * np.gradient(dPdR / R, dR, axis=0) + np.gradient(np.gradient(P, dZ, axis=1), dZ, axis=1)
        J = -ds / (C.MU0 * R)
        m = inside.copy(); m[:2, :] = m[-2:, :] = False; m[:, :2] = m[:, -2:] = False
        I = float(J[m].sum() * dR * dZ)
        if ip[k] > 1e5:
            ratios.append(I / ip[k])
rat = np.array(ratios)
rep = {"file": a.npz, "n_frames": int(len(rat)), "I_map_over_Ip_p5_50_95": [float(x) for x in np.percentile(np.abs(rat), [5, 50, 95])],
       "sign_positive_frac": float(np.mean(rat > 0))}
print("MAPCURRENT " + json.dumps(rep))
