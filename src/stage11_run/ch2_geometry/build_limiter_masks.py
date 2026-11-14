#!/usr/bin/env python3
"""
Stage 11 / Ch2: rule-3-compliant replacements for the scorer envelope inside our MODELS.

For each machine: the 65x65 EFIT grid (read from the INPUT columns efit_grid_R / efit_grid_Z of the
shipped parquet files, "kept on every split"), a PUBLIC limiter polygon, the grid mask of points
inside it, its 1-pixel ring, and a densely resampled polygon (1 cm) for exact limiter-flux sampling.

  MAST: FAIR-MAST level-2 wall/limiter_{r,z} (37 pts, EFIT machine description lim_mast.010_M4,
        shotRange 0..40000; fetched by fetch_geometry.py from non-demo, non-test shots).
  D3D : FreeGS 0.8.2 machine.DIIID() Wall ("boundary from g file and GA webpage") — public.
        (D3D could legally use d3d_envelope.npz, but the D3D side of a Ch2 physics pipeline must
        mimic the MAST conditions, so it uses a public wall too.)

NEVER opens fusion_scoring/masks/mast_envelope.npz (rule 3: scoring derivation only).
Usage: python build_limiter_masks.py --geom-dir <dir with mast_limiter.json> --out <dir> [--check-all]
(.venv_gs python: needs freegs for the D3D wall.)
"""
from __future__ import annotations
import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.path import Path as MPath
from scipy import ndimage

DATA = Path(os.environ["SCR"] + "/data_full")


def grid_of(files, check_all=False):
    ref = None
    n_diff = 0
    for i, f in enumerate(files):
        row = pd.read_parquet(f, columns=["efit_grid_R", "efit_grid_Z"]).iloc[0]
        R = np.asarray(row["efit_grid_R"], float); Z = np.asarray(row["efit_grid_Z"], float)
        if ref is None:
            ref = (R, Z)
        elif not (np.allclose(R, ref[0], atol=1e-9) and np.allclose(Z, ref[1], atol=1e-9)):
            n_diff += 1
        if not check_all and i >= 4:
            break
    return ref[0], ref[1], n_diff, (len(files) if check_all else min(len(files), 5))


def resample_polygon(R, Z, ds=0.01):
    P = np.column_stack([R, Z])
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])
    out = []
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.hypot(*(b - a)) / ds)))
        t = np.arange(n) / n
        out.append(a[None] + t[:, None] * (b - a)[None])
    return np.vstack(out)


def build(R, Z, gR, gZ):
    poly = MPath(np.column_stack([R, Z]))
    RR, ZZ = np.meshgrid(gR, gZ)                     # [iz, ir] like the scorer masks
    inside = poly.contains_points(np.column_stack([RR.ravel(), ZZ.ravel()])).reshape(RR.shape)
    ring = inside & ~ndimage.binary_erosion(inside)
    dense = resample_polygon(R, Z)
    return inside, ring, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check-all", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rep = {}
    # ---- MAST
    lim = json.load(open(Path(args.geom_dir) / "mast_limiter.json"))
    mfiles = sorted((DATA / "mast").glob("*.parquet"))
    assert not any(d in f.name for f in mfiles for d in ("28348", "28350", "28351"))
    gR, gZ, nd, nchk = grid_of(mfiles, args.check_all)
    inside, ring, dense = build(np.asarray(lim["R"]), np.asarray(lim["Z"]), gR, gZ)
    np.savez(out / "mast_limiter_mask.npz", grid_R=gR, grid_Z=gZ, mask=inside, ring=ring,
             limiter_R=np.asarray(lim["R"]), limiter_Z=np.asarray(lim["Z"]), limiter_dense=dense)
    rep["mast"] = {"grid_files_checked": nchk, "grid_files_differing": nd, "grid_R": [gR[0], gR[1] - gR[0], gR[-1]],
                   "grid_Z": [gZ[0], gZ[1] - gZ[0], gZ[-1]], "mask_pixels": int(inside.sum()), "ring_pixels": int(ring.sum()),
                   "limiter_source": lim["provenance"]["limiter_attrs"].get("source"),
                   "limiter_shots_read": lim["provenance"]["shots_read"],
                   "mask_R_range": [float(np.meshgrid(gR, gZ)[0][inside].min()), float(np.meshgrid(gR, gZ)[0][inside].max())],
                   "mask_Z_range": [float(np.meshgrid(gR, gZ)[1][inside].min()), float(np.meshgrid(gR, gZ)[1][inside].max())],
                   "mask_centroid_R": float(np.meshgrid(gR, gZ)[0][inside].mean())}
    # ---- D3D (public FreeGS wall)
    try:
        import freegs
        w = freegs.machine.DIIID().wall
        dR, dZ = np.asarray(w.R, float), np.asarray(w.Z, float)
        src = f"freegs {freegs.__version__} machine.DIIID().wall"
    except Exception as e:  # noqa: BLE001
        dR = dZ = None; src = f"unavailable: {e}"
    if dR is not None:
        dfiles = sorted((DATA / "diii_d_train").glob("*.parquet"))
        gR, gZ, nd, nchk = grid_of(dfiles[:200] if args.check_all else dfiles, args.check_all)
        inside, ring, dense = build(dR, dZ, gR, gZ)
        np.savez(out / "d3d_limiter_mask.npz", grid_R=gR, grid_Z=gZ, mask=inside, ring=ring,
                 limiter_R=dR, limiter_Z=dZ, limiter_dense=dense)
        rep["d3d"] = {"grid_files_checked": nchk, "grid_files_differing": nd, "mask_pixels": int(inside.sum()),
                      "ring_pixels": int(ring.sum()), "limiter_source": src, "n_wall_points": int(len(dR))}
    (out / "limiter_masks_report.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
