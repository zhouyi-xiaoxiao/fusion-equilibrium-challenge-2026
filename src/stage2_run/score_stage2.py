#!/usr/bin/env python3
"""
Stage-2 scorer — real metric (v3.2.0), generalized to both machines.
All scoring functionals imported unmodified from starter/local_score.py +
vendored fusion_scoring/ (same approach as scaleup_run/score_scaleup.py).

Usage:
  score_stage2.py --machine d3d  --gt-dir DIR --skip 1000 --n 150 --pred preds_mlp.npz
  score_stage2.py --machine mast --gt-list mast_demo_files.txt --pred preds_mast_agn.npz
  score_stage2.py --machine d3d --gt-dir DIR --skip 1000 --n 150 --mode perfect|zeros
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STARTER = HERE.parent / "starter"
sys.path.insert(0, str(STARTER))

import matplotlib
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

import local_score as LS  # noqa: E402
from common import N_CONS, N_SCALARS, PSI_SIGNS, SCALARS, SCORING_VERSION  # noqa: E402
from experiments import _as_psirz_stack  # noqa: E402
from metrics import Accum, finalize_machine  # noqa: E402

MASKS = {"d3d": "d3d_envelope.npz", "mast": "mast_envelope.npz"}


def load_gt(files):
    out = []
    for f in files:
        row = pd.read_parquet(f).iloc[0]
        out.append({
            "psi": _as_psirz_stack(row["efit_psirz"]),
            "q95": np.asarray(row["efit_q95"], dtype=np.float64).ravel(),
            "betaN": np.asarray(row["efit_beta_n"], dtype=np.float64).ravel(),
        })
    print(f"  loaded {len(out)} GT shots", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["d3d", "mast"], default="d3d")
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--gt-list", default=None, help="file of parquet paths, one per line")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--n", type=int, default=0, help="0 = all after skip")
    ap.add_argument("--pred", default=None)
    ap.add_argument("--mode", choices=["pred", "perfect", "zeros"], default="pred")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    if args.gt_list:
        files = [Path(l) for l in Path(args.gt_list).read_text().split() if l.strip()]
    else:
        files = sorted(Path(args.gt_dir).glob("*.parquet"))[args.skip:]
        if args.n:
            files = files[:args.n]

    # CRITICAL (stage-4 find): local_score's module-global MACHINE defaults to
    # "DIII-D"; without this override every MAST fold ran LCFS extraction and
    # derivations with AXIS_SIGN["DIII-D"] (wrong orientation) -> boundary and
    # Consistency were structurally zeroed on MAST (stage-2/3 "LCFS fails
    # 100%" was largely THIS, not missing closed surfaces).
    LS.MACHINE = "MAST" if args.machine == "mast" else "DIII-D"

    mask = np.load(STARTER / "fusion_scoring" / "masks" / MASKS[args.machine])
    R, Z = mask["grid_R"], mask["grid_Z"]
    mask_coarse = mask["mask_coarse"].astype(bool)
    mask_f = mask_coarse.astype(np.float64)

    tag = args.tag or (args.mode if args.mode != "pred" else Path(args.pred).stem)
    print(f"Stage-2 scorer (metric v{SCORING_VERSION}, {args.machine.upper()}) — "
          f"{len(files)} GT shots, tag={tag}", flush=True)
    shots = load_gt(files)

    if args.mode == "perfect":
        preds = [{"psirz": s["psi"].astype(np.float64), "q95": s["q95"],
                  "betaN": s["betaN"]} for s in shots]
    elif args.mode == "zeros":
        preds = [{"psirz": np.zeros_like(s["psi"], dtype=np.float64),
                  "q95": np.zeros_like(s["q95"]),
                  "betaN": np.zeros_like(s["betaN"])} for s in shots]
    else:
        z = np.load(args.pred, allow_pickle=False)
        preds = [{"psirz": z[f"shot_{i:04d}_psirz"].astype(np.float64),
                  "q95": z[f"shot_{i:04d}_q95"].astype(np.float64),
                  "betaN": z[f"shot_{i:04d}_betaN"].astype(np.float64)}
                 for i in range(len(shots))]

    print("Building reference targets from ground truth...", flush=True)
    refs = []
    psi_sum = psi_sumsq = psi_n = 0.0
    scal_sum, scal_sumsq, scal_n = np.zeros(N_SCALARS), np.zeros(N_SCALARS), np.zeros(N_SCALARS)
    cons_sum, cons_sumsq, cons_n = np.zeros(N_CONS), np.zeros(N_CONS), np.zeros(N_CONS)
    for si, s in enumerate(shots):
        ref = LS.build_reference(s["psi"], R, Z, mask_coarse, mask_f)
        refs.append(ref)
        g = s["psi"].astype(np.float64)
        fin = np.isfinite(g)
        psi_sum += float(g[fin].sum()); psi_sumsq += float((g[fin] ** 2).sum()); psi_n += int(fin.sum())
        for j, name in enumerate(SCALARS):
            v = np.asarray(s[name], dtype=np.float64)
            v = v[np.isfinite(v)]
            scal_sum[j] += v.sum(); scal_sumsq[j] += (v ** 2).sum(); scal_n[j] += v.size
        cons_gt, cmask = ref[1], ref[2]
        for j in range(N_CONS):
            v = cons_gt[cmask[:, j], j]
            cons_sum[j] += v.sum(); cons_sumsq[j] += (v ** 2).sum(); cons_n[j] += v.size
        if (si + 1) % 25 == 0:
            print(f"  reference {si + 1}/{len(shots)}", flush=True)

    ref_stats = {"psi_sum": psi_sum, "psi_sumsq": psi_sumsq, "psi_n": psi_n,
                 "scal_sum": scal_sum, "scal_sumsq": scal_sumsq, "scal_n": scal_n,
                 "cons_sum": cons_sum, "cons_sumsq": cons_sumsq, "cons_n": cons_n}
    mean_psi = psi_sum / psi_n if psi_n else 0.0
    means = (mean_psi,
             np.where(scal_n > 0, scal_sum / np.where(scal_n > 0, scal_n, 1), 0.0),
             np.where(cons_n > 0, cons_sum / np.where(cons_n > 0, cons_n, 1), 0.0))

    totals = {sg: 0.0 for sg in PSI_SIGNS}
    for s_, p in zip(shots, preds):
        rr, _, _ = LS.psi_residuals(s_["psi"], p["psirz"], mean_psi)
        for sg in PSI_SIGNS:
            totals[sg] += rr[sg]
    psi_sign = min(PSI_SIGNS, key=lambda sg: totals[sg])

    print("Scoring...", flush=True)
    acc = Accum()
    acc.psi_sign = psi_sign
    for si, (s_, ref, p) in enumerate(zip(shots, refs, preds)):
        acc.add(LS.score_shot(s_, ref, p, R, Z, mask_coarse, mask_f, psi_sign, means))
        if (si + 1) % 25 == 0:
            print(f"  scored {si + 1}/{len(shots)}", flush=True)

    res = finalize_machine(acc, ref_stats)

    print("\n" + "=" * 62)
    print(f"  [{tag}] {args.machine.upper()} COMPOSITE S = {res['S']:.4f}   "
          f"({len(shots)} shots)")
    print("=" * 62)
    for label, key, w in [("R2_psi", "r2_psi", 0.55), ("R2_{q95,betaN}", "r2_qb", 0.15),
                          ("1 - D_LCFS", "dlcfs", 0.10), ("Consistency", "consistency", 0.20)]:
        v = (1.0 - min(1.0, res["dlcfs"])) if key == "dlcfs" else res[key]
        print(f"  {label:>16s}  {v:8.4f}   x {w:.2f}  =  {w * max(0.0, v):.4f}")
    print("\n  per-derived-scalar R2 (the Consistency term):")
    for k, v in res["r2_cons_each"].items():
        print(f"      {k:>8s}  {'  n/a' if v is None else f'{v:7.4f}'}")
    print(f"\n  LCFS extraction failed on {res['lcfs_fail_frac']:.1%} of frames; "
          f"derivations on {res['cons_fail_frac']:.1%}")
    print(f"  psi_sign = {res['psi_sign']:+d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
