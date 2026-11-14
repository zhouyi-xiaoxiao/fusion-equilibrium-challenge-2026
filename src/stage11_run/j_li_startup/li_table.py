#!/usr/bin/env python3
"""
Stage 11J / li_startup — li_table.py: per-frame feature table for the li-from-inputs model.

  --machine d3d   DIII-D TRAIN shots (labels allowed): features exactly as the driver computes them in the MAST-like
                  pseudo-horizontal harness (ramp history from the Ip record + real-Thomson core T_e; exponents from the
                  synthesised midplane laser, clean AND stressed) + the SCORER-derived li of the label map (official
                  fusion_scoring derivation with the DIII-D envelope) + the EFIT li label + phase information.
  --machine mast  label-free: rule3_guard installed, input columns only; the same features on every frame (the laser is real).

Usage: li_table.py --machine d3d --data-dir D --files LIST.txt --wall W.npz --out T.parquet [--frames-per-shot 16] [--procs N]
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WS = Path(os.environ["WS"])
G1DIR = HERE.parent / "ch2_G1"
for p in (HERE.parent / "e_mast_inv6", HERE.parent / "ch2d_inverse", G1DIR, HERE.parent / "g_restrict", HERE):
    sys.path.insert(0, str(p))
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402
import li_core as LI  # noqa: E402

STRESS = {"noise": 0.1, "branch": 0.05, "timing": 10.0, "sep": 0.03, "edge_noise": 0.01}
A = None
_OFF = {}


def official():
    if not _OFF:
        sys.path.insert(0, str(WS / "starter/fusion_scoring"))
        import derive
        import lcfs as LC
        m = np.load(WS / "starter/fusion_scoring/masks/d3d_envelope.npz")
        _OFF.update({"derive": derive, "LC": LC, "R": m["grid_R"].astype(float), "Z": m["grid_Z"].astype(float),
                     "MC": m["mask_coarse"].astype(bool)})
        _OFF["MF"] = _OFF["MC"].astype(np.float64)
    return _OFF


def one(job):
    idx, name = job
    try:
        return _one(idx, name)
    except Exception as e:  # noqa: BLE001
        import traceback
        return [{"file": name, "idx": idx, "fatal": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-800:]}]


def _one(idx, name):
    machine = A.machine
    cfg = {**C.DEFAULT_CFG, **json.load(open(G1DIR / "configs/cfg_soft_k50.json"))}
    cfg["li_tau_scale"] = A.tau_scale
    row = D.load_row(str(Path(A.data_dir) / name), machine)
    t = np.asarray(row["efit_times"], float)
    T = len(t)
    th = D.thomson_frames(row, machine, t, cfg)
    inp = D.frame_inputs(row, machine, t)
    te = np.array([LI.te_top(p) for p in th["prof"]], float)
    tip, ip = LI.ip_record(row, machine)
    ramp = LI.ramp_features(tip, ip, t, te, machine, tau_scale=A.tau_scale, fvac=inp["fvac"])
    ipf = np.abs(inp["ip_A"])
    ks = list(range(T))
    if A.frames_per_shot and machine == "d3d":
        ks = sorted(set(np.linspace(0, T - 1, min(A.frames_per_shot, T)).round().astype(int).tolist()))
    rows = []
    base = {"file": name, "idx": idx}
    if machine == "mast":
        for k in ks:
            if ipf[k] < cfg["ip_min"]:
                continue
            ex = None
            try:
                ex = LI.exponents(th["prof"][k], float(cfg["f_edge"])) if th["prof"][k] is not None else None
            except Exception:  # noqa: BLE001
                ex = None
            r = {**base, "k": int(k), "t": float(t[k]), "ip_kA": float(ipf[k] / 1e3), "cond": "real",
                 "ts_has": bool(th["has"][k]), "ts_cens": bool(th["cens"][k]), "ts_cin": bool(th["cin"][k]),
                 "frac_t": float(k / max(T - 1, 1)), "tau_ms": float(ramp["tau_ms"][k]), "theta": float(ramp["theta"][k]),
                 "te_keV": float(ramp["te_keV"][k]), **{n: float(ramp[n][k]) for n in LI.FEATS_RAMP},
                 "aT": ex[0] if ex else np.nan, "an": ex[1] if ex else np.nan}
            rows.append(r)
        return rows
    # ---- DIII-D (labels)
    import run_inv7l as RL
    lab = D.labels_of(row)
    if "psi" not in lab:
        return rows
    off = official()
    vert, tang = RL.d3d_chords(row, t, cfg["th_gap_ms"])
    w = np.load(A.wall)
    cov_lo = float(np.asarray(w["limiter_R"], float).min()) + 0.04
    rng = np.random.default_rng(104729 * idx + 31)
    for k in ks:
        if ipf[k] < cfg["ip_min"] or k >= len(lab["psi"]):
            continue
        g = lab["psi"][k].astype(np.float64)
        if not np.isfinite(g).all():
            continue
        try:
            Cg = off["LC"].extract_lcfs(g, off["R"], off["Z"], "DIII-D", off["MC"], off["MF"], n_points=512)
            sc = off["derive"].derive_frame(g, off["R"], off["Z"], "DIII-D", off["MC"], off["MF"], contour=Cg)
        except Exception:  # noqa: BLE001
            continue
        lic = np.nan
        try:                                             # cylinder-equivalent li of the label map (same functional as in the solver)
            ef = RL.efit_frame(row, lab, k)
            if ef is not None:
                spl_, pa_, pb_, ra_, za_ = ef
                gR_ = np.asarray(row["efit_grid_R"], float); gZ_ = np.asarray(row["efit_grid_Z"], float)
                fR_ = np.linspace(gR_[0], gR_[-1], (len(gR_) - 1) * 2 + 1); fZ_ = np.linspace(gZ_[0], gZ_[-1], (len(gZ_) - 1) * 2 + 1)
                lic = LI.li_cyl(fR_, fZ_, spl_(fR_, fZ_), spl_, pa_, pb_, (ra_, za_))
        except Exception:  # noqa: BLE001
            lic = np.nan
        common = {**base, "gt_lic": lic, "k": int(k), "t": float(t[k]), "ip_kA": float(ipf[k] / 1e3), "frac_t": float(k / max(T - 1, 1)),
                  "tau_ms": float(ramp["tau_ms"][k]), "theta": float(ramp["theta"][k]), "te_keV": float(ramp["te_keV"][k]),
                  **{n: float(ramp[n][k]) for n in LI.FEATS_RAMP}, **{f"gt_{n}": float(v) for n, v in sc.items()},
                  "efit_li": float(lab["efit_li"][k]) if "efit_li" in lab and k < len(lab["efit_li"]) else np.nan,
                  "efit_q95": float(lab["efit_q95"][k]) if "efit_q95" in lab and k < len(lab["efit_q95"]) else np.nan}
        for cond, st in (("clean", {}), ("stress", STRESS)):
            ex, has = None, False
            try:
                pf = RL.pseudo_frame(row, lab, k, t, vert, tang, st, idx, cov_lo, rng)
                if pf is not None:
                    has = True
                    ex = LI.exponents(pf[0], float(cfg["f_edge"]))
            except Exception:  # noqa: BLE001
                ex = None
            rows.append({**common, "cond": cond, "ts_has": has, "aT": ex[0] if ex else np.nan, "an": ex[1] if ex else np.nan})
    return rows


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["d3d", "mast"], required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--files", default=None, help="text file of shot file names (default: every parquet of the directory)")
    ap.add_argument("--wall", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-per-shot", type=int, default=16)
    ap.add_argument("--tau-scale", type=float, default=1.0)
    ap.add_argument("--procs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    A = ap.parse_args()
    if A.machine == "mast":
        import rule3_guard
        rule3_guard.install()
    names = [x for x in Path(A.files).read_text().split("\n") if x] if A.files else sorted(p.name for p in Path(A.data_dir).glob("*.parquet"))
    t0 = time.time()
    rows = []
    with get_context("fork").Pool(A.procs) as pool:
        for n, r in enumerate(pool.imap_unordered(one, list(enumerate(names)), chunksize=2)):
            rows.extend(r)
            if (n + 1) % max(1, len(names) // 10) == 0:
                print(f"  {n + 1}/{len(names)} shots, {len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(A.out)
    nf = int(df["fatal"].notna().sum()) if "fatal" in df else 0
    print("LITABLE " + json.dumps({"machine": A.machine, "shots": len(names), "rows": int(len(df)), "fatal": nf,
                                   "example": (str(df.loc[df['fatal'].notna(), 'tb'].iloc[0])[-500:] if nf else ""),
                                   "wall_s": time.time() - t0}))


if __name__ == "__main__":
    main()
