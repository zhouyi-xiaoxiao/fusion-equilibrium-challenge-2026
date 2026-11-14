#!/usr/bin/env python3
"""Stage 11e / ip_axis — validation of the v2.2 Ip-axis estimator (ip_axis.py) on every DIII-D train + public-test shot.

Per shot (inputs only unless stated):
  branch / shift_ms / mag_t0 / t_bd on the v2.1 and v2.2 axes;
  |Ip| / peak at the first and second EFIT frame on both axes (EFIT has a plasma there by construction);
  TRAIN ONLY, label-based reference (never a model input): Ip_psi(t) = -(1/mu0) * sum(Delta* psi / R) dA over the EFIT grid
  interior; lag scan d in [-400, 400] ms (2 ms) of |Ip|(t - d) on the V2.2 axis against k * |Ip_psi|(t) (k fitted per d);
  reports the best lag relative to the v2.2 axis (delta22) and the normalised misfit at d = 0 on both axes
  (res21 = misfit of the v2.1 axis, res22 = misfit of the v2.2 axis, resbest).
Output $OUT/ipx_diag_{train,test}.npz (+ summary printed).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "starter"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ip_axis  # noqa: E402

SCR = Path(os.environ["SCR"])
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/ip_axis/diag")))
MU0 = 4e-7 * np.pi
LAGS = np.arange(-400.0, 401.0, 2.0)


def _col(tbl, c):
    return tbl.column(c).combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float64)


def _ipfun(axis, ip):
    n = min(len(ip), len(axis))
    a, t = np.abs(ip[:n]), axis[:n]
    ok = np.isfinite(a) & np.isfinite(t)
    a, t = a[ok], t[ok]
    o = np.argsort(t, kind="stable")
    a, t = a[o], t[o]
    g = np.arange(t[0], t[-1], 1.0)
    ig = np.interp(g, t, a)
    ig = np.convolve(np.pad(ig, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid")
    return lambda tq: np.interp(tq, g, ig, left=0.0, right=0.0)


def one(args):
    path, labels = args
    cols = ["efit_times", "magnetics_time", "magnetics_plasma_current", "magnetics_plasma_current_times"]
    if labels:
        cols += ["efit_psirz", "efit_grid_R", "efit_grid_Z"]
    tbl = pq.read_table(path, columns=cols)
    t = _col(tbl, "efit_times")
    row = {"magnetics_plasma_current_times": _col(tbl, "magnetics_plasma_current_times"),
           "magnetics_plasma_current": _col(tbl, "magnetics_plasma_current"), "magnetics_time": _col(tbl, "magnetics_time")}
    info = ip_axis.ip_axis_info(row)
    a21 = np.asarray(ip_axis.original()(row), np.float64)
    a22 = info["axis"]
    ip = row["magnetics_plasma_current"]
    br = {"reorigined": 0, "kept_unchanged": 1, "old_era_8192": 2, "fallback_tbd": 3}[info["branch"]]
    r = {"branch": float(br), "shift_ms": info["shift_ms"], "mag_t0": info["mag_t0"], "t_bd21": info["t_bd_v21"],
         "t_bd22": info["t_bd_v22"], "T": float(len(t)), "efit_first": float(t[0]) if len(t) else np.nan}
    f21, f22 = _ipfun(a21, ip), _ipfun(a22, ip)
    pk = float(np.nanpercentile(np.abs(ip[np.isfinite(ip)]), 99.5))
    for k in (0, 1):
        if len(t) > k:
            r[f"ipfrac21_f{k}"] = float(f21(t[k:k + 1])[0] / pk)
            r[f"ipfrac22_f{k}"] = float(f22(t[k:k + 1])[0] / pk)
    if labels and len(t) > 3:
        R = _col(tbl, "efit_grid_R"); Z = _col(tbl, "efit_grid_Z")
        a = tbl.column("efit_psirz").combine_chunks()
        psi = a.flatten().flatten().flatten().to_numpy(zero_copy_only=False).astype(np.float64).reshape(len(t), len(Z), len(R))
        dR = np.gradient(psi, R, axis=2)
        ds = np.gradient(dR, R, axis=2) - dR / R[None, None, :] + np.gradient(np.gradient(psi, Z, axis=1), Z, axis=1)
        w = (ds / R[None, None, :])[:, 2:-2, 2:-2]
        ipp = np.abs(-w.sum((1, 2)) * abs((R[1] - R[0]) * (Z[1] - Z[0])) / MU0) / 1e3

        def mis(f, d):
            y = f(t - d)
            kk = float(ipp @ y / max(ipp @ ipp, 1e-9))
            e = y - kk * ipp
            return float(e @ e / max(y @ y, 1e-9))
        res = np.array([mis(f22, d) for d in LAGS])
        j = int(np.argmin(res))
        r.update(delta22=float(LAGS[j]), res22=float(res[LAGS == 0][0]), resbest=float(res[j]), res21=mis(f21, 0.0))
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [("train", SCR / "data_full/diii_d_train", True), ("test", SCR / "data_full/diii_d_public_test", False)]
    nproc = int(os.environ.get("NPROC", "70"))
    for name, d, lab in jobs:
        files = sorted(p.name for p in d.glob("*.parquet"))
        if os.environ.get("LIMIT"):
            files = files[: int(os.environ["LIMIT"])]
        with Pool(nproc) as pool:
            rows = pool.map(one, [(str(d / f), lab) for f in files], chunksize=4)
        keys = sorted({k for r in rows for k in r})
        out = {k: np.array([r.get(k, np.nan) for r in rows], np.float64) for k in keys}
        out["files"] = np.array(files)
        np.savez(OUT / f"ipx_diag_{name}.npz", **out)
        b = out["branch"]
        print(f"IPX {name}: {len(files)} shots; branches reorigined {int((b == 0).sum())} kept_unchanged {int((b == 1).sum())} "
              f"old_era_8192 {int((b == 2).sum())} fallback_tbd {int((b == 3).sum())}", flush=True)
        q = lambda x: np.round(np.nanpercentile(x, [0.5, 5, 25, 50, 75, 95, 99.5]), 1)  # noqa: E731
        for lab_, m in (("reorigined", b == 0), ("old-era", b >= 2)):
            if m.any():
                print(f"  {lab_}: t_bd v21 {q(out['t_bd21'][m])}  v22 {q(out['t_bd22'][m])}", flush=True)
                print(f"  {lab_}: shift {q(out['shift_ms'][m])}", flush=True)
                for k in (0, 1):
                    a21, a22 = out[f"ipfrac21_f{k}"][m], out[f"ipfrac22_f{k}"][m]
                    print(f"  {lab_}: frame{k} |Ip|/peak < 5 %: v21 {int((a21 < 0.05).sum())}  v22 {int((a22 < 0.05).sum())} "
                          f"(median v21 {np.nanmedian(a21):.3f} v22 {np.nanmedian(a22):.3f})", flush=True)
                if lab:
                    dd = out["delta22"][m]
                    mad = np.nanmedian(np.abs(dd - np.nanmedian(dd)))
                    print(f"  {lab_}: label ref best lag vs v2.2 axis {q(dd)} MAD {mad:.1f}; misfit v21 {np.nanmedian(out['res21'][m]):.5f} "
                          f"v22 {np.nanmedian(out['res22'][m]):.5f} best {np.nanmedian(out['resbest'][m]):.5f}", flush=True)


if __name__ == "__main__":
    main()
