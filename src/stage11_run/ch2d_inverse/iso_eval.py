#!/usr/bin/env python3
"""
Label-free MAST validation of ANY run's flux maps (G1 or inverse parts) against the midplane Thomson data:
  * isotherm-pair mismatch per Te level: displacement [cm] of the inboard crossing along the chord needed to put it on
    the flux surface of the outboard crossing (signed and |.|), levels {0.20,0.30,0.45,0.60,0.75,0.85} (two unambiguous
    crossings only) - held-out levels of a run fitted on other levels are an out-of-sample check;
  * outer Thomson edge (uncensored) and inner edge (uncensored) distance to the LCFS along the chord [cm];
    censored inner edge: violation = coverage limit outside the plasma.
Reads only: MAST input columns (rule3_guard installed, read_mast_parquet allow-list), our own parts, the public wall.
Usage: python iso_eval.py --run DIR --tag T --data-dir <mast dir> --wall <mast_limiter_mask.npz> --cfg cfg.json [--stride N]
"""
import argparse, json, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from scipy.interpolate import RectBivariateSpline
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "ch2_G1")); sys.path.insert(0, str(HERE.parent / "ch2_geometry"))
import rule3_guard  # noqa: E402
rule3_guard.install()
import g1_core as C  # noqa: E402
import g1_data as D  # noqa: E402
import run_g1 as RG  # noqa: E402
import inv_core as K  # noqa: E402

LEV = [0.20, 0.35, 0.50, 0.65, 0.80]
A = None


def one(job):
    i, name = job
    try:
        row = D.load_row(Path(A.data_dir) / name, "mast")
        S = RG.get_solver("mast", row, A.cfgd, A.wall)
        cfg = S.cfg
        t = np.asarray(row["efit_times"], float)
        th = D.thomson_frames(row, "mast", t, cfg)
        c = np.load(Path(A.run) / "parts" / f"{A.tag}_mast_{i:05d}.npz", allow_pickle=False)
        psi, ok = c["psi"], c["ok"]
        zl = th["chord"][1]
        out = []
        ks = range(0, len(t), A.every)
        for k in ks:
            pk = th["prof"][k]
            if pk is None:
                continue
            P = psi[k].astype(np.float64).T                   # MAST: OUT_SIGN +1, no gauge
            spl = RectBivariateSpline(S.grid.gR, S.grid.gZ, P)
            iso = K.mast_iso_obs(pk, LEV, zl)
            rec = {"idx": i, "k": k, "ok": bool(ok[k]), "ip_kA": float(np.interp(t[k], t, np.abs(c["ip_A"])) / 1e3)}
            for f, v in K.iso_mismatch(spl, iso):
                rec[f"m{f:g}"] = 100 * v
            b = C.analyse(S.grid, S.wall, P, None, None)
            if b is not None:
                for nm, (R0, cens, e) in {"out": (th["edge"][k], th["cens"][k], 1.0), "in": (th["rin"][k], th["cin"][k], -1.0)}.items():
                    if np.isfinite(R0) and not cens:
                        pr = float(spl.ev(R0, zl, dx=1)); pz = float(spl.ev(R0, zl, dy=1))
                        sl = max(-(e * pr), 0.3 * np.hypot(pr, pz), 1e-12)
                        rec[f"e_{nm}"] = 100 * (float(spl.ev(R0, zl)) - b["psi_b"]) / sl
                if th["cin"][k] and np.isfinite(th["covlo"][k]):
                    R0 = th["covlo"][k]
                    pr = float(spl.ev(R0, zl, dx=1)); pz = float(spl.ev(R0, zl, dy=1))
                    sl = max(pr, 0.3 * np.hypot(pr, pz), 1e-12)
                    rec["e_inge"] = 100 * min(0.0, (float(spl.ev(R0, zl)) - b["psi_b"]) / sl)
            out.append(rec)
        return out
    except Exception as e:  # noqa: BLE001
        return [{"idx": i, "fatal": f"{type(e).__name__}: {e}"}]


def summarise(df):
    s = {"n_frames": int(len(df))}
    for f in LEV:
        c = f"m{f:g}"
        if c in df:
            v = df[c].dropna()
            s[c] = {"n": int(len(v)), "mae_cm": float(v.abs().mean()), "med_abs_cm": float(v.abs().median()),
                    "rms_cm": float(np.sqrt((v ** 2).mean())), "bias_cm": float(v.mean())}
    for c in ("e_out", "e_in", "e_inge"):
        if c in df:
            v = df[c].dropna()
            s[c] = {"n": int(len(v)), "mae_cm": float(v.abs().mean()), "bias_cm": float(v.mean()),
                    "frac_within_3cm": float((v.abs() < 3).mean())}
    return s


def main():
    global A
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--wall", required=True)
    ap.add_argument("--cfg", default=None); ap.add_argument("--stride", type=int, default=1); ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--procs", type=int, default=32); ap.add_argument("--out", default=None)
    A = ap.parse_args()
    A.cfgd = json.load(open(A.cfg)) if A.cfg else {}
    names = [x for x in (Path(A.run) / f"files_{A.tag}_mast.txt").read_text().split("\n") if x]
    jobs = [(i, n) for i, n in enumerate(names)][::A.stride]
    with get_context("fork").Pool(A.procs) as pool:
        res = pool.map(one, jobs, chunksize=2)
    import pandas as pd
    df = pd.DataFrame([r for rr in res for r in rr])
    nf = int(df["fatal"].notna().sum()) if "fatal" in df else 0
    if nf:
        df = df[df["fatal"].isna()]
    outp = Path(A.out or (Path(A.run) / f"isoeval_{A.tag}.json"))
    df.to_parquet(outp.with_suffix(".parquet"))
    rep = {"run": A.run, "tag": A.tag, "n_shots": len(jobs), "n_fatal": nf, "all": summarise(df), "converged_only": summarise(df[df["ok"]])}
    json.dump(rep, open(outp, "w"), indent=1)
    print("ISOEVAL " + A.tag + " " + json.dumps({k: (v.get("mae_cm") if isinstance(v, dict) else v) for k, v in rep["all"].items()}))


if __name__ == "__main__":
    main()
