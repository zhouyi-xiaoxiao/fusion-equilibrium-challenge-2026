#!/usr/bin/env python3
"""Stage 11E / fold2_offset — parity of every feature_v21 column under the current-MIRROR map
(ECOILA, F1A..F9B, plasma_current -> negated; bcoil, Thomson, times unchanged).  If the GT psi is sign-normalised
(it is: psi_axis < psi_edge for every shot of both Ip signs), a reversed-Ip shot is the mirror of a normal one and
canonical inputs x_c = sign(Ip) * x_odd make the two populations one.  Writes $OUT/parity.npz (class per column:
1 = odd, 0 = even, 2 = neither) after testing N shots of both Ip signs and both DAQ eras."""
import os, sys, json
from pathlib import Path
import multiprocessing as mp
import numpy as np

WS = Path(os.environ["WS"])
FE = WS / "stage11_run/ch1_features"
for p in (WS / "starter", WS / "starter/fusion_scoring", WS / "stage2_run", FE):
    sys.path.insert(0, str(p))
SCR = Path(os.environ["SCR"])
OUT = Path(os.environ.get("OUT", str(SCR / "stage11_ch1/e_fold2_offset")))
MIRROR = ["magnetics_ECOILA"] + [f"magnetics_F{i}{s}" for i in range(1, 10) for s in "AB"] + ["magnetics_plasma_current"]


def mirror(row, bcoil=False):
    r = row.copy()
    for c in MIRROR + (["magnetics_bcoil"] if bcoil else []):
        r[c] = -np.asarray(row[c], dtype=np.float64)
    return r


def work(path):
    os.environ["OMP_NUM_THREADS"] = "1"
    from feat_v21 import features_v21, read_inputs
    row = read_inputs(path)
    X = features_v21(row).astype(np.float64)
    out = []
    for bc in (False, True):
        Y = features_v21(mirror(row, bc)).astype(np.float64)
        sc = np.nan_to_num(np.nanmax(np.abs(X), 0) if np.isfinite(X).any() else np.zeros(X.shape[1])) + 1e-9
        dodd = np.nan_to_num(np.nanmax(np.abs(Y + X), 0)) / sc
        devn = np.nan_to_num(np.nanmax(np.abs(Y - X), 0)) / sc
        nanm = (np.isnan(X) != np.isnan(Y)).any(0)
        out.append((dodd, devn, nanm))
    return out


def main():
    E = np.load(OUT / "emeta_train.npz")
    rng = np.random.default_rng(1)
    s = np.sign(E["ip_max"]); era = np.round(E["mag_dt"], 2)
    pick = []
    for e in (0.05, 0.5):
        for a in (1, -1):
            q = np.flatnonzero((s == a) & (era == e))
            pick += list(rng.choice(q, 20, replace=False))
    files = [str(SCR / "data_full/diii_d_train" / E["fname"][i]) for i in pick]
    import warnings; warnings.simplefilter("ignore")
    with mp.get_context("fork").Pool(70) as pool:
        res = pool.map(work, files)
    names = [l for l in (SCR / "stage11_ch1/feat_v21/names.txt").read_text().split("\n") if l]
    tol = 1e-5
    save = {"names": np.array(names)}
    for k, lab in ((0, "cur"), (1, "full")):
        dodd = np.max([r[k][0] for r in res], 0); devn = np.max([r[k][1] for r in res], 0); nanm = np.any([r[k][2] for r in res], 0)
        cls = np.where(devn <= tol, 0, np.where(dodd <= tol, 1, 2))
        cls[nanm] = 2
        save[f"cls_{lab}"] = cls; save[f"dodd_{lab}"] = dodd; save[f"devn_{lab}"] = devn
        print(f"[{lab} mirror] parity over {len(files)} shots: odd {int((cls==1).sum())}, even {int((cls==0).sum())}, neither {int((cls==2).sum())}")
        for j in np.flatnonzero(cls == 2):
            print(f"  neither: {j} {names[j]} dodd {dodd[j]:.3g} devn {devn[j]:.3g} nanmismatch {bool(nanm[j])}")
        print("  odd (excluding mag_/v1_F/drv_ECOILA/drv_plasma/drv_bcoil/mag_bcoil):", ", ".join(names[j] for j in np.flatnonzero(cls == 1) if not names[j].startswith(("mag_", "v1_F", "drv_ECOILA", "drv_plasma", "drv_bcoil"))))
        print("  odd count by prefix:", {p: int(sum(1 for j in np.flatnonzero(cls == 1) if names[j].startswith(p))) for p in ("v1_", "mag_", "ipn_", "time_", "drv_", "phys_", "tsc", "tse", "th1", "tsv", "rec")})
    np.savez(OUT / "parity.npz", **save)


if __name__ == "__main__":
    main()
