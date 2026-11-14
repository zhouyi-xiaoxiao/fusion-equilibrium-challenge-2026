#!/usr/bin/env python3
"""Stage 11E / CX (combined) — copy of e_fold2_offset/e_canon.py, unchanged arithmetic; only the parity file is env-selectable
(PARITY=<npz>, default the v2.1 parity file).  Used to build mirror-canonical caches on FEATURE-V2.2 (corrected Ip axis).

Original docstring: MIRROR-CANONICAL feature cache (inputs only, exact, usable on unseen shots).

Finding (an internal development note): the GT psi is sign-normalised (psi_axis < psi_edge for every shot) and a
reversed-Ip shot is an exact mirror of a normal one (all PF/E-coil/Ip currents negated, same psi), but in the raw
feature space the 8.6 % reversed shots sit in a disjoint region, so the nets treat them as a rare regime (20x psi
error rate on both folds).  Every feature_v21 column is exactly odd or even under the mirror map (e_parity.py,
80 shots, no exception), so the canonical features are X_c = X_raw * s for the odd columns, s = time_sign_ip_shot
(+-1, an input), i.e. bit-for-bit the features of the mirrored raw signals when s = -1.

  SRC=<feat dir with train/holdout/test> DST=<new feat dir> VARIANT=cfl|cfp|ccl python e_canon.py
VARIANT  cfl = full mirror (currents + bcoil) and the raw v1_sign_ip kept as a flag column
         cfp = full mirror, pure (every odd column flipped; sign columns become +1)
         ccl = current-only mirror (bcoil untouched) + raw v1_sign_ip flag
Scaler refit on DST/train (f2_build.fit_scaler semantics); X.npy written; other arrays symlinked.
"""
import json, os, sys, shutil
from pathlib import Path
import numpy as np

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "stage11_run/ch1d_fold2"))
from f2_build import fit_scaler, standardise  # noqa: E402

SCR = Path(os.environ["SCR"])
E = SCR / "stage11_ch1/e_fold2_offset"
SRC, DST, VAR = Path(os.environ["SRC"]), Path(os.environ["DST"]), os.environ.get("VARIANT", "cfl")
CLIP = 10.0
LINK = ["X_raw.npy", "y_C80.npy", "y_C120.npy", "y_q95.npy", "y_betaN.npy", "y_aux.npy", "times.npy", "offsets.npy",
        "shot_id.npy", "files.txt"]


def flipmask():
    par = np.load(os.environ.get("PARITY", str(E / "parity.npz")))
    names = list(par["names"])
    odd = (par["cls_full"] == 1) if VAR in ("cfl", "cfp") else (par["cls_cur"] == 1)
    odd = odd.copy()
    if VAR in ("cfl", "ccl"):
        odd[names.index("v1_sign_ip")] = False          # keep the raw Ip-sign flag
    return odd, names.index("time_sign_ip_shot"), names


def canon(Xr, odd, sc):
    b = np.asarray(Xr, np.float64)
    s = b[:, sc].copy()
    bad = ~np.isin(s, (-1.0, 1.0))
    if bad.any():
        raise SystemExit(f"time_sign_ip_shot not +-1 on {bad.sum()} rows")
    b[:, odd] *= s[:, None]
    return b


def main():
    odd, sc, names = flipmask()
    print(f"{VAR}: flipping {odd.sum()} odd columns; sign col {names[sc]}", flush=True)
    DST.mkdir(parents=True, exist_ok=True)
    s0 = np.load(SRC / "scaler_v21.npz")
    is_v1 = s0["is_v1"].astype(bool)
    Xtr = np.load(SRC / "train/X_raw.npy", mmap_mode="r")
    N = Xtr.shape[0]
    tmp = DST / "train_Xc_raw.npy"
    Xc = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=Xtr.shape)
    nneg = 0
    for a in range(0, N, 200_000):
        c = canon(Xtr[a:a + 200_000], odd, sc)
        nneg += int((np.asarray(Xtr[a:a + 200_000, sc]) < 0).sum())
        Xc[a:a + 200_000] = c.astype(np.float32)
    Xc.flush()
    mean, scale = fit_scaler(Xc, np.arange(N), is_v1)
    np.savez(DST / "scaler_v21.npz", mean=mean, scale=scale, is_v1=is_v1, clip=CLIP, names=s0["names"], odd=odd,
             sign_col=sc, variant=VAR)
    print(f"  train: {N} frames, {nneg} reversed-Ip frames ({nneg/N:.3%}); scaler refit", flush=True)
    for sp in ("train", "holdout", "test"):
        d = DST / sp
        d.mkdir(exist_ok=True)
        for f in LINK:
            if (SRC / sp / f).exists() and not (d / f).exists():
                os.symlink((SRC / sp / f).resolve(), d / f)
        Xr = Xc if sp == "train" else np.load(SRC / sp / "X_raw.npy", mmap_mode="r")
        X = np.lib.format.open_memmap(d / "X.npy", mode="w+", dtype=np.float32, shape=Xr.shape)
        for a in range(0, Xr.shape[0], 200_000):
            blk = Xr[a:a + 200_000] if sp == "train" else canon(Xr[a:a + 200_000], odd, sc)
            X[a:a + 200_000] = standardise(blk, mean, scale, is_v1, CLIP)
        X.flush()
        print(f"  {sp}: {Xr.shape[0]} frames written", flush=True)
    del Xc
    tmp.unlink()
    for f in ("names.txt",):
        shutil.copy(SRC / f, DST / f)
    spec = json.load(open(SRC / "spec.json"))
    spec["version"] = spec.get("version", "") + f" | mirror-canonical {VAR} (stage11E e_canon.py)"
    spec["canonical"] = {"variant": VAR, "n_odd_flipped": int(odd.sum()), "sign_col": names[sc]}
    json.dump(spec, open(DST / "spec.json", "w"), indent=1)
    for a, b in (("fold2", "holdout"), ("holdout150", "test")):
        if (SRC / a).exists() and not (DST / a).exists():
            os.symlink(b, DST / a)
    print("done", flush=True)


if __name__ == "__main__":
    main()
