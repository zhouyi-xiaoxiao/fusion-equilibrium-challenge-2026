#!/usr/bin/env python3
"""Stage 11E / CX — parity of every FEATURE-V2.2 column under the full current mirror (inputs only).

FEATURE-V2.2 = feat_v21 code path with the corrected DIII-D Ip axis (e_ip_axis/feat_v22.py).  The canonical map of
e_fold2_offset (X_c = X_raw * s on the odd columns) was validated on v2.1; this re-runs the identical 80-shot test
(same rng seed / same picks as e_parity.py) with feat_v21.features_v21 patched to the v2.2 builder, and checks that
the odd/even classes are identical to the v2.1 parity file.  Writes $CX/parity_v22.npz (same keys as parity.npz).
"""
import os, sys
from pathlib import Path
import multiprocessing as mp
import numpy as np

WS = Path(os.environ["WS"])
sys.path.insert(0, str(WS / "stage11_run/e_ip_axis"))
sys.path.insert(0, str(WS / "stage11_run/e_fold2_offset"))
import feat_v22  # noqa: E402
feat_v22.patch_feat_v21()          # every `from feat_v21 import features_v21` now returns the v2.2 builder
import e_parity  # noqa: E402

SCR = Path(os.environ["SCR"])
E = SCR / "stage11_ch1/e_fold2_offset"
CX = Path(os.environ["CX"])


def work(path):
    import feat_v21
    assert feat_v21.features_v21 is feat_v22.features_v22
    return e_parity.work(path)


def main():
    Em = np.load(E / "emeta_train.npz")
    rng = np.random.default_rng(1)
    s = np.sign(Em["ip_max"]); era = np.round(Em["mag_dt"], 2)
    pick = []
    for e in (0.05, 0.5):
        for a in (1, -1):
            q = np.flatnonzero((s == a) & (era == e))
            pick += list(rng.choice(q, 20, replace=False))
    files = [str(SCR / "data_full/diii_d_train" / Em["fname"][i]) for i in pick]
    import warnings; warnings.simplefilter("ignore")
    with mp.get_context("fork").Pool(70) as pool:
        res = pool.map(work, files)
    names = [l for l in (SCR / "stage11_ch1/feat_v21/names.txt").read_text().split("\n") if l]
    tol = 1e-5
    save = {"names": np.array(names)}
    p21 = np.load(E / "parity.npz")
    ok = True
    for k, lab in ((0, "cur"), (1, "full")):
        dodd = np.max([r[k][0] for r in res], 0); devn = np.max([r[k][1] for r in res], 0); nanm = np.any([r[k][2] for r in res], 0)
        cls = np.where(devn <= tol, 0, np.where(dodd <= tol, 1, 2))
        cls[nanm] = 2
        save[f"cls_{lab}"] = cls; save[f"dodd_{lab}"] = dodd; save[f"devn_{lab}"] = devn
        same = bool(np.array_equal(cls, p21[f"cls_{lab}"]))
        ok &= same
        print(f"[v2.2 {lab} mirror] {len(files)} shots: odd {int((cls==1).sum())}, even {int((cls==0).sum())}, neither {int((cls==2).sum())}"
              f" | identical to v2.1 classes: {same}", flush=True)
        for j in np.flatnonzero(cls != p21[f"cls_{lab}"]):
            print(f"  DIFF {j} {names[j]} v22 {cls[j]} v21 {p21[f'cls_{lab}'][j]} dodd {dodd[j]:.3g} devn {devn[j]:.3g}")
    np.savez(CX / "parity_v22.npz", **save)
    print("PARITY_V22_OK" if ok else "PARITY_V22_MISMATCH", flush=True)
    sys.exit(0 if ok else 3)


if __name__ == "__main__":
    main()
