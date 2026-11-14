#!/usr/bin/env python3
"""
Stage 11d / ch1d_liloss — li WINDOW GUARD (the cheap profile-mode li corrector of review follow-up 1, §2, applied only where the
scorer would NULL the frame's li).

Why: derive._clamp_to_nan nulls a derived li outside 0.1..5 and the reduction mean-substitutes it, so a predicted li of 5.2 on
a frame whose true li is 4.8 costs (4.8 - <li>)^2 ~ 13 instead of 0.16. On the standard holdout the last 2-3 frames of one
shot (true li 4.2-4.9) are exactly that: they carry ~1/3 of the li SSE of every uncorrected map (FM, TW, FM+TW).

What: for every frame, the OFFICIAL prediction-path derivation (extract_lcfs_with_sign(n_iter=22, axis_sign=AXIS_SIGN) +
internal_inductance on the predicted contour).  If li > 5 (or < 0.1) the interior flux is modified with the contour-preserving
monotone profile mode
     phi' = phi + a (phi_a - phi_b) u^2 (1-u)^2,   u = (phi_a - phi)/(phi_a - phi_b),  only at nodes inside the predicted LCFS
(phi = AXIS_SIGN * psi, phi_a = axis value, phi_b = LCFS level; zero value and slope at the axis and at the boundary, so the
axis and the LCFS stay put; grad phi' = grad phi (1 - a g'(u)), |g'| <= 0.19, so phi' stays monotone for |a| < 5.2), |a| <= A_MAX (2), with a found by secant iterations on the EXACT scorer li (re-extracted contour)
towards a fixed, pre-registered target (4.90 for li > 5, 0.12 for li < 0.1). The modified frame is kept only if its li is
inside the window and its contour moved by < HD_MAX = 2 cm (Hausdorff; one frame of extra D_LCFS is ~1e-3 of the cost of a
nulled li); otherwise the frame is left unchanged.
Nothing else changes (q95/betaN pass through, all other frames bit-identical).
Usage: li_guard.py --pred IN.npz --out OUT.npz [--feat-dir DIR (offsets for n shots)] [--workers 280]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np

WS = Path(os.environ["WS"])
MACHINE = "DIII-D"
HI, LO, T_HI, T_LO = 5.0, 0.1, 4.90, 0.12
A_MAX = float(os.environ.get("A_MAX", "2.0"))     # mode A monotone for |a| < 5.2 (max |d/du u^2(1-u)^2| = 0.19)
HD_MAX = float(os.environ.get("HD_MAX", "0.02"))   # [m] contour-change safety bound (a nulled li costs ~1e3x more than 1 frame of D_LCFS)
_G = {}


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    for p in (WS / "starter", WS / "starter/fusion_scoring"):
        sys.path.insert(0, str(p))
    from common import AXIS_SIGN
    m = np.load(WS / "starter/fusion_scoring/masks/d3d_envelope.npz")
    _G["R"], _G["Z"] = m["grid_R"].astype(np.float64), m["grid_Z"].astype(np.float64)
    _G["mc"] = m["mask_coarse"].astype(bool)
    _G["mf"] = _G["mc"].astype(np.float64)
    _G["ax"] = float(AXIS_SIGN[MACHINE])
    RR, ZZ = np.meshgrid(_G["R"], _G["Z"])
    _G["RR"], _G["ZZ"] = RR.ravel(), ZZ.ravel()


def li_pred(psi):
    from derive import internal_inductance
    from lcfs import extract_lcfs_with_sign
    C, _ = extract_lcfs_with_sign(psi, _G["R"], _G["Z"], MACHINE, _G["mc"], _G["mf"], n_iter=22, axis_sign=_G["ax"])
    if C is None or len(C) < 8:
        return np.nan, None
    return internal_inductance(psi, _G["R"], _G["Z"], MACHINE, C), C


MODES = (("A", lambda u: u ** 2 * (1 - u) ** 2, -A_MAX, A_MAX),     # review follow-up 1 mode: zero value+slope at axis and LCFS
         ("B", lambda u: u ** 2 * (1 - u), -0.9, 2.5))           # fallback: zero slope at the axis, boundary |grad| x (1+a);
                                                                  # monotone for -1 < a < 3; contour level set unchanged


def guard_frame(psi):
    """-> (new_psi or None, info)"""
    from contour import symmetric_hausdorff
    from derive import _bilinear, _points_in_polygon
    from o_point import find_o_point
    li0, C0 = li_pred(psi)
    if C0 is None or not np.isfinite(li0) or LO <= li0 <= HI:
        return None, {"li0": float(li0)}
    tgt = T_HI if li0 > HI else T_LO
    s = _G["ax"]
    phi = s * psi.astype(np.float64)
    _, _, iz, ir, _ = find_o_point(phi, _G["R"], _G["Z"], _G["mc"])
    ins = _points_in_polygon(_G["RR"], _G["ZZ"], C0).reshape(phi.shape)
    pa = float(phi[iz, ir]) if ins[iz, ir] else float(phi[ins].max())
    pb = float(np.mean(_bilinear(phi, _G["R"], _G["Z"], C0[:, 0], C0[:, 1])))
    u = (pa - phi) / (pa - pb)
    tried = {}
    for mname, g, amin, amax in MODES:
        bump = np.where(ins & (u > 0) & (u < 1), (pa - pb) * g(np.clip(u, 0, 1)), 0.0)

        def ev(a):
            p = (s * (phi + a * bump)).astype(np.float32)          # written as float32, evaluated as the harness reads it
            l, C = li_pred(p)
            return l, C, p

        pts = [(0.0, li0)]
        for a in (0.05, -0.05):
            l, _, _ = ev(a)
            if np.isfinite(l):
                pts.append((a, l))
        if len(pts) < 2:
            tried[mname] = "probe"
            continue
        slope = float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0])
        best, a_cur, l_cur = None, 0.0, li0
        for it in range(10):
            if slope == 0 or not np.isfinite(slope):
                break
            a_new = float(np.clip(a_cur + (tgt - l_cur) / slope, amin, amax))
            if abs(a_new - a_cur) < 1e-6:
                break
            l, C, p = ev(a_new)
            if not np.isfinite(l):
                break
            sl = (l - l_cur) / (a_new - a_cur)
            if np.isfinite(sl) and sl != 0 and np.sign(sl) == np.sign(slope):
                slope = sl
            a_cur, l_cur = a_new, l
            pts.append((a_new, l))
            if LO <= l <= HI and C is not None:
                hd = symmetric_hausdorff(np.asarray(C0), np.asarray(C))
                if hd < HD_MAX and (best is None or abs(l - tgt) < abs(best[1] - tgt)):
                    best = (a_new, l, p, hd)
                if abs(l - tgt) < 0.03:
                    break
        if best is not None:
            return best[2], {"li0": float(li0), "li1": float(best[1]), "mode": mname, "a": best[0], "haus_m": float(best[3]),
                             "n_eval": len(pts), "rmse_change_over_sd": float(np.sqrt(np.mean((best[2] - psi) ** 2)) / psi.std())}
        tried[mname] = [(round(x, 4), round(y, 4)) for x, y in pts]
    return None, {"li0": float(li0), "fail": "no_valid", "tried": tried}


def work(args):
    s, P = args
    out, infos = P.copy(), []
    for k in range(len(P)):
        newp, info = guard_frame(P[k])
        if newp is not None or "fail" in info:
            info.update(shot=s, frame=k)
            infos.append(info)
        if newp is not None:
            out[k] = newp
    return s, out, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=280)
    ap.add_argument("--halves", action="store_true", help="also write <out>_A (shots 0-74) / _B (75-149) for the 150-shot holdout")
    a = ap.parse_args()
    t0 = time.time()
    z = np.load(a.pred)
    n = len([k for k in z.files if k.endswith("_psirz")])
    tasks = [(s, z[f"shot_{s:04d}_psirz"]) for s in range(n)]
    res = {}
    infos = []
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        for s, P, inf in pool.imap_unordered(work, tasks, chunksize=1):
            res[s] = P
            infos += inf
    d = {}
    for s in range(n):
        d[f"shot_{s:04d}_psirz"] = res[s].astype(np.float32)
        d[f"shot_{s:04d}_q95"] = z[f"shot_{s:04d}_q95"]
        d[f"shot_{s:04d}_betaN"] = z[f"shot_{s:04d}_betaN"]
    np.savez(a.out, **d)
    if a.halves:
        h = n // 2
        for sfx, rng in (("_A", range(h)), ("_B", range(h, n))):
            np.savez(a.out[:-4] + sfx + ".npz", **{f"shot_{j:04d}_{k}": d[f"shot_{s:04d}_{k}"]
                                                     for j, s in enumerate(rng) for k in ("psirz", "q95", "betaN")})
    nf = sum(len(v) for v in res.values())
    ok = [i for i in infos if "li1" in i]
    rep = {"pred": a.pred, "out": a.out, "shots": n, "frames": nf, "guarded": len(ok), "failed": len(infos) - len(ok),
           "infos": infos, "sec": time.time() - t0}
    json.dump(rep, open(a.out + ".guard.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "infos"}), flush=True)
    for i in infos:
        print("  ", json.dumps(i), flush=True)


if __name__ == "__main__":
    main()
