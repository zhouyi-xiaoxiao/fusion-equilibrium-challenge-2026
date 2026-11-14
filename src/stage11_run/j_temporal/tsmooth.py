#!/usr/bin/env python3
"""
Stage 11J / Ch2 temporal — TEMPORAL MODULE, stage A: linear (optionally robust) smoothing of an exported map sequence along the
shot, with fallback / invalid / filled frames treated as MISSING OBSERVATIONS. Post-processing of any G1 / INV "parts" run
(<run>/parts/<tag>_<machine>_<idx>.npz with psi, q95, betaN, ok, t, ip_A, ks, info) -> the same layout under <tag>_<name>.

psi_out(t) = psi_vac(t) [EXACT vacuum flux of the measured coil currents of the frame, the run's own coil model + gauge]
             + Ip(t) x SMOOTH_t{ y },   y = (psi - psi_vac) / Ip  (plasma + virtual-control flux per unit current).
Frame classes (from the run's own per-frame records, label-free):
  P  inverse solution (ok, x_src != g1)                      -> observation, weight 1
  G  G1 fallback shipped (x_src == g1)                       -> miss='g1': missing | 'g1bad': missing only if it fails the
                                                                inverse's own validity test (audit fix F1) | 'none': observation
  F  plasma frame without a solution (time-interpolated fill) -> missing (its content is an interpolation of the neighbours)
  L  Ip < ip_min                                              -> never an observation, output unchanged
Smoothers (all linear in y for fixed weights: y_s = E y, E built per shot from the frame times and weights):
  gauss  Gaussian kernel, window shrinking symmetrically at the shot ends (legacy f_mast_shape 'g<s>'; weights renormalised)
  ll     Gaussian-kernel local-linear regression (exact for linear trends, one-sided at the ends, any missing pattern)
  whit1  random-walk (local-level) smoother  E = (W + lam D1'D1)^-1 W           = steady-state Kalman / RTS, local level
  whit2  integrated-random-walk smoother     E = (W + lam D2'D2)^-1 W           = Kalman / RTS, local linear trend (spline)
Bandwidth s in FRAMES of the processed series (u='f') or in ms (u='ms'); whit: lam from s by the half-power match with a
Gaussian of sigma s (lam1 = s^2/1.386, lam2 = s^4/1.922, differences in units of the median frame spacing).
Robust option rob=n: n re-weighting passes, w_k *= min(1, c*scale/rho_k) with rho_k the leave-one-out residual rms of frame k
inside the wall and scale the shot's MAD-sd of rho (c = robc).
Scalars: 'q': spec / 'b': spec smooth q95 / betaN the same way (missing = non-finite or outside the validity window, and the
missing map classes); absent -> passed through unchanged.
Sub-sampling experiment (DIII-D): --every m --phase p processes ks[p::m] only (bandwidth in frames of THAT series).
MAST: rule3_guard installed, input columns only. DIII-D: no label is read.
Usage: tsmooth.py --machine M --run DIR --tag T --data-dir D --wall W --cfg cfg.json --out ODIR --specs specs.json [--only a,b]
"""
import argparse, json, os, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np

WS = Path(os.environ["WS"])
for p in ("stage11_run/ch2_G1", "stage11_run/ch2_geometry"):
    sys.path.insert(0, str(WS / p))
A = None
SPECS = None
AXIS_RANGE = {"mast": [0.45, 1.25], "d3d": [1.40, 2.05]}       # inv_core.INV_DEFAULT['axis_range'] (the inverse's own test)
CLOSURE_TOL = 0.03                                             # inv_core.INV_DEFAULT['closure_tol']


# ------------------------------------------------------------------------------------------- linear smoothers
def E_gauss(t, w, s):
    n = len(t); E = np.zeros((n, n)); H = int(np.ceil(3 * s))
    for k in range(n):
        hh = min(H, k, n - 1 - k)
        j = np.arange(k - hh, k + hh + 1)
        a = np.exp(-0.5 * ((j - k) / s) ** 2) * w[j]
        if a.sum() > 0:
            E[k, j] = a / a.sum()
    return E


def E_ll(t, w, s):
    """local-linear regression with a Gaussian kernel of sigma s (same units as t)."""
    n = len(t); E = np.zeros((n, n))
    for k in range(n):
        tau = t - t[k]
        a = np.exp(-0.5 * (tau / s) ** 2) * w
        a[np.abs(tau) > 4 * s] = 0.0
        S0, S1, S2 = a.sum(), (a * tau).sum(), (a * tau * tau).sum()
        if S0 <= 0:
            continue
        den = S0 * S2 - S1 * S1
        if (a > 1e-9 * a.max()).sum() >= 3 and den > 1e-9 * S0 * S2:
            E[k] = a * (S2 - S1 * tau) / den
        else:
            E[k] = a / S0
    return E


def E_whit(t, w, s, order, dt):
    n = len(t)
    if n <= order:
        return np.diag((w > 0).astype(float))
    x = (t - t[0]) / dt
    if order == 1:
        D = np.zeros((n - 1, n))
        for i in range(n - 1):
            h = max(x[i + 1] - x[i], 1e-6)
            D[i, i], D[i, i + 1] = -1 / np.sqrt(h), 1 / np.sqrt(h)          # random walk: var of the increment ~ h
        lam = s ** 2 / 1.386
    else:
        D = np.zeros((n - 2, n))
        for i in range(n - 2):
            h1, h2 = max(x[i + 1] - x[i], 1e-6), max(x[i + 2] - x[i + 1], 1e-6)
            sc = np.sqrt(2.0 / (h1 + h2))
            D[i, i], D[i, i + 1], D[i, i + 2] = sc / h1, -sc * (1 / h1 + 1 / h2), sc / h2    # divided second difference
        lam = s ** 4 / 1.922
    M = np.diag(w) + lam * D.T @ D
    try:
        return np.linalg.solve(M, np.diag(w))
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(M, np.diag(w), rcond=None)[0]


def build_E(t, w, sp, dt):
    kind = sp.get("kind", "gauss")
    s = float(sp.get("s", 2.0))
    if sp.get("u", "f") == "ms":
        s_fr, s_t = s / dt, s
    else:
        s_fr, s_t = s, s * dt
    if s_fr <= 0 or kind == "none":
        return np.diag((w > 0).astype(float))
    if kind == "gauss":
        return E_gauss(t, w, s_fr)
    if kind == "ll":
        return E_ll(t, w, s_t)
    if kind in ("whit1", "whit2"):
        return E_whit(t, w, s_fr, 1 if kind == "whit1" else 2, dt)
    raise ValueError(kind)


def smooth_series(t, y, w0, sp, dt, region=None):
    """y: (n, ...) ; w0: (n,) observation weights (0 = missing). Returns (ys, w, E, supported)."""
    n = len(t)
    w = w0.astype(float).copy()
    yf = y.reshape(n, -1)
    E = build_E(t, w, sp, dt)
    for _ in range(int(sp.get("rob", 0))):
        r = yf - E @ yf
        rr = r[:, region] if region is not None else r
        rho = np.sqrt((rr ** 2).mean(axis=1))
        lev = np.clip(np.diag(E), 0.0, 0.95)
        rho = rho / (1.0 - lev)                                   # leave-one-out residual of a linear smoother
        m = w0 > 0
        if m.sum() < 5:
            break
        med = np.median(rho[m]); sc = 1.4826 * np.median(np.abs(rho[m] - med)) + 1e-30
        c = float(sp.get("robc", 3.0))
        w = w0 * np.minimum(1.0, (c * sc) / np.maximum(rho - med, 1e-30))
        w = np.where(rho - med <= c * sc, w0, w)
        E = build_E(t, w, sp, dt)
    ys = (E @ yf).reshape(y.shape)
    return ys, w, E, np.abs(E).sum(axis=1) > 0


# ------------------------------------------------------------------------------------------- frame classes
def g1_valid(r, machine):
    lo, hi = AXIS_RANGE[machine]
    ra = r.get("gs_Rax")
    if ra is None or not np.isfinite(ra) or not (lo <= ra <= hi) or r.get("clip"):
        return False
    d = r.get("d_out")
    if r.get("obs_out") == "eq" and d is not None and np.isfinite(d) and abs(d) > CLOSURE_TOL:
        return False
    return True


def classes(info, T, machine, miss):
    """per frame: 'P' | 'G' | 'F' | 'L' | '-' (not in the run) and the observation weight."""
    cl = np.array(["-"] * T, dtype="<U1"); w = np.zeros(T)
    for r in info:
        k = int(r["k"])
        if r.get("err") == "Ip<min" or r.get("lvl") == -2:
            cl[k] = "L"
        elif not r.get("ok"):
            cl[k] = "F"
        elif r.get("x_src") == "g1":
            cl[k] = "G"
            w[k] = 0.0 if (miss == "g1" or (miss == "g1bad" and not g1_valid(r, machine))) else 1.0
        else:
            cl[k] = "P"; w[k] = 1.0
    return cl, w


def load_series(i, name):
    """parts + exact per-frame vacuum flux -> y = (psi - vac) / Ip on the processed frames."""
    import g1_core as C
    import g1_data as D
    import run_g1 as RG
    cfg = json.load(open(A.cfg))
    src = Path(A.run) / "parts" / f"{A.tag}_{A.machine}_{i:05d}.npz"
    c = dict(np.load(src, allow_pickle=False))
    row = D.load_row(str(Path(A.data_dir) / name), A.machine)
    S = RG.get_solver(A.machine, row, cfg, A.wall)
    cs, ns = S.coils, int(S.cfg["nsub"])
    t = np.asarray(row["efit_times"], float)
    inp = D.frame_inputs(row, A.machine, t)
    sgn = np.where(inp["ip_A"] >= 0, 1.0, -1.0)
    Ip = np.maximum(np.abs(inp["ip_A"]), S.cfg["ip_min"])
    T = len(t)
    info = json.loads(str(c["info"]))
    ks = c["ks"].astype(int)
    if A.every > 1:
        ks = ks[A.phase::A.every]
    psi = c["psi"].astype(np.float64)
    vac = np.zeros_like(psi)
    for k in ks:
        v = S.vacuum(cs.vec({cn: sgn[k] * inp["cur"][cn][k] for cn in C.current_columns(A.machine)}))
        vac[k] = (RG.OUT_SIGN[A.machine] * v)[::ns, ::ns].T + RG.gauge(A.machine, inp, k)
    y = (psi[ks] - vac[ks]) / Ip[ks, None, None]
    region = S.wall.inside[::ns, ::ns].T.ravel()
    dts = np.diff(t[ks]); dt = float(np.median(dts)) if len(dts) else 1.0
    return c, t, T, ks, y, vac, Ip, info, region, dt


def replace_mask(cls, w, sup, maxgap):
    """frames whose plasma part may be replaced: observations, and missing frames BRACKETED by observations within maxgap
    frames on both sides (no extrapolation beyond the first / last observation: there the original content is kept)."""
    n = len(w); obs = w > 0
    idx = np.arange(n)
    left = np.where(obs, idx, -10 ** 9); left = np.maximum.accumulate(left)
    right = np.where(obs, idx, 10 ** 9); right = np.minimum.accumulate(right[::-1])[::-1]
    brack = (idx - left <= maxgap) & (right - idx <= maxgap)
    return (cls != "L") & (cls != "-") & sup & (obs | brack)


def one(job):
    i, name = job
    if A.skip_missing and not (Path(A.run) / "parts" / f"{A.tag}_{A.machine}_{i:05d}.npz").exists():
        return []
    c, t, T, ks, y, vac, Ip, info, region, dt = load_series(i, name)
    st = []
    for nm, sp in SPECS.items():
        cl, w0 = classes(info, T, A.machine, sp.get("miss", "g1"))
        cls, w = cl[ks], w0[ks]
        ys, wf, E, sup = smooth_series(t[ks], y, w, sp, dt, region)
        rep = replace_mask(cls, w, sup, int(sp.get("maxgap", 3)))
        if sp.get("keep_data"):
            rep &= w <= 0                                          # gap filling only (isolates the missing-data effect)
        new = c["psi"].astype(np.float32).copy()
        kk = ks[rep]
        new[kk] = (vac[kk] + ys[rep] * Ip[kk, None, None]).astype(np.float32)
        cc = dict(c); cc["psi"] = new; cc["ks"] = ks
        for key, col, lo, hi in (("q", "q95", 1.2, 25.0), ("b", "betaN", 0.0, 8.0)):
            sq = sp.get(key)
            if not sq:
                continue
            v = c[col].astype(float)[ks] * float(sq.get("factor", 1.0))
            wv = w * (np.isfinite(v) & (v > lo) & (v < hi))
            vv = np.where(wv > 0, v, 0.0)
            if sq.get("log"):
                vv = np.where(wv > 0, np.log(np.maximum(v, 1e-9)), 0.0)
            vs, _, _, sv = smooth_series(t[ks], vv, wv, sq, dt)
            if sq.get("log"):
                vs = np.exp(vs)
            out = c[col].astype(float).copy()
            out[ks] = np.where(sv, vs / float(sq.get("factor", 1.0)), np.nan)
            cc[col] = out
        dst = Path(A.out) / "parts" / f"{A.tag}_{nm}_{A.machine}_{i:05d}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **cc)
        os.replace(tmp, dst)
        st.append((nm, int((cls == "P").sum()), int((cls == "G").sum()), int(((cls == "G") & (w <= 0)).sum()), int((cls == "F").sum()),
                   int((cls == "L").sum()), int(rep.sum()), float((wf[w > 0] < 0.999).mean()) if (w > 0).any() else 0.0))
    return st


def cv_one(job):
    """leave-one-out cross-validation of the map smoother (label-free): per bandwidth, the sum over observation frames of the
    squared leave-one-out residual of y inside the wall, in units of the shot's own frame-to-frame scatter."""
    i, name = job
    c, t, T, ks, y, vac, Ip, info, region, dt = load_series(i, name)
    sp0 = SPECS[A.cv_spec]
    cl, w0 = classes(info, T, A.machine, sp0.get("miss", "g1"))
    w = w0[ks]
    n = len(ks)
    if (w > 0).sum() < 8:
        return None
    yf = y.reshape(n, -1)[:, region]
    obs = w > 0
    d1 = yf[1:] - yf[:-1]
    pair = obs[1:] & obs[:-1]
    out = {"n": int(obs.sum()), "npix": int(yf.shape[1]), "d1": float((d1[pair] ** 2).sum()), "n1": int(pair.sum()),
           "tot": float(((yf[obs] - yf[obs].mean(0)) ** 2).sum()), "cv": {}, "lag": {}, "lagv": {}}
    # INPUT side (noise-free): vacuum flux of the measured coil currents per unit Ip inside the wall, gauge (spatial mean) removed
    vf = (vac[ks] / Ip[ks, None, None]).reshape(n, -1)[:, region]
    vf = vf - vf.mean(axis=1, keepdims=True)
    out["totv"] = float(((vf[obs] - vf[obs].mean(0)) ** 2).sum())
    for h in (1, 2, 3, 4, 6, 8, 12, 16):
        if n > h:
            pr = obs[h:] & obs[:-h]
            if pr.any():
                out["lagv"][h] = (float(((vf[h:] - vf[:-h])[pr] ** 2).sum()), int(pr.sum()))
    for h in (1, 2, 3, 4, 6, 8, 12, 16):                              # variogram of y (label-free noise / signal split)
        if n > h:
            pr = obs[h:] & obs[:-h]
            if pr.any():
                out["lag"][h] = (float(((yf[h:] - yf[:-h])[pr] ** 2).sum()), int(pr.sum()))
    for s_ in A.cv_s:
        sp = dict(sp0); sp["s"] = s_
        E = build_E(t[ks], w, sp, dt)
        r = yf - E @ yf
        lev = np.clip(np.diag(E), 0.0, 0.98)
        r = r[obs] / (1.0 - lev[obs])[:, None]
        out["cv"][s_] = float((r ** 2).sum())
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["mast", "d3d"]); ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--data-dir", required=True); ap.add_argument("--wall", required=True); ap.add_argument("--cfg", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--specs", required=True); ap.add_argument("--only", default=None)
    ap.add_argument("--cv-spec", default=None, help="spec name: leave-one-out CV over --cv-s bandwidths instead of writing parts")
    ap.add_argument("--cv-s", default="0.2,0.3,0.4,0.5,0.6,0.75,1,1.25,1.5,2,2.5,3,4,6,8,12"); ap.add_argument("--cv-out", default=None)
    ap.add_argument("--every", type=int, default=1); ap.add_argument("--phase", type=int, default=0); ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--stride", type=int, default=1); ap.add_argument("--procs", type=int, default=int(os.environ.get("NPROC", "8")))
    A = ap.parse_args()
    if A.machine == "mast":
        import rule3_guard
        rule3_guard.install()
    SPECS = json.load(open(A.specs))
    if A.only:
        SPECS = {k: SPECS[k] for k in A.only.split(",")}
    names = [x for x in (Path(A.run) / f"files_{A.tag}_{A.machine}.txt").read_text().split("\n") if x]
    Path(A.out).mkdir(parents=True, exist_ok=True)
    if A.cv_spec:
        A.cv_s = [float(v) for v in A.cv_s.split(",")]
        jobs = list(enumerate(names))[::A.stride]
        with get_context("fork").Pool(A.procs) as pool:
            res = [r for r in pool.imap_unordered(cv_one, jobs, chunksize=1) if r is not None]
        ntot = sum(r["n"] * r["npix"] for r in res)
        # pooled: every shot weighted by its own noise scale (half the adjacent-frame squared difference per frame-pixel)
        pooled = {s_: float(sum(r["cv"][s_] for r in res) / ntot) for s_ in A.cv_s}
        rel = {s_: float(np.mean([r["cv"][s_] / max(0.5 * r["d1"] / max(r["n1"], 1) * r["n"], 1e-300) for r in res])) for s_ in A.cv_s}
        best = [min(A.cv_s, key=lambda s_: r["cv"][s_]) for r in res]
        lag = {h: float(sum(r["lag"][h][0] for r in res if h in r["lag"]) / max(sum(r["lag"][h][1] * r["npix"] for r in res if h in r["lag"]), 1)) for h in (1, 2, 3, 4, 6, 8, 12, 16)}
        tot = float(sum(r["tot"] for r in res) / ntot)
        totv = float(sum(r["totv"] for r in res) / ntot)
        lagv = {h: float(sum(r["lagv"][h][0] for r in res if h in r["lagv"]) / max(sum(r["lagv"][h][1] * r["npix"] for r in res if h in r["lagv"]), 1)) for h in (1, 2, 3, 4, 6, 8, 12, 16)}
        out = {"machine": A.machine, "tag": A.tag, "every": A.every, "spec": SPECS[A.cv_spec], "n_shots": len(res), "cv_pooled": pooled, "cv_rel_mean": rel,
               "s_pooled": min(pooled, key=pooled.get), "s_rel": min(rel, key=rel.get), "s_shot_median": float(np.median(best)),
               "s_shot_quartiles": [float(v) for v in np.percentile(best, [25, 50, 75])], "variogram": lag, "within_shot_var": tot,
               "variogram_vacuum": lagv, "within_shot_var_vacuum": totv, "frames_per_shot_median": float(np.median([r["n"] for r in res]))}
        if A.cv_out:
            json.dump(out, open(A.cv_out, "w"), indent=1)
        print(f"CVTS {A.machine} {A.tag} e{A.every} [{A.cv_spec}]: s_pooled {out['s_pooled']} s_rel {out['s_rel']} shot median {out['s_shot_median']} quartiles {out['s_shot_quartiles']} | "
              + "cv_rel " + " ".join(f"{s_:g}:{rel[s_]:.4f}" for s_ in A.cv_s) + " | variogram/2 (lag: value / within-shot var) " + " ".join(f"{h}:{0.5 * v / tot:.4f}" for h, v in lag.items())
              + " | INPUT (vacuum shape flux) variogram/2 / within-shot var " + " ".join(f"{h}:{0.5 * v / totv:.4f}" for h, v in lagv.items()) + f" | frames/shot median {out['frames_per_shot_median']:.0f}")
        sys.exit(0)
    for nm in SPECS:
        (Path(A.out) / f"files_{A.tag}_{nm}_{A.machine}.txt").write_text("\n".join(names) + "\n")
    jobs = list(enumerate(names))[::A.stride]
    tot = {}
    with get_context("fork").Pool(A.procs) as pool:
        for st in pool.imap_unordered(one, jobs, chunksize=1):
            for nm, *v in st:
                a = tot.setdefault(nm, np.zeros(7)); a[:6] += v[:6]; a[6] += v[6]
    for nm, a in tot.items():
        print(f"TSMOOTH {A.machine} {A.tag}_{nm} e{A.every}p{A.phase}: P {int(a[0])} G {int(a[1])} (missing {int(a[2])}) F {int(a[3])} L {int(a[4])} replaced {int(a[5])} "
              f"robust-downweighted frac {a[6] / max(len(jobs), 1):.4f} | spec {json.dumps(SPECS[nm])}")
