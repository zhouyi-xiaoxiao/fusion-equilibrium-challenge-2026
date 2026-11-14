#!/usr/bin/env python3
"""Stage 11J / j_dlcfs_contour — LABEL-topology bank (training labels only): the scorer's own reference extraction, kept raw.

For every feat_v21 row of the train (6,891 shots) and holdout (150 shots) splits, from the float32 label psi exactly as the harness
reads it (phi = -psi): the official bisection (jc_bank.ext_level) gives the reference polygon; stored per frame
  ea, eb   grid edge (node a, node b) of every polygon vertex;  pa, pb = label phi at the two nodes (float32, exact)
  crit     the critical node (lo < phi <= hi), phic0 its label value, lvl0 the last valid level
  cosl     per vertex  L_edge * |cos(angle between the edge and the polygon normal)|  [m]  (normal displacement per unit of u)
  wseg     per vertex arc-length weight (half the two adjacent segment lengths), 0 on the padding / the closing duplicate
With these, the first-order normal displacement of the predicted LCFS at label vertex k for a predicted map phi_hat = phi + e is
  dn_k = cosl_k * (e(crit) - (1 - u_k) e(a_k) - u_k e(b_k)) / (pb_k - pa_k),   u_k = (lvl0 - pa_k) / (pb_k - pa_k)
(Pro note §B 1(b) "normal-displacement variant", evaluated on the scorer's raw polygon, level tied to the critical node).
Rows are matched to the parquet frames by efit_times as ch1d_liloss/li_prep.py; the 512-point contour is checked against the
li cache.  Split-level output $J/banklab/<split>/*.npy (feat_v21 row order).
  SPLITS=train,holdout python jc_bank_label.py [--workers 280] [--max-shots 0]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np

import jc_bank as B
import jc_common as C
from jc_common import NG, VMAX, log

FULL_DIR = C.D3D_TRAIN_DIR
FEAT = C.FEAT21
_D: dict = {}


def geom(a, b, P):
    """Polygon P (M,2) closed (P[0] == P[-1]); edges a, b (M,). -> cosl (M,), wseg (M,)"""
    g = B._G
    M = len(P)
    U = P[:-1]                                              # unique vertices
    n = len(U)
    t = np.roll(U, -1, 0) - np.roll(U, 1, 0)
    # degenerate neighbours (two coincident vertices at the critical node): widen the stencil
    bad = (t ** 2).sum(1) < 1e-16
    if bad.any():
        t2 = np.roll(U, -2, 0) - np.roll(U, 2, 0)
        t[bad] = t2[bad]
    tn = np.sqrt((t ** 2).sum(1)) + 1e-300
    nrm = np.stack([t[:, 1], -t[:, 0]], 1) / tn[:, None]
    ev = np.stack([g["R"][b[:-1] % NG] - g["R"][a[:-1] % NG], g["Z"][b[:-1] // NG] - g["Z"][a[:-1] // NG]], 1)
    cosl = np.abs((ev * nrm).sum(1))
    seg = np.sqrt(((np.roll(U, -1, 0) - U) ** 2).sum(1))
    w = 0.5 * (seg + np.roll(seg, 1))
    out_c = np.zeros(M); out_w = np.zeros(M)
    out_c[:n] = cosl; out_w[:n] = w
    out_c[n:] = cosl[0]
    return out_c, out_w


def work(args):
    split, i, fname, tq, a0 = args
    B.init()
    g = B._G
    import pandas as pd
    from experiments import _as_psirz_stack
    row = pd.read_parquet(FULL_DIR / fname, columns=["efit_times", "efit_psirz"]).iloc[0]
    te = np.asarray(row["efit_times"], dtype=np.float64)
    psi = _as_psirz_stack(row["efit_psirz"])
    n = min(len(te), len(psi))
    idx = np.empty(len(tq), np.int64)
    j = 0
    for k, t in enumerate(tq):
        while j < n and te[j] != t:
            j += 1
        if j >= n:
            raise RuntimeError(f"{fname}: time {t} (row {k}) not found")
        idx[k] = j
        j += 1
    ps = psi[idx]
    T = len(ps)
    r = dict(split=split, i=i, ok=np.zeros(T, bool), why=np.zeros(T, np.uint8), nv=np.zeros(T, np.int16),
             ea=np.zeros((T, VMAX), np.int16), eb=np.zeros((T, VMAX), np.int16), pa=np.zeros((T, VMAX), np.float32),
             pb=np.ones((T, VMAX), np.float32), cosl=np.zeros((T, VMAX), np.float32), wseg=np.zeros((T, VMAX), np.float32),
             lvl0=np.zeros(T), crit=np.zeros(T, np.int16), phic0=np.zeros(T, np.float32), cdiff=np.full(T, np.nan, np.float32),
             rgeo=np.full(T, np.nan, np.float32))
    Cli = _D["C"][split]
    for k in range(T):
        p = ps[k]
        if not np.isfinite(p).all():
            r["why"][k] = 5
            continue
        why, o = B.one_frame(-p.astype(np.float64), None)
        if o is not None:
            r["lvl0"][k] = o["lvl0"]; r["nv"][k] = min(o["nv"], 32767)
        r["why"][k] = why
        if why != 0:
            continue
        nvk = o["nv"]
        P, _ = B.replay_np(o["a"], o["b"], (-p.astype(np.float64)).ravel(), o["lvl0"])
        cl, ws = geom(o["a"], o["b"], P)
        for key, src in (("ea", o["a"]), ("eb", o["b"]), ("pa", o["pa"]), ("pb", o["pb"]), ("cosl", cl)):
            r[key][k, :nvk] = src
            r[key][k, nvk:] = src[-1]
        r["wseg"][k, :nvk] = ws
        r["crit"][k], r["phic0"][k] = o["crit"], o["phic0"]
        r["cdiff"][k] = float(np.abs(o["C512"] - np.asarray(Cli[a0 + k], np.float64)).max())
        r["rgeo"][k] = 0.5 * (o["C512"][:, 0].max() + o["C512"][:, 0].min())
        r["ok"][k] = True
    return r


def run_split(split, pool, max_shots):
    t0 = time.time()
    d = FEAT / split
    files = [l.strip() for l in (d / "files.txt").read_text().split("\n") if l.strip()]
    off = np.load(d / "offsets.npy").astype(np.int64)
    tim = np.load(d / "times.npy")
    S = len(files) if not max_shots else min(max_shots, len(files))
    N = int(off[S])
    o = C.J / "banklab" / split
    o.mkdir(parents=True, exist_ok=True)
    arr = {k: np.lib.format.open_memmap(o / f"{k}.npy", mode="w+", dtype=dt, shape=(N,) + sh) for k, dt, sh in
           [("ok", bool, ()), ("why", np.uint8, ()), ("nv", np.int16, ()), ("ea", np.int16, (VMAX,)), ("eb", np.int16, (VMAX,)),
            ("pa", np.float32, (VMAX,)), ("pb", np.float32, (VMAX,)), ("cosl", np.float32, (VMAX,)), ("wseg", np.float32, (VMAX,)),
            ("lvl0", np.float64, ()), ("crit", np.int16, ()), ("phic0", np.float32, ()), ("cdiff", np.float32, ()),
            ("rgeo", np.float32, ())]}
    tasks = [(split, i, files[i], tim[off[i]:off[i + 1]], int(off[i])) for i in range(S)]
    done = 0
    for r in pool.imap_unordered(work, tasks, chunksize=2):
        p, e = int(off[r["i"]]), int(off[r["i"] + 1])
        for k in arr:
            arr[k][p:e] = r[k]
        done += 1
        if done % 1000 == 0:
            log(f"  {split} {done}/{S} shots {time.time()-t0:.0f}s")
    for k in arr:
        arr[k].flush()
    ok, why = np.asarray(arr["ok"]), np.asarray(arr["why"])
    rep = {"split": split, "shots": int(S), "rows": int(N), "ok_frac": float(ok.mean()),
           "why": {B.WHY[int(c)]: int((why == c).sum()) for c in np.unique(why)},
           "cdiff_max_ok_m": float(np.nanmax(np.asarray(arr["cdiff"])[ok])), "nv_max": int(np.asarray(arr["nv"])[ok].max()),
           "sec": time.time() - t0}
    json.dump(rep, open(o / "report.json", "w"), indent=1)
    log(json.dumps(rep))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=int(os.environ.get("NW", "280")))
    ap.add_argument("--max-shots", type=int, default=0)
    a = ap.parse_args()
    for p in (C.STARTER, C.STARTER / "fusion_scoring"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    B.init()
    import pandas  # noqa: F401
    from experiments import _as_psirz_stack  # noqa: F401
    _D["C"] = {sp: np.load(C.LI / "cache" / sp / "C.npy", mmap_mode="r") for sp in ("train", "holdout")}
    os.environ["OMP_NUM_THREADS"] = "1"
    with mp.get_context("fork").Pool(a.workers) as pool:
        for sp in os.environ.get("SPLITS", "holdout,train").split(","):
            run_split(sp, pool, a.max_shots)


if __name__ == "__main__":
    main()
