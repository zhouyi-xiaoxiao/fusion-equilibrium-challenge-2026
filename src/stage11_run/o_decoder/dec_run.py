#!/usr/bin/env python3
"""
Stage 11o / Ch2 decoder — driver: decode every frame of a set into one map (G1 "parts" layout, so ev_ts / the packers work unchanged).

Inputs (all READ-ONLY products of other tracks, or this track's target tables):
  --inv RUN:TAG     inverse parts            --net RUN:TAG   net parts          --blend RUN:TAG   0.5 / 0.5 blend parts (start "B", scalars)
  --real DIR        syn_gen --mode real set of the same shots (vacuum flux of the measured coil currents `vac`, `gauge`)
  --targets X.parquet (dw.py apply: mu_* per (file, k), w_psi_net, rn_*) + X_stats.json (corpus sd of the targets = the R2 normalisers' proxy)
  --prior J.json    dw.py fit output ("prior": robust sds of the start map's control-point errors on the DIII-D TRAIN pool, units of a)
Output: <out>/parts/<tag>_<machine>_<idx>.npz, <out>/files_<tag>_<machine>.txt, <out>/dec_<tag>_<machine>.parquet (per-frame decoder diagnostics).
MAST: rule3_guard installed (dec_core.Geo); only our own parts, the input-derived vacuum flux and the public wall polygon are opened.
"""
import argparse, json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dec_core as DC  # noqa: E402

A = None
G = None
FD = None
CTX = {}


def names_of(run, tag, machine):
    return [x for x in (Path(run) / f"files_{tag}_{machine}.txt").read_text().split("\n") if x]


def part(run, tag, machine, i):
    return Path(run) / "parts" / f"{tag}_{machine}_{i:05d}.npz"


def vstat(job):
    i, name = job
    z = np.load(part(*CTX["blend"], A.machine, i), allow_pickle=False)
    ks = z["ks"].astype(int); ok = z["ok"].astype(bool)
    p = z["psi"][ks[ok[ks]]].astype(np.float64)
    return float(p.sum()), float((p ** 2).sum()), int(p.size)


def work(job):
    i, name = job
    t0 = time.time()
    m = A.machine
    out = part(A.out, A.tag, m, i)
    zb = np.load(part(*CTX["blend"], m, i), allow_pickle=False)
    zi = np.load(part(*CTX["inv"], m, CTX["iI"][name]), allow_pickle=False)
    zn = np.load(part(*CTX["net"], m, CTX["iN"][name]), allow_pickle=False)
    zr = np.load(Path(A.real) / f"{CTX['iR'][name]:05d}.npz", allow_pickle=False)
    assert str(zr["fname"]) == name, (name, str(zr["fname"]))
    kr = {int(k): j for j, k in enumerate(zr["ks"].astype(int))}
    ks = zb["ks"].astype(int); ok = zb["ok"].astype(bool)
    tg = CTX["tg"].get(name)
    psiB, psiI, psiN = zb["psi"], zi["psi"], zn["psi"]
    start = A.cfg.get("start", "B")
    base = {"B": psiB, "I": psiI, "N": psiN}.get(start, psiB)
    po = base.astype(np.float32).copy()
    rows = []
    s, wj, v_psi = CTX["s"], CTX["wj"], CTX["v_psi"]
    for k in ks:
        k = int(k)
        row = {"file": name, "k": k, "status": "skip"}
        if tg is None or k not in tg.index or k not in kr or not ok[k]:
            rows.append(row); continue
        t = tg.loc[k]
        j = kr[k]
        only = A.cfg.get("only_regimes")
        if only is not None and int(t["regime"]) not in only and start != "M":
            row["status"] = "other_regime"           # frames outside the listed regimes keep the start (= blend) map bit for bit
            rows.append(row); continue
        vac = DC.OUT_SIGN[m] * zr["vac"][j].astype(np.float64).T + float(zr["gauge"][j])
        pI, pN, pB = psiI[k].astype(np.float64), psiN[k].astype(np.float64), psiB[k].astype(np.float64)
        if not (np.isfinite(pI).all() and np.isfinite(pN).all()):
            rows.append(row); continue
        w = float(t["w_psi_net"])
        mu_psi = w * pN + (1 - w) * pI
        p0 = {"B": pB, "I": pI, "N": pN, "M": mu_psi}[start]
        mu = np.array([t["mu_" + n] for n in DC.TG], float)
        rn = np.array([t["rn_" + n] for n in DC.TG], float)
        try:
            if only is not None and int(t["regime"]) not in only:      # start = mu_psi: outside the listed regimes mu_psi is the 0.5 / 0.5 blend
                p1, info = pB, {"status": "other_regime", "n_it": 0}
            else:
                p1, info = FD.decode(p0, pI, pN, vac, mu, s, wj, mu_psi, v_psi, rn)
        except Exception as e:  # noqa: BLE001
            p1, info = p0, {"status": f"error:{type(e).__name__}:{str(e)[:80]}", "n_it": 0}
        po[k] = p1.astype(np.float32)
        row.update({"status": info["status"], "n_it": info.get("n_it", 0), "L0": info.get("L0", np.nan), "L1": info.get("L1", np.nan), "rej": info.get("rej", ""),
                    "closure": info.get("closure", np.nan), "neg0": info.get("neg0", np.nan), "neg1": info.get("neg1", np.nan), "tv0": info.get("tv0", np.nan),
                    "tv1": info.get("tv1", np.nan), "dpsi_rms": info.get("dpsi_rms", np.nan), "a": info.get("a", np.nan)})
        if "c" in info:
            for n, v in zip(FD.coords, info["c"]):
                row["c_" + n] = float(v)
            for n, v0, v1, mm in zip(DC.TG, info["v0"], info["v1"], mu):
                row["f0_" + n], row["f1_" + n], row["mu_" + n] = float(v0), float(v1), float(mm)
        rows.append(row)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, psi=po, q95=zb["q95"], betaN=zb["betaN"], ok=zb["ok"], t=zb["t"], ip_A=zb["ip_A"], ks=zb["ks"], info=np.array("[]"))
    os.replace(tmp, out)
    return rows, time.time() - t0


def main():
    global A, G, FD
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", required=True, choices=["d3d", "mast"]); ap.add_argument("--wall", required=True)
    ap.add_argument("--inv", required=True); ap.add_argument("--net", required=True); ap.add_argument("--blend", required=True); ap.add_argument("--real", required=True)
    ap.add_argument("--targets", required=True); ap.add_argument("--prior", required=True)
    ap.add_argument("--cfg", action="append", default=[]); ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--max-shots", type=int, default=0); ap.add_argument("--shard", default=None, help="i/n")
    ap.add_argument("--vpsi", type=float, default=0.0, help="pooled flux variance of the set (0 = computed from the blend parts of ALL shots of the set)")
    ap.add_argument("--procs", type=int, default=int(os.environ.get("NPROC", "8")))
    A = ap.parse_args()
    cfg = {}
    for f in A.cfg:
        cfg.update(json.load(open(f)))
    for kv in A.set:
        k, v = kv.split("=", 1); cfg[k] = json.loads(v)
    A.cfg = cfg
    CTX["inv"], CTX["net"], CTX["blend"] = (tuple(x.split(":")) for x in (A.inv, A.net, A.blend))
    m = A.machine
    names = names_of(*CTX["blend"], m)
    CTX["iI"] = {n: i for i, n in enumerate(names_of(*CTX["inv"], m))}
    CTX["iN"] = {n: i for i, n in enumerate(names_of(*CTX["net"], m))}
    real = Path(A.real)
    CTX["iR"] = {n: i for i, n in enumerate([x for x in (real.parent / f"files_{real.name}.txt").read_text().split("\n") if x])}
    G = DC.Geo(m, A.wall)
    R = json.load(open(A.prior))
    start = cfg.get("start", "B")
    pri = dict(R["prior"]["B" if start in ("B", "M") else start])
    FD = DC.FrameDecoder(G, cfg, pri)
    tgd = pd.read_parquet(A.targets)
    st = json.load(open(A.targets.replace(".parquet", "_stats.json")))
    CTX["s"] = np.array([st["sd"][n] for n in DC.TG], float)
    w_cp = cfg.get("w_cp", 0.10 / 6)
    CTX["wj"] = np.array([0.20 / 7] * 7 + [w_cp] * 6, float)
    off = set(cfg.get("targets_off", []))
    CTX["wj"] = np.where([n in off for n in DC.TG], 0.0, CTX["wj"])
    CTX["tg"] = {n: g.set_index("k") for n, g in tgd.groupby("file")}
    out = Path(A.out); (out / "parts").mkdir(parents=True, exist_ok=True)
    jobs = list(enumerate(names))
    ctx = get_context("fork")
    if A.vpsi > 0:
        CTX["v_psi"] = A.vpsi
    else:
        with ctx.Pool(A.procs) as pool:
            vs = pool.map(vstat, jobs, chunksize=4)
        s1, s2, n = (sum(v[i] for v in vs) for i in range(3))
        CTX["v_psi"] = s2 / n - (s1 / n) ** 2
    if A.max_shots:
        jobs = jobs[:A.max_shots]
    if A.shard:
        si, sn = (int(x) for x in A.shard.split("/"))
        jobs = jobs[si::sn]
    (out / f"files_{A.tag}_{m}.txt").write_text("\n".join(names) + "\n")
    json.dump({"cfg": cfg, "prior": pri, "v_psi": CTX["v_psi"], "s": CTX["s"].tolist(), "wj": CTX["wj"].tolist(), "coords": FD.coords, "args": {k: v for k, v in vars(A).items() if k != "cfg"}},
              open(out / f"cfg_{A.tag}_{m}.json", "w"), indent=1, default=str)
    t0 = time.time()
    with ctx.Pool(A.procs) as pool:
        res = pool.map(work, jobs, chunksize=1)
    d = pd.DataFrame([r for rows, _ in res for r in rows])
    sfx = "" if not A.shard else f".s{A.shard.replace('/', 'of')}"
    d.to_parquet(out / f"dec_{A.tag}_{m}{sfx}.parquet")
    vc = d["status"].value_counts().to_dict()
    dec = d[d["status"] == "decoded"]
    summ = {"tag": A.tag, "machine": m, "n_shots": len(jobs), "n_frames": len(d), "status": vc, "v_psi": CTX["v_psi"], "wall_s": time.time() - t0,
            "cpu_s_per_frame": float(sum(t for _, t in res) / max(len(d), 1)),
            "L_drop_mean": float((dec["L0"] - dec["L1"]).mean()) if len(dec) else 0.0, "closure_max": float(dec["closure"].abs().max()) if len(dec) else 0.0,
            "c_rms": {c: float(np.sqrt((dec["c_" + c] ** 2).mean())) for c in FD.coords} if len(dec) else {}}
    print("DECRUN " + json.dumps(summ), flush=True)


if __name__ == "__main__":
    main()
