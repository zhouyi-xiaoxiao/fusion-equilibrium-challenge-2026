#!/usr/bin/env python3
"""
Stage 11o / Ch2 decoder — SYNTHETIC-MAST populations for the robust weight w_S and for the decoder's in-geometry test (label-free:
truth = OUR OWN solver's synthetic equilibria driven by MAST input frames; no MAST label, no envelope; rule3_guard installed).

Truth sets = k_corrector/ksyn.py gen (READ-ONLY tool; frozen copy of the synth_net generator): per driver shot
  <truth>/synth/<tag>/<idx>.npz  X (net inputs of the real frame), F (Thomson features of the 3 noise realisations), psi (plasma flux, [R, Z])
  <truth>/truth/<tag>/<idx>.npz  psi_truth (total flux, stored orientation), rendered profile, ks
and an inverse run on the rendered diagnostics (ksyn.py inv; G1 parts layout).

  net    amortised net (the MAST twin of the production net) on the synthetic frames -> net parts, 0.5 / 0.5 blend parts with the inverse run,
         vacuum-flux files in the layout dec_run.py expects, and the TRUTH functional table (published functionals, public wall)
  ws     robust weight per functional from several population tables: w_S = argmin_w max_d L_jd(w) + 0.05 (w - 1/2)^2,
         L_jd(w) = E_d[(w e_N + (1 - w) e_I)^2] / Var_d(truth_j); feasible interval of w for the projection L_jd(w) <= L_jd(w_S) + 0.02
"""
import argparse, json, os, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
WS = Path(os.environ["WS"])
OUT_SIGN = {"d3d": -1.0, "mast": 1.0}
G = None
A = None


def truth_rows(job):
    j, name, tfile, parts = job
    z = np.load(tfile, allow_pickle=False)
    P = [np.load(p, allow_pickle=False)["psi"] if p and Path(p).exists() else None for p in parts]
    rows = []
    for i, k in enumerate(z["ks"].astype(int)):
        g = z["psi_truth"][i].astype(np.float64)
        f = G.F(g)
        row = {"file": name, "k": int(k), "psi_s1": float(g.sum()), "psi_s2": float((g ** 2).sum()), "psi_n": int(g.size)}
        v = f["v"] if f is not None else np.full(13, np.nan)
        for n, x in zip(("Ra", "Za", "kappa", "tri_top", "tri_bot", "volume", "li", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro"), v):
            row["g_" + n] = float(x)
        for lab, p in zip(("I", "N", "B"), P):
            if p is not None:
                row[lab + "_psi_sse"] = float(((p[k].astype(np.float64) - g) ** 2).sum())
        rows.append(row)
    return rows


def cmd_net(a):
    global G
    import torch
    import dec_core as DC
    G = DC.Geo(a.machine, a.wall)                       # installs rule3_guard for MAST
    sys.path.insert(0, str(WS / "stage11_run/j_synth_net"))
    from syn_train import Net, DIAG_KEYS
    from syn_gate import gate
    M = torch.load(a.model, map_location="cpu", weights_only=False)
    K = M["K"]
    nets = []
    for st in M["members"]:
        n = Net(M["nin"], K + len(M["names"]), M["width"], M["depth"], M.get("drop", 0.05)); n.load_state_dict(st); n.eval(); nets.append(n)
    comps = torch.tensor(M["comps"]); mu = torch.tensor(M["mu"])
    z0 = (0.0 - M["x_mu"]) / M["x_sd"]
    xlo, xhi = M["x_lo"].copy(), M["x_hi"].copy()
    xlo[M["nXc"]:] = np.minimum(xlo[M["nXc"]:], z0[M["nXc"]:]); xhi[M["nXc"]:] = np.maximum(xhi[M["nXc"]:], z0[M["nXc"]:])
    truth = Path(a.truth)
    anc = json.load(open(truth / f"anchors_{a.tag}.json"))["shots"]
    irun, itag = a.inv.rsplit(":", 1)
    out = Path(a.out); (out / "parts").mkdir(parents=True, exist_ok=True); (out / "real" / a.name).mkdir(parents=True, exist_ok=True)
    names = [x for x in (Path(irun) / f"files_{itag}_{a.machine}.txt").read_text().split("\n") if x]
    kI, kK = list(DIAG_KEYS).index("ip"), list(DIAG_KEYS).index("k")
    sg = OUT_SIGN[a.machine]
    jobs, r2n, r2d = [], [0.0, 0.0], 0
    nfr = 0
    tidx = {s["file"]: int(s["idx"]) for s in anc}
    for j, name in enumerate(names):                      # parts index = position in the inverse run's list; truth file = the generator's idx
        if name not in tidx:
            continue
        ti = tidx[name]
        ft, fs = truth / "truth" / a.tag / f"{ti:05d}.npz", truth / "synth" / a.tag / f"{ti:05d}.npz"
        fi = Path(irun) / "parts" / f"{itag}_{a.machine}_{j:05d}.npz"
        if not (ft.exists() and fs.exists() and fi.exists()):
            continue
        zt, zs, zi = np.load(ft, allow_pickle=False), np.load(fs, allow_pickle=False), np.load(fi, allow_pickle=False)
        ks = zt["ks"].astype(int)
        assert (zs["diag"][:, kK].astype(int) == ks).all()
        ip = zs["diag"][:, kI].astype(float)
        vac = zt["psi_truth"].astype(np.float64) - sg * np.transpose(zs["psi"].astype(np.float64), (0, 2, 1))          # stored orientation, gauge included
        F = zs["F"][:, a.real].astype(np.float32)
        if M.get("gate", False):
            F, _ = gate(F)
        F = F.astype(np.float64) * M["feat_mask"][None, :]
        x = np.concatenate([zs["X"], F], 1)
        xs = np.clip((x - M["x_mu"]) / M["x_sd"], xlo, xhi)
        with torch.no_grad():
            o = torch.stack([m(torch.tensor(xs.astype(np.float32))) for m in nets]).mean(0)
            res = (o[:, :K] @ comps + mu).double().reshape(len(ks), vac.shape[2], vac.shape[1]).numpy()                    # [R, Z] per MA
        sc = o[:, K:].numpy() * M["ys_sd"] + M["ys_mu"]
        psiN = vac + sg * np.transpose(res * (ip / 1e6)[:, None, None], (0, 2, 1))
        T = len(zi["t"])
        pn = np.zeros((T, vac.shape[1], vac.shape[2]), np.float32); pn[ks] = psiN.astype(np.float32)
        okn = np.zeros(T, bool); okn[ks] = True
        aux = np.full((T, sc.shape[1]), np.nan, np.float32); aux[ks] = sc
        q = np.full(T, np.nan); b = np.full(T, np.nan)
        q[ks], b[ks] = sc[:, M["names"].index("q95")], sc[:, M["names"].index("betaN")]
        np.savez_compressed(out / "parts" / f"{a.name}_N_{a.machine}_{j:05d}.npz", psi=pn, q95=q, betaN=b, ok=okn, t=zi["t"], ip_A=zi["ip_A"], ks=ks,
                            info=np.array("[]"), aux=aux, aux_names=np.array(M["names"]))
        pb = np.zeros_like(pn); pb[ks] = (0.5 * psiN + 0.5 * zi["psi"][ks].astype(np.float64)).astype(np.float32)
        np.savez_compressed(out / "parts" / f"{a.name}_B_{a.machine}_{j:05d}.npz", psi=pb, q95=zi["q95"], betaN=zi["betaN"], ok=zi["ok"], t=zi["t"], ip_A=zi["ip_A"],
                            ks=ks, info=np.array("[]"))
        rf = out / "real" / a.name / f"{j:05d}.npz"
        if not rf.exists():
            np.savez_compressed(rf, vac=np.transpose(sg * vac, (0, 2, 1)).astype(np.float32), gauge=np.zeros(len(ks)), ks=ks, fname=np.array(name))
        g = zt["psi_truth"].astype(np.float64)
        r2n[0] += float(((psiN - g) ** 2).sum()); r2n[1] += float(((zi["psi"][ks].astype(np.float64) - g) ** 2).sum()); r2d += float(((g - g.mean()) ** 2).sum())
        nfr += len(ks)
        jobs.append((j, name, str(ft), [str(fi), str(out / "parts" / f"{a.name}_N_{a.machine}_{j:05d}.npz"), str(out / "parts" / f"{a.name}_B_{a.machine}_{j:05d}.npz")]))
    for t_ in ("N", "B"):
        (out / f"files_{a.name}_{t_}_{a.machine}.txt").write_text("\n".join(names) + "\n")
    (out / "real" / f"files_{a.name}.txt").write_text("\n".join(names) + "\n")
    with get_context("fork").Pool(a.procs) as pool:
        rows = [r for rr in pool.map(truth_rows, jobs, chunksize=2) for r in rr]
    d = pd.DataFrame(rows)
    d.to_parquet(out / f"truth_{a.name}.parquet")
    tot = float(d.psi_s2.sum() - d.psi_s1.sum() ** 2 / d.psi_n.sum())
    print(f"DSYNNET {a.name}: {len(jobs)} shots, {nfr} frames; pooled R2psi vs truth: inverse {1 - d.I_psi_sse.sum() / tot:.5f} net {1 - d.N_psi_sse.sum() / tot:.5f} "
          f"blend {1 - d.B_psi_sse.sum() / tot:.5f}; truth contour fail {float(d.g_kappa.isna().mean()):.4f}", flush=True)


def cmd_ws(a):
    FNS = ["R_axis", "Z_axis", "kappa", "tri_top", "tri_bot", "volume", "li", "Rt", "Zt", "Rb", "Zb", "Ri", "Ro"]
    WC = {"R_axis": "Ra", "Z_axis": "Za"}
    REG = ["ordinary", "startup", "inner_cens", "near_dn"]
    pops = {}
    for spec in a.pop:
        nm, f = spec.split("=", 1)
        d = pd.read_parquet(f)
        pops[nm] = d[d["ok"].astype(bool)].reset_index(drop=True)
    wg = np.linspace(0, 1, 201)
    out = {"_pops": list(pops), "_n": {k: [int(len(v)), int(v["file"].nunique())] for k, v in pops.items()}, "_reg": a.reg, "_tol": a.tol, "_neutral": a.neutral.split(",") if a.neutral else []}
    print(f"   {'functional':9s} {'regime':10s} w_S(net)  feasible [lo, hi]   max_d L(w_S)   L_d at w = 0 (inverse) / 0.5 / 1 (net) per population " + " ".join(pops))
    for j in FNS:
        c = WC.get(j, j)
        out[j] = {}
        for rn in ["pooled"] + REG:
            Ls, used = [], []
            for nm, d in pops.items():
                sel = np.ones(len(d), bool) if rn == "pooled" else (d["regime"].to_numpy() == REG.index(rn))
                g, eI, eN = d["g_" + c].to_numpy(float), d["I_" + c].to_numpy(float) - d["g_" + c].to_numpy(float), d["N_" + c].to_numpy(float) - d["g_" + c].to_numpy(float)
                m = sel & np.isfinite(eI) & np.isfinite(eN)
                if d.loc[m, "file"].nunique() < 30:
                    continue
                V = float(np.var(g[m]))
                # variance normaliser of the WHOLE population (the R2 denominator is corpus-level), not of the regime subset
                Vp = float(np.nanvar(g))
                mii, mnn, mi_n = float((eI[m] ** 2).mean()), float((eN[m] ** 2).mean()), float((eI[m] * eN[m]).mean())
                Ls.append(((1 - wg) ** 2 * mii + wg ** 2 * mnn + 2 * wg * (1 - wg) * mi_n) / max(Vp, 1e-30)); used.append(nm)
            if not Ls:
                out[j][rn] = None
                continue
            Ls = np.array(Ls)
            obj = Ls.max(0) + a.reg * (wg - 0.5) ** 2
            i0 = int(np.argmin(obj))
            if j in out["_neutral"]:
                i0 = int(np.argmin(np.abs(wg - 0.5)))
            feas = (Ls <= Ls[:, [i0]] + a.tol).all(0)
            lo = i0
            while lo > 0 and feas[lo - 1]:
                lo -= 1
            hi = i0
            while hi < len(wg) - 1 and feas[hi + 1]:
                hi += 1
            out[j][rn] = {"w_net": float(wg[i0]), "lo": float(wg[lo]), "hi": float(wg[hi]), "maxL": float(Ls[:, i0].max()), "pops": used,
                          "L_inv": Ls[:, 0].tolist(), "L_half": Ls[:, 100].tolist(), "L_net": Ls[:, -1].tolist(), "L_ws": Ls[:, i0].tolist()}
            print(f"   {j:9s} {rn:10s} {wg[i0]:7.3f}   [{wg[lo]:.3f}, {wg[hi]:.3f}]   {Ls[:, i0].max():10.3f}    "
                  + " | ".join(f"{x:.3f}/{y:.3f}/{z_:.3f}" for x, y, z_ in zip(Ls[:, 0], Ls[:, 100], Ls[:, -1])))
        for rn in REG:
            if out[j][rn] is None:
                out[j][rn] = out[j]["pooled"]
    # ---- the flux map itself (0.55 of the score): robust weight of the net in mu_psi, same min-max rule; cross moment from the 0.5 / 0.5 blend's SSE
    out["psi"] = {}
    for rn in ["pooled"] + REG:
        Ls, used = [], []
        for nm, d in pops.items():
            if "B_psi_sse" not in d:
                continue
            sel = np.ones(len(d), bool) if rn == "pooled" else (d["regime"].to_numpy() == REG.index(rn))
            ii, nn, bb = d["I_psi_sse"].to_numpy(float), d["N_psi_sse"].to_numpy(float), d["B_psi_sse"].to_numpy(float)
            m = sel & np.isfinite(ii) & np.isfinite(nn) & np.isfinite(bb)
            if d.loc[m, "file"].nunique() < 30:
                continue
            vpop = float(d["psi_s2"].sum() - d["psi_s1"].sum() ** 2 / d["psi_n"].sum()) / len(d)             # per-frame SST of the population
            mii, mnn = float(ii[m].mean()), float(nn[m].mean())
            mi_n = float((2.0 * bb[m] - 0.5 * (ii[m] + nn[m])).mean())
            Ls.append(((1 - wg) ** 2 * mii + wg ** 2 * mnn + 2 * wg * (1 - wg) * mi_n) / vpop); used.append(nm)
        if not Ls:
            out["psi"][rn] = None
            continue
        Ls = np.array(Ls)
        tol = a.tol * 0.1                                   # R2psi lives on a 10x finer scale than the functionals' R2
        i0 = int(np.argmin(Ls.max(0) + a.reg * 0.1 * (wg - 0.5) ** 2))
        feas = (Ls <= Ls[:, [i0]] + tol).all(0)
        lo = i0
        while lo > 0 and feas[lo - 1]:
            lo -= 1
        hi = i0
        while hi < len(wg) - 1 and feas[hi + 1]:
            hi += 1
        out["psi"][rn] = {"w_net": float(wg[i0]), "lo": float(wg[lo]), "hi": float(wg[hi]), "maxL": float(Ls[:, i0].max()), "pops": used,
                          "L_inv": Ls[:, 0].tolist(), "L_half": Ls[:, 100].tolist(), "L_net": Ls[:, -1].tolist(), "L_ws": Ls[:, i0].tolist()}
        print(f"   {'psi':9s} {rn:10s} {wg[i0]:7.3f}   [{wg[lo]:.3f}, {wg[hi]:.3f}]   {Ls[:, i0].max():10.4f}    "
              + " | ".join(f"{x:.4f}/{y:.4f}/{z_:.4f}" for x, y, z_ in zip(Ls[:, 0], Ls[:, 100], Ls[:, -1])))
    for rn in REG:
        if out["psi"].get(rn) is None:
            out["psi"][rn] = out["psi"].get("pooled")
    json.dump(out, open(a.out, "w"), indent=1)
    print("DSYNWS", a.out, json.dumps(out["_n"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("net")
    p.add_argument("--machine", default="mast"); p.add_argument("--wall", required=True); p.add_argument("--truth", required=True); p.add_argument("--tag", required=True)
    p.add_argument("--inv", required=True); p.add_argument("--model", required=True); p.add_argument("--real", type=int, default=1, help="noise realisation of the truth set")
    p.add_argument("--name", required=True); p.add_argument("--out", required=True); p.add_argument("--procs", type=int, default=int(os.environ.get("NPROC", "8")))
    p = sp.add_parser("ws")
    p.add_argument("--pop", nargs="+", required=True, help="name=est.parquet ..."); p.add_argument("--reg", type=float, default=0.05); p.add_argument("--tol", type=float, default=0.02)
    p.add_argument("--neutral", default="", help="comma list of functionals whose w_S is fixed to 1/2 (the synthetic worlds cannot represent an estimator's information)")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"net": cmd_net, "ws": cmd_ws}[a.cmd](a)
