#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — apply a trained network to REAL frames (inputs only) and write G1-format parts.

psi(frame) = OUT_SIGN * (psi_vac(measured coils; stored by syn_gen --mode real) + Ip * net(x)).T + gauge  (G1's assembly: the same
coil model incl. the P2 wiring repair, the same EFIT layout sign, the DIII-D E-coil gauge rule; MAST: no offset). Frames with
Ip < ip_min get the nearest valid frame's plasma flux (G1's fallback). q95 / betaN = network outputs, uncalibrated (the
calibration factors are applied by eval_frames / the packager exactly as for the inverse).
Outputs (G1 layout, so eval_frames.py / paired_eval.py / package_g1.py / smooth_parts.py work unchanged):
  <out>/parts/<tag>_<machine>_<idx>.npz, <out>/files_<tag>_<machine>.txt, <out>/cfg_<tag>_<machine>.json
  --calib-fit F.json   (DIII-D TRAIN real set only): C_q95 / C_betaN = median label / prediction ratios on its labels.
MAST: no label is read (the real npz hold inputs only); rule3_guard is installed for --machine mast.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT_SIGN = {"d3d": -1.0, "mast": 1.0}
IP_MIN = 5e4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--real", required=True, help="<out>/real/<tag> of syn_gen --mode real")
    ap.add_argument("--machine", choices=["d3d", "mast"], required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--calib-fit", default=None)
    ap.add_argument("--drop-ts", action="store_true", help="diagnostic: ignore every Thomson feature")
    ap.add_argument("--smooth", type=float, default=0.0, help="Gaussian sigma [frames] on the plasma part per unit Ip along the shot (0 = off)")
    ap.add_argument("--smooth-scal", action="store_true", help="also smooth log q95 / betaN with the same kernel")
    ap.add_argument("--fill-nots", action="store_true", help="frames without a measured outer edge (after the gate): plasma flux per unit Ip and "
                    "scalars interpolated in time between the nearest frames of the shot that have one (the inverse's carry rule) instead of E[.|coils]")
    ap.add_argument("--pin", default="", help="axis pin of the exported map to the Rax / Zax heads: '' | r | z | rz (syn_pin.pin_axis)")
    ap.add_argument("--subset-ks", type=int, default=0, help="write only N evenly spaced frames per shot (the harness's frame subset)")
    ap.add_argument("--drop-feat", default="", help="held-out test: comma list of Thomson feature-name prefixes set to 'missing' (raw 0, the training "
                    "dropout value) after the gate, e.g. 'lo0.35,hi0.35,lo0.65,hi0.65,in_'")
    a = ap.parse_args()
    if a.machine == "mast":
        ws = Path(os.environ["WS"])
        sys.path.insert(0, str(ws / "stage11_run/ch2_geometry"))
        import rule3_guard
        rule3_guard.install()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sys.path.insert(0, str(HERE))
    from syn_train import Net
    from syn_gate import gate
    M = torch.load(a.model, map_location="cpu", weights_only=False)
    K = M["K"]
    nets = []
    for st in M["members"]:
        n = Net(M["nin"], K + len(M["names"]), M["width"], M["depth"], M.get("drop", 0.05)).to(dev); n.load_state_dict(st); n.eval(); nets.append(n)
    comps = torch.tensor(M["comps"], device=dev); mu = torch.tensor(M["mu"], device=dev)
    # support clipping: the raw value 0 (= feature missing; produced by the training dropout but absent from the un-augmented
    # percentiles) is always inside the support of every Thomson feature. [fix 21:58 UTC: flags that are ~always 1 in the synthetic
    # set had lo = hi = 1, so missing features of real frames were clipped to "present".]
    z0 = (0.0 - M["x_mu"]) / M["x_sd"]
    M["x_lo"], M["x_hi"] = M["x_lo"].copy(), M["x_hi"].copy()
    M["x_lo"][M["nXc"]:] = np.minimum(M["x_lo"][M["nXc"]:], z0[M["nXc"]:])
    M["x_hi"][M["nXc"]:] = np.maximum(M["x_hi"][M["nXc"]:], z0[M["nXc"]:])
    real = Path(a.real)
    files = [x for x in (real.parent / f"files_{real.name}.txt").read_text().split("\n") if x]
    out = Path(a.out); (out / "parts").mkdir(parents=True, exist_ok=True)
    (out / f"files_{a.tag}_{a.machine}.txt").write_text("\n".join(files) + "\n")
    json.dump({"synth_net_model": str(a.model), "real": str(real), "train_diag": {k: v for k, v in M["diag"].items() if k != "hist"}},
              open(out / f"cfg_{a.tag}_{a.machine}.json", "w"), indent=1, default=float)
    nq, nb = M["names"].index("q95"), M["names"].index("betaN")
    fit = {"q": [], "b": []}
    stats = {"frames": 0, "pred": 0, "fallback": 0, "no_ts": 0, "clipped_frames": 0, "clip_by_feature": np.zeros(M["nin"], int)}
    aux_all, ens_sd, ens_nrm, pin_d0 = [], [], [], []
    for i, name in enumerate(files):
        z = np.load(real / f"{i:05d}.npz")
        assert str(z["fname"]) == name, (name, str(z["fname"]))
        ks, T, t, Ipf, sgn = z["ks"], int(z["T"]), z["t"], z["Ip"], z["sgn"]
        Ip = Ipf[ks]
        F = z["F"].astype(np.float32)
        if M.get("gate", False):
            F, gs = gate(F)
            for k_ in ("prof_dropped", "sides_dropped", "has_prof", "block_dropped"):
                stats[k_] = stats.get(k_, 0) + gs[k_]
        if a.drop_feat:
            from syn_common_names import FEAT_NAMES as _FN
            dcols = [j_ for j_, n_ in enumerate(_FN) if any(n_.startswith(p_) for p_ in a.drop_feat.split(","))]
            F[:, dcols] = 0.0
            stats["dropped_cols"] = [_FN[j_] for j_ in dcols]
        Fg = F.copy()
        F = F.astype(np.float64) * M["feat_mask"][None, :]
        if a.drop_ts:
            F = F * 0.0
        x = np.concatenate([z["X"], F], 1)
        xs = (x - M["x_mu"]) / M["x_sd"]
        oob = (xs < M["x_lo"] - 1e-3) | (xs > M["x_hi"] + 1e-3)          # stats only (tolerance: float32 percentile rounding)
        stats["clipped_frames"] += int(oob.any(1).sum()); stats["clip_by_feature"] += oob.sum(0)
        xs = np.clip(xs, M["x_lo"], M["x_hi"])
        xn = torch.tensor(xs.astype(np.float32), device=dev)
        with torch.no_grad():
            oo = torch.stack([m(xn) for m in nets])
            o = oo.mean(0)
            res = (o[:, :K] @ comps + mu).double()                               # psi_res per MA, [R, Z] flattened
            ens_sd.append(torch.sqrt((oo[:, :, :K].std(0) ** 2).sum(1)).cpu().numpy() if len(nets) > 1 else np.zeros(len(xn)))
            ens_nrm.append(torch.sqrt((o[:, :K] ** 2).sum(1)).cpu().numpy())
        sc = o[:, K:].cpu().numpy() * M["ys_sd"] + M["ys_mu"]
        nx, ny = z["vac"].shape[1:]
        resn = res.reshape(len(ks), nx, ny).cpu().numpy()
        if M.get("zcanon", 0.0) > 0:                         # Z-canonical maps: move the plasma flux to the predicted axis height
            from syn_shift import zshift
            resn = zshift(resn, -sc[:, M["names"].index("Zax")].astype(float) / M["zcanon"])
        px = resn * (Ip / 1e6)[:, None, None]
        q95, bN = sc[:, nq].astype(float).copy(), sc[:, nb].astype(float).copy()
        good = Ip >= IP_MIN
        gi = np.where(good)[0]
        for j in np.where(~good)[0]:
            stats["fallback"] += 1
            if len(gi):
                jj = gi[np.argmin(np.abs(t[ks][gi] - t[ks][j]))]
                px[j] = px[jj]; q95[j], bN[j] = q95[jj], bN[jj]
            else:
                px[j] = 0.0
        if a.fill_nots and len(gi) > 1:
            has_e = (Fg[:, 0] > 0) & good                    # measured outer edge present after the gate
            src, dst = np.where(has_e)[0], np.where(good & ~has_e)[0]
            if len(src) and len(dst):
                tt_ = t[ks].astype(float)
                y = px[src] / (Ip[src] / 1e6)[:, None, None]
                lqs, bbs = np.log(np.clip(q95[src], 0.5, 50.0)), bN[src]
                for j in dst:
                    p_ = int(np.searchsorted(tt_[src], tt_[j]))
                    lo_, hi_ = (p_ - 1 if p_ > 0 else None), (p_ if p_ < len(src) else None)
                    if lo_ is None or hi_ is None:
                        w_, a_, b_ = 0.0, (hi_ if lo_ is None else lo_), (hi_ if lo_ is None else lo_)
                    else:
                        a_, b_ = lo_, hi_
                        w_ = float((tt_[j] - tt_[src[a_]]) / max(tt_[src[b_]] - tt_[src[a_]], 1e-9))
                    px[j] = ((1 - w_) * y[a_] + w_ * y[b_]) * (Ip[j] / 1e6)
                    q95[j] = float(np.exp((1 - w_) * lqs[a_] + w_ * lqs[b_])); bN[j] = float((1 - w_) * bbs[a_] + w_ * bbs[b_])
                    sc[j] = (1 - w_) * sc[src[a_]] + w_ * sc[src[b_]]
                    stats["filled_nots"] = stats.get("filled_nots", 0) + 1
        if a.smooth > 0 and len(gi) > 2:                    # temporal regularisation of the plasma part (kernel shrinks at the ends)
            y = px[gi] / (Ip[gi] / 1e6)[:, None, None]
            H = int(np.ceil(3 * a.smooth))
            y2 = np.empty_like(y)
            lq, bb_ = np.log(np.clip(q95[gi], 0.5, 50.0)), bN[gi].copy()
            lq2, bb2 = lq.copy(), bb_.copy()
            for n_ in range(len(gi)):
                hh = min(H, n_, len(gi) - 1 - n_)
                wk = np.exp(-0.5 * (np.arange(-hh, hh + 1) / a.smooth) ** 2); wk /= wk.sum()
                y2[n_] = np.tensordot(wk, y[n_ - hh:n_ + hh + 1], axes=1)
                lq2[n_], bb2[n_] = float(wk @ lq[n_ - hh:n_ + hh + 1]), float(wk @ bb_[n_ - hh:n_ + hh + 1])
            px[gi] = y2 * (Ip[gi] / 1e6)[:, None, None]
            if a.smooth_scal:
                q95[gi], bN[gi] = np.exp(lq2), bb2
        full = OUT_SIGN[a.machine] * (z["vac"].astype(np.float64) + px)
        psi_k = np.transpose(full, (0, 2, 1)) + z["gauge"][:, None, None]
        if a.pin:
            from syn_pin import pin_axis, AXIS_SIGN, A_PUBLIC
            jr_, jz_ = M["names"].index("Rax"), M["names"].index("Zax")
            gR_, gZ_ = z["gR"].astype(float), z["gZ"].astype(float)
            for j in gi:
                new, pinfo = pin_axis(AXIS_SIGN[a.machine] * psi_k[j], gR_, gZ_, float(sc[j, jr_]), float(sc[j, jz_]), A_PUBLIC[a.machine], a.pin)
                stats["pin_try"] = stats.get("pin_try", 0) + 1
                if pinfo["ok"]:
                    psi_k[j] = AXIS_SIGN[a.machine] * new
                    stats["pin_ok"] = stats.get("pin_ok", 0) + 1
                    pin_d0.append(pinfo["d0"])
        psi_k = psi_k.astype(np.float32)
        psi = np.zeros((T, ny, nx), np.float32); psi[ks] = psi_k
        qf = np.full(T, np.nan); bf = np.full(T, np.nan); okf = np.zeros(T, bool)
        qf[ks], bf[ks], okf[ks] = q95, bN, good
        ks_all, good_all = ks, good
        if a.subset_ks:
            assert len(ks) == T, "--subset-ks needs an all-frames real set"
            ks = np.array(sorted(set(np.linspace(0, T - 1, min(a.subset_ks, T)).round().astype(int).tolist())))
            good = good_all[ks]
        stats["frames"] += T; stats["pred"] += len(ks); stats["no_ts"] += int((z["F"][:, 28] == 0).sum())
        info = [{"file": name, "idx": i, "k": int(k), "t": float(t[k]), "ip_kA": float(Ipf[k] / 1e3), "ok": bool(good[j]),
                 "err": "" if good[j] else "Ip<min", "lvl": 0} for j, k in enumerate(ks)]
        auxf = np.full((T, sc.shape[1]), np.nan, np.float32); auxf[ks_all] = sc
        tmp = out / "parts" / f"{a.tag}_{a.machine}_{i:05d}.tmp.npz"
        np.savez_compressed(tmp, psi=psi, q95=qf, betaN=bf, ok=okf, t=t, ip_A=Ipf * sgn, ks=ks, info=json.dumps(info), aux=auxf,
                            aux_names=np.array(M["names"]))
        os.replace(tmp, out / "parts" / f"{a.tag}_{a.machine}_{i:05d}.npz")
        aux_all.append(sc[good_all])
        if a.calib_fit and a.machine == "d3d":
            for key, arr, gt in (("q", q95, "gt_q95"), ("b", bN, "gt_betaN")):
                if gt in z:
                    fit[key].append(np.stack([arr[good_all], z[gt][good_all]], 1))
    if a.calib_fit:
        qq, bb = np.concatenate(fit["q"]), np.concatenate(fit["b"])
        qq = qq[np.isfinite(qq).all(1) & (qq[:, 1] > 1) & (qq[:, 1] < 20) & (qq[:, 0] > 0)]
        bb = bb[np.isfinite(bb).all(1) & (bb[:, 1] > 0.05) & (bb[:, 1] < 8) & (bb[:, 0] > 0)]
        r2 = lambda y, p: float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())  # noqa: E731
        cq, cb = float(np.median(qq[:, 1] / qq[:, 0])), float(np.median(bb[:, 1] / bb[:, 0]))
        cj = {"C_q95": cq, "C_betaN": cb, "source": f"{real} (DIII-D TRAIN labels; synth_net {a.model})", "n_q": int(len(qq)), "n_b": int(len(bb)),
              "r2_q95_raw": r2(qq[:, 1], qq[:, 0]), "r2_q95_cal": r2(qq[:, 1], cq * qq[:, 0]),
              "r2_betaN_raw": r2(bb[:, 1], bb[:, 0]), "r2_betaN_cal": r2(bb[:, 1], cb * bb[:, 0])}
        json.dump(cj, open(a.calib_fit, "w"), indent=1)
        print("CALIB " + json.dumps(cj), flush=True)
    aux = np.concatenate(aux_all) if aux_all else np.zeros((0, len(M["names"])))
    cb_ = stats.pop("clip_by_feature")
    top = np.argsort(-cb_)[:6]
    stats["clip_top"] = {int(j): int(cb_[j]) for j in top if cb_[j] > 0}
    stats["aux_p5_50_95"] = {n: [float(v) for v in np.percentile(aux[:, j], [5, 50, 95])] for j, n in enumerate(M["names"])} if len(aux) else {}
    if pin_d0:
        stats["pin_d0_p50_90_99"] = [float(v) for v in np.percentile(pin_d0, [50, 90, 99])]
    es, en = np.concatenate(ens_sd), np.concatenate(ens_nrm)
    stats["ens_sd_over_coef_sd_p50_90_99"] = [float(v) for v in np.percentile(es / max(float(np.std(en)), 1e-12), [50, 90, 99])]
    json.dump(stats, open(out / f"predstats_{a.tag}_{a.machine}.json", "w"), indent=1)
    print("PREDICT " + json.dumps({"tag": a.tag, "machine": a.machine, "n_shots": len(files), **stats}), flush=True)


if __name__ == "__main__":
    main()
