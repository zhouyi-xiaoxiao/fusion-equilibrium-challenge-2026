#!/usr/bin/env python3
"""Compact diagnostics of an inverse run (D3D: label diagnostics; MAST: label-free). Usage: inv_summary.py <out> <tag>"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
out, tag = Path(sys.argv[1]), sys.argv[2]
def pct(x, q=(5, 50, 95), f=3):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return ("[" + " ".join(f"{v:.{f}f}" for v in np.percentile(x, q)) + f"] n={len(x)}") if len(x) else "[]"
def col(d, c):
    return d[c] if c in d else pd.Series(np.nan, index=d.index)
for m in ("d3d", "mast"):
    p = out / f"frames_{tag}_{m}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    if "fatal" in d and d["fatal"].notna().any():
        print(f"[{m}] FATAL x{int(d['fatal'].notna().sum())}:", d.loc[d['fatal'].notna(), 'tb'].iloc[0][-700:])
        d = d[d["fatal"].isna()]
    pl = d[d["err"] != "Ip<min"]; ok = pl[pl["ok"].astype(bool)]
    if not len(ok):
        print(f"[{m}] {tag}: no converged frames; errs {pl['err'].value_counts().head(5).to_dict()}"); continue
    print(f"[{m}] {tag}: frames {len(d)} plasma {len(pl)} conv {len(ok)/max(len(pl),1):.4f} pass {ok['pass'].value_counts().to_dict()} x_src {ok['x_src'].value_counts().to_dict()} "
          f"closure {col(ok,'closure').value_counts().to_dict()} kind {ok['gs_kind'].value_counts().to_dict()} fill {col(d,'fill').value_counts().to_dict()}")
    print(f"[{m}]   errs {pl[~pl['ok'].astype(bool)]['err'].astype(str).str.slice(0,60).value_counts().head(4).to_dict()}")
    print(f"[{m}]   solves {pct(ok['n_solves'],f=0)} lm_it {pct(ok['lm_it'],f=0)} rej {pct(ok['n_rej'],f=0)} frozen {pct(col(ok,'n_frozen'),f=0)} secs {pct(ok['secs'],f=2)} shot_secs {pct(ok.groupby('idx')['shot_secs'].first(),f=0)}")
    print(f"[{m}]   cp {pct(ok['cp'])} b1 {pct(ok['b1'])} b2 {pct(ok['b2'])} dr {pct(col(ok,'dr'))} gev {pct(ok['gev'])} f_press {pct(ok['f_press'])} zt {pct(ok['zt'])}")
    print(f"[{m}]   cost {pct(ok['cost0'],f=2)} -> {pct(ok['cost'],f=2)}; sv0 {pct(col(ok,'sv0'),f=2)} sv1 {pct(col(ok,'sv1'),f=2)} sv2 {pct(col(ok,'sv2'),f=2)} sv3 {pct(col(ok,'sv3'),f=2)}")
    print(f"[{m}]   iso_mae_cm {pct(col(ok,'iso_mae_cm'),f=2)} val_mae_cm {pct(col(ok,'val_mae_cm'),f=2)} d_out {pct(col(ok,'d_out'))} d_in {pct(col(ok,'d_in'))}")
    print(f"[{m}]   Rax {pct(ok['gs_Rax'])} Zax {pct(ok['gs_Zax'])} kappa {pct(col(ok,'gs_kappa'))} q95 {pct(col(ok,'gs_q95'))} betaN {pct(col(ok,'gs_betaN'))} li {pct(col(ok,'gs_li'))} betap {pct(col(ok,'gs_betap'))}")
    if m == "mast" and "gs_edge_gs_in" in ok:
        ei = (ok["gs_edge_gs_in"] - ok["ts_rin"])[~ok["ts_cin"].astype(bool)]
        eo = (ok["gs_edge_gs"] - ok["ts_edge"])[~ok["ts_cens"].astype(bool)]
        print(f"[{m}]   LCFS-TS on the laser chord: outer {pct(eo)} MAE {np.nanmean(np.abs(eo)):.4f}; inner(uncens) {pct(ei)} MAE {np.nanmean(np.abs(ei)):.4f}; Rax-ts_sym {pct(ok['gs_Rax']-ok['ts_rsym'])}")
    if "r2_frame" in d:
        print(f"[{m}]   r2_frame all {pct(pl['r2_frame'],(5,25,50))} mean {np.nanmean(pl['r2_frame']):.4f}; ok {pct(ok['r2_frame'],(5,25,50))}")
        print(f"[{m}]   Rax err {pct(ok['gs_Rax']-ok['gt_raxis'])} MAE {np.nanmean(np.abs(ok['gs_Rax']-ok['gt_raxis'])):.4f}; Zax err MAE {np.nanmean(np.abs(ok['gs_Zax']-ok['gt_zaxis'])):.4f}")
        for a, b in (("gs_q95", "gt_q95"), ("gs_betaN", "gt_betaN"), ("gs_li", "gt_li")):
            if a in ok and b in ok:
                v = ok[[a, b]].replace([np.inf, -np.inf], np.nan).dropna()
                cc = np.corrcoef(v[a], v[b])[0, 1] if len(v) > 5 else np.nan
                r2 = 1 - ((v[b] - v[a] * np.median(v[b] / v[a])) ** 2).sum() / ((v[b] - v[b].mean()) ** 2).sum() if len(v) > 5 else np.nan
                print(f"[{m}]   {a}/{b} ratio {pct(v[a]/v[b])} corr {cc:.3f} R2(after median scale) {r2:.3f}")
        if "efit_rout" in ok:
            print(f"[{m}]   pseudo edges: d_out {pct(col(ok,'d_out'))}")
