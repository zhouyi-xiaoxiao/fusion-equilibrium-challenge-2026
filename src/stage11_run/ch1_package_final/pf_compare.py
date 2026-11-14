#!/usr/bin/env python3
"""ch1_package_final — reproduction check of a predict.sh run on the 874 public-test shots against the stored intermediates of the
shipped FcCX chain (label-free; every stage: bit-equal? max abs difference, frames differing).
  pf_compare.py RUN_DIR OUT.json [REF_NAME=REF.npz ...]     (REF = final files to compare the run's submission.npz with)"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

SCR = Path(os.environ["SCR"])
S = SCR / "stage11_ch1"; CX = S / "e_cx"; E = S / "e_fold2_offset"; F = S / "f_corr_canon"
RUN = Path(sys.argv[1]); out = {}
T = "CXFMTWqsc_kb"


def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def arr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"shape": [list(a.shape), list(b.shape)], "bit_equal": False}
    eq = bool(np.array_equal(a, b, equal_nan=True))
    r = {"bit_equal": eq, "dtype": [str(a.dtype), str(b.dtype)]}
    if not eq:
        d = np.abs(a.astype(np.float64) - b.astype(np.float64))
        m = np.isfinite(d)
        r.update(max_abs_diff=float(d[m].max()) if m.any() else None, n_diff=int((~(a == b) & ~(np.isnan(a) & np.isnan(b))).sum()) if a.dtype.kind == "f" else int((a != b).sum()),
                 n=int(a.size), nan_pattern_equal=bool(np.array_equal(np.isnan(a), np.isnan(b))) if a.dtype.kind == "f" else None,
                 max_rel_to_sd=float(d[m].max() / (np.nanstd(b.astype(np.float64)) + 1e-300)) if m.any() else None)
    return r


def npz(pa, pb, keys=("psirz", "q95", "betaN")):
    A, B = np.load(pa), np.load(pb)
    n = len([k for k in B.files if k.endswith("_psirz")])
    r = {"keys_equal": sorted(A.files) == sorted(B.files), "shots": n}
    for k in keys:
        mx, nd, nf, fr = 0.0, 0, 0, []
        for s in range(n):
            kk = f"shot_{s:04d}_{k}"
            if kk not in A.files or kk not in B.files:
                continue
            x, y = A[kk], B[kk]
            if x.shape != y.shape:
                r[k] = {"shape_mismatch_shot": s}; break
            ne = (x != y).reshape(len(x), -1).any(1)
            if ne.any():
                mx = max(mx, float(np.max(np.abs(x.astype(np.float64) - y.astype(np.float64)))))
                nd += int(ne.sum()); fr += [(s, int(j)) for j in np.flatnonzero(ne)[:3]]
            nf += len(x)
        else:
            r[k] = {"bit_equal": nd == 0, "frames_differing": nd, "frames": nf, "max_abs_diff": mx, "first": fr[:12]}
    return r


def step(name, fn):
    try:
        out[name] = fn()
    except Exception as e:  # noqa: BLE001
        out[name] = {"error": f"{type(e).__name__}: {e}"}
    print(f"[cmp] {name}: {json.dumps(out[name])[:900]}", flush=True)


ft = RUN / "feat"
step("files", lambda: {"eq_feat_v21_test": (ft / "files.txt").read_text() == (S / "feat_v21/test/files.txt").read_text(),
                        "offsets": arr(np.load(ft / "offsets.npy"), np.load(S / "feat_v21/test/offsets.npy")), "times": arr(np.load(ft / "times.npy"), np.load(S / "feat_v21/test/times.npy"))})
step("X_raw_v21", lambda: arr(np.load(ft / "raw21/X_raw.npy"), np.load(S / "feat_v21/test/X_raw.npy")))
step("X_raw_v22", lambda: arr(np.load(ft / "raw22/X_raw.npy"), np.load(S / "feat_v22/test/X_raw.npy")))
step("X_base_canonical_v22", lambda: arr(np.load(RUN / "feat_base/X.npy"), np.load(CX / "h150/feat/test/X.npy")))
step("X_heads_canonical_v21", lambda: arr(np.load(RUN / "ccfeat/pkg/X.npy"), np.load(E / "h150_cfl/feat/test/X.npy")))
for sfx in ("", "_cpu"):
    d = RUN / f"fm_raw{sfx}"
    if d.exists():
        def fm(d=d):
            r = {}
            for f in sorted(d.glob("coef_*.npy")):
                m = f.stem[5:]
                r[m] = arr(np.load(f), np.load(CX / "full/frame_mlp/members" / m / "coef_test.npy"))
            return {"members": len(r), "all_bit_equal": all(v["bit_equal"] for v in r.values()), "max_abs_diff": max([v.get("max_abs_diff") or 0.0 for v in r.values()]),
                    "max_rel_to_sd": max([v.get("max_rel_to_sd") or 0.0 for v in r.values()])}
        step(f"FM_member_coefs{sfx}", fm)
    p = RUN / f"tw{sfx}.npz"
    if p.exists():
        def tw(p=p):
            ms = [np.load(CX / f"full/temporal/full/members/test_tR_w9d_s{s}.npz")["coef"].astype(np.float64) for s in range(4)]
            Cb = np.mean(ms, 0)
            pc = np.load(SCR / "stage6_H/pca_80.npz")
            ref = (Cb @ pc["comps_"].astype(np.float64) + pc["mean_"].astype(np.float64)[None]).astype(np.float32)
            z = np.load(p); off = np.load(ft / "offsets.npy")
            mine = np.concatenate([z[f"shot_{s:04d}_psirz"].reshape(-1, 4225) for s in range(len(off) - 1)])
            r = arr(mine, ref)
            bag = np.load(RUN / f"tw_bag4{sfx}.npz")["coef"]
            r["bag_coef_vs_mean_of_training_time_members"] = arr(bag, Cb.astype(np.float32))
            return r
        step(f"TW_psi{sfx}", tw)


def sc():
    z = np.load(RUN / "sc.npz"); r = {}
    for k in [k[4:] for k in z.files if k.startswith("q95_")]:
        m = np.load(CX / f"full/scalars/full/{k}.npz")
        r[k] = {"q95": arr(z[f"q95_{k}"], m["q95"]), "betaN": arr(z[f"betaN_{k}"], m["betaN"])}
    return {"members": len(r), "all_bit_equal": all(v["q95"]["bit_equal"] and v["betaN"]["bit_equal"] for v in r.values()),
            "max_abs_diff": max([max(v["q95"].get("max_abs_diff") or 0.0, v["betaN"].get("max_abs_diff") or 0.0) for v in r.values()])}


step("SC_members", sc)
step("blend", lambda: npz(RUN / "blend.npz", CX / "full/test/preds_test_CX_FMTW_qsc.npz"))
cc = RUN / "cc"; fc = F / "full/corrector"
step("head_predictions", lambda: arr(np.load(cc / "hp_direct2_mlp_pkg.npy"), np.load(fc / "hp_direct2_mlp_test.npy")))


def jac():
    a, b = np.load(cc / "jac_pkg_pk.npz"), np.load(CX / f"full/corrector/jac_test_{T}.npz")
    return {k: arr(a[k], b[k]) for k in b.files if k in a.files}


step("jacobian", jac)
step("base_derivation_f11", lambda: arr(np.load(cc / "f11_pkg_pk.npy"), np.load(CX / f"full/corrector/f11_test_{T}.npy")))
step("rob_targets", lambda: arr(np.load(cc / "tgt_pkg_pk_rob.npy"), np.load(fc / f"tgt_test_{T}_rob.npy")))
step("incumbent_GN", lambda: npz(cc / "preds_pkg_pk_rob_r1e+09_e1_i13.npz", fc / f"preds_test_{T}_rob_r1e+09_e1_i13.npz", ("psirz",)))
step("robmi", lambda: npz(cc / "preds_pkg_pk_robmi_r1e+09_e1_i13.npz", fc / f"preds_test_{T}_robmi_r1e+09_e1_i13.npz", ("psirz",)))
sub = RUN / "submission.npz"
out["submission"] = {"path": str(sub), "md5": md5(sub), "bytes": os.path.getsize(sub)}
for kv in sys.argv[3:]:
    nm, ref = kv.split("=", 1)
    if os.path.exists(ref):
        step(f"final_vs_{nm}", lambda ref=ref: dict(npz(sub, ref), ref_md5=md5(ref), md5_equal=md5(ref) == out["submission"]["md5"]))
json.dump(out, open(sys.argv[2], "w"), indent=1)
print("PF_COMPARE_DONE", json.dumps(out["submission"]))
