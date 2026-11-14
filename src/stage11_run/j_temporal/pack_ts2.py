#!/usr/bin/env python3
"""
Stage 11J / Ch2 temporal — package a temporally regularised MAST candidate from an EXISTING validated submission file
(generalises f_mast_shape/pack_ts.py).

Every key of the source npz is kept in its exact order. shot_XXXX_psirz is replaced by the map of the same shot from a
temporal-module run (tsmooth.py / resolve_ts.py parts; float32 with the mantissa truncated to --psi-bits bits, the same
binary truncation as pack_inv6.py). shot_XXXX_q95 / shot_XXXX_betaN are copied BYTE-IDENTICALLY unless --q-spec / --b-spec
(JSON of a tsmooth scalar spec, e.g. {"kind":"whit1","s":2,"log":1}) is given: then the source series itself is smoothed along
the shot with the frame classes of the run (fallback / filled / low-Ip frames are missing observations) and clipped to the
packager's ranges. Asserts: same key order and shapes, the source psi equals the UNSMOOTHED base run's psi of that shot (so the
parts belong to the source file), finite float32. Label-free: rule3_guard installed; reads only our own files.
Usage: pack_ts2.py --src <validated npz> --base-run DIR --base-tag T --run <module dir> --tag T_variant --out <npz> [--psi-bits 24]
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "ch2_geometry"))
import rule3_guard
rule3_guard.install()
sys.path.insert(0, str(HERE))
import tsmooth as TS  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--base-run", required=True); ap.add_argument("--base-tag", required=True)
ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--psi-bits", type=int, default=24); ap.add_argument("--what", default="")
ap.add_argument("--q-spec", default=None); ap.add_argument("--b-spec", default=None); ap.add_argument("--miss", default="g1")
a = ap.parse_args()
qs = json.loads(a.q_spec) if a.q_spec else None
bs = json.loads(a.b_spec) if a.b_spec else None


def trunc(psi, bits):
    if not bits:
        return psi.astype(np.float32)
    m, e = np.frexp(psi.astype(np.float64))
    return np.ldexp(np.round(m * 2.0 ** bits) / 2.0 ** bits, e).astype(np.float32)


def smooth_scalar(v, t, w, sp, lo, hi):
    v = v.astype(float)
    wv = w * (np.isfinite(v) & (v > lo) & (v < hi))
    if wv.sum() < 3:
        return v.astype(np.float32)
    y = np.where(wv > 0, np.log(np.maximum(v, 1e-9)) if sp.get("log") else v, 0.0)
    dts = np.diff(t); dt = float(np.median(dts)) if len(dts) else 1.0
    ys, _, _, sup = TS.smooth_series(t, y, wv, sp, dt)
    ys = np.exp(ys) if sp.get("log") else ys
    out = np.where(sup, ys, v)
    return np.clip(out, lo, hi).astype(np.float32)


src = np.load(a.src)
keys = list(src.files)
n = len([k for k in keys if k.endswith("_psirz")])
sub, st = {}, {"n_shots": n, "frames": 0, "src_psi_matches_base": 0, "max_rel_change": 0.0, "sum2": 0.0, "npx": 0, "q_changed": 0, "b_changed": 0}
t0 = time.time()
cache = {}
for k in keys:
    i = int(k.split("_")[1])
    if i not in cache:
        cache.clear()
        cache[i] = np.load(Path(a.run) / "parts" / f"{a.tag}_mast_{i:05d}.npz", allow_pickle=False)
    part = cache[i]
    if k.endswith("_psirz"):
        p0 = src[k]
        base = np.load(Path(a.base_run) / "parts" / f"{a.base_tag}_mast_{i:05d}.npz", allow_pickle=False)["psi"]
        assert base.shape == p0.shape, (k, base.shape, p0.shape)
        if np.array_equal(trunc(base, a.psi_bits), p0):
            st["src_psi_matches_base"] += 1
        new = part["psi"]
        assert new.shape == p0.shape and np.isfinite(new).all(), k
        p1 = trunc(new, a.psi_bits)
        assert p1.dtype == np.float32 and np.isfinite(p1).all()
        d = p1.astype(np.float64) - p0.astype(np.float64)
        st["sum2"] += float((d ** 2).sum()); st["npx"] += d.size; st["frames"] += p0.shape[0]
        st["max_rel_change"] = max(st["max_rel_change"], float(np.abs(d).max() / max(float(np.ptp(p0)), 1e-12)))
        sub[k] = p1
        continue
    v0 = src[k]
    sp, lo, hi, cnt = (qs, 1.2, 25.0, "q_changed") if k.endswith("_q95") else (bs, 0.0, 8.0, "b_changed")
    if sp is None:
        sub[k] = v0                                       # identical array
        continue
    info = json.loads(str(part["info"]))
    T = len(v0)
    _, w = TS.classes(info, T, "mast", a.miss)
    v1 = smooth_scalar(v0, part["t"].astype(float), w, sp, lo, hi)
    assert v1.shape == v0.shape and np.isfinite(v1).all() and v1.dtype == np.float32
    st[cnt] += int((v1 != v0).sum())
    sub[k] = v1
assert list(sub.keys()) == keys
assert st["src_psi_matches_base"] == n, f"source psi differs from the base run on {n - st['src_psi_matches_base']} shots"
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(out, **sub)
st["rms_change"] = float(np.sqrt(st.pop("sum2") / max(st.pop("npx"), 1)))
man = {"out": str(out), "md5": hashlib.md5(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size, "source_npz": a.src,
       "source_md5": hashlib.md5(Path(a.src).read_bytes()).hexdigest(), "base_run": a.base_run, "base_tag": a.base_tag,
       "module_run": a.run, "module_tag": a.tag, "what": a.what, "psi_bits": a.psi_bits, "q_spec": qs, "b_spec": bs, **st, "secs": time.time() - t0,
       "rule3": "MAST: own parts + own validated submission file only; module constants from DIII-D labels / label-free statistics; no MAST label, no envelope, no demo"}
json.dump(man, open(out.parent / "manifest_ts.json", "w"), indent=1)
print("PACKTS2 " + json.dumps(man))
