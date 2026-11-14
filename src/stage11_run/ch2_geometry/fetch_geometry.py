#!/usr/bin/env python3
"""
Stage 11 / Ch2 geometry: fetch PUBLIC MAST machine-description geometry from the FAIR-MAST
level-2 store (anonymous HTTPS, CC BY-SA 4.0) and save it with provenance.

RULE-3 COMPLIANCE (read before editing):
  * Reads ONLY machine-description geometry arrays:
      wall/limiter_{r,z,geometry_channel}                (EFIT machine-description limiter polygon,
                                                           shotRange 0..40000 per its own attributes)
      pf_passive/<structure>_{r,z,width,height,shapeAngle1,shapeAngle2,geometry_channel}
                                                          (passive-conductor rectangles; NEVER *_current)
      pf_active/<coil>_{r,z,width,height,coordinate_element}  (cross-check vs the shipped coil_* columns)
      the zarr attributes of wall/limiter_r and of the groups (provenance text only)
  * NEVER opens equilibrium/*, magnetics/*, summary/*, thomson_scattering/*, pf_passive/*_current,
    tf/coil_current or any level-1 efm/* array (equilibrium data of any source is banned for MAST).
    An allow-list check below enforces this.
  * Shots used: a fixed list of reference shots (REFERENCE_SHOTS below); never the three labelled
    demo shots (28348/28350/28351). The geometry is shot-independent.

Usage (login node is fine, a few KB over HTTPS):
  PYTHONPATH=$O/pylib python fetch_geometry.py --out <out dir> --shots 20000 22000 29500
where $O/pylib holds zarr>=3 (stage8_O_scout_mast_archive/pylib).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np

S3 = "https://s3.echo.stfc.ac.uk"
L2 = S3 + "/mast/level2/shots/{shot}.zarr"
DEMO = {28348, 28350, 28351}
REFERENCE_SHOTS = {15000, 20000, 22000, 29500}

ALLOWED = re.compile(
    r"^(wall/limiter_(r|z|geometry_channel)"
    r"|pf_passive/[a-z0-9_]+_(r|z|width|height|shapeAngle1|shapeAngle2|geometry_channel)"
    r"|pf_active/[a-z0-9_]+_(r|z|width|height|coordinate_element))$")
FORBIDDEN = re.compile(r"(^equilibrium|^magnetics|^summary|^thomson|_current$|^efm|psi|q95|beta)")


def allowed(name: str) -> bool:
    return bool(ALLOWED.match(name)) and not FORBIDDEN.search(name)


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def to_py(v):
    if isinstance(v, np.ndarray):
        if v.dtype.kind in "OUST":
            return [str(x) for x in v.tolist()]
        return v.tolist()
    return v


def fetch_shot(shot: int) -> dict:
    import zarr
    assert shot not in DEMO, "demo shots are banned"
    assert shot in REFERENCE_SHOTS, "geometry is read only from the fixed reference shots"
    url = L2.format(shot=shot)
    root = zarr.open_group(url, mode="r", storage_options={"anon": True}) if url.startswith("s3") else \
        zarr.open_group(url, mode="r")
    out = {"shot": shot, "url": url, "arrays": {}, "attrs": {}}
    names = []
    for grp in ("wall", "pf_passive", "pf_active"):
        g = root[grp]
        out["attrs"][grp] = {k: v for k, v in dict(g.attrs).items() if isinstance(v, (str, int, float))}
        for k in g.array_keys():
            n = f"{grp}/{k}"
            if allowed(n):
                names.append(n)
    for n in sorted(names):
        assert allowed(n), n
        a = np.asarray(root[n][...])
        out["arrays"][n] = a
    out["attrs"]["wall/limiter_r"] = {k: v for k, v in dict(root["wall/limiter_r"].attrs).items()
                                      if isinstance(v, (str, int, float))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--shots", type=int, nargs="+", default=[20000, 22000, 29500])
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res = []
    for s in args.shots:
        t0 = time.time()
        try:
            r = fetch_shot(s)
            r["secs"] = round(time.time() - t0, 1)
            res.append(r)
            print(f"shot {s}: {len(r['arrays'])} geometry arrays in {r['secs']} s", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"shot {s}: FAILED {type(e).__name__}: {e}", flush=True)
    if not res:
        raise SystemExit("no shot fetched")
    ref = res[0]
    # cross-shot identity check (geometry must be shot-independent)
    ident = {}
    for n, a in ref["arrays"].items():
        same = []
        for r in res[1:]:
            b = r["arrays"].get(n)
            same.append(bool(b is not None and b.shape == a.shape and
                             ([str(x) for x in a.tolist()] == [str(x) for x in b.tolist()] if a.dtype.kind in "OUST" else np.allclose(a, b, atol=1e-9))))
        ident[n] = all(same) if same else None
    lim_r = np.asarray(ref["arrays"]["wall/limiter_r"], np.float64)
    lim_z = np.asarray(ref["arrays"]["wall/limiter_z"], np.float64)
    passive = {}
    for n, a in ref["arrays"].items():
        if n.startswith("pf_passive/"):
            m = re.match(r"pf_passive/([a-z0-9_]+?)_(r|z|width|height|shapeAngle1|shapeAngle2|geometry_channel)$", n)
            if m:
                passive.setdefault(m.group(1), {})[m.group(2)] = to_py(a)
    active = {}
    for n, a in ref["arrays"].items():
        if n.startswith("pf_active/"):
            m = re.match(r"pf_active/([a-z0-9_]+?)_(r|z|width|height|coordinate_element)$", n)
            if m:
                active.setdefault(m.group(1), {})[m.group(2)] = to_py(a)
    prov = {
        "source": "FAIR-MAST level-2 (UKAEA/STFC), anonymous HTTPS, licence CC BY-SA 4.0",
        "store_url_template": L2,
        "shots_read": [r["shot"] for r in res],
        "rule3_note": ("machine-description geometry only; no equilibrium/magnetics/summary/thomson/"
                       "current arrays opened; demo shots excluded; fixed reference shots only"),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "limiter_attrs": ref["attrs"].get("wall/limiter_r", {}),
        "group_attrs": ref["attrs"],
        "identical_across_shots": ident,
        "sha256_16": {n: (hashlib.sha256("|".join(str(x) for x in np.asarray(a).tolist()).encode()).hexdigest()[:16]
                               if np.asarray(a).dtype.kind in "OUST" else sha(np.asarray(a, np.float64)))
                      for n, a in ref["arrays"].items()},
    }
    (out / "mast_limiter.json").write_text(json.dumps({
        "R": lim_r.tolist(), "Z": lim_z.tolist(),
        "channel": to_py(ref["arrays"].get("wall/limiter_geometry_channel", np.array([]))),
        "n_points": int(len(lim_r)), "provenance": prov}, indent=1, default=str))
    (out / "mast_passive.json").write_text(json.dumps({"structures": passive, "provenance": prov},
                                                       indent=1, default=str))
    (out / "mast_pf_active_l2.json").write_text(json.dumps({"coils": active, "provenance": prov},
                                                            indent=1, default=str))
    np.savez(out / "mast_geometry.npz", limiter_R=lim_r, limiter_Z=lim_z)
    print(json.dumps({"limiter_n": len(lim_r), "R_min": lim_r.min(), "R_max": lim_r.max(),
                      "Z_min": lim_z.min(), "Z_max": lim_z.max(),
                      "n_passive_structures": len(passive), "n_active_groups": len(active),
                      "all_identical": all(v for v in ident.values() if v is not None)}, default=float))


if __name__ == "__main__":
    main()
