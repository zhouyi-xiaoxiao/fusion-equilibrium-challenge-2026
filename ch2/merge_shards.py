#!/usr/bin/env python3
"""Merge the shard outputs of ch2/predict_mast.sh into one run directory in global (sorted) shot order.

  merge_shards.py --run <run dir> --tag <tag> --machine mast --nshard N

Shard k was solved with --shot-list shards/list_k.txt into shards/s<k>/ (parts/<tag>_<machine>_<local index>.npz,
frames_<tag>_<machine>.parquet with the local index in column `idx`, files_<tag>_<machine>.txt). The merged run has
parts/<tag>_<machine>_<global index>.npz (hard link or copy, content unchanged), one frame table with global `idx`
(sorted by idx, k) and the sorted file list. Shots are independent, so the merged run equals a single-process run."""
import argparse, json, os, shutil
from pathlib import Path
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--machine", default="mast"); ap.add_argument("--nshard", type=int, required=True)
a = ap.parse_args()
run = Path(a.run); tm = f"{a.tag}_{a.machine}"
names = [x for x in (run / "shards/all.txt").read_text().split("\n") if x]
assert names == sorted(names)
gidx = {n: i for i, n in enumerate(names)}
(run / "parts").mkdir(exist_ok=True)
frames = []; seen = set(); cfg0 = None; core_h = 0.0
for k in range(a.nshard):
    s = run / f"shards/s{k}"
    loc = [x for x in (s / f"files_{tm}.txt").read_text().split("\n") if x]
    assert loc == [x for x in (run / f"shards/list_{k}.txt").read_text().split("\n") if x], f"shard {k}: file list differs from its shot list"
    cfg = json.load(open(s / f"cfg_{tm}.json")); cfg.pop("_shot_list", None)
    if cfg0 is None: cfg0 = cfg
    assert cfg == cfg0, f"shard {k}: configuration differs"
    core_h += json.load(open(s / f"summary_{tm}.json")).get("cpu_core_hours", 0.0)
    mp = {j: gidx[n] for j, n in enumerate(loc)}
    for j, g in mp.items():
        assert g not in seen; seen.add(g)
        src = s / "parts" / f"{tm}_{j:05d}.npz"; dst = run / "parts" / f"{tm}_{g:05d}.npz"
        if dst.exists(): dst.unlink()
        try: os.link(src, dst)
        except OSError: shutil.copyfile(src, dst)
    df = pd.read_parquet(s / f"frames_{tm}.parquet")
    df["idx"] = df["idx"].map(mp).astype(df["idx"].dtype)
    frames.append(df)
assert seen == set(range(len(names))), f"merged {len(seen)} of {len(names)} shots"
df = pd.concat(frames, ignore_index=True)
df = df.sort_values(["idx", "k"], kind="stable").reset_index(drop=True) if "k" in df else df.sort_values("idx", kind="stable").reset_index(drop=True)
df.to_parquet(run / f"frames_{tm}.parquet")
(run / f"files_{tm}.txt").write_text("\n".join(names) + "\n")
cfg0["_shot_list"] = None
json.dump(cfg0, open(run / f"cfg_{tm}.json", "w"), indent=1)
json.dump({"tag": a.tag, "machine": a.machine, "n_shots": len(names), "n_frames": int(len(df)), "nshard": a.nshard, "cpu_core_hours": core_h}, open(run / f"summary_{tm}.json", "w"), indent=1)
print(f"MERGE_OK {len(names)} shots, {len(df)} frames, {a.nshard} shards, {core_h:.1f} core-h")
