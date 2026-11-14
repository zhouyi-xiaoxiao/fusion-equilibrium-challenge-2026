#!/usr/bin/env python3
"""Run the organisers' validator (src/starter/validate_submission.py, unmodified) against any configuration of any challenge
dataset, e.g. the Final configurations, which the pinned validator does not list.

  validate_final.py <file.npz> --repo Sophelio/fusion-equilibrium-challenge-final --config mast_private_test [--split S] [--max-shots 0]

The validator module is imported as it is; only its dataset id (REPO_ID) and its table of known configurations (CONFIG_INFO) are
extended in memory: config -> (split, machine), machine from the config name (diii_d_* -> DIII-D, mast_* -> MAST), split from the
Hub (the only split of the configuration) unless --split is given. A token is only needed for a gated dataset (HF_TOKEN, never printed)."""
import argparse, importlib.util, os, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", type=Path); ap.add_argument("--repo", required=True); ap.add_argument("--config", required=True)
    ap.add_argument("--split", default=None); ap.add_argument("--max-shots", type=int, default=0)
    a = ap.parse_args()
    ws = Path(os.environ.get("WS") or Path(__file__).resolve().parent.parent / "src")
    spec = importlib.util.spec_from_file_location("validate_submission", ws / "starter" / "validate_submission.py")
    vs = importlib.util.module_from_spec(spec); spec.loader.exec_module(vs)
    machine = "DIII-D" if a.config.startswith("diii_d") else "MAST" if a.config.startswith("mast") else None
    assert machine, f"cannot tell the machine from config {a.config}"
    split = a.split
    if split is None and a.repo == vs.REPO_ID and a.config in vs.CONFIG_INFO:
        split = vs.CONFIG_INFO[a.config][0]                 # a configuration the pinned validator knows: unchanged
    if split is None:
        from datasets import get_dataset_split_names
        names = get_dataset_split_names(a.repo, a.config)
        assert len(names) == 1, f"give --split, one of {names}"
        split = names[0]
    vs.REPO_ID = a.repo
    vs.CONFIG_INFO[a.config] = (split, machine)
    print(f"validator: repo {a.repo} config {a.config} split {split} machine {machine}", flush=True)
    return vs.validate(a.npz, a.config, a.max_shots)


if __name__ == "__main__":
    sys.exit(main())
