#!/usr/bin/env python3
"""
Stage 11 / Ch2 — rule-3 process guard for every MAST-producing pipeline (G1, G2, packaging).

    import rule3_guard; rule3_guard.install()        # first line of every MAST entrypoint / worker

After install() the interpreter REFUSES (RuntimeError) to open
  * fusion_scoring/masks/mast_envelope.npz           (scoring-derivation only; never input/limiter/prior/mask)
  * anything under a mast_demo directory or a file of the three labelled demo shots (28348/28350/28351)
  * FAIR-MAST equilibrium stores (level-1 efm/*, level-2 equilibrium/*)
and `mast_columns()` rejects any parquet column that is not a permitted MAST input.

Implemented with sys.addaudithook ("open" events) so it also covers numpy/pandas/pyarrow file opens
that go through Python's open(); pyarrow's native reader is covered by `read_mast_parquet` below,
which is the only sanctioned way to read a MAST shot file. Audit hooks cannot be removed again.
The D3D side of a pipeline (labels allowed) must run in a separate process without install().
"""
from __future__ import annotations

import os
import re
import sys

DEMO_IDS = ("28348", "28350", "28351")
_BANNED = re.compile(r"(mast_envelope\.npz|[/\\]mast_demo[/\\]|[/\\]efm[/\\]|\.zarr[/\\]equilibrium|"
                     + "|".join(f"shot_?0*{d}(?![0-9a-f])" for d in DEMO_IDS) + r")", re.I)
# Permitted MAST columns: inputs + frame time base + grid/coil descriptions + bookkeeping.
_OK_COL = re.compile(r"^(magnetics_.*|thomson_.*|coil_.*|efit_times|efit_grid_R|efit_grid_Z|shot_id|shot|source|machine)$")
_BAD_COL = re.compile(r"^(efit_(?!times$|grid_R$|grid_Z$).*|magnetics_dsep.*)$")
_installed = False


def banned_path(p) -> bool:
    try:
        return bool(_BANNED.search(os.fspath(p)))
    except TypeError:
        return False


def _hook(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        p = args[0].decode(errors="ignore") if isinstance(args[0], bytes) else os.fspath(args[0])
        if _BANNED.search(p):
            raise RuntimeError(f"rule-3 guard: refusing to open banned MAST artefact: {p}")


def install():
    global _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True


def mast_columns(cols):
    """Validate a MAST column list; returns it unchanged or raises."""
    bad = [c for c in cols if _BAD_COL.match(c) or not _OK_COL.match(c)]
    if bad:
        raise RuntimeError(f"rule-3 guard: non-input MAST columns requested: {bad}")
    return list(cols)


def read_mast_parquet(path, columns):
    """The only sanctioned MAST shot reader: banned-path check + column allow-list + label assertion."""
    import pandas as pd
    if banned_path(path):
        raise RuntimeError(f"rule-3 guard: banned MAST file: {path}")
    df = pd.read_parquet(path, columns=mast_columns(columns))
    assert not any(_BAD_COL.match(c) for c in df.columns)
    return df


if __name__ == "__main__":                     # self-test (no files needed)
    install()
    for p in ("/x/starter/fusion_scoring/masks/mast_envelope.npz", "/x/stage2_run/data/mast_demo/a.parquet",
              "/x/mast_shot_28350.parquet", "/x/level1/30420.zarr/efm/psirz"):
        try:
            open(p, "rb")
            raise SystemExit(f"FAIL: opened {p}")
        except RuntimeError:
            pass
        except OSError:
            raise SystemExit(f"FAIL: guard did not fire for {p}")
    try:
        open("/x/starter/fusion_scoring/masks/d3d_envelope.npz", "rb")
    except FileNotFoundError:
        pass
    assert mast_columns(["efit_times", "efit_grid_R", "magnetics_p4l_current", "thomson_core_Te", "coil_R"])
    for bad in (["efit_psirz"], ["efit_q95"], ["magnetics_dsep"], ["something_else"]):
        try:
            mast_columns(bad); raise SystemExit(f"FAIL: accepted {bad}")
        except RuntimeError:
            pass
    print("rule3_guard self-test OK")
