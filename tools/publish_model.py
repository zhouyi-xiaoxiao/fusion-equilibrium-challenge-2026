#!/usr/bin/env python3
"""Make a publishable copy of a model file that a run trained itself (e.g. the network of a syn_* MAST profile).

  publish_model.py <model.pt> <out.pt>

The trainers record their command line in the checkpoint (`diag/args`), which contains absolute paths of the work directory. This
tool replaces every string that contains a path separator by its last path component, leaves every tensor / array and every other
value untouched, verifies that all arrays of the copy are bit-identical to the original, and prints both sha256. The predict path
reads only the arrays, so `SYN_MODEL_FILE=<out.pt>` reproduces the run's predictions bit for bit."""
import hashlib
import sys

import numpy as np
import torch


def scrub(o, n):
    if isinstance(o, str) and "/" in o:
        n[0] += 1
        return o.rstrip("/").rsplit("/", 1)[-1]
    if isinstance(o, dict):
        return {k: scrub(v, n) for k, v in o.items()}
    if isinstance(o, list):
        return [scrub(v, n) for v in o]
    if isinstance(o, tuple):
        return tuple(scrub(v, n) for v in o)
    return o


def arrays(o, pre=""):
    if isinstance(o, torch.Tensor):
        yield pre, o.detach().cpu().numpy()
    elif isinstance(o, np.ndarray):
        yield pre, o
    elif isinstance(o, dict):
        for k in sorted(o, key=str):
            yield from arrays(o[k], f"{pre}/{k}")
    elif isinstance(o, (list, tuple)):
        for i, v in enumerate(o):
            yield from arrays(v, f"{pre}/{i}")


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


src, dst = sys.argv[1:3]
a = torch.load(src, map_location="cpu", weights_only=False)
n = [0]
b = scrub(a, n)
torch.save(b, dst)
c = torch.load(dst, map_location="cpu", weights_only=False)
x, y = dict(arrays(a)), dict(arrays(c))
assert set(x) == set(y) and all(x[k].shape == y[k].shape and x[k].dtype == y[k].dtype and np.array_equal(x[k], y[k], equal_nan=(x[k].dtype.kind == "f")) for k in x), "arrays changed"
print(f"PUBLISH_MODEL {len(x)} arrays bit-identical; {n[0]} path string(s) shortened; sha256 original {sha(src)} published {sha(dst)}")
