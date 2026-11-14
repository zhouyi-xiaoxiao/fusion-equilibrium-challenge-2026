#!/usr/bin/env python3
"""Vertical shift of flux maps on the uniform Z grid (numpy; Catmull-Rom along Z, linear extrapolation beyond the grid).
zshift(Y, s): Y (n, nR, nZ) in [R, Z] layout, s (n,) shift in CELLS; returns Y evaluated at Z index j + s, i.e. the map moved DOWN by s
cells (a feature at index j0 appears at j0 - s). Used for the Z-canonical representation of the plasma flux: canonical = zshift(res, +Zax/dZ)
(axis at Z = 0), back-transform res = zshift(canonical, -Zax/dZ)."""
import numpy as np

PAD = 6


def zshift(Y, s, chunk=20000):
    Y = np.asarray(Y)
    n, nR, nZ = Y.shape
    out = np.empty_like(Y)
    k = np.arange(1, PAD + 1, dtype=Y.dtype)
    for a in range(0, n, chunk):
        y = Y[a:a + chunk]
        lo = y[:, :, :1] - k[::-1][None, None, :] * (y[:, :, 1:2] - y[:, :, :1])
        hi = y[:, :, -1:] + k[None, None, :] * (y[:, :, -1:] - y[:, :, -2:-1])
        yp = np.concatenate([lo, y, hi], axis=2)
        j = np.arange(nZ)[None, :] + np.clip(np.asarray(s[a:a + chunk], float), -(PAD - 2), PAD - 2)[:, None] + PAD
        j0 = np.floor(j).astype(np.int64)
        t = (j - j0).astype(Y.dtype)[:, None, :]

        def g(idx):
            return np.take_along_axis(yp, np.broadcast_to(idx[:, None, :], (len(y), nR, nZ)), axis=2)
        p0, p1, p2, p3 = g(j0 - 1), g(j0), g(j0 + 1), g(j0 + 2)
        out[a:a + chunk] = p1 + 0.5 * t * (p2 - p0 + t * (2 * p0 - 5 * p1 + 4 * p2 - p3 + t * (3 * (p1 - p2) + p3 - p0)))
    return out


if __name__ == "__main__":
    Z = np.linspace(-1.6, 1.6, 65); R = np.linspace(0.84, 2.54, 65)
    f = lambda z0: np.exp(-((R[:, None] - 1.7) / 0.5) ** 2 - ((Z[None, :] - z0) / 0.9) ** 2)[None]  # noqa: E731
    y = f(0.07).astype(np.float32)
    c = zshift(y, np.array([0.07 / 0.05]))
    print("canonical err", float(np.abs(c - f(0.0)).max()), "roundtrip err", float(np.abs(zshift(c, -np.array([0.07 / 0.05])) - y).max()))
