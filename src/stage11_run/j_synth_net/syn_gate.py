#!/usr/bin/env python3
"""
Stage 11J / Ch2 TRACK synth_net — label-free plausibility gate of the laser-profile features (numpy only; ONE rule for real and
synthetic frames, applied at training and at prediction time to the stored feature vectors).
Feature layout (syn_common.FEAT_NAMES), positions in units (R - R0) / a:
  0-7   out_eq, x, out_ge, x, in_eq, x, in_ge, x        8-27  per level (5): lo_has, lo_x, hi_has, hi_x
  28-32 prof_has, tepk_x, sym_has, sym_x, log_beta_pe
Rules (physics of a monotone Te(psi) on a midplane chord through the axis; no fitted constant):
  (0) a measured outer edge less than 0.6 a outboard of the inner edge / inner coverage limit is not the edge of a plasma of this
      machine at Ip >= 50 kA (junk profile, edge found on the inboard side) -> the whole laser block is dropped (= no Thomson);
  (1) the Te peak must lie inside the measured edges / coverage limits, at least 15 % of the width away from either;
      otherwise the profile is not a core profile (hot spurious channels, laser off-axis ...) -> all profile features dropped;
  (2) an isotherm crossing must be on its own side of the Te peak and not outside the edge by more than 0.1 a -> side dropped;
  (3) crossings of increasing level must move monotonically towards the peak (lo increases, hi decreases) -> offending side dropped.
"""
import numpy as np

N_LVL, L0, P0 = 5, 8, 28


def gate(F):
    F = np.array(F, dtype=np.float32, copy=True)
    sh = F.shape
    F = F.reshape(-1, sh[-1])
    nan = np.float32(np.nan)
    out_x = np.where(F[:, 0] > 0, F[:, 1], np.where(F[:, 2] > 0, F[:, 3], nan))
    in_x = np.where(F[:, 4] > 0, F[:, 5], np.where(F[:, 6] > 0, F[:, 7], nan))
    has = F[:, P0] > 0
    tepk = np.where(has, F[:, P0 + 1], nan)
    w = out_x - in_x
    with np.errstate(invalid="ignore"):
        junk = (F[:, 0] > 0) & (w < 0.6)
    F[junk, :] = 0.0
    out_x = np.where(junk, nan, out_x); in_x = np.where(junk, nan, in_x)
    has = has & ~junk
    tepk = np.where(has, tepk, nan)
    w = out_x - in_x
    with np.errstate(invalid="ignore"):
        bad = has & ((tepk < in_x + 0.15 * w) | (tepk > out_x - 0.15 * w))
        prev_lo, prev_hi = np.full(len(F), -np.inf, np.float32), np.full(len(F), np.inf, np.float32)
        n_side = 0
        for l in range(N_LVL):
            j = L0 + 4 * l
            hl, xl, hh, xh = F[:, j] > 0, F[:, j + 1], F[:, j + 2] > 0, F[:, j + 3]
            bl = hl & ((xl >= tepk) | (xl < in_x - 0.1) | (xl <= prev_lo))
            bh = hh & ((xh <= tepk) | (xh > out_x + 0.1) | (xh >= prev_hi))
            n_side += int(bl.sum() + bh.sum())
            F[bl, j] = 0.0; F[bl, j + 1] = 0.0
            F[bh, j + 2] = 0.0; F[bh, j + 3] = 0.0
            okl, okh = hl & ~bl, hh & ~bh
            prev_lo = np.where(okl, xl, prev_lo); prev_hi = np.where(okh, xh, prev_hi)
    F[bad, L0:] = 0.0
    return F.reshape(sh), {"n": int(len(F)), "prof_dropped": int(bad.sum()), "sides_dropped": n_side, "has_prof": int(has.sum()),
                          "block_dropped": int(junk.sum())}
