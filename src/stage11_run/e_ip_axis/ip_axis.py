#!/usr/bin/env python3
"""Stage 11e / Ch1 ip_axis — label-free DIII-D Ip time-axis estimator (inputs only).

Finding (an internal development note §1): every DIII-D file ships the same generic Ip axis (-858.187 .. 14500.81 ms,
30,719 samples, 0.5 ms).  starter/data_fixes.fix_d3d_ip_times re-origins it to magnetics_time[0] when the 10 %-of-peak
crossing falls outside [-500, 1000] ms (the "new" DAQ era, magnetics record starting at about -4.1 s) and otherwise keeps
the shipped axis.  The kept shots are exactly the "old" DAQ era (magnetics record starting at about -9.2 s, length
24,575.5 ms), and on them the shipped axis is NOT correct: the breakdown time t_bd (10 % crossing) on the shipped axis obeys
    t_bd + magnetics_time[0] = -9034.2 ms   (MAD 2.4 ms over 2,111 train shots; corr(t_bd, mt0) = -0.997),
i.e. the Ip record of an old-era shot starts a FIXED 8,192 ms (2^13 ms) after its magnetics record starts, exactly as the
organiser's rule puts the new-era Ip record at the magnetics start (offset 0).  With
    ip_axis = magnetics_time[0] + 8192 ms + (shipped - shipped[0])
the old-era breakdown distribution (5/50/95 % = 10.2 / 16.0 / 24.1 ms) matches the new era (10.1 / 14.9 / 26.0 ms).

`fix_d3d_ip_times_v22(row)` = that rule on the kept (old-era) shots, guarded:
  * applies only when fix_d3d_ip_times kept the shipped axis AND the magnetics record starts before -6,000 ms;
  * the corrected breakdown must land in [-100, +150] ms, else a fallback that puts the 10 % crossing at 15 ms
    (only when the shipped crossing is later than 40 ms), else the v2.1 axis.
Re-origined shots and anything unusual keep the v2.1 axis bit-for-bit.

`install()` monkeypatches data_fixes.fix_d3d_ip_times so the UNCHANGED feat_v2/feat_v21 code paths (which import it at call
time) build v2.2 features.  `original()` returns the unpatched function.
"""
from __future__ import annotations

import numpy as np

OLD_ERA_IP_OFFSET_MS = 8192.0          # Ip record start - magnetics record start, old DAQ era
OLD_ERA_MAG_T0_MAX_MS = -6000.0        # old era: magnetics record starts at about -9.2 s (new era about -4.1 s)
BD_WINDOW_MS = (-100.0, 150.0)         # plausible corrected breakdown window (new era 0.5-99.5 %: 7..64 ms)
BD_TARGET_MS = 15.0                    # fallback: put the 10 % crossing here (new-era median 14.9 ms)
BD_LATE_MS = 40.0

_ORIG = None


def original():
    global _ORIG
    if _ORIG is None:
        import data_fixes
        _ORIG = getattr(data_fixes, "_fix_d3d_ip_times_v21", data_fixes.fix_d3d_ip_times)
    return _ORIG


def _breakdown(ipt, ip):
    ipa = np.abs(np.asarray(ip, np.float64))
    n = min(len(ipa), len(ipt))
    ipa, ipt = ipa[:n], np.asarray(ipt, np.float64)[:n]
    ok = np.isfinite(ipa) & np.isfinite(ipt)
    if ok.sum() < 10:
        return np.nan
    thr = 0.10 * np.nanpercentile(ipa[ok], 99.5)
    on = ok & (ipa > thr)
    return float(ipt[np.argmax(on)]) if on.any() else np.nan


def ip_axis_info(row):
    """Return dict(axis=<corrected Ip axis>, branch=..., shift_ms=..., t_bd_v21=..., t_bd_v22=..., mag_t0=...)."""
    shipped = np.asarray(row["magnetics_plasma_current_times"], dtype=np.float64)
    fixed = np.asarray(original()(row), dtype=np.float64)
    ip = row["magnetics_plasma_current"]
    mt = np.asarray(row["magnetics_time"], dtype=np.float64)
    mtf = mt[np.isfinite(mt)]
    mt0 = float(mtf[0]) if len(mtf) else np.nan
    tbd0 = _breakdown(fixed, ip)
    info = {"axis": fixed, "branch": "reorigined", "shift_ms": 0.0, "t_bd_v21": tbd0, "t_bd_v22": tbd0, "mag_t0": mt0}
    kept = len(fixed) == len(shipped) and np.array_equal(fixed, shipped)
    if not kept or len(shipped) == 0 or not np.isfinite(shipped[0]):
        return info
    info["branch"] = "kept_unchanged"
    if np.isfinite(mt0) and mt0 < OLD_ERA_MAG_T0_MAX_MS:
        cand = mt0 + OLD_ERA_IP_OFFSET_MS + (shipped - shipped[0])
        tb = _breakdown(cand, ip)
        if np.isfinite(tb) and BD_WINDOW_MS[0] <= tb <= BD_WINDOW_MS[1]:
            info.update(axis=cand, branch="old_era_8192", shift_ms=float(cand[0] - shipped[0]), t_bd_v22=tb)
            return info
    if np.isfinite(tbd0) and tbd0 > BD_LATE_MS:
        sh = BD_TARGET_MS - tbd0
        info.update(axis=shipped + sh, branch="fallback_tbd", shift_ms=float(sh), t_bd_v22=BD_TARGET_MS)
    return info


def fix_d3d_ip_times_v22(row) -> np.ndarray:
    return ip_axis_info(row)["axis"]


def install():
    """Route data_fixes.fix_d3d_ip_times -> v2.2 (idempotent; keeps the original as _fix_d3d_ip_times_v21)."""
    import data_fixes
    if not hasattr(data_fixes, "_fix_d3d_ip_times_v21"):
        data_fixes._fix_d3d_ip_times_v21 = data_fixes.fix_d3d_ip_times
    original()
    data_fixes.fix_d3d_ip_times = fix_d3d_ip_times_v22
    return data_fixes
