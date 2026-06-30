"""
Single-transit candidate flagging.

EXPLICITLY SCOPED AS EXPLORATORY. Long-period planets that transit only once
within the observation baseline cannot be confirmed via periodicity-based
search (BLS/TLS both require >=2 transits). This is a known, openly
documented blind spot shared by every major existing pipeline.

This module performs a sliding-window matched-filter scan for isolated,
statistically significant dips, and flags them SEPARATELY from the main
periodic-detection results with explicit lower confidence. It does not
attempt to confirm or classify these candidates -- only to flag them for
human follow-up.
"""

import numpy as np
from scipy.ndimage import uniform_filter1d


def flag_single_transit_candidates(time, flux, expected_duration_hours=3.0,
                                    cadence_minutes=2.0, sigma_threshold=4.0):
    """Scan for isolated, statistically significant dips that do not
    necessarily repeat periodically within the baseline.

    Returns a list of candidate dicts: {time, depth, significance}.
    """
    cadence_days = cadence_minutes / (24 * 60)
    window_points = max(int((expected_duration_hours / 24) / cadence_days) * 5, 5)

    local_mean = uniform_filter1d(flux, size=window_points)
    residual = local_mean - flux
    local_std = np.sqrt(uniform_filter1d(residual ** 2, size=window_points))

    with np.errstate(divide="ignore", invalid="ignore"):
        significance = np.where(local_std > 0, residual / local_std, 0)

    candidate_mask = significance > sigma_threshold

    # group contiguous flagged points into discrete candidate events
    candidates = []
    idx = np.where(candidate_mask)[0]
    if len(idx) == 0:
        return candidates

    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for g in groups:
        if len(g) == 0:
            continue
        candidates.append({
            "time": float(np.mean(time[g])),
            "depth": float(1 - np.min(flux[g])),
            "significance": float(np.max(significance[g])),
            "confidence": "low -- exploratory single-transit flag, requires follow-up",
        })
    return candidates
