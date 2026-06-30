"""
Periodic signal detection stage.

Two-tier design for computational efficiency at scale:
  Tier 1 (BLS): fast, coarse triage across ALL stars.
  Tier 2 (TLS): slower, more sensitive refinement -- only run on stars that
                pass the Tier 1 power threshold.

This tiered structure is itself a deliberate efficiency decision for
operating at ISRO's full-sector scale (20-30k targets), not just a default.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares


def bls_search(time, flux, period_min=0.5, period_max=15.0, n_periods=10000,
                durations=None):
    """Fast first-pass periodic search using Box Least Squares.

    Returns dict with best_period, best_duration, best_t0, power, periods,
    and a normalized `significance` score analogous to TLS's SDE:
    (best_power - median(power)) / std(power). Raw BLS power is NOT
    comparable across stars (it scales with each star's noise level and
    depth) -- this normalized score is what should be used for thresholding,
    not max_power directly.
    """
    if durations is None:
        durations = np.linspace(0.03, 0.3, 10)  # days

    periods = np.linspace(period_min, period_max, n_periods)
    bls = BoxLeastSquares(time, flux)
    result = bls.power(periods, durations)

    best_idx = np.argmax(result.power)
    power_median = np.median(result.power)
    power_std = np.std(result.power)
    significance = ((result.power[best_idx] - power_median) / power_std
                     if power_std > 0 else 0.0)

    return dict(
        best_period=result.period[best_idx],
        best_duration=result.duration[best_idx],
        best_t0=result.transit_time[best_idx],
        best_depth=result.depth[best_idx],
        max_power=result.power[best_idx],
        significance=significance,
        periods=periods,
        power=result.power,
    )


def tls_search(time, flux, period_min=0.5, period_max=15.0):
    """Refined search using Transit Least Squares -- more sensitive to
    box-shaped periodic signals, and gives an SDE (Signal Detection
    Efficiency) statistic usable directly as a detection significance score.
    """
    from transitleastsquares import transitleastsquares

    model = transitleastsquares(time, flux)
    result = model.power(period_min=period_min, period_max=period_max,
                          show_progress_bar=False, use_threads=1)

    return dict(
        best_period=result.period,
        best_duration=result.duration,
        best_t0=result.T0,
        best_depth=1 - result.depth,
        SDE=result.SDE,
        periods=result.periods,
        power=result.power,
    )


def detection_passes_threshold(sde_or_power, threshold=7.0):
    """Standard literature cutoff: SDE > 7 is the conventional credible-
    candidate threshold."""
    return sde_or_power > threshold
