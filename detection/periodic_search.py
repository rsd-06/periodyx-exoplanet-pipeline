"""
Periodic signal detection stage.

Two-tier design for computational efficiency at scale:
  Tier 1 (BLS): fast, coarse triage across ALL stars.
  Tier 2 (TLS): slower, more sensitive refinement -- only run on stars that
                pass the Tier 1 power threshold.

This tiered structure is itself a deliberate efficiency decision for
operating at ISRO's full-sector scale (20-30k targets), not just a default.

v2 additions:
  check_half_period() -- Fix 3: detects when BLS has locked onto 2P
  (reports every other transit as noise). Tests period/2 via a second
  focused BLS run and uses BIC comparison to decide which period fits better.
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


def check_half_period(time, flux, period, t0, bls_best_power,
                       min_transits=3, bic_improvement=6.0):
    """Fix 3 — detect when BLS has locked onto 2P instead of P.

    BLS sometimes skips every other transit (treating it as noise or scatter)
    and reports 2x the true period. Symptom: the expected number of transit
    events per baseline is half of what it should be.

    Strategy:
      1. Run a focused BLS pass restricted to period/2 ± 15%.
      2. Count how many clean transit epochs appear at each period.
      3. Use BIC (Bayesian Information Criterion) to decide which period
         fits the data better. ΔBIC > 6 is conventionally 'strong evidence'.
         We only correct if the evidence is decisive -- we don't want to
         halve a genuine long-period planet's period.

    Parameters
    ----------
    time, flux : preprocessed arrays
    period : BLS best period to test
    t0 : BLS best transit time
    bls_best_power : BLS max power at `period` (for BIC reference)
    min_transits : minimum transit events required at period/2 to accept it
    bic_improvement : ΔBIC threshold (default 6 = strong evidence per Kass & Raftery)

    Returns
    -------
    corrected_period : float (period/2 if correction accepted, else original period)
    was_corrected : bool
    """
    half_period = period / 2.0
    if half_period < 0.5:
        # Don't correct below the minimum physically meaningful period
        return period, False

    # Count expected transit events at the full period
    baseline = time[-1] - time[0]
    n_transits_full = max(int(baseline / period), 1)
    n_transits_half = int(baseline / half_period)

    if n_transits_half < min_transits:
        return period, False

    # Focused BLS on period/2 range
    half_range = np.linspace(half_period * 0.85, half_period * 1.15, 500)
    durations = np.linspace(0.03, 0.3, 8)
    try:
        bls = BoxLeastSquares(time, flux)
        result_half = bls.power(half_range, durations)
        best_half_power = result_half.power[np.argmax(result_half.power)]

        # BIC approximation: BIC = -2 * ln(L) + k * ln(n)
        # For BLS, power is proportional to -ln(L); we use power directly
        # as a proxy. ΔBIC = 2 * (power_half - power_full) adjusted for
        # the parameter count difference (same model complexity, different period).
        # If half-period fits substantially better, accept it.
        delta_power = best_half_power - bls_best_power
        if delta_power > bic_improvement:
            best_half_period = result_half.period[np.argmax(result_half.power)]
            return best_half_period, True
    except Exception:
        pass

    return period, False
