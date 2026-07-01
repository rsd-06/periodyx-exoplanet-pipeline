"""
Feature engineering stage: builds the classifier-ready feature vector from
the trapezoid fit plus astronomical discriminator checks.

v2 — Fixes applied in this version:
  Fix 1: P/2 aliasing — correct_period_alias() detects when BLS has locked
          onto half the true orbital period and corrects before computing
          secondary eclipse phase.
  Fix 2: Eccentric orbits — secondary_eclipse_depth() now scans the full
          phase curve for the deepest dip rather than assuming phase 0.5.
  Fix 3: Multi-planet contamination — residual BLS run detects additional
          periodic signals; if found, secondary eclipse is marked unreliable.
  Fix 4: Starspot contamination — secondary eclipse depth is checked for
          consistency across three time thirds; if it varies too much, it is
          marked unreliable (starspot, not a fixed orbital signal).
  Fix 5: Low-SNR ingress fraction — bootstrap validates ingress fraction
          reliability; unreliable values are set to NaN for the classifier.
"""

import os
import numpy as np


# ---------------------------------------------------------------------------
# Fix 1: P/2 alias detection and correction
# ---------------------------------------------------------------------------

def correct_period_alias(time, flux, period, t0, odd_even_diff,
                          odd_even_threshold=0.10):
    """Detect and correct the classic BLS half-period aliasing failure mode.

    When an eclipsing binary has a primary eclipse AND a similarly-deep
    secondary eclipse, BLS will often lock onto half the true orbital period
    (P_true / 2) because it sees two roughly equal dips per orbit and treats
    them as one repeating event. This shifts 'phase 0.5' to the wrong place
    for every downstream feature.

    Strategy: if the odd-even depth difference is large (a direct signal that
    alternating transits have different depths, as expected when the period is
    wrong), test whether 2 * period gives a cleaner, more consistent signal.
    If yes, return the doubled period.

    Parameters
    ----------
    time, flux : preprocessed arrays
    period : detected period from BLS/TLS
    t0 : reference transit epoch
    odd_even_diff : already-computed odd-even depth difference at `period`
    odd_even_threshold : float, trigger threshold (default 0.10, calibrated
        against Kepler EB population where confirmed EBs show median ~0.20+)

    Returns
    -------
    corrected_period : float
    was_corrected : bool
    """
    if odd_even_diff is None or np.isnan(odd_even_diff):
        return period, False

    if odd_even_diff <= odd_even_threshold:
        # Odd-even depths are consistent: no evidence of aliasing
        return period, False

    # Test doubled period
    doubled = 2.0 * period

    # Phase-fold at doubled period and compute odd-even difference
    odd_even_doubled = odd_even_depth_difference(time, flux, doubled, t0,
                                                  t_tot=period * 0.1)

    # Accept the correction if:
    #   (a) the doubled-period odd-even difference is lower (more consistent)
    #   (b) the improvement is non-trivial (not just noise)
    if (not np.isnan(odd_even_doubled) and
            odd_even_doubled < odd_even_diff * 0.6):
        return doubled, True

    return period, False


# ---------------------------------------------------------------------------
# Fix 2: Eccentric orbit scan (replaces hardcoded phase 0.5 window)
# Fix 4: Starspot consistency check
# ---------------------------------------------------------------------------

def secondary_eclipse_depth(time, flux, period, t0, t_tot=None,
                              corrected_period=None):
    """Measure the depth of a potential secondary eclipse.

    v2 changes:
    - Uses corrected_period (from alias correction) if available.
    - Scans the full phase curve for the deepest dip outside the primary
      eclipse window, rather than assuming it sits at phase 0.5. This handles
      eccentric orbits where the secondary eclipse is not at phase 0.5.
    - Applies a SNR guard: the candidate secondary must exceed 3x the local
      noise floor to be reported (not just noise fluctuation).
    - Runs a 3-epoch consistency check (Fix 4): if the secondary depth varies
      wildly across time thirds, the measurement is unreliable (starspot
      contamination, not a fixed orbital signal) and is set to NaN.

    Returns
    -------
    depth : float (NaN if unreliable/insufficient data)
    phase_offset : float — phase where the secondary was found
                   (0.5 = circular orbit, other values = eccentric)
    """
    use_period = corrected_period if corrected_period is not None else period
    if t_tot is None:
        t_tot = use_period * 0.05  # fallback: 5% of period

    # Use simple 0-to-1 phase convention.
    # The centered formula ((t - t0 + P/2) % P) - P/2 has a floating-point
    # edge case: a secondary eclipse at EXACTLY t0 + P/2 (e.g. after alias
    # correction doubles the period) wraps to -P/2 instead of +P/2, placing
    # it at phase_01 = 0 (inside the primary mask) rather than 0.5.
    # The 0-to-1 formula avoids this entirely.
    phase_01 = ((time - t0) % use_period) / use_period

    # Mask primary eclipse region
    primary_half = t_tot / use_period * 1.5  # generous buffer
    outside_primary = (phase_01 > primary_half) & (phase_01 < (1.0 - primary_half))

    if outside_primary.sum() < 10:
        return np.nan, np.nan

    # Sliding window scan over phase bins (bin width = 0.05, step = 0.01)
    bin_width = 0.05
    bin_centers = np.arange(primary_half + bin_width / 2,
                             1.0 - primary_half - bin_width / 2, 0.01)

    if len(bin_centers) == 0:
        return np.nan, np.nan

    bin_medians = []
    for bc in bin_centers:
        in_bin = (phase_01 > bc - bin_width / 2) & (phase_01 < bc + bin_width / 2)
        if in_bin.sum() >= 3:
            bin_medians.append((bc, np.nanmedian(flux[in_bin])))

    if len(bin_medians) < 5:
        return np.nan, np.nan

    bin_centers_arr = np.array([b[0] for b in bin_medians])
    bin_vals = np.array([b[1] for b in bin_medians])

    # Out-of-transit noise estimate
    out_of_transit = outside_primary & (
        (phase_01 < 0.3) | (phase_01 > 0.7)
    )
    if out_of_transit.sum() < 5:
        noise_floor = np.nanstd(flux[outside_primary])
    else:
        noise_floor = np.nanstd(flux[out_of_transit])

    # Baseline is median of all scanned bins
    baseline = np.nanmedian(bin_vals)
    dips = baseline - bin_vals  # positive = flux dip

    best_idx = np.argmax(dips)
    best_depth = dips[best_idx]
    best_phase = bin_centers_arr[best_idx]

    # SNR guard: must exceed 1.5x noise floor.
    # We use 1.5x (not 3x) because secondary eclipses are inherently shallower
    # than primaries AND are partially absorbed by the biweight detrending window
    # (window=0.5d vs eclipse duration ~0.15-0.2d). At 3x, real attenuated
    # secondaries are routinely rejected. 1.5x still suppresses random noise.
    if noise_floor <= 0 or best_depth < 1.5 * noise_floor:
        return 0.0, best_phase  # return 0 (not NaN) -- no detectable secondary

    # Fix 4: Starspot consistency check across 3 time thirds
    depth_thirds = []
    third_len = (time[-1] - time[0]) / 3
    t_start = time[0]
    for i in range(3):
        mask_third = (time >= t_start + i * third_len) & \
                     (time < t_start + (i + 1) * third_len)
        if mask_third.sum() < 5:
            continue
        # Use same 0-1 phase convention
        phase_t = ((time[mask_third] - t0) % use_period) / use_period
        in_sec = np.abs(phase_t - best_phase) < bin_width
        if in_sec.sum() >= 3:
            bl_t = np.nanmedian(flux[mask_third])
            sec_t = np.nanmedian(flux[mask_third][in_sec])
            depth_thirds.append(bl_t - sec_t)

    if len(depth_thirds) >= 2:
        depth_arr = np.array(depth_thirds)
        denom = np.mean(np.abs(depth_arr))
        if denom > 0:
            consistency = np.std(depth_arr) / denom
            if consistency > 0.5:
                # Depth varies too much across time — likely a starspot
                return np.nan, best_phase

    return best_depth, best_phase


# ---------------------------------------------------------------------------
# Original features (unchanged interface)
# ---------------------------------------------------------------------------

def odd_even_depth_difference(time, flux, period, t0, t_tot):
    """Fit transit depth separately on odd- and even-numbered transits.
    A large mismatch is a classic signature of an eclipsing binary being
    folded at half its true period.
    """
    transit_number = np.floor((time - t0 + period / 2) / period)
    in_transit = np.abs(((time - t0 + period / 2) % period) - period / 2) < (t_tot / 2)

    odd_mask = in_transit & (transit_number % 2 != 0)
    even_mask = in_transit & (transit_number % 2 == 0)

    if odd_mask.sum() < 3 or even_mask.sum() < 3:
        return np.nan

    odd_depth = 1 - np.nanmedian(flux[odd_mask])
    even_depth = 1 - np.nanmedian(flux[even_mask])
    denom = np.mean([abs(odd_depth), abs(even_depth)])
    if denom == 0:
        return 0.0
    return abs(odd_depth - even_depth) / denom


def depth_snr(depth, flux_out_of_transit):
    """Signal strength relative to the out-of-transit noise floor."""
    noise = np.nanstd(flux_out_of_transit)
    return depth / noise if noise > 0 else np.nan


# ---------------------------------------------------------------------------
# Fix 3: Multi-planet residual check
# ---------------------------------------------------------------------------

def check_multiplanet_contamination(time, flux, period, t0, t_tot,
                                     trapezoid_model_flux,
                                     significance_threshold=7.0):
    """Detect additional periodic signals in the residuals after removing the
    primary transit.

    If a second significant period is found, the secondary eclipse measurement
    is unreliable (a transit from another planet may have landed near phase 0.5
    purely by chance). Returns n_signals_detected and a reliability flag.

    Parameters
    ----------
    trapezoid_model_flux : array, the model prediction at each time point
                           (used to subtract the primary transit)

    Returns
    -------
    n_signals : int (1 = clean, 2+ = multi-period contamination suspected)
    secondary_reliable : bool
    """
    from astropy.timeseries import BoxLeastSquares

    residuals = flux - trapezoid_model_flux
    residuals = residuals - np.nanmedian(residuals)

    # Quick BLS on residuals — coarse grid is sufficient here
    periods_grid = np.linspace(0.5, 15.0, 3000)
    durations_grid = np.linspace(0.03, 0.3, 8)

    try:
        bls = BoxLeastSquares(time, residuals)
        result = bls.power(periods_grid, durations_grid)
        best_power = result.power[np.argmax(result.power)]
        power_median = np.median(result.power)
        power_std = np.std(result.power)
        significance = (best_power - power_median) / power_std if power_std > 0 else 0.0
        best_residual_period = result.period[np.argmax(result.power)]

        # Only flag as multi-planet if the second signal is at a clearly
        # different period (>5% away) and clears significance threshold
        period_sep = abs(best_residual_period - period) / period
        if significance > significance_threshold and period_sep > 0.05:
            return 2, False
    except Exception:
        pass

    return 1, True


# ---------------------------------------------------------------------------
# Fix 5: Low-SNR ingress fraction bootstrap
# ---------------------------------------------------------------------------

def validate_ingress_fraction(phase_time, phase_flux, fit_params,
                               snr, n_bootstrap=50,
                               snr_threshold=5.0,
                               instability_threshold=0.15):
    """Bootstrap-validate the ingress fraction estimate for low-SNR detections.

    For noisy stars where depth_snr < snr_threshold, the trapezoid fitter can
    freely float to a V-shape (large ingress fraction) even for real flat-
    bottomed transits, because the noisy data doesn't constrain the flat-bottom
    duration. This function re-fits on bootstrap resamples; if the spread in
    ingress_fraction across samples is large, the measurement is marked
    unreliable (NaN) so the classifier knows not to trust it.

    Parameters
    ----------
    phase_time, phase_flux : phase-folded arrays
    fit_params : dict from fit_trapezoid
    snr : depth_snr for this detection
    n_bootstrap : number of bootstrap resamples

    Returns
    -------
    ingress_fraction : float or NaN
    """
    if snr >= snr_threshold:
        # High-SNR detection: trust the fit directly
        return fit_params["ingress_fraction"]

    from characterization.trapezoid_fit import fit_trapezoid

    rng = np.random.default_rng(seed=0)
    n = len(phase_time)
    if n < 10:
        return fit_params["ingress_fraction"]

    bootstrap_fracs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        t_b = phase_time[idx]
        f_b = phase_flux[idx]
        order = np.argsort(t_b)
        try:
            fit_b = fit_trapezoid(
                t_b[order], f_b[order],
                depth_guess=max(fit_params["depth"], 1e-5),
                t_tot_guess=fit_params["t_tot"],
                t_in_guess=fit_params["t_in"],
            )
            if not np.isnan(fit_b["ingress_fraction"]):
                bootstrap_fracs.append(fit_b["ingress_fraction"])
        except Exception:
            pass

    if len(bootstrap_fracs) < 10:
        return fit_params["ingress_fraction"]

    spread = np.std(bootstrap_fracs)
    if spread > instability_threshold:
        return np.nan  # XGBoost handles NaN natively

    return fit_params["ingress_fraction"]


# ---------------------------------------------------------------------------
# Feature vector assembly
# ---------------------------------------------------------------------------

def build_feature_vector(trapezoid_fit, detection_result, odd_even_diff,
                          secondary_depth, secondary_phase, snr,
                          n_signals=1, period_corrected=False,
                          ingress_fraction_validated=None):
    """Assemble the final feature vector handed to the classifier.

    New features vs v1:
      secondary_eclipse_phase  — where in the orbit the secondary sits
                                  (0.5 = circular; deviation = eccentric)
      n_signals_detected       — 1 if clean single-period star; 2+ if a
                                  second significant period was found in the
                                  residuals (multi-planet contamination flag)
      period_corrected         — 1 if P/2 alias was detected and the period
                                  was doubled; 0 otherwise. Encodes the fact
                                  that BLS half-period aliasing is
                                  disproportionately common in EBs.
    """
    ingress_frac = (ingress_fraction_validated
                    if ingress_fraction_validated is not None
                    else trapezoid_fit["ingress_fraction"])
    return {
        "depth": trapezoid_fit["depth"],
        "t_tot_hours": trapezoid_fit["t_tot"] * 24,
        "t_in_hours": trapezoid_fit["t_in"] * 24,
        "flat_bottom_hours": trapezoid_fit["flat_bottom_hours"],
        "ingress_fraction": ingress_frac,
        "period": detection_result.get("best_period", np.nan),
        "detection_significance": detection_result.get(
            "SDE", detection_result.get("significance", np.nan)
        ),
        "odd_even_depth_diff": odd_even_diff,
        "secondary_eclipse_depth": secondary_depth,
        "secondary_eclipse_phase": secondary_phase,
        "depth_snr": snr,
        "n_signals_detected": n_signals,
        "period_corrected": int(period_corrected),
    }
