"""
Pipeline orchestrator: chains every stage from raw (time, flux) input
through to a final, reportable result dict.

This is the single entry point the rest of the system (batch runner, demo
script, future dashboard) calls. Each stage is independently swappable
(e.g. BLS-only vs BLS+TLS, synthetic vs real acquisition) without touching
this orchestration logic.

v2 — Wires in all 5 diagnostic fixes:
  Fix 1: P/2 alias correction before secondary eclipse calculation
  Fix 2: Eccentric orbit scan (full-phase secondary search)
  Fix 3: Period-doubling check (BLS locked onto 2P)
  Fix 4: Starspot consistency check (inside secondary_eclipse_depth)
  Fix 5: Low-SNR ingress fraction bootstrap validation
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.detrend import detrend_lightcurve, sigma_clip
from detection.periodic_search import (bls_search, tls_search,
                                        detection_passes_threshold,
                                        check_half_period)
from characterization.trapezoid_fit import phase_fold, fit_trapezoid, _trapezoid
from features.extract import (odd_even_depth_difference, secondary_eclipse_depth,
                               depth_snr, build_feature_vector,
                               correct_period_alias, check_multiplanet_contamination,
                               validate_ingress_fraction)
from singletransit.flagging import flag_single_transit_candidates


def run_pipeline(time, flux, target_name="target", use_tls=True,
                  bls_significance_threshold=9.0, run_single_transit_scan=True,
                  classifier=None):
    """Run the full detection-to-feature-extraction pipeline on one star.

    Parameters
    ----------
    time, flux : raw input arrays
    use_tls : if True, refine BLS result with TLS -- but ONLY for stars whose
        BLS significance clears `bls_significance_threshold`. This is the
        triage step: BLS runs on every star (cheap), TLS (slow) only runs on
        the subset that look like real candidates.
    bls_significance_threshold : normalized BLS score, see
        detection/periodic_search.py:bls_search. Default 9.0 -- chosen
        empirically after observing that pure noise can score ~6 on this
        metric purely from the "look-elsewhere effect" (taking the max over
        thousands of period trials inflates apparent significance even with
        no real signal). This default is a starting point, NOT a substitute
        for calibrating against your own data -- see
        scripts/calibrate_threshold.py.
    classifier : optional ExoplanetClassifier instance. If None or not yet
        fitted, classification is skipped and reported as "pending_training".

    Returns
    -------
    dict containing every intermediate and final pipeline output.
    """
    result = {"target": target_name}

    # Stage A: clean + detrend
    time_c, flux_c = sigma_clip(time, flux, sigma=5.0)
    flux_flat, trend = detrend_lightcurve(time_c, flux_c, window_length=0.5)
    result["n_points"] = len(time_c)

    # Stage B: periodic search (tiered BLS -> TLS)
    bls_result = bls_search(time_c, flux_flat)
    result["bls"] = {k: v for k, v in bls_result.items() if k not in ("periods", "power")}

    # Fix 3: Check for period-doubling (BLS locked onto 2P instead of P).
    # Do this BEFORE TLS, so if we correct the period, TLS refines the right one.
    period_from_bls = bls_result["best_period"]
    t0_from_bls = bls_result["best_t0"]
    bls_max_power = bls_result["max_power"]

    period_halved, halved = check_half_period(
        time_c, flux_flat, period_from_bls, t0_from_bls, bls_max_power
    )
    if halved:
        bls_result["best_period"] = period_halved
        bls_result["best_t0"] = t0_from_bls  # epoch stays valid
        result["bls"]["period_halved"] = True
        result["bls"]["best_period"] = period_halved
    else:
        result["bls"]["period_halved"] = False

    detection = bls_result
    result["tls_ran"] = False
    if use_tls and bls_result["significance"] > bls_significance_threshold:
        try:
            tls_result = tls_search(time_c, flux_flat)
            result["tls"] = {k: v for k, v in tls_result.items() if k not in ("periods", "power")}
            detection = tls_result
            result["tls_ran"] = True
        except Exception as e:
            result["tls_error"] = str(e)

    period = detection["best_period"]
    t0 = detection["best_t0"]
    duration = detection["best_duration"]

    # Use the appropriate significance metric depending on which detector ran.
    # TLS produces SDE (Signal Detection Efficiency), which is a well-established
    # normalized score with a conventional cutoff of ~7.
    # BLS produces our own normalized significance (same concept, same scale)
    # computed inside bls_search(). Both are directly comparable.
    # Do NOT use raw max_power here -- it's not on the same scale.
    if result["tls_ran"]:
        sig_for_threshold = detection.get("SDE", 0)
    else:
        sig_for_threshold = bls_result["significance"]
        # Ensure the feature extractor uses this normalized significance instead of raw max_power
        detection["SDE"] = sig_for_threshold

    result["detection_passes_threshold"] = detection_passes_threshold(sig_for_threshold)
    result["detection_significance_used"] = sig_for_threshold

    # Stage C: phase fold + trapezoid fit
    phase_time, phase_flux = phase_fold(time_c, flux_flat, period, t0)
    fit = fit_trapezoid(
        phase_time, phase_flux,
        depth_guess=detection.get("best_depth", 0.01),
        t_tot_guess=duration,
        t_in_guess=duration * 0.2,
    )
    result["trapezoid_fit"] = {k: v for k, v in fit.items() if k != "fit_result"}

    # Stage D: feature engineering (astronomical discriminators)

    # Compute odd-even depth difference first — needed by Fix 1
    odd_even = odd_even_depth_difference(time_c, flux_flat, period, t0, fit["t_tot"])

    # Fix 1: Correct P/2 aliasing before computing secondary eclipse phase.
    # If BLS locked onto half the true period, "phase 0.5" lands in the
    # wrong place. correct_period_alias() tests 2*period and switches if
    # the doubled period gives a more consistent signal.
    corrected_period, period_was_corrected = correct_period_alias(
        time_c, flux_flat, period, t0, odd_even
    )
    result["period_alias_corrected"] = period_was_corrected

    # Fix 2 + Fix 4: Eccentric orbit scan + starspot consistency check.
    # secondary_eclipse_depth() now scans the full phase curve (not just 0.5)
    # and validates depth consistency across 3 time-thirds.
    secondary, secondary_phase = secondary_eclipse_depth(
        time_c, flux_flat, period, t0,
        t_tot=fit["t_tot"],
        corrected_period=corrected_period if period_was_corrected else None,
    )
    result["secondary_eclipse_phase"] = secondary_phase

    out_of_transit_mask = np.abs(phase_time) > fit["t_tot"]
    snr = depth_snr(fit["depth"], phase_flux[out_of_transit_mask])

    # Fix 3: Multi-planet contamination check.
    # Build the trapezoid model flux for the full time array, then check
    # residuals for additional periodic signals.
    trapezoid_model = _trapezoid(
        time_c,
        f0=fit.get("f0", 1.0),
        depth=fit["depth"],
        t0=t0,
        t_tot=fit["t_tot"],
        t_in=fit["t_in"],
    )
    n_signals, secondary_reliable = check_multiplanet_contamination(
        time_c, flux_flat, period, t0, fit["t_tot"], trapezoid_model
    )
    if not secondary_reliable:
        secondary = np.nan  # contaminated; don't feed a spurious number to the model
    result["n_signals_detected"] = n_signals

    # Fix 5: Bootstrap-validate ingress fraction for low-SNR stars.
    # For noisy detections, the fitter can produce a spuriously large ingress
    # fraction (V-shape) even for genuine flat-bottomed transits. Bootstrap
    # variance > 0.15 sets ingress_fraction = NaN so the classifier knows
    # this measurement is unreliable.
    ingress_validated = validate_ingress_fraction(
        phase_time, phase_flux, fit, snr
    )

    features = build_feature_vector(
        fit, detection, odd_even, secondary, secondary_phase, snr,
        n_signals=n_signals,
        period_corrected=period_was_corrected or halved,
        ingress_fraction_validated=ingress_validated,
    )
    result["features"] = features

    # Stage E: classification (skipped until curated dataset available)
    if classifier is not None and getattr(classifier, "is_fitted", False):
        import pandas as pd
        feat_df = pd.DataFrame([features])
        mean_proba, std_proba = classifier.predict_proba(feat_df)
        
        result["classification"] = dict(zip(classifier.classes_, mean_proba[0].tolist()))
        result["classification_uncertainty"] = dict(zip(classifier.classes_, std_proba[0].tolist()))
    else:
        result["classification"] = "pending_training -- awaiting ISRO curated dataset"
        result["classification_uncertainty"] = {}

    # Stage F: single-transit exploratory flagging (independent side-branch)
    if run_single_transit_scan:
        result["single_transit_candidates"] = flag_single_transit_candidates(
            time_c, flux_flat, expected_duration_hours=duration * 24
        )

    # carry forward arrays needed for plotting
    result["_arrays"] = dict(time=time_c, flux_flat=flux_flat,
                              phase_time=phase_time, phase_flux=phase_flux,
                              fit_params=fit)
    return result
