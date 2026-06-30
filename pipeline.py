"""
Pipeline orchestrator: chains every stage from raw (time, flux) input
through to a final, reportable result dict.

This is the single entry point the rest of the system (batch runner, demo
script, future dashboard) calls. Each stage is independently swappable
(e.g. BLS-only vs BLS+TLS, synthetic vs real acquisition) without touching
this orchestration logic.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.detrend import detrend_lightcurve, sigma_clip
from detection.periodic_search import bls_search, tls_search, detection_passes_threshold
from characterization.trapezoid_fit import phase_fold, fit_trapezoid, _trapezoid
from features.extract import (odd_even_depth_difference, secondary_eclipse_depth,
                               depth_snr, build_feature_vector)
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

    result["detection_passes_threshold"] = detection_passes_threshold(
        detection.get("SDE", detection.get("max_power", 0))
    )

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
    odd_even = odd_even_depth_difference(time_c, flux_flat, period, t0, fit["t_tot"])
    secondary = secondary_eclipse_depth(time_c, flux_flat, period, t0)
    out_of_transit_mask = np.abs(phase_time) > fit["t_tot"]
    snr = depth_snr(fit["depth"], phase_flux[out_of_transit_mask])

    features = build_feature_vector(fit, detection, odd_even, secondary, snr)
    result["features"] = features

    # Stage E: classification (skipped until curated dataset available)
    if classifier is not None and getattr(classifier, "is_fitted", False):
        proba = classifier.predict_proba([features])[0]
        result["classification"] = dict(zip(classifier.classes_, proba.tolist()))
    else:
        result["classification"] = "pending_training -- awaiting ISRO curated dataset"

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
