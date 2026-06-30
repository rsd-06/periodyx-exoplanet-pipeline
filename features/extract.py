"""
Feature engineering stage: builds the classifier-ready feature vector from
the trapezoid fit plus astronomical discriminator checks (odd-even depth
difference, secondary eclipse) -- the same diagnostics used by ExoMiner's
vetting branches, implemented here as lightweight, interpretable features
rather than learned embeddings.
"""

import numpy as np


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
        return np.nan  # insufficient data for either subset

    odd_depth = 1 - np.nanmedian(flux[odd_mask])
    even_depth = 1 - np.nanmedian(flux[even_mask])
    denom = np.mean([abs(odd_depth), abs(even_depth)])
    if denom == 0:
        return 0.0
    return abs(odd_depth - even_depth) / denom


def secondary_eclipse_depth(time, flux, period, t0):
    """Check for a dip at phase 0.5 (secondary eclipse). A significant
    detection implies an eclipsing binary, not a planet (planets are too
    dim to produce a detectable secondary eclipse in optical TESS data).
    """
    phase = ((time - t0) % period) / period
    secondary_window = (phase > 0.45) & (phase < 0.55)
    baseline_window = (phase < 0.1) | (phase > 0.9)

    if secondary_window.sum() < 3 or baseline_window.sum() < 3:
        return np.nan

    baseline = np.nanmedian(flux[baseline_window])
    secondary = np.nanmedian(flux[secondary_window])
    return baseline - secondary


def depth_snr(depth, flux_out_of_transit):
    """Signal strength relative to the out-of-transit noise floor."""
    noise = np.nanstd(flux_out_of_transit)
    return depth / noise if noise > 0 else np.nan


def build_feature_vector(trapezoid_fit, detection_result, odd_even_diff,
                          secondary_depth, snr):
    """Assemble the final feature vector handed to the classifier."""
    return {
        "depth": trapezoid_fit["depth"],
        "t_tot_hours": trapezoid_fit["t_tot"] * 24,
        "t_in_hours": trapezoid_fit["t_in"] * 24,
        "flat_bottom_hours": trapezoid_fit["flat_bottom_hours"],
        "ingress_fraction": trapezoid_fit["ingress_fraction"],
        "period": detection_result.get("best_period", np.nan),
        "detection_significance": detection_result.get(
            "SDE", detection_result.get("max_power", np.nan)
        ),
        "odd_even_depth_diff": odd_even_diff,
        "secondary_eclipse_depth": secondary_depth,
        "depth_snr": snr,
    }
