"""
Detrending stage: remove instrumental/stellar systematics while preserving
the transit signal, via wotan's biweight filter with iterative transit masking.
"""

import numpy as np
from wotan import flatten


def detrend_lightcurve(time, flux, window_length=0.5, transit_mask=None):
    """Flatten a light curve using a biweight filter.

    Parameters
    ----------
    time, flux : arrays
    window_length : float, days. Should be ~2-3x expected transit duration.
    transit_mask : boolean array, True where in-transit points should be
        excluded from the trend fit (second-pass refinement).

    Returns
    -------
    flux_flat, trend
    """
    if transit_mask is not None:
        flux_flat, trend = flatten(
            time, flux, method="biweight", window_length=window_length,
            mask=transit_mask, return_trend=True,
        )
    else:
        flux_flat, trend = flatten(
            time, flux, method="biweight", window_length=window_length,
            return_trend=True,
        )
    return flux_flat, trend


def sigma_clip(time, flux, sigma=5.0):
    """Remove points beyond `sigma` standard deviations from the median."""
    median = np.nanmedian(flux)
    std = np.nanstd(flux)
    mask = np.abs(flux - median) < sigma * std
    return time[mask], flux[mask]
