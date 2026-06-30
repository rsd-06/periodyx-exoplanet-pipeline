"""
Transit shape characterization via trapezoid model fitting.

Deliberately uses a trapezoid model (baseline flux, depth, total duration,
ingress/egress duration) rather than a full limb-darkened physical model
(e.g. batman). This is a direct match to ISRO's own reference example
slides, is faster and more numerically stable on noisy data, and needs no
assumed limb-darkening coefficients -- a real unknown for arbitrary stars.

A full batman-based physical re-fit is left as an optional secondary
refinement step for confirmed transit candidates only (not implemented here
to keep the core pipeline lean).
"""

import numpy as np
from lmfit import Model


def _trapezoid(t, f0, depth, t0, t_tot, t_in):
    tt = np.abs(t - t0)
    flat_half = max((t_tot / 2) - t_in, 1e-6)
    f = np.full_like(t, f0)

    in_flat = tt <= flat_half
    f[in_flat] = f0 - depth

    in_ramp = (tt > flat_half) & (tt <= t_tot / 2)
    frac = np.clip((t_tot / 2 - tt[in_ramp]) / max(t_in, 1e-6), 0, 1)
    f[in_ramp] = f0 - depth * frac

    return f


def phase_fold(time, flux, period, t0):
    """Fold light curve at given period/epoch, centered on transit (phase 0)."""
    phase = ((time - t0 + period / 2) % period) - period / 2
    order = np.argsort(phase)
    return phase[order], flux[order]


def fit_trapezoid(phase_time, flux, depth_guess, t_tot_guess, t_in_guess):
    """Fit the trapezoid model to a phase-folded light curve.

    Returns a dict of best-fit parameters with 1-sigma uncertainties
    (from the fit covariance matrix via lmfit).
    """
    model = Model(_trapezoid)
    params = model.make_params(
        f0=1.0, depth=max(depth_guess, 1e-5), t0=0.0,
        t_tot=max(t_tot_guess, 1e-3), t_in=max(t_in_guess, 1e-4),
    )
    params["depth"].min = 0
    params["t_in"].min = 1e-4
    params["t_tot"].min = 1e-3
    # ingress can never exceed half the total duration
    params["t_in"].max = params["t_tot"].value if params["t_tot"].value > 0 else 1.0

    result = model.fit(flux, params, t=phase_time)

    out = {}
    for name, par in result.params.items():
        out[name] = par.value
        out[f"{name}_err"] = par.stderr if par.stderr is not None else np.nan

    out["flat_bottom_hours"] = (out["t_tot"] - 2 * out["t_in"]) * 24
    out["ingress_fraction"] = out["t_in"] / out["t_tot"] if out["t_tot"] > 0 else np.nan
    out["chisqr"] = result.chisqr
    out["redchi"] = result.redchi
    out["fit_result"] = result
    return out
