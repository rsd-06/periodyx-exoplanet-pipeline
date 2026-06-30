"""
Visualization stage: 3-panel plot per detection
  (1) full detrended light curve
  (2) phase-folded light curve at detected period
  (3) trapezoid model fit overlaid on the folded curve
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_detection_summary(time, flux_flat, phase_time, phase_flux,
                            fit_model_func, fit_params, target_name="target",
                            save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    axes[0].plot(time, flux_flat, ".", ms=1.5, color="#2C3E91", alpha=0.6)
    axes[0].set_title(f"{target_name} — Detrended Light Curve")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Normalized Flux")

    axes[1].plot(phase_time * 24, phase_flux, ".", ms=2, color="#7A1F1F", alpha=0.5)
    axes[1].set_title("Phase-Folded")
    axes[1].set_xlabel("Time from Mid-Transit (hours)")
    axes[1].set_ylabel("Normalized Flux")

    smooth_t = np.linspace(phase_time.min(), phase_time.max(), 500)
    model_flux = fit_model_func(smooth_t, fit_params["f0"], fit_params["depth"],
                                 fit_params["t0"], fit_params["t_tot"], fit_params["t_in"])
    axes[2].plot(phase_time * 24, phase_flux, ".", ms=2, color="#999999", alpha=0.4,
                 label="Observed")
    axes[2].plot(smooth_t * 24, model_flux, "-", color="#B5121B", lw=2,
                 label="Best-Fit Trapezoid Model")
    axes[2].set_title("Transit Shape Fit")
    axes[2].set_xlabel("Time from Mid-Transit (hours)")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
