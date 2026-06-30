"""
End-to-end demo: proves the full pipeline (Stages A-F) runs successfully on
synthetic data with a known, injected transit -- without requiring live
TESS archive access or a trained classifier.

Run: python3 run_demo.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic import make_synthetic_lightcurve, make_false_positive_lightcurve
from pipeline import run_pipeline
from visualization.plots import plot_detection_summary
from characterization.trapezoid_fit import _trapezoid


def print_result(label, result, truth=None):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    print(f"Points after cleaning: {result['n_points']}")
    print(f"BLS best period: {result['bls']['best_period']:.4f} days "
          f"(power={result['bls']['max_power']:.2f})")
    if "tls" in result:
        print(f"TLS best period: {result['tls']['best_period']:.4f} days "
              f"(SDE={result['tls']['SDE']:.2f})")
    print(f"Detection passes significance threshold: {result['detection_passes_threshold']}")

    fit = result["trapezoid_fit"]
    print("\n-- Trapezoid Fit --")
    print(f"  Depth            : {fit['depth']*100:.4f} % (+/- {fit['depth_err']*100:.4f})")
    print(f"  Total duration   : {fit['t_tot']*24:.3f} hrs (+/- {fit['t_tot_err']*24:.3f})")
    print(f"  Ingress duration : {fit['t_in']*24:.3f} hrs (+/- {fit['t_in_err']*24:.3f})")
    print(f"  Flat bottom      : {fit['flat_bottom_hours']:.3f} hrs")
    print(f"  Ingress fraction : {fit['ingress_fraction']:.3f}  "
          f"({'transit-like' if fit['ingress_fraction'] < 0.3 else 'EB/blend-like'})")

    feats = result["features"]
    print("\n-- Discriminator Features --")
    print(f"  Odd-even depth diff      : {feats['odd_even_depth_diff']:.4f}")
    print(f"  Secondary eclipse depth  : {feats['secondary_eclipse_depth']:.6f}")
    print(f"  Depth SNR                : {feats['depth_snr']:.2f}")

    print(f"\nClassification: {result['classification']}")
    print(f"Single-transit candidates flagged: {len(result.get('single_transit_candidates', []))}")

    if truth:
        print("\n-- Ground Truth (injected) --")
        for k, v in truth.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    os.makedirs("demo_outputs", exist_ok=True)

    # --- Case 1: genuine transit-like signal ---
    t, flux, truth = make_synthetic_lightcurve()
    result_transit = run_pipeline(t, flux, target_name="SYNTH-TRANSIT-01", use_tls=True)
    print_result("CASE 1: Injected Transit-Like Signal", result_transit, truth)

    arrs = result_transit["_arrays"]
    plot_detection_summary(
        arrs["time"], arrs["flux_flat"], arrs["phase_time"], arrs["phase_flux"],
        _trapezoid, arrs["fit_params"], target_name="SYNTH-TRANSIT-01",
        save_path="demo_outputs/case1_transit.png",
    )

    # --- Case 2: eclipsing-binary-like false positive ---
    t2, flux2, truth2 = make_false_positive_lightcurve()
    result_fp = run_pipeline(t2, flux2, target_name="SYNTH-FALSEPOS-01", use_tls=True)
    print_result("CASE 2: Injected Eclipsing-Binary-Like False Positive", result_fp, truth2)

    arrs2 = result_fp["_arrays"]
    plot_detection_summary(
        arrs2["time"], arrs2["flux_flat"], arrs2["phase_time"], arrs2["phase_flux"],
        _trapezoid, arrs2["fit_params"], target_name="SYNTH-FALSEPOS-01",
        save_path="demo_outputs/case2_falsepositive.png",
    )

    print(f"\n{'='*60}")
    print("Demo complete. Plots saved to demo_outputs/")
    print(f"{'='*60}")
