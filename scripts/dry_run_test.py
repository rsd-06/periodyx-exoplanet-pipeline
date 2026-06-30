"""
Dry-run test: validates the ENTIRE training pipeline -- label loading,
feature extraction, classifier training, save/load -- using fast synthetic
data instead of real MAST downloads.

Run this FIRST, before any real cloud/GPU spend, to confirm your
environment and code are correct. Takes seconds, not hours.

Usage:
    python3 scripts/dry_run_test.py
"""

import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic import make_synthetic_lightcurve, make_false_positive_lightcurve
from pipeline import run_pipeline
from classification.classifier import ExoplanetClassifier

FEATURE_COLUMNS = [
    "depth", "t_tot_hours", "t_in_hours", "flat_bottom_hours",
    "ingress_fraction", "period", "detection_significance",
    "odd_even_depth_diff", "secondary_eclipse_depth", "depth_snr",
]


def build_fake_dataset(n_transit=15, n_fp=15):
    """Generate a small labeled synthetic dataset using varied parameters,
    mimicking what build_training_set.py would produce from real KOIs."""
    rows = []
    for i in range(n_transit):
        period = 3.0 + i * 0.4
        t, flux, truth = make_synthetic_lightcurve(period=period, t0=1.0 + i * 0.1,
                                                     depth=0.006 + i * 0.0006, seed=100 + i)
        result = run_pipeline(t, flux, target_name=f"FAKE-T-{i}", use_tls=False,
                               run_single_transit_scan=False)
        row = {"label": "transit"}
        row.update(result["features"])
        rows.append(row)

    for i in range(n_fp):
        t, flux, truth = make_false_positive_lightcurve(seed=200 + i)
        result = run_pipeline(t, flux, target_name=f"FAKE-FP-{i}", use_tls=False,
                               run_single_transit_scan=False)
        row = {"label": "eclipsing_binary"}
        row.update(result["features"])
        rows.append(row)

    return rows


def test_tiered_detection_gate():
    """Directly verifies the BLS-significance-gated TLS tiering: a strong,
    clean injected transit should clear the threshold and trigger TLS; a
    pure-noise light curve with no injected signal should not.

    NOTE: an earlier version of this test used threshold=6.0 and found pure
    noise scored ~6.1 -- i.e. ABOVE that threshold, due to the "look-elsewhere
    effect" (taking the max BLS power over thousands of period trials
    inflates apparent significance even with zero real signal). The default
    threshold was raised to 9.0 as a result. This is exactly why
    scripts/calibrate_threshold.py exists -- don't trust a threshold you
    haven't checked against your own noise floor.
    """
    print("=" * 60)
    print("DRY RUN: verifying BLS -> TLS tiering gate...")
    print("=" * 60)

    threshold = 9.0
    t_strong, flux_strong, _ = make_synthetic_lightcurve(
        depth=0.02, noise_ppm=150, seed=1)  # strong, clean signal
    result_strong = run_pipeline(t_strong, flux_strong, target_name="STRONG-SIGNAL",
                                  use_tls=True, bls_significance_threshold=threshold,
                                  run_single_transit_scan=False)
    print(f"Strong injected transit: BLS significance="
          f"{result_strong['bls']['significance']:.2f}, tls_ran={result_strong['tls_ran']}")

    import numpy as np
    rng = np.random.default_rng(0)
    t_noise = np.arange(0, 27, 2 / (24 * 60))
    flux_noise = 1.0 + rng.normal(0, 300e-6, size=t_noise.shape)  # pure noise, no transit
    result_noise = run_pipeline(t_noise, flux_noise, target_name="PURE-NOISE",
                                 use_tls=True, bls_significance_threshold=threshold,
                                 run_single_transit_scan=False)
    print(f"Pure noise (no signal): BLS significance="
          f"{result_noise['bls']['significance']:.2f}, tls_ran={result_noise['tls_ran']}")
    print(f"(threshold={threshold} -- if pure-noise significance ever creeps close to this "
          f"again on your real data, run calibrate_threshold.py and raise it further.)")

    assert result_strong["tls_ran"], "Strong signal should have triggered TLS but didn't!"
    assert not result_noise["tls_ran"], (
        f"Pure noise scored {result_noise['bls']['significance']:.2f}, above threshold "
        f"{threshold} -- the look-elsewhere effect is worse than expected for this data "
        f"length/cadence. Raise bls_significance_threshold further."
    )
    print("Tiering gate behaves correctly: strong signal -> TLS runs, noise -> TLS skipped.\n")


def main():
    test_tiered_detection_gate()

    print("=" * 60)
    print("DRY RUN: building synthetic labeled dataset...")
    print("=" * 60)
    rows = build_fake_dataset()

    os.makedirs("data", exist_ok=True)
    out_path = "data/_dry_run_features.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label"] + FEATURE_COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in ["label"] + FEATURE_COLUMNS})
    print(f"Wrote {len(rows)} synthetic feature rows to {out_path}")

    print("\n" + "=" * 60)
    print("DRY RUN: training classifier on synthetic features...")
    print("=" * 60)
    import pandas as pd
    df = pd.read_csv(out_path).dropna(subset=FEATURE_COLUMNS)
    X, y = df[FEATURE_COLUMNS], df["label"]

    clf = ExoplanetClassifier(n_estimators=50, max_depth=3)
    clf.fit(X, y)
    preds = clf.predict(X)
    acc = (preds == y.values).mean()
    print(f"Training-set accuracy (sanity check, not held-out): {acc:.3f}")
    print("Feature importances:", clf.feature_importance())

    model_path = "models/_dry_run_model.joblib"
    os.makedirs("models", exist_ok=True)
    clf.save(model_path)
    print(f"\nSaved model to {model_path}")

    reloaded = ExoplanetClassifier.load(model_path)
    reloaded_preds = reloaded.predict(X)
    assert (reloaded_preds == preds).all(), "Reloaded model predictions don't match!"
    print("Reload check passed: saved + reloaded model produces identical predictions.")

    print("\n" + "=" * 60)
    print("DRY RUN PASSED. Pipeline mechanics are correct end-to-end.")
    print("Safe to proceed to real data (build_training_set.py + train_classifier.py).")
    print("=" * 60)


if __name__ == "__main__":
    main()
