"""
Calibrates a sensible BLS significance threshold using phase-scrambling --
NOT by comparing significance between class labels (an earlier version of
this script did that, and it's the wrong comparison: eclipsing binaries and
other false positives often produce STRONGER, cleaner periodic signals than
genuine small-planet transits, since planets cause much shallower dips. So
"not_transit" stars frequently score HIGHER BLS significance than real
transits -- comparing class labels tells you almost nothing useful about
where to set a noise-floor cutoff, and can suggest a threshold that's
actually below the real noise floor.)

What this gate is actually for: separating "there is a real periodic signal
here, worth the expensive TLS refinement" from "there is nothing here at
all." It is NOT meant to separate planets from eclipsing binaries -- that
job belongs to the shape-fit features and the classifier, downstream.

The correct way to find the noise floor: take your actual real light curves
and phase-scramble them (randomly shuffle the flux values' time order).
This destroys any real periodicity while preserving the star's genuine
noise characteristics, gaps, and cadence exactly. Running BLS on the
scrambled version measures pure look-elsewhere-effect noise, on your real
data, not a synthetic guess.

Usage:
    python3 scripts/calibrate_threshold.py --koi-csv data/koi_cumulative.csv --sample 150
"""

import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.koi_labels import load_koi_labels
from data.acquisition import fetch_lightcurve
from preprocessing.detrend import detrend_lightcurve, sigma_clip
from detection.periodic_search import bls_search


def phase_scramble(flux, seed):
    """Randomly shuffle flux values' time order -- destroys periodicity,
    preserves the star's real noise distribution exactly."""
    rng = np.random.default_rng(seed)
    return rng.permutation(flux)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koi-csv", default="data/koi_cumulative.csv")
    ap.add_argument("--sample", type=int, default=150,
                     help="Number of real stars to sample (any disposition -- this no "
                          "longer splits by class label) for noise-floor estimation.")
    ap.add_argument("--scrambles-per-star", type=int, default=3,
                     help="Number of independent phase-scrambles per star, to get a "
                          "more robust noise-floor estimate than a single shuffle.")
    args = ap.parse_args()

    labels_df = load_koi_labels(args.koi_csv, drop_unlabeled=True)
    sampled = labels_df.sample(n=min(args.sample, len(labels_df)), random_state=42)
    print(f"\nEstimating noise floor from {len(sampled)} real stars "
          f"({args.scrambles_per_star} scrambles each = "
          f"{len(sampled) * args.scrambles_per_star} noise samples)...\n")

    real_significances = []
    noise_significances = []
    n_fail = 0

    for i, row in enumerate(sampled.itertuples(), 1):
        try:
            time, flux, _ = fetch_lightcurve(f"KIC {row.kepid}", mission="Kepler", author="Kepler")
            time_c, flux_c = sigma_clip(time, flux, sigma=5.0)
            flux_flat, _ = detrend_lightcurve(time_c, flux_c, window_length=0.5)

            # real (unscrambled) significance -- informative context, not used for the cutoff
            real_bls = bls_search(time_c, flux_flat)
            real_significances.append(real_bls["significance"])

            # phase-scrambled significance -- THIS is the noise-floor estimate
            for s in range(args.scrambles_per_star):
                scrambled_flux = phase_scramble(flux_flat, seed=hash((row.kepid, s)) % (2**32))
                noise_bls = bls_search(time_c, scrambled_flux)
                noise_significances.append(noise_bls["significance"])
        except Exception:
            n_fail += 1
        if i % 25 == 0:
            print(f"  [{i}/{len(sampled)}] processed (failures so far: {n_fail})")

    real_significances = np.array(real_significances)
    noise_significances = np.array(noise_significances)

    print(f"\nReal (unscrambled) significance, n={len(real_significances)} "
          f"(mixed classes -- informative only, NOT used to set the threshold):")
    if len(real_significances) > 0:
        print(f"  mean={real_significances.mean():.2f}  median={np.median(real_significances):.2f}")

    print(f"\nPhase-scrambled noise floor, n={len(noise_significances)} "
          f"(THIS is what the threshold should sit above):")
    if len(noise_significances) > 0:
        median = np.median(noise_significances)
        mad = np.median(np.abs(noise_significances - median))
        p95 = np.percentile(noise_significances, 95)
        p99 = np.percentile(noise_significances, 99)
        print(f"  mean={noise_significances.mean():.2f}  median={median:.2f}  MAD={mad:.2f}  "
              f"p95={p95:.2f}  p99={p99:.2f}  max={noise_significances.max():.2f}")

        n_for_tail = max(1, int(len(noise_significances) * 0.01))
        print(f"\nCAUTION: p99 is determined by roughly the top {n_for_tail} values out of "
              f"{len(noise_significances)} -- a small handful of unusually noisy/outlier-prone "
              f"stars can dominate this number on their own. If p99 is far above the median "
              f"(a heavy right tail, like here), treat the percentile-based suggestion below "
              f"as fragile and prefer the MAD-based one, or increase --sample substantially.")

        suggestion_percentile = p99 + 1.0
        suggestion_mad = median + 6 * mad  # ~6 MAD is a conservative, outlier-robust margin

        print(f"\nOption A -- percentile-based (fragile if tail is heavy): {suggestion_percentile:.2f}")
        print(f"Option B -- MAD-based (robust to a few extreme outlier stars): {suggestion_mad:.2f}")
        print("If these two disagree substantially, the gap itself tells you the tail is being "
              "driven by a small number of unusually noisy stars rather than the bulk of your "
              "sample -- prefer Option B in that case.")

        for label, suggestion in [("Option A", suggestion_percentile), ("Option B", suggestion_mad)]:
            frac = (real_significances > suggestion).mean() if len(real_significances) else float("nan")
            print(f"  {label} ({suggestion:.2f}) -> {100*frac:.1f}% of real sample would get TLS-refined")
    else:
        print("No successful noise samples -- check failures and try again.")


if __name__ == "__main__":
    main()