# Feature Engineering

This module bridges the gap between the raw physical outputs of the detection/characterization modules and the XGBoost classifier.

## Key Components

- **`extract.py`**: Calculates the final 1D feature array passed to the model.
    - **Geometric Features**: Calculates the SNR, secondary eclipse depth (checking if there is a second dip at phase 0.5, a strong indicator of an eclipsing binary), and odd/even transit depth differences (to catch blended binaries at 2x the true period).
    - **Physical Priors**: Passes through the archival `koi_srad`, `koi_steff`, `koi_kepmag`, and centroid tracking.
    - **Stellar Density Ratio (`compute_stellar_density_ratio`)**: The V5 physics constraint. Uses Kepler's Third Law to calculate the stellar density implied strictly by the transit geometry, and compares it to the star's actual measured density. This single feature breaks the degeneracy between grazing EBs and real transits.
