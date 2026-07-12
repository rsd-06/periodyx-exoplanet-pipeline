# Detection

This module scans a cleaned, detrended light curve for periodic transit signals. It is the engine that actually "finds" the planets before they are analyzed.

## Key Components

- **`periodic.py`**: Implements a tiered search strategy.
    - **Tier 1 (BLS)**: Uses Astropy's `BoxLeastSquares`. This algorithm is fast and highly optimized for scanning a wide grid of periods and durations. It acts as our first pass. If the Signal Detection Efficiency (SDE) surpasses our calibrated noise floor, it triggers Tier 2.
    - **Tier 2 (TLS - Optional)**: `TransitLeastSquares` is more sensitive to true planet shapes (incorporating limb-darkening) but computationally expensive. It is used to refine the period and epoch of a candidate found by BLS.
