# Preprocessing

This module is responsible for cleaning raw archival light curves. The raw data contains immense instrumental noise (thermal settling, thruster firings) and stellar noise (starspots, pulsations) that must be removed.

## Key Components

- **`clean.py`**: 
    - Removes NaN flux values and normalizes the light curve.
    - Uses `wotan` to apply a time-windowed **biweight filter**. This sliding window dynamically tracks and flattens long-term stellar variability while rigorously preserving the sharp, short-term dips characteristic of planetary transits.
