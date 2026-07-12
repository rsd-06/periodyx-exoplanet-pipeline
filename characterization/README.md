# Characterization

This module is responsible for extracting physical constraints and geometric parameters from detected transit signals. Once a periodic signal is found, the signal must be mathematically described before it can be classified.

## Key Components

- **`trapezoid_fit.py`**: The core fitting engine. It uses `lmfit` to perform non-linear least squares optimization, fitting a physical trapezoid model to the phase-folded light curve. 
    - **Why Trapezoids?** True planetary transits are 'U-shaped' due to the planet's spherical shadow traversing the stellar disk, which closely approximates a trapezoid (with slanted sides representing ingress/egress). Eclipsing binaries are often 'V-shaped' (grazing). Fitting a trapezoid allows us to extract explicit physical parameters: `depth`, `duration` (t_tot), and `ingress_fraction`.
- **Outputs**: Returns a dictionary of optimized parameters (depth, t_tot, t_in, t_flat) and their associated uncertainties, which are later consumed by the `features` module to engineer inputs for the classifier.
