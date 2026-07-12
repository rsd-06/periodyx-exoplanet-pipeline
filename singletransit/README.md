# Single Transit Detection

This module exists as a parallel track to the main periodic detection pipeline.

## Key Components

- **`scanner.py`**: Not all planets transit multiple times during an observation window (especially long-period, habitable-zone planets). This module runs a sliding-window matched filter across the entire light curve to detect significant, isolated V/U-shaped dips that lack a repeating period. 
    - If found, it flags them as `single_transit_candidates`, which triggers a specific alert in the UI for human review.
