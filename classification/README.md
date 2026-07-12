# Classification

This module contains the machine learning models that decide the final disposition of a Threshold Crossing Event (TCE).

## Key Components

- **`classifier.py`**: The original XGBoost baseline (`ExoplanetClassifier`). It wraps `xgboost`, implements the bootstrap ensemble logic (`fit_ensemble`), and handles probability mean/standard deviation calculations for uncertainty quantification.
- **`two_stage_classifier.py`**: The V5 production architecture. Realizing that separating a real planet from noise is fundamentally easier than separating a grazing eclipsing binary from a background blend, we implemented a hierarchical model.
    - **Stage 1**: Binary classifier (Transit vs Non-Transit).
    - **Stage 2**: Multi-class classifier (Eclipsing Binary vs Blend vs Other). Trained exclusively on non-transit examples.
    - **`predict_proba`**: Recombines the probabilities mathematically (e.g. $P(Blend) = P(Not Transit) \times P_{stage2}(Blend)$) ensuring that the model output is perfectly continuous and semantically verified.
