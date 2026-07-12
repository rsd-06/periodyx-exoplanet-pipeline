# Uncertainty

This module provides the statistical framework for interpreting the confidence of the machine learning model.

## Key Components

- **`bootstrap.py`**: Rather than training one XGBoost model, we train an ensemble of $N$ models (default $N=20$). Each model is trained on a different random resample (with replacement) of the training dataset.
    - At inference time, a light curve is passed through all 20 models. 
    - The mean of the predictions becomes the final probability, and the standard deviation across the 20 models provides a robust $\pm \sigma$ uncertainty bound, which is surfaced directly to the user interface.
