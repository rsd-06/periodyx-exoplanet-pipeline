# Scripts

This directory contains standalone execution scripts used to build, calibrate, train, and test the pipeline offline. 

## Key Scripts

- **`build_training_set.py`**: Orchestrates the downloading of raw MAST data, runs the pipeline on thousands of stars concurrently (multiprocessing), and exports the results to `data/training_features.csv`.
- **`calibrate_threshold.py`**: Phase-scrambles real Kepler light curves to empirically determine the BLS noise floor. Crucial for avoiding the "look-elsewhere" effect.
- **`train_classifier.py`**: The training pipeline. Drops rows missing critical physical data (MNAR leakage prevention), runs hyperparameter tuning via Optuna, trains the ensemble, and saves it to `models/`.
- **`verify_semantics.py`**: A manual sanity-check script that forces the Two-Stage classifier to predict on known-label stars, verifying that `predict_proba` probability arrays perfectly map to the correct semantic string labels without implicit sorting drift.
