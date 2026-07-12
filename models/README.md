# Models

This directory stores the serialized, production-ready machine learning models.

## Key Files

- **`exoplanet_classifier.joblib`**: The final, trained Two-Stage Bootstrap Ensemble (V5) saved via `joblib`. 
    - This file is automatically loaded by `main.py` when the FastAPI server boots. 
    - It contains both the Stage 1 and Stage 2 XGBoost models, the label encoders, the explicitly mapped class arrays (`stage1_classes_`, `stage2_classes_`), and the optimal threshold configurations.
