"""
Uncertainty quantification layer.

Three independent sources of uncertainty are combined in this pipeline:
  1. Detection significance  -> BLS power / TLS SDE (detection/periodic_search.py)
  2. Parameter uncertainty   -> lmfit covariance-derived stderr (characterization/trapezoid_fit.py)
  3. Classification confidence -> bootstrap ensemble variance (this module)

Bootstrap ensembling is used instead of MC Dropout because the classifier
is tree-based (XGBoost), not a neural network -- this is the correct
uncertainty technique for that model family.
"""

import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def train_bootstrap_ensemble(X_train, y_train, n_bootstrap=20, xgb_params=None,
                             sample_weight=None):
    """Train an ensemble of bootstrap-resampled classifiers.

    Each bootstrap iteration resamples rows (with replacement). If
    ``sample_weight`` is supplied, the resampled weights are passed directly
    to ``clf.fit`` so class-balance correction is preserved inside every
    bootstrap model — not just the aggregate.

    Returns
    -------
    models : list of fitted xgb.XGBClassifier
    classes_ : array of original string class labels
    """
    import pandas as pd
    xgb_params = xgb_params or dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss",
    )

    X_train_df = pd.DataFrame(X_train) if not hasattr(X_train, "columns") else X_train

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y_train))

    models = []
    print(f"Training bootstrap ensemble of {n_bootstrap} models...")
    for i in range(n_bootstrap):
        idx = resample(np.arange(len(X_train_df)), random_state=i)
        X_boot = X_train_df.iloc[idx]
        y_boot = y_encoded[idx]
        sw_boot = sample_weight[idx] if sample_weight is not None else None
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_boot, y_boot, sample_weight=sw_boot)
        models.append(clf)

    return models, encoder.classes_
