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


def bootstrap_classifier_uncertainty(X_train, y_train, X_query, n_bootstrap=20,
                                      xgb_params=None):
    """Train an ensemble of bootstrap-resampled classifiers and report the
    mean and standard deviation of predicted class probabilities for X_query.

    Returns
    -------
    mean_proba, std_proba : arrays of shape (n_query, n_classes)
    classes : array of original string class labels, in column order
    """
    import pandas as pd
    xgb_params = xgb_params or dict(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        eval_metric="logloss",
    )

    X_train_df = pd.DataFrame(X_train) if not hasattr(X_train, "columns") else X_train
    X_query_df = pd.DataFrame(X_query) if not hasattr(X_query, "columns") else X_query

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(np.asarray(y_train))

    predictions = []
    for i in range(n_bootstrap):
        X_boot, y_boot = resample(X_train_df, y_encoded, random_state=i)
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_boot, y_boot)
        predictions.append(clf.predict_proba(X_query_df))

    predictions = np.array(predictions)  # shape (n_bootstrap, n_query, n_classes)
    mean_proba = predictions.mean(axis=0)
    std_proba = predictions.std(axis=0)
    return mean_proba, std_proba, encoder.classes_
