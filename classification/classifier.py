"""
Classification stage.

IMPORTANT: This module defines the classifier ARCHITECTURE and interface
only. It is intentionally NOT trained on real labels yet, since ISRO's
curated dataset is only released upon selection.

When the curated dataset arrives, the only required change is calling
`.fit(X_curated, y_curated)` -- no architecture or feature changes needed,
because the feature vector (from features/extract.py) is dataset-agnostic.

For now, `label_classes` defines the planned output taxonomy, and the model
can optionally be exercised on small synthetic/public-proxy labels purely to
verify the interface works end-to-end (see run_demo.py).
"""

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

LABEL_CLASSES = ["transit", "eclipsing_binary", "blend", "other"]


class ExoplanetClassifier:
    def __init__(
        self,
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=1.0,
        colsample_bytree=1.0,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
    ):
        self.xgb_params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            eval_metric="logloss",
        )
        self.model = xgb.XGBClassifier(**self.xgb_params)
        self.models = []  # For bootstrap ensemble
        self.is_fitted = False
        self.feature_names_ = None
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y, sample_weight=None):
        """Standard single-model fit."""
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        self.feature_names_ = list(X_df.columns)
        y_encoded = self.label_encoder_.fit_transform(np.asarray(y))
        self.model.fit(X_df, y_encoded, sample_weight=sample_weight)
        self.models = [self.model]
        self.is_fitted = True
        return self

    def fit_ensemble(self, X, y, n_bootstrap=20, sample_weight=None):
        """Train an ensemble of N bootstrap-resampled classifiers.

        Parameters
        ----------
        sample_weight : array-like, optional
            Per-sample weights (e.g. from compute_sample_weight). Applied
            inside every bootstrap iteration so class-balance correction is
            not discarded.
        """
        from uncertainty.bootstrap import train_bootstrap_ensemble
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        self.feature_names_ = list(X_df.columns)
        self.label_encoder_.fit(np.asarray(y))

        sw = np.asarray(sample_weight) if sample_weight is not None else None
        models, classes_ = train_bootstrap_ensemble(
            X_df, y, n_bootstrap=n_bootstrap,
            xgb_params=self.xgb_params,
            sample_weight=sw,
        )
        self.models = models
        self.model = models[0]
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Classifier not yet trained.")
        mean_proba, _ = self.predict_proba(X)
        y_pred_encoded = np.argmax(mean_proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        """Returns (mean_proba, std_proba). std_proba is 0 if no ensemble."""
        if not self.is_fitted:
            raise RuntimeError("Classifier not yet trained.")
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        
        preds = []
        for m in self.models:
            preds.append(m.predict_proba(X_df))
            
        preds = np.array(preds)
        mean_proba = preds.mean(axis=0)
        std_proba = preds.std(axis=0) if len(self.models) > 1 else np.zeros_like(mean_proba)
        
        return mean_proba, std_proba

    @property
    def classes_(self):
        """Original string class labels."""
        return self.label_encoder_.classes_

    def feature_importance(self):
        if not self.is_fitted:
            return None
        return dict(zip(self.feature_names_, self.model.feature_importances_))

    def save(self, path):
        """Persist the trained model + feature schema + label encoder to disk."""
        import joblib
        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained classifier.")
        joblib.dump({
            "models": self.models,
            "model": self.model,
            "feature_names_": self.feature_names_,
            "is_fitted": self.is_fitted,
            "label_encoder_": self.label_encoder_,
        }, path)

    @classmethod
    def load(cls, path):
        """Load a previously trained classifier from disk."""
        import joblib
        payload = joblib.load(path)
        obj = cls()
        obj.model = payload.get("model", payload.get("models", [None])[0])
        obj.models = payload.get("models", [obj.model])
        obj.feature_names_ = payload["feature_names_"]
        obj.is_fitted = payload["is_fitted"]
        obj.label_encoder_ = payload["label_encoder_"]
        return obj
