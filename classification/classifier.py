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
    def __init__(self, n_estimators=200, max_depth=4, learning_rate=0.05):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
        )
        self.is_fitted = False
        self.feature_names_ = None
        self.label_encoder_ = LabelEncoder()

    def fit(self, X, y):
        """X: list/array of feature dicts or DataFrame. y: array of string labels
        (e.g. 'transit', 'eclipsing_binary'). Internally encoded to integers
        for XGBoost, transparently decoded back on predict()."""
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        self.feature_names_ = list(X_df.columns)
        y_encoded = self.label_encoder_.fit_transform(np.asarray(y))
        self.model.fit(X_df, y_encoded)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError(
                "Classifier not yet trained -- awaiting ISRO curated dataset. "
                "Pipeline interface is verified end-to-end via run_demo.py."
            )
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        y_pred_encoded = self.model.predict(X_df)
        return self.label_encoder_.inverse_transform(y_pred_encoded)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Classifier not yet trained.")
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        return self.model.predict_proba(X_df)

    @property
    def classes_(self):
        """Original string class labels, in the order predict_proba columns
        are returned (i.e. matches self.label_encoder_.classes_)."""
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
        obj.model = payload["model"]
        obj.feature_names_ = payload["feature_names_"]
        obj.is_fitted = payload["is_fitted"]
        obj.label_encoder_ = payload["label_encoder_"]
        return obj
