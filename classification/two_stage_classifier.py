"""
Two-stage exoplanet classifier (V5).

Architecture
------------
Stage 1 : transit vs. everything-else  (binary XGBoost ensemble)
Stage 2 : eclipsing_binary / blend / other  (3-class XGBoost ensemble,
           trained ONLY on non-transit rows)

Final 4-class probabilities are reconstructed from both stages:
    P(transit)          = Stage1.P(transit)
    P(eclipsing_binary) = (1 - Stage1.P(transit)) * Stage2.P(eclipsing_binary)
    P(blend)            = (1 - Stage1.P(transit)) * Stage2.P(blend)
    P(other)            = (1 - Stage1.P(transit)) * Stage2.P(other)

The API contract (predict_proba returns (mean_proba, std_proba) arrays with
the same 4-class layout) is identical to ExoplanetClassifier — callers don't
need to know which model architecture is loaded.

Uncertainty
-----------
Each stage is a bootstrap ensemble (N=20 by default), so mean + std are
available for both the transit/not-transit split AND the 3-class split.
The combined std is propagated via first-order approximation.
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight


STAGE1_CLASSES = ["not_transit", "transit"]
STAGE2_CLASSES = ["eclipsing_binary", "blend", "other"]
ALL_CLASSES    = ["transit", "eclipsing_binary", "blend", "other"]


class TwoStageClassifier:
    """Two-stage decomposed exoplanet classifier.

    Exposes the same ``predict_proba(X) -> (mean_proba, std_proba)`` and
    ``save`` / ``load`` interface as ``ExoplanetClassifier``.
    """

    def __init__(self, xgb_params=None):
        self.xgb_params = xgb_params or dict(
            n_estimators=204, max_depth=8, learning_rate=0.11,
            eval_metric="logloss",
        )
        self.stage1_models = []   # binary: transit / not_transit
        self.stage2_models = []   # 3-class: EB / blend / other
        self.feature_names_ = None
        self.is_fitted = False
        # fixed class ordering (matches FEATURE_COLUMNS label ordering)
        self._classes = np.array(ALL_CLASSES)

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit_ensemble(self, X, y, n_bootstrap=20, sample_weight=None):
        """Train both stages as bootstrap ensembles.

        Parameters
        ----------
        X : DataFrame of shape (n, n_features)
        y : array of string labels in ALL_CLASSES
        sample_weight : array of per-sample weights (balanced by class).
            Applied inside every bootstrap iteration.
        """
        import pandas as pd
        from uncertainty.bootstrap import train_bootstrap_ensemble

        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X
        self.feature_names_ = list(X_df.columns)
        y_arr = np.asarray(y)
        sw = np.asarray(sample_weight) if sample_weight is not None else None

        # ── Stage 1: transit vs. not_transit ─────────────────────────────────
        y1 = np.where(y_arr == "transit", "transit", "not_transit")
        sw1 = sw  # same rows, same weights
        print(f"\n[TwoStage] Stage 1 — transit vs. not_transit (N={n_bootstrap})")
        self.stage1_models, _ = train_bootstrap_ensemble(
            X_df, y1, n_bootstrap=n_bootstrap,
            xgb_params=self.xgb_params, sample_weight=sw1,
        )

        # ── Stage 2: EB / blend / other (non-transit rows only) ──────────────
        mask2 = y_arr != "transit"
        X2 = X_df[mask2].reset_index(drop=True)
        y2 = y_arr[mask2]
        sw2 = sw[mask2] if sw is not None else None
        # Recompute balanced weights on the subset so they're correct
        sw2_balanced = compute_sample_weight("balanced", y=y2)
        if sw2 is not None:
            # Multiply original weights by rebalanced subset weights
            sw2 = sw2 * sw2_balanced / sw2_balanced.mean()
        else:
            sw2 = sw2_balanced

        print(f"[TwoStage] Stage 2 — EB/blend/other on {mask2.sum()} non-transit rows (N={n_bootstrap})")
        self.stage2_models, _ = train_bootstrap_ensemble(
            X2, y2, n_bootstrap=n_bootstrap,
            xgb_params=self.xgb_params, sample_weight=sw2,
        )
        self.is_fitted = True
        return self

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_proba(self, X):
        """Returns (mean_proba, std_proba) of shape (n_samples, 4).

        Column order: transit, eclipsing_binary, blend, other
        """
        if not self.is_fitted:
            raise RuntimeError("TwoStageClassifier not yet trained.")
        import pandas as pd
        X_df = pd.DataFrame(X) if not hasattr(X, "columns") else X

        n = len(X_df)

        # Stage 1 probabilities across ensemble
        # Each model returns shape (n, 2): [not_transit, transit]
        s1_preds = []
        for m in self.stage1_models:
            proba = m.predict_proba(X_df)   # (n, 2)
            # find transit column index
            try:
                transit_idx = list(m.classes_).index(1)  # encoded as 1
            except ValueError:
                transit_idx = 1
            s1_preds.append(proba[:, transit_idx])       # (n,)
        s1_arr = np.array(s1_preds)   # (N_boot, n)
        p_transit_mean = s1_arr.mean(axis=0)              # (n,)
        p_transit_std  = s1_arr.std(axis=0)               # (n,)
        p_not_transit_mean = 1 - p_transit_mean

        # Stage 2 probabilities across ensemble
        # Each model returns shape (n, 3): [EB, blend, other]
        s2_preds = []
        for m in self.stage2_models:
            proba = m.predict_proba(X_df)   # (n, 3)
            s2_preds.append(proba)
        s2_arr = np.array(s2_preds)         # (N_boot, n, 3)
        s2_mean = s2_arr.mean(axis=0)       # (n, 3)  [EB, blend, other]
        s2_std  = s2_arr.std(axis=0)        # (n, 3)

        # Reconstruct 4-class output — column order: transit, EB, blend, other
        mean_proba = np.zeros((n, 4))
        std_proba  = np.zeros((n, 4))

        mean_proba[:, 0] = p_transit_mean
        std_proba[:, 0]  = p_transit_std

        for i in range(3):  # EB=1, blend=2, other=3
            mean_proba[:, i+1] = p_not_transit_mean * s2_mean[:, i]
            # Error propagation: var(A*B) ≈ (A*σ_B)² + (B*σ_A)² for uncorrelated
            var_i = ((p_not_transit_mean * s2_std[:, i]) ** 2
                     + (s2_mean[:, i]   * p_transit_std) ** 2)
            std_proba[:, i+1] = np.sqrt(var_i)

        # Renormalize rows to sum to 1 (small numerical drift only)
        row_sums = mean_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        mean_proba = mean_proba / row_sums

        return mean_proba, std_proba

    def predict(self, X):
        mean_proba, _ = self.predict_proba(X)
        idx = np.argmax(mean_proba, axis=1)
        return self._classes[idx]

    @property
    def classes_(self):
        return self._classes

    def feature_importance(self):
        """Average feature importance across Stage 1 ensemble."""
        if not self.is_fitted or not self.stage1_models:
            return None
        importances = np.array([m.feature_importances_
                                 for m in self.stage1_models]).mean(axis=0)
        return dict(zip(self.feature_names_, importances))

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path):
        import joblib
        if not self.is_fitted:
            raise RuntimeError("Cannot save an untrained TwoStageClassifier.")
        joblib.dump({
            "_type": "TwoStageClassifier",
            "stage1_models": self.stage1_models,
            "stage2_models": self.stage2_models,
            "feature_names_": self.feature_names_,
            "xgb_params": self.xgb_params,
            "is_fitted": self.is_fitted,
        }, path)

    @classmethod
    def load(cls, path):
        import joblib
        payload = joblib.load(path)
        obj = cls(xgb_params=payload.get("xgb_params"))
        obj.stage1_models = payload["stage1_models"]
        obj.stage2_models = payload["stage2_models"]
        obj.feature_names_ = payload["feature_names_"]
        obj.is_fitted = payload["is_fitted"]
        return obj
