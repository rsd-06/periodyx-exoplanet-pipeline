"""
Trains the ExoplanetClassifier on the feature table built by
scripts/build_training_set.py, evaluates it with stratified k-fold cross
validation, and saves the trained model to disk.

This step is FAST -- XGBoost on a few thousand rows x ~10 features trains
in seconds on a laptop CPU. No GPU needed. This is not the bottleneck.

Usage:
    python3 scripts/train_classifier.py \
        --features data/training_features.csv \
        --model-out models/exoplanet_classifier.joblib
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from classification.classifier import ExoplanetClassifier, LABEL_CLASSES

FEATURE_COLUMNS = [
    "depth", "t_tot_hours", "t_in_hours", "flat_bottom_hours",
    "ingress_fraction", "period", "detection_significance",
    "odd_even_depth_diff", "secondary_eclipse_depth",
    "depth_snr",
]
# NOTE: secondary_eclipse_phase, n_signals_detected, period_corrected are
# intentionally excluded from v2 training. These columns exist in the CSV
# but were filled with constants (0.5 / 1 / 0) for all 7,449 v1-era rows,
# so they carry zero information for the classifier. They will be re-added
# to FEATURE_COLUMNS once the training set is rebuilt with v2 pipeline code
# (requires re-downloading Kepler light curves — currently blocked by WDAC
# policy on this machine; see README Section 8 for details).



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/training_features.csv")
    ap.add_argument("--model-out", default="models/exoplanet_classifier.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    n_before = len(df)

    # Drop only rows where the TARGET LABEL is missing.
    # Feature NaNs are legitimate "unknown" values (e.g. secondary_eclipse_phase
    # when only 60-day synthetic data is used, or ingress_fraction on very noisy
    # stars). We impute with per-column medians / flag-zeros so no training
    # examples are wasted.
    df = df.dropna(subset=["label"])

    # Per-column imputation strategy:
    #   - period_corrected, n_signals_detected: fill 0 / 1 (conservative defaults)
    #   - secondary_eclipse_phase: fill 0.5 (circular orbit assumption)
    #   - everything else: fill column median (center of distribution)
    fill_defaults = {
        "period_corrected": 0,
        "n_signals_detected": 1,
        "secondary_eclipse_phase": 0.5,
    }
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = fill_defaults.get(col, 0.0)
        elif col in fill_defaults:
            df[col] = df[col].fillna(fill_defaults[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    print(f"Loaded {n_before} rows, {len(df)} usable after NaN imputation.")
    print("Class counts:\n", df["label"].value_counts().to_string())

    X = df[FEATURE_COLUMNS]
    y = df["label"]


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    clf = ExoplanetClassifier()

    # Class weighting: address the known blend/other underperformance.
    # XGBoost handles class imbalance via sample_weight in .fit().
    # We compute inverse-frequency weights so rarer classes get proportionally
    # more influence during training without artificially inflating their count.
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Cross-validated estimate on the training split before final fit.
    # cross_val_score works on the raw XGBoost model, which (as of recent
    # XGBoost versions) requires integer-encoded labels -- encode here only
    # for this CV estimate; the real .fit() below uses the wrapper, which
    # handles string labels internally.
    from sklearn.preprocessing import LabelEncoder
    y_train_encoded = LabelEncoder().fit_transform(y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf.model, X_train, y_train_encoded, cv=cv, scoring="f1_macro")
    print(f"\n5-fold CV F1-macro: {scores.mean():.3f} +/- {scores.std():.3f}")

    clf.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = clf.predict(X_test)
    print("\nHeld-out test set report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=true, cols=predicted):")
    labels_present = sorted(y.unique())
    print(pd.DataFrame(
        confusion_matrix(y_test, y_pred, labels=labels_present),
        index=labels_present, columns=labels_present,
    ))

    print("\nFeature importances:")
    for k, v in sorted(clf.feature_importance().items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {v:.4f}")

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    clf.save(args.model_out)
    print(f"\nSaved trained model to {args.model_out}")


if __name__ == "__main__":
    main()
