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
    "odd_even_depth_diff", "secondary_eclipse_depth", "depth_snr",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/training_features.csv")
    ap.add_argument("--model-out", default="models/exoplanet_classifier.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    n_before = len(df)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])
    print(f"Loaded {n_before} rows, {len(df)} usable after dropping NaNs.")
    print("Class counts:\n", df["label"].value_counts().to_string())

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    clf = ExoplanetClassifier()

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

    clf.fit(X_train, y_train)

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
