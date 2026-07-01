"""
Trains the ExoplanetClassifier on the feature table built by
scripts/build_training_set.py, evaluates it with stratified k-fold cross
validation, and saves the trained model to disk.

v3 additions:
  - 5 new NASA vetting flag features (fpflag_nt/ss/co/ec, koi_prad)
  - Optuna hyperparameter search (100 trials, ~30-60 mins)

Usage:
    python3 scripts/train_classifier.py \
        --features data/training_features.csv \
        --model-out models/exoplanet_classifier.joblib

    # Skip Optuna (use sensible defaults, trains in seconds):
    python3 scripts/train_classifier.py --no-optuna
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from classification.classifier import ExoplanetClassifier, LABEL_CLASSES

# ── v2 pipeline features ────────────────────────────────────────────────────
FEATURE_COLUMNS_V2 = [
    "depth", "t_tot_hours", "t_in_hours", "flat_bottom_hours",
    "ingress_fraction", "period", "detection_significance",
    "odd_even_depth_diff", "secondary_eclipse_depth", "secondary_eclipse_phase",
    "depth_snr", "n_signals_detected", "period_corrected",
]

# ── v3 additions: NASA vetting flags ────────────────────────────────────────
# What each flag means (NASA computed, zero extra compute):
#   fpflag_co  = centroid offset flag  — flux centroid shifts during transit,
#                meaning the dip is coming from a NEARBY source, not the target.
#                This is the strongest blend discriminator that exists.
#   fpflag_ss  = secondary eclipse flag — a significant secondary eclipse was
#                seen, pointing strongly to an eclipsing binary, not a planet.
#   fpflag_ec  = ephemeris contamination — the period/epoch matches a known EB
#                elsewhere on the detector (another blend mechanism).
#   fpflag_nt  = not transit-like — the shape doesn't match a box transit at all
#                (too asymmetric, V-shaped, etc.). Strong "other" indicator.
#   koi_prad   = fitted planet radius (Earth radii). Planets > ~15 R_earth are
#                almost always EBs or giant mis-classified blends.
FEATURE_COLUMNS_V3 = FEATURE_COLUMNS_V2 + [
    "fpflag_co", "fpflag_ss", "fpflag_ec", "fpflag_nt", "koi_prad",
]

FEATURE_COLUMNS = FEATURE_COLUMNS_V3  # active set

FILL_DEFAULTS = {
    "period_corrected": 0,
    "n_signals_detected": 1,
    "secondary_eclipse_phase": 0.5,
    "fpflag_nt": 0,
    "fpflag_ss": 0,
    "fpflag_co": 0,
    "fpflag_ec": 0,
}


def load_and_prepare(features_path):
    df = pd.read_csv(features_path)
    n_before = len(df)
    df = df.dropna(subset=["label"])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = FILL_DEFAULTS.get(col, 0.0)
        elif col in FILL_DEFAULTS:
            df[col] = df[col].fillna(FILL_DEFAULTS[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    print(f"Loaded {n_before} rows, {len(df)} usable after NaN imputation.")
    print("Class counts:\n", df["label"].value_counts().to_string())

    # Log-transform koi_prad: distribution is extremely right-skewed
    # (range 0.08 → 200,000 R_earth). Log scale makes it useful to the model.
    if "koi_prad" in df.columns:
        df["koi_prad"] = np.log1p(df["koi_prad"].clip(lower=0))

    return df


def run_optuna_search(X_train, y_train, n_trials=100):
    """
    Optuna hyperparameter search.

    What it does:
        Optuna runs 100 experiments (called 'trials'). In each trial it picks
        a different combination of XGBoost settings (depth, learning rate, etc.)
        and measures how good that combination is using 3-fold cross-validation
        F1-macro. After all trials it returns the best combination found.

    Why not just grid search?
        A full grid over 5 parameters with 5 values each = 3,125 combinations.
        Optuna uses a smart Bayesian algorithm (TPE) that learns from previous
        trials — it tries promising regions of the search space first, so it
        finds near-optimal settings with far fewer trials than brute force.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    le = LabelEncoder()
    y_enc = le.fit_transform(y_train)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1000),
            "max_depth":         trial.suggest_int("max_depth", 3, 9),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        }
        clf = ExoplanetClassifier(**params)
        scores = cross_val_score(clf.model, X_train, y_enc, cv=cv,
                                 scoring="f1_macro", n_jobs=1)
        return scores.mean()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nOptuna best F1-macro (3-fold): {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k:22s} = {v}")
    return study.best_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/training_features.csv")
    ap.add_argument("--model-out", default="models/exoplanet_classifier.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--no-optuna", action="store_true",
                    help="Skip Optuna search and use default XGBoost parameters. "
                         "Trains in seconds instead of ~30-60 mins.")
    ap.add_argument("--optuna-trials", type=int, default=100,
                    help="Number of Optuna trials (default 100, ~30-60 mins).")
    args = ap.parse_args()

    df = load_and_prepare(args.features)
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    # ── Hyperparameter search ──────────────────────────────────────────────
    best_params = {}
    if not args.no_optuna:
        print(f"\nRunning Optuna search ({args.optuna_trials} trials)...")
        print("This takes ~30-60 mins. Use --no-optuna to skip.\n")
        best_params = run_optuna_search(X_train, y_train,
                                        n_trials=args.optuna_trials)
    else:
        print("\nSkipping Optuna (--no-optuna set). Using default XGBoost params.")

    # ── Final model ────────────────────────────────────────────────────────
    clf = ExoplanetClassifier(**best_params)
    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    # Quick CV estimate on the training split
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf.model, X_train, y_train_enc, cv=cv,
                             scoring="f1_macro")
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
