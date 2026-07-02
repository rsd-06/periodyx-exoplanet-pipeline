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

# ── Features ─────────────────────────────────────────────────────────────
# We use ONLY the 13 purely independent features extracted directly from the
# raw lightcurves by our pipeline.
#
# IMPORTANT: NASA's fpflag_* columns and `koi_prad` are EXCLUDED. 
# The fpflags are the literal answer key. `koi_prad` is subtly leaky because 
# its presence and error bars correlate with NASA's confidence in the disposition.
FEATURE_COLUMNS = [
    "depth", "t_tot_hours", "t_in_hours", "flat_bottom_hours",
    "ingress_fraction", "period", "detection_significance",
    "odd_even_depth_diff", "secondary_eclipse_depth", "secondary_eclipse_phase",
    "depth_snr", "n_signals_detected", "period_corrected",
    # v4 additions:
    "koi_srad", "koi_steff", "koi_slogg", "koi_kepmag",
    "centroid_offset_magnitude", "reconstructed_prad"
]

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
    
    # ── Drop Missing Data to Prevent MNAR Leakage ───────────────────────
    # Missingness in stellar parameters or centroid data correlates heavily
    # with the 'FALSE POSITIVE' disposition. If we impute these (or 0-fill
    # centroid data), XGBoost will learn "missing = false positive" or "missing = planet",
    # which is exactly the data leakage we want to avoid.
    # We must explicitly drop rows missing these fundamental physics requirements.
    req_cols = ["koi_srad", "koi_steff", "koi_slogg", "koi_kepmag", "centroid_offset_magnitude"]
    # Only drop them if they actually exist in the CSV (during v4)
    if all(c in df.columns for c in req_cols):
        n_before_drop = len(df)
        dist_before = df["label"].value_counts()
        df = df.dropna(subset=req_cols)
        n_dropped = n_before_drop - len(df)
        dist_after = df["label"].value_counts()
        
        print(f"\n--- DATASET PRUNING (v4) ---")
        print(f"Dropped {n_dropped} rows ({n_dropped/n_before_drop*100:.1f}%) missing stellar or centroid data.")
        print("Class distribution shift (Before vs After drop):")
        df_dist = pd.DataFrame({"Before": dist_before, "After": dist_after, "Loss %": ((dist_before - dist_after) / dist_before * 100).round(1)})
        print(df_dist.to_string())
        print("----------------------------\n")
        
    # Reconstruct planetary radius using our independent depth & stellar radius.
    # *CRITICAL ASSUMPTION*: For true planets, depth is ~ (Rp/Rs)^2.
    # For blends (background eclipsing binaries), the transit depth is diluted by
    # the target star. Therefore, for blends, `reconstructed_prad` represents
    # an *effective* radius and will be systematically underestimated. This is
    # actually a useful signal: an anomalously small radius for a deep, box-shaped
    # transit is a strong blend indicator.
    if "koi_srad" in df.columns and "depth" in df.columns:
        # 1 Solar Radius = 109.2 Earth Radii
        df["reconstructed_prad"] = np.sqrt(df["depth"].clip(lower=0)) * df["koi_srad"] * 109.2
        # Extreme value suppression (there are some huge stars)
        df["reconstructed_prad"] = np.log1p(df["reconstructed_prad"])

    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = FILL_DEFAULTS.get(col, 0.0)
        elif col in FILL_DEFAULTS:
            df[col] = df[col].fillna(FILL_DEFAULTS[col])
        else:
            df[col] = df[col].fillna(df[col].median())

    print(f"Loaded {n_before} rows, {len(df)} usable after imputation.")
    print("Final Class counts:\n", df["label"].value_counts().to_string())

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

    # ── Ablation Study (v3 vs v4) ──────────────────────────────────────────
    # Train a baseline on the identical row set but *without* the v4 features,
    # to prove how much gain comes from physics vs dataset pruning.
    v3_features = FEATURE_COLUMNS[:13] # The first 13 are the v3 features
    print(f"\n--- ABLATION STUDY: v3 Baseline on Pruned Dataset ---")
    clf_v3 = ExoplanetClassifier() # default params
    sw_v3 = compute_sample_weight(class_weight="balanced", y=y_train)
    
    le_v3 = LabelEncoder()
    y_train_enc_v3 = le_v3.fit_transform(y_train)
    cv_v3 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores_v3 = cross_val_score(clf_v3.model, X_train[v3_features], y_train_enc_v3, cv=cv_v3, scoring="f1_macro")
    
    clf_v3.fit(X_train[v3_features], y_train, sample_weight=sw_v3)
    y_pred_v3 = clf_v3.predict(X_test[v3_features])
    from sklearn.metrics import f1_score
    test_f1_v3 = f1_score(y_test, y_pred_v3, average="macro")
    
    print(f"v3 (13 features) 5-fold CV F1-macro: {scores_v3.mean():.3f}")
    print(f"v3 (13 features) Test F1-macro:      {test_f1_v3:.3f}")
    print("-----------------------------------------------------\n")


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
