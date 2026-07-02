# PeriodyX — Exoplanet Detection Pipeline

ISRO BAH 2026 — Problem Statement #07: AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**Team:** PeriodyX  
**Goal:** Build a robust, interpretable, end-to-end machine learning pipeline to detect exoplanets from raw astronomical light curves, strictly avoiding data leakage and prioritizing physical priors over black-box shape matching.

---

## 1. What This Project Does

This is an end-to-end pipeline that takes raw stellar brightness measurements (light curves) from NASA's Kepler/TESS missions and automatically:

1. **Detrends the signal:** Removes instrumental noise and stellar variability while preserving genuine transit signals.
2. **Searches for periodicity:** Scans for box-shaped dips that indicate a transiting planet.
3. **Fits a physical model:** Uses a trapezoid model to characterize the shape of each detected dip (depth, duration, ingress/egress time).
4. **Extracts physical priors & features:** Pulls independent stellar properties (radius, temperature) and centroid tracking (pixel-level motion) directly from archival data.
5. **Classifies detections (XGBoost):** Separates detections into four categories:
   - **transit** (real planet)
   - **eclipsing_binary** (two stars orbiting each other)
   - **blend** (contamination from a background source)
   - **other** (noise, starspots, artifacts)

**Design Philosophy:** Unlike many published systems (e.g., AstroNet) that rely on NASA proprietary pipelines to do the actual detection and feed cropped signals into black-box CNNs, PeriodyX owns the entire chain end-to-end. It uses physically interpretable features and runs on a single commodity machine without requiring a GPU.

---

## 2. Tech Stack

- **Language:** Python 3
- **Data Acquisition:** `lightkurve` (MAST archive access), `urllib` (Exoplanet Archive TAP API)
- **Detrending:** `wotan` (Biweight filter)
- **Periodic Search:** `astropy.timeseries.BoxLeastSquares`, `transitleastsquares`
- **Shape Fitting:** `lmfit` (Non-linear least squares)
- **Classification:** `xgboost`, `scikit-learn`
- **Data Handling:** `pandas`, `numpy`
- **Model Persistence:** `joblib`
- **Deployment:** Hugging Face Hub (`huggingface_hub`)

---

## 3. Pipeline Architecture

```text
Raw Light Curve (Kepler/TESS, via MAST)
        |
        v
  [1] DETRENDING (wotan, biweight filter)
        |
        v
  [2] PERIODIC SEARCH -- tiered
        BLS (fast triage, runs on every star)
              |
              v  (only if BLS significance > noise threshold)
        TLS (slow, high-sensitivity refinement)
        |
        v
  [3] SHAPE CHARACTERIZATION (lmfit)
        -> Fits depth, total duration, ingress duration, etc.
        |
        v
  [4] FEATURE ENGINEERING & PHYSICAL PRIORS (The "v4" architecture)
        -> Extracts lightcurve shapes (SNR, secondary eclipses)
        -> Merges independent stellar priors (Radius, Temp, Gravity, Mag)
        -> Computes Reconstructed Planetary Radius (Rs * sqrt(depth))
        -> Merges Centroid Motion (Pixel offset magnitudes)
        |
        v
  [5] CLASSIFICATION (XGBoost)
        -> transit / eclipsing_binary / blend / other
```

---

## 4. The Evolution of the Pipeline (Version History & Data Leakage)

Building an astronomical ML pipeline is notoriously susceptible to **data leakage**—where the model secretly learns human vetting confidence rather than actual physics. Here is how we iteratively debugged and defeated leakage to achieve our final result.

### v1: The Pure Shape Baseline
- **Approach:** Fed the model 13 pure geometric features extracted from the lightcurve (depth, duration, odd-even differences, secondary eclipses).
- **Results:** ~40-46% F1-macro score. 
- **Learning:** 1D shapes are fundamentally degenerate. A background eclipsing binary (Blend) looks mathematically identical to a small transiting planet. The model plateaued because the physical information required to separate them simply wasn't in the 1D shape.

### v2/v3: The Leakage Trap
- **Approach:** Attempted to boost performance by including NASA vetting flags (`fpflag_*`) and the NASA-computed planetary radius (`koi_prad`).
- **Results:** F1-macro skyrocketed to ~98%.
- **The Catch (Data Leakage):** We proved this was a massive leak. The `fpflag_*` columns were the literal answer key. Even worse, `koi_prad` was subtly leaky: we ran diagnostic scripts and proved that NASA only computed a tight error bar for `koi_prad` when they were *already confident* the candidate was a planet. Optuna (the hyperparameter tuner) built a massive 800-tree forest just to memorize the exact error-bar signatures of the dataset, ignoring the astrophysics entirely. 
- **Action:** Stripped out all vetting flags and `koi_prad`. F1 dropped back to the honest 46% baseline.

### v4: Physical Priors & Centroid Tracking (The Breakthrough)
- **Approach:** If we can't use NASA's confidence, we must use real physics. We upgraded the feature set to include:
  1. **Independent Stellar Priors:** Star radius (`koi_srad`), temperature (`steff`), and surface gravity.
  2. **Reconstructed Radius:** We manually computed planetary radius (`sqrt(depth) * koi_srad`) to isolate the physics while dodging NASA's vetting proxies.
  3. **Centroid Motion:** We merged `koi_dicco_mra` and `mdec` to measure if the center of light shifted during the dip (the literal definition of a Blend).
- **Leakage Prevention:** Missing data in stellar parameters correlates heavily with False Positives. If we zero-filled or imputed NaNs, XGBoost would cheat by learning "NaN = False Positive". To prevent this, we **aggressively dropped 9% of the dataset** (any row missing stellar/centroid data), guaranteeing an honest, mathematically rigorous training set.

---

## 5. Final Results (v4)

Optuna hyperparameter tuning on the clean, mathematically honest v4 dataset found a robust, regularized optimum (`n_estimators=204`, `max_depth=8`).

**Overall Held-Out Test Accuracy: 73%**

| Class | Precision | Recall | F1-Score | Improvement over v1/v3 Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Transit (Planet)** | 0.87 | 0.89 | **0.88** | **+0.26** |
| **Eclipsing Binary** | 0.81 | 0.65 | **0.72** | **+0.07** |
| **Blend** | 0.56 | 0.65 | **0.61** | **+0.28** |
| **Other** | 0.40 | 0.49 | **0.44** | **+0.12** |

### Key Takeaways
1. **Centroid Motion works:** The centroid offset magnitude became the **#1 most important feature** (17.4% importance). It successfully broke the blend degeneracy, nearly doubling the Blend F1 score.
2. **Physics > Shapes:** The bespoke hand-engineered shape features from v1 (like odd-even depth) dropped to the bottom of the importance rankings (<4%). The model proved that independent physical priors (Stellar Radius, Temperature, Centroid Tracking) are vastly superior to 1D shape approximations.
3. **The Debugging Trail Matters:** Achieving 73% accuracy by blindly feeding CSVs to XGBoost is easy (and usually leaky). Achieving 73% accuracy *after* proving and eliminating multiple vectors of human-in-the-loop data leakage represents a rigorous, scientifically defensible pipeline ready for fresh, unseen ISRO data.

---

## 6. How to Recreate the Pipeline

### Step 0: Clone and Setup
```bash
git clone https://github.com/rsd-06/periodyx-exoplanet-pipeline.git
cd periodyx-exoplanet-pipeline
pip install -r requirements.txt
```

### Step 1: Download the KOI Data (Label Source)
This downloads the Kepler KOI Cumulative Table, injecting the specific physical columns needed for v4 (`koi_srad`, centroid motion, etc.).
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,kepoi_name,koi_disposition,koi_srad,koi_steff,koi_slogg,koi_kepmag,koi_dicco_mra,koi_dicco_mdec,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,koi_period,koi_duration,koi_depth+from+cumulative&format=csv', 'data/koi_cumulative.csv')"
```

### Step 2: Mechanical Sanity Check
Runs the pipeline entirely on synthetic, locally generated data to ensure your environment (lightkurve, wotan, XGBoost) is functioning.
```bash
python3 scripts/validate_fixes.py
```

### Step 3: Threshold Calibration
Phase-scrambles real Kepler data to establish an empirical noise floor, determining the exact Signal Detection Efficiency (SDE) threshold required to trigger TLS.
```bash
python3 scripts/calibrate_threshold.py --koi-csv data/koi_cumulative.csv --sample 150
```

### Step 4: Build the Training Set
The heavy lifter. Downloads raw FITS light curves from MAST, detrends, searches, fits trapezoids, and exports the 1D shape features.
```bash
python3 scripts/build_training_set.py --koi-csv data/koi_cumulative.csv --bls-threshold 10.77 --workers 8
```
*(Replace `10.77` with your calibrated threshold. Note: This step takes hours for the full 7000+ star dataset).*

### Step 5: Train the Classifier (V4)
Merges the extracted 1D shape features with the NASA archival stellar priors, drops NaNs to prevent MNAR leakage, computes reconstructed radii, runs an Ablation Study against the v3 baseline, optimizes hyperparameters using Optuna (50 trials), and saves the final `.joblib` model.
```bash
python3 scripts/train_classifier.py --features data/training_features.csv --optuna-trials 50
```

### Step 6: Deploy (Optional)
Upload the saved `.joblib` to Hugging Face.
```bash
export HF_TOKEN="your_token"
python3 upload_model.py
```

---

## 7. Key Learnings
1. **Beware the Proxy:** If an archival data column was generated by a human-in-the-loop vetting process (like NASA calculating tight error bars only for highly confident planet candidates), your machine learning model will weaponize it.
2. **Missing Data is Signal:** Dropping missing data is painful, but imputing physical properties (like a star's radius) guarantees your model will learn that "NaN = False Positive". In astrophysics, you cannot zero-fill physics.
3. **Physics beats Geometry:** 1D light curve geometry is fundamentally degenerate. Breaking the degeneracy requires introducing true physical constraints (like centroid pixel shifts).