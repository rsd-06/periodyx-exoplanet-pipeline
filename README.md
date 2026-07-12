---
title: PeriodyX Exoplanet Pipeline
emoji: 🪐
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# PeriodyX — Exoplanet Detection Pipeline

ISRO BAH 2026 — Problem Statement #07: AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**Team:** PeriodyX
**Goal:** Build a robust, interpretable, end-to-end machine learning pipeline to detect exoplanets from raw astronomical light curves, strictly avoiding data leakage and prioritizing physical priors over black-box shape matching.

---

## Domain Background (How it works)

To find exoplanets, we monitor the brightness of stars over time. When a planet passes in front of its host star (a "transit"), it temporarily blocks a fraction of the star's light. This creates a periodic, box-shaped dip in the star's light curve:

![Transit Animation](docs/transit.gif)

A detected periodic dip signal is called a **Threshold Crossing Event (TCE)**. A TCE is characterized by three main parameters:
- **Period**: The number of days between each occurrence of the detected signal.
- **Duration**: The time elapsed by each occurrence of the signal.
- **Epoch**: The time of the first observed occurrence of the signal.

Our goal is to build a machine learning model that classifies whether a given TCE is a genuine planet transit or a false positive caused by another astronomical phenomenon — most commonly an eclipsing binary star or background stellar contamination (a "blend").

---

## 1. What This Project Does

This is an end-to-end pipeline that takes raw stellar brightness measurements (light curves) from NASA's Kepler/TESS missions and automatically:

1. **Detrends the signal** — removes instrumental noise and stellar variability while preserving genuine transit signal.
2. **Searches for periodicity** — scans for box-shaped dips that indicate a transiting planet, using a fast/slow tiered search.
3. **Fits a physical model** — a trapezoid fit characterizes each detected dip's depth, duration, and ingress/egress time.
4. **Extracts physical priors and features** — pulls independent stellar properties (radius, temperature) and centroid motion (pixel-level positional shift) directly from archival data.
5. **Classifies each detection (XGBoost)** into one of four categories:
   - **transit** — a real planet
   - **eclipsing_binary** — two stars orbiting each other
   - **blend** — contamination from a background source
   - **other** — noise, starspots, or instrumental artifacts

**Design philosophy:** unlike many published systems (e.g. AstroNet) that rely on NASA's proprietary detection pipeline to isolate signals before feeding them into a classifier, PeriodyX owns the entire chain end-to-end — detection through classification — using physically interpretable features rather than a black-box network, and runs on a single commodity machine with no GPU required.

---

## 2. Tech Stack

- **Language:** Python 3
- **Data acquisition:** `lightkurve` (MAST archive access), `urllib` (NASA Exoplanet Archive TAP API)
- **Detrending:** `wotan` (biweight filter)
- **Periodic search:** `astropy.timeseries.BoxLeastSquares`, `transitleastsquares`
- **Shape fitting:** `lmfit` (non-linear least squares)
- **Classification:** `xgboost`, `scikit-learn`, `optuna` (hyperparameter search)
- **Data handling:** `pandas`, `numpy`
- **Model persistence:** `joblib`
- **Web backend:** `FastAPI`, `uvicorn`
- **Deployment:** Hugging Face Spaces (Docker), GitHub

---

## 2b. Interactive Web Application

The pipeline is deployed as a full-stack web application on Hugging Face Spaces.

**Architecture:**
- **Backend (`main.py`):** FastAPI loads the v5 Two-Stage `.joblib` model at startup and exposes three endpoints:
  - `POST /api/run_synthetic` — generates a synthetic Kepler-like light curve, runs the full pipeline, and returns classification probabilities plus a 3-panel visualization.
  - `POST /api/run_custom` — accepts a user-uploaded CSV (`time`, `flux` columns) plus astrophysical priors (stellar radius, temperature, etc.), runs the same pipeline, and returns results.
  - `POST /api/run_batch` — accepts multiple CSV files for batch processing, perfect for reviewing multiple light curves at once.
- **Frontend (`static/`):** a dark glassmorphism UI in vanilla HTML/CSS/JS, featuring animated pipeline step indicators, batch-upload support, per-class confidence bars, and a physical diagnostics grid.
- **No core pipeline changes:** `pipeline.py` and all core modules are untouched by the web layer. The backend only *calls* the pipeline and forwards results to the frontend.

---

## 3. Pipeline Architecture

```text
Raw Light Curve (Kepler/TESS, via MAST)
        |
        v
  [1] DETRENDING (wotan, biweight filter)
        |--------------------------------------------+
        v                                            v
  [2] PERIODIC SEARCH -- tiered                [2b] SINGLE-TRANSIT SCAN
        BLS (fast triage, runs on every star)       (Sliding window matched filter
              |                                      for non-repeating long-period
              v  (if BLS > noise floor)              dips)
        TLS (slow, high-sensitivity refinement)      |
        |                                            v
        v                                         (Flagged for human review)
  [3] SHAPE CHARACTERIZATION (lmfit)
        -> Fits depth, duration, ingress/egress
        |
        v
  [4] FEATURE ENGINEERING & PHYSICAL PRIORS (the "v4/v5" architecture)
        -> 1D shape features (SNR, odd-even depth diff, secondary eclipse)
        -> Independent stellar priors (radius, temperature, gravity, magnitude)
        -> Centroid motion (pixel-level offset magnitude)
        -> Stellar Density Ratio (Kepler's Third Law sanity check constraint)
        |
        v
  [5] CLASSIFICATION ENSEMBLE (N=20 XGBoost models)
        -> Stage 1: Transit vs. Not Transit
        -> Stage 2: Eclipsing Binary vs. Blend vs. Other
        -> Outputs final 4-class probabilities + Bootstrap Uncertainty Bounds
```

---

## 4. The Evolution of the Pipeline (Version History)

Building an astronomical ML pipeline is notoriously susceptible to **data leakage**. This section documents how the pipeline was iteratively debugged and improved.

### v1: The Pure Shape Baseline
- **Approach:** 13 pure geometric features extracted from the light curve.
- **Result:** ~40–46% F1-macro.
- **Finding:** 1D shape is fundamentally degenerate.

### v2/v3: The Leakage Trap
- **Approach:** Added NASA's own vetting flags (`fpflag_*`) and NASA-computed planetary radius (`koi_prad`).
- **Result:** F1-macro jumped to ~98%.
- **The catch:** This was leakage. The model memorized NASA's missing-data and error-bar signatures rather than learning astrophysics. F1 returned to ~46% once these were stripped.

### v4: Physical Priors & Centroid Tracking
- **Approach:** Added real, independent physics: independent stellar priors (`koi_srad`, `koi_steff`) and pixel-level centroid motion (`koi_dicco_mra/mdec`) to break the blend degeneracy.
- **Result:** F1-macro 0.66. Held-out accuracy 73%. Centroid motion proved to be the most important feature (17.4%).

### v5: Decomposed Classifier & Density Sanity Check
- **Approach:** 
  1. **Stellar Density Ratio:** A physical sanity check directly derived from Kepler's Third Law (using the Winn 2010 geometric form). Deep, short-duration grazing binaries imply mathematically impossible ultra-dense stars. We compute this implied density and take the log ratio against the measured density.
  2. **Two-Stage Classifier:** Decomposed the classification task into a hierarchy (Transit vs Non-Transit, then EB vs Blend vs Other) to better handle class imbalances and differing difficulty boundaries.
- **Result:** The Two-Stage model cleanly beats the baseline (F1-macro ~0.67), while precision on `eclipsing_binary` returned to high levels.

---

## 5. Final Results (v5)

**Overall held-out test accuracy: ~74%** (F1-macro 0.67)

| Class | Precision | Recall | F1-Score | v1/v3 Baseline F1 | v4 F1 |
|---|---|---|---|---|---|
| Transit (Planet) | 0.87 | 0.89 | **0.88** | 0.62 | 0.88 |
| Eclipsing Binary | 0.84 | 0.60 | **0.70** | 0.65 | 0.72 |
| Blend | 0.56 | 0.68 | **0.62** | 0.33 | 0.61 |
| Other | 0.38 | 0.51 | **0.44** | 0.32 | 0.44 |

### Why it worked

1. **Two-Stage Separation:** Separating the easiest distinction (Transit vs Noise) from the hardest distinction (Blend vs Eclipsing Binary) prevents the model from smearing its decision boundaries.
2. **Physical Discriminants:** Features like the `stellar_density_ratio` give the XGBoost ensemble direct access to physics constraints (Kepler's Third Law) that it would otherwise struggle to infer from raw duration/period values alone.


---

## 6. How to Recreate the Pipeline

### Step 0: Clone and Setup
```bash
git clone https://github.com/rsd-06/periodyx-exoplanet-pipeline.git
cd periodyx-exoplanet-pipeline
pip install -r requirements.txt
```

### Step 1: Download the KOI Data (Label Source)
Downloads the Kepler KOI Cumulative Table, including the physical columns required for v4 (`koi_srad`, centroid motion, etc.).
```bash
python -c "import urllib.request; urllib.request.urlretrieve('https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,kepoi_name,koi_disposition,koi_srad,koi_steff,koi_slogg,koi_kepmag,koi_dicco_mra,koi_dicco_mdec,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,koi_period,koi_duration,koi_depth+from+cumulative&format=csv', 'data/koi_cumulative.csv')"
```

### Step 2: Mechanical Sanity Check
Runs the pipeline entirely on synthetic, locally generated data to confirm the environment (lightkurve, wotan, XGBoost) is functioning correctly.
```bash
python3 scripts/validate_fixes.py
```

### Step 3: Threshold Calibration
Phase-scrambles real Kepler data to establish an empirical noise floor, determining the Signal Detection Efficiency (SDE) threshold required to trigger TLS.
```bash
python3 scripts/calibrate_threshold.py --koi-csv data/koi_cumulative.csv --sample 150
```

### Step 4: Build the Training Set
The heavy step. Downloads raw FITS light curves from MAST, detrends, searches, fits trapezoids, and exports the shape + physical-prior feature table.
```bash
python3 scripts/build_training_set.py --koi-csv data/koi_cumulative.csv --bls-threshold 10.77 --workers 8
```
*(Replace `10.77` with your own calibrated threshold from Step 3. This step takes hours for the full 7,000+ star dataset.)*

### Step 5: Train the Classifier (v5)
Merges extracted shape features with NASA archival stellar priors, drops rows with missing physical data (MNAR-safe), optimizes hyperparameters via Optuna, and trains the Two-Stage Bootstrap Ensemble (N=20) for uncertainty quantification.
```bash
python3 scripts/train_classifier.py --features data/training_features.csv --optuna-trials 50
```

### Step 6: Deploy (Optional)
```bash
export HF_TOKEN="your_token"
python3 upload_model.py
```

---

## 7. Key Learnings

1. **Beware the proxy.** If an archival column was generated by a human-in-the-loop vetting process (like NASA computing tight error bars only for confident planet candidates), a model will exploit it if given the chance — regardless of whether that was the intent.
2. **Missing data is signal.** Dropping missing rows is costly, but imputing physical properties (like stellar radius) risks teaching a model "NaN = false positive" instead of teaching it physics. In this domain, silently filling gaps is not a neutral choice.
3. **A leakage check has to look at more than presence/absence.** Confirming a feature isn't *missing* in a biased way is necessary but not sufficient — its *precision* (error bars, relative uncertainty) can carry the same bias even when the value itself is present, as seen with both `koi_prad` and (initially, before the physical explanation held up) the centroid data.
4. **Physics beats geometry, but not every physically-motivated feature earns its place.** Centroid motion broke a real degeneracy that shape alone couldn't. Reconstructed radius, despite being equally well-motivated physically, turned out to be redundant once centroid was present — ablation testing, not intuition, is what caught that.