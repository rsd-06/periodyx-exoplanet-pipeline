# PeriodyX — Exoplanet Detection Pipeline

ISRO BAH 2026 — Problem Statement #07: AI-enabled Detection of Exoplanets
from Noisy Astronomical Light Curves

Team PeriodyX

---

## 1. What This Project Does

This is an end-to-end pipeline that takes raw stellar brightness measurements
(light curves) from NASA's Kepler/TESS missions and automatically:

1. Cleans and detrends the signal (removes instrumental noise and stellar
   variability while preserving genuine transit signal)
2. Searches for periodic, box-shaped dips that could indicate a transiting
   planet
3. Fits a physical trapezoid model to characterize the shape of each
   detected dip (depth, duration, ingress/egress time)
4. Extracts a set of physically meaningful, interpretable features from
   that fit
5. Classifies each detection into one of four categories: **transit**
   (real planet), **eclipsing binary**, **blend** (contamination from a
   background source), or **other** (noise, starspots, artifacts)
6. Reports a statistically grounded confidence level for every output,
   rather than a single bare prediction

The design philosophy, deliberately: most published systems (AstroNet,
ExoMiner, ExoMiner++) classify candidates that someone else's pipeline
already detected — they depend on NASA's proprietary SPOC system to do the
actual detection step. This project owns the entire chain itself, end to
end, using interpretable shape parameters instead of a black-box deep
network, and is built to run on a single commodity machine with no GPU.

---

## 2. Why This Problem Is Hard

A transiting planet blocks a tiny fraction of its star's light — often
0.01%–1%, frequently smaller than the noise introduced by the detector
itself, spacecraft jitter, or the star's own natural variability (spots,
flares). On top of that, several unrelated astrophysical phenomena produce
visually similar periodic dips:

- A genuine planet transit
- An eclipsing binary star system (two stars orbiting each other)
- A "blend" — light from a background eclipsing system contaminating the
  measurement of the target star
- Starspots rotating across the stellar surface

A single TESS/Kepler observing sector can contain 20,000+ stars, making
manual inspection infeasible. The pipeline has to separate real transits
from these look-alikes automatically, at scale, with enough statistical
rigor to be trustworthy.

---

## 3. Pipeline Architecture

```
Raw Light Curve (Kepler/TESS, via MAST)
        |
        v
  [1] DETRENDING (wotan, biweight filter)
        |
        v
  [2] PERIODIC SEARCH -- tiered
        BLS (fast, runs on every star)
              |
              v  (only if BLS significance clears a calibrated threshold)
        TLS (slow, more sensitive refinement)
        |
        v
  [3] PHASE FOLDING
        |
        v
  [4] TRAPEZOID SHAPE FIT (lmfit)
        -> depth, total duration, ingress/egress duration,
           flat-bottom duration, with covariance-based uncertainty
        |
        v
  [5] FEATURE EXTRACTION
        -> ingress fraction, odd-even depth difference,
           secondary eclipse depth, depth SNR, detection significance
        |
        v
  [6] CLASSIFICATION (XGBoost)
        -> transit / eclipsing_binary / blend / other
        |
        v
  [7] UNCERTAINTY QUANTIFICATION
        -> bootstrap ensemble variance on classifier confidence
        |
        v
  Output: label + confidence + physical parameters + error bars
```

A parallel, independent module (`singletransit/flagging.py`) scans for
isolated, non-periodic dips that the periodic search structurally cannot
detect (see Section 7).

---

## 4. Module-by-Module Breakdown

| Module | File | What it does |
|---|---|---|
| Data acquisition (real) | `data/acquisition.py` | Downloads and lightly cleans real light curves from MAST via `lightkurve` |
| Data acquisition (synthetic) | `data/synthetic.py` | Generates synthetic transit-injected light curves for pipeline validation without needing real data access |
| Label loading | `data/koi_labels.py` | Loads the Kepler KOI cumulative table and maps disposition + diagnostic flags to the 4-class taxonomy |
| Detrending | `preprocessing/detrend.py` | Biweight filter (`wotan`) + sigma clipping |
| Periodic search | `detection/periodic_search.py` | Tiered BLS (fast triage) → TLS (sensitive refinement), with a normalized, per-star significance score |
| Shape characterization | `characterization/trapezoid_fit.py` | Trapezoid model fit (`lmfit`) with parameter uncertainties from the fit covariance matrix |
| Feature engineering | `features/extract.py` | Odd-even depth difference, secondary eclipse depth, ingress fraction, depth SNR |
| Classification | `classification/classifier.py` | XGBoost wrapper with label encoding, save/load persistence |
| Uncertainty | `uncertainty/bootstrap.py` | Bootstrap ensemble for classification confidence |
| Single-transit flagging | `singletransit/flagging.py` | Sliding-window matched filter for non-periodic candidates |
| Visualization | `visualization/plots.py` | 3-panel detection summary plots |
| Orchestration | `pipeline.py` | Chains every stage into a single `run_pipeline()` call |
| Training set builder | `scripts/build_training_set.py` | Parallelized, resumable batch feature extraction across thousands of stars |
| Threshold calibration | `scripts/calibrate_threshold.py` | Phase-scrambling noise-floor estimation for the BLS→TLS gate |
| Training | `scripts/train_classifier.py` | Trains, cross-validates, and saves the classifier |
| Sanity test | `scripts/dry_run_test.py` | End-to-end mechanical validation on synthetic data, no real downloads needed |

---

## 5. Algorithms Used, and Why

**Detrending — biweight filter (`wotan`)**
Removes long-timescale instrumental and stellar trends while preserving
short-timescale transit signal. Chosen over a simple moving average because
it's robust to the transit dip itself being pulled into the trend estimate.

**Box Least Squares (BLS)**
Fast first-pass periodic search. Tests a grid of candidate periods and
durations, looking for the period that best matches a box-shaped dip. Used
as a cheap triage step run on every star.

**Transit Least Squares (TLS)**
A slower, more sensitive refinement of BLS, used only on stars whose BLS
result clears a significance threshold. Produces a Signal Detection
Efficiency (SDE) score used as the detection-significance feature.

**Why tiered, not TLS-on-everything:** TLS is far more compute-expensive
than BLS. Running it on every star in a 20,000+-star sector is wasteful;
running it only on the subset BLS flags as promising is what makes
full-sector processing computationally realistic. This tiering is a
deliberate scalability decision, not just a speed shortcut.

**Trapezoid transit model fit (`lmfit`)**
Rather than a full physical limb-darkened model (e.g. `batman`), the
characterization stage fits a 4-parameter trapezoid: baseline flux, depth,
total duration, ingress/egress duration. This choice deliberately mirrors
ISRO's own reference methodology slides (which show a trapezoid fit, not a
smooth physical model), avoids needing assumed limb-darkening coefficients
for unknown stars, and is faster and more numerically stable on noisy data.

**Feature engineering — odd-even depth difference, secondary eclipse depth**
Standard astronomical vetting checks (the same diagnostics ExoMiner uses as
dedicated branches), implemented here as lightweight, interpretable
features rather than learned embeddings:
- *Odd-even depth difference*: if a periodic search locks onto half the
  true period of an eclipsing binary, alternating transits will have
  different depths — a strong tell that the signal isn't a planet.
- *Secondary eclipse depth*: a real secondary eclipse (visible at phase
  0.5) implies two stars eclipsing each other, not a planet (planets are
  too dim to produce a detectable secondary eclipse in this data).

**XGBoost classifier**
Chosen over a neural network because the feature set is small (~10 hand-
engineered numeric features) and the labeled dataset, while substantial, is
not at the scale where deep learning has a clear advantage. Gradient-
boosted trees train in seconds, need no GPU, and provide feature
importances for free.

**Bootstrap ensemble uncertainty**
Trains multiple classifiers on bootstrap-resampled subsets of the training
data; the variance across their predictions on a given star is used as a
confidence estimate. Chosen over Monte Carlo Dropout because the classifier
is tree-based, not a neural network — MC Dropout doesn't apply here.

**Phase-scrambling for threshold calibration**
The BLS→TLS gate needs a significance threshold above which a star is
"worth" the expensive TLS refinement. Rather than guessing a fixed number,
`calibrate_threshold.py` randomly shuffles each real star's flux values in
time (destroying any real periodicity while preserving the star's actual
noise characteristics exactly), runs BLS on the scrambled version, and uses
the resulting distribution as a real, data-grounded estimate of what "pure
noise" looks like on this specific dataset.

---

## 6. Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3 |
| Data acquisition | `lightkurve` (MAST archive access) |
| Detrending | `wotan` |
| Periodic search | `astropy.timeseries.BoxLeastSquares`, `transitleastsquares` |
| Shape fitting | `lmfit` |
| Classification | `xgboost`, `scikit-learn` |
| Uncertainty | bootstrap ensembling (`scikit-learn` resampling) |
| Data handling | `pandas`, `numpy` |
| Visualization | `matplotlib` |
| Model persis## 7. Known Limitations (Explicit, Not Hidden)

**Long-period, single-transit planets are structurally undetectable by the main pipeline.** BLS and TLS both require at least two observed transits to establish periodicity. A planet whose orbital period exceeds the observation baseline shows only one dip, with no way to confirm periodicity. `singletransit/flagging.py` is an exploratory side-module that flags isolated, statistically significant dips for manual follow-up — it does not attempt to confirm or classify them, and is reported separately from the main classification output, with explicit lower confidence. This is a known, openly documented blind spot shared by every major existing detection pipeline, not unique to this project.

**Training dataset rebuild blocked by system WDAC policy.** The core pipeline has been upgraded (v2) to fix five major feature-extraction failure modes (see Section 9). However, because rebuilding the training dataset requires re-downloading raw data for all 7,500 stars, and a Windows Defender Application Control (WDAC) policy is currently blocking `lightkurve` from loading a required C extension (`_c_internal_utils.pyd`), the dataset cannot be rebuilt on this machine. The current model (Section 8) is trained on legacy v1 data with missing values filled via imputation, so it does not yet benefit from the v2 pipeline fixes.

**Class imbalance is present but mitigated.** Label distribution across 7,449 processed stars: transit 2,701, eclipsing_binary 2,187, blend 1,388, other 1,173. `blend` and `other` are smaller and harder for the model to separate. The classifier applies `sample_weight="balanced"` to address this, though physical overlap remains a challenge.

**This pipeline is trained on the Kepler KOI cumulative table, not on ISRO's curated dataset.** ISRO's curated dataset is released only upon selection; the TOI catalogue (TESS's equivalent candidate table) was explicitly confirmed off-limits at the ideation stage by the hackathon organizers. This project uses the *Kepler* KOI table instead, which is a separate, older mission's public catalogue, to build and validate the pipeline independently while the project's eventual recalibration target remains ISRO's own curated set.

---

## 8. Real Results — Full Kepler KOI Run

**Data acquisition:** 7,586 KOI entries attempted; 7,449 stars successfully downloaded and processed (using v1 pipeline). 

**Class distribution (7,449 labeled stars):**

| Class | Count |
|---|---|
| transit | 2,701 |
| eclipsing_binary | 2,187 |
| blend | 1,388 |
| other | 1,173 |

**Held-out test set performance (XGBoost, 5-fold CV F1-macro: 0.452):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| blend | 0.32 | 0.49 | 0.39 | 278 |
| eclipsing_binary | 0.80 | 0.62 | 0.70 | 437 |
| other | 0.29 | 0.43 | 0.35 | 235 |
| transit | 0.66 | 0.50 | 0.57 | 540 |
| **accuracy** | | | **0.52** | 1490 |
| **macro avg** | 0.52 | 0.51 | **0.50** | 1490 |

**Confusion matrix (rows = true label, columns = predicted):**

| True \ Pred | blend | eclipsing_binary | other | transit |
|---|---|---|---|---|
| blend | 137 | 15 | 87 | 39 |
| eclipsing_binary | 45 | 270 | 66 | 56 |
| other | 69 | 15 | 101 | 50 |
| transit | 131 | 28 | 112 | 269 |

**Top 5 feature importances:**

1. `depth_snr` — 31.2%
2. `depth` — 12.4%
3. `period` — 9.6%
4. `t_tot_hours` — 9.1%
5. `detection_significance` — 8.7%

**Honest reading of these results:** The macro F1 of 0.452 is the honest summary number. The top feature importances are generic signal-strength metrics (depth, SNR, duration). The two features specifically engineered to distinguish cause (`odd_even_depth_diff`, `secondary_eclipse_depth`) have low importance (6–7%) in this dataset because they were extracted with the v1 pipeline, which had known failure modes (see Section 9). Once the WDAC block is lifted and the dataset is rebuilt using v2 code, these discriminative features are expected to jump in importance and boost overall accuracy.

---

## 9. Problems Encountered and Fixed During Development (v2 Upgrades)

This section documents real debugging history, specifically the physical and mathematical edge cases that broke the v1 pipeline and were fixed in v2.

**The "P/2 Alias" (Period Halving):** BLS often locks onto exactly half the true orbital period of an eclipsing binary because folding the light curve at P/2 stacks the primary and secondary eclipses on top of each other, creating a deeper, cleaner-looking "box". *Fix:* Added an explicit check comparing signal strength at P vs. 2P; if the odd/even transits differ significantly in depth, the pipeline doubles the period before searching for a secondary eclipse.

**Eccentric Orbits hiding Secondary Eclipses:** v1 assumed secondary eclipses only happen at exactly phase 0.5 (circular orbits). Eccentric orbits shift the secondary eclipse to a different phase. *Fix:* The secondary eclipse search now scans the entire out-of-transit phase space, rather than just a narrow window at 0.5.

**Period Doubling on Noisy Transits:** Sometimes BLS would double a planet's period and find a "secondary" dip that was just random noise. *Fix:* Implemented a Bayesian Information Criterion (BIC) check. A secondary dip is only accepted if a 2-dip model explains the data significantly better than a 1-dip model, penalizing unnecessary complexity.

**Starspots Mimicking Eclipses:** A large starspot can rotate into view periodically and mimic a secondary eclipse. However, starspots migrate and change over months, whereas physical eclipses are rigidly periodic. *Fix:* The pipeline splits the light curve into two halves (early/late epochs). A true secondary eclipse must be present and aligned in both halves; wandering starspots fail this check and are discarded.

**Bootstrap Uncertainty on Low-SNR Planets:** At low SNR, the `ingress_fraction` feature (the slope of the dip) becomes chaotic due to noise. *Fix:* If SNR < 5.0, the pipeline bootstraps the light curve (resampling residuals with replacement) 100 times and computes the median ingress fraction, stabilizing the feature against random noise spikes.

---

## 10. How to Run This

### Step 0: Get the Data
Before running the pipeline, you need the label source (`data/koi_cumulative.csv`), which tells the project whether a star is a confirmed planet or a false positive. It is not included in the repository because it's pulled live from NASA. 

Download it by visiting this URL in a browser or using a tool like `curl`:
[NASA Exoplanet Archive Data](https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+kepid,kepoi_name,koi_disposition,koi_fpflag_nt,koi_fpflag_ss,koi_fpflag_co,koi_fpflag_ec,koi_period,koi_duration,koi_depth,koi_prad+from+cumulative&format=csv)

Save the result as `data/koi_cumulative.csv` inside your project folder.

### Step 1: Mechanical Sanity Check
Install the required libraries and run a self-contained test on fake data. This checks that your environment is set up correctly (engine starts) before downloading real data.

```bash
pip install -r requirements.txt
python3 scripts/validate_fixes.py
```
*Run this first! This runs the full v2 pipeline on pure synthetic data. If it fails, something is wrong with your environment.*

### Step 2: Calibrate the Detection Threshold
Determine the cutoff score for "interesting signals" vs. "pure noise". This script scrambles real data to establish a noise baseline and suggests a safe threshold number.

```bash
python3 scripts/calibrate_threshold.py --koi-csv data/koi_cumulative.csv --sample 150
```
*It prints a recommended threshold number at the end, which you'll need for Step 3.*

### Step 3: Build the Training Dataset
This is the heavy-lifting step. It downloads real light curves for every star, runs them through the pipeline, and builds a table of measurements for the classifier.

```bash
python3 scripts/build_training_set.py \
    --koi-csv data/koi_cumulative.csv \
    --bls-threshold <value from calibration step> \
    --workers 8
```
- `--bls-threshold`: Replace `<value from calibration step>` with the number printed in Step 2.
- `--workers`: Processes multiple stars in parallel. Set this to the number of CPU cores on your machine.

*Note: This step can take hours depending on the number of stars.*

### Step 4: Train the Classifier
Teach the model to recognize patterns in the data prepared in Step 3. Since the heavy lifting is done, this step only takes seconds.

```bash
python3 scripts/train_classifier.py \
    --features data/training_features.csv \
    --model-out models/exoplanet_classifier.joblib
```
*It saves the trained model as a `.joblib` file so you can reuse it without retraining, and prints a performance report.*

---

## 11. What's Next

1. **Rebuild the Training Dataset:** Run the full dataset through the upgraded v2 pipeline once the WDAC policy allows `lightkurve` data acquisition. The v2 fixes are proven on synthetic data but need to be applied to the real Kepler dataset.
2. **Re-evaluate the Classifier:** Retrain the XGBoost model on the newly extracted v2 features. The improved `secondary_eclipse_depth` and `odd_even_depth_diff` features are expected to drastically improve separation between `transit`, `eclipsing_binary`, and `blend`.
3. **ISRO Dataset:** Recalibrate and retrain on ISRO's curated dataset once released, using the same pipeline and feature interface — no architectural changes required, only a swapped label source.