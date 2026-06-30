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
| Model persistence | `joblib` |

---

## 7. Known Limitations (Explicit, Not Hidden)

**Long-period, single-transit planets are structurally undetectable by the
main pipeline.** BLS and TLS both require at least two observed transits to
establish periodicity. A planet whose orbital period exceeds the
observation baseline shows only one dip, with no way to confirm
periodicity. `singletransit/flagging.py` is an exploratory side-module that
flags isolated, statistically significant dips for manual follow-up — it
does not attempt to confirm or classify them, and is reported separately
from the main classification output, with explicit lower confidence. This
is a known, openly documented blind spot shared by every major existing
detection pipeline, not unique to this project.

**`secondary_eclipse_depth` is currently not contributing meaningful signal
— root cause identified, fix not yet implemented.** Investigation found
this feature clusters within ~1e-5 of zero for 98% of all 7,449 processed
stars, including stars labeled as confirmed eclipsing binaries
specifically. Isolating just the 2,187 EB-labeled stars and re-checking the
distribution showed no meaningful difference from the full population
(median -0.000021 vs. -0.000014 in the unfiltered set) — ruling out "real
signal, just small" as the explanation. Most likely cause: periodic search
locking onto half the true orbital period for a meaningful fraction of
eclipsing binaries (a well-documented BLS/TLS failure mode), which would
shift the true secondary eclipse away from the phase-0.5 location this
function assumes. Planned fix: detect this aliasing condition directly
(e.g. compare signal strength at the detected period vs. double that
period) before computing secondary eclipse phase, rather than assuming the
detected period is correct.

**Class imbalance is present and currently unweighted.** Label distribution
across 7,449 processed stars: transit 2,701, eclipsing_binary 2,187, blend
1,388, other 1,173. `blend` and `other` are both smaller and, based on the
confusion matrix (Section 8), harder for the model to separate from
`transit`. The classifier currently does not apply class weighting or
`scale_pos_weight` adjustment.

**This pipeline is trained on the Kepler KOI cumulative table, not on
ISRO's curated dataset.** ISRO's curated dataset is released only upon
selection; the TOI catalogue (TESS's equivalent candidate table) was
explicitly confirmed off-limits at the ideation stage by the hackathon
organizers. This project uses the *Kepler* KOI table instead, which is a
separate, older mission's public catalogue, to build and validate the
pipeline independently while the project's eventual recalibration target
remains ISRO's own curated set.

---

## 8. Real Results — Full Kepler KOI Run

**Data acquisition:** 7,586 KOI entries attempted; 7,449 stars successfully
downloaded and processed (137 unavailable/missing on MAST servers); 5,682
stars yielded a complete feature vector with no missing values across every
feature column. The full 7,449-row dataset (including rows with some
missing odd-even/secondary-eclipse values, dropped via `dropna()` at
training time) was used for the final reported model below.

**Class distribution (7,449 labeled stars):**

| Class | Count |
|---|---|
| transit | 2,701 |
| eclipsing_binary | 2,187 |
| blend | 1,388 |
| other | 1,173 |

**Held-out test set performance (XGBoost, n=1,137):**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| blend | 0.34 | 0.15 | 0.21 | 203 |
| eclipsing_binary | 0.73 | 0.62 | 0.67 | 347 |
| other | 0.36 | 0.20 | 0.25 | 168 |
| transit | 0.54 | 0.85 | 0.66 | 419 |
| **accuracy** | | | **0.56** | 1137 |
| **macro avg** | 0.49 | 0.45 | **0.45** | 1137 |
| weighted avg | 0.53 | 0.56 | 0.52 | 1137 |

**Confusion matrix (rows = true label, columns = predicted):**

| True \ Pred | blend | eclipsing_binary | other | transit |
|---|---|---|---|---|
| blend | 31 | 26 | 24 | **122** |
| eclipsing_binary | 18 | 214 | 21 | 94 |
| other | 21 | 26 | 33 | **88** |
| transit | 20 | 29 | 14 | 356 |

**Top 5 feature importances:**

1. `depth_snr` — 20.8%
2. `depth` — 20.7%
3. `period` — 10.7%
4. `detection_significance` — 8.5%
5. `t_tot_hours` — 8.2%

**Honest reading of these results:** Transit recall (85%) looks strong in
isolation, but the confusion matrix shows this is partly inflated by the
model defaulting to "transit" whenever uncertain — 60% of true blends and
52% of true "other" cases get misclassified as transit specifically, not
spread evenly across the other classes. The macro F1 of 0.45 is the more
honest summary number than any single-class recall figure. The top feature
importances are all generic signal-strength metrics (depth, SNR, duration);
the two features specifically engineered to distinguish *cause* (odd-even
difference, secondary eclipse) do not appear in the top 5, consistent with
the secondary-eclipse investigation in Section 7. This is real, useful
diagnostic information about where the pipeline needs further work, not a
finished result.

---

## 9. Problems Encountered and Fixed During Development

This section documents real debugging history, not just the final design.

**XGBoost label encoding.** Recent `xgboost` versions require integer-
encoded class labels internally and reject raw strings. Fixed by wrapping
`sklearn.preprocessing.LabelEncoder` inside `ExoplanetClassifier`,
transparent to callers (`.fit()`/`.predict()` still accept/return string
labels like `"transit"`).

**Hardcoded `objective="multi:softprob"` broke on small class counts.**
When testing on small synthetic datasets with only 2 classes present,
XGBoost's multiclass objective failed because it expects a `num_class`
matching what was hardcoded. Fixed by removing the explicit objective and
letting XGBoost auto-infer binary vs. multiclass from the data.

**BLS→TLS gating threshold was a no-op bug.** The original gate compared
*raw* BLS power against a default threshold of `0.0`. Raw power is almost
always positive, so the condition was true for nearly every star — TLS ran
on everyone regardless of the BLS result, defeating the purpose of having a
triage tier at all. Fixed by computing a normalized, per-star significance
score: `(best_power - median(power)) / std(power)`, comparable across stars
regardless of individual noise level.

**Look-elsewhere effect inflated the noise floor.** Initial testing with a
significance threshold of 6.0 found that *pure synthetic noise* (no
injected transit at all) could score ~6.1 — above the supposed cutoff —
because testing thousands of candidate periods means even pure chance will
occasionally produce a result that looks locally significant. Threshold
raised to 9.0 as an interim default, with a regression test
(`scripts/dry_run_test.py`) added that explicitly checks pure noise stays
below threshold and a strong injected signal clears it.

**Threshold calibration via class-label comparison was methodologically
wrong.** An earlier version of `calibrate_threshold.py` compared BLS
significance between `transit`-labeled and `not_transit`-labeled real
stars, suggesting a threshold of 5.99. This was the wrong comparison:
eclipsing binaries (a large share of `not_transit`) often produce *stronger*
periodic signals than genuine small-planet transits, since stellar eclipses
are typically deeper than planetary transits. The suggested threshold was
actually below the real noise floor. Replaced with phase-scrambling
calibration (Section 5), which measures the noise floor directly rather
than inferring it from class labels.

**Percentile-based threshold was fragile with a heavy-tailed, small
sample.** With only 450 noise samples, the 99th percentile was determined
by roughly the top 4 values — a handful of unusually noisy outlier stars
could single-handedly set the entire threshold. Added a MAD (median
absolute deviation) based alternative, which is robust to a small number of
extreme outliers, alongside the percentile estimate, with explicit guidance
to prefer the MAD-based value when the two disagree substantially.

**`secondary_eclipse_depth` returns a number even on statistically weak
input.** The function's only safeguard is a minimum point-count check
(`< 3` points in the relevant phase window), which is almost never triggered
on Kepler's typically dense cadence — so it returns a value even when that
value is dominated by noise rather than a real secondary signal. This is
the root cause identified in Section 7 and is an open item, not yet fixed.

---

## 10. How to Run This

**Mechanical sanity check (no real data, seconds to run):**
```bash
pip install -r requirements.txt
python3 scripts/dry_run_test.py
```

**Calibrate the BLS→TLS threshold on real data:**
```bash
python3 scripts/calibrate_threshold.py --koi-csv data/koi_cumulative.csv --sample 150
```

**Build the training feature set (slow, CPU-bound, resumable):**
```bash
python3 scripts/build_training_set.py \
    --koi-csv data/koi_cumulative.csv \
    --bls-threshold <value from calibration step> \
    --workers 8
```

**Train and evaluate the classifier (fast, seconds):**
```bash
python3 scripts/train_classifier.py \
    --features data/training_features.csv \
    --model-out models/exoplanet_classifier.joblib
```

---

## 11. What's Next

In priority order, based on the diagnostic findings above:

1. Fix `secondary_eclipse_depth` to detect and correct for period-aliasing
   in eclipsing binaries before computing secondary-eclipse phase.
2. Add class weighting (`scale_pos_weight` / `sample_weight`) to address
   the `blend`/`other` underperformance visible in the confusion matrix.
3. Re-evaluate feature importances and per-class F1 after both fixes, to
   check whether the discriminator features actually become useful once
   they're computed correctly.
4. Recalibrate and retrain on ISRO's curated dataset once released, using
   the same pipeline and feature interface — no architectural changes
   required, only a swapped label source.