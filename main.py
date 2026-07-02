"""
FastAPI backend for the PeriodyX Exoplanet Detection Pipeline.

Serves the custom HTML/CSS/JS frontend and exposes two API endpoints:
  POST /api/run_synthetic  — generate a synthetic lightcurve and run pipeline
  POST /api/run_custom     — accept a CSV upload and run pipeline

The V4 classifier is loaded ONCE at startup from the saved .joblib file.
Core pipeline logic (pipeline.py, etc.) is NOT modified — we only call it.
"""

import io
import os
import base64
import traceback
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# ── Pipeline imports (untouched core) ────────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.synthetic import make_synthetic_lightcurve, make_false_positive_lightcurve
from pipeline import run_pipeline
from characterization.trapezoid_fit import _trapezoid
from classification.classifier import ExoplanetClassifier

# ── V4 Feature column order (must match training exactly) ────────────────────
FEATURE_COLUMNS = [
    "depth", "t_tot_hours", "t_in_hours", "flat_bottom_hours",
    "ingress_fraction", "period", "detection_significance",
    "odd_even_depth_diff", "secondary_eclipse_depth", "secondary_eclipse_phase",
    "depth_snr", "n_signals_detected", "period_corrected",
    "koi_srad", "koi_steff", "koi_slogg", "koi_kepmag",
    "centroid_offset_magnitude", "reconstructed_prad"
]

FILL_DEFAULTS = {
    "period_corrected": 0,
    "n_signals_detected": 1,
    "secondary_eclipse_phase": 0.5,
    "secondary_eclipse_depth": 0.0,
}

# ── Default Sol-like stellar parameters for synthetic lightcurves ─────────────
SOL_DEFAULTS = {
    "koi_srad": 1.0,
    "koi_steff": 5778.0,
    "koi_slogg": 4.44,
    "koi_kepmag": 12.0,
    "centroid_offset_magnitude": 0.0,
}

app = FastAPI(title="PeriodyX Exoplanet Pipeline API")

# ── Load V4 classifier at startup ─────────────────────────────────────────────
MODEL_PATH = "models/exoplanet_classifier.joblib"
classifier = None
if os.path.exists(MODEL_PATH):
    try:
        classifier = ExoplanetClassifier.load(MODEL_PATH)
        print(f"✅ V4 classifier loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"⚠️  Failed to load classifier: {e}")
else:
    print(f"⚠️  Model not found at {MODEL_PATH} — classification will be skipped")


def _build_feature_row(pipeline_result: dict, stellar_params: dict) -> pd.DataFrame:
    """
    Merge pipeline extracted features with the stellar physical priors to build
    the full V4 feature vector. Does NOT touch any pipeline logic.
    """
    feats = pipeline_result["features"]
    fit = pipeline_result["trapezoid_fit"]

    row = {
        "depth": feats.get("depth", fit.get("depth", 0.0)),
        "t_tot_hours": fit.get("t_tot", 0.0) * 24,
        "t_in_hours": fit.get("t_in", 0.0) * 24,
        "flat_bottom_hours": fit.get("flat_bottom_hours", 0.0),
        "ingress_fraction": fit.get("ingress_fraction", 0.5),
        "period": pipeline_result["bls"]["best_period"],
        "detection_significance": pipeline_result.get("detection_significance_used", 0.0),
        "odd_even_depth_diff": feats.get("odd_even_depth_diff", 0.0),
        "secondary_eclipse_depth": feats.get("secondary_eclipse_depth", 0.0),
        "secondary_eclipse_phase": feats.get("secondary_eclipse_phase", 0.5),
        "depth_snr": feats.get("depth_snr", 0.0),
        "n_signals_detected": feats.get("n_signals_detected", 1),
        "period_corrected": int(pipeline_result.get("period_alias_corrected", False)),
        # V4 physical priors (from user input or Sol defaults)
        "koi_srad": stellar_params.get("koi_srad", 1.0),
        "koi_steff": stellar_params.get("koi_steff", 5778.0),
        "koi_slogg": stellar_params.get("koi_slogg", 4.44),
        "koi_kepmag": stellar_params.get("koi_kepmag", 12.0),
        "centroid_offset_magnitude": stellar_params.get("centroid_offset_magnitude", 0.0),
    }

    # Reconstruct planetary radius using same formula as training
    koi_srad = row["koi_srad"]
    depth_val = max(row["depth"], 0.0)
    reconstructed_prad = np.sqrt(depth_val) * koi_srad * 109.2
    row["reconstructed_prad"] = float(np.log1p(reconstructed_prad))

    df = pd.DataFrame([row])
    # Ensure column order matches training exactly
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = FILL_DEFAULTS.get(col, 0.0)
    return df[FEATURE_COLUMNS]


def _classify(feature_row: pd.DataFrame) -> dict:
    """Run the classifier and return probabilities.
    
    Model-agnostic: slices the feature row to exactly the columns the loaded
    model was trained on (classifier.feature_names_). This means the backend
    works correctly regardless of whether the model on disk is v1 (10 features),
    v3 (13 features), or v4 (19 features).
    """
    if classifier is None or not classifier.is_fitted:
        return {"error": "Model not loaded"}

    # Use only the exact feature columns this specific model was trained on
    model_features = classifier.feature_names_
    # Fill any columns the model needs but we didn't compute with 0
    for col in model_features:
        if col not in feature_row.columns:
            feature_row[col] = 0.0
    aligned_row = feature_row[model_features]

    proba = classifier.predict_proba(aligned_row)[0]
    classes = classifier.classes_
    return {cls: float(p) for cls, p in zip(classes, proba)}


def _plot_to_base64(pipeline_result: dict, target_name: str) -> str:
    """Render the 3-panel pipeline plot in-memory and return as Base64 PNG."""
    arrs = pipeline_result["_arrays"]
    fit = arrs["fit_params"]
    phase_time = arrs["phase_time"]
    phase_flux = arrs["phase_flux"]
    time = arrs["time"]
    flux_flat = arrs["flux_flat"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), facecolor="#0d1117")
    for ax in axes:
        ax.set_facecolor("#0d1117")
        ax.tick_params(colors="#8b949e")
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")
        ax.title.set_color("#e6edf3")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # Panel 1: detrended light curve
    axes[0].plot(time, flux_flat, ".", ms=1.5, color="#58a6ff", alpha=0.6)
    axes[0].set_title(f"{target_name} — Detrended Light Curve")
    axes[0].set_xlabel("Time (days)")
    axes[0].set_ylabel("Normalized Flux")

    # Panel 2: phase-folded
    axes[1].plot(phase_time * 24, phase_flux, ".", ms=2, color="#d2a8ff", alpha=0.5)
    axes[1].set_title("Phase-Folded")
    axes[1].set_xlabel("Time from Mid-Transit (hours)")
    axes[1].set_ylabel("Normalized Flux")

    # Panel 3: trapezoid fit overlay
    smooth_t = np.linspace(phase_time.min(), phase_time.max(), 500)
    model_flux = _trapezoid(smooth_t, fit["f0"], fit["depth"], fit["t0"], fit["t_tot"], fit["t_in"])
    axes[2].plot(phase_time * 24, phase_flux, ".", ms=2, color="#8b949e", alpha=0.35, label="Observed")
    axes[2].plot(smooth_t * 24, model_flux, "-", color="#f78166", lw=2, label="Trapezoid Fit")
    axes[2].set_title("Transit Shape Fit")
    axes[2].set_xlabel("Time from Mid-Transit (hours)")
    axes[2].legend(fontsize=8, facecolor="#161b22", labelcolor="#e6edf3")

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _safe(val):
    """Convert any numpy/Python value to a JSON-serializable native Python type."""
    import math
    import numpy as np
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        v = float(val)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(val, float):
        return None if (math.isnan(val) or math.isinf(val)) else val
    return val


def _format_pipeline_diagnostics(result: dict) -> dict:
    """Extract a clean set of diagnostic metrics to return to the frontend.
    All values are cast to native Python types so JSONResponse can serialize them.
    """
    fit = result["trapezoid_fit"]
    feats = result["features"]
    raw = {
        "bls_period_days":          round(float(result["bls"]["best_period"]), 4),
        "tls_sde":                  round(float(result.get("tls", {}).get("SDE", result.get("detection_significance_used", 0))), 2),
        "tls_ran":                  bool(result.get("tls_ran", False)),
        "detection_passes":         bool(result.get("detection_passes_threshold", False)),
        "depth_pct":                round(float(fit.get("depth", 0)) * 100, 5),
        "depth_err_pct":            round(float(fit.get("depth_err", 0)) * 100, 5),
        "duration_hours":           round(float(fit.get("t_tot", 0)) * 24, 3),
        "duration_err_hours":       round(float(fit.get("t_tot_err", 0)) * 24, 3),
        "ingress_fraction":         round(float(fit.get("ingress_fraction", 0)), 4),
        "depth_snr":                round(float(feats.get("depth_snr", 0)), 2),
        "odd_even_diff":            round(float(feats.get("odd_even_depth_diff", 0)), 6),
        "secondary_eclipse_depth":  round(float(feats.get("secondary_eclipse_depth", 0)), 6),
        "period_alias_corrected":   bool(result.get("period_alias_corrected", False)),
        "n_signals_detected":       int(result.get("n_signals_detected", 1)),
        "single_transit_candidates": int(len(result.get("single_transit_candidates", []))),
    }
    # Final pass: make every value safe (catches any edge cases from numpy)
    return {k: _safe(v) for k, v in raw.items()}



# ── API Routes ────────────────────────────────────────────────────────────────

@app.post("/api/run_synthetic")
async def run_synthetic(case: str = Form("transit")):
    """
    Generate a synthetic lightcurve, run the full pipeline, classify, and return results.
    case: 'transit' | 'false_positive'
    """
    try:
        if case == "transit":
            t, flux, _ = make_synthetic_lightcurve(depth=0.0015)
            target_name = "SYNTH-TRANSIT-01"
        else:
            t, flux, _ = make_false_positive_lightcurve()
            target_name = "SYNTH-FALSEPOS-01"

        result = run_pipeline(t, flux, target_name=target_name, use_tls=True)

        feature_row = _build_feature_row(result, SOL_DEFAULTS)
        probabilities = _classify(feature_row)
        plot_b64 = _plot_to_base64(result, target_name)
        diagnostics = _format_pipeline_diagnostics(result)

        return JSONResponse({
            "target": target_name,
            "plot_base64": plot_b64,
            "probabilities": probabilities,
            "diagnostics": diagnostics,
            "stellar_params": SOL_DEFAULTS,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {traceback.format_exc()}")


@app.post("/api/run_custom")
async def run_custom(
    file: UploadFile = File(...),
    koi_srad: float = Form(1.0),
    koi_steff: float = Form(5778.0),
    koi_slogg: float = Form(4.44),
    koi_kepmag: float = Form(12.0),
    centroid_offset_magnitude: float = Form(0.0),
):
    """
    Accept a user CSV with 'time' and 'flux' columns, run the full pipeline,
    and return classification results.
    """
    try:
        contents = await file.read()
        df_input = pd.read_csv(io.BytesIO(contents))

        required = {"time", "flux"}
        if not required.issubset(set(df_input.columns)):
            raise HTTPException(
                status_code=422,
                detail=f"CSV must contain 'time' and 'flux' columns. Found: {list(df_input.columns)}"
            )

        t = df_input["time"].values.astype(float)
        flux = df_input["flux"].values.astype(float)

        target_name = os.path.splitext(file.filename)[0] if file.filename else "CUSTOM-STAR"
        result = run_pipeline(t, flux, target_name=target_name, use_tls=True)

        stellar_params = {
            "koi_srad": koi_srad,
            "koi_steff": koi_steff,
            "koi_slogg": koi_slogg,
            "koi_kepmag": koi_kepmag,
            "centroid_offset_magnitude": centroid_offset_magnitude,
        }

        feature_row = _build_feature_row(result, stellar_params)
        probabilities = _classify(feature_row)
        plot_b64 = _plot_to_base64(result, target_name)
        diagnostics = _format_pipeline_diagnostics(result)

        return JSONResponse({
            "target": target_name,
            "plot_base64": plot_b64,
            "probabilities": probabilities,
            "diagnostics": diagnostics,
            "stellar_params": stellar_params,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {traceback.format_exc()}")


# ── Static file serving ───────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root():
    return FileResponse("static/index.html")
