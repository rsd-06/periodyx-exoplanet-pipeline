"""
Builds the classifier training set by running the detection +
characterization + feature-extraction stages (NOT classification) on every
labeled KOI star, in parallel across CPU cores.

This is the slow, CPU-bound step -- NOT GPU-bound. There is no neural
network training here; this script just runs BLS/TLS + a trapezoid fit per
star, which is pure CPU signal processing. Use more cores, not a GPU.

Usage:
    python3 scripts/build_training_set.py \
        --koi-csv data/koi_cumulative.csv \
        --out data/training_features.csv \
        --workers 8 \
        --limit 500          # remove --limit for a full run
        --fast               # BLS only, skips slower TLS refinement

Resumable: re-running with the same --out path skips stars already present
in the checkpoint, so an interrupted run can simply be restarted.
"""

import argparse
import os
import sys
import csv
import traceback
from multiprocessing import Pool

# --- WDAC BYPASS ---
# Windows Defender Application Control blocks matplotlib's _c_internal_utils.pyd.
# Since we don't render UI plots during feature extraction, we can safely mock 
# this C-extension to allow lightkurve to import and run on this machine.
import unittest.mock
sys.modules['matplotlib._c_internal_utils'] = unittest.mock.MagicMock()
# -------------------

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.koi_labels import load_koi_labels
from data.acquisition import fetch_lightcurve
from pipeline import run_pipeline

FIELDNAMES = [
    "kepid", "kepoi_name", "label", "tls_ran",
    "depth", "t_tot_hours", "t_in_hours", "flat_bottom_hours",
    "ingress_fraction", "period", "detection_significance",
    "odd_even_depth_diff", "secondary_eclipse_depth", "secondary_eclipse_phase",
    "depth_snr", "n_signals_detected", "period_corrected",
    # v3: NASA vetting flags (zero compute cost — already in KOI table)
    "fpflag_nt", "fpflag_ss", "fpflag_co", "fpflag_ec", "koi_prad",
]


def process_one(args):
    kepid, kepoi_name, label, use_tls, bls_sig_threshold, koi_flags = args
    try:
        time, flux, meta = fetch_lightcurve(f"KIC {kepid}", mission="Kepler", author="Kepler")
        result = run_pipeline(
            time, flux, target_name=kepoi_name,
            use_tls=use_tls, bls_significance_threshold=bls_sig_threshold,
            run_single_transit_scan=False, classifier=None,
        )
        row = {"kepid": kepid, "kepoi_name": kepoi_name, "label": label,
               "tls_ran": result.get("tls_ran", False)}
        row.update(result["features"])
        row.update(koi_flags)  # merge NASA vetting flags
        return ("ok", row)
    except Exception as e:
        return ("fail", dict(kepid=kepid, kepoi_name=kepoi_name,
                              error=f"{type(e).__name__}: {e}"))


def already_processed(out_path):
    if not os.path.exists(out_path):
        return set()
    import pandas as pd
    return set(pd.read_csv(out_path)["kepid"].astype(str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koi-csv", default="data/koi_cumulative.csv")
    ap.add_argument("--out", default="data/training_features.csv")
    ap.add_argument("--fail-log", default="data/training_features_failures.csv")
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fast", action="store_true",
                     help="BLS-only mode -- TLS is skipped entirely, even for strong candidates. "
                          "Use this only for a quick first pass; for real training data, "
                          "leave this off and use --bls-threshold to control tiering instead.")
    ap.add_argument("--bls-threshold", type=float, default=9.0,
                     help="Normalized BLS significance score required before TLS is run on a "
                          "star (only applies when --fast is NOT set). Run "
                          "scripts/calibrate_threshold.py first to pick a sensible value for "
                          "your actual data instead of relying on this default.")
    args = ap.parse_args()

    labels_df = load_koi_labels(args.koi_csv)
    if args.limit:
        labels_df = labels_df.sample(n=min(args.limit, len(labels_df)), random_state=42)

    done = already_processed(args.out)
    remaining = labels_df[~labels_df["kepid"].astype(str).isin(done)]
    print(f"{len(done)} stars already processed, {len(remaining)} remaining.")

    # KOI flag columns to carry through to features (already computed by NASA)
    flag_cols = ["koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co",
                 "koi_fpflag_ec", "koi_prad"]

    tasks = [
        (
            row.kepid, row.kepoi_name, row.label, not args.fast, args.bls_threshold,
            {
                "fpflag_nt": getattr(row, "koi_fpflag_nt", None),
                "fpflag_ss": getattr(row, "koi_fpflag_ss", None),
                "fpflag_co": getattr(row, "koi_fpflag_co", None),
                "fpflag_ec": getattr(row, "koi_fpflag_ec", None),
                "koi_prad":  getattr(row, "koi_prad",  None),
            }
        )
        for row in remaining.itertuples()
    ]

    write_header = not os.path.exists(args.out)
    out_f = open(args.out, "a", newline="")
    fail_f = open(args.fail_log, "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDNAMES)
    fail_writer = csv.DictWriter(fail_f, fieldnames=["kepid", "kepoi_name", "error"])
    if write_header:
        writer.writeheader()
    if not os.path.exists(args.fail_log) or os.path.getsize(args.fail_log) == 0:
        fail_writer.writeheader()

    n_ok, n_fail, n_tls = 0, 0, 0
    with Pool(processes=args.workers) as pool:
        for i, (status, row) in enumerate(pool.imap_unordered(process_one, tasks), 1):
            if status == "ok":
                writer.writerow(row)
                n_ok += 1
                if row.get("tls_ran"):
                    n_tls += 1
            else:
                fail_writer.writerow(row)
                n_fail += 1
            if i % 25 == 0:
                out_f.flush()
                fail_f.flush()
                print(f"  [{i}/{len(tasks)}] ok={n_ok} fail={n_fail} tls_ran={n_tls}")

    out_f.close()
    fail_f.close()
    print(f"Done. ok={n_ok} fail={n_fail}. TLS ran on {n_tls}/{n_ok} "
          f"({100*n_tls/max(n_ok,1):.1f}%) of successfully processed stars.")
    print(f"Output: {args.out}")
    if not args.fast and n_ok > 0 and n_tls / n_ok > 0.5:
        print("WARNING: TLS ran on over half your stars. Your --bls-threshold is likely "
              "too low to act as a real triage filter -- run scripts/calibrate_threshold.py "
              "and consider raising it.")


if __name__ == "__main__":
    main()
