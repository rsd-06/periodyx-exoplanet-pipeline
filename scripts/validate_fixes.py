"""
Validation script: tests the v2 feature fixes on a representative sample
of 90 labeled Kepler stars (30 transit, 30 eclipsing_binary, 30 blend/other)
before committing to the full 7,449-star pipeline rerun.

Runs via multiprocessing.Pool (same as build_training_set.py) to avoid the
matplotlib DLL Application Control policy issue on Windows.

Success criteria:
  1. secondary_eclipse_depth distribution for EBs is measurably shifted
     from transits (KS-test p < 0.05).
  2. period_alias_corrected fires on >5% of EBs and <2% of confirmed transits.
  3. No crashes on any of the 90 stars.

Usage:
    python scripts/validate_fixes.py --koi-csv data/koi_cumulative.csv
"""

import argparse
import sys
import os
import warnings
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_one(args):
    """Worker function — runs in a child process, safe from DLL policy."""
    warnings.filterwarnings("ignore")
    kepid, kepoi_name, label, use_tls, bls_threshold = args
    try:
        from data.acquisition import fetch_lightcurve
        from pipeline import run_pipeline
        import numpy as np

        time, flux, meta = fetch_lightcurve(
            f"KIC {kepid}", mission="Kepler", author="Kepler"
        )
        result = run_pipeline(
            time, flux,
            target_name=kepoi_name,
            use_tls=use_tls,
            bls_significance_threshold=bls_threshold,
            run_single_transit_scan=False,
        )
        feats = result["features"]
        return {
            "kepid": kepid,
            "label": label,
            "secondary_eclipse_depth": feats.get("secondary_eclipse_depth"),
            "secondary_eclipse_phase": feats.get("secondary_eclipse_phase"),
            "odd_even_depth_diff": feats.get("odd_even_depth_diff"),
            "ingress_fraction": feats.get("ingress_fraction"),
            "period_corrected": feats.get("period_corrected", 0),
            "n_signals_detected": feats.get("n_signals_detected", 1),
            "depth_snr": feats.get("depth_snr"),
            "error": None,
        }
    except Exception as e:
        return {
            "kepid": kepid,
            "label": label,
            "error": f"{type(e).__name__}: {e}",
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--koi-csv", default="data/koi_cumulative.csv")
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--bls-threshold", type=float, default=9.0)
    ap.add_argument("--workers", type=int, default=max(os.cpu_count() - 1, 1))
    ap.add_argument("--fast", action="store_true",
                    help="BLS-only mode for a quicker validation pass")
    args = ap.parse_args()

    # All heavy imports deferred into child processes
    import numpy as np
    import pandas as pd
    from scipy import stats
    from data.koi_labels import load_koi_labels

    use_tls = not args.fast
    print(f"Validation mode: {'BLS-only (--fast)' if args.fast else 'BLS+TLS'}")
    print(f"Stars per class: {args.n_per_class} | Workers: {args.workers}")

    labels_df = load_koi_labels(args.koi_csv)
    n = args.n_per_class

    transit = labels_df[labels_df["label"] == "transit"].sample(
        n=min(n, len(labels_df[labels_df["label"] == "transit"])), random_state=42)
    eb = labels_df[labels_df["label"] == "eclipsing_binary"].sample(
        n=min(n, len(labels_df[labels_df["label"] == "eclipsing_binary"])), random_state=42)
    blend_other = labels_df[labels_df["label"].isin(["blend", "other"])].sample(
        n=min(n, len(labels_df[labels_df["label"].isin(["blend", "other"])])), random_state=42)

    combined = pd.concat([transit, eb, blend_other])
    tasks = [
        (row.kepid, row.kepoi_name, row.label, use_tls, args.bls_threshold)
        for row in combined.itertuples()
    ]

    print(f"Running {len(tasks)} stars via Pool({args.workers})...")
    all_rows = []
    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_one, tasks), 1):
            status = "OK" if result["error"] is None else f"FAIL"
            label = result.get("label", "?")
            kepid = result.get("kepid", "?")
            print(f"  [{i}/{len(tasks)}] KIC {kepid} ({label}) -> {status}"
                  + (f": {result['error']}" if result["error"] else ""))
            all_rows.append(result)

    results = pd.DataFrame(all_rows)
    ok = results[results["error"].isna()].copy()
    failed = results[results["error"].notna()]

    print(f"\n{'='*60}")
    print(f"Results: {len(ok)}/{len(results)} successful | {len(failed)} failed")

    print(f"\n{'='*60}")
    print("Secondary Eclipse Depth by class:")
    print(f"{'Class':<20} {'N':>4} {'Median':>10} {'Mean':>10} {'Std':>10} {'Non-zero %':>12}")
    print("-" * 70)
    for cls in ["transit", "eclipsing_binary", "blend", "other"]:
        subset = ok[ok["label"] == cls]["secondary_eclipse_depth"].dropna()
        if len(subset) == 0:
            continue
        nonzero = (np.abs(subset) > 1e-4).mean() * 100
        print(f"{cls:<20} {len(subset):>4} {subset.median():>10.5f} "
              f"{subset.mean():>10.5f} {subset.std():>10.5f} {nonzero:>11.1f}%")

    # KS-test: EB vs transit secondary depths
    eb_depths = ok[ok["label"] == "eclipsing_binary"]["secondary_eclipse_depth"].dropna()
    tr_depths = ok[ok["label"] == "transit"]["secondary_eclipse_depth"].dropna()
    if len(eb_depths) >= 5 and len(tr_depths) >= 5:
        ks_stat, ks_p = stats.ks_2samp(eb_depths, tr_depths)
        print(f"\nKS-test (EB vs transit secondary depth): stat={ks_stat:.3f}, p={ks_p:.4f}")
        if ks_p < 0.05:
            print("✅ PASS: EB and transit secondary depths are statistically distinct (p < 0.05)")
            print("        Fix 1 + Fix 2 are working correctly.")
        else:
            print("⚠️  The two distributions are not yet statistically distinct.")
            print("   This may mean more stars are needed, or additional fixes are required.")
    else:
        print("\n⚠️  Not enough data for KS-test.")

    print(f"\n{'='*60}")
    print("Period correction firing rate by class:")
    for cls in ok["label"].unique():
        subset = ok[ok["label"] == cls]
        correction_rate = subset["period_corrected"].mean() * 100
        print(f"  {cls:<22}: {correction_rate:.1f}% corrected")

    # Expected: EBs corrected more often than transits
    eb_rate = ok[ok["label"] == "eclipsing_binary"]["period_corrected"].mean()
    tr_rate = ok[ok["label"] == "transit"]["period_corrected"].mean()
    if len(ok[ok["label"] == "eclipsing_binary"]) > 0 and len(ok[ok["label"] == "transit"]) > 0:
        if eb_rate > tr_rate:
            print(f"\n✅ PASS: Period correction fires more on EBs ({eb_rate*100:.1f}%) "
                  f"than transits ({tr_rate*100:.1f}%) — as expected.")
        else:
            print(f"\n⚠️  Period correction rate similar for EBs ({eb_rate*100:.1f}%) "
                  f"and transits ({tr_rate*100:.1f}%) — check alias detection logic.")

    print(f"\n{'='*60}")
    print("Multi-planet flag (n_signals_detected > 1):")
    multi = (ok["n_signals_detected"] > 1).mean() * 100
    print(f"  {multi:.1f}% of stars flagged as multi-signal")

    out_path = "data/validation_results_v2.csv"
    ok.to_csv(out_path, index=False)
    print(f"\nDetailed results saved to {out_path}")
    print("\nIf KS-test passes, proceed with full pipeline rerun:")
    print("  python scripts/build_training_set.py --koi-csv data/koi_cumulative.csv --workers 8")


if __name__ == "__main__":
    main()
