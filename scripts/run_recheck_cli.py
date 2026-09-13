#!/usr/bin/env python3
"""Headless recheck report: load exported evaluations, recompute metrics,
compare against reported values, print a summary, and exit non-zero if
anything looks wrong (validation errors, or recomputed metrics that don't
match what the original pipeline reported beyond tolerance).

Usage:
    python scripts/run_recheck_cli.py --data-dir data/evaluations
    python scripts/run_recheck_cli.py --data-dir data/evaluations --csv report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recheck.aggregate import (
    best_model_counts,
    comparisons_to_dataframe,
    mismatch_table,
    sector_breakdown,
    win_rate_summary,
)
from recheck.compare import DEFAULT_TOLERANCE, compare_all
from recheck.loader import load_all_exports


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/evaluations", help="Directory of exported <SYMBOL>.json files")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE, help="Numerical tolerance for reported-vs-recomputed comparison")
    parser.add_argument("--csv", default=None, help="Optional path to write the full per-company/per-model comparison table as CSV")
    arguments = parser.parse_args(argv)

    result = load_all_exports(Path(arguments.data_dir))

    if result.load_errors:
        print(f"WARNING: {len(result.load_errors)} file(s) failed to load/validate:", file=sys.stderr)
        for symbol, error in result.load_errors.items():
            print(f"  {symbol}: {error}", file=sys.stderr)

    if not result.exports:
        print("No valid company exports found - nothing to recheck.", file=sys.stderr)
        return 2

    validation_failed = False
    for symbol, export in result.exports.items():
        if export.has_errors:
            validation_failed = True
            for issue in export.issues:
                if issue.severity == "error":
                    print(f"VALIDATION ERROR [{symbol}]: {issue.message}", file=sys.stderr)
        else:
            for issue in export.issues:
                print(f"VALIDATION WARNING [{symbol}]: {issue.message}", file=sys.stderr)

    comparisons = compare_all(result.exports, tolerance=arguments.tolerance)
    df = comparisons_to_dataframe(comparisons, result.exports)

    print(f"\nLoaded {len(result.exports)} companies, evaluation window "
          f"{df['evaluation_start'].min()} .. {df['evaluation_end'].max()}\n")

    print("=== Win rate vs. Naive (recomputed, principal models only) ===")
    print(win_rate_summary(df).to_string(index=False))

    print("\n=== Best model by company (lowest recomputed RMSE) ===")
    print(best_model_counts(df).to_string(index=False))

    print("\n=== Median MASE by sector ===")
    breakdown = sector_breakdown(df)
    if breakdown.empty:
        print("(no sector information available)")
    else:
        print(breakdown.to_string(index=False))

    mismatches = mismatch_table(df)
    print(f"\n=== Reported vs. recomputed mismatches (tolerance={arguments.tolerance}) ===")
    if mismatches.empty:
        print("None - every reported metric matches this repo's independent recomputation.")
    else:
        print(
            mismatches[
                ["symbol", "model_label", "reported_rmse", "recomputed_rmse",
                 "reported_mase", "recomputed_mase", "max_abs_diff"]
            ].to_string(index=False)
        )

    if arguments.csv:
        df.to_csv(arguments.csv, index=False)
        print(f"\nFull comparison table written to {arguments.csv}")

    if validation_failed:
        print("\nRESULT: FAILED - one or more company exports had validation errors.", file=sys.stderr)
        return 1
    if not mismatches.empty:
        print(f"\nRESULT: MISMATCHES FOUND - {len(mismatches)} model/company combination(s) "
              f"outside tolerance.", file=sys.stderr)
        return 1

    print("\nRESULT: PASSED - all recomputed metrics match reported values within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
