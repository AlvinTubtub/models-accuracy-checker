#!/usr/bin/env python3
"""Headless CLI runner for the PSE Model Accuracy Recheck pipeline.

Loads exported evaluation JSON files, audits schema invariants, recomputes
metrics from scratch, compares them against reported values within tolerance,
and summarizes whether models outperform the Naive baseline.

Exits with:
  0: All invariants hold, metrics match within tolerance.
  1: Invariant violations, file load errors, or metric discrepancies found.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from recheck.aggregate import (
    best_model_counts,
    comparisons_to_dataframe,
    mismatch_table,
    sector_breakdown,
    win_rate_summary,
)
from recheck.compare import DEFAULT_TOLERANCE, compare_all, mismatches_only
from recheck.loader import load_all_exports


def format_table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> str:
    """Simple terminal table formatter without third-party dependencies."""
    if not rows:
        return f"{indent}(no data)"
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(val)))

    header_line = indent + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
    sep_line = indent + "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    data_lines = [
        indent + " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def run_cli(
    data_dir: Path | str,
    *,
    tolerance: float = 1e-4,
    report_path: Path | str | None = None,
) -> int:
    data_dir = Path(data_dir)
    print(f"\n{'='*70}")
    print(f"  PSE MODEL ACCURACY RECHECK AUDIT (TIER 2)")
    print(f"  Target directory: {data_dir.resolve()}")
    print(f"  Numerical tolerance: {tolerance}")
    print(f"{'='*70}\n")

    if not data_dir.is_dir():
        print(f"ERROR: Directory '{data_dir}' does not exist.", file=sys.stderr)
        return 1

    # 1. Ingestion
    result = load_all_exports(data_dir)
    total_files = len(result.exports) + len(result.load_errors)

    print(f"[*] Found {total_files} file(s) ({len(result.exports)} valid, {len(result.load_errors)} failed)")

    has_errors = False

    if result.load_errors:
        print("\n[!] INGESTION / SCHEMA ERRORS:")
        for sym, err in result.load_errors.items():
            print(f"    - {sym}: {err}")
        has_errors = True

    # Check for soft validation issues inside loaded exports
    validation_issue_count = 0
    for export in result.exports.values():
        if export.issues:
            for issue in export.issues:
                prefix = "[!] ERROR" if issue.severity == "error" else "[*] WARNING"
                print(f"    {prefix} ({issue.symbol}): {issue.message}")
                if issue.severity == "error":
                    has_errors = True
                    validation_issue_count += 1

    if not result.exports:
        print("\nERROR: No valid evaluation JSON files could be loaded. Aborting.", file=sys.stderr)
        return 1

    # 2. Recompute and Compare
    comparisons = compare_all(result.exports, tolerance=tolerance)
    mismatches = mismatches_only(comparisons)
    df = comparisons_to_dataframe(comparisons, result.exports)

    # 3. Print Discrepancy Status
    print(f"\n[*] METRIC VERIFICATION (Recomputed vs. Reported within tolerance {tolerance}):")
    if mismatches:
        print(f"    FAILED: Found {len(mismatches)} metric mismatch(es) exceeding tolerance:")
        mismatch_df = mismatch_table(df)
        headers = ["Symbol", "Model", "Metric Diff", "Rec. RMSE", "Rep. RMSE", "Rec. MASE", "Rep. MASE"]
        rows = [
            [
                row["symbol"],
                row["model"],
                f"{row['max_abs_diff']:.6f}",
                f"{row['recomputed_rmse']:.4f}",
                f"{row['reported_rmse']:.4f}",
                f"{row['recomputed_mase']:.4f}",
                f"{row['reported_mase']:.4f}",
            ]
            for _, row in mismatch_df.iterrows()
        ]
        print(format_table(headers, rows, indent="    "))
        has_errors = True
    else:
        print(f"    PASSED: 100% agreement across all {len(df)} evaluated model series.")

    # 4. Print Accuracy vs. Naive Benchmark Win-Rate Summary
    print("\n[*] ACCURACY VS. NAIVE BENCHMARK (MASE < 1.0 means beating Naive):")
    win_summary = win_rate_summary(df)
    if not win_summary.empty:
        headers = ["Model", "Beats Naive", "Win Rate %", "Median MASE", "Median RMSE"]
        rows = [
            [
                row["model_label"],
                f"{int(row['companies_beating_naive'])}/{int(row['total_companies'])}",
                f"{row['win_rate_pct']:.1f}%",
                f"{row['median_mase']:.4f}",
                f"{row['median_rmse']:.4f}",
            ]
            for _, row in win_summary.iterrows()
        ]
        print(format_table(headers, rows, indent="    "))

    # 5. Best Model Counts
    print("\n[*] PRINCIPAL MODEL RANKINGS (Lowest Recomputed RMSE):")
    best_counts = best_model_counts(df)
    if not best_counts.empty:
        headers = ["Model", "Times Ranked Best", "Share %"]
        total_comp = len(result.exports)
        rows = [
            [
                row["model_label"],
                str(row["companies_best"]),
                f"{(row['companies_best'] / total_comp * 100.0):.1f}%",
            ]
            for _, row in best_counts.iterrows()
        ]
        print(format_table(headers, rows, indent="    "))

    # 6. Sector Breakdown
    print("\n[*] SECTOR SUMMARY (Median Recomputed MASE by Sector):")
    sector_df = sector_breakdown(df)
    if not sector_df.empty:
        headers = ["Sector", "Model", "Median MASE"]
        rows = [
            [row["sector"], row["model_label"], f"{row['median_mase']:.4f}"]
            for _, row in sector_df.iterrows()
        ]
        print(format_table(headers, rows, indent="    "))

    # Optional report export
    if report_path:
        out_report = Path(report_path)
        audit_payload = {
            "data_directory": str(data_dir.resolve()),
            "tolerance": tolerance,
            "total_companies": len(result.exports),
            "invariant_pass": not has_errors,
            "mismatch_count": len(mismatches),
            "win_rate_summary": win_summary.to_dict(orient="records") if not win_summary.empty else [],
            "best_models": best_counts.to_dict(orient="records") if not best_counts.empty else [],
        }
        with out_report.open("w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)
        print(f"\n[*] Audit report saved to {out_report}")

    print(f"\n{'='*70}")
    if has_errors:
        print("  OVERALL AUDIT VERDICT: FAILED (Discrepancies or schema errors detected)")
        print(f"{'='*70}\n")
        return 1
    else:
        print("  OVERALL AUDIT VERDICT: PASSED (All invariants valid & metrics verified)")
        print(f"{'='*70}\n")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless PSE Model Accuracy Recheck CLI")
    parser.add_argument(
        "--data-dir",
        "-d",
        default="data/evaluations",
        help="Path to directory containing exported company JSON files (default: data/evaluations)",
    )
    parser.add_argument(
        "--tolerance",
        "-t",
        type=float,
        default=1e-4,
        help="Float comparison tolerance (default: 1e-4 to account for exported rounding)",
    )
    parser.add_argument(
        "--report",
        "-r",
        default=None,
        help="Optional path to write a JSON audit summary report",
    )
    args = parser.parse_args()
    return run_cli(args.data_dir, tolerance=args.tolerance, report_path=args.report)


if __name__ == "__main__":
    raise SystemExit(main())
