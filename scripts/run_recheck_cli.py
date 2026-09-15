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
    dm_csv: Path | str | None = None,
    cross_company_csv: Path | str | None = None,
    wilcoxon_csv: Path | str | None = None,
    alpha: float = 0.05,
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

    # Provenance Classification
    tiers = {export.provenance_tier.value for export in result.exports.values()}
    tier_str = ", ".join(sorted(tiers))
    print(f"[*] Provenance Classification: {tier_str}")

    # Independent Naive Verification Preflight
    print("\n[*] INDEPENDENT NAIVE RECONSTRUCTION & VERIFICATION:")
    has_formal = any(e.provenance_tier.value == "FORMAL_FROZEN" for e in result.exports.values())
    if has_formal:
        raw_dir = Path("data/raw")
        naive_failures = 0
        for sym, export in result.exports.items():
            if export.provenance_tier.value == "FORMAL_FROZEN":
                csv_path = raw_dir / f"{sym}.csv"
                if not csv_path.is_file():
                    print(f"    [!] FAILED ({sym}): Required frozen raw CSV not found at {csv_path} (RAW_DATA_UNAVAILABLE)")
                    has_errors = True
                    naive_failures += 1
                else:
                    import pandas as pd
                    from recheck.naive import audit_naive_predictions
                    raw_df = pd.read_csv(csv_path)
                    audit_res = audit_naive_predictions(
                        symbol=sym,
                        raw_dates=raw_df["Date"],
                        raw_closes=raw_df["Close"],
                        target_dates=export.target_dates,
                        exported_naive_predictions=export.predicted_closes_by_model["naive"],
                        tolerance=tolerance,
                    )
                    if not audit_res.all_naive_predictions_match:
                        print(f"    [!] FAILED ({sym}): Reconstructed Naive mismatch ({audit_res.mismatch_count} discrepancies)")
                        has_errors = True
                        naive_failures += 1
                    else:
                        print(f"    [*] PASSED ({sym}): Reconstructed Naive matched 100% across {audit_res.total_sessions} sessions")
        if naive_failures == 0:
            print("    PASSED: All FORMAL_FROZEN evaluations verified against independent Naive benchmark.")
    else:
        print("    STATUS: NOT_RUN / DEMO_TIER")
        print("    NOTE: Loaded exports are classified as DEMO. Demo data powers test/dashboard workflows")
        print("          but cannot be certified as formal research evidence without verified raw inputs.")

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

    # 4. Print Development-Period MASE Scaling Reference Summary
    print("\n[*] DEVELOPMENT-PERIOD MASE SCALING REFERENCE:")
    print("    (MASE < 1.0 indicates error below the development-period Naive scaling reference.")
    print("     Direct holdout superiority is evaluated separately against the aligned holdout Naive forecast.)")
    win_summary = win_rate_summary(df)
    if not win_summary.empty:
        headers = ["Model", "MASE < 1.0", "Ratio %", "Median MASE", "Median RMSE"]
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

    # 4b. Direct Out-of-Sample (OOS) Comparison
    from recheck.oos import evaluate_company_direct_oos, oos_comparisons_to_dataframe
    oos_summaries = [evaluate_company_direct_oos(exp) for exp in result.exports.values()]
    oos_df = oos_comparisons_to_dataframe(oos_summaries)
    if not oos_df.empty:
        print("\n[*] DIRECT OUT-OF-SAMPLE (OOS) HOLDOUT COMPARISON VS. NAIVE:")
        oos_rows = []
        labels = {"lag_reg": "Lag-Informed Regression", "arima": "ARIMA", "lstm": "LSTM"}
        for model_name in ("lag_reg", "arima", "lstm"):
            sub = oos_df[oos_df["model"] == model_name]
            if sub.empty:
                continue
            total = len(sub)
            rmse_wins = int((sub["rmse_comparison_status"] == "WIN").sum())
            rmse_ties = int((sub["rmse_comparison_status"] == "TIE").sum())
            rmse_losses = int((sub["rmse_comparison_status"] == "LOSS").sum())
            valid_rmse_skill = sub["rmse_skill_value"].dropna()
            med_rmse_skill = (valid_rmse_skill.median() * 100.0) if not valid_rmse_skill.empty else None
            rmse_skill_str = f"{med_rmse_skill:+.2f}%" if med_rmse_skill is not None else "N/A"
            mae_wins = int((sub["mae_comparison_status"] == "WIN").sum())
            valid_mae_skill = sub["mae_skill_value"].dropna()
            med_mae_skill = (valid_mae_skill.median() * 100.0) if not valid_mae_skill.empty else None
            mae_skill_str = f"{med_mae_skill:+.2f}%" if med_mae_skill is not None else "N/A"
            oos_rows.append([
                labels.get(model_name, model_name),
                f"{rmse_wins}/{total} (ties: {rmse_ties})",
                rmse_skill_str,
                f"{mae_wins}/{total}",
                mae_skill_str,
            ])
        oos_headers = ["Model", "Beats Naive (RMSE)", "Median RMSE Skill", "Beats Naive (MAE)", "Median MAE Skill"]
        print(format_table(oos_headers, oos_rows, indent="    "))

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

    # 7. Within-Company Diebold-Mariano Statistical Testing
    from recheck.statistics import (
        REQUIRED_PAIRS,
        dm_results_to_dataframe,
        evaluate_company_pairwise_dm,
    )

    dm_results_by_company = {}
    for sym, exp in result.exports.items():
        try:
            dm_results_by_company[sym] = evaluate_company_pairwise_dm(exp, alpha=alpha)
        except Exception as ex:
            print(f"    [!] Error running DM test for {sym}: {ex}")

    print("\n[*] WITHIN-COMPANY DIEBOLD-MARIANO TESTS (Pairwise Holdout Equal-Accuracy Tests):")
    if any(e.provenance_tier.value == "DEMO" for e in result.exports.values()):
        print("    NOTE: Provenance is DEMO tier. Statistical tests on demo data are illustrative")
        print("          and cannot be cited as formal research evidence.")
    print(f"    Holm step-down family-wise error rate control per company (alpha = {alpha}, 6 pairs / family)")

    dm_df = dm_results_to_dataframe(dm_results_by_company)
    if not dm_df.empty:
        dm_summary_rows = []
        for m_a, m_b in REQUIRED_PAIRS:
            pair_label = f"{m_a} vs {m_b}"
            sq_sub = dm_df[(dm_df["pair"] == pair_label) & (dm_df["loss_function"] == "squared")]
            sq_sig_a = int((sq_sub["status"] == "A_SIGNIFICANTLY_LOWER_LOSS").sum())
            sq_sig_b = int((sq_sub["status"] == "B_SIGNIFICANTLY_LOWER_LOSS").sum())
            sq_not_sig = len(sq_sub) - sq_sig_a - sq_sig_b

            abs_sub = dm_df[(dm_df["pair"] == pair_label) & (dm_df["loss_function"] == "absolute")]
            abs_sig_a = int((abs_sub["status"] == "A_SIGNIFICANTLY_LOWER_LOSS").sum())
            abs_sig_b = int((abs_sub["status"] == "B_SIGNIFICANTLY_LOWER_LOSS").sum())
            abs_not_sig = len(abs_sub) - abs_sig_a - abs_sig_b

            dm_summary_rows.append([
                pair_label,
                f"{sq_sig_a} / {sq_sig_b} / {sq_not_sig}",
                f"{abs_sig_a} / {abs_sig_b} / {abs_not_sig}",
            ])

        dm_headers = [
            "Model Pair (A vs B)",
            "Squared Loss (Sig A / Sig B / Not Sig)",
            "Absolute Loss (Sig A / Sig B / Not Sig)",
        ]
        print(format_table(dm_headers, dm_summary_rows, indent="    "))

    if dm_csv:
        out_csv = Path(dm_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        dm_df.to_csv(out_csv, index=False)
        print(f"\n[*] Full Diebold-Mariano test results ({len(dm_df)} rows) saved to {out_csv}")

    # 8. Cross-Company Supporting Analysis
    from recheck.cross_company import (
        SUPPORTING_EVIDENCE_NOTE,
        evaluate_cross_company,
        method_stats_to_dataframe,
        rank_matrix_to_dataframe,
        wilcoxon_results_to_dataframe,
    )

    print("\n[*] CROSS-COMPANY SUPPORTING ANALYSIS (15 Matched Company Evaluations):")
    if any(e.provenance_tier.value == "DEMO" for e in result.exports.values()):
        print("    DEMO ONLY — NOT FORMAL RESEARCH EVIDENCE")
    print(f"    Provenance Tier: {tier_str}")
    print(f"    Matched Companies: {len(result.exports)}")
    print(f"    Primary Metric: Recomputed Out-of-Sample MASE (Scale-Independent)")
    print(f"    Note: {SUPPORTING_EVIDENCE_NOTE}")

    is_formal_run = any(e.provenance_tier.value == "FORMAL_FROZEN" for e in result.exports.values())
    cross_summary = evaluate_cross_company(result.exports, alpha=alpha, strict_formal=is_formal_run)

    # Descriptive Method Table
    stats_df = method_stats_to_dataframe(cross_summary)
    c_headers = [
        "Method",
        "Median MASE",
        "Mean Rank",
        "Median Rank",
        "Wins (Rank 1)",
        "Tied 1st",
        "MASE < 1.0",
    ]
    c_rows = [
        [
            row["method_label"],
            f"{row['median_mase']:.4f}",
            f"{row['mean_rank']:.2f}",
            f"{row['median_rank']:.1f}",
            str(row["count_ranked_first"]),
            str(row["count_tied_first"]),
            str(row["mase_below_one_count"]),
        ]
        for _, row in stats_df.iterrows()
    ]
    print("\n    Descriptive Method Summary:")
    print(format_table(c_headers, c_rows, indent="    "))

    # Friedman Omnibus Test
    f_res = cross_summary.friedman
    print(f"\n    Friedman Omnibus Test (df = {f_res.degrees_of_freedom}, alpha = {f_res.alpha}):")
    f_stat_str = f"{f_res.statistic:.4f}" if f_res.statistic is not None else "N/A"
    f_p_str = f"{f_res.raw_p_value:.6e}" if f_res.raw_p_value is not None else "N/A"
    kw_str = f"{f_res.kendalls_w:.4f}" if f_res.kendalls_w is not None else "N/A"
    print(f"      Friedman Statistic (chi^2): {f_stat_str}")
    print(f"      Raw p-value:                {f_p_str}")
    print(f"      Kendall's W (Effect Size):  {kw_str}")
    print(f"      Significance Status:        {f_res.status} (Significant: {f_res.significant})")

    # Conditional Wilcoxon Post-Hoc Tests
    if cross_summary.friedman.significant:
        print(f"\n    Pairwise Wilcoxon Post-Hoc Tests (Holm-Adjusted, m = {len(cross_summary.pairwise_wilcoxon)}):")
        w_headers = [
            "Pair (A vs B)",
            "Median MASE (A / B)",
            "Method",
            "Wilcoxon Stat",
            "Raw p-val",
            "Holm Adj p-val",
            "Status",
        ]
        w_rows = [
            [
                f"{r.model_a} vs {r.model_b}",
                f"{r.median_mase_a:.4f} / {r.median_mase_b:.4f}",
                r.actual_method_used,
                f"{r.wilcoxon_statistic:.1f}" if r.wilcoxon_statistic is not None else "N/A",
                f"{r.raw_p_value:.6f}" if r.raw_p_value is not None else "N/A",
                f"{r.holm_adjusted_p_value:.6f}" if r.holm_adjusted_p_value is not None else "N/A",
                r.status,
            ]
            for r in cross_summary.pairwise_wilcoxon
        ]
        print(format_table(w_headers, w_rows, indent="    "))
    else:
        print("\n    Post-hoc Wilcoxon: NOT RUN")
        print("    Reason: Friedman omnibus test was not significant.")

    if cross_company_csv:
        rank_df = rank_matrix_to_dataframe(cross_summary)
        out_cross = Path(cross_company_csv)
        out_cross.parent.mkdir(parents=True, exist_ok=True)
        rank_df.to_csv(out_cross, index=False)
        print(f"\n[*] Cross-company rank matrix saved to {out_cross}")

    if wilcoxon_csv:
        w_df = wilcoxon_results_to_dataframe(cross_summary.pairwise_wilcoxon)
        out_w = Path(wilcoxon_csv)
        out_w.parent.mkdir(parents=True, exist_ok=True)
        w_df.to_csv(out_w, index=False)
        print(f"\n[*] Pairwise Wilcoxon results saved to {out_w}")

    # Optional report export
    if report_path:
        out_report = Path(report_path)
        out_report.parent.mkdir(parents=True, exist_ok=True)
        audit_payload = {
            "data_directory": str(data_dir.resolve()),
            "tolerance": tolerance,
            "total_companies": len(result.exports),
            "invariant_pass": not has_errors,
            "mismatch_count": len(mismatches),
            "win_rate_summary": win_summary.to_dict(orient="records") if not win_summary.empty else [],
            "best_models": best_counts.to_dict(orient="records") if not best_counts.empty else [],
            "diebold_mariano_summary": dm_summary_rows if not dm_df.empty else [],
            "cross_company_analysis": {
                "provenance_tier": cross_summary.provenance_tier,
                "company_count": cross_summary.company_count,
                "friedman": {
                    "statistic": f_res.statistic,
                    "degrees_of_freedom": f_res.degrees_of_freedom,
                    "raw_p_value": f_res.raw_p_value,
                    "kendalls_w": f_res.kendalls_w,
                    "significant": f_res.significant,
                    "status": f_res.status,
                },
                "posthoc_status": cross_summary.posthoc_status,
                "lowest_median_mase_method": cross_summary.lowest_median_mase_method,
                "best_mean_rank_method": cross_summary.best_mean_rank_method,
                "most_company_wins_method": cross_summary.most_company_wins_method,
            },
        }
        with out_report.open("w", encoding="utf-8") as f:
            json.dump(audit_payload, f, indent=2)
        print(f"\n[*] Audit report saved to {out_report}")

    print(f"\n{'='*70}")
    is_formal = any(e.provenance_tier.value == "FORMAL_FROZEN" for e in result.exports.values())
    if has_errors:
        if is_formal:
            print("  FORMAL AUDIT VERDICT: FAILED (Discrepancies or schema errors detected)")
        else:
            print("  DEMO PIPELINE VERIFICATION: FAILED (Discrepancies or schema errors detected)")
            print("  FORMAL AUDIT VERDICT: NOT APPLICABLE (Demo data cannot certify formal experiment)")
        print(f"{'='*70}\n")
        return 1
    else:
        if is_formal:
            print("  FORMAL AUDIT VERDICT: PASSED (All invariants valid & metrics verified)")
        else:
            print("  DEMO PIPELINE VERIFICATION: PASSED (Pipeline executed cleanly on demo data)")
            print("  FORMAL AUDIT VERDICT: NOT APPLICABLE (Formal verification requires FORMAL_FROZEN experiment)")
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
    parser.add_argument(
        "--dm-csv",
        default=None,
        help="Optional path to export full within-company Diebold-Mariano test results to CSV",
    )
    parser.add_argument(
        "--cross-company-csv",
        default=None,
        help="Optional path to export cross-company per-company rank matrix to CSV",
    )
    parser.add_argument(
        "--wilcoxon-csv",
        default=None,
        help="Optional path to export pairwise Wilcoxon post-hoc test results to CSV",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical hypothesis tests (default: 0.05)",
    )
    args = parser.parse_args()
    return run_cli(
        args.data_dir,
        tolerance=args.tolerance,
        report_path=args.report,
        dm_csv=args.dm_csv,
        cross_company_csv=args.cross_company_csv,
        wilcoxon_csv=args.wilcoxon_csv,
        alpha=args.alpha,
    )


if __name__ == "__main__":
    raise SystemExit(main())
