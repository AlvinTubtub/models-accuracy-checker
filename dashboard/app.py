"""Interactive Streamlit Dashboard for the PSE Model Accuracy Recheck (Tier 2).

Provides comprehensive auditing, visualization, and cross-checking of
ForecastPH's model predictions against the Naive benchmark over the
Jan 2, 2020 – Sep 11, 2026 evaluation window.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd
import streamlit as st

from dashboard.components.charts import (
    plot_actual_vs_predicted,
    plot_metric_distributions,
    plot_residuals,
    plot_time_series,
    plot_win_rates,
)
from dashboard.components.tables import render_company_metrics_table, render_mismatch_table
from recheck.aggregate import (
    best_model_counts,
    comparisons_to_dataframe,
    sector_breakdown,
    win_rate_summary,
)
from recheck.compare import compare_all, compare_company, mismatches_only
from recheck.loader import load_all_exports


st.set_page_config(
    page_title="PSE Model Accuracy Recheck",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner="Loading and validating evaluation exports...")
def get_evaluation_data(data_dir_str: str, tolerance: float):
    data_dir = Path(data_dir_str)
    result = load_all_exports(data_dir)
    comparisons = compare_all(result.exports, tolerance=tolerance)
    df = comparisons_to_dataframe(comparisons, result.exports)
    return result, comparisons, df


def main():
    st.title("📈 PSE Model Accuracy Recheck (Tier 2)")
    st.caption(
        "Independent Out-of-Sample Accuracy Audit vs. Naive Benchmark | "
        "Evaluation Span: Jan 2, 2020 – Sep 11, 2026"
    )

    # Sidebar Controls
    st.sidebar.header("⚙️ Audit Configuration")

    default_data_dir = str(REPO_ROOT / "data" / "evaluations")
    data_dir_input = st.sidebar.text_input(
        "Evaluation Data Directory",
        value=default_data_dir,
        help="Path to folder containing exported <SYMBOL>.json files",
    )

    tolerance = st.sidebar.select_slider(
        "Numerical Discrepancy Tolerance",
        options=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2],
        value=1e-4,
        format_func=lambda x: f"{x:.1e}",
        help="Tolerance threshold for comparing recomputed metrics vs reported metrics",
    )

    data_dir = Path(data_dir_input)

    # Empty directory helper
    if not data_dir.is_dir() or not list(data_dir.glob("*.json")):
        st.sidebar.warning("⚠️ No evaluation JSON files found in data directory.")
        if st.sidebar.button("Generate Synthetic Demo Data"):
            from scripts.generate_sample_data import main as gen_main
            gen_main()
            st.rerun()

    if st.sidebar.button("🔄 Refresh / Reload Data"):
        st.cache_data.clear()
        st.rerun()

    if not data_dir.is_dir():
        st.error(f"Specified directory does not exist: `{data_dir}`")
        st.info("Please verify the folder path or run `scripts/generate_sample_data.py` to create demo data.")
        return

    result, comparisons, df = get_evaluation_data(str(data_dir), tolerance)

    if not result.exports:
        st.warning("No valid `<SYMBOL>.json` files found in the specified path.")
        st.markdown(
            """
            ### Next Steps:
            1. **Export from Original Repo**: Run `bridge/export_evaluations.py` in the original forecasting repository and copy the output here.
            2. **Generate Demo Data**: Click **'Generate Synthetic Demo Data'** in the sidebar to test immediately.
            """
        )
        return

    # Dataset Manifest info
    if result.manifest:
        is_syn = result.manifest.get("is_synthetic", False)
        badge = "🟡 Synthetic Demo Data" if is_syn else "🟢 Production Fresh Model Artifacts"
        st.sidebar.markdown(f"**Data Source:** {badge}")
        exported_at = result.manifest.get("exported_at_utc") or result.manifest.get("generated_at_utc")
        if exported_at:
            st.sidebar.caption(f"Exported: {exported_at[:19]} UTC")

    # High-level metrics
    total_companies = len(result.exports)
    mismatches = mismatches_only(comparisons)
    audit_passed = len(mismatches) == 0 and len(result.load_errors) == 0

    tabs = st.tabs([
        "📊 Executive Summary & Naive Benchmark",
        "🔍 Company Deep-Dive",
        "🌐 Cross-Model & Sector Analysis",
        "🛡️ Data Audit & Integrity Inspector",
    ])

    # =========================================================================
    # TAB 1: Executive Summary
    # =========================================================================
    with tabs[0]:
        st.subheader("Executive Accuracy Summary")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Companies Evaluated", f"{total_companies} / 15")
        kpi2.metric("Total Model Series Audited", f"{len(df)}")
        kpi3.metric(
            "Audit Integrity Verdict",
            "PASSED" if audit_passed else "FLAGGED",
            delta="100% Agreement" if audit_passed else f"{len(mismatches)} Discrepancies",
            delta_color="normal" if audit_passed else "inverse",
        )
        # Average win rate across principal models
        win_sum = win_rate_summary(df)
        avg_win_rate = win_sum["win_rate_pct"].mean() if not win_sum.empty else 0.0
        kpi4.metric("Average Model Win-Rate", f"{avg_win_rate:.1f}%")

        st.markdown("---")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.markdown("#### Model Win Rate vs. Naive Baseline")
            st.caption("Percentage of companies where the model achieves **MASE < 1.0** (outperforming the Naive lag-1 persistence benchmark).")
            if not win_sum.empty:
                fig_win = plot_win_rates(win_sum)
                st.plotly_chart(fig_win, use_container_width=True)

        with col_right:
            st.markdown("#### Best Model Frequency (Lowest RMSE)")
            st.caption("How often each principal forecasting model was selected as optimal per company.")
            best_counts_df = best_model_counts(df)
            if not best_counts_df.empty:
                st.dataframe(
                    best_counts_df.rename(
                        columns={
                            "model_label": "Model",
                            "companies_best": "Companies Ranked #1",
                        }
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        st.markdown("---")
        st.markdown("#### All Companies: Performance Overview")
        
        # Overview table
        overview_rows = []
        for symbol, comp_list in comparisons.items():
            export = result.exports[symbol]
            principal_comps = [c for c in comp_list if c.model in ["lag_reg", "arima", "lstm"]]
            best_c = min(principal_comps, key=lambda c: c.recomputed.rmse)
            naive_c = next((c for c in comp_list if c.model == "naive"), None)
            
            beats_naive = any(c.beats_naive_recomputed for c in principal_comps)
            overview_rows.append({
                "Symbol": symbol,
                "Company Name": export.name or symbol,
                "Sector": export.sector or "Unknown",
                "Best Model": best_c.model.upper(),
                "Best RMSE": f"₱{best_c.recomputed.rmse:.3f}",
                "Best MASE": f"{best_c.recomputed.mase:.3f}",
                "Naive RMSE": f"₱{naive_c.recomputed.rmse:.3f}" if naive_c else "N/A",
                "Beats Naive?": "🏆 Yes" if best_c.beats_naive_recomputed else "❌ No",
                "Audit Status": "✅ OK" if all(c.within_tolerance for c in comp_list) else "⚠️ Diff",
            })
        
        st.dataframe(pd.DataFrame(overview_rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Company Deep-Dive
    # =========================================================================
    with tabs[1]:
        st.subheader("Individual Company Deep-Dive")

        selected_symbol = st.selectbox(
            "Select Company to Inspect",
            options=sorted(result.exports.keys()),
            format_func=lambda s: f"{s} — {result.exports[s].name or s} ({result.exports[s].sector or 'Sector'})",
        )

        export = result.exports[selected_symbol]
        comp_comparisons = comparisons[selected_symbol]

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        c_col1.metric("Sector", export.sector or "N/A")
        c_col2.metric("Evaluation Trading Sessions", f"{len(export.target_dates)} days")
        c_col3.metric("MASE Scale Denominator", f"₱{export.mase_denominator:.4f}")
        c_col4.metric("Reported Best Model", export.reported_best_model.upper() or "N/A")

        # Interactive Price Forecast Chart
        st.markdown("#### Out-of-Sample Price Forecasts vs. Actual Market Close")
        fig_ts = plot_time_series(
            export.target_dates,
            export.actual_closes,
            export.predicted_closes_by_model,
            selected_symbol,
            name=export.name,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # Residuals Chart
        st.markdown("#### Prediction Residuals over Time ($Predicted - Actual$)")
        fig_res = plot_residuals(
            export.target_dates,
            export.actual_closes,
            export.predicted_closes_by_model,
            selected_symbol,
        )
        st.plotly_chart(fig_res, use_container_width=True)

        # Side-by-side Recomputed vs Reported Metrics
        st.markdown("#### Metric Verification & Discrepancy Diff")
        st.caption(
            "Comparing independently recomputed metrics from raw predicted vs. actual arrays "
            f"against values reported by the original pipeline (Tolerance: {tolerance:.1e})."
        )
        metrics_table_df = render_company_metrics_table(comp_comparisons)
        st.dataframe(metrics_table_df, use_container_width=True, hide_index=True)

        # Scatter Parity Plot
        st.markdown("#### Parity Scatter Plot")
        chosen_model = st.selectbox(
            "Select model for Actual vs. Predicted parity check:",
            options=["lag_reg", "arima", "lstm", "naive"],
            format_func=lambda x: {"lag_reg": "Lag Regression (LIR)", "arima": "ARIMA", "lstm": "LSTM", "naive": "Naive"}[x],
        )
        if chosen_model in export.predicted_closes_by_model:
            fig_scat = plot_actual_vs_predicted(
                export.actual_closes,
                export.predicted_closes_by_model[chosen_model],
                chosen_model.upper(),
            )
            st.plotly_chart(fig_scat, use_container_width=True)

    # =========================================================================
    # TAB 3: Cross-Model & Sector Analysis
    # =========================================================================
    with tabs[2]:
        st.subheader("Cross-Model & Sector Breakdown")

        metric_to_analyze = st.selectbox(
            "Select Metric to Analyze:",
            options=["recomputed_mase", "recomputed_rmse", "recomputed_mae", "recomputed_r2"],
            format_func=lambda m: {"recomputed_mase": "MASE (Mean Absolute Scaled Error)", "recomputed_rmse": "RMSE", "recomputed_mae": "MAE", "recomputed_r2": "R²"}[m],
        )

        fig_dist = plot_metric_distributions(df, metric=metric_to_analyze)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Sector-Level Median MASE")
        st.caption("Median recomputed MASE per sector. Lower is better; MASE < 1.0 beats the Naive baseline.")
        sector_df = sector_breakdown(df)
        if not sector_df.empty:
            st.dataframe(sector_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### Raw Tabular Comparisons (All Companies)")
        st.dataframe(df, use_container_width=True)
        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Audit Comparisons as CSV",
            data=csv_data,
            file_name="pse_models_accuracy_recheck.csv",
            mime="text/csv",
        )

    # =========================================================================
    # TAB 4: Data Audit & Integrity Inspector
    # =========================================================================
    with tabs[3]:
        st.subheader("Data Audit & Integrity Verification")

        # Invariant checks summary
        st.markdown("#### 1. Invariant Integrity Checklist")
        invariants = [
            ("Chronological Order", "Dates strictly increasing with no reversed ordering", True),
            ("No Duplicate Dates", "Every trading session has a unique target date", True),
            ("Length Alignment", "Length of predicted series strictly equals target dates count", True),
            ("Consensus Actual Prices", "Actual close values agree across all model records", True),
            ("Finite Numbers", "No NaN, Inf, or unparseable values present", True),
        ]

        inv_cols = st.columns(len(invariants))
        for i, (name, desc, default_status) in enumerate(invariants):
            has_fail = any(
                name.lower() in issue.message.lower()
                for export in result.exports.values()
                for issue in export.issues
            )
            status = not has_fail
            inv_cols[i].metric(name, "PASSED" if status else "FAILED")

        st.markdown("---")
        st.markdown("#### 2. Metric Discrepancy Log")
        if mismatches:
            st.error(f"Found {len(mismatches)} metric comparison(s) exceeding tolerance {tolerance:.1e}:")
            st.dataframe(render_mismatch_table(df), use_container_width=True, hide_index=True)
        else:
            st.success(f"✅ Zero discrepancies found! Recomputed metrics match reported figures within tolerance {tolerance:.1e}.")

        st.markdown("---")
        st.markdown("#### 3. Raw Export JSON Viewer")
        inspect_sym = st.selectbox("Inspect JSON for company:", options=sorted(result.exports.keys()), key="json_view")
        if inspect_sym:
            json_file = data_dir / f"{inspect_sym}.json"
            if json_file.is_file():
                with json_file.open("r", encoding="utf-8") as f:
                    raw_content = json.load(f)
                with st.expander("Show Raw JSON Structure", expanded=False):
                    st.json(raw_content)


if __name__ == "__main__":
    main()
