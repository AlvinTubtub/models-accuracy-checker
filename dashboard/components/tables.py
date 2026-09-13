"""Table formatters and styling components for the Streamlit dashboard."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from recheck.compare import ModelComparison


def render_company_metrics_table(comparisons: list[ModelComparison]) -> pd.DataFrame:
    """Format a clean side-by-side table comparing recomputed vs reported metrics."""

    rows = []
    for c in comparisons:
        label_map = {
            "lag_reg": "LIR (Linear Regression)",
            "arima": "ARIMA",
            "lstm": "LSTM",
            "naive": "Naive Benchmark",
        }
        rows.append(
            {
                "Model": label_map.get(c.model, c.model),
                "Rec. RMSE": f"{c.recomputed.rmse:.4f}",
                "Rep. RMSE": f"{float(c.reported['rmse']):.4f}",
                "Rec. MAE": f"{c.recomputed.mae:.4f}",
                "Rep. MAE": f"{float(c.reported['mae']):.4f}",
                "Rec. MASE": f"{c.recomputed.mase:.4f}",
                "Rep. MASE": f"{float(c.reported['mase']):.4f}",
                "Rec. R²": f"{c.recomputed.r2:.4f}",
                "Rep. R²": f"{float(c.reported['r2']):.4f}",
                "Max Diff": f"{c.max_abs_diff:.6f}",
                "Status": "✅ Match" if c.within_tolerance else "❌ Mismatch",
                "Beats Naive": "🏆 Yes" if c.beats_naive_recomputed else "No",
            }
        )
    return pd.DataFrame(rows)


def render_mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Format table of only mismatched metrics."""

    if df.empty:
        return pd.DataFrame()

    mismatches = df[~df["within_tolerance"]].copy()
    if mismatches.empty:
        return pd.DataFrame()

    return mismatches[
        [
            "symbol",
            "sector",
            "model_label",
            "max_abs_diff",
            "recomputed_rmse",
            "reported_rmse",
            "recomputed_mase",
            "reported_mase",
        ]
    ].rename(
        columns={
            "symbol": "Symbol",
            "sector": "Sector",
            "model_label": "Model",
            "max_abs_diff": "Discrepancy",
            "recomputed_rmse": "Rec. RMSE",
            "reported_rmse": "Rep. RMSE",
            "recomputed_mase": "Rec. MASE",
            "reported_mase": "Rep. MASE",
        }
    )
