"""Cross-company roll-ups: per-model win rates against Naive, sector views."""

from __future__ import annotations

import pandas as pd

from recheck.compare import ModelComparison
from recheck.schema import PRINCIPAL_MODELS, REQUIRED_MODELS, CompanyEvaluationExport

MODEL_LABELS = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive",
}


def comparisons_to_dataframe(
    comparisons_by_symbol: dict[str, list[ModelComparison]],
    exports: dict[str, CompanyEvaluationExport],
) -> pd.DataFrame:
    """One row per (company, model): reported and recomputed metrics side by side."""

    rows = []
    for symbol, comparisons in comparisons_by_symbol.items():
        export = exports[symbol]
        for comparison in comparisons:
            rows.append(
                {
                    "symbol": symbol,
                    "name": export.name or symbol,
                    "sector": export.sector or "Unknown",
                    "model": comparison.model,
                    "model_label": MODEL_LABELS.get(comparison.model, comparison.model),
                    "is_principal": comparison.model in PRINCIPAL_MODELS,
                    "recomputed_rmse": comparison.recomputed.rmse,
                    "recomputed_mae": comparison.recomputed.mae,
                    "recomputed_mase": comparison.recomputed.mase,
                    "recomputed_r2": comparison.recomputed.r2,
                    "reported_rmse": float(comparison.reported["rmse"]),
                    "reported_mae": float(comparison.reported["mae"]),
                    "reported_mase": float(comparison.reported["mase"]),
                    "reported_r2": float(comparison.reported["r2"]),
                    "max_abs_diff": comparison.max_abs_diff,
                    "within_tolerance": comparison.within_tolerance,
                    "beats_naive_recomputed": comparison.beats_naive_recomputed,
                    "beats_naive_reported": comparison.beats_naive_reported,
                    "observations": comparison.recomputed.observations,
                    "evaluation_start": export.evaluation_start.isoformat(),
                    "evaluation_end": export.evaluation_end.isoformat(),
                }
            )
    return pd.DataFrame(rows)


def win_rate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per principal model: how many / what % of companies beat Naive (recomputed)."""

    if df.empty:
        return pd.DataFrame()

    principal = df[df["is_principal"]].copy()
    total_companies = principal["symbol"].nunique()
    if total_companies == 0:
        return pd.DataFrame()

    summary = (
        principal.groupby(["model", "model_label"], as_index=False)
        .agg(
            companies_beating_naive=("beats_naive_recomputed", "sum"),
            median_mase=("recomputed_mase", "median"),
            median_rmse=("recomputed_rmse", "median"),
            best_model_count=("symbol", "size"),  # placeholder
        )
        .drop(columns=["best_model_count"])
    )
    summary["total_companies"] = total_companies
    summary["win_rate_pct"] = (
        summary["companies_beating_naive"] / summary["total_companies"] * 100.0
    ).round(1)
    return summary.sort_values("median_mase").reset_index(drop=True)


def best_model_counts(df: pd.DataFrame) -> pd.DataFrame:
    """How often each principal model has the lowest recomputed RMSE per company."""

    if df.empty:
        return pd.DataFrame()

    principal = df[df["is_principal"]].copy()
    if principal.empty:
        return pd.DataFrame()

    idx = principal.groupby("symbol")["recomputed_rmse"].idxmin()
    best_per_company = principal.loc[idx, ["symbol", "model", "model_label"]]
    counts = (
        best_per_company.groupby(["model", "model_label"])
        .size()
        .reset_index(name="companies_best")
        .sort_values("companies_best", ascending=False)
    )
    return counts.reset_index(drop=True)


def sector_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Median recomputed MASE per (sector, model), principal models only."""

    if df.empty:
        return pd.DataFrame()

    principal = df[df["is_principal"] & df["sector"].notna()].copy()
    if principal.empty:
        return pd.DataFrame()

    return (
        principal.groupby(["sector", "model_label"], as_index=False)["recomputed_mase"]
        .median()
        .rename(columns={"recomputed_mase": "median_mase"})
        .sort_values(["sector", "median_mase"])
        .reset_index(drop=True)
    )


def mismatch_table(df: pd.DataFrame) -> pd.DataFrame:
    """Only the rows where recomputed vs. reported metrics disagree beyond tolerance."""

    if df.empty:
        return pd.DataFrame()

    return df[~df["within_tolerance"]].sort_values("max_abs_diff", ascending=False).reset_index(
        drop=True
    )
