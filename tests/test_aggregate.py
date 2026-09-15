"""Unit tests for recheck/aggregate.py cross-company aggregation and table helpers."""

from __future__ import annotations

from datetime import date
import pandas as pd
import pytest

from recheck.aggregate import (
    best_model_counts,
    best_model_counts_mase,
    comparisons_to_dataframe,
    mismatch_table,
    sector_breakdown,
    win_rate_summary,
)
from recheck.compare import ModelComparison
from recheck.metrics import Metrics
from recheck.schema import CompanyEvaluationExport, ProvenanceTier


def _make_export_and_comparisons():
    actuals = [10.0, 11.0, 12.0]
    dates = [date(2020, 1, 2), date(2020, 1, 3), date(2020, 1, 4)]
    preds = {
        "lag_reg": [10.1, 11.1, 12.1],
        "arima": [10.2, 11.2, 12.2],
        "lstm": [10.3, 11.3, 12.3],
        "naive": [10.0, 11.0, 12.0],
    }
    from recheck.metrics import compute_metrics
    metrics = {}
    for m in ["lag_reg", "arima", "lstm", "naive"]:
        met = compute_metrics(actuals, preds[m], mase_denominator=0.5)
        metrics[m] = {
            "rmse": met.rmse,
            "mae": met.mae,
            "mase": met.mase,
            "r2": met.r2,
        }

    export_ali = CompanyEvaluationExport(
        symbol="ALI",
        name="Ayala Land",
        sector="Property",
        evaluation_proportion=0.2,
        development_pairs=100,
        evaluation_pairs=3,
        evaluation_start=dates[0],
        evaluation_end=dates[-1],
        mase_denominator=0.5,
        reported_metrics=metrics,
        reported_best_model="lag_reg",
        target_dates=dates,
        actual_closes=actuals,
        predicted_closes_by_model=preds,
        provenance_tier=ProvenanceTier.DEMO,
    )

    export_bdo = CompanyEvaluationExport(
        symbol="BDO",
        name="BDO Unibank",
        sector="Financials",
        evaluation_proportion=0.2,
        development_pairs=100,
        evaluation_pairs=3,
        evaluation_start=dates[0],
        evaluation_end=dates[-1],
        mase_denominator=0.5,
        reported_metrics=metrics,
        reported_best_model="arima",
        target_dates=dates,
        actual_closes=actuals,
        predicted_closes_by_model=preds,
        provenance_tier=ProvenanceTier.DEMO,
    )

    exports = {"ALI": export_ali, "BDO": export_bdo}
    from recheck.compare import compare_all
    comparisons_by_symbol = compare_all(exports)
    return exports, comparisons_by_symbol


def test_comparisons_to_dataframe():
    exports, comparisons = _make_export_and_comparisons()
    df = comparisons_to_dataframe(comparisons, exports)
    assert not df.empty
    assert len(df) == 8  # 2 companies * 4 models
    assert "symbol" in df.columns
    assert "recomputed_mase" in df.columns
    assert "within_tolerance" in df.columns


def test_win_rate_summary():
    exports, comparisons = _make_export_and_comparisons()
    df = comparisons_to_dataframe(comparisons, exports)
    summary = win_rate_summary(df)
    assert not summary.empty
    assert len(summary) == 3  # 3 principal models
    assert "win_rate_pct" in summary.columns
    assert "companies_beating_naive" in summary.columns


def test_best_model_counts():
    exports, comparisons = _make_export_and_comparisons()
    df = comparisons_to_dataframe(comparisons, exports)
    counts = best_model_counts(df)
    assert not counts.empty
    assert "companies_best" in counts.columns


def test_best_model_counts_mase():
    exports, comparisons = _make_export_and_comparisons()
    df = comparisons_to_dataframe(comparisons, exports)
    counts = best_model_counts_mase(df)
    assert not counts.empty
    assert "companies_best_mase" in counts.columns


def test_sector_breakdown():
    exports, comparisons = _make_export_and_comparisons()
    df = comparisons_to_dataframe(comparisons, exports)
    sec = sector_breakdown(df)
    assert not sec.empty
    assert "sector" in sec.columns
    assert "median_mase" in sec.columns


def test_mismatch_table_empty_when_all_within_tolerance():
    exports, comparisons = _make_export_and_comparisons()
    df = comparisons_to_dataframe(comparisons, exports)
    mismatches = mismatch_table(df)
    assert mismatches.empty


def test_mismatch_table_filters_discrepancies():
    exports, comparisons = _make_export_and_comparisons()
    # Artificially modify one comparison to violate tolerance
    comparisons["ALI"][0] = ModelComparison(
        symbol="ALI",
        model="lag_reg",
        recomputed=Metrics(rmse=0.5, mae=0.5, mase=1.5, r2=0.5, observations=3),
        reported={"rmse": 0.1, "mae": 0.1, "mase": 0.8, "r2": 0.9},
        max_abs_diff=0.7,
        within_tolerance=False,
        beats_naive_recomputed=False,
        beats_naive_reported=True,
    )
    df = comparisons_to_dataframe(comparisons, exports)
    mismatches = mismatch_table(df)
    assert len(mismatches) == 1
    assert mismatches.iloc[0]["symbol"] == "ALI"


def test_empty_dataframe_handling():
    empty = pd.DataFrame()
    assert win_rate_summary(empty).empty
    assert best_model_counts(empty).empty
    assert best_model_counts_mase(empty).empty
    assert sector_breakdown(empty).empty
    assert mismatch_table(empty).empty
