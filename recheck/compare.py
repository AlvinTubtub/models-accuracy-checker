"""Recompute metrics from scratch and diff against what the original reported."""

from __future__ import annotations

from dataclasses import dataclass

from recheck.metrics import Metrics, compute_metrics
from recheck.oos import (
    CompanyOOSSummary,
    ComparisonStatus,
    ModelOOSComparison,
    evaluate_company_direct_oos,
)
from recheck.schema import REQUIRED_MODELS, CompanyEvaluationExport
from recheck.tolerances import (
    DEMO_RECOMPUTATION_TOLERANCE,
    RECOMPUTATION_TOLERANCE,
    TIE_TOLERANCE,
)

DEFAULT_TOLERANCE = RECOMPUTATION_TOLERANCE


@dataclass(frozen=True)
class ModelComparison:
    symbol: str
    model: str
    recomputed: Metrics
    reported: dict[str, float]
    max_abs_diff: float
    within_tolerance: bool
    beats_naive_recomputed: bool  # True if recomputed MASE < 1.0 (development-period reference)
    beats_naive_reported: bool  # True if reported MASE < 1.0 (development-period reference)


def compare_company(
    export: CompanyEvaluationExport,
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> list[ModelComparison]:
    """Recompute RMSE/MAE/MASE/R2 for every model in one company's export
    and compare against the `reported_metrics` shipped in that export.
    """

    comparisons: list[ModelComparison] = []
    for model in REQUIRED_MODELS:
        predicted = export.predicted_closes_by_model[model]
        recomputed = compute_metrics(
            export.actual_closes,
            predicted,
            mase_denominator=export.mase_denominator,
        )
        reported = export.reported_metrics[model]
        diffs = [
            abs(recomputed.rmse - float(reported["rmse"])),
            abs(recomputed.mae - float(reported["mae"])),
            abs(recomputed.mase - float(reported["mase"])),
            abs(recomputed.r2 - float(reported["r2"])),
        ]
        max_abs_diff = max(diffs)
        comparisons.append(
            ModelComparison(
                symbol=export.symbol,
                model=model,
                recomputed=recomputed,
                reported=reported,
                max_abs_diff=max_abs_diff,
                within_tolerance=max_abs_diff <= tolerance,
                beats_naive_recomputed=recomputed.mase < 1.0,
                beats_naive_reported=float(reported["mase"]) < 1.0,
            )
        )
    return comparisons


compare_company_evaluation = compare_company


def compare_all(
    exports: dict[str, CompanyEvaluationExport],
    *,
    tolerance: float = DEFAULT_TOLERANCE,
) -> dict[str, list[ModelComparison]]:
    return {
        symbol: compare_company(export, tolerance=tolerance)
        for symbol, export in exports.items()
    }


def mismatches_only(
    comparisons_by_symbol: dict[str, list[ModelComparison]],
) -> list[ModelComparison]:
    """Flatten to only the comparisons that fell outside tolerance."""

    return [
        comparison
        for comparisons in comparisons_by_symbol.values()
        for comparison in comparisons
        if not comparison.within_tolerance
    ]


@dataclass(frozen=True)
class DirectOOSComparison:
    """Direct out-of-sample comparison between a model and the Naive baseline on the aligned holdout."""

    symbol: str
    model: str
    model_rmse: float
    naive_rmse: float
    model_mae: float
    naive_mae: float
    model_mase: float
    mase_below_one: bool
    beats_naive_rmse: bool
    beats_naive_mae: bool
    rmse_skill_vs_naive: float
    mae_skill_vs_naive: float


def evaluate_direct_oos_comparisons(
    export: CompanyEvaluationExport,
) -> dict[str, DirectOOSComparison]:
    """Independently compute direct holdout comparisons of each principal model vs Naive.

    Strictly separates scale-normalized MASE from actual direct RMSE/MAE superiority
    on the aligned out-of-sample holdout.
    """
    actual = export.actual_closes
    naive_preds = export.predicted_closes_by_model["naive"]
    naive_metrics = compute_metrics(actual, naive_preds, mase_denominator=export.mase_denominator)

    comparisons: dict[str, DirectOOSComparison] = {}
    for model in ("lag_reg", "arima", "lstm"):
        if model not in export.predicted_closes_by_model:
            continue
        model_preds = export.predicted_closes_by_model[model]
        model_metrics = compute_metrics(actual, model_preds, mase_denominator=export.mase_denominator)

        comparisons[model] = DirectOOSComparison(
            symbol=export.symbol,
            model=model,
            model_rmse=model_metrics.rmse,
            naive_rmse=naive_metrics.rmse,
            model_mae=model_metrics.mae,
            naive_mae=naive_metrics.mae,
            model_mase=model_metrics.mase,
            mase_below_one=model_metrics.mase < 1.0,
            beats_naive_rmse=model_metrics.rmse < naive_metrics.rmse,
            beats_naive_mae=model_metrics.mae < naive_metrics.mae,
            rmse_skill_vs_naive=1.0 - (model_metrics.rmse / naive_metrics.rmse) if naive_metrics.rmse > 0 else 0.0,
            mae_skill_vs_naive=1.0 - (model_metrics.mae / naive_metrics.mae) if naive_metrics.mae > 0 else 0.0,
        )
    return comparisons
