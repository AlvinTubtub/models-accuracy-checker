"""Recompute metrics from scratch and diff against what the original reported."""

from __future__ import annotations

from dataclasses import dataclass

from recheck.metrics import Metrics, compute_metrics
from recheck.schema import REQUIRED_MODELS, CompanyEvaluationExport

DEFAULT_TOLERANCE = 1e-6


@dataclass(frozen=True)
class ModelComparison:
    symbol: str
    model: str
    recomputed: Metrics
    reported: dict[str, float]
    max_abs_diff: float
    within_tolerance: bool
    beats_naive_recomputed: bool
    beats_naive_reported: bool


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
