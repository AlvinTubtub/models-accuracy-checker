"""Independent RMSE / MAE / MASE / R^2 implementation.

This module is written from scratch against standard textbook definitions.
It does not import, copy, or depend on anything from the original ForecastPH
repository. Its only job is to let this repo compute its own opinion of a
model's accuracy from raw actual/predicted arrays, so that opinion can be
compared against whatever the original pipeline reported, rather than trusted
blindly.

Definitions used:
    error_i        = predicted_i - actual_i
    RMSE           = sqrt( mean(error_i^2) )
    MAE            = mean(|error_i|)
    MASE           = MAE / mase_denominator
    R^2            = 1 - sum(error_i^2) / sum((actual_i - mean(actual))^2)

`mase_denominator` is a scale, computed once per company from the
*development* (training) partition's day-over-day Close changes:
    mase_denominator = mean(|Close[t] - Close[t-1]|)   over development days
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence


class MetricComputationError(ValueError):
    """Raised when inputs are unsuitable for an honest metric computation."""


@dataclass(frozen=True)
class Metrics:
    rmse: float
    mae: float
    mase: float
    r2: float
    observations: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "rmse": self.rmse,
            "mae": self.mae,
            "mase": self.mase,
            "r2": self.r2,
            "observations": self.observations,
        }


def _to_float_list(values: Sequence[float], *, name: str) -> list[float]:
    if len(values) < 1:
        raise MetricComputationError(f"{name} cannot be empty")
    result = [float(v) for v in values]
    if not all(math.isfinite(v) for v in result):
        raise MetricComputationError(f"{name} must contain only finite values")
    return result


def compute_metrics(
    actual_closes: Sequence[float],
    predicted_closes: Sequence[float],
    *,
    mase_denominator: float,
) -> Metrics:
    """Compute RMSE, MAE, MASE, and R^2 from raw actual/predicted Close arrays."""

    actual = _to_float_list(actual_closes, name="actual_closes")
    predicted = _to_float_list(predicted_closes, name="predicted_closes")
    if len(actual) != len(predicted):
        raise MetricComputationError(
            f"actual_closes ({len(actual)}) and predicted_closes ({len(predicted)}) "
            "must have equal length"
        )
    if not math.isfinite(mase_denominator) or mase_denominator <= 0.0:
        raise MetricComputationError(
            "mase_denominator must be finite and strictly positive"
        )

    n = len(actual)
    errors = [p - a for p, a in zip(predicted, actual)]
    squared_errors = [e * e for e in errors]
    absolute_errors = [abs(e) for e in errors]

    mean_squared_error = sum(squared_errors) / n
    rmse = math.sqrt(mean_squared_error)
    mae = sum(absolute_errors) / n
    mase = mae / mase_denominator

    mean_actual = sum(actual) / n
    total_sum_of_squares = sum((a - mean_actual) ** 2 for a in actual)
    residual_sum_of_squares = sum(squared_errors)
    if total_sum_of_squares == 0.0:
        r2 = 1.0 if residual_sum_of_squares == 0.0 else 0.0
    else:
        r2 = 1.0 - residual_sum_of_squares / total_sum_of_squares

    return Metrics(rmse=rmse, mae=mae, mase=mase, r2=r2, observations=n)


def mase_denominator_from_closes(development_closes: Sequence[float]) -> float:
    """Recompute the naive-scale MASE denominator from a raw Close series.

    mean(|Close[t] - Close[t-1]|) over the supplied (development-only)
    chronological Close values.
    """

    closes = _to_float_list(development_closes, name="development_closes")
    if len(closes) < 2:
        raise MetricComputationError("At least two development Close values are required")
    diffs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    denominator = sum(diffs) / len(diffs)
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise MetricComputationError("Computed MASE denominator must be finite and positive")
    return denominator


def beats_naive(mase: float) -> bool:
    """MASE below 1.0 means the model beats the naive Close(t)=Close(t+1) baseline."""

    return mase < 1.0
