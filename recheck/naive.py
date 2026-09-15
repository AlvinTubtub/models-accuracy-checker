"""Independent Naive benchmark reconstruction and audit.

This repository does NOT train a Naive model.

Independently reconstructs:
    prediction for target Date[t+1] = actual Close at origin Date[t]
using the frozen chronological raw Close series and the formal target dates.

Never trusts `predicted_closes_by_model["naive"]` as the source of truth.
Directly audits exported Naive predictions against reconstructed ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import math
from typing import Sequence
import pandas as pd


class NaiveReconstructionError(ValueError):
    """Raised when data is insufficient or invalid for Naive reconstruction."""


@dataclass(frozen=True)
class NaiveStepEvidence:
    """Structured evidence for one target session's Naive benchmark comparison."""

    target_date: str
    origin_date: str
    origin_close: float
    reconstructed_naive: float
    exported_naive: float
    absolute_difference: float
    matches: bool


@dataclass(frozen=True)
class NaiveAuditSummary:
    """Full independent audit summary of exported Naive predictions vs ground truth."""

    symbol: str
    all_naive_predictions_match: bool
    mismatch_count: int
    total_sessions: int
    max_absolute_difference: float
    steps: list[NaiveStepEvidence]


def compute_development_mase_denominator(development_closes: Sequence[float]) -> float:
    """Compute mean absolute first differences strictly on DEVELOPMENT data only.

    Formula:
        mean(|Close[t] - Close[t-1]|) for all development pairs.

    Zero final-holdout observations are permitted in this computation.
    """
    if len(development_closes) < 2:
        raise NaiveReconstructionError(
            "At least 2 development Close values are required to compute MASE denominator"
        )

    closes = [float(c) for c in development_closes]
    if not all(math.isfinite(c) and c > 0.0 for c in closes):
        raise NaiveReconstructionError(
            "Development closes must contain only positive, finite numbers"
        )

    diffs = [abs(closes[i] - closes[i - 1]) for i in range(1, len(closes))]
    denominator = sum(diffs) / len(diffs)

    if not math.isfinite(denominator) or denominator <= 0.0:
        raise NaiveReconstructionError(
            "Computed development MASE denominator must be finite and positive"
        )
    return denominator


# Alias for backward compatibility
compute_mase_denominator = compute_development_mase_denominator


def audit_naive_predictions(
    symbol: str,
    raw_dates: Sequence[date | str],
    raw_closes: Sequence[float],
    target_dates: Sequence[date | str],
    exported_naive_predictions: Sequence[float],
    *,
    tolerance: float = 1e-6,
) -> NaiveAuditSummary:
    """Independently reconstruct Naive shift-1 predictions and audit exported Naive series.

    Parameters
    ----------
    symbol : str
        Ticker symbol.
    raw_dates : Sequence[date | str]
        Complete chronological dates of the raw price series (development + evaluation).
    raw_closes : Sequence[float]
        Complete chronological closing prices corresponding to `raw_dates`.
    target_dates : Sequence[date | str]
        Evaluation target dates from the formal holdout.
    exported_naive_predictions : Sequence[float]
        The exported Naive predictions reported by ForecastPH.
    tolerance : float
        Maximum allowed absolute difference for matching (default 1e-6).
    """
    if len(raw_dates) != len(raw_closes):
        raise NaiveReconstructionError("raw_dates and raw_closes must have identical length")
    if len(target_dates) != len(exported_naive_predictions):
        raise NaiveReconstructionError(
            "target_dates and exported_naive_predictions must have identical length"
        )
    if len(target_dates) == 0:
        raise NaiveReconstructionError("target_dates cannot be empty")

    # Standardize dates to YYYY-MM-DD strings
    str_raw_dates = [
        d.isoformat() if isinstance(d, (date, datetime)) else str(d) for d in raw_dates
    ]
    str_target_dates = [
        d.isoformat() if isinstance(d, (date, datetime)) else str(d) for d in target_dates
    ]

    # Map raw dates to their index in the chronological series
    date_to_idx = {d: idx for idx, d in enumerate(str_raw_dates)}

    steps: list[NaiveStepEvidence] = []
    mismatch_count = 0
    max_abs_diff = 0.0

    for i, t_date in enumerate(str_target_dates):
        if t_date not in date_to_idx:
            raise NaiveReconstructionError(
                f"{symbol}: Target date {t_date} not found in raw chronological price series"
            )

        target_idx = date_to_idx[t_date]
        if target_idx == 0:
            raise NaiveReconstructionError(
                f"{symbol}: Target date {t_date} is the first record in raw series; "
                "no preceding origin Close exists to reconstruct Naive"
            )

        origin_idx = target_idx - 1
        origin_date = str_raw_dates[origin_idx]
        origin_close = float(raw_closes[origin_idx])
        reconstructed_val = origin_close

        exported_val = float(exported_naive_predictions[i])
        abs_diff = abs(reconstructed_val - exported_val)
        matches = abs_diff <= tolerance

        if abs_diff > max_abs_diff:
            max_abs_diff = abs_diff

        if not matches:
            mismatch_count += 1

        steps.append(
            NaiveStepEvidence(
                target_date=t_date,
                origin_date=origin_date,
                origin_close=origin_close,
                reconstructed_naive=reconstructed_val,
                exported_naive=exported_val,
                absolute_difference=abs_diff,
                matches=matches,
            )
        )

    all_match = mismatch_count == 0
    return NaiveAuditSummary(
        symbol=symbol,
        all_naive_predictions_match=all_match,
        mismatch_count=mismatch_count,
        total_sessions=len(target_dates),
        max_absolute_difference=max_abs_diff,
        steps=steps,
    )


def reconstruct_naive_holdout(
    last_development_close: float,
    evaluation_actual_closes: Sequence[float],
    development_closes: Sequence[float] | None = None,
) -> list[float]:
    """Reconstruct Naive predictions array directly when given the last development close."""
    last_dev = float(last_development_close)
    if not math.isfinite(last_dev) or last_dev <= 0.0:
        raise NaiveReconstructionError("last_development_close must be a positive finite number")

    eval_actual = [float(c) for c in evaluation_actual_closes]
    if len(eval_actual) == 0:
        raise NaiveReconstructionError("evaluation_actual_closes cannot be empty")
    if not all(math.isfinite(c) and c > 0.0 for c in eval_actual):
        raise NaiveReconstructionError("evaluation_actual_closes must be positive finite numbers")

    n = len(eval_actual)
    predicted: list[float] = [0.0] * n
    predicted[0] = last_dev
    for t in range(1, n):
        predicted[t] = eval_actual[t - 1]

    return predicted
