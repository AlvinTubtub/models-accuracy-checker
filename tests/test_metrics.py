"""Unit tests for independent metrics computation."""

import math
import pytest

from recheck.metrics import (
    MetricComputationError,
    beats_naive,
    compute_metrics,
    mase_denominator_from_closes,
)


def test_perfect_predictions():
    actual = [10.0, 20.0, 30.0, 40.0]
    predicted = [10.0, 20.0, 30.0, 40.0]
    mase_den = 5.0

    m = compute_metrics(actual, predicted, mase_denominator=mase_den)
    assert m.rmse == pytest.approx(0.0)
    assert m.mae == pytest.approx(0.0)
    assert m.mase == pytest.approx(0.0)
    assert m.r2 == pytest.approx(1.0)
    assert m.observations == 4
    assert beats_naive(m.mase) is True


def test_known_analytical_values():
    # Errors: [1, -2, 3, -4]
    # Squared errors: [1, 4, 9, 16] -> mean = 30 / 4 = 7.5 -> sqrt = ~2.7386
    # Absolute errors: [1, 2, 3, 4] -> mean = 10 / 4 = 2.5
    # Actual mean = 25.0, sum of squares = (10-25)^2 + (20-25)^2 + (30-25)^2 + (40-25)^2
    #                                    = 225 + 25 + 25 + 225 = 500
    # SS_res = 30 -> R^2 = 1 - 30/500 = 0.94
    actual = [10.0, 20.0, 30.0, 40.0]
    predicted = [11.0, 18.0, 33.0, 36.0]
    mase_den = 2.5

    m = compute_metrics(actual, predicted, mase_denominator=mase_den)
    assert m.rmse == pytest.approx(math.sqrt(7.5))
    assert m.mae == pytest.approx(2.5)
    assert m.mase == pytest.approx(1.0)  # exactly 1.0
    assert m.r2 == pytest.approx(0.94)
    assert beats_naive(m.mase) is False  # MASE == 1.0 is not strictly below scaling reference (< 1.0)


def test_length_mismatch_raises():
    actual = [10.0, 20.0]
    predicted = [10.0]
    with pytest.raises(MetricComputationError, match="must have equal length"):
        compute_metrics(actual, predicted, mase_denominator=1.0)


def test_invalid_mase_denominator():
    actual = [10.0, 20.0]
    predicted = [10.0, 20.0]
    with pytest.raises(MetricComputationError, match="mase_denominator must be finite and strictly positive"):
        compute_metrics(actual, predicted, mase_denominator=0.0)

    with pytest.raises(MetricComputationError, match="mase_denominator must be finite and strictly positive"):
        compute_metrics(actual, predicted, mase_denominator=-2.0)


def test_non_finite_values_raise():
    actual = [10.0, float("nan")]
    predicted = [10.0, 20.0]
    with pytest.raises(MetricComputationError, match="must contain only finite values"):
        compute_metrics(actual, predicted, mase_denominator=1.0)


def test_mase_denominator_from_closes():
    closes = [10.0, 12.0, 11.0, 14.0]
    # Diffs: |12 - 10| = 2, |11 - 12| = 1, |14 - 11| = 3
    # Mean diff = (2 + 1 + 3) / 3 = 2.0
    denom = mase_denominator_from_closes(closes)
    assert denom == pytest.approx(2.0)

    with pytest.raises(MetricComputationError, match="At least two"):
        mase_denominator_from_closes([10.0])
