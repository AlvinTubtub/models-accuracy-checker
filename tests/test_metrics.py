from __future__ import annotations

import math

import pytest

from recheck.metrics import (
    MetricComputationError,
    beats_naive,
    compute_metrics,
    mase_denominator_from_closes,
)


def test_compute_metrics_hand_worked_example():
    actual = [10.0, 20.0, 30.0]
    predicted = [12.0, 18.0, 33.0]
    mase_denominator = 2.0

    # Worked by hand, independent of the implementation under test:
    # errors = [2, -2, 3]
    # squared errors = [4, 4, 9] -> mean = 17/3 -> rmse = sqrt(17/3)
    # abs errors = [2, 2, 3] -> mae = 7/3
    # mase = mae / 2.0 = 7/6
    # mean(actual) = 20; total_ss = 100 + 0 + 100 = 200; residual_ss = 17
    # r2 = 1 - 17/200 = 0.915
    expected_rmse = math.sqrt(17 / 3)
    expected_mae = 7 / 3
    expected_mase = (7 / 3) / 2.0
    expected_r2 = 1 - 17 / 200

    metrics = compute_metrics(actual, predicted, mase_denominator=mase_denominator)

    assert metrics.rmse == pytest.approx(expected_rmse)
    assert metrics.mae == pytest.approx(expected_mae)
    assert metrics.mase == pytest.approx(expected_mase)
    assert metrics.r2 == pytest.approx(expected_r2)
    assert metrics.observations == 3


def test_compute_metrics_perfect_prediction_gives_zero_error_and_r2_one():
    actual = [5.0, 6.0, 7.0, 8.0]
    metrics = compute_metrics(actual, actual, mase_denominator=1.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.mase == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)


def test_compute_metrics_rejects_mismatched_lengths():
    with pytest.raises(MetricComputationError):
        compute_metrics([1.0, 2.0], [1.0], mase_denominator=1.0)


def test_compute_metrics_rejects_empty_input():
    with pytest.raises(MetricComputationError):
        compute_metrics([], [], mase_denominator=1.0)


def test_compute_metrics_rejects_non_positive_denominator():
    with pytest.raises(MetricComputationError):
        compute_metrics([1.0, 2.0], [1.0, 2.0], mase_denominator=0.0)
    with pytest.raises(MetricComputationError):
        compute_metrics([1.0, 2.0], [1.0, 2.0], mase_denominator=-1.0)


def test_compute_metrics_rejects_non_finite_values():
    with pytest.raises(MetricComputationError):
        compute_metrics([1.0, float("nan")], [1.0, 2.0], mase_denominator=1.0)
    with pytest.raises(MetricComputationError):
        compute_metrics([1.0, 2.0], [1.0, float("inf")], mase_denominator=1.0)


def test_mase_denominator_from_closes_hand_worked_example():
    # diffs = |11-10|, |9-11|, |15-9| = 1, 2, 6 -> mean = 3.0
    closes = [10.0, 11.0, 9.0, 15.0]
    assert mase_denominator_from_closes(closes) == pytest.approx(3.0)


def test_mase_denominator_from_closes_requires_at_least_two_values():
    with pytest.raises(MetricComputationError):
        mase_denominator_from_closes([10.0])


def test_beats_naive_threshold():
    assert beats_naive(0.99) is True
    assert beats_naive(1.0) is False
    assert beats_naive(1.01) is False
