"""Tests for independent Naive baseline reconstruction and audit."""

from datetime import date
import pytest

from recheck.naive import (
    NaiveReconstructionError,
    audit_naive_predictions,
    compute_development_mase_denominator,
    compute_mase_denominator,
    reconstruct_naive_holdout,
)


def test_compute_development_mase_denominator():
    dev_closes = [10.0, 12.0, 11.0, 14.0]
    # diffs: |12 - 10| = 2, |11 - 12| = 1, |14 - 11| = 3 -> mean = 6 / 3 = 2.0
    denom = compute_development_mase_denominator(dev_closes)
    assert pytest.approx(denom, 1e-6) == 2.0

    # Ensure alias works
    assert compute_mase_denominator(dev_closes) == denom


def test_compute_mase_denominator_insufficient_or_invalid():
    with pytest.raises(NaiveReconstructionError, match="At least 2"):
        compute_development_mase_denominator([10.0])

    with pytest.raises(NaiveReconstructionError, match="finite"):
        compute_development_mase_denominator([10.0, float("nan")])

    with pytest.raises(NaiveReconstructionError, match="positive"):
        compute_development_mase_denominator([-5.0, -2.0])


def test_naive_reconstruction_from_known_close_sequence():
    raw_dates = ["2026-08-28", "2026-09-01", "2026-09-02", "2026-09-03"]
    raw_closes = [100.0, 105.0, 102.0, 108.0]
    target_dates = ["2026-09-01", "2026-09-02", "2026-09-03"]
    exported_naive = [100.0, 105.0, 102.0]

    audit = audit_naive_predictions(
        symbol="ALI",
        raw_dates=raw_dates,
        raw_closes=raw_closes,
        target_dates=target_dates,
        exported_naive_predictions=exported_naive,
    )

    assert audit.symbol == "ALI"
    assert audit.all_naive_predictions_match is True
    assert audit.mismatch_count == 0
    assert audit.total_sessions == 3
    assert audit.max_absolute_difference == 0.0

    # Inspect first step
    step0 = audit.steps[0]
    assert step0.target_date == "2026-09-01"
    assert step0.origin_date == "2026-08-28"
    assert step0.origin_close == 100.0
    assert step0.reconstructed_naive == 100.0
    assert step0.exported_naive == 100.0
    assert step0.matches is True


def test_incorrect_exported_naive_fails_audit():
    raw_dates = ["2026-08-28", "2026-09-01", "2026-09-02", "2026-09-03"]
    raw_closes = [100.0, 105.0, 102.0, 108.0]
    target_dates = ["2026-09-01", "2026-09-02", "2026-09-03"]
    # Corrupted exported Naive at step 1: reports 999.0 instead of 105.0
    exported_naive_corrupt = [100.0, 999.0, 102.0]

    audit = audit_naive_predictions(
        symbol="ALI",
        raw_dates=raw_dates,
        raw_closes=raw_closes,
        target_dates=target_dates,
        exported_naive_predictions=exported_naive_corrupt,
    )

    assert audit.all_naive_predictions_match is False
    assert audit.mismatch_count == 1
    assert audit.steps[1].matches is False
    assert pytest.approx(audit.steps[1].absolute_difference, 1e-6) == 894.0  # |105 - 999|


def test_missing_target_date_in_raw_series_raises():
    raw_dates = ["2026-09-01", "2026-09-02"]
    raw_closes = [100.0, 102.0]
    target_dates = ["2026-09-05"]  # missing from raw series

    with pytest.raises(NaiveReconstructionError, match="not found in raw chronological price series"):
        audit_naive_predictions(
            symbol="ALI",
            raw_dates=raw_dates,
            raw_closes=raw_closes,
            target_dates=target_dates,
            exported_naive_predictions=[100.0],
        )


def test_reconstruct_naive_holdout_array():
    last_dev = 100.0
    eval_actual = [105.0, 102.0, 108.0]
    result = reconstruct_naive_holdout(last_dev, eval_actual)

    assert result == [100.0, 105.0, 102.0]
