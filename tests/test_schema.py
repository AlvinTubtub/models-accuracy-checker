"""Unit tests for schema parsing and invariant validation."""

import pytest

from recheck.schema import (
    ExportValidationError,
    parse_company_export,
)


def _valid_payload():
    return {
        "symbol": "ALI",
        "name": "Ayala Land, Inc.",
        "sector": "Property",
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": 100,
            "evaluation_pairs": 3,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-03",
        },
        "mase_denominator": 0.5,
        "metrics": {
            "lag_reg": {"rmse": 0.2, "mae": 0.15, "mase": 0.3, "r2": 0.9, "observations": 3},
            "arima": {"rmse": 0.3, "mae": 0.2, "mase": 0.4, "r2": 0.85, "observations": 3},
            "lstm": {"rmse": 0.25, "mae": 0.18, "mase": 0.36, "r2": 0.88, "observations": 3},
            "naive": {"rmse": 0.4, "mae": 0.3, "mase": 0.6, "r2": 0.7, "observations": 3},
        },
        "principal_ranking": {"best_model": "lag_reg"},
        "backtest": {
            "target_dates": ["2026-09-01", "2026-09-02", "2026-09-03"],
            "actual_closes": [30.0, 31.0, 32.0],
            "predicted_closes_by_model": {
                "lag_reg": [30.1, 30.9, 32.2],
                "arima": [30.2, 30.8, 32.3],
                "lstm": [30.0, 31.1, 32.1],
                "naive": [30.0, 30.0, 31.0],
            },
        },
    }


def test_valid_payload_parses():
    payload = _valid_payload()
    export = parse_company_export(payload)
    assert export.symbol == "ALI"
    assert export.has_errors is False
    assert len(export.issues) == 0
    assert len(export.target_dates) == 3


def test_missing_required_root_key():
    payload = _valid_payload()
    del payload["split"]
    with pytest.raises(ExportValidationError, match="missing required key 'split'"):
        parse_company_export(payload)


def test_missing_required_model():
    payload = _valid_payload()
    del payload["metrics"]["lstm"]
    with pytest.raises(ExportValidationError, match="metrics missing model 'lstm'"):
        parse_company_export(payload)


def test_non_chronological_dates_flagged_as_error():
    payload = _valid_payload()
    # Reverse dates: 2026-09-03 before 2026-09-01
    payload["backtest"]["target_dates"] = ["2026-09-03", "2026-09-02", "2026-09-01"]
    export = parse_company_export(payload)
    assert export.has_errors is True
    assert any("strictly increasing" in issue.message for issue in export.issues)


def test_duplicate_dates_flagged_as_error():
    payload = _valid_payload()
    payload["backtest"]["target_dates"] = ["2026-09-01", "2026-09-01", "2026-09-03"]
    export = parse_company_export(payload)
    assert export.has_errors is True
    assert any("duplicate" in issue.message for issue in export.issues)


def test_length_mismatch_flagged_as_error():
    payload = _valid_payload()
    payload["backtest"]["actual_closes"] = [30.0, 31.0]  # length 2 instead of 3
    export = parse_company_export(payload)
    assert export.has_errors is True
    assert any("actual_closes length" in issue.message for issue in export.issues)
