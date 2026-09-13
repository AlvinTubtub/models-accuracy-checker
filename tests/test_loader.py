"""Unit tests for file loading and batch ingestion."""

import json
from pathlib import Path
import pytest

from recheck.loader import load_all_exports, load_company_export
from recheck.schema import ExportValidationError


def test_load_company_export(tmp_path: Path):
    sample_file = tmp_path / "SMPH.json"
    content = {
        "symbol": "SMPH",
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": 50,
            "evaluation_pairs": 2,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-02",
        },
        "mase_denominator": 1.0,
        "metrics": {
            "lag_reg": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
            "arima": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
            "lstm": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
            "naive": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
        },
        "backtest": {
            "target_dates": ["2026-09-01", "2026-09-02"],
            "actual_closes": [30.0, 31.0],
            "predicted_closes_by_model": {
                "lag_reg": [30.0, 31.0],
                "arima": [30.0, 31.0],
                "lstm": [30.0, 31.0],
                "naive": [30.0, 30.0],
            },
        },
    }
    with sample_file.open("w") as f:
        json.dump(content, f)

    export = load_company_export(sample_file)
    assert export.symbol == "SMPH"
    assert len(export.target_dates) == 2


def test_load_all_exports_skips_manifest_and_records_corrupted(tmp_path: Path):
    # Valid file
    valid_file = tmp_path / "JFC.json"
    content = {
        "symbol": "JFC",
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": 50,
            "evaluation_pairs": 2,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-02",
        },
        "mase_denominator": 1.0,
        "metrics": {
            "lag_reg": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
            "arima": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
            "lstm": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
            "naive": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.9, "observations": 2},
        },
        "backtest": {
            "target_dates": ["2026-09-01", "2026-09-02"],
            "actual_closes": [240.0, 242.0],
            "predicted_closes_by_model": {
                "lag_reg": [240.0, 242.0],
                "arima": [240.0, 242.0],
                "lstm": [240.0, 242.0],
                "naive": [240.0, 240.0],
            },
        },
    }
    with valid_file.open("w") as f:
        json.dump(content, f)

    # Manifest file (should be skipped)
    manifest_file = tmp_path / "manifest.json"
    with manifest_file.open("w") as f:
        json.dump({"test": "manifest"}, f)

    # Corrupted file
    corrupt_file = tmp_path / "CORRUPT.json"
    with corrupt_file.open("w") as f:
        f.write("{invalid_json:")

    result = load_all_exports(tmp_path)
    assert "JFC" in result.exports
    assert "CORRUPT" in result.load_errors
    assert "manifest" not in result.exports
    assert result.manifest == {"test": "manifest"}


def test_missing_data_directory():
    with pytest.raises(FileNotFoundError):
        load_all_exports(Path("/nonexistent/directory/path"))
