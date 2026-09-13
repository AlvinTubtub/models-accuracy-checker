from __future__ import annotations

import json

import pytest

from recheck.loader import load_all_exports, load_company_export
from recheck.schema import ExportValidationError, parse_company_export


def test_parse_valid_minimal_payload(minimal_payload):
    export = parse_company_export(minimal_payload)
    assert export.symbol == "TEST"
    assert export.sector == "Financials"
    assert len(export.target_dates) == 3
    assert export.actual_closes == [100.0, 102.0, 101.0]
    assert not export.has_errors


def test_parse_rejects_missing_required_key(minimal_payload):
    del minimal_payload["metrics"]
    with pytest.raises(ExportValidationError):
        parse_company_export(minimal_payload)


def test_parse_rejects_missing_model_in_metrics(minimal_payload):
    del minimal_payload["metrics"]["arima"]
    with pytest.raises(ExportValidationError):
        parse_company_export(minimal_payload)


def test_parse_flags_non_chronological_dates(minimal_payload):
    minimal_payload["backtest"]["target_dates"] = ["2026-09-10", "2026-09-09", "2026-09-08"]
    export = parse_company_export(minimal_payload)
    assert export.has_errors
    assert any("strictly increasing" in issue.message for issue in export.issues)


def test_parse_flags_duplicate_dates(minimal_payload):
    minimal_payload["backtest"]["target_dates"] = ["2026-09-08", "2026-09-08", "2026-09-10"]
    export = parse_company_export(minimal_payload)
    assert export.has_errors
    assert any("duplicate" in issue.message.lower() for issue in export.issues)


def test_parse_flags_length_mismatch(minimal_payload):
    minimal_payload["backtest"]["actual_closes"] = [100.0, 102.0]  # one short
    export = parse_company_export(minimal_payload)
    assert export.has_errors


def test_parse_flags_actual_close_disagreement_across_models(minimal_payload):
    records = minimal_payload["backtest"]["records"]
    # Corrupt one arima record's actual_close so it disagrees with lag_reg's
    # for the same target_date.
    for record in records:
        if record["model"] == "arima" and record["target_date"] == "2026-09-08":
            record["actual_close"] = 999.0
    export = parse_company_export(minimal_payload)
    assert export.has_errors
    assert any("disagrees across models" in issue.message for issue in export.issues)


def test_loader_round_trip(tmp_path, minimal_payload):
    path = tmp_path / "TEST.json"
    with path.open("w") as handle:
        json.dump(minimal_payload, handle)

    export = load_company_export(path)
    assert export.symbol == "TEST"


def test_load_all_exports_skips_manifest_and_reports_bad_files(tmp_path, minimal_payload):
    good_path = tmp_path / "TEST.json"
    with good_path.open("w") as handle:
        json.dump(minimal_payload, handle)

    manifest_path = tmp_path / "manifest.json"
    with manifest_path.open("w") as handle:
        json.dump({"note": "irrelevant"}, handle)

    bad_path = tmp_path / "BAD.json"
    with bad_path.open("w") as handle:
        handle.write("{not valid json")

    result = load_all_exports(tmp_path)
    assert result.symbols == ["TEST"]
    assert "BAD" in result.load_errors
    assert result.manifest == {"note": "irrelevant"}


def test_load_all_exports_raises_for_missing_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_all_exports(tmp_path / "does-not-exist")
