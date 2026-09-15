"""Tests for data provenance, raw CSV auditing, and calendar verification."""

from pathlib import Path
import pandas as pd
import pytest

from recheck.provenance import (
    CalendarAuditReport,
    ProvenanceRecord,
    ProvenanceTier,
    audit_raw_csv,
    compute_file_sha256,
    compute_string_sha256,
)


def test_provenance_tier_enum():
    assert ProvenanceTier.DEMO.value == "DEMO"
    assert ProvenanceTier.FORMAL_FROZEN.value == "FORMAL_FROZEN"
    assert ProvenanceTier.PROSPECTIVE.value == "PROSPECTIVE"


def test_compute_file_sha256(tmp_path: Path):
    f = tmp_path / "sample.txt"
    f.write_text("ForecastPH test string", encoding="utf-8")
    h = compute_file_sha256(f)
    assert len(h) == 64
    assert h == compute_string_sha256("ForecastPH test string")


def test_audit_raw_csv(tmp_path: Path):
    csv_file = tmp_path / "ALI.csv"
    data = {
        "Date": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        "Close": [30.0, 30.5, 31.0, 31.2],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    report = audit_raw_csv(csv_file, symbol="ALI")
    assert report.symbol == "ALI"
    assert report.row_count == 4
    assert report.first_date == "2026-01-05"
    assert report.last_date == "2026-01-08"
    assert len(report.duplicate_dates) == 0
    assert len(report.anomalous_returns) == 0
    assert report.is_complete is True


def test_audit_raw_csv_detects_duplicates(tmp_path: Path):
    csv_file = tmp_path / "BPI.csv"
    data = {
        "Date": ["2026-01-05", "2026-01-05", "2026-01-06"],
        "Close": [100.0, 100.0, 102.0],
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)

    report = audit_raw_csv(csv_file, symbol="BPI")
    assert len(report.duplicate_dates) == 1
    assert report.is_complete is False
