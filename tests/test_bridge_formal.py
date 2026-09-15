"""Tests for bridge/export_formal_experiment.py helper functions."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from bridge.export_formal_experiment import (
    BRIDGE_SCHEMA_VERSION,
    BRIDGE_VERSION,
    MissingEvidenceRegistry,
    canonical_dump_json,
    clean_floats,
    compute_file_sha256,
    extract_environment_evidence,
    extract_git_evidence,
)


def test_sha256_generation(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("ForecastPH formal evidence bridge test payload\n", encoding="utf-8")
    sha = compute_file_sha256(test_file)
    assert isinstance(sha, str)
    assert len(sha) == 64
    # Deterministic re-check
    assert sha == compute_file_sha256(test_file)


def test_canonical_json_serialization(tmp_path: Path):
    target = tmp_path / "canonical.json"
    data = {
        "z_key": 1.23456789,
        "a_key": "first",
        "nested": {"m_key": [1, 2, 3], "b_key": True},
    }
    canonical_dump_json(data, target)
    raw = target.read_text(encoding="utf-8")
    # Verify sorted keys
    lines = [l.strip() for l in raw.splitlines() if l.strip()]
    assert lines[1].startswith('"a_key"')
    assert lines[-2].startswith('"z_key"')
    # Verify exact JSON roundtrip
    assert json.loads(raw) == data


def test_clean_floats_rejects_nan_and_inf():
    with pytest.raises(ValueError, match="Non-finite float"):
        clean_floats({"val": float("nan")})

    with pytest.raises(ValueError, match="Non-finite float"):
        clean_floats([1.0, float("inf")])

    valid = {"rmse": 0.4521, "list": [1.0, 2.5]}
    assert clean_floats(valid) == valid


def test_missing_evidence_registry():
    registry = MissingEvidenceRegistry()
    registry.record(
        symbol="ALI",
        component="arima",
        field="diagnostics.roots",
        requirement_level="HIGH",
        status="MISSING_EVIDENCE",
        reason="Roots not persisted",
    )
    registry.record(
        symbol="BPI",
        component="raw_data",
        field="raw_sha256",
        requirement_level="CRITICAL",
        status="MISSING_EVIDENCE",
        reason="Checksum mismatch",
    )
    registry.record(
        symbol="GLO",
        component="logs",
        field="console_log",
        requirement_level="OPTIONAL",
        status="MISSING_EVIDENCE",
        reason="Log not saved",
    )

    assert registry.critical_missing_count() == 1
    assert registry.high_missing_count() == 1
    assert len(registry.as_list()) == 3


def test_environment_extraction():
    py_info, plat_info, pkg_info, _, freeze = extract_environment_evidence()
    assert "version" in py_info
    assert "system" in plat_info
    assert "packages" in pkg_info
    assert pkg_info["count"] > 0
    assert isinstance(freeze, str)


def test_git_evidence_extraction(tmp_path: Path):
    # Running in current repo
    git_ev = extract_git_evidence(Path.cwd())
    assert git_ev["bridge_schema_version"] == BRIDGE_SCHEMA_VERSION
    assert "export_timestamp_utc" in git_ev
    assert git_ev["commit_sha"] is not None
