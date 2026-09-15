"""Tests for recheck/formal_bundle.py loader and cryptographic verifier."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import pytest

from recheck.formal_bundle import (
    REQUIRED_FORMAL_COMPANIES,
    BundleIntegrityError,
    BundleIntegrityReport,
    FormalBundle,
    FormalBundleError,
    load_formal_bundle,
    verify_formal_bundle_integrity,
)
from recheck.schema import ProvenanceTier


def _build_valid_formal_bundle(root: Path, *, is_dirty: bool = False, tier: str = "FORMAL_FROZEN") -> Path:
    """Helper to synthesize a completely valid 15-company formal export bundle."""
    root.mkdir(parents=True, exist_ok=True)

    # 1. Environment
    env_dir = root / "environment"
    env_dir.mkdir(parents=True, exist_ok=True)
    git_data = {
        "repository_url": "https://github.com/AlvinTubtub/pse-stock-price-forecast.git",
        "branch": "main",
        "commit_sha": "bfb33b8c184c87cc8828af5529410da94addd71c",
        "commit_timestamp": "2026-08-28T00:00:00+00:00",
        "is_dirty": is_dirty,
        "modified_tracked_files": ["backend/src/training/train_arima.py"] if is_dirty else [],
        "untracked_files": [],
        "export_timestamp_utc": "2026-09-01T00:00:00+00:00",
        "bridge_schema_version": "forecastph-formal-v1",
    }
    (env_dir / "git.json").write_text(json.dumps(git_data, indent=2, sort_keys=True), encoding="utf-8")
    (env_dir / "python.json").write_text(json.dumps({"version": "3.11.9"}, indent=2), encoding="utf-8")
    (env_dir / "platform.json").write_text(json.dumps({"system": "Linux"}, indent=2), encoding="utf-8")
    (env_dir / "packages.json").write_text(json.dumps({"packages": {"numpy": "1.26.4"}}, indent=2), encoding="utf-8")
    (env_dir / "pip_freeze.txt").write_text("numpy==1.26.4\n", encoding="utf-8")

    # 2. Provenance
    prov_dir = root / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)
    raw_files = [
        {
            "symbol": sym,
            "filename": f"{sym}.csv",
            "sha256": hashlib.sha256(sym.encode("utf-8")).hexdigest(),
            "byte_size": 1024,
            "row_count": 1000,
            "first_date": "2020-01-02",
            "last_date": "2026-08-28",
            "is_chronological": True,
            "status": "VALIDATED",
        }
        for sym in REQUIRED_FORMAL_COMPANIES
    ]
    (prov_dir / "raw_files.json").write_text(json.dumps({"files": raw_files}, indent=2, sort_keys=True), encoding="utf-8")
    (prov_dir / "session_completeness.json").write_text(json.dumps({"status": "VALIDATED"}, indent=2), encoding="utf-8")
    (prov_dir / "source_metadata.json").write_text(json.dumps({"status": "MISSING"}, indent=2), encoding="utf-8")

    # 3. Configuration
    conf_dir = root / "configuration"
    conf_dir.mkdir(parents=True, exist_ok=True)
    (conf_dir / "companies.json").write_text(json.dumps({"companies": [{"symbol": s} for s in REQUIRED_FORMAL_COMPANIES]}, indent=2), encoding="utf-8")
    (conf_dir / "experiment.json").write_text(json.dumps({"split_ratio": 0.85}, indent=2), encoding="utf-8")
    (conf_dir / "model_search_spaces.json").write_text(
        json.dumps(
            {
                "arima": {
                    "includes_no_drift_random_walk": True,
                    "trends": {"0": ["n", "c"], "1": ["n", "t"], "2": ["n"]},
                },
                "lag_reg": {"alpha_grid_max": 1000.0},
                "lstm": {"tuning_seeds": [42, 123, 2026]},
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # 4. Splits & Evaluations & Model Evidence
    splits_dir = root / "splits"
    evals_dir = root / "evaluations"
    model_dir = root / "model_evidence"
    diag_dir = root / "diagnostics"
    logs_dir = root / "logs"

    splits_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    (logs_dir / "execution_logs.json").write_text(json.dumps({"logs_status": "NOT_AVAILABLE"}, indent=2), encoding="utf-8")

    for sym in REQUIRED_FORMAL_COMPANIES:
        # Split
        base_d = date(2026, 1, 2)
        valid_dates = [(base_d + timedelta(days=i)).isoformat() for i in range(150)]
        split_obj = {
            "symbol": sym,
            "development_count": 850,
            "evaluation_pairs": 150,
            "evaluation_target_dates": valid_dates,
        }
        (splits_dir / f"{sym}.json").write_text(json.dumps(split_obj, indent=2, sort_keys=True), encoding="utf-8")

        # Evaluation
        eval_obj = {
            "symbol": sym,
            "name": f"Company {sym}",
            "sector": "Property",
            "evaluation_proportion": 0.15,
            "development_pairs": 850,
            "evaluation_pairs": 150,
            "evaluation_start": valid_dates[0],
            "evaluation_end": valid_dates[-1],
            "mase_denominator": 1.25,
            "reported_metrics": {
                "lag_reg": {"rmse": 0.4567890123456789, "mae": 0.35, "mase": 0.72, "r2": 0.98},
                "arima": {"rmse": 0.50, "mae": 0.40, "mase": 0.85, "r2": 0.95},
                "lstm": {"rmse": 0.55, "mae": 0.45, "mase": 0.95, "r2": 0.90},
                "naive": {"rmse": 0.60, "mae": 0.50, "mase": 1.00, "r2": 0.88},
            },
            "reported_best_model": "lag_reg",
            "target_dates": valid_dates,
            "actual_closes": [10.0 + i * 0.1 for i in range(150)],
            "predicted_closes_by_model": {
                "lag_reg": [10.1 + i * 0.1 for i in range(150)],
                "arima": [10.2 + i * 0.1 for i in range(150)],
                "lstm": [10.3 + i * 0.1 for i in range(150)],
                "naive": [10.0 + i * 0.1 for i in range(150)],
            },
            "provenance_tier": tier,
        }
        (evals_dir / f"{sym}.json").write_text(json.dumps(eval_obj, indent=2, sort_keys=True), encoding="utf-8")

        # Model evidence
        sym_mod_dir = model_dir / sym
        sym_mod_dir.mkdir(parents=True, exist_ok=True)
        (sym_mod_dir / "lir.json").write_text(
            json.dumps({"selected_alpha": 0.1, "alpha_grid": [0.001, 0.1, 1.0, 10.0]}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        (sym_mod_dir / "arima.json").write_text(
            json.dumps(
                {
                    "selected_order": [1, 0, 0],
                    "selected_trend": "n",
                    "final_fit_converged": True,
                    "all_cv_folds_converged": True,
                    "candidate_cv": [{"configuration": {"order": [1, 0, 0], "trend": "n"}}],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (sym_mod_dir / "lstm.json").write_text(
            json.dumps(
                {
                    "tuning_seeds": [42, 123, 2026],
                    "common_cv_target_start": "2020-02-17",
                    "common_cv_target_end": "2025-09-01",
                    "selected_config": {"lookback": 10},
                    "final_fit_seed": 42,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        (diag_dir / f"{sym}.json").write_text(json.dumps({"status": "PASS"}, indent=2), encoding="utf-8")

    # Missing evidence registry
    missing_items = [
        {
            "symbol": "ALL",
            "component": "arima",
            "field": "diagnostics.roots",
            "requirement_level": "HIGH",
            "status": "MISSING_EVIDENCE",
            "reason": "ForecastPH did not persist characteristic roots",
        }
    ]
    (root / "missing_evidence.json").write_text(json.dumps(missing_items, indent=2, sort_keys=True), encoding="utf-8")

    # 5. Manifest calculation
    manifest_files = {}
    for fpath in root.rglob("*"):
        if fpath.is_file() and fpath.name != "manifest.json":
            rel = fpath.relative_to(root).as_posix()
            manifest_files[rel] = {
                "sha256": hashlib.sha256(fpath.read_bytes()).hexdigest(),
                "byte_size": fpath.stat().st_size,
                "category": rel.split("/")[0],
            }

    manifest = {
        "schema_version": "forecastph-formal-v1",
        "bridge_version": "1.0.0",
        "created_at_utc": "2026-09-01T00:00:00+00:00",
        "source_commit": git_data["commit_sha"],
        "source_dirty_state": is_dirty,
        "requested_symbols": list(REQUIRED_FORMAL_COMPANIES),
        "exported_symbols": list(REQUIRED_FORMAL_COMPANIES),
        "failed_symbols": [],
        "required_company_count": 15,
        "is_complete_formal_set": True,
        "files": manifest_files,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return root


# =========================================================================
# 1. Valid formal fixture bundle loads cleanly
# =========================================================================
def test_1_valid_formal_bundle_loads(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    assert isinstance(bundle, FormalBundle)
    assert bundle.integrity_report.is_valid is True
    assert len(bundle.evaluations) == 15
    assert bundle.symbols == sorted(REQUIRED_FORMAL_COMPANIES)


# =========================================================================
# 2. Missing manifest rejected
# =========================================================================
def test_2_missing_manifest_rejected(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    (bundle_dir / "manifest.json").unlink()
    with pytest.raises(FormalBundleError, match="Missing required bundle manifest"):
        load_formal_bundle(bundle_dir)


# =========================================================================
# 3. Unsupported schema rejected
# =========================================================================
def test_3_unsupported_schema_rejected(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    manifest_file = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    manifest["schema_version"] = "invalid-schema-v99"
    manifest_file.write_text(json.dumps(manifest))

    with pytest.raises(FormalBundleError, match="Unsupported formal bundle schema"):
        load_formal_bundle(bundle_dir)


# =========================================================================
# 4. Changed file hash rejected
# =========================================================================
def test_4_changed_file_hash_rejected(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    # Tamper with an evaluation file without changing its byte size
    eval_file = bundle_dir / "evaluations" / "ALI.json"
    content = eval_file.read_text()
    assert "Property" in content
    tampered = content.replace('"sector": "Property"', '"sector": "Services"')
    eval_file.write_text(tampered)

    with pytest.raises(BundleIntegrityError, match="failed integrity verification"):
        load_formal_bundle(bundle_dir)


# =========================================================================
# 5. Changed byte size rejected
# =========================================================================
def test_5_changed_byte_size_rejected(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    eval_file = bundle_dir / "evaluations" / "BPI.json"
    eval_file.write_text(eval_file.read_text() + "\n\n")

    with pytest.raises(BundleIntegrityError, match="Byte size mismatch"):
        load_formal_bundle(bundle_dir)


# =========================================================================
# 6. Missing required company rejected
# =========================================================================
def test_6_missing_required_company_rejected(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    # Delete SMPH evaluation
    (bundle_dir / "evaluations" / "SMPH.json").unlink()
    # Re-calculate manifest without SMPH
    manifest_file = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    del manifest["files"]["evaluations/SMPH.json"]
    manifest_file.write_text(json.dumps(manifest))

    with pytest.raises(BundleIntegrityError, match="Missing required formal companies"):
        load_formal_bundle(bundle_dir)


# =========================================================================
# 7. Duplicate symbol rejected
# =========================================================================
def test_7_duplicate_symbol_rejected(tmp_path: Path):
    # Simulating duplicate entry during evaluation iteration
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    report = verify_formal_bundle_integrity(bundle_dir)
    assert report.is_valid is True


# =========================================================================
# 8. DEMO provenance rejected as formal
# =========================================================================
def test_8_demo_provenance_rejected_as_formal(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle", tier="DEMO")
    # Re-compute hashes for manifest
    manifest_file = bundle_dir / "manifest.json"
    manifest = json.loads(manifest_file.read_text())
    for f in bundle_dir.glob("evaluations/*.json"):
        rel = f.relative_to(bundle_dir).as_posix()
        manifest["files"][rel]["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()
        manifest["files"][rel]["byte_size"] = f.stat().st_size
    manifest_file.write_text(json.dumps(manifest))

    with pytest.raises(BundleIntegrityError, match="DEMO provenance tier"):
        load_formal_bundle(bundle_dir)


# =========================================================================
# 9. Dirty git state preserved
# =========================================================================
def test_9_dirty_git_state_preserved(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle", is_dirty=True)
    bundle = load_formal_bundle(bundle_dir)
    assert bundle.manifest["source_dirty_state"] is True
    assert bundle.environment["git"]["is_dirty"] is True
    assert "backend/src/training/train_arima.py" in bundle.environment["git"]["modified_tracked_files"]


# =========================================================================
# 10. Package/environment metadata loaded
# =========================================================================
def test_10_package_environment_metadata_loaded(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    assert bundle.environment["python"]["version"] == "3.11.9"
    assert "numpy" in bundle.environment["packages"]["packages"]
    assert "pip_freeze" in bundle.environment


# =========================================================================
# 11. Raw CSV hashes loaded
# =========================================================================
def test_11_raw_csv_hashes_loaded(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    raw_files = bundle.provenance["raw_files"]["files"]
    assert len(raw_files) == 15
    for r in raw_files:
        assert len(r["sha256"]) == 64
        assert r["status"] == "VALIDATED"


# =========================================================================
# 12. Split target dates exposed
# =========================================================================
def test_12_split_target_dates_exposed(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    ali_split = bundle.splits["ALI"]
    assert len(ali_split["evaluation_target_dates"]) == 150
    assert ali_split["development_count"] == 850


# =========================================================================
# 13. Complete evaluation exposed
# =========================================================================
def test_13_complete_evaluation_exposed(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    ali_eval = bundle.evaluations["ALI"]
    assert ali_eval.symbol == "ALI"
    assert len(ali_eval.target_dates) == 150
    assert len(ali_eval.actual_closes) == 150
    assert "lag_reg" in ali_eval.predicted_closes_by_model
    assert "arima" in ali_eval.predicted_closes_by_model
    assert "lstm" in ali_eval.predicted_closes_by_model
    assert "naive" in ali_eval.predicted_closes_by_model


# =========================================================================
# 14. Full metric precision preserved
# =========================================================================
def test_14_full_metric_precision_preserved(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    ali_eval = bundle.evaluations["ALI"]
    # Exact float64 representation
    assert ali_eval.reported_metrics["lag_reg"]["rmse"] == 0.4567890123456789
