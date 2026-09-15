"""Comprehensive tests for recheck/model_audit.py Model Audit Verifier."""

from __future__ import annotations

from pathlib import Path
import pytest

from recheck.formal_bundle import load_formal_bundle
from recheck.model_audit import (
    audit_arima_candidate_space,
    audit_company_arima_evidence,
    audit_company_lir_evidence,
    audit_company_lstm_evidence,
    audit_formal_bundle,
)
from tests.test_formal_bundle import _build_valid_formal_bundle


# =========================================================================
# ARIMA Tests (17-22)
# =========================================================================

def test_17_arima_010n_candidate_pass():
    space = {
        "includes_no_drift_random_walk": True,
        "trends": {"0": ["n", "c"], "1": ["n", "t"], "2": ["n"]},
    }
    checks = audit_arima_candidate_space(space)
    no_drift = next(c for c in checks if c.check_id == "ARIMA_CANDIDATE_NO_DRIFT")
    assert no_drift.status == "PASS"


def test_18_missing_no_drift_candidate_fail():
    space = {
        "includes_no_drift_random_walk": False,
        "no_drift_candidate": None,
        "trends": {"0": ["n", "c"]},
    }
    checks = audit_arima_candidate_space(space)
    no_drift = next(c for c in checks if c.check_id == "ARIMA_CANDIDATE_NO_DRIFT")
    assert no_drift.status == "FAIL"


def test_19_invalid_trend_d_combination_fail():
    # d=1 with trend='c' or 'ct' is invalid in the strict protocol
    space = {
        "includes_no_drift_random_walk": True,
        "trends": {"0": ["n", "c"], "1": ["n", "c", "t"], "2": ["n"]},
    }
    checks = audit_arima_candidate_space(space)
    trend_check = next(c for c in checks if c.check_id == "ARIMA_TREND_RULES_COMPLIANT")
    assert trend_check.status == "FAIL"


def test_20_selected_candidate_outside_grid_fail():
    ev = {
        "selected_order": [3, 1, 3],
        "selected_trend": "t",
        "candidate_cv": [{"configuration": {"order": [1, 0, 0], "trend": "n"}}],
    }
    checks = audit_company_arima_evidence("ALI", ev)
    sel_check = next(c for c in checks if c.check_id == "ARIMA_SELECTED_IN_GRID")
    assert sel_check.status == "FAIL"


def test_21_missing_convergence_evidence_missing_evidence():
    ev = {
        "selected_order": [1, 0, 0],
        "selected_trend": "n",
        # convergence flags omitted
    }
    checks = audit_company_arima_evidence("ALI", ev)
    final_conv = next(c for c in checks if c.check_id == "ARIMA_FINAL_FIT_CONVERGED")
    assert final_conv.status == "MISSING_EVIDENCE"


def test_22_failed_convergence_evidence_fail():
    ev = {
        "selected_order": [1, 0, 0],
        "selected_trend": "n",
        "final_fit_converged": False,
    }
    checks = audit_company_arima_evidence("ALI", ev)
    final_conv = next(c for c in checks if c.check_id == "ARIMA_FINAL_FIT_CONVERGED")
    assert final_conv.status == "FAIL"


# =========================================================================
# LIR / LASSO Tests (23-27)
# =========================================================================

def test_23_lasso_grid_above_one_pass():
    ev = {
        "selected_alpha": 1.0,
        "tuning_metadata": {"grid": [0.01, 1.0, 10.0]},
    }
    checks = audit_company_lir_evidence("ALI", ev)
    grid_check = next(c for c in checks if c.check_id == "LIR_ALPHA_EXPANSION_ABOVE_ONE")
    assert grid_check.status == "PASS"


def test_24_no_alpha_above_one_fail():
    ev = {
        "selected_alpha": 0.5,
        "tuning_metadata": {"grid": [0.01, 0.1, 0.5, 1.0]},  # Max is 1.0, no expansion above 1.0
    }
    checks = audit_company_lir_evidence("ALI", ev)
    grid_check = next(c for c in checks if c.check_id == "LIR_ALPHA_EXPANSION_ABOVE_ONE")
    assert grid_check.status == "FAIL"


def test_25_selected_alpha_boundary_warning_present_pass():
    ev = {
        "selected_alpha": 0.01,  # Hits lower boundary
        "tuning_metadata": {
            "grid": [0.01, 0.1, 1.0, 10.0],
            "selected_at_boundary": True,
        },
    }
    checks = audit_company_lir_evidence("ALI", ev)
    b_check = next(c for c in checks if c.check_id == "LIR_BOUNDARY_WARNING_AUDIT")
    assert b_check.status == "PASS"


def test_26_boundary_winner_without_warning_fail():
    ev = {
        "selected_alpha": 0.01,  # Hits lower boundary
        "tuning_metadata": {
            "grid": [0.01, 0.1, 1.0, 10.0],
            "selected_at_boundary": False,  # Missing boundary warning flag!
        },
    }
    checks = audit_company_lir_evidence("ALI", ev)
    b_check = next(c for c in checks if c.check_id == "LIR_BOUNDARY_WARNING_AUDIT")
    assert b_check.status == "FAIL"


def test_27_missing_tuning_evidence_missing_evidence():
    ev = {}
    checks = audit_company_lir_evidence("ALI", ev)
    assert any(c.status == "MISSING_EVIDENCE" for c in checks)


# =========================================================================
# LSTM Tests (28-33)
# =========================================================================

def test_28_lstm_exactly_three_tuning_seeds_pass():
    ev = {
        "training_metadata": {
            "tuning_seeds": [42, 123, 2026],
            "common_cv_target_start": "2020-02-17",
            "common_cv_target_end": "2025-09-01",
            "selected_config": {"lookback": 10},
            "final_fit_seed": 42,
        }
    }
    checks = audit_company_lstm_evidence("ALI", ev)
    seed_check = next(c for c in checks if c.check_id == "LSTM_THREE_TUNING_SEEDS")
    assert seed_check.status == "PASS"


def test_29_lstm_missing_or_incorrect_seeds_fail_or_missing():
    # Only 2 seeds instead of 3
    ev = {
        "training_metadata": {
            "tuning_seeds": [42, 123],
        }
    }
    checks = audit_company_lstm_evidence("ALI", ev)
    seed_check = next(c for c in checks if c.check_id == "LSTM_THREE_TUNING_SEEDS")
    assert seed_check.status == "FAIL"

    # No seeds
    ev_empty = {"training_metadata": {}}
    checks_empty = audit_company_lstm_evidence("ALI", ev_empty)
    seed_check_empty = next(c for c in checks_empty if c.check_id == "LSTM_THREE_TUNING_SEEDS")
    assert seed_check_empty.status == "MISSING_EVIDENCE"


def test_30_lstm_common_date_window_evidence():
    ev = {
        "training_metadata": {
            "common_cv_target_start": "2020-02-17",
            "common_cv_target_end": "2025-09-01",
        }
    }
    checks = audit_company_lstm_evidence("ALI", ev)
    win_check = next(c for c in checks if c.check_id == "LSTM_COMMON_CV_WINDOW")
    assert win_check.status == "PASS"


def test_31_lstm_final_refit_seed_evidence():
    ev = {
        "training_metadata": {
            "final_fit_seed": 42,
        }
    }
    checks = audit_company_lstm_evidence("ALI", ev)
    seed_check = next(c for c in checks if c.check_id == "LSTM_FINAL_FIT_SEED")
    assert seed_check.status == "PASS"


# =========================================================================
# Readiness and Portability Tests (34-36)
# =========================================================================

def test_34_and_35_readiness_summary_and_high_evidence_visibility(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    readiness = audit_formal_bundle(bundle)

    # Valid bundle has zero critical failures
    assert readiness.ready_for_formal_run is True
    assert readiness.summary_verdict == "READY FOR FORMAL EXPERIMENT PREPARATION"

    # But high/optional missing items (e.g. roots) remain visible
    assert readiness.arima_diagnostics_status == "PARTIALLY_AVAILABLE_ROOTS_MISSING"
    assert readiness.high_missing_evidence_count > 0


def test_36_bundle_remains_portable_without_forecastph(tmp_path: Path):
    bundle_dir = _build_valid_formal_bundle(tmp_path / "bundle")
    bundle = load_formal_bundle(bundle_dir)
    # Auditing requires no ForecastPH imports
    readiness = audit_formal_bundle(bundle)
    assert len(readiness.checks) > 0
