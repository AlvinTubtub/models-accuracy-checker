"""Integration test for Phase 1 independent Naive reconstruction connected to formal audit."""

import pytest

from recheck.naive import audit_naive_predictions
from recheck.oos import DirectOOSVerificationError, evaluate_company_direct_oos
from recheck.schema import ProvenanceTier, parse_company_export


def _create_formal_export(exported_naive: list[float]) -> dict:
    return {
        "symbol": "ALI",
        "provenance_tier": "FORMAL_FROZEN",
        "split": {
            "evaluation_proportion": 0.2,
            "development_pairs": 10,
            "evaluation_pairs": 3,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-03",
        },
        "mase_denominator": 0.5,
        "metrics": {
            "lag_reg": {"rmse": 0.5, "mae": 0.4, "mase": 0.8, "r2": 0.9, "observations": 3},
            "arima": {"rmse": 0.6, "mae": 0.5, "mase": 1.0, "r2": 0.85, "observations": 3},
            "lstm": {"rmse": 0.7, "mae": 0.6, "mase": 1.2, "r2": 0.80, "observations": 3},
            "naive": {"rmse": 0.8, "mae": 0.7, "mase": 1.4, "r2": 0.75, "observations": 3},
        },
        "backtest": {
            "target_dates": ["2026-09-01", "2026-09-02", "2026-09-03"],
            "actual_closes": [105.0, 102.0, 108.0],
            "predicted_closes_by_model": {
                "lag_reg": [104.0, 103.0, 107.0],
                "arima": [103.5, 103.0, 106.5],
                "lstm": [103.0, 103.5, 106.0],
                "naive": exported_naive,
            },
        },
    }


def test_formal_audit_path_with_valid_raw_data_passes():
    # Raw sequence:
    # 2026-08-31: 100.0 (origin for 2026-09-01)
    # 2026-09-01: 105.0 (origin for 2026-09-02)
    # 2026-09-02: 102.0 (origin for 2026-09-03)
    # 2026-09-03: 108.0
    raw_dates = ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03"]
    raw_closes = [100.0, 105.0, 102.0, 108.0]
    expected_naive = [100.0, 105.0, 102.0]

    export = parse_company_export(_create_formal_export(exported_naive=expected_naive))
    assert export.provenance_tier == ProvenanceTier.FORMAL_FROZEN

    # Flow: frozen raw Close data -> independent Naive reconstruction -> compare with exported Naive -> PASS
    summary = evaluate_company_direct_oos(
        export,
        raw_dates=raw_dates,
        raw_closes=raw_closes,
    )

    assert summary.naive_audit_status == "PASSED"
    assert summary.naive_source == "reconstructed"
    assert summary.best_principal_beats_naive is True


def test_formal_audit_path_fails_on_corrupted_exported_naive():
    raw_dates = ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03"]
    raw_closes = [100.0, 105.0, 102.0, 108.0]
    # Exported Naive has corrupted value 999.0
    corrupted_naive = [100.0, 999.0, 102.0]

    export = parse_company_export(_create_formal_export(exported_naive=corrupted_naive))

    # Flow: independent reconstruction detects mismatch -> formal verification FAILS
    with pytest.raises(DirectOOSVerificationError, match="failed independent reconstruction audit"):
        evaluate_company_direct_oos(
            export,
            raw_dates=raw_dates,
            raw_closes=raw_closes,
        )


def test_formal_audit_path_fails_when_raw_data_unavailable():
    export = parse_company_export(_create_formal_export(exported_naive=[100.0, 105.0, 102.0]))

    # FORMAL_FROZEN without raw data MUST NOT pass or fabricate verification
    with pytest.raises(DirectOOSVerificationError, match="RAW_DATA_UNAVAILABLE"):
        evaluate_company_direct_oos(export, raw_dates=None, raw_closes=None)


def test_demo_audit_path_remains_usable_without_formal_claim():
    payload = _create_formal_export(exported_naive=[100.0, 105.0, 102.0])
    payload["provenance_tier"] = "DEMO"
    export = parse_company_export(payload)

    # DEMO can evaluate without raw data, but clearly labeled NOT_RUN_DEMO
    summary = evaluate_company_direct_oos(export, raw_dates=None, raw_closes=None)
    assert summary.provenance_tier == ProvenanceTier.DEMO
    assert summary.naive_audit_status == "NOT_RUN_DEMO"
    assert summary.naive_source == "exported_demo"
