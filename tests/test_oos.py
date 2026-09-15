"""Comprehensive tests for Direct Out-of-Sample (OOS) model-vs-Naive comparison.

Covers all 22 Phase 2 requirements:
1. principal model RMSE beats Naive
2. principal model RMSE loses to Naive
3. exact RMSE tie
4. principal model MAE beats Naive
5. principal model MAE loses to Naive
6. exact MAE tie
7. positive RMSE skill
8. negative RMSE skill
9. positive MAE skill
10. negative MAE skill
11. zero Naive RMSE handled safely
12. zero Naive MAE handled safely
13. best principal differs from best evaluated method
14. Naive is best evaluated method
15. all principals worse than Naive
16. principal model has MASE < 1 but still loses direct holdout comparison
17. negative R² remains negative
18. R² is not used for ranking
19. exact same target dates required
20. FORMAL_FROZEN refuses direct verification without independently reconstructed Naive evidence
21. DEMO remains usable but clearly non-formal
22. deterministic tie behavior
"""

import math
import pytest

from recheck.oos import (
    ComparisonStatus,
    DirectOOSVerificationError,
    compute_skill,
    determine_comparison_status,
    evaluate_company_direct_oos,
    oos_comparisons_to_dataframe,
)
from recheck.schema import (
    CompanyEvaluationExport,
    ExportValidationError,
    ProvenanceTier,
    parse_company_export,
)


def _build_test_export(
    symbol: str = "ALI",
    actual_closes: list[float] | None = None,
    lag_reg: list[float] | None = None,
    arima: list[float] | None = None,
    lstm: list[float] | None = None,
    naive: list[float] | None = None,
    mase_denominator: float = 1.0,
    provenance_tier: str = "DEMO",
) -> CompanyEvaluationExport:
    n = 3
    if actual_closes is None:
        actual_closes = [100.0, 102.0, 105.0]
    n = len(actual_closes)
    dates = [f"2026-09-{i+1:02d}" for i in range(n)]

    def_preds = [c + 0.1 for c in actual_closes]
    def_naive = [actual_closes[0]] + actual_closes[:-1]

    payload = {
        "symbol": symbol,
        "sector": "Property",
        "provenance_tier": provenance_tier,
        "split": {
            "evaluation_proportion": 0.2,
            "development_pairs": 50,
            "evaluation_pairs": n,
            "evaluation_start": dates[0],
            "evaluation_end": dates[-1],
        },
        "mase_denominator": mase_denominator,
        "metrics": {
            "lag_reg": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": n},
            "arima": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": n},
            "lstm": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": n},
            "naive": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": n},
        },
        "principal_ranking": {"best_model": "lag_reg"},
        "backtest": {
            "target_dates": dates,
            "actual_closes": actual_closes,
            "predicted_closes_by_model": {
                "lag_reg": lag_reg or def_preds,
                "arima": arima or [c + 0.2 for c in actual_closes],
                "lstm": lstm or [c + 0.3 for c in actual_closes],
                "naive": naive or def_naive,
            },
        },
    }
    return parse_company_export(payload)


def test_1_principal_model_rmse_beats_naive():
    # actual: [100, 100, 100]
    # lag_reg err = 0.5 (RMSE = 0.5)
    # naive err = 1.0 (RMSE = 1.0)
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[100.5, 100.5, 100.5],
        naive=[101.0, 101.0, 101.0],
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    assert comp.beats_naive_rmse is True
    assert comp.rmse_comparison_status == ComparisonStatus.WIN
    assert comp.rmse_difference < 0.0


def test_2_principal_model_rmse_loses_to_naive():
    # lag_reg err = 2.0 (RMSE = 2.0)
    # naive err = 0.5 (RMSE = 0.5)
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[102.0, 102.0, 102.0],
        naive=[100.5, 100.5, 100.5],
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    assert comp.beats_naive_rmse is False
    assert comp.rmse_comparison_status == ComparisonStatus.LOSS
    assert comp.rmse_difference > 0.0


def test_3_exact_rmse_tie():
    # Both lag_reg and naive have err = 1.0
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[101.0, 101.0, 101.0],
        naive=[101.0, 101.0, 101.0],
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    # EXACT TIE IS NOT A WIN
    assert comp.beats_naive_rmse is False
    assert comp.rmse_comparison_status == ComparisonStatus.TIE
    assert pytest.approx(comp.rmse_difference, 1e-9) == 0.0


def test_4_principal_model_mae_beats_naive():
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[100.2, 100.2, 100.2],  # MAE = 0.2
        naive=[101.0, 101.0, 101.0],    # MAE = 1.0
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    assert comp.beats_naive_mae is True
    assert comp.mae_comparison_status == ComparisonStatus.WIN
    assert comp.mae_difference < 0.0


def test_5_principal_model_mae_loses_to_naive():
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[101.5, 101.5, 101.5],  # MAE = 1.5
        naive=[100.5, 100.5, 100.5],    # MAE = 0.5
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    assert comp.beats_naive_mae is False
    assert comp.mae_comparison_status == ComparisonStatus.LOSS
    assert comp.mae_difference > 0.0


def test_6_exact_mae_tie():
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[101.0, 99.0, 101.0],  # errors: +1, -1, +1 -> MAE = 1.0
        naive=[99.0, 101.0, 99.0],    # errors: -1, +1, -1 -> MAE = 1.0
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    # EXACT TIE IS NOT A WIN
    assert comp.beats_naive_mae is False
    assert comp.mae_comparison_status == ComparisonStatus.TIE
    assert pytest.approx(comp.mae_difference, 1e-9) == 0.0


def test_7_and_8_rmse_skill_positive_and_negative():
    # Positive skill: model_rmse = 0.5, naive_rmse = 1.0 -> skill = 1 - 0.5/1.0 = +0.50 (+50%)
    skill_pos = compute_skill(0.5, 1.0)
    assert skill_pos.skill_defined is True
    assert pytest.approx(skill_pos.skill_value, 1e-6) == 0.50
    assert skill_pos.skill_reason is None

    # Negative skill: model_rmse = 1.5, naive_rmse = 1.0 -> skill = 1 - 1.5/1.0 = -0.50 (-50%)
    skill_neg = compute_skill(1.5, 1.0)
    assert skill_neg.skill_defined is True
    assert pytest.approx(skill_neg.skill_value, 1e-6) == -0.50
    assert skill_neg.skill_reason is None


def test_9_and_10_mae_skill_positive_and_negative():
    skill_pos = compute_skill(0.2, 0.8)
    assert skill_pos.skill_defined is True
    assert pytest.approx(skill_pos.skill_value, 1e-6) == 0.75

    skill_neg = compute_skill(1.2, 0.8)
    assert skill_neg.skill_defined is True
    assert pytest.approx(skill_neg.skill_value, 1e-6) == -0.50


def test_11_and_12_zero_naive_rmse_and_mae_handled_safely():
    # Naive is perfect (0.0 error)
    # When model is also 0.0 -> TIE, skill is undefined (finite JSON-safe: None)
    res_tie = compute_skill(0.0, 0.0)
    assert res_tie.skill_defined is False
    assert res_tie.skill_value is None
    assert res_tie.skill_reason == "NAIVE_ERROR_ZERO"
    beats_tie, status_tie = determine_comparison_status(0.0, 0.0)
    assert beats_tie is False
    assert status_tie == ComparisonStatus.TIE

    # When model has error > 0 while naive is 0 -> LOSS, skill is undefined without -inf or ZeroDivisionError
    res_loss = compute_skill(0.5, 0.0)
    assert res_loss.skill_defined is False
    assert res_loss.skill_value is None
    assert res_loss.skill_reason == "NAIVE_ERROR_ZERO"
    beats_loss, status_loss = determine_comparison_status(0.5, 0.0)
    assert beats_loss is False
    assert status_loss == ComparisonStatus.LOSS


def test_13_best_principal_differs_from_best_evaluated_method():
    # Principals: lag_reg RMSE = 2.0, arima = 2.5, lstm = 3.0 -> Best principal is lag_reg
    # Naive: RMSE = 0.5 -> Best evaluated method is NAIVE
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[102.0, 102.0, 102.0],  # RMSE = 2.0
        arima=[102.5, 102.5, 102.5],    # RMSE = 2.5
        lstm=[103.0, 103.0, 103.0],     # RMSE = 3.0
        naive=[100.5, 100.5, 100.5],    # RMSE = 0.5
    )
    summary = evaluate_company_direct_oos(export)
    assert summary.best_principal_model == "lag_reg"
    assert summary.best_evaluated_method == "naive"
    assert summary.best_principal_model != summary.best_evaluated_method


def test_14_and_15_all_principals_worse_than_naive():
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[102.0, 102.0, 102.0],
        arima=[102.5, 102.5, 102.5],
        lstm=[103.0, 103.0, 103.0],
        naive=[100.1, 100.1, 100.1],
    )
    summary = evaluate_company_direct_oos(export)
    assert summary.all_principals_worse_than_naive is True
    assert summary.naive_is_best_evaluated_method is True
    assert summary.best_principal_beats_naive is False
    assert "None of the principal forecasting models outperformed the Naive benchmark" in summary.overall_status_message


def test_16_mase_below_one_but_loses_direct_holdout():
    # Development variation is very high (e.g. mase_denominator = 10.0)
    # Model MAE = 5.0 -> MASE = 5.0 / 10.0 = 0.5 (< 1.0!)
    # But on this exact holdout, Naive MAE is 1.0!
    # So model MAE (5.0) > Naive MAE (1.0) -> model LOSES direct comparison despite MASE < 1!
    export = _build_test_export(
        actual_closes=[100.0, 100.0, 100.0],
        lag_reg=[105.0, 105.0, 105.0],  # MAE = 5.0
        naive=[101.0, 101.0, 101.0],    # MAE = 1.0
        mase_denominator=10.0,
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    assert comp.mase_below_one is True   # MASE is 0.5 < 1.0
    assert comp.beats_naive_mae is False  # But lost direct MAE comparison to Naive!
    assert comp.beats_naive_rmse is False # Lost direct RMSE comparison to Naive!
    assert comp.mae_comparison_status == ComparisonStatus.LOSS


def test_17_and_18_negative_r2_preserved_and_not_used_for_ranking():
    # Model with extreme errors -> R^2 is heavily negative
    export = _build_test_export(
        actual_closes=[100.0, 105.0, 110.0],
        lag_reg=[200.0, 200.0, 200.0],  # heavily negative R^2
        arima=[102.0, 106.0, 111.0],    # close predictions
        lstm=[103.0, 107.0, 112.0],
    )
    summary = evaluate_company_direct_oos(export)
    comp = summary.models["lag_reg"]
    assert comp.model_r2 < 0.0  # preserved as negative, NOT clipped to 0
    # Ranking is strictly based on RMSE, NOT R^2
    assert summary.best_principal_model == "arima"


def test_19_exact_same_target_dates_required():
    # Attempting to parse payload where prediction length differs from target dates
    payload = {
        "symbol": "ALI",
        "split": {
            "evaluation_proportion": 0.2,
            "development_pairs": 50,
            "evaluation_pairs": 3,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-03",
        },
        "mase_denominator": 1.0,
        "metrics": {
            "lag_reg": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": 3},
            "arima": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": 3},
            "lstm": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": 3},
            "naive": {"rmse": 1.0, "mae": 1.0, "mase": 1.0, "r2": 0.8, "observations": 3},
        },
        "backtest": {
            "target_dates": ["2026-09-01", "2026-09-02", "2026-09-03"],
            "actual_closes": [100.0, 102.0, 105.0],
            "predicted_closes_by_model": {
                "lag_reg": [100.1, 102.1],  # 2 != 3
                "arima": [100.2, 102.2, 105.2],
                "lstm": [100.3, 102.3, 105.3],
                "naive": [100.0, 102.0, 105.0],
            },
        },
    }
    export = parse_company_export(payload)
    assert export.has_errors is True
    assert any("predicted_closes_by_model['lag_reg'] length" in i.message for i in export.issues)


def test_20_formal_frozen_refuses_without_independent_naive_evidence():
    export = _build_test_export(provenance_tier="FORMAL_FROZEN")
    # Calling evaluate_company_direct_oos on FORMAL_FROZEN without raw data MUST raise DirectOOSVerificationError
    with pytest.raises(DirectOOSVerificationError, match="RAW_DATA_UNAVAILABLE"):
        evaluate_company_direct_oos(export, raw_dates=None, raw_closes=None)


def test_21_demo_remains_usable_but_clearly_non_formal():
    export = _build_test_export(provenance_tier="DEMO")
    # DEMO can evaluate without raw data, but its status is NOT_RUN_DEMO
    summary = evaluate_company_direct_oos(export)
    assert summary.provenance_tier == ProvenanceTier.DEMO
    assert summary.naive_audit_status == "NOT_RUN_DEMO"
    assert summary.naive_source == "exported_demo"


def test_22_deterministic_tie_behavior():
    # Two models have identical RMSE: 1.0
    # Model A: lag_reg, Model B: arima
    # Tie-breaker 1: MAE. lag_reg has MAE = 0.8, arima has MAE = 0.9
    export = _build_test_export(
        actual_closes=[100.0, 100.0],
        lag_reg=[101.0, 99.0],   # errors: +1, -1 -> RMSE = 1.0, MAE = 1.0
        arima=[101.0, 101.0],    # errors: +1, +1 -> RMSE = 1.0, MAE = 1.0
        lstm=[105.0, 105.0],     # RMSE = 5.0
    )
    summary = evaluate_company_direct_oos(export)
    # Both RMSE and MAE are tied: deterministic alphabetical tie formatting
    assert summary.is_principal_tie is True
    assert summary.best_principal_model == "TIE:arima/lag_reg"


def test_oos_comparisons_to_dataframe():
    export = _build_test_export()
    summary = evaluate_company_direct_oos(export)
    df = oos_comparisons_to_dataframe([summary])
    assert not df.empty
    assert len(df) == 3  # lag_reg, arima, lstm
    expected_cols = [
        "symbol", "sector", "model", "model_rmse", "naive_rmse", "rmse_difference",
        "rmse_skill_vs_naive", "beats_naive_rmse", "rmse_comparison_status",
        "model_mae", "naive_mae", "mae_difference", "mae_skill_vs_naive",
        "beats_naive_mae", "mae_comparison_status", "model_mase", "naive_mase",
        "mase_below_one", "model_r2", "naive_r2", "best_principal_model",
        "best_evaluated_method", "best_principal_beats_naive",
        "all_principals_worse_than_naive", "naive_is_best_evaluated_method",
    ]
    for col in expected_cols:
        assert col in df.columns
