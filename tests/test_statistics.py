"""Unit and integration tests for independent within-company Diebold-Mariano statistical testing.

Covers all 26 Phase 3 test requirements:
1. exactly six pairs generated
2. squared-loss family generated
3. absolute-loss family generated
4. correct loss-differential sign convention
5. clearly superior A produces negative mean differential
6. clearly superior B produces positive mean differential
7. identical forecasts -> IDENTICAL_LOSS
8. identical losses -> p = 1
9. zero variance handled safely
10. insufficient observations rejected/undefined
11. non-finite values rejected
12. invalid HAC lag rejected
13. HLN h=1 factor equals sqrt((n-1)/n)
14. DM symmetry: swapping A/B negates statistic but preserves p-value
15. deterministic p-values
16. Holm adjusted p >= raw p
17. Holm adjusted p <= 1
18. Holm family size = 6
19. squared and absolute families corrected separately
20. different companies corrected separately
21. large deliberate difference becomes statistically significant
22. small/noisy difference is not falsely significant
23. alpha configurability
24. complete holdout date alignment enforced
25. 60-session subset cannot accidentally enter formal DM
26. reported ForecastPH DM mismatch can be surfaced if fixture supplied
"""

import math
import numpy as np
import pytest
import scipy.stats as stats

from datetime import date, timedelta
import math
import numpy as np
import pytest
import scipy.stats as stats

from recheck.schema import CompanyEvaluationExport, ProvenanceTier, parse_company_export
from recheck.statistics import (
    DMResult,
    DMStatus,
    LossFunction,
    REQUIRED_PAIRS,
    compute_diebold_mariano,
    compute_hln_factor,
    cross_check_forecastph_dm,
    dm_results_to_dataframe,
    evaluate_company_pairwise_dm as run_pairwise_dm,
    holm_step_down,
)


def _build_test_export(
    symbol: str = "TEST",
    n: int = 20,
    tier: str = "DEMO",
    actual: list[float] | None = None,
    lag_reg: list[float] | None = None,
    arima: list[float] | None = None,
    lstm: list[float] | None = None,
    naive: list[float] | None = None,
) -> CompanyEvaluationExport:
    act = actual if actual is not None else [100.0 + i for i in range(n)]
    start_d = date(2025, 1, 1)
    dates = [(start_d + timedelta(days=i)).isoformat() for i in range(n)]
    p_lr = lag_reg if lag_reg is not None else [act[i] + 0.1 * (1.0 + 0.5 * ((-1) ** i)) for i in range(n)]
    p_ar = arima if arima is not None else [act[i] + 0.2 * (1.0 + 0.5 * ((-1) ** (i + 1))) for i in range(n)]
    p_ls = lstm if lstm is not None else [act[i] + 0.3 * (1.0 + 0.5 * ((-1) ** i)) for i in range(n)]
    p_nv = naive if naive is not None else [act[i] + 0.4 * (1.0 + 0.5 * ((-1) ** (i + 1))) for i in range(n)]

    payload = {
        "symbol": symbol,
        "sector": "Financials",
        "provenance_tier": tier,
        "split": {
            "evaluation_proportion": 0.2,
            "development_pairs": 50,
            "evaluation_pairs": n,
            "evaluation_start": dates[0],
            "evaluation_end": dates[-1],
        },
        "mase_denominator": 1.0,
        "metrics": {
            "lag_reg": {"rmse": 0.1, "mae": 0.1, "mase": 0.1, "r2": 0.99, "observations": n},
            "arima": {"rmse": 0.2, "mae": 0.2, "mase": 0.2, "r2": 0.98, "observations": n},
            "lstm": {"rmse": 0.3, "mae": 0.3, "mase": 0.3, "r2": 0.97, "observations": n},
            "naive": {"rmse": 0.4, "mae": 0.4, "mase": 0.4, "r2": 0.96, "observations": n},
        },
        "principal_ranking": {"best_model": "lag_reg"},
        "backtest": {
            "target_dates": dates,
            "actual_closes": act,
            "predicted_closes_by_model": {
                "lag_reg": p_lr,
                "arima": p_ar,
                "lstm": p_ls,
                "naive": p_nv,
            },
        },
    }
    return parse_company_export(payload)


# 1. Exactly six pairs generated
def test_1_exactly_six_pairs_generated():
    export = _build_test_export()
    results = run_pairwise_dm(export)
    assert len(results.squared_family) == 6
    assert len(results.absolute_family) == 6
    pairs_in_results = [(r.model_a, r.model_b) for r in results.squared_family]
    assert pairs_in_results == REQUIRED_PAIRS


# 2. Squared-loss family generated
def test_2_squared_loss_family_generated():
    export = _build_test_export()
    results = run_pairwise_dm(export)
    assert all(r.loss_function == LossFunction.SQUARED for r in results.squared_family)


# 3. Absolute-loss family generated
def test_3_absolute_loss_family_generated():
    export = _build_test_export()
    results = run_pairwise_dm(export)
    assert all(r.loss_function == LossFunction.ABSOLUTE for r in results.absolute_family)


# 4. Correct loss-differential sign convention: d_t = loss_a_t - loss_b_t
def test_4_correct_loss_differential_sign_convention():
    actual = [10.0, 10.0, 10.0, 10.0]
    pred_a = [12.0, 12.0, 12.0, 12.0]  # error = -2 -> loss_sq = 4, loss_abs = 2
    pred_b = [11.0, 11.0, 11.0, 11.0]  # error = -1 -> loss_sq = 1, loss_abs = 1
    # For A vs B: d_t = loss_a - loss_b = 4 - 1 = +3 (mean_d > 0 since B is better)
    res_sq = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert pytest.approx(res_sq.mean_loss_a, 1e-9) == 4.0
    assert pytest.approx(res_sq.mean_loss_b, 1e-9) == 1.0
    assert pytest.approx(res_sq.mean_loss_differential, 1e-9) == 3.0


# 5. Clearly superior A produces negative mean differential
def test_5_clearly_superior_a_produces_negative_mean_differential():
    actual = [10.0, 10.0, 10.0, 10.0, 10.0]
    pred_a = [10.0, 10.0, 10.0, 10.0, 10.0]  # perfect: loss = 0
    pred_b = [15.0, 15.0, 15.0, 15.0, 15.0]  # error = -5: loss = 25
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert res.mean_loss_differential < 0
    assert res.lower_loss_model == "A"


# 6. Clearly superior B produces positive mean differential
def test_6_clearly_superior_b_produces_positive_mean_differential():
    actual = [10.0, 10.0, 10.0, 10.0, 10.0]
    pred_a = [15.0, 15.0, 15.0, 15.0, 15.0]  # error = -5: loss = 25
    pred_b = [10.0, 10.0, 10.0, 10.0, 10.0]  # perfect: loss = 0
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert res.mean_loss_differential > 0
    assert res.lower_loss_model == "B"


# 7. Identical forecasts -> IDENTICAL_LOSS
def test_7_identical_forecasts_produce_identical_loss():
    actual = [10.0, 20.0, 30.0, 40.0, 50.0]
    pred_a = [10.5, 20.5, 30.5, 40.5, 50.5]
    pred_b = [10.5, 20.5, 30.5, 40.5, 50.5]
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B")
    assert res.status == DMStatus.IDENTICAL_LOSS
    assert res.dm_statistic_hln == 0.0
    assert res.raw_p_value == 1.0
    assert res.significant_raw is False


# 8. Identical losses -> p = 1
def test_8_identical_losses_produce_p_value_one():
    actual = [10.0, 10.0, 10.0, 10.0, 10.0]
    # Errors are +1 and -1, so squared loss is 1.0 for both models everywhere
    pred_a = [11.0, 11.0, 11.0, 11.0, 11.0]
    pred_b = [9.0, 9.0, 9.0, 9.0, 9.0]
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert res.status == DMStatus.IDENTICAL_LOSS
    assert res.raw_p_value == 1.0
    assert res.significant_raw is False


# 9a. Zero variance with nonzero constant mean produces DEGENERATE status
def test_9a_zero_variance_nonzero_mean_produces_degenerate():
    actual = [10.0, 10.0, 10.0, 10.0]
    pred_a = [12.0, 12.0, 12.0, 12.0]  # error = -2, loss = 4
    pred_b = [11.0, 11.0, 11.0, 11.0]  # error = -1, loss = 1
    # differential d_t = 3 everywhere -> sample variance is 0, but mean is +3!
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert res.status == DMStatus.DEGENERATE
    assert res.reason == "ZERO_LONG_RUN_VARIANCE_WITH_NONZERO_MEAN"
    assert res.significant_raw is False
    assert res.dm_statistic_hln is None
    assert res.raw_p_value is None
    assert res.long_run_variance == 0.0
    assert pytest.approx(res.mean_loss_differential, 1e-9) == 3.0


# 9b. Zero variance with zero mean produces IDENTICAL_LOSS status
def test_9b_zero_variance_zero_mean_produces_identical_loss():
    actual = [10.0, 10.0, 10.0, 10.0]
    pred_a = [12.0, 12.0, 12.0, 12.0]  # loss = 4
    pred_b = [12.0, 12.0, 12.0, 12.0]  # loss = 4
    # differential d_t = 0 everywhere
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert res.status == DMStatus.IDENTICAL_LOSS
    assert res.reason == "IDENTICAL_LOSS_SERIES"
    assert res.significant_raw is False
    assert res.dm_statistic_hln == 0.0
    assert res.raw_p_value == 1.0
    assert res.mean_loss_differential == 0.0


# 10. Insufficient observations rejected/undefined
def test_10_insufficient_observations_rejected_or_undefined():
    actual = [10.0, 20.0]  # n = 2 < 3
    pred_a = [11.0, 21.0]
    pred_b = [12.0, 22.0]
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B")
    assert res.status == DMStatus.UNDEFINED
    assert res.reason == "INSUFFICIENT_OBSERVATIONS"
    assert res.significant_raw is False


# 11. Non-finite values rejected
def test_11_non_finite_values_rejected():
    actual = [10.0, float("nan"), 30.0, 40.0]
    pred_a = [10.0, 20.0, 30.0, 40.0]
    pred_b = [10.0, 20.0, 30.0, 40.0]
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B")
    assert res.status == DMStatus.UNDEFINED
    assert res.reason == "NON_FINITE_DATA"


# 12. Invalid HAC lag rejected
def test_12_invalid_hac_lag_rejected():
    actual = [10.0, 20.0, 30.0, 40.0, 50.0]
    pred_a = [11.0, 21.0, 31.0, 41.0, 51.0]
    pred_b = [12.0, 22.0, 32.0, 42.0, 52.0]
    res_neg = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", hac_lag=-1)
    assert res_neg.status == DMStatus.UNDEFINED
    assert res_neg.reason == "INVALID_HAC_LAG"

    res_large = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", hac_lag=10)
    assert res_large.status == DMStatus.UNDEFINED
    assert res_large.reason == "INVALID_HAC_LAG"


# 13. HLN h=1 factor equals sqrt((n-1)/n)
def test_13_hln_factor_equals_formula():
    for n in (5, 10, 20, 60, 331):
        factor = compute_hln_factor(n, h=1)
        expected = math.sqrt((n - 1) / n)
        assert pytest.approx(factor, 1e-12) == expected


# 14. DM symmetry: swapping A/B negates statistic but preserves p-value
def test_14_dm_symmetry():
    actual = [10.0, 10.0, 10.0, 10.0, 10.0]
    pred_a = [10.0, 10.0, 10.0, 10.0, 10.0]
    pred_b = [12.0, 11.0, 12.0, 11.0, 12.0]
    res_ab = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED, hac_lag=0)
    res_ba = compute_diebold_mariano("T", actual, pred_b, pred_a, "B", "A", LossFunction.SQUARED, hac_lag=0)

    assert pytest.approx(res_ab.dm_statistic_hln, 1e-10) == -res_ba.dm_statistic_hln
    assert pytest.approx(res_ab.dm_statistic_uncorrected, 1e-10) == -res_ba.dm_statistic_uncorrected
    assert pytest.approx(res_ab.raw_p_value, 1e-10) == res_ba.raw_p_value
    assert res_ab.lower_loss_model == "A"
    assert res_ba.lower_loss_model == "A"


# 15. Deterministic p-values match known numerical fixture
def test_15_deterministic_p_values_analytical_fixture():
    # Analytically derived fixture: n=5, actual=[10]*5, pred_a=[10]*5, pred_b=[12, 11, 12, 11, 12]
    # mean_d = -2.8, gamma_0 = 2.16, LRV (lag=0) = 2.16
    # uncorrected DM = -2.8 * sqrt(5) / sqrt(2.16)
    # HLN factor = sqrt(4/5)
    # DM_HLN = -5.6 / sqrt(2.16) ~= -3.81031737766
    # df = 4, p-value ~= 0.0189349787
    actual = [10.0, 10.0, 10.0, 10.0, 10.0]
    pred_a = [10.0, 10.0, 10.0, 10.0, 10.0]
    pred_b = [12.0, 11.0, 12.0, 11.0, 12.0]
    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED, hac_lag=0)

    expected_dm = -5.6 / math.sqrt(2.16)
    expected_p = 2.0 * float(stats.t.sf(abs(expected_dm), df=4))

    assert pytest.approx(res.mean_loss_differential, 1e-12) == -2.8
    assert pytest.approx(res.long_run_variance, 1e-12) == 2.16
    assert pytest.approx(res.dm_statistic_hln, 1e-12) == expected_dm
    assert pytest.approx(res.raw_p_value, 1e-12) == expected_p


# 16. Holm adjusted p >= raw p
def test_16_holm_adjusted_p_greater_or_equal_to_raw():
    export = _build_test_export()
    res = run_pairwise_dm(export)
    for r in res.all_results:
        assert r.holm_adjusted_p_value >= r.raw_p_value - 1e-12


# 17. Holm adjusted p <= 1
def test_17_holm_adjusted_p_bounded_by_one():
    export = _build_test_export()
    res = run_pairwise_dm(export)
    for r in res.all_results:
        assert r.holm_adjusted_p_value <= 1.0


# 18. Holm family size = 6
def test_18_holm_family_size_is_six():
    export = _build_test_export()
    res = run_pairwise_dm(export)
    for r in res.all_results:
        assert r.holm_family_size == 6


# 19. Squared and absolute families corrected separately
def test_19_squared_and_absolute_families_corrected_separately():
    export = _build_test_export()
    res = run_pairwise_dm(export)
    assert len(res.squared_family) == 6
    assert len(res.absolute_family) == 6
    assert all(r.holm_family_size == 6 for r in res.squared_family)
    assert all(r.holm_family_size == 6 for r in res.absolute_family)


# 20. Different companies corrected separately
def test_20_different_companies_corrected_separately():
    export1 = _build_test_export("BPI")
    export2 = _build_test_export("ALI")
    res1 = run_pairwise_dm(export1)
    res2 = run_pairwise_dm(export2)
    assert all(r.holm_family_size == 6 for r in res1.squared_family)
    assert all(r.holm_family_size == 6 for r in res2.squared_family)


# 21. Large deliberate difference becomes statistically significant
def test_21_large_deliberate_difference_statistically_significant():
    n = 100
    actual = [100.0] * n
    # Model A is perfect with small noise
    pred_a = [100.0 + ((-1) ** i) * 0.05 for i in range(n)]
    # Model B has huge persistent errors
    pred_b = [120.0 + ((-1) ** i) * 0.1 for i in range(n)]

    export = _build_test_export(
        n=n,
        actual=actual,
        lag_reg=pred_a,
        arima=pred_b,
        lstm=pred_b,
        naive=pred_b,
    )
    res = run_pairwise_dm(export)
    lr_vs_ar = next(r for r in res.squared_family if r.model_a == "lag_reg" and r.model_b == "arima")
    assert lr_vs_ar.significant_holm is True
    assert lr_vs_ar.status == DMStatus.A_SIGNIFICANTLY_LOWER_LOSS
    assert lr_vs_ar.lower_loss_model == "lag_reg"


# 22. Small/noisy difference is not falsely significant
def test_22_small_noisy_difference_not_falsely_significant():
    np.random.seed(42)
    n = 60
    actual = [100.0] * n
    noise_a = np.random.normal(0, 1.0, n)
    noise_b = np.random.normal(0, 1.0, n)
    pred_a = [100.0 + noise_a[i] for i in range(n)]
    pred_b = [100.0 + noise_b[i] for i in range(n)]

    res = compute_diebold_mariano("T", actual, pred_a, pred_b, "A", "B", LossFunction.SQUARED)
    assert res.significant_raw is False
    assert res.status == DMStatus.LOWER_LOSS_NOT_SIGNIFICANT


# 23. Alpha configurability
def test_23_alpha_configurability():
    export = _build_test_export()
    res_05 = run_pairwise_dm(export, alpha=0.05)
    res_01 = run_pairwise_dm(export, alpha=0.01)
    assert res_05.alpha == 0.05
    assert res_01.alpha == 0.01
    assert all(r.alpha == 0.05 for r in res_05.all_results)
    assert all(r.alpha == 0.01 for r in res_01.all_results)


# 24. Complete holdout date alignment enforced
def test_24_complete_holdout_date_alignment_enforced():
    import dataclasses
    export = _build_test_export()
    corrupt_export = dataclasses.replace(
        export,
        actual_closes=export.actual_closes[:-1],  # 19 items instead of 20!
    )
    with pytest.raises(ValueError, match="alignment"):
        run_pairwise_dm(corrupt_export)


# 25. 60-session subset cannot accidentally enter formal DM
def test_25_60_session_subset_cannot_enter_formal_dm():
    export_60 = _build_test_export(n=60, tier="FORMAL_FROZEN")
    with pytest.raises(ValueError, match="60-session display subset cannot enter formal DM"):
        run_pairwise_dm(export_60)


# 26. Reported ForecastPH DM mismatch can be surfaced if fixture supplied
def test_26_reported_forecastph_dm_mismatch_surfaced():
    export = _build_test_export(n=20)
    ind_res = run_pairwise_dm(export)

    fixture = {
        "statistical_tests": {
            "diebold_mariano": [
                {
                    "model_1": "lag_reg",
                    "model_2": "arima",
                    "loss_function": "squared",
                    "dm_statistic": 999.99,  # Deliberate mismatch
                    "raw_p_value": 0.0001,
                }
            ]
        }
    }
    checks = cross_check_forecastph_dm(fixture, ind_res, tolerance=1e-4)
    assert len(checks) == 1
    chk = checks[0]
    assert chk.is_consistent is False
    assert "Discrepancy" in chk.discrepancy_note


# Auxiliary: Test DataFrame conversion
def test_dm_results_to_dataframe():
    export = _build_test_export()
    res = run_pairwise_dm(export)
    df = dm_results_to_dataframe(res)
    assert not df.empty
    assert len(df) == 12  # 6 squared + 6 absolute
    assert "pair" in df.columns
    assert "holm_adjusted_p_value" in df.columns
    assert "status" in df.columns
