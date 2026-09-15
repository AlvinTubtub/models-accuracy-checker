"""Comprehensive tests for Phase 4 Cross-Company Statistical Testing and Aggregation.

Covers all 27 required validation cases:
1. complete 15-company x 4-method matrix accepted
2. missing company rejected for FORMAL_FROZEN
3. missing method rejected
4. non-finite MASE rejected
5. within-company ranking correct
6. tied MASE values receive tied average ranks
7. Friedman detects deliberately separated methods
8. Friedman nonsignificant on equivalent methods
9. nonsignificant Friedman skips Wilcoxon
10. significant Friedman triggers exactly six Wilcoxon tests
11. Wilcoxon operates on paired company values
12. deterministic Wilcoxon method policy
13. zero differences handled
14. Holm family size exactly 6
15. adjusted p >= raw p
16. adjusted p <= 1
17. Holm monotonicity
18. Kendall's W calculation
19. raw peso RMSE is not used for cross-company winner selection
20. high-priced company cannot dominate aggregate ranking
21. negative R² cannot influence aggregate ranking
22. MASE < 1 count remains separate from direct Naive win count
23. significant pair direction agrees with paired MASE evidence
24. method ties remain ties statistically
25. results invariant to company input ordering
26. DEMO provenance remains labeled DEMO
27. cross-company output explicitly identifies analysis as supporting evidence
"""

from __future__ import annotations

from datetime import date
import math
import numpy as np
import pandas as pd
import pytest

from recheck.cross_company import (
    DEFAULT_CROSS_COMPANY_METHODS,
    PAIRWISE_METHOD_PAIRS,
    REQUESTED_WILCOXON_POLICY,
    SUPPORTING_EVIDENCE_NOTE,
    CrossCompanySummary,
    CrossCompanyVerificationError,
    FriedmanResult,
    MethodDescriptiveStats,
    PairwiseWilcoxonResult,
    build_mase_matrix,
    compute_friedman_test,
    compute_pairwise_wilcoxon,
    compute_within_company_ranks,
    determine_wilcoxon_method,
    evaluate_cross_company,
    evaluate_cross_company_from_matrix,
    method_stats_to_dataframe,
    normalize_kendalls_w,
    rank_matrix_to_dataframe,
    wilcoxon_results_to_dataframe,
)
from recheck.schema import CompanyEvaluationExport, ProvenanceTier
from recheck.tolerances import NUMERICAL_BOUNDARY_TOLERANCE, TIE_TOLERANCE


def _make_dummy_export(
    symbol: str,
    *,
    tier: ProvenanceTier = ProvenanceTier.DEMO,
    mase_values: dict[str, float] | None = None,
    denom: float = 1.0,
    n_days: int = 10,
    r2_values: dict[str, float] | None = None,
) -> CompanyEvaluationExport:
    """Helper to construct a synthetically consistent CompanyEvaluationExport."""
    if mase_values is None:
        mase_values = {"lag_reg": 0.8, "arima": 0.9, "lstm": 1.1, "naive": 1.0}

    # actuals = [10.0, 10.0, ...]
    actuals = [10.0] * n_days
    # For each method, craft predicted closes so MAE = mase_value * denom
    preds: dict[str, list[float]] = {}
    for m, mase in mase_values.items():
        mae_target = mase * denom
        preds[m] = [10.0 + mae_target] * n_days

    # reported metrics
    metrics: dict[str, dict[str, float]] = {}
    for m, mase in mase_values.items():
        mae_target = mase * denom
        r2_val = r2_values.get(m, 0.5) if r2_values else 0.5
        metrics[m] = {
            "rmse": mae_target,
            "mae": mae_target,
            "mase": mase,
            "r2": r2_val,
        }

    return CompanyEvaluationExport(
        symbol=symbol,
        name=f"Company {symbol}",
        sector="Financials",
        evaluation_proportion=0.2,
        development_pairs=100,
        evaluation_pairs=n_days,
        evaluation_start=date(2020, 1, 2),
        evaluation_end=date(2020, 1, 15),
        mase_denominator=denom,
        reported_metrics=metrics,
        reported_best_model="lag_reg",
        target_dates=[date(2020, 1, 2 + i) for i in range(n_days)],
        actual_closes=actuals,
        predicted_closes_by_model=preds,
        provenance_tier=tier,
    )


# =========================================================================
# 1. Complete 15-company x 4-method matrix accepted
# =========================================================================
def test_1_complete_15_company_x_4_method_matrix_accepted():
    exports = {
        f"SYM{i:02d}": _make_dummy_export(f"SYM{i:02d}", tier=ProvenanceTier.FORMAL_FROZEN)
        for i in range(15)
    }
    symbols, matrix = build_mase_matrix(exports, strict_formal=True)
    assert len(symbols) == 15
    for s in symbols:
        assert set(matrix[s].keys()) == set(DEFAULT_CROSS_COMPANY_METHODS)

    summary = evaluate_cross_company(exports, strict_formal=True)
    assert summary.company_count == 15
    assert len(summary.methods) == 4


# =========================================================================
# 2. Missing company rejected for FORMAL_FROZEN
# =========================================================================
def test_2_missing_company_rejected_for_formal_frozen():
    # Only 14 companies instead of required 15
    exports = {
        f"SYM{i:02d}": _make_dummy_export(f"SYM{i:02d}", tier=ProvenanceTier.FORMAL_FROZEN)
        for i in range(14)
    }
    with pytest.raises(CrossCompanyVerificationError, match="FORMAL_FROZEN experiment requires 15 matched companies"):
        evaluate_cross_company(exports, strict_formal=True)


# =========================================================================
# 3. Missing method rejected
# =========================================================================
def test_3_missing_method_rejected():
    exports = {
        f"SYM{i:02d}": _make_dummy_export(f"SYM{i:02d}")
        for i in range(15)
    }
    # Deliberately remove arima from SYM00
    del exports["SYM00"].predicted_closes_by_model["arima"]

    with pytest.raises(CrossCompanyVerificationError, match="missing required evaluation series for method 'arima'"):
        evaluate_cross_company(exports, strict_formal=False)


# =========================================================================
# 4. Non-finite MASE rejected
# =========================================================================
def test_4_non_finite_mase_rejected():
    exports = {
        f"SYM{i:02d}": _make_dummy_export(f"SYM{i:02d}")
        for i in range(15)
    }
    # Set non-finite actual close in SYM01
    exports["SYM01"].actual_closes[0] = float("nan")

    with pytest.raises((CrossCompanyVerificationError, ValueError)):
        evaluate_cross_company(exports, strict_formal=False)


# =========================================================================
# 5. Within-company ranking correct
# =========================================================================
def test_5_within_company_ranking_correct():
    mase_matrix = {
        "ALI": {"lag_reg": 0.50, "arima": 0.80, "lstm": 1.20, "naive": 1.00}
    }
    ranks = compute_within_company_ranks(mase_matrix, ["ALI"], DEFAULT_CROSS_COMPANY_METHODS)
    # lag_reg (0.50) -> 1.0, arima (0.80) -> 2.0, naive (1.00) -> 3.0, lstm (1.20) -> 4.0
    assert ranks["ALI"]["lag_reg"] == 1.0
    assert ranks["ALI"]["arima"] == 2.0
    assert ranks["ALI"]["naive"] == 3.0
    assert ranks["ALI"]["lstm"] == 4.0


# =========================================================================
# 6. Tied MASE values receive tied average ranks
# =========================================================================
def test_6_tied_mase_values_receive_tied_average_ranks():
    mase_matrix = {
        "BDO": {"lag_reg": 0.70, "arima": 0.70, "lstm": 1.10, "naive": 1.50}
    }
    ranks = compute_within_company_ranks(mase_matrix, ["BDO"], DEFAULT_CROSS_COMPANY_METHODS)
    # lag_reg and arima tie for 1st and 2nd -> average rank (1+2)/2 = 1.5
    assert ranks["BDO"]["lag_reg"] == 1.5
    assert ranks["BDO"]["arima"] == 1.5
    assert ranks["BDO"]["lstm"] == 3.0
    assert ranks["BDO"]["naive"] == 4.0


# =========================================================================
# 7. Friedman detects deliberately separated methods
# =========================================================================
def test_7_friedman_detects_deliberately_separated_methods():
    # Construct 15 companies where lag_reg < arima < lstm < naive consistently
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.01 * i,
            "arima": 0.75 + 0.01 * i,
            "lstm": 1.00 + 0.01 * i,
            "naive": 1.25 + 0.01 * i,
        }
        for i in range(15)
    }
    symbols = sorted(mase_matrix.keys())
    res = compute_friedman_test(mase_matrix, symbols, DEFAULT_CROSS_COMPANY_METHODS)
    assert res.significant is True
    assert res.status == "SIGNIFICANT"
    assert res.raw_p_value < 0.001
    assert res.kendalls_w == pytest.approx(1.0, abs=1e-6)


# =========================================================================
# 8. Friedman nonsignificant on equivalent methods
# =========================================================================
def test_8_friedman_nonsignificant_on_equivalent_methods():
    # Construct balanced rank distributions where methods take turns winning
    orders = [
        {"lag_reg": 0.5, "arima": 0.6, "lstm": 0.7, "naive": 0.8},
        {"lag_reg": 0.8, "arima": 0.5, "lstm": 0.6, "naive": 0.7},
        {"lag_reg": 0.7, "arima": 0.8, "lstm": 0.5, "naive": 0.6},
        {"lag_reg": 0.6, "arima": 0.7, "lstm": 0.8, "naive": 0.5},
    ]
    mase_matrix = {f"SYM{i:02d}": orders[i % 4] for i in range(16)}
    symbols = sorted(mase_matrix.keys())
    res = compute_friedman_test(mase_matrix, symbols, DEFAULT_CROSS_COMPANY_METHODS)
    assert res.significant is False
    assert res.status == "NOT_SIGNIFICANT"
    assert res.raw_p_value == pytest.approx(1.0, abs=1e-4)


# =========================================================================
# 9. Nonsignificant Friedman skips Wilcoxon
# =========================================================================
def test_9_nonsignificant_friedman_skips_wilcoxon():
    orders = [
        {"lag_reg": 0.5, "arima": 0.6, "lstm": 0.7, "naive": 0.8},
        {"lag_reg": 0.8, "arima": 0.5, "lstm": 0.6, "naive": 0.7},
        {"lag_reg": 0.7, "arima": 0.8, "lstm": 0.5, "naive": 0.6},
        {"lag_reg": 0.6, "arima": 0.7, "lstm": 0.8, "naive": 0.5},
    ]
    mase_matrix = {f"SYM{i:02d}": orders[i % 4] for i in range(16)}
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    assert summary.friedman.significant is False
    assert summary.posthoc_status == "NOT_RUN_FRIEDMAN_NOT_SIGNIFICANT"
    assert len(summary.pairwise_wilcoxon) == 0


# =========================================================================
# 10. Significant Friedman triggers exactly six Wilcoxon tests
# =========================================================================
def test_10_significant_friedman_triggers_exactly_six_wilcoxon_tests():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.01 * i,
            "arima": 0.75 + 0.01 * i,
            "lstm": 1.00 + 0.01 * i,
            "naive": 1.25 + 0.01 * i,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    assert summary.friedman.significant is True
    assert summary.posthoc_status == "RUN"
    assert len(summary.pairwise_wilcoxon) == 6


# =========================================================================
# 11. Wilcoxon operates on paired company values
# =========================================================================
def test_11_wilcoxon_operates_on_paired_company_values():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.60 + 0.01 * i,
            "arima": 0.80 + 0.01 * i,
            "lstm": 1.00,
            "naive": 1.20,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    pair = next(p for p in summary.pairwise_wilcoxon if p.model_a == "lag_reg" and p.model_b == "arima")
    assert pair.n_companies == 15
    assert pair.n_nonzero_differences == 15
    # Every paired difference is 0.60 - 0.80 = -0.20
    assert pair.median_paired_difference == pytest.approx(-0.20, abs=1e-6)


# =========================================================================
# 12. Deterministic Wilcoxon method policy
# =========================================================================
def test_12_deterministic_wilcoxon_method_policy():
    # Case A: Constant differences across companies -> tied absolute differences -> approx
    mase_matrix_tied = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50,
            "arima": 0.80,
            "lstm": 1.00,
            "naive": 1.20,
        }
        for i in range(15)
    }
    summary_tied = evaluate_cross_company_from_matrix(mase_matrix_tied)
    for pair in summary_tied.pairwise_wilcoxon:
        assert pair.requested_method_policy == REQUESTED_WILCOXON_POLICY
        assert pair.actual_method_used == "approx"
        assert pair.method_used == "approx"
        assert pair.zero_method == "wilcox"
        assert pair.alternative == "two-sided"
        assert pair.n_companies == 15
        assert pair.n_pairs == 15
        assert pair.n_nonzero_differences == 15
        assert pair.has_zero_differences is False
        assert pair.has_tied_absolute_differences is True

    # Case B: Clean untied nonzero differences (N <= 50) -> exact
    mase_matrix_clean = {
        f"SYM{i:02d}": {
            "lag_reg": 0.40 + 0.01 * i,
            "arima": 0.70 + 0.03 * i,
            "lstm": 1.00 + 0.05 * i,
            "naive": 1.40 + 0.07 * i,
        }
        for i in range(15)
    }
    summary_clean = evaluate_cross_company_from_matrix(mase_matrix_clean)
    for pair in summary_clean.pairwise_wilcoxon:
        assert pair.requested_method_policy == REQUESTED_WILCOXON_POLICY
        assert pair.actual_method_used == "exact"
        assert pair.method_used == "exact"
        assert pair.zero_method == "wilcox"
        assert pair.alternative == "two-sided"
        assert pair.n_companies == 15
        assert pair.n_pairs == 15
        assert pair.n_nonzero_differences == 15
        assert pair.has_zero_differences is False
        assert pair.has_tied_absolute_differences is False


def test_determine_wilcoxon_method_unit_cases():
    # 1. Clean sample: N <= 50, no zeros, no ties -> exact
    x1 = np.array([1.0, 2.0, 3.0, 4.0])
    y1 = np.array([0.5, 1.2, 2.1, 2.9])  # diffs: 0.5, 0.8, 0.9, 1.1
    method, has_zeros, has_ties, n_nonz = determine_wilcoxon_method(x1, y1)
    assert method == "exact"
    assert has_zeros is False
    assert has_ties is False
    assert n_nonz == 4

    # 2. Sample with zero differences -> approx
    x2 = np.array([1.0, 2.0, 3.0])
    y2 = np.array([1.0, 1.2, 2.1])  # diffs: 0.0, 0.8, 0.9
    method, has_zeros, has_ties, n_nonz = determine_wilcoxon_method(x2, y2)
    assert method == "approx"
    assert has_zeros is True
    assert has_ties is False
    assert n_nonz == 2

    # 3. Sample with tied absolute differences -> approx
    x3 = np.array([1.0, 2.0, 3.0, 4.0])
    y3 = np.array([0.5, 1.5, 2.5, 3.5])  # diffs: 0.5, 0.5, 0.5, 0.5
    method, has_zeros, has_ties, n_nonz = determine_wilcoxon_method(x3, y3)
    assert method == "approx"
    assert has_zeros is False
    assert has_ties is True
    assert n_nonz == 4

    # 4. All zero differences -> none
    x4 = np.array([1.0, 2.0, 3.0])
    y4 = np.array([1.0, 2.0, 3.0])
    method, has_zeros, has_ties, n_nonz = determine_wilcoxon_method(x4, y4)
    assert method == "none"
    assert has_zeros is True
    assert has_ties is False
    assert n_nonz == 0

    # 5. Clean untied sample but N > 50 -> approx
    n = 55
    x5 = np.arange(n, dtype=float) * 2.0
    y5 = np.arange(n, dtype=float) * 1.0  # diffs: 0.0, 1.0, 2.0, ...
    # make all nonzero and untied
    x5 = x5 + 1.0
    # diffs = 1.0, 2.0, 3.0, ... untied and nonzero
    method, has_zeros, has_ties, n_nonz = determine_wilcoxon_method(x5, y5)
    assert method == "approx"
    assert has_zeros is False
    assert has_ties is False
    assert n_nonz == 55


# =========================================================================
# 13. Zero differences handled
# =========================================================================
def test_13_zero_differences_handled():
    # 5 companies tie, 10 companies have lag_reg better than arima
    mase_matrix = {}
    for i in range(15):
        if i < 5:
            mase_matrix[f"SYM{i:02d}"] = {"lag_reg": 0.5, "arima": 0.5, "lstm": 1.0, "naive": 1.2}
        else:
            mase_matrix[f"SYM{i:02d}"] = {"lag_reg": 0.4, "arima": 0.8, "lstm": 1.0, "naive": 1.2}

    symbols = sorted(mase_matrix.keys())
    results = compute_pairwise_wilcoxon(mase_matrix, symbols)
    pair = next(p for p in results if p.model_a == "lag_reg" and p.model_b == "arima")
    assert pair.n_companies == 15
    assert pair.n_nonzero_differences == 10
    assert pair.wilcoxon_statistic is not None
    assert pair.raw_p_value is not None


# =========================================================================
# 14. Holm family size exactly 6
# =========================================================================
def test_14_holm_family_size_exactly_six():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50,
            "arima": 0.70,
            "lstm": 0.90,
            "naive": 1.10,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    for pair in summary.pairwise_wilcoxon:
        assert pair.holm_family_size == 6


# =========================================================================
# 15. Adjusted p >= raw p
# =========================================================================
def test_15_adjusted_p_greater_or_equal_to_raw():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.02 * (i % 3),
            "arima": 0.60 + 0.01 * (i % 4),
            "lstm": 0.90,
            "naive": 1.10,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    for pair in summary.pairwise_wilcoxon:
        if pair.raw_p_value is not None and pair.holm_adjusted_p_value is not None:
            assert pair.holm_adjusted_p_value >= pair.raw_p_value - 1e-9


# =========================================================================
# 16. Adjusted p <= 1
# =========================================================================
def test_16_adjusted_p_bounded_by_one():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.05 * (i % 2),
            "arima": 0.60 + 0.05 * ((i + 1) % 2),
            "lstm": 0.90,
            "naive": 1.10,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    for pair in summary.pairwise_wilcoxon:
        if pair.holm_adjusted_p_value is not None:
            assert pair.holm_adjusted_p_value <= 1.0


# =========================================================================
# 17. Holm monotonicity
# =========================================================================
def test_17_holm_monotonicity():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.01 * i,
            "arima": 0.65 + 0.02 * (i % 3),
            "lstm": 0.85 + 0.01 * (i % 2),
            "naive": 1.05,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    # Sort by raw p-value
    sorted_pairs = sorted(summary.pairwise_wilcoxon, key=lambda p: p.raw_p_value or 0.0)
    for i in range(1, len(sorted_pairs)):
        assert sorted_pairs[i].holm_adjusted_p_value >= sorted_pairs[i - 1].holm_adjusted_p_value - 1e-9


# =========================================================================
# 18. Kendall's W calculation and boundary handling
# =========================================================================
def test_18_kendalls_w_calculation():
    # Perfect rank concordance: W = 1.0
    mase_matrix = {
        f"SYM{i:02d}": {"lag_reg": 0.5, "arima": 0.7, "lstm": 0.9, "naive": 1.1}
        for i in range(15)
    }
    symbols = sorted(mase_matrix.keys())
    res = compute_friedman_test(mase_matrix, symbols, DEFAULT_CROSS_COMPANY_METHODS)
    assert res.kendalls_w == pytest.approx(1.0, abs=1e-6)

    # Identical methods: W = 0.0
    mase_matrix_ties = {
        f"SYM{i:02d}": {"lag_reg": 1.0, "arima": 1.0, "lstm": 1.0, "naive": 1.0}
        for i in range(15)
    }
    res_ties = compute_friedman_test(mase_matrix_ties, symbols, DEFAULT_CROSS_COMPANY_METHODS)
    assert res_ties.kendalls_w == 0.0


def test_kendalls_w_boundary_normalization_cases():
    # 1. Tiny negative within tolerance -> adjusted to 0.0
    assert normalize_kendalls_w(-1e-10) == 0.0
    assert normalize_kendalls_w(-NUMERICAL_BOUNDARY_TOLERANCE) == 0.0

    # 2. Tiny above 1.0 within tolerance -> adjusted to 1.0
    assert normalize_kendalls_w(1.0 + 1e-10) == 1.0
    assert normalize_kendalls_w(1.0 + NUMERICAL_BOUNDARY_TOLERANCE) == 1.0

    # 3. Legitimate exact boundaries and intermediate
    assert normalize_kendalls_w(0.0) == 0.0
    assert normalize_kendalls_w(1.0) == 1.0
    assert normalize_kendalls_w(0.5) == 0.5
    assert normalize_kendalls_w(0.725) == 0.725

    # 4. Materially negative outside tolerance -> raises CrossCompanyVerificationError
    with pytest.raises(CrossCompanyVerificationError, match="materially outside theoretical bounds"):
        normalize_kendalls_w(-0.05)

    with pytest.raises(CrossCompanyVerificationError, match="materially outside theoretical bounds"):
        normalize_kendalls_w(-1.0001e-9)

    # 5. Materially above 1.0 outside tolerance -> raises CrossCompanyVerificationError
    with pytest.raises(CrossCompanyVerificationError, match="materially outside theoretical bounds"):
        normalize_kendalls_w(1.05)

    with pytest.raises(CrossCompanyVerificationError, match="materially outside theoretical bounds"):
        normalize_kendalls_w(1.0 + 1.0001e-9)

    # 6. Non-finite values -> raises CrossCompanyVerificationError
    with pytest.raises(CrossCompanyVerificationError, match="non-finite"):
        normalize_kendalls_w(float("nan"))
    with pytest.raises(CrossCompanyVerificationError, match="non-finite"):
        normalize_kendalls_w(float("inf"))


def test_kendalls_w_invalid_statistic_in_friedman(monkeypatch):
    # If Kendall's W is materially invalid, compute_friedman_test returns INVALID_STATISTIC
    def _mock_normalize(w, eps=1e-9):
        raise CrossCompanyVerificationError("Mock material violation")

    monkeypatch.setattr("recheck.cross_company.normalize_kendalls_w", _mock_normalize)

    mase_matrix = {
        f"SYM{i:02d}": {"lag_reg": 0.5, "arima": 0.7, "lstm": 0.9, "naive": 1.1}
        for i in range(15)
    }
    symbols = sorted(mase_matrix.keys())
    res = compute_friedman_test(mase_matrix, symbols, DEFAULT_CROSS_COMPANY_METHODS)
    assert res.status == "INVALID_STATISTIC"
    assert res.kendalls_w is None
    assert "Mock material violation" in res.reason


# =========================================================================
# 19. Raw peso RMSE is not used for cross-company winner selection
# =========================================================================
def test_19_raw_peso_rmse_is_not_used_for_cross_company_winner_selection():
    mase_matrix = {
        f"SYM{i:02d}": {"lag_reg": 0.6, "arima": 0.8, "lstm": 1.0, "naive": 1.2}
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    # Ensure there is no overall_best_model or cross-company RMSE winner field
    assert not hasattr(summary, "overall_best_model")
    assert not hasattr(summary, "mean_rmse_winner")
    assert hasattr(summary, "lowest_median_mase_method")
    assert hasattr(summary, "best_mean_rank_method")
    assert hasattr(summary, "most_company_wins_method")


# =========================================================================
# 20. High-priced company cannot dominate aggregate ranking
# =========================================================================
def test_20_high_priced_company_cannot_dominate_aggregate_ranking():
    """Regression test proving high-priced stock cannot dominate aggregate rankings.

    Scenario:
    - SYM00 is an ultra-high-priced blue chip (Price ~ ₱2,000, daily change ~ ₱50).
      Model A has RMSE = 100.0 (MASE = 2.0).
      Model B has RMSE = 20.0 (MASE = 0.4) -> Model B wins SYM00 by 80 pesos!
    - SYM01..SYM14 (14 companies) are penny / low-priced stocks (Price ~ ₱5, daily change ~ ₱0.10).
      Model A has RMSE = 0.05 (MASE = 0.5) -> Model A wins 14 out of 15 stocks.
      Model B has RMSE = 0.50 (MASE = 5.0).

    Raw RMSE averaging:
      Mean RMSE(A) = (100.0 + 14 * 0.05) / 15 = 6.713 pesos
      Mean RMSE(B) = (20.0 + 14 * 0.50) / 15 = 1.800 pesos
      If raw RMSE was used, Model B would falsely be declared the winner!

    Scale-free MASE & rank aggregation:
      Model A achieves lower MASE in 14 of 15 companies.
      Model A mean rank = (2.0 + 14 * 1.0) / 15 = 1.067.
      Model B mean rank = (1.0 + 14 * 2.0) / 15 = 1.933.
      Model A median MASE = 0.50, Model B median MASE = 5.00.
      Model A is the scale-correct descriptive leader.
    """
    mase_matrix = {}
    mase_matrix["SYM00"] = {"model_a": 2.0, "model_b": 0.4, "lstm": 3.0, "naive": 1.0}
    for i in range(1, 15):
        mase_matrix[f"SYM{i:02d}"] = {"model_a": 0.5, "model_b": 5.0, "lstm": 3.0, "naive": 1.0}

    methods = ("model_a", "model_b", "lstm", "naive")
    summary = evaluate_cross_company_from_matrix(mase_matrix, methods=methods)

    # Model A must win on scale-free ranks and median MASE
    assert summary.method_stats["model_a"].mean_rank < summary.method_stats["model_b"].mean_rank
    assert summary.method_stats["model_a"].median_mase < summary.method_stats["model_b"].median_mase
    assert summary.method_stats["model_a"].count_ranked_first == 14
    assert summary.method_stats["model_b"].count_ranked_first == 1
    assert summary.best_mean_rank_method == "model_a"
    assert summary.lowest_median_mase_method == "model_a"
    assert summary.most_company_wins_method == "model_a"


# =========================================================================
# 21. Negative R² cannot influence aggregate ranking
# =========================================================================
def test_21_negative_r2_cannot_influence_aggregate_ranking():
    # Build exports with extreme negative R2 values for lag_reg
    exports_a = {
        f"SYM{i:02d}": _make_dummy_export(
            f"SYM{i:02d}",
            mase_values={"lag_reg": 0.6, "arima": 0.8, "lstm": 1.0, "naive": 1.2},
            r2_values={"lag_reg": -999.0, "arima": 0.5, "lstm": 0.5, "naive": 0.0},
        )
        for i in range(15)
    }
    exports_b = {
        f"SYM{i:02d}": _make_dummy_export(
            f"SYM{i:02d}",
            mase_values={"lag_reg": 0.6, "arima": 0.8, "lstm": 1.0, "naive": 1.2},
            r2_values={"lag_reg": 0.99, "arima": 0.5, "lstm": 0.5, "naive": 0.0},
        )
        for i in range(15)
    }
    summary_a = evaluate_cross_company(exports_a, strict_formal=False)
    summary_b = evaluate_cross_company(exports_b, strict_formal=False)

    # R2 difference must have zero impact on Friedman statistic, p-value, or rankings
    assert summary_a.friedman.statistic == pytest.approx(summary_b.friedman.statistic, abs=1e-9)
    assert summary_a.friedman.raw_p_value == pytest.approx(summary_b.friedman.raw_p_value, abs=1e-9)
    assert summary_a.best_mean_rank_method == summary_b.best_mean_rank_method


# =========================================================================
# 22. MASE < 1 count remains separate from direct Naive win count
# =========================================================================
def test_22_mase_below_one_count_remains_separate_from_direct_naive_win_count():
    exports = {
        f"SYM{i:02d}": _make_dummy_export(
            f"SYM{i:02d}",
            mase_values={"lag_reg": 0.8, "arima": 0.9, "lstm": 1.1, "naive": 0.95},
        )
        for i in range(15)
    }
    summary = evaluate_cross_company(exports, strict_formal=False)
    # For Naive: MASE is 0.95 < 1.0 on all 15 companies
    assert summary.method_stats["naive"].mase_below_one_count == 15
    # But Naive cannot have direct wins vs itself
    assert summary.method_stats["naive"].direct_rmse_wins_vs_naive is None


# =========================================================================
# 23. Significant pair direction agrees with paired MASE evidence
# =========================================================================
def test_23_significant_pair_direction_agrees_with_paired_mase_evidence():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.01 * i,
            "arima": 0.80 + 0.01 * i,
            "lstm": 1.10,
            "naive": 1.30,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    pair = next(p for p in summary.pairwise_wilcoxon if p.model_a == "lag_reg" and p.model_b == "arima")
    assert pair.significant_holm is True
    assert pair.lower_median_method == "lag_reg"
    assert pair.status == "A_SIGNIFICANTLY_LOWER_MASE"


# =========================================================================
# 24. Method ties remain ties statistically
# =========================================================================
def test_24_method_ties_remain_ties_statistically():
    mase_matrix = {
        f"SYM{i:02d}": {
            "lag_reg": 0.70,
            "arima": 0.70,  # identical to lag_reg
            "lstm": 1.00,
            "naive": 1.20,
        }
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    pair = next(p for p in summary.pairwise_wilcoxon if p.model_a == "lag_reg" and p.model_b == "arima")
    assert pair.n_nonzero_differences == 0
    assert pair.significant_holm is False
    assert pair.status == "IDENTICAL_DATA"
    assert pair.raw_p_value == 1.0


# =========================================================================
# 25. Results invariant to company input ordering
# =========================================================================
def test_25_results_invariant_to_company_input_ordering():
    mase_matrix_1 = {
        f"SYM{i:02d}": {
            "lag_reg": 0.50 + 0.02 * i,
            "arima": 0.65 + 0.01 * i,
            "lstm": 0.85 + 0.03 * (i % 2),
            "naive": 1.05,
        }
        for i in range(15)
    }
    # Reverse dictionary order
    mase_matrix_2 = {k: mase_matrix_1[k] for k in reversed(list(mase_matrix_1.keys()))}

    summary_1 = evaluate_cross_company_from_matrix(mase_matrix_1)
    summary_2 = evaluate_cross_company_from_matrix(mase_matrix_2)

    assert summary_1.friedman.statistic == pytest.approx(summary_2.friedman.statistic, abs=1e-9)
    assert summary_1.friedman.raw_p_value == pytest.approx(summary_2.friedman.raw_p_value, abs=1e-9)
    assert summary_1.friedman.kendalls_w == pytest.approx(summary_2.friedman.kendalls_w, abs=1e-9)
    for p1, p2 in zip(summary_1.pairwise_wilcoxon, summary_2.pairwise_wilcoxon):
        assert p1.raw_p_value == pytest.approx(p2.raw_p_value, abs=1e-9)
        assert p1.holm_adjusted_p_value == pytest.approx(p2.holm_adjusted_p_value, abs=1e-9)


# =========================================================================
# 26. DEMO provenance remains labeled DEMO
# =========================================================================
def test_26_demo_provenance_remains_labeled_demo():
    exports = {
        f"SYM{i:02d}": _make_dummy_export(f"SYM{i:02d}", tier=ProvenanceTier.DEMO)
        for i in range(15)
    }
    summary = evaluate_cross_company(exports, strict_formal=False)
    assert summary.provenance_tier == "DEMO"
    assert summary.is_demo is True


# =========================================================================
# 27. Cross-company output explicitly identifies analysis as supporting evidence
# =========================================================================
def test_27_cross_company_output_explicitly_identifies_as_supporting_evidence():
    exports = {
        f"SYM{i:02d}": _make_dummy_export(f"SYM{i:02d}")
        for i in range(15)
    }
    summary = evaluate_cross_company(exports, strict_formal=False)
    assert SUPPORTING_EVIDENCE_NOTE in summary.note
    assert "supporting cross-company evidence" in summary.note
    assert "not be treated as fully independent experimental units" in summary.note


# =========================================================================
# Additional test: DataFrame export formats
# =========================================================================
def test_dataframe_export_formats():
    mase_matrix = {
        f"SYM{i:02d}": {"lag_reg": 0.6, "arima": 0.8, "lstm": 1.0, "naive": 1.2}
        for i in range(15)
    }
    summary = evaluate_cross_company_from_matrix(mase_matrix)
    rank_df = rank_matrix_to_dataframe(summary)
    assert rank_df.shape == (15, 9)
    assert "lag_reg_mase" in rank_df.columns
    assert "lag_reg_rank" in rank_df.columns

    stats_df = method_stats_to_dataframe(summary)
    assert stats_df.shape == (4, 15)

    w_df = wilcoxon_results_to_dataframe(summary.pairwise_wilcoxon)
    assert w_df.shape == (6, 24)
    expected_cols = [
        "pair",
        "model_a",
        "model_b",
        "metric",
        "n_companies",
        "n_pairs",
        "n_nonzero_differences",
        "has_zero_differences",
        "has_tied_absolute_differences",
        "median_mase_a",
        "median_mase_b",
        "median_paired_difference",
        "wilcoxon_statistic",
        "raw_p_value",
        "holm_adjusted_p_value",
        "significant_holm",
        "lower_median_method",
        "status",
        "reason",
        "requested_method_policy",
        "actual_method_used",
        "method_used",
        "zero_method",
        "alternative",
    ]
    for col in expected_cols:
        assert col in w_df.columns
