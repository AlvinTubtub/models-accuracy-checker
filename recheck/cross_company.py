"""Cross-company statistical testing and scale-correct aggregation.

Evaluates matched company evaluations using independently recomputed
scale-free MASE values.

Key invariants:
- Scale-independent: Primary cross-company metric is independently recomputed MASE.
  No raw-peso RMSE or MAE averaging is permitted to select an overall winner.
- Non-independence acknowledgement: Companies share Philippine market conditions;
  this analysis is supporting cross-company evidence, not independent replications.
- Within-company ranking: Ranks 1 to 4 using scipy.stats.rankdata(..., method="average").
  Lower MASE is better. Statistical rank ties remain ties.
- Friedman omnibus test: scipy.stats.friedmanchisquare (df = 3) with Kendall's W.
- Conditional Wilcoxon post-hoc: GATED strictly by Friedman significance.
  If Friedman is not significant, post-hoc tests are NOT run (posthoc_status = NOT_RUN_FRIEDMAN_NOT_SIGNIFICANT).
- Predeclared Wilcoxon policy: two-sided alternative, zero_method="wilcox", method="auto".
- Multiplicity control: Holm step-down adjustment across exactly the 6 pairwise tests (family size m = 6).
- Strict separation: MASE < 1.0 scaling reference counts are never conflated with direct Naive holdout wins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.stats

from recheck.metrics import MetricComputationError, compute_metrics
from recheck.oos import ComparisonStatus, evaluate_company_direct_oos
from recheck.schema import (
    PRINCIPAL_MODELS,
    REQUIRED_MODELS,
    CompanyEvaluationExport,
    ProvenanceTier,
)
from recheck.tolerances import NUMERICAL_BOUNDARY_TOLERANCE, TIE_TOLERANCE

DEFAULT_CROSS_COMPANY_METHODS: tuple[str, ...] = ("lag_reg", "arima", "lstm", "naive")
EXPECTED_FORMAL_COMPANY_COUNT: int = 15
REQUESTED_WILCOXON_POLICY: str = "exact_if_no_zeros_or_ties_else_approx"

PAIRWISE_METHOD_PAIRS: tuple[tuple[str, str], ...] = (
    ("lag_reg", "arima"),
    ("lag_reg", "lstm"),
    ("lag_reg", "naive"),
    ("arima", "lstm"),
    ("arima", "naive"),
    ("lstm", "naive"),
)

SUPPORTING_EVIDENCE_NOTE: str = (
    "Companies share common Philippine market conditions and should not be "
    "treated as fully independent experimental units; this analysis is supporting "
    "cross-company evidence."
)

METHOD_LABELS: dict[str, str] = {
    "lag_reg": "Lag-Informed Regression",
    "arima": "ARIMA",
    "lstm": "LSTM",
    "naive": "Naive",
}


class CrossCompanyVerificationError(ValueError):
    """Raised when formal cross-company requirements or invariants are violated."""


@dataclass(frozen=True)
class FriedmanResult:
    metric: str = "mase"
    company_count: int = 0
    method_count: int = 4
    methods: list[str] = field(default_factory=lambda: list(DEFAULT_CROSS_COMPANY_METHODS))
    statistic: float | None = None
    degrees_of_freedom: int = 3
    raw_p_value: float | None = None
    alpha: float = 0.05
    significant: bool = False
    status: str = "INITIALIZED"
    reason: str = ""
    kendalls_w: float | None = None


@dataclass(frozen=True)
class PairwiseWilcoxonResult:
    metric: str = "mase"
    model_a: str = ""
    model_b: str = ""
    n_companies: int = 0
    n_nonzero_differences: int = 0
    median_mase_a: float = 0.0
    median_mase_b: float = 0.0
    median_paired_difference: float = 0.0
    wilcoxon_statistic: float | None = None
    raw_p_value: float | None = None
    holm_adjusted_p_value: float | None = None
    significant_holm: bool = False
    holm_family_size: int = 6
    requested_method_policy: str = REQUESTED_WILCOXON_POLICY
    actual_method_used: str = "approx"
    method_used: str = "approx"
    zero_method: str = "wilcox"
    alternative: str = "two-sided"
    has_zero_differences: bool = False
    has_tied_absolute_differences: bool = False
    lower_median_method: str | None = None
    status: str = "INITIALIZED"
    reason: str = ""

    @property
    def n_pairs(self) -> int:
        return self.n_companies


@dataclass(frozen=True)
class MethodDescriptiveStats:
    method: str
    method_label: str
    companies_evaluated: int
    median_mase: float
    mean_rank: float
    median_rank: float
    count_ranked_first: int
    count_tied_first: int
    mase_below_one_count: int
    direct_rmse_wins_vs_naive: int | None = None
    direct_rmse_ties_vs_naive: int | None = None
    direct_rmse_losses_vs_naive: int | None = None
    direct_mae_wins_vs_naive: int | None = None
    direct_mae_ties_vs_naive: int | None = None
    direct_mae_losses_vs_naive: int | None = None


@dataclass(frozen=True)
class CrossCompanySummary:
    provenance_tier: str
    company_count: int
    symbols: list[str]
    methods: list[str]
    friedman: FriedmanResult
    posthoc_status: str
    pairwise_wilcoxon: list[PairwiseWilcoxonResult]
    method_stats: dict[str, MethodDescriptiveStats]
    lowest_median_mase_method: str | None
    best_mean_rank_method: str | None
    most_company_wins_method: str | None
    rank_matrix: dict[str, dict[str, float]]
    mase_matrix: dict[str, dict[str, float]]
    note: str = SUPPORTING_EVIDENCE_NOTE
    is_demo: bool = False


def compute_mase(
    actual_closes: Sequence[float],
    predicted_closes: Sequence[float],
    mase_denominator: float,
) -> float:
    """Compute recomputed MASE from actual, predicted, and development denominator."""
    return compute_metrics(
        actual_closes=actual_closes,
        predicted_closes=predicted_closes,
        mase_denominator=mase_denominator,
    ).mase


def build_mase_matrix(
    exports: dict[str, CompanyEvaluationExport],
    methods: Sequence[str] = DEFAULT_CROSS_COMPANY_METHODS,
    strict_formal: bool = True,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """Recompute and validate the matched company x method MASE matrix.

    Requirements:
    - Expected company set: sorted deterministically.
    - All specified methods must exist for each company.
    - MASE values must be finite.
    - For FORMAL_FROZEN: requires complete EXPECTED_FORMAL_COMPANY_COUNT (15) companies.
    """
    sorted_symbols = sorted(exports.keys())
    is_formal = any(e.provenance_tier == ProvenanceTier.FORMAL_FROZEN for e in exports.values())

    if is_formal and strict_formal:
        if len(sorted_symbols) < EXPECTED_FORMAL_COMPANY_COUNT:
            raise CrossCompanyVerificationError(
                f"FORMAL_FROZEN experiment requires {EXPECTED_FORMAL_COMPANY_COUNT} matched companies, "
                f"found {len(sorted_symbols)}."
            )

    mase_matrix: dict[str, dict[str, float]] = {}

    for sym in sorted_symbols:
        export = exports[sym]
        mase_matrix[sym] = {}
        actual = np.asarray(export.actual_closes, dtype=float)

        for m in methods:
            if m not in export.predicted_closes_by_model:
                msg = f"Company '{sym}' is missing required evaluation series for method '{m}'."
                raise CrossCompanyVerificationError(msg)

            pred = np.asarray(export.predicted_closes_by_model[m], dtype=float)
            if len(pred) != len(actual):
                raise CrossCompanyVerificationError(
                    f"Length mismatch for '{sym}' method '{m}': {len(pred)} != {len(actual)}"
                )

            mase_val = compute_mase(actual, pred, export.mase_denominator)
            if not math.isfinite(mase_val):
                raise CrossCompanyVerificationError(
                    f"Non-finite recomputed MASE ({mase_val}) for company '{sym}', method '{m}'."
                )
            mase_matrix[sym][m] = float(mase_val)

    return sorted_symbols, mase_matrix


def compute_within_company_ranks(
    mase_matrix: dict[str, dict[str, float]],
    symbols: Sequence[str],
    methods: Sequence[str] = DEFAULT_CROSS_COMPANY_METHODS,
) -> dict[str, dict[str, float]]:
    """Compute within-company tied average ranks from recomputed MASE.

    Lower MASE = better rank (rank 1 is best).
    Uses scipy.stats.rankdata(..., method='average').
    Statistical ties remain ties.
    """
    rank_matrix: dict[str, dict[str, float]] = {}
    for sym in symbols:
        mases = [mase_matrix[sym][m] for m in methods]
        ranks = scipy.stats.rankdata(mases, method="average")
        rank_matrix[sym] = {m: float(r) for m, r in zip(methods, ranks)}
    return rank_matrix


def normalize_kendalls_w(
    w_raw: float,
    eps: float = NUMERICAL_BOUNDARY_TOLERANCE,
) -> float:
    """Validate and adjust Kendall's W against numerical boundary noise.

    - If -eps <= W < 0.0: returns 0.0
    - If 1.0 < W <= 1.0 + eps: returns 1.0
    - If 0.0 <= W <= 1.0: returns float(W)
    - If W < -eps or W > 1.0 + eps: raises CrossCompanyVerificationError (materially invalid)
    """
    if not math.isfinite(w_raw):
        raise CrossCompanyVerificationError(f"Kendall's W is non-finite: {w_raw}")
    if -eps <= w_raw < 0.0:
        return 0.0
    if 1.0 < w_raw <= 1.0 + eps:
        return 1.0
    if 0.0 <= w_raw <= 1.0:
        return float(w_raw)
    raise CrossCompanyVerificationError(
        f"Kendall's W ({w_raw}) is materially outside theoretical bounds [0.0, 1.0]."
    )


def determine_wilcoxon_method(
    x: np.ndarray,
    y: np.ndarray,
    tolerance: float = TIE_TOLERANCE,
) -> tuple[str, bool, bool, int]:
    """Deterministically determine the exact Wilcoxon method to execute.

    Policy:
    - Compute paired differences d = x - y.
    - If all differences are zero within tolerance: returns ("none", True, False, 0).
    - Checks whether zero differences exist (|d| <= tolerance).
    - Checks whether tied absolute differences exist in the nonzero differences.
    - If there are no zero differences, no tied absolute differences, and n_nonzero <= 50:
          actual_method_used = "exact"
      otherwise:
          actual_method_used = "approx"

    Returns:
        (actual_method_used, has_zero_differences, has_tied_absolute_differences, n_nonzero)
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    is_zero = np.abs(d) <= tolerance
    has_zero_diffs = bool(np.any(is_zero))
    n_nonzero = int(np.sum(~is_zero))

    if n_nonzero == 0:
        return ("none", True, False, 0)

    d_nonzero = d[~is_zero]
    abs_d = np.abs(d_nonzero)
    if len(abs_d) > 1:
        sorted_abs = np.sort(abs_d)
        has_ties = bool(np.any(np.diff(sorted_abs) <= tolerance))
    else:
        has_ties = False

    if (not has_zero_diffs) and (not has_ties) and (n_nonzero <= 50):
        actual_method = "exact"
    else:
        actual_method = "approx"

    return (actual_method, has_zero_diffs, has_ties, n_nonzero)


def compute_friedman_test(
    mase_matrix: dict[str, dict[str, float]],
    symbols: Sequence[str],
    methods: Sequence[str] = DEFAULT_CROSS_COMPANY_METHODS,
    alpha: float = 0.05,
) -> FriedmanResult:
    """Run Friedman omnibus test across methods with Kendall's W effect size.

    df = methods - 1
    Kendall's W = chi_square / (N * (K - 1))
    """
    n_companies = len(symbols)
    k_methods = len(methods)

    if n_companies < 2:
        return FriedmanResult(
            metric="mase",
            company_count=n_companies,
            method_count=k_methods,
            methods=list(methods),
            statistic=None,
            degrees_of_freedom=max(1, k_methods - 1),
            raw_p_value=None,
            alpha=alpha,
            significant=False,
            status="INSUFFICIENT_COMPANIES",
            reason="At least 2 matched companies are required for Friedman omnibus test.",
            kendalls_w=None,
        )

    if k_methods < 2:
        return FriedmanResult(
            metric="mase",
            company_count=n_companies,
            method_count=k_methods,
            methods=list(methods),
            statistic=None,
            degrees_of_freedom=0,
            raw_p_value=None,
            alpha=alpha,
            significant=False,
            status="FEWER_THAN_REQUIRED_METHODS",
            reason="At least 2 methods are required for Friedman omnibus test.",
            kendalls_w=None,
        )

    # Check for non-finite values
    for s in symbols:
        for m in methods:
            val = mase_matrix.get(s, {}).get(m)
            if val is None or not math.isfinite(val):
                return FriedmanResult(
                    metric="mase",
                    company_count=n_companies,
                    method_count=k_methods,
                    methods=list(methods),
                    statistic=None,
                    degrees_of_freedom=k_methods - 1,
                    raw_p_value=None,
                    alpha=alpha,
                    significant=False,
                    status="NON_FINITE_VALUES",
                    reason="Non-finite MASE value detected in company evaluations.",
                    kendalls_w=None,
                )

    # Prepare method columns across subjects
    cols = [np.asarray([mase_matrix[s][m] for s in symbols], dtype=float) for m in methods]

    # Check if all methods have identical values within every company
    all_identical = True
    for s in symbols:
        row_vals = [mase_matrix[s][m] for m in methods]
        if max(row_vals) - min(row_vals) > TIE_TOLERANCE:
            all_identical = False
            break

    if all_identical:
        return FriedmanResult(
            metric="mase",
            company_count=n_companies,
            method_count=k_methods,
            methods=list(methods),
            statistic=0.0,
            degrees_of_freedom=k_methods - 1,
            raw_p_value=1.0,
            alpha=alpha,
            significant=False,
            status="IDENTICAL_RANKS",
            reason="All evaluated methods have identical values across all companies.",
            kendalls_w=0.0,
        )

    # Run SciPy Friedman test
    try:
        res = scipy.stats.friedmanchisquare(*cols)
        stat = float(res.statistic)
        raw_p = float(res.pvalue)

        if math.isnan(stat) or math.isnan(raw_p):
            return FriedmanResult(
                metric="mase",
                company_count=n_companies,
                method_count=k_methods,
                methods=list(methods),
                statistic=0.0,
                degrees_of_freedom=k_methods - 1,
                raw_p_value=1.0,
                alpha=alpha,
                significant=False,
                status="IDENTICAL_RANKS",
                reason="Rank sums have zero variance across evaluated methods.",
                kendalls_w=0.0,
            )

        # Kendall's W = chi^2 / (N * (K - 1))
        denom = float(n_companies * (k_methods - 1))
        w = stat / denom if denom > 0 else 0.0

        try:
            w_norm = normalize_kendalls_w(w, eps=NUMERICAL_BOUNDARY_TOLERANCE)
        except CrossCompanyVerificationError as err:
            return FriedmanResult(
                metric="mase",
                company_count=n_companies,
                method_count=k_methods,
                methods=list(methods),
                statistic=stat,
                degrees_of_freedom=k_methods - 1,
                raw_p_value=raw_p,
                alpha=alpha,
                significant=False,
                status="INVALID_STATISTIC",
                reason=str(err),
                kendalls_w=None,
            )

        sig = bool(raw_p < alpha)
        status = "SIGNIFICANT" if sig else "NOT_SIGNIFICANT"
        reason = (
            "Friedman omnibus test detected significant differences among methods."
            if sig
            else "No statistically detectable difference among methods under Friedman test."
        )

        return FriedmanResult(
            metric="mase",
            company_count=n_companies,
            method_count=k_methods,
            methods=list(methods),
            statistic=stat,
            degrees_of_freedom=k_methods - 1,
            raw_p_value=raw_p,
            alpha=alpha,
            significant=sig,
            status=status,
            reason=reason,
            kendalls_w=w_norm,
        )

    except Exception as ex:
        return FriedmanResult(
            metric="mase",
            company_count=n_companies,
            method_count=k_methods,
            methods=list(methods),
            statistic=None,
            degrees_of_freedom=k_methods - 1,
            raw_p_value=None,
            alpha=alpha,
            significant=False,
            status="ERROR",
            reason=f"Friedman computation error: {ex}",
            kendalls_w=None,
        )


def compute_pairwise_wilcoxon(
    mase_matrix: dict[str, dict[str, float]],
    symbols: Sequence[str],
    pairs: Sequence[tuple[str, str]] = PAIRWISE_METHOD_PAIRS,
    alpha: float = 0.05,
) -> list[PairwiseWilcoxonResult]:
    """Execute predeclared post-hoc Wilcoxon signed-rank tests with Holm correction.

    Policy:
    - alternative = "two-sided"
    - zero_method = "wilcox"
    - deterministic method selection:
      * "exact" if no zero differences, no tied absolute differences, and n_nonzero <= 50.
      * "approx" otherwise.
      * "none" if all differences are zero.
    - Holm step-down applied across exactly the declared pairwise family (m = 6).
    """
    raw_results: list[PairwiseWilcoxonResult] = []

    for m_a, m_b in pairs:
        x = np.asarray([mase_matrix[s][m_a] for s in symbols], dtype=float)
        y = np.asarray([mase_matrix[s][m_b] for s in symbols], dtype=float)
        d = x - y

        n_comp = len(x)
        med_a = float(np.median(x))
        med_b = float(np.median(y))
        med_diff = float(np.median(d))

        if med_a < med_b - TIE_TOLERANCE:
            lower_med = m_a
        elif med_b < med_a - TIE_TOLERANCE:
            lower_med = m_b
        else:
            lower_med = None

        actual_method, has_zero_diffs, has_ties, n_nonzero = determine_wilcoxon_method(
            x, y, tolerance=TIE_TOLERANCE
        )

        if actual_method == "none":
            raw_results.append(
                PairwiseWilcoxonResult(
                    metric="mase",
                    model_a=m_a,
                    model_b=m_b,
                    n_companies=n_comp,
                    n_nonzero_differences=0,
                    median_mase_a=med_a,
                    median_mase_b=med_b,
                    median_paired_difference=med_diff,
                    wilcoxon_statistic=0.0,
                    raw_p_value=1.0,
                    holm_adjusted_p_value=1.0,
                    significant_holm=False,
                    holm_family_size=len(pairs),
                    requested_method_policy=REQUESTED_WILCOXON_POLICY,
                    actual_method_used="none",
                    method_used="none",
                    zero_method="wilcox",
                    alternative="two-sided",
                    has_zero_differences=True,
                    has_tied_absolute_differences=False,
                    lower_median_method=lower_med,
                    status="IDENTICAL_DATA",
                    reason="All paired differences are zero within tolerance.",
                )
            )
            continue

        try:
            res = scipy.stats.wilcoxon(
                x,
                y,
                zero_method="wilcox",
                alternative="two-sided",
                method=actual_method,
            )
            stat = float(res.statistic)
            raw_p = float(res.pvalue)

            raw_results.append(
                PairwiseWilcoxonResult(
                    metric="mase",
                    model_a=m_a,
                    model_b=m_b,
                    n_companies=n_comp,
                    n_nonzero_differences=n_nonzero,
                    median_mase_a=med_a,
                    median_mase_b=med_b,
                    median_paired_difference=med_diff,
                    wilcoxon_statistic=stat,
                    raw_p_value=raw_p,
                    holm_adjusted_p_value=None,  # to be filled by Holm step-down
                    significant_holm=False,
                    holm_family_size=len(pairs),
                    requested_method_policy=REQUESTED_WILCOXON_POLICY,
                    actual_method_used=actual_method,
                    method_used=actual_method,
                    zero_method="wilcox",
                    alternative="two-sided",
                    has_zero_differences=has_zero_diffs,
                    has_tied_absolute_differences=has_ties,
                    lower_median_method=lower_med,
                    status="TESTED",
                    reason="",
                )
            )
        except Exception as ex:
            raw_results.append(
                PairwiseWilcoxonResult(
                    metric="mase",
                    model_a=m_a,
                    model_b=m_b,
                    n_companies=n_comp,
                    n_nonzero_differences=n_nonzero,
                    median_mase_a=med_a,
                    median_mase_b=med_b,
                    median_paired_difference=med_diff,
                    wilcoxon_statistic=None,
                    raw_p_value=None,
                    holm_adjusted_p_value=None,
                    significant_holm=False,
                    holm_family_size=len(pairs),
                    requested_method_policy=REQUESTED_WILCOXON_POLICY,
                    actual_method_used=actual_method,
                    method_used=actual_method,
                    zero_method="wilcox",
                    alternative="two-sided",
                    has_zero_differences=has_zero_diffs,
                    has_tied_absolute_differences=has_ties,
                    lower_median_method=lower_med,
                    status="ERROR",
                    reason=f"Wilcoxon computation failed ({actual_method}): {ex}",
                )
            )

    # Apply Holm step-down across the 6 tests
    m = len(raw_results)
    valid_indices = [i for i, r in enumerate(raw_results) if r.raw_p_value is not None]
    sorted_indices = sorted(valid_indices, key=lambda i: raw_results[i].raw_p_value)

    adjusted_p: dict[int, float | None] = {i: None for i in range(m)}
    running_max = 0.0

    for rank, orig_idx in enumerate(sorted_indices):
        k = rank + 1
        multiplier = m - k + 1
        raw_p = raw_results[orig_idx].raw_p_value
        cand_p = min(1.0, multiplier * raw_p)
        running_max = max(running_max, cand_p)
        adj = min(1.0, max(running_max, raw_p))
        adjusted_p[orig_idx] = adj

    # Build final adjusted results
    final_results: list[PairwiseWilcoxonResult] = []
    for i, res in enumerate(raw_results):
        adj_p = adjusted_p[i]
        sig_holm = (adj_p is not None and adj_p < alpha) and res.status not in (
            "IDENTICAL_DATA",
            "ERROR",
        )

        if res.status == "IDENTICAL_DATA":
            status = "IDENTICAL_DATA"
            reason = res.reason
        elif res.status == "ERROR":
            status = "ERROR"
            reason = res.reason
        elif sig_holm:
            if res.lower_median_method == res.model_a:
                status = "A_SIGNIFICANTLY_LOWER_MASE"
                reason = f"{METHOD_LABELS.get(res.model_a, res.model_a)} had significantly lower company-level MASE than {METHOD_LABELS.get(res.model_b, res.model_b)} after Holm adjustment."
            elif res.lower_median_method == res.model_b:
                status = "B_SIGNIFICANTLY_LOWER_MASE"
                reason = f"{METHOD_LABELS.get(res.model_b, res.model_b)} had significantly lower company-level MASE than {METHOD_LABELS.get(res.model_a, res.model_a)} after Holm adjustment."
            else:
                status = "SIGNIFICANT_HOLM_EQUAL_MEDIANS"
                reason = "Significant difference detected after Holm correction with equal medians."
        else:
            status = "NOT_SIGNIFICANT"
            reason = "No statistically detectable difference between methods after Holm adjustment."

        final_results.append(
            PairwiseWilcoxonResult(
                metric=res.metric,
                model_a=res.model_a,
                model_b=res.model_b,
                n_companies=res.n_companies,
                n_nonzero_differences=res.n_nonzero_differences,
                median_mase_a=res.median_mase_a,
                median_mase_b=res.median_mase_b,
                median_paired_difference=res.median_paired_difference,
                wilcoxon_statistic=res.wilcoxon_statistic,
                raw_p_value=res.raw_p_value,
                holm_adjusted_p_value=adj_p,
                significant_holm=sig_holm,
                holm_family_size=res.holm_family_size,
                requested_method_policy=res.requested_method_policy,
                actual_method_used=res.actual_method_used,
                method_used=res.method_used,
                zero_method=res.zero_method,
                alternative=res.alternative,
                has_zero_differences=res.has_zero_differences,
                has_tied_absolute_differences=res.has_tied_absolute_differences,
                lower_median_method=res.lower_median_method,
                status=status,
                reason=reason,
            )
        )

    return final_results


def build_descriptive_summary(
    mase_matrix: dict[str, dict[str, float]],
    rank_matrix: dict[str, dict[str, float]],
    symbols: Sequence[str],
    methods: Sequence[str] = DEFAULT_CROSS_COMPANY_METHODS,
    direct_oos_stats: dict[str, dict[str, int]] | None = None,
) -> dict[str, MethodDescriptiveStats]:
    """Compute descriptive statistics per method: median MASE, mean/median ranks,
    win counts, and direct holdout Naive wins/losses.
    """
    stats_by_method: dict[str, MethodDescriptiveStats] = {}

    for m in methods:
        mases = [mase_matrix[s][m] for s in symbols]
        ranks = [rank_matrix[s][m] for s in symbols]

        med_mase = float(np.median(mases)) if mases else 0.0
        mean_r = float(np.mean(ranks)) if ranks else 0.0
        med_r = float(np.median(ranks)) if ranks else 0.0

        ranked_first = 0
        tied_first = 0
        mase_lt_one = 0

        for s in symbols:
            r = rank_matrix[s][m]
            min_r = min(rank_matrix[s].values())
            # Check if this method achieved the best rank in this company
            if abs(r - min_r) <= TIE_TOLERANCE:
                # Check if multiple methods tied for best rank
                methods_at_min = sum(
                    1 for other_m in methods if abs(rank_matrix[s][other_m] - min_r) <= TIE_TOLERANCE
                )
                if methods_at_min == 1:
                    ranked_first += 1
                else:
                    tied_first += 1

            if mase_matrix[s][m] < 1.0 - TIE_TOLERANCE:
                mase_lt_one += 1

        oos = direct_oos_stats.get(m, {}) if direct_oos_stats else {}

        stats_by_method[m] = MethodDescriptiveStats(
            method=m,
            method_label=METHOD_LABELS.get(m, m),
            companies_evaluated=len(symbols),
            median_mase=med_mase,
            mean_rank=mean_r,
            median_rank=med_r,
            count_ranked_first=ranked_first,
            count_tied_first=tied_first,
            mase_below_one_count=mase_lt_one,
            direct_rmse_wins_vs_naive=oos.get("rmse_wins"),
            direct_rmse_ties_vs_naive=oos.get("rmse_ties"),
            direct_rmse_losses_vs_naive=oos.get("rmse_losses"),
            direct_mae_wins_vs_naive=oos.get("mae_wins"),
            direct_mae_ties_vs_naive=oos.get("mae_ties"),
            direct_mae_losses_vs_naive=oos.get("mae_losses"),
        )

    return stats_by_method


def evaluate_cross_company(
    exports: dict[str, CompanyEvaluationExport],
    *,
    alpha: float = 0.05,
    strict_formal: bool = True,
) -> CrossCompanySummary:
    """Primary entry point for cross-company statistical evaluation from exports."""
    # 1. Build & validate MASE matrix
    symbols, mase_matrix = build_mase_matrix(
        exports, methods=DEFAULT_CROSS_COMPANY_METHODS, strict_formal=strict_formal
    )

    # 2. Determine provenance
    is_formal = any(e.provenance_tier == ProvenanceTier.FORMAL_FROZEN for e in exports.values())
    is_demo = any(e.provenance_tier == ProvenanceTier.DEMO for e in exports.values())
    tier_str = "FORMAL_FROZEN" if is_formal else ("DEMO" if is_demo else "HYBRID")

    # 3. Direct OOS holdout wins vs Naive
    direct_oos_stats: dict[str, dict[str, int]] = {}
    for p_model in PRINCIPAL_MODELS:
        direct_oos_stats[p_model] = {
            "rmse_wins": 0,
            "rmse_ties": 0,
            "rmse_losses": 0,
            "mae_wins": 0,
            "mae_ties": 0,
            "mae_losses": 0,
        }

    for exp in exports.values():
        try:
            oos_summary = evaluate_company_direct_oos(exp)
        except Exception:
            oos_summary = None

        if oos_summary is not None:
            for p_model in PRINCIPAL_MODELS:
                comp = oos_summary.models.get(p_model)
                if comp is not None:
                    if comp.rmse_comparison_status == ComparisonStatus.WIN:
                        direct_oos_stats[p_model]["rmse_wins"] += 1
                    elif comp.rmse_comparison_status == ComparisonStatus.TIE:
                        direct_oos_stats[p_model]["rmse_ties"] += 1
                    else:
                        direct_oos_stats[p_model]["rmse_losses"] += 1

                    if comp.mae_comparison_status == ComparisonStatus.WIN:
                        direct_oos_stats[p_model]["mae_wins"] += 1
                    elif comp.mae_comparison_status == ComparisonStatus.TIE:
                        direct_oos_stats[p_model]["mae_ties"] += 1
                    else:
                        direct_oos_stats[p_model]["mae_losses"] += 1
        else:
            # Fallback when independent raw data series is not provided at cross-company level
            actual = np.asarray(exp.actual_closes, dtype=float)
            naive_pred = np.asarray(exp.predicted_closes_by_model.get("naive", []), dtype=float)
            if len(naive_pred) == len(actual) and len(actual) > 0:
                naive_rmse = math.sqrt(float(np.mean((naive_pred - actual) ** 2)))
                naive_mae = float(np.mean(np.abs(naive_pred - actual)))

                for p_model in PRINCIPAL_MODELS:
                    if p_model in exp.predicted_closes_by_model:
                        m_pred = np.asarray(exp.predicted_closes_by_model[p_model], dtype=float)
                        if len(m_pred) == len(actual):
                            m_rmse = math.sqrt(float(np.mean((m_pred - actual) ** 2)))
                            m_mae = float(np.mean(np.abs(m_pred - actual)))

                            if m_rmse < naive_rmse - TIE_TOLERANCE:
                                direct_oos_stats[p_model]["rmse_wins"] += 1
                            elif abs(m_rmse - naive_rmse) <= TIE_TOLERANCE:
                                direct_oos_stats[p_model]["rmse_ties"] += 1
                            else:
                                direct_oos_stats[p_model]["rmse_losses"] += 1

                            if m_mae < naive_mae - TIE_TOLERANCE:
                                direct_oos_stats[p_model]["mae_wins"] += 1
                            elif abs(m_mae - naive_mae) <= TIE_TOLERANCE:
                                direct_oos_stats[p_model]["mae_ties"] += 1
                            else:
                                direct_oos_stats[p_model]["mae_losses"] += 1

    return evaluate_cross_company_from_matrix(
        mase_matrix=mase_matrix,
        symbols=symbols,
        methods=DEFAULT_CROSS_COMPANY_METHODS,
        provenance_tier=tier_str,
        alpha=alpha,
        direct_oos_stats=direct_oos_stats,
        is_demo=is_demo,
    )


def evaluate_cross_company_from_matrix(
    mase_matrix: dict[str, dict[str, float]],
    symbols: Sequence[str] | None = None,
    methods: Sequence[str] = DEFAULT_CROSS_COMPANY_METHODS,
    provenance_tier: str = "DEMO",
    alpha: float = 0.05,
    direct_oos_stats: dict[str, dict[str, int]] | None = None,
    is_demo: bool = False,
) -> CrossCompanySummary:
    """Execute cross-company evaluation directly from an existing MASE matrix."""
    if symbols is None:
        sorted_symbols = sorted(mase_matrix.keys())
    else:
        # Guarantee deterministic symbol ordering
        sorted_symbols = sorted(symbols)

    # 1. Within-company ranks
    rank_matrix = compute_within_company_ranks(mase_matrix, sorted_symbols, methods)

    # 2. Friedman omnibus test
    friedman_res = compute_friedman_test(mase_matrix, sorted_symbols, methods, alpha=alpha)

    # 3. Conditional post-hoc Wilcoxon
    if friedman_res.significant:
        # Only test pairs where both models are in methods
        active_pairs = [p for p in PAIRWISE_METHOD_PAIRS if p[0] in methods and p[1] in methods]
        pairwise_wilcoxon = compute_pairwise_wilcoxon(
            mase_matrix, sorted_symbols, active_pairs, alpha=alpha
        )
        posthoc_status = "RUN"
    else:
        pairwise_wilcoxon = []
        posthoc_status = "NOT_RUN_FRIEDMAN_NOT_SIGNIFICANT"

    # 4. Descriptive summary
    method_stats = build_descriptive_summary(
        mase_matrix, rank_matrix, sorted_symbols, methods, direct_oos_stats
    )

    # 5. Descriptive winner labels (strictly non-universal, descriptive only)
    lowest_med_method = None
    best_mean_rank_method = None
    most_wins_method = None

    if method_stats:
        # Lowest median MASE
        sorted_by_med_mase = sorted(method_stats.values(), key=lambda s: s.median_mase)
        if len(sorted_by_med_mase) > 1 and abs(
            sorted_by_med_mase[0].median_mase - sorted_by_med_mase[1].median_mase
        ) <= TIE_TOLERANCE:
            lowest_med_method = None  # tied
        else:
            lowest_med_method = sorted_by_med_mase[0].method

        # Best mean rank (lowest rank is best)
        sorted_by_mean_rank = sorted(method_stats.values(), key=lambda s: s.mean_rank)
        if len(sorted_by_mean_rank) > 1 and abs(
            sorted_by_mean_rank[0].mean_rank - sorted_by_mean_rank[1].mean_rank
        ) <= TIE_TOLERANCE:
            best_mean_rank_method = None  # tied
        else:
            best_mean_rank_method = sorted_by_mean_rank[0].method

        # Most company wins
        sorted_by_wins = sorted(method_stats.values(), key=lambda s: s.count_ranked_first, reverse=True)
        if len(sorted_by_wins) > 1 and sorted_by_wins[0].count_ranked_first == sorted_by_wins[1].count_ranked_first:
            most_wins_method = None  # tied
        else:
            most_wins_method = sorted_by_wins[0].method

    return CrossCompanySummary(
        provenance_tier=provenance_tier,
        company_count=len(sorted_symbols),
        symbols=sorted_symbols,
        methods=list(methods),
        friedman=friedman_res,
        posthoc_status=posthoc_status,
        pairwise_wilcoxon=pairwise_wilcoxon,
        method_stats=method_stats,
        lowest_median_mase_method=lowest_med_method,
        best_mean_rank_method=best_mean_rank_method,
        most_company_wins_method=most_wins_method,
        rank_matrix=rank_matrix,
        mase_matrix=mase_matrix,
        note=SUPPORTING_EVIDENCE_NOTE,
        is_demo=is_demo or (provenance_tier == "DEMO"),
    )


# =========================================================================
# DataFrame Export Helpers for CLI and Dashboard Preparation
# =========================================================================

def rank_matrix_to_dataframe(summary: CrossCompanySummary) -> pd.DataFrame:
    """Format per-company recomputed MASE and within-company ranks as a DataFrame."""
    rows = []
    for sym in summary.symbols:
        row = {"symbol": sym}
        for m in summary.methods:
            row[f"{m}_mase"] = summary.mase_matrix[sym][m]
        for m in summary.methods:
            row[f"{m}_rank"] = summary.rank_matrix[sym][m]
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_results_to_dataframe(results: list[PairwiseWilcoxonResult]) -> pd.DataFrame:
    """Format pairwise Wilcoxon post-hoc results as a DataFrame."""
    rows = []
    for r in results:
        rows.append(
            {
                "pair": f"{r.model_a} vs {r.model_b}",
                "model_a": r.model_a,
                "model_b": r.model_b,
                "metric": r.metric,
                "n_companies": r.n_companies,
                "n_pairs": r.n_pairs,
                "n_nonzero_differences": r.n_nonzero_differences,
                "has_zero_differences": r.has_zero_differences,
                "has_tied_absolute_differences": r.has_tied_absolute_differences,
                "median_mase_a": r.median_mase_a,
                "median_mase_b": r.median_mase_b,
                "median_paired_difference": r.median_paired_difference,
                "wilcoxon_statistic": r.wilcoxon_statistic,
                "raw_p_value": r.raw_p_value,
                "holm_adjusted_p_value": r.holm_adjusted_p_value,
                "significant_holm": r.significant_holm,
                "lower_median_method": r.lower_median_method,
                "status": r.status,
                "reason": r.reason,
                "requested_method_policy": r.requested_method_policy,
                "actual_method_used": r.actual_method_used,
                "method_used": r.method_used,
                "zero_method": r.zero_method,
                "alternative": r.alternative,
            }
        )
    return pd.DataFrame(rows)


def method_stats_to_dataframe(summary: CrossCompanySummary) -> pd.DataFrame:
    """Format cross-company descriptive method statistics as a DataFrame."""
    rows = []
    for m in summary.methods:
        s = summary.method_stats[m]
        rows.append(
            {
                "method": s.method,
                "method_label": s.method_label,
                "companies_evaluated": s.companies_evaluated,
                "median_mase": s.median_mase,
                "mean_rank": s.mean_rank,
                "median_rank": s.median_rank,
                "count_ranked_first": s.count_ranked_first,
                "count_tied_first": s.count_tied_first,
                "mase_below_one_count": s.mase_below_one_count,
                "direct_rmse_wins_vs_naive": s.direct_rmse_wins_vs_naive,
                "direct_rmse_ties_vs_naive": s.direct_rmse_ties_vs_naive,
                "direct_rmse_losses_vs_naive": s.direct_rmse_losses_vs_naive,
                "direct_mae_wins_vs_naive": s.direct_mae_wins_vs_naive,
                "direct_mae_ties_vs_naive": s.direct_mae_ties_vs_naive,
                "direct_mae_losses_vs_naive": s.direct_mae_losses_vs_naive,
            }
        )
    return pd.DataFrame(rows)
