"""Independent Within-Company Diebold-Mariano Statistical Testing.

Tests for equal predictive accuracy between every pair of evaluated forecasting
methods on the COMPLETE aligned out-of-sample holdout.

Conventions:
- Models evaluated: lag_reg, arima, lstm, naive.
- Exactly 6 pairwise combinations per company.
- Two loss functions: squared-error loss and absolute-error loss.
- Sign convention: d_t = loss_a_t - loss_b_t
    * mean(d) < 0 => model A has lower average loss
    * mean(d) > 0 => model B has lower average loss
    * mean(d) == 0 => equal average loss
- HAC / Newey-West long-run variance with Bartlett kernel.
- Harvey-Leybourne-Newbold (HLN) small-sample modification.
- Two-sided Student-t distribution with n - 1 degrees of freedom.
- Holm step-down multiplicity correction applied strictly PER COMPANY and
  PER LOSS FUNCTION (6 tests per family).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import math
from typing import Sequence
import numpy as np
import scipy.stats as stats

from recheck.schema import CompanyEvaluationExport
from recheck.tolerances import TIE_TOLERANCE

# Exactly six required unordered pairs of evaluated models
REQUIRED_PAIRS: list[tuple[str, str]] = [
    ("lag_reg", "arima"),
    ("lag_reg", "lstm"),
    ("lag_reg", "naive"),
    ("arima", "lstm"),
    ("arima", "naive"),
    ("lstm", "naive"),
]


class LossFunction(str, Enum):
    SQUARED = "squared"
    ABSOLUTE = "absolute"


class DMStatus(str, Enum):
    A_SIGNIFICANTLY_LOWER_LOSS = "A_SIGNIFICANTLY_LOWER_LOSS"
    B_SIGNIFICANTLY_LOWER_LOSS = "B_SIGNIFICANTLY_LOWER_LOSS"
    LOWER_LOSS_NOT_SIGNIFICANT = "LOWER_LOSS_NOT_SIGNIFICANT"
    IDENTICAL_LOSS = "IDENTICAL_LOSS"
    DEGENERATE = "DEGENERATE"
    UNDEFINED = "UNDEFINED"


@dataclass(frozen=True)
class DMResult:
    symbol: str
    loss_function: LossFunction
    model_a: str
    model_b: str
    n: int
    forecast_horizon: int
    mean_loss_a: float
    mean_loss_b: float
    mean_loss_differential: float
    hac_lag: int
    long_run_variance: float
    dm_statistic_uncorrected: float | None
    hln_factor: float
    dm_statistic_hln: float | None
    degrees_of_freedom: int
    raw_p_value: float | None
    lower_loss_model: str | None
    significant_raw: bool
    status: DMStatus
    reason: str | None
    holm_adjusted_p_value: float | None = 1.0
    significant_holm: bool = False
    holm_family_size: int = 6
    alpha: float = 0.05


@dataclass(frozen=True)
class CompanyDMResults:
    symbol: str
    squared_family: list[DMResult]
    absolute_family: list[DMResult]
    alpha: float
    hac_lag_policy: str

    @property
    def all_results(self) -> list[DMResult]:
        return self.squared_family + self.absolute_family


@dataclass(frozen=True)
class DMCrossCheckResult:
    symbol: str
    model_a: str
    model_b: str
    loss_function: str
    independent_dm: float
    reported_dm: float
    independent_p: float
    reported_p: float
    dm_difference: float
    p_difference: float
    is_consistent: bool
    discrepancy_note: str | None = None


def compute_hln_factor(n: int, h: int = 1) -> float:
    """Harvey-Leybourne-Newbold (1997) small-sample correction factor.

    Formula: sqrt((n + 1 - 2*h + h*(h - 1)/n) / n).
    For h = 1, simplifies exactly to sqrt((n - 1) / n).
    """
    if n <= 0:
        return 0.0
    numerator = n + 1 - 2 * h + (h * (h - 1) / n)
    if numerator <= 0:
        return 0.0
    return math.sqrt(numerator / n)


def compute_newey_west_bandwidth(n: int) -> int:
    """Automatic deterministic Newey-West (1994) lag selection rule:

    floor(4 * (n / 100)^(2/9)).
    Bounded between 0 and n - 1.
    """
    if n <= 1:
        return 0
    raw_lag = int(math.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0))))
    return max(0, min(n - 1, raw_lag))


def compute_long_run_variance(d: np.ndarray, lag: int) -> float:
    """Compute Newey-West long-run variance estimate with Bartlett kernel.

    LRV = gamma_0 + 2 * sum_{j=1}^lag (1 - j / (lag + 1)) * gamma_j
    where gamma_j = (1 / n) * sum_{t=j+1}^n (d_t - mean_d) * (d_{t-j} - mean_d)
    """
    n = len(d)
    if n <= 1:
        return 0.0
    mean_d = float(np.mean(d))
    dem = d - mean_d

    # Sample variance (gamma_0) with 1/n denominator
    gamma_0 = float(np.dot(dem, dem) / n)
    if gamma_0 <= TIE_TOLERANCE:
        return 0.0

    lrv = gamma_0
    for j in range(1, lag + 1):
        weight = 1.0 - (j / (lag + 1.0))
        gamma_j = float(np.dot(dem[j:], dem[:-j]) / n)
        lrv += 2.0 * weight * gamma_j

    # Ensure non-negative LRV
    return max(0.0, lrv)


def holm_step_down(
    results: Sequence[DMResult],
    alpha: float = 0.05,
) -> list[DMResult]:
    """Apply Holm step-down multiplicity correction to a family of DM results.

    Requirements:
    - Family is strictly PER COMPANY and PER LOSS FUNCTION (e.g. 6 tests).
    - Adjusted p-values:
        * >= corresponding raw p-value
        * <= 1.0
        * satisfy Holm monotonic step-down: p_tilde_{(k)} = max(p_tilde_{(k-1)}, (m - k + 1) * p_{(k)})
    """
    m = len(results)
    if m == 0:
        return []

    # Sort tests with valid numeric p-values ascending; keep family size m = len(results)
    valid_tests = [(i, r) for i, r in enumerate(results) if r.raw_p_value is not None]
    indexed = sorted(valid_tests, key=lambda x: x[1].raw_p_value)

    adjusted_p: dict[int, float | None] = {i: None for i in range(m)}
    running_max = 0.0

    for rank, (orig_idx, res) in enumerate(indexed):
        k = rank + 1  # 1-based index
        multiplier = m - k + 1
        raw_p = res.raw_p_value  # known float

        # Calculate candidate adjusted p-value
        cand_p = min(1.0, multiplier * raw_p)
        running_max = max(running_max, cand_p)
        adjusted_val = min(1.0, max(running_max, raw_p))
        adjusted_p[orig_idx] = adjusted_val

    # Reconstruct updated DMResult items with Holm adjusted statistics
    updated: list[DMResult] = []
    for orig_idx, res in enumerate(results):
        adj_p = adjusted_p[orig_idx]
        sig_holm = (
            (adj_p is not None and adj_p <= alpha)
            if res.status not in (DMStatus.UNDEFINED, DMStatus.IDENTICAL_LOSS, DMStatus.DEGENERATE)
            else False
        )

        # Determine final interpretation status based on Holm significance
        if res.status in (DMStatus.UNDEFINED, DMStatus.IDENTICAL_LOSS, DMStatus.DEGENERATE):
            new_status = res.status
        elif sig_holm:
            if res.mean_loss_differential < -TIE_TOLERANCE:
                new_status = DMStatus.A_SIGNIFICANTLY_LOWER_LOSS
            elif res.mean_loss_differential > TIE_TOLERANCE:
                new_status = DMStatus.B_SIGNIFICANTLY_LOWER_LOSS
            else:
                new_status = DMStatus.LOWER_LOSS_NOT_SIGNIFICANT
        else:
            new_status = DMStatus.LOWER_LOSS_NOT_SIGNIFICANT

        updated.append(
            DMResult(
                symbol=res.symbol,
                loss_function=res.loss_function,
                model_a=res.model_a,
                model_b=res.model_b,
                n=res.n,
                forecast_horizon=res.forecast_horizon,
                mean_loss_a=res.mean_loss_a,
                mean_loss_b=res.mean_loss_b,
                mean_loss_differential=res.mean_loss_differential,
                hac_lag=res.hac_lag,
                long_run_variance=res.long_run_variance,
                dm_statistic_uncorrected=res.dm_statistic_uncorrected,
                hln_factor=res.hln_factor,
                dm_statistic_hln=res.dm_statistic_hln,
                degrees_of_freedom=res.degrees_of_freedom,
                raw_p_value=res.raw_p_value,
                lower_loss_model=res.lower_loss_model,
                significant_raw=res.significant_raw,
                status=new_status,
                reason=res.reason,
                holm_adjusted_p_value=adj_p,
                significant_holm=sig_holm,
                holm_family_size=m,
                alpha=alpha,
            )
        )

    return updated


def compute_diebold_mariano(
    symbol: str,
    actual: Sequence[float],
    pred_a: Sequence[float],
    pred_b: Sequence[float],
    model_a: str,
    model_b: str,
    loss_function: LossFunction = LossFunction.SQUARED,
    forecast_horizon: int = 1,
    hac_lag: int | None = None,
    alpha: float = 0.05,
) -> DMResult:
    """Independently compute the Diebold-Mariano test between model_a and model_b.

    Sign convention: d_t = loss_a_t - loss_b_t
    mean(d) < 0 => model A has lower average loss
    mean(d) > 0 => model B has lower average loss
    """
    n_act = len(actual)
    n_a = len(pred_a)
    n_b = len(pred_b)

    # 1. Validate complete alignment and non-empty inputs
    if n_act != n_a or n_act != n_b:
        return DMResult(
            symbol=symbol,
            loss_function=loss_function,
            model_a=model_a,
            model_b=model_b,
            n=n_act,
            forecast_horizon=forecast_horizon,
            mean_loss_a=0.0,
            mean_loss_b=0.0,
            mean_loss_differential=0.0,
            hac_lag=0,
            long_run_variance=0.0,
            dm_statistic_uncorrected=0.0,
            hln_factor=0.0,
            dm_statistic_hln=0.0,
            degrees_of_freedom=max(0, n_act - 1),
            raw_p_value=1.0,
            lower_loss_model=None,
            significant_raw=False,
            status=DMStatus.UNDEFINED,
            reason="LENGTH_MISMATCH_OR_ALIGNMENT_FAILURE",
            alpha=alpha,
        )

    n = n_act
    if n < 3:
        return DMResult(
            symbol=symbol,
            loss_function=loss_function,
            model_a=model_a,
            model_b=model_b,
            n=n,
            forecast_horizon=forecast_horizon,
            mean_loss_a=0.0,
            mean_loss_b=0.0,
            mean_loss_differential=0.0,
            hac_lag=0,
            long_run_variance=0.0,
            dm_statistic_uncorrected=0.0,
            hln_factor=0.0,
            dm_statistic_hln=0.0,
            degrees_of_freedom=max(0, n - 1),
            raw_p_value=1.0,
            lower_loss_model=None,
            significant_raw=False,
            status=DMStatus.UNDEFINED,
            reason="INSUFFICIENT_OBSERVATIONS",
            alpha=alpha,
        )

    act_arr = np.asarray(actual, dtype=np.float64)
    a_arr = np.asarray(pred_a, dtype=np.float64)
    b_arr = np.asarray(pred_b, dtype=np.float64)

    # Check finite numbers
    if not (np.all(np.isfinite(act_arr)) and np.all(np.isfinite(a_arr)) and np.all(np.isfinite(b_arr))):
        return DMResult(
            symbol=symbol,
            loss_function=loss_function,
            model_a=model_a,
            model_b=model_b,
            n=n,
            forecast_horizon=forecast_horizon,
            mean_loss_a=0.0,
            mean_loss_b=0.0,
            mean_loss_differential=0.0,
            hac_lag=0,
            long_run_variance=0.0,
            dm_statistic_uncorrected=0.0,
            hln_factor=0.0,
            dm_statistic_hln=0.0,
            degrees_of_freedom=n - 1,
            raw_p_value=1.0,
            lower_loss_model=None,
            significant_raw=False,
            status=DMStatus.UNDEFINED,
            reason="NON_FINITE_DATA",
            alpha=alpha,
        )

    # Validate or compute HAC lag
    if hac_lag is not None:
        if hac_lag < 0 or hac_lag >= n:
            return DMResult(
                symbol=symbol,
                loss_function=loss_function,
                model_a=model_a,
                model_b=model_b,
                n=n,
                forecast_horizon=forecast_horizon,
                mean_loss_a=0.0,
                mean_loss_b=0.0,
                mean_loss_differential=0.0,
                hac_lag=hac_lag,
                long_run_variance=0.0,
                dm_statistic_uncorrected=0.0,
                hln_factor=0.0,
                dm_statistic_hln=0.0,
                degrees_of_freedom=n - 1,
                raw_p_value=1.0,
                lower_loss_model=None,
                significant_raw=False,
                status=DMStatus.UNDEFINED,
                reason="INVALID_HAC_LAG",
                alpha=alpha,
            )
        used_lag = hac_lag
    else:
        used_lag = compute_newey_west_bandwidth(n)

    # 2. Calculate loss sequences
    err_a = act_arr - a_arr
    err_b = act_arr - b_arr

    if loss_function == LossFunction.SQUARED:
        loss_a = err_a ** 2
        loss_b = err_b ** 2
    elif loss_function == LossFunction.ABSOLUTE:
        loss_a = np.abs(err_a)
        loss_b = np.abs(err_b)
    else:
        raise ValueError(f"Unsupported loss function: {loss_function}")

    mean_loss_a = float(np.mean(loss_a))
    mean_loss_b = float(np.mean(loss_b))

    # Loss differential vector: d_t = loss_a_t - loss_b_t
    d = loss_a - loss_b
    mean_d = float(np.mean(d))

    # Determine which model has lower mean loss
    if mean_d < -TIE_TOLERANCE:
        lower_loss_model = model_a
    elif mean_d > TIE_TOLERANCE:
        lower_loss_model = model_b
    else:
        lower_loss_model = None

    # Check for identical loss / zero differential
    max_abs_diff = float(np.max(np.abs(d)))
    if max_abs_diff <= TIE_TOLERANCE:
        return DMResult(
            symbol=symbol,
            loss_function=loss_function,
            model_a=model_a,
            model_b=model_b,
            n=n,
            forecast_horizon=forecast_horizon,
            mean_loss_a=mean_loss_a,
            mean_loss_b=mean_loss_b,
            mean_loss_differential=0.0,
            hac_lag=used_lag,
            long_run_variance=0.0,
            dm_statistic_uncorrected=0.0,
            hln_factor=compute_hln_factor(n, forecast_horizon),
            dm_statistic_hln=0.0,
            degrees_of_freedom=n - 1,
            raw_p_value=1.0,
            lower_loss_model=None,
            significant_raw=False,
            status=DMStatus.IDENTICAL_LOSS,
            reason="IDENTICAL_LOSS_SERIES",
            alpha=alpha,
        )

    # 3. Compute Long-Run Variance
    lrv = compute_long_run_variance(d, used_lag)
    if lrv <= TIE_TOLERANCE:
        # Zero variance in differential with non-zero constant mean (e.g. d = [-1, -1, -1, -1])
        return DMResult(
            symbol=symbol,
            loss_function=loss_function,
            model_a=model_a,
            model_b=model_b,
            n=n,
            forecast_horizon=forecast_horizon,
            mean_loss_a=mean_loss_a,
            mean_loss_b=mean_loss_b,
            mean_loss_differential=mean_d,
            hac_lag=used_lag,
            long_run_variance=0.0,
            dm_statistic_uncorrected=None,
            hln_factor=compute_hln_factor(n, forecast_horizon),
            dm_statistic_hln=None,
            degrees_of_freedom=n - 1,
            raw_p_value=None,
            lower_loss_model=lower_loss_model,
            significant_raw=False,
            status=DMStatus.DEGENERATE,
            reason="ZERO_LONG_RUN_VARIANCE_WITH_NONZERO_MEAN",
            alpha=alpha,
        )

    # 4. Asymptotic DM statistic and HLN correction
    dm_uncorrected = (mean_d * math.sqrt(n)) / math.sqrt(lrv)
    hln = compute_hln_factor(n, forecast_horizon)
    dm_hln = hln * dm_uncorrected
    df = n - 1

    # Two-sided Student-t p-value
    raw_p = 2.0 * float(stats.t.sf(abs(dm_hln), df=df))
    raw_p = min(1.0, max(0.0, raw_p))

    sig_raw = raw_p <= alpha
    status = DMStatus.LOWER_LOSS_NOT_SIGNIFICANT

    return DMResult(
        symbol=symbol,
        loss_function=loss_function,
        model_a=model_a,
        model_b=model_b,
        n=n,
        forecast_horizon=forecast_horizon,
        mean_loss_a=mean_loss_a,
        mean_loss_b=mean_loss_b,
        mean_loss_differential=mean_d,
        hac_lag=used_lag,
        long_run_variance=lrv,
        dm_statistic_uncorrected=dm_uncorrected,
        hln_factor=hln,
        dm_statistic_hln=dm_hln,
        degrees_of_freedom=df,
        raw_p_value=raw_p,
        lower_loss_model=lower_loss_model,
        significant_raw=sig_raw,
        status=status,
        reason=None,
        alpha=alpha,
    )


def test_company_pairwise_dm(
    export: CompanyEvaluationExport,
    forecast_horizon: int = 1,
    hac_lag: int | None = None,
    alpha: float = 0.05,
) -> CompanyDMResults:
    """Perform complete independent pairwise DM testing for all 6 model pairs.

    Runs squared-loss and absolute-loss families separately, applying Holm
    step-down correction strictly within each 6-test family.
    """
    symbol = export.symbol
    actual = export.actual_closes
    preds = export.predicted_closes_by_model

    # Formal DM runs cannot run on a truncated display window (e.g. 60 sessions)
    if export.provenance_tier.value == "FORMAL_FROZEN":
        if len(export.target_dates) <= 60:
            raise ValueError(
                f"{symbol}: Formal DM testing requires complete formal holdout; "
                f"truncated 60-session display subset cannot enter formal DM."
            )

    # Validate complete date and length alignment
    if len(export.target_dates) != len(actual):
        raise ValueError(
            f"{symbol}: Date alignment failure: target_dates length {len(export.target_dates)} "
            f"!= actual_closes length {len(actual)}."
        )

    for m_name, p_series in preds.items():
        if len(p_series) != len(actual):
            raise ValueError(
                f"{symbol}: Alignment failure: model '{m_name}' predictions length {len(p_series)} "
                f"!= actual_closes length {len(actual)}."
            )

    hac_policy = f"fixed_{hac_lag}" if hac_lag is not None else "newey_west_auto"

    squared_results: list[DMResult] = []
    absolute_results: list[DMResult] = []

    for m_a, m_b in REQUIRED_PAIRS:
        if m_a not in preds or m_b not in preds:
            # Missing model
            for lf, container in ((LossFunction.SQUARED, squared_results), (LossFunction.ABSOLUTE, absolute_results)):
                container.append(
                    DMResult(
                        symbol=symbol,
                        loss_function=lf,
                        model_a=m_a,
                        model_b=m_b,
                        n=len(actual),
                        forecast_horizon=forecast_horizon,
                        mean_loss_a=0.0,
                        mean_loss_b=0.0,
                        mean_loss_differential=0.0,
                        hac_lag=0,
                        long_run_variance=0.0,
                        dm_statistic_uncorrected=0.0,
                        hln_factor=0.0,
                        dm_statistic_hln=0.0,
                        degrees_of_freedom=max(0, len(actual) - 1),
                        raw_p_value=1.0,
                        lower_loss_model=None,
                        significant_raw=False,
                        status=DMStatus.UNDEFINED,
                        reason=f"MISSING_PREDICTIONS: {m_a if m_a not in preds else m_b}",
                        alpha=alpha,
                    )
                )
            continue

        pred_a = preds[m_a]
        pred_b = preds[m_b]

        # Squared error test
        res_sq = compute_diebold_mariano(
            symbol=symbol,
            actual=actual,
            pred_a=pred_a,
            pred_b=pred_b,
            model_a=m_a,
            model_b=m_b,
            loss_function=LossFunction.SQUARED,
            forecast_horizon=forecast_horizon,
            hac_lag=hac_lag,
            alpha=alpha,
        )
        squared_results.append(res_sq)

        # Absolute error test
        res_abs = compute_diebold_mariano(
            symbol=symbol,
            actual=actual,
            pred_a=pred_a,
            pred_b=pred_b,
            model_a=m_a,
            model_b=m_b,
            loss_function=LossFunction.ABSOLUTE,
            forecast_horizon=forecast_horizon,
            hac_lag=hac_lag,
            alpha=alpha,
        )
        absolute_results.append(res_abs)

    # Apply Holm step-down correction separately to squared and absolute families
    adjusted_squared = holm_step_down(squared_results, alpha=alpha)
    adjusted_absolute = holm_step_down(absolute_results, alpha=alpha)

    return CompanyDMResults(
        symbol=symbol,
        squared_family=adjusted_squared,
        absolute_family=adjusted_absolute,
        alpha=alpha,
        hac_lag_policy=hac_policy,
    )


# Pytest compatibility: prevent pytest from collecting this function as a test
test_company_pairwise_dm.__test__ = False
evaluate_company_pairwise_dm = test_company_pairwise_dm


def cross_check_forecastph_dm(
    raw_payload: dict,
    independent_results: CompanyDMResults,
    tolerance: float = 1e-4,
) -> list[DMCrossCheckResult]:
    """Cross-check ForecastPH reported DM results against independently computed results.

    If raw_payload contains statistical_tests -> diebold_mariano, parses each
    reported result and compares against independent calculations.
    Returns list of comparisons with discrepancy flags.
    """
    comparisons: list[DMCrossCheckResult] = []
    stat_tests = raw_payload.get("statistical_tests", {})
    if not isinstance(stat_tests, dict):
        return comparisons

    dm_tests = stat_tests.get("diebold_mariano", [])
    if not isinstance(dm_tests, list) or not dm_tests:
        return comparisons

    # Index independent results by (model_a, model_b, loss_function)
    ind_map: dict[tuple[str, str, str], DMResult] = {}
    for res in independent_results.all_results:
        ind_map[(res.model_a, res.model_b, res.loss_function.value)] = res
        # Also index symmetric pair
        ind_map[(res.model_b, res.model_a, res.loss_function.value)] = res

    for item in dm_tests:
        if not isinstance(item, dict):
            continue
        m_a = item.get("model_1") or item.get("model_a")
        m_b = item.get("model_2") or item.get("model_b")
        loss_fn = item.get("loss_function") or item.get("loss_type", "squared")
        rep_dm = item.get("dm_statistic") or item.get("statistic")
        rep_p = item.get("raw_p_value") or item.get("p_value")

        if m_a is None or m_b is None or rep_dm is None or rep_p is None:
            continue

        key = (str(m_a), str(m_b), str(loss_fn))
        if key in ind_map:
            ind_res = ind_map[key]
            if ind_res.dm_statistic_hln is not None and ind_res.raw_p_value is not None:
                comp_dm = ind_res.dm_statistic_hln if ind_res.model_a == m_a else -ind_res.dm_statistic_hln
                dm_diff = abs(float(rep_dm) - comp_dm)
                p_diff = abs(float(rep_p) - ind_res.raw_p_value)
                is_consistent = (dm_diff <= tolerance) and (p_diff <= tolerance)
                note = None if is_consistent else f"Discrepancy: diff_dm={dm_diff:.6f}, diff_p={p_diff:.6f}"
                ind_p = ind_res.raw_p_value
            else:
                comp_dm = float("nan")
                ind_p = float("nan")
                dm_diff = float("nan")
                p_diff = float("nan")
                is_consistent = False
                note = f"Independent calculation is {ind_res.status.value}"

            comparisons.append(
                DMCrossCheckResult(
                    symbol=independent_results.symbol,
                    model_a=str(m_a),
                    model_b=str(m_b),
                    loss_function=str(loss_fn),
                    independent_dm=comp_dm,
                    reported_dm=float(rep_dm),
                    independent_p=ind_p,
                    reported_p=float(rep_p),
                    dm_difference=dm_diff,
                    p_difference=p_diff,
                    is_consistent=is_consistent,
                    discrepancy_note=note,
                )
            )

    return comparisons


def dm_results_to_dataframe(
    results: Sequence[CompanyDMResults] | dict[str, CompanyDMResults] | Sequence[DMResult],
) -> pd.DataFrame:
    """Convert CompanyDMResults or DMResults to a flat pandas DataFrame."""
    import pandas as pd

    if isinstance(results, dict):
        result_list = list(results.values())
    elif isinstance(results, Sequence):
        result_list = list(results)
    else:
        result_list = [results]

    dm_items: list[DMResult] = []
    for item in result_list:
        if isinstance(item, CompanyDMResults):
            dm_items.extend(item.all_results)
        elif isinstance(item, DMResult):
            dm_items.append(item)

    rows: list[dict] = []
    for r in dm_items:
        rows.append(
            {
                "symbol": r.symbol,
                "loss_function": r.loss_function.value,
                "model_a": r.model_a,
                "model_b": r.model_b,
                "pair": f"{r.model_a} vs {r.model_b}",
                "n": r.n,
                "forecast_horizon": r.forecast_horizon,
                "mean_loss_a": r.mean_loss_a,
                "mean_loss_b": r.mean_loss_b,
                "mean_loss_differential": r.mean_loss_differential,
                "lower_loss_model": r.lower_loss_model,
                "hac_lag": r.hac_lag,
                "long_run_variance": r.long_run_variance,
                "dm_statistic_uncorrected": r.dm_statistic_uncorrected,
                "hln_factor": r.hln_factor,
                "dm_statistic_hln": r.dm_statistic_hln,
                "degrees_of_freedom": r.degrees_of_freedom,
                "raw_p_value": r.raw_p_value,
                "significant_raw": r.significant_raw,
                "holm_adjusted_p_value": r.holm_adjusted_p_value,
                "significant_holm": r.significant_holm,
                "holm_family_size": r.holm_family_size,
                "alpha": r.alpha,
                "status": r.status.value,
                "reason": r.reason,
            }
        )
    return pd.DataFrame(rows)

