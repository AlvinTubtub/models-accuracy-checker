"""Direct Out-of-Sample (OOS) model-vs-Naive comparison on aligned holdouts.

Strictly decouples scale-normalized MASE from actual holdout superiority:
- Evaluates lag_reg vs naive, arima vs naive, and lstm vs naive.
- Uses exactly the same aligned holdout target dates and actual closes.
- For FORMAL_FROZEN runs, uses independently reconstructed Naive predictions
  from frozen raw Close series.
- Explicitly flags WIN, TIE, and LOSS. An exact tie is NOT a win.
- R^2 is supplementary only and is NEVER used for ranking or winner selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
import math
from typing import Sequence
import pandas as pd

from recheck.metrics import compute_metrics
from recheck.naive import audit_naive_predictions
from recheck.schema import (
    PRINCIPAL_MODELS,
    REQUIRED_MODELS,
    CompanyEvaluationExport,
    ProvenanceTier,
)
from recheck.tolerances import TIE_TOLERANCE

TOLERANCE = TIE_TOLERANCE


class ComparisonStatus(str, Enum):
    WIN = "WIN"
    TIE = "TIE"
    LOSS = "LOSS"


class DirectOOSVerificationError(ValueError):
    """Raised when formal direct-OOS verification requirements are violated."""


@dataclass(frozen=True)
class SkillScore:
    skill_defined: bool
    skill_value: float | None
    skill_reason: str | None = None


@dataclass(frozen=True)
class ModelOOSComparison:
    symbol: str
    sector: str | None
    model: str
    model_rmse: float
    naive_rmse: float
    rmse_difference: float
    rmse_skill: SkillScore
    beats_naive_rmse: bool
    rmse_comparison_status: ComparisonStatus
    model_mae: float
    naive_mae: float
    mae_difference: float
    mae_skill: SkillScore
    beats_naive_mae: bool
    mae_comparison_status: ComparisonStatus
    model_mase: float
    naive_mase: float
    mase_below_one: bool
    model_r2: float
    naive_r2: float

    @property
    def rmse_skill_defined(self) -> bool:
        return self.rmse_skill.skill_defined

    @property
    def rmse_skill_value(self) -> float | None:
        return self.rmse_skill.skill_value

    @property
    def rmse_skill_reason(self) -> str | None:
        return self.rmse_skill.skill_reason

    @property
    def rmse_skill_vs_naive(self) -> float | None:
        return self.rmse_skill.skill_value

    @property
    def mae_skill_defined(self) -> bool:
        return self.mae_skill.skill_defined

    @property
    def mae_skill_value(self) -> float | None:
        return self.mae_skill.skill_value

    @property
    def mae_skill_reason(self) -> str | None:
        return self.mae_skill.skill_reason

    @property
    def mae_skill_vs_naive(self) -> float | None:
        return self.mae_skill.skill_value


@dataclass(frozen=True)
class CompanyOOSSummary:
    symbol: str
    sector: str | None
    provenance_tier: ProvenanceTier
    target_dates: list[date]
    models: dict[str, ModelOOSComparison]
    naive_source: str
    naive_audit_status: str
    best_principal_model: str
    best_principal_rmse: float
    is_principal_tie: bool
    best_evaluated_method: str
    best_evaluated_rmse: float
    is_evaluated_tie: bool
    best_principal_beats_naive: bool
    all_principals_worse_than_naive: bool
    naive_is_best_evaluated_method: bool
    overall_status_message: str


def compute_skill(
    model_val: float, naive_val: float, tolerance: float = TIE_TOLERANCE
) -> SkillScore:
    """Compute relative percentage skill score: 1.0 - (model_val / naive_val).

    Handles zero naive baseline explicitly and safely without division by zero:
    If naive_val <= tolerance, returns skill_defined=False, skill_value=None,
    skill_reason='NAIVE_ERROR_ZERO'.
    Ensures all exported structures remain finite and JSON-safe.
    """
    if naive_val <= tolerance:
        return SkillScore(
            skill_defined=False,
            skill_value=None,
            skill_reason="NAIVE_ERROR_ZERO",
        )
    val = 1.0 - (model_val / naive_val)
    if not math.isfinite(val):
        return SkillScore(
            skill_defined=False,
            skill_value=None,
            skill_reason="NON_FINITE_SKILL",
        )
    return SkillScore(
        skill_defined=True,
        skill_value=val,
        skill_reason=None,
    )


def determine_comparison_status(
    model_val: float, naive_val: float, tolerance: float = TOLERANCE
) -> tuple[bool, ComparisonStatus]:
    """Determine WIN/TIE/LOSS status. An exact tie is NOT a win."""
    diff = model_val - naive_val
    if diff < -tolerance:
        return True, ComparisonStatus.WIN
    elif abs(diff) <= tolerance:
        return False, ComparisonStatus.TIE
    else:
        return False, ComparisonStatus.LOSS


def evaluate_company_direct_oos(
    export: CompanyEvaluationExport,
    *,
    raw_dates: Sequence[date | str] | None = None,
    raw_closes: Sequence[float] | None = None,
    tolerance: float = 1e-6,
) -> CompanyOOSSummary:
    """Perform direct OOS comparison between principal models and the Naive benchmark.

    For FORMAL_FROZEN tier:
        Requires frozen raw dates and closes to independently reconstruct Naive.
        If missing or mismatching, raises DirectOOSVerificationError.
    For DEMO tier:
        Exported synthetic Naive is permitted for demonstration, but labeled NOT_RUN_DEMO.
    """
    symbol = export.symbol
    tier = export.provenance_tier
    actual = export.actual_closes
    n_dates = len(export.target_dates)

    # 1. Obtain and verify Naive baseline predictions
    naive_source = "exported_demo"
    naive_audit_status = "NOT_RUN_DEMO"

    if tier == ProvenanceTier.FORMAL_FROZEN:
        if raw_dates is None or raw_closes is None:
            raise DirectOOSVerificationError(
                f"{symbol}: FORMAL_FROZEN evaluation requires raw price data for independent "
                "Naive reconstruction (status: RAW_DATA_UNAVAILABLE)"
            )
        # Independently audit exported Naive against reconstructed
        audit_res = audit_naive_predictions(
            symbol=symbol,
            raw_dates=raw_dates,
            raw_closes=raw_closes,
            target_dates=export.target_dates,
            exported_naive_predictions=export.predicted_closes_by_model["naive"],
            tolerance=tolerance,
        )
        if not audit_res.all_naive_predictions_match:
            raise DirectOOSVerificationError(
                f"{symbol}: Exported Naive predictions failed independent reconstruction audit "
                f"({audit_res.mismatch_count} mismatches, max diff: {audit_res.max_absolute_difference:.6f})"
            )
        naive_source = "reconstructed"
        naive_audit_status = "PASSED"
        naive_preds = [step.reconstructed_naive for step in audit_res.steps]
    else:
        # DEMO tier: verify if raw data is provided, but keep status clearly non-formal
        if raw_dates is not None and raw_closes is not None:
            audit_res = audit_naive_predictions(
                symbol=symbol,
                raw_dates=raw_dates,
                raw_closes=raw_closes,
                target_dates=export.target_dates,
                exported_naive_predictions=export.predicted_closes_by_model["naive"],
                tolerance=tolerance,
            )
            naive_source = "reconstructed_demo"
            naive_audit_status = "DEMO_MATCH" if audit_res.all_naive_predictions_match else "DEMO_MISMATCH"
            naive_preds = [step.reconstructed_naive for step in audit_res.steps]
        else:
            naive_preds = export.predicted_closes_by_model["naive"]

    # Compute Naive benchmark metrics on the aligned holdout
    naive_metrics = compute_metrics(
        actual_closes=actual,
        predicted_closes=naive_preds,
        mase_denominator=export.mase_denominator,
    )

    model_comparisons: dict[str, ModelOOSComparison] = {}
    rmse_by_model: dict[str, float] = {}
    mae_by_model: dict[str, float] = {}

    rmse_by_model["naive"] = naive_metrics.rmse
    mae_by_model["naive"] = naive_metrics.mae

    # 2. Compute metrics and comparisons for principal models
    for model in ("lag_reg", "arima", "lstm"):
        if model not in export.predicted_closes_by_model:
            continue
        p_preds = export.predicted_closes_by_model[model]
        m_metrics = compute_metrics(
            actual_closes=actual,
            predicted_closes=p_preds,
            mase_denominator=export.mase_denominator,
        )
        rmse_by_model[model] = m_metrics.rmse
        mae_by_model[model] = m_metrics.mae

        beats_rmse, rmse_status = determine_comparison_status(m_metrics.rmse, naive_metrics.rmse)
        beats_mae, mae_status = determine_comparison_status(m_metrics.mae, naive_metrics.mae)

        rmse_diff = m_metrics.rmse - naive_metrics.rmse
        mae_diff = m_metrics.mae - naive_metrics.mae

        rmse_skill = compute_skill(m_metrics.rmse, naive_metrics.rmse)
        mae_skill = compute_skill(m_metrics.mae, naive_metrics.mae)

        model_comparisons[model] = ModelOOSComparison(
            symbol=symbol,
            sector=export.sector,
            model=model,
            model_rmse=m_metrics.rmse,
            naive_rmse=naive_metrics.rmse,
            rmse_difference=rmse_diff,
            rmse_skill=rmse_skill,
            beats_naive_rmse=beats_rmse,
            rmse_comparison_status=rmse_status,
            model_mae=m_metrics.mae,
            naive_mae=naive_metrics.mae,
            mae_difference=mae_diff,
            mae_skill=mae_skill,
            beats_naive_mae=beats_mae,
            mae_comparison_status=mae_status,
            model_mase=m_metrics.mase,
            naive_mase=naive_metrics.mase,
            mase_below_one=m_metrics.mase < 1.0,
            model_r2=m_metrics.r2,  # preserved exactly, never clipped
            naive_r2=naive_metrics.r2,
        )

    # 3. Deterministic Ranking & Tie Policy
    # Rank Best Principal Model among ("lag_reg", "arima", "lstm")
    principals = [m for m in ("lag_reg", "arima", "lstm") if m in rmse_by_model]
    min_p_rmse = min(rmse_by_model[m] for m in principals)
    tied_principals_rmse = [m for m in principals if abs(rmse_by_model[m] - min_p_rmse) <= TOLERANCE]

    if len(tied_principals_rmse) == 1:
        best_principal_model = tied_principals_rmse[0]
        is_p_tie = False
    else:
        # Tie-breaker 1: MAE
        min_p_mae = min(mae_by_model[m] for m in tied_principals_rmse)
        tied_principals_mae = [m for m in tied_principals_rmse if abs(mae_by_model[m] - min_p_mae) <= TOLERANCE]
        if len(tied_principals_mae) == 1:
            best_principal_model = tied_principals_mae[0]
            is_p_tie = True
        else:
            # Deterministic alphabetical tie representation
            is_p_tie = True
            best_principal_model = "TIE:" + "/".join(sorted(tied_principals_mae))

    best_p_rmse = min_p_rmse

    # Rank Best Evaluated Method among ("lag_reg", "arima", "lstm", "naive")
    all_evaluated = principals + ["naive"]
    min_eval_rmse = min(rmse_by_model[m] for m in all_evaluated)
    tied_eval_rmse = [m for m in all_evaluated if abs(rmse_by_model[m] - min_eval_rmse) <= TOLERANCE]

    if len(tied_eval_rmse) == 1:
        best_eval_method = tied_eval_rmse[0]
        is_eval_tie = False
    else:
        min_eval_mae = min(mae_by_model[m] for m in tied_eval_rmse)
        tied_eval_mae = [m for m in tied_eval_rmse if abs(mae_by_model[m] - min_eval_mae) <= TOLERANCE]
        if len(tied_eval_mae) == 1:
            best_eval_method = tied_eval_mae[0]
            is_eval_tie = True
        else:
            is_eval_tie = True
            best_eval_method = "TIE:" + "/".join(sorted(tied_eval_mae))

    # 4. Status flags and narrative verdict
    # An exact tie is NOT beating Naive
    best_p_beats_naive = best_p_rmse < (naive_metrics.rmse - TOLERANCE)
    all_worse_than_naive = all(
        rmse_by_model[m] >= (naive_metrics.rmse - TOLERANCE) for m in principals
    )
    naive_is_best = (best_eval_method == "naive")

    if all_worse_than_naive:
        msg = "None of the principal forecasting models outperformed the Naive benchmark during the stated holdout period."
    elif best_p_beats_naive:
        msg = f"Principal model '{best_principal_model}' outperformed the Naive benchmark on the aligned holdout."
    else:
        msg = "Principal models tied with the Naive benchmark."

    return CompanyOOSSummary(
        symbol=symbol,
        sector=export.sector,
        provenance_tier=tier,
        target_dates=export.target_dates,
        models=model_comparisons,
        naive_source=naive_source,
        naive_audit_status=naive_audit_status,
        best_principal_model=best_principal_model,
        best_principal_rmse=best_p_rmse,
        is_principal_tie=is_p_tie,
        best_evaluated_method=best_eval_method,
        best_evaluated_rmse=min_eval_rmse,
        is_evaluated_tie=is_eval_tie,
        best_principal_beats_naive=best_p_beats_naive,
        all_principals_worse_than_naive=all_worse_than_naive,
        naive_is_best_evaluated_method=naive_is_best,
        overall_status_message=msg,
    )


def oos_comparisons_to_dataframe(
    summaries: Sequence[CompanyOOSSummary] | dict[str, CompanyOOSSummary],
) -> pd.DataFrame:
    """Convert OOS summaries into an aggregate dataframe suitable for CLI and dashboards."""
    summary_list = summaries.values() if isinstance(summaries, dict) else summaries
    rows: list[dict] = []

    for s in summary_list:
        for model_name, comp in s.models.items():
            rows.append(
                {
                    "symbol": s.symbol,
                    "sector": s.sector,
                    "model": comp.model,
                    "model_rmse": comp.model_rmse,
                    "naive_rmse": comp.naive_rmse,
                    "rmse_difference": comp.rmse_difference,
                    "rmse_skill_defined": comp.rmse_skill_defined,
                    "rmse_skill_value": comp.rmse_skill_value,
                    "rmse_skill_reason": comp.rmse_skill_reason,
                    "rmse_skill_vs_naive": comp.rmse_skill_vs_naive,
                    "beats_naive_rmse": comp.beats_naive_rmse,
                    "rmse_comparison_status": comp.rmse_comparison_status.value,
                    "model_mae": comp.model_mae,
                    "naive_mae": comp.naive_mae,
                    "mae_difference": comp.mae_difference,
                    "mae_skill_defined": comp.mae_skill_defined,
                    "mae_skill_value": comp.mae_skill_value,
                    "mae_skill_reason": comp.mae_skill_reason,
                    "mae_skill_vs_naive": comp.mae_skill_vs_naive,
                    "beats_naive_mae": comp.beats_naive_mae,
                    "mae_comparison_status": comp.mae_comparison_status.value,
                    "model_mase": comp.model_mase,
                    "naive_mase": comp.naive_mase,
                    "mase_below_one": comp.mase_below_one,
                    "model_r2": comp.model_r2,
                    "naive_r2": comp.naive_r2,
                    "best_principal_model": s.best_principal_model,
                    "best_evaluated_method": s.best_evaluated_method,
                    "best_principal_beats_naive": s.best_principal_beats_naive,
                    "all_principals_worse_than_naive": s.all_principals_worse_than_naive,
                    "naive_is_best_evaluated_method": s.naive_is_best_evaluated_method,
                }
            )

    return pd.DataFrame(rows)
