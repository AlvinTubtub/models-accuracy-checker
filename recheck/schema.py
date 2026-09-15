"""Schema and independent validation for bridge-exported evaluation JSON.

This module enforces strict structural invariants before any downstream metric
or statistical analysis is permitted.

Required Invariants for Every Company:
- evaluation_pairs == len(target_dates)
- each reported model observation count == len(target_dates)
- len(actual_closes) == len(target_dates)
- len(lag_reg predictions) == len(target_dates)
- len(arima predictions) == len(target_dates)
- len(lstm predictions) == len(target_dates)
- len(naive predictions) == len(target_dates)
- first target_date == evaluation_start
- last target_date == evaluation_end
- target_dates strictly increasing
- no duplicate target dates
- no NaN / +inf / -inf in actual closes or predictions
- required four evaluated methods (lag_reg, arima, lstm, naive) must be present
- missing Naive is a hard validation failure
- actual values are consistently aligned
- no silent truncation to shortest array
- evaluation dates remain the complete formal holdout
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
import math
from typing import Any

REQUIRED_MODELS: tuple[str, ...] = ("lag_reg", "arima", "lstm", "naive")
PRINCIPAL_MODELS: tuple[str, ...] = ("lag_reg", "arima", "lstm")


class ProvenanceTier(str, Enum):
    DEMO = "DEMO"
    FORMAL_FROZEN = "FORMAL_FROZEN"
    PROSPECTIVE = "PROSPECTIVE"


class ExportValidationError(ValueError):
    """Raised when an exported company evaluation file fails an integrity check."""


@dataclass(frozen=True)
class ValidationIssue:
    symbol: str
    severity: str  # "error" or "warning"
    message: str


@dataclass(frozen=True)
class CompanyEvaluationExport:
    """Parsed, validated representation of one company's exported evaluation."""

    symbol: str
    sector: str | None
    name: str | None
    evaluation_proportion: float
    development_pairs: int
    evaluation_pairs: int
    evaluation_start: date
    evaluation_end: date
    mase_denominator: float
    reported_metrics: dict[str, dict[str, float]]
    reported_best_model: str
    target_dates: list[date]
    actual_closes: list[float]
    predicted_closes_by_model: dict[str, list[float]]
    issues: list[ValidationIssue] = field(default_factory=list)
    provenance_tier: ProvenanceTier = ProvenanceTier.DEMO

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def is_formal_finalizable(self) -> bool:
        """Formal research finalization requires FORMAL_FROZEN tier and zero validation errors."""
        return self.provenance_tier == ProvenanceTier.FORMAL_FROZEN and not self.has_errors


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportValidationError(message)


def parse_company_export(payload: dict[str, Any], *, source_label: str = "") -> CompanyEvaluationExport:
    """Parse and independently validate one exported company JSON payload.

    Raises ExportValidationError for structurally unusable input or missing required models.
    Data alignment and completeness issues are collected as ValidationIssue records.
    """
    label = source_label or payload.get("symbol", "<unknown>")
    for required_key in ("symbol", "split", "mase_denominator", "metrics", "backtest"):
        _require(required_key in payload, f"{label}: missing required key '{required_key}'")

    symbol = str(payload["symbol"])
    split = payload["split"]
    for key in (
        "evaluation_proportion",
        "development_pairs",
        "evaluation_pairs",
        "evaluation_start",
        "evaluation_end",
    ):
        _require(key in split, f"{symbol}: split missing '{key}'")

    metrics = payload["metrics"]
    for model in REQUIRED_MODELS:
        _require(
            model in metrics,
            f"{symbol}: missing required model '{model}' in metrics (missing Naive is a hard failure)"
            if model == "naive"
            else f"{symbol}: metrics missing model '{model}'",
        )
        for field_name in ("rmse", "mae", "mase", "r2", "observations"):
            _require(
                field_name in metrics[model],
                f"{symbol}: metrics['{model}'] missing '{field_name}'",
            )

    backtest = payload["backtest"]
    for key in ("target_dates", "actual_closes", "predicted_closes_by_model"):
        _require(key in backtest, f"{symbol}: backtest missing '{key}'")

    for model in REQUIRED_MODELS:
        _require(
            model in backtest["predicted_closes_by_model"],
            f"{symbol}: backtest.predicted_closes_by_model missing '{model}'",
        )

    # Provenance tier validation
    raw_tier = payload.get("provenance_tier") or payload.get("provenance")
    if raw_tier is not None:
        try:
            provenance_tier = ProvenanceTier(str(raw_tier).upper())
        except ValueError as exc:
            raise ExportValidationError(
                f"{symbol}: invalid provenance_tier '{raw_tier}'. Must be DEMO, FORMAL_FROZEN, or PROSPECTIVE"
            ) from exc
    else:
        # Default to DEMO to prevent accidental unearned formal claims
        provenance_tier = ProvenanceTier.DEMO

    issues: list[ValidationIssue] = []

    # 1. Target dates parsing and chronological verification
    target_dates_raw = backtest["target_dates"]
    try:
        target_dates = [datetime.fromisoformat(str(d)).date() for d in target_dates_raw]
    except ValueError as exc:
        raise ExportValidationError(f"{symbol}: unparseable target_date - {exc}") from exc

    n_dates = len(target_dates)
    if n_dates < 2:
        issues.append(
            ValidationIssue(symbol, "error", f"only {n_dates} evaluation date(s); too few to score")
        )

    # Strictly increasing check
    for previous, current in zip(target_dates, target_dates[1:]):
        if previous >= current:
            issues.append(
                ValidationIssue(
                    symbol,
                    "error",
                    f"target_dates are not strictly increasing at {previous} -> {current}",
                )
            )
            break

    # Duplicate dates check
    if len(set(target_dates)) != len(target_dates):
        issues.append(ValidationIssue(symbol, "error", "duplicate target_dates found"))

    # 2. Evaluation start and end alignment
    try:
        eval_start = datetime.fromisoformat(str(split["evaluation_start"])).date()
        eval_end = datetime.fromisoformat(str(split["evaluation_end"])).date()
    except ValueError as exc:
        raise ExportValidationError(f"{symbol}: unparseable split dates - {exc}") from exc

    if target_dates:
        if target_dates[0] != eval_start:
            issues.append(
                ValidationIssue(
                    symbol,
                    "error",
                    f"first target_date ({target_dates[0]}) != split.evaluation_start ({eval_start})",
                )
            )
        if target_dates[-1] != eval_end:
            issues.append(
                ValidationIssue(
                    symbol,
                    "error",
                    f"last target_date ({target_dates[-1]}) != split.evaluation_end ({eval_end})",
                )
            )

    # 3. Evaluation pairs alignment (evaluation_pairs == len(target_dates))
    eval_pairs = int(split["evaluation_pairs"])
    if eval_pairs != n_dates:
        issues.append(
            ValidationIssue(
                symbol,
                "error",
                f"split.evaluation_pairs ({eval_pairs}) != len(target_dates) ({n_dates})",
            )
        )

    # 4. Observation counts per model (metrics[m]['observations'] == len(target_dates))
    for model in REQUIRED_MODELS:
        m_obs = int(metrics[model]["observations"])
        if m_obs != n_dates:
            issues.append(
                ValidationIssue(
                    symbol,
                    "error",
                    f"metrics['{model}']['observations'] ({m_obs}) != len(target_dates) ({n_dates})",
                )
            )

    # 5. Actual closes length and finite validation
    raw_actuals = backtest["actual_closes"]
    actual_closes = [float(v) for v in raw_actuals]
    if len(actual_closes) != n_dates:
        issues.append(
            ValidationIssue(
                symbol,
                "error",
                f"actual_closes length ({len(actual_closes)}) != target_dates length ({n_dates})",
            )
        )

    for value in actual_closes:
        if not math.isfinite(value):
            issues.append(ValidationIssue(symbol, "error", "non-finite value in actual_closes"))
            break

    # 6. Predictions length and finite validation per required model (no silent truncation)
    predicted_closes_by_model: dict[str, list[float]] = {}
    for model in REQUIRED_MODELS:
        series_raw = backtest["predicted_closes_by_model"][model]
        series = [float(v) for v in series_raw]
        predicted_closes_by_model[model] = series

        if len(series) != n_dates:
            issues.append(
                ValidationIssue(
                    symbol,
                    "error",
                    f"predicted_closes_by_model['{model}'] length ({len(series)}) "
                    f"!= target_dates length ({n_dates})",
                )
            )

        for p_val in series:
            if not math.isfinite(p_val):
                issues.append(
                    ValidationIssue(
                        symbol,
                        "error",
                        f"non-finite prediction in predicted_closes_by_model['{model}']",
                    )
                )
                break

    # 7. Cross-check actual_close consistency across models in records if present
    records = backtest.get("records")
    if records:
        actual_by_model_date: dict[tuple[str, str], float] = {}
        for record in records:
            key = (str(record.get("model")), str(record.get("target_date")))
            actual_by_model_date[key] = float(record.get("actual_close", math.nan))
        reference_actuals: dict[str, float] = {}
        mismatch_found = False
        for (model, target_date_str), actual_value in actual_by_model_date.items():
            if target_date_str not in reference_actuals:
                reference_actuals[target_date_str] = actual_value
            elif reference_actuals[target_date_str] != actual_value and not mismatch_found:
                issues.append(
                    ValidationIssue(
                        symbol,
                        "error",
                        f"actual_close disagrees across models on {target_date_str} "
                        f"({reference_actuals[target_date_str]} vs {actual_value})",
                    )
                )
                mismatch_found = True

    return CompanyEvaluationExport(
        symbol=symbol,
        sector=payload.get("sector"),
        name=payload.get("name"),
        evaluation_proportion=float(split["evaluation_proportion"]),
        development_pairs=int(split["development_pairs"]),
        evaluation_pairs=eval_pairs,
        evaluation_start=eval_start,
        evaluation_end=eval_end,
        mase_denominator=float(payload["mase_denominator"]),
        reported_metrics=metrics,
        reported_best_model=str(
            payload.get("principal_ranking", {}).get("best_model", "")
        ),
        target_dates=target_dates,
        actual_closes=actual_closes,
        predicted_closes_by_model=predicted_closes_by_model,
        issues=issues,
        provenance_tier=provenance_tier,
    )
