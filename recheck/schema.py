"""Schema and independent validation for bridge-exported evaluation JSON.

This re-checks the same kind of invariants the original pipeline claims to
enforce (chronological dates, no duplicates, consistent actual_close across
models for the same date) but does so from scratch here, against the
exported JSON, rather than trusting the export was internally consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
import math

REQUIRED_MODELS: tuple[str, ...] = ("lag_reg", "arima", "lstm", "naive")
PRINCIPAL_MODELS: tuple[str, ...] = ("lag_reg", "arima", "lstm")


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

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExportValidationError(message)


def parse_company_export(payload: dict, *, source_label: str = "") -> CompanyEvaluationExport:
    """Parse and independently validate one exported company JSON payload.

    Raises ExportValidationError for structurally unusable input. Softer
    integrity concerns (that don't prevent computing metrics, but should be
    surfaced to the user) are collected as ValidationIssue warnings instead
    of raising.
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
        _require(model in metrics, f"{symbol}: metrics missing model '{model}'")
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

    issues: list[ValidationIssue] = []

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

    # Chronological, strictly increasing, no duplicates - checked independently here.
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
    if len(set(target_dates)) != len(target_dates):
        issues.append(ValidationIssue(symbol, "error", "duplicate target_dates found"))

    actual_closes = [float(v) for v in backtest["actual_closes"]]
    if len(actual_closes) != n_dates:
        issues.append(
            ValidationIssue(
                symbol,
                "error",
                f"actual_closes length ({len(actual_closes)}) != target_dates length ({n_dates})",
            )
        )

    predicted_closes_by_model: dict[str, list[float]] = {}
    for model in REQUIRED_MODELS:
        series = [float(v) for v in backtest["predicted_closes_by_model"][model]]
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

    # Cross-check actual_close consistency using the per-record list too, if present -
    # this is the same invariant the original pipeline is supposed to enforce
    # (BacktestAlignmentError in its own code), re-verified here independently.
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

    for value in actual_closes:
        if value != value or value in (float("inf"), float("-inf")):  # NaN/inf check
            issues.append(ValidationIssue(symbol, "error", "non-finite value in actual_closes"))
            break

    return CompanyEvaluationExport(
        symbol=symbol,
        sector=payload.get("sector"),
        name=payload.get("name"),
        evaluation_proportion=float(split["evaluation_proportion"]),
        development_pairs=int(split["development_pairs"]),
        evaluation_pairs=int(split["evaluation_pairs"]),
        evaluation_start=datetime.fromisoformat(str(split["evaluation_start"])).date(),
        evaluation_end=datetime.fromisoformat(str(split["evaluation_end"])).date(),
        mase_denominator=float(payload["mase_denominator"]),
        reported_metrics=metrics,
        reported_best_model=str(
            payload.get("principal_ranking", {}).get("best_model", "")
        ),
        target_dates=target_dates,
        actual_closes=actual_closes,
        predicted_closes_by_model=predicted_closes_by_model,
        issues=issues,
    )
