"""Optional cross-check: recompute the MASE denominator from raw OHLCV.

This is the one place this repo reaches past the bridge-exported JSON and
back to raw data — but even here, it never touches the original repo's
code, only its *data*. `backend/data/raw/<SYMBOL>.csv` is committed to the
original repo's git history (unlike `backend/artifacts/`), so this file can
be obtained with a plain `git clone`, no bridge export needed.

Why this matters: the MASE denominator the original pipeline reports
(`mean(|diff(development Close)|)`) is a single number, computed once, that
every model's MASE score depends on. Diffing recomputed vs. reported
RMSE/MAE/R2 (see `recheck/compare.py`) already catches most arithmetic
mistakes, but it can't catch a bad denominator baked equally into every
model's reported MASE. This module recomputes that denominator completely
independently, from the raw prices and the reported evaluation_start date
alone, so it can be diffed too.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from recheck.metrics import MetricComputationError, mase_denominator_from_closes


class RawDataError(ValueError):
    """Raised when a raw OHLCV CSV can't be used for the denominator recheck."""


@dataclass(frozen=True)
class DenominatorCheck:
    symbol: str
    recomputed_denominator: float
    reported_denominator: float
    development_rows_used: int
    abs_diff: float
    relative_diff_pct: float
    within_tolerance: bool


def _load_raw_closes(csv_path: Path) -> pd.Series:
    frame = pd.read_csv(csv_path)
    columns = {c.lower(): c for c in frame.columns}
    for required in ("date", "close"):
        if required not in columns:
            raise RawDataError(f"{csv_path}: missing required column '{required}'")
    frame = frame.rename(columns={columns["date"]: "date", columns["close"]: "close"})
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise RawDataError(f"{csv_path}: duplicate dates found in raw CSV")
    return frame.set_index("date")["close"].astype(float)


def recompute_mase_denominator(
    symbol: str,
    csv_path: Path,
    *,
    evaluation_start: date,
    reported_denominator: float,
    tolerance_pct: float = 0.5,
) -> DenominatorCheck:
    """Recompute the MASE denominator using only raw Close prices strictly
    before `evaluation_start` (i.e. the development/training partition),
    and compare it against what the original pipeline reported.
    """

    closes = _load_raw_closes(csv_path)
    development_closes = closes[closes.index < evaluation_start]
    if len(development_closes) < 2:
        raise RawDataError(
            f"{symbol}: fewer than 2 development rows before {evaluation_start} in {csv_path}"
        )

    try:
        recomputed = mase_denominator_from_closes(development_closes.tolist())
    except MetricComputationError as exc:
        raise RawDataError(f"{symbol}: {exc}") from exc

    abs_diff = abs(recomputed - reported_denominator)
    relative_diff_pct = (
        abs_diff / reported_denominator * 100.0 if reported_denominator else float("inf")
    )

    return DenominatorCheck(
        symbol=symbol,
        recomputed_denominator=recomputed,
        reported_denominator=reported_denominator,
        development_rows_used=len(development_closes),
        abs_diff=abs_diff,
        relative_diff_pct=relative_diff_pct,
        within_tolerance=relative_diff_pct <= tolerance_pct,
    )
