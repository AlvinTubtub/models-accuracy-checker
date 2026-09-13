"""Chronological dataset partitioning and cross-validation generator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterator
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitPlan:
    symbol: str
    evaluation_proportion: float
    total_records: int
    dev_count: int
    eval_count: int
    dev_indices: np.ndarray
    eval_indices: np.ndarray
    eval_start_date: date
    eval_end_date: date
    mase_denominator: float


def create_split_plan(
    df: pd.DataFrame,
    symbol: str,
    evaluation_proportion: float = 0.15,
) -> SplitPlan:
    """Divide chronological observations into development (~85%) and evaluation (~15%)."""
    df_sorted = df.sort_values("Date").reset_index(drop=True)
    total_n = len(df_sorted)
    if total_n < 50:
        raise ValueError(f"Insufficient historical records for {symbol}: {total_n}")

    eval_count = int(total_n * evaluation_proportion)
    dev_count = total_n - eval_count

    dev_indices = np.arange(dev_count)
    eval_indices = np.arange(dev_count, total_n)

    eval_dates = pd.to_datetime(df_sorted["Date"].iloc[eval_indices])
    eval_start = eval_dates.iloc[0].date()
    eval_end = eval_dates.iloc[-1].date()

    # Calculate MASE denominator over development Close changes
    dev_closes = df_sorted["Close"].iloc[dev_indices].to_numpy(dtype=float)
    diffs = np.abs(np.diff(dev_closes))
    mase_denom = float(np.mean(diffs))

    return SplitPlan(
        symbol=symbol,
        evaluation_proportion=evaluation_proportion,
        total_records=total_n,
        dev_count=dev_count,
        eval_count=eval_count,
        dev_indices=dev_indices,
        eval_indices=eval_indices,
        eval_start_date=eval_start,
        eval_end_date=eval_end,
        mase_denominator=mase_denom,
    )


def time_series_splits(
    n_samples: int,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Chronological expanding-window cross validation folds."""
    from sklearn.model_selection import TimeSeriesSplit

    tscv = TimeSeriesSplit(n_splits=n_splits)
    indices = np.arange(n_samples)
    for train_idx, val_idx in tscv.split(indices):
        yield train_idx, val_idx
