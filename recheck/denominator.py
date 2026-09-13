"""Optional raw-CSV MASE denominator cross-check.

Allows independent recalculation of the MASE scale factor directly from
a company's raw OHLCV CSV file (from backend/data/raw/<SYMBOL>.csv).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import pandas as pd

from recheck.metrics import mase_denominator_from_closes


def calculate_mase_denominator_from_csv(
    csv_path: Path | str,
    *,
    evaluation_proportion: float = 0.15,
    close_column: str = "Close",
    date_column: str = "Date",
) -> float:
    """Read a raw CSV file and compute the development partition's MASE denominator.

    Args:
        csv_path: Path to the raw OHLCV CSV file.
        evaluation_proportion: Proportion of data reserved for out-of-sample evaluation (default: 0.15).
        close_column: Column name for closing prices (default: 'Close').
        date_column: Column name for trade dates (default: 'Date').

    Returns:
        float: mean(|diff(Close)|) over the chronological development partition.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    df = pd.read_csv(path)
    if close_column not in df.columns:
        # Try case-insensitive lookup
        matched = [c for c in df.columns if c.lower() == close_column.lower()]
        if matched:
            close_column = matched[0]
        else:
            raise KeyError(f"Column '{close_column}' not found in {path}")

    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column).reset_index(drop=True)

    closes: Sequence[float] = df[close_column].dropna().astype(float).tolist()
    total_samples = len(closes)
    if total_samples < 10:
        raise ValueError(f"Insufficient samples in {path} ({total_samples})")

    # Split into development (1 - eval_prop) and evaluation
    eval_count = int(total_samples * evaluation_proportion)
    dev_count = total_samples - eval_count
    dev_closes = closes[:dev_count]

    return mase_denominator_from_closes(dev_closes)
