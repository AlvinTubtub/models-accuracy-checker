"""Data provenance and trading calendar completeness verification.

Enforces clear separation among data tiers:
- DEMO: Simulated or synthetic data. Cannot be used for formal research evidence or promotion eligibility.
- FORMAL_FROZEN: Cryptographically hashed historical data used for fixed holdout evaluation.
- PROSPECTIVE: Forward-looking data captured pre-settlement and verified against PSE settlement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
import hashlib
from pathlib import Path
from typing import Sequence
import pandas as pd


class ProvenanceTier(str, Enum):
    DEMO = "DEMO"
    FORMAL_FROZEN = "FORMAL_FROZEN"
    PROSPECTIVE = "PROSPECTIVE"


@dataclass(frozen=True)
class CalendarAuditReport:
    symbol: str
    row_count: int
    first_date: str
    last_date: str
    duplicate_dates: list[str]
    suspected_missing_dates: list[str]
    anomalous_returns: list[dict[str, str | float]]
    is_complete: bool


@dataclass(frozen=True)
class ProvenanceRecord:
    tier: ProvenanceTier
    source_identifier: str
    source_sha256: str
    row_count: int
    first_date: str
    last_date: str
    retrieval_timestamp_utc: str | None = None
    is_synthetic: bool = False
    calendar_audit: CalendarAuditReport | None = None
    notes: str = ""

    def as_dict(self) -> dict:
        return {
            "tier": self.tier.value,
            "source_identifier": self.source_identifier,
            "source_sha256": self.source_sha256,
            "row_count": self.row_count,
            "first_date": self.first_date,
            "last_date": self.last_date,
            "retrieval_timestamp_utc": self.retrieval_timestamp_utc,
            "is_synthetic": self.is_synthetic,
            "notes": self.notes,
        }


def compute_file_sha256(file_path: Path | str) -> str:
    """Calculate hex SHA-256 checksum of a file."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_string_sha256(text: str) -> str:
    """Calculate hex SHA-256 checksum of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def audit_raw_csv(csv_path: Path | str, symbol: str = "") -> CalendarAuditReport:
    """Audit a raw price CSV for date monotonicity, duplicates, and gaps.

    Expected columns: Date, Close (and optionally Open, High, Low, Volume).
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    sym = symbol or path.stem

    if "Date" not in df.columns or "Close" not in df.columns:
        raise ValueError(f"{path.name} must contain 'Date' and 'Close' columns")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    row_count = len(df)
    if row_count == 0:
        raise ValueError(f"{path.name} is empty")

    first_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
    last_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")

    # Check duplicates
    duplicate_series = df[df["Date"].duplicated(keep=False)]
    duplicate_dates = [d.strftime("%Y-%m-%d") for d in duplicate_series["Date"].unique()]

    # Check date gaps: consecutive trading days should not have gaps > 4 calendar days (long weekend)
    # under normal circumstances, though holiday clusters occur.
    suspected_missing: list[str] = []
    for i in range(1, len(df)):
        gap = (df["Date"].iloc[i] - df["Date"].iloc[i - 1]).days
        if gap > 5:  # more than 5 calendar days indicates potential prolonged gap / missing records
            prev_d = df["Date"].iloc[i - 1].strftime("%Y-%m-%d")
            curr_d = df["Date"].iloc[i].strftime("%Y-%m-%d")
            suspected_missing.append(f"{prev_d} to {curr_d} ({gap} days)")

    # Check for extreme anomalous price movements (> 50% single-day change without split adjustment)
    anomalies: list[dict[str, str | float]] = []
    closes = df["Close"].astype(float).values
    for i in range(1, len(closes)):
        prev_c = closes[i - 1]
        curr_c = closes[i]
        if prev_c > 0:
            pct_change = abs(curr_c - prev_c) / prev_c
            if pct_change >= 0.50:
                anomalies.append({
                    "date": df["Date"].iloc[i].strftime("%Y-%m-%d"),
                    "prev_close": float(prev_c),
                    "curr_close": float(curr_c),
                    "pct_change": float(pct_change),
                })

    is_complete = len(duplicate_dates) == 0

    return CalendarAuditReport(
        symbol=sym,
        row_count=row_count,
        first_date=first_date,
        last_date=last_date,
        duplicate_dates=duplicate_dates,
        suspected_missing_dates=suspected_missing,
        anomalous_returns=anomalies,
        is_complete=is_complete,
    )
