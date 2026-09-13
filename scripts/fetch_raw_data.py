#!/usr/bin/env python3
"""Fetch the 15 raw PSE OHLCV CSV files from the source repository.

Downloads directly from github.com/AlvinTubtub/pse-stock-price-forecast
into `data/raw/<SYMBOL>.csv` and verifies file integrity and date ranges.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import urllib.request
import urllib.error

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.companies import COMPANIES

RAW_BASE_URL = (
    "https://raw.githubusercontent.com/AlvinTubtub/pse-stock-price-forecast/main/backend/data/raw"
)


def download_company_csv(symbol: str, target_dir: Path) -> Path:
    """Download one company's raw CSV file."""
    url = f"{RAW_BASE_URL}/{symbol}.csv"
    destination = target_dir / f"{symbol}.csv"

    print(f"[*] Downloading {symbol} from {url}...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (models-accuracy-checker)"},
    )

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            content = resp.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP error downloading {symbol}: {exc.code} {exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"Error downloading {symbol}: {exc}") from exc

    with destination.open("wb") as f:
        f.write(content)

    return destination


def verify_csv(csv_path: Path) -> tuple[int, str, str]:
    """Verify CSV has required columns and return (row_count, start_date, end_date)."""
    import pandas as pd

    df = pd.read_csv(csv_path)
    required_cols = {"Date", "Close"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"{csv_path.name} missing required columns: {required_cols - set(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    start_date = df["Date"].iloc[0].strftime("%Y-%m-%d")
    end_date = df["Date"].iloc[-1].strftime("%Y-%m-%d")
    return len(df), start_date, end_date


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch 15 PSE raw CSV files")
    parser.add_argument(
        "--output",
        "-o",
        default="data/raw",
        help="Target directory for CSV files (default: data/raw)",
    )
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  FETCHING RAW PSE HISTORICAL DATA (Jan 2, 2020 – Sep 11, 2026)")
    print(f"  Destination: {out_dir}")
    print(f"{'='*70}\n")

    failed: list[str] = []
    success: list[tuple[str, int, str, str]] = []

    for comp in COMPANIES:
        try:
            dest = download_company_csv(comp.symbol, out_dir)
            rows, start, end = verify_csv(dest)
            success.append((comp.symbol, rows, start, end))
            print(f"    -> OK: {comp.symbol} ({rows} rows: {start} to {end})")
        except Exception as exc:
            failed.append(f"{comp.symbol}: {exc}")
            print(f"    -> FAILED {comp.symbol}: {exc}", file=sys.stderr)

    print(f"\n[*] Summary: Downloaded {len(success)}/{len(COMPANIES)} company files.")
    if failed:
        print(f"[!] {len(failed)} failure(s) occurred:")
        for err in failed:
            print(f"    - {err}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
