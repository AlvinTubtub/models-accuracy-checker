#!/usr/bin/env python3
"""Manual CLI runner to train models and produce evaluation artifacts.

Runs exact model computations (LIR, ARIMA, LSTM, and Naive) on raw PSE OHLCV CSVs
over the Jan 2, 2020 – Sep 11, 2026 data window, and exports validated JSON
artifacts directly to `data/evaluations/`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.companies import COMPANIES, ALL_SYMBOLS
from training.pipeline import train_and_export_company


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train LIR, ARIMA, LSTM, and Naive models on raw PSE data"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Train all 15 PSE companies")
    group.add_argument("--symbol", "-s", type=str, help="Train only this company symbol (e.g. ALI)")

    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Directory containing raw <SYMBOL>.csv files (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="data/evaluations",
        help="Directory to write exported JSON evaluations (default: data/evaluations)",
    )
    parser.add_argument(
        "--skip-lstm",
        action="store_true",
        help="Skip deep LSTM training for faster iteration",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of epochs for LSTM training (default: 30)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="PyTorch device: 'cpu', 'cuda', or 'mps' (auto-detected by default)",
    )
    args = parser.parse_args()

    raw_dir = (REPO_ROOT / args.raw_dir).resolve()
    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.is_dir():
        print(
            f"ERROR: Raw data directory '{raw_dir}' does not exist.\n"
            f"Please run 'python scripts/fetch_raw_data.py' first.",
            file=sys.stderr,
        )
        return 1

    symbols_to_train = ALL_SYMBOLS if args.all else [args.symbol.upper()]

    print(f"\n{'='*70}")
    print(f"  PSE FRESH MODEL TRAINING & OOS EVALUATION (Jan 2, 2020 – Sep 11, 2026)")
    print(f"  Companies to train : {len(symbols_to_train)} ({', '.join(symbols_to_train)})")
    print(f"  Raw data directory : {raw_dir}")
    print(f"  Evaluation output  : {out_dir}")
    print(f"  LSTM training      : {'SKIPPED' if args.skip_lstm else f'{args.epochs} epochs'}")
    print(f"{'='*70}\n")

    start_total = time.time()
    successful: list[str] = []
    failed: list[str] = []

    for i, symbol in enumerate(symbols_to_train, 1):
        t0 = time.time()
        print(f"[{i}/{len(symbols_to_train)}] Training {symbol}...")
        try:
            dest = train_and_export_company(
                symbol=symbol,
                raw_dir=raw_dir,
                output_dir=out_dir,
                skip_lstm=args.skip_lstm,
                lstm_epochs=args.epochs,
                device=args.device,
            )
            elapsed = time.time() - t0
            print(f"    -> COMPLETED in {elapsed:.1f}s: {dest.name}")
            successful.append(symbol)
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    -> FAILED in {elapsed:.1f}s: {exc}", file=sys.stderr)
            failed.append(f"{symbol}: {exc}")

    # Write manifest
    manifest = {
        "exported_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_synthetic": False,
        "description": "Exact trained model evaluation artifacts (LIR, ARIMA, LSTM, Naive)",
        "source_data_window": "2020-01-02 to 2026-09-11",
        "requested_symbols": list(symbols_to_train),
        "exported_symbols": successful,
        "failed_symbols": failed,
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    total_time = time.time() - start_total
    print(f"\n{'='*70}")
    print(f"  TRAINING RUN SUMMARY")
    print(f"  Completed: {len(successful)}/{len(symbols_to_train)} companies in {total_time:.1f}s")
    if failed:
        print(f"  Failures : {len(failed)}")
        for err in failed:
            print(f"    - {err}", file=sys.stderr)
    print(f"{'='*70}\n")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
