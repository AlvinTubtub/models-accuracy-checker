#!/usr/bin/env python3
"""Generate realistic synthetic evaluation export JSON files for all 15 PSE companies.

This allows immediate testing of the recheck engine, test suite, and Streamlit
dashboard without requiring access to the original repository or trained models.
All generated files strictly conform to the bridge export schema.
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random

import sys

# Ensure repository root is on Python path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from config.companies import COMPANIES, CompanyMetadata
from recheck.metrics import compute_metrics
from recheck.schema import REQUIRED_MODELS, PRINCIPAL_MODELS

# Baseline representative price scales for PSE tickers (PHP)
BASE_PRICES = {
    "ALI": 32.5,
    "APX": 3.1,
    "BPI": 118.0,
    "GLO": 1950.0,
    "ICT": 360.0,
    "JFC": 245.0,
    "MBT": 68.5,
    "MEG": 2.15,
    "MER": 380.0,
    "NIKL": 4.8,
    "PGOLD": 27.0,
    "SCC": 34.0,
    "SECB": 72.0,
    "SHLPH": 12.5,
    "SMPH": 30.0,
}


def _generate_business_days(start_date: date, count: int) -> list[date]:
    """Generate chronological business days (Monday to Friday)."""
    days: list[date] = []
    current = start_date
    while len(days) < count:
        if current.weekday() < 5:  # Monday to Friday
            days.append(current)
        current += timedelta(days=1)
    return days


def generate_company_data(
    company: CompanyMetadata,
    *,
    eval_count: int = 245,
    dev_count: int = 1388,
    seed: int = 42,
    introduce_mismatch: bool = False,
) -> dict:
    rng = random.Random(seed + sum(ord(c) for c in company.symbol))

    base_price = BASE_PRICES.get(company.symbol, 50.0)
    volatility = base_price * 0.015

    # Simulate development series for realistic MASE denominator
    dev_closes = [base_price]
    for _ in range(dev_count):
        change = rng.gauss(0.0, volatility)
        new_val = max(0.5, dev_closes[-1] + change)
        dev_closes.append(new_val)

    dev_diffs = [abs(dev_closes[i] - dev_closes[i - 1]) for i in range(1, len(dev_closes))]
    mase_denominator = sum(dev_diffs) / len(dev_diffs)

    # Simulate evaluation trading dates up to 2026-09-11
    end_date = date(2026, 9, 11)
    # Walk backwards to find start date
    test_days: list[date] = []
    curr = end_date
    while len(test_days) < eval_count:
        if curr.weekday() < 5:
            test_days.append(curr)
        curr -= timedelta(days=1)
    target_dates = sorted(test_days)

    # Actual closes in evaluation window
    actual_closes: list[float] = []
    current_close = dev_closes[-1]
    for _ in range(eval_count):
        current_close = max(0.5, current_close + rng.gauss(0.0, volatility))
        actual_closes.append(round(current_close, 2))

    # Generate predictions per model
    # Naive model: Close(t-1)
    naive_preds = [round(dev_closes[-1], 2)] + actual_closes[:-1]

    # Principal models with varying accuracy characteristics
    # LIR: usually solid, slightly damped mean reversion
    lir_preds = [
        round(max(0.5, actual_closes[i] + rng.gauss(0.0, volatility * 0.75)), 2)
        for i in range(eval_count)
    ]
    # ARIMA: smooth autoregressive tracking
    arima_preds = [
        round(max(0.5, actual_closes[i] + rng.gauss(0.0, volatility * 0.85)), 2)
        for i in range(eval_count)
    ]
    # LSTM: non-linear tracking
    lstm_preds = [
        round(max(0.5, actual_closes[i] + rng.gauss(0.0, volatility * 0.95)), 2)
        for i in range(eval_count)
    ]

    predicted_by_model: dict[str, list[float]] = {
        "lag_reg": lir_preds,
        "arima": arima_preds,
        "lstm": lstm_preds,
        "naive": naive_preds,
    }

    # Compute genuine metrics
    computed_metrics: dict[str, dict] = {}
    for model_name in REQUIRED_MODELS:
        m = compute_metrics(
            actual_closes,
            predicted_by_model[model_name],
            mase_denominator=mase_denominator,
        )
        reported_dict = {
            "rmse": round(m.rmse, 4),
            "mae": round(m.mae, 4),
            "mase": round(m.mase, 4),
            "r2": round(m.r2, 4),
            "observations": m.observations,
        }
        if introduce_mismatch and model_name == "lstm":
            # Artificially alter reported metric for testing discrepancy detection
            reported_dict["rmse"] = round(m.rmse + 0.15, 4)
        computed_metrics[model_name] = reported_dict

    # Ranking by RMSE among principal models
    ranking = sorted(
        [{"model": m, "rmse": computed_metrics[m]["rmse"]} for m in PRINCIPAL_MODELS],
        key=lambda x: x["rmse"],
    )
    ranked_models = [
        {"rank": idx + 1, "model": r["model"], "criterion": "rmse", "value": r["rmse"]}
        for idx, r in enumerate(ranking)
    ]
    best_model = ranked_models[0]["model"]

    # Build detailed backtest records
    records: list[dict] = []
    for i, t_date in enumerate(target_dates):
        origin_date = (
            (t_date - timedelta(days=3)).isoformat()
            if t_date.weekday() == 0
            else (t_date - timedelta(days=1)).isoformat()
        )
        origin_close = dev_closes[-1] if i == 0 else actual_closes[i - 1]
        act_close = actual_closes[i]

        for model_name in REQUIRED_MODELS:
            pred_close = predicted_by_model[model_name][i]
            records.append(
                {
                    "symbol": company.symbol,
                    "model": model_name,
                    "origin_date": origin_date,
                    "target_date": t_date.isoformat(),
                    "origin_close": origin_close,
                    "actual_close": act_close,
                    "predicted_close": pred_close,
                    "error": round(pred_close - act_close, 4),
                }
            )

    return {
        "symbol": company.symbol,
        "name": company.name,
        "sector": company.sector,
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": dev_count,
            "evaluation_pairs": eval_count,
            "evaluation_start": target_dates[0].isoformat(),
            "evaluation_end": target_dates[-1].isoformat(),
        },
        "mase_denominator": round(mase_denominator, 6),
        "metrics": computed_metrics,
        "principal_ranking": {
            "criterion": "rmse",
            "best_model": best_model,
            "ranked_models": ranked_models,
        },
        "backtest": {
            "symbol": company.symbol,
            "target_dates": [d.isoformat() for d in target_dates],
            "actual_closes": actual_closes,
            "predicted_closes_by_model": predicted_by_model,
            "records": records,
            "source": "chronological_out_of_sample_evaluation",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic evaluation JSON files for 15 PSE companies"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/evaluations",
        help="Target directory (default: data/evaluations)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--mismatch-symbol",
        default=None,
        help="Symbol to deliberately inject an audit mismatch into (for testing)",
    )
    args = parser.parse_args()

    out_dir = Path(args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    exported: list[str] = []
    for comp in COMPANIES:
        inject = args.mismatch_symbol == comp.symbol
        data = generate_company_data(comp, seed=args.seed, introduce_mismatch=inject)
        file_path = out_dir / f"{comp.symbol}.json"
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        exported.append(comp.symbol)
        print(f"Generated {comp.symbol} -> {file_path}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "is_synthetic": True,
        "description": "Synthetic evaluation dataset for testing and dashboard verification",
        "exported_symbols": exported,
        "failed_symbols": [],
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    print(f"\nSuccessfully generated {len(exported)} evaluation files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
