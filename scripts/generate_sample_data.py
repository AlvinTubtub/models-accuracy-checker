#!/usr/bin/env python3
"""Generate synthetic per-company evaluation exports for demoing the
dashboard/CLI without needing the original repo or the bridge step.

The generated data is clearly synthetic (random-walk prices, seeded noise
standing in for model error) but uses the exact same JSON schema
`bridge/export_evaluations.py` produces, so it exercises every part of
`recheck/` and `dashboard/app.py` for real.

One company (the last one generated) has its `reported_metrics` for one
model deliberately corrupted by a fixed offset, so the dashboard's
recomputed-vs-reported mismatch view has something real to show. This is
noted in that company's manifest entry and printed when this script runs -
it is not hidden.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recheck.metrics import compute_metrics

# For demo purposes only - real exports get sector/name from the original
# repo's own config/companies.py via the bridge script, not from here.
DEMO_COMPANIES = [
    ("BPI", "Bank of the Philippine Islands", "Financials"),
    ("MBT", "Metropolitan Bank & Trust Co.", "Financials"),
    ("SECB", "Security Bank Corporation", "Financials"),
    ("MER", "Manila Electric Company", "Industrial"),
    ("JFC", "Jollibee Foods Corporation", "Industrial"),
    ("SHLPH", "Splash Corporation", "Industrial"),
    ("MEG", "Megaworld Corporation", "Property"),
    ("ALI", "Ayala Land, Inc.", "Property"),
    ("SMPH", "SM Prime Holdings, Inc.", "Property"),
    ("GLO", "Globe Telecom, Inc.", "Services"),
    ("PGOLD", "Puregold Price Club, Inc.", "Services"),
    ("ICT", "International Container Terminal Services", "Services"),
    ("APX", "Apex Mining Company, Inc.", "Mining & Oil"),
    ("NIKL", "Nickel Asia Corporation", "Mining & Oil"),
    ("SCC", "Semirara Mining and Power Corporation", "Mining & Oil"),
]

EVALUATION_START_DATE = date(2020, 1, 2)
EVALUATION_END_DATE = date(2026, 9, 11)
EVALUATION_PROPORTION = 0.15

# Rough per-model noise scale relative to daily price moves - deliberately
# varied per model so different companies "win" with different models,
# rather than one model trivially dominating every synthetic company.
MODEL_NOISE_SCALE = {"lag_reg": 0.55, "arima": 0.70, "lstm": 0.60, "naive": 1.0}


def _business_days(start: date, end: date) -> list[date]:
    days = []
    current = start
    while current <= end:
        if current.weekday() < 5:  # skip weekends only, for demo simplicity
            days.append(current)
        current += timedelta(days=1)
    return days


def _random_walk_closes(rng: random.Random, n: int, start_price: float) -> list[float]:
    closes = [start_price]
    for _ in range(n - 1):
        step = rng.gauss(0.0, start_price * 0.012)
        closes.append(max(0.50, closes[-1] + step))
    return closes


def _build_company(symbol: str, name: str, sector: str, seed: int, corrupt: bool) -> dict:
    rng = random.Random(seed)
    dates = _business_days(EVALUATION_START_DATE, EVALUATION_END_DATE)
    start_price = rng.uniform(8.0, 400.0)
    closes = _random_walk_closes(rng, len(dates), start_price)

    n_pairs = len(dates) - 1  # (origin, target) pairs
    evaluation_pairs = max(30, round(n_pairs * EVALUATION_PROPORTION))
    development_pairs = n_pairs - evaluation_pairs
    evaluation_target_indices = list(range(development_pairs + 1, len(dates)))

    development_closes = closes[: development_pairs + 1]
    development_diffs = [
        abs(development_closes[i] - development_closes[i - 1])
        for i in range(1, len(development_closes))
    ]
    mase_denominator = sum(development_diffs) / len(development_diffs)

    target_dates = [dates[i] for i in evaluation_target_indices]
    actual_closes = [closes[i] for i in evaluation_target_indices]
    origin_closes = [closes[i - 1] for i in evaluation_target_indices]

    predicted_closes_by_model: dict[str, list[float]] = {}
    predicted_closes_by_model["naive"] = list(origin_closes)
    for model in ("lag_reg", "arima", "lstm"):
        scale = MODEL_NOISE_SCALE[model] * (start_price * 0.01)
        predicted_closes_by_model[model] = [
            actual + rng.gauss(0.0, scale) for actual in actual_closes
        ]

    reported_metrics: dict[str, dict[str, float]] = {}
    for model, predicted in predicted_closes_by_model.items():
        metrics = compute_metrics(actual_closes, predicted, mase_denominator=mase_denominator)
        reported_metrics[model] = metrics.as_dict()

    if corrupt:
        # Deliberately introduce a reported-vs-actual discrepancy so the
        # dashboard's mismatch view has a real example to display.
        reported_metrics["arima"]["rmse"] = reported_metrics["arima"]["rmse"] + 1.75
        reported_metrics["arima"]["mase"] = reported_metrics["arima"]["mase"] + 0.15

    principal_rmse = {
        model: reported_metrics[model]["rmse"] for model in ("lag_reg", "arima", "lstm")
    }
    ranked = sorted(principal_rmse.items(), key=lambda item: item[1])
    best_model = ranked[0][0]

    records = []
    for model, predicted in predicted_closes_by_model.items():
        for idx, (target_date, origin_close, actual_close, predicted_close) in enumerate(
            zip(target_dates, origin_closes, actual_closes, predicted)
        ):
            origin_date = dates[evaluation_target_indices[idx] - 1]
            records.append(
                {
                    "symbol": symbol,
                    "model": model,
                    "origin_date": origin_date.isoformat(),
                    "target_date": target_date.isoformat(),
                    "origin_close": origin_close,
                    "actual_close": actual_close,
                    "predicted_close": predicted_close,
                    "error": predicted_close - actual_close,
                }
            )

    return {
        "symbol": symbol,
        "sector": sector,
        "name": name,
        "split": {
            "evaluation_proportion": EVALUATION_PROPORTION,
            "development_pairs": development_pairs,
            "evaluation_pairs": evaluation_pairs,
            "evaluation_start": target_dates[0].isoformat(),
            "evaluation_end": target_dates[-1].isoformat(),
        },
        "mase_denominator": mase_denominator,
        "metrics": reported_metrics,
        "principal_ranking": {
            "criterion": "rmse",
            "best_model": best_model,
            "ranked_models": [
                {"rank": rank + 1, "model": model, "criterion": "rmse", "value": value}
                for rank, (model, value) in enumerate(ranked)
            ],
        },
        "backtest": {
            "symbol": symbol,
            "target_dates": [d.isoformat() for d in target_dates],
            "actual_closes": actual_closes,
            "predicted_closes_by_model": predicted_closes_by_model,
            "records": records,
            "source": "SYNTHETIC_DEMO_DATA - not from the real pipeline",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/evaluations", help="Directory to write JSON into")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed (default: 42)")
    arguments = parser.parse_args(argv)

    output_dir = Path(arguments.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for index, (symbol, name, sector) in enumerate(DEMO_COMPANIES):
        is_last = index == len(DEMO_COMPANIES) - 1
        payload = _build_company(
            symbol, name, sector, seed=arguments.seed + index, corrupt=is_last
        )
        destination = output_dir / f"{symbol}.json"
        with destination.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        exported.append(symbol)
        if is_last:
            print(f"Generated {symbol} (synthetic) -- deliberately corrupted ARIMA metrics "
                  f"to demonstrate the mismatch-detection view")
        else:
            print(f"Generated {symbol} (synthetic)")

    manifest = {
        "note": "SYNTHETIC DEMO DATA - generated by scripts/generate_sample_data.py, "
        "not exported from the real ForecastPH pipeline.",
        "exported_symbols": exported,
        "seed": arguments.seed,
    }
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"\nWrote {len(exported)} synthetic company exports to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
