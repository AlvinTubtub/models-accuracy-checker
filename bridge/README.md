# Bridge export — schema reference

`export_evaluations.py` calls the original repo's own
`CompanyEvaluation.as_dict()` (from `backend/src/evaluation/evaluator.py`)
and writes it to `<SYMBOL>.json`, with two extra fields (`sector`, `name`)
added for convenience. Nothing else is changed, added, or recomputed.

## Per-company JSON shape

```jsonc
{
  "symbol": "ALI",
  "sector": "Property",           // added by the bridge script
  "name": "Ayala Land, Inc.",      // added by the bridge script

  "split": {
    "evaluation_proportion": 0.15,
    "development_pairs": 1388,
    "evaluation_pairs": 245,
    "evaluation_start": "2025-09-15",
    "evaluation_end": "2026-09-11"
  },

  "mase_denominator": 0.1234567,   // mean(|diff(development Close))|,
                                    // shared by all four models

  "metrics": {
    "lag_reg": {"rmse": 0.45, "mae": 0.32, "mase": 0.85, "r2": 0.96, "observations": 245},
    "arima":   {"rmse": 0.48, "mae": 0.35, "mase": 0.92, "r2": 0.95, "observations": 245},
    "lstm":    {"rmse": 0.52, "mae": 0.38, "mase": 1.01, "r2": 0.94, "observations": 245},
    "naive":   {"rmse": 0.55, "mae": 0.40, "mase": 1.00, "r2": 0.93, "observations": 245}
  },

  "principal_ranking": {
    "criterion": "rmse",
    "best_model": "lag_reg",       // one of lag_reg / arima / lstm (never naive)
    "ranked_models": [
      {"rank": 1, "model": "lag_reg", "criterion": "rmse", "value": 0.45},
      {"rank": 2, "model": "arima",   "criterion": "rmse", "value": 0.48},
      {"rank": 3, "model": "lstm",    "criterion": "rmse", "value": 0.52}
    ]
  },

  "backtest": {
    "symbol": "ALI",
    "target_dates": ["2025-09-15", "...", "2026-09-11"],   // every evaluation date
    "actual_closes": [32.5, 32.8, "..."],                   // same length, same order
    "predicted_closes_by_model": {
      "lag_reg": [32.4, 32.7, "..."],
      "arima":   [32.6, 32.9, "..."],
      "lstm":    [32.3, 32.6, "..."],
      "naive":   [32.5, 32.8, "..."]
    },
    "records": [
      {
        "symbol": "ALI", "model": "lag_reg",
        "origin_date": "2025-09-12", "target_date": "2025-09-15",
        "origin_close": 32.5, "actual_close": 32.8,
        "predicted_close": 32.7, "error": -0.1
      }
      // ... one record per model per target date (4 x evaluation_pairs total)
    ],
    "source": "chronological_out_of_sample_evaluation"
  }
}
```

The important part for the independent recheck: `backtest.records` (or
equivalently `backtest.target_dates` / `actual_closes` /
`predicted_closes_by_model`) covers the **complete** evaluation window — every
date, not a 60-session slice. `metrics` is what the original pipeline
*reported*; the recheck repo treats it only as "reported_metrics" to diff
against, never as ground truth on its own.

## Why `sector`/`name` are added but nothing else

The independent repo needs a way to group companies without re-implementing
the original's internal database. Sector and display name are static reference
facts, not computed results, so copying them across doesn't compromise
independence the way reusing a metric formula would.

## Multiple runs over time

If you re-run Fresh Training quarterly and want to track whether the
recheck's verdict is stable across retrains, export into a dated
subdirectory instead of overwriting:

```bash
python export_evaluations.py --output ../../evaluation_export/2026-11-28
```

The recheck repo's loader (`recheck/loader.py`) reads whatever directory
of `<SYMBOL>.json` files you point it at.
