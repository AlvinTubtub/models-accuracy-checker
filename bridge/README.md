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
    "evaluation_start": "2025-XX-XX",
    "evaluation_end": "2026-09-11"
  },

  "mase_denominator": 0.1234567,   // mean(|diff(development Close))|,
                                    // shared by all four models

  "metrics": {
    "lag_reg": {"rmse": ..., "mae": ..., "mase": ..., "r2": ..., "observations": 245},
    "arima":   {"rmse": ..., "mae": ..., "mase": ..., "r2": ..., "observations": 245},
    "lstm":    {"rmse": ..., "mae": ..., "mase": ..., "r2": ..., "observations": 245},
    "naive":   {"rmse": ..., "mae": ..., "mase": ..., "r2": ..., "observations": 245}
  },

  "principal_ranking": {
    "criterion": "rmse",
    "best_model": "lag_reg",       // one of lag_reg / arima / lstm (never naive)
    "ranked_models": [
      {"rank": 1, "model": "lag_reg", "criterion": "rmse", "value": ...},
      {"rank": 2, "model": "arima",   "criterion": "rmse", "value": ...},
      {"rank": 3, "model": "lstm",    "criterion": "rmse", "value": ...}
    ]
  },

  "backtest": {
    "symbol": "ALI",
    "target_dates": ["2025-XX-XX", "...", "2026-09-11"],   // every evaluation date
    "actual_closes": [..., ...],                            // same length, same order
    "predicted_closes_by_model": {
      "lag_reg": [...], "arima": [...], "lstm": [...], "naive": [...]
    },
    "records": [
      {
        "symbol": "ALI", "model": "lag_reg",
        "origin_date": "...", "target_date": "...",
        "origin_close": ..., "actual_close": ...,
        "predicted_close": ..., "error": ...   // predicted_close - actual_close
      },
      // ... one record per model per target date (4 x evaluation_pairs total)
    ],
    "source": "chronological_out_of_sample_evaluation"
  }
}
```

The important part for the independent recheck: `backtest.records` (or
equivalently `backtest.target_dates` / `actual_closes` /
`predicted_closes_by_model`) is the **complete** evaluation window — every
date, not a 60-session slice. `metrics` is what the original pipeline
*reported*; the recheck repo treats it only as "reported_metrics" to diff
against, never as ground truth on its own.

## Why `sector`/`name` are added but nothing else

The independent repo needs a way to group companies (e.g. "does LSTM do
better in Mining & Oil than in Financials?") without re-implementing the
original's `config/companies.py` lookup logic. Sector and display name are
static reference facts, not computed results, so copying them across
doesn't compromise independence the way reusing a metric formula would.

## Multiple runs over time

If you re-run Fresh Training quarterly and want to track whether the
recheck's verdict is stable across retrains, export into a dated
subdirectory instead of overwriting, e.g.:

```
python export_evaluations.py --output ../../evaluation_export/2026-11-28
```

The recheck repo's loader (`recheck/loader.py`) just reads whatever
directory of `<SYMBOL>.json` files you point it at, so this works without
any change on that side.
