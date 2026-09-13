from __future__ import annotations

import copy

import pytest


def _build_minimal_payload() -> dict:
    """A small, hand-built, internally-consistent export payload used across tests."""

    target_dates = ["2026-09-08", "2026-09-09", "2026-09-10"]
    actual_closes = [100.0, 102.0, 101.0]
    origin_closes = [99.0, 100.0, 102.0]

    predicted_closes_by_model = {
        "lag_reg": [100.5, 101.5, 101.5],
        "arima": [99.5, 103.0, 100.0],
        "lstm": [100.2, 101.8, 101.2],
        "naive": list(origin_closes),
    }

    def _metrics_for(model: str) -> dict:
        predicted = predicted_closes_by_model[model]
        errors = [p - a for p, a in zip(predicted, actual_closes)]
        mae = sum(abs(e) for e in errors) / len(errors)
        rmse = (sum(e * e for e in errors) / len(errors)) ** 0.5
        mase_denominator = 1.5
        mase = mae / mase_denominator
        mean_actual = sum(actual_closes) / len(actual_closes)
        total_ss = sum((a - mean_actual) ** 2 for a in actual_closes)
        residual_ss = sum(e * e for e in errors)
        r2 = 1.0 - residual_ss / total_ss if total_ss else 1.0
        return {"rmse": rmse, "mae": mae, "mase": mase, "r2": r2, "observations": len(errors)}

    records = []
    for model, predicted in predicted_closes_by_model.items():
        for target_date, origin_close, actual_close, predicted_close in zip(
            target_dates, origin_closes, actual_closes, predicted
        ):
            records.append({
                "symbol": "TEST",
                "model": model,
                "origin_date": target_date,
                "target_date": target_date,
                "origin_close": origin_close,
                "actual_close": actual_close,
                "predicted_close": predicted_close,
                "error": predicted_close - actual_close,
            })

    return {
        "symbol": "TEST",
        "sector": "Financials",
        "name": "Test Company Inc.",
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": 100,
            "evaluation_pairs": 3,
            "evaluation_start": target_dates[0],
            "evaluation_end": target_dates[-1],
        },
        "mase_denominator": 1.5,
        "metrics": {model: _metrics_for(model) for model in predicted_closes_by_model},
        "principal_ranking": {
            "criterion": "rmse",
            "best_model": "lag_reg",
            "ranked_models": [
                {"rank": 1, "model": "lag_reg", "criterion": "rmse", "value": 0.1},
                {"rank": 2, "model": "lstm", "criterion": "rmse", "value": 0.2},
                {"rank": 3, "model": "arima", "criterion": "rmse", "value": 0.3},
            ],
        },
        "backtest": {
            "symbol": "TEST",
            "target_dates": target_dates,
            "actual_closes": actual_closes,
            "predicted_closes_by_model": predicted_closes_by_model,
            "records": records,
            "source": "chronological_out_of_sample_evaluation",
        },
    }


@pytest.fixture
def minimal_payload() -> dict:
    return copy.deepcopy(_build_minimal_payload())
