"""Unit tests for metrics comparison and diffing."""

from recheck.compare import compare_company
from recheck.metrics import compute_metrics
from recheck.schema import parse_company_export


def test_compare_company_exact_match():
    # Build a consistent payload where reported metrics match calculated metrics
    dates = ["2026-09-01", "2026-09-02", "2026-09-03"]
    actuals = [10.0, 12.0, 14.0]
    preds = {
        "lag_reg": [10.5, 12.2, 13.8],
        "arima": [10.1, 11.9, 14.2],
        "lstm": [9.9, 12.1, 14.1],
        "naive": [10.0, 10.0, 12.0],
    }
    mase_denom = 1.5

    reported_metrics = {}
    for m, p in preds.items():
        res = compute_metrics(actuals, p, mase_denominator=mase_denom)
        reported_metrics[m] = res.as_dict()

    payload = {
        "symbol": "BPI",
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": 50,
            "evaluation_pairs": 3,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-03",
        },
        "mase_denominator": mase_denom,
        "metrics": reported_metrics,
        "principal_ranking": {"best_model": "arima"},
        "backtest": {
            "target_dates": dates,
            "actual_closes": actuals,
            "predicted_closes_by_model": preds,
        },
    }

    export = parse_company_export(payload)
    comparisons = compare_company(export, tolerance=1e-6)

    assert len(comparisons) == 4
    for c in comparisons:
        assert c.within_tolerance is True
        assert c.max_abs_diff < 1e-6


def test_compare_company_detects_discrepancy():
    dates = ["2026-09-01", "2026-09-02"]
    actuals = [10.0, 12.0]
    preds = {
        "lag_reg": [10.0, 12.0],
        "arima": [10.0, 12.0],
        "lstm": [10.0, 12.0],
        "naive": [10.0, 10.0],
    }
    mase_denom = 1.0

    reported_metrics = {}
    for m, p in preds.items():
        res = compute_metrics(actuals, p, mase_denominator=mase_denom)
        reported_metrics[m] = res.as_dict()

    # Tamper with reported LSTM RMSE to simulate a discrepancy
    reported_metrics["lstm"]["rmse"] = 99.9

    payload = {
        "symbol": "MEG",
        "split": {
            "evaluation_proportion": 0.15,
            "development_pairs": 50,
            "evaluation_pairs": 2,
            "evaluation_start": "2026-09-01",
            "evaluation_end": "2026-09-02",
        },
        "mase_denominator": mase_denom,
        "metrics": reported_metrics,
        "principal_ranking": {"best_model": "lag_reg"},
        "backtest": {
            "target_dates": dates,
            "actual_closes": actuals,
            "predicted_closes_by_model": preds,
        },
    }

    export = parse_company_export(payload)
    comparisons = compare_company(export, tolerance=1e-4)

    lstm_comp = next(c for c in comparisons if c.model == "lstm")
    assert lstm_comp.within_tolerance is False
    assert lstm_comp.max_abs_diff > 50.0
