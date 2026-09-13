"""ARIMA model selection and walk-forward out-of-sample forecasting."""

from __future__ import annotations

import warnings
import numpy as np
from statsmodels.tsa.arima.model import ARIMA


def find_best_arima_order(
    dev_closes: np.ndarray,
    candidate_orders: list[tuple[int, int, int]] | None = None,
) -> tuple[int, int, int]:
    """Identify optimal ARIMA (p, d, q) order on development Close series by AIC."""
    if candidate_orders is None:
        candidate_orders = [
            (1, 1, 0),
            (0, 1, 1),
            (1, 1, 1),
            (2, 1, 1),
            (1, 1, 2),
            (2, 1, 2),
            (0, 1, 0),
            (1, 0, 1),
        ]

    best_aic = float("inf")
    best_order = (1, 1, 1)

    for order in candidate_orders:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(dev_closes, order=order, trend="t" if order[1] == 1 else "c")
                res = model.fit()
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = order
        except Exception:
            continue

    return best_order


def train_and_predict_arima(
    dev_closes: np.ndarray,
    eval_closes: np.ndarray,
    *,
    order: tuple[int, int, int] | None = None,
) -> tuple[list[float], tuple[int, int, int]]:
    """Fit ARIMA on development data and generate walk-forward OOS predictions.

    Uses statsmodels ARIMA with one-step-ahead rolling predictions.
    """
    if order is None:
        order = find_best_arima_order(dev_closes)

    trend = "t" if order[1] == 1 else "c"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 1. Fit parameters on development series
        base_model = ARIMA(dev_closes, order=order, trend=trend)
        fitted_res = base_model.fit()

        # 2. Walk-forward one-step-ahead forecasting across evaluation window
        full_series = np.concatenate([dev_closes, eval_closes])
        # .apply() updates the state filter across the full series without refitting params
        applied_res = fitted_res.apply(full_series, refit=False)

        # The one-step-ahead forecast for eval index t comes from prediction at t
        dev_len = len(dev_closes)
        eval_len = len(eval_closes)
        forecasts = applied_res.predict(start=dev_len, end=dev_len + eval_len - 1)

    return [round(float(p), 4) for p in forecasts], order
