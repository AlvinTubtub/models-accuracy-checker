"""Lag-Informed Regression (LIR) with LASSO regularized feature selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from training.split import time_series_splits

ALPHA_GRID: tuple[float, ...] = (
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
)


def train_and_predict_lir(
    X_dev: np.ndarray,
    y_dev: np.ndarray,
    X_eval: np.ndarray,
    *,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
    cv_splits: int = 5,
    random_state: int = 42,
) -> tuple[list[float], float]:
    """Train Lag-Informed Regression via LASSO with chronological CV.

    Returns:
        (predicted_eval_closes, best_alpha)
    """
    n_dev = len(X_dev)
    best_alpha = alpha_grid[0]
    best_cv_rmse = float("inf")

    # 1. Hyperparameter tuning across alpha grid using expanding TimeSeriesSplit
    for alpha in alpha_grid:
        fold_rmses = []
        for train_idx, val_idx in time_series_splits(n_dev, n_splits=cv_splits):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_dev[train_idx])
            y_tr = y_dev[train_idx]

            X_va = scaler.transform(X_dev[val_idx])
            y_va = y_dev[val_idx]

            model = Lasso(alpha=alpha, max_iter=100_000, tol=1e-7, random_state=random_state)
            model.fit(X_tr, y_tr)

            preds = model.predict(X_va)
            rmse = float(np.sqrt(np.mean((y_va - preds) ** 2)))
            fold_rmses.append(rmse)

        mean_rmse = float(np.mean(fold_rmses))
        if mean_rmse < best_cv_rmse:
            best_cv_rmse = mean_rmse
            best_alpha = alpha

    # 2. Final development refit using best alpha
    final_scaler = StandardScaler()
    X_dev_scaled = final_scaler.fit_transform(X_dev)
    final_model = Lasso(alpha=best_alpha, max_iter=100_000, tol=1e-7, random_state=random_state)
    final_model.fit(X_dev_scaled, y_dev)

    # 3. Out-of-sample prediction
    X_eval_scaled = final_scaler.transform(X_eval)
    eval_preds = final_model.predict(X_eval_scaled)

    return [round(float(p), 4) for p in eval_preds], best_alpha
