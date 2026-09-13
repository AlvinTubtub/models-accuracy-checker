"""Unit tests for feature engineering, splitting, and model training."""

import numpy as np
import pandas as pd
import pytest

from training.features import compute_rsi, create_feature_dataframe, select_pacf_lags
from training.models.arima import train_and_predict_arima
from training.models.lir import train_and_predict_lir
from training.models.lstm import train_and_predict_lstm
from training.models.naive import generate_naive_predictions
from training.split import create_split_plan


def _make_dummy_ohlcv(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    np.random.seed(42)
    closes = 50.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "Date": dates,
        "Open": closes * 0.99,
        "High": closes * 1.01,
        "Low": closes * 0.98,
        "Close": closes,
        "Volume": np.random.randint(1000, 50000, size=n),
    })


def test_compute_rsi():
    series = pd.Series([10.0, 11.0, 12.0, 11.5, 13.0, 14.0, 13.5, 15.0, 16.0, 17.0])
    rsi = compute_rsi(series, period=5)
    assert len(rsi) == len(series)
    assert not rsi.isna().any()
    assert (rsi >= 0.0).all() and (rsi <= 100.0).all()


def test_select_pacf_lags():
    np.random.seed(42)
    series = np.cumsum(np.random.randn(200))
    lags = select_pacf_lags(series, max_lag=10)
    assert 1 in lags  # Always includes at least lag 1
    assert all(1 <= lag <= 10 for lag in lags)


def test_create_feature_dataframe():
    df = _make_dummy_ohlcv(100)
    feat_df = create_feature_dataframe(df)
    assert "target_close" in feat_df.columns
    assert "origin_close" in feat_df.columns
    assert "rsi_14" in feat_df.columns
    assert len(feat_df) == 99  # last row dropped since target_close is NaN


def test_create_split_plan():
    df = _make_dummy_ohlcv(100)
    plan = create_split_plan(df, "TEST", evaluation_proportion=0.15)
    assert plan.eval_count == 15
    assert plan.dev_count == 85
    assert plan.mase_denominator > 0.0


def test_naive_model():
    origins = [10.0, 11.5, 12.0]
    preds = generate_naive_predictions(origins)
    assert preds == [10.0, 11.5, 12.0]


def test_lir_training():
    np.random.seed(42)
    X = np.random.randn(60, 5)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 10.0
    X_eval = np.random.randn(10, 5)

    preds, best_alpha = train_and_predict_lir(X, y, X_eval, cv_splits=3)
    assert len(preds) == 10
    assert best_alpha > 0.0


def test_arima_training():
    np.random.seed(42)
    dev = 30.0 + np.cumsum(np.random.randn(80) * 0.2)
    eval_series = 30.0 + np.cumsum(np.random.randn(15) * 0.2)

    preds, order = train_and_predict_arima(dev, eval_series, order=(1, 1, 0))
    assert len(preds) == 15
    assert order == (1, 1, 0)


def test_lstm_training():
    dev = np.linspace(20.0, 30.0, 60, dtype=np.float32)
    eval_series = np.linspace(30.0, 32.0, 10, dtype=np.float32)

    preds, meta = train_and_predict_lstm(
        dev, eval_series, lookback=5, hidden_size=8, epochs=3, device="cpu"
    )
    assert len(preds) == 10
    assert meta["epochs"] == 3
