"""Feature engineering without future leakage.

Computes technical indicators (RSI, EMA, MACD, Bollinger Bands) and fold-local
PACF significant lag selections using past observations strictly up to time t
to predict Close at t+1.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using exponential moving averages."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """MACD line and Signal line."""
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    macd_signal = compute_ema(macd_line, signal)
    return macd_line, macd_signal


def compute_bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: Middle, Upper, Lower, and Percent B."""
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / middle.replace(0, np.nan)
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return middle, upper, lower, percent_b.fillna(0.5)


def select_pacf_lags(
    close_series: pd.Series | np.ndarray,
    max_lag: int = 20,
    significance_z: float = 1.96,
) -> list[int]:
    """Identify significant PACF lags on development data (lag 1 to max_lag)."""
    n = len(close_series)
    if n < max_lag + 5:
        return [1]

    threshold = significance_z / math.sqrt(n)
    pacf_values = pacf(close_series, nlags=max_lag, method="ols")

    significant_lags = []
    for lag in range(1, len(pacf_values)):
        if abs(pacf_values[lag]) > threshold:
            significant_lags.append(lag)

    # Always ensure at least lag 1 is present
    if 1 not in significant_lags:
        significant_lags.insert(0, 1)
    return sorted(significant_lags)


def create_feature_dataframe(
    df: pd.DataFrame,
    pacf_lags: list[int] | None = None,
) -> pd.DataFrame:
    """Create feature matrix X and target y (Close_{t+1}) from raw OHLCV.

    Features at row t use prices up to row t.
    Target at row t is Close_{t+1}.
    """
    df_sorted = df.sort_values("Date").reset_index(drop=True).copy()
    close = df_sorted["Close"]

    features = pd.DataFrame(index=df_sorted.index)

    # 1. Price and Return Lags
    lags = pacf_lags or [1, 2, 3, 5]
    for lag in lags:
        features[f"lag_close_{lag}"] = close.shift(lag - 1)
        # Log return over lag
        features[f"return_lag_{lag}"] = (close.shift(lag - 1) / close.shift(lag) - 1.0).fillna(0)

    # 2. Technical Indicators
    features["rsi_14"] = compute_rsi(close, 14)
    macd_line, macd_signal = compute_macd(close, 12, 26, 9)
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    _, upper_bb, lower_bb, pct_b = compute_bollinger_bands(close, 20, 2.0)
    features["bb_pct_b"] = pct_b

    # Current Close (origin close)
    features["origin_close"] = close
    features["Date"] = df_sorted["Date"]

    # Target: next day close
    features["target_close"] = close.shift(-1)
    features["target_date"] = df_sorted["Date"].shift(-1)

    # Drop the very last row because target_close is NaN
    clean = features.dropna(subset=["target_close"]).copy()
    # Forward fill or zero fill any indicator warmup NaNs
    clean = clean.bfill().fillna(0.0)
    return clean
