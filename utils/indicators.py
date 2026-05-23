"""
utils/indicators.py
Computes technical indicators used throughout the dashboard.
"""

import pandas as pd
import numpy as np


def add_moving_averages(df: pd.DataFrame,
                        windows: list[int] = [7, 20, 50]) -> pd.DataFrame:
    """Add Simple Moving Averages for given windows."""
    df = df.copy()
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(w).mean()
    return df


def add_bollinger_bands(df: pd.DataFrame, window: int = 20,
                        num_std: float = 2.0) -> pd.DataFrame:
    """Add Bollinger Bands (upper, lower, middle)."""
    df = df.copy()
    df["BB_mid"]   = df["Close"].rolling(window).mean()
    std            = df["Close"].rolling(window).std()
    df["BB_upper"] = df["BB_mid"] + num_std * std
    df["BB_lower"] = df["BB_mid"] - num_std * std
    return df


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index."""
    df   = df.copy()
    delta = df["Close"].diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs        = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame,
             fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD line, Signal line, and Histogram."""
    df = df.copy()
    ema_fast        = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow        = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]      = ema_fast - ema_slow
    df["MACD_sig"]  = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_hist"] = df["MACD"] - df["MACD_sig"]
    return df


def add_daily_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily percentage returns and cumulative returns."""
    df = df.copy()
    df["Daily_Return"]      = df["Close"].pct_change() * 100
    df["Cumulative_Return"] = (1 + df["Close"].pct_change()).cumprod() - 1
    return df


def summary_stats(df: pd.DataFrame) -> dict:
    """Return a dict of key statistics for a stock."""
    close = df["Close"]
    stats = {
        "Current Price":   round(close.iloc[-1], 2),
        "52W High":        round(close.max(), 2),
        "52W Low":         round(close.min(), 2),
        "Avg Price":       round(close.mean(), 2),
        "Volatility (σ)":  round(close.std(), 2),
        "Avg Daily Return":round(df["Close"].pct_change().mean() * 100, 4),
        "Total Return (%)":round(
            (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100, 2
        ),
    }
    return stats
