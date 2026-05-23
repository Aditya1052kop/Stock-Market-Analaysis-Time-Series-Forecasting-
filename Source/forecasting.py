"""
models/forecasting.py
ARIMA-based time series forecasting for stock closing prices.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def check_stationarity(series: pd.Series) -> dict:
    """
    Run Augmented Dickey-Fuller test.
    Returns p-value and is_stationary flag.
    """
    result = adfuller(series.dropna())
    return {
        "adf_stat":     round(result[0], 4),
        "p_value":      round(result[1], 4),
        "is_stationary": result[1] < 0.05,
    }


def run_arima(series: pd.Series,
              order: tuple = (1, 1, 1),
              steps: int = 10) -> dict:
    """
    Fit ARIMA and return forecast.

    Parameters
    ----------
    series : pd.Series  — Close price series with DatetimeIndex
    order  : (p, d, q)  — ARIMA order
    steps  : int        — number of days to forecast

    Returns
    -------
    dict with keys: forecast_values, conf_int, model_summary, aic, bic
    """
    series = series.dropna().asfreq("B").ffill()   # business-day frequency

    try:
        model     = ARIMA(series, order=order)
        model_fit = model.fit()

        forecast_result = model_fit.get_forecast(steps=steps)
        forecast_vals   = forecast_result.predicted_mean
        conf_int        = forecast_result.conf_int()

        # Build a future date index
        last_date    = series.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1),
                                      periods=steps)
        forecast_vals.index = future_dates
        conf_int.index      = future_dates

        return {
            "success":         True,
            "forecast_values": forecast_vals,
            "conf_int":        conf_int,
            "aic":             round(model_fit.aic, 2),
            "bic":             round(model_fit.bic, 2),
            "model_summary":   model_fit.summary().as_text(),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def auto_order(series: pd.Series) -> tuple:
    """
    Very simple heuristic to choose (p, d, q):
    - d=1 if series is not stationary, else d=0
    - p=1, q=1 always (safe default)
    """
    stat = check_stationarity(series.dropna())
    d = 0 if stat["is_stationary"] else 1
    return (1, d, 1)
