# services/metrics_intraday.py
"""
Intraday Metrics (5m bars)
--------------------------
Lightweight, fast metrics used by planner/ranker. Works on CLOSED 5-min bars.

What it computes:
- RSI(window) via ta.momentum.RSIIndicator
- ADX(window) via ta.trend.ADXIndicator
- OBV via ta.volume.OnBalanceVolumeIndicator
- Slopes of RSI/ADX over a short rolling window using linear regression
- Optional VWAP (disabled unless explicitly enabled via config)

Strict config (NO in-code defaults). Required keys in entry_config.json:
{
  "metrics_min_bars_5m": 20,
  "metrics_rsi_len": 14,
  "metrics_adx_len": 14,
  "metrics_slope_window": 3,
  "metrics_drop_last_forming": true,
  "metrics_enable_vwap": false
}
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import linregress

from config.logging_config import get_agent_logger
from config.filters_setup import load_filters
from utils.time_util import ensure_naive_ist_index, _drop_forming_last_bar

# Use the official 'ta' module classes (not function shortcuts)
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator
from ta.volume import OnBalanceVolumeIndicator

logger = get_agent_logger()


def calculate_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Rolling linear slope of a series over `window` points.
    Returns NaN where there aren't enough points or NaNs are present.
    """
    try:
        return series.rolling(window).apply(
            lambda x: linregress(range(len(x)), x).slope if np.isfinite(x).all() else np.nan,
            raw=False,
        )
    except Exception as e:
        logger.exception(f"metrics: calculate_slope error: {e}")
        # Return an all-NaN series with same index
        return pd.Series(index=series.index, dtype="float64")


def compute_intraday_breakout_score(
    df: pd.DataFrame,
    symbol: str | None = None,    # optional log context
    mode: str = "normal",         # optional log context
) -> pd.DataFrame:
    """
    Enrich a 5m OHLCV DataFrame with RSI/ADX/OBV and their slopes.
    - Assumes df has columns: ['open','high','low','close','volume'].
    - Index must be naive IST; function enforces/normalizes it.
    - Drops the *forming* last bar if config says so.

    Returns a copy of df with added columns:
      ['RSI','ADX','OBV','RSI_slope','ADX_slope', (optional) 'vwap']
    """
    try:
        if df is None or df.empty:
            return pd.DataFrame()

        cfg = load_filters()
        d = ensure_naive_ist_index(df).copy()

        # Optionally drop forming bar (strictly controlled by config)
        if bool(cfg["metrics_drop_last_forming"]) and len(d) > 2:
            try:
                d = _drop_forming_last_bar(d, "5min")
            except Exception as e:
                logger.exception(f"metrics: drop_forming_last_bar failed: {e}")

        min_bars = int(cfg["metrics_min_bars_5m"])
        if len(d) < min_bars:
            logger.warning(f"metrics: insufficient 5m bars len={len(d)} < {min_bars}")
            return pd.DataFrame()

        rsi_len = int(cfg["metrics_rsi_len"])
        adx_len = int(cfg["metrics_adx_len"])
        slope_w = int(cfg["metrics_slope_window"])
        enable_vwap = bool(cfg["metrics_enable_vwap"])

        # --- Compute indicators (robust to NaNs where data insufficient) ---
        try:
            d["RSI"] = RSIIndicator(close=d["close"], window=rsi_len, fillna=False).rsi()
        except Exception as e:
            logger.exception(f"metrics: RSI error: {e}")
            d["RSI"] = np.nan

        try:
            adx_ind = ADXIndicator(
                high=d["high"],
                low=d["low"],
                close=d["close"],
                window=adx_len,
                fillna=False,
            )
            d["ADX"] = adx_ind.adx()
        except Exception as e:
            logger.exception(f"metrics: ADX error: {e}")
            d["ADX"] = np.nan

        try:
            d["OBV"] = OnBalanceVolumeIndicator(close=d["close"], volume=d["volume"], fillna=False).on_balance_volume()
        except Exception as e:
            logger.exception(f"metrics: OBV error: {e}")
            d["OBV"] = np.nan

        # Slopes
        d["RSI_slope"] = calculate_slope(d["RSI"], slope_w)
        d["ADX_slope"] = calculate_slope(d["ADX"], slope_w)

        # Optional VWAP (disabled unless explicitly enabled)
        if enable_vwap:
            try:
                tp = (d["high"] + d["low"] + d["close"]) / 3.0
                d["VWAP"] = (tp * d["volume"]).cumsum() / d["volume"].cumsum()
            except Exception as e:
                logger.exception(f"metrics: VWAP error: {e}")
                d["VWAP"] = np.nan

        logger.info(
            f"metrics: computed "
            f"RSI_len={rsi_len}, ADX_len={adx_len}, slope_w={slope_w}, "
            f"vwap={'on' if enable_vwap else 'off'}, sym={symbol}, mode={mode}, last_ts={d.index[-1]}"
        )
        if not _validate_last_row(d, symbol):
            return pd.DataFrame()
        return d

    except Exception as e:
        logger.exception(f"metrics: compute_intraday_breakout_score fatal: {e}")
        return pd.DataFrame()
    
def _validate_last_row(d: pd.DataFrame, symbol: str | None) -> bool:
    try:
        if d is None or d.empty:
            return False
        row = d.iloc[-1]
        required = ["RSI", "ADX", "close", "volume"]
        missing = [k for k in required if k not in row.index]
        if missing:
            logger.warning(f"metrics: missing fields {missing} sym={symbol}")
            return False
        # NaN checks
        for k in required:
            val = row[k]
            if pd.isna(val):
                logger.warning(f"metrics: NaN {k} at {d.index[-1]} sym={symbol}")
                return False
        return True
    except Exception as e:
        logger.exception(f"metrics: validate_last_row error sym={symbol}: {e}")
        return False
