"""
Unified 5m Bar Enrichment — Single Source of Truth
====================================================

One function that computes all indicators on 5m OHLCV bars.
Used in exactly two places:
  1. Offline precomputation script (tools/precompute_5m_cache.py)
  2. Paper/live runtime enrichment (services/screener_live.py)

Backtest reads precomputed output from feather cache — zero runtime cost.
Parity is guaranteed because the precomputed data was created by this function.

Requires warmup bars (previous trading day) prepended to stabilize
rolling/EMA indicators from bar 1 of the trading day.
"""

import pandas as pd

from services.indicators.indicators import calculate_adx, calculate_rsi


def enrich_5m_bars(
    df: pd.DataFrame,
    session_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """
    Compute indicators on 5m OHLCV bars. Single source of truth.

    Args:
        df: DataFrame with columns [open, high, low, close, volume] and
            a DatetimeIndex (IST-naive). Should include previous-day warmup
            bars prepended (≥30 bars) so rolling/EMA indicators stabilize.
        session_date: If provided, trim output to only this date's bars.
            If None, return all bars (caller handles trimming).

    Returns:
        DataFrame with added columns [vwap, bb_width_proxy, adx, rsi].
        If session_date is set, only that day's bars are returned.
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # ── VWAP: cumulative from market open each day ──
    # Resets at each day boundary (standard institutional VWAP)
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    tp_vol = tp * df["volume"]
    day = df.index.date
    cum_tp_vol = tp_vol.groupby(day).cumsum()
    cum_vol = df["volume"].groupby(day).cumsum()
    df["vwap"] = cum_tp_vol / cum_vol.replace(0, float("nan"))
    df["vwap"] = df["vwap"].ffill().fillna(tp)

    # ── BB width proxy: rolling(20) std / SMA  (standard Bollinger Band width) ──
    sma_20 = df["close"].rolling(20, min_periods=5).mean()
    std_20 = df["close"].rolling(20, min_periods=5).std(ddof=0)
    df["bb_width_proxy"] = (std_20 / sma_20).fillna(0.0)

    # ── ADX(14): Wilder smoothing from shared library ──
    df["adx"] = calculate_adx(df, period=14).fillna(0.0)

    # ── RSI(14): Wilder smoothing from shared library ──
    df["rsi"] = calculate_rsi(df["close"], period=14).fillna(50.0)

    # ── Trim warmup bars if session_date specified ──
    if session_date is not None:
        target = pd.Timestamp(session_date).date()
        df = df[df.index.date == target]

    return df
