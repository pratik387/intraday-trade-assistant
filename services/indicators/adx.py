"""
Technical Indicators - Centralized Implementation
==================================================

Single source of truth for technical indicator calculations across the entire codebase.
Avoids duplicate implementations with potential inconsistencies.

All indicators use industry-standard methods:
- ATR: Wilder's smoothing (alpha=1/period)
- ADX: Wilder's smoothing (alpha=1/period)
- RSI: Wilder's smoothing (alpha=1/period)
- MACD: EMA-based (span smoothing)
- EMA: Standard exponential moving average

Usage:
    from services.indicators.adx import (
        calculate_atr,
        calculate_adx, calculate_adx_with_di,
        calculate_rsi, calculate_macd, calculate_ema
    )

    # ADX
    adx = calculate_adx(df, period=14)
    adx, plus_di, minus_di = calculate_adx_with_di(df, period=14)

    # RSI
    rsi = calculate_rsi(close_series, period=14)

    # MACD
    macd_dict = calculate_macd(close_series, fast=12, slow=26, signal=9)

    # EMA
    ema = calculate_ema(series, span=20)
"""

import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """
    Calculate ATR (Average True Range) using Wilder's smoothing.

    Uses Wilder's smoothing (alpha=1/period) which is the industry standard.

    Args:
        df: DataFrame with columns ['high', 'low', 'close']
        period: ATR period (default 14, Wilder's standard)

    Returns:
        float: Scalar ATR value (last value from series)

    Example:
        >>> atr = calculate_atr(df, period=14)
        >>> print(f"ATR: {atr:.2f}")
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    # True Range calculation
    # TR = max(H-L, |H-PC|, |L-PC|) where PC = previous close
    prev_close = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    atr_series = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return float(atr_series.iloc[-1])


def calculate_atr_series(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ATR (Average True Range) as a full series using Wilder's smoothing.

    Same as calculate_atr but returns the full series instead of scalar.

    Args:
        df: DataFrame with columns ['high', 'low', 'close']
        period: ATR period (default 14)

    Returns:
        pd.Series: ATR values for entire DataFrame

    Example:
        >>> atr_series = calculate_atr_series(df, period=14)
        >>> print(atr_series.tail())
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()

    # True Range calculation
    prev_close = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_close), np.abs(l - prev_close)))

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    atr_series = pd.Series(tr).ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return atr_series


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate ADX (Average Directional Index) from OHLC data.

    Uses Wilder's smoothing (alpha=1/period) which is the standard method.

    Args:
        df: DataFrame with columns ['high', 'low', 'close']
        period: ADX period (default 14, Wilder's standard)

    Returns:
        pd.Series: ADX values (0-100 scale)
    """
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # Directional Movement
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    # True Range
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

    # Directional Index (DX)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)

    # ADX (smoothed DX)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx.bfill()


def calculate_adx_with_di(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate ADX along with +DI and -DI for regime detection.

    Args:
        df: DataFrame with columns ['high', 'low', 'close']
        period: ADX period (default 14)

    Returns:
        tuple: (adx, plus_di, minus_di) as pd.Series
    """
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)

    # Directional Movement
    up = h.diff()
    down = -l.diff()
    plus_dm = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)

    # True Range
    tr = pd.concat([
        (h - l),
        (h - c.shift(1)).abs(),
        (l - c.shift(1)).abs()
    ], axis=1).max(axis=1)

    # Wilder's smoothing
    alpha = 1.0 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

    # DX and ADX
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0)
    adx = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return adx.bfill(), plus_di.fillna(0), minus_di.fillna(0)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) using Wilder's smoothing.

    Uses Wilder's smoothing (alpha=1/period) which is the standard method.

    Args:
        series: Price series (typically close prices)
        period: RSI period (default 14, Wilder's standard)

    Returns:
        pd.Series: RSI values (0-100 scale)
    """
    s = series.astype(float)
    delta = s.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / period
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    # RSI calculation
    rs = avg_gain / avg_loss.replace(0.0, pd.NA)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    """
    Calculate EMA (Exponential Moving Average).

    Args:
        series: Price series (typically close prices)
        span: EMA period (e.g., 20, 50, 200)

    Returns:
        pd.Series: EMA values
    """
    return series.astype(float).ewm(span=span, adjust=False).mean()


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Uses standard EMA-based calculation.

    Args:
        series: Price series (typically close prices)
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        dict: {
            'macd': MACD line (fast EMA - slow EMA),
            'signal': Signal line (EMA of MACD),
            'histogram': MACD histogram (MACD - signal)
        }
    """
    s = series.astype(float)
    ema_fast = s.ewm(span=fast, adjust=False).mean()
    ema_slow = s.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }
