"""
DataFrame Validation Utilities
===============================

Centralized utilities for common DataFrame validation patterns.

This eliminates the repeated pattern of checking:
  if df is not None and not df.empty and len(df) >= min_rows:
      ...

Used by: screener_live.py, planner_internal.py, exit_executor.py, features.py, levels.py, etc.

Usage:
    from utils.dataframe_utils import validate_df, has_column, validate_df_columns

    # Basic validation
    if validate_df(df, min_rows=5):
        # Process df safely
        ...

    # Column existence
    if has_column(df, "volume"):
        vol_ratio = df["volume"].iloc[-1] / df["volume"].mean()

    # Multiple columns
    if validate_df_columns(df, ["high", "low", "close"]):
        # All required columns present
        ...
"""

import pandas as pd
from typing import List, Optional


def validate_df(df: pd.DataFrame, min_rows: int = 1) -> bool:
    """
    Validate DataFrame is not None, not empty, and has minimum rows.

    This replaces the common pattern:
        if df is not None and not df.empty and len(df) >= min_rows:

    Args:
        df: DataFrame to validate
        min_rows: Minimum number of rows required (default 1)

    Returns:
        bool: True if DataFrame is valid, False otherwise

    Examples:
        >>> if validate_df(df5, min_rows=5):
        >>>     # Process df5 safely
        >>>     close = df5["close"].iloc[-1]

        >>> if validate_df(daily_df, min_rows=10):
        >>>     # Compute daily indicators
        >>>     ema = daily_df["close"].ewm(span=20).mean()
    """
    return df is not None and not df.empty and len(df) >= min_rows


def has_column(df: pd.DataFrame, column: str) -> bool:
    """
    Safe column existence check.

    This replaces the pattern:
        if "volume" in df.columns:

    Args:
        df: DataFrame to check
        column: Column name to check for

    Returns:
        bool: True if DataFrame exists and has the column, False otherwise

    Examples:
        >>> if has_column(df, "volume"):
        >>>     vol_ratio = df["volume"].iloc[-1] / df["volume"].mean()

        >>> if has_column(df, "vwap"):
        >>>     distance_from_vwap = abs(close - df["vwap"].iloc[-1])
    """
    return df is not None and column in df.columns


def validate_df_columns(df: pd.DataFrame, required: List[str]) -> bool:
    """
    Validate DataFrame has all required columns.

    This replaces manual checking of multiple columns.

    Args:
        df: DataFrame to validate
        required: List of required column names

    Returns:
        bool: True if DataFrame exists and has ALL required columns, False otherwise

    Examples:
        >>> if validate_df_columns(df, ["high", "low", "close"]):
        >>>     # Compute ATR safely
        >>>     tr = max(h - l, abs(h - prev_close), abs(l - prev_close))

        >>> if validate_df_columns(df, ["open", "high", "low", "close", "volume"]):
        >>>     # Full OHLCV data available
        >>>     ...
    """
    if df is None or df.empty:
        return False
    return all(col in df.columns for col in required)


def validate_df_range(df: pd.DataFrame, column: str, min_val: Optional[float] = None, max_val: Optional[float] = None) -> bool:
    """
    Validate DataFrame column values are within a range.

    Useful for filtering outliers or checking data quality.

    Args:
        df: DataFrame to validate
        column: Column name to check
        min_val: Minimum allowed value (inclusive), None for no minimum
        max_val: Maximum allowed value (inclusive), None for no maximum

    Returns:
        bool: True if all values in column are within range, False otherwise

    Examples:
        >>> if validate_df_range(df, "volume", min_val=100):
        >>>     # All volume bars >= 100
        >>>     ...

        >>> if validate_df_range(df, "close", min_val=10, max_val=10000):
        >>>     # Price within reasonable range
        >>>     ...
    """
    if not has_column(df, column):
        return False

    series = df[column]

    if min_val is not None and (series < min_val).any():
        return False

    if max_val is not None and (series > max_val).any():
        return False

    return True


def safe_get_last(df: pd.DataFrame, column: str, default=None):
    """
    Safely get the last value from a DataFrame column.

    This replaces:
        value = df[column].iloc[-1] if not df.empty else default

    Args:
        df: DataFrame to get value from
        column: Column name
        default: Default value if DataFrame invalid or column missing

    Returns:
        Last value from column, or default if unavailable

    Examples:
        >>> close = safe_get_last(df5, "close", default=0.0)
        >>> volume = safe_get_last(df5, "volume", default=1.0)
        >>> vwap = safe_get_last(sess, "vwap", default=close)
    """
    if not has_column(df, column) or df.empty:
        return default

    try:
        value = df[column].iloc[-1]
        return value if pd.notna(value) else default
    except (IndexError, KeyError):
        return default


def filter_valid_rows(df: pd.DataFrame, required_columns: List[str]) -> pd.DataFrame:
    """
    Filter DataFrame to rows where all required columns have non-null values.

    This replaces:
        df = df[df["high"].notna() & df["low"].notna() & df["close"].notna()]

    Args:
        df: DataFrame to filter
        required_columns: List of columns that must be non-null

    Returns:
        Filtered DataFrame with only valid rows

    Examples:
        >>> # Filter to rows with valid OHLC data
        >>> df_clean = filter_valid_rows(df, ["open", "high", "low", "close"])

        >>> # Filter to rows with price and volume
        >>> df_liquid = filter_valid_rows(df, ["close", "volume"])
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Build combined mask
    mask = pd.Series(True, index=df.index)
    for col in required_columns:
        if col in df.columns:
            mask &= df[col].notna()

    return df[mask]
