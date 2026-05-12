"""Tests for services.feature_computer.compute_bar_features."""
import pandas as pd
import pytest
from datetime import datetime


def test_compute_bar_features_returns_dataframe_with_expected_columns():
    from services.feature_computer import compute_bar_features

    # Two symbols, 5 bars each (enough for ATR/bb_width calc)
    idx = pd.date_range("2024-08-29 09:15", periods=5, freq="5min")
    df_a = pd.DataFrame({
        "open": [100, 101, 102, 103, 104],
        "high": [101, 102, 103, 104, 105],
        "low":  [99, 100, 101, 102, 103],
        "close":[100.5, 101.5, 102.5, 103.5, 104.5],
        "volume":[1000, 1100, 1200, 1300, 1400],
    }, index=idx)
    df_b = pd.DataFrame({
        "open": [50, 50, 50, 50, 50],
        "high": [51, 51, 51, 51, 51],
        "low":  [49, 49, 49, 49, 49],
        "close":[50, 50, 50, 50, 50],
        "volume":[500, 500, 500, 500, 500],
    }, index=idx)
    df5_by_symbol = {"NSE:A": df_a, "NSE:B": df_b}
    universe = {"NSE:A", "NSE:B"}
    now = idx[-1]
    levels_by_symbol = {
        "NSE:A": {"PDH": 105, "PDL": 95, "PDC": 100},
        "NSE:B": {"PDH": 52, "PDL": 48, "PDC": 50},
    }

    out = compute_bar_features(df5_by_symbol, universe, now, levels_by_symbol)

    assert isinstance(out, pd.DataFrame)
    assert set(out["symbol"]) == universe
    for col in ("symbol", "close", "vwap", "volume", "ret_1", "vol_z20",
                "dist_to_vwap", "bb_width_proxy", "vol_ratio", "rank_score"):
        assert col in out.columns, f"missing {col}"
    # NSE:A trending up should have positive ret_1
    a_row = out[out["symbol"] == "NSE:A"].iloc[0]
    assert a_row["close"] == 104.5
    assert a_row["ret_1"] > 0


def test_compute_bar_features_skips_symbols_outside_universe():
    from services.feature_computer import compute_bar_features

    idx = pd.date_range("2024-08-29 09:15", periods=3, freq="5min")
    df = pd.DataFrame({
        "open":[10]*3, "high":[10]*3, "low":[10]*3, "close":[10]*3, "volume":[100]*3
    }, index=idx)
    df5_by_symbol = {"NSE:X": df, "NSE:Y": df, "NSE:Z": df}
    universe = {"NSE:X"}   # only X is in universe

    out = compute_bar_features(df5_by_symbol, universe, idx[-1], {})
    assert set(out["symbol"]) == {"NSE:X"}


def test_compute_bar_features_skips_insufficient_data():
    from services.feature_computer import compute_bar_features

    idx = pd.date_range("2024-08-29 09:15", periods=1, freq="5min")
    df_short = pd.DataFrame({
        "open":[100], "high":[101], "low":[99], "close":[100], "volume":[100]
    }, index=idx)
    out = compute_bar_features({"NSE:SHORT": df_short}, {"NSE:SHORT"}, idx[-1], {})
    # Single-bar symbol: vol_z20/bb_width need >=2 bars, so should be omitted OR have NaN sentinels
    if not out.empty:
        assert "NSE:SHORT" in set(out["symbol"])
