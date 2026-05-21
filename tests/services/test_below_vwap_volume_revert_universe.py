"""Universe builder for below_vwap_volume_revert_long.

Static cap-segment + MIS-eligible filter using nse_all.json metadata.
Returns symbols matching cell_lock_cap_segment="unknown".
"""
from datetime import date

import pandas as pd
import pytest

from services.setup_universe import below_vwap_volume_revert_long_universe


@pytest.fixture
def daily_dict():
    """Three symbols with the daily history shape expected by the builder."""
    idx = pd.date_range("2026-04-01", "2026-05-19", freq="B")
    return {
        "NSE:UNKNOWNSTOCK": pd.DataFrame({
            "close": [100.0] * len(idx),
            "high": [101.0] * len(idx),
            "low": [99.0] * len(idx),
            "volume": [200_000] * len(idx),
        }, index=idx),
        "NSE:RELIANCE": pd.DataFrame({
            "close": [2500.0] * len(idx),
            "high": [2510.0] * len(idx),
            "low": [2490.0] * len(idx),
            "volume": [10_000_000] * len(idx),
        }, index=idx),
        "NSE:LOWVOLSME": pd.DataFrame({
            "close": [50.0] * len(idx),
            "high": [51.0] * len(idx),
            "low": [49.0] * len(idx),
            "volume": [5_000] * len(idx),  # below 50k daily-vol floor
        }, index=idx),
    }


def test_universe_returns_only_unknown_cap_symbols(monkeypatch, daily_dict):
    """Builder includes UNKNOWNSTOCK (cap=unknown), excludes RELIANCE (large_cap)."""
    from services import symbol_metadata

    cap_map = {
        "NSE:UNKNOWNSTOCK": "unknown",
        "NSE:RELIANCE": "large_cap",
        "NSE:LOWVOLSME": "unknown",
    }
    mis_map = {
        "NSE:UNKNOWNSTOCK": {"mis_enabled": True, "mis_leverage": 5.0},
        "NSE:RELIANCE": {"mis_enabled": True, "mis_leverage": 5.0},
        "NSE:LOWVOLSME": {"mis_enabled": True, "mis_leverage": 5.0},
    }
    monkeypatch.setattr(symbol_metadata, "get_cap_segment",
                        lambda s: cap_map.get(s, "unknown"))
    monkeypatch.setattr(symbol_metadata, "get_mis_info",
                        lambda s: mis_map.get(s, {"mis_enabled": False, "mis_leverage": None}))

    cfg = {
        "cell_lock_cap_segment": "unknown",
        "min_daily_avg_volume": 50_000,
        "min_trading_days_required": 30,
    }
    qual = below_vwap_volume_revert_long_universe(daily_dict, date(2026, 5, 20), cfg)
    assert "NSE:UNKNOWNSTOCK" in qual
    assert "NSE:RELIANCE" not in qual
    assert "NSE:LOWVOLSME" not in qual  # rejected by daily-volume floor
