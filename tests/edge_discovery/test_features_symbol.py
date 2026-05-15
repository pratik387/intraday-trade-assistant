import pandas as pd
import numpy as np
from tools.edge_discovery.types import Event
from tools.edge_discovery.features.symbol_features import SymbolFeaturesTierA


def _make_session_bars(start_date: str = "2024-06-15") -> pd.DataFrame:
    """75 5m bars from 09:15 to 15:30 IST."""
    idx = pd.date_range(f"{start_date} 09:15:00", periods=75, freq="5min")
    return pd.DataFrame({
        "date": idx,
        "open": np.linspace(100.0, 110.0, 75),
        "high": np.linspace(100.5, 110.5, 75),
        "low": np.linspace(99.5, 109.5, 75),
        "close": np.linspace(100.2, 110.2, 75),
        "volume": np.full(75, 5000),
    })


def test_symbol_features_emits_expected_names():
    bars = _make_session_bars()
    event = Event(symbol="X", event_time=bars["date"].iloc[10], metadata={})
    meta = {"cap_segment": "small_cap", "mis_leverage": 5.0}
    sf = SymbolFeaturesTierA()
    pdh, pdl = 99.0, 95.0
    prior_close = 99.5
    out = sf.compute(event, bars, symbol_meta=meta, pdh=pdh, pdl=pdl, prior_close=prior_close)
    for k in (
        "cap_segment", "adv_bucket", "mis_leverage",
        "dist_from_pdh_pct", "dist_from_pdl_pct",
        "prior_session_pct_change", "gap_pct",
        "bar_range_pct", "bar_body_pct",
        "bar_upper_wick_ratio", "bar_lower_wick_ratio",
    ):
        assert k in out, f"missing feature: {k}"


def test_dist_from_pdh_for_above_pdh_entry():
    bars = _make_session_bars()
    event = Event(symbol="X", event_time=bars["date"].iloc[10], metadata={})
    meta = {"cap_segment": "small_cap"}
    sf = SymbolFeaturesTierA()
    entry_price = float(bars["close"].iloc[10])
    pdh = entry_price * 0.98  # entry is ~2% above PDH
    out = sf.compute(event, bars, symbol_meta=meta, pdh=pdh, pdl=pdh * 0.95, prior_close=pdh * 0.99)
    assert abs(out["dist_from_pdh_pct"] - (1/0.98 - 1)) < 1e-6


def test_gap_pct_when_open_is_above_prior_close():
    bars = _make_session_bars()
    bars.loc[0, "open"] = 105.0
    event = Event(symbol="X", event_time=bars["date"].iloc[0], metadata={})
    meta = {"cap_segment": "small_cap"}
    sf = SymbolFeaturesTierA()
    out = sf.compute(event, bars, symbol_meta=meta, pdh=104.5, pdl=99.0, prior_close=100.0)
    assert abs(out["gap_pct"] - 0.05) < 1e-9


def test_bar_upper_wick_ratio_correctly_signed():
    bars = _make_session_bars()
    bars.loc[10, "open"] = 100.0
    bars.loc[10, "high"] = 102.0
    bars.loc[10, "low"] = 99.5
    bars.loc[10, "close"] = 100.5
    event = Event(symbol="X", event_time=bars["date"].iloc[10], metadata={})
    meta = {"cap_segment": "small_cap"}
    sf = SymbolFeaturesTierA()
    out = sf.compute(event, bars, symbol_meta=meta, pdh=99.0, pdl=95.0, prior_close=99.5)
    # range = 2.5, upper wick = 102 - max(100, 100.5) = 1.5 → ratio 0.6
    assert abs(out["bar_upper_wick_ratio"] - 0.6) < 1e-9
