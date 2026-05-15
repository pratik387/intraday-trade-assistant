import pandas as pd
import numpy as np
from tools.edge_discovery.types import Event
from tools.edge_discovery.outcomes.returns import ForwardReturns


def _make_bars(n: int = 60, start: str = "2024-06-15 09:15:00") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min")
    return pd.DataFrame({
        "date": idx,
        "open": np.linspace(100.0, 101.0, n),
        "high": np.linspace(100.5, 101.5, n),
        "low": np.linspace(99.5, 100.5, n),
        "close": np.linspace(100.2, 101.2, n),
        "volume": np.full(n, 1000),
    })


def test_forward_returns_5m_against_known_close_ladder():
    bars = _make_bars(n=30)
    bars["close"] = [100.0 + 0.5 * i for i in range(30)]
    bars["high"] = bars["close"] + 0.3
    bars["low"] = bars["close"] - 0.3
    event = Event(symbol="X", event_time=bars["date"].iloc[5], metadata={"direction": "long"})
    fr = ForwardReturns(horizons_minutes=[5, 15, 30], eod=False)
    out = fr.compute(event, bars)
    # entry at bar[5].close=102.5; bar[6].close=103.0 → +5m ret = 0.4878%
    assert abs(out["ret_5m"] - (103.0 - 102.5) / 102.5) < 1e-9
    assert abs(out["ret_15m"] - (104.0 - 102.5) / 102.5) < 1e-9


def test_forward_returns_short_direction_flips_sign():
    bars = _make_bars(n=30)
    bars["close"] = [100.0 + 0.5 * i for i in range(30)]
    bars["high"] = bars["close"] + 0.3
    bars["low"] = bars["close"] - 0.3
    event = Event(symbol="X", event_time=bars["date"].iloc[5], metadata={"direction": "short"})
    fr = ForwardReturns(horizons_minutes=[5], eod=False)
    out = fr.compute(event, bars)
    assert out["ret_5m"] < 0


def test_mfe_mae_at_60m():
    bars = _make_bars(n=30)
    bars["close"] = [100.0] * 30
    bars["high"] = [100.0] * 30
    bars["low"] = [100.0] * 30
    bars.loc[7, "high"] = 102.0
    bars.loc[8, "low"] = 98.5
    event = Event(symbol="X", event_time=bars["date"].iloc[5], metadata={"direction": "long"})
    fr = ForwardReturns(horizons_minutes=[60], eod=False)
    out = fr.compute(event, bars)
    assert abs(out["mfe_60m"] - 0.02) < 1e-9
    assert abs(out["mae_60m"] + 0.015) < 1e-9
