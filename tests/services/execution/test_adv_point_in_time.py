"""ADV must be point-in-time, or a participation cap is calibrated on the future.

The 2026-08-24 participation study first used "the last 60 sessions in the
file" for every trade, which for a July trade included data through late July —
look-ahead. It mattered: these setups fire on volume spikes, so the trade day is
systematically atypical, and the error inflated the measured execution component
(corr(participation, impact) 0.599 -> 0.361 once corrected).

The engine must not repeat it. `_adv_for_symbol` is fed the batch that spans
session_date-30d .. session_date-1d, so every bar precedes the signal by
construction — these tests pin the properties that make that true.
"""
import numpy as np
import pandas as pd
import pytest

from services.execution.close_dn_baseline_build import _adv_for_symbol


def _bars(days, per_day=75, close=100.0, vol=1000.0, start="2026-07-01"):
    idx, rows = [], []
    for d in pd.date_range(start, periods=days, freq="B"):
        for i in range(per_day):
            idx.append(pd.Timestamp(d) + pd.Timedelta(minutes=5 * i))
            rows.append({"close": close, "volume": vol})
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_adv_is_rupee_turnover_not_share_volume():
    """A 500k-share name at Rs10 and at Rs2000 are different capacity problems."""
    cheap = _adv_for_symbol(_bars(30, close=10.0, vol=1000.0), 20)
    dear = _adv_for_symbol(_bars(30, close=2000.0, vol=1000.0), 20)
    assert dear == pytest.approx(200 * cheap), "ADV must scale with price"


def test_adv_is_a_daily_total_not_a_bar_average():
    df = _bars(30, per_day=75, close=100.0, vol=1000.0)
    # 75 bars x 100 x 1000 = 7,500,000 per day
    assert _adv_for_symbol(df, 20) == pytest.approx(7_500_000)


def test_adv_uses_the_median_so_one_spike_cannot_dominate():
    df = _bars(30, close=100.0, vol=1000.0)
    spike = df.index.normalize().unique()[-1]
    df.loc[df.index.normalize() == spike, "volume"] = 1_000_000.0
    # median is unmoved by a single 1000x day; a mean would be wrecked
    assert _adv_for_symbol(df, 20) == pytest.approx(7_500_000)


def test_adv_refuses_when_history_is_too_short():
    """Too few sessions to size against — record None rather than guess."""
    assert _adv_for_symbol(_bars(5), 20) is None


def test_adv_window_respects_n_sessions():
    df = _bars(40, close=100.0, vol=1000.0)
    last10 = df.index.normalize().unique()[-10:]
    df.loc[df.index.normalize().isin(last10), "volume"] = 2000.0
    a20 = _adv_for_symbol(df, 20)   # 10 days at 2x, 10 at 1x -> median between
    a5 = _adv_for_symbol(df, 5)     # all 5 most recent are 2x
    assert a5 == pytest.approx(15_000_000)
    assert a20 < a5, "a longer window must dilute the recent regime"


def test_degenerate_inputs_return_none_not_zero():
    assert _adv_for_symbol(None, 20) is None
    assert _adv_for_symbol(pd.DataFrame(), 20) is None
    assert _adv_for_symbol(_bars(30).drop(columns=["volume"]), 20) is None


def test_zero_volume_sessions_are_excluded_not_counted_as_liquidity():
    df = _bars(30, close=100.0, vol=1000.0)
    halted = df.index.normalize().unique()[:12]
    df.loc[df.index.normalize().isin(halted), "volume"] = 0.0
    # 12 halted days dropped; the median reflects the sessions that traded
    assert _adv_for_symbol(df, 20) == pytest.approx(7_500_000)


def test_builder_passes_a_pre_signal_window():
    """The guarantee that makes ADV point-in-time lives in the caller."""
    import inspect
    from services.execution import close_dn_baseline_build as B
    src = inspect.getsource(B.build_baseline_and_candidates)
    assert "session_date - _td(days=1)" in src, \
        "fetch window must END the day BEFORE the signal, or ADV sees the trade day"
    assert "_adv_for_symbol(df, adv_sessions)" in src
