"""Tests for services.daily_panel_provider — feather + live backends + factory."""
from datetime import date, timedelta
from pathlib import Path
import pandas as pd
import pytest

from services.daily_panel_provider import (
    FeatherDailyPanelProvider, LiveDailyPanelProvider, make_provider, _window_calendar_days,
)

CFG = {"selection_mode": "trailing_loser_decile", "lookback_days": 5,
       "shock_lookback_days": 20, "data_source": "x.feather"}


def _mk_df(symbols, days, start=date(2025, 1, 1)):
    rows = []
    for i in range(days):
        d = start + timedelta(days=i)
        for s in symbols:
            rows.append((pd.Timestamp(d), s, 100.0, 101.0, 99.0, 100.0, 1000))
    return pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])


def test_window_covers_required_trailing_days():
    # need = max(5,20)+5 = 25 trading days -> >= 35 calendar + buffer
    assert _window_calendar_days(CFG) >= 35


def test_feather_provider_slices_to_window(tmp_path):
    f = tmp_path / "panel.feather"
    _mk_df(["AAA", "BBB"], 60).to_feather(f)
    p = FeatherDailyPanelProvider(f, CFG)
    as_of = date(2025, 1, 60 - 30 + 1)  # mid-range
    panel = p.get_panel(as_of)
    assert (panel["date"].dt.normalize() <= pd.Timestamp(as_of)).all()  # no future leakage
    assert set(panel["symbol"]) == {"AAA", "BBB"}
    assert len(panel) > 0


def test_feather_provider_no_future_leakage(tmp_path):
    f = tmp_path / "panel.feather"
    _mk_df(["AAA"], 60).to_feather(f)
    p = FeatherDailyPanelProvider(f, CFG)
    as_of = date(2025, 1, 20)
    panel = p.get_panel(as_of)
    assert panel["date"].max() <= pd.Timestamp(as_of)


def test_feather_provider_missing_file_raises(tmp_path):
    p = FeatherDailyPanelProvider(tmp_path / "nope.feather", CFG)
    with pytest.raises(FileNotFoundError):
        p.get_panel(date(2025, 1, 10))


def test_live_provider_uses_fetcher_and_strips_future(tmp_path):
    captured = {}
    def fake_fetch(symbols, start, end):
        captured["symbols"] = list(symbols)
        captured["start"], captured["end"] = start, end
        # return some rows incl one beyond as_of to prove it's stripped
        df = _mk_df(["AAA", "BBB"], 40, start=start)
        return df
    as_of = date(2025, 2, 10)
    p = LiveDailyPanelProvider(fake_fetch, ["NSE:AAA", "bbb"], CFG)
    panel = p.get_panel(as_of)
    assert captured["symbols"] == ["AAA", "BBB"]  # normalized
    assert captured["end"] == as_of
    assert panel["date"].max() <= pd.Timestamp(as_of)  # future stripped


def test_live_provider_empty_fetch_returns_empty():
    p = LiveDailyPanelProvider(lambda s, a, b: pd.DataFrame(), ["AAA"], CFG)
    assert p.get_panel(date(2025, 1, 10)).empty


def test_factory_backtest_returns_feather(tmp_path):
    f = tmp_path / "p.feather"; _mk_df(["AAA"], 30).to_feather(f)
    cfg = {**CFG, "data_source": "p.feather"}
    prov = make_provider(cfg, dry_run=True, repo_root=tmp_path)
    assert isinstance(prov, FeatherDailyPanelProvider)


def test_factory_live_requires_fetcher():
    with pytest.raises(ValueError):
        make_provider(CFG, dry_run=False)
