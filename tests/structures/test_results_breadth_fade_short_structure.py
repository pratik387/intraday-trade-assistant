"""Results-day breadth cells (2026-11-03): up_spike's fade on completed candles,
results-tagged symbol-days only, events named after their own setup."""
import io
import json
from pathlib import Path

import pandas as pd
import pytest

from structures import results_breadth_fade_short_structure as mod
from structures.data_models import MarketContext
from structures.results_breadth_fade_short_structure import ResultsBreadthFadeShortStructure

CFG = json.load(io.open(Path(__file__).resolve().parents[2] / "config" / "configuration.json",
                        encoding="utf-8"))["setups"]
DAY, LATE = "results_breadth_day_fade_short", "results_breadth_late_fade_short"
SPIKE_BAR = pd.Timestamp("2026-11-03 10:05")


def _df(r3=0.04, last_vol=2_000_000.0, prior_vol=200_000.0, day="2026-11-03", forming=None):
    """8 flat bars from 09:15, then three bars rising r3 into the 10:05 spike bar;
    forming = (close, volume) of a still-forming 10:10 candle, or None."""
    rows, ts, p = [], pd.Timestamp(f"{day} 09:15"), 100.0
    for _ in range(8):
        rows.append((ts, p, p, p, p, prior_vol)); ts += pd.Timedelta(minutes=5)
    for c, v in ((p, prior_vol), (p * (1 + r3 * 0.5), prior_vol), (p * (1 + r3), last_vol)):
        rows.append((ts, c, c * 1.001, c * 0.999, c, v)); ts += pd.Timedelta(minutes=5)
    if forming is not None:
        c, v = forming
        rows.append((ts, c, c, c, c, v))
    return pd.DataFrame({"open": [r[1] for r in rows], "high": [r[2] for r in rows],
                         "low": [r[3] for r in rows], "close": [r[4] for r in rows],
                         "volume": [r[5] for r in rows]}, index=[r[0] for r in rows])


def _ctx(df, scan_bar_ts=SPIKE_BAR, sym="NSE:TESTCO"):
    return MarketContext(symbol=sym, current_price=float(df["close"].iloc[-1]),
                         timestamp=df.index[-1], df_5m=df,
                         session_date=df.index[-1].date(), cap_segment="small_cap",
                         scan_bar_ts=scan_bar_ts)


@pytest.fixture
def tagged(monkeypatch):
    seen = {}

    def fake(sym, session, path=None):
        seen["path"] = path
        return "board_meeting_today"
    monkeypatch.setattr(mod, "results_context", fake)
    return seen


def _det(name=DAY):
    return ResultsBreadthFadeShortStructure(dict(CFG[name]))


def test_fires_on_a_tagged_day_and_names_its_setup(tagged):
    a = _det().detect(_ctx(_df()))
    assert a.structure_detected
    assert a.events[0].structure_type == DAY
    assert a.events[0].timestamp == SPIKE_BAR
    assert str(tagged["path"]).replace("\\", "/").endswith(CFG[DAY]["event_feed"]["path"])


def test_untagged_day_is_rejected(monkeypatch):
    monkeypatch.setattr(mod, "results_context", lambda sym, session, path=None: "none")
    a = _det().detect(_ctx(_df()))
    assert not a.structure_detected and "results_context=none" in a.rejection_reason


def test_forming_candle_is_dropped(tagged):
    # A still-forming 10:10 candle that gave back the spike on thin volume: the
    # cell judges the completed 10:05 candle, as validated.
    df = _df(forming=(100.5, 50_000.0))
    a = _det().detect(_ctx(df, scan_bar_ts=SPIKE_BAR))
    assert a.structure_detected and a.events[0].timestamp == SPIKE_BAR


def test_unknown_scan_bar_fails_closed(tagged):
    a = _det().detect(_ctx(_df(), scan_bar_ts=None))
    assert not a.structure_detected and "scan_bar_ts unknown" in a.rejection_reason


def test_below_the_cell_threshold_does_not_fire(tagged):
    assert not _det().detect(_ctx(_df(r3=0.03))).structure_detected            # day cell needs 3.5%


def test_late_cell_window_and_threshold():
    late, day = CFG[LATE], CFG[DAY]
    assert (late["active_window_start"], late["active_window_end"], late["spike_r3_pct_min"]) == ("14:05", "15:00", 0.055)
    assert (day["active_window_start"], day["active_window_end"], day["spike_r3_pct_min"]) == ("09:45", "14:00", 0.035)


def test_cells_are_paper_only_and_inherit_up_spike_execution():
    up = CFG["up_spike_fade_short"]
    for name in (DAY, LATE):
        c = CFG[name]
        assert c["enabled"] is True and c["live_enabled"] is False
        assert c["event_setup_type"] == name
        assert c["use_completed_bars_only"] is True
        assert c["required_results_context"] == ["results_preopen", "board_meeting_today"]
        assert c["event_feed"]["path"] == up["event_feed"]["path"]
        for k in ("vol_burst_min", "min_bars_before_signal", "per_day_turnover_floor_rs", "allowed_cap_segments",
                  "catastrophe_stop_pct", "time_stop_at", "immediate_execution", "circuit_cover_order",
                  "target_notional_pct", "circuit_risk_budget_pct_of_capital", "min_daily_avg_turnover_inr"):
            assert c[k] == up[k], k


@pytest.mark.parametrize("tags", [[], "results_preopen", ["results_preopen", "typo"], ["none"]])
def test_bad_results_gate_is_a_config_error(tags):
    cfg = dict(CFG[DAY], required_results_context=tags)
    with pytest.raises(ValueError):
        ResultsBreadthFadeShortStructure(cfg)
