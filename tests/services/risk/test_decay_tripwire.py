"""Tests for DecayTripwire."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from services.risk.decay_tripwire import DecayTripwire


def _pwd(tmp_path: Path):
    return tmp_path / "decay_tripwire_test.json"


def _trade(pnl: float, days_offset: int = 0, base: str = "2026-01-01T09:30:00"):
    base_dt = datetime.fromisoformat(base)
    return pnl, (base_dt + timedelta(days=days_offset)).isoformat()


def test_init_with_no_state_starts_empty(tmp_path):
    tw = DecayTripwire("test", _pwd(tmp_path),
                        window_trades=30, pf_floor=1.20, sustained_weeks=6)
    assert not tw.is_paused()
    s = tw.state_summary()
    assert s["trade_count"] == 0
    assert s["current_rolling_pf"] is None


def test_records_trades(tmp_path):
    tw = DecayTripwire("test", _pwd(tmp_path),
                        window_trades=30, pf_floor=1.20, sustained_weeks=6)
    base = datetime.fromisoformat("2026-01-01T09:30:00")
    for i in range(10):
        tw.record_trade(net_pnl_inr=100.0, ts_iso=(base + timedelta(days=i)).isoformat())
    s = tw.state_summary()
    assert s["trade_count"] == 10
    # Less than window_trades=30 → rolling PF is None
    assert s["current_rolling_pf"] is None


def test_does_not_pause_when_pf_above_floor(tmp_path):
    tw = DecayTripwire("test", _pwd(tmp_path),
                        window_trades=10, pf_floor=1.20, sustained_weeks=6)
    base = datetime.fromisoformat("2026-01-01T09:30:00")
    # 30 trades, mostly winners: 25 wins of 200, 5 losses of 100 → PF = 5000 / 500 = 10
    for i in range(25):
        tw.record_trade(200.0, (base + timedelta(days=i)).isoformat())
    for i in range(25, 30):
        tw.record_trade(-100.0, (base + timedelta(days=i)).isoformat())
    assert not tw.is_paused()
    s = tw.state_summary()
    assert s["current_rolling_pf"] is not None
    assert s["current_rolling_pf"] >= 1.20


def test_pauses_when_pf_below_floor_for_sustained_period(tmp_path):
    """30 losing trades over >6 weeks → tripwire pauses."""
    tw = DecayTripwire("test", _pwd(tmp_path),
                        window_trades=10, pf_floor=1.20, sustained_weeks=4)
    base = datetime.fromisoformat("2026-01-01T09:30:00")
    # First 10 trades: PF = 1.0 (10 wins of 100, 10 losses of 100 spread)
    # Actually: to ensure PF < 1.2, do mostly losses
    for i in range(10):
        ts = (base + timedelta(days=i)).isoformat()
        tw.record_trade(50.0, ts)  # win of 50
    # Now 10 trades, all winners → PF = inf, not paused
    assert not tw.is_paused()

    # Now add 10 losses (replaces window): PF becomes 0 (no gains in window)
    for i in range(10, 20):
        ts = (base + timedelta(days=i)).isoformat()
        tw.record_trade(-100.0, ts)
    # Now rolling 10-trade window is all losses → PF < 1.2, breach starts
    s = tw.state_summary()
    assert s["current_rolling_pf"] is not None and s["current_rolling_pf"] < 1.20
    # First trade was at day 0; check the first_below_floor_ts was set
    # ... but we still need sustained_weeks=4 weeks above the first breach to pause
    # Add more losses extending the timeline past 4 weeks (28+ days)
    for i in range(20, 25):
        ts = (base + timedelta(days=i + 30)).isoformat()  # >4 weeks past first breach
        tw.record_trade(-100.0, ts)
    assert tw.is_paused()


def test_unpauses_after_pf_recovers(tmp_path):
    """If PF recovers above floor BEFORE sustained_weeks elapses, breach watch clears."""
    tw = DecayTripwire("test", _pwd(tmp_path),
                        window_trades=10, pf_floor=1.20, sustained_weeks=6)
    base = datetime.fromisoformat("2026-01-01T09:30:00")
    # 10 trades at PF = 1.0 (5 wins of 100, 5 losses of 100)
    for i in range(5):
        tw.record_trade(100.0, (base + timedelta(days=i)).isoformat())
    for i in range(5, 10):
        tw.record_trade(-100.0, (base + timedelta(days=i)).isoformat())
    # PF = 500/500 = 1.0 < 1.2 → breach starts
    assert tw.state_summary()["first_below_floor_ts"] is not None
    assert not tw.is_paused()  # not yet sustained

    # Add 5 strong winners — pushes window PF back above 1.2
    for i in range(10, 15):
        tw.record_trade(500.0, (base + timedelta(days=i)).isoformat())
    s = tw.state_summary()
    assert s["current_rolling_pf"] >= 1.20
    assert s["first_below_floor_ts"] is None  # breach cleared


def test_state_persists_across_instances(tmp_path):
    """Write state in one instance, read in another."""
    path = _pwd(tmp_path)
    tw1 = DecayTripwire("test", path,
                         window_trades=30, pf_floor=1.20, sustained_weeks=6)
    tw1.record_trade(100.0, "2026-01-01T09:30:00")
    tw1.record_trade(200.0, "2026-01-02T09:30:00")
    # New instance loads same state
    tw2 = DecayTripwire("test", path,
                         window_trades=30, pf_floor=1.20, sustained_weeks=6)
    assert tw2.state_summary()["trade_count"] == 2


def test_state_setup_name_mismatch_raises(tmp_path):
    path = _pwd(tmp_path)
    tw = DecayTripwire("setupA", path, 30, 1.20, 6)
    tw.record_trade(100.0, "2026-01-01T09:30:00")
    with pytest.raises(ValueError, match="state file is for setup"):
        DecayTripwire("setupB", path, 30, 1.20, 6)


def test_corrupt_state_raises(tmp_path):
    path = _pwd(tmp_path)
    path.write_text("not valid json {{{", encoding="utf-8")
    with pytest.raises(ValueError, match="corrupt"):
        DecayTripwire("test", path, 30, 1.20, 6)


def test_reset_clears_pause_state(tmp_path):
    """reset() should clear paused_since AND first_below_floor_ts but keep trade history."""
    tw = DecayTripwire("test", _pwd(tmp_path),
                        window_trades=10, pf_floor=1.20, sustained_weeks=4)
    base = datetime.fromisoformat("2026-01-01T09:30:00")
    # Force a pause
    for i in range(10):
        tw.record_trade(-100.0, (base + timedelta(days=i)).isoformat())
    for i in range(10, 15):
        tw.record_trade(-100.0, (base + timedelta(days=i + 30)).isoformat())
    assert tw.is_paused()

    tw.reset()
    assert not tw.is_paused()
    s = tw.state_summary()
    assert s["paused_since"] is None
    assert s["first_below_floor_ts"] is None
    # Trade history preserved
    assert s["trade_count"] == 15


def test_records_optional_fees_and_gross(tmp_path):
    """record_trade can carry fees_inr + gross_pnl_inr; persisted and reloaded."""
    path = _pwd(tmp_path)
    tw = DecayTripwire("test", path, 30, 1.20, 6)
    tw.record_trade(net_pnl_inr=850.0, ts_iso="2026-01-01T09:30:00",
                    fees_inr=120.0, gross_pnl_inr=970.0)
    data = json.loads(path.read_text(encoding="utf-8"))
    rec = data["trades"][0]
    assert rec["net_pnl_inr"] == 850.0
    assert rec["fees_inr"] == 120.0
    assert rec["gross_pnl_inr"] == 970.0
    # Reload preserves the fields
    tw2 = DecayTripwire("test", path, 30, 1.20, 6)
    assert tw2.state_summary()["trade_count"] == 1


def test_legacy_records_without_fees_load(tmp_path):
    """Old ledgers (net_pnl_inr only) must still load — fees default to None."""
    path = _pwd(tmp_path)
    path.write_text(json.dumps({
        "setup_name": "test",
        "trades": [{"net_pnl_inr": 100.0, "ts_iso": "2026-01-01T09:30:00"}],
    }), encoding="utf-8")
    tw = DecayTripwire("test", path, 30, 1.20, 6)
    assert tw.state_summary()["trade_count"] == 1
    # Appending a fee-bearing trade alongside the legacy one round-trips cleanly
    tw.record_trade(net_pnl_inr=50.0, ts_iso="2026-01-02T09:30:00", fees_inr=10.0)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["trades"][0].get("fees_inr") is None
    assert data["trades"][1]["fees_inr"] == 10.0


def test_invalid_window_trades_raises(tmp_path):
    with pytest.raises(ValueError, match="window_trades"):
        DecayTripwire("test", _pwd(tmp_path), window_trades=3, pf_floor=1.2, sustained_weeks=6)


def test_invalid_pf_floor_raises(tmp_path):
    with pytest.raises(ValueError, match="pf_floor"):
        DecayTripwire("test", _pwd(tmp_path), window_trades=30, pf_floor=0.0, sustained_weeks=6)
