"""Settles from the PREVIOUS regime must not enter the live ledger.

Context: on 2026-08-12 the multi-day book's capital management changed. The
ledgers were reset so DecayTripwire's rolling PF measures one regime. But
positions ENTERED under the old rules were still open at the boundary and
settle days later -- routing on settle date would have dripped old-regime
trades into the fresh ledger, which is exactly what the reset was for.

Routing is therefore on ENTRY date: a position opened before the boundary
belongs to the old regime however late it closes.
"""
import json
from pathlib import Path

import pytest

from services.risk.decay_tripwire import DecayTripwire

SETUP = "zscore_oversold_revert_long"
BOUNDARY = "2026-08-13"
LABEL = "pre-sizing-2026-08-12"
REGIME = "pre_sizing"


def _tw(tmp_path: Path, *, boundary=BOUNDARY, label=LABEL, regime=REGIME,
        window=30) -> DecayTripwire:
    return DecayTripwire(
        setup_name=SETUP,
        state_path=tmp_path / "state" / f"decay_tripwire_{SETUP}.json",
        window_trades=window, pf_floor=1.2, sustained_weeks=6,
        archive_entries_before=boundary, archive_label=label,
        archive_regime=regime,
    )


def _archive_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / "archive" / f"decay_tripwire_{SETUP}.{LABEL}.json"


def _live_path(tmp_path: Path) -> Path:
    return tmp_path / "state" / f"decay_tripwire_{SETUP}.json"


def _record(tw, *, entry_date, net=100.0, symbol="NSE:RUPA", ts="2026-08-14T15:30:00"):
    tw.record_trade(net_pnl_inr=net, ts_iso=ts, symbol=symbol, entry_date=entry_date)


def test_pre_boundary_entry_goes_to_archive_not_live(tmp_path):
    """The exact case: entered 2026-08-11, settles 2026-08-14."""
    tw = _tw(tmp_path)
    _record(tw, entry_date="2026-08-11")
    assert not _live_path(tmp_path).exists() or \
        json.loads(_live_path(tmp_path).read_text())["trades"] == []
    arch = json.loads(_archive_path(tmp_path).read_text())
    assert len(arch["trades"]) == 1
    assert arch["trades"][0]["entry_date"] == "2026-08-11"
    assert arch["_regime"] == REGIME


def test_post_boundary_entry_goes_to_live(tmp_path):
    tw = _tw(tmp_path)
    _record(tw, entry_date="2026-08-13", ts="2026-08-18T15:30:00")
    assert json.loads(_live_path(tmp_path).read_text())["trades"][0]["entry_date"] == "2026-08-13"
    assert not _archive_path(tmp_path).exists()


def test_boundary_date_itself_is_current_regime(tmp_path):
    """archive_entries_before is exclusive: entries ON the date are current."""
    tw = _tw(tmp_path)
    _record(tw, entry_date=BOUNDARY)
    assert len(json.loads(_live_path(tmp_path).read_text())["trades"]) == 1


def test_archived_settles_do_not_move_the_rolling_pf(tmp_path):
    """The whole point: old-regime P&L must not reach the gate."""
    tw = _tw(tmp_path, window=5)
    for i in range(10):  # 10 catastrophic old-regime losses
        _record(tw, entry_date="2026-08-11", net=-5000.0, symbol=f"OLD{i}")
    assert tw._rolling_pf() is None, "archived trades leaked into the PF window"
    assert not tw.is_paused()


def test_archive_appends_to_an_existing_file(tmp_path):
    """The archive already holds the reset's history — never clobber it."""
    p = _archive_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "setup_name": SETUP, "_regime": REGIME,
        "trades": [{"net_pnl_inr": 1.0, "ts_iso": "2026-07-01T15:30:00"}],
    }), encoding="utf-8")
    _record(_tw(tmp_path), entry_date="2026-08-11", net=222.0)
    arch = json.loads(p.read_text())
    assert [t["net_pnl_inr"] for t in arch["trades"]] == [1.0, 222.0]


def test_archive_rows_keep_full_detail(tmp_path):
    tw = _tw(tmp_path)
    tw.record_trade(net_pnl_inr=-750.0, ts_iso="2026-08-14T15:30:00",
                    fees_inr=120.0, gross_pnl_inr=-630.0, symbol="NSE:ISGEC",
                    entry_price=100.0, exit_price=97.0, exit_reason="kday_close_moc",
                    qty=370, attributed=True, entry_date="2026-08-11")
    row = json.loads(_archive_path(tmp_path).read_text())["trades"][0]
    assert row["symbol"] == "NSE:ISGEC" and row["qty"] == 370
    assert row["fees_inr"] == 120.0 and row["attributed"] is True
    assert row["exit_reason"] == "kday_close_moc"


def test_reload_ignores_the_archive(tmp_path):
    """A fresh tripwire must not read archived rows back into the window."""
    tw = _tw(tmp_path)
    _record(tw, entry_date="2026-08-11", net=-9999.0)
    _record(tw, entry_date="2026-08-20", net=50.0)
    reloaded = _tw(tmp_path)
    assert [t.net_pnl_inr for t in reloaded._trades] == [50.0]


def test_no_boundary_configured_records_everything_live(tmp_path):
    """Books that were never reset (overnight) keep the old behaviour."""
    tw = _tw(tmp_path, boundary=None, label=None, regime=None)
    _record(tw, entry_date="2020-01-01")
    assert len(json.loads(_live_path(tmp_path).read_text())["trades"]) == 1


def test_missing_entry_date_stays_live_and_warns(tmp_path, caplog):
    tw = _tw(tmp_path)
    with caplog.at_level("WARNING"):
        tw.record_trade(net_pnl_inr=10.0, ts_iso="2026-08-14T15:30:00", symbol="NSE:X")
    assert len(json.loads(_live_path(tmp_path).read_text())["trades"]) == 1
    assert "no entry_date" in caplog.text


def test_partial_archive_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be set together"):
        _tw(tmp_path, label=None, regime=None)


def test_malformed_boundary_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        _tw(tmp_path, boundary="12-08-2026")


def test_archive_for_a_different_setup_is_refused(tmp_path):
    p = _archive_path(tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"setup_name": "someone_else", "trades": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="is for setup"):
        _record(_tw(tmp_path), entry_date="2026-08-11")
