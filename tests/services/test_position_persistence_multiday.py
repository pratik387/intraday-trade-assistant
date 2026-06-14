"""Multi-day (CNC/MTF) extension of PositionPersistence.

Verifies the stale-snapshot recovery split: a multi-day hold whose
`exit_on_date` is today-or-later is SALVAGED across a restart; legacy
intraday / 1-night (close_dn) positions and expired holds are dropped exactly
as before. Also checks save_position carries the new fields round-trip.

"Today" is pinned by monkeypatching the module clock so the freshness logic is
deterministic regardless of the wall clock.
"""
import json
from datetime import datetime, date
from pathlib import Path

import pytest

import services.state.position_persistence as pp
from services.state.position_persistence import PositionPersistence, PersistedPosition


def _pin_today(monkeypatch, y, m, d):
    monkeypatch.setattr(pp, "_now_naive_ist", lambda: datetime(y, m, d, 9, 30))


def _write_snapshot(state_dir: Path, ts_iso: str, positions: dict) -> None:
    payload = {"timestamp": ts_iso, "positions": positions}
    (state_dir / "positions_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")


def _legacy(sym):
    """A legacy intraday / 1-night position: no multi-day fields."""
    return PersistedPosition(
        symbol=sym, side="BUY", qty=10, avg_price=100.0, trade_id=f"T-{sym}"
    ).to_dict()


def _multiday(sym, entry_date, exit_on_date, product="MTF"):
    return PersistedPosition(
        symbol=sym, side="BUY", qty=10, avg_price=100.0, trade_id=f"T-{sym}",
        entry_date=entry_date, exit_on_date=exit_on_date, product=product,
    ).to_dict()


def test_fresh_snapshot_loads_everything(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 15)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 15, 9, 0).isoformat(),
        {"NSE:AAA": _legacy("NSE:AAA"),
         "NSE:BBB": _multiday("NSE:BBB", "2026-06-15", "2026-06-17")},
    )
    p = PositionPersistence(tmp_path)
    loaded = p.load_snapshot()
    assert set(loaded) == {"NSE:AAA", "NSE:BBB"}  # fresh => load all verbatim


def test_stale_snapshot_drops_legacy(tmp_path, monkeypatch):
    # close_dn 1-night / intraday positions have no exit_on_date => dropped on
    # a stale snapshot, exactly as the original behavior.
    _pin_today(monkeypatch, 2026, 6, 16)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 15, 15, 30).isoformat(),
        {"NSE:AAA": _legacy("NSE:AAA")},
    )
    p = PositionPersistence(tmp_path)
    assert p.load_snapshot() == {}


def test_stale_snapshot_salvages_open_multiday(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 16)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 15, 15, 30).isoformat(),
        {"NSE:BBB": _multiday("NSE:BBB", "2026-06-15", "2026-06-17")},  # exit in future
    )
    p = PositionPersistence(tmp_path)
    loaded = p.load_snapshot()
    assert set(loaded) == {"NSE:BBB"}
    assert loaded["NSE:BBB"].exit_on_date == "2026-06-17"
    assert loaded["NSE:BBB"].product == "MTF"


def test_stale_snapshot_salvages_when_exit_is_today(tmp_path, monkeypatch):
    # exit_on_date == today is still open (exit happens at today's close).
    _pin_today(monkeypatch, 2026, 6, 17)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 16, 15, 30).isoformat(),
        {"NSE:BBB": _multiday("NSE:BBB", "2026-06-15", "2026-06-17")},
    )
    assert set(PositionPersistence(tmp_path).load_snapshot()) == {"NSE:BBB"}


def test_stale_snapshot_drops_expired_multiday(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 18)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 17, 15, 30).isoformat(),
        {"NSE:BBB": _multiday("NSE:BBB", "2026-06-15", "2026-06-17")},  # exit already passed
    )
    assert PositionPersistence(tmp_path).load_snapshot() == {}


def test_stale_snapshot_mixed(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 18)
    _write_snapshot(
        tmp_path, datetime(2026, 6, 17, 15, 30).isoformat(),
        {
            "NSE:LEG": _legacy("NSE:LEG"),                                  # drop
            "NSE:OPEN": _multiday("NSE:OPEN", "2026-06-16", "2026-06-19"),  # salvage
            "NSE:EXP": _multiday("NSE:EXP", "2026-06-12", "2026-06-16"),    # drop (expired)
        },
    )
    assert set(PositionPersistence(tmp_path).load_snapshot()) == {"NSE:OPEN"}


def test_unparseable_timestamp_treated_as_stale(tmp_path, monkeypatch):
    # Corrupt/absent timestamp => stale => per-position recovery: legacy dropped,
    # still-open multi-day salvaged. Safer than blindly loading unknown-age data.
    _pin_today(monkeypatch, 2026, 6, 16)
    _write_snapshot(
        tmp_path, "not-a-date",
        {"NSE:LEG": _legacy("NSE:LEG"),
         "NSE:OPEN": _multiday("NSE:OPEN", "2026-06-15", "2026-06-18")},
    )
    assert set(PositionPersistence(tmp_path).load_snapshot()) == {"NSE:OPEN"}


def test_missing_timestamp_treated_as_stale(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 16)
    payload = {"positions": {"NSE:LEG": _legacy("NSE:LEG")}}  # no "timestamp" key
    (tmp_path / "positions_snapshot.json").write_text(json.dumps(payload), encoding="utf-8")
    assert PositionPersistence(tmp_path).load_snapshot() == {}


def test_save_position_round_trips_multiday_fields(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 16)
    p = PositionPersistence(tmp_path)
    p.save_position(
        symbol="NSE:BBB", side="BUY", qty=10, avg_price=100.0, trade_id="T1",
        entry_date="2026-06-16", exit_on_date="2026-06-18", product="MTF",
    )
    # Reload from disk into a fresh instance (same pinned today => fresh snapshot).
    again = PositionPersistence(tmp_path).get_position("NSE:BBB")
    assert again is not None
    assert again.entry_date == "2026-06-16"
    assert again.exit_on_date == "2026-06-18"
    assert again.product == "MTF"


def test_save_position_legacy_has_no_multiday_fields(tmp_path, monkeypatch):
    _pin_today(monkeypatch, 2026, 6, 16)
    p = PositionPersistence(tmp_path)
    p.save_position(symbol="NSE:AAA", side="BUY", qty=10, avg_price=100.0, trade_id="T1")
    pos = p.get_position("NSE:AAA")
    assert pos.exit_on_date is None and pos.entry_date is None and pos.product is None
