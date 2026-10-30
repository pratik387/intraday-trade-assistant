"""Tests for cron-triggered overnight handlers (run_entry + run_verify_exit)."""
import json
import sys
from datetime import date, time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT))

from services.execution.overnight_handlers import (  # noqa: E402
    run_entry,
    run_verify_exit,
    _next_trading_day,
    _insufficient_funds_retry_qty,
    _live_poll_fill_ex,
)


_MTF_SNAPSHOT = _REPO_ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"


# ---------------------------------------------------------------------------
# _next_trading_day
# ---------------------------------------------------------------------------

def test_next_trading_day_monday_to_thursday():
    """Mon, Tue, Wed, Thu -> +1 day."""
    assert _next_trading_day(date(2026, 5, 18)) == date(2026, 5, 19)  # Mon -> Tue
    assert _next_trading_day(date(2026, 5, 21)) == date(2026, 5, 22)  # Thu -> Fri


def test_next_trading_day_friday_to_monday():
    """Friday -> Monday (skip Sat+Sun)."""
    assert _next_trading_day(date(2026, 5, 22)) == date(2026, 5, 25)


def test_next_trading_day_weekend_normalization():
    """Saturday + Sunday -> Monday."""
    assert _next_trading_day(date(2026, 5, 23)) == date(2026, 5, 25)  # Sat -> Mon
    assert _next_trading_day(date(2026, 5, 24)) == date(2026, 5, 25)  # Sun -> Mon


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _minimal_config(state_path: Path) -> dict:
    """Minimal config so handlers can load without a full configuration.json."""
    return {
        "setups": {
            "close_dn_overnight_long": {
                "enabled": False,
                "paper_enabled": True,
                "mode": "overnight",
                "detector_class": (
                    "structures.close_dn_overnight_long_structure."
                    "CloseDnOvernightLongStructure"
                ),
                "universe_builder": "services.setup_universe.close_dn_overnight_long_universe",
                "universe_trigger": "session_start",
                "active_window_start": "15:25",
                "active_window_end": "15:25",
                "signed_vol_ratio_max": -0.5,
                "closing_25m_volume_z_min": 1.0,
                "min_signal_bar_count": 4,
                "cell_volume_z_min": 2.0,
                "cell_prior_day_return_pct_min": 3.0,
                "baseline_rolling_days": 20,
                "min_daily_avg_volume": 50000,
                "min_trading_days_required": 30,
                "universe_max_symbols": 1500,
                "exclude_no_band_names": False,
                "catastrophe_stop_pct": 5.0,
                "gtt_limit_buffer_pct": 0.5,
                "failsafe_fill_confirm_sec": 0,
                "entry_limit_buffer_pct": 1.0,
                "tick_size_fallback_inr": 0.05,
                "insufficient_funds_retry_haircut": 0.95,
                "fill_poll_timeout_sec": 20,
                "entry_fetch_rps": 30.0,
                "entry_fetch_concurrency": 30,
                "entry_placement_cutoff_hhmmss": "15:29:20",
                "slot_ranking_mode": "unbiased_hash",
                "capital_allocation": {
                    "active_margin_inr": 400000,
                    "cushion_inr": 100000,
                    "max_concurrent_slots": 4,
                    "margin_per_slot_inr": 100000,
                    "max_new_positions_per_day": 2,
                    "state_file": str(state_path),
                },
                "mtf": {
                    "approved_list_snapshot_path": str(_MTF_SNAPSHOT),
                    "interest_pct_per_day": 0.0004,
                    "exclude_etf": True,
                    "fallback_to_cnc_if_not_mtf": True,
                    "stale_snapshot_warn_days": 7,
                },
            }
        }
    }


def _seed_state(state_path: Path, slots: list) -> None:
    """Persist a seed slot pool state file with the given slot list (length 4)."""
    state = {
        "max_slots": 4,
        "margin_per_slot_inr": 100000,
        "max_new_per_day": 2,
        "slots": slots,
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state))


def _empty_free_slot(slot_id: int) -> dict:
    return {
        "slot_id": slot_id, "status": "free",
        "symbol": None, "product": None, "leverage": 1.0,
        "margin_inr": 0.0, "notional_inr": 0.0,
        "buy_fill_price": None, "buy_fill_ts": None, "buy_order_id": None,
        "amo_sell_order_id": None, "expected_exit_date": None,
        "sell_fill_price": None, "sell_fill_ts": None,
        "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
        "reserved_today": None,
    }


# ---------------------------------------------------------------------------
# run_entry
# ---------------------------------------------------------------------------

def test_run_entry_with_disabled_setup_exits_cleanly(tmp_path):
    """If paper_enabled=False AND enabled=False, run_entry returns early with no fires."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["paper_enabled"] = False
    broker = MagicMock()
    summary = run_entry(cfg, broker, now_ist=pd.Timestamp("2026-05-21 15:25:00"))
    assert summary["fired_count"] == 0
    assert summary["skipped_count"] == 0
    assert summary["rejected_count"] == 0


def test_run_entry_skips_setup_paused_by_edge_integrity(tmp_path):
    """cb_state='paused_precondition' (jobs/check_edge_integrity.py) blocks NEW
    overnight entries; run_entry exits with the pause recorded in the summary."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["cb_state"] = "paused_precondition"
    broker = MagicMock()
    summary = run_entry(cfg, broker, now_ist=pd.Timestamp("2026-05-21 15:25:00"))
    assert summary["cb_paused_setups"] == ["close_dn_overnight_long"]
    assert summary["fired_count"] == 0
    broker.place_order.assert_not_called()


def test_run_entry_with_empty_broker_universe_exits_cleanly(tmp_path):
    """If broker lists no symbols, universe is empty: no fires, no errors."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    broker = MagicMock()
    broker.list_symbols.return_value = []
    broker.get_daily.return_value = pd.DataFrame()
    summary = run_entry(cfg, broker, now_ist=pd.Timestamp("2026-05-21 15:25:00"))
    assert summary["fired_count"] == 0


# ---------------------------------------------------------------------------
# run_verify_exit
# ---------------------------------------------------------------------------

def test_run_verify_exit_with_no_state_file_exits_cleanly(tmp_path):
    """No state file -> no settles, no errors."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    broker = MagicMock()
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-22 09:30:00"))
    assert summary["settled_count"] == 0
    assert summary["released_count"] == 0


def test_run_verify_exit_settles_t0_slot_at_expected_date(tmp_path):
    """A slot in t0_open with expected_exit_date today gets settled."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:RELIANCE",
            "product": "MTF", "leverage": 3.85,
            "margin_inr": 100000.0, "notional_inr": 385000.0,
            "buy_fill_price": 2500.0, "buy_fill_ts": "2026-05-21T15:30:00",
            "buy_order_id": "BUY-001",
            "amo_sell_order_id": "AMO-001",
            "expected_exit_date": "2026-05-22",
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": "2026-05-21",
        },
        _empty_free_slot(2),
        _empty_free_slot(3),
        _empty_free_slot(4),
    ])

    broker = MagicMock()
    enriched = {
        "RELIANCE": pd.DataFrame(
            {"open": [2515.0], "high": [2520.0], "low": [2510.0],
             "close": [2517.0], "volume": [1000]},
            index=[pd.Timestamp("2026-05-22 09:15:00")],
        )
    }
    broker._load_enriched_5m = MagicMock(return_value=enriched)

    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-22 09:30:00"))
    assert summary["settled_count"] == 1

    new_state = json.loads(state_path.read_text())
    slot1 = next(s for s in new_state["slots"] if s["slot_id"] == 1)
    assert slot1["status"] == "t1_settling"
    assert slot1["sell_fill_price"] == 2515.0
    assert slot1["realized_pnl_inr"] is not None
    # qty=floor(385000/2500)=154; gross=(2515-2500)*154=2310; fees+interest<1000
    assert slot1["realized_pnl_inr"] > 1000.0


def test_run_verify_exit_releases_t1_slot_at_t2(tmp_path):
    """A slot in t1_settling whose T+2 cash settle date has passed gets released."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t1_settling", "symbol": "NSE:RELIANCE",
            "product": "MTF", "leverage": 3.85,
            "margin_inr": 100000.0, "notional_inr": 385000.0,
            "buy_fill_price": 2500.0, "buy_fill_ts": "2026-05-21T15:30:00",
            "buy_order_id": "BUY-001",
            "amo_sell_order_id": "AMO-001",
            "expected_exit_date": "2026-05-22",
            "sell_fill_price": 2515.0, "sell_fill_ts": "2026-05-22T09:15:00",
            "realized_pnl_inr": 1650.0, "fees_inr": 500.0, "interest_inr": 160.0,
            "reserved_today": "2026-05-21",
        },
        _empty_free_slot(2),
        _empty_free_slot(3),
        _empty_free_slot(4),
    ])

    broker = MagicMock()
    # T+2 settle: Fri 2026-05-22 -> Mon 2026-05-25 per _next_trading_day
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-25 09:30:00"))
    assert summary["released_count"] == 1

    new_state = json.loads(state_path.read_text())
    slot1 = next(s for s in new_state["slots"] if s["slot_id"] == 1)
    assert slot1["status"] == "free"
    assert slot1["symbol"] is None


def test_run_verify_exit_idempotent_for_already_released(tmp_path):
    """Running verify-exit twice on the same settled+released slot is a no-op."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t1_settling", "symbol": "NSE:RELIANCE",
            "product": "MTF", "leverage": 3.85,
            "margin_inr": 100000.0, "notional_inr": 385000.0,
            "buy_fill_price": 2500.0, "buy_fill_ts": "2026-05-21T15:30:00",
            "buy_order_id": "BUY-001", "amo_sell_order_id": "AMO-001",
            "expected_exit_date": "2026-05-22",
            "sell_fill_price": 2515.0, "sell_fill_ts": "2026-05-22T09:15:00",
            "realized_pnl_inr": 1650.0, "fees_inr": 500.0, "interest_inr": 160.0,
            "reserved_today": "2026-05-21",
        },
        _empty_free_slot(2),
        _empty_free_slot(3),
        _empty_free_slot(4),
    ])
    broker = MagicMock()

    run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-25 09:30:00"))
    state_after_first = json.loads(state_path.read_text())

    s2 = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-25 09:35:00"))
    state_after_second = json.loads(state_path.read_text())

    # Second run should be a no-op for the released slot
    assert state_after_first["slots"] == state_after_second["slots"]
    assert s2["released_count"] == 0
    assert s2["settled_count"] == 0


def test_run_verify_exit_skips_t0_before_expected_exit(tmp_path):
    """If today < expected_exit_date, t0_open slot is NOT settled (AMO not fired yet)."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:RELIANCE",
            "product": "MTF", "leverage": 3.85,
            "margin_inr": 100000.0, "notional_inr": 385000.0,
            "buy_fill_price": 2500.0, "buy_fill_ts": "2026-05-21T15:30:00",
            "buy_order_id": "BUY-001", "amo_sell_order_id": "AMO-001",
            "expected_exit_date": "2026-05-22",
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": "2026-05-21",
        },
        _empty_free_slot(2),
        _empty_free_slot(3),
        _empty_free_slot(4),
    ])
    broker = MagicMock()
    # Today is same as buy day, before expected_exit
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-21 16:00:00"))
    assert summary["settled_count"] == 0
    assert summary["orphan_t0_count"] == 0

    # State should be unchanged
    state = json.loads(state_path.read_text())
    slot1 = next(s for s in state["slots"] if s["slot_id"] == 1)
    assert slot1["status"] == "t0_open"


def test_run_verify_exit_uses_cnc_fees_when_product_is_cnc(tmp_path):
    """A CNC slot should use calc_fee_cnc (no interest)."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:TATAMOTORS",
            "product": "CNC", "leverage": 1.0,
            "margin_inr": 100000.0, "notional_inr": 100000.0,
            "buy_fill_price": 1000.0, "buy_fill_ts": "2026-05-21T15:30:00",
            "buy_order_id": "BUY-002", "amo_sell_order_id": "AMO-002",
            "expected_exit_date": "2026-05-22",
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": "2026-05-21",
        },
        _empty_free_slot(2),
        _empty_free_slot(3),
        _empty_free_slot(4),
    ])

    broker = MagicMock()
    enriched = {
        "TATAMOTORS": pd.DataFrame(
            {"open": [1010.0], "high": [1015.0], "low": [1005.0],
             "close": [1012.0], "volume": [500]},
            index=[pd.Timestamp("2026-05-22 09:15:00")],
        )
    }
    broker._load_enriched_5m = MagicMock(return_value=enriched)

    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-22 09:30:00"))
    assert summary["settled_count"] == 1

    new_state = json.loads(state_path.read_text())
    slot1 = next(s for s in new_state["slots"] if s["slot_id"] == 1)
    # CNC interest must be zero
    assert slot1["interest_inr"] == 0.0
    assert slot1["fees_inr"] is not None and slot1["fees_inr"] > 0


def test_run_verify_exit_orphan_t0_without_expected_exit_date(tmp_path):
    """A t0_open slot lacking expected_exit_date AND its entry day is an orphan
    (with the entry day known, the exit is derived: see the no-AMO tests)."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:RELIANCE",
            "product": "MTF", "leverage": 3.85,
            "margin_inr": 100000.0, "notional_inr": 385000.0,
            "buy_fill_price": 2500.0, "buy_fill_ts": "2026-05-21T15:30:00",
            "buy_order_id": "BUY-001",
            "amo_sell_order_id": None,
            "expected_exit_date": None,  # orphan
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": None,
        },
        _empty_free_slot(2),
        _empty_free_slot(3),
        _empty_free_slot(4),
    ])

    broker = MagicMock()
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-05-22 09:30:00"))
    assert summary["orphan_t0_count"] == 1
    assert summary["settled_count"] == 0


# ---------------------------------------------------------------------------
# _insufficient_funds_retry_qty (2026-07-06 incident: async REJECTED burn)
# ---------------------------------------------------------------------------

# Exact live status_message from the 2026-07-06 Kite rejection.
_LIVE_REJECT_MSG = (
    "Insufficient funds. Margin required: 152010.12. "
    "Margin available: 149081.20. Check the funds statement for more details."
)


def test_insufficient_funds_retry_qty_live_message():
    """Jul-6 live message, shortfall-based: shrink THIS order by the account
    shortfall S against its OWN margin M — not by the cumulative ratio."""
    qty, price, lev = 133, 571.5, 2.93
    own_margin = qty * price / lev                      # ~25,940
    shortfall = 152010.12 - 149081.20                   # 2,928.92
    expected = int(qty * ((own_margin - shortfall) / own_margin) * 0.95)
    out = _insufficient_funds_retry_qty(_LIVE_REJECT_MSG, qty, 0.95, price, lev)
    assert out == expected and 1 <= out < qty
    # the retry must actually FIT: its own margin <= M - S
    assert out * price / lev <= own_margin - shortfall


def test_insufficient_funds_retry_qty_tvselect_2026_07_09():
    """The incident that exposed the old math: cumulative required 189k vs
    available 150k. Old ratio-scaling gave qty 145 (still bounced); shortfall
    math must produce a qty whose own margin fits inside M - S."""
    msg = ("Insufficient funds. Margin required: 189442.72. "
           "Margin available: 150775.50. Check orderbook for open orders.")
    qty, price, lev = 193, 750.0, 2.9
    own_margin = qty * price / lev                      # ~49,913
    shortfall = 189442.72 - 150775.50                   # 38,667.22
    out = _insufficient_funds_retry_qty(msg, qty, 0.95, price, lev)
    assert out is not None and out < 145                # strictly better than old math
    assert out * price / lev <= own_margin - shortfall  # actually affordable


def test_insufficient_funds_retry_qty_non_matching_message():
    """A rejection that is NOT insufficient-funds -> None (no retry)."""
    assert _insufficient_funds_retry_qty(
        "Order rejected: circuit limit exceeded", 133, 0.95, 571.5, 2.93) is None
    assert _insufficient_funds_retry_qty("", 133, 0.95, 571.5, 2.93) is None
    assert _insufficient_funds_retry_qty(None, 133, 0.95, 571.5, 2.93) is None


def test_insufficient_funds_retry_qty_shortfall_exceeds_own_margin():
    """Shortfall bigger than this order's whole margin -> None (shrinking this
    order alone can never fit; don't place dust)."""
    msg = ("Insufficient funds. Margin required: 152010.12. "
           "Margin available: 100.00.")
    assert _insufficient_funds_retry_qty(msg, 133, 0.95, 571.5, 2.93) is None


def test_insufficient_funds_retry_qty_no_positive_shortfall():
    """available >= required -> no shortfall -> None (nothing to fix; the
    rejection wasn't really about this order's size)."""
    msg = ("Insufficient funds. Margin required: 100.00. "
           "Margin available: 500.00.")
    assert _insufficient_funds_retry_qty(msg, 133, 0.95, 571.5, 2.93) is None


def test_insufficient_funds_retry_qty_zero_required_guard():
    """Margin required 0 -> shortfall negative -> None (no divide-by-zero)."""
    msg = "Insufficient funds. Margin required: 0. Margin available: 100.00."
    assert _insufficient_funds_retry_qty(msg, 133, 0.95, 571.5, 2.93) is None
    # degenerate order params guard
    assert _insufficient_funds_retry_qty(_LIVE_REJECT_MSG, 133, 0.95, 0.0, 2.93) is None
    assert _insufficient_funds_retry_qty(_LIVE_REJECT_MSG, 0, 0.95, 571.5, 2.93) is None


# ---------------------------------------------------------------------------
# _live_poll_fill_ex (fast-fail on REJECTED/CANCELLED)
# ---------------------------------------------------------------------------

def test_live_poll_fill_ex_rejected_fast_fails_without_burning_timeout():
    """A REJECTED order stops polling on the FIRST status check (the 2026-07-06
    incident burned the whole poll window on an already-rejected order)."""
    broker = MagicMock()
    broker.get_order_status.return_value = {
        "order_id": "X1", "status": "REJECTED", "average_price": 0.0,
        "status_message": _LIVE_REJECT_MSG,
    }
    price, status = _live_poll_fill_ex(broker, "X1", timeout_sec=60)
    assert price is None
    assert status is not None and status["status"] == "REJECTED"
    assert status["status_message"] == _LIVE_REJECT_MSG
    assert broker.get_order_status.call_count == 1  # no repeat polling


def test_live_poll_fill_ex_cancelled_fast_fails():
    broker = MagicMock()
    broker.get_order_status.return_value = {
        "order_id": "X2", "status": "CANCELLED", "average_price": 0.0,
    }
    price, status = _live_poll_fill_ex(broker, "X2", timeout_sec=60)
    assert price is None
    assert status["status"] == "CANCELLED"
    assert broker.get_order_status.call_count == 1


def test_live_poll_fill_ex_complete_returns_price_and_status():
    broker = MagicMock()
    broker.get_order_status.return_value = {
        "order_id": "X3", "status": "COMPLETE", "average_price": 101.25,
    }
    price, status = _live_poll_fill_ex(broker, "X3", timeout_sec=60)
    assert price == 101.25
    assert status["status"] == "COMPLETE"


def test_live_poll_fill_ex_timeout_returns_last_status():
    """An OPEN order that never fills -> (None, last_status) after timeout."""
    broker = MagicMock()
    broker.get_order_status.return_value = {
        "order_id": "X4", "status": "OPEN", "average_price": 0.0,
    }
    price, status = _live_poll_fill_ex(broker, "X4", timeout_sec=0)
    assert price is None
    assert status is not None and status["status"] == "OPEN"


# ---------------------------------------------------------------------------
# _rank_detections (slot-allocation order)
# ---------------------------------------------------------------------------

def test_rank_detections_conviction_mode_deepest_first_cheap_tiebreak():
    """Legacy 'conviction' mode: deepest |svr| desc, cheaper entry tiebreak."""
    from types import SimpleNamespace as NS
    from services.execution.overnight_handlers import _rank_detections

    dets = [
        ("NSE:MILD",  NS(confidence=0.55), NS(entry_price=100.0)),
        ("NSE:DEEP",  NS(confidence=0.95), NS(entry_price=500.0)),
        ("NSE:TIE_EXPENSIVE", NS(confidence=0.80), NS(entry_price=900.0)),
        ("NSE:TIE_CHEAP",     NS(confidence=0.80), NS(entry_price=50.0)),
    ]
    ranked = [s for s, _, _ in _rank_detections(
        dets, mode="conviction", session_date=date(2026, 8, 4))]
    assert ranked == ["NSE:DEEP", "NSE:TIE_CHEAP", "NSE:TIE_EXPENSIVE", "NSE:MILD"]
    assert dets[0][0] == "NSE:MILD"  # pure function: input untouched


def test_rank_detections_unbiased_hash_deterministic_and_conviction_blind():
    """'unbiased_hash' (production): same date -> same order (idempotent
    re-runs); different date -> reshuffled; confidence plays no role."""
    from types import SimpleNamespace as NS
    from services.execution.overnight_handlers import _rank_detections

    dets = [
        (f"NSE:SYM{i}", NS(confidence=0.5 + i * 0.05), NS(entry_price=100.0 + i))
        for i in range(8)
    ]
    d1 = date(2026, 8, 4)
    r1 = [s for s, _, _ in _rank_detections(dets, mode="unbiased_hash", session_date=d1)]
    r2 = [s for s, _, _ in _rank_detections(dets, mode="unbiased_hash", session_date=d1)]
    assert r1 == r2  # deterministic within a session
    r3 = [s for s, _, _ in _rank_detections(
        dets, mode="unbiased_hash", session_date=date(2026, 8, 5))]
    assert r3 != r1  # date salt reshuffles across days
    # conviction-blind: order is NOT the conviction order
    conv = [s for s, _, _ in _rank_detections(dets, mode="conviction", session_date=d1)]
    assert r1 != conv
    assert sorted(r1) == sorted(conv)  # same members, different order


def test_rank_detections_unknown_mode_fails_fast():
    from services.execution.overnight_handlers import _rank_detections
    with pytest.raises(ValueError):
        _rank_detections([], mode="best_first", session_date=date(2026, 8, 4))
# ---------------------------------------------------------------------------
# Partial-fill handling (2026-07-15 AJOONI: cancelled remainder orphaned the
# filled part — 1,751 real shares left with no AMO exit)
# ---------------------------------------------------------------------------

def test_partial_fill_from_status():
    from services.execution.overnight_handlers import _partial_fill_from_status
    assert _partial_fill_from_status({"status": "CANCELLED", "filled_quantity": 1751,
                                      "average_price": 4.14}) == (1751, 4.14)
    assert _partial_fill_from_status({"status": "CANCELLED", "filled_quantity": 0,
                                      "average_price": 0}) is None
    assert _partial_fill_from_status({"status": "REJECTED"}) is None
    assert _partial_fill_from_status(None) is None
    assert _partial_fill_from_status({"filled_quantity": "bad"}) is None
    # filled qty but no price -> not attachable
    assert _partial_fill_from_status({"filled_quantity": 10, "average_price": 0}) is None


def test_cancel_unfilled_buy_returns_status_dict():
    """Arity + passthrough: callers need the final status to detect partials."""
    from types import SimpleNamespace as NS
    from unittest.mock import MagicMock
    import services.execution.overnight_handlers as oh
    broker = MagicMock()
    broker.get_order_status.return_value = {"status": "CANCELLED",
                                            "filled_quantity": 100, "average_price": 9.5}
    st, fp, status = oh._cancel_unfilled_buy(broker, "OID", "NSE:X", confirm_timeout_sec=1)
    assert st == "CANCELLED" and fp is None
    assert status["filled_quantity"] == 100  # partial detectable by caller


# ---------------------------------------------------------------------------
# Next-day circuit-band estimate for the exit AMO (2026-07-20 SABEVENTS: AMO
# limit 7.37 sat inside Friday's band but below Monday's re-centered 7.50
# floor -> REJECTED at the 08:58 exchange send, position left unhedged)
# ---------------------------------------------------------------------------

def test_est_next_day_lower_circuit_math():
    from services.execution.overnight_handlers import _est_next_day_lower_circuit
    # Friday band from Thursday close 7.55 at +-5%: lc=7.1725 uc=7.9275.
    # Friday close 7.89 -> Monday floor estimate = 7.89 * 0.95 = 7.4955.
    est = _est_next_day_lower_circuit(7.89, 7.9275, 7.1725)
    assert est == pytest.approx(7.4955, abs=1e-6)
    # Missing quote fields -> None (caller falls back to today's lc)
    assert _est_next_day_lower_circuit(None, 7.9275, 7.1725) is None
    assert _est_next_day_lower_circuit(7.89, None, 7.1725) is None
    assert _est_next_day_lower_circuit(7.89, 7.9275, None) is None
    # Degenerate band -> None
    assert _est_next_day_lower_circuit(7.89, 7.0, 7.5) is None


def test_place_exit_amo_limit_respects_next_day_band(tmp_path):
    """The AMO SELL limit must clear the ESTIMATED next-day lower circuit,
    not just today's band (SABEVENTS 2026-07-20 rejection)."""
    from services.execution.overnight_handlers import run_place_exit
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True  # live path
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:SABEVENTS",
            "product": "CNC", "leverage": 1.0,
            "margin_inr": 50000.0, "notional_inr": 50000.0,
            "buy_fill_price": 7.7563, "buy_fill_ts": "2026-07-17T15:26:00",
            "buy_order_id": "BUY-001", "buy_qty": 6426,
            "amo_sell_order_id": None, "expected_exit_date": None,
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": "2026-07-17",
        },
        _empty_free_slot(2), _empty_free_slot(3), _empty_free_slot(4),
    ])
    broker = MagicMock()
    broker._data_sdk = None
    broker.get_quote.return_value = {
        "last_price": 7.89, "upper_circuit_limit": 7.9275,
        "lower_circuit_limit": 7.1725,
    }
    broker.place_order.return_value = "AMO-NEW"
    broker.place_gtt_stop.return_value = "GTT-NEW"

    summary = run_place_exit(cfg, broker,
                             now_ist=pd.Timestamp("2026-07-17 16:05:00"),
                             paper_mode=False)
    assert summary["placed_count"] == 1
    # catastrophe floor 7.7563*0.95=7.3685 clamped UP to est next-day lc
    # 7.4955, then ceil to tick -> 7.50 (Monday's actual floor).
    amo_call = broker.place_order.call_args
    assert amo_call.kwargs["variety"] == "amo"
    assert amo_call.kwargs["price"] == pytest.approx(7.50)


def test_verify_exit_adopts_out_of_band_manual_sell(tmp_path):
    """AMO rejected + user flattened manually -> settle at the manual fill,
    do NOT place a failsafe (it would reject on zero holdings / double-sell)."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True  # live path
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:SABEVENTS",
            "product": "CNC", "leverage": 1.0,
            "margin_inr": 50000.0, "notional_inr": 50000.0,
            "buy_fill_price": 7.7563, "buy_fill_ts": "2026-07-17T15:26:00",
            "buy_order_id": "BUY-001", "buy_qty": 6426,
            "amo_sell_order_id": "AMO-001", "expected_exit_date": "2026-07-20",
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": "2026-07-17",
        },
        _empty_free_slot(2), _empty_free_slot(3), _empty_free_slot(4),
    ])
    broker = MagicMock()
    broker._data_sdk = None
    broker.get_order_status.return_value = {"status": "REJECTED"}
    broker.get_orders.return_value = [
        # our own rejected AMO (engine tag) — must be ignored
        {"order_id": "AMO-001", "tag": "ITDA_2026-07-17_1",
         "tradingsymbol": "SABEVENTS", "transaction_type": "SELL",
         "status": "REJECTED", "filled_quantity": 0, "average_price": 0},
        # the user's manual market sell — must be adopted
        {"order_id": "MANUAL-1", "tag": None,
         "tradingsymbol": "SABEVENTS", "transaction_type": "SELL",
         "status": "COMPLETE", "filled_quantity": 6426, "average_price": 7.51},
    ]

    summary = run_verify_exit(cfg, broker,
                              now_ist=pd.Timestamp("2026-07-20 09:33:00"),
                              paper_mode=False)
    assert summary["settled_count"] == 1
    slot1 = next(s for s in json.loads(state_path.read_text())["slots"]
                 if s["slot_id"] == 1)
    assert slot1["status"] == "t1_settling"
    assert slot1["sell_fill_price"] == 7.51
    # no failsafe order placed
    broker.place_order.assert_not_called()


def _live_slot_state(state_path, *, amo_id="AMO-001"):
    """Seed one t0_open live slot expiring 2026-07-20 (a Monday)."""
    _seed_state(state_path, [
        {
            "slot_id": 1, "status": "t0_open", "symbol": "NSE:REGENCERAM",
            "product": "CNC", "leverage": 1.0,
            "margin_inr": 50000.0, "notional_inr": 50000.0,
            "buy_fill_price": 36.3757, "buy_fill_ts": "2026-07-17T15:26:00",
            "buy_order_id": "BUY-001", "buy_qty": 1388,
            "amo_sell_order_id": amo_id, "expected_exit_date": "2026-07-20",
            "sell_fill_price": None, "sell_fill_ts": None,
            "realized_pnl_inr": None, "fees_inr": None, "interest_inr": None,
            "reserved_today": "2026-07-17",
        },
        _empty_free_slot(2), _empty_free_slot(3), _empty_free_slot(4),
    ])


def test_verify_exit_auth_preflight_aborts_cleanly(tmp_path):
    """Dead token -> abort before any state/broker half-actions (2026-07-24)."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    broker = MagicMock()
    broker.check_auth.return_value = False
    summary = run_verify_exit(cfg, broker,
                              now_ist=pd.Timestamp("2026-07-20 09:33:00"),
                              paper_mode=False)
    assert summary.get("auth_failed") is True
    assert summary["settled_count"] == 0
    broker.get_order_status.assert_not_called()
    broker.place_order.assert_not_called()


def test_verify_exit_open_partial_amo_gets_repriced_not_doubled(tmp_path):
    """Still-OPEN partially-filled AMO locks holdings: re-price it via
    modify_order, never place a second SELL (REGENCERAM 2026-07-24)."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    broker = MagicMock()
    broker._data_sdk = None
    broker.check_auth.return_value = True
    broker.get_order_status.return_value = {
        "status": "OPEN", "filled_quantity": 286, "average_price": 34.6}
    broker.get_orders.return_value = []
    broker.get_quote.return_value = {
        "last_price": 34.0, "upper_circuit_limit": 41.56,
        "lower_circuit_limit": 27.72}
    broker.modify_order.return_value = True
    summary = run_verify_exit(cfg, broker,
                              now_ist=pd.Timestamp("2026-07-20 09:33:00"),
                              paper_mode=False)
    assert summary["settled_count"] == 0  # settle deferred to next run
    broker.modify_order.assert_called_once()
    args, kwargs = broker.modify_order.call_args
    assert args[0] == "AMO-001"
    assert args[1] <= 34.0  # marketable: at/below LTP
    broker.place_order.assert_not_called()
    slot1 = next(s for s in json.loads(state_path.read_text())["slots"]
                 if s["slot_id"] == 1)
    assert slot1["status"] == "t0_open"  # untouched


def test_verify_exit_dead_partial_amo_failsafes_remainder_blended(tmp_path):
    """CANCELLED AMO with a partial fill: failsafe only the unsold remainder
    and settle at the qty-blended price."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    broker = MagicMock()
    broker._data_sdk = None
    broker.check_auth.return_value = True
    broker.get_order_status.side_effect = _statuses({
        "AMO-001": {"status": "CANCELLED", "filled_quantity": 286, "average_price": 34.6},
        "FAILSAFE-1": {"status": "COMPLETE", "filled_quantity": 1102, "average_price": 33.9},
    })
    broker.get_orders.return_value = []
    broker.get_quote.return_value = {
        "last_price": 34.0, "upper_circuit_limit": 41.56,
        "lower_circuit_limit": 27.72}
    broker.place_order.return_value = "FAILSAFE-1"
    summary = run_verify_exit(cfg, broker,
                              now_ist=pd.Timestamp("2026-07-20 09:33:00"),
                              paper_mode=False)
    assert summary["settled_count"] == 1
    _, po_kwargs = broker.place_order.call_args
    assert po_kwargs["qty"] == 1388 - 286  # remainder only
    assert po_kwargs["side"] == "SELL"
    slot1 = _slot1(state_path)
    # blended on the failsafe's REAL fill, not LTP
    expected = (286 * 34.6 + (1388 - 286) * 33.9) / 1388
    assert slot1["sell_fill_price"] == pytest.approx(expected, abs=1e-6)


def _statuses(by_id):
    return lambda oid: by_id.get(oid)


def _slot1(state_path):
    return next(s for s in json.loads(state_path.read_text())["slots"] if s["slot_id"] == 1)


def _failsafe_broker(statuses):
    broker = MagicMock()
    broker._data_sdk = None
    broker.check_auth.return_value = True
    broker.get_order_status.side_effect = _statuses(statuses)
    broker.get_orders.return_value = []
    broker.get_quote.return_value = {
        "last_price": 28.4, "upper_circuit_limit": 33.74, "lower_circuit_limit": 22.5}
    broker.modify_order.return_value = True
    return broker


def test_verify_exit_unfilled_failsafe_keeps_slot_open(tmp_path):
    """ATL 2026-10-19: AMO rejected, failsafe only partly filled when the pass
    ended -> the slot must NOT settle (and its GTT must stay), the failsafe id
    is recorded for the next pass."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    broker = _failsafe_broker({
        "AMO-001": {"status": "REJECTED", "status_message": "16418 : Order with Invalid attributes"},
        "FS-1": {"status": "OPEN", "filled_quantity": 350, "average_price": 28.26},
    })
    broker.place_order.return_value = "FS-1"
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:16:30"),
                              paper_mode=False)
    assert summary["settled_count"] == 0
    s = _slot1(state_path)
    assert s["status"] == "t0_open"
    assert s["failsafe_sell_order_id"] == "FS-1"
    broker.cancel_gtt.assert_not_called()


def test_verify_exit_next_pass_settles_completed_failsafe(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(failsafe_sell_order_id="FS-1", exit_sold_qty=0, exit_sold_value=0.0)
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({
        "AMO-001": {"status": "REJECTED"},
        "FS-1": {"status": "COMPLETE", "filled_quantity": 1388, "average_price": 28.26},
    })
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:30:30"),
                              paper_mode=False)
    assert summary["settled_count"] == 1
    s = _slot1(state_path)
    assert s["status"] == "t1_settling"
    assert s["sell_fill_price"] == pytest.approx(28.26)
    broker.place_order.assert_not_called()


def test_verify_exit_next_pass_reprices_working_failsafe(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(failsafe_sell_order_id="FS-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({"FS-1": {"status": "OPEN", "filled_quantity": 350,
                                        "average_price": 28.26}})
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:30:30"),
                              paper_mode=False)
    assert summary["settled_count"] == 0
    args, _ = broker.modify_order.call_args
    assert args[0] == "FS-1" and args[1] <= 28.4
    broker.place_order.assert_not_called()


def test_verify_exit_dead_failsafe_banks_partial_and_refails_remainder(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(failsafe_sell_order_id="FS-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({
        "FS-1": {"status": "CANCELLED", "filled_quantity": 350, "average_price": 28.26},
        "FS-2": {"status": "COMPLETE", "filled_quantity": 1038, "average_price": 28.0},
    })
    broker.place_order.return_value = "FS-2"
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:30:30"),
                              paper_mode=False)
    assert summary["settled_count"] == 1
    assert broker.place_order.call_args.kwargs["qty"] == 1388 - 350
    s = _slot1(state_path)
    assert s["sell_fill_price"] == pytest.approx((350 * 28.26 + 1038 * 28.0) / 1388)
    assert s["exit_banked_order_id"] == "FS-1"
    # released slot later clears the new fields
    from services.capital_manager import OvernightSlotPool
    pool = OvernightSlotPool(state_path, max_slots=4, margin_per_slot=100000, max_new_per_day=2)
    pool.release(1, date(2026, 7, 21))
    slot = pool._get_slot(1)  # noqa: SLF001
    assert (slot.failsafe_sell_order_id, slot.exit_sold_qty, slot.exit_sold_value,
            slot.exit_banked_order_id) == (None, 0, 0.0, None)


def test_verify_exit_dead_failsafe_partial_banked_once_when_replacement_fails(tmp_path):
    """Replacement failsafe fails to place twice -> the dead order's 350 is
    banked ONCE, the slot never settles while shares are held."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(failsafe_sell_order_id="FS-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({
        "FS-1": {"status": "REJECTED", "filled_quantity": 350, "average_price": 28.26}})
    broker.place_order.side_effect = RuntimeError("16418 : Order with Invalid attributes")
    for ts in ("2026-07-20 09:30:30", "2026-07-20 10:30:30"):
        summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp(ts), paper_mode=False)
        assert summary["settled_count"] == 0
    s = _slot1(state_path)
    assert s["status"] == "t0_open"
    assert s["exit_sold_qty"] == 350
    assert broker.place_order.call_args.kwargs["qty"] == 1388 - 350


def test_verify_exit_stale_slot_with_pending_failsafe_keeps_gtt(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(failsafe_sell_order_id="FS-1", gtt_id="GTT-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({})        # yesterday's FS-1: no status today
    broker.get_holdings.return_value = [
        {"tradingsymbol": "REGENCERAM", "quantity": 1388, "t1_quantity": 0, "used_quantity": 0}]
    broker.place_order.return_value = "FS-2"
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-21 09:16:30"),
                              paper_mode=False)
    # Shares still held: sold now; the GTT stays until that sell fills.
    assert summary["orphan_t0_count"] == 1
    assert broker.place_order.call_args.kwargs["qty"] == 1388
    broker.cancel_gtt.assert_not_called()
    assert _slot1(state_path)["gtt_id"] == "GTT-1"
    assert _slot1(state_path)["failsafe_sell_order_id"] == "FS-2"


def test_verify_exit_stale_slot_still_held_sells_and_settles(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(gtt_id="GTT-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({"FS-2": {"status": "COMPLETE", "filled_quantity": 1388,
                                        "average_price": 28.3}})
    broker.get_holdings.return_value = [
        {"tradingsymbol": "REGENCERAM", "quantity": 0, "t1_quantity": 1388, "used_quantity": 0}]
    broker.place_order.return_value = "FS-2"
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-21 09:16:30"),
                              paper_mode=False)
    assert summary["settled_count"] == 1
    s1 = _slot1(state_path)
    # Settled at the real fill; the exit day is past, so the same pass releases it.
    assert summary["released_count"] == 1 and abs(summary["events"][0]["sell_price"] - 28.3) < 1e-9
    broker.cancel_gtt.assert_called_once_with("GTT-1")


def test_verify_exit_stale_slot_flat_cancels_gtt(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(gtt_id="GTT-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({})
    broker.get_holdings.return_value = []
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-21 09:16:30"),
                              paper_mode=False)
    # Flat reading -> manual settlement; the GTT is kept (a false 'flat' must not
    # strip real shares of their stop).
    assert summary["orphan_t0_count"] == 1
    broker.place_order.assert_not_called()
    broker.cancel_gtt.assert_not_called()
    assert _slot1(state_path)["gtt_id"] == "GTT-1"


@pytest.mark.parametrize("row, held", [
    # real Kite rows 2026-10-30 after the open sale: fields are already net of it
    ({"tradingsymbol": "MKEXIM", "quantity": 0, "t1_quantity": 0, "used_quantity": 426}, 0),
    ({"tradingsymbol": "MKEXIM", "quantity": 0, "t1_quantity": 0, "used_quantity": 0,
      "mtf": {"quantity": 0, "used_quantity": 72}}, 0),
    # partly sold: 400 of 1388 gone -> 988 still held (used_quantity not taken off again)
    ({"tradingsymbol": "MKEXIM", "quantity": 0, "t1_quantity": 988, "used_quantity": 400}, 988),
    ({"tradingsymbol": "MKEXIM", "quantity": 0, "t1_quantity": 0, "used_quantity": 400,
      "mtf": {"quantity": 988, "used_quantity": 400}}, 988),
])
def test_held_unsold_qty_reads_kite_quantities_as_net(row, held):
    from services.execution.overnight_handlers import _held_unsold_qty
    broker = MagicMock()
    broker.get_holdings.return_value = [row]
    slot = MagicMock(symbol="NSE:MKEXIM", slot_id=1)
    pool = MagicMock()
    pool.active.return_value = []
    assert _held_unsold_qty(broker, slot, pool) == held


def test_verify_exit_stale_slot_holdings_unreadable_keeps_gtt(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(gtt_id="GTT-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({})
    broker.get_holdings.side_effect = RuntimeError("timeout")
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-21 09:16:30"),
                              paper_mode=False)
    assert summary["orphan_t0_count"] == 1
    broker.place_order.assert_not_called()
    broker.cancel_gtt.assert_not_called()
    assert _slot1(state_path)["gtt_id"] == "GTT-1"


def test_verify_exit_stale_slot_other_slot_same_symbol_not_counted(tmp_path):
    """Holdings that belong to another open slot in the same symbol do not make
    the stale slot look held."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(gtt_id="GTT-1")
    st["slots"][1].update(slot_id=2, status="t0_open", symbol="NSE:REGENCERAM", product="CNC",
                          notional_inr=20000.0, buy_fill_price=28.0, buy_qty=700,
                          amo_sell_order_id="AMO-2", expected_exit_date="2026-07-22",
                          reserved_today="2026-07-21")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({})
    broker.get_holdings.return_value = [
        {"tradingsymbol": "REGENCERAM", "quantity": 0, "t1_quantity": 700, "used_quantity": 0}]
    run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-21 09:16:30"), paper_mode=False)
    broker.place_order.assert_not_called()


def test_verify_exit_stale_slot_resumes_todays_failsafe_by_tag(tmp_path):
    """A failsafe placed by an earlier pass today is found by its tag and driven
    to completion, never a second SELL."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(failsafe_sell_order_id="FS-OLD", gtt_id="GTT-1")
    state_path.write_text(json.dumps(st))
    broker = _failsafe_broker({"FS-9": {"status": "COMPLETE", "filled_quantity": 1388,
                                        "average_price": 28.1}})
    broker.get_orders.return_value = [
        {"order_id": "FS-9", "tag": "ITDA_S_26-07-17F1", "tradingsymbol": "REGENCERAM",
         "transaction_type": "SELL", "status": "COMPLETE"}]
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-21 10:30:00"),
                              paper_mode=False)
    broker.place_order.assert_not_called()
    assert summary["settled_count"] == 1


def test_failsafe_carries_deterministic_tag(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    broker = _failsafe_broker({"AMO-001": {"status": "REJECTED"}})
    broker.place_order.return_value = "FS-1"
    run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:16:30"), paper_mode=False)
    assert broker.place_order.call_args.kwargs["trade_id"][-12:] == "S_26-07-17F1"


def test_verify_exit_no_amo_unreadable_orderbook_places_nothing(tmp_path):
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _no_amo_state(state_path)
    broker = _failsafe_broker({})
    broker.get_orders.side_effect = RuntimeError("429")
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:16:30"),
                              paper_mode=False)
    broker.place_order.assert_not_called()
    assert summary["settled_count"] == 0


def _no_amo_state(state_path):
    _live_slot_state(state_path, amo_id=None)
    st = json.loads(state_path.read_text())
    st["slots"][0].update(expected_exit_date=None)
    state_path.write_text(json.dumps(st))


def test_verify_exit_failed_place_exit_sells_at_open(tmp_path):
    """16:05 place-exit failed (no AMO, no exit date): the next morning's verify
    sells the position with a failsafe and settles on its fill."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _no_amo_state(state_path)
    broker = _failsafe_broker({"FS-1": {"status": "COMPLETE", "filled_quantity": 1388,
                                        "average_price": 37.1}})
    broker.place_order.return_value = "FS-1"
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:16:30"),
                              paper_mode=False)
    assert summary["settled_count"] == 1 and summary["orphan_t0_count"] == 0
    assert broker.place_order.call_args.kwargs["qty"] == 1388
    s1 = _slot1(state_path)
    assert s1["expected_exit_date"] == "2026-07-20" and abs(s1["sell_fill_price"] - 37.1) < 1e-9


def test_verify_exit_failed_place_exit_adopts_unrecorded_amo(tmp_path):
    """The AMO reached the broker but the placement raised: adopt it, never a
    second SELL."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _no_amo_state(state_path)
    broker = _failsafe_broker({"AMO-X": {"status": "COMPLETE", "average_price": 37.4,
                                         "filled_quantity": 1388}})
    broker.get_orders.return_value = [
        {"order_id": "AMO-X", "tag": "ITDA_2026-07-17_1", "tradingsymbol": "REGENCERAM",
         "transaction_type": "SELL", "status": "COMPLETE"}]
    summary = run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:16:30"),
                              paper_mode=False)
    broker.place_order.assert_not_called()
    assert summary["settled_count"] == 1
    assert _slot1(state_path)["amo_sell_order_id"] == "AMO-X"


def test_run_entry_excludes_no_band_names(tmp_path, monkeypatch, caplog):
    """F&O / CAS names (NSE 'No Band') are dropped from the entry universe;
    names absent from the list are kept."""
    import logging
    import services.price_bands as pbm
    import services.setup_universe as su

    class _Sess:
        bands = {"RELIANCE": float("nan"), "SMALLCO": 20.0}
    class _Store:
        def session(self, d):
            return _Sess()
    monkeypatch.setattr(pbm, "get_price_band_store", lambda cfg: _Store())
    monkeypatch.setattr(su, "close_dn_overnight_long_universe",
                        lambda dd, d, c: {"NSE:RELIANCE", "NSE:SMALLCO", "NSE:RENAMED"})
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["exclude_no_band_names"] = True
    broker = MagicMock()
    broker.list_symbols.return_value = []
    broker._data_sdk = None
    with caplog.at_level(logging.INFO):
        run_entry(cfg, broker, now_ist=pd.Timestamp("2026-05-21 15:25:00"))
    assert any("F&O/No-Band filter" in r.getMessage() and "3 -> 2" in r.getMessage()
               for r in caplog.records)


def test_run_entry_no_band_list_means_no_entries(tmp_path, monkeypatch):
    import services.price_bands as pbm
    import services.setup_universe as su

    class _Store:
        def session(self, d):
            raise pbm.PriceBandDataError("missing")
    monkeypatch.setattr(pbm, "get_price_band_store", lambda cfg: _Store())
    monkeypatch.setattr(su, "close_dn_overnight_long_universe", lambda dd, d, c: {"NSE:SMALLCO"})
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["exclude_no_band_names"] = True
    broker = MagicMock()
    broker.list_symbols.return_value = []
    broker._data_sdk = None
    summary = run_entry(cfg, broker, now_ist=pd.Timestamp("2026-05-21 15:25:00"))
    assert summary["fired_count"] == 0
    broker.place_order.assert_not_called()


def test_verify_exit_failsafe_id_persisted_before_poll(tmp_path):
    """The failsafe id must be on disk before the fill poll, so a run that dies
    mid-poll cannot lead the next pass to place a second SELL."""
    state_path = tmp_path / "overnight_slots.json"
    cfg = _minimal_config(state_path)
    cfg["setups"]["close_dn_overnight_long"]["enabled"] = True
    _live_slot_state(state_path)
    broker = _failsafe_broker({"AMO-001": {"status": "REJECTED"}})
    broker.place_order.return_value = "FS-1"

    def die_on_failsafe_poll(oid):
        if oid == "FS-1":
            raise SystemExit("killed mid-poll")
        return {"status": "REJECTED"}
    broker.get_order_status.side_effect = die_on_failsafe_poll
    with pytest.raises(SystemExit):
        run_verify_exit(cfg, broker, now_ist=pd.Timestamp("2026-07-20 09:16:30"),
                        paper_mode=False)
    assert _slot1(state_path)["failsafe_sell_order_id"] == "FS-1"


# ---------------------------------------------------------------------------
# Cancel-fill race reconcile (2026-07-24 CREATIVEYE: cancel-confirm status
# said CANCELLED/filled=0, slot freed, 7,520 real shares left unhedged)
# ---------------------------------------------------------------------------

def _pool(tmp_path):
    from services.capital_manager import OvernightSlotPool
    state_path = tmp_path / "overnight_slots.json"
    _seed_state(state_path, [_empty_free_slot(i) for i in range(1, 5)])
    return OvernightSlotPool(state_path, max_slots=4,
                             margin_per_slot=50000, max_new_per_day=2), state_path


def _orph_order(**over):
    o = {"order_id": "B-1", "tag": "ITDA_2026-07-24_5", "tradingsymbol": "CREATIVEYE",
         "transaction_type": "BUY", "product": "CNC", "status": "CANCELLED",
         "filled_quantity": 7520, "average_price": 6.63}
    o.update(over)
    return o


def test_reconcile_attaches_orphaned_buy_fill(tmp_path):
    from services.execution.overnight_handlers import _reconcile_unattached_buys
    pool, state_path = _pool(tmp_path)
    broker = MagicMock()
    broker.get_orders.return_value = [
        _orph_order(),
        _orph_order(order_id="X-1", tag=None),                 # manual — ignored
        _orph_order(order_id="X-2", transaction_type="SELL"),  # sell — ignored
        _orph_order(order_id="X-3", product="MIS"),            # intraday — ignored
        _orph_order(order_id="X-4", filled_quantity=0),        # no fill — ignored
    ]
    summary = {}
    n = _reconcile_unattached_buys(broker, pool, pd.Timestamp("2026-07-24 15:28:00"), summary)
    assert n == 1
    slots = json.loads(state_path.read_text())["slots"]
    s = next(x for x in slots if x["status"] == "t0_open")
    assert s["symbol"] == "NSE:CREATIVEYE"
    assert s["buy_qty"] == 7520
    assert s["buy_fill_price"] == pytest.approx(6.63)
    assert s["buy_order_id"] == "B-1"
    assert summary["reconciled_attach"][0]["symbol"] == "NSE:CREATIVEYE"


def test_reconcile_skips_already_attached(tmp_path):
    from services.execution.overnight_handlers import _reconcile_unattached_buys
    pool, _ = _pool(tmp_path)
    slot = pool.reserve("NSE:CREATIVEYE", "CNC", 1.0, date(2026, 7, 24))
    pool.attach_buy_fill(slot.slot_id, fill_price=6.63,
                         fill_ts_iso="2026-07-24T15:26:00", order_id="B-1", qty=7520)
    broker = MagicMock()
    broker.get_orders.return_value = [_orph_order()]
    n = _reconcile_unattached_buys(broker, pool, pd.Timestamp("2026-07-24 15:28:00"), {})
    assert n == 0


def test_reconcile_bypasses_daily_cap_but_not_full_pool(tmp_path):
    from services.execution.overnight_handlers import _reconcile_unattached_buys
    pool, _ = _pool(tmp_path)
    # exhaust the daily-new cap (max_new_per_day=2) with other fills
    for i, sym in enumerate(["NSE:AAA", "NSE:BBB"]):
        s = pool.reserve(sym, "CNC", 1.0, date(2026, 7, 24))
        pool.attach_buy_fill(s.slot_id, 10.0, "2026-07-24T15:26:00", f"O-{i}", qty=100)
    broker = MagicMock()
    broker.get_orders.return_value = [_orph_order()]
    summary = {}
    n = _reconcile_unattached_buys(broker, pool, pd.Timestamp("2026-07-24 15:28:00"), summary)
    assert n == 1  # cap bypassed — the fill is a fact
    # now fill the pool completely and try another orphan
    for sym in ["NSE:CCC"]:
        free = [s for s in pool._slots if s.status == "free"]
        s = free[0]; s.status = "t0_open"; s.symbol = sym
    broker.get_orders.return_value = [_orph_order(order_id="B-2", tradingsymbol="ZZZ")]
    summary2 = {}
    n2 = _reconcile_unattached_buys(broker, pool, pd.Timestamp("2026-07-24 15:28:00"), summary2)
    assert n2 == 0
    assert summary2["reconcile_unhedged"] == ["NSE:ZZZ"]


def test_find_out_of_band_sell_filters():
    from services.execution.overnight_handlers import _find_out_of_band_sell
    orders = [
        {"order_id": "A1", "tag": "ITDA_x", "tradingsymbol": "X",
         "transaction_type": "SELL", "status": "COMPLETE",
         "filled_quantity": 10, "average_price": 5.0},   # engine-tagged
        {"order_id": "A2", "tag": None, "tradingsymbol": "X",
         "transaction_type": "SELL", "status": "COMPLETE",
         "filled_quantity": 9, "average_price": 5.0},    # qty mismatch
        {"order_id": "A3", "tag": None, "tradingsymbol": "X",
         "transaction_type": "BUY", "status": "COMPLETE",
         "filled_quantity": 10, "average_price": 5.0},   # wrong side
        {"order_id": "A4", "tag": None, "tradingsymbol": "X",
         "transaction_type": "SELL", "status": "COMPLETE",
         "filled_quantity": 10, "average_price": 5.2},   # match
    ]
    broker = MagicMock()
    broker.get_orders.return_value = orders
    assert _find_out_of_band_sell(broker, "NSE:X", 10) == ("A4", 5.2)
    # excluded id is skipped even if COMPLETE
    assert _find_out_of_band_sell(broker, "NSE:X", 10,
                                  exclude_order_id="A4") is None
    # broker without get_orders -> None
    class NoOrders:
        pass
    assert _find_out_of_band_sell(NoOrders(), "NSE:X", 10) is None


# ---------------------------------------------------------------------------
# Per-instrument tick size (2026-07-02 NPST/MIDWESTLTD rejections)
# ---------------------------------------------------------------------------

def test_instrument_tick_prefers_broker_value():
    from services.execution.overnight_handlers import _instrument_tick
    broker = MagicMock()
    broker.get_tick_size.return_value = 0.10
    assert _instrument_tick(broker, "NSE:NPST", {"tick_size_fallback_inr": 0.05}) == 0.10


def test_instrument_tick_falls_back_on_none_mock_or_missing():
    from services.execution.overnight_handlers import _instrument_tick
    cfg = {"tick_size_fallback_inr": 0.05}
    # None (dry_run / symbol not in dump) -> fallback
    broker = MagicMock()
    broker.get_tick_size.return_value = None
    assert _instrument_tick(broker, "NSE:X", cfg) == 0.05
    # Non-numeric return (raw MagicMock: float() == 1.0 would poison prices
    # if it leaked through) -> fallback
    broker2 = MagicMock()
    assert _instrument_tick(broker2, "NSE:X", cfg) == 0.05
    # Broker without the method at all -> fallback
    class NoTick:
        pass
    assert _instrument_tick(NoTick(), "NSE:X", cfg) == 0.05


def test_place_buy_rounds_to_instrument_tick():
    """Regression for 2026-07-02: a 0.10-tick scrip priced at a 0.05-odd limit
    was REJECTED outright ('Tick size for this script is 0.10'), silently
    dropping the top-2 ranked fires of the day. The BUY limit must round to
    the INSTRUMENT tick, not the blanket 0.05."""
    from services.execution.overnight_handlers import _place_buy
    broker = MagicMock()
    broker.place_order.return_value = "OID-1"
    _place_buy(broker, symbol="NSE:NPST", qty=10, product="MTF",
               paper_mode=False, trade_id="T", limit_price=256.85,
               tick_size=0.10)
    sent = broker.place_order.call_args.kwargs["price"]
    assert sent == pytest.approx(256.8)  # nearest 0.10 multiple (half-even), valid tick
    assert round(sent / 0.10) * 0.10 == pytest.approx(sent)


def test_place_amo_sell_and_failsafe_round_to_instrument_tick():
    from services.execution.overnight_handlers import (
        _place_amo_sell, _place_failsafe_sell)
    broker = MagicMock()
    broker.place_order.return_value = "OID-2"
    _place_amo_sell(broker, symbol="NSE:NPST", qty=10, product="MTF",
                    paper_mode=False, trade_id="T", limit_price=241.65,
                    tick_size=0.10)
    assert broker.place_order.call_args.kwargs["price"] == pytest.approx(241.6)
    _place_failsafe_sell(broker, symbol="NSE:NPST", qty=10, product="MTF",
                         limit_price=239.95, tick_size=0.10, trade_id="OVERNIGHT_FS_26-07-17F1")
    sent = broker.place_order.call_args.kwargs["price"]
    assert sent == pytest.approx(239.9)  # float repr of 239.95/0.10 rounds down
    assert round(sent / 0.10) * 0.10 == pytest.approx(sent)


# ---------------------------------------------------------------------------
# Entry placement cutoff (2026-07-30 'Markets are closed' overrun)
# ---------------------------------------------------------------------------

def test_entry_cutoff_never_fires_in_paper_mode():
    from services.execution.overnight_handlers import _entry_cutoff_reached
    cfg = {"entry_placement_cutoff_hhmmss": "15:29:20"}
    assert _entry_cutoff_reached(cfg, paper_mode=True) is False


def test_entry_cutoff_live_respects_wall_clock(monkeypatch):
    import services.execution.overnight_handlers as oh
    cfg = {"entry_placement_cutoff_hhmmss": "15:29:20"}
    monkeypatch.setattr(
        oh, "_now_naive_ist", lambda: pd.Timestamp("2026-07-30 15:29:20"))
    assert oh._entry_cutoff_reached(cfg, paper_mode=False) is True
    monkeypatch.setattr(
        oh, "_now_naive_ist", lambda: pd.Timestamp("2026-07-30 15:28:59"))
    assert oh._entry_cutoff_reached(cfg, paper_mode=False) is False


def test_entry_cutoff_missing_config_key_fails_fast():
    from services.execution.overnight_handlers import _entry_cutoff_reached
    with pytest.raises(KeyError):
        _entry_cutoff_reached({}, paper_mode=False)
