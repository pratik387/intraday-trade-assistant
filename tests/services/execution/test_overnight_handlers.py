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
                "catastrophe_stop_pct": 5.0,
                "gtt_limit_buffer_pct": 0.5,
                "entry_limit_buffer_pct": 1.0,
                "insufficient_funds_retry_haircut": 0.95,
                "fill_poll_timeout_sec": 60,
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
    """A t0_open slot lacking expected_exit_date is reported as orphan."""
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
            "reserved_today": "2026-05-21",
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

def test_rank_detections_deepest_svr_first_cheap_tiebreak():
    """Slots are allocated deepest-capitulation-first (confidence = |svr| desc),
    tiebreak cheaper entry price — replacing arbitrary set-iteration order."""
    from types import SimpleNamespace as NS
    from services.execution.overnight_handlers import _rank_detections

    dets = [
        ("NSE:MILD",  NS(confidence=0.55), NS(entry_price=100.0)),
        ("NSE:DEEP",  NS(confidence=0.95), NS(entry_price=500.0)),
        ("NSE:TIE_EXPENSIVE", NS(confidence=0.80), NS(entry_price=900.0)),
        ("NSE:TIE_CHEAP",     NS(confidence=0.80), NS(entry_price=50.0)),
    ]
    ranked = [s for s, _, _ in _rank_detections(dets)]
    assert ranked == ["NSE:DEEP", "NSE:TIE_CHEAP", "NSE:TIE_EXPENSIVE", "NSE:MILD"]
    # pure function: input untouched
    assert dets[0][0] == "NSE:MILD"
