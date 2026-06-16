"""Tests for the overnight ledger-detail backfill matcher.

The tripwire ledger stored net-only rows before the detail-persist change.
Each settled trade's full detail survived in that day's 16:00 OCI snapshot of
overnight_slots.json (captured while the slot was still t1_settling). The
backfill matches snapshot slots to ledger rows on (settle_date, net_pnl) and
enriches in place — preserving net_pnl_inr + ts_iso exactly.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.backfill_overnight_ledger_detail import slot_to_detail, enrich_ledger_trades


def _slot(symbol, buy, sell, fees, interest, realized, notional):
    return {
        "status": "t1_settling", "symbol": symbol,
        "buy_fill_price": buy, "sell_fill_price": sell,
        "fees_inr": fees, "interest_inr": interest,
        "realized_pnl_inr": realized, "notional_inr": notional,
    }


def test_slot_to_detail_derives_fields():
    d = slot_to_detail(_slot("NSE:AIIL", 521.9, 531.0, 598.35, 181.24, 3597.51, 521.9 * 100))
    assert d["symbol"] == "NSE:AIIL"
    assert d["entry_price"] == 521.9
    assert d["exit_price"] == 531.0
    assert round(d["fees_inr"], 2) == round(598.35 + 181.24, 2)   # fees + interest
    assert d["exit_reason"] == "t1_settle"
    assert d["qty"] == 100
    # gross = net + total cost
    assert round(d["gross_pnl_inr"], 2) == round(3597.51 + 598.35 + 181.24, 2)


def test_enrich_matches_on_date_and_pnl():
    trades = [
        {"net_pnl_inr": 135.5972946944051, "ts_iso": "2026-06-15T09:30:01"},
        {"net_pnl_inr": 2945.41724696, "ts_iso": "2026-06-15T09:30:01"},
    ]
    slots_by_date = {
        "2026-06-15": [
            _slot("NSE:SAMPANN", 27.9, 28.0, 222.8, 0.0, 135.5972946944051, 27.9 * 100),
            _slot("NSE:CLSEL", 293.0, 297.0, 662.45, 216.13, 2945.41724696, 293.0 * 100),
        ]
    }
    enriched, unmatched = enrich_ledger_trades(trades, slots_by_date)
    assert unmatched == []
    assert enriched[0]["symbol"] == "NSE:SAMPANN"
    assert enriched[1]["symbol"] == "NSE:CLSEL"
    # net + ts preserved exactly
    assert enriched[0]["net_pnl_inr"] == 135.5972946944051
    assert enriched[0]["ts_iso"] == "2026-06-15T09:30:01"


def test_enrich_skips_already_detailed_and_reports_unmatched():
    trades = [
        {"net_pnl_inr": 100.0, "ts_iso": "2026-06-15T09:30:01", "symbol": "NSE:X"},  # already has detail
        {"net_pnl_inr": 999.0, "ts_iso": "2026-06-16T09:30:01"},                      # no slot for it
    ]
    slots_by_date = {"2026-06-15": [_slot("NSE:Y", 10, 11, 1, 0, 100.0, 1000)]}
    enriched, unmatched = enrich_ledger_trades(trades, slots_by_date)
    # already-detailed row untouched (not re-matched to NSE:Y)
    assert enriched[0]["symbol"] == "NSE:X"
    assert len(unmatched) == 1 and unmatched[0]["net_pnl_inr"] == 999.0
