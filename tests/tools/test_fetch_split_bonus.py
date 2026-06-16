"""Tests for tools.corporate_actions.fetch_split_bonus parsing.

Produces data/corporate_actions/split_bonus_events.parquet — the CA-events file the
multi-day CNC/MTF executor reads to exclude names with a split/bonus ex-date inside
the K-day hold (mtf_capitulation_handlers._load_ca_ex_dates). Output must carry
`symbol` (NSE:TICKER) + `ex_date`, the keys that loader reads.
"""
from datetime import date

from tools.corporate_actions.fetch_split_bonus import (
    classify_ca_type,
    parse_split_bonus_actions,
)


def test_classify_bonus():
    assert classify_ca_type("Bonus 1:1") == "bonus"
    assert classify_ca_type("BONUS issue in ratio 2:1") == "bonus"


def test_classify_split():
    assert classify_ca_type("Face Value Split (Sub-Division) - From Rs 10/- to Re 1/-") == "split"
    assert classify_ca_type("Sub-division of shares") == "split"
    assert classify_ca_type("Stock Split From Rs 10 to Rs 2") == "split"


def test_classify_non_ca_returns_none():
    assert classify_ca_type("Interim Dividend Rs 5 Per Share") is None
    assert classify_ca_type("Annual General Meeting") is None
    assert classify_ca_type("") is None


def test_parse_keeps_only_bonus_split_with_exdate():
    items = [
        {"symbol": "INFY", "subject": "Bonus 1:1", "exDate": "15-Jun-2026", "series": "EQ"},
        {"symbol": "TCS", "subject": "Face Value Split - From Rs 10 to Re 1", "exDate": "20-Jun-2026"},
        {"symbol": "HDFCBANK", "subject": "Interim Dividend Rs 19", "exDate": "10-Jun-2026"},  # drop
        {"symbol": "NOEX", "subject": "Bonus 1:1", "exDate": "-"},   # no ex-date -> drop
        {"symbol": "", "subject": "Bonus 1:1", "exDate": "15-Jun-2026"},  # no symbol -> drop
    ]
    rows = parse_split_bonus_actions(items)
    syms = {r["symbol"]: r for r in rows}
    assert set(syms) == {"NSE:INFY", "NSE:TCS"}
    assert syms["NSE:INFY"]["ca_type"] == "bonus"
    assert syms["NSE:INFY"]["ex_date"] == date(2026, 6, 15)
    assert syms["NSE:TCS"]["ca_type"] == "split"
    assert syms["NSE:TCS"]["ex_date"] == date(2026, 6, 20)
    # schema the executor's _load_ca_ex_dates reads
    for r in rows:
        assert "symbol" in r and "ex_date" in r and r["symbol"].startswith("NSE:")
