"""Tests for KiteBroker MTF + AMO order routing.

Task 5 of close_dn_overnight_long paper-trade implementation.

Verifies:
- product='MTF' is accepted (maps to kc.PRODUCT_MTF)
- variety='amo' is accepted (maps to kc.VARIETY_AMO)
- MTF + AMO combination works (the actual overnight use case)
- Invalid products/varieties still raise ValueError
- MIS-availability check stays MIS-only (not triggered for MTF/CNC)
- Existing MIS/CNC orders still work (regression coverage)
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for all tests (KITE_API_KEY/KITE_ACCESS_TOKEN)."""
    with patch('broker.kite.kite_broker.env') as mock_env:
        mock_env.KITE_API_KEY = "test_api_key"
        mock_env.KITE_ACCESS_TOKEN = "test_access_token"
        yield mock_env


@pytest.fixture
def dry_broker():
    """Dry-run KiteBroker with stubbed KC constants.

    Uses dry_run=True so the order takes the _simulate_order path
    (no real kc.place_order call). We still stub the KC constants
    in case any are referenced before the dry-run early return.
    """
    with patch("broker.kite.kite_broker.KiteConnect") as MockKC:
        instance = MockKC.return_value
        instance.PRODUCT_MIS = "MIS"
        instance.PRODUCT_CNC = "CNC"
        instance.PRODUCT_MTF = "MTF"
        instance.PRODUCT_NRML = "NRML"
        instance.VARIETY_REGULAR = "regular"
        instance.VARIETY_AMO = "amo"
        instance.TRANSACTION_TYPE_BUY = "BUY"
        instance.TRANSACTION_TYPE_SELL = "SELL"
        instance.ORDER_TYPE_MARKET = "MARKET"
        instance.ORDER_TYPE_LIMIT = "LIMIT"

        from broker.kite.kite_broker import KiteBroker

        b = KiteBroker(api_key="test", access_token="test", dry_run=True)
        b.kc = instance
        # Stub LTP fetch so _simulate_order doesn't make network calls
        b.get_ltp = MagicMock(return_value=100.0)
        yield b


def test_place_mtf_order_accepted(dry_broker):
    """product='MTF' is accepted (no ValueError)."""
    order_id = dry_broker.place_order(
        symbol="NSE:RELIANCE", side="BUY", qty=10, product="MTF",
        check_margins=False,
    )
    assert order_id.startswith("PAPER_")


def test_place_amo_variety_accepted(dry_broker):
    """variety='amo' is accepted."""
    order_id = dry_broker.place_order(
        symbol="NSE:RELIANCE", side="SELL", qty=10, product="MTF",
        variety="amo", check_margins=False,
    )
    assert order_id.startswith("PAPER_")


def test_place_mtf_amo_combination(dry_broker):
    """product=MTF + variety=amo combination works (the actual use case)."""
    order_id = dry_broker.place_order(
        symbol="NSE:RELIANCE", side="SELL", qty=10,
        product="MTF", variety="amo", check_margins=False,
    )
    assert order_id.startswith("PAPER_")
    # Verify variety was recorded on the paper order
    assert dry_broker._paper_orders[-1]["variety"] == "amo"
    assert dry_broker._paper_orders[-1]["product"] == "MTF"


def test_invalid_product_rejected(dry_broker):
    """product='BO' (bracket order, not supported) raises ValueError."""
    with pytest.raises(ValueError, match="product must be"):
        dry_broker.place_order(
            symbol="NSE:RELIANCE", side="BUY", qty=10, product="BO",
            check_margins=False,
        )


def test_invalid_variety_rejected(dry_broker):
    """variety='co' (cover order, not in our supported set) raises ValueError."""
    with pytest.raises(ValueError, match="variety must be"):
        dry_broker.place_order(
            symbol="NSE:RELIANCE", side="BUY", qty=10, product="MIS",
            variety="co", check_margins=False,
        )


def test_mis_check_skipped_for_mtf(dry_broker):
    """check_margins=True should NOT trigger MIS-availability check for MTF orders."""
    # If check_mis_availability is called, the test fails.
    with patch.object(dry_broker, "check_mis_availability") as mock_check:
        order_id = dry_broker.place_order(
            symbol="NSE:RELIANCE", side="BUY", qty=10, product="MTF",
            check_margins=True,
        )
        mock_check.assert_not_called()
        assert order_id.startswith("PAPER_")


def test_existing_mis_order_unaffected(dry_broker):
    """Regression: product='MIS' with variety='regular' still works."""
    order_id = dry_broker.place_order(
        symbol="NSE:RELIANCE", side="BUY", qty=10, product="MIS",
        check_margins=False,
    )
    assert order_id.startswith("PAPER_")


def test_existing_cnc_order_unaffected(dry_broker):
    """Regression: product='CNC' still works."""
    order_id = dry_broker.place_order(
        symbol="NSE:RELIANCE", side="BUY", qty=10, product="CNC",
        check_margins=False,
    )
    assert order_id.startswith("PAPER_")


# ---------------------------------------------------------------------------
# Tick-size retry (2026-07-02: 0.10-tick names cost the day's top-2 MTF picks)
# ---------------------------------------------------------------------------

def _live_broker_with_kc():
    """Live-mode broker whose kc is a MagicMock with real constant strings."""
    with patch("broker.kite.kite_broker.KiteConnect") as MockKC:
        kc = MagicMock()
        kc.TRANSACTION_TYPE_BUY = "BUY"; kc.TRANSACTION_TYPE_SELL = "SELL"
        kc.ORDER_TYPE_LIMIT = "LIMIT"; kc.ORDER_TYPE_MARKET = "MARKET"
        kc.PRODUCT_MIS = "MIS"; kc.PRODUCT_CNC = "CNC"; kc.PRODUCT_MTF = "MTF"
        kc.VARIETY_REGULAR = "regular"; kc.VARIETY_AMO = "amo"
        MockKC.return_value = kc
        from broker.kite.kite_broker import KiteBroker
        return KiteBroker(dry_run=False), kc


def test_tick_size_rejection_retries_with_parsed_tick():
    """Kite rejects a 0.05-rounded price on a 0.10-tick script; the broker must
    parse the tick from the error, re-round DIRECTIONALLY (BUY floors) and
    retry once."""
    broker, kc = _live_broker_with_kc()
    calls = []

    def _po(**kw):
        calls.append(dict(kw))
        if len(calls) == 1:
            raise Exception("Tick size for this script is 0.10. Kindly enter price "
                            "in the multiple of tick size for this script")
        return "OID2"

    kc.place_order.side_effect = _po
    oid = broker.place_order(symbol="NSE:NPST", side="BUY", qty=10,
                             order_type="LIMIT", price=101.25, product="MTF",
                             variety="regular", check_margins=False)
    assert oid == "OID2"
    assert len(calls) == 2
    assert calls[0]["price"] == 101.25
    assert calls[1]["price"] == 101.20          # floored to 0.10 tick (BUY)


def test_tick_size_retry_sell_ceils():
    broker, kc = _live_broker_with_kc()
    calls = []

    def _po(**kw):
        calls.append(dict(kw))
        if len(calls) == 1:
            raise Exception("Tick size for this script is 0.10.")
        return "OID2"

    kc.place_order.side_effect = _po
    broker.place_order(symbol="NSE:NPST", side="SELL", qty=10,
                       order_type="LIMIT", price=95.05, product="CNC",
                       variety="amo", check_margins=False)
    assert calls[1]["price"] == 95.10           # ceiled to 0.10 tick (SELL)


def test_non_tick_error_does_not_retry():
    broker, kc = _live_broker_with_kc()
    kc.place_order.side_effect = Exception("Insufficient stock holding in MTF")
    with pytest.raises(RuntimeError):
        broker.place_order(symbol="NSE:X", side="SELL", qty=5,
                           order_type="LIMIT", price=100.0, product="MTF",
                           variety="amo", check_margins=False)
    assert kc.place_order.call_count == 1       # no retry on unrelated errors
