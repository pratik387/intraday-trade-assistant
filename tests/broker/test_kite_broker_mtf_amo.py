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
