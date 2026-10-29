"""
Unit tests for KiteBroker - MIS blocking and position reconciliation.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock env before importing broker
@pytest.fixture(autouse=True)
def mock_env():
    """Mock environment variables for all tests."""
    with patch('broker.kite.kite_broker.env') as mock_env:
        mock_env.KITE_API_KEY = "test_api_key"
        mock_env.KITE_ACCESS_TOKEN = "test_access_token"
        yield mock_env


class TestMISBlocking:
    """Tests for MIS-blocked stock handling (no CNC fallback)."""

    @pytest.fixture
    def mock_kite_broker(self):
        """Create a KiteBroker instance with mocked KiteConnect."""
        with patch('broker.kite.kite_broker.KiteConnect') as mock_kc:
            from broker.kite.kite_broker import KiteBroker

            # Mock the KiteConnect instance
            mock_instance = MagicMock()
            mock_kc.return_value = mock_instance

            broker = KiteBroker(
                api_key="test_key",
                access_token="test_token",
                dry_run=False
            )
            broker.kc = mock_instance
            yield broker, mock_instance

    def test_mis_blocked_raises_error(self, mock_kite_broker):
        """When MIS is blocked by exchange, should raise RuntimeError (no CNC fallback)."""
        broker, mock_kc = mock_kite_broker

        # Simulate MIS rejection from Zerodha at order placement
        mock_kc.place_order.side_effect = Exception(
            "MIS orders are currently blocked for KUANTUM. Place a CNC order instead."
        )
        mock_kc.PRODUCT_MIS = "MIS"
        mock_kc.VARIETY_REGULAR = "regular"

        with pytest.raises(RuntimeError) as exc_info:
            broker.place_order(
                symbol="NSE:KUANTUM",
                side="BUY",
                qty=100,
                order_type="MARKET",
                product="MIS",
                check_margins=False  # Skip pre-check to test actual order rejection
            )

        assert "MIS blocked" in str(exc_info.value)
        assert "KUANTUM" in str(exc_info.value)

        # Verify no retry with CNC was attempted
        assert mock_kc.place_order.call_count == 1

    def test_mis_blocked_short_raises_error(self, mock_kite_broker):
        """SHORT on MIS-blocked stock should raise RuntimeError."""
        broker, mock_kc = mock_kite_broker

        mock_kc.place_order.side_effect = Exception(
            "MIS orders are currently blocked for XYZSTOCK."
        )
        mock_kc.PRODUCT_MIS = "MIS"
        mock_kc.VARIETY_REGULAR = "regular"

        with pytest.raises(RuntimeError) as exc_info:
            broker.place_order(
                symbol="NSE:XYZSTOCK",
                side="SELL",
                qty=50,
                order_type="MARKET",
                product="MIS",
                check_margins=False  # Skip pre-check to test actual order rejection
            )

        assert "MIS blocked" in str(exc_info.value)

    def test_successful_mis_order(self, mock_kite_broker):
        """Normal MIS order should succeed."""
        broker, mock_kc = mock_kite_broker

        mock_kc.place_order.return_value = "123456789"
        mock_kc.PRODUCT_MIS = "MIS"
        mock_kc.VARIETY_REGULAR = "regular"

        order_id = broker.place_order(
            symbol="NSE:RELIANCE",
            side="BUY",
            qty=10,
            order_type="MARKET",
            product="MIS",
            check_margins=False  # Skip margin check for this test
        )

        assert order_id == "123456789"
        mock_kc.place_order.assert_called_once()


class TestPositionReconciliation:
    """Tests for position reconciliation with broker."""

    @pytest.fixture
    def mock_kite_broker_live(self):
        """Create a live KiteBroker instance with mocked KiteConnect."""
        with patch('broker.kite.kite_broker.KiteConnect') as mock_kc:
            from broker.kite.kite_broker import KiteBroker

            mock_instance = MagicMock()
            mock_kc.return_value = mock_instance

            broker = KiteBroker(
                api_key="test_key",
                access_token="test_token",
                dry_run=False
            )
            broker.kc = mock_instance
            yield broker, mock_instance

    @pytest.fixture
    def mock_kite_broker_paper(self):
        """Create a paper trading KiteBroker instance."""
        with patch('broker.kite.kite_broker.KiteConnect') as mock_kc:
            from broker.kite.kite_broker import KiteBroker

            broker = KiteBroker(
                api_key="test_key",
                access_token="test_token",
                dry_run=True
            )
            yield broker

    def test_reconcile_position_found(self, mock_kite_broker_live):
        """Position found at broker should return position info."""
        broker, mock_kc = mock_kite_broker_live

        # Mock order status as COMPLETE
        mock_kc.orders.return_value = [
            {
                "order_id": "123456",
                "status": "COMPLETE",
                "average_price": 150.50,
                "filled_quantity": 100
            }
        ]

        # Mock positions response (Kite API format)
        mock_kc.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "RELIANCE",
                    "exchange": "NSE",
                    "quantity": 100,  # Positive = LONG
                    "average_price": 150.50,
                    "product": "MIS",
                    "pnl": 25.00
                }
            ],
            "net": []
        }

        result = broker.reconcile_position(
            symbol="NSE:RELIANCE",
            order_id="123456",
            expected_qty=100,
            expected_side="BUY",
            timeout=0.01  # Short timeout for test
        )

        assert result is not None
        assert result["qty"] == 100
        assert result["avg_price"] == 150.50
        assert result["product"] == "MIS"
        assert result["verified"] is True

    def test_reconcile_position_not_found(self, mock_kite_broker_live):
        """Position not at broker should return None."""
        broker, mock_kc = mock_kite_broker_live

        # Mock order status as REJECTED
        mock_kc.orders.return_value = [
            {
                "order_id": "123456",
                "status": "REJECTED",
                "average_price": 0,
                "filled_quantity": 0
            }
        ]

        result = broker.reconcile_position(
            symbol="NSE:RELIANCE",
            order_id="123456",
            expected_qty=100,
            expected_side="BUY",
            timeout=0.01
        )

        assert result is None

    def test_reconcile_position_short(self, mock_kite_broker_live):
        """SHORT position should have negative quantity at broker."""
        broker, mock_kc = mock_kite_broker_live

        mock_kc.orders.return_value = [
            {
                "order_id": "789",
                "status": "COMPLETE",
                "average_price": 200.00,
                "filled_quantity": 50
            }
        ]

        mock_kc.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "INFY",
                    "exchange": "NSE",
                    "quantity": -50,  # Negative = SHORT
                    "average_price": 200.00,
                    "product": "MIS",
                    "pnl": -10.00
                }
            ],
            "net": []
        }

        result = broker.reconcile_position(
            symbol="NSE:INFY",
            order_id="789",
            expected_qty=50,
            expected_side="SELL",
            timeout=0.01
        )

        assert result is not None
        assert result["qty"] == 50  # Returns absolute qty
        assert result["verified"] is True

    def test_reconcile_paper_trading(self, mock_kite_broker_paper):
        """Paper trading reconciliation uses internal order list."""
        broker = mock_kite_broker_paper

        # Simulate a paper order
        broker._paper_orders = [
            {
                "order_id": "PAPER_00000001",
                "status": "COMPLETE",
                "quantity": 100,
                "average_price": 150.00,
                "product": "MIS"
            }
        ]

        result = broker.reconcile_position(
            symbol="NSE:TESTSTOCK",
            order_id="PAPER_00000001",
            expected_qty=100,
            expected_side="BUY",
            timeout=0
        )

        assert result is not None
        assert result["qty"] == 100
        assert result["avg_price"] == 150.00
        assert result["verified"] is True

    def test_reconcile_paper_order_not_found(self, mock_kite_broker_paper):
        """Paper trading with non-existent order returns None."""
        broker = mock_kite_broker_paper
        broker._paper_orders = []

        result = broker.reconcile_position(
            symbol="NSE:TESTSTOCK",
            order_id="PAPER_NONEXISTENT",
            expected_qty=100,
            expected_side="BUY",
            timeout=0
        )

        assert result is None


class TestCapitalManagerMISBlocking:
    """Tests for MIS blocking in CapitalManager (paper trading)."""

    @pytest.fixture
    def capital_manager_with_mis_fetcher(self):
        """Create CapitalManager with mocked MIS fetcher."""
        from services.capital_manager import CapitalManager

        # Mock the MIS fetcher
        mock_fetcher = Mock()
        mock_fetcher.is_mis_allowed.return_value = False  # Stock not in MIS list

        cm = CapitalManager(
            enabled=True,
            initial_capital=100000,
            max_positions=10,
            min_notional_pct=0.01,
            capital_utilization=0.9,
            max_allocation_per_trade=0.2,
            risk_mode="percentage",
            risk_fixed_amount=0,
            risk_percentage=1.0,
            mis_enabled=True,
            mis_fetcher=mock_fetcher
        )
        yield cm, mock_fetcher

    def test_mis_blocked_buy_rejected(self, capital_manager_with_mis_fetcher):
        """BUY on MIS-blocked stock should be rejected (no CNC fallback)."""
        cm, mock_fetcher = capital_manager_with_mis_fetcher
        mock_fetcher.is_mis_allowed.return_value = False

        allowed, reason = cm.is_mis_allowed("NSE:BLOCKEDSTOCK", "BUY")

        assert allowed is False
        assert "mis_blocked" in reason

    def test_mis_blocked_sell_rejected(self, capital_manager_with_mis_fetcher):
        """SELL on MIS-blocked stock should be rejected."""
        cm, mock_fetcher = capital_manager_with_mis_fetcher
        mock_fetcher.is_mis_allowed.return_value = False

        allowed, reason = cm.is_mis_allowed("NSE:BLOCKEDSTOCK", "SELL")

        assert allowed is False
        assert "mis_blocked" in reason

    def test_mis_allowed_stock_passes(self, capital_manager_with_mis_fetcher):
        """Stock in MIS list should be allowed."""
        cm, mock_fetcher = capital_manager_with_mis_fetcher
        mock_fetcher.is_mis_allowed.return_value = True

        allowed, reason = cm.is_mis_allowed("NSE:RELIANCE", "BUY")

        assert allowed is True
        assert reason == "ok"

    def test_can_enter_rejects_mis_blocked(self, capital_manager_with_mis_fetcher):
        """can_enter_position should reject MIS-blocked stocks."""
        cm, mock_fetcher = capital_manager_with_mis_fetcher
        mock_fetcher.is_mis_allowed.return_value = False

        can_enter, qty, reason = cm.can_enter_position(
            symbol="NSE:BLOCKEDSTOCK",
            qty=100,
            price=50.0,
            side="BUY"
        )

        assert can_enter is False
        assert qty == 0
        assert "mis_blocked" in reason


class TestExitReconciliation:
    """Tests for exit reconciliation with broker."""

    @pytest.fixture
    def mock_kite_broker_live(self):
        """Create a live KiteBroker instance with mocked KiteConnect."""
        with patch('broker.kite.kite_broker.KiteConnect') as mock_kc:
            from broker.kite.kite_broker import KiteBroker

            mock_instance = MagicMock()
            mock_kc.return_value = mock_instance

            broker = KiteBroker(
                api_key="test_key",
                access_token="test_token",
                dry_run=False
            )
            broker.kc = mock_instance
            yield broker, mock_instance

    @pytest.fixture
    def mock_kite_broker_paper(self):
        """Create a paper trading KiteBroker instance."""
        with patch('broker.kite.kite_broker.KiteConnect') as mock_kc:
            from broker.kite.kite_broker import KiteBroker

            broker = KiteBroker(
                api_key="test_key",
                access_token="test_token",
                dry_run=True
            )
            yield broker

    def test_reconcile_exit_full_position_closed(self, mock_kite_broker_live):
        """Full exit should show position closed (qty=0 at broker)."""
        broker, mock_kc = mock_kite_broker_live

        # Mock order as COMPLETE with fill price
        mock_kc.orders.return_value = [
            {
                "order_id": "EXIT123",
                "status": "COMPLETE",
                "average_price": 155.00,
                "filled_quantity": 100
            }
        ]

        # Position is now closed (not in positions list)
        mock_kc.positions.return_value = {
            "day": [],
            "net": []
        }

        result = broker.reconcile_exit(
            symbol="NSE:RELIANCE",
            order_id="EXIT123",
            expected_qty=100,
            position_qty_before=100,
            timeout=0.01
        )

        assert result is not None
        assert result["avg_price"] == 155.00
        assert result["qty_exited"] == 100
        assert result["remaining_qty"] == 0
        assert result["verified"] is True

    def test_reconcile_exit_partial_position_reduced(self, mock_kite_broker_live):
        """Partial exit should show reduced position at broker."""
        broker, mock_kc = mock_kite_broker_live

        mock_kc.orders.return_value = [
            {
                "order_id": "EXIT456",
                "status": "COMPLETE",
                "average_price": 160.00,
                "filled_quantity": 50
            }
        ]

        # Position reduced from 100 to 50
        mock_kc.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "INFY",
                    "exchange": "NSE",
                    "quantity": 50,  # Remaining long position
                    "average_price": 150.00,
                    "product": "MIS",
                    "pnl": 500.00
                }
            ],
            "net": []
        }

        result = broker.reconcile_exit(
            symbol="NSE:INFY",
            order_id="EXIT456",
            expected_qty=50,
            position_qty_before=100,
            timeout=0.01
        )

        assert result is not None
        assert result["avg_price"] == 160.00
        assert result["qty_exited"] == 50
        assert result["remaining_qty"] == 50
        assert result["verified"] is True

    def test_reconcile_exit_order_not_filled(self, mock_kite_broker_live):
        """Exit order that didn't fill should return None."""
        broker, mock_kc = mock_kite_broker_live

        # Order rejected or pending
        mock_kc.orders.return_value = [
            {
                "order_id": "EXIT789",
                "status": "REJECTED",
                "average_price": 0,
                "filled_quantity": 0
            }
        ]

        result = broker.reconcile_exit(
            symbol="NSE:RELIANCE",
            order_id="EXIT789",
            expected_qty=100,
            position_qty_before=100,
            timeout=0.01
        )

        assert result is None

    def test_reconcile_exit_short_position(self, mock_kite_broker_live):
        """Closing a short position (negative qty becomes less negative)."""
        broker, mock_kc = mock_kite_broker_live

        mock_kc.orders.return_value = [
            {
                "order_id": "EXITSHORT",
                "status": "COMPLETE",
                "average_price": 195.00,
                "filled_quantity": 30
            }
        ]

        # Short position reduced: was -50, now -20 (exited 30)
        mock_kc.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "HDFCBANK",
                    "exchange": "NSE",
                    "quantity": -20,  # Negative = short position remaining
                    "average_price": 200.00,
                    "product": "MIS",
                    "pnl": 150.00
                }
            ],
            "net": []
        }

        result = broker.reconcile_exit(
            symbol="NSE:HDFCBANK",
            order_id="EXITSHORT",
            expected_qty=30,
            position_qty_before=50,  # We track absolute qty internally
            timeout=0.01
        )

        assert result is not None
        assert result["avg_price"] == 195.00
        assert result["qty_exited"] == 30
        assert result["remaining_qty"] == 20  # abs(-20) = 20
        assert result["verified"] is True

    def test_reconcile_exit_eod_squareoff_all_closed(self, mock_kite_broker_live):
        """EOD square-off should verify all positions closed."""
        broker, mock_kc = mock_kite_broker_live

        mock_kc.orders.return_value = [
            {
                "order_id": "EOD001",
                "status": "COMPLETE",
                "average_price": 1520.00,
                "filled_quantity": 25
            }
        ]

        # All positions closed at EOD
        mock_kc.positions.return_value = {
            "day": [],
            "net": []
        }

        result = broker.reconcile_exit(
            symbol="NSE:TCS",
            order_id="EOD001",
            expected_qty=25,
            position_qty_before=25,
            timeout=0.01
        )

        assert result is not None
        assert result["verified"] is True
        assert result["remaining_qty"] == 0

    def test_reconcile_exit_partial_fill_detected(self, mock_kite_broker_live):
        """Detect when exit only partially filled."""
        broker, mock_kc = mock_kite_broker_live

        # Order filled but only partially (market conditions)
        mock_kc.orders.return_value = [
            {
                "order_id": "PARTIAL001",
                "status": "COMPLETE",
                "average_price": 300.00,
                "filled_quantity": 50  # Only 50 of 100 filled
            }
        ]

        # Position only reduced by 50, not 100
        mock_kc.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "WIPRO",
                    "exchange": "NSE",
                    "quantity": 50,  # Was 100, expected 0, got 50
                    "average_price": 290.00,
                    "product": "MIS",
                    "pnl": 500.00
                }
            ],
            "net": []
        }

        result = broker.reconcile_exit(
            symbol="NSE:WIPRO",
            order_id="PARTIAL001",
            expected_qty=100,  # Tried to exit 100
            position_qty_before=100,
            timeout=0.01
        )

        assert result is not None
        assert result["avg_price"] == 300.00
        # Actual exited = 100 - 50 = 50
        assert result["qty_exited"] == 50
        assert result["remaining_qty"] == 50
        # Verified is True because some exit happened
        assert result["verified"] is True

    def test_reconcile_exit_paper_trading(self, mock_kite_broker_paper):
        """Paper trading exit reconciliation uses internal state."""
        broker = mock_kite_broker_paper

        broker._paper_orders = [
            {
                "order_id": "PAPER_EXIT001",
                "status": "COMPLETE",
                "quantity": 75,
                "average_price": 125.50,
                "product": "MIS"
            }
        ]

        result = broker.reconcile_exit(
            symbol="NSE:TESTSTOCK",
            order_id="PAPER_EXIT001",
            expected_qty=75,
            position_qty_before=100,
            timeout=0
        )

        assert result is not None
        assert result["avg_price"] == 125.50
        assert result["qty_exited"] == 75
        assert result["remaining_qty"] == 25  # 100 - 75
        assert result["verified"] is True

    def test_reconcile_exit_paper_order_not_found(self, mock_kite_broker_paper):
        """Paper trading with non-existent exit order returns None."""
        broker = mock_kite_broker_paper
        broker._paper_orders = []

        result = broker.reconcile_exit(
            symbol="NSE:TESTSTOCK",
            order_id="PAPER_NONEXISTENT",
            expected_qty=50,
            position_qty_before=50,
            timeout=0
        )

        assert result is None

    def test_reconcile_exit_ignores_a_cnc_line_in_the_same_stock(self, mock_kite_broker_live):
        """reconcile_exit verifies an intraday MIS exit. A CNC/MTF holding in the same
        stock (the overnight book shares the account) is not that position and must
        not be read as qty still open (2026-10-29)."""
        broker, mock_kc = mock_kite_broker_live

        mock_kc.orders.return_value = [
            {
                "order_id": "NET001",
                "status": "COMPLETE",
                "average_price": 450.00,
                "filled_quantity": 20
            }
        ]

        # Position in net, not day (e.g., carried forward position)
        mock_kc.positions.return_value = {
            "day": [],
            "net": [
                {
                    "tradingsymbol": "SBIN",
                    "exchange": "NSE",
                    "quantity": 30,
                    "average_price": 440.00,
                    "product": "CNC",
                    "pnl": 300.00
                }
            ]
        }

        result = broker.reconcile_exit(
            symbol="NSE:SBIN",
            order_id="NET001",
            expected_qty=20,
            position_qty_before=50,
            timeout=0.01
        )

        assert result is not None
        assert result["remaining_qty"] == 0
        assert result["verified"] is True

    def test_reconcile_exit_flags_a_flipped_position(self, mock_kite_broker_live):
        """SAKSOFT 2026-01-28: two BUYs closed a short and opened a long."""
        broker, mock_kc = mock_kite_broker_live
        mock_kc.orders.return_value = [{"order_id": "F1", "status": "COMPLETE",
                                        "average_price": 100.0, "filled_quantity": 287}]
        mock_kc.positions.return_value = {"day": [{"tradingsymbol": "SAKSOFT", "quantity": 287,
                                                   "product": "MIS", "average_price": 100.0}], "net": []}
        r = broker.reconcile_exit(symbol="NSE:SAKSOFT", order_id="F1", expected_qty=287,
                                  position_qty_before=287, timeout=0.01, position_side="SELL")
        assert r["flipped"] is True and r["verified"] is False

    def test_reconcile_exit_positions_failure_is_not_flat(self, mock_kite_broker_live):
        broker, mock_kc = mock_kite_broker_live
        mock_kc.orders.return_value = [{"order_id": "F2", "status": "COMPLETE",
                                        "average_price": 100.0, "filled_quantity": 10}]
        mock_kc.positions.side_effect = RuntimeError("api down")
        r = broker.reconcile_exit(symbol="NSE:X", order_id="F2", expected_qty=10,
                                  position_qty_before=10, timeout=0.01, position_side="SELL")
        assert r is not None and r.get("verified") is not True

    def test_reconcile_exit_day_position_zero_qty(self, mock_kite_broker_live):
        """Position exists in day with qty=0 should use day value, not check net."""
        broker, mock_kc = mock_kite_broker_live

        mock_kc.orders.return_value = [
            {
                "order_id": "DAYZERO",
                "status": "COMPLETE",
                "average_price": 500.00,
                "filled_quantity": 100
            }
        ]

        # Position in day with 0 qty (fully closed), also has stale net entry
        mock_kc.positions.return_value = {
            "day": [
                {
                    "tradingsymbol": "ICICIBANK",
                    "exchange": "NSE",
                    "quantity": 0,  # Fully exited in day
                    "average_price": 0,
                    "product": "MIS",
                    "pnl": 1000.00
                }
            ],
            "net": [
                {
                    "tradingsymbol": "ICICIBANK",
                    "exchange": "NSE",
                    "quantity": 50,  # Stale net position (should not use this)
                    "average_price": 480.00,
                    "product": "CNC",
                    "pnl": 0
                }
            ]
        }

        result = broker.reconcile_exit(
            symbol="NSE:ICICIBANK",
            order_id="DAYZERO",
            expected_qty=100,
            position_qty_before=100,
            timeout=0.01
        )

        assert result is not None
        assert result["remaining_qty"] == 0  # Use day qty (0), not net qty (50)
        assert result["verified"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
