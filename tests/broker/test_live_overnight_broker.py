import os, sys
from unittest.mock import MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _make():
    from broker.live_overnight_broker import LiveOvernightBroker
    data_sdk = MagicMock()
    data_sdk._equity_instruments = ["NSE:RELIANCE", "NSE:TCS"]
    kite = MagicMock()
    return LiveOvernightBroker(data_sdk=data_sdk, kite=kite), data_sdk, kite


def test_data_methods_route_to_data_sdk():
    b, data_sdk, kite = _make()
    b.get_intraday_5m("NSE:RELIANCE")
    data_sdk.get_intraday_5m.assert_called_once_with("NSE:RELIANCE")
    b.get_daily("NSE:RELIANCE", days=30)
    data_sdk.get_daily.assert_called_once_with("NSE:RELIANCE", days=30)
    assert b.list_symbols() == ["NSE:RELIANCE", "NSE:TCS"]
    assert b._data_sdk is data_sdk
    kite.get_intraday_5m.assert_not_called()


def test_order_methods_route_to_kite():
    b, data_sdk, kite = _make()
    kite.place_order.return_value = "ORD1"
    oid = b.place_order(symbol="NSE:RELIANCE", side="BUY", qty=10, product="MTF", variety="regular")
    assert oid == "ORD1"
    kite.place_order.assert_called_once()
    b.get_order_status("ORD1"); kite.get_order_status.assert_called_once_with("ORD1")
    b.place_gtt_stop(symbol="NSE:RELIANCE", qty=10, trigger_price=1.0, limit_price=0.9, product="MTF")
    kite.place_gtt_stop.assert_called_once()
    b.cancel_gtt("G1"); kite.cancel_gtt.assert_called_once_with("G1")
    data_sdk.place_order.assert_not_called()


def test_has_no_dry_session_date_attr():
    # _build_market_context treats presence of _dry_session_date as backtest mode.
    b, _, _ = _make()
    assert getattr(b, "_dry_session_date", None) is None
