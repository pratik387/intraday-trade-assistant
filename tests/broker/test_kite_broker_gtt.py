import os, sys
from unittest.mock import patch, MagicMock
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def mock_env():
    with patch('broker.kite.kite_broker.env') as m:
        m.KITE_API_KEY = "k"; m.KITE_ACCESS_TOKEN = "t"
        yield m


@pytest.fixture
def live_broker():
    with patch("broker.kite.kite_broker.KiteConnect") as MockKC:
        inst = MockKC.return_value
        inst.GTT_TYPE_SINGLE = "single"
        inst.TRANSACTION_TYPE_SELL = "SELL"
        inst.ORDER_TYPE_LIMIT = "LIMIT"
        inst.PRODUCT_MTF = "MTF"
        inst.PRODUCT_CNC = "CNC"
        from broker.kite.kite_broker import KiteBroker
        b = KiteBroker(api_key="k", access_token="t", dry_run=False)
        b.kc = inst
        b.get_ltp = MagicMock(return_value=145.0)
        yield b, inst


def test_place_gtt_stop_calls_kc_with_single_leg(live_broker):
    b, inst = live_broker
    inst.place_gtt.return_value = {"trigger_id": 777}
    gid = b.place_gtt_stop(symbol="NSE:RELIANCE", qty=68, trigger_price=138.5, limit_price=137.0, product="MTF")
    assert gid == "777"
    kwargs = inst.place_gtt.call_args.kwargs
    assert kwargs["trigger_type"] == "single"
    assert kwargs["trigger_values"] == [138.5]
    assert kwargs["orders"][0]["transaction_type"] == "SELL"
    assert kwargs["orders"][0]["product"] == "MTF"
    assert kwargs["orders"][0]["price"] == 137.0


def test_cancel_gtt_calls_delete(live_broker):
    b, inst = live_broker
    assert b.cancel_gtt("777") is True
    inst.delete_gtt.assert_called_once_with(777)


def test_cancel_gtt_swallows_error(live_broker):
    b, inst = live_broker
    inst.delete_gtt.side_effect = RuntimeError("already gone")
    assert b.cancel_gtt("777") is False
