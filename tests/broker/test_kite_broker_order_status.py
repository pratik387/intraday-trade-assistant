import os, sys
from unittest.mock import patch
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
        from broker.kite.kite_broker import KiteBroker
        b = KiteBroker(api_key="k", access_token="t", dry_run=False)
        b.kc = inst
        yield b, inst


def test_get_order_status_returns_status_and_price(live_broker):
    b, inst = live_broker
    inst.orders.return_value = [
        {"order_id": "111", "status": "COMPLETE", "average_price": 145.8},
        {"order_id": "222", "status": "OPEN", "average_price": 0},
    ]
    out = b.get_order_status("111")
    assert out["status"] == "COMPLETE"
    assert out["average_price"] == 145.8


def test_get_order_status_unknown_order(live_broker):
    b, inst = live_broker
    inst.orders.return_value = []
    out = b.get_order_status("999")
    assert out["status"] == "UNKNOWN"
    assert out["average_price"] == 0.0


@pytest.fixture
def dry_broker():
    with patch("broker.kite.kite_broker.KiteConnect"):
        from broker.kite.kite_broker import KiteBroker
        b = KiteBroker(api_key="k", access_token="t", dry_run=True)
        yield b


def test_get_order_status_dry_run_found(dry_broker):
    dry_broker._paper_orders.append({"order_id": "P1", "status": "COMPLETE", "average_price": 99.5})
    out = dry_broker.get_order_status("P1")
    assert out["status"] == "COMPLETE"
    assert out["average_price"] == 99.5


def test_get_order_status_dry_run_unknown(dry_broker):
    out = dry_broker.get_order_status("NOPE")
    assert out["status"] == "UNKNOWN"
    assert out["average_price"] == 0.0
