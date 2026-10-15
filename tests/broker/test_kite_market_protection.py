"""KiteBroker sends market_protection on every live MARKET order.

Zerodha rejected the first live intraday entry (2026-10-15, NSE:BAJAJELEC):
'Market orders without market protection are not allowed via API'.
"""
from unittest.mock import MagicMock, patch

import pytest

from broker.kite import kite_broker as kb


def _broker(mp=-1, dry_run=False):
    kc = MagicMock()
    kc.ORDER_TYPE_LIMIT, kc.ORDER_TYPE_MARKET = "LIMIT", "MARKET"
    kc.TRANSACTION_TYPE_BUY, kc.TRANSACTION_TYPE_SELL = "BUY", "SELL"
    kc.PRODUCT_MIS, kc.PRODUCT_CNC = "MIS", "CNC"
    kc.VARIETY_REGULAR, kc.VARIETY_AMO = "regular", "amo"
    kc._post.return_value = {"order_id": "123"}
    with patch.object(kb, "KiteConnect", return_value=kc), \
         patch.object(kb.env, "KITE_API_KEY", "k"), \
         patch.object(kb.env, "KITE_ACCESS_TOKEN", "t"), \
         patch("config.filters_setup.load_filters", return_value={"kite_orders": {"market_protection": mp}}):
        return kb.KiteBroker(dry_run=dry_run), kc


def test_market_order_carries_market_protection():
    b, kc = _broker(mp=-1)
    oid = b.place_order(symbol="NSE:BAJAJELEC", side="SELL", qty=181, check_margins=False)
    assert oid == "123"
    route, = kc._post.call_args.args
    kw = kc._post.call_args.kwargs
    assert route == "order.place"
    assert kw["url_args"] == {"variety": "regular"}
    assert kw["params"]["order_type"] == "MARKET"
    assert kw["params"]["market_protection"] == -1
    assert "price" not in kw["params"]


def test_custom_percent_is_passed_through():
    b, kc = _broker(mp=2)
    b.place_order(symbol="NSE:X", side="BUY", qty=1, check_margins=False)
    assert kc._post.call_args.kwargs["params"]["market_protection"] == 2


def test_limit_order_has_no_market_protection():
    b, kc = _broker()
    b.place_order(symbol="NSE:X", side="SELL", qty=5, order_type="LIMIT", price=100.0,
                  product="CNC", variety="amo", check_margins=False)
    p = kc._post.call_args.kwargs["params"]
    assert p["order_type"] == "LIMIT" and p["price"] == 100.0
    assert "market_protection" not in p
    assert kc._post.call_args.kwargs["url_args"] == {"variety": "amo"}


def test_paper_mode_never_calls_kite():
    b, kc = _broker(dry_run=True)
    b.place_order(symbol="NSE:X", side="SELL", qty=5, check_margins=False)
    kc._post.assert_not_called()


@pytest.mark.parametrize("bad", [0, -2, 101, "auto", None, True])
def test_invalid_config_fails_fast(bad):
    with pytest.raises(RuntimeError, match="market_protection"):
        _broker(mp=bad)


def test_missing_config_key_fails_fast():
    with patch.object(kb, "KiteConnect", return_value=MagicMock()), \
         patch.object(kb.env, "KITE_API_KEY", "k"), \
         patch.object(kb.env, "KITE_ACCESS_TOKEN", "t"), \
         patch("config.filters_setup.load_filters", return_value={}):
        with pytest.raises(KeyError):
            kb.KiteBroker(dry_run=False)


def test_shipped_config_value_is_valid():
    from config.filters_setup import load_filters
    mp = load_filters()["kite_orders"]["market_protection"]
    assert mp == -1 or 0 < mp <= 100
