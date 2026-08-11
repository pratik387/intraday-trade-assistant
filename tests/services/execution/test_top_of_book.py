"""Tests for `_safe_top_of_book` (overnight entry observability).

Added 2026-08-11. The helper reads 5-level depth out of the SAME Kite quote()
response the entry path already fetches (it was previously discarded), so the
bid/ask spread is visible on every fire. Nothing prices off it yet — but it runs
inside the LIVE entry loop, so its failure modes must never propagate: a broker
without depth, a malformed payload, or a raising quote call all have to degrade
to None rather than block an order.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.execution.overnight_handlers import _safe_top_of_book  # noqa: E402


class _Broker:
    def __init__(self, payload=None, raises=False):
        self._payload = payload
        self._raises = raises

    def get_quote(self, symbol):
        if self._raises:
            raise RuntimeError("quote blew up")
        return self._payload


def _depth(bid, ask, bq=100, aq=200):
    return {
        "last_price": 100.0,
        "depth": {
            "buy": [{"price": bid, "quantity": bq, "orders": 1}],
            "sell": [{"price": ask, "quantity": aq, "orders": 1}],
        },
    }


def test_reads_top_of_book():
    bid, ask, bq, aq = _safe_top_of_book(_Broker(_depth(99.5, 100.5)), "NSE:X")
    assert (bid, ask, bq, aq) == (99.5, 100.5, 100, 200)


def test_spread_is_computable_from_the_result():
    """The whole point: spread in bp must be derivable at order time."""
    bid, ask, _, _ = _safe_top_of_book(_Broker(_depth(100.0, 101.0)), "NSE:X")
    assert round(1e4 * (ask / bid - 1.0)) == 100  # 1% == 100bp


def test_missing_depth_key_returns_nones():
    """Older/partial payloads must not raise inside the live entry loop."""
    assert _safe_top_of_book(_Broker({"last_price": 100.0}), "NSE:X") == (None, None, None, None)


def test_empty_depth_sides_return_nones():
    payload = {"depth": {"buy": [], "sell": []}}
    assert _safe_top_of_book(_Broker(payload), "NSE:X") == (None, None, None, None)


def test_zero_prices_treated_as_absent():
    """Kite fills missing fields with 0.0; 0 is not a tradeable price."""
    bid, ask, _, _ = _safe_top_of_book(_Broker(_depth(0.0, 0.0)), "NSE:X")
    assert bid is None and ask is None


def test_raising_quote_never_propagates():
    assert _safe_top_of_book(_Broker(raises=True), "NSE:X") == (None, None, None, None)


def test_broker_without_get_quote_returns_nones():
    class _Bare:
        pass
    assert _safe_top_of_book(_Bare(), "NSE:X") == (None, None, None, None)


def test_none_payload_returns_nones():
    assert _safe_top_of_book(_Broker(None), "NSE:X") == (None, None, None, None)


@pytest.mark.parametrize("bad", [{"depth": None}, {"depth": {"buy": None, "sell": None}}])
def test_malformed_depth_shapes_return_nones(bad):
    assert _safe_top_of_book(_Broker(bad), "NSE:X") == (None, None, None, None)
