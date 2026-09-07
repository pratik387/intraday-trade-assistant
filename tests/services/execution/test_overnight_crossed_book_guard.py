"""A crossed top-of-book is a bad read, not a market, and must be discarded.

`_safe_top_of_book` exists to measure the spread on every fire so that the entry
LIMIT can eventually be priced off the ask instead of a stale LTP. Its own
docstring defers that change until the measured distribution exists.

That makes quote hygiene load-bearing: the distribution IS the calibration
input. Two of the 43 ENTRY_BASIS lines captured so far are crossed —

    2026-09-03  NSE:RBLBANK  bid 421.35 / ask 396.85   spread -581.5bp
    2026-09-04  NSE:SWIGGY   bid 287.75 / ask 271.05   spread -580.4bp

— both liquid names, both within a basis point of the same impossible spread,
which reads as the wrong depth level rather than anything the market did. A
crossed book cannot persist on an exchange; it would have traded.

Left in, those two would drag the measured mean spread by ~27bp on n=43 and
would later seed the ask-based pricing with a number below the bid.
"""
import inspect

import services.execution.overnight_handlers as oh


class _Broker:
    """Minimal broker exposing get_quote with a settable depth."""

    def __init__(self, buys, sells):
        self._buys, self._sells = buys, sells

    def get_quote(self, symbol):
        return {"depth": {"buy": self._buys, "sell": self._sells}}


def _lvl(price, qty):
    return [{"price": price, "quantity": qty}]


def test_normal_book_passes_through():
    b = _Broker(_lvl(100.0, 500), _lvl(100.5, 400))
    assert oh._safe_top_of_book(b, "NSE:X") == (100.0, 100.5, 500, 400)


def test_crossed_book_is_discarded():
    """The RBLBANK shape: bid above ask."""
    b = _Broker(_lvl(421.35, 100), _lvl(396.85, 100))
    assert oh._safe_top_of_book(b, "NSE:RBLBANK") == (None, None, None, None)


def test_locked_book_is_discarded():
    """bid == ask is equally impossible — it would have traded."""
    b = _Broker(_lvl(250.0, 10), _lvl(250.0, 10))
    assert oh._safe_top_of_book(b, "NSE:Y") == (None, None, None, None)


def test_one_tick_spread_still_passes():
    """The guard must not eat legitimately tight books."""
    b = _Broker(_lvl(250.00, 10), _lvl(250.05, 10))
    bid, ask, _, _ = oh._safe_top_of_book(b, "NSE:Z")
    assert (bid, ask) == (250.00, 250.05)


def test_partial_book_unaffected():
    """Only one side quoted: nothing to compare, pass through as before."""
    b = _Broker(_lvl(100.0, 5), [])
    assert oh._safe_top_of_book(b, "NSE:W") == (100.0, None, 5, None)
    b2 = _Broker([], _lvl(100.0, 5))
    assert oh._safe_top_of_book(b2, "NSE:W") == (None, 100.0, None, 5)


def test_crossed_book_is_logged_loudly(caplog):
    b = _Broker(_lvl(287.75, 8431018), _lvl(271.05, 14456014))
    with caplog.at_level("WARNING"):
        oh._safe_top_of_book(b, "NSE:SWIGGY")
    assert any("TOB_CROSSED" in r.message for r in caplog.records), (
        "a crossed book must be visible, not silently swallowed")


def test_guard_cannot_block_order_placement():
    """This helper is best-effort by contract — it must never raise."""
    class Boom:
        def get_quote(self, symbol):
            raise RuntimeError("feed down")

    assert oh._safe_top_of_book(Boom(), "NSE:X") == (None, None, None, None)


def test_helper_is_still_observability_only():
    """Guard against this quietly becoming a pricing input without calibration.

    The docstring defers ask-based pricing until the spread distribution is
    measured. If that changes, this test should be updated deliberately.
    """
    src = inspect.getsource(oh._safe_top_of_book)
    assert "OBSERVABILITY ONLY" in src
