"""Shared-account funds gate for the intraday book."""
import pytest

from services.execution.intraday_live_guards import (
    IntradayLiveGuards,
    IntradayLiveGuardsConfigError,
    funds_verdict,
    validate_cfg,
)


def _cfg(reserve=100000, on=True, ttl=20):
    return {"funds_gate": {"enabled": on, "overnight_reserve_inr": reserve, "balance_cache_secs": ttl}}


class FakeBroker:
    def __init__(self, *, dry_run=False, balance=200000.0, margin=15000.0, balance_exc=None, margin_exc=None):
        self.dry_run = dry_run
        self.balance, self.margin = balance, margin
        self.balance_exc, self.margin_exc = balance_exc, margin_exc
        self.balance_calls = self.margin_calls = 0

    def get_free_balance(self):
        self.balance_calls += 1
        if self.balance_exc:
            raise self.balance_exc
        return self.balance

    def get_order_margin(self, symbol, side, qty, product="MIS"):
        self.margin_calls += 1
        if self.margin_exc:
            raise self.margin_exc
        return self.margin


@pytest.mark.parametrize("mutate", [
    lambda c: c.pop("funds_gate"),
    lambda c: c["funds_gate"].pop("overnight_reserve_inr"),
    lambda c: c["funds_gate"].pop("balance_cache_secs"),
    lambda c: c["funds_gate"].pop("enabled"),
    lambda c: c["funds_gate"].__setitem__("overnight_reserve_inr", -1),
    lambda c: c["funds_gate"].__setitem__("balance_cache_secs", -1),
])
def test_missing_or_invalid_config_fails_fast(mutate):
    c = _cfg()
    mutate(c)
    with pytest.raises(IntradayLiveGuardsConfigError):
        validate_cfg(c)


def test_missing_block_fails_fast_at_construction():
    with pytest.raises(IntradayLiveGuardsConfigError):
        IntradayLiveGuards(None, FakeBroker())


def test_funds_verdict_boundary():
    assert funds_verdict(115000, 15000, 100000)[0] is True     # exactly the reserve left
    assert funds_verdict(114999, 15000, 100000)[0] is False
    assert funds_verdict(-16000, 15000, 100000)[0] is False    # negative balance (seen 2026-10-13)


def test_live_passes_when_reserve_is_left():
    ok, reason = IntradayLiveGuards(_cfg(), FakeBroker(balance=200000, margin=15000)).check_entry("NSE:X", "SELL", 100, 100.0)
    assert (ok, reason) == (True, "ok")


def test_live_blocks_when_reserve_would_be_breached():
    ok, reason = IntradayLiveGuards(_cfg(reserve=100000), FakeBroker(balance=110000, margin=15000)).check_entry("NSE:X", "SELL", 100, 100.0)
    assert ok is False and reason.startswith("funds_gate:")


@pytest.mark.parametrize("kw,prefix", [
    (dict(balance_exc=RuntimeError("api down")), "funds_gate:funds_unavailable"),
    (dict(margin_exc=RuntimeError("api down")), "funds_gate:funds_unavailable"),
    (dict(margin=0.0), "funds_gate:margin_unavailable"),
])
def test_live_fails_closed(kw, prefix):
    ok, reason = IntradayLiveGuards(_cfg(), FakeBroker(**kw)).check_entry("NSE:X", "SELL", 100, 100.0)
    assert ok is False and reason.startswith(prefix)


def test_balance_is_cached_then_dropped_after_an_entry():
    b = FakeBroker()
    g = IntradayLiveGuards(_cfg(ttl=3600), b)
    g.check_entry("NSE:X", "SELL", 100, 100.0)
    g.check_entry("NSE:Y", "SELL", 100, 100.0)
    assert b.balance_calls == 1
    g.note_entry()
    g.check_entry("NSE:Z", "SELL", 100, 100.0)
    assert b.balance_calls == 2


def test_failed_read_is_not_cached():
    b = FakeBroker(margin_exc=RuntimeError("x"))
    g = IntradayLiveGuards(_cfg(ttl=3600), b)
    g.check_entry("NSE:X", "SELL", 100, 100.0)
    b.margin_exc = None
    assert g.check_entry("NSE:X", "SELL", 100, 100.0)[0] is True
    assert b.balance_calls == 2


def test_disabled_gate_passes_without_broker_calls():
    b = FakeBroker(balance=0.0)
    ok, _ = IntradayLiveGuards(_cfg(on=False), b).check_entry("NSE:X", "SELL", 100, 100.0)
    assert ok is True and b.balance_calls == 0 and b.margin_calls == 0


def test_paper_never_blocks_and_makes_no_broker_call():
    b = FakeBroker(dry_run=True, balance=-50000)
    g = IntradayLiveGuards(_cfg(), b)
    assert g.live is False
    assert g.check_entry("NSE:X", "SELL", 100, 100.0) == (True, "paper_not_evaluated")
    assert b.balance_calls == 0 and b.margin_calls == 0


def test_broker_without_dry_run_attribute_is_treated_as_paper():
    g = IntradayLiveGuards(_cfg(), object())
    assert g.live is False and g.check_entry("NSE:X", "SELL", 1, 100.0)[0] is True
