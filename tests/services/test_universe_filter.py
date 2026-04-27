"""universe_filter tests (sub8-T1)."""
import pandas as pd
import pytest

from services.universe_filter import (
    in_nifty50,
    in_banknifty,
    in_fno_liquid_200,
    in_universe,
)


def test_nifty50_member_returns_true():
    # RELIANCE is always in Nifty 50
    assert in_nifty50("NSE:RELIANCE") is True


def test_nifty50_non_member_returns_false():
    # SWSOLAR is small-cap — never in Nifty 50
    assert in_nifty50("NSE:SWSOLAR") is False


def test_banknifty_member_returns_true():
    # HDFCBANK is always in Bank Nifty
    assert in_banknifty("NSE:HDFCBANK") is True


def test_banknifty_non_member_returns_false():
    assert in_banknifty("NSE:RELIANCE") is False  # in Nifty50, not BankNifty


def test_fno_liquid_200_includes_index_majors():
    assert in_fno_liquid_200("NSE:RELIANCE") is True
    assert in_fno_liquid_200("NSE:HDFCBANK") is True


def test_in_universe_dispatches_by_key():
    assert in_universe("NSE:RELIANCE", "nifty50") is True
    assert in_universe("NSE:RELIANCE", "banknifty") is False
    assert in_universe("NSE:RELIANCE", "nifty50_banknifty") is True
    assert in_universe("NSE:HDFCBANK", "nifty50_banknifty") is True
    assert in_universe("NSE:RELIANCE", "fno_liquid_200") is True


def test_in_universe_unknown_key_raises():
    with pytest.raises(KeyError):
        in_universe("NSE:RELIANCE", "nonexistent_universe")
