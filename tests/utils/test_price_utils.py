"""Tests for utils.price_utils — NSE tick rounding + circuit-aware LIMIT pricing.

Regression cover for the 2026-06-29 live overnight failures:
  - NSE:DOLLAR rejected: "Tick size for this script is 0.05" (price not a tick multiple)
  - NSE:TIRUPATIFL rejected: order price > upper circuit limit 58.03
"""
import math

from utils.price_utils import round_to_tick, clamp_round_limit


def test_round_to_tick_basic():
    assert round_to_tick(287.43) == 287.45      # nearest 0.05
    assert round_to_tick(133.0) == 133.0        # already valid
    assert round_to_tick(0.0) == 0.0


def test_buy_floors_to_tick_below_upper_circuit():
    # TIRUPATIFL case: ref*(1+buf) overshoots a NON-tick circuit cap 58.03.
    # BUY must end up tick-valid AND <= 58.03 -> 58.00 (floor), never 58.05.
    px = clamp_round_limit(60.0, "BUY", upper_circuit=58.03)
    assert px == 58.00
    assert px <= 58.03
    assert abs((px / 0.05) - round(px / 0.05)) < 1e-9  # on a 0.05 tick


def test_buy_normal_floors_to_tick_no_circuit():
    # DOLLAR case: a non-tick desired price with no binding circuit -> tick-valid.
    px = clamp_round_limit(287.43, "BUY", upper_circuit=None)
    assert px == 287.40          # floor to tick (stays marketable under the 1% buffer)
    assert abs((px / 0.05) - round(px / 0.05)) < 1e-9


def test_sell_ceils_to_tick_above_lower_circuit():
    # A SELL must end up tick-valid AND >= the (non-tick) lower circuit.
    px = clamp_round_limit(10.0, "SELL", lower_circuit=12.07)
    assert px == 12.10           # ceil to tick, never below 12.07
    assert px >= 12.07


def test_sell_normal_ceils_to_tick():
    px = clamp_round_limit(133.02, "SELL", lower_circuit=None)
    assert px == 133.05


def test_zero_or_negative_passthrough():
    assert clamp_round_limit(0.0, "BUY") == 0.0
    assert clamp_round_limit(-5.0, "SELL") == -5.0


def test_already_tick_valid_unchanged():
    assert clamp_round_limit(133.00, "SELL", lower_circuit=100.0) == 133.00
    assert clamp_round_limit(133.00, "BUY", upper_circuit=200.0) == 133.00
