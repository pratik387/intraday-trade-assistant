# @role: Price utilities — NSE tick rounding, price validation
# @used_by: trigger_aware_executor.py, exit_executor.py, overnight_handlers.py
# @filter_type: utility
# @tags: utility, pricing, NSE

import math


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round price to the nearest NSE tick size.

    NSE equity cash segment:
    - Scrips >= Re.1: tick = Re.0.05
    - Scrips < Re.1: tick = Re.0.01

    Default tick_size=0.05 covers 99.9%+ of intraday-tradeable stocks
    (0.05 is itself a multiple of 0.01, so a 0.05-rounded price is always a
    valid tick for sub-Re.1 scrips too — never rejected for tick size).
    """
    if price <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 2)


def clamp_round_limit(
    price: float,
    side: str,
    *,
    tick_size: float = 0.05,
    upper_circuit: float = None,
    lower_circuit: float = None,
) -> float:
    """Return a Kite-acceptable LIMIT price: tick-valid AND inside the circuit band.

    Kite rejects a LIMIT order whose price is not a multiple of the instrument
    tick (e.g. "Tick size for this script is 0.05") or whose price violates the
    circuit band (e.g. "order price is higher than the upper circuit limit").
    Circuit limits are NOT themselves tick-aligned (they are prev_close x band%),
    so rounding must be DIRECTIONAL to stay inside the band:

      BUY  -> clamp to <= upper_circuit, then FLOOR to tick (never exceeds the cap)
      SELL -> clamp to >= lower_circuit, then CEIL  to tick (never undercuts the floor)

    Circuit bounds are optional; pass None (e.g. paper mode or quote unavailable)
    to skip the clamp and tick-round in the marketable direction only.
    """
    if price <= 0:
        return price
    if str(side).upper() == "BUY":
        if upper_circuit and upper_circuit > 0:
            price = min(price, float(upper_circuit))
        p = math.floor(price / tick_size) * tick_size
    else:  # SELL
        if lower_circuit and lower_circuit > 0:
            price = max(price, float(lower_circuit))
        p = math.ceil(price / tick_size) * tick_size
    return round(p, 2)
