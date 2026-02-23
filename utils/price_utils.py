# @role: Price utilities — NSE tick rounding, price validation
# @used_by: trigger_aware_executor.py, exit_executor.py
# @filter_type: utility
# @tags: utility, pricing, NSE


def round_to_tick(price: float, tick_size: float = 0.05) -> float:
    """Round price to the nearest NSE tick size.

    NSE equity cash segment:
    - Scrips >= Re.1: tick = Re.0.05
    - Scrips < Re.1: tick = Re.0.01

    Default tick_size=0.05 covers 99.9%+ of intraday-tradeable stocks.
    """
    if price <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 2)
