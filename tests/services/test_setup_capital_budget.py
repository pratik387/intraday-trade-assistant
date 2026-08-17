"""Per-setup capital budgets must BLOCK, not merely be accounted.

`capital_budget_pct` is research-derived. earnings_downshock_continuation_short's
brief fixes its footprint at "Rs 1L notional ... 5 slots = Rs 100k = 20%" and
states that max_concurrent_positions: 5 / capital_budget_pct: 20 "truncates
nothing" — the measured expectancy holds only inside that footprint.

screener_live wired the budgets in ("so it can block setups that monopolize
total capital") and CapitalManager tracked usage on open and release, but no
code ever compared usage against the budget. Measured 2026-08-17: four
concurrent earnings_downshock positions held Rs100k margin each = 80% of Rs5L
against a 20% budget.
"""
import pytest

from services.capital_manager import CapitalManager

SETUP = "earnings_downshock_continuation_short"


def _cm(budget_pct=20.0, capital=500_000.0):
    cm = CapitalManager(
        enabled=True, initial_capital=capital, max_positions=50,
        min_notional_pct=0.02, capital_utilization=1.0,
        max_allocation_per_trade=1.0, risk_mode="fixed",
        risk_fixed_amount=1000.0, risk_percentage=0.01, mis_enabled=False,
    )
    cm.setup_budgets_pct = {SETUP: budget_pct}
    cm.setup_budget_used = {SETUP: 0.0}
    return cm


def _enter(cm, qty, price=100.0, setup=SETUP):
    return cm.can_enter_position(symbol="NSE:X", qty=qty, price=price,
                                 setup_type=setup)


def test_budget_blocks_once_exhausted():
    """The 2026-08-17 case: 20% of Rs5L is Rs100k of margin, not Rs400k."""
    cm = _cm()
    cm.setup_budget_used[SETUP] = 100_000.0        # budget fully consumed
    ok, adj, reason = _enter(cm, qty=1000)          # wants Rs100k more
    assert ok is False and adj == 0
    assert "setup_budget_exhausted" in reason


def test_partial_room_scales_instead_of_rejecting():
    cm = _cm()
    cm.setup_budget_used[SETUP] = 70_000.0          # Rs30k of room left
    ok, adj, _ = _enter(cm, qty=1000, price=100.0)  # wants Rs100k
    assert ok is True
    assert adj == 300, "should size into the remaining Rs30k, not the full request"


def test_within_budget_is_untouched():
    cm = _cm()
    ok, adj, _ = _enter(cm, qty=500, price=100.0)   # Rs50k of a Rs100k budget
    assert ok is True and adj == 500


def test_budget_is_a_percent_not_a_fraction():
    """capital_budget_pct is 20 meaning 20%, per the briefs. Reading it as a
    fraction would give a Rs100 budget and block everything."""
    cm = _cm(budget_pct=20.0)
    ok, adj, _ = _enter(cm, qty=900, price=100.0)   # Rs90k < Rs100k budget
    assert ok is True and adj == 900


def test_setups_without_a_budget_are_unaffected():
    cm = _cm()
    ok, adj, _ = _enter(cm, qty=5000, price=100.0, setup="some_other_setup")
    assert ok is True and adj == 5000


def test_four_concurrent_positions_cannot_exceed_the_budget():
    """Reproduces the incident: repeated entries must stop at the budget."""
    cm = _cm()
    granted = []
    for _ in range(6):
        ok, adj, _ = _enter(cm, qty=1000, price=100.0)   # Rs100k each
        if not ok or adj < 1:
            break
        granted.append(adj)
        cm.setup_budget_used[SETUP] += (adj * 100.0)     # simulate the open
    total_margin = sum(g * 100.0 for g in granted)
    assert total_margin <= 100_000.0 + 1e-6, (
        f"setup took Rs{total_margin:,.0f} against a Rs100,000 budget")


def test_released_budget_becomes_available_again():
    cm = _cm()
    cm.setup_budget_used[SETUP] = 100_000.0
    assert _enter(cm, qty=100)[0] is False
    cm.setup_budget_used[SETUP] = 0.0                    # positions closed
    assert _enter(cm, qty=100)[0] is True
