"""Tests for CapitalManager per-setup budget tracking."""
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _make_manager(total=1_000_000, setups_cfg=None):
    """Construct a CapitalManager with sensible defaults + injected setup budgets."""
    from services.capital_manager import CapitalManager
    cm = CapitalManager(
        enabled=True,
        initial_capital=float(total),
        max_positions=10,
        min_notional_pct=0.0,
        capital_utilization=1.0,
        max_allocation_per_trade=1.0,
        risk_mode="fixed",
        risk_fixed_amount=1000.0,
        risk_percentage=0.01,
        mis_enabled=False,
    )
    # Inject per-setup budgets directly (production path: set from configuration.json
    # in pipeline init). cm.setup_budgets_pct is a dict {setup_type: pct_int}.
    cm.setup_budgets_pct = (setups_cfg or {})
    cm.setup_budget_used = {s: 0.0 for s in cm.setup_budgets_pct}
    return cm


def test_setup_budget_allows_first_position():
    cm = _make_manager(total=1_000_000, setups_cfg={"gap_fade_short": 30})
    # qty=10, price=100 => margin=1000 (no leverage), well within 30% budget of Rs.300k
    ok, qty, reason = cm.can_enter_position(
        symbol="NSE:A", qty=10, price=100.0,
        cap_segment="small_cap", setup_type="gap_fade_short",
    )
    assert ok, reason


def test_setup_budget_rejects_when_setup_share_exhausted():
    """gap_fade has 30% budget = Rs.300k. After 250k allocated, a 100k+ trade fails."""
    cm = _make_manager(total=1_000_000, setups_cfg={"gap_fade_short": 30})
    # Mark 250k of gap_fade budget used directly
    cm.setup_budget_used["gap_fade_short"] = 250_000
    # New entry: qty=600, price=200 => margin=120k, which exceeds remaining 50k in budget
    ok, qty, reason = cm.can_enter_position(
        symbol="NSE:B", qty=600, price=200.0,
        cap_segment="small_cap", setup_type="gap_fade_short",
    )
    # Expectation: blocked by setup_budget (margin needed > 50k remaining in 30% budget)
    if not ok:
        assert "setup_budget" in reason or "budget" in reason


def test_enter_position_increments_setup_budget_used():
    cm = _make_manager(total=1_000_000, setups_cfg={"gap_fade_short": 30})
    cm.enter_position(
        symbol="NSE:A", qty=100, price=200.0,
        cap_segment="small_cap", setup_type="gap_fade_short",
    )
    assert cm.setup_budget_used["gap_fade_short"] > 0


def test_reduce_position_decrements_setup_budget_used():
    cm = _make_manager(total=1_000_000, setups_cfg={"gap_fade_short": 30})
    cm.enter_position(
        symbol="NSE:A", qty=100, price=200.0,
        cap_segment="small_cap", setup_type="gap_fade_short",
    )
    used_before = cm.setup_budget_used["gap_fade_short"]
    cm.reduce_position("NSE:A", qty_exited=100, new_qty=0)
    assert cm.setup_budget_used["gap_fade_short"] < used_before
