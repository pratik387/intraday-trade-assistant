"""R-multiple must be measured against INITIAL (decision-time) risk.

Bug: both exit sites computed risk_per_unit from the CURRENT stop. After T1 the
stop is moved to breakeven (exit_t1_move_sl_to_be), collapsing the denominator
and inflating R on every scaled winner (paper showed +5.15R avg win realizing
₹365 while ₹1,000-risk losers read −0.89R). Fix: use _decision_sl, fall back to
the live stop only when the decision SL wasn't snapshotted.
"""
import logging
from unittest.mock import MagicMock

import services.execution.exit_executor as ee_mod
ee_mod.logger = logging.getLogger("test_r_multiple")
ee_mod.trade_logger = MagicMock()

from services.execution.exit_executor import _r_multiple


def test_scaled_winner_uses_initial_risk_not_breakeven_stop():
    # SHORT entry 100, decision SL 102 (risk 2/unit). After T1, live stop -> BE 100.1.
    # Cover 100 sh at 99 -> pnl = (100-99)*100 = 100. True R = 100/(100*2) = 0.5.
    r = _r_multiple(entry_price=100.0, pnl=100.0, qty=100,
                    decision_sl=102.0, current_sl=100.1)
    assert r == 0.5  # NOT 10.0 (the BE-denominator bug)


def test_falls_back_to_current_stop_when_decision_sl_missing():
    r = _r_multiple(entry_price=100.0, pnl=100.0, qty=100,
                    decision_sl=None, current_sl=102.0)
    assert r == 0.5


def test_none_when_no_stop_available():
    assert _r_multiple(100.0, 100.0, 100, decision_sl=None, current_sl=None) is None


def test_none_when_zero_risk():
    assert _r_multiple(100.0, 100.0, 100, decision_sl=100.0, current_sl=100.0) is None


def test_loser_r_is_negative_full_stop():
    # SHORT entry 100, decision SL 102, price runs to 102 -> pnl = (100-102)*100 = -200
    # R = -200/(100*2) = -1.0 (full initial-risk stop)
    r = _r_multiple(entry_price=100.0, pnl=-200.0, qty=100,
                    decision_sl=102.0, current_sl=102.0)
    assert r == -1.0
