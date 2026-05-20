"""Regression tests for target-side validation in target_recalc._recalc_structural.

The bug: detectors emit structural T1/T2 levels validated against their
detection-time `close`. The trade actually enters at `actual_entry` (from
TICK_TRIGGERED price inside the entry zone), which can be lower than `close`
for SHORT (or higher for LONG). When a structural target sits between
`actual_entry` and `close`, the detector's "favorable side" override missed
it — and T2_FULL then fires immediately as a small loss.

Concrete cases (gap_fade_short, both May 19 OCI runs):
  - 2023-01-04 NSE:INFOBEAN  entry=532.50  exit=533.15  T2_FULL  loss
  - 2023-01-05 NSE:MALLCOM  entry=718.30  exit=718.55  T2_FULL  loss
  - 2024-12-02 NSE:KKCL      entry=630.90  exit=631.90  T2_FULL  loss

Fix: at target_recalc time (when actual_entry is known), detect any target
on the adverse side and override using the same R-multiple fallback the
detector uses (T1=0.5R, T2=1.0R favorable from actual_entry).
"""
from services.target_recalc import recalculate_targets_for_actual_entry


def _short_plan_with_adverse_target(actual_entry=630.90, decision_close=632.05,
                                     hard_sl=652.90, t2_level=631.90):
    """KKCL-shaped plan: SHORT with T2 between actual_entry and decision close."""
    return {
        "symbol": "NSE:KKCL",
        "strategy": "gap_fade_short",
        "entry_ref_price": decision_close,
        "price": decision_close,
        "stop": {"hard": hard_sl, "risk_per_share": hard_sl - decision_close},
        "sizing": {"risk_per_share": hard_sl - decision_close, "qty": 47},
        "risk_per_share": hard_sl - decision_close,
        "target_anchor_type": "structural",
        "targets": [
            {"name": "T1", "level": 631.50, "rr": 0.026, "qty_pct": 0.5, "action": "partial_exit"},
            {"name": "T2", "level": t2_level, "rr": 0.007, "qty_pct": 0.5, "action": "exit_full"},
        ],
    }


def test_short_target_on_adverse_side_gets_overridden():
    """REGRESSION (2026-05-20): T2 above actual_entry for SHORT should be
    overridden to a favorable (below-entry) level using the 1.0R fallback."""
    plan = _short_plan_with_adverse_target(
        actual_entry=630.90, decision_close=632.05,
        hard_sl=652.90, t2_level=631.90,
    )
    adjusted = recalculate_targets_for_actual_entry(plan, actual_entry=630.90, side="SELL")

    targets = adjusted["targets"]
    t1, t2 = targets[0], targets[1]

    # Both T1 and T2 were on adverse side -> both should be overridden
    assert t1["level"] < 630.90, f"T1 must be below entry for SHORT, got {t1['level']}"
    assert t2["level"] < 630.90, f"T2 must be below entry for SHORT, got {t2['level']}"

    # Verify the override formula: actual_rps = 652.90 - 630.90 = 22.00
    # T1 at 0.5R favorable: 630.90 - 11.00 = 619.90
    # T2 at 1.0R favorable: 630.90 - 22.00 = 608.90
    assert abs(t1["level"] - 619.90) < 0.01
    assert abs(t2["level"] - 608.90) < 0.01


def test_short_target_already_favorable_unchanged():
    """If targets are already on favorable side (below actual_entry), they
    must NOT be touched."""
    plan = _short_plan_with_adverse_target(
        actual_entry=630.90, decision_close=632.05,
        hard_sl=652.90, t2_level=610.00,
    )
    # T2 at 610.00 is below 630.90 entry — favorable. T1 at 631.50 IS adverse.
    adjusted = recalculate_targets_for_actual_entry(plan, actual_entry=630.90, side="SELL")
    targets = adjusted["targets"]
    # T1 (adverse) was overridden
    assert targets[0]["level"] < 630.90
    # T2 (already favorable) was kept
    assert abs(targets[1]["level"] - 610.00) < 0.01


def test_long_adverse_target_overridden():
    """LONG: T2 BELOW actual_entry is adverse, gets overridden to 1.0R above."""
    plan = {
        "symbol": "NSE:TEST",
        "strategy": "long_panic_gap_down",
        "entry_ref_price": 100.0,
        "price": 100.0,
        "stop": {"hard": 95.0, "risk_per_share": 5.0},
        "sizing": {"risk_per_share": 5.0, "qty": 100},
        "risk_per_share": 5.0,
        "target_anchor_type": "structural",
        "targets": [
            {"name": "T1", "level": 99.0, "rr": 0.0, "qty_pct": 0.5, "action": "partial_exit"},
            {"name": "T2", "level": 98.0, "rr": 0.0, "qty_pct": 0.5, "action": "exit_full"},
        ],
    }
    # Actual entry filled HIGHER than decision close (e.g., entry zone trigger from above)
    adjusted = recalculate_targets_for_actual_entry(plan, actual_entry=101.0, side="BUY")

    targets = adjusted["targets"]
    # Both T1=99 and T2=98 are BELOW actual_entry 101 -> adverse for LONG
    assert targets[0]["level"] > 101.0, f"T1 must be above entry for LONG, got {targets[0]['level']}"
    assert targets[1]["level"] > 101.0, f"T2 must be above entry for LONG, got {targets[1]['level']}"

    # actual_rps = 101.0 - 95.0 = 6.00. T1 at 0.5R = 104.0; T2 at 1.0R = 107.0
    assert abs(targets[0]["level"] - 104.0) < 0.01
    assert abs(targets[1]["level"] - 107.0) < 0.01


def test_no_override_when_actual_entry_equals_decision():
    """When actual_entry == detection close, targets are unchanged
    (degenerate but valid case)."""
    plan = {
        "symbol": "NSE:OK",
        "strategy": "gap_fade_short",
        "entry_ref_price": 100.0,
        "price": 100.0,
        "stop": {"hard": 102.0, "risk_per_share": 2.0},
        "sizing": {"risk_per_share": 2.0, "qty": 50},
        "risk_per_share": 2.0,
        "target_anchor_type": "structural",
        "targets": [
            {"name": "T1", "level": 99.0, "rr": 0.5, "qty_pct": 0.5, "action": "partial_exit"},
            {"name": "T2", "level": 98.0, "rr": 1.0, "qty_pct": 0.5, "action": "exit_full"},
        ],
    }
    adjusted = recalculate_targets_for_actual_entry(plan, actual_entry=100.0, side="SELL")
    targets = adjusted["targets"]
    # Both targets already below actual_entry → unchanged
    assert abs(targets[0]["level"] - 99.0) < 0.01
    assert abs(targets[1]["level"] - 98.0) < 0.01
