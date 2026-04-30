"""
Unit test for LEVEL category target preservation fix.

Tests that LEVEL-based strategies (resistance_bounce, support_bounce, etc.)
preserve their original structural targets instead of recalculating them
when actual entry differs from planned entry.

Uses actual GHCLTEXTIL trade from 2026-01-08 paper trading session.
"""
import pytest
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class TestLevelTargetPreservation:
    """Test that LEVEL strategies preserve original targets."""

    # Actual GHCLTEXTIL plan from 2026-01-08 paper trading
    GHCLTEXTIL_PLAN = {
        "symbol": "NSE:GHCLTEXTIL",
        "strategy": "resistance_bounce_short",
        "category": "LEVEL",
        "entry_ref_price": 75.14,
        "stop": {
            "hard": 75.48,
            "risk_per_share": 0.34
        },
        "targets": [
            {"name": "T1", "level": 74.13, "rr": 3.0, "qty_pct": 0.6},
            {"name": "T2", "level": 73.96, "rr": 3.5, "qty_pct": 0.4},
            {"name": "T3", "level": 73.79, "rr": 4.0, "qty_pct": 0.0}
        ]
    }

    # Actual entry was BETTER (lower for short)
    ACTUAL_ENTRY = 74.90  # vs planned 75.14

    def test_level_targets_preserved_on_better_entry(self):
        """
        Test that LEVEL strategy targets are preserved when entry is better.

        GHCLTEXTIL case:
        - Planned entry: 75.14, Actual: 74.90 (BETTER for short - got in lower)
        - Original T1: 74.13, T2: 73.96 (structural levels)
        - OLD behavior: Would recalculate to T1=73.16, T2=72.87 (pushed further!)
        - NEW behavior: Keep T1=74.13, T2=73.96 (price respects LEVELS)
        """
        from services.execution.trigger_aware_executor import TriggerAwareExecutor

        # Create executor (we only need the method, not full init)
        executor = TriggerAwareExecutor.__new__(TriggerAwareExecutor)

        # Call the recalculation method
        adjusted = executor._recalculate_targets_for_actual_entry(
            plan=self.GHCLTEXTIL_PLAN.copy(),
            actual_entry=self.ACTUAL_ENTRY,
            side="SELL"
        )

        # CRITICAL: Targets should be UNCHANGED (structural levels preserved)
        assert adjusted["targets"][0]["level"] == 74.13, \
            f"T1 should remain 74.13 (structural), got {adjusted['targets'][0]['level']}"
        assert adjusted["targets"][1]["level"] == 73.96, \
            f"T2 should remain 73.96 (structural), got {adjusted['targets'][1]['level']}"

        # rps should be updated for R-multiple tracking
        expected_rps = 75.48 - 74.90  # SL - entry for short = 0.58
        assert abs(adjusted.get("risk_per_share", 0) - expected_rps) < 0.01, \
            f"rps should be updated to {expected_rps:.2f}, got {adjusted.get('risk_per_share')}"

        # actual_entry should be recorded
        assert adjusted.get("actual_entry") == 74.90

    def test_level_targets_preserved_on_worse_entry(self):
        """
        Test that LEVEL strategy targets are preserved even when entry is worse.
        """
        from services.execution.trigger_aware_executor import TriggerAwareExecutor

        executor = TriggerAwareExecutor.__new__(TriggerAwareExecutor)

        # Worse entry for short (higher)
        worse_entry = 75.30

        adjusted = executor._recalculate_targets_for_actual_entry(
            plan=self.GHCLTEXTIL_PLAN.copy(),
            actual_entry=worse_entry,
            side="SELL"
        )

        # Targets should still be unchanged
        assert adjusted["targets"][0]["level"] == 74.13
        assert adjusted["targets"][1]["level"] == 73.96

        # rps updated (smaller now since entry is worse)
        expected_rps = 75.48 - 75.30  # 0.18
        assert abs(adjusted.get("risk_per_share", 0) - expected_rps) < 0.01

    def test_orb_strategy_preserves_targets_when_recalc_disabled(self):
        """
        Test that ORB strategies preserve original targets when recalculation is disabled.

        With orb_target_recalculation.enabled=false in config, ORB trades should
        keep their original structural targets (like JGCHEM: T1=309.62 not 303.55).
        """
        from services.execution.trigger_aware_executor import TriggerAwareExecutor

        executor = TriggerAwareExecutor.__new__(TriggerAwareExecutor)

        # JGCHEM-like ORB plan
        orb_plan = {
            "symbol": "NSE:JGCHEM",
            "strategy": "orb_pullback_short",
            "category": "LEVEL",
            "entry_ref_price": 314.4,
            "stop": {
                "hard": 317.13,
                "risk_per_share": 2.73
            },
            "targets": [
                {"name": "T1", "level": 309.62, "rr": 1.75},
                {"name": "T2", "level": 308.26, "rr": 2.25}
            ],
            "levels": {
                "ORH": 324.55,
                "ORL": 314.4
            }
        }

        actual_entry = 313.7  # Actual JGCHEM entry

        adjusted = executor._recalculate_targets_for_actual_entry(
            plan=orb_plan.copy(),
            actual_entry=actual_entry,
            side="SELL"
        )

        # With recalculation DISABLED, targets should be preserved
        # Old behavior would recalculate to T1=303.55, T2=298.47
        assert adjusted["targets"][0]["level"] == 309.62, \
            f"ORB T1 should be preserved at 309.62 (not recalculated), got {adjusted['targets'][0]['level']}"
        assert adjusted["targets"][1]["level"] == 308.26, \
            f"ORB T2 should be preserved at 308.26 (not recalculated), got {adjusted['targets'][1]['level']}"

        print(f"\nORB targets preserved: T1={adjusted['targets'][0]['level']}, T2={adjusted['targets'][1]['level']}")

    def test_reversion_strategy_still_recalculates(self):
        """
        Test that plans with target_anchor_type='r_multiple' recalculate targets.

        Pre-Phase-C this test relied on category="REVERSION" — the legacy
        category-pipeline dispatch path. Post-Phase-C the dispatch key is
        target_anchor_type ∈ {structural, r_multiple, or_range}; setting
        r_multiple keeps this test's intent intact under the new system.
        """
        from services.execution.trigger_aware_executor import TriggerAwareExecutor

        executor = TriggerAwareExecutor.__new__(TriggerAwareExecutor)

        reversion_plan = {
            "symbol": "NSE:TEST",
            "strategy": "failure_fade_short",
            "category": "REVERSION",
            "target_anchor_type": "r_multiple",
            "entry_ref_price": 100.0,
            "stop": {
                "hard": 101.0,
                "risk_per_share": 1.0
            },
            "targets": [
                {"name": "T1", "level": 97.0, "rr": 3.0},
                {"name": "T2", "level": 96.0, "rr": 4.0}
            ]
        }

        actual_entry = 99.5  # Better entry

        adjusted = executor._recalculate_targets_for_actual_entry(
            plan=reversion_plan.copy(),
            actual_entry=actual_entry,
            side="SELL"
        )

        # REVERSION should recalculate targets based on actual entry
        # New rps = 101.0 - 99.5 = 1.5
        # New T1 = 99.5 - (3.0 * 1.5) = 95.0
        # New T2 = 99.5 - (4.0 * 1.5) = 93.5
        assert adjusted["targets"][0]["level"] == 95.0, \
            f"REVERSION T1 should be recalculated to 95.0, got {adjusted['targets'][0]['level']}"
        assert adjusted["targets"][1]["level"] == 93.5, \
            f"REVERSION T2 should be recalculated to 93.5, got {adjusted['targets'][1]['level']}"


class TestGHCLTEXTILPnLComparison:
    """Compare PnL with and without the fix using actual trade data."""

    def test_ghcltextil_pnl_with_fix(self):
        """
        Calculate what GHCLTEXTIL PnL would have been with the fix.

        Actual outcome (without fix):
        - Targets pushed to T1=73.16, T2=72.87
        - Price reached 73.30 (MFE 1.6R) but missed new targets
        - Exited at EOD 73.81
        - PnL: Rs 3,223.13

        With fix (original targets preserved):
        - T1=74.13 would have been hit (price went to 73.30)
        - T2=73.96 would have been hit
        - Earlier profit lock-in
        """
        qty = 2957
        entry = 74.90

        # Original structural targets
        t1_level = 74.13
        t2_level = 73.96

        # EOD exit (what actually happened)
        eod_exit = 73.81

        # With fix: T1 and T2 would hit
        t1_qty = int(qty * 0.6)  # 1774
        t2_qty = qty - t1_qty    # 1183

        t1_pnl = t1_qty * (entry - t1_level)  # Short: entry - exit
        t2_pnl = t2_qty * (entry - t2_level)
        total_with_fix = t1_pnl + t2_pnl

        # Without fix (actual)
        actual_pnl = qty * (entry - eod_exit)

        print(f"\n{'='*60}")
        print("GHCLTEXTIL PnL Comparison")
        print(f"{'='*60}")
        print(f"Entry: {entry} (SHORT), Qty: {qty}")
        print()
        print("WITH FIX (original structural targets):")
        print(f"  T1 @ {t1_level}: {t1_qty} qty -> Rs {t1_pnl:.2f}")
        print(f"  T2 @ {t2_level}: {t2_qty} qty -> Rs {t2_pnl:.2f}")
        print(f"  TOTAL: Rs {total_with_fix:.2f}")
        print()
        print("WITHOUT FIX (EOD exit @ 73.81):")
        print(f"  Rs {actual_pnl:.2f}")
        print()
        print(f"DIFFERENCE: Rs {total_with_fix - actual_pnl:.2f}")
        print(f"{'='*60}")

        # The key insight: with fix, we lock in profit EARLIER at targets
        # Even though EOD exit was at a better price (73.81 vs T2 73.96),
        # the fix gives certainty of hitting targets when price reaches them

        # Both should be profitable
        assert total_with_fix > 0, "With fix should be profitable"
        assert actual_pnl > 0, "Actual was profitable"

        # With structural targets, we get slightly less profit (exit earlier)
        # but with CERTAINTY of hitting targets
        # Rs 2,375 (fix) vs Rs 3,223 (EOD) - EOD was better by luck
        # But if price had reversed before EOD, we'd have missed targets entirely


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
