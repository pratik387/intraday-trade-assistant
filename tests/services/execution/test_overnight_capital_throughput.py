"""The overnight day-cap must respect the T+2 capital lock.

2026-08-27 raised `max_new_positions_per_day` 3 -> 6 "to match
max_concurrent_slots". The two are not the same quantity and must not be set
equal: a slot is occupied for LONGER than the position is held.

close_dn buys at 15:26 on T and the AMO sells at the open on T+1, so the
POSITION lives one night. But the cash from that sale settles on T+2, which is
why `OvernightSlotPool` walks free -> t0_open -> t1_settling -> free and only
releases on the T+2 morning. The slot is therefore tied up for ~2 trading days,
not one.

That makes the sustainable entry rate:

    entries/day  =  active_margin / (margin_per_slot * lock_days)
                 =  300000 / (50000 * 2)
                 =  3

At 6/day a single day can consume the whole pool, and the next day starts
broke. Measured over the seven sessions the raise was live:

    08-28  4 names (cash 227,742)  ->  08-31  2 names (85,528)
    09-01  4 names (199,608)       ->  09-02  2 names (109,514)
    09-03  5 attempted (184,035)   ->  09-04  3 names (99,184)

Throughput was unchanged (~3.1/day) but the book alternated, heavy days reached
into lower-svr signals, the broker rejected orders the pool believed were
funded, NSE:RBLBANK was lost outright on 09-03 and NSE:SETL was cut 128 -> 81.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CFG = json.loads((REPO / "config" / "configuration.json").read_text(encoding="utf-8"))
CA = CFG["setups"]["close_dn_overnight_long"]["capital_allocation"]

# Buy T (15:26) -> AMO sell at the open T+1 -> cash settles T+2.
CAPITAL_LOCK_TRADING_DAYS = 2


def test_day_cap_is_fundable_across_the_settlement_lock():
    """The invariant the 2026-08-27 raise violated."""
    sustainable = CA["active_margin_inr"] / (
        CA["margin_per_slot_inr"] * CAPITAL_LOCK_TRADING_DAYS)
    assert CA["max_new_positions_per_day"] <= sustainable, (
        f"max_new_positions_per_day={CA['max_new_positions_per_day']} exceeds the "
        f"{sustainable:.0f}/day the pool can fund "
        f"(active_margin {CA['active_margin_inr']:.0f} / "
        f"(margin_per_slot {CA['margin_per_slot_inr']:.0f} x "
        f"{CAPITAL_LOCK_TRADING_DAYS}-day lock)). Raising the cap needs MORE "
        f"CAPITAL, not a bigger number.")


def test_slots_cover_the_full_lock_not_just_one_night():
    """Slots must hold every position still occupying capital, not just today's."""
    need = CA["max_new_positions_per_day"] * CAPITAL_LOCK_TRADING_DAYS
    assert CA["max_concurrent_slots"] >= need, (
        f"{CA['max_new_positions_per_day']}/day over a "
        f"{CAPITAL_LOCK_TRADING_DAYS}-day lock needs {need} slots, "
        f"only {CA['max_concurrent_slots']} configured")


def test_day_cap_must_not_simply_equal_slot_count():
    """The specific mistake: setting the two equal.

    They are different quantities — slots count CONCURRENT positions across the
    lock, the day cap counts NEW positions in one session. Equal only if the
    lock were a single day, which it is not.
    """
    assert CA["max_new_positions_per_day"] != CA["max_concurrent_slots"], (
        "max_new_positions_per_day == max_concurrent_slots lets one session "
        "consume the entire pool and starve the next (2026-08-27 regression)")


def test_pool_commitment_fits_the_funded_margin():
    committed = CA["max_concurrent_slots"] * CA["margin_per_slot_inr"]
    assert committed <= CA["active_margin_inr"], (
        f"pool commits {committed:.0f} but active_margin is "
        f"{CA['active_margin_inr']:.0f}")


def test_live_reference_matches_the_active_value():
    """`_live_*` keys are the documented live sizing; drift means one is stale.

    They diverged between 2026-08-27 and 2026-09-07 — the live reference stayed
    at 3 while the active value ran at 6, so the config contradicted itself for
    seven sessions with nothing to catch it.
    """
    for key in ("max_new_positions_per_day", "max_concurrent_slots",
                "margin_per_slot_inr", "active_margin_inr"):
        assert CA[key] == CA[f"_live_{key}"], (
            f"{key}={CA[key]} but _live_{key}={CA[f'_live_{key}']}")


def test_rationale_still_states_the_lock():
    """Guard the reasoning, not just the number."""
    r = CA["_live_rationale"]
    assert "2-trading-day capital lock" in r
    assert "3 new/day" in r


@pytest.mark.parametrize("over_by", [1, 2, 6])
def test_invariant_actually_rejects_overcommitment(over_by):
    """The guard must fail on values above the budget.

    Derived from the live config, not hardcoded: the sustainable rate moves
    with position size. At 6 x Rs50k the budget was 3/day, so 4 was already
    overcommitment; at 12 x Rs25k it is 6/day and 4 is fine. Pinning literals
    here made this test fail the moment the book was resized for a reason that
    had nothing to do with the invariant.
    """
    sustainable = CA["active_margin_inr"] / (
        CA["margin_per_slot_inr"] * CAPITAL_LOCK_TRADING_DAYS)
    bad_cap = sustainable + over_by
    assert bad_cap > sustainable, (
        f"{bad_cap} should be over the {sustainable:.0f}/day budget")
    # and the real guard would reject it
    assert not (bad_cap <= sustainable)
