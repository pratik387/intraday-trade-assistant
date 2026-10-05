"""The participation cap must bound order size by name liquidity.

Measured on 96 live overnight trades with point-in-time ADV: positions above 5%
of a name's median daily rupee turnover returned -2.234%/trade and produced 92%
of the book's loss; positions below 1% were profitable. corr(participation,
return) = -0.69 on the clean subset. The paper mirror shows no such effect
(corr -0.03) and thin names are not worse trades there (thinnest ADV quintile
+0.45%, n=302), so the cause is order size, not name selection.

The structural argument stands without the P&L: REGENCERAM was a Rs50k order in
a name trading Rs54k/day (92.8%), and its exit AMO sat above the market because
there was no one to sell to.
"""
import io
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
CFG = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))
PC = CFG["setups"]["close_dn_overnight_long"]["participation_cap"]
SRC = (ROOT / "services" / "execution" / "overnight_handlers.py").read_text(encoding="utf-8")


def _cap_qty(notional, adv, px, pct):
    """The arithmetic the handler performs."""
    if notional <= pct * adv:
        return int(notional // px)
    return int((pct * adv) // px)


def test_config_is_present_and_sane():
    # ENABLED after a slot-constrained backtest: -Rs16,496 -> +Rs3,658 on the
    # live ledger, still positive with conservative replacement pricing. The
    # earlier paper comparison was invalid — it applied the filter to all 340
    # fires when live only ever takes 3/day from a ranked list.
    assert PC["enabled"] is True
    # 2026-10-05: 10%, the level above which the only two live fills lost 3.7pp
    # to execution; below it 13 fills cost ~0.25pp. See
    # test_overnight_floor_and_tick_gate.test_fixed_size_rule_2026_10_05.
    assert 0 < PC["max_participation_pct"] <= 0.10, "a cap above 10% is not a cap"
    assert PC["skip_when_adv_missing"] is True


def test_the_regenceram_order_is_skipped_not_cut():
    """Rs50,489 into a name trading Rs54,402/day — 92.8% participation.
    2026-08-25 cut it to Rs1k; 2026-10-05 fixed-size rule skips it: the capped
    size is under the floor, and a Rs1k position earns nothing live anyway."""
    adv, px, notional = 54_402.0, 33.85, 50_489.0
    q = _cap_qty(notional, adv, px, PC["max_participation_pct"])
    capped_notional = q * px
    assert capped_notional <= PC["max_participation_pct"] * adv * 1.001
    assert capped_notional < PC["min_notional_after_cap_inr_cnc"], "must be skipped, never traded as a token"


def test_a_liquid_name_is_untouched():
    """0.05% participation must pass through unchanged — the cap is for the tail."""
    adv, px = 100_000_000.0, 500.0
    notional = 50_000.0
    assert notional / adv < PC["max_participation_pct"]
    assert _cap_qty(notional, adv, px, PC["max_participation_pct"]) == int(notional // px)


@pytest.mark.parametrize("adv", [1_000_000.0, 10_000_000.0, 100_000_000.0])
def test_capped_notional_never_exceeds_the_limit(adv):
    px, notional = 100.0, 50_000.0
    q = _cap_qty(notional, adv, px, PC["max_participation_pct"])
    assert q * px <= max(PC["max_participation_pct"] * adv, notional) + 1e-6


def test_tiny_positions_are_skipped_not_traded():
    """A capped-to-token position pays fixed MTF costs and burns a slot.
    Measured round-trip on the MTF path: 7.46% of notional at Rs544, 1.66% at
    Rs5,000, 0.41% at Rs50,000 — against a ~0.35% gross edge."""
    floor = PC["min_notional_after_cap_inr"]
    assert floor >= 20_000, "floor must exceed the size where MTF fees swamp the edge"
    # REGENCERAM capped to Rs544 must be SKIPPED, not traded
    adv, px, notional = 54_402.0, 33.85, 50_489.0
    q = _cap_qty(notional, adv, px, PC["max_participation_pct"])
    assert q * px < floor, "the pathological case must fall below the floor"


def test_the_floor_implies_a_known_liquidity_threshold():
    """cap+floor together mean a name needs ADV >= floor/cap to trade at all.
    Measured 2026-08-25: that is Rs25 lakh, which excludes 457 of 2,114
    candidates (21.6%). Documented rather than accidental — the config note
    carries the number, and changing either knob moves the threshold."""
    # 2026-10-05: cap 10%, floor = the Rs25k slot -> a name needs ADV >= Rs2.5L.
    # (2026-08-25 it was Rs25L at the 1% cap; the fixed-size rule widened it.)
    need = PC["min_notional_after_cap_inr"] / PC["max_participation_pct"]
    assert need == pytest.approx(250_000), "threshold moved; update the config note"
    # a Rs5L-turnover name is above it and trades at FULL size
    assert PC["max_participation_pct"] * 500_000.0 >= PC["min_notional_after_cap_inr"]


def test_the_exclusion_is_justified_by_untradeability_not_by_signal_quality():
    """A Rs5L/day name is untradeable at ANY size: full slot is 10x its daily
    volume (impact), capped size loses to fixed MTF fees. That is the reason —
    NOT 'thin names are bad', which the paper mirror refutes (+0.45%, n=302)."""
    # 2026-10-05: the untradeable line is 10% of ADV, measured on live fills
    # (two fills above it lost 3.7pp to execution; 13 below it ~0.25pp).
    # A Rs2L/day name: Rs25k is 12.5% of its day -> skipped.
    # A Rs5L/day name: Rs25k is 5% -> full size. Not shrunk, not skipped.
    slot = 25_000.0
    assert PC["max_participation_pct"] * 200_000.0 < PC["min_notional_after_cap_inr"], "Rs2L name must skip"
    assert PC["max_participation_pct"] * 500_000.0 >= slot, "Rs5L name trades at full size"


def test_skipped_trade_releases_the_slot_for_the_next_candidate():
    i = SRC.index("slot released")
    seg = SRC[i:i + 500]
    assert "_rollback_slot_to_free(slot)" in seg
    assert "continue" in seg
    # reserve() must sit INSIDE the ranked loop or the freed slot is wasted
    loop = SRC.index("for rank_i, (symbol, evt, plan) in enumerate(ranked):")
    res = SRC.index("slot = pool.reserve(", loop)
    nxt = SRC.index("PARTICIPATION_CAP", loop)
    assert loop < res < nxt, "reserve() must be per-candidate inside the loop"


def test_no_silent_uncapped_path():
    """Every branch either caps, skips loudly, or logs that it is sending
    uncapped. A silent pass-through defeats the control."""
    blk = SRC[SRC.index("# ---- PARTICIPATION CAP (enforced) ----"):]
    blk = blk[:blk.index('summary["participation_capped"]')]
    assert "no usable price" in blk, "price<=0 must not silently skip the cap"
    assert "order sent UNCAPPED" in blk, "adv-missing fallback must announce itself"


def test_entry_basis_reports_the_size_actually_sent():
    """ENTRY_BASIS previously logged the PRE-cap qty, so the observability
    disagreed with the order it was meant to explain."""
    cap = SRC.index("# ---- PARTICIPATION CAP (enforced) ----")
    eb = SRC.index("ENTRY_BASIS | %s")
    place = SRC.index("buy_order_id = _place_buy")
    assert cap < eb < place, "ENTRY_BASIS must log after the cap, before placement"


def test_handler_enforces_and_does_not_merely_log():
    assert "PARTICIPATION_CAP" in SRC
    assert "plan.qty = _capped_qty" in SRC, "the cap must actually resize the order"
    assert 'summary["participation_capped"]' in SRC, "capping must be counted"


def test_missing_adv_skips_rather_than_trading_blind():
    i = SRC.index("PARTICIPATION_CAP | %s | SKIP — no ADV")
    seg = SRC[i:i + 400]
    assert "_rollback_slot_to_free(slot)" in seg, "a skipped entry must free its slot"
    assert "continue" in seg


def test_skip_paths_free_the_slot_and_count_it():
    """A reserved slot left behind is the ghost-slot incident (register 1.2).

    The count is pinned so a NEW skip path cannot be added without also adding
    its rollback — which is the whole point of the assertions below. Fourth
    path added 2026-09-07: tick cost above max_tick_pct_of_price.
    """
    n_skip = SRC.count("PARTICIPATION_CAP | %s | SKIP")
    assert n_skip == 4, (
        "expected skips for: no ADV, no usable price, tick too wide, below floor")
    blk = SRC[SRC.index("# ---- PARTICIPATION CAP (enforced) ----"):]
    blk = blk[:blk.index('summary["participation_capped"]')]
    # every skip must free the slot and count it, or a reserved slot leaks
    # (the ghost-slot incident, register 1.2)
    assert blk.count("_rollback_slot_to_free(slot)") == n_skip
    assert blk.count('summary["skipped_count"] += 1') == n_skip


def test_cap_can_be_disabled_without_code_change():
    assert 'bool(_pc.get("enabled", False))' in SRC, "must be flag-gated"


def test_tripwire_window_matches_the_stated_review_horizon():
    """Un-pausing reset the 6-week clock; the horizon was shortened to match
    'a few more weeks' rather than silently running to October."""
    tw = CFG["setups"]["close_dn_overnight_long"]["decay_tripwire"]
    assert tw["sustained_weeks"] == 3
    # other books must be untouched
    assert CFG["setups"]["zscore_oversold_revert_long"]["decay_tripwire"]["sustained_weeks"] == 6
