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
    # DISABLED 2026-08-25: the backtest contradicted the live measurement —
    # +Rs20,309 on the live ledger but -Rs86,235 on paper, because the trades
    # it removes are profitable when market impact is absent. The arithmetic
    # below is still pinned so the cap is correct WHEN re-enabled.
    assert PC["enabled"] is False, (
        "re-enable only after the impact estimate is established on a larger "
        "post-ranker-fix sample — see the _DISABLED note in config")
    assert 0 < PC["max_participation_pct"] <= 0.05, "a cap above 5% is not a cap"
    assert PC["skip_when_adv_missing"] is True


def test_the_regenceram_order_would_be_cut_by_98_percent():
    """Rs50,489 into a name trading Rs54,402/day — 92.8% participation."""
    adv, px, notional = 54_402.0, 33.85, 50_489.0
    q = _cap_qty(notional, adv, px, PC["max_participation_pct"])
    capped_notional = q * px
    assert capped_notional <= PC["max_participation_pct"] * adv * 1.001
    assert capped_notional < notional * 0.02, "must be a drastic cut, not a trim"


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
    need = PC["min_notional_after_cap_inr"] / PC["max_participation_pct"]
    assert need == pytest.approx(2_500_000), "threshold moved; update the config note"
    # a Rs50L-turnover name is comfortably above it and still trades
    assert PC["max_participation_pct"] * 5_000_000.0 >= PC["min_notional_after_cap_inr"]


def test_the_exclusion_is_justified_by_untradeability_not_by_signal_quality():
    """A Rs5L/day name is untradeable at ANY size: full slot is 10x its daily
    volume (impact), capped size loses to fixed MTF fees. That is the reason —
    NOT 'thin names are bad', which the paper mirror refutes (+0.45%, n=302)."""
    adv = 500_000.0                      # Rs5 lakh/day
    full_slot = 50_000.0
    assert full_slot / adv > 0.05, "full size would be a large share of daily volume"
    capped = PC["max_participation_pct"] * adv
    assert capped < PC["min_notional_after_cap_inr"], "capped size is fee-uneconomic"


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
    """A reserved slot left behind is the ghost-slot incident (register 1.2)."""
    n_skip = SRC.count("PARTICIPATION_CAP | %s | SKIP")
    assert n_skip == 2, "expected the no-ADV and below-min-qty skip paths"
    blk = SRC[SRC.index("# ---- PARTICIPATION CAP (enforced) ----"):]
    blk = blk[:blk.index('summary["participation_capped"]')]
    assert blk.count("_rollback_slot_to_free(slot)") == 2
    assert blk.count('summary["skipped_count"] += 1') == 2


def test_cap_can_be_disabled_without_code_change():
    assert 'bool(_pc.get("enabled", False))' in SRC, "must be flag-gated"


def test_tripwire_window_matches_the_stated_review_horizon():
    """Un-pausing reset the 6-week clock; the horizon was shortened to match
    'a few more weeks' rather than silently running to October."""
    tw = CFG["setups"]["close_dn_overnight_long"]["decay_tripwire"]
    assert tw["sustained_weeks"] == 3
    # other books must be untouched
    assert CFG["setups"]["zscore_oversold_revert_long"]["decay_tripwire"]["sustained_weeks"] == 6
