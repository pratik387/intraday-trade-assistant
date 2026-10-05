"""The small-position gate must key on tick cost, and the notional floor on product.

2026-09-07: nine fires, four taken. The day cap (6), the slots (12) and the cash
(Rs300k) all had room — the Rs25,000 min-notional floor removed five, every one
of them CNC:

    ASHOKAMET  capped Rs1,943   ADV Rs195,451   Rs14.61/share
    UDAYJEW    capped Rs4,172   ADV Rs421,347   Rs166.88/share
    MALUPAPER  capped Rs2,457   ADV Rs245,826   Rs33.66/share
    MITTAL     capped Rs5,698   ADV Rs569,932   Rs0.92/share
    MGEL       capped Rs9,263   ADV Rs927,282   Rs15.70/share

That floor was derived from MTF economics — MTF round trip carries a FIXED
pledge/unpledge component (7.46% of notional at Rs544, 1.66% at Rs5,000). CNC
has none: Zerodha delivery brokerage is Rs0, and 128 live fills put CNC at 0.22%
of notional FLAT down to Rs1,465 notional. Applying the MTF number to CNC was
simply the wrong cost model.

The backtest agrees the floor was cutting good trades. With ADV rebuilt the way
production computes it, the names the Rs25,000 floor removes EARN MORE than the
ones it keeps, in all three splits: Disc +0.710% vs +0.518%, OOS +1.047% vs
+0.312%, HO +1.166% vs +0.191%.

But notional is the wrong variable to gate on either way. What makes a small
position unusable is the tick: MITTAL at Rs0.92 has a 1.09% tick against a
~0.35% edge, and no amount of size fixes that. UDAYJEW at Rs166.88 has a 0.03%
tick and is a perfectly clean trade. A notional floor cannot tell those apart;
a tick gate can.

The threshold comes from economics, not from the research maximum. One tick is
a LOWER BOUND on the spread, so a threshold at or above the ~0.35% gross edge
would admit names whose cheapest possible round trip cannot be paid for. The
research names the old floor removed sit at a median 0.05-0.075% with p90
0.16%; the 0.495% max is one tail case, not the mass. 0.2% keeps the bulk of
what the backtest validated and still leaves the edge intact — which blocks
MGEL (0.32%) and ASHOKAMET (0.34%) as well as MITTAL, and passes UDAYJEW and
MALUPAPER.

An earlier draft set this at 0.5% on the reasoning "admit everything research
saw". That was fitting to a single observation and would have admitted names
whose minimum round trip exceeded the edge.
"""
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
CFG = json.loads((REPO / "config" / "configuration.json").read_text(encoding="utf-8"))
PC = CFG["setups"]["close_dn_overnight_long"]["participation_cap"]
SRC = (REPO / "services" / "execution" / "overnight_handlers.py").read_text(encoding="utf-8")


# ---------------------------------------------------------------- config ---

def test_floor_is_product_aware():
    assert "min_notional_after_cap_inr" in PC, "MTF floor must remain"
    assert "min_notional_after_cap_inr_cnc" in PC, "CNC needs its own floor"


def test_fixed_size_rule_2026_10_05():
    """FIXED SIZE: a position is the full slot or nothing. Never shrunk.

    Live ledger since 2026-09-08: 13 positions under Rs5k (the 1%-of-ADV shrink
    rule at work) earned Rs-1 in total while occupying slots that bind on 10 of
    16 days; full Rs25k CNC slots earned Rs285 each. Paper had said the shrunk
    names earn +2.5% - the fill gap on tiny illiquid positions ate all of it.
    Fill cost measured on 139 live trades paired to paper: up to 10% of ADV
    ~0.25pp, above 10% ~3.7pp. So the rule is: full size if that is <= 10% of
    the day's turnover, else skip. With the CNC floor EQUAL to the slot, any
    name the cap would shrink is under the floor and skips - the shrink path
    is dead by construction.
    """
    slot = float(CFG["setups"]["close_dn_overnight_long"]["capital_allocation"]["margin_per_slot_inr"])
    assert PC["max_participation_pct"] == 0.1
    assert float(PC["min_notional_after_cap_inr_cnc"]) == slot, "CNC floor must equal the slot so nothing is shrunk"
    assert float(PC["min_notional_after_cap_inr"]) >= slot
    assert PC["fallback_to_cnc_below_mtf_floor"] is False, "the fallback only ever produced tiny positions"


def test_tick_gate_configured_and_calibrated():
    thr = PC["max_tick_pct_of_price"]
    assert 0 < thr < 5, f"implausible tick threshold {thr}"
    # Must keep the BULK of what the backtest validated: across the research
    # names the old floor removed, one tick is a median 0.05-0.075% of price
    # with p90 0.16%. The 0.495% max is a single tail case, not the mass.
    assert thr >= 0.16, (
        f"threshold {thr}% cuts below the p90 of research-validated names (0.16%)")
    # And it must leave the edge intact. One tick is a LOWER BOUND on the
    # spread, so a threshold at or above the ~0.35% gross edge admits names
    # whose minimum possible round trip cannot be paid for.
    assert thr < 0.35, (
        f"threshold {thr}% >= the ~0.35% gross edge — admits unpayable names")
    # Blocks the 2026-09-07 MITTAL case (Rs0.92, 0.01 tick = 1.09%)
    assert thr < 100.0 * 0.01 / 0.92


@pytest.mark.parametrize("price,tick,blocked", [
    (0.92, 0.01, True),      # MITTAL    1.09%/tick — unpayable
    (14.61, 0.05, True),     # ASHOKAMET 0.34% — at the edge, blocked
    (15.70, 0.05, True),     # MGEL      0.32% — at the edge, blocked
    (33.66, 0.05, False),    # MALUPAPER 0.15% — clean
    (166.88, 0.05, False),   # UDAYJEW   0.03% — clean
    (0.50, 0.01, True),      # 2%/tick
    (5.00, 0.05, True),      # 1%/tick
])
def test_gate_verdict_on_the_real_names(price, tick, blocked):
    """Every name from the 2026-09-07 session, plus two synthetic edges."""
    thr = PC["max_tick_pct_of_price"]
    assert ((100.0 * tick / price) > thr) is blocked, (
        f"Rs{price}/share with a {tick} tick is {100*tick/price:.2f}%/tick; "
        f"expected blocked={blocked} at threshold {thr}%")


def test_the_2026_09_07_names_under_the_fixed_size_rule():
    """Under the fixed-size rule a name is full size or skipped, never shrunk.
    The 2026-09-07 version of this test asserted the five names be taken at
    Rs2k-9k. Live then showed such positions earn nothing and burn a slot.

    Rs25k vs 20-day ADV: ASHOKAMET 12.8% and MALUPAPER 10.2% are over the 10%
    ceiling -> skipped. UDAYJEW 5.9%, MITTAL 4.4%, MGEL 2.7% are within it ->
    full size as far as the cap is concerned (MITTAL and MGEL are then blocked
    by the tick gate, which runs first)."""
    cap = PC["max_participation_pct"]
    floor = float(PC["min_notional_after_cap_inr_cnc"])
    adv = {"ASHOKAMET": 195_451, "UDAYJEW": 421_347, "MALUPAPER": 245_826,
           "MGEL": 927_282, "MITTAL": 569_932}
    verdict = {n: ("skip" if cap * a < floor else "full") for n, a in adv.items()}
    assert verdict == {"ASHOKAMET": "skip", "MALUPAPER": "skip",
                       "UDAYJEW": "full", "MGEL": "full", "MITTAL": "full"}


# ------------------------------------------------------------------ code ---

def test_tick_gate_precedes_the_cap():
    """Tick cost is a property of the instrument, not of our size.

    Capping to full size cannot rescue a 1.1%-per-tick name, so the check has
    to run before the cap rather than after it.
    """
    i_tick = SRC.index("_max_tick_pct = float(_pc[")
    i_cap = SRC.index('_max_notional = float(_pc["max_participation_pct"])')
    assert i_tick < i_cap, "the tick gate must be evaluated before the cap"


def test_tick_gate_releases_the_slot():
    """A skip must hand the slot to the next candidate, not burn it."""
    i = SRC.index("_max_tick_pct = float(_pc[")
    body = SRC[i:i + 1400]
    assert "_rollback_slot_to_free(slot)" in body
    assert "pool.persist()" in body
    assert 'summary["skipped_count"] += 1' in body
    assert "continue" in body


def test_floor_selects_by_product_not_a_single_constant():
    i = SRC.index("_is_mtf = str(evt.context[")
    body = SRC[i:i + 400]
    assert "min_notional_after_cap_inr_cnc" in body
    assert "min_notional_after_cap_inr" in body


def test_new_config_keys_are_required_not_defaulted():
    """Mandatory rule: a missing config key must fail fast, never silently default."""
    i = SRC.index("_max_tick_pct = float(_pc[")
    assert '_pc["max_tick_pct_of_price"]' in SRC[i:i + 120], (
        "must index, not .get() with a default")
    j = SRC.index("_is_mtf = str(evt.context[")
    body = SRC[j:j + 400]
    assert '.get("min_notional_after_cap_inr' not in body, (
        "product floors must be required keys")


def test_skip_reason_is_diagnosable():
    """The log must say WHY, so the next session is explainable without a rerun."""
    i = SRC.index("_max_tick_pct = float(_pc[")
    body = SRC[i:i + 1000]
    for token in ("one tick", "% of ", "execution cost"):
        assert token in body, f"tick-skip log should record {token!r}"


def test_cost_model_in_the_log_is_the_measured_one():
    """The skip log must quote the MEASURED cost, not a guessed multiple.

    An earlier version printed `2.0 * tick_pct` as "round trip". Across 120 live
    fills scored against their idealized entry/exit references, execution cost
    in bp actually runs 3.9 + 101 x tick% — about ONE tick, because the entry
    crosses half a spread and the exit is an opening-auction print that crosses
    nothing. Doubling it overstated the cost of every borderline name.
    """
    i = SRC.index("_max_tick_pct = float(_pc[")
    body = SRC[i:i + 1400]
    assert "2.0 * _tick_pct" not in body, "the 2x round-trip model was wrong"
    assert "101.0 * _tick_pct" in body, "should use the measured slope"


# ------------------------------------------------- MTF -> CNC fallback ---
#
# 2026-10-01. Live Sep-2026 at Rs25k slots: 14 of 118 paper signals were skipped
# because they were MTF names whose 1%-of-ADV size fell under the Rs25k MTF
# floor. In the paper mirror those 14 returned +2.53% gross with an 86% hit rate,
# against +0.54% for the 81 that were placed. The floor is a statement about
# MTF's fixed pledge cost, not about the signal; the same shares as CNC cost
# 0.22% flat. So a sub-floor MTF name now falls back to CNC at the capped size
# instead of being thrown away. The CNC floor and the tick gate still apply.

from services.execution.overnight_handlers import floor_verdict  # noqa: E402


def _pc(**over):
    base = {"min_qty_after_cap": 1, "min_notional_after_cap_inr": 25000.0,
            "min_notional_after_cap_inr_cnc": 1000.0, "fallback_to_cnc_below_mtf_floor": True}
    base.update(over)
    return base


def test_fallback_flag_is_configured_and_off():
    """Built 2026-10-01, switched off 2026-10-05: see test_fixed_size_rule_2026_10_05."""
    assert PC["fallback_to_cnc_below_mtf_floor"] is False


def test_fallback_flag_is_required_not_defaulted():
    i = SRC.index("def floor_verdict(")
    body = SRC[i:i + 1600]
    assert 'pc["fallback_to_cnc_below_mtf_floor"]' in body
    assert '.get("fallback_to_cnc_below_mtf_floor"' not in body


@pytest.mark.parametrize("is_mtf, qty, notional, expect", [
    (True, 100, 30000.0, "take"),           # MTF above its floor: unchanged
    (True, 100, 12000.0, "fallback_cnc"),   # MTF under Rs25k, above Rs1k: take as CNC
    (True, 3, 800.0, "skip"),               # MTF under the CNC floor too: still skipped
    (False, 100, 12000.0, "take"),          # CNC above its own floor
    (False, 3, 800.0, "skip"),              # CNC under its floor
    (True, 0, 0.0, "skip"),                 # zero shares is never a trade
])
def test_floor_verdict(is_mtf, qty, notional, expect):
    assert floor_verdict(is_mtf, qty, notional, _pc(), cash_reserved_inr=25000.0) == expect


def test_fallback_can_be_switched_off_to_restore_the_skip():
    assert floor_verdict(True, 100, 12000.0, _pc(fallback_to_cnc_below_mtf_floor=False), 25000.0) == "skip"


def test_fallback_is_bounded_by_the_cash_the_slot_reserved():
    """CNC needs the full notional in cash. If the slot reserved less than the
    capped notional (possible if the MTF floor and the slot margin ever diverge),
    the fallback must not place a position the pool cannot fund."""
    assert floor_verdict(True, 100, 12000.0, _pc(), cash_reserved_inr=12000.0) == "fallback_cnc"
    assert floor_verdict(True, 100, 12000.0, _pc(), cash_reserved_inr=11999.0) == "skip"


def test_fallback_call_site_passes_the_slot_cash():
    i = SRC.index("_verdict = floor_verdict(")
    assert "float(slot.margin_inr)" in SRC[i:i + 200]


def test_paper_open_snapshot_is_rewritten_after_placement():
    """The dashboard reads product from this file; a fallback changes product."""
    assert SRC.count("_write_paper_open_snapshot()") >= 3, "define + before loop + after loop"
    i_loop = SRC.index("for rank_i, (symbol, evt, plan) in enumerate(ranked):")
    assert SRC.rindex("_write_paper_open_snapshot()") > i_loop


def test_fallback_rewrites_product_on_both_event_and_slot():
    """The order, the ledger and the EXIT all read product; all three must agree."""
    i = SRC.index('if _verdict == "fallback_cnc":')
    body = SRC[i:i + 1400]
    for token in ('evt.context["product"] = "CNC"', 'evt.context["leverage"] = 1.0',
                  'slot.product = "CNC"', 'slot.leverage = 1.0'):
        assert token in body, f"fallback must set {token}"
    assert "MTF->CNC" in body, "the log must say the product changed"
    assert body.index('evt.context["product"] = "CNC"') < body.index('elif _verdict == "skip":')


def test_fallback_happens_before_the_cap_is_applied():
    """The capped qty is then used for the order, same path as any capped trade."""
    i = SRC.index('if _verdict == "fallback_cnc":')
    j = SRC.index("plan.qty = _capped_qty", i)
    assert j > i
    assert "continue" not in SRC[SRC.index('summary["mtf_to_cnc_fallback"]', i):j].split("elif")[0]
