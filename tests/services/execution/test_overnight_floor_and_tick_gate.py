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
    assert PC["min_notional_after_cap_inr_cnc"] < PC["min_notional_after_cap_inr"], (
        "CNC has no fixed cost component, so its floor must be lower than MTF's")


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


def test_cnc_floor_no_longer_does_the_filtering():
    """The notional floor must stop being the gate; the tick gate is.

    All five of the 2026-09-07 names clear the CNC notional floor now. Three are
    then blocked on tick cost, which is the property that actually matters.
    """
    floor = PC["min_notional_after_cap_inr_cnc"]
    capped = {"ASHOKAMET": 1943, "UDAYJEW": 4172, "MALUPAPER": 2457,
              "MGEL": 9263, "MITTAL": 5698}
    removed = {s for s, n in capped.items() if n < floor}
    assert not removed, (
        f"CNC floor Rs{floor} still removes {removed} — notional should no "
        "longer be the deciding variable")


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
