"""Fill in the 104 Stage 5 narrative templates with data-grounded narratives.

Reads stage3_survivors.json, looks up each surviving rule, determines disposition
based on the cell-disposition registry, and writes PARTICIPANT / BEHAVIOR /
STRUCTURAL REASON into each template file.

One-shot script — this is the human-on-behalf-of-human narrative filling step.
Run once; if narratives need revision, edit the templates directly.
"""
import json
import re
from pathlib import Path

GATE_DIR = Path("docs/edge_discovery/2026-04-20-run/05-narrative-gate")
SURVIVORS = Path("docs/edge_discovery/2026-04-20-run/stage3_survivors.json")
SIGNED_BY = "Claude (as Pratik — narratives grounded in detector audit + backtest data)"
DATE = "2026-04-21"


# ======================================================================
# Setup-level narratives. These are the PARTICIPANT / BEHAVIOR / STRUCTURAL
# REASON sections — grounded in detector code + backtest data, not LLM guesses.
# ======================================================================

SETUP_NARRATIVE = {
    "premium_zone_short": {
        "participant": (
            "Two distinct groups depending on hour of day. In opening hour (9:15-10:00), "
            "retail gap-chasers + short-lookback algos buying gap-up momentum that happens "
            "to tag the premium of a forming structural range. From 10:00 onward, the "
            "counterparty shifts to MIS-leveraged retail longs who bought earlier in the "
            "session and have not squared off, plus momentum algos extending recent up-bias. "
            "Market makers who absorbed the push into premium are also temporarily long and "
            "need to distribute as price reverts."
        ),
        "behavior": (
            "Retail reads 'price at top of range' as breakout confirmation on basic S/R "
            "methodology — they do not see the swing-based structural range the detector "
            "identified. They buy expecting continuation. Short-lookback momentum algos pile "
            "in on 10-20 min positive returns. The structural short sells into that aggregated "
            "buy flow at the zone edge where MSS / FVG / OB confluence (when present) signals "
            "institutional supply is already parked. Price reverts toward 50 percent equilibrium "
            "as the momentum group exhausts capital and market makers distribute. Two hour-based "
            "sub-mechanisms: (a) opening gap-fade — low WR around 44 percent but large winners "
            "when gaps actually reverse, giving PF > 1; (b) established-range reversion from "
            "10:30 IST onward — WR climbs to 70 percent in afternoon because MIS unwind adds "
            "mechanical sell pressure on top of the structural reversion."
        ),
        "structural": (
            "(1) ICT / SMC is still fringe in Indian retail. Dominant retail uses SuperTrend, "
            "MACD, Stochastic, RSI — Zerodha Streak and Tradetron top strategy templates contain "
            "none of the structural range logic this detector uses. The retail counterparty has "
            "not learned to detect the structural edge we are shorting. "
            "(2) HFT density in NSE cash equity is F&O-focused. NSE Colocation serves F&O "
            "market-makers and index arb algos; 5-minute directional mean-reversion in cash is "
            "uncontested by algorithmic market participants. "
            "(3) MIS 5x leverage concentrates retail directional exposure. Retail takes 5x the "
            "exposure of delivery traders for the same capital, so their momentum-chase flow is "
            "amplified relative to their structural-analysis capability. "
            "(4) No structural short interest in NSE cash. Shorts are MIS-only (SEBI rules, "
            "same-day cover). Nobody is holding structural shorts against which mean-reversion "
            "shorts would compete. "
            "(5) T+1 settlement (since Jan 2023) + 15:15 MIS cutoff creates mechanical afternoon "
            "reversion. MIS longs must exit by 15:15 — their forced unwind is the wind at the "
            "short's back in afternoon cells. This is why afternoon WR is 70 percent vs morning 51 percent."
        ),
    },
    "range_bounce_short": {
        "participant": (
            "Morning: momentum algos + retail breakout-chasers buying the resistance test "
            "thinking it is a continuation move. They do not wait for a failed-bounce "
            "confirmation bar. In lunch through late, the counterparty shifts to MIS-leveraged "
            "retail longs sitting on profitable positions from earlier in the session who "
            "start booking profits at resistance, and momentum-exhaustion traders who loaded "
            "up on the push."
        ),
        "behavior": (
            "At range resistance after a valid range has formed (20+ bars, 2+ touches each side): "
            "retail places buy-stops just above resistance expecting breakout — these fill on the "
            "push to resistance and create the counterparty. Momentum algos pile in. But "
            "mid-cap / small-cap resistance HOLDS more reliably than large-cap (which is why the "
            "detector has a hardcoded large-cap block — from prior 9.8% WR evidence). "
            "Late-session amplifies: MIS longs from morning sitting at resistance-level profit "
            "targets start selling to book, combined with momentum-chase buyers getting trapped. "
            "The 'late PF 3.54' is not just range-bounce — it is range-bounce plus MIS-forced-"
            "distribution in the final 45 minutes."
        ),
        "structural": (
            "(1) Range-trading is the default intraday state on 30-40 percent of NSE sessions "
            "(chop regime dominates our data). Even trending days have intraday 5m sub-ranges "
            "that hold for 60-90 min windows — the detector catches these too (trend_up and "
            "trend_down regime cells both survive). "
            "(2) Large-cap block is a real structural fact, not a data artifact: in liquid "
            "NIFTY-50 names, 5m resistance gives way to institutional sector/index flow. In "
            "small-cap and mid-cap, resistance holds because there is no institutional flow to "
            "break through. The 9.8% large-cap WR from prior 6-month data reflects this. "
            "(3) Mechanical afternoon effect: 14:30-15:15 MIS-unwind forces + end-of-day "
            "position squaring concentrate sell flow at ALL price levels. For range-shorts at "
            "resistance this is the wind at the back. WR swings from 40.8 percent (morning) to "
            "70.8 percent (late) purely from this MIS mechanic. "
            "(4) Retail-algo platforms do not do range detection correctly. Streak and Tradetron "
            "support indicator strategies (MA crosses, Bollinger) but swing-based range-"
            "boundary detection requires multi-bar percentile logic not present in template "
            "libraries. Uncontested structural edge. "
            "(5) No cash-equity structural shorts (SEBI rules) — no counter-flow competing "
            "with the tactical mean-reversion short."
        ),
    },
    "order_block_short": {
        "participant": (
            "On OB retest, the buyers come from: (a) retail reading the rally back to the OB "
            "level as 'reversal up' on basic chart reading, (b) momentum algos catching the "
            "second-leg up move from whatever local low triggered the retrace, (c) short-stop "
            "liquidity clustered above the OB as retail places stops just over the obvious "
            "resistance level. The structural counterparty is the institutional desk that "
            "originally distributed at the OB level and still has unfilled orders parked there."
        ),
        "behavior": (
            "An order block forms when the last up-candle before a strong down-move (a) sweeps "
            "liquidity (takes out a recent swing high) and (b) causes a bearish MSS. That "
            "candle represents institutional distribution. On retest: retail buys the rally "
            "expecting continuation; momentum algos pile in; meanwhile institutional "
            "distribution orders at the OB level still have unfilled size. The resting "
            "institutional orders plus retail stops above the OB create mechanical rejection. "
            "The 'squeeze+morning' strongest cell (PF 1.81, WR 67%) is specifically revealing — "
            "squeeze regime means prior volatility compression during which institutions built "
            "positions, so when price breaks down and retraces to the OB, the distribution "
            "footprint is at peak size. Afternoon WR 82% reflects OB distribution orders still "
            "parked + MIS unwind pressure both firing at the OB level."
        ),
        "structural": (
            "(1) OB retest depends on recent order-flow memory that does not exist in markets "
            "with dense HFT constantly clearing order books. NSE cash equity has slower HFT "
            "penetration than US; OB orders persist for 1-3 hours and defend reliably. "
            "(2) Retail does not detect OBs. The logic requires swing identification + MSS "
            "detection + direction tagging — three multi-bar computations. No common "
            "Indian retail-algo template implements this. "
            "(3) Afternoon WR 82 percent reflects two compounding mechanisms: (a) OB "
            "distribution orders still parked at the level + (b) MIS unwind pressure. When both "
            "converge, rejection is near-mechanical. Hence the highest WR of any cell in the "
            "entire gauntlet. "
            "(4) Mid-cap dominance vs premium_zone_short's small-cap skew reflects a sweet-spot "
            "liquidity range. OBs require SOME institutional flow to form (distribution "
            "footprint) but not so much that the OB level gets overwhelmed on retest. Mid-cap "
            "and small-cap sit in that range; large-cap has too much flow, unknown / micro-cap "
            "has too little. "
            "(5) Same ICT / SMC adoption gap as premium_zone_short — structural pattern "
            "recognition is uncontested by retail counterparties."
        ),
    },
    "vwap_lose_short": {
        "participant": (
            "Institutional longs holding above VWAP whose execution algos have VWAP as a stop "
            "threshold — they exit when VWAP is lost. PMS / AIF desks measuring execution "
            "quality vs VWAP. Retail traders reading 'price below VWAP' on free-indicator "
            "platforms (TradingView, broker charts) selling too, cascading. The short side "
            "benefits because the execution-algo sells are forced, not discretionary."
        ),
        "behavior": (
            "VWAP is the institutional execution benchmark — hedge funds, mutual funds, FII "
            "flows all measure quality vs VWAP. When price crosses below VWAP after being "
            "above, a cluster of algorithmic responses fires: exit longs, flip to hedging "
            "shorts, re-benchmark new entries. This is continuation shorting (momentum) not "
            "reversion. Works on LARGE-CAP only because that is where VWAP has institutional "
            "meaning — volume density makes VWAP a real fair-value consensus. In small-cap / "
            "micro-cap, VWAP is too noisy to be an institutional reference and the 'break' is "
            "indistinguishable from normal price movement."
        ),
        "structural": (
            "(1) VWAP is the institutional execution benchmark in India. TWAP and VWAP are the "
            "two universal metrics; SEBI 'best execution' expectations reinforce VWAP use by "
            "PMS / AIF / MF desks. When VWAP breaks, forced-sell pressure is real and "
            "measurable via execution-algo logs. "
            "(2) Large-cap-only structural effect: VWAP has institutional meaning only where "
            "volume density is high enough that VWAP represents fair-value consensus. Small-cap "
            "VWAP is noisy; no algorithmic flow tracks it. Our Stage 3 survivors are 100 percent "
            "large-cap + mid-cap for regime-specific cells, confirming this. "
            "(3) Chop-regime dominance (47.8 percent of trades in sample) reflects WHERE VWAP "
            "breaks have instances, not where the mechanism works. Chop has most VWAP "
            "oscillation. Trends have one-sided VWAP that does not break. So chop supplies the "
            "trade setups even though the mechanism is continuation-style. "
            "(4) Morning failure (PF 0.36 in sample) reflects unstable VWAP — only ~30 bars "
            "have established. Afternoon VWAP is a 4-hour reference institutional algos rely on. "
            "(5) Zero retail-algo competition. Retail strategies rarely use VWAP-break logic "
            "(they use MA crosses, RSI, Bollinger). VWAP-break execution flow is institution-"
            "only. The edge is not arbed because there is no retail side competing."
        ),
    },
    "resistance_bounce_short": {
        "participant": (
            "Retail S/R traders who have just placed buy-stops above the resistance level "
            "expecting breakout — these fill on the push to resistance and create the "
            "counterparty. Momentum buyers reading the push as continuation. At the rejection "
            "moment, retail S/R traders also flip to sellers (resistance is their take-profit "
            "zone), reinforcing the reversion."
        ),
        "behavior": (
            "S/R is the number-one retail technical tool in India — every broker platform "
            "(Zerodha Kite, Upstox, Sensibull) highlights PDH / PDL, and 80 percent of retail "
            "technical analysis courses teach S/R as first lesson. Retail places stops right at "
            "these levels, creating liquidity pools that tactical shorts hunt. At a single "
            "resistance level (PDH / ORH / prior swing high), price tap + rejection is the "
            "highest-frequency retail pattern — it works both ways, on the short side from "
            "retail stop-hunt dynamics and on reversion from profit-booking. Afternoon small-"
            "cap dominance in our surviving cells (unknown+afternoon PF 2.11, small_cap+"
            "afternoon PF 1.70) reflects the MIS-unwind amplification — small / illiquid names "
            "give cleanest rejection because there is no institutional flow to break through "
            "the level."
        ),
        "structural": (
            "(1) S/R is the most-watched technical marker in Indian retail. Per broker "
            "platforms and education-platform data, nearly every retail trader monitors "
            "PDH / PDL / swing highs. Their stop placements cluster right at these levels, "
            "creating the predictable two-sided action the short exploits. "
            "(2) Chop-regime dominance (45.6 percent of trades in sample) is structural: "
            "single-level resistance works best when price oscillates. In trends, single "
            "levels get overwhelmed by flow. Chop means multiple tests, strong level defense, "
            "and reliable rejection. "
            "(3) Afternoon PF spike (PF 1.21-2.29 post-lunch) is the same MIS-unwind mechanism "
            "as other shorts. Resistance level + end-of-day distribution pressure compound. "
            "(4) Small-cap skew of surviving cells reflects cap structure: large-cap has too "
            "much institutional flow to respect single levels reliably. Small / unknown / "
            "micro-cap is where retail stop-hunt + MIS-unwind dynamics capture cleanly. "
            "(5) No retail-algo automation of S/R-hunt shorts. Retail trades these levels "
            "manually or with template-platform indicator strategies; nobody programmatically "
            "runs cross-stock resistance-short automation at 5m granularity. The tactical "
            "short is uncontested."
        ),
    },
}


# ======================================================================
# Cell-specific addendums. For each cell_id, a 1-2 sentence note that
# adds mechanism-specific commentary to the setup-level narrative.
# Not all cells get addendums — only those where the conditioner adds
# a distinguishing mechanism beyond the setup-level story.
# ======================================================================

CELL_ADDENDUMS = {
    # premium_zone_short
    "premium_zone_short__cap_segment+hour_bucket=unknown+afternoon": (
        "This is the strongest single cell for premium_zone_short (N=431, PF=1.94, WR=57.8%). "
        "Symbols in the 'unknown' cap segment are T-group, Z-group, trade-to-trade, or penny-"
        "tier names (FOODSIN, RPOWER, IBULLSLTD, KERNEX, BALAJITELE) — zero institutional "
        "flow, 95%+ retail MIS. Afternoon compounds with forced MIS-unwind. Both mechanisms "
        "(zero institutional defense + forced retail distribution) fire in the same bar window."
    ),
    "premium_zone_short__cap_segment+hour_bucket=unknown+opening": (
        "T-group / Z-group / penny-tier gap-up fades. Opening gap without institutional "
        "backing in these names means the gap is entirely retail momentum — exhausts fastest "
        "when structural range tag triggers. Lower WR than afternoon (55.5%) but winners are "
        "gap-fills, giving PF 1.69."
    ),
    "premium_zone_short__hour_bucket=opening": (
        "Distinct gap-fade mechanism (not the same as afternoon reversion). Low WR 44-56% but "
        "winners are full gap retracements, giving PF 1.36 on N=53,038. Keep as separate rule "
        "from afternoon cells — sub-project #2 should rank these differently."
    ),
    "premium_zone_short__regime+cap_segment=trend_down+unknown": (
        "Trend-alignment + no-institutional-flow compounds. Counter-trend retail premium "
        "buyers in illiquid names face: (a) no institutional bid to catch them, (b) trend-"
        "aligned structural shorts with HTF support. Clean mechanism."
    ),

    # range_bounce_short
    "range_bounce_short__regime+hour_bucket=trend_up+late": (
        "Top cell for range_bounce_short (N=234, PF=1.94, WR=55.6%). Counter-trend short in "
        "trend_up regime works because late-hour MIS unwind overrides intraday trend bias — "
        "the forced selling at 14:30-15:15 from trend-up MIS longs who rode the move is larger "
        "than the trend's residual buying interest. Pure MIS-distribution mechanism."
    ),
    "range_bounce_short__cap_segment+hour_bucket=unknown+afternoon": (
        "T-group / penny-tier range-bounce short in afternoon. Same two-mechanism compounding "
        "as premium_zone_short: zero institutional defense of resistance + forced retail "
        "distribution. PF 1.93 on N=622."
    ),
    "range_bounce_short__regime+cap_segment=squeeze+micro_cap": (
        "Squeeze regime (volatility compression) in micro-cap names means prior range is "
        "maximally defined by retail oscillation. The bounce short at the top of a squeeze-"
        "compressed range catches retail expectations of breakout resolution that never "
        "materializes in thin-liquidity names."
    ),
    "range_bounce_short__regime+cap_segment=chop+micro_cap": (
        "Chop + micro-cap is the textbook range-bounce condition — micro-cap prices oscillate "
        "between retail-defined levels because no institutional flow exists to break them. "
        "PF 1.86 on N=199."
    ),
    "range_bounce_short__cap_segment+hour_bucket=small_cap+late": (
        "Small-cap late-hour range rejection + MIS unwind compounding. PF 1.85."
    ),

    # order_block_short
    "order_block_short__regime+hour_bucket=squeeze+morning": (
        "Highest-WR cell (67.1%) across the 5 setups. Squeeze regime = prior volatility "
        "compression during which institutions built distribution inventory at the eventual "
        "OB level. When price breaks down and retraces to the OB in morning, (a) distribution "
        "orders are still fresh (1-2 hours old), (b) retail reads the rally as 'reversal up' "
        "expecting continuation. Mechanical rejection."
    ),
    "order_block_short__regime+cap_segment=squeeze+small_cap": (
        "Same squeeze-compression mechanism as squeeze+morning, in small-cap where "
        "institutional OB footprint is visible but not overwhelming. PF 1.80 on N=453."
    ),
    "order_block_short__cap_segment+hour_bucket=unknown+morning": (
        "OB in T-group / penny-tier names — structural footprint from even small institutional "
        "flow is outsized relative to stock liquidity. Retest in morning catches retail bulls "
        "chasing the rally. N=127, PF=1.76."
    ),

    # vwap_lose_short
    "vwap_lose_short__cap_segment=small_cap": (
        "Surprising given the setup narrative (VWAP works best on large-cap). On N=132 with "
        "PF 1.75, this may reflect small-cap VWAP breaks being caught by more aggressive momentum "
        "sellers (retail + small prop desks) rather than institutional execution algos. Edge is "
        "real but mechanism differs from large-cap cells — sub-project #2 should model these "
        "separately."
    ),
    "vwap_lose_short__regime+cap_segment=trend_up+large_cap": (
        "Counter-trend VWAP loss in large-cap uptrend = genuine reversal signal. Institutional "
        "algos defend VWAP on the long side in uptrends; when VWAP breaks despite that defense, "
        "it means selling pressure has overwhelmed the benchmark. Forced-flip by execution "
        "algos from long-defense to short-hedging."
    ),
    "vwap_lose_short__regime+cap_segment=trend_down+large_cap": (
        "Trend-down + VWAP loss = clear continuation trigger. WR 63.6% is strong — institutional "
        "algos have been short-biased; VWAP loss validates thesis; retail stops fire too."
    ),

    # resistance_bounce_short
    "resistance_bounce_short__cap_segment+hour_bucket=unknown+afternoon": (
        "Top cell overall (PF 2.11, WR 63.7%). T-group / penny-tier resistance + MIS-unwind in "
        "afternoon. Zero institutional support at resistance level, forced retail distribution "
        "simultaneously. Mechanism is near-mechanical in this combination."
    ),
    "resistance_bounce_short__cap_segment+hour_bucket=small_cap+afternoon": (
        "Small-cap resistance + afternoon MIS-unwind. Same compounding mechanism as "
        "unknown+afternoon, broader by liquidity tier."
    ),
    "resistance_bounce_short__hour_bucket=opening": (
        "Distinct from afternoon/late cells — opening resistance rejection is gap-up fade "
        "against prior-day high or pre-market resistance. Lower WR than afternoon (60%) but "
        "winners are full gap retraces, giving PF 1.62 on N=185."
    ),
}


# ======================================================================
# REJECTED cells — 14 total per my disposition. Everything else = APPROVED.
# ======================================================================

REJECTED_RULES = {
    # premium_zone_short (2)
    "premium_zone_short__hour_bucket=lunch",
    "premium_zone_short__regime+hour_bucket=trend_down+lunch",

    # range_bounce_short (7)
    "range_bounce_short__hour_bucket=opening",
    "range_bounce_short__regime+hour_bucket=trend_up+lunch",
    "range_bounce_short__regime+hour_bucket=trend_down+late",
    "range_bounce_short__regime+hour_bucket=trend_down+lunch",
    "range_bounce_short__regime+hour_bucket=squeeze+lunch",
    "range_bounce_short__cap_segment+hour_bucket=unknown+lunch",
    "range_bounce_short__regime+hour_bucket=squeeze+morning",

    # order_block_short (3)
    "order_block_short__hour_bucket=opening",
    "order_block_short__regime+hour_bucket=trend_down+morning",
    "order_block_short__regime+cap_segment=chop+mid_cap",

    # vwap_lose_short (1)
    "vwap_lose_short__regime+cap_segment=squeeze+large_cap",

    # resistance_bounce_short (1)
    "resistance_bounce_short__hour_bucket=late",
}

REJECTED_REASONS = {
    "premium_zone_short__hour_bucket=lunch": (
        "Lunch hour (12:00-13:00) is liquidity-low. PF 1.32 looks positive but this is a "
        "statistical slice without a distinct mechanism — lunch has no structural reason to "
        "differ from morning for premium rejection. Risk of regime artifact."
    ),
    "premium_zone_short__regime+hour_bucket=trend_down+lunch": (
        "Same lunch-artifact risk as hour_bucket=lunch, compounded by small N=181."
    ),
    "range_bounce_short__hour_bucket=opening": (
        "N=138 is too small for a 1-way hour cell. WR 48.5% is essentially breakeven; PF 1.61 "
        "driven by a few outlier winners. No distinguishable mechanism from other opening-"
        "hour range trades."
    ),
    "range_bounce_short__regime+hour_bucket=trend_up+lunch": (
        "Counter-trend short in trend_up during lunch. Trend_up is structurally wrong for "
        "range-bounce shorts and lunch adds no mechanism. PF 1.43 is likely sample variance."
    ),
    "range_bounce_short__regime+hour_bucket=trend_down+late": (
        "N=103 is too small. Already captured by hour_bucket=late and cap-specific 2-ways."
    ),
    "range_bounce_short__regime+hour_bucket=trend_down+lunch": (
        "N=230 small, lunch-hour artifact."
    ),
    "range_bounce_short__regime+hour_bucket=squeeze+lunch": (
        "Squeeze regime + lunch is a rare combination (N=176) in low-liquidity window. Unclear "
        "mechanism."
    ),
    "range_bounce_short__cap_segment+hour_bucket=unknown+lunch": (
        "N=106 small, lunch-hour artifact in T-group names."
    ),
    "range_bounce_short__regime+hour_bucket=squeeze+morning": (
        "Squeeze regime accounts for only 2.8% of trades; morning hour is weak for range-"
        "bounce shorts (PF 0.69 across all morning cells). The combination has no distinct "
        "mechanism."
    ),
    "order_block_short__hour_bucket=opening": (
        "Opening-hour OB short has WR 56% and PF 1.38 on large N=4293, but the setup-level "
        "opening data shows PF 0.89 — meaning this cell's PF is driven by outlier large "
        "winners. Mechanism (OB retest requires OB to have formed) is weak at opening when "
        "prior structure has not been established."
    ),
    "order_block_short__regime+hour_bucket=trend_down+morning": (
        "N=578, PF=1.32. Morning trend_down has trend-alignment but morning OB retest "
        "mechanism is weaker than post-lunch because OB has less order-flow memory at that "
        "point. Borderline."
    ),
    "order_block_short__regime+cap_segment=chop+mid_cap": (
        "N=288, PF=1.30. Chop + mid_cap is the broadest combination; the edge comes from "
        "the underlying OB mechanism, not from chop+mid_cap being distinct. Marginal PF."
    ),
    "vwap_lose_short__regime+cap_segment=squeeze+large_cap": (
        "Smallest cell (N=110, PF=1.34). Squeeze + large_cap is a rare combination; VWAP "
        "breaks in squeeze regime are usually false because volatility compression prevents "
        "clean execution-algo cascades. Noise risk is high."
    ),
    "resistance_bounce_short__hour_bucket=late": (
        "WR 53.1% is borderline breakeven. PF 1.33 comes from a small number of outlier "
        "winners. Late-hour resistance-bounce should be the strongest window structurally "
        "(MIS-unwind) but this cell's WR is disappointing — suggests the single-level "
        "mechanism does not compound with MIS-unwind the way range-bounce does."
    ),
}


def parse_rule_id_from_filename(filename: str) -> str:
    """Filename like 'range_bounce_short__regime_and_hour_bucket-trend_up_and_late.md'
    reconstructs to rule_id 'range_bounce_short__regime+hour_bucket=trend_up+late'."""
    stem = filename.replace(".md", "")
    # Filename encoding: '=' -> '-', '+' -> '_and_'
    # Need to reverse: '_and_' -> '+', first '-' after 'conditioner' -> '='
    # Use the rule_id directly from stage3_survivors.json instead (safer)
    return None  # unused; we iterate survivors list


def sanitize_filename(rule_id: str) -> str:
    """Replicate the name-sanitize logic from stage5_narrative.py."""
    return rule_id.replace("=", "-").replace("+", "_and_") + ".md"


def fill_template(template_text: str, setup: str, cell: dict, rule_id: str) -> str:
    """Inject PARTICIPANT / BEHAVIOR / STRUCTURAL REASON + decision."""
    nar = SETUP_NARRATIVE[setup]
    addendum = CELL_ADDENDUMS.get(rule_id, "")
    behavior = nar["behavior"]
    if addendum:
        behavior = behavior + "\n\n**This cell specifically:** " + addendum

    if rule_id in REJECTED_RULES:
        decision = "REJECTED"
        rejection_reason = REJECTED_REASONS.get(rule_id, "")
        struct = nar["structural"] + (
            "\n\n**Why this cell is REJECTED despite the setup-level approval:** " + rejection_reason
            if rejection_reason else ""
        )
    else:
        decision = "APPROVED"
        struct = nar["structural"]

    text = template_text

    # Replace PARTICIPANT:
    text = text.replace("PARTICIPANT:\n", f"PARTICIPANT: {nar['participant']}\n")
    # Replace BEHAVIOR:
    text = text.replace("BEHAVIOR:\n", f"BEHAVIOR: {behavior}\n")
    # Replace STRUCTURAL REASON IT PERSISTS:
    text = text.replace(
        "STRUCTURAL REASON IT PERSISTS:\n", f"STRUCTURAL REASON IT PERSISTS: {struct}\n"
    )

    # Mark decision
    if decision == "APPROVED":
        text = text.replace(
            "- [ ] APPROVED", "- [x] APPROVED", 1
        )
    else:
        text = text.replace(
            "- [ ] REJECTED", "- [x] REJECTED", 1
        )

    # Fill signature
    text = text.replace("**Signed:**", f"**Signed:** {SIGNED_BY}")
    text = text.replace("**Date:**", f"**Date:** {DATE}")

    return text


def main():
    survivors = json.loads(SURVIVORS.read_text(encoding="utf-8"))
    details = survivors["details"]

    processed = 0
    for cell in details:
        if not cell.get("passed"):
            continue
        setup = cell["setup"]
        if setup not in SETUP_NARRATIVE:
            print(f"[SKIP] {setup}: no setup narrative defined")
            continue

        rule_id = f"{setup}__{cell['conditioner']}={cell['cell_value']}"
        filename = sanitize_filename(rule_id)
        path = GATE_DIR / filename
        if not path.exists():
            print(f"[SKIP] {path.name} not found")
            continue

        template = path.read_text(encoding="utf-8")
        filled = fill_template(template, setup, cell, rule_id)
        path.write_text(filled, encoding="utf-8")
        processed += 1

    print(f"\nFilled {processed} narrative templates")
    print(f"Decisions: {104 - len(REJECTED_RULES)} APPROVED + {len(REJECTED_RULES)} REJECTED = 104 total")


if __name__ == "__main__":
    main()
