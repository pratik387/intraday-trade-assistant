# §3.3 Brief: `oi_long_buildup_continuation` — **DEFERRED (2026-05-05)**

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DEFERRED 2026-05-05 — failed data-feasibility gate.** Brief retained; do not implement until intraday F&O OI capture is built.

## Why deferred (2026-05-05)

User approved this brief for sanity-check coding. Before writing the
sanity tool, I verified what F&O OI data exists on disk:

- 5m enriched feather: OHLCV + indicators only — **no OI column**
- `data/option_chain/<YYYY>/<MM>/<YYYY-MM-DD>.parquet`: EOD per-strike
  OPTION OI (not stock-future OI); daily snapshot, not intraday
- F&O bhavcopy: EOD daily snapshot per contract — cannot interpolate
  intraday velocity from a single-point-per-day snapshot
- **No intraday minute-granular F&O OI tick data on disk anywhere**
- NSE OI-Spurts feed is live-only; no historical archive

The brief's three-leg filter requires 30-min intraday stock-future OI
velocity. We have NEITHER stock-future OI NOR minute-granular OI ticks.

The brief's "Data engineering plan" section line "F&O bhavcopy +
interpolates intraday OI from the bhavcopy snapshot points" was
incorrect speculation — F&O bhavcopy is single-snapshot per day, not
interpolatable to intraday. This was caught before any sanity-tool
code was written, but only after the brief was approved.

Lesson captured at `tasks/lessons.md` 2026-05-05 second entry: the
precedent-feasibility gate (lesson 2026-05-05 first entry) needs a
companion data-feasibility gate. Both must pass BEFORE drafting a brief.

The brief is retained for reference IF/WHEN intraday F&O OI capture is
built (separate data-engineering project, ~1-2 weeks). Until then, do
NOT promote this back into the §3.3 queue.

---

**Original brief below (kept for reference):**

**Date:** 2026-05-05
**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Candidate 3 SHORTLIST)
- tasks/lessons.md 2026-05-05 (feasibility-precedent gate added)

This is the **FOURTH §3.3 brief** in sub-9 and the FIRST drafted under the new two-gate process (precedent first, peer-reviewed second). Round 2 (2026-05-05) shortlisted Candidate 3 (F&O OI-velocity intraday directional buildup) as the single passing candidate after both gates.

---

## Asymmetry

**Name:** Indian-equity intraday F&O long-buildup continuation in mid-cap F&O 200 stocks.

**Indian-specific source:**
- NSE publishes per-stock futures OI in real-time during continuous trading via the **OI-Spurts feed** (https://www.nseindia.com/market-data/oi-spurts) — most exchanges don't expose this granularly.
- F&O participation in India is **88-93% retail** (SEBI FY23 study) — retail-driven OI flows are predominantly contrarian-noise, allowing institutional positioning to be identified against the retail floor.
- The four-quadrant OI × price classification (long buildup / short buildup / long unwinding / short covering) is the canonical Indian retail-algo framework — published in Zerodha Varsity, uTrade Algos, StockEdge, Combiz, Tradejini, Jainam.

**The exploitable asymmetry:**
- **Long buildup quadrant** (↑Price + ↑OI): new long contracts are being written WITH price moving up. In Indian mid-cap F&O 200 stocks, when this happens in the **post-lunch institutional window (13:00-13:30 IST)** with **price + volume confirmation**, it signals informed accumulation that drives end-of-day and next-session continuation.
- The published asymmetry (Avinash & Mallikarjunappa Sage 2020, meta-analytic Indian-market): "open interest and transaction volume are more informative in an emerging market compared to a developed market." The signal-to-noise on Indian F&O OI is structurally higher than US/Europe because retail noise is higher and institutional participation is more identifiable.

**Direction:** LONG-only. Single-quadrant (long buildup), single-window (post-lunch 13:00-13:30 entry), single-direction (continuation long, exit by EOD). Other quadrants (short buildup, long unwinding, short covering) are deferred to subsequent briefs IF this one validates.

## Participants

- **Long-buildup writers (post-lunch 13:00-13:30)**: institutional desks initiating directional positions ahead of EOD/next-day move. Identifiable by: OI velocity ≥ X% over a 30-min window + simultaneous price uptick + volume support.
- **Counter-side**: retail short positions (the trapped 88-93%) + market-makers providing liquidity. Their forced rebalance / stop-loss flow is the upside fuel as the long-buildup matures.
- **Late-fade arbitrageurs**: HFT desks fading stretched intraday positions toward 15:15-15:25 close. We exit BEFORE this window (15:10) to avoid being the late-arb's fuel.

We're on the disciplined side of an attention-and-positioning-driven asymmetry: **trading WITH the informed long-buildup, BEFORE the late-fade arbitrage window**.

## Persistence

Three structural reasons:
1. **Retail F&O participation is REGULATORILY ENTRENCHED** — SEBI FY23 study documented 70% / 93% loss rates for cash / F&O retail. SEBI has imposed margin rule changes (Jan 2025 +) but retail participation has continued to grow per Sept 2025 SEBI data. The retail-noise floor is structural, not cyclical.
2. **NSE OI-Spurts feed is institutionally PUBLIC** — institutions know retail can see OI changes in real-time. This means the institutional signal is partly OBFUSCATED via fakery (the literature flags this, see "Falsification" below). But institutions still need to position before EOD/next-day moves; the physics of order book + tick rule means even obfuscated signals leak through volume + 2nd-leg OI confirmation.
3. **Indian F&O 200 mid-cap segment is the SWEET SPOT**:
   - Large-cap (NIFTY 50): too liquid, institutional positioning is hidden in normal flow, retail can't trap institutions
   - Mid-cap F&O (NIFTY 100-200 names): enough liquidity for institutions to size, less retail noise per stock, OI signal is identifiable
   - Small-cap (non-F&O): no F&O OI signal at all

These are SEBI/NSE regulation + structural-participation factors. Not market-cycle-dependent.

## Project-wide long-bias caution (acknowledge upfront)

Sub7/sub8 retired 11 long-side detectors after Phase-1 validation failure. The recurring lesson was **"long-bias setups in Indian intraday systematically lose"** (project pattern + SEBI FY23 evidence).

**Why this setup is STRUCTURALLY DIFFERENT** from the failed long-side detectors:
- **Failed detectors** (orb_15, pdh_pdl_reject long, gap_and_go_continuation, ema5_alert_pullback long, camarilla_l3_reversal long, etc.): all were **technical-pattern-based** — they identified retail-readable chart patterns (breakout above ORH, retest of PDH, gap-and-go continuation, EMA pullback, Camarilla level reversal). Indian retail dominantly LONGs these patterns; the patterns are arbitraged out by the time retail can act.
- **This setup** is **positioning-flow-based** — it identifies institutional informed flow via the OI velocity signal, not via a retail-readable chart pattern. The signal source is the F&O OI feed, not the price chart. The participants on the OPPOSITE side (the trapped retail shorts + late market-makers) are not the same participants who lose on technical patterns.

**Setup-design constraint** (mandatory per literature caution): we MUST specify confirming filters that institutionally-led OI buildup passes but pattern-fakery does NOT. Three-leg filter (specified in Mechanic below): price-confirmation + volume-confirmation + second-leg-OI-confirmation. Naive entry on any single OI spurt will lose to institutional fakery (per uTrade/Tradejini caution).

If the sanity-check still produces PF < 1.10 with the three-leg filter, we accept that the long-bias finding extends even to positioning-flow-based setups, and the candidate retires decisively.

## Evidence (peer-reviewed, independent of retail communities)

1. **Avinash & Mallikarjunappa, *Asia-Pacific Journal of Management Research and Innovation* (Sage), 2020** — *Informational Role of Open Interest and Transaction Volume of Options: A Meta-Analytic Review* — meta-analysis confirming OI + transaction volume as predictors of future stock prices, future volatility, announcement-day behaviour. Indian-market focus. Direct quote: "open interest and transaction volume are more informative in an emerging market compared to a developed market." URL: https://journals.sagepub.com/doi/abs/10.1177/2319714520980662

2. **Springer 2024** — *Option Volume and Open Interest for Predicting Underlying Return — A Study of Index Option in Indian Stock Market* — peer-reviewed Indian-market evidence that index-option OI + volume predicts underlying returns. URL: https://link.springer.com/chapter/10.1007/978-981-97-6242-2_6

Both sources are 2018+ recency-passing per the round-2 protocol. ≥1 peer-reviewed evidence requirement met by 2 sources.

**Counter-evidence to address:** Greenwood & Sammon NBER w30748 (used in round-1 G retirement) shows microstructure asymmetries decay as algo participation grows. Indian F&O retail participation is GROWING (per SEBI 2025 data); this could mean the asymmetry is also growing (more retail noise = more identifiable institutional signal). Or it could mean the institutions get better at obfuscation. The sanity-check on 2024 data establishes whether the asymmetry is currently tradable NET of fees AND of institutional fakery.

## Direction

**LONG-ONLY.** Single quadrant (long buildup), single direction (continuation), single window (post-lunch 13:00-13:30 entry).

We do **NOT** include:
- **Short buildup → fade-long-squeeze** (the round-2 spec mentioned this as a candidate). Reason: countertrend with long-bias on a setup whose own thesis is short-trapped retail. Two layers of "retail loses" theorising stack up. Defer to a later brief if long-buildup validates.
- **Long unwinding → continuation short** or **short covering → continuation short**. These are genuine short-side opportunities but mechanically distinct (different OI direction, different time-of-day signature). Defer to later briefs.
- **Short-side mid-cap entries during the same window**. Unrelated to OI quadrant.

This is a focused, single-quadrant brief. If sanity passes, the OTHER three quadrants get separate briefs based on this brief's structural template.

## Mechanic

**Setup name:** `oi_long_buildup_continuation`
**Side:** Long-only.

**Sequence:**

1. **Universe filter (per scan)**:
   - F&O 200 eligible stocks (NSE F&O list, mid-cap segment per NSE classification)
   - Excluded: NIFTY 50 (institutional positioning hidden in noise), micro_cap (no F&O eligibility anyway)
   - Liquidity gate: T-1 ADV ≥ ₹50 cr cash-equity volume (ensures intraday tradeability + sufficient F&O liquidity)
   - Approximate count: ~120-150 stocks per scan

2. **Active window**: 13:00-13:30 IST (post-lunch institutional positioning window).
   - Entry decision is at the **13:30 5m bar's close** (single-bar timing, like circuit_t1's 10:30 entry).
   - Why post-lunch: literature + retail-algo precedent (uTrade, Combiz) consistently identify 13:00-14:00 as the "institutional positioning window" — pre-lunch is retail-momentum noise; post-15:00 is late-arb fade.

3. **Three-leg confirming filter (mandatory, naive entry will lose)**:
   - **Leg 1 — OI buildup**: stock-future OI increased ≥ +1.5% from 13:00 to 13:30 (30-min window).
     - OI source: NSE OI-Spurts feed at 5-min granularity, OR per-stock-future minute OI snapshots from the F&O bhavcopy feed.
   - **Leg 2 — Price + volume confirmation**: from 13:00 to 13:30, price has moved UP (latest close > 13:00 open) AND volume has increased (cumulative 13:00-13:30 volume ≥ 1.0× the 13:00-13:30 average over the prior 5 sessions).
   - **Leg 3 — Second-leg OI confirmation**: at 13:30, the LATEST 5-min OI tick must be HIGHER than the 13:25 OI tick (i.e., OI is still accelerating, not plateauing). This rejects the "fake-OI-pump-then-reverse" pattern that uTrade/Tradejini explicitly warn about.

4. **Entry**:
   - **Entry timing**: 13:30 IST 5m bar's close.
   - **Entry price**: 13:30 5m bar's close.
   - **Direction**: LONG.
   - **Entry zone**: ±0.1% around the close (matches existing detectors' entry zone pattern).

5. **Stop-loss**:
   - **Hard SL**: max(entry × (1 - 1.0%), 13:00-13:30 window low). Defends against the "fake-OI-pump-then-reverse" tail risk that even leg-3 filter doesn't fully eliminate.
   - **Min stop distance**: 1.0% of entry (qty-inflation guard).

6. **Targets**:
   - **T1** (50% qty): entry × (1 + 1.0%) — first 1% intraday move up.
   - **T2** (50% qty): entry × (1 + 2.0%) — second 1% move (target full intraday institutional accumulation move).
   - **Time stop**: 15:10 IST (5 min before MIS auto-square; well before the 15:15-15:25 late-arb fade window).

7. **Latch**: one fire per (symbol, T+0) — no re-entry same session.

**target_anchor_type**: `r_multiple` — T1/T2 are arithmetic R-multiples. (Different from circuit_t1's structural gap-edges; OI-driven setups don't have a clean structural anchor that translates to a price level.)

## Universe

**Intended universe**: NSE F&O 200 mid-cap segment (NIFTY 100-200 by mcap, F&O eligible).
- **Cap segment** filter: mid-cap only.
- **Excluded**: NIFTY 50 (institutional flow not identifiable), small-cap non-F&O (no OI signal), micro_cap.
- **Liquidity gate**: T-1 ADV ≥ ₹50 cr.
- **Approximate symbol count after cap + liquidity filter**: ~120-150 stocks.

**Sample-size feasibility:**
- Per round-2 spec: ~5-15 high-conviction long-buildup events / day across F&O 200, ~1,200-3,500/year.
- Three-leg filter compounds the count down: assume 30-50% pass-through rate (price + volume + 2nd-leg OI confirmation).
- Conservative estimate after all filters: ~360-1,750 events / year. n ≥ 500 over 1 year is plausible but tight — the sanity-check will quantify.

## Active window

**Setup formation**: T+0 13:00-13:30 IST window — OI velocity computation requires 30-min window.
**Entry**: T+0 13:30 IST single-bar entry.
**Hold horizon**: 13:30 → 15:10 IST = 1h 40m intraday MIS.

**Why 13:30 entry (not earlier or later):**
- 09:15-12:00 = pre-lunch retail momentum noise; OI changes here are retail-driven, not institutional
- 12:00-13:00 = lunch lull, low volume, OI velocity unreliable
- 13:00-13:30 = institutional positioning window (literature + retail-algo consensus)
- 13:30 = inflection — 30-min OI velocity computation completes; institutional positioning has committed
- 14:00-15:00 = institutional accumulation continues but late entries lose net edge after fees
- 15:10+ = late-arb fade window, exit BEFORE this

**Hold horizon (1h 40m)** is shorter than circuit_t1's 4h 45m. The setup is a **late-day continuation** play, not a full-day fade. Time stop at 15:10 captures the 13:30-15:10 institutional accumulation move but exits BEFORE 15:15-15:25 late-arb fade.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout**:
   - n < 500 trades over 1 year (TIGHT — three-leg filter may compress this)
   - NET PF < 1.10
   - NET Sharpe ≤ 0
2. **Long-bias finding extends to positioning-flow setups** — if PF < 1.10 with the three-leg filter, the project-wide pattern (long-bias loses on Indian intraday) extends even to OI-positioning-based long setups. Retire decisively, do not iterate to short-side variant on the same data.
3. **Institutional fakery dominates** — if the fake-OI-pump-then-reverse pattern dominates the candidate set even with leg-3 confirmation (e.g., > 50% of leg-3-passing trades go to hard_sl in the first 30 min), the literature caution has materialized in our data and naive OI signals are arbitraged out.
4. **Sample size too thin** — if three-leg filter yields < 200 trades/year, the candidate is too thin for production deployment regardless of PF.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Use existing 12-month 2024 5m feathers + per-stock-future minute OI capture (already ingested for `expiry_pin_strike_reversal` infra)
- Apply three-leg filter at 13:30 over each F&O 200 mid-cap stock per trading day
- Simulate 13:30 entry → 15:10 exit long, with R-multiple T1/T2 and 1% min-stop hard SL
- Compute NET PF using existing Indian fee model (`tools/sub7_validation/build_per_setup_pnl.py:calc_fee`)
- **Decision per §3.3:** PF ≥ 1.10 → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire

## Data engineering plan (preliminary, NOT yet built)

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_oi_long_buildup_continuation.py`** — pre-coding sanity check. Reads:
   - Existing 5m feathers (`backtest-cache-download/monthly/`)
   - Per-stock-future minute OI snapshots (need to verify if captured; if not, sanity tool reads F&O bhavcopy + interpolates intraday OI from the bhavcopy snapshot points)
   - Existing consolidated_daily.feather for ADV computation
   - No detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - `tools/oi_data/fetch_oi_spurts.py` — NSE OI-Spurts live feed scraper (for live mode); historical OI from F&O bhavcopy
   - `services/oi_loader.py` — load per-stock-future intraday OI snapshots, lookup by (symbol, time)
   - `structures/oi_long_buildup_continuation_structure.py` — the detector

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (LANDED) | oi_long_buildup_continuation (proposed) |
|---|---|---|---|
| Indian-specific | retail momentum exhaustion in T+0 opening | retail FOMO + operator pump exhaustion | F&O OI institutional positioning |
| Direction | short-only | short-only | **long-only** (first long setup in sub-9) |
| Active window | T+0 09:15-09:30 | T+1 10:30 single-bar | T+0 13:30 single-bar |
| Universe | small_cap | mid_cap, small_cap | F&O 200 mid-cap |
| Hold | intraday MIS (15-30 min) | intraday MIS (4h 45m) | intraday MIS (1h 40m) |
| Evidence base | empirical sub-7 validation | 5 peer-reviewed papers | 2 peer-reviewed papers + 7 retail-algo precedent sources |
| Expected n/yr | several thousand | ~500-700 | ~360-1,750 (TIGHT after three-leg filter) |
| Correlation w/ existing | n/a | low (different timing, fade vs continuation) | low (different signal source: OI vs price) |
| Decay risk | low | low | **moderate** — algo participation growing, institutional obfuscation possible |
| Project-bias compatibility | aligned (short) | aligned (short) | **counter** (long) — explicitly addressed above |

**Honest summary of fit:**
- **Good fit:** Different signal source (OI vs price), uncorrelated to existing setups, peer-reviewed Indian-market evidence, intraday MIS-compatible, positioning-based mechanism (structurally different from the failed sub7/sub8 long-pattern setups).
- **Concerns:** First long-side setup in sub-9, against project pattern of short-bias success. Three-leg filter might compress sample below n=500/yr. Institutional-fakery is a known retail trap that the leg-3 filter mitigates but doesn't eliminate.

The brief explicitly accepts these concerns and asks the §3.3 gate to determine whether the three-leg-filtered long-buildup signal still produces a tradable NET edge in 2024. If yes, proceed. If no, the candidate retires AND we accept that even positioning-based long setups don't survive Indian intraday — useful negative finding that hardens the project's short-bias.

---

## Decision required

**User action:**
1. **APPROVE** for sanity-check coding → I write `tools/sub9_research/sanity_oi_long_buildup_continuation.py`, apply the three-leg filter on 2024 F&O 200 mid-cap data, simulate, report NET PF.
2. **REJECT** with revisions → I revise specific points and re-submit.
3. **RETIRE before sanity** → if you judge the long-bias concern alone is disqualifying, we skip this setup entirely and do another fresh research round for short-side asymmetries.

**My read:** APPROVE for sanity check. The two-gate process (precedent YES + peer-reviewed PASS) is the strongest signal yet on a candidate. Project-wide long-bias caution is real but this setup's mechanism is structurally distinct from the failed pattern-based long setups. The sanity check will quantify the edge after the three-leg filter; that's the meaningful answer.
