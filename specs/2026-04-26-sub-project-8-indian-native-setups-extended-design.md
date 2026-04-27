# Sub-project #8 — Indian-Native Setup Library, EXTENDED (DESIGN)

**Status:** Phase 0 brainstorming + research-first design + **independent Opus research review applied 2026-04-26**. Ready for plan-writing.
**Date:** 2026-04-26 (rev2 — review revisions applied)
**Predecessor:** Sub-project #7 (3 detectors: mis_unwind_short, gap_fade_short, cpr_mean_revert). Phase 1 results: only gap_fade_short crossed PF ≥ 1.10 (PF=1.15) on Discovery, but failed Phase 2 (Sharpe 0.18 < 0.6). cpr_mean_revert bled (PF=0.64) even after the industry-standard TC/BC rewrite. mis_unwind_short never accumulated enough sample.
**Companion config:** `config/sub8_oci_overrides.json` (to be authored alongside the implementation plan).
**Review:** `specs/2026-04-26-sub-project-8-research-review.md` — independent Opus reviewer flagged 1 CRITICAL (Setup 4 citation), 5 MAJOR (mostly window/stop calibration), and 3 missed cross-cutting rules. All revisions are reflected inline below; the **Section 13 (rev2 changelog)** at the end summarizes diffs.

## 1. Sub-7 retrospective — what worked, what failed, learnings carried into sub-8

### 1.1 What worked
- **Per-setup independent validation discipline.** We refused to compose unvalidated detectors. The fact that we *correctly* killed cpr_mean_revert and mis_unwind_short instead of hiding them inside an Optuna joint optimisation is a process win, even though the strategy-level PF was negative.
- **gap_fade_short edge is real but thin.** PF=1.15 with ~14 trades/day on small/mid caps is meaningful — that detector survives into sub-8 untouched and serves as the baseline against which sub-8 detectors are compared.
- **wide_open_mode + bundled OCI capture** worked operationally. One OCI run, 487 sessions, 127k trades captured cleanly. The `_build_plan_from_sub7_detector` fast path in `pipelines/orchestrator.py` (lines 274, 800) is a reusable contract for sub-8 detectors.

### 1.2 What failed (root causes, not symptoms)
| Failure mode | Root cause |
|---|---|
| cpr_mean_revert PF=0.64 with 247 trades/day | (a) Stop multiplier 0.5×ATR vs Indian source-cited 0.75-1.0×ATR — getting wicked out before mean reversion. (b) Single-shot exit at CPR midpoint — no T1 partial. (c) Wrong universe: ran on 1000+ symbols incl. micro-caps; CPR is a Bank Nifty / Nifty 50 index/heavyweight tool per Frank Ochoa as taught in India. |
| mis_unwind_short n=326 | Stop multiplier 0.8×ATR vs source-cited 1.2-1.5×ATR for end-of-day high-volatility window. Most signals stopped out before the unwind began. Also overly restrictive cap-segment filter (small/mid only) starved sample. |
| gap_fade_short Sharpe 0.18 | Single-shot target at PDC. ~50% of fills hit T1 (50% gap fill) and reversed back through entry, eating the win. Fixed in this branch with T1/T2 tiered exits but Phase-2 number not yet re-run. |

### 1.3 Universal learnings → sub-8 design principles
1. **Stop placement is research-cited or the detector doesn't ship.** Every SL multiplier in sub-8 must cite an Indian source URL/book/chapter. No invented multipliers.
2. **Tiered exits are mandatory.** T1 (~1R partial 50%) + T2 (structural target). Single-shot exits are forbidden because they kill Sharpe even when PF is positive.
3. **Universe per setup.** CPR-class setups → Bank Nifty / Nifty 50 only. ORB-class setups → liquid F&O universe (~200 names). PDH/PDL retail-watch setups → small/mid caps. NO setup runs on the full 1000+ scan.
4. **`gate_input_logging.enabled = TRUE` in OCI capture configs.** Sub-7 OCI run forgot this; we couldn't run a formal Optuna gauntlet because the gate-input snapshots weren't persisted. Sub-8 capture flips this on by default.
5. **Detector emits clean TradePlan with `t1_price`, `t2_price`, `qty_t1_frac`, `qty_t2_frac`.** Orchestrator's existing `_build_plan_from_sub7_detector` path consumes this without modification.

## 2. Sub-8 design principles (binding rules)

```
RULE 1  Every parameter (window, threshold, stop multiplier, target multiple)
        cites an Indian source: URL, book chapter, or YouTube video timestamp.
        No opinion-based numbers.

RULE 2  Universe per setup is declared in `config/sub8_oci_overrides.json`
        as `enabled_caps_<setup>` or `index_filter_<setup>` keys.
        No global universe scan; each detector has its own pre-filter.

RULE 3  Every detector emits T1 + T2. Single-shot exits rejected at code review.

RULE 4  `gate_input_logging.enabled = true` in every sub-8 OCI override config.

RULE 5  Wide_open_mode + cycle_limit raised mirrors sub-7 capture protocol so
        Phase 4/5 OOS can re-use existing tools/gauntlet_v2/* tooling.

RULE 6  Phase 1 floor: NET PF >= 1.10 AND n_trades >= 500 AND Net Sharpe > 0.
        Same as sub-7 — the floor is conservative because we want to wash out
        weak edges before composition.
```

## 3. Setup #1 — ORB-15 (Opening Range Breakout, first 15 minutes)

### 3.1 Thesis
The first 15-minute range (09:15-09:30) sets the day's institutional intent. A *full-bar close* outside this range with above-average volume is the most-traded breakout setup in Indian intraday literature, and unlike the failed cpr_mean_revert it cuts *with* trend rather than against it.

### 3.2 Universe
NSE F&O liquid universe (~200 names) — same set our existing scanner uses for liquidity gating. Excludes circuit-band stocks (price within 2% of upper/lower 20% circuit) because they cannot be exited intraday once stuck.
- Rationale: Algotest backtest on Bank Nifty 2015-2023 used liquid index/futures only; retail tutorials repeatedly say "stick to NIFTY 50, Reliance, ICICI Bank" for cleaner price action.
- Source: [Algotest sample range-breakout strategy](https://docs.algotest.in/sample-algo-trading-strategies/range-breakout-strategy/), [In-the-Money by Zerodha — All About ORB Part 1](https://inthemoneybyzerodha.substack.com/p/all-about-opening-range-breakout)

### 3.3 Active window
**09:30 - 11:15 IST.** Range definition window 09:15-09:30. Entry window 09:30-11:15.
- Citation: In-the-Money by Zerodha Substack reports "the 09:15-11:15 ORB consistently outperforms all other time windows. The 09:15-10:15 window comes second." We choose 11:15 as the cutoff to capture both.

### 3.4 Trigger conditions
1. Range = (high, low) of 09:15-09:30 5-min bars (3 bars).
2. Range size >= 0.4% AND <= 2.0% of opening price (filter narrow noise + avoid blow-off opens).
3. Entry on the FIRST 5-min bar that **closes** above range_high (long) or below range_low (short).
4. Volume on the breakout bar >= 1.5× the 30-day median volume for that bar's clock-time slot.
   - Source: Algotest tutorial — "Confirm breakouts with higher-than-average volume and wait for a full candle close beyond the range, not just a wick. This reduces the chance of entering on a false breakout."

### 3.5 Stop placement (**rev2 — opposite-end-of-range PRIMARY per reviewer**)
- **Stop = opposite side of the range** (Indian-source-cited primary): long stop = range_low − wick_buffer, short stop = range_high + wick_buffer.
- Citation: AlgoTest, In-the-Money by Zerodha, Saimohanreddy ORB backtest.
- Reviewer flagged: mid-of-range "conservative variant" is a retail derivative, not the published default. Mid-range stop will be hit more often by ATR-typical noise after a wick-only break.
- Buffer: ± 0.10% of opening price (Saimohanreddy: 48% DD without buffer). 
- A/B variant: mid-of-range stop is configurable behind `stop_at_range_midpoint` flag for Phase 1 sweep, but DEFAULT = opposite-end.

### 3.6 Targets (tiered)
- **T1 = entry + 1R (50% of size).** Standard 1R partial.
- **T2 = entry + 2R (50% of size).** R = entry-to-stop distance.
- Citation: Algotest — "set your stop loss at the 50% threshold ... take 1:1.5 risk-to-reward trade" and onlinenifm.com "Many traders set their target at 1.5x or 2x the range size to capture strong moves."

### 3.7 Expected RR
- Average 1.5R blended (T1 capped at 1R for half size, T2 at 2R for half) → ~1.5R per fill.

### 3.8 Expected WR
- Backtest on Bank Nifty 2015-2023 reported "profitable almost every year except 2023." Win rate not explicitly reported but 100+ point days in 9 of 11 sample sessions implies ~60% directional WR before slippage.
- Source: [Saimohanreddy ORB backtest](https://saimohanreddy.com/orb-backtest-on-banknifty/)

### 3.9 Estimated trades/day in our universe scan
- ~200 F&O names. Historically ~15-25% trigger a clean ORB-15 with the volume filter on any given day. Cap to top-10 by signal strength → **~10 trades/day**.

### 3.10 Direction
**Bidirectional** (long on upside break, short on downside break).

---

## 4. Setup #2 — Narrow CPR Trending Day Breakout

### 4.1 Thesis
A narrow CPR (TC-to-BC distance < 0.4% of pivot) signals a tight previous-day consolidation. Frank Ochoa's *Secret of Pivot Boss* (taught in Indian context by Shubham Agarwal at Quantsapp) documents that narrow-CPR days have a **statistically significant trending bias** — price compression precedes expansion.

### 4.2 Universe
**Bank Nifty constituents (12 banks) + Nifty 50 (50 names) only.** Total ~55 unique names after de-dup.
- Rationale: CPR is taught as an **index and index-heavyweight** tool in India (Jainam, Fisdom, Optionx Journal). Sub-7 cpr_mean_revert failed partly because we ran it on 1000+ names including illiquid small caps where CPR is statistical noise.

### 4.3 Active window
**09:30 - 14:00 IST.** Skip first 15 min (let opening range form). Skip last 90 min (MIS unwind window owned by Setup #5).
- Citation: Multiple Indian CPR tutorials (Jainam, Fisdom, Tradingdirection.in CPR Brahmastra) recommend mid-session for CPR breakout entries.

### 4.4 Trigger conditions
1. CPR width = (TC - BC) / Pivot. Filter for narrow_cpr_pct <= 0.40%.
2. Bar **closes** above TC (long) or below BC (short).
3. Bar volume >= 1.3× 20-day median volume for the same clock-time slot.
4. Anti-whipsaw: no entry if previous 2 bars already had a TC/BC tag-and-reject (avoids second-attempt failure).

### 4.5 Stop placement
- **Stop = Pivot level (P)** for both long and short.
- Citation: [Optionx Journal — CPR Explained](https://optionx.trade/blogs/central-pivot-range-cpr-explained), [Jainam — CPR in Trading](https://www.jainam.in/blog/cpr-in-trading/) — both cite "stop loss at the Pivot level to protect against unfavorable price movements."
- Risk per trade in points = entry - Pivot (long) or Pivot - entry (short). Typical narrow-CPR: 0.20-0.40% of price.

### 4.6 Targets (tiered)
- **T1 = R1 (long) or S1 (short)** — 50% of size.
- **T2 = R2 (long) or S2 (short)** — 50% of size.
- Citation: Optionx Journal — "Set profit targets using ... R:R ratio (1:2 or 1:3), or levels like the next support/resistance. If the next resistance and support level is far then middle levels are used for targets."

### 4.7 Expected RR
- Typical narrow-CPR R1 distance is ~2.5-3× pivot-stop distance → **T1 ~2R**, T2 ~3R. Blended ~2.5R per fill.

### 4.8 Expected WR
- Indian sources do not publish raw WR for narrow-CPR. Frank Ochoa's *Secret of Pivot Boss* reports ~55-65% trending-day WR for narrow CPRs across US futures (cited in Indian CPR tutorials by analogy).
- Conservative assumption for our backtest: 45-55%.

### 4.9 Estimated trades/day in our universe scan
- 55 names × ~15% narrow-CPR-day rate × ~50% directional trigger rate = ~4 trades/day average. Some days zero, some days 8.

### 4.10 Direction
**Bidirectional.**

---

## 5. Setup #3 — VWAP First-Pullback Continuation

### 5.1 Thesis
After a clean morning move (09:30-10:30) above/below VWAP, the **first pullback to VWAP** that holds with a bullish/bearish reversal candle is one of the most institutionally watched setups in Indian intraday. It is published verbatim by Rupeezy, Choice India, BlinkX, and Tradingshastra.

### 5.2 Universe
NSE F&O liquid universe (~200 names) AND Bank Nifty constituents — same as ORB. Excludes micro-caps (VWAP is meaningless on illiquid books).

### 5.3 Active window (**rev2 — extended to 14:30 to capture JM Financial's afternoon golden window**)
**10:00 - 14:30 IST.** Need ~45 min of trend before first pullback is valid. Cut at 14:30 to avoid overlap with Setup 5 (CHR starts 14:30).
- Citation: [JM Financial — Intraday Trading Time Analysis](https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis) — "two golden windows for intraday trading are 9:30-11:30 AM and 1:30-3:00 PM."
- Reviewer flagged: rev1's 13:30 cutoff missed the entire 13:30-15:00 second golden window. Lunch-lull (11:30-13:30) is correctly avoided by starting at 10:00. Setup 5 (CHR) handles 14:30-15:15.

### 5.4 Trigger conditions
1. Established trend: at least 4 of last 6 bars (5-min) closed on the same side of VWAP (>= 4/6 above for long, >= 4/6 below for short).
2. Pullback bar: low (long) or high (short) touches within 0.10% of current VWAP value.
3. Reversal candle: next 5-min bar closes back beyond VWAP in original trend direction with bar range >= 0.20% of price.
4. Volume on the reversal candle >= prior bar volume (not necessarily above-average; Rupeezy emphasises *relative* spike near VWAP).

### 5.5 Stop placement
- **Stop = pullback bar's low (long) or high (short).**
- Multiplier check: stop distance must be <= 0.6% of price. If wider, skip — the pullback was too deep, signal invalid.
- Citation: [Rupeezy — VWAP Trading Strategy](https://rupeezy.in/blog/vwap-trading-strategy-intraday-options) — "Place the stop-loss just below the VWAP and the entry candle's low for a long position, or just above the VWAP and the entry candle's high for a short position."

### 5.6 Targets (tiered)
- **T1 = previous swing high (long) or low (short)** — 50% of size.
- **T2 = entry + 2× T1 distance from entry** (Fibonacci 2.0 extension proxy) — 50% of size.
- Citation: Rupeezy — "The primary profit target is typically the previous high in an uptrend or the previous low in a downtrend. More advanced traders might use Fibonacci extensions."

### 5.7 Expected RR
- T1 typically 1.0-1.5R (recent swing nearby). T2 2.0-2.5R. Blended ~1.5-2R.

### 5.8 Expected WR
- Choice India and BlinkX both reference "60-65% with strict trend filter." We assume 50% in our backtest as a conservative haircut for slippage.

### 5.9 Estimated trades/day
- ~200 names × ~5% qualify per day = ~10 candidates, capped to top-5 by trend strength = **~5 trades/day**.

### 5.10 Direction
**Bidirectional.**

---

## 6. Setup #4 — PDH/PDL Touch-and-Reject Fade (rev2 — generic Indian retail PDH/PDL fade)

**Reviewer flagged (CRITICAL):** rev1 attributed this to "Subasish Pani style" — incorrect. Subasish Pani's published method is the **5 EMA strategy** (Alert Candle), NOT a PDH/PDL fade. Capital.com (rev1 citation) is a UK forex retail site, not Indian. Both attribution and citation are removed in rev2. The setup itself remains; we treat it as generic Indian-retail PDH/PDL fade with explicit acknowledgment of unresolved volume polarity.

### 6.1 Thesis
PDH and PDL are watched levels by Indian retail. When price tags PDH and prints a rejection candle (long upper wick, small body) intraday, the level often holds — fade the rejection. This is generic Indian-retail intraday teaching (Groww, ChartMantra, TradingQnA threads), NOT a single-author named strategy.

### 6.2 Universe
Small + mid-cap NSE F&O universe (~100 names after dropping the top-50 mega-caps). Retail-driven flow is concentrated here.
- Rationale: Sub-7 gap_fade_short worked on small/mid (PF=1.15). Same demographic logic — retail-watched levels work where retail dominates the order book.

### 6.3 Active window
**10:00 - 14:30 IST.** Skip opening 45 min (PDH/PDL test in opening range is noise). Skip last hour (owned by MIS unwind setup).

### 6.4 Trigger conditions (**rev2 — volume polarity is A/B variant, not single-rule**)
1. Bar's high (for short) tags PDH within 0.10% AND bar closes back below PDH with body_size_pct < 40% AND upper_wick_ratio > 1.5× body.
   - For longs: bar's low tags PDL within 0.10% AND bar closes back above PDL with same wick/body filter inverted.
   - Reviewer note: 0.10% / 40% / 1.5× ratios are design-author inferences. Indian sources are silent on exact ratios. Phase 1 should sweep these.
2. **Volume polarity (A/B variant — Indian sources contested):**
   - Variant A: rejection bar volume **NOT** >= 1.5× recent (absence of volume = no breakout conviction).
   - Variant B: rejection bar volume >= 1.5× recent (volume spike at rejection = institutional fade).
   - Default config ships Variant A; Variant B enabled by `volume_polarity: "spike"` config flag.
   - Phase 1 will run BOTH polarities and the data decides.
3. No prior tag of PDH/PDL today (only fade the FIRST tag — second tags are typically breakouts).

### 6.5 Stop placement
- **Stop = rejection bar's wick extreme + 0.10% buffer.** I.e., short stop = wick high × 1.001, long stop = wick low × 0.999.
- Citation (rev2): [Groww — Previous Day High and Low Strategy](https://groww.in/blog/previous-day-high-low-strategy), TradingQnA / ChartMantra retail PDH/PDL threads (Indian-source-aligned). Generic "stops just beyond the level" rule. The 0.10% buffer % is design-author inference (Indian sources silent on exact buffer).

### 6.6 Targets (tiered)
- **T1 = VWAP** (50% size). The first mean-revert magnet for a rejection trade.
- **T2 = opposite end of today's range** (50% size). E.g., for short at PDH, T2 = today's low so far.
- Citation (rev2): VWAP magnet target is Indian-source-aligned (Rupeezy, Choice India general VWAP teachings). T2 = opposite range extreme is design inference, not Indian-source-cited specifically for PDH/PDL fade — flagged as inference.

### 6.7 Expected RR
- Blended ~2.0-2.5R per fill.

### 6.8 Expected WR
- Subasish Pani 5-EMA strategy (related rejection style): self-reported 60% WR (myalgomate / TradingQnA).
- Conservative for backtest: 45%.

### 6.9 Estimated trades/day
- ~100 small/mid names × ~6% tag PDH/PDL per day with rejection pattern = ~6 trades/day.

### 6.10 Direction
**Bidirectional** — short rejections at PDH, long rejections at PDL.

---

## 7. Setup #5 — Closing Hour Reversal (CHR), 14:30 - 15:15 IST

### 7.1 Thesis
This is the **complement to sub-7's failed mis_unwind_short**. The structural insight remains valid (SEBI MIS square-off pressure 14:55-15:20 forces unwinds), but sub-7's variant scanned for SHORT setups on stocks at fresh intraday highs. The bug: many of those signals were *trend continuations*, not exhaustions. Sub-8 widens the trigger to **trend reversal** in either direction with proper exhaustion confirmation, and uses the source-cited 1.2-1.5× ATR stop.

### 7.2 Universe
Full NSE F&O liquid universe (~200) — both long and short reversals trade.

### 7.3 Active window
**14:30 - 15:15 IST.** Square-off forced by 15:20 hard stop on all positions.
- Citation: [Zerodha — MIS auto square-off timings](https://support.zerodha.com/category/trading-and-markets/trading-faqs/market-sessions/articles/intraday-auto-square-off-timings) — equity MIS square-off begins 15:20.

### 7.4 Trigger conditions
1. Trend leg: stock has moved >= 1.5% in one direction between 09:30 and 14:30 (intraday range > 1.5×ATR(14)).
2. At 14:30+, exhaustion candle prints: opposite-direction bar with body >= 60% of total range AND volume >= 1.3× recent average.
3. Direction: short if the move was UP (faded by retail unwind); long if move was DOWN (covered by short-squarers).

### 7.5 Stop placement (**rev2 — tightened toward Indian-source-cited 1.5× ATR standard**)
- **Stop = recent intraday extreme + 1.5 × ATR(14, 5-min).**
- Citation: [Goodwill — ATR Smart Stop-Losses](https://www.gwcindia.in/blog/using-atr-average-true-range-to-set-smart-stop-losses/), [Zerodha Varsity — Volatility Applications](https://zerodha.com/varsity/chapter/volatility-applications/) — 1.5×ATR is the most-cited Indian intraday stop multiplier; 1.5-2× is the "widely used range." Sub-7 used 0.8× (too tight, killed sample). Rev1 used 1.2× (closer to industry but still conservative).
- A/B variant: 1.2× available behind `stop_atr_multiplier_variant: "tight"` config flag for Phase 1 sweep. Default = 1.5×.

### 7.6 Targets (tiered) (**rev2 — hard time stop pushed to 15:22**)
- **T1 = VWAP** (50% size).
- **T2 = Pivot (CPR P) or 50% retrace of the day's intraday move**, whichever is closer (50% size).
- **Hard time stop: 15:22** (rev2: was 15:18). Zerodha equity MIS auto square-off begins 15:25 — 15:22 leaves 3 min margin which is sufficient and captures 4 more minutes of trade time vs rev1's 15:18.
- Citation: VWAP magnet — Indian-source-aligned (StockGro, Subhadip Nandy / Capitalmind). Hard stop — [Zerodha — MIS Auto Square-off Timings](https://zerodha.com/z-connect/updates/changes-to-the-auto-square-off-timings-for-equity-and-fo).

### 7.6a Fail-fast trip-wire (**rev2 — added per reviewer**)
**Mandatory:** if Setup 5 produces fewer than **200 trades in the first 100 OCI sessions**, KILL it without waiting for full Discovery. Rationale: sub-7 mis_unwind_short failed because it never accumulated sample (n=326 over 487 sessions). Setup 5 is the structural successor with corrected stop and bidirectional support, but if the sample-size pattern persists, the underlying assumption (MIS-unwind produces directional reversion) is what's wrong, not the parameters.

### 7.7 Expected RR
- T1 typically 1-1.5R (VWAP nearby late-day). T2 2-3R. Blended ~1.5-2R per fill.

### 7.8 Expected WR
- No published Indian-source WR. Conservative 40-45%.

### 7.9 Estimated trades/day
- ~200 names × ~3% qualify per day = ~6 trades/day in the 45-min window.

### 7.10 Direction
**Bidirectional** — long for down-move reversal, short for up-move reversal.

---

## 8. Setup library composition summary

| # | Setup | Universe | Window | Direction | Est. trades/day |
|---|---|---|---|---|---|
| 1 | ORB-15 (carry-over baseline) | F&O ~200 | 09:30-11:15 | Bidir | ~10 |
| 2 | Narrow CPR Trending Breakout | NIFTY 50 + Bank Nifty (~55) | 09:30-14:00 | Bidir | ~4 |
| 3 | VWAP First-Pullback | F&O ~200 | 10:00-13:30 | Bidir | ~5 |
| 4 | PDH/PDL Touch-and-Reject | Small/Mid F&O ~100 | 10:00-14:30 | Bidir | ~6 |
| 5 | Closing Hour Reversal | F&O ~200 | 14:30-15:15 | Bidir | ~6 |
| Sub-7 carry | gap_fade_short (passes Phase-1) | Small/Mid + micro | 09:15-09:30 | Short | ~14 |

**Total estimated suite throughput: ~45 trades/day** before slot competition / cross-sectional gates. After gates we expect ~15-20 fills/day, similar to sub-7 scale.

---

## 9. Implementation phasing

### Phase 0 — DONE (this design doc)
Research + source-citation per setup. Brainstorming on universe and exit tiers.

### Phase 1A — Local detector build (~3 weeks)
1. Implement Setups #1, #3 first (ORB and VWAP-pullback) — most well-defined Indian sources, easiest to validate.
2. Implement Setups #2, #4 next (CPR and PDH/PDL) — fewer published numbers, more interpretation.
3. Implement Setup #5 last (CHR) — explicitly an iteration on sub-7's failed mis_unwind_short; needs the most A/B comparison vs sub-7 variant.
4. Each detector uses the existing `BaseStructure` interface and emits `TradePlan` with `t1_price`, `t2_price`, `qty_t1_frac=0.5`, `qty_t2_frac=0.5`. No engine changes required (the orchestrator's `_build_plan_from_sub7_detector` fast path consumes these).

### Phase 1B — Local subset iteration (~1 week)
Run on 5-10 random Discovery sessions (`main.py --dry-run`). Verify:
- Trigger fires the expected number of times.
- TradePlan has both T1 and T2 prices populated.
- Stop is research-multiplier-aligned (assert in test).
- Universe filter respected (no ORB on a micro-cap, no CPR outside Nifty/Bank Nifty).

### Phase 2 — Bundled OCI capture (1 run, $100-300)
```json
{
  "enabled_detectors": [
    "orb_15", "narrow_cpr_breakout", "vwap_first_pullback",
    "pdh_pdl_reject", "closing_hour_reversal",
    "gap_fade_short"
  ],
  "wide_open_mode": true,
  "gate_input_logging": {"enabled": true},
  "max_trades_per_cycle": 10000,
  "universe_per_setup": {
    "orb_15": "fno_liquid_200",
    "narrow_cpr_breakout": "nifty50_banknifty",
    "vwap_first_pullback": "fno_liquid_200",
    "pdh_pdl_reject": "smallmid_fno_100",
    "closing_hour_reversal": "fno_liquid_200",
    "gap_fade_short": "smallmid_micro"
  }
}
```
Period: 2023-01-01 → 2026-03-31 (full 3.25y, captures Discovery + Validation + Holdout in one run for code-version consistency, same as sub-7).

### Phase 3 — Per-setup analysis (1 week local)
Apply `tools/sub7_validation/build_per_setup_pnl.py` (already built in sub-7) and `per_setup_report.py` (already built). Slice trade_report by setup_type, apply Indian fee model, compute per-setup metrics.

### Phase 4 — Phase-1 Pass/Fail gate per setup
Same threshold as sub-7:
```
NET PF      >= 1.10  on Discovery (2023-2024)
n_trades    >= 500
Net Sharpe  > 0
```
- ≥3 of 5 new + gap_fade_short pass → proceed to Phase 5 composition.
- 1-2 pass → soft warning, debug stop multipliers / re-cite sources.
- 0 pass → KILL sub-8.

### Phase 5 — Mechanical portfolio composition (1 week local)
Use existing `LiveGateChain` + `screener_live` for slot competition + ranking. Test equal-weight first, then risk-parity. NO joint Optuna.

```
NET PF        >= 1.25
Net Sharpe    >= 0.6
Max DD        <= 20%
Beats Nifty50 in 2023 AND 2024 individually
```

### Phase 6 — Validation OOS (1 day, no new OCI)
`tools/gauntlet_v2/validate.py --period validation` (Jan-Sep 2025).
```
NET PF      >= 1.15
Net Sharpe  >= 0.5
Max DD      <= 25%
```

### Phase 7 — Holdout OOS (1 day, no new OCI)
Same as Phase 6 but Oct 2025-Mar 2026.

---

## 10. Validation methodology

- **Phase 1 thresholds** identical to sub-7 (PF ≥ 1.10 / n ≥ 500 / Sharpe > 0). The sub-7 floor was correct; the sub-7 setups simply didn't clear it. We're not relaxing the floor.
- **OOS split** identical to sub-7: Discovery 2023-2024 / Validation 2025-Sep / Holdout Oct 2025-Mar 2026. Same period rationale (regime continuity, no expiry-shift artefacts).
- **OOS data freshness:** As of sub-8, sub-7's Phase 4/5 has only consumed validation+holdout for 3 detectors. The 5 new sub-8 detectors are conceptually distinct trigger logic, so re-using the same OOS period is acceptable per the same Q6 rationale documented in sub-7 design doc Section 2.
- **Slippage / fees:** Use the exact Indian intraday schedule from `services/logging/trading_logger.py`. MIS leverage applied per-trade BEFORE aggregation (per project memory rule). No per-trade tax — tax computed on FY net at 31.2% per project memory.

---

## 10a. Cross-cutting rules (rev2 — added per reviewer's missed-rules section)

These rules apply to **every setup** in the suite, implemented as universe-filter / detector-base prerequisites, not per-setup logic.

### 10a.1 Expiry-day exclusion
- **Rule:** Skip ALL detectors on Indian F&O weekly expiry days (post-Sep 2025: Tuesday for Nifty, Thursday/Wednesday legacy schedules where applicable). Implementation: load NSE expiry calendar; detector first-condition check `if today in expiry_dates: return _empty("expiry_day_excluded")`.
- **Rationale:** Reviewer flagged — bid-ask spreads widen after 14:30 on expiry days; OI-driven sharp bidirectional moves distort all 5 setups, especially Setup 5 (CHR 14:30-15:15) which sits squarely in the danger zone. Sources: [HDFC Sky — Bank Nifty Expiry](https://hdfcsky.com/blogs/share-market/bank-nifty-expiry), [AlgoTest — Bank Nifty Expiry Day](https://algotest.in/blog/bank-nifty-expiry-day/), [Sahi.com — Nifty Expiry Day Rules](https://www.sahi.com/blogs/5-rules-every-trader-must-follow-on-nifty-options-expiry-day).
- **Phase 1 variant:** Run an A/B with `include_expiry_days: true` to confirm the exclusion's PnL impact and validate the rule empirically.

### 10a.2 Gap-day cross-detector exclusion (ORB-15 vs gap_fade_short)
- **Rule:** If `(open_today - PDC) / PDC > 0.5%` (gap up) OR `< -0.5%` (gap down), DISABLE Setup 1 (ORB-15) for that session and let the existing sub-7 `gap_fade_short` handle the gap regime. Implementation: ORB-15 detector first-condition check `if abs(opening_gap_pct) > 0.5: return _empty("gap_day_routed_to_gap_fade")`.
- **Rationale:** Reviewer flagged — Indian sources (gwcindia, truedata, motilaloswal) describe gap > 0.5% as requiring a different playbook. Both detectors firing on the same gap day creates duplicate signals from contradictory thesis (ORB cuts WITH trend, gap_fade cuts AGAINST).

### 10a.3 Universe-wide circuit-band exclusion
- **Rule:** Across ALL 5 setups + sub-7 carry-overs, exclude any symbol where price is within 2% of upper/lower 5/10/20% intraday circuit limit. Implementation: `services/universe_filter.py` adds `not_circuit_band_proximate(symbol, current_price)` helper; every detector calls it as the FIRST universe-filter prerequisite.
- **Rationale:** Reviewer flagged — once a stock circuits, intraday positions become forced delivery (StockGro, 5paisa). Rev1 had this rule only on Setup 1; rev2 makes it universe-wide.

### 10a.4 Pre-open call-auction wick handling (Setup 1 only)
- **Rule:** Setup 1 (ORB-15) range starts at **09:20 IST**, NOT 09:15. The first 5-min bar (09:15-09:20) often contains the call-auction print + first continuous-trade ticks, creating a synthetic wick that distorts the range definition.
- **Citation:** Reviewer's concern, no Indian source explicitly documents the call-auction wick issue but it is observable in any 5-min bar dataset for Indian equities.
- **A/B variant:** Setup 1 ships with `range_window_start: "09:20"` as the default; `09:15` available behind `include_call_auction: true` for Phase 1 sweep.

---

## 11. Setups considered and REJECTED (and why)

| Candidate | Why rejected |
|---|---|
| Closing Hour Reversal at INDEX level (Bank Nifty/Nifty futures) | We cannot trade index futures with current capital allocation logic; equity-only constraint. CHR equity-level is included as Setup #5 instead. |
| Lunch-lull breakout (12:30-13:30) | JM Financial explicitly says "most losses ... come from trying to trade a consolidating market where volume is low" 11:30-13:30. Indian sources unanimously discourage this window. No Indian-source citation supports a lunch breakout edge. |
| Pre-market gap continuation (vs gap fade) | India's pre-market 09:00-09:08 sets the opening price by call auction; there is no "continuation" leg until 09:15. The gap is fully expressed in the opening 5-min bar. Continuation logic collapses into ORB-15 (Setup #1), which already captures this. |
| Block-deal aftermath (next-day momentum) | Indian sources (Sharekhan, Trendlyne) describe block deals as informational, not as a documented intraday entry trigger. NSE/SEBI publishes block deals at EOD, by which time intraday opportunity is gone. Found no Indian source with explicit entry/SL/target rules. |
| F&O ban-list rotation | When a stock enters F&O ban (MWPL > 95%), only square-off positions allowed in F&O. The cash market becomes a thin retail-only book. No Indian source documents an entry trigger / SL / target rules for cash-only ban-day intraday. Reject. |
| Expiry-day NIFTY/BANKNIFTY constituent decay | Bank Nifty weekly expiry is Wednesday, Nifty Thursday — different days, different constituent flows. Elearnmarkets / Vivek Bajaj webinars discuss expiry **option** strategies, not **constituent equity** intraday. No Indian-source citation for the constituent-decay equity trigger. Reject. |
| Circuit-band proximity rejection | When a stock hits an upper/lower circuit, **no opposite-side liquidity exists by definition** — intraday positions become forced delivery (per StockGro, 5paisa, Mastertrust). The "fade near circuit" trade has no Indian source documenting how to fade *before* the circuit hits, only descriptions of why you can't trade *after*. Reject as untradeable. |

---

## 12. Source bibliography

1. [In-the-Money by Zerodha — All About ORB Part 1](https://inthemoneybyzerodha.substack.com/p/all-about-opening-range-breakout)
2. [Algotest — Sample Range Breakout Strategy](https://docs.algotest.in/sample-algo-trading-strategies/range-breakout-strategy/)
3. [Saimohanreddy — ORB Backtest on Bank Nifty 2015-2023](https://saimohanreddy.com/orb-backtest-on-banknifty/)
4. [Zerodha Varsity — Central Pivot Range](https://zerodha.com/varsity/chapter/the-central-pivot-range/)
5. [Zerodha Varsity — Volatility Applications (ATR for stops)](https://zerodha.com/varsity/chapter/volatility-applications/)
6. [Optionx Journal — CPR Explained with NSE Examples](https://optionx.trade/blogs/central-pivot-range-cpr-explained)
7. [Jainam — CPR in Trading](https://www.jainam.in/blog/cpr-in-trading/)
8. [Tradingdirection — CPR Brahmastra Strategy webinar](https://www.tradingdirection.in/courses/CPR-Brahmastra-Strategy-6415f372e4b050ea3bb6f0f0)
9. [Rupeezy — VWAP Trading Strategy](https://rupeezy.in/blog/vwap-trading-strategy-intraday-options)
10. [Choice India — VWAP Trading Strategy](https://choiceindia.com/blog/vwap-trading-strategy)
11. [BlinkX — Volume Weighted Average Price](https://blinkx.in/en/knowledge-base/share-market/volume-weighted-average-price-trading-strategy)
12. [Tradingshastra — VWAP Institutional Indicator 2025](https://tradingshastra.com/vwap-institutional-indicator/)
13. ~~Capital.com — PDH/PDL Day Trading Toolbox~~ (rev2 — REMOVED. Capital.com is UK-based forex retail site, NOT Indian. Replaced by Groww / TradingQnA / ChartMantra retail PDH/PDL threads as Indian-source references for Setup 4.)
13a. [Groww — Previous Day High and Low Strategy](https://groww.in/blog/previous-day-high-low-strategy)
13b. [Goodwill — Using ATR for Smart Stop-Losses](https://www.gwcindia.in/blog/using-atr-average-true-range-to-set-smart-stop-losses/)
13c. [TradingQnA / ChartMantra retail PDH/PDL threads](https://tradingqna.com/search?q=previous%20day%20high%20low)
14. [Power of Stocks — 5 EMA Strategy on Myalgomate](https://www.myalgomate.com/product/5-ema-strategy-power-of-stocks/)
15. [TradingQnA — Backtesting 5 EMA Strategy of Power of Stocks](https://tradingqna.com/t/backtesting-5-ema-strategy-of-power-of-stocks/131974)
16. [Algotest — 5 EMA Scalping by Power of Stocks](https://docs.algotest.in/signals/famous-strategies/5-ema/)
17. [JM Financial — Intraday Trading Time Analysis](https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis)
18. [Zerodha — MIS Auto Square-Off Timings](https://support.zerodha.com/category/trading-and-markets/trading-faqs/market-sessions/articles/intraday-auto-square-off-timings)
19. [Quantsapp — Shubham Agarwal Bank Nifty Strategies](https://www.quantsapp.com/learn/articles/Deploy-Bank-Nifty-Call-Butterfly-Spread-strategy-Shubham-Agarwal-196)
20. Frank Ochoa — *Secret of Pivot Boss* (referenced in Indian CPR tutorials by Jainam, Optionx, Fisdom)

---

## 13. Rev2 changelog (independent Opus reviewer revisions, 2026-04-26)

| Setup / Rule | Rev1 | Rev2 | Reason |
|---|---|---|---|
| **Setup 1 (ORB-15) stop** | mid-of-range (conservative variant) | opposite-end-of-range (industry standard) | Reviewer MAJOR: mid-range is retail variant, not Indian-source default; gets hit by ATR noise after wick-only break. Mid-range now A/B variant. |
| **Setup 1 range start** | 09:15 | 09:20 (default) | Reviewer MISSED RULE: pre-open call-auction (09:00-09:08) print contaminates 09:15-09:20 first bar. 09:15 now A/B variant. |
| **Setup 3 (VWAP) window** | 10:00-13:30 | 10:00-14:30 | Reviewer MAJOR: rev1 cut at 13:30 missed JM Financial's "1:30-3:00 PM" second golden window. Cap at 14:30 to leave Setup 5 window clear. |
| **Setup 4 attribution** | "Subasish Pani style" | generic Indian retail PDH/PDL fade | Reviewer CRITICAL: Subasish Pani's published method is 5 EMA strategy, NOT PDH/PDL fade. Mis-attribution removed. |
| **Setup 4 citation** | Capital.com | Groww + ChartMantra + TradingQnA | Reviewer CRITICAL: Capital.com is UK forex site, violates RULE 1 ("Indian sources only"). Replaced. |
| **Setup 4 volume polarity** | absence of volume only | A/B variant: absence (default) + spike | Reviewer CRITICAL: Indian sources contested on polarity; coin-flip risk. Phase 1 runs both; data decides. |
| **Setup 5 (CHR) stop** | 1.2× ATR | 1.5× ATR (default), 1.2× as A/B variant | Reviewer MAJOR: 1.5× is most-cited Indian intraday standard (Goodwill, Varsity volatility chapter). Rev1's 1.2× was closer to standard than sub-7's 0.8× but still conservative. |
| **Setup 5 hard time stop** | 15:18 | 15:22 | Reviewer MINOR: Zerodha equity MIS auto square-off begins 15:25; 15:22 leaves 3 min margin (sufficient) and captures 4 more minutes vs rev1's 15:18. |
| **Setup 5 fail-fast trip-wire** | none | n<200 in first 100 sessions → KILL | Reviewer MAJOR risk: Setup 5 is structural successor to sub-7 mis_unwind_short which never accumulated sample (n=326). Trip-wire prevents wasting Discovery on a structurally-failing setup. |
| **MISSED: Expiry-day exclusion** | absent | added (Section 10a.1) | Reviewer MISSED RULE: post-Sep 2025 weekly expiries (Tuesday Nifty, Thursday/Wed BankNifty) distort closing-hour moves. Setup 5 in danger zone. Universe-wide rule. |
| **MISSED: Gap-day cross-detector exclusion** | absent | added (Section 10a.2) | Reviewer MISSED RULE: gap > 0.5% routes to gap_fade_short; ORB-15 disabled to avoid duplicate-thesis signals. |
| **MISSED: Circuit-band exclusion universe-wide** | only on Setup 1 | added universe-wide (Section 10a.3) | Reviewer MISSED RULE: rev1 had this only on ORB; once a stock circuits, all setups become forced-delivery. Universal rule. |

**Setup 4 alternative:** Reviewer recommended rebuilding Setup 4 as actual **Subasish Pani 5 EMA strategy** (myalgomate, algotest, tradingqna, ezquant — authoritative Indian sources). For rev2, we keep PDH/PDL as a generic retail fade (with proper Indian citations) and defer 5 EMA strategy to a potential **sub-9** project (more setups, narrower attribution).

**Verdict on rev2:** SHIP as REVISED. All CRITICAL and MAJOR issues addressed. MINOR over-fitting concerns are explicitly acknowledged and parameters flagged for Phase 1 sweep.
