# Sub-project #8 — Independent Research Review

**Reviewer mode:** Independent / blind. Indian sources only.
**Date:** 2026-04-26
**Subject:** `specs/2026-04-26-sub-project-8-indian-native-setups-extended-design.md`

---

## Section 1 — Independent research findings per setup

### Setup 1: Opening Range Breakout (ORB-15)

| Parameter | Indian-source finding | Source |
|---|---|---|
| Range definition window | Two camps. **15-min (09:15-09:30)** is the most-taught default per Streak/AlgoTest tutorials and TradingQnA threads. **30-min (09:15-09:45)** appears in retail academy material (sahi.com, AngelOne). **60-min (09:15-10:15)** is the window used in Zerodha Varsity / In-the-Money's published backtest | https://inthemoneybyzerodha.substack.com/p/all-about-opening-range-breakout, https://tradingqna.com/t/orb-15-mins-break-out-or-break-down/52978, https://saimohanreddy.com/orb-backtest-on-banknifty/ |
| Entry confirmation | Full-bar close above range high (long) / below range low (short). Volume above 30-day average emphasised by AlgoTest. "Wait for full candle close beyond the range, not just a wick" | https://docs.algotest.in/sample-algo-trading-strategies/range-breakout-strategy/ |
| Stop placement | **Industry standard: opposite end of the range.** Mid-range stop is a tighter retail variant but not the published default. Saimohanreddy notes 48% drawdown without buffer | https://saimohanreddy.com/orb-backtest-on-banknifty/, https://docs.algotest.in/sample-algo-trading-strategies/range-breakout-strategy/ |
| Targets | "Many traders use 1.5x or 2x range size" / structural S/R levels | https://www.onlinenifm.com/blog/post/326/technical-analysis/what-is-opening-range-breakout-orb-trading-strategy-explained |
| Active window cutoff | **In-the-Money / Zerodha Substack backtest: 09:15-11:15 outperforms all other windows; 09:15-10:15 second.** Rest of day decays | https://inthemoneybyzerodha.substack.com/p/all-about-opening-range-breakout |
| Universe | Bank Nifty / Nifty 50 / liquid F&O. AlgoTest tutorial uses index futures. Some retail tutorials say "stick to NIFTY 50, Reliance, ICICI Bank" | https://docs.algotest.in/sample-algo-trading-strategies/range-breakout-strategy/ |
| Win Rate / RR | In-the-Money: ~60% long, ~50% short. Saimohanreddy: profitable most years, **NOT profitable on Bank Nifty after slippage + brokerage** | https://saimohanreddy.com/orb-backtest-on-banknifty/, https://inthemoneybyzerodha.substack.com/p/all-about-opening-range-breakout |
| Note: directional asymmetry | Long ~60%, short ~50% — Indian sources document long-bias on ORB |  |

### Setup 2: Narrow CPR Trending Breakout

| Parameter | Indian-source finding | Source |
|---|---|---|
| "Narrow" CPR threshold | **Two cited values: 0.5% and 0.25% of pivot.** OptionX / TradingView Indian script community cite 0.5% as trending-day boundary. 0.25% is "very narrow / strong trending." 0.2% appears in Chartink scanners. | https://optionx.trade/blogs/central-pivot-range-cpr-explained, https://in.tradingview.com/scripts/cpr/, https://www.scribd.com/document/659685125/ |
| Direction trigger | Bar close above TC = long, below BC = short. Confirmation with 5-min close and volume | https://www.gwcindia.in/blog/top-5-indicator-setups-for-bank-nifty-traders-in-india/ |
| Stop placement | **Pivot (P)** is most-cited stop level (Jainam, OptionX). Some sources cite S1/R1 or below CPR for tighter R:R | https://www.jainam.in/blog/cpr-in-trading/, https://optionx.trade/blogs/central-pivot-range-cpr-explained |
| Targets | R1/R2 (long), S1/S2 (short), or 1:2/1:3 R:R | https://optionx.trade/blogs/central-pivot-range-cpr-explained |
| Active window | Indian sources are SILENT on a specific window. Recommend mid-session per Jainam/Fisdom — 09:30+ is fine; no published cutoff | https://www.jainam.in/blog/cpr-in-trading/ |
| Universe | **Indian CPR teachings center on Bank Nifty + Nifty 50 / index heavyweights.** Frank Ochoa's Pivot Boss (US futures) is the original source. Retail tutorials in India (Fisdom, OptionX, Jainam) all reference index/Nifty constituents | https://www.fisdom.com/central-pivot-range/, https://optionx.trade/blogs/central-pivot-range-cpr-explained |
| Anti-whipsaw rule | **Indian sources are SILENT on lookback for failed previous breakouts.** No formal rule documented. |  |
| Backtest WR | Zerodha Varsity / Samco mention "70% with volume confirmation" but no public auditable backtest | https://www.samco.in/knowledge-center/articles/cpr-trading-and-use-of-cpr-indicator/ |

### Setup 3: VWAP First-Pullback Continuation

| Parameter | Indian-source finding | Source |
|---|---|---|
| Trend definition | Sources do not specify "X of last Y bars same side." Generic guidance: "established trend in direction of VWAP." Fyers / Choice India recommend ≥30-60 min for VWAP to stabilize | https://fyers.in/blog/how-to-build-a-profitable-vwap-strategy-for-intraday-trading/, https://choiceindia.com/blog/vwap-trading-strategy |
| Pullback proximity | Indian sources are SILENT on an exact % threshold. Choice India / Rupeezy say "price retraces TO VWAP" or touches VWAP. Some advanced systems use 0.15% | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |
| Reversal confirmation | "Bullish/bearish candle close back beyond VWAP, with volume spike" | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |
| Stop placement | **"Just below VWAP and entry candle's low for long; just above VWAP and entry candle's high for short."** Indian-source-cited verbatim | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |
| Targets | Previous swing high (long) / low (short); Fibonacci extensions for advanced traders | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |
| Active window | JM Financial: "Two golden windows: 9:30-11:30 AM and 1:30-3:00 PM." 11:30-13:30 is lunch lull (avoid) | https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis |
| Universe | High-beta / Nifty Futures / Bank Nifty / liquid F&O | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |
| Win Rate | "60% historically with 1:1 RR" cited by Rupeezy; Choice India cites 60-65% with strict trend filter | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |

### Setup 4: PDH/PDL Touch-and-Reject (Subasish Pani style)

| Parameter | Indian-source finding | Source |
|---|---|---|
| Subasish Pani primary published method | **Subasish Pani / Power of Stocks IS NOT a PDH/PDL fade source.** His core published method is the **5 EMA strategy** (Alert Candle close fully above/below 5 EMA, entry on next candle break of Alert Candle high/low, SL = Alert Candle low/high, target 1:3). PDH/PDL is taught generically in Indian retail but NOT as a named Subasish Pani strategy | https://www.myalgomate.com/product/5-ema-strategy-power-of-stocks/, https://tradingqna.com/t/backtesting-5-ema-strategy-of-power-of-stocks/131974 |
| Level proximity tolerance | **Indian sources are SILENT on a specific tolerance.** Generic "tag and rejection" with no published % threshold |  |
| Rejection candle definition | Long upper wick (for short at PDH) / long lower wick (for long at PDL); pin-bar / hammer style. Body in lower 40% of range is a US ICT-style convention, NOT explicitly Indian-source-cited |  |
| Volume rule | **Mixed.** General trading wisdom: "rejection on low/declining volume → fade is valid." But several Indian sources also say "volume spike at rejection confirms it." There is NO unanimous Indian-source-cited rule on this | https://erranteacademy.com/previous-day-high-and-low-strategy-pdh-pdl-the-magic-levels-banks-see/ (non-Indian; Indian sources silent) |
| Stop placement | "Stops just beyond the level" (wick extreme + buffer is standard). Buffer % not Indian-source-cited |  |
| Targets | VWAP is a common magnet target; opposite intraday range extreme is logical but not Indian-source-cited as a documented target rule for this specific setup |  |
| Active window | Indian sources do not name PDH/PDL specifically as a time-of-day setup. JM Financial general guidance: 10:15-14:30 is optimal "trend" window | https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis |
| Universe | Indian sources are SILENT on cap-segment for PDH/PDL specifically. The universe choice (small/mid F&O) appears to be a design inference from gap_fade_short retail-flow logic, NOT cited |  |
| Backtest WR | No Indian-source-published WR for PDH/PDL fade strategy. Subasish Pani's 5 EMA self-reports 60% — different setup |  |

### Setup 5: Closing Hour Reversal (CHR), 14:30-15:15

| Parameter | Indian-source finding | Source |
|---|---|---|
| Trend qualifier | Indian sources are SILENT on a published intraday-move % threshold. JM Financial / StockGro discuss "closing hour volatility" qualitatively only | https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis, https://www.stockgro.club/blogs/intraday-trading/intraday-closing-time/ |
| Exhaustion candle | Generic "engulfing / pin-bar with volume spike." 60% body of range is not Indian-source-cited; that figure is design-author inference |  |
| Stop multiplier | **1.5× ATR is the most-cited intraday stop multiplier in Indian sources** (Goodwill, alphaexcapital). 1.5x-2x is the "widely used range." Bank Nifty intraday specifically cites 1.5x | https://www.gwcindia.in/blog/using-atr-average-true-range-to-set-smart-stop-losses/, https://www.alphaexcapital.com/stocks/technical-analysis-for-stock-trading/trading-strategies-using-technical-analysis/atr-based-stop-loss |
| Targets | VWAP is the most-cited "EOD magnet." Pivot / 50% retrace are logical but NOT Indian-source-cited specifically for this CHR setup |  |
| Bidirectional vs short-only | **Indian sources describe MIS unwind as bidirectional in nature** (long covers from shorts squarers, short squarers from long unwinders). StockGro / sahi.com note "sharp bidirectional moves" in the closing hour. The design's switch from sub-7's short-only to bidirectional is research-supported | https://www.stockgro.club/blogs/intraday-trading/intraday-closing-time/ |
| Hard time stop | **Zerodha equity MIS auto square-off begins 15:25 (recently extended from 15:20). F&O MIS at 15:26.** Design's 15:18 hard stop is conservatively earlier than Zerodha's actual square-off | https://zerodha.com/z-connect/updates/changes-to-the-auto-square-off-timings-for-equity-and-fo, https://www.threads.com/@zerodhaonline/post/DStqLy6iIZq/ |
| Universe | Indian sources are SILENT on cap-segment for CHR specifically |  |
| Backtest WR | No Indian-source-published WR. Multiple sources call closing-hour reversal "high-risk, requires careful management" | https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis |

---

## Section 2 — Setup-by-setup comparison tables

### Setup 1: ORB-15

| Parameter | Design value | Independent finding | Source | Severity | Recommendation |
|---|---|---|---|---|---|
| Range window | 09:15-09:30 (15-min) | 15-min IS most-taught Indian default | AlgoTest, Streak | NONE | Keep |
| Entry confirmation | Full-bar close + volume ≥ 1.5× 30-day median | Indian-source-aligned (AlgoTest) | docs.algotest.in | NONE | Keep |
| Stop placement | **Mid-of-range (conservative variant)** | Indian sources teach **OPPOSITE end of range** as default; mid-range is a retail variant, not the published default | AlgoTest, Saimohanreddy | **MAJOR** | Reconsider. Mid-range stop will more frequently be hit by ATR-typical noise after a wick-only break. Recommend ship opposite-end-of-range as primary, A/B test mid-range as variant |
| Buffer | ±0.10% wick buffer | Saimohanreddy cites buffer as material to avoid 48% DD | saimohanreddy.com | NONE | Keep |
| Targets | T1=1R, T2=2R | "1.5x or 2x range" is published. T2=2R aligns | onlinenifm.com | NONE | Keep |
| Active window | 09:30-11:15 | **In-the-Money cites 09:15-11:15 outperforms all others** | inthemoneybyzerodha.substack.com | NONE | Keep — well-cited |
| Universe | F&O ~200, exclude circuit-band | Indian-source-aligned. Retail bias toward Nifty 50 / heavy index names is also defensible | AlgoTest | NONE | Keep |
| Long-bias asymmetry | Bidirectional, no long-bias flag | **In-the-Money: longs ~60% WR vs shorts ~50% WR.** Asymmetry is documented | inthemoneybyzerodha.substack.com | **MINOR** | Consider: long-only variant, OR weight shorts lower in ranking |
| Slippage realism | Implicit | **Saimohanreddy: ORB on Bank Nifty NOT profitable post-slippage/brokerage.** This is a critical concern that the design acknowledges only obliquely | saimohanreddy.com | **MAJOR** | Phase 1 NET PF threshold (1.10) must be applied AFTER full Indian fee schedule. Design says this. Verify in implementation. |

### Setup 2: Narrow CPR Trending Breakout

| Parameter | Design value | Independent finding | Source | Severity | Recommendation |
|---|---|---|---|---|---|
| "Narrow" CPR threshold | ≤ 0.40% | Indian sources cite 0.5% (loose) or 0.25%/0.2% (strict). 0.40% is between these two but **not the dominant Indian-source value** | OptionX, Chartink, TradingView CPR scripts | **MINOR** | Acceptable middle ground but consider sweep: 0.25%, 0.40%, 0.50% in Phase 1 to find optimal |
| Direction trigger | Close above TC (long) / below BC (short) | Indian-source aligned | gwcindia, OptionX | NONE | Keep |
| Stop placement | Pivot (P) | Indian-source aligned (Jainam, OptionX) | jainam.in | NONE | Keep |
| Volume filter | ≥ 1.3× 20-day median | Indian sources cite "volume confirmation" qualitatively; no specific multiplier published. 1.3× is design inference | gwcindia | **MINOR** | Defensible; document as inference |
| Targets | T1=R1, T2=R2 | OptionX cites R1/R2 explicitly | optionx.trade | NONE | Keep |
| Active window | 09:30-14:00 | Indian sources silent on cutoff. Skipping last 90 min for setup #5 is design coordination logic (defensible) |  | NONE | Keep |
| Anti-whipsaw rule | "No entry if previous 2 bars had TC/BC tag-and-reject" | **Indian sources SILENT on this rule.** It's design-author inference. May overfit |  | **MINOR** | Acceptable as filter but flag as untested addition |
| Universe | Bank Nifty + Nifty 50 (~55 names) | Indian-source-aligned. CPR is index/heavyweight tool | fisdom, optionx, jainam | NONE | Keep — strong improvement vs sub-7 |
| Expected WR | 45-55% conservative | Indian sources don't publish; design honest about the gap | — | NONE | Keep |

### Setup 3: VWAP First-Pullback Continuation

| Parameter | Design value | Independent finding | Source | Severity | Recommendation |
|---|---|---|---|---|---|
| Trend definition | ≥ 4 of last 6 5-min bars same side | **Indian sources SILENT on exact bar count.** Design-author inference |  | **MINOR** | Acceptable but document as unvalidated. Sweep 3/6, 4/6, 5/6 in Phase 1 |
| Pullback proximity | within 0.10% of VWAP | Indian sources cite "touches VWAP" qualitatively; 0.15% mentioned in advanced systems. **0.10% is design-author inference, not published** | rupeezy, choiceindia | **MINOR** | Acceptable; sweep 0.05/0.10/0.20 |
| Reversal confirmation | Range ≥ 0.20% AND volume > prior bar | Indian sources mention "candle confirmation + volume spike" but exact 0.20% range is design inference | rupeezy | **MINOR** | Document as inference |
| Stop | Pullback bar's extreme; reject if > 0.6% of price | **Rupeezy cites verbatim "just below VWAP and entry candle's low for long" — design matches** | rupeezy.in | NONE | Keep |
| 0.6% width cap | Design-author cap | **Indian sources SILENT on this cap** | — | **MINOR** | Acceptable filter but unvalidated |
| Targets | T1 = previous swing, T2 = 2× T1 | Rupeezy explicitly cites previous swing high/low. 2× is Fibonacci-extension inference but reasonable | rupeezy | NONE | Keep |
| Active window | 10:00-13:30 | **JM Financial: "9:30-11:30 AND 1:30-3:00 PM" — design's 13:30 cutoff misses afternoon golden window 13:30-15:00** | jmfinancialservices.in | **MAJOR** | Extend window to 15:00 OR explain (per design: avoid lunch-lull whipsaws — but 13:30 already past lunch lull per JM Financial). The design CUTS the second-best window short. |
| Universe | F&O liquid 200 + Bank Nifty | Indian-source-aligned (high-beta, F&O liquid) | rupeezy, fyers | NONE | Keep |
| Expected WR | 50% (haircut from 60-65%) | Conservative haircut from cited Rupeezy/Choice India | rupeezy, choiceindia | NONE | Keep |

### Setup 4: PDH/PDL Touch-and-Reject

| Parameter | Design value | Independent finding | Source | Severity | Recommendation |
|---|---|---|---|---|---|
| **Attribution to Subasish Pani** | "Subasish Pani style" | **Subasish Pani's published method is 5 EMA, NOT PDH/PDL fade.** The PDH/PDL setup is generic Indian-retail teaching, not a Power of Stocks named strategy. Capital.com (cited in design) is **non-Indian** (UK forex site) | myalgomate, tradingqna | **MAJOR — attribution error** | Either re-attribute (drop "Subasish Pani style" → "Indian retail PDH/PDL fade") OR rebuild as actual Subasish Pani 5 EMA strategy |
| Capital.com citation | Cited as Indian-derivative | **Capital.com is UK-based forex retail site, NOT an Indian source.** Violates own RULE 1 of design | capital.com | **CRITICAL — citation rule violation** | Replace with actual Indian source. The strict "Indian sources only" rule was violated. |
| Level proximity | within 0.10% | Indian sources SILENT | — | **MINOR** | Sweep 0.05/0.10/0.15 |
| Rejection candle: body < 40%, wick > 1.5× body | Design value | **Indian sources SILENT on these specific ratios.** This is design-author inference / appears optimized | — | **MINOR / over-fit risk** | Document as inference; sweep |
| Volume rule | "NOT ≥ 1.5× recent" (absence of volume) | **Indian sources are CONFLICTED.** Some sources say "rejection on low volume confirms fade"; others say "volume spike at rejection." Design picks one side without strong citation | erranteacademy (non-Indian) | **MAJOR** | Either A/B test both polarities, or document unresolved disagreement. Picking "absence of volume" without clear Indian citation is a coin flip. |
| First-tag-only rule | Design value | Indian sources SILENT but logical ("second tag = breakout") | — | NONE | Defensible |
| Stop | Wick extreme + 0.10% buffer | Indian sources SILENT on buffer %, generic "beyond level" | — | **MINOR** | Defensible |
| T1 = VWAP, T2 = opposite intraday range extreme | Design value | T1 VWAP is Indian-source aligned (mean-revert magnet). T2 = opposite range extreme is design inference | (general) | NONE | Keep |
| Active window | 10:00-14:30 | Indian sources SILENT on time window for PDH/PDL | — | **MINOR** | Defensible by JM Financial general intraday window |
| Universe | Small/mid F&O ~100 | **Indian sources SILENT on cap-segment specifically for PDH/PDL.** Design transfers logic from gap_fade_short. Plausible but unvalidated | — | **MINOR / over-fit risk** | Document as inference; consider including some large caps in Phase 1 sweep |
| Expected WR | 45% conservative | No published WR; honest haircut from Subasish Pani 5 EMA's claimed 60% | — | NONE | Keep |

### Setup 5: Closing Hour Reversal (CHR)

| Parameter | Design value | Independent finding | Source | Severity | Recommendation |
|---|---|---|---|---|---|
| Trend qualifier | Move ≥ 1.5% AND range > 1.5×ATR(14) | **Indian sources SILENT on specific %.** Design-author inference | — | **MINOR / over-fit risk** | Sweep 1.0%, 1.5%, 2.0% |
| Exhaustion candle: body ≥ 60% of range, vol ≥ 1.3× | Design value | Indian sources SILENT on specific ratios. Design inference | — | **MINOR / over-fit risk** | Document as inference; sweep |
| Direction (bidirectional) | Bidirectional | StockGro / sahi.com describe closing hour as "bidirectional sharp moves." Design fix from sub-7 short-only is research-supported | stockgro.club | NONE | Keep |
| Stop multiplier | 1.2× ATR(14, 5-min) | **Indian-source-cited 1.5× ATR is more standard.** Design's 1.2× is between sub-7's 0.8× (too tight) and standard 1.5×. Sources also cite 1.5-2× as "widely used range" | gwcindia.in, alphaexcapital | **MINOR** | Consider 1.5× as primary; 1.2× as A/B variant. The design is closer than sub-7 but not at industry-standard. |
| T1 = VWAP, T2 = Pivot or 50% retrace | Design value | VWAP is Indian-source-aligned EOD magnet. Pivot/50% retrace is logical inference | — | NONE | Keep |
| Hard time stop | 15:18 | **Zerodha equity MIS auto square-off is 15:25 (extended). 15:18 leaves 7 min margin.** The choice is conservative and acceptable, but 15:20-15:22 would still leave safe square-off margin and capture more time | zerodha.com/z-connect, threads.com/@zerodhaonline | **MINOR** | Consider 15:20 or 15:22 instead of 15:18 |
| Universe | F&O liquid ~200 | Indian sources SILENT on cap-segment | — | NONE | Defensible |
| Expected WR | 40-45% conservative | No published WR; honest | — | NONE | Keep |
| Reuses sub-7 mis_unwind_short architecture | Acknowledged in design | Sub-7 failed (n=326). The design's structural fix (bidirectional + wider stop) is logical but **risk of inheriting same problem** if window/qualifier still wrong | — | **MAJOR (RISK)** | Mandatory: do NOT re-use any sub-7 mis_unwind_short candidate filtering logic. Build CHR cleanly from scratch. |

---

## Section 3 — Critical issues summary (ranked by expected PnL impact)

### CRITICAL #1: Setup 4 cites Capital.com (UK forex site) as a "Subasish Pani derivative" Indian source

The design's RULE 1 says "Every parameter cites an Indian source: URL, book chapter, or YouTube video timestamp. No opinion-based numbers." Capital.com is a UK-based retail-forex broker, not an Indian source. The Subasish Pani attribution is also wrong — his published method is the 5 EMA strategy (Alert Candle, body fully outside 5 EMA, target 1:3), not a PDH/PDL fade. **This is the only CRITICAL because it violates the design's own foundational rule.**

**Fix options:**
- (a) Re-attribute Setup 4 from "Subasish Pani style" to "Indian retail PDH/PDL fade" and replace Capital.com with an Indian source (Errante is also non-Indian; need to find a real Indian PDH/PDL tutorial e.g., Groww, ChartMantra, TradingQnA threads)
- (b) Re-design Setup 4 as the actual Subasish Pani 5 EMA strategy (Alert Candle entry, target 1:3)

### CRITICAL #2: Setup 4 volume rule (absence of volume) is contested in Indian sources

The design picks "rejection bar volume NOT ≥ 1.5× recent" as a positive trigger. Indian retail sources are split: some say "low volume on rejection = fade is valid (no breakout conviction)"; others say "volume spike at rejection confirms it." Without an authoritative Indian citation for the polarity chosen, this rule is a coin flip and could systematically lose money if the wrong polarity is chosen for Indian retail-flow dynamics. **Phase 1 PF 1.10 floor is at material risk if the polarity is wrong.**

**Fix:** Ship both polarities as A/B variants in the Phase 2 OCI capture and let the data decide.

---

## Section 4 — Major issues summary

### MAJOR #1: Setup 1 mid-of-range stop is NOT the published Indian default

Industry standard per AlgoTest, Saimohanreddy, In-the-Money is opposite-end-of-range stop. The design's mid-range "conservative variant" optimizes for tighter R:R but will be hit more often by ATR-typical intraday noise after a wick-only breakout. Design rationale ("PF threshold tolerates lower R:R if WR is higher") assumes WR will be higher with mid-range stop, but this is unproven for Indian intraday. **Recommend ship opposite-end as primary.**

### MAJOR #2: Setup 1 ORB on Bank Nifty has documented post-slippage failure

Saimohanreddy's published Bank Nifty 2015-2023 backtest shows ORB is "not profitable at all" once 0.03% slippage and ₹20/lot brokerage are applied. The design uses Indian fee model (per Section 10) but doesn't acknowledge this specific Bank Nifty result. **Risk:** the strategy may pass on equity universe but fail on Bank Nifty constituents specifically. Consider excluding Bank Nifty index futures and treating Bank Nifty constituents (HDFCBANK, ICICIBANK, etc.) only after Phase 1 evidence.

### MAJOR #3: Setup 3 active window cuts off second JM Financial golden window

JM Financial explicitly names "9:30-11:30 AM and 1:30-3:00 PM" as the two golden windows. Design's 10:00-13:30 cutoff stops at 13:30 — exactly where JM Financial says the SECOND golden window starts. Lunch-lull rationale (cited in design) actually applies to 11:30-13:30, which the design correctly avoids by starting at 10:00... but cutting at 13:30 misses 13:30-15:00 trending opportunity entirely.

**Fix:** Extend Setup 3 window to 15:00 (or 14:30 to avoid overlap with Setup 5).

### MAJOR #4: Setup 5 inherits sub-7 mis_unwind_short risk

Sub-7's mis_unwind_short never accumulated enough sample (n=326). Setup 5's design fixes the symptom (bidirectional + wider stop) but is structurally the same idea. **Risk:** if the underlying assumption (MIS-unwind pressure produces directional reversion) is what's wrong — not the parameters — then Setup 5 will fail like sub-7. The design should explicitly include a Phase 1 fail-fast trip-wire: if Setup 5 produces < 200 trades in the first 100 OCI sessions, kill it.

### MAJOR #5: Setup 4 attribution to Subasish Pani

Even setting aside Capital.com (CRITICAL above), labeling this "Subasish Pani style" is misleading. His audience-known method is 5 EMA. Mis-attribution can lead to false confidence in the setup's prior validation. The design should either rename or rebuild.

---

## Section 5 — Minor issues, missed rules, over-fitting concerns

### Minor design choices (defensible but unvalidated)

1. **Setup 1 long-bias asymmetry not exploited.** In-the-Money cites long ~60% WR vs short ~50%. Design treats both directions symmetrically. (Minor — could be Phase 1 enhancement.)
2. **Setup 2 narrow-CPR threshold 0.40%.** Indian sources cite 0.5% (loose) or 0.25% (strict). Design picks 0.40%. (Minor — sweep in Phase 1.)
3. **Setup 2 anti-whipsaw "previous 2 bars" rule.** Indian sources SILENT. (Minor over-fit risk.)
4. **Setup 3 trend definition "≥ 4 of 6 bars."** Indian sources SILENT on bar counts. (Minor over-fit risk.)
5. **Setup 3 pullback proximity 0.10% / range ≥ 0.20% / stop cap 0.6%.** All design-author inferences. (Minor — sweep.)
6. **Setup 5 trend qualifier 1.5%.** Indian sources SILENT. (Minor over-fit risk.)
7. **Setup 5 exhaustion candle 60% body / 1.3× volume.** Specific ratios design-inferred. (Minor over-fit risk.)
8. **Setup 5 stop 1.2× ATR.** Indian-source-cited industry standard is 1.5× ATR; design's 1.2× is closer to standard than sub-7 but not on it. (Minor.)
9. **Setup 5 hard-stop 15:18.** Zerodha auto square-off is 15:25; 15:18 over-conservative. Could use 15:20-15:22. (Minor.)

### MISSED RULES — important conditions Indian sources document but design doesn't include

1. **Expiry-day exclusion.** Indian sources unanimously document expiry-day intraday volatility distortions. Bank Nifty/Nifty options expire Tuesday (post-Sep 2025 SEBI rule change from Thursday). On expiry days bid-ask spreads widen after 14:30; OI-driven sharp bidirectional moves are common. **Setup 5 (CHR 14:30-15:15) is in the danger zone.** No expiry-day filter is mentioned in design. → **Add expiry-day exclusion or expiry-day separate ruleset.** Sources: hdfcsky.com, algotest.in/blog/bank-nifty-expiry-day, sahi.com/blogs/5-rules-every-trader-must-follow.
2. **Gap-day filter for Setup 1 (ORB).** Indian sources (gwcindia, truedata, motilaloswal) describe gap > 0.5% (Bank Nifty) or > 50-60 points (Nifty) as requiring a different playbook than ORB. Sub-7 has a separate gap_fade_short detector for this. Design should explicitly route gap-days to gap_fade_short and exclude ORB on gap days, otherwise both detectors fire on the same setup. → **Add cross-detector exclusion: if gap > X% then ORB-15 disabled for that day.**
3. **Circuit-band proximity exclusion.** Setup 1 mentions "exclude circuit-band stocks (price within 2% of upper/lower 20% circuit)." Setups 2-5 don't mention this. Indian sources (StockGro, 5paisa) note that once a stock circuits, intraday positions become forced delivery. → **Apply circuit-band exclusion universe-wide, not just to ORB.**
4. **Pre-open call-auction price ≠ regular session open.** ORB-15 range starts 09:15. The opening 09:15 print is from pre-open call auction (09:00-09:08). Design assumes 09:15 5-min bars are normal continuous trading, but the first 5-min bar (09:15-09:20) often contains the call-auction print + first continuous trades, creating a synthetic wick. → **Consider starting ORB range at 09:20 instead of 09:15** OR **trim opening synthetic wick.**
5. **Lot-size minimum (₹15 lakh per SEBI Nov 2024 rule).** Affects F&O contract sizing for Setup 5 if F&O instruments are traded. Equity is unaffected.

### OVER-FITTING RISKS

1. **Setup 2 anti-whipsaw "previous 2 bars" lookback** — specific number, no Indian citation.
2. **Setup 3 trend definition "4 of 6 bars"** — specific ratio, no Indian citation.
3. **Setup 4 entire rejection-candle spec** (body < 40%, wick > 1.5× body, proximity 0.10%) — multiple specific ratios stacked, none Indian-source-cited.
4. **Setup 5 trend qualifier 1.5% AND range > 1.5×ATR(14)** — compound specific filter.
5. **Setup 5 exhaustion 60% body + 1.3× volume** — compound specific filter.

The design has internal consistency (multiple "1.3× volume" choices across Setups 2, 5) which is good for cross-comparison but creates the appearance of unified "magic numbers" rather than per-setup-cited values. Recommend running parameter sweeps in Phase 1 to confirm these are not coincidental.

---

## Section 6 — Final recommendation

### Verdict: **REVISE BEFORE IMPLEMENTATION**

**Reasoning:** The design is research-driven on Setups 1, 2, 3, and 5 with defensible interpretations. Setup 4 has a CRITICAL citation-rule violation (Capital.com is non-Indian; Subasish Pani attribution is wrong) and a contested volume polarity. The other major issues (Setup 1 mid-range stop, Setup 3 window cut, Setup 5 inheriting sub-7 risk) are addressable without redesign.

### Per-setup verdicts

| # | Setup | Verdict | Top issue |
|---|---|---|---|
| 1 | ORB-15 | **REVISE** (minor) | Mid-of-range stop is non-standard; ship opposite-end as primary |
| 2 | Narrow CPR Trending | **SHIP with caveats** | Sweep 0.25/0.40/0.50 narrow threshold; drop unvalidated anti-whipsaw rule or document |
| 3 | VWAP First-Pullback | **REVISE** (window) | Extend active window past 13:30 to capture JM Financial's second golden window |
| 4 | PDH/PDL Touch-and-Reject | **REVISE** (citation + polarity) | Drop Subasish Pani attribution; replace Capital.com with Indian source; A/B test volume polarity |
| 5 | Closing Hour Reversal | **SHIP with caveats** | Tighten ATR multiplier toward 1.5× standard; add fail-fast n<200 trip-wire vs sub-7 risk |

### Top 3 critical issues (cross-cutting)

1. **Setup 4 cites Capital.com (UK forex) as Indian source — violates RULE 1.** Re-cite or redesign.
2. **Setup 4 volume polarity (absence of volume) is contested in Indian literature.** Could systematically lose money if polarity is wrong. A/B test.
3. **Missed rule: expiry-day exclusion.** Setup 5 in particular runs in the closing-hour window where post-Sep-2025 Tuesday expiries cause distorted moves and widened spreads. Add expiry-day filter or treat expiry days separately.

### Top 3 missed rules

1. **Expiry-day filter** (affects all setups, especially #5).
2. **Gap-day cross-detector exclusion** between ORB-15 and gap_fade_short.
3. **Circuit-band exclusion universe-wide** (not just ORB).

### Setups recommended to drop entirely

**None.** All five are addressable. However:
- Setup 4 should be substantially revised. If revising Capital.com citation cannot be done (no Indian source supports the exact rejection-candle ratios + volume rule), recommend **rebuild Setup 4 as actual Subasish Pani 5 EMA strategy**, which has authoritative Indian sources (myalgomate, algotest, tradingqna, ezquant).
- Setup 5 should ship with an explicit fail-fast trip-wire (n<200 trades in first 100 sessions → kill) given sub-7 mis_unwind_short precedent.

---

## Sources cited (Indian-only, with one flagged exception)

- [In-the-Money by Zerodha — All About ORB](https://inthemoneybyzerodha.substack.com/p/all-about-opening-range-breakout)
- [AlgoTest — Range Breakout Strategy](https://docs.algotest.in/sample-algo-trading-strategies/range-breakout-strategy/)
- [Saimohanreddy — ORB Backtest Bank Nifty](https://saimohanreddy.com/orb-backtest-on-banknifty/)
- [Zerodha Varsity — Dow Theory Part 2](https://zerodha.com/varsity/chapter/dow-theory-part-2/)
- [TradingQnA — ORB 15 mins](https://tradingqna.com/t/orb-15-mins-break-out-or-break-down/52978)
- [TradingQnA — Backtesting 5 EMA Strategy of Power of Stocks](https://tradingqna.com/t/backtesting-5-ema-strategy-of-power-of-stocks/131974)
- [Myalgomate — 5 EMA Strategy by Power of Stocks](https://www.myalgomate.com/product/5-ema-strategy-power-of-stocks/)
- [EzQuant — 5 EMA Strategy Walkthrough](https://ezquant.in/5-ema-strategy-parameters-walk-through/)
- [Goodwill — CPR Bank Nifty Setups](https://www.gwcindia.in/blog/top-5-indicator-setups-for-bank-nifty-traders-in-india/)
- [Goodwill — Using ATR for Smart Stop-Losses](https://www.gwcindia.in/blog/using-atr-average-true-range-to-set-smart-stop-losses/)
- [Goodwill — Gap Up Gap Down NSE](https://www.gwcindia.in/blog/gap-up-gap-down-how-to-trade-market-gaps-on-nse-with-confidence/)
- [Fisdom — Central Pivot Range](https://www.fisdom.com/central-pivot-range/)
- [OptionX Journal — CPR Explained](https://optionx.trade/blogs/central-pivot-range-cpr-explained)
- [Jainam — CPR in Trading](https://www.jainam.in/blog/cpr-in-trading/)
- [Samco — CPR Indicator](https://www.samco.in/knowledge-center/articles/cpr-trading-and-use-of-cpr-indicator/)
- [Rupeezy — VWAP Trading Strategy](https://rupeezy.in/blog/vwap-trading-strategy-intraday-options)
- [Choice India — VWAP](https://choiceindia.com/blog/vwap-trading-strategy)
- [Fyers — VWAP Strategy](https://fyers.in/blog/how-to-build-a-profitable-vwap-strategy-for-intraday-trading/)
- [Tradejini — Scalping Nifty Options VWAP](https://www.tradejini.com/blogs/introduction-to-scalping-in-nifty-options)
- [JM Financial — Intraday Trading Time Analysis](https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis)
- [Zerodha Z-Connect — MIS Auto Square-Off Update](https://zerodha.com/z-connect/updates/changes-to-the-auto-square-off-timings-for-equity-and-fo)
- [Zerodha Threads — MIS Auto Square-off Times](https://www.threads.com/@zerodhaonline/post/DStqLy6iIZq/)
- [StockGro — Intraday Closing Time](https://www.stockgro.club/blogs/intraday-trading/intraday-closing-time/)
- [HDFC Sky — Bank Nifty Expiry](https://hdfcsky.com/blogs/share-market/bank-nifty-expiry)
- [AlgoTest Blog — Bank Nifty Expiry Day](https://algotest.in/blog/bank-nifty-expiry-day/)
- [Sahi.com — Nifty Expiry Day Rules](https://www.sahi.com/blogs/5-rules-every-trader-must-follow-on-nifty-options-expiry-day)
- [TrueData — Gap Up Gap Down Strategy](https://www.truedata.in/blog/Gap-up-and-gap-down-intraday-trading-strategy)
- [TradingDirection — Narrow CPR Stocks](https://www.tradingdirection.in/blog/narrow-cpr-for-intraday-trading)
- [Chartink — Narrow CPR Scanner](https://www.scribd.com/document/659685125/Chartink-Tomorrow-0-2-narrow-CPR-range-Technical-Analysis-Scanner)
- [TradingView India — CPR Scripts](https://in.tradingview.com/scripts/cpr/)
- [Booming Bulls — VWAP Pivots Stochastic RSI Strategy](https://www.tradingview.com/script/TQzcMHrR-Booming-Bull-VWAP-Stoch-RSI-Multi-Pivot/)
- (Flagged) Capital.com — UK forex site, **non-Indian**: cited in design Setup 4, recommend removal.
