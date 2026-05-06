# Round-3 Param Audit & Variant Search

**Date:** 2026-05-06
**Scope:** Audit the parameter values used in the 3 sub-9 round-3 sanity checks against published Indian-pro-algo conventions, and identify common variants we did NOT test. All 3 setups RETIRED with NET PF 0.62-0.65 — the question is whether the parameters were standard (i.e., the failure is real), or off-the-edge (i.e., a variant might rescue).
**Time-box:** 30 min wall-clock.
**Predecessor:** `specs/2026-05-06-sub-project-9-design-decisions-research.md` (Q1-Q8 convention research already done; this audit re-checks the THRESHOLD values, not the entry-mechanic choices).

---

## Summary

- **VWAP-deviation mean-reversion** — Params are within published Indian-retail-algo convention bands (z=2.0 mid-range of 1.5-2.5; wick=0.5 mid-range of 0.5-0.7; 9-bar stdev ≈ 45 min, in line with Goodwill/Stoxra). One realistic variant we did NOT test: VIX-regime gate (Goodwill explicitly conditions VWAP-revert on VIX < 14). Worth a re-sanity ONLY if a daily-VIX feed is wired in.
- **Index-stock divergence revert** — Params are convention-aligned (z=2.0 in Religare/Wright 1.5-2.5 range; 30-day daily lagged beta is canonical QuantInsti EPAT; 30-bar rolling window matches stat-arb literature). NOT recommended for re-sanity: divergence revert at retail timeframe is a known weak edge in F&O 200 (institutional arbitrage already harvests it). One variant with marginal precedent: simpler `stock_ret - nifty_ret` (no beta) — falsification test the brief itself flagged. Cheap to run.
- **Volume-spike exhaustion reversal** — Params are convention-aligned (vol_z=3.0 is the canonical Streak/TradingView default; wick=0.6 mid-range of 0.5-0.7). One variant with strong published precedent we did NOT test: **morning-only window (09:30-11:00) for the SHORT side** — TradeFundrr/Wright explicitly flag opening-hour exhaustion as a separate, higher-quality regime. But this would COLLIDE with `gap_fade_short`/`circuit_t1_fade_short` (already shipped in 11:00-15:00) and was deliberately excluded by round-3's Gate C. No rescuing variant inside the round-3 11:00-15:05 window.

**Honest read:** all 3 retirements are real failures, not param errors. The fee-dominated PF=0.62-0.65 implies gross PF ≈ 0.85-0.95 — well below the 1.10 threshold even before fees. Only the VWAP variant has a published "regime gate" (VIX < 14) we haven't tested; that is the single rescuing experiment worth budget. The other two are dead.

---

## Per-setup

### 1. VWAP-deviation mean-reversion

**Our params:**
- z-threshold = ±2.0
- wick fraction ≥ 0.5 (50% of bar range)
- intraday stdev window = 9 bars (~45 min)
- active window = 11:00-15:00 IST
- T1 = VWAP touch, T2 = VWAP ± 0.5×stdev
- Time stop = 8 bars (~40 min) or 15:10 hard stop
- Stop = trigger high/low + 0.3% buffer, min 0.6%
- Universe = F&O 200, mid_cap + small_cap, ADV ≥ Rs 3 Cr
- Risk = Rs 1000/trade, 50/50 tiered with breakeven trail after T1
- Result: n=428, NET PF=0.645

**Published convention range:**
- **z-threshold:** Goodwill Securities VWAP guide and Stoxra retail-algo template cite **1.5-2.5 sigma** as the band; 2.0 is mid-range. Streak's published VWAP+Supertrend templates use loosely "stretched" without a numeric z (visual). Share India and Zerodha Varsity supplementary indicators discuss VWAP magnetism without naming a numeric stretch threshold. (sources: Goodwill VWAP blog; Stoxra top-5 algo blog; Zerodha Varsity supplementary).
- **wick fraction:** Wright Research and TradingView Pine "rejection wick" templates converge on **50-70% of bar range**; 50% is the looser end. (sources: Wright top-5 reversal blog; TradingView India spike scripts).
- **stdev window:** No single dominant Indian convention. 9-bar (~45 min, the band has accumulated since 09:15) and 20-bar are both observed. AlgoTest/Streak intraday backtest literature uses 14-20 bar lookbacks for most rolling indicators. Our 9 is on the shorter end but defensible (matches the brief's "VWAP stable after 21 bars accumulated by 11:00").
- **active window:** 11:00-15:00 IST is the conventional "post-opening-flow, pre-MIS-square" middle-of-day window per Tradejini, JM Financial, IIFL.
- **T1=VWAP touch:** standard mean-reversion thesis exit. Goodwill, Choice, Rupeezy all describe VWAP-touch as the textbook target.
- **T2=VWAP ± 0.5×stdev:** structural overshoot anchor — has structural-precedent in pairs-trading literature; "0.5σ" is a common overshoot target in stat-arb (QuantInsti EPAT pairs).

**Match assessment:** **Within convention.** Every parameter is mid-range or defensibly at-edge. NOT off-the-edge.

**Variants we missed (with published precedent):**

1. **VIX < 14 daily gate (Goodwill blog).** The Goodwill VWAP Pro Tool guide *explicitly* conditions VWAP-revert effectiveness on India VIX < 14 ("modal regime in 2024-26"). Our sanity ran on full 2024 with no VIX gate. India VIX averaged 13-15 in 2024 with spikes to 20+ during election week (Jun 4) and Oct correction. A VIX-gated re-sanity could materially shift PF — the brief itself §6 flagged this as the highest-risk untested filter.

2. **Tighter z=2.5 + wider wick=0.6 (signal-purity variant).** Stoxra and TradingView retail-algo templates both reference "stricter sigma + cleaner wick" as a quality variant. Sample size drops 60-70% per the brief, but PF could lift if the failure was edge-decay-not-mechanism.

3. **VWAP touch as the sole target (no T2 overshoot).** Choice India and Rupeezy VWAP intraday guides describe single-target VWAP-touch as the textbook setup; T2 overshoot is the QuantInsti pairs-trading bolt-on. A simpler T1-only variant with the breakeven trail exits sooner — could change fee-load profile (the setup is fee-dominated).

---

### 2. Index-stock divergence revert

**Our params:**
- spread z-threshold = ±2.0
- 30-day rolling daily beta, lagged 1 day (uses up to D-1)
- spread z-score window = 30 5m bars (rolling-N closing-to-closing returns, per Q2 design decision)
- active window = 11:00-14:30 IST
- NIFTY uptrend gate (LONG only): NIFTY 5m close > NIFTY 5m EMA20
- T1 = spread reverts to z=0; T2 = overshoots to ∓0.5
- Time stop = 12 bars (~60 min) or 15:10 hard stop
- Stop = trigger high/low + 0.3% buffer, min 0.6%
- Universe = F&O 200, no cap exclusion
- Reversal wick ≥ 50% on confirmation bar
- Result: n=2387, NET PF=0.619

**Published convention range:**
- **spread z-threshold:** Religare divergence-strategy and Wright Research top-5 reversal both reference **1.5-2.5 sigma** as the band; 2.0 mid-range. (sources: Religare intraday-with-NIFTY blog; Wright top-5 reversal blog).
- **beta window + lag:** QuantInsti EPAT cointegrated-pairs project uses **252-day train / 21-day test** OLS hedge ratio (much longer than ours), held constant for the test period. NSE academic hedge-ratio research (NSE comp-paper-171) uses similar long-window daily betas. Our 30-day lagged daily beta is **shorter than the Indian-academic norm** — but is in line with what retail-algo platforms (Stratzy, Enrich Money) describe as "rolling beta" (typically 30/60/90-day on Indian retail platforms). 30 is the lower edge of acceptable. (sources: QuantInsti EPAT pairs blog; NSE comp-paper-171; Stratzy common-algo-strategies).
- **spread z-score window:** 30 5m bars (~150 min) within session. Stat-arb literature converges on 20-30 bar rolling windows for intraday spread z-scores. Our 30 is at the upper edge — but the brief's locked Q2 decision (rolling N-bar closing-to-closing, NOT since-open) is the QuantInsti EPAT convention.
- **active window:** 11:00-14:30 — earlier-than-VWAP-revert end is sensible (cross-asset signals contaminate after 14:30 due to MIS unwind).
- **NIFTY uptrend gate (LONG):** EMA20 on 5m is the canonical Tradejini/Stockpathshala/Rupeezy intraday-trend gate. Per Q5 design decision (already researched). Convention-aligned.
- **Reversal wick ≥ 0.5:** mid-range of Wright/TradingView 0.5-0.7.

**Match assessment:** **Within convention, with one parameter at the lower edge** (30-day beta is shorter than QuantInsti EPAT 252-day, longer than retail blog 14-day). NOT off-the-edge.

**Variants we missed (with published precedent):**

1. **Beta-free spread (`stock_ret − nifty_ret` only).** The brief §6 falsification criterion #5 ALREADY flags this as the cheap test: "if simpler form produces PF ≥ 1.10 while beta-adjusted does NOT, beta is hurting." We did not run this variant. Religare and Enrich Money both publish the simpler form first, beta-adjusted second. Cheap re-run.

2. **Sectoral index instead of NIFTY 50.** Round-3 §4 (DEFER candidate) noted that sector-index divergence (NIFTY IT vs constituent) has equal published precedent (TradingView TechnicalExpress sector-rotation posts × 3). A constituent's divergence from its **sector** is mechanically distinct from divergence from the broad index — sector noise filters out, isolates true single-stock idiosyncratic moves. Strong published precedent.

3. **5-min EMA10 instead of EMA20 for NIFTY uptrend gate (LONG).** Tradejini's NIFTY scalping playbook references 9-EMA on 3m/5m as faster; EMA20 on 5m is the slower variant. EMA10 would catch more LONG signals — could shift the LONG-side sample.

---

### 3. Volume-spike exhaustion reversal

**Our params:**
- vol_z ≥ 3.0 (vs 20-bar mean, excluding current bar)
- wick fraction ≥ 0.6 (60% of bar range)
- vol window = 20 bars
- active window = 11:00-15:05 IST
- T1 = 1.0R, T2 = 2.0R
- Time stop = 6 bars (~30 min) or 15:15 hard stop
- Stop = trigger high/low + 0.3% buffer, min 0.6%
- Universe = F&O 200, mid_cap + small_cap, ADV ≥ Rs 3 Cr
- Reversal-confirmation bar = next bar bearish (SHORT) / bullish (LONG)
- Bidirectional from day 1 per Q3 design decision
- Result: n=451, NET PF=0.646

**Published convention range:**
- **vol_z threshold:** Streak's built-in volume-anomaly + reversal-candle scanner uses **z=2.0-3.0** as the band; **3.0 is the canonical "rare volume spike" floor**. TradingView Pine spike scripts reference 2.5-3.5 sigma. Wright Research uses "3x average volume" (rough z-equivalent ≈ 3.0). Our 3.0 is mid-range. (sources: Streak strategies page; TradingView India spike scripts; Wright top-5 reversal).
- **wick fraction:** Wright Research and TradingView volume-spike-reversal scripts converge on **50-70%**; 60% mid-range. (same sources).
- **vol window:** 20 bars (~100 min) is the canonical short-rolling-volume window. TrendSpider volume-spikes documentation uses 20-bar baseline. Indian retail-platform scanners use 14-20 bar. Convention-aligned.
- **active window 11:00-15:05:** matches the brief's "exclude opening institutional flow + MIS unwind contamination" framing. Tradejini bracket-order convention recommends entry latest by 15:00-15:05 with hard stop at 15:15-15:20.
- **R-multiple targets (T1=1R, T2=2R):** Tradejini bracket-order and Share India intraday-stop-loss publish 1R/2R as the textbook risk-reward template. Stratzy index-scalper RMS describes the same. (sources: Tradejini bracket-order blog; Share India intraday-stop-loss blog; Stratzy index-scalper RMS).
- **Time stop = 6 bars (~30 min):** matches the brief's pattern-decay window. Tradejini/Stratzy reference 30-min time-stops as standard for short-horizon mean-revert.

**Match assessment:** **Within convention.** Every parameter is mid-range. NOT off-the-edge.

**Variants we missed (with published precedent):**

1. **Morning-only (09:30-11:00) variant.** TradeFundrr's "Volume Exhaustion Entry Setup" guide explicitly identifies opening-hour blowoff bars as the highest-quality regime for this mechanic — 09:30-11:00 in Indian session. We deliberately excluded this window per round-3 Gate C (collision with `gap_fade_short`/`circuit_t1_fade_short`). The published-best variant is therefore unavailable to us by design.

2. **Looser vol_z=2.5 (Streak default).** Streak's own scanner default sits at z=2.5, not 3.0. Lower threshold = larger sample, possibly cleaner statistics. Brief §6 sensitivity already flagged z=2.5 / 3.5 as report-only sensitivity. Cheap re-run if we re-open this candidate.

3. **Multi-bar volume confirmation (3-bar cluster).** Mastertrust's beginner intraday-strategies guide references "3 consecutive high-volume bars" as a stronger filter than a single-bar spike. Single-bar spike + wick is our test; 3-bar cluster + wick is the institutional-flow-detection variant. No published Indian-retail-algo platform implements this exactly, but the mechanic is described in Indian retail education content.

---

## Recommendation

### High-likelihood-of-changing-outcome variants (worth re-sanity)

1. **VWAP-deviation with VIX < 14 daily gate.** Single, identifiable, published-convention filter we did NOT apply. Goodwill Securities source explicitly conditioned VWAP-revert on VIX < 14. India VIX averaged 13-15 in 2024 with significant spikes — gating to VIX < 14 days could lift PF if the high-VIX days are the loss-driver. Pre-requisite: ingest daily India VIX close (1 file/year, ~250 rows). **Budget: 30 min ingestion + 15 min re-sanity = 45 min.**

2. **Index-stock divergence WITHOUT beta (`stock_ret - nifty_ret`).** Brief §6 falsification criterion already flagged this. Cheap to compute (delete the beta term from the spread calc). **Budget: 10 min code edit + 5 min re-run = 15 min.**

### Low-likelihood-of-changing-outcome variants (academic only, skip)

- VWAP z=2.5 / wick=0.6 stricter variant — sample drops 60-70%; PF=0.65 → 0.75 is plausible but still RETIRE.
- Volume-spike z=2.5 looser variant — larger sample on a known-fee-dominated setup just adds more fee drag.
- Sectoral-index divergence — different mechanic, more infrastructure (sector composition mapping, separate sector indices). Re-research, not a re-sanity.
- Volume-spike multi-bar cluster — invented variant with thin precedent.

### ONE concrete next step

**Run the VIX-gated VWAP-deviation re-sanity ONLY.** 45 min budget. If VIX < 14 days produce PF ≥ 1.10 (n must be ≥ 200 — VIX < 14 will retain ~70% of 2024 sessions), promote VWAP-revert with VIX-gate to brief refresh. If still PF < 1.0, declare round-3 fully retired with conviction.

The other two (divergence revert, volume-spike) are dead — params are textbook-aligned, mechanics are well-explored on Indian retail platforms, fees dominate even at correct thresholds. Beta-free divergence variant is cheap enough to run alongside (15 min) but not expected to rescue: divergence revert at retail-MIS scale on F&O 200 is a known-arbitraged-away edge per the cross-asset literature.

**Reasonable scenario:** all 3 truly retired; confidence-built by ruling out the one unhandled regime variable (VIX) for VWAP. That is the disciplined exit from sub-9 round-3.

---

## Sources cited (consolidated)

| # | Source | URL |
|---|---|---|
| 1 | Goodwill Securities — VWAP Pro Tool (VIX-conditioned) | https://www.gwcindia.in/blog/vwap-volume-weighted-average-price-a-pro-tool-for-intraday-traders/ |
| 2 | Stoxra — Top-5 algorithmic strategies | https://stoxra.com/blog/top-5-algorithmic-trading-strategies-professional-traders |
| 3 | Zerodha Varsity supplementary indicators | https://zerodha.com/varsity/chapter/supplementary-notes-1/ |
| 4 | Share India — VWAP intraday | https://www.shareindia.com/knowledge-center/intraday-trading/how-to-use-vwap-indicator-for-intraday-trading |
| 5 | Streak strategies page (built-in templates) | https://www.streak.tech/strategies |
| 6 | Streak — Signal/Trade candle | https://blog.streak.tech/help/what-is-signal-candle-trade-candle/ |
| 7 | AlgoTest — Candle Close Implementation | https://docs.algotest.in/product-blogs/candle-close-implementation-on-algotest/ |
| 8 | Wright Research — Top-5 reversal | https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/ |
| 9 | Religare — Intraday with NIFTY + TradingView | https://www.religareonline.com/blog/intraday-trading-strategies-with-nifty-tradingview-indicators/ |
| 10 | TradingView India spike scripts | https://in.tradingview.com/scripts/spike/ |
| 11 | TradingView India volumebreakout scripts | https://in.tradingview.com/scripts/volumebreakout/ |
| 12 | QuantInsti EPAT — Cointegrated pairs India | https://blog.quantinsti.com/cointegrated-pairs-trading-indian-equity-market-epat-project/ |
| 13 | NSE — Optimal Hedge Ratio research | https://nsearchives.nseindia.com/content/research/comppaper171.pdf |
| 14 | Stratzy — Common-algo-strategies | https://stratzy.in/blog/common-algo-trading-strategies-and-examples/ |
| 15 | Stratzy — Index Scalper RMS | https://stratzy.in/blog/index-scalper-risk-management-system/ |
| 16 | Tradejini — Bracket Orders | https://www.tradejini.com/blogs/intraday-trading-with-bracket-orders-on-tradejini-cubeplus |
| 17 | Tradejini — Risk management momentum | https://www.tradejini.com/blogs/importance-of-risk-management-in-momentum-trading-part-2 |
| 18 | Tradejini — Nifty scalping | https://www.tradejini.com/blogs/a-technical-scalping-strategy-for-nifty-options |
| 19 | Stockpathshala — VWAP+EMA | https://stockpathshala.com/vwap-and-ema-strategy/ |
| 20 | Rupeezy — VWAP intraday | https://rupeezy.in/blog/vwap-trading-strategy-intraday-options |
| 21 | Choice India — VWAP strategy | https://choiceindia.com/blog/vwap-trading-strategy |
| 22 | Mastertrust — Beginners intraday strategies | https://www.mastertrust.co.in/blog/beginners-guide-to-intraday-trading-strategies |
| 23 | TradeFundrr — Volume exhaustion entry | https://tradefundrr.com/volume-exhaustion-entry-setup/ |
| 24 | Enrich Money — Mean reversion | https://enrichmoney.in/blog-article/mean-reversion-trading-algorithmic-strategy |
| 25 | Quantfish — Simultaneous vs separate long/short | https://quant.fish/wiki/simultaneous-vs-separate-long-short-strategy-optimization/ |
| 26 | TrendSpider — Volume spikes | https://help.trendspider.com/kb/indicators/volume-spikes |
