# Sub-project #9 — Asymmetry Feasibility Round 3 (Indian Microstructure)

**Date:** 2026-05-06
**Predecessors:**
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)

**Lessons driving this round:**
- 2026-05-05 — feasibility / pro-precedent check BEFORE any §3.3 brief
- 2026-05-05 — never recommend candidates that fail Gate B (data feasibility); F&O OI velocity died here (no historical intraday OI ticks)
- 2026-05-01 — stop padding shortlists; recommend only what passes

**Round-3 mandate:** find Indian intraday-MIS asymmetries that pass three gates simultaneously — (A) Indian retail-algo precedent at Zerodha/Upstox-scale infra, (B) data already on disk in this repo, (C) time-of-day fits the 11:00-15:15 IST gap left by `gap_fade_short` (09:15-09:30) and `circuit_t1_fade_short` (10:30). Academic peer-review is secondary; if A + B both pass, candidate advances and the spec notes "in-house event study to produce evidence."

---

## Executive summary

Nine candidate directions evaluated. Three pass all three gates and form the round-3 shortlist:

| # | Candidate | Gate A | Gate B | Gate C | Status |
|---|---|:---:|:---:|:---:|---|
| 1 | VWAP-deviation mean-reversion (5m) | YES | YES | YES (intraday all day) | **SHORTLIST** |
| 2 | Index-vs-stock divergence revert (NIFTY 5m × stock 5m) | YES | YES | YES | **SHORTLIST** |
| 3 | Volume-spike exhaustion reversal (5m) | YES | YES | YES | **SHORTLIST** |
| 4 | Sector-relative momentum (NIFTY IT/BANK/AUTO sectoral × stock) | YES | YES | YES | DEFER (overlap with #2; revisit if #2 underperforms) |
| 5 | EOD-PCR mean-revert intraday on T+1 | PARTIAL | YES | YES | DEFER (precedent on intraday-PCR not EOD-PCR; data shape mismatch) |
| 6 | Late-day liquidation fade (14:30-15:15) | PARTIAL | YES | YES | DEFER (retail-blog precedent; no algo-platform-published strategy located) |
| 7 | Range-day breakout (NR4/NR7 → T-day breakout) | YES | YES | NO (entry 09:30-10:30 collides with existing setups) | RETIRE for round-3; consider as morning-setup expansion later |
| 8 | Earnings-day intraday | YES (precedent) | NO (no earnings calendar on disk) | n/a | RETIRE |
| 9 | Lunch-hour fade (12:00-13:00) | NO (consensus is "do not trade this window") | YES | YES | RETIRE |

**Recommended top brief target: Candidate 2 — Index-vs-stock divergence revert.** It is the most differentiated mechanic from existing setups (uses NIFTY 50 + stock cross-asset signal, not pure single-stock OHLCV), has direct retail-algo precedent (TradingView "relative strength to NIFTY" scripts, Wright Research, Religare), uses cleanly available 1m index + 5m enriched stock data, and naturally fires in the 11:00-14:30 trend-confirmation window where existing setups don't operate. Expected fire rate is moderate (~3-8 events/day across F&O 200), low correlation with `gap_fade_short` (different participants) and zero overlap with `circuit_t1_fade_short` (different mechanic).

**Discovered new mechanism not in the candidate list:** **first-pullback-to-VWAP after 11:00 morning-trend-day** (a hybrid of #1 and #2 that uses NIFTY trend-day classification as the regime filter and individual stock VWAP-touch as the entry trigger). Not formally evaluated as a 10th candidate due to budget; flagged as "natural setup-design extension of #1 + #2" should those advance.

---

## Per-candidate findings

### 1. VWAP-deviation mean-reversion (5m bars)

**Mechanic.** On a 5m bar, compute `(close - vwap) / vwap_intraday_stdev` for the symbol. When this z-score breaches ±N (e.g., ±2.0 or ±2.5) and the next bar prints a reversal candle (close back inside the band), enter mean-revert in the direction of VWAP. Exit when price touches VWAP, or after K bars, whichever first. Indian-flavoured because (a) NSE intraday is heavily VWAP-anchored due to MIS auto-square dynamics and institutional VWAP-target execution algos, (b) India VIX < 14 regimes (the modal regime in 2024-26) make VWAP behave as a strong magnet per multiple Indian retail-algo blogs, (c) the 5m enriched feather already has a `vwap` column attached.

**Gate A — YES.** Precedent at Indian retail/pro intraday-MIS scale:
1. **Zerodha Streak** — published VWAP-strategy templates incl. "VWAP & Supertrend Intraday Strategy" YouTube walkthrough on Streak, deployable on Zerodha Kite for retail MIS. https://www.youtube.com/watch?v=9O6ZHoVSLcQ , https://streak.zerodha.com/
2. **Zerodha Varsity** — VWAP chapter in supplementary technical-indicators module (canonical foundational reference). https://zerodha.com/varsity/chapter/supplementary-notes-1/
3. **Goodwill Securities** — "VWAP: A Pro Tool for Intraday Traders" — Indian broker guide combining VIX < 14 regime filter with VWAP mean-reversion entries. https://www.gwcindia.in/blog/vwap-volume-weighted-average-price-a-pro-tool-for-intraday-traders/
4. **Share India** — "VWAP Indicator for Intraday Trading" — Indian broker how-to. https://www.shareindia.com/knowledge-center/intraday-trading/how-to-use-vwap-indicator-for-intraday-trading
5. **Stoxra Blog** — published top-5 algo strategies for India-retail-intraday including VWAP deviation. https://stoxra.com/blog/top-5-algorithmic-trading-strategies-professional-traders

**Gate B — YES.** Data on disk:
- `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` — has `open, high, low, close, volume, vwap, bb_width_proxy, adx, rsi`. VWAP column already attached. 75 bars/day × ~750 trading days = ~56K bars/symbol; F&O 200 universe → 11M bars total. Verified the file exists for RELIANCE (1500 rows for one month indicates per-symbol coverage is fine).
- `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_1minutes.feather` — for replay-precision entries (verified RELIANCE 2023-01-02 to 2026-04-30, 303,335 rows).

**Gate C — YES.** Fires anywhere price stretches from VWAP with a reversal candle. The 11:00-15:15 window is naturally the most active because morning gap-noise has settled and afternoon trend-day stretches happen in this window.

**Direction.** Either (long when stretched-below-VWAP, short when stretched-above). Asymmetry: in the F&O universe, short-fade-of-stretched-above-VWAP is statistically stronger because retail-led greed-buy-the-rip drives the stretch (analogous to gap_fade_short's mechanic, but applied any-time-of-day not just open).

**Expected fire rate.** 5-15/day across F&O 200 at ±2.0 sigma; ~2-5/day at ±2.5 sigma. Conservative annual count: 1,200-3,500 events.

**Peer-reviewed evidence (secondary).** General mean-reversion in Indian equity markets documented in Springer "Mean-Reverting Tendency in Stock Returns" chapter (Indian-data study). VWAP-specific Indian intraday peer-reviewed paper not located in budget; in-house event study expected to produce the formal evidence.
- https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4
- VWAP general framework: Zarattini & Aziz, SSRN 4631351 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351

**Recommendation: SHORTLIST.** Strong precedent, easy data path, fits the time-of-day gap.

---

### 2. Index-vs-stock divergence revert

**Mechanic.** On a 5m bar, compute the cumulative return of NIFTY 50 since 09:15 vs the cumulative return of an F&O-200 stock since 09:15. When the spread (`stock_return − beta × nifty_return`, or simpler `stock_return − nifty_return`) exceeds ±N stdev of its rolling-30-bar history AND the stock's 5m bar prints a reversal candle, enter mean-revert toward the index (long stock if it has lagged NIFTY by N stdev with positive NIFTY trend; short stock if it has stretched above NIFTY by N stdev). Exit on spread reversion to zero, on time stop (e.g., 60 minutes), or on intraday close 15:15.

This is a cross-sectional / cross-asset mechanic. It is fundamentally different from candidate 1 (single-stock VWAP) — here the alpha is in the *spread* between two assets, not a single-asset deviation. Intuition: NIFTY 50 represents the "market regime" intraday; stocks that diverge from this regime mid-day are usually a single-stock micro-event (retail-driven impulse, news headline, sector rotation) that mean-reverts back to the index drift over 30-60 minutes once the impulse fades.

**Gate A — YES.** Precedent:
1. **Wright Research** — published reversal-strategy guide that explicitly enumerates "Top 5 Reversal Trading Strategies" for Indian retail traders, including momentum-divergence and RSI-divergence reversal variants. https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/
2. **Religare** — "Strategies for Intraday Trading with Nifty & TradingView Indicators" — published divergence-based intraday strategies tied to NIFTY. https://www.religareonline.com/blog/intraday-trading-strategies-with-nifty-tradingview-indicators/
3. **TradingView India / Knowledgehub** — "Divergence" published indicators and intraday playbooks for NIFTY. https://in.tradingview.com/chart/NIFTY/akf9L2u5-Divergence/ ; "Mastering the Intraday Sutra" includes index-relative-strength reads. https://in.tradingview.com/chart/NIFTY/hAT315Ke-Mastering-the-Intraday-Sutra-An-intraday-trading-strategy/
4. **Stratzy** — common-algo-strategies post enumerates pairs / index-stock divergence as a mainstream algo strategy on Indian retail platforms. https://stratzy.in/blog/common-algo-trading-strategies-and-examples/
5. **Enrich Money** — algorithmic mean-reversion explainer using Indian-stock + index examples. https://enrichmoney.in/blog-article/mean-reversion-trading-algorithmic-strategy

**Gate B — YES.** Data on disk:
- `backtest-cache-download/index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1minutes.feather` — verified 290,267 rows from 2023-01-19 to 2026-01-16 (>3 years, sufficient depth). Resample to 5m for alignment with stock 5m enriched feathers.
- `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` — F&O-200 universe 5m bars.
- `assets/fno_liquid_200.csv` — universe filter.
- Beta computation: `cache/preaggregate/consolidated_daily.feather` (rolling 30/60/90-day daily beta vs NIFTY).

**Gate C — YES.** Spread divergences mature in the trending part of the day. Empirically (sub7/sub8 backtest patterns) the 11:00-14:30 window is when single-stock noise diverges from the index. Aligns directly with the gap left by existing setups.

**Direction.** Either, depending on quadrant: stock-stretched-above-NIFTY → short stock; stock-stretched-below-NIFTY (when NIFTY trend is up) → long stock. The "fight-the-stock-not-the-market" framing is the natural Indian retail-algo translation of the mechanic.

**Expected fire rate.** 3-8 events/day across F&O 200 at ±2.0 sigma on the spread. Annual count: 700-2,000 events.

**Peer-reviewed evidence (secondary).** Liu, Liu, Wang, Zhou, Zhu — "Overnight-Intraday Reversal Everywhere" (SSRN) documents intraday-reversal patterns globally; not Indian-specific but the cross-asset spread variant is the natural India translation. https://papers.ssrn.com/sol3/Delivery.cfm/2730304.pdf?abstractid=2730304 . In-house event study expected to confirm Indian persistence.

**Recommendation: SHORTLIST. Top brief target.** Most differentiated mechanic from existing setups; cross-asset signal eliminates single-stock-only failure modes.

---

### 3. Volume-spike exhaustion reversal (5m)

**Mechanic.** On a 5m bar, compute `volume_z = (volume − rolling_30_bar_mean) / rolling_30_bar_std`. When `volume_z >= 3.0` AND the bar's `(close − low) / range` (for a long-tail-down) or `(high − close) / range` (for a long-tail-up) indicates exhaustion (e.g., wick ≥ 60% of range), AND the next bar prints a confirming reversal close, enter counter-direction. Exit on N-bar time stop or 0.5R move. The thesis: 3x+ normal volume on a single 5m bar in the middle of the day is almost never a fundamental news event (those mostly fire at open or on press releases) — it is a retail capitulation / squeeze, which is statistically prone to immediate reversal.

**Gate A — YES.** Precedent:
1. **Wright Research** — reversal strategies explicitly include "volume-confirmed exhaustion" as a top-5 setup for Indian retail. https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/
2. **TradingView India** — multiple published Pine indicators for volume-spike-reversal on NSE charts under the `Spike` and `VOLUMEBREAKOUT` tags. https://in.tradingview.com/scripts/spike/ , https://in.tradingview.com/scripts/volumebreakout/
3. **Mastertrust** — published intraday-strategies tips post including volume-spike reversal as a beginner playbook. https://www.mastertrust.co.in/blog/beginners-guide-to-intraday-trading-strategies
4. **Streak** — built-in scanner templates for volume-anomaly + reversal-candle combinations. https://www.streak.tech/strategies
5. **MEMORY.md project_detector_bugs_and_improvements.md** notes "volume spike reversal P1-P6" as an existing internal priority — independent confirmation that the mechanic was already on the team's radar.

**Gate B — YES.** Data on disk:
- `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` — `volume` column directly available, `vwap` for confirmation, `rsi` for divergence overlay. No additional ingestion needed.

**Gate C — YES.** Fires anywhere a volume spike happens. The afternoon (11:00-14:30) is exactly where volume spikes are most likely to be retail-driven (not institutional opening-flow, not closing-rebalance), which is the participant identification that gives this setup its edge over generic spike strategies.

**Direction.** Either, contingent on the wick direction of the spike bar.

**Expected fire rate.** 8-20 events/day across F&O 200 at z >= 3.0 with wick ≥ 60% filter. Annual count: ~2,000-5,000.

**Peer-reviewed evidence (secondary).** Generic volume-spike-and-reversal patterns documented in Springer Mean-Reverting chapter (above). In-house event study expected.

**Recommendation: SHORTLIST.** Lowest infrastructural lift of all candidates (single-symbol, no cross-asset wiring), strong precedent. Likely the easiest of the three to ship.

---

### 4. Sector-relative momentum (sectoral index × constituent)

**Mechanic.** Like #2 but with sector index (NIFTY IT, NIFTY BANK, NIFTY AUTO, NIFTY METAL, NIFTY PHARMA, NIFTY FMCG, NIFTY ENERGY, NIFTY FIN_SERVICE, NIFTY PSU_BANK, NIFTY REALTY) instead of NIFTY 50. A constituent that diverges from its sector index is the candidate trade.

**Gate A — YES.** TradingView TechnicalExpress sector-rotation strategy posts (multiple). https://in.tradingview.com/chart/NIFTY/zoVMSO0t-Introduction-to-Sector-Rotation-Strategies-in-Trading/ , https://in.tradingview.com/chart/BANKNIFTY/sFjTg4R9-Sector-Rotation-Strategy/ , https://in.tradingview.com/chart/BANKNIFTY/9bkUVeFa-Sector-Rotation-Strategies/ . Religare intraday-strategies blog discusses sector-relative-strength explicitly.

**Gate B — YES (revised vs Round-2).** Round-2 incorrectly stated only NIFTY 50 + NIFTY BANK indices were available. **Verified actually on disk:**
```
backtest-cache-download/index_ohlcv/
  NSE_NIFTY_50/
  NSE_NIFTY_AUTO/
  NSE_NIFTY_BANK/
  NSE_NIFTY_ENERGY/
  NSE_NIFTY_FIN_SERVICE/
  NSE_NIFTY_FMCG/
  NSE_NIFTY_IT/
  NSE_NIFTY_METAL/
  NSE_NIFTY_PHARMA/
  NSE_NIFTY_PSU_BANK/
  NSE_NIFTY_REALTY/
```
Each has `_1minutes.feather` and `_1days.feather`. Sector composition lists in `assets/ind_nifty{it,bank,auto,fmcg,finance,metal,oilgas,pharma,psubank,realty}list.csv` (verified, e.g., IT list has Coforge/HCL/Infosys/LTIM/Mphasis/OFSS/Persistent/TCS/Tech-M/Wipro). Round-2's data limitation statement was wrong; this candidate is fully feasible on existing disk.

**Gate C — YES.** Same as #2.

**Direction.** Either.

**Expected fire rate.** Lower than #2 because the sectoral universe is smaller (each sector has 5-15 names). 2-5 events/day across all sectors. Annual count: ~500-1,200.

**Recommendation: DEFER.** Mechanically a strict subset of #2 (NIFTY 50 contains all sectoral names). Ship #2 first; revisit this only if #2 underperforms specifically on sector-rotation regimes (e.g., IT moves +2% while NIFTY +0.3% — single-stock IT divergences from NIFTY-50 will be muted but they will be visible against NIFTY-IT). Not the round-3 brief target because it would dilute focus.

---

### 5. EOD-PCR mean-revert intraday on T+1

**Mechanic.** Compute symbol-level PCR (put OI / call OI) from EOD `data/option_chain/{YYYY}/{MM}/{YYYY-MM-DD}.parquet`. When PCR is at z-score ±2 of its rolling 60-day distribution, hypothesise mean-reversion of the underlying on T+1 intraday.

**Gate A — PARTIAL.** PCR-mean-reversion precedent is on **intraday-PCR** (live, real-time PCR), which we cannot replicate (no historical intraday PCR ticks). NiftyTrader, Upstox, NiftyInvest, BlinkX, Sensibull all publish intraday PCR; the strategy literature uses *intraday* PCR-spikes for entry, not EOD PCR for next-day entry. https://www.niftytrader.in/nifty-put-call-ratio , https://upstox.com/fno-discovery/open-interest-analysis/nifty-pcr/ , https://niftyinvest.com/put-call-ratio/NIFTY , https://web.sensibull.com/live-options-charts?tradingsymbol=NIFTY . No published Indian retail-algo strategy targets *EOD PCR → next-day intraday* mean-reversion.

**Gate B — YES (data shape).** EOD parquets verified — 11 columns including `oi`, `oi_change`, `vol`, `option_type` (CE/PE). Aggregable to per-symbol PCR per day.

**Gate C — YES.**

**Recommendation: DEFER.** Same precedent failure mode as Round-2 candidate F&O OI velocity — the mechanic that has precedent uses live data we don't have, and the variant that fits our data has no Indian-retail-algo precedent. Lessons.md 2026-05-05 explicitly retires this pattern. Re-route as an in-house event study (not a §3.3 brief) to test whether EOD-PCR has *any* T+1 intraday signal; if event study yields >2x baseline edge, promote.

---

### 6. Late-day liquidation fade (14:30-15:15)

**Mechanic.** Stocks that are heavily down by 14:30 IST often see further pressure from MIS auto-square unwinding 15:00-15:25 and from stop-loss cascades — but if intraday-low is reclaimed by ~14:45, the reclaim itself is a signal of late-stage smart-money accumulation. Long the reclaim, exit by 15:15.

**Gate A — PARTIAL.** Retail-blog discussion of last-hour reversals exists (JM Financial blog, IIFL knowledge centre, 5paisa best-time-frame guide), but **no Indian retail-algo platform publishes an operationalised algo for this exact mechanic**. The only formally-published algo for the closing window is `mis_unwind_short` (sub7, REJECTED) which traded the *opposite* direction. Multiple sources warn the closing-hour is unpredictable due to institutional flow. https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis , https://www.indiainfoline.com/knowledge-center/share-market/what-is-the-timing-of-intraday-trading

**Gate B — YES.** Pure 5m OHLCV — already on disk.

**Gate C — YES (literally targets 14:30-15:15).**

**Recommendation: DEFER.** Precedent at the level required by Gate A (≥2 distinct algo-platform-published strategies) does not exist; same failure mode as Round-2 anchor-lock-in candidate. Promote only via in-house event study or after retail-algo-platform precedent emerges. The setup also has the same time-window collision concern as `mis_unwind_short` — must ensure the long-reclaim signal is statistically distinct from the (rejected) auto-square-short hypothesis.

---

### 7. Range-day breakout (NR4/NR7 → T-day breakout)

**Mechanic.** Identify NR4/NR7 (narrowest 4-day or 7-day range) on T-1 from daily OHLC. On T, enter long above T-1 high or short below T-1 low. Standard volatility-contraction → expansion play.

**Gate A — YES.** Strong Indian retail-algo precedent: Elearnmarkets (StockEdge family), HDFC Sky, Wealthpedia, Unofficed, IntradayScreener. https://blog.elearnmarkets.com/nr4-and-nr7-trading-strategy-setup/ , https://hdfcsky.com/sky-learn/trading-strategies/narrow-range-nr-4-nr-7 , https://www.wealthpedia.in/what-is-nr4-and-nr7-for-intraday-trading/ , https://intradayscreener.com/nr4-nr7-stock-screener . 5+ distinct sources; precedent is unambiguous.

**Gate B — YES.** `cache/preaggregate/consolidated_daily.feather` for daily OHLC; `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_1minutes.feather` for intraday entry trigger. Both verified on disk.

**Gate C — NO.** NR-breakout strategies fire at the 09:30-10:30 window (after the opening range establishes), which directly overlaps with `gap_fade_short` (09:15-09:30) and `circuit_t1_fade_short` (10:30). Adding a third morning setup increases morning concentration risk and does not fill the 11:00-15:15 gap that round-3 mandate prioritises.

**Recommendation: RETIRE for round-3.** Strong A and B; fails C. Re-evaluate as part of a separate "morning-setup expansion" sub-project after the 11:00-15:15 gap is filled.

---

### 8. Earnings-day intraday

**Mechanic.** Post-earnings drift / fade on the announcement day, intraday window after the announcement.

**Gate A — YES.** Standard global setup with Indian-market presence on retail platforms (StockEdge results-calendar scans, Tickertape, MoneyControl).

**Gate B — NO.** No earnings/results calendar found on disk. `assets/` has sector lists, holiday list, NIFTY daily, FNO 200, traded-symbols list, but no earnings calendar. Without an event-aligned calendar, intraday data alone cannot be sliced to "announcement day, post-announcement window" reliably.

**Gate C — YES (would have been).**

**Recommendation: RETIRE.** Fails Gate B definitively. Acquiring an earnings calendar (NSE results calendar scrape or third-party feed) is a pre-requisite that is out-of-scope for round-3. Revisit if earnings-calendar ingestion is added to the data layer in a separate sub-project.

---

### 9. Lunch-hour fade (12:00-13:00)

**Mechanic.** Hypothesis: low-volume drift in the lunch hour reverts at 13:00 onwards as institutional desks return.

**Gate A — NO.** The Indian-trading-community **consensus** (multiple distinct sources: JM Financial Services, ICICI-aligned broker blogs, 5paisa, Lakshmishree, JainamGold) is that the 11:30-13:30 window is "midday doldrums" with low volume, choppy price action, and false breakouts that **trap retail traders on both sides**. The dominant retail-algo recommendation is to *not* trade this window. https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis , https://tradesmartonline.in/blog/best-time-to-trade-in-indian-market/ , https://www.5paisa.com/blog/best-time-frame-for-intraday-trading

**Gate B — YES.** Pure 5m OHLCV.

**Gate C — YES (covers 12:00-13:00).**

**Recommendation: RETIRE.** Precedent is *against* trading this window, not *for*. Same lesson as Round-2 candidate 4 (T+1 settlement liquidation) — when the literature points opposite to the hypothesis, the hypothesis dies.

---

## Shortlist for §3.3 brief stage

Three passing candidates, ranked. Per 2026-05-01 lesson, no padding below this — these are the only round-3 candidates that pass all three gates.

1. **Candidate 2 — Index-vs-stock divergence revert (TOP BRIEF TARGET)**
   - Most differentiated mechanic from existing setups
   - Cross-asset spread (NIFTY 5m × stock 5m) — robust to single-stock failure modes
   - Fires 11:00-14:30 — fills the time-of-day gap precisely
   - Data already on disk; new infra = NIFTY 5m loader + spread computation
   - Setup-design caution: `(stock_return − beta × nifty_return)` is the production form; the simpler `(stock_return − nifty_return)` will create false signals for high-beta names. The brief must specify rolling-30-day beta sourcing from `consolidated_daily.feather`

2. **Candidate 3 — Volume-spike exhaustion reversal**
   - Lowest infrastructural lift (single-symbol, single-feed)
   - Strongest fire rate (8-20/day across F&O 200)
   - Cleanest data path (single 5m enriched feather, no cross-asset wiring)
   - Setup-design caution: must distinguish 11:00-14:30 retail-spike from 09:15-10:30 opening-flow spike (the latter is institutional, will not mean-revert) — the time-of-day filter is critical

3. **Candidate 1 — VWAP-deviation mean-reversion**
   - Established mechanic with 5+ Indian-retail-algo sources
   - Naturally fires throughout the day; needs time-of-day filter to focus on 11:00-15:15 window
   - Setup-design caution: VIX-regime-conditional — Goodwill blog states VWAP-revert works in VIX < 14 (modal regime); needs a India VIX gate. India VIX intraday history not currently on disk; daily India VIX would need to be added or proxied via NIFTY-50 IV from the EOD option chain

**Round-1 + Round-2 production setups remain the dominant body of work**:
- `gap_fade_short` (round-1 D variant, shipped)
- `circuit_t1_fade_short` (round-1 E, shipped at thin alpha)
- F&O OI-velocity (round-2 candidate 3, retired at Gate B per 2026-05-05 lessons)

Round-3 shortlist adds three uncorrelated middle-of-day setups that share zero data-infrastructure with the existing morning setups.

---

## Sources (consolidated)

### VWAP mean-reversion (Candidate 1 — SHORTLISTED)
- Zerodha Streak — https://streak.zerodha.com/ , https://www.youtube.com/watch?v=9O6ZHoVSLcQ
- Zerodha Varsity supplementary indicators — https://zerodha.com/varsity/chapter/supplementary-notes-1/
- Goodwill Securities (VIX-regime-aware) — https://www.gwcindia.in/blog/vwap-volume-weighted-average-price-a-pro-tool-for-intraday-traders/
- Share India — https://www.shareindia.com/knowledge-center/intraday-trading/how-to-use-vwap-indicator-for-intraday-trading
- Stoxra Blog — https://stoxra.com/blog/top-5-algorithmic-trading-strategies-professional-traders
- Trading Q&A by Zerodha — https://tradingqna.com/t/what-is-vwap-how-do-i-read-this-indicator/7936
- VWAP general framework: Zarattini & Aziz, SSRN 4631351 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4631351

### Index-vs-stock divergence (Candidate 2 — SHORTLISTED, top brief)
- Wright Research — https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/
- Religare — https://www.religareonline.com/blog/intraday-trading-strategies-with-nifty-tradingview-indicators/
- TradingView Knowledgehub Divergence — https://in.tradingview.com/chart/NIFTY/akf9L2u5-Divergence/
- TradingView VishalSubandh Intraday Sutra — https://in.tradingview.com/chart/NIFTY/hAT315Ke-Mastering-the-Intraday-Sutra-An-intraday-trading-strategy/
- Stratzy — https://stratzy.in/blog/common-algo-trading-strategies-and-examples/
- Enrich Money — https://enrichmoney.in/blog-article/mean-reversion-trading-algorithmic-strategy
- Liu et al., "Overnight-Intraday Reversal Everywhere" SSRN 2730304 — https://papers.ssrn.com/sol3/Delivery.cfm/2730304.pdf?abstractid=2730304

### Volume-spike exhaustion reversal (Candidate 3 — SHORTLISTED)
- Wright Research (above)
- TradingView spike scripts — https://in.tradingview.com/scripts/spike/ , https://in.tradingview.com/scripts/volumebreakout/
- Mastertrust — https://www.mastertrust.co.in/blog/beginners-guide-to-intraday-trading-strategies
- Streak strategies — https://www.streak.tech/strategies
- TrendSpider Volume Spikes (mechanism reference) — https://help.trendspider.com/kb/indicators/volume-spikes
- Internal: MEMORY.md `project_detector_bugs_and_improvements.md` — volume spike reversal P1-P6

### Sector-relative momentum (Candidate 4 — DEFER)
- TradingView TechnicalExpress sector-rotation posts — https://in.tradingview.com/chart/NIFTY/zoVMSO0t-Introduction-to-Sector-Rotation-Strategies-in-Trading/ , https://in.tradingview.com/chart/BANKNIFTY/sFjTg4R9-Sector-Rotation-Strategy/ , https://in.tradingview.com/chart/BANKNIFTY/9bkUVeFa-Sector-Rotation-Strategies/

### EOD-PCR (Candidate 5 — DEFER)
- NiftyTrader — https://www.niftytrader.in/nifty-put-call-ratio
- Upstox PCR — https://upstox.com/fno-discovery/open-interest-analysis/nifty-pcr/
- NiftyInvest PCR — https://niftyinvest.com/put-call-ratio/NIFTY
- BlinkX — https://blinkx.in/put-call-ratio
- Sensibull — https://web.sensibull.com/live-options-charts?tradingsymbol=NIFTY
- IIFL — https://www.indiainfoline.com/markets/derivatives/put-call-ratio
- Bajaj Finserv PCR — https://www.bajajfinserv.in/put-call-ratio
- Groww — https://groww.in/p/put-call-ratio

### Late-day liquidation fade (Candidate 6 — DEFER)
- JM Financial intraday-time analysis — https://www.jmfinancialservices.in/blogs-and-articles/intraday-trading-time-analysis
- IIFL share-market timing — https://www.indiainfoline.com/knowledge-center/share-market/what-is-the-timing-of-intraday-trading
- 5paisa best-time-frame — https://www.5paisa.com/blog/best-time-frame-for-intraday-trading

### Range-day breakout NR4/NR7 (Candidate 7 — RETIRE for round-3)
- Elearnmarkets (StockEdge) — https://blog.elearnmarkets.com/nr4-and-nr7-trading-strategy-setup/
- HDFC Sky — https://hdfcsky.com/sky-learn/trading-strategies/narrow-range-nr-4-nr-7
- Wealthpedia — https://www.wealthpedia.in/what-is-nr4-and-nr7-for-intraday-trading/
- Unofficed — https://unofficed.com/courses/narrow-range-strategy/lessons/intraday-strategy-nr7/
- IntradayScreener — https://intradayscreener.com/nr4-nr7-stock-screener
- StockeZee — https://www.stockezee.com/stock-screener
- AutoBuySellSignal — https://autobuysellsignal.in/2025/01/nr4-nr7-intraday-trading-strategy/
- Academic: Wang & Gangwar (SSRN 5198458) "Optimizing Intraday Breakout Strategies on the NSE" — https://papers.ssrn.com/sol3/Delivery.cfm/5198458.pdf?abstractid=5198458

### Lunch-hour fade (Candidate 9 — RETIRE)
- JM Financial (above)
- TradeSmart — https://tradesmartonline.in/blog/best-time-to-trade-in-indian-market/
- 5paisa (above)
- Lakshmishree — https://lakshmishree.com/blog/best-intraday-trading-tips/
- ICFM India — https://www.icfmindia.com/blog/intraday-trading-for-beginners-in-india-2026-how-to-start-strategies-risk-management-real-market-insights

### General mean-reversion / academic framing
- Springer "Mean-Reverting Tendency in Stock Returns" — https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4
- Liu et al. SSRN 2730304 (above)
- Damora Capital "Algorithm-Based Intraday Trading Strategies and their Market Impact" — https://damoracapital.com/wp-content/uploads/2021/04/Momentum-Mean-Reversion-and-Statistical-Arbitrage-id3785503.pdf
- Vu & Bhattacharyya "Design and Development of Mean Reversion Strategies on QuantConnect Platform" SSRN 4878676 — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4878676

---

## Methodology notes (round 3 vs round 2)

Round-2 imposed gates A (precedent) and B-equivalent (academic). Round-3 reorders to A (precedent) → B (data-on-disk) → C (time-of-day fit). Two changes from round-2:

1. **Data-on-disk verified before drafting.** The Round-2 F&O OI velocity candidate passed Gate A but died at the unfaced Gate B; the lesson 2026-05-05 prompted moving disk verification ahead of academic search. This round, every YES on Gate B has a verified file path or confirmed file existence.
2. **Time-of-day diversification gate added.** The portfolio currently has two morning setups; adding a third morning setup adds little portfolio value. Round-3 explicitly favours 11:00-15:15 IST candidates. Candidate 7 (NR-breakout) was strong on A and B but failed C and was retired *for this round only* — flagged for a future "morning-expansion" sub-project.

Round-1 and Round-2 candidates are NOT re-evaluated under round-3's three gates in this spec. The carryover production setups (`gap_fade_short`, `circuit_t1_fade_short`) remain in place as the morning leg of the portfolio; round-3 shortlist adds the middle-of-day leg.

The discovered candidate (first-pullback-to-VWAP after 11:00 morning-trend-day, hybrid of #1 and #2) is flagged as a natural setup-design extension to surface during §3.3 brief work on Candidate 2 — not a 10th formal candidate, given the budget.
