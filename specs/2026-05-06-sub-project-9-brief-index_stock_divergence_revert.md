# §3.3 Brief: `index_stock_divergence_revert`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-06

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3, this candidate is §2 — top recommendation)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template)

This is one of three round-3 §3.3 briefs (the others are `vwap_deviation_meanrevert` and `volume_spike_exhaustion_reversal`). All three target the 11:00-15:15 IST middle-of-day window left empty by the existing morning-only setups.

**Note on round-3 ranking:** round-3 spec named this candidate the "top recommendation" because it is the most differentiated mechanic (cross-asset spread, not single-stock OHLCV). Per user equal-rigor mandate, the brief depth here is identical to the other two — sanity-check budget will decide which advances, not pre-judgment in brief depth.

---

## Asymmetry

**Name:** Indian-equity intraday stock-vs-NIFTY-50 divergence mean-reversion (5m).

**Indian-specific source:**
- **NIFTY 50 represents the consensus Indian market regime intraday.** ~70% of FII/DII flow is benchmarked to NIFTY 50 (or its derivatives). Within an intraday session, the index drift represents the aggregate flow direction; individual stocks that diverge mid-session from this drift are usually doing so for an *idiosyncratic* reason (single-stock news headline, sector rotation impulse, retail-attention spike) — and these idiosyncratic moves are the most mean-reversion-prone of all intraday moves because the index itself doesn't ratify them.
- **Indian retail trades stocks individually, not as index spreads.** Cross-asset thinking (stock vs index) is institutional/algo territory. Retail momentum-chase fires on the *single-stock* breakout signal — without checking whether NIFTY itself is moving. This creates a pocket of inefficiency: when retail piles into a stretching stock that the index isn't validating, the divergent move is structurally fragile.
- **TradingView India Knowledgehub + Religare** explicitly publish "stock vs NIFTY divergence" intraday strategies for Indian retail — confirming the mechanic has retail-platform precedent at Indian intraday-MIS scale.
- **Stratzy + Wright Research** list pairs / index-divergence in their "common algo strategies" — confirming Indian-algo-platform precedent.

The exploitable asymmetry is the **5m bar where (stock_return − beta × nifty_return) z-score breaches ±N over its rolling-30-bar history** AND the stock prints a reversal candle, where the spread reverts to zero over the next 30-60 minutes as the idiosyncratic impulse fades.

## Participants

- **Stretched-above-NIFTY (SHORT side, primary edge):** retail momentum-chasers + sector-rotation operators piling into single-stock momentum that the index is NOT validating. Common single-stock impulse sources: news-headline pop, sector-leader breakout halo (e.g., HDFCBANK pops → other banking pops), retail screener "top gainers" attention. The fade is filled by mean-reversion algos + institutional desk arbitrage that systematically short stretches against index.
- **Stretched-below-NIFTY (LONG side, secondary edge):** intraday capitulation in single-stock without index ratification (often a stop-cascade or single-block sale). The long side is buying-the-dip-against-market-direction. Per project-wide caveat (long-bias 11-failure pattern), this side carries higher failure risk and is gated.

We are on the disciplined side of an idiosyncratic mispricing — explicitly in the cross-asset framing, which is harder for retail to see.

## Persistence

Three structural reasons:

1. **Retail single-stock framing is structural.** Indian retail brokerages display individual stock charts; the NIFTY chart is a *separate* tab. Retail decisions (especially algo-retail with simple breakout/RSI signals) fire on single-stock data without the cross-asset context. As long as Indian retail keeps using single-stock decision frames (i.e., as long as Zerodha Kite UI looks like Zerodha Kite UI), this mispricing persists.
2. **Institutional benchmark is NIFTY-relative.** FII/DII performance is reported NIFTY-relative; large-account flow systematically rebalances divergent stocks back toward index drift over the session. This creates a guaranteed counter-flow on stretches. Not market-cycle-dependent.
3. **Beta-stable universe property.** F&O 200 stocks have stable 30-day rolling betas vs NIFTY 50 (mostly 0.6-1.5 with 90-day stability). Spread mean-reversion math holds because beta is approximately stationary at the daily timescale — the intraday divergence is genuinely transient noise around a stable relationship.

Greenwood/Sammon decay caveat: cross-asset arbitrage strategies decay AS LONG AS arbitrageurs adapt. The Indian-retail single-stock framing protects the strategy from the typical "everyone runs the same algo" arbitrage decay because most retail won't adopt the spread frame in this decade. Decay is plausible but slower than VWAP-revert (candidate 1) which is more saturated. Annual re-validation.

## Project-wide caveats addressed

- **Long-bias caveat:** the brief proposes a **bidirectional** detector but prefers SHORT side. Per sub7/sub8 11-failure long-bias pattern, the LONG-side ship gate requires sanity-check long-side PF ≥ short-side PF × 0.85. The cross-asset mechanic is *less* long-bias-failure-prone than single-stock long setups because the index gives a cross-asset confirmation (LONG only fires when stock has stretched DOWN against an UP-trending NIFTY — i.e., the index trend supports the long bias). However the gate still applies — short-only ship if long-side underperforms.
- **Decay risk (Greenwood/Sammon):** Acknowledged but plausibly slower than candidates 1 and 3 because the Indian-retail single-stock framing is sticky.
- **Beta-instability tail risk:** for stocks with regime-shift news (acquisition, ratings change, sector-leadership change), the 30-day rolling beta is no longer the right reference. Sanity-check must report PF excluding "ex-corporate-action" days; live detector defers to a daily news-flag exclusion.

## Evidence

Round-3 spec §2 lists 5 Indian retail-algo platform sources (Wright Research, Religare, TradingView India Knowledgehub, Stratzy, Enrich Money). All 5 are operational publications on Indian intraday divergence/reversal at retail-MIS scale. Per round-3 mandate (Gate A passed), Indian retail-algo precedent is primary; in-house event study during sanity-check is the formal evidence substitute.

Peer-reviewed (secondary):
1. **Liu, Liu, Wang, Zhou, Zhu — "Overnight-Intraday Reversal Everywhere", SSRN 2730304** — global documentation of intraday-reversal patterns, applicable as cross-market framing. https://papers.ssrn.com/sol3/Delivery.cfm/2730304.pdf?abstractid=2730304
2. **Springer "Mean-Reverting Tendency in Stock Returns" chapter** (Indian-data study). https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4

Indian retail-algo sources (Gate A primary):
1. Wright Research top-5 reversal strategies — https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/
2. Religare intraday with NIFTY + TradingView indicators — https://www.religareonline.com/blog/intraday-trading-strategies-with-nifty-tradingview-indicators/
3. TradingView Knowledgehub Divergence — https://in.tradingview.com/chart/NIFTY/akf9L2u5-Divergence/
4. TradingView VishalSubandh Intraday Sutra — https://in.tradingview.com/chart/NIFTY/hAT315Ke-Mastering-the-Intraday-Sutra-An-intraday-trading-strategy/
5. Stratzy common-algo-strategies — https://stratzy.in/blog/common-algo-trading-strategies-and-examples/
6. Enrich Money mean-reversion — https://enrichmoney.in/blog-article/mean-reversion-trading-algorithmic-strategy

## Direction

**Bidirectional**, with **SHORT side as the primary edge** and additional gate on long-side ship decision (see project-wide caveat above).

- **SHORT** when (stock_intraday_return − β × nifty_intraday_return) ≥ +N stdev over rolling 30-bar history of the spread.
- **LONG** when the same spread ≤ −N stdev AND NIFTY is in an uptrend (5m close > 5m EMA20 on NIFTY).

The NIFTY-uptrend gate on the LONG side is the cross-asset confirmation that protects against catching a single-stock-falling-knife in a falling market — which is the structural failure mode of single-stock long setups.

## Mechanic

**Setup name:** `index_stock_divergence_revert`
**Side:** Bidirectional (SHORT primary, LONG conditional ship + cross-asset gate).

**Sequence:**

1. **Beta computation (daily, pre-session):**
   - From `cache/preaggregate/consolidated_daily.feather`, compute rolling 30-day daily beta of each F&O-200 stock vs NIFTY 50: β = cov(stock_daily_ret, nifty_daily_ret) / var(nifty_daily_ret).
   - Cache as a per-day per-symbol parquet: `data/beta_cache/<YYYY-MM-DD>.parquet` (columns: symbol, beta_30d).
   - Used as the `β` constant for the entire next session.

2. **Continuous detection (every 5m bar close, 11:00-14:30 IST):**
   - For each F&O-200 symbol, compute on the closing 5m bar:
     - `stock_intraday_ret = (close − today_open) / today_open`
     - `nifty_intraday_ret = (nifty_close − nifty_today_open) / nifty_today_open` (NIFTY 5m bar resampled from 1m feather)
     - `spread = stock_intraday_ret − β × nifty_intraday_ret`
     - `spread_z = (spread − rolling_30bar_mean) / rolling_30bar_stdev` where the rolling history is the past 30 5m bars of the spread time series within today's session
   - **Trigger:** `|spread_z| ≥ 2.0` (research-locked threshold; post-sanity may reveal need to revise but DO NOT iterate on validation data per lessons.md 2026-05-01).
     - The 2.0 threshold rationale: at typical Indian-intraday vol levels, a 2-sigma spread breach corresponds to roughly 80-150 bps of single-stock divergence — empirically the level above which retail-driven idiosyncratic moves are statistically distinct from index-tracking noise (per Religare and Wright Research divergence-strategy parameter ranges of 1.5-2.5 sigma).

3. **Reversal-candle confirmation (next 5m bar):**
   - For SHORT entry: next bar's close < open AND close < prior bar's close AND (high − close) / (high − low) ≥ 0.5.
   - For LONG entry: next bar's close > open AND close > prior bar's close AND (close − low) / (high − low) ≥ 0.5 AND NIFTY 5m close > NIFTY 5m EMA20 (cross-asset confirmation gate).

4. **Entry:**
   - **Entry price:** confirmation bar's CLOSE.
   - **Direction:** as triggered (with NIFTY-uptrend gate for LONG).
   - **Active window gate:** 11:00-14:30 IST inclusive on the trigger bar. After 14:30, late-day liquidation flow contaminates the signal (sub7's `mis_unwind_short` failure pattern); the spread can mean-revert for non-idiosyncratic reasons.

5. **Stop-loss:**
   - **Hard SL (SHORT):** trigger bar's high + 0.3% buffer.
   - **Hard SL (LONG):** trigger bar's low − 0.3% buffer.
   - **Min stop distance:** 0.6% of entry.
   - Cross-asset addition: trail stop if NIFTY moves against position by ≥ 0.5% post-entry (e.g., for SHORT, NIFTY rips up 0.5%+ → tighten stop or exit, the index regime turning against the trade is a thesis kill).

6. **Targets:**
   - **T1** (50% qty): spread reverts to zero (`spread_z ≤ 0` for SHORT, `spread_z ≥ 0` for LONG). Literal mean-reversion thesis cashing in.
   - **T2** (50% qty): spread overshoots to −0.5σ (SHORT) or +0.5σ (LONG) — pendulum-overshoot capture.
   - **Time stop:** 12 bars from entry (≈ 60 min) OR 15:10 IST hard stop, whichever first.

7. **Latch:** one fire per (symbol, day, direction) — no re-entry same direction same session.

**target_anchor_type:** `structural` — T1/T2 are anchored to the live spread z-score (a cross-asset structural signal). Not arithmetic R-multiples. The structural anchor matches the asymmetry source (cross-asset divergence is the structural condition; reversion is the structural exit).

## Universe

**Universe:** F&O 200 (full).
- **No cap-segment exclusion:** unlike candidates 1 and 3, this setup's edge is in the cross-asset spread, not in retail-FOMO concentration. Large-cap F&O names like RELIANCE, HDFCBANK, INFY, TCS exhibit divergence behaviour just as cleanly as mid/small caps (and are MORE liquid for short-side execution).
- **Universe filter file:** `assets/fno_liquid_200.csv` (verified on disk per round-3 §2 Gate B).
- **Liquidity gate:** F&O 200 inclusion already implies liquidity; no additional ADV filter.
- **Beta computability gate:** stock must have ≥ 60 days of daily history in `consolidated_daily.feather` for beta computation. Brand-new IPOs excluded for first 60 trading days.

**Why this universe:**
- Largest possible F&O universe maximises sample-size feasibility (since cross-asset edge applies broadly, no cap-bias).
- F&O eligibility ensures borrowable + liquid for MIS short.
- The mechanic is universe-agnostic in a way candidates 1 and 3 are not.

**Approximate symbol count after filters:** ~200 stocks. Sample-size feasibility per round-3 §2: 3-8/day at ±2.0 sigma; conservative annual count 700-2,000 events.

## Active window

**Setup formation + entry:** 11:00-14:30 IST. Entry on the post-trigger reversal-confirmation 5m bar (latest entry = 14:25-14:30 bar close = entry at 14:30).
**Hold horizon:** 12 bars (≈ 60 min) OR 15:10 IST hard stop.
**Latest possible exit:** 15:10 IST.

**Why 11:00-14:30 entry window (not 09:15 or 14:30+):**
- 09:15-10:55 = opening flow, beta unstable (NIFTY itself is gappy + intraday return denominators are tiny → noisy z-scores). Overlap with `gap_fade_short` and `circuit_t1_fade_short` doesn't help.
- 11:00 onwards = NIFTY has accumulated ≥ 21 5m bars; spread denominators are stable; idiosyncratic divergences become measurable.
- 11:00-14:30 = peak afternoon trend window per round-3 §2 — this is where single-stock impulses diverge from index drift most cleanly.
- 14:30+ = late-day liquidation contaminates spread (broad MIS unwind moves index AND stocks together; idiosyncratic-divergence interpretation breaks down). Earlier cutoff than candidate 1 (14:30 vs 15:00) precisely because of the cross-asset interpretation issue.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout:**
   - n < 500 trades over 12 months
   - NET PF < 1.10
   - NET Sharpe (daily) ≤ 0
2. **Direction asymmetry collapses:** if SHORT-side PF < 1.10 while LONG-side PF passes, reconsider — but the sub7/sub8 11-failure pattern means trust SHORT here even if LONG looks better in-sample.
3. **Beta drift kills the edge:** if PF in months following a market-regime change (e.g., sectoral leadership rotation, election event) is ≤ 0.9 the beta-stationarity assumption is broken; gate to 60-day rolling beta with monthly recompute, re-validate.
4. **NIFTY trend-day failures:** if PF on classified trend-up days (NIFTY close > 1% above open by 14:00) is ≤ 0.9, the SHORT side is being eaten by trend continuation; add trend-day gate to detector.
5. **Cross-asset signal fails simpler tests:** if the simpler `(stock_ret − nifty_ret)` (no beta) sanity test produces PF ≥ 1.10 while the beta-adjusted version does NOT, beta computation is hurting more than helping; revert to simpler form (acknowledge & lock).
6. **Decay signal:** rolling-60-trade NET PF drops below 1.05 sustained for 90 calendar days post-launch.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Use existing 12-month 2024 5m enriched feathers + NIFTY 1m feather (resample to 5m) on disk
- Compute 30-day rolling daily beta per symbol from `consolidated_daily.feather`
- Per (symbol, day), compute spread time series, rolling 30-bar z-score within session, identify trigger bars where `|spread_z| ≥ 2.0` AND time ∈ [11:00, 14:30]
- Apply reversal-candle confirmation (and NIFTY-uptrend gate for LONG)
- Simulate entry → 12-bar / 15:10 exit with structural T1/T2 + 0.6% min-stop hard SL
- Compute NET PF using existing Indian fee model
- **Sensitivity analysis** (locked, not re-tuned): also report PF at z-thresholds 1.5, 2.5 — report-only, no result-driven re-selection per lessons.md 2026-05-01
- Report PF / WR / Sharpe per direction (SHORT vs LONG), per cap segment (large vs mid/small), per NIFTY-regime (trend up / trend down / range)
- **Decision per §3.3:** PF ≥ 1.10 (SHORT side) → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire. LONG-side PF ≥ SHORT-side PF × 0.85 → ship bidirectional; else SHORT-only.

## Data engineering plan

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_index_stock_divergence_revert.py`** — pre-coding sanity check. Reads NIFTY 1m feather + 5m enriched stock feathers + consolidated_daily; computes spread + z-score; no detector code yet. Will be retired after used.

2. **(post-sanity-check, only if APPROVED for full implementation):**
   - `services/beta_cache_builder.py` — daily pre-session beta computation; emits `data/beta_cache/<YYYY-MM-DD>.parquet`.
   - `services/nifty_intraday_loader.py` — NIFTY 1m → 5m resample helper, live + replay compatible (CLAUDE.md rule 3).
   - `services/spread_zscore_calculator.py` — rolling 30-bar spread z-score, incremental for live mode.
   - `structures/index_stock_divergence_revert_structure.py` — the detector.
   - Config keys (added to `config/configuration.json`, NO hardcoded defaults per CLAUDE.md rule 1):
     - `index_stock_divergence_revert.spread_z_threshold` = 2.0
     - `index_stock_divergence_revert.beta_window_days` = 30
     - `index_stock_divergence_revert.spread_zscore_window_bars` = 30
     - `index_stock_divergence_revert.entry_window_start` = "11:00"
     - `index_stock_divergence_revert.entry_window_end` = "14:30"
     - `index_stock_divergence_revert.time_stop_bars` = 12
     - `index_stock_divergence_revert.time_stop_hard_ist` = "15:10"
     - `index_stock_divergence_revert.sl_buffer_pct` = 0.003
     - `index_stock_divergence_revert.min_stop_pct` = 0.006
     - `index_stock_divergence_revert.t2_overshoot_stdev` = 0.5
     - `index_stock_divergence_revert.nifty_uptrend_ema_bars` = 20
     - `index_stock_divergence_revert.nifty_adverse_move_stop_pct` = 0.005
     - `index_stock_divergence_revert.long_side_enabled` = false
     - `index_stock_divergence_revert.use_beta_adjustment` = true   # toggled false if simpler form wins sanity

   No new ingestion needed for stock or NIFTY data (verified on disk per round-3 §2 Gate B). Beta cache is computed from existing daily feather. New cross-asset wiring (NIFTY loader + spread calculator) is the only structural addition — single feed, well-bounded scope.

## Sample-size feasibility

Per round-3 §2: 3-8 events/day across F&O 200 at ±2.0 sigma; annual 700-2,000 events. After reversal-candle confirmation (~50% pass rate), NIFTY-uptrend gate on LONG side (~50% of LONG candidates), and active-window gate (11:00-14:30 = 47% of session):
- SHORT entries/year: ~400-1,000
- LONG entries/year: ~200-500
- Combined: ~600-1,500. n ≥ 500 met but tight on the lower bound.

Honest acknowledgement: this is the LOWEST-fire-rate of the three round-3 candidates. If sanity-check sample is ~700/year and validation halves it, the n ≥ 500 floor is at risk. Mitigation: extend sanity-check to 24 months 2023-24 to compensate.

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | index_stock_divergence_revert (proposed) |
|---|---|---|---|
| Indian-specific source | retail open momentum exhaustion | T+0 retail FOMO + operator pump | retail single-stock framing vs institutional NIFTY-relative benchmark |
| Direction | short-only | short-only | bidirectional, SHORT primary, NIFTY-trend gated LONG |
| Active window | T+0 09:15-09:30 | T+1 10:30 single bar | T+0 11:00-14:30 continuous |
| Universe | small_cap | mid_cap, small_cap (no F&O restriction) | full F&O-200 (no cap filter) |
| Hold | 15-30 min MIS | 4h 45m MIS | 60 min MIS |
| Evidence base | empirical sub-7 validation | 5 peer-reviewed papers | 5+ Indian retail-algo platforms + 2 academic |
| Anchor | structural (PDC) | structural (gap edges) | structural (spread z-score) |
| Mechanic class | single-asset OHLCV | single-asset + price-band CSV | **cross-asset** (only sub-9 setup using NIFTY 5m as input) |
| Correlation w/ existing | n/a | low | very low (different time + cross-asset trigger + larger universe) |

The setup complements: most differentiated mechanic in sub-9 (cross-asset spread), broadest universe (full F&O 200), distinct trigger (z-score on spread, not single-bar pattern). Lowest fire rate but highest mechanical novelty. Bidirectional design with NIFTY-trend gate on LONG is the cross-asset confirmation that distinguishes this from the long-bias-failed sub7/sub8 patterns.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check script
- [ ] REJECTED — reason
- [ ] RETIRE — kill candidate

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10).
