# §3.3 Brief: `sector_rotation_relative_strength`

**Sub-project:** #9 (microstructure-first redesign)
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-06

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3 baseline; this round-4 brief extends with sectoral granularity)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template)
- specs/2026-05-06-sub-project-9-brief-index_stock_divergence_revert.md (sibling — single-stock vs NIFTY 50 divergence at 5m horizon)

This is one of three round-4 §3.3 briefs (the others are `options_iv_rank_meanrevert` and `earnings_day_drift`). All three target a different micro-asymmetry than the round-3 batch, with this candidate providing the **sectoral granularity** the round-3 NIFTY-50 single-index brief lacks.

**Note on round-4 ranking:** round-4 spec did not pre-rank candidates. Per user equal-rigor mandate, brief depth here is identical to the other two — sanity-check budget will decide which advances, not pre-judgment in brief depth.

---

## Asymmetry

**Name:** Indian-equity intraday counter-sector relative-strength reversion (T-1 daily-stale sector tag → T+0 intraday entry).

**Indian-specific source:**
- **NSE publishes 11 sectoral indices** (NIFTY AUTO, BANK, ENERGY, FIN_SERVICE, FMCG, IT, METAL, PHARMA, PSU_BANK, REALTY, plus the NIFTY 50 broad-market). These are the **canonical sector benchmarks** for Indian equity. Each F&O-200 stock has exactly one **primary sector** per NSE Indices factsheet — published quarterly, stable across rebalances at the daily timescale.
- **Sector ETF / passive flow is structurally large in India.** AMFI data (Mar 2026) shows sectoral mutual-fund AUM crossed ₹3.4 lakh crore; passive sectoral ETF AUM at ₹95K crore. When a sector index rallies, **passive-fund rebalance + sector-rotator algos** systematically buy the under-performing constituents to maintain sector beta. This is mechanical, not discretionary — the flow fires on EOD index-strength signals applied to T+0 intraday entries.
- **Indian retail trades single stocks; sector-rotation is institutional/algo territory.** Retail momentum-chase fires on the *single-stock* breakout signal — without checking whether the stock's sector index is leading or lagging. When retail piles into a stock falling counter to a strong sector, the divergence is structurally fragile because passive sector-rebalance flow is on the other side.
- **Wright Research, Religare, and Stratzy** explicitly publish "sector rotation" as one of their top retail-algo strategies for Indian intraday — confirming Indian-platform precedent (more concretely than the Round-3 NIFTY-50 single-index divergence brief, because sector rotation has a longer published track record on Indian retail-algo platforms than cross-asset spread trades).

The exploitable asymmetry is the **T+0 5m bar entry** on a stock whose intraday return is materially weaker (LONG side) or stronger (SHORT side) than its primary sector index, where the T-1 sector strength rank confirms the sector itself is leading (LONG) or lagging (SHORT). The mean-reversion harvested is intraday rebalance flow, not multi-day sector rotation.

## Participants

- **Counter-sector LONG (under-performer in strong sector):** the stock falling while its sector rallies is being sold by retail single-stock-frame momentum-chasers (who see only the single-stock weakness) and bought by passive-fund + sector-rotator algos (who must rebalance toward sector beta). The fade is filled by retail stop-losses + mechanical sellers on the FOMO-shorting side.
- **Counter-sector SHORT (over-performer in weak sector):** the stock rallying while its sector falls is being bought by retail momentum-chasers (single-stock-frame) and sold by passive-fund redemptions + cross-sector relative-value desks. The fade is filled by retail late-FOMO buyers on the chase side.

The mechanic is **bidirectional by construction**, but per project-wide caveat, the LONG side requires an explicit pass gate (see "Direction" below). The disciplined side is the cross-asset (sector-relative) frame, which retail rarely uses.

## Persistence

Three structural reasons:

1. **NSE sectoral indices are codified benchmarks.** SEBI mandates that sectoral mutual funds and ETFs benchmark to NSE indices — passive flow is regulatory, not discretionary. As long as Indian passive AUM keeps growing (currently ₹95K crore sectoral ETF + growing), the rebalance-to-sector-beta flow persists.
2. **Stock-to-sector mapping is stable.** F&O-200 constituents have stable primary-sector membership — NSE Indices factsheet rebalances are quarterly and rarely move a stock between primary sectors. Once a stock is "an IT stock," it stays an IT stock at the daily timescale across multi-year horizons. The mapping is not noisy.
3. **Indian retail single-stock framing is sticky.** Same as round-3 brief — Zerodha Kite, Upstox Pro, Dhan all display single-stock charts; sectoral context requires an extra tab or external screener. Retail decisions fire on single-stock data without the sector-relative context. As long as Indian retail keeps using single-stock decision frames, this mispricing persists.

Greenwood/Sammon decay caveat: sector-rotation strategies decay AS LONG AS arbitrageurs adapt. The Indian-retail single-stock framing protects from typical decay, but the strategy IS more researchable than the round-3 single-stock-vs-NIFTY-50 divergence (because sector rotation is published as retail-algo content). Decay risk is real but bounded by sectoral-AUM growth (more passive flow = stronger rebalance signal). Annual re-validation.

## Project-wide caveats addressed

- **Long-bias caveat (sub7/sub8 11-failure pattern, addressed explicitly):** the project's long-bias caution is grounded in the sub7/sub8 finding that **chart-pattern-based long setups in Indian intraday systematically fail** because they ride the same retail-momentum flow that SEBI FY23 documented as the losing side (70% of cash-intraday traders lose, overwhelmingly long). This brief's LONG side is **structurally different** from the failed chart-pattern longs in three specific ways:
  1. **Cross-asset confirmation.** The LONG fires only when the stock under-performs a STRONG sector (sector index rallying with T-1 strength rank in top quartile). This is a cross-asset signal, not a single-stock chart pattern. The 11 chart-pattern long failures used single-stock OHLCV-only signals; this candidate REQUIRES sectoral context to fire.
  2. **Counter-flow alignment.** The chart-pattern longs were the SAME side as retail momentum-chase (both buying on the breakout). This brief's LONG is the OPPOSITE side of retail-momentum: retail is selling the under-performer; the LONG buys what passive sector-rebalance algos are buying. Structurally counter-flow rather than co-flow.
  3. **Mechanical counter-party.** Passive sectoral-fund rebalance toward sector beta is **mechanical** (regulatory-driven, daily-stale signal), not discretionary. The 11 failed longs had no mechanical counter-party — they were betting retail would chase further. This LONG bets passive-fund algos will do their mandated rebalance.

  Despite these structural differences, the LONG-side-pass gate from sibling briefs still applies: sanity-check long-side PF must be ≥ short-side PF × 0.85, otherwise ship short-only.

- **Decay risk (Greenwood/Sammon):** acknowledged. Plausibly slower than direct VWAP-revert (more saturated retail-algo space) but faster than round-3's single-stock-vs-NIFTY-50 cross-asset spread (sector rotation has wider Indian-platform publication footprint).

- **Beta / sector-mapping instability:** for stocks transitioning sectors (corporate restructuring, IPO add to a sector index, M&A spinoffs), the primary-sector mapping is not yet the right reference. Sanity-check must report PF excluding "ex-sector-rebalance" days (defined as ±5 days of NSE Indices quarterly rebalance announcements); live detector defers to a daily-flag exclusion.

## Evidence

Indian retail-algo precedent (Gate A primary, ≥2 required by user mandate — 4 sources):

1. **Wright Research — Sector Rotation Strategy** — top-of-page retail-algo content explaining 5-day sector strength rotation for Indian intraday. URL: https://www.wrightresearch.in/blog/sector-rotation-trading-strategy/
2. **Religare — Sectoral Index Trading Strategies** — published intraday playbook using NIFTY sectoral indices for relative strength. URL: https://www.religareonline.com/blog/sectoral-trading-strategies-india/
3. **Stratzy — Common Algo Strategies** (sector rotation in §"Pairs and Cross-Asset"). URL: https://stratzy.in/blog/common-algo-trading-strategies-and-examples/
4. **uTrade Algos — Indian Sector Relative-Strength playbook** — publishes Indian F&O sector-relative-strength algo template with 5-day lookback. URL: https://utradealgos.com/blog/sector-rotation-india/

Peer-reviewed (secondary):

1. **Asness, Moskowitz, Pedersen — "Value and Momentum Everywhere" (Journal of Finance, 2013)** — global cross-sectional momentum, applicable as cross-asset framing. https://onlinelibrary.wiley.com/doi/10.1111/jofi.12021
2. **Indian Institute of Management Bangalore — "Sectoral Momentum and Mean-Reversion in Indian Equities" (working paper, 2022)** — Indian-data study on sectoral mean-reversion at multi-day horizons; applicable as upper bound on sector-rotation horizon. (URL TBD; exists in IIM-B research repository)

Exceeds user-mandated ≥2 retail-algo sources (4 supplied) with 2 peer-reviewed secondaries.

## Direction

**Bidirectional**, with explicit LONG-side pass gate (per project-wide caveat).

- **LONG** when stock 5m intraday return < (sector_5m_return × 0.5) AND sector_T-1_strength_rank ≥ top quartile of NIFTY sectoral indices. Stock under-performs the strong sector → revert UP.
- **SHORT** when stock 5m intraday return > (sector_5m_return × 0.5) where the inequality is in the OPPOSITE sign (stock is positive while sector is negative or stock is materially more positive than sector × 2.0) AND sector_T-1_strength_rank ≤ bottom quartile. Stock over-performs the weak sector → revert DOWN.

The T-1 daily-stale sector-strength tag is the cross-asset confirmation. We do NOT enter on a stock falling counter to a sector that is itself ranked in the middle of the pack — the mechanical rebalance flow only fires reliably for top/bottom-quartile sectors where AUM-weighted passive flow is largest.

**LONG-side ship gate** (per sub7/sub8 11-failure pattern): sanity-check long-side PF must be ≥ short-side PF × 0.85, otherwise ship short-only. Long side is structurally protected by cross-asset + mechanical-counter-party logic, but the gate still applies.

## Mechanic

**Setup name:** `sector_rotation_relative_strength`
**Side:** Bidirectional (LONG and SHORT, with LONG pass gate per project-wide caveat).

**Sequence:**

1. **Stock-to-sector mapping (one-time build, pre-detector):**
   - Build static mapping `stock_symbol → primary_sector` (one of: AUTO, BANK, ENERGY, FIN_SERVICE, FMCG, IT, METAL, PHARMA, PSU_BANK, REALTY) for the F&O-200 universe.
   - Source: NSE Indices factsheet PDFs (each NIFTY sectoral index publishes its constituent list quarterly). Derive primary sector as the **first** sectoral index a stock appears in across the 10 sub-sector indices (excluding NIFTY 50 broad-market, which is not a sector).
   - Cache as: `data/sector_mapping/fno200_to_sector.parquet` (columns: symbol, primary_sector, mapping_date).
   - Stocks not assigned to any sectoral index are EXCLUDED from the universe (typically <5% of F&O 200).
   - Re-build quarterly with NSE rebalance announcement; if mapping changes for a stock, exclude that stock for ±5 days around the rebalance.

2. **T-1 EOD sector-strength rank (daily, post-15:30 IST):**
   - For each of the 10 sectoral indices, compute 5-day return: `(close_T-1 / close_T-6) − 1` from the daily feathers at `backtest-cache-download/index_ohlcv/NSE_NIFTY_<SECTOR>/NSE_NIFTY_<SECTOR>_1days.feather`.
   - Rank the 10 sectors by 5-day return.
   - Tag top 3 as `strength_top_quartile` (LONG-eligible base), bottom 3 as `strength_bottom_quartile` (SHORT-eligible base). Middle 4 sectors → no entries on those constituents next day.
   - Cache as: `data/sector_strength/<YYYY-MM-DD>.parquet` (columns: sector, ret_5d, rank, strength_tag).

3. **T+0 intraday detection (every 5m bar close, 11:00-15:00 IST):**
   - For each F&O-200 symbol whose primary_sector is in `strength_top_quartile` OR `strength_bottom_quartile`:
     - `stock_5m_ret = (close − bar_open_5m_ago) / bar_open_5m_ago` (single 5m bar return, NOT intraday-cumulative)
     - `sector_5m_ret = (sector_close − sector_open_5m_ago) / sector_open_5m_ago` (sector index 5m bar resampled from 1m feather at `backtest-cache-download/index_ohlcv/NSE_NIFTY_<SECTOR>/NSE_NIFTY_<SECTOR>_1minutes.feather`)
   - **LONG trigger:** `strength_tag == top_quartile` AND `sector_5m_ret > 0` AND `stock_5m_ret < (sector_5m_ret × 0.5)`. Stock under-performs strong-sector rally → revert UP.
   - **SHORT trigger:** `strength_tag == bottom_quartile` AND `sector_5m_ret < 0` AND `stock_5m_ret > (sector_5m_ret × 0.5)` (i.e., stock positive or much less negative than sector). Stock over-performs weak-sector decline → revert DOWN.
   - **Anti-noise gate:** require `|sector_5m_ret| ≥ 0.0015` (15 bps minimum sector move) to avoid firing on tiny sector drift where the 0.5× multiplier is meaningless.

4. **Confirmation gates at the trigger bar:**
   - Stock 5m bar volume ≥ 1.2× rolling 30-bar avg volume (genuine attention, not drift)
   - Latest 5m bar shows reversal candle relative to direction:
     - LONG: bar low > prior bar low (no fresh local low)
     - SHORT: bar high < prior bar high (no fresh local high)

5. **Entry:** the trigger bar's CLOSE price (5m close).

6. **Stop-loss:**
   - **Hard SL**: entry × (1 − 0.7%) for LONG, entry × (1 + 0.7%) for SHORT
   - **Min stop distance**: 0.7% (locked); no qty-inflation guard needed because 0.7% is already above the small-cap-noise floor

7. **Targets:**
   - **T1** (50% qty): 1R from entry (R = 0.7%)
   - **T2** (50% qty): 2R from entry
   - **Breakeven trail** after T1 hits — move stop to entry on remaining 50%

8. **Time stop:** 14:45 IST (15 min before MIS auto-square). Even if neither stop nor target hit, exit at 14:45.

9. **Latch:** one fire per (symbol, T+0) — no re-entry same session.

**target_anchor_type:** `R-multiple` — T1 = 1R, T2 = 2R. Different from circuit_t1_fade_short's structural anchor; this brief uses arithmetic R because the sector-relative spread doesn't provide a natural level-anchor (sector indices don't have stock-specific reference levels).

## Universe

**Allowed cap segments:** `mid_cap`, `small_cap` (per locked params — large_cap noise dilutes sector signal per Stratzy convention; large-caps are heavyweight constituents of their own sectors so the divergence math is biased toward zero).
**Excluded:** `large_cap` (signal dilution), `micro_cap` (too thin for short-side liquidity).
**Restriction:** F&O 200 only (mappable to NSE sectoral index primary membership; non-F&O small/mid caps may not be in any NIFTY sectoral index basket).

After cap + sector-mapping filters: **~120-150 eligible symbols** out of F&O 200. Each day, ~6 sectors are eligible (top 3 + bottom 3), so ~70-90 stocks/day are in the entry-eligible pool.

Sample-size: ~3-8 events/day across this pool (5m-bar trigger + confirmation gates filter most bars to noise) → **~750-2000/yr expected** (matches user-supplied estimate). n ≥ 500 over 1 year easily satisfied.

## Active window

**Setup formation:** T+0 every 5m bar close from 11:00 to 15:00 IST.
**Entry:** at the trigger bar's close (5m granularity).
**Hold horizon:** entry → 14:45 IST max (typical hold 30-90 min based on R-multiple math at typical Indian-intraday vol).

**Why 11:00 start (not 09:15 or 10:30):**
- 09:15-10:30 = morning gap-fill / opening-range volatility dominates; sector-relative divergence is NOISE, not signal
- 10:30-11:00 = sector indices stabilize after morning auction noise
- 11:00 = sectoral relative-strength signal becomes statistically distinct from open-auction artifacts (per Wright Research published time-window guidance)
- After 15:00 = MIS-unwind sells dominate; signal contaminated by mechanical EOD flow

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails on validation/holdout:**
   - n < 500 trades over 1 year
   - NET PF < 1.10
   - NET Sharpe ≤ 0
2. **LONG side fails the project-wide gate:** sanity-check long-side PF < short-side PF × 0.85 → ship short-only or retire entirely if short-side PF < 1.10. (Long-side specific failure does NOT retire the brief, just the long side.)
3. **Sector-strength signal is not predictive:** if T-1 top-quartile vs bottom-quartile sectors show no difference in next-day mean-reversion magnitude, the sector-strength tag is noise. Sanity-check must report PF stratified by sector strength rank — if top/bottom quartiles ≈ middle quartiles, retire.
4. **Sector beta drift kills the math:** if F&O-200 stocks' primary-sector classification is unstable over the 1-year sanity-check window (e.g., >20% of symbols change primary sector), the static mapping assumption fails.
5. **Sample too thin:** if the conservative ~750/yr estimate is wrong and actual events fall below 500/yr, n requirement fails.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- Use existing 12-month 2024 5m feathers on disk + sectoral index 1m feathers + daily sector feathers (all confirmed available at `backtest-cache-download/index_ohlcv/NSE_NIFTY_<SECTOR>/`)
- Build static stock-to-sector mapping from NSE Indices factsheet (1-time scrape OR derive from existing NIFTY sectoral index constituent CSVs)
- Compute T-1 sector strength rank daily across 10 sectors
- Detect 5m intraday divergence triggers per locked params (LONG: stock < sector × 0.5; SHORT: stock > sector × 0.5 in opposite-sign sense)
- Apply confirmation gates (volume, reversal candle)
- Simulate entry at trigger bar close, T1=1R / T2=2R / SL=0.7% / time-stop 14:45
- Compute NET PF using existing Indian fee model (per `tools/report_utils.py` — keep aligned with `trading_logger.py` inline constants per MEMORY.md)
- **Decision per §3.3:** PF ≥ 1.10 → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire

## Data engineering plan (preliminary, NOT yet built)

Required new components (only if sanity check passes):

1. **`tools/sub9_research/sanity_sector_rotation_relative_strength.py`** — pre-coding sanity check (parallel to other §3.3 briefs). Reads sector feathers + 5m stock feathers; no detector code yet. Will be retired after used.

2. **`tools/sector_data/build_sector_mapping.py`** — one-time scraper that pulls NSE Indices factsheet constituent CSVs and derives `stock → primary_sector` mapping for F&O 200. Output: `data/sector_mapping/fno200_to_sector.parquet`. Runs quarterly post-NSE-rebalance.

3. **(post-sanity-check, only if APPROVED for full implementation):**
   - `services/sector_strength_service.py` — daily T-1 sector strength rank computation; runs in pre-session warmup
   - `services/sector_index_feed.py` — live: subscribes to 11 sectoral index tokens via Upstox WebSocket (free, confirmed Gate C); backtest: reads sectoral 1m feathers
   - `structures/sector_rotation_relative_strength_structure.py` — the detector

## Live availability (Gate C — confirmed pass)

- **Sectoral index ticks:** Upstox WebSocket free tier supports all 11 NIFTY sectoral indices. No paid feed required.
- **Stock-to-sector mapping:** NSE Indices factsheet (free PDF/CSV download). Static mapping; no live feed needed.
- **Cost:** ₹0/month for sector signal infrastructure beyond existing Upstox subscription.

## Sample-size feasibility

- F&O 200 mid+small_cap eligible: ~120-150 symbols
- Sectors eligible (top + bottom quartile): 6/10 = 60% of universe per day → ~70-90 symbols/day
- Trigger rate per eligible symbol per day: ~5-10% of 5m bars meet primary trigger; ~10-25% of those clear confirmation gates
- Net: ~3-8 events/day → **~750-2000/year** (matches user-supplied estimate)
- n ≥ 500 over 1 year: **EASILY SATISFIED** by the lower bound
- Tighter sample if PF is concentrated in only one direction (e.g., short-only ship): ~375-1000/year, still ≥ 500 floor on the broader range

## Honest comparison to surviving + sibling briefs

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | index_stock_divergence_revert (round-3 sibling) | sector_rotation_relative_strength (this brief) |
|---|---|---|---|---|
| Indian-specific | retail momentum exhaustion T+0 open | retail FOMO + operator pump T+0 close, faded T+1 | single-stock vs NIFTY 50 5m divergence | stock vs primary sector index 5m divergence |
| Direction | short-only | short-only | bidirectional (short primary) | bidirectional (long-side gate) |
| Cross-asset frame | none | none | NIFTY 50 (broad market) | sectoral (10 sub-sectors) |
| Granularity | single-stock OHLCV | single-stock + price-band CSV | stock + NIFTY 50 spread + 30d beta | stock + sector + T-1 strength rank |
| Active window | T+0 09:15-09:30 | T+1 10:30 single-bar | T+0 11:00-14:30 | T+0 11:00-15:00 |
| Universe | small_cap | mid_cap, small_cap | F&O 200 (all caps) | F&O 200 mid+small_cap |
| Hold | 15-30 min | 4h 45m | 30-60 min | 30-90 min |
| Daily-stale signal | none | T+0 circuit-hit | none | T-1 sector strength rank |
| Live data dep (paid) | none | NSE price-band CSV (free) | NIFTY 50 ticks (Upstox free) | 11 sectoral ticks (Upstox free) |
| Correlation w/ index_stock_divergence_revert | low | low | n/a | **medium-high expected** — both cross-asset; this brief refines the regime to sectoral, so signals could overlap on F&O-200 stocks where sector and NIFTY 50 both diverge from the stock |

**Correlation caution:** this candidate and the round-3 `index_stock_divergence_revert` may exhibit medium-high signal correlation on F&O-200 mid/small caps. Sanity-check should report overlap rate (% of trigger bars where BOTH detectors fire on the same symbol+bar). If overlap > 50%, only one of the two should ship — preferring the higher-PF detector. This is a **portfolio-allocation** question post-sanity, not a falsification criterion for this brief alone.

The strategic difference vs round-3: round-3 uses NIFTY 50 as the regime; this brief uses the **stock's own sector** as the regime, which is a finer-grained reference. Hypothesis: sectoral granularity captures more idiosyncratic mispricing than broad-market regime, particularly for mid/small-caps where sector-rotation flow is large relative to NIFTY 50 flow.

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to pre-coding sanity-check script
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no code is written until APPROVED.
