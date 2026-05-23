# `nifty_100_sector_divergence_intraday_revert` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:** Post 11-KILL meta-reframe (2026-05-22). User critique: today's candidates were retail-screenable patterns (Streak/Chartink-scannable), no large-cap focus. This brief responds: cross-sectional flow-asymmetry in NIFTY 100, NOT retail-screenable.
**Direction:** BILATERAL — LONG-revert for underperformers, SHORT-revert for outperformers (direction determined per-event by sign of divergence)
**Window:** Intraday MIS (square 15:25). Signal scan 10:30-13:30; entry at divergence-bar close; exit at 14:30 or VWAP-convergence.
**Portfolio rationale:** First large-cap setup AT ALL (existing 5 active setups are all small/mid-cap). First cross-sectional (pair/sector-relative) candidate. Bilateral direction fills BOTH portfolio gaps (currently 4 SHORT + 1 LONG, 0 large-cap).

## 1. Mechanism statement (ONE sentence)

NIFTY 100 stocks (top-100 by market cap, F&O-eligible) where the intraday stock-vs-sector relative return — sector being one of {NIFTY Bank, NIFTY IT, NIFTY Auto, NIFTY Pharma, NIFTY FMCG, NIFTY Metal, NIFTY Energy, NIFTY Realty, NIFTY PSU Bank, NIFTY Oil & Gas, NIFTY Finance} — diverges by >= ±1.0% at any 5m bar in 10:30-13:30 IST AND the divergence-bar has vol_ratio >= 1.5× cumulative-prior-mean (real institutional flow, not noise) revert toward the sector index within 60-120 minutes because sector-ETF tracker-fund AUM (NIFTY Bank ETF, NIFTY IT ETF, etc., aggregate Rs 50K+ cr) and pair-trade desks mechanically compress single-stock-vs-sector dispersion; trade is MEAN-REVERT (SHORT outpacers > +1%, LONG underperformers < -1%).

## 2. Falsifiers (3)

1. **Volume-flow signature (institutional vs noise):** Divergence bar must be elevated-volume to qualify as real institutional flow. Test: across 200+ fires, signal-bar vol_ratio median >= 1.5×. If median < 1.2× OR if >40% of fires have vol_ratio < 1.2×, divergences are noise (retail trades) not institutional positioning, and the mean-revert thesis doesn't apply → KILL.

2. **Convergence-direction signature:** Across signal events, the post-signal 60-min stock-sector relative return must show MEAN-REVERT direction (positive divergence → subsequent negative relative return; negative divergence → subsequent positive). Test: signed mean (post-signal_relative_return × -signal_sign) across all signals must be > 0 with t > 2.0. If random/no-direction, mechanism is wrong → KILL.

3. **Regime (sector ETF AUM dependency):** Mechanism depends on sector ETF AUM scale (compression flow). Pre-2023 NIFTY Bank ETF + NIFTY IT ETF AUM was a fraction of current scale. Per-regime PF CI lower bound > 1.0 must hold in at least 4 of 7 regimes including R1, R2, R5. If <4 regimes pass, mechanism is regime-conditioned to recent-only.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `sectoral_lag_catchup_long` | RETIRED | LONG | Sector-INDEX-breaks-PDH + LAGGARD continuation. DIFFERENT mechanism class (continuation, not mean-revert). | 0 (retired + opposite mechanism) |
| All current active setups (gap_fade, circuit_t1, or_window_failure, long_panic, delivery_pct) | active or disabled | various | Single-stock retail-FOMO triggers in small/mid-cap. **Zero overlap with NIFTY 100 cross-sectional cohort.** | 0 |

**Effective M estimate:** 0 — fully orthogonal mechanism class. Cleanest M-cost candidate in any 2026 brainstorm.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian sources to find (≥2 required)

1. **AMFI ETF AUM reports** — confirm NIFTY Bank ETF / NIFTY IT ETF / sector ETF AUM scale (cumulative Rs 50K+ cr indicates mechanical-compression force)
2. **NSE Indices methodology** — sector index composition + rebalance mechanics
3. **Tradetron / Streak pair-trade documentation** — Indian retail-pro operationalization of pair / sector-relative arbitrage
4. **Indian academic literature on cross-sectional momentum / dispersion in NSE** — Acharya/Pedersen-style analysis adapted to Indian equity
5. **Indian broker quant reports (Motilal Oswal, ICICI Sec, Kotak Sec)** — sector pair-trade or dispersion-arb publications

Acceptance: AMFI ETF AUM confirms compression-flow scale AND ≥1 source operationalizes "stock-vs-sector intraday divergence mean-revert" specifically on Indian large-caps.

### Gate B — Data feasibility (predicted PASS with infra work)

| Required data | On disk? |
|---|---|
| 5m bars per symbol | ✅ |
| NIFTY 100 list (top-100 by market cap) | ⚠️ Need build (similar to `assets/nifty_heavyweights.csv` but 100 names) |
| Sector mapping (stock → sector) | ⚠️ Need build (11 sectors per NSE Indices methodology) |
| NIFTY sector index intraday returns | ⚠️ Need compute (equal-weighted constituent basket as proxy if direct index data unavailable; market-cap-weighted preferred) |
| `ProductionUniverseGate` (Lesson #19) | ✅ |

**Verdict:** Gate B passes with one-time ~30min infra task (build NIFTY 100 list + sector mapping). All raw data is on disk.

## 5. Phase 2 plan (preview)

- **Universe:** NIFTY 100 (top-100 by market cap as of T-1), F&O-eligible (proxy via top-200 ADV from consolidated_daily), `ProductionUniverseGate` per-date
- **Sector mapping:** static map of stock → sector_id. 11 sectors.
- **Intraday signal:** for each 5m bar in 10:30-13:30 IST per (sym, date):
  - Compute `stock_return_intraday = (close[i] / open_of_day) - 1`
  - Compute `sector_return_intraday = market-cap-weighted (or equal-weighted) mean of all NIFTY 100 stocks in same sector` at the same bar
  - `divergence = stock_return_intraday - sector_return_intraday`
  - Trigger if `abs(divergence) >= 0.01` (1%) AND `vol_ratio >= 1.5×` (cumulative-prior-mean excluding current bar)
  - First-fire-per-day-per-stock latch
- **Direction:** `SHORT if divergence > 0`, `LONG if divergence < 0` (mean-revert)
- **Target:** measure post-signal 60-min stock-sector relative return (`signal_bar+12 close - signal_close`) and 120-min relative return. PnL_pct (raw, side-aware, no fees no leverage).
- **Acceptance:** signed-mean post-signal relative return must be MEAN-REVERT direction (signal_sign × post_relative_return < 0) with mean magnitude >= 0.20%. n_signal >= 500.
- **Required splits:** pre/post-2023 (ETF AUM scale), pre/post-SEBI-Oct-2025, per-sector (BankNIFTY vs IT vs Auto etc.), divergence-magnitude buckets, cap=large_cap-only vs mid_cap-only

## 6. Status checklist

- [ ] Gate A — ≥2 Indian sources cited (AMFI + pair-trade operationalization)
- [ ] Gate B — data feasibility (NIFTY 100 + sector mapping build)
- [ ] Falsifier #1 (volume-flow) pre-registered
- [ ] Falsifier #2 (mean-revert direction) pre-registered
- [ ] Phase 2 sector mapping methodology locked (market-cap-weighted vs equal-weighted)

## 7. Next action

Phase 1 supplementary research + NIFTY 100 list build + sector mapping build, then Phase 2 dispatch.
