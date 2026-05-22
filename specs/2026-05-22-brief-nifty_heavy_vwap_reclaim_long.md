# `nifty_heavy_vwap_reclaim_long` — Stage 0 brief

> **STATUS: KILLED at Phase 2 (2026-05-22)** — preserved as negative knowledge.
>
> **Reason:** mechanism has no measurable 5m-bar footprint over 2023-2026 Discovery window. Phase 1 Gate A + Gate B both PASSED cleanly, but empirical signature on top-10 NIFTY 50 heavyweights produced drift delta of just **+0.0064%** (signal n=424 vs baseline n=2383), 16× below the +0.10% Stage-2 floor.
>
> **Phase 2 evidence (script: `tools/sub9_research/phase2_nifty_heavy_vwap_reclaim_signature.py`, output: `reports/sub9_sanity/_phase2_nifty_heavy_vwap_reclaim_signature.csv`, 2807 rows):**
>
> | Metric | Value |
> |---|---|
> | Signal events (n) | 424 (well above 200 floor) |
> | Baseline events (n) | 2,383 |
> | Signal mean ret_to_1515 | -0.0069% |
> | Baseline mean ret_to_1515 | -0.0134% |
> | **DRIFT DELTA** | **+0.0064%** (vs ≥+0.10% floor) |
>
> **Pre/Post 2024 split is the key insight (Lesson #18 asymmetry — INVERSE of brief's thesis):**
>
> | Cohort | n_signal | delta |
> |---|---|---|
> | pre_2024 | 133 | +0.0266% (still below floor, but POSITIVE) |
> | post_2024 | 291 | **-0.0037%** (SIGN FLIPPED) |
>
> The brief's thesis was that FY25 +278% YoY inflow growth → post-2024 should be stronger. **Actual data shows the opposite** — post-AUM-acceleration period has WEAKER (slightly negative) signal. The Lesson #18 asymmetry warning fires: documented passive-AUM growth does NOT manifest in 5m-bar price footprint.
>
> **Per-symbol distribution balanced** (32-49 fires per name across 10 symbols, per-symbol means -0.086% to +0.066% centered near zero). Refreshing to top-20 (deferred Phase 4 prerequisite) is unlikely to convert +0.006% into +0.10%.
>
> **Conditions for revival:**
> 1. **Test at slower timescale.** ETF rebalance flow is documented at daily resolution; 5m bars may be too granular to capture the mechanical buy pressure. Try 15:00-15:25 hold window (peak ETF window per Indian sources) or 30-min target instead of 15:15 exit.
> 2. **Direct ETF-flow detection** (sub-5m): observe NIFTY 50 ETF (NIFTYBEES) tick flow directly and fade against the heavyweight basket. Requires sub-5m intraday ETF data which is not cached.
> 3. **Wait for post-2026-05-27 cohort to accumulate** (NSE rebalance methodology changed effective 2026-05-27 close — 5 days post-brief; no post-data exists yet). Even if accumulated, the regime-conditioned warning above suggests the methodology change is unlikely to restore a +0.1% footprint that wasn't there before.
>
> ---

**Date:** 2026-05-22
**Stage:** 0 — Idea (awaiting Phase 1 Indian-market research)
**Predecessor:** Brainstorming session 2026-05-22 (3-candidate batch)
**Direction:** LONG
**Window:** Intraday MIS (square 15:25)
**Portfolio rationale:** No active large-cap setup. Operates in a 14:30-15:15 window not covered by any active setup. Captures structural passive-flow tailwind (AMFI passive AUM growing yearly).

## 1. Mechanism statement (ONE sentence)

NIFTY-50 top-20-weight heavyweights trading below VWAP at 13:00-14:00 IST that reclaim VWAP after 14:30 on rising volume (vol >= 1.3x cumulative-prior-bars-mean) see passive ETF closing-flow plus index-rebalance-flow push price 0.3-0.5% above VWAP into 15:15 close, capturing the structural mechanical buy that grows yearly with AMFI passive AUM (Rs 50K+ cr NIFTY 50 ETF AUM as of 2026, +25% YoY).

## 2. Falsifiers (3 conditions that would invalidate the thesis)

1. **Mechanism falsifier (passive-flow correlation must hold):** If the passive-ETF thesis is right, fire-day performance should correlate positively with that day's NIFTY 50 ETF (NIFTYBEES, SETFNIF50) net inflow direction. Test: across 100+ events, signal-day mean return should be > baseline return on net-inflow days AND < baseline on net-outflow days. If correlation is zero or negative, mechanism is wrong → KILL.

2. **Regime falsifier (FII-exit regime + war-vol regimes):** Mechanism depends on net passive AUM inflow regime. During FII-exit regimes (R4) or war-volatility regimes (R7), ETF redemptions can dominate inflows, reversing closing-flow direction. Per-regime breakdown should show PF CI lower bound > 1.0 in R1+R2+R3+R5 but may fail in R4+R7. If PF CI lower bound < 1.0 in 4+ regimes including R1/R2/R5, mechanism is wrong → KILL.

3. **Infra falsifier (SEBI / NSE rebalance methodology change) — LIVE RISK 2026-05-27:** NSE Indices revised the rebalance methodology effective **2026-05-27 close** (5 days from brief date). Phase 2 sanity MUST split data into pre-2026-05-27 vs post-2026-05-27 cohorts and compare drift deltas. If the post-cutover cohort shows materially weaker drift (>= 50% reduction in mean drift OR PF CI lower bound below 1.0), the mechanism is regime-broken by the methodology change. Also monitor SEBI MF circulars for ETF rebalance-rule changes. Source: Business Standard 2026-05-18 article + NSE Methodology PDF.

## 3. Adjacent setups + correlation/effective-M assessment

| Setup | Status | Direction | Universe | Mechanism overlap | Correlation est. | M penalty |
|---|---|---|---|---|---|---|
| `gap_fade_short` | active | SHORT | small-cap | Different universe (large-cap vs small-cap) | Low | 0 |
| `circuit_t1_fade_short` | active | SHORT | mostly small/mid | Different universe + window | Low | 0 |
| `or_window_failure_fade_short` | active | SHORT | mid/small | Different universe + window | Low | 0 |
| `long_panic_gap_down` | active | LONG | small-cap | Different universe, different trigger | Low | 0 |
| `delivery_pct_anomaly_short` | disabled | SHORT | mostly small | Different universe + window | Low | 0 |

**Effective M estimate (Harvey-Liu input):** 0 — large-cap universe is independent of all current setups, and 14:30-15:15 window has no existing fires. This is the cleanest M-cost candidate in the brainstorm.

**Portfolio impact if shipped:** would add first large-cap LONG to portfolio, occupying a previously empty (universe × window) cell.

## 4. Phase 1 research outline (Gate A + Gate B)

### Gate A — Precedent (PASSED 2026-05-22)

Sources confirmed via Phase 1 research agent:
1. **AMFI Annual MF Report FY2025** — ETF + index-fund AUM grew 27% (Dec 2024 Rs 11.11 lakh cr → Nov 2025 Rs 14.07 lakh cr); index-fund inflows +278% YoY to Rs 59,306 cr in FY25. Confirms passive-flow scale >> Rs 50K cr threshold and structural growth.
2. **SBI Nifty 50 ETF (SETFNIF50) AUM** — Rs 1,91,909 cr single-fund AUM (May 2026), ~4x the brief's original Rs 50K cr assumption. NIFTYBEES is comparable scale.
3. **NSE Indices Methodology** — quarterly rebalance with tracker funds mechanically buying/selling at index reconstitution; weight redistribution rule **effective 2026-05-27 close** (Phase 2 must split pre/post).
4. **Motilal Oswal — Index Rebalancing (Jun 2025)** — Indian broker quant note explicitly identifies that index funds/ETFs tracking NIFTY 50 must purchase shares to align with index composition, causing predictable close-flow price impact.

**Acceptance threshold met:** AMFI passive-AUM scale + NSE methodology + Motilal Oswal broker operationalization. All Indian-specific, no US/forex cargo-culting.

### Gate B — Data feasibility (PASSED 2026-05-22 with prerequisite task)

| Required data | On disk? | Source | Notes |
|---|---|---|---|
| 5m bars for NIFTY heavyweights | ✅ | `backtest-cache-download/monthly/*_5m_enriched.feather` | Top-20 by weight always have data (large-cap full history) |
| NIFTY heavyweight list | ⚠️ PARTIAL FAIL | `assets/nifty_heavyweights.csv` | **STALE: last refresh 2025-04-29 (~13 months), only 10 names** (commit `9e7c99c` from retired `expiry_pin_strike_reversal`). **PHASE 2 PREREQUISITE:** refresh to current top-20 NIFTY 50 weights before sanity run. |
| VWAP cumulative compute | ✅ | Standard intraday cumulative | Use bars[:i+1] only (Lesson #5 anti-look-ahead) |
| ProductionUniverseGate (Lesson #19) | ✅ | `tools/sub9_research/production_universe.py` | Supports cap filter + zero legacy filters per Lesson #17 |
| ETF net inflow daily | ⚠️ SCRAPEABLE | NOT on disk; available at AMFI monthly notes + Nifty Indices quarterly Passive Insights PDF | Phase 5 falsifier #1 covariate; scrape feasibility confirmed by Phase 1 research |
| AMFI passive AUM | ✅ | Phase 1 sources cited (AMFI Annual MF Report FY2025) | Confirmed Rs 14.07L cr aggregate, Rs 1.92L cr in SETFNIF50 alone |

**Verdict:** Gate B passes for core mechanic. Heavyweight CSV refresh is a Phase 2 prerequisite task (not a blocker for moving forward).

## 5. Phase 2 empirical signature plan (preview only)

Once Phase 1 confirms precedent:

- **Universe:** top-20 NIFTY 50 names by weight (refresh from current NSE methodology).
- **Signal definition:** for each (sym, date), check if 14:00 close < VWAP. Then identify the first 14:30-15:00 bar where close >= VWAP AND vol_ratio (vs cumulative prior-bars-mean) >= 1.3.
- **Baseline:** all (sym, date) where 14:00 close < VWAP (no further filter).
- **Drift measure:** signed mean return signal_event → 15:15, vs baseline.
- **Acceptance threshold:** ≥ +0.1% drift delta (Stage 2 kill floor).
- **Secondary check:** stratify by NIFTYBEES daily volume (proxy for ETF flow if direct data unavailable).

## 6. Status checklist for advance to Phase 2

- [x] Gate A — ≥2 Indian sources cited (AMFI + NSE methodology + Motilal Oswal broker quant; PASS 2026-05-22)
- [x] Gate B — data feasibility verified (PASS with prerequisite task: heavyweight CSV refresh)
- [ ] **Prerequisite: Index-weight refresh for `assets/nifty_heavyweights.csv` to current top-20 NIFTY 50 weights** (current file is 10 names, 13mo stale)
- [x] Adjacent setup correlation re-confirmed (currently 0)
- [ ] **Phase 2 split required: pre-2026-05-27 vs post-2026-05-27 (NSE rebalance methodology change effective close 2026-05-27)**
- [ ] **Phase 2 stability check: pre-2024 vs post-2024 cohorts** — most passive AUM accumulated 2024-2025 (FY25 +278% inflow growth). Required to distinguish structural edge from regime-conditioned (post-2024 only) edge. Lesson #18 asymmetry warning applies.

## 7. Next action

Phase 1 research (Gate A + Gate B verification) — runs as a parallel agent task per session plan 2026-05-22.
