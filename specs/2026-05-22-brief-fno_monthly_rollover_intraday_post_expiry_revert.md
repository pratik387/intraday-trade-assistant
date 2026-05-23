# `fno_monthly_rollover_intraday_post_expiry_revert` — Stage 0 brief

**Date:** 2026-05-22
**Stage:** 0 — Idea (Phase 1 Gate A + Gate B complete in this brief)
**Predecessor:** Brainstorming session 2026-05-22 (F&O rollover candidate, direction-picking from 5 hypothesis menu)
**Direction:** Mean-revert (intraday LONG-and-SHORT, side picked by morning gap sign on T+1)
**Window:** Intraday MIS (square 15:25)
**Portfolio rationale:** No active setup operates on **T+1 morning after monthly expiry** specifically. Covers a once-a-month (~12 days/year) gap in the calendar where institutional position-reset + cash-settlement clearing produces measurable mean-revert footprint.

---

## Direction-picking rationale (why this and not the others)

The brainstorm menu had 5 candidate directions on the F&O rollover calendar. Pre-existing work in `specs/` shows:

| # | Hypothesis | Status in repo | Decision |
|---|---|---|---|
| 1 | T-2/T-3 institutional roll basis dislocation | `specs/2026-05-09-sub-project-9-brief-fno_monthly_rollover_basis_dislocation_T2.md` — **CONDITIONAL-KILLED**: sibling `stock_futures_basis_convergence_T1` already failed sanity (NET PnL **-₹106,221 on n=2670**, mean -₹39/trade, WR 41.9%; reports/sub9_sanity/stock_futures_basis_convergence_trades.csv). Same cost structure kills T-2 the same way. | **SKIP** |
| 2 | **Post-expiry T+1 mean-revert** | **NOT in repo. No spec, no detector, no sanity.** Grep `post.expiry|T_plus_1|day.after.expiry|expiry.spillover` returned zero matches. | **PICK** |
| 3 | Expiry-Thursday afternoon pin-magnet | `structures/expiry_pin_strike_reversal_structure.py` + brief; pending OI re-capture | **SKIP (covered)** |
| 4 | Pre-expiry Wed+Thu morning institutional unwind SHORT | Adjacent to (1)/(3); shares cost structure with sibling; not novel from a cash-equity directional angle | **SKIP** |
| 5 | Other | n/a | n/a |

**Picked direction: #2 (post-expiry T+1).** Reason: it is the only F&O-rollover-calendar mechanism with **zero prior implementation, zero prior brief, and Indian-equity peer-reviewed evidence that explicitly documents the day-after-expiry footprint** (see Gate A below — Narang & Vij 2013 names "build up further on the day after expiry" as the key finding).

---

## 1. Mechanism statement (ONE sentence)

On T+1 (the trading session immediately AFTER NSE monthly F&O expiry — Wednesday post-Tuesday-expiry under the FAOP68747 calendar effective 2025-09-01; Friday post-Thursday-expiry under the prior calendar), F&O-liquid-200 single-stock-futures underlyings exhibit a measurable **morning gap reversal** in the 09:15-10:30 window because (a) overnight institutional position resets (FII/MF basket adjustments after expiring leg closes), (b) cash-settlement payment clearing produces unwind flow into the spot leg, and (c) retail F&O traders who held to T+0 close are forced to re-enter into new front-month — the three flows create a transient supply/demand imbalance that mean-reverts within the first hour (Narang & Vij 2013 IJSRN: "spot returns, volume, and volatility are high on expiration day and they build up further on the day after expiry").

## 2. Falsifiers (3 conditions that would invalidate the thesis)

1. **Mechanism falsifier (must be tied to F&O calendar, not generic morning-gap).** If T+1 morning gap-reversal performs no better than a matched non-expiry-week Wednesday/Friday baseline (`mean_drift_T+1 - mean_drift_matched_non_expiry < +0.10%`), the F&O-rollover anchor is fake — the mechanism is just generic morning mean-revert dressed up with a calendar filter. Phase 2 must compute matched-day baseline. KILL.

2. **Regime falsifier (calendar-change asymmetry).** NSE shifted monthly expiry from **last Thursday** to **last Tuesday** effective 2025-09-01 (NSE Circular FAOP68747). The post-expiry T+1 day therefore shifted from **Friday → Wednesday**. Friday and Wednesday have different intraday seasonality (Friday end-of-week unwind vs Wednesday mid-week steady). If pre-Sep-2025 PF (Friday T+1) ≠ post-Sep-2025 PF (Wednesday T+1) with |ΔPF| > 0.4 absolute, the mechanism is calendar-day-of-week-dependent and not robust → KILL. Note: in `data/futures_basis/` 30 of 41 expiries are Thursday (pre-2025-09), 8 are Tuesday (post-2025-09), 2 Wednesday, 1 Monday (holiday-adjusted). Phase 2 must report side-by-side.

3. **Infra falsifier (5m bar coverage on F&O 200 universe).** F&O liquid 200 includes some thinly-tracked tickers (e.g., `AEGISLOG`, `ANANTRAJ` in `assets/fno_liquid_200.csv`); if these have spotty 5m coverage in `backtest-cache-download/monthly/*_5m_enriched.feather`, the per-event filter must rely on `ProductionUniverseGate` (Lesson #19, Lesson #16). If post-gating n_events < 500 over 41 expiries × 200 symbols = 8,200 possible events worst-case, the candidate has insufficient power for stable PF estimate → KILL pre-OOS.

## 3. Adjacent setups + correlation/effective-M assessment

| Setup | Status | Direction | Universe | Window | Mechanism overlap | Correlation est. | M penalty |
|---|---|---|---|---|---|---|---|
| `gap_fade_short` | active | SHORT | small-cap | T+0 09:15-09:30 | Same morning window, **different universe (small-cap not F&O 200)** + different trigger (no expiry anchor) | Low-mod | 0.5 |
| `circuit_t1_fade_short` | active | SHORT | mid/small | T+1 10:30 | Different trigger (T-1 circuit) + different universe (mostly non-F&O small/mid) | Low | 0 |
| `or_window_failure_fade_short` | active | SHORT | mid/small | T+0 morning OR | Different anchor (opening range failure, not expiry calendar) | Low | 0 |
| `long_panic_gap_down` | active | LONG | small-cap | T+0 morning | Different trigger (panic gap-down, not expiry) | Low | 0 |
| `expiry_pin_strike_reversal` | pending OI | both | NIFTY heavyweights | **expiry T+0 13:00-15:00** | Same calendar, **different DAY (T+0 vs T+1)** and different universe (NIFTY 50 heavyweights vs F&O 200) | Low | 0.5 |
| `stock_futures_basis_convergence_T1` | **FAILED sanity** | both | F&O 200 | T-1 11:00 | Same universe, **different DAY (T-1 vs T+1)** and **different mechanic (basis convergence vs post-reset revert)** | Low | 0 (failed peer, no live overlap) |
| `fno_monthly_rollover_basis_dislocation_T2` | conditional-killed | both | F&O 200 | T-2/T-3 11:00 | Same universe, different day (pre-expiry vs post-expiry) | Low | 0 (killed peer) |

**Effective M estimate (Harvey-Liu input):** 1 (one half-penalty from `gap_fade_short` because of shared 09:15-10:30 morning window, even though universe and trigger differ; one half-penalty from `expiry_pin_strike_reversal` because of shared monthly-expiry-calendar anchor). Conservative penalty: **M = 1** for Lesson #15 confidence-card haircut.

**Portfolio impact if shipped:** adds first F&O-eligible-universe directional setup that fires on a **named calendar slot (12 days/year)** rather than a price-pattern-trigger setup. Complementary to `expiry_pin_strike_reversal` (T+0 afternoon) and `gap_fade_short` (any morning).

## 4. Phase 1 research outline (Gate A + Gate B)

### Gate A — Indian precedent (PASS — 2026-05-22)

Sources verified via WebSearch this session:

1. **Narang & Vij 2013 — "Long-Term Effects of Expiration of Derivatives on Indian Spot Volatility," International Scholarly Research Notices (Wiley):** explicitly documents "spot returns, volume, and volatility are high on expiration day and they **build up further on the day after expiry** which shows that the Indian market is weakly efficient." This is the load-bearing Indian peer-reviewed evidence for the T+1 effect (not T-1, not T+0 — specifically the day after). URL: https://onlinelibrary.wiley.com/doi/10.1155/2013/718538

2. **Agarwalla & Pandey 2013 — "Expiration-Day Effects and the Impact of Short Trading Breaks on Intraday Volatility: Evidence from the Indian Market," Journal of Futures Markets:** IIM-Ahmedabad paper, peer-reviewed, sample Jan 2001-Dec 2009, documents that "volatility of stocks increases in the last half-an-hour trade on the expiry day" — establishes that expiry-day end-of-day cash-settled volatility spike exists, which is the substrate that mean-reverts T+1. URL: https://onlinelibrary.wiley.com/doi/10.1002/fut.21632

3. **Motilal Oswal Derivative Rollover Note (named monthly publication):** Indian broker-quant operationalisation of rollover percentages on Bank NIFTY and single-stock futures (~75-79% rollover on monthly cycle). Confirms institutional rollover behaviour is large and tracked. URL: https://www.research360.in/future-and-options/rollover (live data) + monthly notes archived at investmentguruindia.com (e.g., "Derivative Rollover Note 01st August 2025").

4. **Agarwalla, Jacob & Varma — "High Frequency Manipulation at Futures Expiry: The Case of Cash Settled Indian Single Stock Futures" (IIM-A working paper, SSRN 2395159):** Indian-specific microstructure paper on cash-settlement-induced price distortions at expiry; describes the rebound dynamics that motivate T+1 revert. URL: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2395159

**Acceptance threshold met:** ≥2 Indian peer-reviewed sources (Narang & Vij 2013 + Agarwalla & Pandey 2013) + 1 Indian-broker operationalisation (Motilal Oswal) + 1 IIM-A working paper. NO US-futures-rollover literature cited. The brief is fully Indian-microstructure anchored.

**Novelty check (Chartink/Streak scanner):** Chartink offers a generic "Expiry day Scan" that buys stocks with RSI > 70 the **next day** (URL: https://chartink.com/screener/expiry-day-scan) — but this is a momentum scanner (RSI-extension follow-through), NOT a **mean-revert intraday gap-fade on F&O underlyings filtered to T+1 of monthly expiry only**. The 5m-intraday mean-revert on a calendar-anchored T+1 morning window is **NOT** a published Chartink/Streak/Stratzy/Wright preset. **Novelty PRESERVED.**

### Gate B — Data feasibility (PASS — 2026-05-22)

| Required data | On disk? | Source | Notes |
|---|---|---|---|
| 5m bars on F&O 200 underlyings (T+1 morning) | YES | `backtest-cache-download/monthly/*_5m_enriched.feather` (Jan 2023 to Apr 2026, verified 80 files = 2 series × 40 months) | Full coverage for F&O liquid 200 |
| F&O liquid 200 universe | YES | `assets/fno_liquid_200.csv` (153 NSE symbols) | Already used by sibling brief |
| F&O monthly expiry calendar 2023-2026 | **YES — derivable from disk, NO scraping needed** | `data/futures_basis/2023_2026_basis.parquet` distinct `expiry_date` field = 41 monthly expiries spanning 2023-01-25 → 2026-05-26. Verified via pd.read_parquet on session: 30 Thursdays (pre-2025-09), 8 Tuesdays (post-2025-09), 2 Wednesdays, 1 Monday (holiday-adjusted). T+1 = next NSE trading session after each expiry, computable from `assets/nse_holidays.json`. | The "last Thursday → last Tuesday" calendar shift effective 2025-09-01 (NSE Circular FAOP68747) is automatically captured because the parquet stores actual settled expiry dates. |
| NSE holidays calendar | YES | `assets/nse_holidays.json` | For T+1-trading-session derivation |
| ProductionUniverseGate (Lesson #19) | YES | `tools/sub9_research/production_universe.py` | Supports F&O-200 filter via cap=any + custom symbol whitelist |
| FUTSTK vol/OI on T+0 close (optional liquidity gate) | YES | `data/futures/{YYYY}/{MM}/*.parquet` (per-day F&O EOD, verified) | Allows filtering for "active rollover" cohort if needed |

**Verdict:** Gate B PASSES cleanly. All data on disk; no scraping; no external dependencies. F&O monthly expiry calendar is a 2-line pandas derivation from the existing basis parquet.

## 5. Phase 2 empirical signature plan (preview — DO NOT CODE YET)

Once user approves brief:

- **Universe:** F&O liquid 200 (`assets/fno_liquid_200.csv`), gated through `ProductionUniverseGate` for each event date.
- **Event identification:** for each of 41 monthly expiries in `data/futures_basis/`, compute `T_plus_1 = next_nse_trading_session(expiry_date)` using `assets/nse_holidays.json`.
- **Signal definition (mean-revert):** for each (symbol, T+1 date), measure 09:15-09:30 net gap (`gap_pct = (open_09:15 - prev_close) / prev_close`). If `|gap_pct| >= 0.5%`, signal fires for **gap-fade direction** (gap-up → SHORT, gap-down → LONG) with entry at first 5m bar showing reversal confirmation (close back through 09:15 high/low).
- **Baseline (mechanism falsifier #1 test):** identical scan on **matched non-expiry-week Wednesdays/Fridays** (same DOW as the T+1, but during a NON-expiry-week to remove the calendar anchor). Mechanism passes only if drift_signal − drift_baseline ≥ +0.10%.
- **Drift measure:** signed mean return entry → 11:00 IST first-target window AND → 15:15 IST hard-square. Report both.
- **Acceptance threshold:** ≥ +0.10% drift delta vs matched-DOW baseline (Stage 2 floor) AND n_events ≥ 500 over the 41-expiry window.
- **Secondary diagnostic (regime falsifier #2):** split into **pre-2025-09 (Friday T+1)** vs **post-2025-09 (Wednesday T+1)** cohorts; report PF for each. If |ΔPF| > 0.4 absolute → KILL (calendar-shift regime-broke the mechanism).

## 6. Status checklist for advance to Phase 2

- [x] Gate A — ≥2 Indian sources cited (Narang & Vij 2013 IJSRN + Agarwalla & Pandey 2013 JFM + Motilal Oswal rollover note + Agarwalla/Jacob/Varma SSRN 2014; all peer-reviewed or named-Indian-broker; **PASS 2026-05-22**)
- [x] Gate B — data feasibility verified (5m bars + F&O 200 list + 41-expiry calendar derivable from `data/futures_basis/` + holidays JSON; **PASS 2026-05-22**, zero data engineering)
- [x] Novelty check — Chartink "Expiry day Scan" is momentum/RSI flavour, not 5m intraday mean-revert on F&O underlyings; **NOVELTY PRESERVED**
- [x] Sibling-precedent check — `stock_futures_basis_convergence_T1` already failed (NET PnL -₹106K); however this candidate's **mechanism is mean-revert intraday gap-fade, NOT basis convergence**, so the cost-structure-kill risk is the SAME as `gap_fade_short` (which lives) rather than the basis-convergence sibling. Cost-side risk is acceptable.
- [x] Adjacent setup correlation re-confirmed (M = 1 conservative; half-penalty each from `gap_fade_short` and `expiry_pin_strike_reversal`)
- [ ] **Phase 2 sanity script (deferred — not part of this brief).** Estimated ~250 LOC, mirror of `tools/sub9_research/sanity_gap_fade_short.py` pattern with expiry-calendar filter wrapper.
- [ ] **Phase 2 cohort split required:** pre-2025-09-01 (Friday T+1, 30 expiries) vs post-2025-09-01 (Wednesday T+1, 8 expiries). The 8-expiry post-cohort has insufficient power for standalone PF estimate; Phase 2 must report both pooled AND split, and flag the post-cohort as low-N caveat in confidence card.

## 7. Next action

**Phase 2 empirical signature run.** User approves brief → write `tools/sub9_research/sanity_fno_monthly_rollover_intraday_post_expiry_revert.py` (~250 LOC, ~1 day), compute drift delta vs matched-DOW baseline + pre/post-2025-09 split, decide at Stage 2 floor (+0.10%) whether to advance to OCI / Holdout / cell-mine.

---

## Phase 1 verdict: **PROCEED to Phase 2**

Gate A passed (4 Indian sources, 2 peer-reviewed, 1 broker-quant, 1 IIM-A working paper). Gate B passed (all data on disk, calendar derivable from existing parquet). Novelty preserved (no Chartink/Streak preset arbs this). M penalty conservative at 1 (half-share each with `gap_fade_short` and `expiry_pin_strike_reversal`). Sibling-failure precedent applies to basis-convergence mechanism class, NOT to the post-expiry-revert mechanism class; cost-structure risk profile is closer to surviving `gap_fade_short` than to failed `stock_futures_basis_convergence_T1`.
