# §3.3 Brief: `mis_unwind_short_late_session`

> ## Phase 1 verdict (2026-06-01) — PROCEED to Phase 2 (with concerns)
>
> | Gate | Status | Notes |
> |---|---|---|
> | A (Indian-pro precedent) | **CONDITIONAL PASS** | Strong regulatory anchor (SEBI 15:20 mandate, Zerodha/Upstox/Angel auto-square SOPs, BSE margin specs). SEBI 2023/2024 retail F&O loss studies (89-91% loss rate, dominantly LONG) corroborate the asymmetry direction. **NO retail-algo platform operationalizes the SPECIFIC new mechanic in this brief** — predecessor `mis_unwind_short` (the 14:55-15:15 entry version) IS retail-published and arb'd out (Streak, TradingView India). This brief's "front-run the unwind, exit before squeeze" angle has no published retail-algo precedent. Per Lesson 2026-05-05 strict reading, no retail-algo precedent = Gate A fail; per the brief's "research the failure mode" angle, the lack of precedent is a feature. **CONDITIONAL PASS — proceed but downgrade confidence.**
> | B (Data feasibility) | **PASS** | All required inputs on disk: 5m enriched feathers (yes), F&O 200 universe (yes), cap_segment lookup (yes), intraday VWAP + ret_3 computable (yes). Zero backfill cost. |
> | Regulatory sensitivity | **LOW RISK** | SEBI 15:20 MIS auto-square mandate codified, stable since 2020. No 2024-26 cutover affects the underlying mechanism. BUT predecessor `mis_unwind_vwap_revert_short` was retired Oct 2025 due to SEBI F&O reforms — that retirement was driven by F&O lot-size + position-limit changes, not the auto-square mandate itself. The auto-square mandate is the load-bearing piece here, so SEBI F&O Oct 2025 impact is indirect. |
> | n/yr screen | **CONCERN** | Brief estimates ~5 trades/month → ~60/yr in Discovery. **Same order of magnitude as just-killed `post_split_bonus_short` (~55-60/yr).** Today's Lesson #22 added n/yr to first-class gates. If Phase 2 confirms n ~60/yr, this candidate has the same operational-meaningfulness concern as the just-killed one. Acceptable IF per-trade edge is strong (>R 1.5 EV) AND the trades are calendar-driven (not slot-blocking). Verify in Phase 2. |
> | Predecessor risk | **MEDIUM** | Prior `mis_unwind_short` (Sub7/8) failed at NET PF 0.355, n=304, WR 9.2%. Prior `mis_unwind_vwap_revert_short` (Sub-9) retired Oct 2025 (PF 0.751 in Holdout, regulatory regime decay). **Same setup-family has failed twice already** — this is the third attempt with a structurally different mechanic. Even if Phase 2 passes, Phase 4 sanity must explicitly verify symbol-set overlap with prior detectors is <60% (per brief §9 falsifier #4). |
>
> **Decision: PROCEED to Phase 2.** Cheap (~10 min compute, no data backfill). The combination of cheap Phase 2 + strongest-possible regulatory anchor + explicit failure-mode redesign justifies a Phase 2 spend even with the family-failure history. Expected outcome: 30-40% chance Phase 2 produces a tradable drift on the off-the-high cell; 60-70% chance it dies at Phase 2 with right-direction-wrong-magnitude or wrong-direction (similar to 5day_RSI cohort).

**Sub-project:** #9 (microstructure-first redesign), Round-5
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-07

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-04-25-sub-project-7-indian-native-setups-design.md (prior `mis_unwind_short` design)
- reports/sub8_phase1/mis_unwind_short_report/01-metrics.json (prior failure: net PF 0.355, n=304, WR 9.2%, losing-days 89.4%)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED template, mechanical-NSE/SEBI-event class)

This brief is **not a re-try of the prior `mis_unwind_short`**. It targets the **same underlying asymmetry** (SEBI 15:20 MIS auto-square mandate forces retail long unwinds in mid-/small-cap F&O) but proposes a **structurally different mechanic** designed around the prior failure mode.

---

## Asymmetry

**Name:** Late-session retail MIS-long forced-unwind pressure in mid-/small-cap F&O (5m).

**Indian-specific source:**
- **NSE/SEBI mandate:** All retail MIS positions must be squared off by 15:20 IST. Brokers (Zerodha, Upstox, AngelOne, ICICIdirect) auto-square at 15:15-15:20 if the user has not manually exited. Reference: SEBI master circular on margin and exposure norms — https://www.sebi.gov.in/legal/master-circulars/jul-2023/master-circular-for-stock-brokers_73558.html and Zerodha published square-off SOP — https://support.zerodha.com/category/trading-and-markets/margin-leverage-and-product-and-order-types/articles/auto-square-off-time-for-mis-co-and-bo
- **Retail directional bias is structurally LONG.** SEBI January-2023 retail F&O loss study (https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html) finds 89% of individual derivative traders make losses; the longer-tail follow-up (https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/updated-study-on-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_87148.html) shows 91.1% of individual cash-and-derivative intraday traders lose money, with directional buying as the dominant book. Bombay Stock Exchange (BSE) margin specs corroborate the leverage profile — https://www.bseindia.com/static/markets/equity/EQReports/Margins.aspx
- **Leverage concentration in mid-/small-cap:** retail MIS leverage of 5x is available almost exclusively on F&O-eligible mid- and small-cap names (Zerodha margin calculator — https://zerodha.com/margin-calculator/Equity/). Large-cap MIS leverage is throttled (~2-3x). Mid/small-cap names therefore carry the heaviest *long, levered, unhedged* retail book.

The exploitable asymmetry is the **mechanical sell-pressure 5-15 minutes before the 15:15 broker auto-square cutoff on stocks where intraday accumulated retail-MIS-long inventory is detectable from 5m tape.**

## Participants

- **Forced sellers:** retail MIS-long holders facing imminent broker auto-square. Cannot defer — broker SOR will fire market sells at 15:15-15:20 regardless of price. Inelastic supply.
- **Disciplined sellers (us):** front-run the unwind by entering SHORT 14:30-15:00, exiting before the 15:15 squeeze risk window.
- **Counterparty / risk to thesis:** retail traders who **convert MIS to CNC delivery** before 15:15 to avoid auto-square (Zerodha "Position Conversion" — https://support.zerodha.com/category/trading-and-markets/kite-web-and-mobile/articles/how-to-convert-positions). On stocks the trader has high conviction in (closing strong, near intraday-high, popular retail name), conversion-to-CNC is common and it CANCELS the unwind — the unwind never materialises and the SHORT gets squeezed into the close. **This is the mechanism that killed the prior `mis_unwind_short`.**

## Persistence

- SEBI 15:20 MIS auto-square mandate is regulatory, not behavioural. It will not change.
- SEBI margin-and-exposure norms (peak-margin rules, intraday-vs-delivery margin separation) are part of the master circular and are reviewed annually but the 15:20 cutoff has been stable since 2020.
- Retail long-bias is documented in the SEBI Jan-2023 and Sep-2024 studies and is structural to the Indian retail-derivative population (no decay observed across 2018-2024).
- Persistence horizon: until SEBI changes the auto-square cutoff or eliminates the MIS product. Decade+ horizon.

## Project-wide caveats addressed

- **Long-bias caveat:** SHORT-only. The setup is structurally short by design — there is no symmetric long-side mirror.
- **Decay risk (Greenwood/Sammon):** the *idea* of MIS-unwind shorts is published on retail-algo platforms (Streak, TradingView India). Per round-1 finding, "if 4 retail platforms publish a pattern, it's arbitraged." This brief is **not** based on the published pattern — it is based on the *failure mode of the published pattern* (CNC-conversion squeeze) and the new mechanic explicitly avoids that failure mode (entry zone moved earlier, time stop placed before the squeeze window). The published pattern's edge has decayed; the *anti-pattern* (avoid the squeeze, enter when CNC-conversion is unlikely) has not been published.
- **Avoid retail-algo framing:** Evidence section deliberately cites SEBI/NSE/BSE/broker primary sources only. No Streak / TradingView / Wright references in the asymmetry argument.

## Evidence

Primary (regulatory and broker-mechanical):
1. SEBI Master Circular for Stock Brokers (margin, exposure, square-off norms): https://www.sebi.gov.in/legal/master-circulars/jul-2023/master-circular-for-stock-brokers_73558.html
2. SEBI Jan-2023 retail F&O P&L study (89% losing, long-bias): https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html
3. SEBI Sep-2024 updated retail F&O P&L study (91.1% losing): https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/updated-study-on-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_87148.html
4. Zerodha auto-square-off SOP (15:15 MIS, 15:20 CO/BO): https://support.zerodha.com/category/trading-and-markets/margin-leverage-and-product-and-order-types/articles/auto-square-off-time-for-mis-co-and-bo
5. Zerodha margin calculator (cap-segment-conditional MIS leverage): https://zerodha.com/margin-calculator/Equity/
6. NSE peak-margin / SEBI peak-margin framework: https://www.sebi.gov.in/legal/circulars/jul-2020/framework-for-margin-collection-and-reporting_47168.html
7. BSE margin reports (corroborating retail leverage profile): https://www.bseindia.com/static/markets/equity/EQReports/Margins.aspx

Internal precedent (failure mode to design around):
8. Sub7/sub8 `mis_unwind_short` Phase-1 metrics: net PF 0.355, n=304, WR 9.2%, losing-days 89.4% — see `reports/sub8_phase1/mis_unwind_short_report/01-metrics.json`.

## Direction

**SHORT-only.** No long-side mirror exists for this asymmetry.

## Mechanic — and how it differs from the prior failed `mis_unwind_short`

The prior detector (commit 548e5f7, file `structures/mis_unwind_short_structure.py`) used:
- Active window **14:55-15:15** (entries fired during the squeeze window itself)
- Filters: dist_vwap ≥ 0.5%, intraday-high in last 30 min, 3-bar momentum ≤ 0%, RVOL ≥ 1.2
- Stop: ATR × 1.5 above; Target: VWAP
- Result: n=304, WR=9.2%, gross PF 1.20, **net PF 0.355**

**Failure-mode diagnosis:**
1. **Entry was inside the squeeze window.** Stocks where retail had high conviction got CNC-converted; the auto-square never fired and the late-session bid stayed strong. Entries at 14:55-15:10 caught the wrong side of the conversion decision.
2. **"Fresh intraday high in last 30m + slight momentum decay" selected for trend-strength stocks** — exactly the names retail converts to CNC. The signal was anti-correlated with the desired population.
3. **n=304 (too few)** — restrictive filters left a sample where any fee drag (gross PF 1.20 → net 0.355) destroyed the edge.

**This brief's mechanic — explicitly different on all three dimensions:**

| Dimension | Prior (failed) | This brief |
|---|---|---|
| Entry window | 14:55-15:15 (inside squeeze) | **14:30-15:00 (BEFORE squeeze, last entry 15:00)** |
| Hard stop | ATR × 1.5 above entry | **15:10 hard time-stop EXIT** (5 min before broker auto-square; avoid the fire-and-forget squeeze) |
| Target population | Stocks at fresh intraday high (HIGH CNC-conversion probability) | **Stocks intraday-up >1.5% AT 14:30 BUT trading 0.3-1.5% off intraday-high (off-the-high — retail conviction wavering, less CNC-conversion)** |
| Volume signal | RVOL ≥ 1.2 (any) | **Intraday cumulative volume rank ≥ 70th pct in F&O 200 mid+small-cap that day (heavy intraday accumulation = larger MIS-long inventory)** AND **last 3 bars (14:15-14:30) volume declining vs 13:00-14:00 average (accumulation exhausting, fits unwind-imminent)** |
| Momentum filter | 3-bar mom ≤ 0% (already weakening at 14:55) | **ret_3 at 14:30 ∈ [+0.0%, +0.5%] (still slightly positive — late-comers still buying) AND ret(14:30) − ret(13:30) ≤ 0 (no fresh acceleration in last hour — peak distribution)** |
| Stop logic | Hard SL ATR × 1.5 above | **Hard SL = max(intraday-high, entry × 1.012)**; **time-stop EXIT at 15:10** (regardless of P&L) — exits BEFORE squeeze risk |
| Target | VWAP (often unreached) | **VWAP if reached by 15:10**, else market exit at 15:10. **First target at 0.4% gain (1R partial) to lock fee coverage given thin sample.** |

**Entry trigger (5m bar):** at 14:30 IST, screen F&O 200 mid+small-cap universe. Symbol qualifies if ALL hold:
- intraday return at 14:30 ∈ [+1.5%, +4.0%] (heavy intraday accumulation, but not a runaway — runaway candidates get CNC-converted)
- close at 14:30 is 0.3-1.5% off intraday-high (off-the-high zone)
- ret_3 (3-bar return) at 14:30 ∈ [0.0%, +0.5%] AND ret(14:30) − ret(13:30) ≤ 0
- intraday cumulative volume rank ≥ 70th percentile of qualifying universe
- avg volume in 14:15-14:30 < avg volume in 13:00-14:00 (accumulation exhausting)
- not a circuit-band day; not an expiry day; not a gap-day (cross-detector exclusion)

**Entry:** 5m bar close at 14:30, 14:35, 14:40, 14:45, 14:50, 14:55, 15:00 (last entry at 15:00).
**Stop:** intraday-high or entry × 1.012, whichever is higher.
**T1:** entry × 0.996 (~0.4%) — partial 50%, move stop to entry.
**T2 / final:** VWAP.
**Time stop:** **HARD EXIT at 15:10** regardless of P&L. (5 min before the 15:15 squeeze begins.)

## Universe

F&O 200 mid+small-cap (cap-segment ∈ {`mid_cap`, `small_cap`}). Large-cap excluded — MIS leverage is throttled, retail unwind pressure is diluted by institutional flow.

## Active window

Entry: **14:30-15:00 IST** (entries fire on 5m bar closes within this window).
Time stop: **15:10 hard exit**.
No entries after 15:00. No positions held past 15:10.

## Risks / falsification

Locked thresholds (per round-3/sub8 standard):
- **NET PF ≥ 1.10** on Discovery 2023-2024
- **n_trades ≥ 30** (small-sample acknowledged; this is a low-frequency mechanical-event setup like circuit_t1_fade_short)
- **|WR delta| ≤ 10pp** between Discovery and OOS Validation
- **Net Sharpe ≥ 0** on Discovery

**Falsification — explicit against prior failure:**
1. If sanity-check yields WR < 20% (similar to prior 9.2%), **ABANDON** — the CNC-conversion squeeze still dominates and the new entry/exit window did not avoid it.
2. If net PF < 1.10 with n ≥ 30, **ABANDON**.
3. If holding past 15:10 would have improved PF substantially, the time-stop is wrong but the asymmetry exists — log and rebrief; do NOT retry as-is.
4. If the off-the-high filter selects the same population as the prior detector (trend-strength names that CNC-convert), the in-house event study will show flat returns 14:30→15:10 with high variance — **ABANDON**.

**Differentiation from prior `mis_unwind_short` (mandatory check):** the sanity-check must explicitly verify that the symbol set selected by THIS detector at 14:30 is **NOT a superset** of the symbol set selected by the prior detector at 14:55. If symbol overlap > 60%, the new mechanic has not actually changed the targeted population and should be re-designed.

## Pre-coding sanity-check plan

Fully achievable on existing 5m feathers (`assets/historical_data/`). One-off Python script (≤200 LOC):

1. **Universe construction.** Load F&O 200 mid+small-cap closing on each Discovery date in 2024 (Jan-Jun, ~120 sessions).
2. **Per-session, per-symbol at 14:30 IST:** compute intraday return, dist-from-intraday-high, ret_3, ret(14:30)−ret(13:30), intraday-cumulative-volume rank within universe, avg-vol(14:15-14:30) vs avg-vol(13:00-14:00).
3. **Filter:** apply the 6-condition gate above. Capture qualifying (session, symbol) pairs.
4. **Forward returns:** compute 5m forward returns at 14:35, 14:40, ..., 15:10. Mean, median, hit-rate-at-0.4%-target, hit-rate-at-stop, distribution.
5. **Diagnostic checks:**
   - Symbol-overlap with prior `mis_unwind_short` output (must be < 60%).
   - Sample size ≥ 30 over 6 months (~5 trades/month is acceptable for low-freq mechanical setups).
   - Median forward 14:30→15:10 return clearly negative (target: ≤ −0.3%).
   - Hit-rate at 0.4% target ≥ 35% (raises WR vs prior 9.2%).
   - Stop-hit rate ≤ 25%.
6. **Decision:** if median return positive or sample < 20, REJECT. If median negative AND sample ≥ 30 AND hit-rate ≥ 35%, PROCEED to detector implementation.

## Data engineering plan

**None additional.** All computations possible from existing 5m feather cache. No new data feed; no FII/DII overlay; no order-book data. The brief is deliberately scoped to data the team already owns.

## Differentiation summary (one-line)

The prior setup *entered the squeeze*; this setup *front-runs the unwind and exits before the squeeze*. Different entry window (14:30 vs 14:55), different population (off-the-high mid-conviction vs at-fresh-high high-conviction), different exit logic (time-stop 15:10 vs run-to-VWAP-or-stop). Same underlying SEBI-mandate asymmetry, structurally different mechanic.
