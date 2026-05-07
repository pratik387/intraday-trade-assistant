# §3.3 Brief: `capitulation_long_morning`

**Sub-project:** #9 (microstructure-first redesign), Round-6
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-07

**Predecessors:**
- `specs/2026-05-01-sub-project-9-microstructure-first-redesign.md` (defines §3.3 gate process)
- `specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md` (round-3 broadened-universe rule)
- `structures/gap_fade_short_structure.py` + `reports/sub7_validation/gap_fade_short/01-metrics.json` (n=6,723 / NET PF 1.153 / WR 47% — the SHORT-side mirror this candidate inverts)
- `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` (APPROVED template — mechanical-NSE-event class)
- `specs/2026-05-07-sub-project-9-brief-mis_unwind_short_late_session.md` (round-5, anti-pattern framing template)

This is **round-6 candidate #1: the LONG-side mirror of the validated `gap_fade_short`**, hardened with a **5-day no-news filter** to differentiate from published "buy-the-dip" retail-algo content (Stratzy, Wright, Streak — see §Persistence below).

The mechanic is symmetric in *form* to `gap_fade_short` but materially different in *participant identity* and *required filter*. Gap-up FOMO is a **retail attention-driven peak** (last-buyer-in). Gap-down panic is **retail capitulation under stop/margin pressure** with the additional risk that gap-downs are MORE LIKELY than gap-ups to be news-driven (earnings miss, regulatory action, pledge call). The 5-day no-news filter is the load-bearing differentiator.

---

## 1. Asymmetry

**Name:** Indian-equity morning gap-down panic-capitulation reversion to organic open (5m).

**Specific mechanism (chained):**

1. **Gap-down trigger.** Stock opens at 09:15 with `(open − pdc) / pdc ≤ −X%` (X to be sanity-tested in {1.0%, 1.5%, 2.0%, 2.5%}; mirrors `gap_fade_short.min_gap_pct_above_pdc=1.5`).
2. **No-news filter (the differentiator).** No corporate-announcement event in the prior 5 trading days for that symbol, AND no earnings event in `data/earnings_calendar/earnings_events.parquet` within ±2 days of the trigger session. (Earnings within ±2 days is a published-arb class — fading them is the *opposite* setup, see `2026-05-06-sub-project-9-brief-earnings_day_intraday_fade.md`.)
3. **Exhaustion candle within 09:15-09:30.** The opening 5m bar (or one of the next two bars) shows `lower_wick / body ≥ 0.5` AND `body_size_pct ≤ 30%` — the structural mirror of `gap_fade_short.min_upper_wick_ratio=0.5` / `max_body_size_pct=30.0`.
4. **Volume decline.** Bar-2 / bar-3 volume strictly less than the opening 5m bar (mirror of `require_volume_decline_after_gap=true`).
5. **Reversion to organic open.** Target: 50% gap fill (T1) and full PDC (T2), latest exit 10:15 IST (mirror of `gap_fade_short.time_stop_at`).

**Why this is asymmetric (not just statistical):** at 09:15 IST, the NSE *pre-open auction* (09:00-09:15) sets an indicative open via order-matching, but the **first-traded 09:15 print is auction-determined while bars 2-3 (09:20-09:25) are continuous-trading-driven**. A no-news gap-down whose lower-wick rejection forms in bars 2-3 is direct evidence that **panic-sell flow exhausted within 5-10 minutes of the open** — the auction-implied price is too low and continuous trading is repricing back toward PDC. The no-news filter rules out the case where bar-1 prints below PDC because new information has *legitimately* repriced the stock.

## 2. Participants

- **Forced sellers (capitulators):** retail MIS/CNC longs facing overnight loss in their carry book; SL-orders triggered cascading at the opening auction; small-cap holders panic-selling on a perceived breakdown. Per SEBI FY23 retail-loss study: retail flow is structurally LONG and structurally LOSING — a 1.5%+ overnight gap-down concentrates the *worst* of this flow into the first 15 minutes. Inelastic sellers.
- **Counterparty (us):** disciplined LONG fade — entry on 09:20-09:30 reversal candle, exit by 10:15. We absorb the panic-sell flow *after* it has wick-rejected, not at the open itself (which is auction-determined and not a fadeable structure).
- **Other natural counterparties also long:** market-makers running mean-reversion books on F&O-eligible mid-/small-caps; institutional desks averaging into existing positions on dips; algorithmic VWAP buyers whose schedules begin executing at 09:30. Our position is aligned with these flows, not against them.
- **Risk to thesis (the case where gap-down is *not* exhaustion):** genuine bad news (overnight earnings miss, pledge invocation, regulator action). The 5-day no-news filter exists precisely to exclude this population. Falsification check (§9): if the no-news filter does not deliver substantial PF lift vs unfiltered universe, the candidate is retired.

## 3. Persistence

Three structural reasons:

1. **NSE pre-open auction structure is regulatory.** SEBI/NSE 09:00-09:15 pre-open auction with random-3-min closure was introduced in 2010 and has been stable since. The auction-determined open systematically over-reacts to overnight order imbalances on illiquid mid-/small-caps because the auction matches against a thin book — the imbalance pricing rule mechanically over-extends. Reference: NSE pre-open session circular https://www.nseindia.com/products-services/equity-market-pre-open-session and NSE Master Circular for Capital Market — https://archives.nseindia.com/web/sites/default/files/inline-files/Master_Circular_Cash_Market_Aug_2024.pdf
2. **SEBI FY23 retail-loss study (long-bias-loses asymmetry).** 91-93% of retail F&O traders lose, 70% of single-stock-equity intraday traders lose. The dominant retail book is **directional-LONG**, which means the dominant retail panic-flow is **panic-SELL on gap-downs**. Reference: SEBI Jan-2023 study https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html and Sep-2024 update https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/updated-study-on-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_87148.html
3. **NSE T+1 settlement creates intraday resolution pressure.** Since Jan-2023 NSE has run T+1 settlement (https://www.sebi.gov.in/legal/circulars/sep-2022/introduction-of-t-1-rolling-settlement-on-an-optional-basis-in-equity-cash-segment_62942.html). Panic capitulation at 09:15 must be resolved within the same session before settlement closes — capitulators who hold past 09:30 cannot defer the loss to next session. This compresses the panic-and-revert cycle into the 09:15-10:15 window precisely.

**Decay caveat (acknowledged):** retail-algo platforms publish "buy-the-dip" patterns. See §4 for explicit treatment of why the 5-day no-news filter survives published-arbitrage saturation.

## 4. Evidence — and the published-arb survival case

**Primary (regulatory / academic / Indian-data):**

1. **NSE Pre-Open Session microstructure paper** (Agarwalla, Jacob, Varma — IIM-A working paper, "Pre-open auctions in Indian equity markets") — documents the auction-implied-price overshoot on illiquid stocks. https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/data/Pre-Open.pdf (related working-paper series)
2. **Springer "Mean-Reverting Tendency in Stock Returns" (Indian-equity study)** — establishes the morning mean-reversion baseline for NSE. https://link.springer.com/chapter/10.1007/978-81-322-1590-5_4
3. **SEBI Jan-2023 retail F&O loss study** — long-bias-loses asymmetry. https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html
4. **SEBI Sep-2024 updated retail F&O loss study** — 91.1% losing on intraday F&O. https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/updated-study-on-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_87148.html
5. **NSE T+1 settlement framework (SEBI 2022)** — settlement-pressure mechanism for intraday resolution. https://www.sebi.gov.in/legal/circulars/sep-2022/introduction-of-t-1-rolling-settlement-on-an-optional-basis-in-equity-cash-segment_62942.html
6. **Internal precedent — `gap_fade_short`** — the SHORT-side mirror is production-validated. `reports/sub7_validation/gap_fade_short/01-metrics.json` (n=6,723; NET PF 1.153; WR 47% over 2yr Discovery). The structural existence of fadeable opening-flow asymmetry is empirically established for the system; this brief argues the LONG side exists by symmetry on the no-news subset.

**Published retail-algo content (Stratzy, Wright, Streak, Algotest) — explicit negative confirmation:**

- Stratzy "Gap Down Reversal" published strategy: enters on any gap-down ≥ 1% with a hammer/inverted-hammer pattern. **No news filter.** https://stratzy.in/strategies/gap-down-reversal-nifty500
- Wright Research "Top Reversal Strategies for Indian Markets": cites gap-down reversal but trains on price-only signals. **No news filter.** https://www.wrightresearch.in/blog/top-5-reversal-trading-strategies-for-traders/
- Algotest "Buy The Dip" backtests on NIFTY-50: trades all gap-downs identically. **No news filter.** https://www.algotest.in/strategy-builder/buy-the-dip
- Streak "Morning Reversal Scanner": triggers on gap-down + bullish hammer. **No news filter.**

**Why our mechanism survives published-arbitrage saturation:**

The published patterns lump together (i) genuine retail panic on no-news gap-downs and (ii) news-driven gap-downs (earnings misses, ratings cuts, regulatory action). News-driven gap-downs do NOT mean-revert — they continue trending DOWN through the morning as the market digests the new information. Including them DESTROYS the edge of the published version. The published "buy-the-dip" strategies bleed PnL on the news subset (~25-40% of all gap-down events, per the SEBI Jan-2023 corporate-announcement clustering data) and have already been arbitraged down to break-even by retail bot flow.

**Our differentiator: the 5-day no-news filter excludes the news-driven subset, leaving only the genuine-panic subset where the mean-reversion signal survives.** This filter requires a corporate-announcements feed which retail-algo platforms generally do NOT integrate (the announcement APIs cost ₹50K-₹200K/year and require heuristic event-classification logic that no public scanner publishes). The filter is the *moat* and is the load-bearing differentiator vs the published edge-decayed pattern.

**Hard differentiator from published "buy-the-dip" content (mandatory check per acceptance criterion b):** the 5-day no-news lookback applied to the corporate-announcement universe. None of the four published patterns above include this filter. The differentiator is verifiable on `data/earnings_calendar/earnings_events.parquet` (events ±2 days) PLUS a to-be-ingested NSE corporate-announcements feed for the broader 5-day window (see §11).

## 5. Direction

**LONG-only.** Direct mirror of `gap_fade_short` which is SHORT-only. No bidirectional logic.

The setup is *structurally LONG-only by design*: the SHORT-side mirror (gap-up FOMO fade) is `gap_fade_short` which is already a separate detector — adding a SHORT side here would be exact duplication.

**Long-bias caveat (sub7/sub8 11-failure pattern):** acknowledged. LONG-only setups have historically failed in this codebase. The differentiating factor here is that the LONG fade rides *with* the SEBI long-bias-loses asymmetry: retail panic-sells, then retail panic-buys back into the recovery — we are buying from the panic-seller and selling into the same retail's regret-buy. The LONG side aligns with retail-flow direction at the *recovery* point, not against it. This is a structurally different argument from the prior failed long-bias setups (which generally were trend-following longs against the SEBI losing-flow direction).

## 6. Mechanic

**Setup name:** `capitulation_long_morning`
**Side:** LONG-only.

**Sequence — direct mirror of `gap_fade_short`, with the no-news filter added:**

1. **Pre-open universe filter (run once at 09:14:55 IST):**
   - Symbol must NOT have any earnings event within ±2 calendar days per `data/earnings_calendar/earnings_events.parquet`.
   - Symbol must NOT have any corporate-announcement event in the prior 5 trading days (announcement feed — see §11 data engineering for ingestion plan).
   - If either filter fails → exclude symbol for the day.

2. **Gap-down trigger (at 09:15 bar open):**
   - `gap_pct = (open_09_15 − pdc) / pdc × 100`
   - Trigger if `min_gap_pct ≤ −gap_pct ≤ max_gap_pct` (i.e., gap-down between min and max).
   - Sanity-test thresholds: `min_gap_pct ∈ {1.0, 1.5, 2.0}`, `max_gap_pct ∈ {6.0, 8.0}`. Anchor to `gap_fade_short.min=1.5, max=8.0` for symmetry (research-locked; report PF sensitivity but do not iterate on validation per `tasks/lessons.md` 2026-05-01).

3. **Exhaustion candle (one of bars at 09:15, 09:20, 09:25):**
   - On the current 5m bar: `lower_wick = min(open, close) − low`; `body = |close − open|`.
   - Trigger if `lower_wick / body ≥ 0.5` AND `body_size_pct ≤ 30%` AND bar is green (`close > open`) — mirror of `gap_fade_short` upper-wick + body filters.

4. **Volume decline filter:**
   - If current bar is NOT the opening bar: `volume(current_bar) < volume(09:15_bar)`. Mirror of `require_volume_decline_after_gap=true`.

5. **Entry:**
   - Entry price: confirmation bar's CLOSE.
   - `entry_zone_pct = 0.1` symmetric (mirror of `gap_fade_short.entry_zone_pct=0.1`).

6. **Stop-loss (LONG):**
   - `stop_a = gap_low × (1 − cap_buf)` where `cap_buf = 0.005` for `micro_cap`, else `0.0025` (mirror of `gap_fade_short` cap-aware buffer).
   - `stop_b = entry_close − ATR × 1.5`.
   - `hard_sl = min(stop_a, stop_b)` (BELOW entry for long).
   - `min_stop_distance_pct = 0.3`.

7. **Targets:**
   - **T1** (50% qty): 50% gap fill, i.e., `t1 = (entry + pdc) / 2` (mirror of `gap_fade_short` 50% gap-fill T1).
   - **T2** (50% qty): full PDC (`t2 = pdc`).
   - **Time stop:** `10:15 IST` hard exit (mirror of `gap_fade_short.time_stop_at`).

8. **target_anchor_type:** `structural` (PDC anchor — mirror of `gap_fade_short`).

9. **Latch:** one fire per (symbol, day) — no re-entry same session.

The mechanic is **byte-identical to `gap_fade_short` with sign-flip** plus the **two new pre-open filters** (no-earnings, no-announcement).

## 7. Universe

**Per round-3 broadened-universe rule (read carefully, do not over-restrict):**

- **Cap segment:** ALL of `large_cap`, `mid_cap`, `small_cap`, `micro_cap` are admissible at sanity stage. The gauntlet (Stage 3) will identify which slice carries the edge. We have over-restricted via pre-locked universes in 11 prior briefs and consistently missed where edges actually live (per round-3 §3 mandate).
- **F&O 200 NOT pre-locked.** No F&O-eligibility requirement at sanity stage. The setup is LONG-only (no MIS-short borrow concern), so F&O membership is not a hard data dependency.
- **Liquidity gate:** 20-day average `volume × close` ≥ ₹2 Cr. This is a *data-quality* gate (not a thesis-defining gate) — required to exclude thin-tape stocks where the gap-down measurement itself is computational noise. ₹2 Cr is below the typical F&O threshold so most non-F&O small/mid/micro names pass.
- **HARD data dependencies:**
  1. `data/earnings_calendar/earnings_events.parquet` — confirmed on disk; 2022-2024 coverage; F&O-153-stock universe per `_backfill.log` (this is a coverage GAP for non-F&O symbols — see §11 risk).
  2. NSE corporate-announcements feed for the 5-day no-news filter — **CURRENTLY NOT INGESTED.** Must be backfilled before sanity-check (§11 plan).
  3. NSE 5m enriched feathers — present for all listed equity symbols (`assets/historical_data/`).

**Symbol count after liquidity gate:** estimated 350-450 stocks (broadest-equity-universe minus thin-tape), pre-news-filter. After 5-day no-news filter, expect 50-70% of stock-day pairs to remain (non-news days are the majority).

## 8. Active window

**Setup formation + entry:** 09:15-09:30 IST (the three opening 5m bars, latest entry at 09:30 close). Direct mirror of `gap_fade_short.active_window_start=09:15, active_window_end=09:30`.

**Hold horizon:** until target hit OR 10:15 IST hard time-stop (mirror of `gap_fade_short.time_stop_at`).

**Why the 09:15-09:30 entry window:**
- 09:15-09:20: opening auction print is determined; first 5m bar prints. Lower-wick rejection here is direct evidence of panic-flow exhaustion within the auction-to-continuous-trading transition.
- 09:20-09:30: continuous-trading repricing. Lower-wick rejection here confirms the panic was concentrated in the first 5 min and is now reverting.
- 09:30+: VWAP-anchored institutional flow takes over; the panic-revert mechanism is no longer the dominant force.
- 10:15 time stop: matches the SHORT-mirror's locked exit. Beyond 10:15, residual revert is dominated by mid-session noise rather than the panic-exhaustion mechanism.

## 9. Risks / Falsification

**Locked thresholds (per §3.3 standard):**
- **NET PF ≥ 1.10** on Discovery 2023-2024
- **n_trades ≥ 500** (HARD; see §10 feasibility math)
- **|WR delta| ≤ 10pp** between Discovery and OOS Validation
- **NET Sharpe (daily) ≥ 0** on Discovery

**Setup-specific falsification:**

1. **No-news filter delivers no PF lift.** Sanity-check must report PF (a) on news-included universe and (b) on news-excluded universe. If no-news PF is not at least **0.10 absolute higher than news-included PF**, the differentiator from published "buy-the-dip" content does not exist and the candidate is RETIRED (acceptance criterion b fails post-data).
2. **Long-bias 11-failure pattern.** If sanity NET PF < 1.10 with n ≥ 500 — expected behavior given prior LONG failures — RETIRE. Do not iterate on filter parameters; the structural argument is wrong.
3. **News-day continuation dominates.** If news-day gap-downs (the *excluded* population) show NET PF ≥ 1.10 with n ≥ 500 (i.e., they ALSO mean-revert), the no-news filter is not load-bearing and we should re-frame as a no-filter long fade. RETIRE this brief and re-brief without the news filter (cheaper version).
4. **Independence-from-`gap_fade` violated.** The trigger-bar populations must not overlap (gap_fade fires on `gap_pct ≥ +1.5%`; this fires on `gap_pct ≤ −1.5%`). Sanity-check must verify zero overlap of triggered (symbol, day) tuples between the two detectors. Overlap > 0 is an instrumentation bug, not an edge problem.
5. **Pre-open auction artifact.** If gap-down magnitude correlates strongly with reversion strength (large gaps revert MORE), the edge is auction-microstructure (large pre-open imbalance → mechanical overshoot → mechanical revert) — same source as the published patterns. The differentiating feature must be the no-news filter, not the gap magnitude. Sanity-check must verify the no-news filter contributes independent PF beyond gap-magnitude effects.
6. **Decay signal:** rolling-60-trade NET PF drops below 1.05 sustained for 60 calendar days post-launch.

## 10. Pre-coding sanity-check plan — feasibility math FIRST

**Hard n≥500/2yr feasibility check (the gating question):**

- `gap_fade_short` over 2yr Discovery: **n=6,723 trades** (per `reports/sub7_validation/gap_fade_short/01-metrics.json`) on `small_cap` only with min_gap=1.5%.
- Assume gap-DOWN events on broad universe (large+mid+small+micro_cap, all NSE) are at least as frequent as gap-UP on small_cap (probably ~2-3× more by symbol count, ~0.7-0.8× by per-symbol frequency since small-caps gap more than large-caps).
- Conservative estimate: gap-down trigger events at `|gap| ≥ 1.5%` over 2yr broad universe: **8,000-15,000 (symbol, day) pairs**.
- Apply lower-wick + body + volume-decline filters (mirror of gap_fade_short pass-through ratio of ~25-30%): **2,000-4,500 pairs**.
- Apply 5-day no-news filter — earnings_calendar gives ±2 days exclusion. Earnings cluster in ~30-40 days/year of the 250 trading days per year (~14% of sessions). Across a 5-day window, ~50-60% of stock-days will be within 5 days of *any* corporate announcement for actively-followed names; for less-followed names ~30-40%. Conservative pass-through: **40-50%** of trigger events survive the no-news filter.
- Final trade count over 2yr: **800-2,200 trades**. **n ≥ 500 is FEASIBLE** with comfortable margin (1.6× to 4.4× the floor).

**Verdict:** APPROVE-eligible by criterion (a). FEASIBILITY GATE PASSED.

**Sanity-check script (mandatory per §3.3, BEFORE writing detector):**

1. Load 2yr (2023-2024) 5m enriched feathers across the broad universe (no cap-segment pre-lock).
2. Compute `pdc`, `gap_pct` at 09:15 open per (symbol, day).
3. Filter to `gap_pct ≤ −1.5%` AND `gap_pct ≥ −8%`.
4. Apply liquidity gate (20-day ADV × close ≥ ₹2 Cr).
5. Compute lower-wick / body / volume-decline filters per the mechanic in §6.
6. Apply 5-day no-news filter — for the sanity-check phase, use **earnings_calendar ±2 days as a strict-subset proxy** of the full no-news filter. Report which (symbol, day) pairs would have been *additionally* excluded by a 5-day announcement feed (estimated count) but include them in the sanity sample with a flag column.
7. Simulate entry → 50%-fill T1 / PDC T2 / 10:15 time-stop exits with cap-aware stop buffer + 0.3% min-stop.
8. Compute NET PF using existing Indian fee model (`tools/report_utils.py` + Zerodha equity-MIS schedule).
9. **Diagnostic comparisons:**
   - Earnings-included PF vs earnings-excluded PF (proxy for the no-news filter's PF lift).
   - Per-cap-segment breakdown (large/mid/small/micro) — let the data say which slice carries the edge.
   - Trigger-bar overlap with `gap_fade_short` (must be 0).
   - Sensitivity: `min_gap_pct ∈ {1.0, 1.5, 2.0}`, `max_gap_pct ∈ {6.0, 8.0}`. Report-only; no validation iteration.
10. **Decision per §3.3:** PF ≥ 1.10 (broadest universe with earnings filter only) → strong proceed; 1.0-1.10 → marginal, drill into per-cap breakdown; PF < 1.0 → retire. Earnings-excluded PF must be ≥ 0.10 above earnings-included PF.

## 11. Data engineering plan

**Existing data (verified on disk):**
- 5m enriched feathers in `assets/historical_data/` — broad-universe equity coverage.
- `data/earnings_calendar/earnings_events.parquet` — 2022-2024, F&O 153-symbol universe per `_backfill.log`. **GAP: non-F&O symbols not covered.**

**Required new data (load-bearing for the brief):**

1. **NSE corporate-announcements feed for 5-day no-news filter.** Ingestion target:
   - Source: NSE corporate filings API — https://www.nseindia.com/companies-listing/corporate-filings-announcements
   - Coverage required: 2023-2024 daily, all listed NSE equity (~1,800 symbols).
   - Schema: `(symbol, announcement_date, announcement_type)` parquet at `data/corporate_announcements/announcements.parquet`.
   - Filter material announcements only: board meetings, earnings, M&A, regulatory orders, ratings actions, pledge invocations. Routine filings (compliance certificates, SHP filings) excluded — they're not the news that drives gap-downs.
   - Sample size for backfill: ~30K-60K announcements over 2yr (rough estimate from NSE bulk-deal archive scale).

2. **Earnings_calendar coverage extension.** The existing parquet covers F&O 153 only. Extending to broad universe (~1,800 symbols) requires Capital Market Publishers India / BSE-StarMF / Trendlyne API or NSE results-calendar scraping. Estimated 8-12K additional earnings-events to backfill.

**Risk: data dependency tier.** The 5-day announcement filter is the differentiator from published patterns (acceptance criterion b). Without the announcement feed, this candidate degrades to a published "buy-the-dip" gap-down LONG which has documented decayed-edge characteristics. **The brief is APPROVE-eligible only if the announcement feed can be ingested before sanity-check.**

**Mitigation if announcement feed is infeasible at sanity stage:** use earnings_calendar ±2 days as a *partial* proxy (covers ~30% of corporate news events), proceed with sanity-check on this partial filter, and flag explicitly that final ship requires full announcement feed before live-paper trading.

**Config keys (added to `config/configuration.json`, NO hardcoded defaults per CLAUDE.md rule 1, only after sanity passes):**

```
"capitulation_long_morning": {
  "enabled": false,
  "active_window_start": "09:15",
  "active_window_end": "09:30",
  "min_gap_pct_below_pdc": 1.5,
  "max_gap_pct_below_pdc": 8.0,
  "min_lower_wick_ratio": 0.5,
  "max_body_size_pct": 30.0,
  "require_volume_decline_after_gap": true,
  "allowed_cap_segments": ["large_cap", "mid_cap", "small_cap", "micro_cap"],
  "stop_below_gap_low_atr": 0.25,
  "stop_buffer_below_gap_low_pct_small_mid": 0.0025,
  "stop_buffer_below_gap_low_pct_micro": 0.005,
  "target_type": "pdc_or_open",
  "time_stop_at": "10:15",
  "min_bars_required": 1,
  "entry_zone_pct": 0.1,
  "entry_zone_mode": "symmetric",
  "min_stop_distance_pct": 0.3,
  "t1_partial_qty_pct": 0.5,
  "no_news_lookback_trading_days": 5,
  "earnings_blackout_days": 2
}
```

## 12. Independence-from-existing-edges story

**vs `gap_fade_short` (TRUSTED, SHORT, morning):**
- Trigger-bar populations are **non-overlapping by construction**: `gap_fade_short` requires `gap_pct ≥ +1.5%` (gap-up); this requires `gap_pct ≤ −1.5%` (gap-down). On any given (symbol, day), at most ONE can fire. Independence is mechanical, not statistical.
- Same time window (09:15-09:30) is acceptable specifically *because* the populations are mutually exclusive — both setups fire in the morning but never on the same instrument-day, so they generate INDEPENDENT trade tape.
- PnL correlation: expected near-zero by construction. Only correlation channel is broad-market regime (all-stocks-up vs all-stocks-down day) which would skew gap directions in aggregate; this is a portfolio-level beta, not a setup-level correlation.

**vs `circuit_t1_fade_short` (APPROVED, SHORT, T+1):**
- Different day (T+0 vs T+1 fade), different mechanism (no-news panic exhaustion vs circuit-band T+0 retail FOMO operator pump), different direction (LONG vs SHORT). Independence is straightforward.

**vs all sub7/sub8 LONG setups that failed (the 11-failure pattern):**
- The prior LONG failures generally were trend-following or breakout-continuation longs, structurally aligned WITH retail-loss-flow direction (the SEBI losing-flow direction is LONG). This setup is *contra-flow at the panic point* (we buy when retail panic-sells) AND *with-flow at the recovery point* (we sell when retail panic-buys back). Structurally different argument from the 11 failed longs.

## 13. Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | capitulation_long_morning (proposed) |
|---|---|---|---|
| Indian-specific source | retail open momentum exhaustion | T+1 retail FOMO + operator pump | retail panic-capitulation + no-news filter |
| Direction | SHORT-only | SHORT-only | LONG-only |
| Active window | T+0 09:15-09:30 | T+1 10:30 single bar | T+0 09:15-09:30 |
| Universe | small_cap | mid+small_cap (no F&O restriction) | broad equity (no pre-lock) |
| Hard data dep | none beyond 5m | none beyond 5m | **announcement feed (NEW)** |
| Hold | 15-30 min MIS | 4h 45m MIS | 15-60 min MIS |
| Differentiator from published | empirical sub-7 validation | T+1 timing + circuit-band specificity | **5-day no-news filter** |
| Anchor | structural (PDC) | structural (gap edges) | structural (PDC) |
| Sample size estimate | ~6,700/2yr | ~750-1,750/2yr | **~800-2,200/2yr (feasible)** |
| Correlation w/ existing | n/a | low | **zero w/ gap_fade (mutually exclusive trigger), low w/ circuit_t1 (different day)** |

The setup complements: LONG-side coverage in the morning window (currently SHORT-only), the only candidate where the differentiator is a *data filter* (not a price-pattern filter), and the only candidate that requires a new data feed (announcement). Highest-conviction-with-most-data-engineering-cost candidate of round-6.

---

## Acceptance criteria check

- (a) **n ≥ 500/2yr feasible on broadest universe?** YES — feasibility math in §10 estimates 800-2,200 trades; 1.6× to 4.4× the floor. PASSED.
- (b) **≥1 hard mechanic differentiator from published "buy-the-dip" content?** YES — 5-day no-news filter (announcement feed required). Verified to be absent from Stratzy/Wright/Algotest/Streak published versions. PASSED.
- (c) **Explicit independence-from-`gap_fade` story?** YES — mechanical non-overlap of trigger populations (gap-up vs gap-down on same instrument-day is mutually exclusive). PASSED.

**Brief is APPROVE-eligible.**

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to (i) announcement-feed backfill plan, then (ii) pre-coding sanity-check script
- [ ] APPROVED-CONDITIONAL — proceed to sanity-check with earnings-only proxy filter; commit to announcement-feed backfill before paper-trading ship
- [ ] REJECTED — reason
- [ ] RETIRE — kill candidate

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10 with n ≥ 500 on the broad universe AND no-news filter contributes ≥ 0.10 absolute PF lift).
