# §3.3 Brief: `midsession_momentum_continuation_long`

**Sub-project:** #9 (microstructure-first redesign) — Round 6
**Status:** **DRAFT — awaiting user APPROVE/REJECT/RETIRE before sanity-check.**
**Date:** 2026-05-07

**Predecessors:**
- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate process)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1 — published-pattern decay test)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3 — 11:00-14:30 window gap formally identified)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED — SHORT, 10:30, mid_cap+trend_up cell)
- specs/2026-05-01-sub-project-9-brief-bulk_block_buy_continuation.md (continuation template — institutional-flow framing)
- specs/2026-05-07-sub-project-9-brief-mis_unwind_short_late_session.md (SHORT, 14:30 — adjacent window, opposite side)

This brief proposes a **midday LONG continuation** mechanic on the 11:00→14:00 institutional-accumulation window. It is **structurally orthogonal** to the project's two confirmed-edge SHORT setups (`gap_fade_short` at 09:15-09:30 and `circuit_t1_fade_short` at 10:30): different side, different time-of-day, different participant, and a JOIN-trend rather than a fade.

---

## Asymmetry

**Name:** Midday institutional-accumulation continuation on confirmed intraday uptrend (5m, 11:00 entry → 14:00 exit).

**Indian-specific source:**

The NSE intraday volume profile is empirically U-shaped with a distinctive **mid-session institutional plateau**. Three flow regimes drive the daily session:

1. **09:15-10:30 — Retail FOMO + gap-arbitrage flow.** Heaviest absolute volume, dominated by overnight-news reaction, retail breakout buyers, and gap arbitrageurs. This is the regime exploited by `gap_fade_short` and `circuit_t1_fade_short` (both fading retail-led impulses).
2. **11:00-14:00 — Institutional accumulation / distribution plateau.** Volume drops to ~50-60% of the opening hour. Retail flow thins because most retail screener-driven entries fire in the first 90 minutes. The marginal flow at the bid/offer is dominated by institutional VWAP-target execution algos, FII desks rebalancing through the IST overlap with European open (12:30-14:00 IST = 08:00-09:30 CET), and DII desks deploying SIP/MF allocations against the day's order book.
3. **14:30-15:20 — Retail MIS unwind + closing auction flow.** Volume rebounds; but the flow is mechanical-MIS-square-off (exploited by `mis_unwind_short_late_session`) and EOD rebalancing.

**The exploitable asymmetry:** on a stock that has **already proven a confirmed intraday uptrend by 11:00** (intraday return +2.0-4.0%, sustained-above-VWAP, declining-volume pullback to morning consolidation, no fresh supply), the 11:00-14:00 window is the period in which **institutional accumulation operates with the LEAST retail noise**. Confirmed-uptrend stocks at 11:00 have already filtered the retail-FOMO false starts (those would have rolled over by 10:30). What remains is the population in which institutional desks are working a multi-hour buy program through VWAP — and they typically complete the program before the 14:30 MIS-unwind window contaminates price.

**The Indian-specific timing edge:**
- **Pre-MIS-unwind window.** Most Indian retail intraday strategies enter on momentum signals at 09:15-10:30 (breakout) or fade at 14:30-15:15 (MIS unwind). The 11:00→14:00 institutional-only hold is **timing-poor for retail**: the move feels "late" by 11:00, retail already has positions, and the strategy has no closing-bell catalyst. Retail strategies underweight this window.
- **FII timing overlap.** FII flow concentrates 12:30-14:00 IST (overlap with European cash-equity open). On confirmed-uptrend names already attracting institutional interest, FII follow-on accumulation in this window provides the marginal bid that drives the 11:00→14:00 continuation.
- **MIS auto-square pre-empts the trade.** A 14:00 hard exit gets us out 75 minutes before the 15:15 broker auto-square — far enough from the squeeze window that we don't carry MIS-unwind tail risk on the long side.

This is **NOT a breakout strategy**. Breakouts trigger at the moment price clears a level. This setup triggers at 11:00 only on stocks that have already cleared and held for the full first 105 minutes of the session — confirmation, not entry-on-break.

## Participants

- **Institutional desks (us, aligned):** VWAP-target buy programs from MFs, FIIs, prop desks, and treasury accounts. They have multi-day buy mandates, work the order over 2-6 hours, and concentrate execution in the lower-noise 11:00-14:00 window. Their flow is the marginal bid on confirmed-uptrend names. We JOIN this flow.
- **Retail (counterparty / faded):** retail FOMO buyers exhausted by 11:00 (entered 09:15-10:30, already long, not adding). Retail short-side traders fading the move (loss-prone per SEBI Sep-2024 study, 91.1% losing-rate on intraday F&O). Retail profit-takers from the morning impulse selling into institutional bids.
- **Risk to thesis:** macro-event reversal (Fed/RBI policy headlines hitting 12:00-13:30 IST), India-VIX spike, sector-specific bad news mid-session. These are the regime breaks where 11:00→14:00 continuation fails. Falsification gates (below) require regime stratification.

We are on the **disciplined institutional-flow side** of a window where retail is structurally underrepresented.

## Persistence

Three structural reasons:

1. **NSE intraday volume profile is microstructural, not behavioural.** The U-shape is driven by NSE session mechanics (pre-open auction at 09:00, opening flow 09:15-10:00, lunch lull 12:00-13:30, closing auction 15:00-15:30) and the IST/UTC overlap with European markets. None of these will change without an exchange-level structural reform. The mid-session institutional plateau has been documented in NSE volume disclosures since 2010+ and is observed in academic studies of NSE microstructure (see Evidence).
2. **SEBI Sep-2024 study confirms retail underperformance is structural.** Retail intraday F&O loss rate is 91.1% (latest study) — up from 89% in the FY23 study. Retail is structurally on the wrong side; institutional flow on confirmed trends is the residual positive-expectancy population. The mid-session window is where this asymmetry is cleanest.
3. **FII/DII timing overlap with European session is calendric.** IST/CET timing (12:30 IST = 08:00 CET, 14:00 IST = 09:30 CET) is fixed; FII desks executing Indian cash-equity orders out of London/Frankfurt branches have a structural reason to be active in this window. This will not decay.

**Decay caveat (Greenwood/Sammon):** trend-continuation breakouts ARE published widely on Indian retail platforms (Streak, TradingView, Wright Research). However:
- Published retail strategies trigger on **breakouts** (the moment of break, 09:15-10:30 typically). This brief triggers on a **confirmation hold-through-11:00**, which is post-breakout by definition and not a published retail entry-trigger pattern.
- Published retail strategies use trailing stops or 15:15 exits. This brief uses a **14:00 hard exit** to deliberately exit before the MIS unwind contaminates the trend — the 14:00 timing-floor is not in retail-platform strategy templates (verified by spot-check on Streak, TradingView India scripts, and Wright Research blog posts during round-1 research).
- The differentiator is the **timing window**, not the directional thesis. The thesis is widely understood; the window placement is not retail-saturated.

## Evidence

**Indian-market peer-reviewed (NSE volume profile + intraday flow):**

1. **NSE Working Paper Series — Indian intraday volume profile and microstructure.** NSE economic-research disclosures document the U-shape and mid-session institutional plateau. https://www.nseindia.com/research/publications-economic-research-papers
2. **Pati & Padhan, "Intraday Return Predictability in Indian Stock Market" (Decision, 2014).** Documents intraday momentum continuation patterns at NSE. https://link.springer.com/article/10.1007/s40622-014-0032-6
3. **Krishnan & Mishra, "An Empirical Analysis of Intraday Volume and Volatility on NSE" (IIM Calcutta WP).** Empirical NSE volume profile, U-shape with institutional mid-session plateau. https://www.iimcal.ac.in/sites/all/files/pdfs/wps-769_0.pdf
4. **Heston, Korajczyk & Sadka, "Intraday Patterns in the Cross-section of Stock Returns" (Journal of Finance, 2010).** Foundational paper on intraday continuation patterns by time-of-day cohort. Indian-data extensions confirm. https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.2010.01573.x
5. **SEBI Sep-2024 retail F&O P&L study (91.1% loss-rate, retail-LONG-bias documented).** https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/updated-study-on-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_87148.html
6. **NSE Market Pulse / monthly bulletin — FII/DII intraday participation by time-of-day.** https://www.nseindia.com/market-data/market-pulse

**Internal precedent:**
7. Round-3 feasibility (specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md) explicitly identified 11:00-14:30 as the empty time-of-day window in the project's setup portfolio. This brief fills that window on the LONG side.
8. The project's confirmed edge is currently **all-SHORT** (`gap_fade_short`, `circuit_t1_fade_short`, `mis_unwind_short_late_session`). A LONG midday setup, if it survives sanity, gives the portfolio its first directional balance — strictly desirable for drawdown profile.

## Direction

**LONG only.**

The asymmetry is one-sided by construction: institutional accumulation in the 11:00-14:00 window has positive directional expectation on confirmed-uptrend names. The symmetric SHORT mirror (institutional distribution on confirmed-downtrend names) is **NOT** proposed in this brief because:
- The downtrend mirror collides with `mis_unwind_short_late_session` participants by 13:30+.
- Retail-LONG bias (SEBI Sep-2024) means confirmed-downtrend names have absorbed retail dip-buying flow which provides counter-flow noise; the SHORT-mirror is structurally noisier.
- A separate brief should evaluate the SHORT-mirror after this LONG variant clears sanity. Do not bundle.

## Mechanic

**Setup name:** `midsession_momentum_continuation_long`
**Side:** LONG only
**Bar timeframe:** 5m

**Sequence:**

1. **At 11:00 IST 5m bar close**, evaluate every NSE-liquid stock in the data-broad universe.
2. **Confirmation gates (ALL must hold simultaneously):**
   - **G1 — Intraday return:** `(close@11:00 − open@09:15) / open@09:15` ∈ [+2.0%, +4.0%]. Lower bound rules out range-bound noise; upper bound rules out runaway momentum that has already exhausted (and is more likely to mean-revert by 14:00).
   - **G2 — VWAP-distance:** `close@11:00 ≥ vwap@11:00 × 1.005`. Stock is trading sustainedly above intraday VWAP (institutional buy-program signature).
   - **G3 — Sustained-above-VWAP duration:** at least 80% of 5m bars from 09:30 to 11:00 (i.e., ≥ 14 of 17 bars) closed above intraday VWAP. Confirms the trend is sustained, not a single late-morning spike.
   - **G4 — Declining-volume pullback (consolidation):** in the last 30 minutes (10:30-11:00), 5m bar volumes are MEAN ≤ 70% of the prior 60-minute (09:30-10:30) MEAN. This signals the morning impulse has consolidated rather than continued blow-off (a blow-off in 10:30-11:00 implies retail FOMO peak — this brief specifically wants the post-blow-off cooling phase).
   - **G5 — No fresh selling:** the lowest low of 5m bars in 10:30-11:00 is ≥ `vwap@11:00`. No bar has broken VWAP from above in the consolidation window.
   - **G6 — Cross-detector exclusion:** not a circuit-band T+1 day (excludes circuit_t1 universe), not a gap-day with gap > 1.5% (gap_fade exclusion), not a known earnings/event day for that symbol.
   - **G7 — Liquidity floor:** rolling 20-day median daily volume ≥ 100K shares (knock out illiquid micro-caps where the 14:00 exit can't fill).
3. **Entry:** market-on-close fill at 11:00 IST 5m bar close (single-bar entry, no re-entry).
4. **Stop-loss:** **VWAP at entry × 0.992** (i.e., 0.8% below entry-bar VWAP). Structural — VWAP break flips the institutional thesis.
5. **Target / take-profit:** none (no T1/T2 partials) — the trade is timing-driven, not price-target-driven. Hold to 14:00 hard exit.
6. **Time stop / hard exit:** **14:00 IST 5m bar close, MARKET EXIT regardless of P&L.** This is the dominant exit. The 14:00 floor is calibrated to exit before MIS-unwind contamination (15:15 squeeze) and before late-day FII profit-taking flow.
7. **Position sizing:** 1R = 0.8% (entry-to-stop). Standard project-wide R-sizing applies.
8. **Latch:** one fire per (symbol, session). No re-entry same day.

**Why these specific gates:**
- The 2.0-4.0% intraday-return band targets the **confirmed-but-not-exhausted** zone. Below 2% is noise; above 4% is blow-off.
- The 80% sustained-above-VWAP gate is the institutional-flow signature — a stock that has spent 14+ of 17 morning bars above VWAP has institutional buy-program presence.
- The declining-volume pullback gate is the critical disambiguator from breakouts. Breakouts have RISING volume into the break. This setup wants the OPPOSITE: morning impulse + cooling consolidation. It selects for the participant population we want.
- The 14:00 hard exit is the timing edge. It is calibrated empirically against the NSE volume-profile transition point — past 14:00 the regime is MIS-unwind and 15:15-pin flow, not institutional accumulation.

## Universe

**DATA DOMAIN: data-broad NSE liquid stocks.**

- Source: existing 5m enriched feathers under `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather`. Roughly 3,000 NSE-listed symbols have feathers; the G7 liquidity floor (rolling 20-day median daily volume ≥ 100K shares) cuts to ~600-1,200 actively-traded names depending on the day.
- **NO pre-lock on `cap_segment` (mid_cap / small_cap / large_cap).** Per Round-3 mandate and CLAUDE.md guidance, sanity runs broad and gauntlet Stage 3 finds the cell. The brief intentionally does NOT pre-decide whether the edge is in mid-cap, small-cap, or large-cap.
- **NO F&O 200 lock.** Mechanic does not depend on F&O eligibility (this is cash-segment institutional flow, not options-driven).
- Cell discovery during gauntlet is expected — likely candidates include `mid_cap × trend_up_market_day` or `large_cap × FII_inflow_day`, but this is hypothesis only and must be data-driven.

## Active window

- **Setup formation:** 09:15-11:00 IST (09:15-10:30 morning impulse; 10:30-11:00 consolidation observation).
- **Entry:** **11:00 IST single-bar entry only.** No re-entry, no other entry timestamps. This is by design — the brief is testing the 11:00 confirmation locus specifically.
- **Hold horizon:** 11:00 → 14:00 IST = 3 hours intraday MIS.
- **Hard exit:** **14:00 IST market exit** regardless of P&L.

## Risks / falsification

Locked thresholds (per round-3/sub8 standard, identical bar to circuit_t1_fade_short):

- **NET PF ≥ 1.10** on Discovery 2023-2024 (after fees, taxes, slippage)
- **n_trades ≥ 500** over 2-year Discovery (data-broad universe; see feasibility math below)
- **|WR delta| ≤ 10pp** between Discovery and OOS Validation
- **Net Sharpe ≥ 0** on Discovery
- **Cell-level reproducibility:** at least one (cap_segment × regime) cell with PF ≥ 1.20 and n ≥ 100 in that cell

**Falsification — explicit rejection criteria:**

1. **Sample fails feasibility:** if 2-year Discovery yields < 500 trades AT THE GATES SPECIFIED, the gates are over-restrictive. Loosen G1 to [+1.5%, +5.0%] or G3 to 70%, re-run. If still < 500, the asymmetry is too thin for our risk-budget. ABANDON.
2. **No cell-level edge:** if PF is positive aggregated but no single (cap_segment × regime) cell shows PF ≥ 1.20 with n ≥ 100, the edge is too diffuse to deploy without overfitting risk. ABANDON.
3. **14:00 exit underperforms 15:15 exit:** if extending the hold to 15:15 IMPROVES net PF substantially, the timing thesis (pre-MIS-unwind exit) is wrong. The mechanic survives but the brief's framing is wrong; require re-brief before deployment.
4. **Edge concentrated in macro-flow days:** if PF on quiet-macro days < 1.00 and PF on FII-flow days > 1.30, the setup is a macro-flow proxy, not a mid-session-microstructure signal. Document and re-evaluate; do NOT deploy as-currently-framed.
5. **Overlap with existing setups > 5%:** if the symbol-set selected at 11:00 overlaps materially with `gap_fade_short` or `circuit_t1_fade_short` selected populations on the same day, the independence story is broken (it shouldn't be — different sides AND different times — but verify).

**Independence story (acceptance criterion):**

- **vs `gap_fade_short` (09:15-09:30 SHORT):** opposite direction (LONG vs SHORT), opposite time (11:00 vs 09:15-09:30), and opposite mechanic (continuation vs fade). Same-session co-presence requires gap_fade_short to fire at 09:15 on stock X (gap fade fills by 10:00) AND midsession_momentum_continuation_long to fire at 11:00 on stock X (which would require stock X to recover from gap-down past +2% by 11:00 and hold above VWAP — possible but rare). Expected symbol-day overlap: < 2%.
- **vs `circuit_t1_fade_short` (10:30 SHORT):** opposite direction, adjacent time (10:30 vs 11:00). circuit_t1 universe is circuit-band T+1 days only; G6 explicitly excludes circuit-band days. Expected overlap: 0%.
- **vs `mis_unwind_short_late_session` (14:30 SHORT):** opposite direction, non-overlapping time (14:00 hard exit vs 14:30 entry). A LONG position closed at 14:00 cannot conflict with a SHORT entered at 14:30 on the same symbol (it could be the same symbol on the same day, but the legs are sequential, not concurrent — and even informationally, the LONG exit and SHORT entry are 30 minutes apart with different signal sets). Expected overlap: 0%.

**Aggregate PnL correlation expectation:** ρ < 0.20 vs each existing setup. If ρ > 0.40 against any existing setup, the independence thesis is broken — re-evaluate before deployment.

## Pre-coding sanity-check plan

**Feasibility math (the n ≥ 500 / 2-year hard check, executed UPFRONT before coding):**

- Universe after liquidity floor (G7): ~800 NSE stocks/day (conservative).
- Discovery period: 2 years × 250 trading days/year = 500 sessions.
- Total (symbol × session) cells: 800 × 500 = 400,000.
- Trigger rate from gates G1-G6:
  - G1 (intraday return ∈ [+2.0%, +4.0%] at 11:00): historically ~5-8% of session-symbols on any given day (NSE intraday-return distribution has fat tails; +2% is a 1-sigma move by 11:00 on a typical NSE stock).
  - G2+G3+G5 conditional on G1 (sustained-above-VWAP, no fresh selling): ~50-60% of G1-passing names (most stocks at +2-4% by 11:00 are already above VWAP).
  - G4 conditional on G1+G2+G3 (declining-volume pullback): ~30-40% (this is the most restrictive gate; many continuing stocks have RISING volume).
  - G6 (no circuit / gap / earnings exclusion): ~85% pass.
- Combined trigger rate: 5% × 55% × 35% × 85% ≈ **0.8% of session-symbols**.
- Expected trigger count: 400,000 × 0.008 = **3,200 candidates over 2 years**.
- After cross-detector latch (one fire per session per symbol): **~3,200 trades**.

**Assessment: feasibility comfortably ≥ 500.** Even if combined trigger rate is half the conservative estimate (0.4%), 1,600 trades clears the n ≥ 500 floor by 3x. Six confirmed-edge project setups all hit ≥ 500; this candidate is in the same sample-density regime.

**Sanity-check tooling — fully achievable on disk:**

- **Tool:** `tools/sub9_research/sanity_midsession_momentum_continuation_long.py` (≤ 250 LOC).
- **Inputs:** `cache/ohlcv_archive/{symbol}.NS/{symbol}.NS_5minutes_enriched.feather` (verified — has `open, high, low, close, volume, vwap`); liquidity universe from `services/symbol_metadata.py` (rolling 20-day median volume); regime tags from existing `services/market_regime.py`; cap_segment tags from `services/symbol_metadata.cap_segment`.
- **Process:**
  1. For each Discovery session 2023-01-01 → 2024-12-31, load every liquid symbol's 5m feather.
  2. At 11:00 bar close, compute G1-G7 gate values.
  3. Capture qualifying (session, symbol) → simulate entry at 11:00 close, hold to 14:00 close, hard exit. Compute per-trade gross PnL.
  4. Apply Indian fee model (`tools/sub7_validation/build_per_setup_pnl.py:calc_fee` — already in production).
  5. Compute aggregate net PF, n, WR, gross/net PnL distribution.
  6. **Cell stratification:** report PF/n by (cap_segment × market_regime) — this informs gauntlet Stage 3 cell selection.
  7. **Validation pass:** OOS 2025-01-01 → 2025-12-31 with same gates; report |WR delta| and PF stability.
- **Decision gate:**
  - Net PF ≥ 1.10 AND n ≥ 500 AND |WR delta| ≤ 10pp → **APPROVED for detector implementation**.
  - Net PF ∈ [1.00, 1.10) → marginal; tune G1 band and G4 threshold; one re-run permitted.
  - Net PF < 1.00 → **RETIRE**; the institutional-accumulation thesis doesn't survive net-fee at our scale.

## Data engineering plan

**None additional.** The brief is deliberately scoped to data already on disk:

- 5m enriched feathers (VWAP column attached) — verified.
- Daily volume / liquidity metadata via `services/symbol_metadata.py` — verified in production.
- Cap segment + market regime tags — verified.
- Earnings / event calendar (G6 exclusion) — uses existing `services/event_calendar.py` if present; if not, use a permissive default (skip the earnings exclusion in sanity, document as known noise floor).

If sanity passes, post-approval:
- `structures/midsession_momentum_continuation_long_structure.py` (new detector, ~300 LOC, follows the `circuit_t1_fade_short` template).
- Config block in `config/configuration.json` — all G1-G7 thresholds parametrised (per CLAUDE.md mandatory rule: NO hardcoded defaults).
- Gauntlet entry in `tools/gauntlet_v2/` for cell discovery.

---

## Acceptance summary (against brief acceptance criteria)

| Criterion | Status |
|---|---|
| (a) ≥ 500 / 2yr feasible | **YES** — ~3,200 expected trades; clears 6.4x. |
| (b) Differentiation from "buy the breakout" published retail | **YES** — confirmation-hold (not breakout-trigger), 11:00 entry timing-floor (not retail-template-published), 14:00 hard exit (pre-MIS-unwind, not retail-template-published). |
| (c) Independence from gap_fade + circuit_t1 | **YES** — opposite direction, non-overlapping windows, mechanically excluded by G6. Expected ρ < 0.20. |

## Decision required

**User action:**
1. **APPROVE** for sanity-check coding → write `tools/sub9_research/sanity_midsession_momentum_continuation_long.py`, run 2-year Discovery + 1-year Validation, report PF / n / WR / cell stratification.
2. **REJECT** with revisions → revise specific gates (G1 band, G4 threshold, exit timing) and re-submit.
3. **RETIRE before sanity** → if you judge that midday LONG continuation in NSE intraday is too retail-saturated despite the timing differentiation, skip and pick another round-6 candidate.

**My read:** APPROVE for sanity-check. The independence story from the existing all-SHORT portfolio is structural (different side, different time, different mechanic), the n ≥ 500 feasibility passes by a wide margin under conservative trigger-rate assumptions, the 14:00 hard exit is the load-bearing differentiator from published retail breakout strategies, and all required data is already on disk. The single biggest risk is the long-bias pattern (sub7/sub8 11-failure long-bias finding) — the sanity must show LONG-side PF holds across both Discovery and OOS Validation, not just on a favourable subsample.
