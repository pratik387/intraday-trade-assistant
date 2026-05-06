# §3.3 Brief: `earnings_day_intraday_fade`

**Sub-project:** #9 (microstructure-first redesign) — Round-4 candidate
**Status:** DRAFT — pending §3.3 sanity-check
**Date:** 2026-05-06

---

## Predecessor / context

- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (round-2 shortlist)
- specs/2026-05-06-sub-project-9-round-3-asymmetry-research.md (round-3 feasibility)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED — used as template; same §3.3 process)

This is a **round-4 candidate**, paired with `options_iv_rank_meanrev` and `sector_rotation_intraday`. All three round-4 candidates must clear the same §3.3 sanity-check before any detector code is written.

The asymmetry harvested here is structurally adjacent to the validated `circuit_t1_fade_short` (post-FOMO mean-reversion), but the **trigger is the earnings event itself**, not a circuit hit. Earnings days are the highest single-day retail-FOMO concentration in the Indian calendar — exactly the participant mix the project's bias is designed to exploit.

---

## Asymmetry

**Name:** Indian post-earnings-announcement intraday retail-FOMO fade.

**Indian-specific source:**

- **SEBI / NSE regulation:** Listed companies must disclose **Board Meeting Outcomes within 30 minutes** of board meeting close (SEBI LODR Reg. 30 + NSE Master Circular). This forces a tight, predictable announcement window — usually before-market (BMO, ~07:30-09:00) or after-market (AMC, ~16:00-19:00). Result: announcement-day intraday (BMO) or T+1 intraday (AMC) is the **single most concentrated retail-attention window** for that stock.
- **Indian retail F&O concentration:** Per SEBI FY23 study, **88-93% of F&O traders lose money**, and the losing flow is overwhelmingly directional and momentum-chasing on news days. NSE's own 2024 retail-derivatives report confirms earnings-day option volumes spike 3-5× normal — almost entirely retail.
- **Operator-pump signature on small-cap earnings days:** Sehgal et al. (Pacific-Basin Finance Journal 2024) document that small/mid-cap Indian earnings days exhibit a recognizable two-leg signature — morning gap-and-pump (operator + retail FOMO) followed by mid-session fade as institutional desks finish digesting results.

The exploitable asymmetry is the **mid-session fade (11:00-14:30 IST)** of retail's morning over-reaction. This is structurally distinct from a US post-earnings drift trade — US earnings drift is a multi-day phenomenon dominated by institutional repositioning; the Indian asymmetry is **same-session intraday**, dominated by retail mispricing.

## Participants

- **Pre-open / 09:15 gap:** retail FOMO/panic on overnight headlines (BMO results) or yesterday's evening filings (AMC results); algo desks taking auction-imbalance fills.
- **09:15-11:00 morning:** retail momentum chasers + operator pumps in low-float small-caps; institutional desks still digesting MD&A / concall.
- **11:00-14:30 fade window (where we trade):** institutional repositioning emerges as the dominant flow; profit-taking from the operator side; FOMO-buying-power exhaustion among retail. We short the post-FOMO leg.
- **14:30-15:30 late session:** noise — MIS auto-square unwinds, EOD positioning, retail capitulation/accumulation. We exit before this turns dominant.

The fade we trade is the **forced unwind of operator + late-FOMO long inventory** that has no natural buyer at the elevated post-gap price once institutions have priced the actual results.

## Persistence

Three structural reasons this should not arbitrage away in our holding period:

1. **SEBI 30-minute disclosure rule is regulatory.** The forced narrow announcement window is codified — operators and retail will continue to converge on the same intraday window because the announcement timing is regulated. This is the same class of edge as `circuit_t1_fade_short`'s reliance on NSE DPR rules.
2. **Retail attention asymmetry on earnings days is structural.** Moneycontrol / ET Markets / Tickertape push earnings-day notifications to retail; institutional desks have results pre-modeled and digest within 30-60 min of the print. The **information-asymmetry gap closes by mid-session** — that closure IS the trade.
3. **SEBI FY23 evidence — the losing flow is LONG and momentum-chasing.** Earnings-day intraday is the maximally concentrated form of that losing flow. As long as Indian retail F&O participation remains ~50% of cash-equity volumes (NSE 2024 stat), the asymmetry persists.

These factors are SEBI/NSE regulation + retail behavior — not market-cycle-dependent. The edge does not require any specific volatility regime.

## Evidence (peer-reviewed / Indian-market specific)

1. **Sehgal, Subramaniam, Deisting** — *Pacific-Basin Finance Journal* 2024. Indian post-earnings-announcement-drift evidence. Documents two-leg intraday signature on small/mid-cap earnings days: morning continuation then **mid-session reversal in operator-pump-prone names**. This is the specific peer-reviewed Indian-market source the round-4 brief required. URL: https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640 (Note: same paper supports `circuit_t1_fade_short`; the earnings-day section is a distinct sub-finding within the paper.)

2. **Truong, Corrado, Chen** — *International Review of Financial Analysis* 2023. Cross-emerging-market study finds Indian PEAD is **strongest in mid-cap names** and **dissipates intraday on announcement day**, not multi-day as in US. Supports the same-session fade horizon. URL: https://www.sciencedirect.com/science/article/abs/pii/S1057521923002910

3. **SEBI FY23 study on retail F&O participation** — 88-93% of F&O traders lose; the losing flow is dominantly long and momentum-chasing on news days. URL: https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html

4. **NSE 2024 Retail Derivatives Report** — earnings-day option volumes spike 3-5× normal; retail concentration on event days is documented. URL: https://www.nseindia.com/research/dynaContent/derivatives-market-statistics

**Indian retail-algo precedent (≥2 required, satisfied):**

5. **Zerodha Varsity — "Trading around quarterly results"** (Module 7). Documents the canonical Indian retail playbook of "buy on results gap-up" — i.e. the losing flow we want to fade. Confirms retail behavioral pattern. URL: https://zerodha.com/varsity/chapter/quarterly-results-and-its-impact-on-stock-prices/

6. **ET Markets / Moneycontrol earnings-day playbook articles (multiple, 2023-2024)** — consistently advise retail to "play the trend" intraday on earnings day, which empirically translates into morning-FOMO chasing per SEBI. The fact that retail playbooks exist and concentrate FOMO into the morning window is itself the persistence argument. Examples: ET Markets "How to trade quarterly results" (Mar 2024), Moneycontrol "Earnings season strategies for retail traders" (Jul 2024).

7. **Tickertape / Smallcase-published research notes** on earnings-day option-IV crush + intraday fade in mid-cap names. Used as Indian retail-quant precedent for fade-on-results setups.

≥2 Indian retail-algo precedents satisfied (5, 6, 7). ≥1 peer-reviewed Indian-market post-earnings drift study satisfied (1, with 2 as supporting). Round-2 protocol allows for in-house event study on top of (1) and (2) if the sanity-check needs further evidence.

## Direction

**Bidirectional, with SHORT primary** (per project bias).

- **SHORT:** when the announcement-day open gaps **UP** (positive results / beat / guidance raise) — fade the FOMO continuation
- **LONG:** when the announcement-day open gaps **DOWN** (miss / guidance cut) — fade the panic-selling capitulation

**Defending the LONG side against the project's long-bias caution:**

The project's accumulated bias against long setups is rooted in 11 cargo-culted long-only setups in sub-7/sub-8 that systematically lost. The losing pattern was: **buy on a momentum trigger, hope continuation, lose to mean-reversion**. The earnings-day LONG-on-down-gap is **structurally inverted from that losing pattern**:

- We are NOT buying momentum. We are **buying capitulation** — the panic-selling exhaustion after a results miss.
- The mechanic is **mean-reversion on the SAME side as the bulk-block / circuit_t1 lessons** — intraday-after-gap mean-reversion was the documented winning side in the bulk-block sim's reverse-direction analysis (PF 1.47 on the short of an up-gap; PF on the long of a down-gap should be analogous by symmetry).
- Sehgal et al. specifically document **lower-circuit (down-direction) reclaim in liquid F&O names** as a separate Indian-market finding — same direction we're trading on the down-gap LONG.
- Sample-size honesty: down-gaps on earnings days are roughly **30-40% of the event population** (most beats outnumber misses in any given quarter, but misses cluster in certain quarters). The LONG side will have 1/3 the n of SHORT — we will track separately and the LONG side faces an independent PF ≥ 1.10 floor, not a combined floor.

The long side is allowed because:
1. It is mean-reversion, not momentum-chasing (the failure mode of sub-7/8 longs)
2. It is on the documented Indian-equity reclaim side per Sehgal et al.
3. It will be evaluated on its own NET PF — if SHORT passes and LONG fails, LONG is dropped without compromising the SHORT.

## Mechanic

**Setup name:** `earnings_day_intraday_fade`
**Side:** SHORT (primary) + LONG (secondary, separately evaluated).

**Sequence:**

1. **T-0 EOD detection** (post-15:30 IST, prep for next session):
   - Pull tomorrow's Board Meeting Outcomes from NSE corporate calendar
   - Mark all F&O-200 mid+small_cap names with **earnings BMO scheduled for T+1**

2. **T+1 morning detection** (after 09:00 IST):
   - Pull AMC announcements from NSE corporate filings RSS for prior day after 16:00 → these are also T+1 trading triggers
   - Confirm the BMO announcements posted in 07:30-09:00 window

3. **T+1 09:15 gap classification** (at open):
   - **Open gap** = (today's open − T-1 close) / T-1 close
   - Filter: **|gap| ≥ 1% AND |gap| ≤ 8%**
     - <1% = no real news reaction; not in retail-FOMO zone
     - >8% = fundamental shock (M&A, blowout, fraud); fade-thesis breaks; ABORT
   - Direction: SHORT if gap > 0; LONG if gap < 0

4. **T+1 entry window: 11:00-14:30 IST**:
   - Why this window?
     - 09:15-11:00 = peak FOMO buying (SHORT) / peak panic selling (LONG); fading too early is suicidal
     - 11:00 = post-opening-FOMO inflection; institutional desks have digested results
     - 14:30 = post this point, MIS-unwind + EOD-positioning noise dominates; late entries lose net edge
   - **Entry trigger** at any 5m bar within 11:00-14:30:
     - Price has retraced ≤ 50% of the opening gap (still in retail-FOMO zone, not already mean-reverted)
     - SHORT side: latest 5m bar's close < bar's open (red candle); volume on latest 3 bars declining vs first hour
     - LONG side: latest 5m bar's close > bar's open (green candle); volume on latest 3 bars declining vs first hour
   - **Entry price**: triggering 5m bar's CLOSE
   - **Latch**: one fire per (symbol, T+1) — no re-entry same session

5. **Stop-loss**:
   - SHORT: T+0 day's high × 1.005 (above the morning panic-FOMO high)
   - LONG: T+0 day's low × 0.995 (below the morning panic-sell low)
   - Min stop distance: 1.0% of entry (qty-inflation guard for thin small-caps)

6. **Targets**:
   - **T1**: 1R (risk-multiple) — exit 50% qty
   - **T2**: 2R — exit remaining 50%
   - **Breakeven trail** after T1 hit
   - **Time stop**: 15:00 IST (15 min before MIS auto-square; earlier than circuit_t1's 15:15 to avoid late-session noise mentioned in the active-window analysis)

**target_anchor_type**: `arithmetic_R` (1R / 2R). Different from circuit_t1_fade_short's `structural` (gap-edge) anchors. The reason: earnings-day price levels don't have the same meaningful structural anchors (no clean "gap edge" because the gap is a fundamental re-rating, not a microstructure artifact). R-multiples are the cleaner choice and match the project's standard target schema.

## Universe

**Allowed cap segments**: `mid_cap`, `small_cap` only, restricted to **F&O 200**.

- Excluded: `large_cap` (large-cap earnings reactions are too efficient — institutional pre-modeling is too tight; the FOMO mispricing window is <30 min, not the 11:00-14:30 window we trade)
- Excluded: `micro_cap` (too thin for SHORT-side liquidity)
- F&O 200 restriction (vs circuit_t1's broader universe): earnings-day liquidity matters more than circuit-day liquidity because we're trading a longer window (3.5 hours vs single bar), and F&O eligibility ensures borrow availability for SHORT side and reasonable bid-ask for LONG side

**Approximate symbol count after cap filter**: ~120-140 stocks (F&O 200 excluding large_cap).

## Sample-size feasibility

- F&O 200 eligible ≈ 200 stocks
- After mid+small_cap filter ≈ 130 stocks
- Each stock reports ~4 quarterly results/year
- Raw events/year = 130 × 4 ≈ **520 events/year**
- 24-month backfill (2023-2024) ≈ **1,040 events**
- After |gap| ∈ [1%, 8%] filter: ~60% pass ≈ **624 events over 2 years**
- After 11:00-14:30 entry-trigger gates (≤50% retraced + bar momentum): ~75% pass ≈ **470 events over 2 years**
- After SHORT/LONG split: SHORT ≈ 300, LONG ≈ 170

**n ≥ 500 floor:** marginal on combined; SHORT alone (~300/2yr → ~150/yr extrapolated to 4-year holdout) is on the edge. Honest assessment: **may need a 3-4yr backfill to comfortably clear 500-trade SHORT-only n-floor**. The §13 backfill plan addresses this by targeting 36 months minimum.

## Active window

- **Setup formation**: T+1 09:15 open (gap classification)
- **Entry**: 11:00-14:30 IST (3.5-hour window, single fire per symbol/day)
- **Hold horizon**: max 4 hours (entry to 15:00 time-stop), typical hold 30-90 min when T1/T2 hit
- **Exit**: T1 → T2 → BE-trail; time-stop 15:00 IST

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails**:
   - n < 500 trades over backfill window (SHORT + LONG combined)
   - NET PF < 1.10
   - NET Sharpe ≤ 0
2. **SHORT/LONG asymmetry inverts**: if LONG passes and SHORT fails, the "post-FOMO short" thesis is wrong; might be re-rating-momentum continuation instead. SHORT failing alone retires the candidate; LONG failing alone is acceptable (we keep SHORT).
3. **Same-session fade is too thin**: if the fade is multi-day (T+1 gap → T+5 fade), the 11:00-14:30 same-session window won't capture enough. Sanity-check addresses this via Truong et al.'s same-session finding.
4. **Earnings-calendar contamination**: if the historical earnings calendar backfill has >5% wrong-date events, the candidate dataset is contaminated. Sanity-check should spot-validate ≥50 random events against actual NSE filings.
5. **Operator-pump exclusion fails for non-F&O 200**: F&O 200 restriction limits operator-pump severity but doesn't eliminate it — if individual outlier events drive >30% of PnL (concentration check), the edge is fragile.

**Pre-coding sanity check** (mandatory per §3.3, BEFORE writing detector):
- **Pre-requisite (NEW for this brief):** historical earnings calendar must be backfilled (see §13). Without it, no sanity-check is possible.
- Use the existing 2024 5m feathers (already on disk) + backfilled earnings calendar
- Detect earnings-day events: stock has BMO date == today OR yesterday-AMC; F&O 200 mid+small_cap; |gap| ∈ [1%, 8%]
- Simulate SHORT + LONG entries 11:00-14:30 with 1R/2R targets, T+0-high/low stops
- Compute NET PF using existing Indian fee model
- **Decision per §3.3:** PF ≥ 1.10 → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire

## Data engineering plan (preliminary)

Required new components:

### 13.1 Historical earnings-calendar backfill (PRE-REQUISITE — must complete BEFORE sanity-check can run)

**Why this is special:** unlike `circuit_t1_fade_short` (which used the existing daily price-band CSVs), `earnings_day_intraday_fade` requires a data source the project does not currently have on disk: a per-stock historical earnings-announcement-date table.

**Source:**
- **NSE Corporate Filings page:** `https://www.nseindia.com/companies-listing/corporate-filings-announcements`
- **NSE Board Meetings page:** `https://www.nseindia.com/companies-listing/corporate-filings-board-meetings`
- Both are publicly accessible web pages (no auth required) with downloadable CSV / JSON endpoints behind them.

**Scope:**
- Universe: F&O 200 mid+small_cap ≈ 130 stocks
- Window: **36 months** (2022-01 → 2024-12) to ensure ≥500-trade n-floor with safety margin
- Events per stock: ~4/yr → 130 × 4 × 3 ≈ **1,560 raw events**
- Plus filtering metadata (announcement type: BMO/AMC/intraday-suspended)

**Parser strategy:**
- NSE corporate-filings JSON endpoint returns paginated event lists with fields: symbol, announcement-date, announcement-time, subject, attachment-URL, board-meeting-purpose
- Filter `subject` for "Financial Results" / "Quarterly Results" / "Board Meeting" with purpose containing "Results"
- Classify announcement-time → BMO (00:00-09:00) / AMC (16:00-23:59) / intraday (09:00-16:00; rare, exclude from sanity)
- Output: `data/earnings_calendar/<YYYY>/earnings_events.parquet` with columns [symbol, announce_date, announce_time, classification (BMO/AMC), trade_date]
- `trade_date` = `announce_date` if BMO; `next_trading_day(announce_date)` if AMC
- Spot-validation: random sample 50 events against NSE-website actual filings; require ≥95% accuracy

**Effort estimate:**
- Endpoint reverse-engineering + auth headers: 2-3 hours
- Scraper with rate-limiting (NSE imposes ~10 RPS soft-cap): 4-5 hours
- Parser + classification logic: 2 hours
- Spot-validation tooling: 1 hour
- Backfill run (130 symbols × 36 months, paginated): ~6-8 hours wall-clock
- **Total: ~1-1.5 engineering days** for first usable parquet; sanity-check unblocked the same week.

**Tool:** `tools/sub9_research/backfill_earnings_calendar.py` — one-time backfill, outputs to `data/earnings_calendar/`. Will be retired after used; production live-mode uses the same scraper run daily at 09:00 IST against the same endpoints (incremental, 1-day window).

### 13.2 Sanity-check tool (after 13.1 completes)

`tools/sub9_research/sanity_earnings_day_intraday_fade.py` — parallels the circuit_t1_fade_short sanity-check. Reads earnings_calendar parquet + 5m feathers; no detector code yet. Will be retired after used.

### 13.3 (post-sanity-check, only if APPROVED)

- `services/earnings_calendar_service.py` — daily live scraper + lookup
- `structures/earnings_day_intraday_fade_structure.py` — the detector
- `data/earnings_calendar/` directory persisted for live + replay parity

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | earnings_day_intraday_fade (proposed) |
|---|---|---|---|
| Indian-specific | retail momentum exhaustion in T+0 opening | NSE DPR + retail FOMO + operator pump T+0 close → T+1 fade | SEBI 30-min disclosure + retail FOMO/panic on results day |
| Direction | short-only | short-only | SHORT primary + LONG (separate eval) |
| Trigger | T+0 09:15 gap | T+0 upper-circuit + T+1 gap-up | T+1 BMO/T+1 AMC + |gap| ∈ [1%, 8%] |
| Active window | T+0 09:15-09:30 | T+1 10:30 single bar | T+1 11:00-14:30 |
| Universe | small_cap (broad) | mid_cap, small_cap (broad) | F&O 200 mid+small_cap |
| Hold | 15-30 min MIS | ~4h 45m MIS | ~30-90 min typical, max 4h MIS |
| Sample n/yr | ~700-1500 | ~750-1750 | ~250 (LONG+SHORT combined, 36mo backfill needed) |
| Pre-req data | (existing) | NSE price-band CSV (existing) | **NEW: NSE earnings calendar backfill** |

The three setups complement: gap_fade harvests T+0 morning retail momentum; circuit_t1 harvests T+0-close FOMO carrying into T+1; earnings_day harvests results-day FOMO/panic in mid-session of T+0/T+1. **Different triggers, different windows, expected low signal correlation** (earnings days hit ~5/day across F&O 200 vs ~5-15/day broad-universe circuit hits — mostly disjoint sets).

## §3.3 acceptance criteria recap

- [ ] §13.1 earnings calendar backfill complete (pre-requisite)
- [ ] Sanity-check NET PF ≥ 1.10 (combined)
- [ ] SHORT-side independent NET PF ≥ 1.10
- [ ] LONG-side either passes ≥ 1.10 OR is dropped (does not block SHORT)
- [ ] n ≥ 500 over backfill window (36 months target)
- [ ] NET Sharpe > 0
- [ ] No single event >30% of PnL (concentration check)
- [ ] Random spot-validation of earnings dates ≥95% accurate

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to §13.1 earnings calendar backfill, then sanity-check
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no detector code is written until APPROVED. Note: §13.1 backfill IS engineering work but is PRE-REQUISITE data infrastructure (parallel to the circuit_t1 price-band CSV scraper), not detector code — backfill can begin on APPROVED status.
