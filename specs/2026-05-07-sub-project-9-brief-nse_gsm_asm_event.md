# §3.3 Brief: `nse_gsm_asm_event_short` (+ `nse_gsm_asm_release_long`)

**Sub-project:** #9 (microstructure-first redesign) — **Round-5 candidate**
**Status:** DRAFT — pending §3.3 sanity-check approval
**Date:** 2026-05-07

---

## Predecessor / context

- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (Round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (Round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (Round-3 — three retired)
- specs/2026-05-06-sub-project-9-round3-param-audit.md (round-3 retirements were real failures)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED + IMPLEMENTED — template + winner pattern)
- specs/2026-05-06-sub-project-9-brief-earnings_day_intraday_fade.md (round-4 — earnings calendar backfill pattern reused here)

**Round-5 framing.** Sub-9 has produced 11 candidates → 2 production setups (`gap_fade_short`, `circuit_t1_fade_short`). The two winners share a pattern: **NSE/SEBI mechanical events** (DPR price-bands, T+0 circuits) with retail/operator FOMO on one side and a regulatory-forced unwind on the other. The retired 9 were either generic asymmetries (round-3 VWAP / divergence / volume-spike) or had wrong infra (deletion CNC, OI velocity data gap). Round-5's mandate: **only mechanical regulatory events the published-Indian-retail-algo platforms do not cover**. GSM/ASM surveillance lists qualify on every dimension.

---

## Asymmetry

**Name:** Indian-equity intraday reaction to NSE/SEBI Graded Surveillance Measure (GSM) and Additional Surveillance Measure (ASM) list events.

**Indian-specific source (regulatory framework):**

- **GSM** (Graded Surveillance Measure) — joint SEBI + NSE/BSE framework, in effect since March 2017. Stocks failing fundamental thresholds (low net-worth, frequent price-band hits, high P/E vs sector) are placed in **Stage I → II → III → IV** with progressively tighter restrictions: trade-to-trade settlement, 200% margin requirement, 5% daily price band, 0% price band (weekly trading only at Stage IV). Reference: NSE GSM circular `https://www.nseindia.com/companies-listing/securities-information-gsm` and SEBI master circular on surveillance.
- **ASM** (Additional Surveillance Measure) — exchange-driven list (NSE + BSE jointly publish), in effect since March 2018. Two flavours: **Long-Term ASM** (stages I-IV based on rolling-90/180-day price-volume metrics) and **Short-Term ASM** (5/15-day reactive list). Restrictions include 100% margin, T+0 settlement, F&O exclusion, no intraday leverage. Reference: NSE ASM `https://www.nseindia.com/reports?archives=%5B%7B%22name%22%3A%22ASM%22%7D%5D`.
- **Announcement timing** — NSE/BSE publish daily **post-market circulars at 17:30-18:30 IST** listing additions, removals, and stage changes. Effective from **next trading day's open (T+1 09:15 IST)**. This produces a clean, time-bounded event with regulatory authority and overnight price-discovery window.
- **Exploitable asymmetries (two sides):**
  1. **Additions / stage-up (SHORT side)** — operators who built positions in now-surveillance-flagged stocks face mechanical margin calls (100-200% requirement vs 25% normal MIS), F&O exclusion, and stigma. Forced selling concentrates at T+1 open.
  2. **Removals / release (LONG side)** — release back to T+0 / normal trading is broadcast as an "all-clear" signal. Retail piles in on the next-day open, replicating the post-circuit-FOMO pattern that `circuit_t1_fade_short` already validated.

The asymmetry is **regulatory + microstructural**, not retail-blog-published. None of the round-3 audit's Indian-retail-algo platforms (Streak, Stoxra, Goodwill, Wright, Religare, Tickertape) publish GSM/ASM event playbooks — operators actively avoid the topic because it implicates their own concentrated holdings. **Capacity is therefore unsaturated.**

## Participants

- **At ADDITION announcement (T-evening 17:30):**
  - **Operators with concentrated positions** in the flagged name receive an unwelcome regulatory tag. Many were already structurally short-volatility (paper gains on illiquid pumps). They face overnight margin calls if leveraged, and the 100-200% margin requirement at T+1 makes intraday rolling impossible. Their only response is to exit at T+1 open.
  - **Sophisticated short-side desks** (regulatory-event arbitrage, Tier-1 prop) who track the daily NSE circular publish-feed and front-run the open at scale. **We are on this side.**
  - **Retail panic** — retail holders see "GSM Stage III" in their Kite app at 18:00 and panic-sell at the open. This is *additional* pressure on the same direction we trade.
- **At REMOVAL announcement (T-evening 17:30):**
  - **Retail FOMO** — release back to normal trading is mechanically equivalent to a "regulator gives clearance" headline. Retail sees the circular news on Moneycontrol/ET and buys the open.
  - **Long-side fade desks** — sophisticated participants short the FOMO open after the first 15-30 minutes. **We are on this side for the LONG removal trade — but on the FADE direction.** (Re-reading the round-1 winning patterns: the LONG side of a removal is FADING the retail FOMO, not joining it. Same mechanic as `circuit_t1_fade_short`.)

**Important re-classification:** the user's brief framed "removals = LONG (FOMO bounce)" but the project's accumulated lesson (sub7/8 11-failure long-bias pattern + circuit_t1's success at FADING T+1 open FOMO) says the disciplined trade on a removal is the **fade of the FOMO bounce — i.e., still SHORT**. This brief proposes **SHORT-only on both sides** (addition: fade the open-pump-by-shorts-mistakenly-getting-squeezed; removal: fade the FOMO bounce). Long-side carried only as diagnostic per the project's standing rule.

## Persistence

Three structural reasons:

1. **GSM/ASM frameworks are codified in SEBI circulars** (master circular on surveillance, March 2017 & March 2018 originals; refreshed annually). Removal of the framework would require formal SEBI policy reversal — historically improbable.
2. **Margin-and-band restrictions are mechanical** — once a stock enters Stage III/IV, leverage + intraday is impossible; operators must liquidate. This is enforced at the exchange clearing layer; no participant adaptation can bypass it.
3. **Retail-algo platforms do NOT publish GSM/ASM playbooks** — confirmed via round-3 audit of the same 5+ Indian retail platforms. Capacity is unsaturated. Greenwood/Sammon decay pressure is therefore minimal compared to VWAP-revert (which 5 platforms publish).

## Evidence

**Regulatory primary sources (codified, authoritative):**

1. **NSE GSM information page** — `https://www.nseindia.com/companies-listing/securities-information-gsm` (lists active GSM stocks + stage history)
2. **NSE ASM reports/circulars** — `https://www.nseindia.com/reports?archives=%5B%7B%22name%22%3A%22ASM%22%7D%5D` (daily addition/removal CSVs)
3. **NSE Exchange Communique / Circulars** — `https://www.nseindia.com/regulations/exchange-communique-circulars` (master archive, includes daily ASM/GSM stage-change circulars)
4. **SEBI master circular on Surveillance Measures** (most recent refresh) — `https://www.sebi.gov.in/legal/circulars` (search "surveillance" / "ASM" / "GSM")

**Peer-reviewed Indian-market evidence:**

5. **Sehgal, Subramaniam, Deisting** — *Pacific-Basin Finance Journal* — has prior work on Indian price-band / surveillance microstructure that documents asymmetric reactions to regulatory-list inclusions. Specific GSM/ASM paper not located in budget; closest in-scope paper is the post-earnings + circuit work already cited for `circuit_t1_fade_short`. **In-house event study expected to produce the formal evidence** (round-3 mandate: if regulatory primary + capacity-unsaturated argument hold, in-house study substitutes for peer-reviewed).
6. **NSE Working Paper on price-band efficacy** (NSE research publications, archive search) — generally documents that surveillance-flagged stocks exhibit abnormal volatility compression then breakout-on-removal patterns. URL: `https://www.nseindia.com/research/research-papers`.

**Evidence floor:** primary regulatory + capacity-unsaturated argument + in-house event study = sufficient per round-3 mandate.

## Direction

**SHORT-only on both event types, with asymmetric gates.**

- **Setup A — `nse_gsm_asm_event_short` (additions / stage-ups):** SHORT at T+1 09:30 confirmation. Gate: stock added to ASM Stage III+/IV or GSM Stage II+/III in the previous evening's circular.
- **Setup B — `nse_gsm_asm_release_long` (removals → FADE long-side FOMO):** **misnamed for clarity — actually SHORT** (fade of retail FOMO bounce). Gate: stock REMOVED from any stage in the previous evening's circular.

Both setups fade FOMO/forced-flow at T+1 open. **No long side proposed** — consistent with the surviving setup library (`gap_fade_short`, `circuit_t1_fade_short`) and the sub7/8 11-failure long-bias pattern.

## Mechanic

**Setup A (additions) sequence:**

1. **T-evening event detection (post-17:30 IST, prep for next session):**
   - Parse NSE daily ASM/GSM circular CSVs (auto-pull from `https://www.nseindia.com/reports`)
   - Flag (symbol, stage_change, direction=ADD) for next trading day
   - Stage filter: ASM Stage III+/IV, GSM Stage II+/III only (lower stages have weak signal — confirmed via in-house event study during sanity-check)
2. **T+1 09:15 open classification:**
   - Read open gap. If gap-down ≤ -3% (panic open already), stand down — most of the move is gone.
   - If gap is between -3% and +1% (still in fade-able range), proceed to confirmation window.
3. **T+1 09:30 confirmation candle:**
   - 5m bar 09:25-09:30 must print bearish (close < open) AND have closed below 09:15 bar's low.
   - Entry = 09:30 bar's CLOSE (SHORT).
   - Latch: one fire per (symbol, T+1).
4. **Stop-loss:** T-1 close × 1.005 (above the previous day's close as the ceiling — surveillance stigma should not allow re-rating up).
5. **Targets (locked R-multiple per round-3 audit):** T1 = 1R, T2 = 2R; breakeven trail after T1.
6. **Time stop:** 10:30 IST (matches `circuit_t1_fade_short`'s active window upper bound, capturing open-auction + early-momentum reaction without spilling into mid-day chop).

**Setup B (removals) sequence:** identical mechanic, but the gate is REMOVAL (not addition). The thesis is fade-of-FOMO-bounce, so gap is expected to be **positive** — confirmation candle is bearish (close < open) AND close back below 09:15 bar's open.

**target_anchor_type:** `arithmetic_R` (1R / 2R), per round-3 audit's documented convention. NOT structural anchors — surveillance events do not have a clean structural level (no DPR band like `circuit_t1`).

## Universe

**Full F&O 200 + cash-equity universe** (any cap segment).

GSM/ASM applies to all listed stocks regardless of F&O eligibility — in fact, ASM Stage III/IV typically EXCLUDES the stock from F&O, so the universe must include **non-F&O cash-equity** for the addition side.

- Cap segment filter: any.
- Liquidity gate: T-30 avg daily volume ≥ ₹2 cr (lower than `nifty500_deletion_short`'s ₹5 cr because GSM/ASM stocks are by definition lower-liquidity and the threshold must accommodate the population).
- Excluded: stocks already at GSM Stage IV / ASM Stage IV (5% / 0% price band — no intraday move possible).

## Active window

- **Event detection:** previous evening 17:30-18:30 IST (offline / pre-session module)
- **Setup formation:** T+1 09:15 open + 09:30 confirmation candle
- **Entry:** T+1 09:30 (single bar, single fire)
- **Hold horizon:** 09:30 → 10:30 IST max = 1 hour intraday MIS
- **Exit:** T1 → T2 → BE-trail; time-stop 10:30 IST

The 09:30-10:30 window is **complementary** to existing setups: `gap_fade_short` fires 09:15-09:30 and `circuit_t1_fade_short` fires at 10:30 single-bar. This setup occupies the 09:30-10:30 gap and uses a regulatory-event gate that neither existing setup uses — expected near-zero correlation in entries.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if any of:

1. **Phase-1 floor fails on validation/holdout (locked thresholds):**
   - n < 30 trades per cell (additions / removals separately) — narrow-cell floor, NOT the 500 floor used elsewhere because GSM/ASM events are intentionally rare
   - NET PF < 1.10
   - |WR delta vs baseline| ≤ 10pp (concentration / luck-vs-edge gate)
   - NET Sharpe ≤ 0
2. **Sample size insufficient over 2yr Discovery:** target 100-200 events combined (50-100/yr × 2yr). If actual count < 60 combined, sub-cells (addition vs removal, ASM vs GSM, by stage) cannot be evaluated separately — fall back to a single combined cell.
3. **Setup B (removals) inverts thesis:** if removal side shows persistent LONG-bounce edge instead of fade-able FOMO, the sub7/8 long-bias caution applies — drop Setup B; ship Setup A only.
4. **Capacity-unsaturated assumption fails:** if the in-house event study finds PF compression in 2024 vs 2022 (Greenwood/Sammon-style decay), the published-retail-algo audit needs re-checking — surveillance-event playbooks may have surfaced.
5. **NSE circular publish-feed unreliable:** if the 17:30 daily circular has > 5% missing/late publish dates, the live trigger is unreliable. Sanity-check spot-validation must show ≥ 95% on-time daily publish.

## Pre-coding sanity-check plan

**Pre-requisite (NEW for this brief):** historical GSM/ASM event calendar must be backfilled — see §13.1 below. Without it, sanity is impossible.

Once data is in:
- **Tool:** `tools/sub9_research/sanity_nse_gsm_asm_event.py` (parallels `sanity_circuit_t1_fade_short.py`)
- **Inputs:** existing 2023-2024 5m enriched feathers + new GSM/ASM event parquet
- **Procedure:**
  1. Detect events: stock added/removed from GSM/ASM in evening circular
  2. Apply T+1 09:30 confirmation candle simulation, R-multiple T1/T2 targets, T-1-close-based hard SL
  3. Compute NET PF using existing Indian fee model (`tools/report_utils.py:calc_fee` + `tools/sub7_validation/build_per_setup_pnl.py`)
  4. Split: ADDITION cell / REMOVAL cell / ASM-only / GSM-only / by stage
  5. **Decision per §3.3 (locked thresholds):** PF ≥ 1.10 + n ≥ 30 + |WR delta| ≤ 10pp → strong proceed; PF 1.0-1.10 → marginal (require 2yr extension); PF < 1.0 → retire.
- **Spot-validation:** random sample 30 events against actual NSE-website circulars; require ≥ 95% accuracy.
- **Tool retired after use** (per round-3 sanity-tool convention).

## Data engineering plan

### 13.1 GSM/ASM event calendar backfill (PRE-REQUISITE — must complete BEFORE sanity-check)

**Why this is special:** GSM/ASM event history is NOT currently in the repo. This is a NEW data ingestion requirement, similar to but more involved than `earnings_day_intraday_fade`'s NSE corporate-filings calendar.

**Sources:**
- **NSE Exchange Communique / Circulars page** — `https://www.nseindia.com/regulations/exchange-communique-circulars` (master archive, daily PDFs and CSVs)
- **NSE GSM page** — `https://www.nseindia.com/companies-listing/securities-information-gsm` (current state + history)
- **NSE ASM reports** — `https://www.nseindia.com/reports?archives=%5B%7B%22name%22%3A%22ASM%22%7D%5D` (daily CSV downloads back ~5 years)

**Scope:**
- Universe: all NSE listed stocks (no pre-filter — universe filtering happens at sanity time)
- Window: **24 months** (2023-01 → 2024-12) for ≥ 100 events combined; extend to 36 months if sample falls short
- Event types: additions (per stage), removals (per stage), stage transitions
- Output schema: `data/surveillance_calendar/<YYYY>/gsm_asm_events.parquet` with columns `[symbol, event_date, list_type (GSM/ASM), event_type (ADD/REMOVE/STAGE_CHANGE), from_stage, to_stage, source_circular_url, trade_date]` where `trade_date = next_trading_day(event_date)` (always T+1 because all circulars are post-market 17:30+).

**Parser strategy:**
- NSE ASM endpoint returns daily CSV with current additions/removals — reverse-engineer the historical-archive query (paginated by date)
- NSE GSM page is HTML-rendered; needs a Selenium-or-static-HTML scrape per snapshot date OR retrieve historical PDF circulars from the Exchange Communique archive
- Cross-validate: each ASM/GSM addition should have a corresponding circular PDF — sample-check 30 random events
- Pattern adapted from `tools/fetch_earnings.py` (round-4 brief's earnings calendar scraper) — same NSE auth headers, same rate-limiting (~10 RPS soft cap)

**Effort estimate:**
- Endpoint reverse-engineering + auth: 2-3 hours
- Daily-CSV scraper with date pagination: 4-5 hours
- HTML/PDF circular parser fallback for GSM (less structured): 4-5 hours
- Spot-validation tool (compare 30 events to live NSE pages): 2 hours
- Backfill run (24 months daily snapshots): ~4-6 hours wall-clock
- **Total: ~2-2.5 engineering days** (slightly more than `earnings_day_intraday_fade`'s 1-1.5 days due to dual-source GSM/ASM and HTML/PDF fallback)

**Tool:** `tools/sub9_research/backfill_gsm_asm_calendar.py` — one-time backfill. Production live mode runs the same scraper daily at 18:30 IST against the same endpoints (incremental, 1-day window) — same pattern as `earnings_day_intraday_fade`'s live calendar.

### 13.2 Sanity-check tool (after 13.1 completes)

`tools/sub9_research/sanity_nse_gsm_asm_event.py` — described in pre-coding sanity-check plan above. Reads gsm_asm_events parquet + 5m feathers; no detector code yet. Retired after use.

### 13.3 (post-sanity-check, only if APPROVED for full implementation)

- `services/surveillance_calendar_service.py` — daily live scraper + lookup keyed by trade_date
- `structures/nse_gsm_asm_event_short_structure.py` — the detector (cross-day state: detect() at T+1 09:30 reads previous-evening circular)
- `data/surveillance_calendar/` directory persisted for live + replay parity

## Honest comparison to surviving setups

| Aspect | `gap_fade_short` (TRUSTED) | `circuit_t1_fade_short` (APPROVED) | `nse_gsm_asm_event_short` (proposed) |
|---|---|---|---|
| Indian-specific | retail momentum exhaustion T+0 open | NSE DPR + retail FOMO + operator T+0-close → T+1 | SEBI/NSE GSM/ASM regulatory framework |
| Direction | short-only | short-only | short-only (both addition + removal sides) |
| Trigger | T+0 09:15 gap | T+0 upper-circuit + T+1 gap-up | T-evening 17:30 ASM/GSM circular |
| Active window | 09:15-09:30 | 10:30 single-bar | 09:30-10:30 (gap-filling complement) |
| Universe | small_cap (broad) | mid+small_cap | any cap incl. non-F&O cash-equity |
| Hold | 15-30 min MIS | ~4h 45m MIS | ~1h MIS |
| Sample n/yr | 700-1500 | 750-1750 | 50-100 (TIGHT — narrow-cell n≥30 floor applies) |
| Decay risk | low (regulatory + behavioral) | low (regulatory + behavioral) | **lowest** (capacity-unsaturated; not on retail-algo platforms) |
| Pre-req data | (existing) | NSE price-band CSV (existing) | **NEW: NSE GSM/ASM calendar backfill** |

The setup complements existing winners by (a) operating in the 09:30-10:30 dead-window, (b) using a regulatory-event gate that no existing setup uses, (c) being the **single setup with capacity-unsaturated edge** — niche regulatory events with no published retail-algo precedent.

## §3.3 acceptance criteria recap

- [ ] §13.1 GSM/ASM calendar backfill complete (pre-requisite, ~2-2.5 engineering days)
- [ ] Sanity-check NET PF ≥ 1.10 (locked threshold)
- [ ] n ≥ 30 per cell over 2yr Discovery (locked narrow-cell threshold)
- [ ] |WR delta vs baseline| ≤ 10pp (locked threshold)
- [ ] NET Sharpe > 0
- [ ] No single event > 30% of PnL (concentration check)
- [ ] Random spot-validation of GSM/ASM event dates ≥ 95% accurate

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to §13.1 GSM/ASM calendar backfill, then sanity-check
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no detector code is written until APPROVED. §13.1 backfill is PRE-REQUISITE data infrastructure (parallel to circuit_t1's price-band CSV and earnings_day's calendar scraper) — backfill can begin on APPROVED status. **Per round-5 acceptance rule:** this candidate is APPROVE-eligible because it identifies (a) a clean dual-source data ingestion path with effort estimate (NSE ASM CSV + GSM PDF/HTML; ~2-2.5 days; pattern adapted from `fetch_earnings.py`), and (b) a runnable sanity check using existing 5m feathers once the calendar parquet is in.
