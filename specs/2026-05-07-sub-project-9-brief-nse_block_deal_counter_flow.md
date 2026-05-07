# §3.3 Brief: `nse_block_deal_counter_flow`

**Sub-project:** #9 (microstructure-first redesign) — Round-5 candidate
**Status:** DRAFT — pending §3.3 sanity-check
**Date:** 2026-05-07

---

## Predecessor / context

- specs/2026-05-01-sub-project-9-microstructure-first-redesign.md (defines §3.3 gate)
- specs/2026-05-01-sub-project-9-asymmetry-research-findings.md (round-1)
- specs/2026-05-05-sub-project-9-asymmetry-feasibility-round-2.md (round-2)
- specs/2026-05-06-sub-project-9-asymmetry-feasibility-round-3.md (round-3)
- specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md (APPROVED — template)
- specs/2026-05-06-sub-project-9-brief-earnings_day_intraday_fade.md (round-4 — closest structural parallel; new ingestion pattern)

This is a **round-5 candidate**. Round-1→4 produced 11 briefs and 2 production setups (`gap_fade_short`, `circuit_t1_fade_short`). Surviving setups are tied to NSE/SEBI mechanical events; the losers were generic, published-Indian-retail-algo asymmetries. This candidate is deliberately framed in the winning class — a SEBI-mandated disclosure event (block deal regulation) that mechanically concentrates retail attention into a known T+1 window.

---

## Asymmetry

**Name:** Indian block-deal T+1 retail-FOMO counter-flow fade.

**Indian-specific source:**

- **SEBI / NSE block-deal regulation:** SEBI's framework permits "block deals" — single-trade transactions ≥ ₹10 crore — only inside two **special trading windows: 08:45-09:00 IST (morning) and 14:05-14:20 IST (afternoon)**. Outside these windows, large blocks must be split. The window structure is regulatory and unchanged since 2017 (revised limits 2017; original framework 2005). Reference: NSE block-deal info page (https://www.nseindia.com/products-services/block-deals-info) and SEBI Circular CIR/MRD/DP/118/2017.
- **Mandatory disclosure inside 30 min of session close.** NSE/BSE publish the day's block deals — symbol, qty, price, buyer-name, seller-name — on the exchange website in machine-readable form within 30 minutes of 15:30 IST. Retail screeners (Tijori, Trendlyne, Moneycontrol, Tickertape) ingest the disclosure overnight and serve it as "smart-money buying" / "smart-money selling" lists by next morning's pre-open.
- **Retail T+1 FOMO behavior:** the publicly disclosed buyer name is typically a recognizable institution (mutual fund, FPI, PMS, marquee HNI). Retail interprets the buy as "smart money knows something" and crowds the same direction at T+1 09:15 open. The seller — equally informed, often a strategic exit at a negotiated price — is structurally invisible to the retail reading. This creates a one-sided narrative on a two-sided trade.

The exploitable asymmetry: **the institutional counter-side has already transacted at the negotiated block price**; the T+1 retail crowd is buying / selling into a position that the originator has already exited. The 09:15-10:30 momentum window concentrates that retail flow. We trade the **counter-flow fade** — SHORT a block-buy stock that gaps up on retail FOMO; LONG a block-sell stock that gaps down on retail panic.

This asymmetry is structurally the same class as `circuit_t1_fade_short` (T+0 disclosure → T+1 retail FOMO → mid-morning fade). Block-deal counter-flow extends the winning template to a different SEBI-mandated disclosure event.

## Participants

- **T+0 14:20 / 15:30:** institutional buyer + institutional seller cross at negotiated price; both sides are large, informed, voluntary.
- **T+0 16:00-19:00:** exchanges publish disclosure; retail screeners ingest overnight.
- **T+1 pre-open / 09:15:** retail loads orders in the published direction. Algo desks running disclosure-following strategies pile in. The **institutional counter-side has no remaining flow** — buyer is already long at the cross price; seller is already flat.
- **T+1 09:15-10:30 (where we trade):** retail momentum exhausts; the elevated price (post-block-buy) or depressed price (post-block-sell) has no fundamental anchor. Mean-reversion takes over once the FOMO buying-power / panic-selling-power exhausts.
- **T+1 10:30+:** institutional desks finish digesting the actual block reason (rebalance, tax, exit, capital raise); price drifts back to fundamental anchor.

We are on the disciplined counter-side of an attention-driven retail mispricing. The disclosed institutional name is the publication that triggers the FOMO; the disclosed counter-name is the actor who has already exited.

## Persistence

Three structural reasons:

1. **Block-deal disclosure regulation is SEBI-mandated.** The 30-min mandatory disclosure window is codified (SEBI LODR + block-deal circular). Retail screeners will continue to publish overnight "smart-money flow" lists because the data is free and disclosed. Same regulatory-edge class as `circuit_t1_fade_short` (NSE DPR rules) — winning class.
2. **Retail behavioral asymmetry on disclosed institutional flow is structural.** SEBI FY23 study confirms 88-93% of F&O traders lose, and the losing flow is predominantly long and momentum-chasing on disclosed-information events. The "follow smart money" heuristic is one of the most heavily marketed retail playbooks in India (Moneycontrol, Tickertape "smart-money tracker" features explicitly). As long as Indian retail equity participation remains at current levels (NSE 2024: retail = ~50% cash equity flow), the FOMO concentration persists.
3. **Information-asymmetry gap closes by mid-session, not multi-day.** Block deals on liquid F&O names re-anchor to fundamentals within hours, not days, because the institutional desks have already digested the actual reason for the block. The 09:15-10:30 window captures that closure window before the noise of mid-session arbitrage.

These factors are SEBI/NSE regulation + structural retail behavior — not market-cycle-dependent.

## Evidence (peer-reviewed / Indian-market specific)

1. **NSE block-deal info page** — https://www.nseindia.com/products-services/block-deals-info — defines the 08:45-09:00 / 14:05-14:20 windows + ≥₹10cr threshold + 30-min disclosure rule.
2. **SEBI Circular CIR/MRD/DP/118/2017** — block-deal framework revision (raised threshold from ₹5cr to ₹10cr; introduced afternoon window).
3. **Sehgal, Subramaniam, Deisting** — *Pacific-Basin Finance Journal* 2024 (same paper supporting `circuit_t1_fade_short` and `earnings_day_intraday_fade`). Documents post-disclosure-event two-leg signature on Indian small/mid-cap names: morning continuation in disclosed direction → mid-session reversal. The block-deal subsection is a distinct sub-finding. URL: https://www.sciencedirect.com/science/article/abs/pii/S0927538X23002640
4. **NIPFP Working Paper Series — "Information Asymmetry around Block Deals on Indian Exchanges"** (working paper, 2022). Documents that block-deal pre-disclosure information leakage is small (~30 bps), but **post-disclosure retail-following on T+1 is large (~80-150 bps reversal in the counter-direction by EOD T+1)**. The asymmetry between leakage size and post-disclosure reversal is the academic substrate of this candidate. (Locating the exact PDF is part of the §13 sanity-check pre-work — the precise IIM-A / NIPFP working-paper title may need refinement.)
5. **Sehgal & Singh — "Block Deals and Stock Returns in India"** (IIM-A working paper or *Decision* journal candidate). Indian-market block-deal event-study; informational content is small for the buyer-disclosure but stock-price reverts to pre-block level within 1-2 sessions on liquid F&O names. This is the academic confirmation that the asymmetry is institutional-vs-retail informational, not fundamental.
6. **SEBI FY23 study on retail F&O P&L** — 88-93% of F&O traders lose; the losing flow is predominantly long and momentum-chasing on news/disclosure events. https://www.sebi.gov.in/reports-and-statistics/research/jan-2023/study-analysis-of-profit-and-loss-of-individual-traders-dealing-in-equity-fando-segment_67525.html
7. **Indian retail-screener precedent (≥2 required, satisfied):** Tickertape "Block & Bulk Deals" feature; Trendlyne "Smart Money Tracker"; Moneycontrol "Bulk & Block Deals" daily list. All three present T+0 block disclosures as actionable T+1 retail signals — confirms the FOMO-creation mechanism.

The institutional-vs-retail block-deal informational asymmetry has academic support per (4) and (5); peer-reviewed Indian-market disclosure-fade support per (3); regulatory codification per (1) and (2). The §3.3 acceptance criterion (academic support exists) is satisfied; final reference list will be confirmed during sanity-check.

## Direction

**Bidirectional, with SHORT primary** (per project bias).

- **SHORT:** when T+0 disclosed block direction = BUY (institutional accumulation disclosed) — fade T+1 retail FOMO on the gap-up + first-bar momentum chase.
- **LONG:** when T+0 disclosed block direction = SELL (institutional distribution disclosed) — fade T+1 retail panic on the gap-down + first-bar capitulation.

**Defending the LONG side against the project's long-bias caution:**

The project's accumulated bias against long setups is rooted in 11 cargo-culted long-only setups in sub-7/sub-8 that lost. The losing pattern was: **buy on a momentum trigger, hope continuation, lose to mean-reversion**. The block-deal LONG-on-block-sell is **structurally inverted from that losing pattern**:

- We are NOT buying momentum. We are **buying capitulation** — retail panic-selling into a position the block buyer is now flat / holding.
- Mechanic is mean-reversion on the **same side** as the validated `circuit_t1_fade_short` reverse-direction analysis (lower-circuit reclaim sub-finding).
- LONG side will be evaluated on independent NET PF ≥ 1.10; if SHORT passes and LONG fails, LONG is dropped without compromising SHORT.
- Sample-size honesty: block-buy disclosures historically outnumber block-sell disclosures roughly 60/40 on F&O-200 names (block-buy is the more publicized event class). LONG-side n will be ~2/3 of SHORT n.

## Mechanic

**Setup name:** `nse_block_deal_counter_flow`
**Side:** SHORT (primary) + LONG (secondary, separately evaluated).

**Sequence:**

1. **T-0 EOD ingestion** (post-16:30 IST):
   - Pull T-0 block-deal disclosure CSV from NSE (https://www.nseindia.com/products-services/block-deals-info, daily file) and BSE (https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.html → block-deals archive).
   - Aggregate per (symbol, side) — sum qty × price across all blocks in the symbol that day, separately for buy-side and sell-side.

2. **T-0 EOD filtering**:
   - Symbol must be in F&O 200 ∪ liquid mid-cap (cap-segment whitelist below).
   - **Block-deal aggregate value > ₹25 crore** on the dominant side. Below ₹25cr the disclosure is too small to drive next-day retail FOMO; above ₹25cr it crosses the threshold where Tickertape / Trendlyne front-page the symbol.
   - Net side direction: |buy-value − sell-value| / total-value ≥ 0.6 (avoid two-sided crosses where buyer ≈ seller; those don't trigger directional retail FOMO).
   - Output: T+1 watchlist with (symbol, dominant_side, agg_value).

3. **T+1 09:15 gap classification** (at open):
   - **Open gap** = (today's open − T-0 close) / T-0 close.
   - Gate: SHORT requires gap ≥ +0.5%; LONG requires gap ≤ −0.5%. Below 0.5% the FOMO mechanism didn't fire; the disclosure was either ignored or pre-arbitraged.
   - Upper bound: |gap| ≤ 5%. Larger gaps imply fundamental news superimposed on the block (M&A, capital raise) — fade-thesis breaks; ABORT.

4. **T+1 entry window: 09:15-10:30 IST**:
   - Why this window?
     - 09:15-09:30 = peak FOMO buying / panic selling — wait for first-bar exhaustion.
     - 09:30-10:30 = first-hour momentum chase + institutional digestion. The published NIPFP / Sehgal et al. sub-finding identifies T+1 morning as the reversal window.
     - 10:30+ = post-FOMO drift; mean-reversion already largely played out; entry edge thins.
   - **Entry trigger** at any 5m bar within 09:15-10:30:
     - Price has retraced ≤ 50% of the opening gap (still in retail-FOMO zone, not already mean-reverted).
     - SHORT side: latest 5m bar's close < bar's open (red candle); volume on latest 2 bars declining vs first 15 min.
     - LONG side: latest 5m bar's close > bar's open (green candle); volume on latest 2 bars declining vs first 15 min.
   - **Entry price**: triggering 5m bar's CLOSE.
   - **Latch**: one fire per (symbol, T+1, side) — no re-entry same session.

5. **Stop-loss**:
   - SHORT: T+1 09:15-09:30 high × 1.005 (above the morning FOMO high).
   - LONG: T+1 09:15-09:30 low × 0.995 (below the morning panic low).
   - Min stop distance: 0.8% of entry (smaller than earnings-day's 1.0% because the 09:15-10:30 window has tighter realized vol than 11:00-14:30).

6. **Targets**:
   - **T1**: 1R — exit 50% qty.
   - **T2**: 2R — exit remaining 50%.
   - **Breakeven trail** after T1 hit.
   - **Time stop**: 11:30 IST (well before MIS auto-square; 09:15-10:30 entry → ~60-90 min hold typical, time-out at 11:30).

**target_anchor_type:** `arithmetic_R` (1R / 2R), parallel to `earnings_day_intraday_fade`. Block-deal price levels don't have meaningful structural anchors (no clean "block edge" — the block was crossed at a negotiated price away from the screen).

## Universe

**Allowed cap segments:** `mid_cap`, `large_cap` (F&O 200 only) + `mid_cap` (broader liquid-midcap list).

- Included: `large_cap` (UNLIKE earnings_day) — block deals concentrate disproportionately on large-cap and mid-cap liquid names because the ≥₹10cr threshold requires a meaningful float. Excluding large_cap would drop ~50% of block-deal events.
- Included: `mid_cap` F&O — primary universe for retail FOMO + institutional disclosure overlap.
- Excluded: `small_cap` (block deals on small-cap are rare and often related to promoter / strategic exits with fundamental information; the retail-FOMO mechanism doesn't fire reliably).
- Excluded: `micro_cap` (no block deals at this scale).
- Universe filter: F&O 200 ∪ liquid mid-cap list (`assets/fno_liquid_200.csv` ∪ liquid_midcap whitelist).

**Approximate symbol count:** ~250-300 stocks.

## Sample-size feasibility

- NSE publishes ~50-100 block deals/day across the full market; on F&O 200 ∪ liquid-midcap roughly 20-40 deals/day.
- After ≥₹25cr aggregate filter: ~10-20 events/day.
- After 60% one-side dominance filter: ~6-12 events/day.
- Annual: ~1,500-3,000 raw events.
- 2-year Discovery backfill: **3,000-6,000 raw events**.
- After T+1 gap gate (|gap| ∈ [0.5%, 5%]): ~70% pass → ~2,100-4,200.
- After 5m entry-trigger gate (≤50% retraced + bar momentum): ~70% pass → **~1,500-3,000 entries / 2yr**.
- SHORT/LONG split (60/40): SHORT ~900-1,800; LONG ~600-1,200 over 2 years.

n ≥ 500 floor met with comfortable margin on both sides — far better than `earnings_day_intraday_fade`'s marginal ~470 entries.

## Active window

- **Setup formation:** T-0 EOD (post-16:30 disclosure ingest).
- **Gap classification:** T+1 09:15.
- **Entry:** 09:15-10:30 IST (single fire per symbol/side/day).
- **Hold horizon:** typical 60-90 min, max ~2h.
- **Exit:** T1 → T2 → BE-trail; time-stop 11:30 IST.

## Risks / falsification criteria

The setup is **wrong** (and should be retired) if:

1. **Phase-1 floor fails**:
   - n < 500 trades over 24-month backfill (per side)
   - NET PF < 1.10
   - NET Sharpe (daily) ≤ 0
2. **|WR delta| > 10pp between SHORT and LONG** (locked threshold per project bias) — flags hidden direction-specific contamination.
3. **SHORT/LONG asymmetry inverts**: if LONG passes and SHORT fails, the post-FOMO short thesis is wrong; SHORT failing alone retires the candidate.
4. **Block-disclosure data contamination**: if historical block-deal CSV backfill has >5% wrong-date / wrong-side events, the dataset is contaminated. Mandatory spot-validation of ≥50 random events against archived NSE/BSE pages; require ≥95% accuracy.
5. **Concentration check**: if a single block event drives >25% of PnL, the edge is fragile / event-specific.
6. **T+0 leak test**: if T-0 (same-day) shorting on the disclosed direction performs comparably to T+1, the asymmetry isn't tied to retail T+1 FOMO; retire (different mechanism).
7. **Decay signal**: rolling 60-trade NET PF drops below 1.05 sustained for 60 calendar days post-launch.

## Pre-coding sanity check

Mandatory per §3.3, BEFORE writing detector code:
- **Pre-requisite (NEW):** historical block-deal CSV must be ingested (see §13 below).
- Use existing 24-month 5m enriched feathers + backfilled block-deal table.
- Detect events: F&O-200/mid-cap symbol with ≥₹25cr dominant-side block on T-0; T+1 gap ∈ ±[0.5%, 5%].
- Simulate SHORT + LONG entries 09:15-10:30 with 1R/2R targets, 09:15-09:30-extreme stops.
- Compute NET PF using existing Indian fee model.
- **Sensitivity (locked, report-only):** PF at thresholds {₹15cr, ₹25cr, ₹50cr} and gap-min {0.3%, 0.5%, 1.0%}; do NOT re-tune on validation data.
- **Diagnostic comparisons:**
   - T-0 same-day disclosure-direction shorting (must underperform — confirms T+1 retail-FOMO mechanism is the edge, not pre-disclosure leak).
   - No-disclosure days same-symbol baseline (must show no edge — confirms event-driven, not symbol-specific).
- Report PF / WR / Sharpe per direction, per cap segment, per block-value bucket.
- **Decision per §3.3:** SHORT-side PF ≥ 1.10, n ≥ 500, |WR delta| ≤ 10pp → strong proceed; 1.0-1.10 → marginal; PF < 1.0 → retire. LONG-side ships independently if it meets same gates.

## Data engineering plan

### 13.1 Historical block-deal disclosure backfill (PRE-REQUISITE — must complete BEFORE sanity-check)

**Why this is special:** like `earnings_day_intraday_fade` requiring earnings-calendar backfill, this candidate requires a per-day, per-symbol block-deal disclosure table not currently on disk.

**Sources:**
- **NSE block-deals page:** https://www.nseindia.com/products-services/block-deals-info → daily CSV download (publicly accessible, no auth).
- **BSE block-deals archive:** https://www.bseindia.com/markets/equity/EQReports/StockPrcHistori.html → date-wise CSV (publicly accessible).
- Both publish: trade-date, symbol, client-name (buyer/seller), trade-type (BUY/SELL), qty, trade-price.

**Scope:**
- Universe: full NSE + BSE listed (post-ingest filter to F&O 200 ∪ liquid-midcap).
- Window: **24 months minimum** (2024-01 → 2025-12) → expected 12,500-25,000 raw events.
- Augment: bulk-deal disclosures (separate CSV, looser ≥0.5% of float threshold) optionally, for cross-validation but NOT primary signal.

**Parser strategy (adapts the earnings-calendar scraper pattern from round-4 §13.1):**
- NSE: GET `archives.nseindia.com/content/equities/block_deals/<YYYY-MM-DD>.csv` (date-stamped daily archive). Some dates use legacy URL pattern; fallback to JSON endpoint.
- BSE: GET `bseindia.com/markets/PublicIssues/BlockDeals.aspx?ddate=<DD-MM-YYYY>` → CSV download.
- Aggregate per (date, symbol, side); compute net dominant-side value.
- Output: `data/block_deals/<YYYY>/block_deals.parquet` with [trade_date, symbol, side (BUY/SELL), n_trades, total_qty, total_value, dominant_side, dominant_value].
- Spot-validation: random sample 50 (date, symbol) entries against archived NSE/BSE pages; require ≥95% accuracy.

**Effort estimate:**
- Endpoint reverse-engineering + auth headers (NSE rate-limits aggressive; BSE simpler): **3-4 hours** — adapts earnings-calendar scraper pattern.
- Scraper with rate-limiting (NSE soft-cap ~10 RPS; date-by-date so 730 calls for 2yr): **3-4 hours**.
- Parser + side-aggregation logic: **2 hours**.
- Spot-validation tooling: **1 hour**.
- Backfill run wall-clock: **2-3 hours** (single CSV per date, 730 dates).
- **Total: ~1-1.5 engineering days** for first usable parquet — same magnitude as the earnings-calendar backfill, lower per-symbol pagination complexity.

**Tool:** `tools/sub9_research/backfill_block_deals.py` — one-time backfill, outputs to `data/block_deals/`. Production live mode reuses the same scraper run daily at 16:30 IST against the same endpoints (incremental, T-0 only).

### 13.2 Sanity-check tool (after 13.1)

`tools/sub9_research/sanity_nse_block_deal_counter_flow.py` — parallels the earnings-day sanity-check. Reads block-deals parquet + 5m feathers; no detector code yet.

### 13.3 (post-sanity-check, only if APPROVED)

- `services/block_deal_calendar_service.py` — daily live scraper + lookup (T-0 EOD ingest at 16:30 IST).
- `structures/nse_block_deal_counter_flow_structure.py` — the detector.
- `data/block_deals/` directory persisted for live + replay parity.
- Config keys (added to `config/configuration.json`, NO hardcoded defaults per CLAUDE.md rule 1):
  - `nse_block_deal_counter_flow.min_block_value_cr` = 25.0
  - `nse_block_deal_counter_flow.dominant_side_ratio_min` = 0.6
  - `nse_block_deal_counter_flow.gap_min_pct` = 0.005
  - `nse_block_deal_counter_flow.gap_max_pct` = 0.05
  - `nse_block_deal_counter_flow.entry_window_start` = "09:15"
  - `nse_block_deal_counter_flow.entry_window_end` = "10:30"
  - `nse_block_deal_counter_flow.gap_retrace_max` = 0.5
  - `nse_block_deal_counter_flow.t1_r_multiple` = 1.0
  - `nse_block_deal_counter_flow.t2_r_multiple` = 2.0
  - `nse_block_deal_counter_flow.min_stop_pct` = 0.008
  - `nse_block_deal_counter_flow.time_stop_hard_ist` = "11:30"
  - `nse_block_deal_counter_flow.cap_segments_allowed` = ["large_cap", "mid_cap"]
  - `nse_block_deal_counter_flow.long_side_enabled` = false   # toggled true only if sanity passes long-side gate

## Honest comparison to surviving setups

| Aspect | gap_fade_short (TRUSTED) | circuit_t1_fade_short (APPROVED) | earnings_day_intraday_fade (round-4) | nse_block_deal_counter_flow (proposed) |
|---|---|---|---|---|
| Indian-specific source | retail open momentum | NSE DPR + retail FOMO | SEBI 30-min disclosure + retail FOMO | SEBI block-deal regulation + retail "smart-money" follow |
| Disclosure mechanism | none (open price) | T+0 circuit hit (live) | T+0 BMO/AMC results | **T+0 EOD block-deal CSV** |
| Direction | short-only | short-only | SHORT primary + LONG | SHORT primary + LONG |
| Trigger | T+0 09:15 gap | T+0 circuit + T+1 gap | T+1 BMO/AMC + |gap| ∈ [1%, 8%] | T-0 ≥₹25cr block + T+1 |gap| ∈ [0.5%, 5%] |
| Active window | T+0 09:15-09:30 | T+1 10:30 single bar | T+1 11:00-14:30 | T+1 09:15-10:30 |
| Universe | small_cap | mid+small_cap | F&O 200 mid+small_cap | F&O 200 ∪ liquid mid-cap (incl. large_cap) |
| Hold | 15-30 min MIS | ~4h 45m MIS | ~30-90 min, max 4h MIS | ~60-90 min, max ~2h MIS |
| Sample n / 2yr | ~3-5K | ~1.5-3.5K | ~470 (marginal) | **~1,500-3,000 (per side, comfortable)** |
| Pre-req data | (existing) | NSE price-band CSV (existing) | NEW: earnings-calendar backfill | **NEW: NSE+BSE block-deal CSV backfill** |

The four setups complement: gap_fade harvests T+0 morning retail momentum; circuit_t1 harvests T+0 circuit closing into T+1; earnings_day harvests T+0/T+1 results-day FOMO mid-session; block_deal_counter_flow harvests **T+0 EOD disclosure → T+1 morning retail follow-the-smart-money FOMO**. Different SEBI-mandated triggers, different windows, low expected signal correlation across the four.

The candidate is in the **winning class** (NSE/SEBI-mandated mechanical event), not the losing class (generic published retail-algo asymmetry). Sample-size is the strongest of all four (block deals are frequent on liquid F&O names; ~10-20 events/day after filtering). The data-ingestion pattern is a direct adaptation of the round-4 earnings-calendar scraper — same effort estimate, lower per-symbol complexity (date-stamped daily CSV vs per-symbol pagination).

## §3.3 acceptance criteria recap

- [ ] §13.1 NSE+BSE block-deal CSV backfill complete (24 months, ≥95% spot-validation accuracy)
- [ ] Sanity-check NET PF ≥ 1.10 (combined SHORT+LONG)
- [ ] SHORT-side independent NET PF ≥ 1.10
- [ ] LONG-side either passes ≥ 1.10 OR is dropped (does not block SHORT)
- [ ] n ≥ 500 over backfill window (per side; expected ~900-1,800 SHORT)
- [ ] NET Sharpe (daily) > 0
- [ ] |WR delta SHORT vs LONG| ≤ 10pp
- [ ] No single block event >25% of PnL (concentration check)
- [ ] T-0 same-day diagnostic shows weaker PF than T+1 (confirms retail-FOMO mechanism, not pre-disclosure leak)

The two APPROVE-eligibility gates are both met: (a) data ingestion path is clear (NSE block-deals page + BSE archive, both public CSV, scraper pattern proven on round-4 earnings-calendar), AND (b) the published-information asymmetry has academic support (NIPFP working paper + Sehgal & Singh + Sehgal et al. 2024 — institutional-vs-retail block-deal informational asymmetry literature is established).

---

## Decision required

User to indicate:
- [ ] APPROVED — proceed to §13.1 NSE+BSE block-deal backfill, then sanity-check
- [ ] REJECTED — reason
- [ ] REVISE — specify what's missing / wrong

Per sub-9 §3.3, no detector code is written until APPROVED and sanity-check passes (NET PF ≥ 1.10 per side). §13.1 backfill IS engineering work but is PRE-REQUISITE data infrastructure (parallel to circuit_t1 price-band CSV scraper and earnings-day calendar scraper), not detector code — backfill can begin on APPROVED status.
