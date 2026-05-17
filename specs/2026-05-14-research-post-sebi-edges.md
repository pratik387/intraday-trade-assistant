# Research: Post-SEBI Edge Setups

**Branch:** `research/post-sebi-edge-setups`
**Started:** 2026-05-14
**Status:** Pruned to 3 candidates (2026-05-16). All 3 now resolved:
- ✅ #5 COMPLETE — paper-trade comparison parked, see `specs/2026-05-16-gap-fade-short-paper-trade-validation.md`
- ❌ #2 KILLED at falsifier stage — see `docs/retired_setups.md → fno_ban_t1_fade_short`
- ❌ #4 KILLED at data-recon stage (NSE retired pre-open archive endpoint) — see `docs/retired_setups.md → pre_open_auction_direction_follow`

**Net outcome of this research branch:** No new setups shipped to production. Existing 3-setup portfolio (`gap_fade_short`, `circuit_t1_fade_short`, `delivery_pct_anomaly_short`) remains the active configuration. Negative-knowledge captured in `docs/retired_setups.md` (2 more entries) and `memory/feedback_data_availability_pre_check.md` (n-check the parquet before scoring a brief).

## Pruned candidates (2026-05-16 decision)

Dropped from active research — kept below for archival reference only:
- ❌ **#1 Post-MWPL-Recalc Forced Rebalance Fade** — sample too thin (3-4 events/yr), high data-scrape cost
- ❌ **#3 Single-Stock F&O Removal Pre-Announcement Drift** — borderline sample (n=30-50), regulatory-window dependent
- ❌ **#6 Lot-Size-Doubled NIFTY Pinning** — requires options-trading broker integration (currently cash-equity only)

Active research order: **#5 (STT-aware gap_fade re-opt) → #2 (F&O Ban-Entry) → #4 (Pre-Open Auction).**

## Context

Our production gauntlet for `delivery_pct_anomaly_short` shipped a setup that
worked Discovery (2023-24) + OOS (Jan-Sep 2025) — PF 1.245 — but broke in
Holdout (Oct 25-Apr 26) — PF 0.879. Root cause: SEBI's Oct 1, 2025 F&O rule
changes (MWPL formula tightened, single-stock position limits cut) eliminated
the speculator-inventory-unwind mechanism the setup fades.

Our gauntlet methodology was blind to regulatory regime breaks. This research
branch corrects that by:

1. Working **only on post-SEBI data** (Oct 1, 2025 onwards)
2. **Pre-registering depends_on tags** so the regime-break detector can validate windows
3. Brainstorming candidate setups whose **mechanism is created by Oct 1, 2025+ rules** rather than vestigial from older rules
4. Using **paper-trade as Holdout** (live-data, post-war, post-SEBI)

## Hard Learnings Locked

### 1. Regulatory regimes break setups silently
Gauntlets must NEVER straddle a high/critical rule change for any of the
strategy's `depends_on` tags. Use `services.regime_break_detector.check_window`
as the pre-flight gate.

### 2. Edge decay has three flavors — diagnose first, then act
| Flavor | Signature | Fix |
|---|---|---|
| Arbitrage | Slow gradual PF drift over 12+ months | Wider universe, tighter cells |
| **Regulatory** | **Sharp drop on a specific dated cutover** | **Retire or rebuild for new regime** |
| Regime | Cyclical, reversible (e.g., war volatility) | Wait for regime to revert |

`delivery_pct` failed because we treated regulatory decay as if it were
arbitrage decay.

### 3. War volatility (Jan-Apr 2026) is a tailwind, not a stress test
Higher realized vol = larger wins on short setups (any volatility-favoring
mechanism wins). Holdout PF readings during this window over-estimated true
edge for fade setups. Cross-check by splitting Holdout into pre-war post-SEBI
(Oct-Dec 2025) and war-period (Jan-Apr 2026).

### 4. Fee model is load-bearing
MIS-leveraged PnL with non-leveraged fees flatters results. Apr 1, 2026 STT
hike (futures 0.02%→0.05%, options premium 0.1%→0.15%) further compresses
edge. Any new setup's break-even must be modeled with leverage-aware fees AND
post-Apr-2026 STT rates.

### 5. Mechanism > parameters
Cell-mining without a clearly-stated economic mechanism produces data-mined
artifacts. Every candidate below states its mechanism in one sentence. If you
can't, don't ship.

---

## Candidate Setups to Research

Each candidate is tagged with `regulatory_sensitivity` and `depends_on`. See
`data/sebi_calendar/README.md` for the dependency tag vocabulary.

### Candidate 1: Post-MWPL-Recalc Forced Rebalance Fade

**Mechanism:** SEBI recalculates MWPL quarterly using rolling 3-month cash
volumes. Stocks whose new MWPL is materially below current open interest must
see forced position cuts at recalc time, creating predictable downside drift
in the 5-10 trading days post-recalc.

**Direction:** SHORT post-recalc on names with current_OI / new_MWPL > 0.85.

**regulatory_sensitivity:** `rule_creating`

**depends_on:** `["MWPL", "single_stock_FO", "F&O_speculation"]`

**Data needed:**
- MWPL quarterly recalc dates (NSE publishes — first business day of each quarter)
- Historical FutEq OI (post-Oct 2025 — pre-Oct uses old methodology, not comparable)
- Current OI / new MWPL ratio per symbol

**Hypothesis falsifier:** if post-recalc stocks with high OI/MWPL ratio don't
show negative drift over 5-10 days at PF >= 1.20 on n >= 50 events, kill it.

**Sample availability:** quarterly cycle means ~3-4 events in Oct 25-Apr 26
window. Tight sample. May need to scrape historical MWPL changes from NSE
archives.

---

### Candidate 2: F&O Ban-Entry Reaction Pattern

**Mechanism:** Under new intraday FutEq OI monitoring (Nov 3, 2025), stocks
can enter ban intraday at any of 4 random snapshots. Once a stock enters ban,
fresh positions are blocked — only existing-position closures allowed. This
creates a one-way exit flow with no fresh buying, leading to predictable
downside drift in the next session.

**Direction:** SHORT next session after a stock enters ban list.

**regulatory_sensitivity:** `rule_creating`

**depends_on:** `["MWPL", "intraday_ban", "single_stock_FO", "F&O_speculation"]`

**Data needed:**
- NSE F&O ban list daily history (already partially scraped — `tools/asm_gsm_history/`)
- Per-stock entry/exit timestamps for intraday ban
- T+1 price action post-ban-entry

**Hypothesis falsifier:** if T+1 sessions post-ban-entry don't show negative
drift on at least 55% of events with median return < -0.5%, kill it.

**Sample availability:** intraday ban entries are common — ~5-15 per week
under new regime. Sample n=200+ achievable in Oct-Apr.

**Note:** This was retired in sub-9 round 1 (different signal — pre-rule).
Post-rule mechanism is fundamentally different (intraday monitoring vs end-of-
day). Worth re-investigating from scratch.

---

### Candidate 3: Single-Stock F&O Removal Pre-Announcement Drift

**Mechanism:** NSE publishes F&O eligibility changes quarterly. Stocks being
removed from F&O segment lose institutional hedge demand → forced cash-segment
unwind. The drift starts 2-3 days BEFORE the official announcement (rumor +
positioning) and continues 5-10 days after.

**Direction:** SHORT stocks rumored / confirmed for F&O removal.

**regulatory_sensitivity:** `rule_dependent`

**depends_on:** `["single_stock_FO", "F&O_speculation"]`

**Data needed:**
- NSE F&O addition/removal circulars (quarterly cycle, ~4-8 stocks per cycle)
- Pre-announcement price/volume in the 5 trading days before circular
- Post-announcement continuation pattern

**Hypothesis falsifier:** if removal-bound stocks don't show statistically
significant negative drift in T-3 to T+10 window, kill it.

**Sample availability:** Sep 2024 removed 16 stocks, Apr 2026 added 6.
Historical sample n=30-50 across 2023-2026. Smallish but workable.

---

### Candidate 4: Pre-Open Auction Edge Post-Option-Premium-Upfront

**Mechanism:** Feb 1, 2025 rule eliminated leverage on long options. Retail
option-buying flow that used to position pre-open (low-premium-cost speculative
calls) is now gone or much smaller. Pre-open auction depth and pattern may
have shifted predictably — opening gap behavior in retail-favorite names
should be more reliable post-rule.

**Direction:** Test both LONG and SHORT on first-15-minute follow-through
patterns in NIFTY-50 stocks.

**regulatory_sensitivity:** `rule_dependent`

**depends_on:** `["option_premium", "F&O_speculation"]`

**Data needed:**
- Pre-open auction prints (NSE publishes — already partially in
  `tools/pre_open_auction/`)
- 09:08 IPO price (final auction discovery price)
- 09:15-09:30 follow-through bar

**Hypothesis falsifier:** if pre-open auction direction + first 5m bar
direction don't predict next 60min with WR >= 60%, kill it.

**Sample availability:** every session × ~50 NIFTY stocks = ~7000 events in
Oct-Apr 2026. Very thick sample.

---

### Candidate 5: STT-Aware Fee-Adjusted Gap-Fade Re-Optimization

**Mechanism:** Apr 1, 2026 STT hike (futures +150%, options +50%) materially
shifts the break-even for fee-sensitive setups. Existing gap_fade_short was
locked at SL/target combinations validated under pre-Apr-2026 fee structure.
Re-running the sweep with post-Apr-1 fees should reveal a new optimum.

**Direction:** Same as gap_fade_short (SHORT small-cap gap-up exhaustion).
Just re-tune SL/T1/T2 for new fees.

**regulatory_sensitivity:** Recalibration of existing setup, not new mechanism.

**depends_on:** `["MIS_leverage", "STT_drag", "dpr_circuit"]`

**Data needed:** Already have. Just re-run the SL/target sweep in
`tools/sub9_research/_gap_fade_short_sl_target_sweep.py` with `is_post_apr_2026=True`
in the fee model.

**Hypothesis falsifier:** if no SL/target combo achieves PF >= 1.30 on
post-Apr-2026 data, gap_fade is below break-even under new STT and should
be sized down or retired.

**Sample availability:** unlimited (existing gap_fade dataset). Could finish
this in 1 day.

---

### Candidate 6: Lot-Size-Doubled Index-Option Pinning (NIFTY only)

**Mechanism:** Dec 26, 2024 NIFTY lot size 25 → 75 (3x). With 3x larger lots,
option writer positions are 3x larger gamma exposure. Expiry-day pinning
dynamics may have changed — pinning may be SHARPER (less retail noise, more
institutional precision) at the highest-OI strike. This is a long-vol setup
near expiry close.

**Direction:** Long ATM NIFTY straddles in the last hour of weekly expiry
when spot is pinning toward a high-OI strike.

**regulatory_sensitivity:** `rule_creating`

**depends_on:** `["index_options", "weekly_expiry", "option_premium"]`

**Data needed:**
- NIFTY weekly expiry option-chain snapshots (need OI by strike intraday)
- Historical pinning behavior pre- and post-Dec-26-2024

**Hypothesis falsifier:** if post-lot-size-doubling expiry-day pinning isn't
materially tighter (measured by intraday range of spot vs nearest
high-OI strike), kill it.

**Sample availability:** ~30 weekly NIFTY expiries Oct-Apr 2026. Thin per
event but high-frequency intraday data per event.

**Note:** options-trading setup — only viable if we add option-trading to the
broker integration. Currently the system trades cash equity SHORT only.

---

## Validation Methodology (Post-SEBI Aware)

For each candidate, the gauntlet is:

| Phase | Window | Sample | Purpose |
|---|---|---|---|
| **Discovery** | Oct 1, 2025 - Dec 31, 2025 | ~60 trading days, clean post-SEBI pre-war | Cell-mine + lock thresholds |
| **OOS** | Jan 1, 2026 - Apr 30, 2026 | ~80 trading days, war-volatility stress test | War-aware split: pre-war OOS PF and war PF reported separately |
| **Holdout** | May 14, 2026 onwards | Paper-trade live data | Forward-only test |

**Pre-registration requirements (per setup brief):**
1. Mechanism in one sentence
2. Direction (LONG / SHORT / both)
3. `depends_on` tags
4. Falsification thresholds (must declare BEFORE looking at data)
5. Minimum sample size for OOS
6. Kill criteria (rolling 30-day PF, WR, drawdown)

**Regime-break check (mandatory):** Before running any gauntlet, call
`services.regime_break_detector.check_window(...)` for each phase. If any
phase contains a high/critical rule change affecting this strategy's deps,
the gauntlet refuses to run.

---

## Execution Plan

### Phase 0 — Wait for paper-trade window (2026-06-14)
Don't start research execution until we have 30 days of fresh paper data
under the current regime. The paper data informs which candidates are worth
prioritizing.

### Phase 1 — Prioritize candidates (1 day after 2026-06-14)
Score candidates by:
- Mechanism clarity (1-5)
- Sample availability under post-SEBI window (1-5)
- Data scrape cost (1-5; 5 = data already there)
- Independence from F&O lot-size / option-premium changes (regime-decay risk)

Pick top 2 to develop briefs for.

### Phase 2 — Brief drafting per candidate (1 day each)
Use `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` as the
template. Mandatory sections: mechanism, direction, cells (pre-registered),
falsifiers, depends_on, sample needs.

### Phase 3 — Sanity check per candidate (3-5 days each)
Standalone Python script that simulates the setup on Discovery data only.
Must NOT touch OOS until sanity PF >= 1.30.

### Phase 4 — OOS validation (1-2 days each)
Run on Jan-Apr 2026 with war-period split. Each candidate must pass:
- Clean window (Oct-Dec 2025): PF >= 1.20
- War window (Jan-Apr 2026): PF >= 1.00 (war is tailwind, don't double-count)
- Combined: PF >= 1.15

### Phase 5 — Detector implementation (1 week each)
Standard sub-9 pattern: structure_type subclass, plan_long/plan_short_strategy,
unit tests, regime_break_detector pre-flight in tests.

### Phase 6 — Paper-trade as Holdout (60 days minimum)
Each new setup goes live in paper mode for 60+ days under live SEBI rules
+ post-war volatility regime before any real-capital commitment.

---

## Open Questions

- Should we expand broker integration to options trading? Candidate 6
  requires it. Currently the system is cash-equity SHORT only.
- Do we want to wait for Apr 1, 2026 STT hike data to mature before shipping
  any new fee-sensitive setup? (Some of Q1 2026 is pre-STT-hike from this
  perspective.)
- Is there enough sample size in post-SEBI window for cell-mining? May need
  to accept thinner samples per cell (n=50 instead of n=200).

---

## Files

- `specs/2026-05-14-research-post-sebi-edges.md` — this roadmap
- (future) `specs/2026-MM-DD-brief-<setup-name>.md` — per-candidate briefs
- (future) `tools/research/post_sebi/sanity_<setup>.py` — sanity scripts
- `data/sebi_calendar/rule_changes.csv` — regulatory calendar (use to validate windows)
- `services/regime_break_detector.py` — mandatory pre-flight for any gauntlet

## Cross-references

- Failed delivery_pct attribution: see commit `ship(paper): 3-setup portfolio with Cell B narrow delivery_pct` for the SEBI-Oct-1 root-cause analysis
- Regulatory calendar + regime-break detector: commit `feat(gauntlet): regulatory calendar + regime-break detector`
- Paper-trade portfolio: `config/configuration.json` `setups.{gap_fade,circuit_t1,delivery_pct}_*`
- Retired setups archive (negative knowledge): `docs/retired_setups.md`
