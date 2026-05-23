# `nifty_100_block_sell_afternoon_continuation_short` — Stage 0 brief (REFRAMED)

**Date:** 2026-05-22 (REFRAMED — original HF3 framing KILLED by Phase 1 Gate A as miscited-Varsity / cargo-cult Wyckoff)
**Stage:** 0 — Idea (awaiting Phase 1 re-run)
**Branch:** `research/2026-05-22-new-setups-batch`
**Predecessor:**
- Original HF3 brief framed as "sustained-above-VWAP + volume climax + red bar = Wyckoff distribution" → KILLED 2026-05-22 because (a) Zerodha Varsity Volume chapter does NOT discuss climax/distribution/Wyckoff (it covers price-volume tables + smart-money intro only); (b) Wyckoff is US-origin distribution literature (cargo-cult risk per §3.2 binding rules); (c) icfmindia.com refs were generic absorption/exhaustion language with no tri-combo operationalization.
- Reframed to a directly Indian-microstructure-anchored mechanism: SEBI/NSE regulated block-deal afternoon window (14:05-14:20 IST) where large negotiated SELL crosses on NIFTY 100 heavyweights leave residual institutional supply that continues working into 14:30-15:25.
- Related (different mechanic, different window): `nse_block_deal_counter_flow` brief (KILLED) tested **T+1 retail-FOMO counter-flow fade** on F&O 200 BIDIRECTIONAL (PF 0.74-0.79); the present setup is **same-day continuation** (NOT counter-flow), **SHORT-only on SELL** (NOT bidirectional), and **NIFTY 100 heavyweight cap-locked** (NOT F&O 200 large+mid).

**Direction:** SHORT
**Window:** Intraday MIS (square 15:25). Block-deal afternoon window closes 14:20 IST; signal evaluated at 14:30 5m bar close; entry at 14:30 close; exit at 15:25 (close of 15:20-15:25 bar) or SL.
**Portfolio rationale:** First **NIFTY 100 heavyweight SHORT** with **regulatory-event anchor** (NSE block-window timestamp). Distinct from existing SHORTs: `gap_fade_short` (small-cap, opening), `or_window_failure_fade_short` (mid-cap, 09:30-10:30), `circuit_t1_fade_short` (T+1 retail FOMO). Same regulatory-edge class as `circuit_t1_fade_short` (NSE/SEBI disclosure-anchored).

## 1. Mechanism statement (ONE sentence)

NIFTY 100 stocks (top-100 by market cap, F&O-eligible, large_cap segment) on which NSE publishes ≥1 BLOCK-DEAL SELL print of trade_value ≥ ₹25 cr (post-Dec-2025 SEBI minimum) in the afternoon block window (14:05-14:20 IST) — when the disclosed institutional seller's negotiated-window inventory is structurally larger than the ±1% (pre-Dec-2025) / ±3% (post-Dec-2025) price-band can absorb in 15 minutes, the residual supply is worked into the lit order book between 14:20 and 15:25 — SHORT at the 14:30 5m bar's close, exit at 15:25 (square pre-15:30) or SL, harvesting the iceberg-residual drift.

## 2. Falsifiers (3)

1. **Mechanism falsifier — block-window timestamp asymmetry:** The mechanism requires the SELL print to be in the AFTERNOON window (14:05-14:20). Control: same-symbol SELL prints in the MORNING window (08:45-09:00) should NOT produce the 14:30-15:25 drift (no time-adjacent residual). Test: across signal cohort, partition by morning vs afternoon block prints. Afternoon-cohort drift to 15:25 must be ≤ -0.20%; morning-cohort drift must be ≥ -0.10% (no residual-supply effect by 14:30). If both cohorts drift symmetrically, the mechanism is "block-deal SELL = bearish day in general" (already in `nse_block_deal_counter_flow` T+1 result, KILLED) → KILL.

   **DATA NOTE (per Gate B below):** On-disk `data/block_deals/block_deals_events.parquet` carries only `trade_date` (no intraday timestamp). Approximation: any block print of size ≥₹25 cr is by SEBI rule in one of the two windows. Proxy gate: control = "block-SELL date with no SAME-DATE block-BUY in NIFTY 100 within ±1 cap-bucket" vs full population. If approximation is too noisy, escalate to NSE block-deal scraper rebuild to capture session-time per row (pre-Phase 2).

2. **Same-day residual signature (institutional-flow tell):** Mechanism requires elevated 14:20-15:25 volume in the lit book vs same-stock 14:20-15:25 5-day baseline (the iceberg-residual signature). Test: across 200+ fires, signal-cohort `late_session_vol_ratio_vs_5day` (cumulative 14:20-15:25 / 5-day-same-window mean) median ≥ 1.3×. If median < 1.1× (no above-baseline lit-book flow), the residual is fully absorbed in the negotiated price and there is no continuation → KILL.

3. **Regime split — pre/post SEBI Dec-7-2025 block-deal reform:** SEBI revised block-deal framework took effect 2025-12-07: minimum size ₹10 cr → ₹25 cr, price band ±1% → ±3%. The wider price band lets the negotiated cross absorb MORE supply on the day, shrinking residual. **Mandatory split:** Pre-2025-12-07 cohort PF and post-2025-12-07 cohort PF must BOTH have lower-CI > 1.0 in their majority regime, OR per-regime PF ranking must be stable across regimes 1-7. Catastrophic post-reform PF degradation (post PF < 0.9, n_post ≥ 50) → KILL (mechanism died with the reform). Notional gate is set at ≥₹25 cr (the higher of the two regimes) to ensure consistent block-size selectivity.

## 3. Adjacent setups + correlation/effective-M

| Setup | Status | Direction | Mechanism overlap | M penalty |
|---|---|---|---|---|
| `nse_block_deal_counter_flow` | KILLED (Round-5, sub-9) | bidirectional | Same data source (block deals) but DIFFERENT direction (counter-flow vs continuation), DIFFERENT window (T+1 09:15-10:30 vs T+0 14:30-15:25), DIFFERENT universe (F&O 200 large+mid vs NIFTY 100 heavyweight only) | **0.2** (orthogonal trigger; rejected hypothesis being inverted, not replicated) |
| `circuit_t1_fade_short` | active | SHORT | Same regulatory-event class (disclosure-anchored), DIFFERENT event (DPR circuit vs block-deal), DIFFERENT window (T+1 morning vs T+0 afternoon) | 0.3 |
| `gap_fade_short` | active | SHORT | Different trigger (gap vs block-print), different window, different cap | 0.3 |
| `or_window_failure_fade_short` | active | SHORT | Different window, different trigger, different cap | 0.3 |
| `nifty_100_gap_up_low_volume_followthrough_fade_short` (sibling brief HF2) | Stage 0 | SHORT | Same universe + cap, very different trigger (gap-up-low-vol morning vs block-SELL afternoon). Mutually exclusive entry windows. | 0.4 (same universe but different trigger/window) |
| `nifty_100_sector_divergence_intraday_revert` (sibling brief) | Stage 0 | bidirectional | Same universe; different mechanism (relative-value); minor overlap | 0.3 |

**Effective M estimate:** 0.4 within batch (vs HF2 — same NIFTY 100 universe). Phase 5 confidence card Bonferroni at M=2+.

## 4. Phase 1 outline (Gate A + Gate B)

### Gate A — Indian-source operationalization (≥2 required) — PRE-VALIDATED IN THIS REFRAME

1. **NSE Member FAQ on Block Deal Window (April 2024)** — https://nsearchives.nseindia.com/web/sites/default/files/inline-files/FAQs_Block_Deal_Window.pdf — defines the afternoon window 14:05-14:20 IST, the VWAP reference computed from cash-segment trades 13:45-14:00, the dissemination window 14:00-14:05, the ±1% price band, and the 30-min mandatory post-disclosure rule. **This is the regulatory anchor for the 14:05-14:20 → 14:30-15:25 residual mechanism.**

2. **NSE-IIM Ahmedabad Research — "Price Impact of Block Trades and Price Behavior Surrounding Them" (NSE Proposal #216)** — https://nsearchives.nseindia.com/content/research/NSE_Proposal_216_Final_Paper.pdf — NSE-funded Indian academic study at IIM-A on block-trade price impact and post-event price behavior on the NSE. **This is the Indian-academic substrate for residual-supply continuation post-block-print on the Indian market** (NOT US Kraus-Stoll, NOT US Wyckoff).

3. **SEBI revised block-deal framework (effective 2025-12-07)** — https://www.medianama.com/2025/10/223-sebi-revamps-block-deal-norms-hikes-25-crore/ — minimum size ₹10cr → ₹25cr, price band ±1% → ±3%. **This is the regime-break that mandates our Falsifier #3 pre/post split** and confirms continued SEBI regulatory attention to the block-deal mechanism (regulatory persistence).

4. **Zerodha Varsity Volume chapter — "Smart Money" sub-section** — https://zerodha.com/varsity/chapter/volumes/ — defines smart-money detection via above-10-day-average volume signature and explicitly notes block/bulk deals are reported and **added back to lit-volume on the day** (this is the volume-trail observable that Falsifier #2 tests for). NOT citing this chapter for "climax/distribution" framing (which was the previous-brief miscite); citing it for the smart-money + block-deal-volume-attribution mechanic only.

**Gate A acceptance:** PASS — 3 Indian regulatory/academic sources + 1 Indian retail-broker source, all directly operationalizing the same-day block-SELL → residual-supply mechanism on NSE/SEBI infrastructure. No Wyckoff, no US sources, no cargo-cult.

### Gate B — Data feasibility

| Required data | On disk? | Notes |
|---|---|---|
| 5m bars per NIFTY 100 symbol (2023-01 → 2026-04) | ✅ | monthly/*_5m_enriched.feather |
| Block-deal events with date/symbol/side/notional | ✅ | `data/block_deals/block_deals_events.parquet` (8,596 rows, NSE+BSE, 2023-01-05 → 2026-04-29; 1,211 NSE SELL events; 627 events ≥₹50cr; 507 distinct (date, sym) pairs ≥₹50cr) |
| Block-deal **intraday timestamp** (morning vs afternoon window) | ❌ | Falsifier #1 requires this; current archive has only `trade_date`. Mitigation: rely on SEBI rule (any ≥₹25cr block is in the two windows), use control-cohort proxies, and budget a Phase 1.5 NSE-archive scraper enhancement to capture session-time if Phase 2 results warrant. |
| NIFTY 100 list | ⚠️ Shared with HF1+HF2; build planned |
| `ProductionUniverseGate` (Lesson #19) | ✅ |
| `consolidated_daily.feather` for 5-day late-session vol baseline | ✅ |

Predicted Gate B: **PASS** for Phase 2 with the timestamp-proxy compromise; **FULL PASS** only after a one-time scraper enhancement (deferred unless Phase 2 directional drift confirms).

## 5. Phase 2 plan (preview)

- **Universe:** NIFTY 100 (top-100 by market cap as of T-1), F&O-eligible, `cap_segment == 'large_cap'`, `ProductionUniverseGate` per-date.
- **Signal day qualifier (T+0, computed at 14:30):**
  - ≥1 NSE block-deal SELL print on (sym, T0) with `trade_value_cr ≥ 25.0` (post-reform floor applied universally for cohort comparability).
  - NO NSE block-deal BUY print on the same (sym, T0) with `trade_value_cr ≥ 25.0` (clean SELL-side; avoids two-sided crosses where the buyer is also size-committed).
  - At least one fire per (sym, date) — first-fire-per-day latch.
- **Pre-signal context (computed at 14:30 close):**
  - Intraday return at 14:30 ≤ +2.0% (exclude strong rallying names where late-session momentum dominates the residual-supply effect).
  - Late-session lit-book volume ratio: `lit_vol(14:20-14:30) / mean(lit_vol(14:20-14:30) over prior 5 trading days, same symbol)` ≥ 1.2× (early signature that residual is hitting the lit book — used as **trigger gate**, not just a falsifier check).
- **Entry:** SHORT at 14:30 5m bar's close (Mode A: signal bar's close).
- **Exit:**
  - HARD time-stop 15:25 (close of 15:20-15:25 bar) — pre-15:30 MIS square.
  - SL: `max(intraday_high_so_far, entry * 1.012)` (1.2% adverse).
  - T1: entry * 0.995 (0.5%), 50% partial, breakeven trail after.
- **Baseline (anchor):** same universe + same intraday-return ceiling but NO block-SELL print on T0; same 14:30 close entry; same 15:25 exit. Tests whether the block-print is the trigger or just a same-day artifact of "large-cap names that drifted down in the afternoon."
- **Falsifier #1 (block-window proxy):** Use SEBI rule that any ≥₹25cr block is in one of the two windows. Partition by whether the same (sym, T0) also has a morning-side news/gap event (proxy: `|gap_pct| > 1.0%` AND first-hour vol > 1.5× 5d-avg) — if YES, the block was probably executed in the morning window and the afternoon-residual mechanism doesn't apply; exclude. If NO, treat as afternoon-window candidate.
- **Falsifier #2 (residual signature):** Compute `late_session_vol_ratio = cum_vol(14:20-15:25) / mean(cum_vol(14:20-15:25) over prior 5 sessions)`. Across signal cohort, require median ≥ 1.3×.
- **Falsifier #3 (regime split):** Pre-2025-12-07 cohort PF vs post-2025-12-07 cohort PF. **Mandatory pre-registered split.** Per-regime (1-7) PF report.
- **Acceptance gates:**
  - n ≥ 200 (post all filters, Discovery window 2023-01-05 → 2025-06-30)
  - Net PF ≥ 1.20 (post-fee; ≥ 1.30 for STRONG PROCEED)
  - Per-month winning months ≥ 55%
  - Falsifier #2 median residual-vol-ratio ≥ 1.3× (else mechanism not confirmed)
  - Falsifier #3 pre/post both PF ≥ 1.0 OR mechanism formally retired post-reform
- **Required splits:** pre/post 2025-12-07 (SEBI reform), notional buckets (₹25-50 cr, ₹50-150 cr, ≥₹150 cr), regime 1-7, intraday-return buckets at 14:30 ([-2,0]%, [0,1]%, [1,2]%).

## 6. Status checklist

- [x] Gate A — ≥2 genuinely Indian sources operationalized (NSE FAQ + NSE-IIM-A NSE Proposal #216 + SEBI Dec-2025 reform + Zerodha Varsity volume — **NO Wyckoff, NO US sources**)
- [ ] Gate B — NIFTY 100 list built (shared with HF1, HF2)
- [ ] Falsifier #1 (block-window proxy) pre-registered (with scraper-enhancement note)
- [ ] Falsifier #2 (late-session lit-vol ≥ 1.3× residual signature) pre-registered
- [ ] Falsifier #3 (pre/post 2025-12-07 SEBI reform mandatory split) pre-registered
- [ ] Within-batch M=2 acknowledged vs HF2 + sector-divergence sibling

## 7. Next action

1. Phase 1 re-run by an agent: confirm Gate A acceptance with this reframed source pack; verify NIFTY 100 list build (shared infra with HF1, HF2).
2. Build sanity script `tools/sub9_research/sanity_block_sell_afternoon_continuation_short.py` using existing `data/block_deals/block_deals_events.parquet` + 5m feathers + ProductionUniverseGate.
3. Phase 2 dispatch with the pre-registered acceptance gates above.

## Appendix — Novelty check (Chartink/Streak retail-arb risk)

Searched standard Indian retail scanner directories (Chartink, Streak, Tickertape) for:
- "block deal SHORT 14:30 same day continuation"
- "afternoon block sell intraday short"
- "block deal residual supply scanner"

Result: **NOT in standard retail scanners.** Public retail tooling for block deals is uniformly framed as T+1 "follow smart money" (BUY-side LONG / SELL-side LONG-fade) — Tickertape "Block & Bulk Deals", Trendlyne "Smart Money Tracker", Moneycontrol "Bulk & Block Deals". None operationalize same-day afternoon-window SHORT-continuation on NIFTY 100 heavyweights with a residual-volume gate. **Edge preserved.** The retail-vs-our-side asymmetry is: retail interprets a published block-SELL as "smart money bearish" but enters T+1 at the open (after the residual has finished working into close T+0); we enter at 14:30 T+0 and exit by 15:25, harvesting the residual T+0 leg they trade against the next morning.
