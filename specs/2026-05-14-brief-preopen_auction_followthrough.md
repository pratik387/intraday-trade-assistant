# Brief: `preopen_auction_followthrough`

**Research branch:** `research/post-sebi-edge-setups`
**Roadmap parent:** `specs/2026-05-14-research-post-sebi-edges.md` (Candidate 4)
**Status:** **DATA-BLOCKED — RETIRE / DEFER until forward-only IEP collection accumulates ~12 months (~2027-05).** See §11.
**Date:** 2026-05-14
**Predecessors:**
- `specs/2026-05-07-sub-project-9-brief-pre_open_auction_imbalance_fade.md` — sister brief (RETIRED 2026-05-07: mechanism-falsified + data-blocked)
- `specs/2026-05-07-pre_open_url_discovery.md` — exhaustive 30-URL probe showing NSE archive endpoint dead
- `specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md` — template (APPROVED §3.3 brief structure)

This is **Candidate 4** of the post-SEBI-Oct-1-2025 research roadmap. The roadmap framing differs from the May-7 sister brief: instead of fading the IEP-vs-09:15-open divergence (mechanism-falsified — the 09:15 print IS the final IEP by construction), this brief asks whether the **Feb 1, 2025 option-premium-upfront rule** has materially shifted **first-15-minute follow-through** patterns in NIFTY-50 retail-favorite names, such that the post-09:15 5m bars now carry exploitable mean-reversion or continuation signal post-rule that was absent / weaker pre-rule.

---

## 1. Mechanism (one sentence)

The Feb 1, 2025 upfront-option-premium rule eliminated retail's pre-open positioning in low-premium speculative calls (the "₹500 lottery-ticket" trade), removing a structural buy-imbalance at the open in NIFTY-50 retail-favorite names — the 09:15-09:30 follow-through pattern should now be more reliable (less retail noise, more institutional/algo-dominated price action), making the **direction of the first 5m bar a stronger predictor of the 09:15-10:15 hour** post-rule than pre-rule.

## 2. Direction (declared after considering both LONG and SHORT)

**Bidirectional with asymmetric thresholds, sign-locked to first-5m bar direction. Pre-registered direction-specific declaration:**

- **LONG side:** `first_5m_bar.close > first_5m_bar.open` AND `first_5m_bar.body_pct >= 0.30` → enter LONG at 09:20 close, expect 09:20-10:15 follow-through to upside.
- **SHORT side:** `first_5m_bar.close < first_5m_bar.open` AND `first_5m_bar.body_pct >= 0.30` → enter SHORT at 09:20 close, expect 09:20-10:15 follow-through to downside.

**Project bias note (asymmetric gates):** Per SEBI FY23 retail-loss study + this project's own sub-7/8 record, **LONG-bias setups have systematically lost on Indian intraday.** All 11 long-side asymmetric explorations in sub-7/8 underperformed gap_fade SHORT. The roadmap allows testing both directions in research, but the **LONG-side falsifier bar is set 5pp WR higher than SHORT-side** (see §6) to compensate for known long-bias loss asymmetry, and per-side metrics must be reported separately. **Default expected outcome (pre-registered): SHORT side passes, LONG side fails — but both are tested for completeness.**

**Implicit baseline test:** the "no-signal" cell (first_5m bar body < 0.30) is recorded but not traded; it serves as a baseline for the cells above.

## 3. Regulatory dependencies

- **regulatory_sensitivity:** `rule_dependent`
- **depends_on:** `["option_premium", "F&O_speculation", "MIS_leverage"]`
- **reasoning:** The setup's mechanism is created by the Feb 1, 2025 option-premium-upfront rule eliminating cheap retail option-buying. Without that rule, retail option-buying flow contaminates pre-open and first-15-min price discovery in NIFTY-50 names. The rule is in the SEBI calendar (`data/sebi_calendar/rule_changes.csv`, row 2025-02-01, severity=high). The setup mechanism is sensitive to:
  - Any rollback / amendment of the Feb 1 2025 upfront-premium rule (would re-introduce retail option-buying noise) → retire.
  - Apr 1, 2026 STT hike on options premium 0.1%→0.15% (further reduces retail option-buying viability) → tailwind (mechanism gets stronger post-Apr 2026), but fee-aware backtest required.
  - MIS leverage changes affecting NIFTY-50 cash trading → re-validate.
- **Regime-break detector pre-flight (mandatory per roadmap §2):**
  - Discovery window (Oct 1 2025 – Dec 31 2025): clean — no rule changes in this window affecting these deps.
  - OOS window (Jan 1 2026 – Apr 30 2026): **contains 2026-04-01 STT hike (critical severity, affects `STT_drag;option_premium;F&O_speculation`)**. War-period split + post-Apr-2026 fee-model recompute REQUIRED. Split OOS into pre-STT-hike (Jan-Mar 2026) and post-STT-hike (Apr 2026 onwards) cells.
  - Holdout (paper, May 2026 onwards): clean.

## 4. Pre-open auction mechanics (educational)

NSE runs a 15-minute pre-open call auction in the equity cash segment, structured per the NSE master circular (Aug 2024) and the public NSE pre-open product page:

| Window | Activity |
|---|---|
| **09:00 – 09:07/09:08 IST** | **Order entry phase.** Limit, market, and "at-market" (ATO) orders accepted; modification/cancellation allowed. NSE applies a **random system-driven closure between the 7th and 8th minute** (anti-gaming measure introduced by NSE Circular 23232, 2013). |
| **09:08 – 09:12 IST** | **Order matching / price discovery phase.** No new orders. NSE's matching engine computes the **Indicative Equilibrium Price (IEP)** — the single price that maximises total tradable quantity (max-volume rule), with tie-breakers on minimum order imbalance and reference-price proximity. IEP is published every few seconds on the NSE feed. |
| **09:12 – 09:15 IST** | **Buffer phase.** Market transitions from auction to continuous trading. The final IEP at 09:14:55-ish is **locked**; that price becomes the 09:15:00 official open. |
| **09:15:00 IST** | **Continuous market opens.** The first continuous-trading print IS the locked IEP from 09:14:55 (by NSE rule — the auction settles AT the final IEP, and the auction trade is the first 09:15 print). |

**Critical mechanic finding (from the May-7 sister-brief retirement):** the auction's matched-quantity book is settled AT the final IEP and that price becomes the 09:15 open. The two prices are **the same number by NSE rule** — any "divergence to fade" is structurally near-zero on liquid names because the order book freezes at 09:14:30. This kills the May-7 brief's "fade the IEP-vs-open divergence" thesis.

**The roadmap's Candidate 4 is a different question:** does the **directional information** in the auction (IEP relative to PDC, or the first 5m bar's body) predict the **next 60 minutes** more reliably *post-Feb-2025* than *pre-Feb-2025*, because retail option-buying noise has been removed? This question is independent of the IEP-vs-open mismatch and is NOT mechanically falsified.

**Academic anchor:** Agarwalla, Jacob, Varma (IIM-A, 2014) "Pre-Open Auctions in Indian Equity Markets" — documents auction over-reaction in illiquid stocks with mean-reversion in continuous trading. Comerton-Forde & Rydge (2006, *J. Financial Markets*) — call-auction imbalance manipulation is mean-reverting once continuous trading begins. These support a follow-through / fade signal, but neither tests the Feb-2025-Indian-retail-flow regime explicitly.

## 5. Existing infra audit (`tools/pre_open_auction/`)

**Files (verified May 14, 2026):**
- `tools/pre_open_auction/__init__.py` — empty package marker.
- `tools/pre_open_auction/fetch_pre_open.py` (28181 bytes, dated 2026-05-07) — historical NSE pre-open archive scraper. Targets `archives.nseindia.com/archives/equities/preopen/{MA,eq,op}{DDMMYYYY}.csv` (and nsearchives mirrors). Implements polite cookie bootstrap, 5-sec sleep cadence, exponential backoff, CSV alias-driven parser, parquet write to `data/pre_open_auction/pre_open_events.parquet`. **Also exposes a `--use-live-api` flag that snapshots `https://www.nseindia.com/api/market-data-pre-open?key=ALL` — the only T-0 path that works.**
- `tools/pre_open_auction/verify_pre_open.py` (17321 bytes, dated 2026-05-07) — validation harness that joins `actual_open` from 5m feathers, computes `divergence_pct`, writes per-quarter coverage stats + top-20 most-divergent cells + 5-event spot-validation URL set to a markdown report.

**Captured data (verified May 14, 2026):**
- `data/pre_open_auction/_backfill.log` — shows the backfill aborted on Day 25 of 818 trading days (range 2023-01-01 → 2026-04-30). **rows=0 across every probed session. last_status=404 on every URL.**
- **`data/pre_open_auction/pre_open_events.parquet` does not exist on disk** (the scraper writes the parquet only if at least one row collected; zero rows = no file).
- **`data/pre_open_iep/` directory does NOT exist** (referenced in the May-7 sister brief but never populated).

**Format usability for backtesting:** the scraper is **production-quality** (cookie bootstrap, retry, alias-driven CSV parser, parquet persistence, dedupe-on-(date,symbol), incremental merge). If a working URL pattern existed, the scraper would Just Work. **But it does not.** Per `specs/2026-05-07-pre_open_url_discovery.md`:

1. NSE archive URLs (all 11 patterns probed): **404 on every trading-day-and-host combination**. Confirmed structural (NSE retired the public preopen archive sometime between 2023-12 and 2024-04 with no replacement).
2. The PR daily ZIP at `/archives/equities/bhavcopy/pr/PR<DDMMYY>.zip` (alive) bundles 17 file types per its Readme.txt; **none are the preopen IEP report.** The `Pr` and `Pd` CSVs carry `OPEN_PRICE` (09:15 continuous open) but not the IEP.
3. NSE live API `/api/market-data-pre-open?key=ALL`: live and healthy, returns full schema (`IEP, totalBuyQuantity, totalSellQuantity, atoBuyQty, atoSellQty, finalQuantity`) for ~2055 symbols per day. **Strictly T-0 / live only — `?date=` parameter is silently ignored.** Caches the in-session capture and serves it through the rest of the day.
4. Local 1m feathers (`backtest-cache-download/monthly/<YYYY>_<MM>_1m.feather`): earliest bar per session is **exactly 09:15:00 IST**. Zero pre-09:15 bars across 26.6k symbol-days probed.
5. Upstox V3 historical API: same constraint — earliest candle 09:15:00 IST, no pre-open data.
6. Kite/Zerodha: `quote()` returns running IEP during 09:00-09:15 LIVE; no historical IEP endpoint.

**Verdict on existing infra:** scaffolding is ready; **data is not.** The scraper would unblock if NSE re-exposed the archive (unlikely). Otherwise, the **only forward path is a T-0 cron at 09:14 IST** writing today's snapshot to `data/pre_open_iep/<YYYY>/<MM>/<YYYYMMDD>.parquet` via the `--use-live-api` path of the existing scraper.

## 6. Cells (pre-registered, must NOT change after seeing data)

**Universe:** NIFTY-50 constituents as of session-start.
**Time grid:** every NSE trading day, Oct 1 2025 – Apr 30 2026 (~140 trading days = ~7000 symbol-day events).

**Trigger evaluation:** at 09:20:00 IST (close of first 5m bar 09:15-09:20).

**Cell definitions (declared BEFORE looking at data):**

| Cell ID | Trigger condition | Direction | Expected n (Oct-Apr) |
|---|---|---|---|
| `C1_short_strong` | first_5m_bar: close < open AND body_pct ≥ 0.50 AND volume ≥ 1.2× 20d-median first-5m volume | SHORT | ~600 |
| `C2_short_moderate` | first_5m_bar: close < open AND 0.30 ≤ body_pct < 0.50 | SHORT | ~900 |
| `C3_long_strong` | first_5m_bar: close > open AND body_pct ≥ 0.50 AND volume ≥ 1.2× 20d-median first-5m volume | LONG | ~600 |
| `C4_long_moderate` | first_5m_bar: close > open AND 0.30 ≤ body_pct < 0.50 | LONG | ~900 |
| `C5_baseline_noise` | first_5m_bar: body_pct < 0.30 (doji-ish) | NONE (baseline) | ~2000 |

**Definitions:**
- `body_pct = abs(close - open) / open * 100` (in pct of open).
- `first_5m volume 20d-median` = median over prior 20 trading days of the same symbol's 09:15-09:20 5m bar volume.

**Per-cell mechanic (LONG and SHORT cells):**
1. **Entry:** market order at 09:20:00 close (i.e., open of the 09:20-09:25 bar).
2. **Stop-loss:**
   - SHORT: `stop = first_5m_bar.high + 0.20 × ATR_5m` (ATR computed from prior 20 trading days' first three 5m bars).
   - LONG: `stop = first_5m_bar.low - 0.20 × ATR_5m`.
   - Min stop distance: 0.25% of entry.
3. **Target:** `T1 = entry ± 1.5R` (50% qty), `T2 = entry ± 3.0R` (50% qty), where `R = abs(entry - stop)`.
4. **Time stop:** **10:15 IST hard exit** (60 min hold; matches first-hour follow-through hypothesis).
5. **Latch:** one fire per (symbol, session_date).

**Splits to report:**
- Per-cell metrics (PF, WR, Sharpe, n) — 5 cells.
- Pre-war vs war-period split for OOS: Discovery (Oct-Dec 2025) vs OOS-pre-war (Jan 2026) vs OOS-war (Feb-Apr 2026).
- Pre-STT-hike vs post-STT-hike split: Jan-Mar 2026 vs Apr 2026 onwards (Apr 1 2026 STT hike).
- IEP-relative-to-PDC sign × first-5m-bar-direction interaction matrix (4 quadrants).

**These cells, thresholds, exits, and splits are LOCKED at brief approval. Any change after seeing data = invalidated brief.**

## 7. Falsification thresholds (pre-registered)

**Locked thresholds (must declare BEFORE looking at data):**

**A. Sample-size feasibility (HARD gate):**
- `n_C1 + n_C2 ≥ 300` SHORT trades total across Oct-Apr.
- `n_C3 + n_C4 ≥ 300` LONG trades total across Oct-Apr.
- Below either → kill the side; bidirectional becomes unilateral or full retire.

**B. Per-side PF thresholds (post-fee, post-STT-hike fee model):**
- SHORT side combined (C1+C2): **NET PF ≥ 1.20** on Discovery (Oct-Dec 2025).
- LONG side combined (C3+C4): **NET PF ≥ 1.25** on Discovery (Oct-Dec 2025). [Higher bar per long-bias-loses-on-Indian-intraday lesson.]

**C. WR + Sharpe gates:**
- SHORT side: WR ≥ 38%, daily Sharpe ≥ 0 on Discovery.
- LONG side: WR ≥ 43%, daily Sharpe ≥ 0 on Discovery. [+5pp WR vs SHORT per long-bias correction.]

**D. OOS gates (must hold on Jan-Apr 2026):**
- Clean window OOS (Jan-Mar 2026): PF degrade ≤ 20% relative to Discovery (per side).
- War-period OOS (Feb-Apr 2026): PF ≥ 1.00 (war is tailwind for fades; cannot double-count).
- Post-STT-hike (Apr 2026 only): PF ≥ 1.10 with post-Apr-2026 STT fee model.

**E. Roadmap hypothesis-specific falsifier (load-bearing):**
- **`pre_open_auction_direction + first_5m_bar_direction` must predict next-60min direction with WR ≥ 60%** in at least one of the four cells (C1/C2/C3/C4). Per the roadmap: "if pre-open auction direction + first 5m bar direction don't predict next 60min with WR >= 60%, kill it." This is the roadmap's explicit kill criterion and supersedes the per-side PF gates above when in conflict.

**F. Implicit-regime-shift gate (the unique-to-Candidate-4 falsifier):**
- The mechanism's claim is that the **Feb 1 2025 rule made the signal more reliable**. To validate this, the sanity must compare:
  - Same cells, same mechanic, on a **pre-Feb-2025 sample** (Aug-Dec 2024, ~80 trading days, before the rule landed).
  - Same cells on a **post-Feb-2025 sample** (Mar-Sep 2025, ~140 trading days, post-rule but pre-Oct-2025-MWPL-rule).
  - Difference in per-cell PF (post − pre) must be ≥ +0.15 absolute on at least 2 of 4 traded cells.
  - **If pre-Feb-2025 PF ≥ post-Feb-2025 PF, the mechanism claim is false and the candidate retires regardless of post-SEBI PF passing.** This is the load-bearing test of the brief's mechanism.

**G. Decay signal:** rolling 30-trade NET PF drops below 1.05 for ≥ 30 calendar days post-paper-launch → halt.

## 8. Data requirements

**On-disk (CONFIRMED present, May 14 2026):**
- 5m enriched feathers: `backtest-cache-download/monthly/<YYYY>_<MM>_5m_enriched.feather` — schema includes date, symbol, OHLC, volume, vwap, ATR-related fields. Earliest bar per session = 09:15:00. Covers all NIFTY-50 names from 2023-01 to 2026-04.
- NIFTY-50 constituents: presumed available via `assets/nse_all.json` or similar (needs verification).
- SEBI calendar: `data/sebi_calendar/rule_changes.csv` (16 rows including 2025-02-01 high-severity row for `option_premium;F&O_speculation;MIS_leverage`).
- Regime-break detector: `services/regime_break_detector.py` (referenced in the roadmap; needs pre-flight wiring per §3).

**On-disk (PARTIALLY present, scaffolding only):**
- `tools/pre_open_auction/fetch_pre_open.py` + `verify_pre_open.py` — production-quality, but the archive endpoint is dead.

**MISSING (load-bearing for the IEP-direction quadrant analysis):**
- **`data/pre_open_iep/<YYYY>/<MM>/<YYYYMMDD>.parquet`** — does not exist. The IEP-relative-to-PDC sign is one half of the load-bearing direction-pair the roadmap requires (§6 cell defs alone use only first_5m_bar direction, but the roadmap's falsifier §7E references the **(IEP, first_5m_bar) pair**).
- **Pre-Feb-2025 IEP history** for the implicit-regime-shift gate §7F: requires Aug 2024 – Jan 2025 IEP archive. **Not available from any source** (see §5 verdict on existing infra and the May-7 URL discovery doc).

**Forward-only collection cost:** ~4 engineering hours to schedule a daily 09:14:30 IST cron hitting `--use-live-api` and parquet-writing to `data/pre_open_iep/`. Yields ~9000 rows/day × ~250 trading days/year ≈ 2.25M rows/year at ~50 KB compressed/day.

**Wait time for usable backtest sample (post-cron-start):** ~12 months minimum (to get a Discovery + OOS split of the proper post-SEBI-Oct-1-2025 window with IEP attached). If forward collection started today (May 14, 2026), the minimum-viable-Discovery date would be **May 14 2027**.

## 9. Implementation sketch

### Timing challenge: 09:08 auction print → 09:15 entry decision

**The hard timing question:** can our existing screener_live act on the 09:08 auction discovery price BEFORE the 09:15 first continuous bar opens?

**Answer: NO — and it doesn't matter for THIS brief's mechanic, BUT it does matter for a related "front-run the open" variant.**

Detail:

1. **Production screener-live timing (verified):**
   - `services/market_hours_manager.py` line 84: market-open hardcoded to `09:15`. `is_market_open()` returns False before 09:15.
   - `services/screener_live.py::_timer_scan_loop` line 1045: `market_open = dtime(9, 20)` — the first scan slot fires at 09:20 IST (after the 09:15-09:20 5m bar closes). No scan slot exists between 09:00 and 09:20.
   - `services/ingest/live_tick_handler.py::on_tick` lines 122-125: explicit `_clear_pre_market()` on the first tick at/after 09:15 IST. **Any tick received before 09:15 is dropped** (not aggregated into the bar store; LTP is kept but bars are wiped at the 09:15 boundary).
   - **There is NO hook in the live pipeline for pre-09:15 IEP ingestion.** The system is designed assuming the auction is opaque and the market starts at 09:15.

2. **Implication for THIS brief's mechanic (§6):**
   - The trigger evaluation is at **09:20:00 IST close of first 5m bar** — well after 09:15. The scan slot at 09:20 (production timer) is exactly when this brief's trigger fires. **No new hook is needed for the §6 mechanic.**
   - The IEP value (if collected forward-going) only needs to be present in a sidecar parquet by 09:19:55 IST. The cron at 09:14:30 IST writing to `data/pre_open_iep/<YYYY-MM-DD>.parquet` gives a 5-minute buffer — comfortable.

3. **Implication for a "09:08-front-run" variant (NOT this brief, but worth a parking-lot note):**
   - If a future brief wanted to enter at **09:14:55 IST (auction lock) or 09:15:00:00X IST (first continuous tick)**, the system would need:
     - A new pre-market scheduler running 09:00-09:15.
     - WebSocket subscription started by 09:13 to capture the first tick (or the existing tick handler's pre-09:15 drop-rule disabled for the IEP-fetcher path).
     - Order routing capable of placing orders **before** 09:15 (Zerodha/Upstox accept pre-open orders 09:00-09:08 only — the order entry phase — but they MUST be price-limit orders that participate in the auction itself, not a continuous order at 09:14:59).
   - This is materially harder than the §6 mechanic. **Out of scope for this brief.**

**Implementation flow for the §6 mechanic (assumes forward-only IEP cron is running):**

```
[T-1 EOD: 09:14:30 IST cron]
  → tools/pre_open_auction/fetch_pre_open.py --use-live-api
    writes data/pre_open_iep/{YYYY}/{MM}/{YYYYMMDD}.parquet
    schema: [session_date, symbol, iep, feq, atoBuyQty, atoSellQty, finalQuantity, imbalance_qty]

[Session start: 09:15 IST]
  → screener_live.py initialises NIFTY-50 universe
  → pre_load IEP sidecar for today (lazy-load on first 09:15 bar arrival)
  → setup-universe builder: NIFTY-50 stocks with IEP present + 20d-median first-5m volume baseline

[09:20:00 IST: first scan slot]
  → bar_scheduler.schedule_admits gets PreopenFollowthroughStructure.detect() plan list
  → detector evaluates each NIFTY-50 symbol:
    - read first_5m_bar (09:15-09:20)
    - read pdc, iep (sidecar), first_5m_bar 20d volume baseline
    - apply C1/C2/C3/C4 cell triggers
    - emit plan with strategy=preopen_auction_followthrough_{long,short}, priority from config
  → bar_scheduler admits subject to capital + risk
  → executor fires orders at 09:20:00 close (i.e., into the 09:20-09:25 bar)

[Hold horizon: 09:20 → 10:15 IST]
  → standard exit-engine: hard SL, T1 (1.5R), T2 (3.0R), time-stop at 10:15

[Detector + plan + tests]
  → structures/preopen_auction_followthrough_structure.py
  → services/plan_long/plan_short_preopen_auction_followthrough.py (or unified plan generator)
  → tests/structures/test_preopen_auction_followthrough.py
  → regime_break_detector pre-flight: refuse if Discovery/OOS straddles 2025-02-01 or 2026-04-01 unsplit
```

**Effort estimate (assuming IEP data exists):**
- Forward-only IEP cron deployment: ~4 hours (scaffolding already in `fetch_pre_open.py::get_live_api`).
- Detector + plan + tests: ~3 engineering days.
- Sanity script (parallel to `tools/sub9_research/sanity_circuit_t1_fade_short.py`): ~2 engineering days.
- **But: zero engineering work is justified before the IEP data accumulates ~12 months of forward history.** Without IEP data, only the body_pct-based cells §6 work, and the load-bearing §7E falsifier (60% WR requires the (IEP, bar) PAIR) cannot be evaluated.

## 10. Risks / open questions

**Risk: candidate is structurally data-blocked.** The IEP history needed for the load-bearing §7E and §7F falsifiers does not exist on any historical source. Forward-only collection requires a 12-month wait minimum. This makes the candidate **DEFER-ineligible-now**, not *retire-permanent*, but the wait is real.

**Risk: the body_pct-only variant collapses onto first_hour_momentum / gap_fade_short.**
- `gap_fade_short` (TRUSTED, SHORT, 09:15-09:30 small/micro-cap) triggers on `(open − pdc) / pdc ≥ +1.5%` exhaustion candle. Its universe is small/micro-cap; this brief's universe is NIFTY-50. Trigger universe overlap on the SHORT side: bounded by universe disjointness — large_cap in NIFTY-50 vs small/micro-cap in gap_fade. Expected overlap < 5%.
- `first_hour_momentum` (sub-7 brief from May-8, status unknown) — likely overlaps if it also uses first 5m bar direction. **Must check overlap at sanity-check stage.**
- `capitulation_long_morning` (round-6 LONG-side mirror of gap_fade) — different universe (broad equity), different sign convention; expected overlap with C3/C4 LONG cells: < 10%.

**Risk: NIFTY-50 universe has structural reasons signal won't work that small-caps don't share.**
- NIFTY-50 names have heavy institutional flow that smooths microstructure noise. The "less noisy post-Feb-2025" hypothesis assumes retail option-buying flow was a meaningful noise source in NIFTY-50 pre-Feb-2025. **Is that true?** The Feb-1-2025 rule affected option premium upfront — relevant to options trading, not directly to cash NIFTY-50 stocks. The hypothesis chain assumes:
  - retail bought cheap NIFTY-50 calls pre-9:15 → broker hedged → hedge flowed into cash NIFTY-50 pre-open → cash NIFTY-50 09:15 open had retail-hedge-driven imbalance.
  - This is a 4-step inferential chain. **Each step has alternative explanations** (e.g., retail buys NIFTY index calls — not single-stock calls — so the hedge flow goes to NIFTY futures, not NIFTY-50 stocks individually). The mechanism may be weaker than the roadmap text suggests.

**Risk: 6000-event sample drops to <300 per traded cell after universe + cell filters.**
- Roadmap says 50 stocks × 120 days = 6000 events. After C1/C2/C3/C4 cell filters (~30% trigger rate based on first_5m_bar body_pct ≥ 0.30 distribution), expected ≈ 1800 trades total across all four traded cells. Per-cell sample sizes:
  - C1 (short strong): ~300 (HARD MARGINAL — at the §7A 300/side floor for SHORT total but the strong cell alone may be ~300, not 600 as listed).
  - C3 (long strong): ~300 (same caveat — see §7A long total floor).
  - These per-cell sample sizes are at the edge of statistical viability. **Cell-internal variance from sub-300-sample noise may dominate the regime-shift signal §7F is meant to test.**

**Risk: Apr 1 2026 STT hike contaminates OOS.**
- The roadmap's OOS window Jan-Apr 2026 spans the 2026-04-01 STT hike (critical severity). The regime_break_detector should refuse to run a single OOS window across this boundary. **Required mitigation:** split OOS into pre-STT-hike (Jan-Mar 2026) and post-STT-hike (Apr 2026 onwards). Report PF separately for each split.

**Risk: war volatility (Jan-Apr 2026) inflates fade-side OOS PF artificially.**
- Per the roadmap: "war is tailwind for fades; cannot double-count." Specific gate: post-war PF must be ≥ 1.00 on the war-period subset; if PF only passes due to war, the setup hasn't shown post-war edge.

**Open question 1:** is forward-only IEP collection worth scheduling NOW, with the candidate parked for ~12 months until data accumulates? **Recommendation: SCHEDULE THE CRON (4-hour cost) but DO NOT BUILD THE DETECTOR.** The cron preserves optionality at near-zero engineering cost; the detector cannot be sanity-checked without ~12 months of data anyway.

**Open question 2:** does the brief's mechanism actually require IEP at all, or can the body_pct-only cells C1-C4 stand alone? **Recommendation: NO — without IEP, the falsifier §7E (the load-bearing "60% WR on (IEP, bar) pair" gate) cannot be tested, and without §7E the brief becomes "a regression on first 5m bar direction" which is mechanically near-identical to first_hour_momentum and unlikely to show incremental edge.**

**Open question 3:** if the body_pct-only cells were retroactively backtested on existing 5m feathers (Oct 2025 – Apr 2026, ~140 trading days, NIFTY-50, ~7000 events), would the SHORT-side numbers pass §7B-C? **Recommendation:** this would be a **scope-violation** of the brief (the brief's mechanism rests on the IEP+bar pair, not the bar alone) — running it would be exploratory and any pass would constitute data-snooping toward a setup the brief never authorised. Resist the temptation.

**Open question 4:** should the brief be re-framed entirely as "post-Feb-2025 NIFTY-50 first-15-min-bar direction follow-through (cash-only, no IEP dependency)"? This would be a different brief — different mechanism (no longer about pre-open auction; just first-bar continuation in a regime where retail noise is reduced), different falsifiers (no §7E/F), different name. **Recommendation: this is a separate candidate worth its own brief — write it as a sibling candidate, not as a redefinition of Candidate 4. Mechanism integrity > convenience.**

## 11. Decision (per roadmap §3.3 + §11)

**DEFER, do not build, do not retire-permanent.**

Rationale:
1. **No historical IEP data exists.** All paths (NSE archive, broker historical, local feathers) probed and confirmed dead by `specs/2026-05-07-pre_open_url_discovery.md`.
2. **Forward-only collection requires ~12 months of wait** before the brief's §7E + §7F falsifiers can be evaluated.
3. **The body_pct-only variant is a different brief** and collapses onto first_hour_momentum / gap_fade_short axes — not an incremental edge worth pursuing.
4. **A 4-hour cron deployment preserves optionality at near-zero cost.** Schedule the forward-only IEP cron now; revisit Candidate 4 in ~2027-05 once 12 months of IEP data have accumulated.

**Action items (no-decision-required actions):**
1. **Optional now (~4 hours):** schedule a daily 09:14:30 IST cron running `tools/pre_open_auction/fetch_pre_open.py --start <today> --end <today> --use-live-api` writing to `data/pre_open_iep/<YYYY>/<MM>/<YYYYMMDD>.parquet`. Cost: ~50KB/day disk; preserves the option to revisit in 2027.
2. **Parking-lot for ~2027-05:** revisit this brief with 12+ months of IEP data. Re-evaluate the post-Feb-2025-vs-pre-Feb-2025 regime-shift claim §7F directly with primary data once Oct-2025-Sep-2026 OOS sample exists.

**Action items (require user decision):**
- [ ] **APPROVE-DEFER:** schedule the IEP cron and park the brief until ~2027-05.
- [ ] **APPROVE-NO-CRON:** park the brief; do not schedule the cron either (accept that the candidate may never be re-investigated).
- [ ] **REJECT:** retire the candidate permanently and remove the `tools/pre_open_auction/` scaffolding.
- [ ] **REVISE:** re-scope to a body_pct-only NIFTY-50 first-bar-follow-through brief (no IEP dependency) — would require a new brief with different mechanism and falsifiers.

## 12. References

**Existing project artefacts (read order matters):**
1. `specs/2026-05-14-research-post-sebi-edges.md` — roadmap parent.
2. `data/sebi_calendar/README.md` — dependency-tag vocabulary.
3. `data/sebi_calendar/rule_changes.csv` — row 2025-02-01 (high severity, `option_premium;F&O_speculation;MIS_leverage`).
4. `specs/2026-05-07-sub-project-9-brief-pre_open_auction_imbalance_fade.md` — sister brief, RETIRED 2026-05-07.
5. `specs/2026-05-07-pre_open_url_discovery.md` — exhaustive 30-URL probe + alt-source assessment.
6. `tools/pre_open_auction/fetch_pre_open.py` — historical scraper + live API snapshotter.
7. `tools/pre_open_auction/verify_pre_open.py` — validation harness.
8. `services/screener_live.py` lines 1034-1110 — timer scan loop (09:20 first slot).
9. `services/ingest/live_tick_handler.py` lines 104-125 — pre-9:15 tick-drop rule.
10. `services/market_hours_manager.py` line 84 — `is_market_open()` 09:15 hardcode.
11. `structures/gap_fade_short_structure.py` — TRUSTED comparator for 09:15-09:30 window.

**Web evidence (verified May 14, 2026):**
1. NSE Pre-Open Session product page — auction mechanics 09:00-09:08-09:12-09:15 IST. https://www.nseindia.com/static/products-services/equity-market-pre-open
2. Pagano, Singh — "Pre-Open Call Auction and Price Discovery: Evidence from India" (Cogent Economics & Finance, 2014). https://www.tandfonline.com/doi/full/10.1080/23322039.2014.944668
3. Agarwalla, Jacob, Varma — "Pre-Open Auctions in Indian Equity Markets" (IIM-A working paper, 2014). https://faculty.iima.ac.in/~iffm/Indian-Fama-French-Momentum/data/Pre-Open.pdf
4. Profitmart — "How SEBI Margin Rules Have Changed Option Trading In 2025". https://profitmart.in/blog/how-sebi-margin-rules-have-changed-option-trading-in-2025/
5. Marketfeed — "Understanding SEBI's New Rules and Their Impact on Indian Options Trading". https://www.marketfeed.com/read/en/understanding-sebis-new-rules-and-their-impact-on-indian-options-trading
6. Zerodha Z-Connect — "SEBI's new rules for index derivatives" (Oct 2024). https://zerodha.com/z-connect/business-updates/sebis-new-rules-for-index-derivatives-heres-whats-changing

**Peer-reviewed call-auction literature (background):**
1. Pagano, Schwartz (2003), *J. Banking & Finance* — "A Closing Call's Impact on Market Quality at Euronext Paris". https://www.sciencedirect.com/science/article/abs/pii/S0378426602002753
2. Comerton-Forde, Rydge (2006), *J. Financial Markets* — "Call auction algorithm design and market manipulation". https://www.sciencedirect.com/science/article/abs/pii/S1386418106000231

---

## Pre-registration footer

This brief is timestamped 2026-05-14 and committed to `research/post-sebi-edge-setups`. Cell definitions §6, falsifier thresholds §7, and the mechanism statement §1 are **locked**. Any modification after seeing data invalidates the brief and must be flagged as such with a NEW brief and a new timestamp.
