# §3.3 Brief: `mtf_capitulation_revert_long`

**Sub-project:** #9 (microstructure-first redesign) — CNC/MTF track (Track A/B of `docs/research/2026-06-02-algo-research-backlog.md`)
**Status:** **CONFIDENCE-CARD PASSED on clean CA-adjusted data — awaiting structure code (new CNC/MTF machine) + paper.**
**Date:** 2026-06-14
**Type:** Daily cross-sectional mean-reversion, **2-day hold, CNC/MTF delivery** (NOT intraday-MIS — new execution architecture).
**Origin:** CNC track. Plain A1 short-term reversal was KILLED (bad-print artifact on unadjusted `consolidated_daily`); the capitulation-conditioned variant on a clean CA-adjusted panel is the survivor. Trail: memory `project_cnc_mtf_capitulation_revert`.

---

## 1. Asymmetry

**Name:** Illiquid daily capitulation over-shoot → 2-day snapback (the CNC/MTF horizon expression of the system's proven `gap_fade_short` / `panic_crash_revert_long` over-extension-reversion family).

A thin small/mid name that falls into the bottom-decile of trailing-5d return **on a turnover/volume shock (≥2× its 20-day average)** is experiencing forced/panic selling into a liquidity vacuum — the same capitulation mechanism the system already monetizes intraday, but resolving over 1-2 sessions rather than within one. The volume shock is the discriminator: a *quiet* drift-down loser continues; a *high-volume* capitulation over-shoots and reverts. The Indian-illiquid specificity is the thin book that lets the over-shoot happen; in liquid names the move is efficient (confirmed: the edge dies monotonically from illiquid→liquid ADV tiers).

## 2. Participants

- **Forced/panic sellers** (stop cascades, margin calls, retail fear) drive the deep drop on a volume spike — they sell into a vacuum and over-shoot.
- **Absent stabilizers** — no institutional desk provides bid-side depth in these illiquid names intraday, so nothing arrests the over-shoot until the panic exhausts.
- **We** provide liquidity into the capitulation (long, via MTF leverage), harvesting the 2-day snapback. Counter-party = the forced sellers' over-shoot.

## 3. Persistence

1. Structural illiquidity is permanent (thin book → over-shoot). Not arbitrage-scalable (impact eats institutional size — this is why MTF tier-1 names, the *least*-liquid leverageable cohort, carry the edge).
2. Forced-selling + retail panic are permanent behavioral/structural features.
3. **Inverse-confirmed and mechanism-coherent:** winner-decile + same volume-shock *falls* (continuation), only the loser+shock reverts → spread +1.5-1.8%. Monotonic illiq>liq gradient. Same asymmetry family as the two LIVE reversion setups.

## 4. Evidence (clean CA-adjusted daily panel, Discovery-locked, one-shot OOS/2026)

- **Cost model (MTF round-trip):** delivery STT 0.20% + brokerage + charges + 2-day MTF interest = 0.347% fixed; tested at +10/20/30bp slippage.
- **Cell-mine (full grid on Discovery, 540 cells, n_disc≥200):** 80 ship-eligible (Disc netPF≥1.20); **52 net-positive in all 3 periods** (coherent cluster, all tier-1 + volume-shock). Most-stable cell selected by min ΔPF (lesson: stability over top-PF).
- **Confidence card (Lesson #15, production framework) on the MOST-STABLE cell (n=1115 pooled):**
  - BCa CI: **PF 1.356 [1.17, 1.60]** (lower bound >1.0), expectancy ₹341/trade [173, 535], WR 47.8%.
  - Per-regime **6/7 positive**; only loser = post_tariff_consolidation (PF 0.92, low-vol regime); **positive in war_vol_2026 (1.57)** → diversifies the all-short-fade book.
  - **Harvey-Liu sign-preserving at the TRUE sweep count M=540:** raw Sharpe 1.60 → **adj 0.80 (M=540 Bonferroni, 50% haircut)** / 0.91 (M=80 ship-eligible). Survives the honest selection-bias penalty.
- **Inverse-edge:** winner+shock decile gross OOS −0.71% / 2026 −0.77% (continuation), loser+shock bounces → confirms directional reversal, not noise.
- **Adjacent proven edge:** `panic_crash_revert_long` + `gap_fade_short` (LIVE) — same illiquid over-extension→reversion family.

## 5. Direction

**LONG only.** Buy the capitulated loser, hold 2 days, exit. MTF leverage (~3× on tier-1 names) amplifies the per-trade net into capital ROI (+0.27%/trade net → ~0.8%/cycle on capital). No short leg (the winner side continues, not fade-able cleanly net-of-cost).

## 6. Mechanic — CELL-LOCKED (Discovery-only selection)

- **Universe:** MTF-eligible (Zerodha approved list, exclude ETF) × **ADV-tier-1** (most-illiquid quintile by trailing-20d turnover, ≥₹20L floor).
- **Signal (day-t close):** trailing-5d return in the **bottom 5%** cross-sectionally (most-stable cut from the 540-cell mine; ΔPF 0.030) AND **day-t turnover ≥ 2× trailing-20d-avg turnover** (causal, excludes day t).
- **Entry:** T+1 **open** (signal known at day-t close). **Hold 2 days. Exit T+1+2 close.**
- **Product:** CNC/MTF delivery (NOT MIS — overnight, leveraged).
- All thresholds config-driven, no hardcoded defaults (CLAUDE.md rule 1).

## 7. Active window
Daily EOD signal computation; T+1-open entry; 2-day hold. Not intraday-time-of-day sensitive.

## 8. Risks / falsification (pre-registered, locked)

1. **Bad-print/CA contamination (THE killer of plain A1):** MUST run on the clean CA-adjusted panel (`clean_daily_from5m.feather`), NEVER `consolidated_daily`. First sanity for any re-run: re-confirm cleaning the extreme-negative tail doesn't change the verdict. *(Holds on clean data.)*
2. **Survivorship / point-in-time MTF list:** the 2026-05 MTF snapshot applied to 2023-25 is anachronistic (no historical MTF list, cf Lesson #27). **Forward paper is the production-faithful gate** — backtest cannot resolve this.
3. **Low-vol regime weakness:** post_tariff_consolidation PF 0.84. Capitulation-reversion needs volatility; size for the low-vol drawdown regime. If a fresh low-vol window is sustained-negative → regime-gate or pause.
4. **Tier-1 slippage realism:** edge lives at ≤20bp slippage; at ≥30bp it thins. Confirm real open→close slippage on tier-1 MTF names in paper.
5. **Frequency:** ~1819 trades / 3.3yr ≈ 550/yr (≥30/cell floor cleared).

## 9. Next steps (pre-registration + build)
0. **Lock the cell JSON** (`tools/sub9_research/mtf_capitulation_revert_long_cell_lock.json`). ✅ DONE (commit b207e6d)
1. **Structure: the new CNC/MTF machine** (does NOT exist; cf `illiquid_momentum_long` impl §): EOD ranking module, multi-day CNC/MTF position store, MTF basket executor (diff held vs target, MTF order routing), corporate-action handling. code-reviewer agent.
   - ✅ Component 1 — `services/cross_sectional_ranker.py` (EOD ranker), 7 tests (b207e6d)
   - ✅ Component 2 — `services/daily_panel_provider.py` (clean daily panel, feather+live), 8 tests (849f75c)
   - ✅ Component 3 — `services/state/position_persistence.py` multi-day extension (entry_date/exit_on_date/product + stale-snapshot salvage), 10 tests (67b3276)
   - ✅ Component 4 — `services/execution/mtf_capitulation_handlers.py` (basket executor: `run_eod` exits+entries, `run_verify_entries` fills), 7 tests. code-reviewer PASS w/ fixes applied.
2. **Impact-aware cost re-confirm** (tier-1 slippage).
3. **Paper-trade ≥ 4-8 weeks** (the load-bearing gate, per risk #2).

### Open follow-ups (deferred from Component 4 code review — do BEFORE live, not blocking paper)
- **catastrophe_stop_pct (15.0) is configured but NOT implemented.** Needs a mid-hold price check (e.g. in `run_verify_entries` each morning: if open-vs-entry drawdown ≥ stop, force-exit). Dead config until then — flagged, not silently ignored.
- **No `events.jsonl` / TradingLogger wiring.** Entries/exits don't appear in events.jsonl or analytics.jsonl, so `recover_from_events` has nothing to replay and trades are absent from the session report. Mirror `overnight_handlers.py`'s diag_event_log pattern.
- **Live exit/failsafe fill uses `get_ltp`, not the order's `average_price`.** `_place_moc_sell_and_fill` / `_failsafe_market_buy` should poll `get_order_status` for the real fill (matters for illiquid spreads). Paper path is unaffected (uses bar prices).
- **`_live_poll_fill` timeout (60s) is hardcoded** — promote to a config key before live.
- **Register `mtf_capitulation_revert_long` in `config/setup_categories.py`** (→ REVERSION) so category-grouped reporting sees it.
- **Wire the two crons** (EOD ~15:25, morning ~09:30) + a daily live `fetch_daily_window` on the broker for `LiveDailyPanelProvider` (make_provider fails fast without it — honest gate).

---

## Decision required
- [ ] APPROVED — lock cell + build the CNC/MTF machine + paper
- [ ] REVISE — specify
- [ ] RETIRE — reason

Per sub-9 §3.3: no detector/executor code until APPROVED. The CNC/MTF executor is a substantial new architecture (multi-day overnight position state + MTF routing) — flagged, not deferred.
