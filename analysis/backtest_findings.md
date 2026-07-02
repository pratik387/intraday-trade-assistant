# Post-OHLC-Fix Backtest Findings

Findings discovered during gates-off backtest runs (Feb 2026).
Review each after the full 3-year backtest completes and pipeline_health_analyzer produces data.

---

## 0b. [2026-06-29] Multi-day composite selection layer: replay parity = perfect fills + expected dedupe (NOT a regression)

**Found by:** Stage-8 replay (`tools/sub9_research/backtest_mtf_replay.py`), Q1-2024, all 4 setups (A2/C1/C4/C6), on branch `feature/multiday-composite-selection` AFTER the cross-setup composite selector shipped.

**Observed:** 399 book exits, exit 0. Fill parity perfect (`fill_diff_bps p50/p90 = 0.0/0.0`, 98–100% matched). Per-setup `exec_n ≪ res_n` (399 unique book positions vs 886 standalone-basket rows = ~55% cross-setup dup/held-carryover). Per-setup `exec_PF < res_PF` (e.g. crash2d 0.964 vs 1.780; low52 1.898 vs 2.774). **Only 4 stale exits** (vs 871 in finding #0 — the holiday-calendar bug is fixed).

**Cause (by design, not a bug):** The composite layer dedupes a name flagged by several setups into ONE owner-attributed book position + applies cross-day held-union filtering. The research ledger (`research_ledger`) is per-setup standalone (no cross-setup dedupe), so it double-counts shared names → far more rows. `exec_PF < res_PF` is an **owner-attribution artifact of this tool**: it records owner-labeled book events and DISABLES the decay tripwire (line 131), so a setup that loses its strong shared names to a higher-`cap_score` owner (zscore_oversold wins most overlaps → low52 drops 59→9) shows a thinner owned-only ledger. The real per-setup edge is measured by the tripwire, which Task 5 feeds for ALL contributors — validated forward in paper (design §6.1), not by this owner-only replay.

**Action:** None blocking. This is the intended hidden-concentration removal (886 overlapping → 399 unique). Open question for paper/§6.1: is owner = max-`cap_score` the right attribution, and does `zscore_oversold` over-dominating ownership warrant per-setup composite weights. Spec: `specs/2026-06-29-multiday-composite-selection-design.md`.

## 0. [2026-06-15] Multi-day CNC/MTF executor: trading-day math ignores NSE holiday calendar (BLOCKER before paper)

**Found by:** Stage-8 replay parity harness (`tools/sub9_research/backtest_mtf_replay.py`) on the new multi-day batch (A2/C1/C4/C6), Q1-2024 sample. Branch `research/2026-06-14-mtf-capitulation-revert`.

**Observed:** Fill-source parity is perfect (executor 5m 09:15-open/15:25-close == research clean_daily open/close, fill_diff p50 = 0.0 bps). BUT `exec_n ≈ 0.5 × res_n` and `exec_PF << res_PF` for every setup (A2 0.86 vs 1.30; low52 1.27 vs 2.77; zscore 0.94 vs 1.65; crash2d 1.44 vs 1.78), with **871 stale exits** in one quarter.

**Root cause:** `services/execution/overnight_handlers.py:_next_trading_day` and `services/execution/mtf_capitulation_handlers.py:_add_trading_days` use **weekday** arithmetic (Mon-Fri), NOT the NSE holiday calendar. So `exit_on_date` (and AMO `entry_date`) routinely lands on a holiday → the exit cron never fires that day → `today > exit_on_date` → position flagged stale, **never settled, and leaks in the position store** (re-counted every subsequent session). ~half of trades never close → the validated research PF does not transfer.

**Scope:** Real LIVE bug (the cron uses the same weekday math), not a backtest artifact. **Also affects `close_dn_overnight_long`** (shares `_next_trading_day` for its AMO exit date) — likely the same latent orphan risk on holiday-adjacent exits.

**Fix (before any paper):**
1. Wire `assets/nse_holidays.json` (79 entries, `tradingDate` = `DD-Mon-YYYY`) into `_next_trading_day`/`_add_trading_days` so they skip holidays.
2. Fix the stale-exit path: settle at the next available session's close (or remove + reconcile) instead of leaking the position in the store forever.
3. Re-run `backtest_mtf_replay.py` and confirm `exec_PF → res_PF`, `exec_n → res_n`, stale → ~0.
4. Regression-test `close_dn` (its 1-night exit-date arithmetic) after touching the shared `_next_trading_day`.

**Status (2026-06-15, commit d5aa95f): HOLIDAY BUG FIXED + VERIFIED.** `_next_trading_day` now holiday-aware; stale exits self-heal. Re-run Stage-8 (Q1-2024): stale 871→4, fill-diff 0.0/0.0 bps all 4 setups. close_dn regression clean.

**NEW sub-finding 0b (open) — confidence-card PFs over-count overlapping signals (lesson #19 class).** The Stage-8 replay also revealed `exec_n ≈ 60% of res_n` with `exec_PF < res_PF`, because the PRODUCTION executor dedups (one position per symbol while held) but the research cell-mine counts EVERY signal as an independent trade. Deduping the research ledger converges trade counts toward the executor (e.g. zscore 559→398 vs exec 336; PF 1.65→1.27) and confirms it; residual gap is Q1-window-boundary + min_universe floor + small-n noise. **Implication:** the validated confidence-card PFs (A2 1.36 / low52 1.86 / zscore 1.43 / crash2d 1.36) are OPTIMISTIC — production-faithful PF requires re-running the Stage-5/6 cell-mine + confidence cards WITH per-symbol-hold-window dedup (one trade per symbol per overlapping hold). Effect is setup-dependent (notably ~−0.4 PF for zscore). Do BEFORE paper. All 4 remain `paper_enabled: false`.

---

## 1. Fill Quality Gate Disabled — Near-Zero T1 Partials

**Observed**: GREAVESCOT range_bounce_long entered at 175.00, T1 at ~175.05 = Rs.0.05 profit (0.03R).
Planned entry was 174.27 but trigger fired at 175.00 (above zone top 174.92).

**Root cause**: `LEVEL_TARGET_PRESERVED` logic keeps structural targets unchanged when actual entry overshoots planned entry. Good concept (structural levels matter), but without fill quality validation, trades with collapsed R:R proceed.

**Existing guard**: `fill_quality.enabled: false` in configuration.json (line 337).
Code at `trigger_aware_executor.py:548-613` — `_validate_fill_quality()` checks `min_rr_to_t1: 0.3` and `max_slippage_pct: 0.75`. Would have caught this case (0.03R < 0.3R threshold).

**Action after backtest**:
- Enable `fill_quality` for live/paper
- Use pipeline_health_analyzer Section 3C (parameter_deciles on `slippage_bps` and structural_rr) to validate 0.3R threshold
- Check how many trades in the backtest had actual_rr < 0.3 and what their aggregate PnL was

---

## 1b. T1 Minimum Profit Guard — Skip Wasteful Partials (FIX APPLIED)

**Observed**: 321/907 trades (35%) had T1 partial profit < Rs.100. All came from entry slippage (0 bad plans).

**Root cause**: LEVEL_TARGET_PRESERVED keeps structural targets when actual entry overshoots plan. Good concept, but creates near-zero T1 partials that pay ~Rs.52 charges for Rs.0 profit.

**Research backing** (Minervini, Grimes, Bellafiore, Velez — unanimous):
- Skip partial when < 0.3R — charges eat 87% of 0.1R partial on NSE
- SL-to-BE is triggered by BOOKING PROFIT, not price reaching level ("free trade" = cash concept)
- Do NOT move SL to BE when partial is skipped — tiny favorable move doesn't justify tightening
- At 0.2R partial on 60% qty: Rs.60 profit vs Rs.360 remaining risk → free trade condition impossible

**Fix applied**: `exit_t1_min_partial_r: 0.3` in configuration.json. Guard in `exit_executor.py:_partial_exit_t1()`:
- When T1 partial < 0.3R: skip partial, set `t1_done=True` (prevent per-tick re-check), NO SL-to-BE move
- `t1_skipped_low_r` flag in state for analytics tracking
- Full qty continues to T2/trailing/EOD with original SL intact

**NSE charge analysis at 0.3R threshold** (risk_per_trade Rs.1,000):
| Partial R | Gross | Charges | Net | Charge % |
|---|---|---|---|---|
| 0.1R | Rs.60 | ~Rs.52 | Rs.8 | 87% |
| 0.3R | Rs.180 | ~Rs.52 | Rs.128 | 29% |
| 0.5R | Rs.300 | ~Rs.52 | Rs.248 | 17% |

**Action after backtest**: Validate 0.3R threshold in full 3-year data. Track `t1_skipped_low_r` trades separately — compare their T2 hit rate and total PnL vs trades that took normal T1 partials.

---

## 2. OR_KILL Disabled — Was Causing Mass Zero-PnL Exits

**Observed**: ~80+ OR_KILL_PARTIAL exits with Rs.0 or negative PnL. Late buffer (0.25x ATR after 14:00) made it trigger on almost everything.

**Root cause**: OR_KILL is NOT a professional trading concept. OR-based exits only apply to OR-based entries (Crabel, Fisher ACD). Blanket application across 40+ setup types is wrong.

**Fix applied**: Added `or_kill_enabled: false` in configuration.json, early return in `exit_executor.py`.

**Action after backtest**:
- Keep disabled permanently
- If ORB-specific time-based exits are needed, implement them ONLY for orb_breakout/orb_breakdown setups
- pipeline_health_analyzer Section 3A exit_reason breakdown will show if any setup benefited from OR_KILL

---

## 3. _DryRunBroker Missing get_daily — Volume Confirmation Broken in Backtest

**Observed**: `'_DryRunBroker' object has no attribute 'get_daily'` during exit executor volume confirmation.

**Fix applied**: Added `get_daily()` delegate to `_DryRunBroker` in main.py. Also changed volume check error handler from `return False` (block exit) to `return True` (allow protective exit when data unavailable).

**Action after backtest**: Verify volume confirmation is actually working now by checking exit logs for volume-based decisions.

---

## 4. Gates-Off Config State — What's Disabled

Current state for the full backtest run. All need re-evaluation after results:

| Gate/Filter | Config Location | Status | Original Value |
|---|---|---|---|
| fill_quality | configuration.json:337 | **disabled** | min_rr_to_t1=0.3, max_slip=0.75% |
| or_kill | configuration.json | **disabled** | was active with late_buffer_mult=0.25 |
| quality_filters | base_config.json:91 | **disabled** | min_structural_rr=1.2, min_t1_rr=1.2 |
| cap_strategy_blocking | base_config.json:14 | **disabled** | blocked FHM in micro/small/large cap |
| time_of_day ranking | base_config.json:165 | **disabled** | late afternoon penalty 1.5x-2.5x |
| blacklist | base_config.json:213 | **disabled** | 10 strategies blacklisted |
| rsi_dead_zone | base_config.json:236 | **disabled** | long RSI 35-50 blocked |
| volatility_expansion | base_config.json:6 | **disabled** | ATR expansion > 0.8 blocks levels |

**Action after backtest**: pipeline_health_analyzer Section 7 will produce evidence-backed recommendations on which to re-enable.

---

## 5. New Detector: volume_breakout — Needs Validation

**Added**: `structures/volume_breakout_structure.py` — swing level breakout with institutional volume surge (2x+).

**Distinct from**:
- `level_breakout`: static PDH/PDL levels only
- `momentum_breakout`: 3-bar momentum, no level reference
- `volume_spike_reversal`: trades AGAINST the spike (reversion)

**Config**: Full 15-key config in configuration.json for both long/short. Enabled.

**Action after backtest**:
- Check pipeline_health_analyzer Section 3A for volume_breakout_long/short stats
- Compare with level_breakout and momentum_breakout on same metrics
- Verify it's detecting distinct setups (not just duplicating level_breakout signals)

---

## 6. MDS_DIAG Logs Removed

**Removed**: 4 `MDS_DIAG` diagnostic log statements from `screener_live.py`. These were temporary debugging logs from market data service integration.

**No action needed** — cleanup only.

---

## 7. Missing Config Keys Fixed

**level_breakout_short**: Was missing `entry_mode`, `min_stop_distance_pct`, and all dual-mode entry keys. Added to match level_breakout_long.

**orb_level_breakout_short**: Was missing `min_stop_distance_pct`. Added (0.3).

**Action after backtest**: Verify both short configs produce trades. If zero trades, the config values may need tuning.

---

## Pipeline Health Analyzer Cross-Reference

Once `tools/pipeline_health_analyzer.py` runs on the full backtest, map findings to these sections:

| Finding | Analyzer Section |
|---|---|
| #1 Fill quality threshold | Sec 3C: parameter deciles on slippage_bps, structural_rr |
| #2 OR_KILL impact | Sec 3A: exit_reason breakdown per setup |
| #4 Which gates to re-enable | Sec 4: gate block audit + Sec 7: actionable findings |
| #5 volume_breakout validation | Sec 3A: per-setup stats, Sec 3D: regime matrix |
| #7 Short config validation | Sec 3A: check for volume_breakout_short, level_breakout_short trades |

---

## 8. Sanity vs OCI Drift Investigation — May 2026

Triggered by user concern that OCI 1-year forward backtest (`20260518-154136_full`) underperformed expected sanity numbers. After systematic debugging (Phase 1-4) across all 4 production setups, the conclusion is that **most of the apparent drift was sanity script methodological bias, not production performance gaps**.

### Per-setup drift table (in-sample 2025-01 → 2025-09)

| Setup | OCI PF | Raw Sanity PF | %same-bar | Corrected Sanity PF | Real Drift |
|---|---|---|---|---|---|
| mis_unwind_vwap_revert_short | 1.024 | 3.16 | 88.2% (look-ahead) | **1.007** (clean re-run) | -0.017 (none) |
| or_window_failure_fade_short | 1.35 | 1.06 | 0.7% | 1.08 | +0.27 (OCI beats) |
| round_number_sweep_short | 1.06 | 1.01 | 0.4% | 1.02 | +0.04 (matches) |
| circuit_release_fade_short | 1.20 | 3.97 | 5.5% | TBD (re-run pending) | partial sanity bias + concurrency |

### 8a. Sanity CSV look-ahead bias (the dominant "drift" cause)

The sanity CSVs at `reports/sub9_sanity/_*_trades_oos.csv` were generated **May 16, 2026 — BEFORE the fix in commit 78f4642 (May 17)**. The old script entered at signal-bar CLOSE and walked the SAME bar for exits, making same-bar T2_full physically impossible but counted as wins.

**For mis_unwind_vwap_revert_short cell-locked (RSI≥75 + vol≥7):**
- OLD (buggy): n=2,111, PF=3.16, NET=+Rs 1.66M
- CLEAN (fixed): n=1,662, PF realized=**1.007**, PF net=**0.804**, NET=**-Rs 219K**
- OCI delivered: PF=1.024

**Verdict for mis_unwind**: Setup is **net-negative after Zerodha charges** (PF_net=0.80). Was shipped on a fake number.

Clean CSV: `reports/sub9_sanity/_mis_unwind_vwap_revert_trades_oos_CLEAN.csv`

**Cell sweep result (May 18)**: Exhaustive sweep across RSI/vol_ratio/vwap_ext/hour/cap dimensions + R-multiplier variations. Only ONE cell clears ship gates by a fragile margin:

```
Cell: RSI>=65 & vol_ratio>=5 & vwap_ext>=1.0 & 14:45-14:55 & small_cap
  n=507, PF_net=1.154 (barely > 1.15 floor), 6/9 winning months, maxLossShare 0.37
  H1 (Jan-May): PF_net 1.22
  H2 (Jun-Sep): PF_net 1.07 (decaying)
```

R-multiplier sweep on the original cell-lock (RSI>=75 + vol>=7) — every SL/T2 combo tested fails. Best is SL=0.7 / T2=3.0 with PF_net 0.853.

**The "shippable" cell lives in 14:45-14:55, small_cap.** But Zerodha MIS auto-square is 15:00-15:15, so 14:45-14:55 timing is mechanism-ADJACENT, not the actual unwind. Edge is generic small-cap mean reversion, not MIS-unwind specifically. Mechanism thesis NOT supported by data.

**Recommendation: RETIRE `mis_unwind_vwap_revert_short`.** Add to `docs/retired_setups.md` with original cell-lock PF look-ahead inflation note (3.16 → 1.007 realized → 0.804 net) and H2 decay trajectory.

**UPDATE 2026-05-19 — RETIREMENT EXECUTED.** After the user pushed back on my OOS-overfit cell sweep, ran the disciplined Phase 1-5 protocol from scratch:
- Phase 1 (Indian markets research): confirmed broker auto-square mechanism is real and heterogeneous (Upstox/Angel 15:15, Zerodha 15:20, ICICI 15:15-20). All ₹50+GST = ₹59 penalty per auto-squared order.
- Phase 2 (empirical signature on Discovery 374K symbol-days): forced-sell volume bulge at 15:15-15:25 confirmed (2.5-3.0x baseline). Directional drift WEAK in aggregate but real.
- Phase 4 (sanity v2 with anti-bias guards verified per retired_setups.md common failure modes)
- Phase 5 (Discovery cell sweep with REAL window 15:00-15:10 entry, 15:15 exit = 15:20:00 IST self-exit before Zerodha penalty)

Locked cell from OOS+Discovery parity: **RSI≥85 + vol≥15 + SL=0.5 + T2=3.0R**

| Period | n | PF_net | mwin | Net PnL |
|---|---|---|---|---|
| Discovery (2023-24) | 622 | **1.213** | 19/24 | +Rs 72,728 |
| OOS (2025 Q1-Q3) | 261 | **1.216** | 7/9 | +Rs 32,195 |
| **Holdout (2025-10 → 2026-04)** | **198** | **0.751** ❌ | ? | **-Rs 34,562** |

Disc↔OOS parity was near-perfect (Δ=0.003 PF) but BOTH predate SEBI Oct 2025 changes. Holdout post-SEBI collapsed PF_net 1.22 → 0.75. Setup is regulatory-decay-dead — same failure pattern as `delivery_pct_anomaly_short`. See `docs/retired_setups.md` for full evidence chain. See `tasks/lessons.md` 2026-05-19 for the methodological lesson (Disc+OOS parity is necessary but not sufficient when both predate a regulatory cutover).

### 8c. `round_number_sweep_short` (C-02) — RETIRED 2026-05-19 (cell-mining illusion)

Production OCI showed PF_real 1.075 with 50% monthly winning months — marginal, below ship gate.

Post-hoc cell-lock recompute of aggregate sanity CSVs (37K Discovery / 17K OOS / 11K Holdout trades) with production filter (small_cap, 11:00-12:30, Rs.100-250) + T1-full-exit geometry:

| Period | n | PF_real | **PF_net** | mwin |
|---|---|---|---|---|
| Discovery | 683 | 0.86 | **0.69** | 4/24 (17%) |
| OOS | 314 | 0.85 | **0.69** | 0/9 (0%) |
| Holdout | 270 | 0.94 | **0.73** | 1/7 (14%) |

ALL THREE PERIODS NET-LOSING. The `_status_2026_05_16` claim of 3-window PF 1.24/1.21/1.17 does NOT reproduce from the underlying CSVs — cell-mining produced an illusion. RETIRED per `docs/retired_setups.md`.

### 8d. Active portfolio after May 2026 retirement wave

**5 active setups remain (down from 8):**
- gap_fade_short
- circuit_t1_fade_short (war-period weakness; pre-war PF 1.88; kept active per portfolio decision)
- delivery_pct_anomaly_short
- long_panic_gap_down (RESILIENT through war, PF 1.31)
- or_window_failure_fade_short (war PF 0.92, marginal loss)

Retirements 2026-05-19:
- `mis_unwind_vwap_revert_short` — regulatory regime break, Holdout collapse 1.22 → 0.75
- `round_number_sweep_short` — cell-mining illusion, all periods net-losing (PF_net 0.69-0.73)
- `circuit_release_fade_short` — regulatory decay starting Oct 2025 (PRE-WAR), PF 1.26 OOS → 0.84 HO_pre → 0.60 HO_war. No 3D cell salvages.

### 8e. Per-period diagnostic (full 6-setup portfolio, OOS+HO)

Computed from `20260519-123643_full` (raw PnL, before MIS/tax/fees):

| Period | Months | n | PF | Avg/trade | Sum |
|---|---|---|---|---|---|
| OOS (Jan-Sep 2025) | 9 | 2,042 | 1.46 | +Rs 160 | +Rs 326K |
| HO pre-war (Oct-Dec 2025) | 3 | 657 | 1.39 | +Rs 135 | +Rs 89K |
| **HO war (Jan-Apr 2026)** | 4 | 719 | **0.88** | -Rs 49 | **-Rs 36K** |

**Per-setup × per-period:**

| Setup | OOS PF | HO pre-war PF | HO war PF | Pattern |
|---|---|---|---|---|
| long_panic_gap_down | 1.32 | 1.09 | **1.31** | RESILIENT (LONG balances portfolio) |
| gap_fade_short | 1.81 | 1.81 | 1.08 | Graceful war drag |
| delivery_pct_anomaly_short | 1.43 | 1.29 | 1.09 | Graceful war drag |
| or_window_failure_fade_short | 1.48 | 1.66 | 0.92 | Mild war loss, deploy with caution |
| circuit_t1_fade_short | 1.29 | **1.88** | **0.53** | War-only collapse (kept active) |
| circuit_release_fade_short (retired) | 1.26 | **0.84** | 0.60 | Decay PRE-war, retired |

**Lesson:** circuit_release decay started in HO_pre_war (post-SEBI-Oct-2025 regulatory cutover) — distinct from circuit_t1 which only failed in the war period. Both face the same war headwind but only circuit_release had pre-existing regulatory decay. Decision principle: war-only collapse = regime-temporary (keep active); pre-war decay = structural alpha decay (retire).

### 8b. circuit_release_fade_short — clean sanity STILL diverges 3.7× from OCI

Re-ran clean sanity for circuit_release with fixed entry timing. Results:

| Metric | OLD buggy | CLEAN (fixed) | OCI IS |
|---|---|---|---|
| n | 493 | 476 | 434 |
| PF_realized | 3.97 | **4.44** | 1.20 |
| PF_net (after fees) | 3.13 | **3.48** | ~1.0 |
| WR | 60.6% | **64.5%** | 36.1% (HO worse) |
| Exit mix | — | 72% time_stop / 28% t2_full / **0% hard_sl** | 47% time_stop / 28% hard_sl / 19% t2_full |

**Counterintuitively, removing the `close_off_high_pct` (EOD-leak) filter RAISED clean PF +0.46.** The filter was rejecting winners along with losers. So the EOD-day_high bias is NOT the primary cause of the gap.

**The 3.7× gap (PF 3.48 vs 1.20) lives in EXECUTION MECHANICS, not detection logic.**

Diagnostic: clean sanity has **zero hard_sl exits** (max mae_r = 1.0, but 0 trades exceed). OCI has 34% hard_sl. The intra-bar tick paths in production reach prices that 5m-bar-low sanity never sees, OR production's SL geometry differs in some subtle way.

**ROOT CAUSE CONFIRMED (May 18 deep-dive)**: 100% sanity look-ahead bias, NOT a production bug.

Sanity script `sanity_circuit_release_fade.py:209-210`:
```python
if morning_high < day_high * 0.999:
    return None  # day_high reached AFTER morning -> not a morning pin
```

`day_high` is the FINAL end-of-day high. Sanity uses HINDSIGHT to reject any day where the morning high gets broken later. Production can't know the future at 13:00 signal time, so it correctly fires on signals that LOOK like morning pins at that moment — but some of them turn out to break out higher in the afternoon.

**Partition analysis of OCI trades vs sanity**:

| Partition | n | PF | WR | hard_sl exits | PnL |
|---|---|---|---|---|---|
| OVERLAP (both took) | 237 | **4.95** (production BEATS sanity 3.48) | 66.2% | 0 | +Rs 151K |
| OCI-ONLY (sanity rejected via look-ahead) | 197 | **0.24** | 16.2% | 124 (63%) | -Rs 114K |

**Quantitative confirmation** (random sample of 50 OCI-only trades):
- 46/50 (92%) had `day_high > morning_high * 1.001` — broke out later
- Of the 46 broke-out: 35 hit hard_sl (76%)
- Of the 4 non-broke-out: 1 hit hard_sl

**Mode A test** (zone re-touch, 15-min expiry) tested as alternate execution mechanic:
- Mode A: n=469, PF_net 2.98 (vs Mode B PF_net 3.48). Only 14% drag.
- 98.7% of zone-touches happen at offset=1 bar (functionally same as Mode B).
- Mode A does NOT close the gap to OCI 1.20.

**Verdict for circuit_release**: Production is correctly identifying the right signals at signal time. The 197 "failed morning pin" days are unavoidable without future knowledge. Sanity's PF 3.48 is achievable only with hindsight. The realistic production PF expectation is OCI's 1.20.

**Potential remediation**: Make production's `session_high_so_far` check STRICTER — require N bars of no-new-high after morning before firing. Trade-off: stricter filter reduces winners too. Worth a parameter sweep but no guarantee of net improvement.

Clean CSV: `reports/sub9_sanity/_circuit_release_fade_short_trades_oos_CLEAN.csv`

**Verdict**: circuit_release has a real edge in idealized fills (PF_net 3.48 with n=476) but production fills aren't realizing it. Need to either:
- Run sanity with `--entry-mode A` to model tick-zone-touch and verify the gap is execution-mechanic-driven
- Investigate whether the cap can be raised given the strong sanity edge
- Accept that real OCI PF is the production reality and the sanity number is unrealistic ceiling

### 8c. Five production bugs found and fixed during investigation

| # | Bug | File | Severity |
|---|---|---|---|
| 1 | `regime='chop'` 100% of trades | `services/screener_live.py:1946-1949` + `_index_df5` synth | Low — no setup filters on regime |
| 2 | R-mismatch — stale `sizing.rps` after fill | `services/target_recalc.py` (3 branches) + `services/execution/exit_executor.py:2086` | **HIGH** — fast-scalp auto-BE used wrong threshold |
| 3 | ORH/ORL NaN — multi-day feather tripped "late-start" recovery | `services/screener_live.py:2366-2372, 2705-2712, 2515-2596` | Medium — 528 afternoon trades had no ORB |
| 4 | MIS list time-travel — live Zerodha sheet applied to historical | `services/screener_live.py:1932-1946` | Medium — 31% of circuit_release missed signals |
| 5 | `bar_scheduler.py` used dead logger | `services/bar_scheduler.py:1-20` | High visibility — silent admission drops |

Bug 5 in particular: rejection messages at `bar_scheduler.py:68, 88, 95-98` were going to a logger with no handler. After fix, all `BAR_SCHED_BLOCK | ... | risk:` and `capital:` rejections appear in `agent.log`.

### 8d. Lessons for sanity script construction

- **Entry timing**: enter at NEXT bar's open (`bars.iloc[i+1].open`), walk from `i+1`. Never include the signal bar in path-walk.
- **Look-ahead filters**: never use EOD daily data (`daily.high`, `daily.close`) to validate intra-day "as-of" conditions. Use only data available at signal time.
- **Production parity for entry execution**: sanity Mode B (next-bar-open) overstates production Mode A (zone re-touch with N-bar expiry). Document which mode the sanity uses in the file header.
- **Fees matter for low-PF setups**: report BOTH `PF_realized` and `PF_net` — Zerodha charges flip many marginal setups from PF>1 to PF<1.
- **Cell-locked filters MUST be applied at signal time, not via CSV post-filter** — if features (RSI, vol_ratio) are computed differently in production vs sanity, the same CSV row may NOT correspond to a production-fired trade.

### 8e. Action items

- [x] All 5 production bugs fixed
- [ ] Re-run sanity for circuit_release_fade_short with fixed entry timing (pending)
- [ ] Cell sweep + R-multiplier sweep on clean mis_unwind CSV to find any shippable cell (pending)
- [ ] Decision: retire mis_unwind_vwap_revert_short if no cell ships
- [ ] Update spec docs (`specs/2026-05-14-research-post-sebi-edges.md`, related) to reflect realistic PF expectations
- [ ] Investigate the 5 other modules with dead loggers (`circuit_breaker.py`, `market_hours_manager.py`, `runtime_rvol_baseline.py`, `scan/energy_scanner.py`, `setup_universe.py`) — same fix pattern as bar_scheduler
