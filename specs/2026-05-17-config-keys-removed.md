# Config Keys Removed 2026-05-17 (Phase 0 PR A)

Per `specs/2026-05-17-dispatch-architecture-plan.md` Task A2. Each entry verified by deep grep (substring + `.get()` + bracket + variable-indirection + nested-parent + non-SEARCH_DIRS paths).

Verification source: `tools/audit_config_keys.py` (commit 75a17a1) flagged 114 candidates. Deep verification found **0 false positives** — all 114 keys are confirmed-dead.

## Reclassified to USED (false positives from audit)

| key | reason |
|-----|--------|
| *(none from the 114 candidates)* | All 114 candidates verified dead across 7-pattern check |

## Near-miss 1: `ws_flush_interval_ms` (NOT a candidate — accidental collateral removal)

During deletion of the `ws_*` candidate block (lines 260-266 in original), `ws_flush_interval_ms`
(line 263) was accidentally removed as it was sandwiched between dead keys. Caught by smoke test:

```
KeyError: 'SubscriptionManager: missing config key `ws_flush_interval_ms`'
```

Used by `services/ingest/subscription_manager.py:49`. Restored to config with value `500`.
This key was **never** a deletion candidate — not in the 114 list.

## Near-miss 2: `trigger_expiry_minutes` + `cancel_existing_pending` (NOT candidates — accidental collateral removal)

During deletion of the large flat `trigger_*` block, two live keys sandwiched between dead candidates
were accidentally removed:

- `trigger_expiry_minutes: 15` — used by `services/execution/trigger_aware_executor.py:842`:
  `expiry_minutes = int(self.cfg.get("trigger_expiry_minutes"))` — **no default**, raises on missing.
- `cancel_existing_pending: true` — used by `trigger_aware_executor.py:864`:
  `if not _wide_open and self.cfg.get("cancel_existing_pending", True)` — has default, but key was live.

Caught by second smoke test run which produced 0 ENTRY events (executor crashed before dispatching).
Both keys restored: `trigger_expiry_minutes: 15`, `cancel_existing_pending: true`.
Neither key was ever in the 114 candidate list.

**Special note on `execution_mode`:** `main.py:270` does `execution_mode = "in_process"` — this is a
local variable assignment, **not** a config read. The key is never accessed via `cfg.get()` or
`cfg["execution_mode"]`. Confirmed dead.

**Special note on `time_policy`:** `screener_live.py` has a method `_blocked_by_time_policy()` but
the method name is coincidence — the config key `"time_policy"` is never accessed via cfg. Confirmed dead.

**Special note on `trail_to`:** `data_models.py` has a dataclass field `trail_to: Optional[str]`
and `delivery_pct_anomaly_short_structure.py` passes `trail_to="breakeven"` as a keyword argument.
Neither accesses the top-level config key `cfg["trail_to"]` or `cfg.get("trail_to")`. Confirmed dead.

**Special note on `hold_bars`:** Many files reference `min_hold_bars` (a dataclass field), not the
top-level config key `hold_bars`. Confirmed dead.

**Special note on `dedupe_cooloff_bars`/`dedupe_require_setup_change`:** Explicitly tagged for
removal in `dedup_gate._comment` ("Legacy keys … will be removed in T14"). Zero code references.
Confirmed dead.

**Special note on `structure_manager`:** Config key `"structure_manager"` is never accessed via cfg.
The file `structures/structure_manager.py` exists but does not read this config key. Confirmed dead.

**Special note on `news_gate`, `time_policy`, `structure`, `cap_strategy_preferences`,
`regime_scores`, `bar_density_gate`, `ict_quality_filters`:** Nested-dict keys verified — no code
does `cfg.get("<key>", {})` or `cfg["<key>"]` to access children. Confirmed dead.

## Removed top-level keys (114 total)

| key | type | verification rationale |
|-----|------|------------------------|
| `adx_strength_min` | scalar | 0 code hits (all patterns) |
| `adx_trend_min` | scalar | 0 code hits (all patterns) |
| `bar_1m_freq` | scalar | 0 code hits (all patterns) |
| `bar_5m_freq` | scalar | 0 code hits (all patterns) |
| `bar_density_gate` | dict | 0 code hits; nested children not accessed via cfg |
| `breakout_momentum_candle_min_ratio` | scalar | 0 code hits (all patterns) |
| `breakout_momentum_candle_min_ratio_short` | scalar | 0 code hits (all patterns) |
| `breakout_volume_min_ratio` | scalar | 0 code hits (all patterns) |
| `breakout_volume_surge_min` | scalar | 0 code hits (all patterns) |
| `breakout_volume_surge_min_short` | scalar | 0 code hits (all patterns) |
| `cache_cleanup_interval_minutes` | scalar | 0 code hits (all patterns) |
| `cap_strategy_preferences` | dict | 0 code hits; nested children not accessed via cfg |
| `condition_state_cleanup_hours` | scalar | 0 code hits (all patterns) |
| `daily_trend_ema_period` | scalar | 0 code hits (all patterns) |
| `decision_require_structure_event` | scalar | 0 code hits (all patterns) |
| `dedupe_cooloff_bars` | scalar | 0 code hits; explicitly tagged for removal in dedup_gate._comment |
| `dedupe_require_setup_change` | scalar | 0 code hits; explicitly tagged for removal in dedup_gate._comment |
| `exec_heartbeat` | dict | 0 code hits; no cfg.get("exec_heartbeat") anywhere |
| `execution_mode` | scalar | 0 cfg reads; main.py:270 assigns local variable, does not read config |
| `expiry_allow_breakout` | scalar | 0 code hits (all patterns) |
| `expiry_allow_fade` | scalar | 0 code hits (all patterns) |
| `expiry_size_multiplier` | scalar | 0 code hits (all patterns) |
| `fill_quality` | dict | 0 code hits; nested children not accessed via cfg |
| `hold_bars` | scalar | 0 cfg hits; `min_hold_bars` is a dataclass field, unrelated |
| `ict_quality_filters` | dict | 0 code hits; nested children not accessed via cfg |
| `k_atr` | scalar | 0 code hits (all patterns) |
| `market_sentiment` | dict | 0 code hits; nested children not accessed via cfg |
| `min_entry_sl_distance_abs` | scalar | 0 code hits (all patterns) |
| `min_entry_sl_distance_pct` | scalar | 0 code hits (all patterns) |
| `momentum_min_ema_slope` | scalar | 0 code hits (all patterns) |
| `news_gate` | dict | 0 code hits; nested children not accessed via cfg |
| `news_min_hold_bars` | scalar | 0 code hits (all patterns) |
| `orb_pullback_hold_bars` | scalar | 0 code hits (all patterns) |
| `orb_pullback_max_minutes` | scalar | 0 code hits (all patterns) |
| `orb_pullback_tolerance_pct` | scalar | 0 code hits (all patterns) |
| `orb_require_index_alignment` | scalar | 0 code hits (all patterns) |
| `order_exception_delay_ms` | scalar | 0 code hits (all patterns) |
| `order_loop_idle_ms` | scalar | 0 code hits (all patterns) |
| `order_rate_limit` | scalar | 0 code hits (all patterns) |
| `order_retry_delay_ms` | scalar | 0 code hits (all patterns) |
| `order_retry_max` | scalar | 0 code hits (all patterns) |
| `pdh_break_tolerance_pct` | scalar | 0 code hits (all patterns) |
| `pdh_hold_bars` | scalar | 0 code hits (all patterns) |
| `pdl_break_tolerance_pct` | scalar | 0 code hits (all patterns) |
| `planner_atr_period` | scalar | 0 code hits (all patterns) |
| `planner_choppiness_high` | scalar | 0 code hits (all patterns) |
| `planner_choppiness_lookback` | scalar | 0 code hits (all patterns) |
| `planner_choppiness_low` | scalar | 0 code hits (all patterns) |
| `planner_entry_zone_atr_frac` | scalar | 0 code hits (all patterns) |
| `planner_entry_zone_breakout_mult` | scalar | 0 code hits (all patterns) |
| `planner_entry_zone_default_mult` | scalar | 0 code hits (all patterns) |
| `planner_entry_zone_fade_mult` | scalar | 0 code hits (all patterns) |
| `planner_max_gap_pct_for_trend` | scalar | 0 code hits (all patterns) |
| `planner_vwap_reclaim_min_bars_above` | scalar | 0 code hits (all patterns) |
| `range_break_retest_max_pct` | scalar | 0 code hits (all patterns) |
| `regime_cap_allocation` | dict | 0 code hits; nested children not accessed via cfg |
| `regime_scores` | dict | 0 code hits; nested children not accessed via cfg |
| `rejection_min_body_pct` | scalar | 0 code hits (all patterns) |
| `require_breakout_bar_volume_surge` | scalar | 0 code hits (all patterns) |
| `require_higher_timeframe_alignment` | scalar | 0 code hits (all patterns) |
| `router_assume_epoch_utc` | scalar | 0 code hits (all patterns) |
| `router_dayvolume_fields` | array | 0 code hits (all patterns) |
| `router_default_timezone` | scalar | 0 code hits (all patterns) |
| `router_price_fields` | array | 0 code hits (all patterns) |
| `router_symbol_fields` | array | 0 code hits (all patterns) |
| `router_timestamp_fields` | array | 0 code hits (all patterns) |
| `router_tradeqty_fields` | array | 0 code hits (all patterns) |
| `rsi_bearish_max` | scalar | 0 code hits (all patterns) |
| `rsi_bearish_min` | scalar | 0 code hits (all patterns) |
| `rsi_bullish_max` | scalar | 0 code hits (all patterns) |
| `rsi_bullish_min` | scalar | 0 code hits (all patterns) |
| `scan_min_bars_5m` | scalar | 0 code hits (all patterns) |
| `scan_output_max_rows` | scalar | 0 code hits (all patterns) |
| `scan_ret1_min` | scalar | 0 code hits (all patterns) |
| `scan_ret3_min` | scalar | 0 code hits (all patterns) |
| `scan_ret5_min` | scalar | 0 code hits (all patterns) |
| `scan_vol_z_min` | scalar | 0 code hits (all patterns) |
| `scan_vol_z_window` | scalar | 0 code hits (all patterns) |
| `scan_vwap_dist_bps_max` | scalar | 0 code hits (all patterns) |
| `session_close_hhmm` | scalar | 0 code hits; `session_open_hhmm` is used but not this key |
| `structure` | dict | 0 cfg reads; `cfg.get("structure")` not called anywhere |
| `structure_manager` | dict | 0 cfg reads; filename match only, not a cfg access |
| `submgr_debounce_ms` | scalar | 0 code hits (all patterns) |
| `time_policy` | dict | 0 cfg reads; method `_blocked_by_time_policy` is coincidence |
| `time_restricted_trading` | scalar | 0 code hits (all patterns) |
| `trade_plan_queue_key` | scalar | 0 code hits (all patterns) |
| `trail_to` | scalar | 0 cfg reads; `trail_to=` keyword arg in detector is not a cfg access |
| `trigger_adx_period` | scalar | 0 code hits (all patterns) |
| `trigger_conditions` | dict | 0 code hits; nested children not accessed via cfg |
| `trigger_default_consecutive_bars` | scalar | 0 code hits (all patterns) |
| `trigger_ema_period` | scalar | 0 code hits (all patterns) |
| `trigger_level_tolerance_pct` | scalar | 0 code hits (all patterns) |
| `trigger_levels` | dict | 0 code hits; nested children not accessed via cfg |
| `trigger_monitoring` | dict | 0 code hits; nested children not accessed via cfg |
| `trigger_patterns` | dict | 0 code hits; nested children not accessed via cfg |
| `trigger_rsi_period` | scalar | 0 code hits (all patterns) |
| `trigger_volume_min_ratio` | scalar | 0 code hits (all patterns) |
| `trigger_vwap_tolerance_pct` | scalar | 0 code hits (all patterns) |
| `use_enhanced_triggers` | scalar | 0 code hits (all patterns) |
| `validation_cache_max_bars_per_symbol` | scalar | 0 code hits (all patterns) |
| `validation_cache_max_symbols` | scalar | 0 code hits (all patterns) |
| `validation_max_time_ms` | scalar | 0 code hits (all patterns) |
| `vol_z_required` | scalar | 0 code hits (all patterns) |
| `volume_persistence_bars` | scalar | 0 code hits (all patterns) |
| `volume_persistence_min_ratio` | scalar | 0 code hits (all patterns) |
| `vwap_min_resistance_distance_atr` | scalar | 0 code hits (all patterns) |
| `vwap_reclaim_min_consecutive` | scalar | 0 code hits (all patterns) |
| `vwap_reclaim_volume_confirmation` | scalar | 0 code hits (all patterns) |
| `vwap_volume_persistence_bars` | scalar | 0 code hits (all patterns) |
| `ws_flush_thread_enabled` | scalar | 0 code hits (all patterns) |
| `ws_max_batch_size` | scalar | 0 code hits (all patterns) |
| `ws_max_retries` | scalar | 0 code hits (all patterns) |
| `ws_reconnect_backoff_secs` | scalar | 0 code hits (all patterns) |
| `ws_subscribe_batch_ms` | scalar | 0 code hits (all patterns) |

Also removed paired `_comment_*` keys where present.

## Verification methodology

For each key, ran 7 checks:
1. `Grep("\"<key>\"")` — quoted reference
2. `Grep("'<key>'")` — single-quoted reference
3. `Grep(".get(\"<key>\"")` — explicit `.get()` call
4. `Grep("[\"<key>\"]")` — bracket access
5. Bare substring grep — variable indirection check
6. Nested-dict parent check: `cfg.get("<key>", {})` or `cfg["<key>"]`
7. Checked `tools/`, `oci/`, `tests/` (non-audit SEARCH_DIRS paths)

Scope: full codebase including `tools/`, `oci/`, `tests/`, `broker/`, `structures/`, `services/`

## Test & smoke results

- `pytest tests/ -v` baseline (pre-A2, commit 75a17a1): 46 failed, 354 passed, 1 skipped (pre-existing failures, not introduced by this change)
- `pytest tests/ -v` post-A2 (after all key deletions + restorations): 46 failed, 354 passed, 1 skipped (no new failures introduced)
- `2024-05-03 smoke` (post all restorations): 34 ENTRY events (matches pre-deletion baseline)
