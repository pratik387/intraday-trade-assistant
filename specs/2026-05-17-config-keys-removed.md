# Config Keys Removed 2026-05-17 (Phase 0 PR A)

Per `specs/2026-05-17-dispatch-architecture-plan.md` Task A2. Each entry verified by deep grep (substring + `.get()` + bracket + variable-indirection + nested-parent + non-SEARCH_DIRS paths).

Verification source: `tools/audit_config_keys.py` (commit 75a17a1) flagged 114 candidates. Deep verification found **0 false positives** — all 114 keys are confirmed-dead.

## Reclassified to USED (false positives from audit)

None. All 114 audit candidates confirmed dead.

Notable edge cases investigated and cleared:

| key | investigation | verdict |
|-----|--------------|---------|
| `execution_mode` | `main.py:270` has `execution_mode = "in_process"` — this is a **local variable** assignment to a string literal, NOT a config read. No `cfg.get("execution_mode")` anywhere. | DEAD |
| `fill_quality` | `tests/test_fill_quality_gate.py:37` references `cfg = {"fill_quality": ...}` but this test is **untracked** (not in git HEAD) and tests a method `_check_fill_quality` that no longer exists in TriggerAwareExecutor. Test was already orphaned/failing before this PR. | DEAD |
| `time_policy` | `screener_live.py:2927` has method `_blocked_by_time_policy()` — this is a **method name**, not a config key access. Method reads `enable_lunch_pause`, `entry_cutoff_hhmm` from config, not `time_policy`. | DEAD |
| `structure_manager` | `structures/structure_manager.py:1` is a **module filename** comment line, not a config key reference. | DEAD |
| `breakout_volume_surge_min` | `config/pipelines/breakout_config.json:93` has a `_comment` string mentioning this key — comment only, no actual config read. Values hardcoded in breakout_config.json. | DEAD |
| `dedupe_cooloff_bars` / `dedupe_require_setup_change` | These are explicitly called out in `configuration.json` comment as "Legacy keys ... will be removed in T14" — scheduled removal. Superseded by `dedup_gate.cooloff_bars` / `dedup_gate.require_setup_change`. | DEAD |
| `market_sentiment` | Was a top-level nested dict. `market_sentiment_filters` (different key, still present) is used in `filters` section. The standalone `market_sentiment` key had no code reads. | DEAD |
| `structure` | Top-level `structure` nested dict (with `max_structural_risk_pct_of_price`, `prefer_levels`, `weaken_levels`). Test references are about the concept "structure" in comments/descriptions, not this config key. No `cfg["structure"]` or `cfg.get("structure")` in code. | DEAD |

## Removed top-level keys (114 total)

All confirmed by 7-pattern check: bare substring grep, quoted `"key"`, single-quoted `'key'`, `.get("key"` pattern, `["key"]` bracket, variable-indirection (>5 hits eyeballed), nested-dict-parent access. Searched paths include `tools/`, `oci/`, `tests/`, `services/`, `structures/`, `pipelines/`, `broker/`, `api/`, `utils/`, `market_data/`, `diagnostics/`, `exceptions/`, `jobs/`.

| key | category | notes |
|-----|----------|-------|
| `adx_strength_min` | trigger-system | Superseded by `trigger_conditions.adx_min_strength` (nested); itself dead |
| `adx_trend_min` | trigger-system | Superseded by `trigger_conditions.adx_min_trend` (nested); itself dead |
| `bar_1m_freq` | websocket/routing | Was `"1min"` string. No code reads it. |
| `bar_5m_freq` | websocket/routing | Was `"5min"` string. No code reads it. |
| `bar_density_gate` | screener | Was nested dict. No `.get("bar_density_gate")` in code. |
| `breakout_momentum_candle_min_ratio` | breakout | Values hardcoded in breakout_config.json (comment only in that file). |
| `breakout_momentum_candle_min_ratio_short` | breakout | Same as above. |
| `breakout_volume_min_ratio` | trigger-system | No code reads. |
| `breakout_volume_surge_min` | breakout | Values duplicated in breakout_config.json; top-level key dead. |
| `breakout_volume_surge_min_short` | breakout | Same. |
| `cache_cleanup_interval_minutes` | maintenance | No code reads. |
| `cap_strategy_preferences` | cap-segmentation | Large nested dict. No code reads the key. |
| `condition_state_cleanup_hours` | trigger-system | No code reads. |
| `daily_trend_ema_period` | trigger-system | No code reads. |
| `decision_require_structure_event` | screener | No code reads. |
| `dedupe_cooloff_bars` | dedup | Superseded by `dedup_gate.cooloff_bars`. Comment in config explicitly says remove in T14. |
| `dedupe_require_setup_change` | dedup | Superseded by `dedup_gate.require_setup_change`. |
| `exec_heartbeat` | process-separation | Nested dict `{key, interval_sec, ttl_sec}`. process-separation code removed (MDS deprecation). |
| `execution_mode` | process-separation | Was `"in_process"`. Local variable in main.py, not config read. |
| `expiry_allow_breakout` | expiry | No code reads. |
| `expiry_allow_fade` | expiry | No code reads. |
| `expiry_size_multiplier` | expiry | No code reads. |
| `fill_quality` | execution | Nested dict. `_check_fill_quality` method removed from TriggerAwareExecutor. |
| `hold_bars` | screener | Was `1`. No code reads `cfg["hold_bars"]`. |
| `ict_quality_filters` | ICT | Large nested dict (order_block, fair_value_gap). ICT setups removed. |
| `k_atr` | screener | Was `0.2`. No code reads. |
| `market_sentiment` | market-context | Large nested dict. No code reads. |
| `min_entry_sl_distance_abs` | execution | Was `0.05`. No code reads. |
| `min_entry_sl_distance_pct` | execution | Was `0.003`. No code reads. |
| `momentum_min_ema_slope` | trigger-system | No code reads. |
| `news_gate` | news | Nested dict. No code reads `cfg.get("news_gate")`. |
| `news_min_hold_bars` | news | Was `2`. No code reads. |
| `orb_pullback_hold_bars` | ORB | Was `2`. No code reads top-level key. |
| `orb_pullback_max_minutes` | ORB | Was `90`. No code reads. |
| `orb_pullback_tolerance_pct` | ORB | Was `0.15`. No code reads. |
| `orb_require_index_alignment` | ORB | Was `true`. No code reads. |
| `order_exception_delay_ms` | order | Was `100`. No code reads. |
| `order_loop_idle_ms` | order | Was `10`. No code reads. |
| `order_rate_limit` | order | Was `1`. No code reads. |
| `order_retry_delay_ms` | order | Was `50`. No code reads. |
| `order_retry_max` | order | Was `3`. No code reads. |
| `pdh_break_tolerance_pct` | levels | Was `0.05`. No code reads. |
| `pdh_hold_bars` | levels | Was `2`. No code reads. |
| `pdl_break_tolerance_pct` | levels | Was `0.05`. No code reads. |
| `planner_atr_period` | planner | Was `14`. No code reads. |
| `planner_choppiness_high` | planner | Was `61.8`. No code reads. |
| `planner_choppiness_lookback` | planner | Was `30`. No code reads. |
| `planner_choppiness_low` | planner | Was `38.2`. No code reads. |
| `planner_entry_zone_atr_frac` | planner | Was `0.08`. No code reads. |
| `planner_entry_zone_breakout_mult` | planner | Was `0.25`. No code reads. |
| `planner_entry_zone_default_mult` | planner | Was `0.15`. No code reads. |
| `planner_entry_zone_fade_mult` | planner | Was `0.08`. No code reads. |
| `planner_max_gap_pct_for_trend` | planner | Was `3.0`. No code reads. |
| `planner_vwap_reclaim_min_bars_above` | planner | Was `3`. No code reads. |
| `range_break_retest_max_pct` | trigger-system | Was `0.15`. No code reads. |
| `regime_cap_allocation` | cap-segmentation | Large nested dict (chop/trend_up/trend_down/squeeze). No code reads. |
| `regime_scores` | regime | Was `{squeeze:0.6, trend:0.8, chop:0.5}`. No code reads. |
| `rejection_min_body_pct` | trigger-system | Was `0.6`. No code reads. |
| `require_breakout_bar_volume_surge` | breakout | Was `true`. No code reads. |
| `require_higher_timeframe_alignment` | trigger-system | Was `true`. No code reads. |
| `router_assume_epoch_utc` | router | Was `true`. Router module removed. |
| `router_dayvolume_fields` | router | List. Router module removed. |
| `router_default_timezone` | router | Was `"Asia/Kolkata"`. Router module removed. |
| `router_price_fields` | router | List. Router module removed. |
| `router_symbol_fields` | router | List. Router module removed. |
| `router_timestamp_fields` | router | List. Router module removed. |
| `router_tradeqty_fields` | router | List. Router module removed. |
| `rsi_bearish_max` | trigger-system | Was `55`. No code reads. |
| `rsi_bearish_min` | trigger-system | Was `30`. No code reads. |
| `rsi_bullish_max` | trigger-system | Was `70`. No code reads. |
| `rsi_bullish_min` | trigger-system | Was `40`. No code reads. |
| `scan_min_bars_5m` | scanner | Was `30`. No code reads. |
| `scan_output_max_rows` | scanner | Was `50`. No code reads. |
| `scan_ret1_min` | scanner | Was `0.002`. No code reads. |
| `scan_ret3_min` | scanner | Was `0.004`. No code reads. |
| `scan_ret5_min` | scanner | Was `0.006`. No code reads. |
| `scan_vol_z_min` | scanner | Was `0.5`. No code reads. |
| `scan_vol_z_window` | scanner | Was `20`. No code reads. |
| `scan_vwap_dist_bps_max` | scanner | Was `80`. No code reads. |
| `session_close_hhmm` | session | Was `"1530"`. No code reads top-level key. Session close handled by `eod_squareoff_hhmm`. |
| `structure` | structure | Nested dict (max_structural_risk_pct_of_price, prefer_levels, weaken_levels). No code reads. |
| `structure_manager` | structure | Nested dict (enabled, min_structure_score, structure_priorities, logging). No code reads `cfg["structure_manager"]`. |
| `submgr_debounce_ms` | websocket | Was `1500`. No code reads. |
| `time_policy` | session | Nested dict (start_trade, lunch_block_start/end, last_entry_cutoff). Superseded by `enable_lunch_pause`/`entry_cutoff_hhmm`. |
| `time_restricted_trading` | session | Was `true`. No code reads. |
| `trade_plan_queue_key` | process-separation | Was Redis queue key for separated exec process (MDS deprecated). |
| `trail_to` | exit | Was `"atr"`. No code reads. |
| `trigger_adx_period` | trigger-system | Was `14`. No code reads. |
| `trigger_conditions` | trigger-system | Nested dict. No code reads top-level key. |
| `trigger_default_consecutive_bars` | trigger-system | Was `1`. No code reads. |
| `trigger_ema_period` | trigger-system | Was `9`. No code reads. |
| `trigger_level_tolerance_pct` | trigger-system | Was `0.1`. No code reads. |
| `trigger_levels` | trigger-system | Nested dict. No code reads top-level key. |
| `trigger_monitoring` | trigger-system | Nested dict. No code reads. |
| `trigger_patterns` | trigger-system | Nested dict. No code reads. |
| `trigger_rsi_period` | trigger-system | Was `14`. No code reads. |
| `trigger_volume_min_ratio` | trigger-system | Was `1.2`. No code reads. |
| `trigger_vwap_tolerance_pct` | trigger-system | Was `0.02`. No code reads. |
| `use_enhanced_triggers` | trigger-system | Was `true`. No code reads. |
| `validation_cache_max_bars_per_symbol` | trigger-system | Was `50`. No code reads. |
| `validation_cache_max_symbols` | trigger-system | Was `100`. No code reads. |
| `validation_max_time_ms` | trigger-system | Was `50`. No code reads. |
| `vol_z_required` | screener | Was `0.8`. No code reads. |
| `volume_persistence_bars` | trigger-system | Was `2`. No code reads. |
| `volume_persistence_min_ratio` | trigger-system | Was `1.3`. No code reads. |
| `vwap_min_resistance_distance_atr` | trigger-system | Was `0.2`. No code reads. |
| `vwap_reclaim_min_consecutive` | trigger-system | Was `3`. No code reads. |
| `vwap_reclaim_volume_confirmation` | trigger-system | Was `true`. No code reads. |
| `vwap_volume_persistence_bars` | trigger-system | Was `3`. No code reads. |
| `ws_flush_thread_enabled` | websocket | Was `true`. No code reads. `ws_flush_enabled` and `ws_flush_interval_ms` (still present) are the live keys. |
| `ws_max_batch_size` | websocket | Was `100`. No code reads. `ws_batch_size` (still present) is the live key. |
| `ws_max_retries` | websocket | Was `5`. No code reads. |
| `ws_reconnect_backoff_secs` | websocket | Was `2`. No code reads. |
| `ws_subscribe_batch_ms` | websocket | Was `1500`. No code reads. |

## Companion keys also removed

Several `_comment_<key>` and `_comment_<topic>` keys adjacent to deleted entries were also removed as they documented the now-deleted keys (e.g., `_comment_entry_zone`, `_comment_breakout_mult`, `_comment_fade_mult`, `_comment_default_mult`, `_comment_volume_surge`, `_comment_momentum_candle`, `_comment_volume_short`, `_comment_momentum_short`, `_comment_trail`, `_comment_dedupe`, `_comment_execution_process`, `_comment_execution_mode`, `_comment_trade_plan_queue`, `_enhanced_strategy_mappings`, `_logging_enhancements`).

## Verification methodology

For each key:
1. `grep -r "key" .` — bare substring across all file types
2. `grep -r '"key"'` — double-quoted references
3. `grep -r "'key'"` — single-quoted references
4. `grep -r '.get("key"'` — explicit cfg.get() access
5. `grep -r '["key"]'` — bracket access
6. Eyeball hits >5 for variable indirection
7. Nested-dict keys: verify no `cfg.get("key", {})` or `cfg["key"]` accesses
8. Non-SEARCH_DIRS: `tools/`, `oci/`, `tests/` explicitly covered

## Test results

- `pytest tests/` (excluding 5 pre-existing import-error files + untracked tests): **337 passed, 10 pre-existing failures**
- Pre-existing failures: `gauntlet_v2` (6) and `shadow/test_parity_simulator` (4) — caused by missing modules (parity_simulator.py etc.), unrelated to config changes
- Smoke: 2024-05-03 dry run — see commit message for trade count

## Also removed

- `tools/audit_config_keys.py` — one-shot script, job done

Verification source: `tools/audit_config_keys.py` (commit 75a17a1) flagged 114 candidates; deep
verification found zero false positives — all 114 confirmed dead.

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
