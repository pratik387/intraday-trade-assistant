# Backtest bar-fetch narrowing — Design

**Date:** 2026-05-21
**Status:** Design — pending implementation
**Author:** brainstorm session 2026-05-21
**Scope:** `services/screener_live.py` — `_run_5m_scan` build_df5_map loop

## Problem

In backtest mode, every 5-minute bar iteration of `_run_5m_scan` builds `df5_by_symbol` by looping over **all 2,266 `core_symbols`** and calling `_get_precomputed_5m` for each. The downstream dispatch path narrows to `tag_map.active_symbols()` (typically 200-1500), so the upstream loop is wasted compute.

Concretely, between 10:30 and 14:55 (59 bars per day), only `below_vwap_volume_revert_long` is active with a ~432-symbol universe, but the loop still slices precomputed bars for all 2,266. Observed per-bar wallclock in the smoke is ~4-5s; the dominant cost is the 2,266× slice + validate_df.

Live/paper mode is already narrowed at `services/screener_live.py:1357`:

```python
# Narrow to tag_map active symbols (~99) instead of all core_symbols (~1500).
# Falls back to full universe if tag_map is empty (pre-09:15).
_univ = self._tag_map.active_symbols()
fetch_symbols = sorted(_univ & set(self.core_symbols)) if _univ else list(self.core_symbols)
```

Backtest is the gap.

## Goal

Apply the same narrowing pattern to the backtest build_df5_map loop so backtest wallclock per bar mirrors live behavior. Preserve correctness for 5-arg universe builders (`gap_fade_universe`, `long_panic_gap_down_universe`) that iterate `df5_by_symbol` to build their universes.

## Non-goals

- Change live/paper behavior (already narrowed).
- Modify the dispatch path itself (already correctly narrows at line 2174).
- Pre-compute calendars or restructure the TransitionCalendar.
- Touch the exit_executor (uses tick subscription separately; not on the bar-fetch path).
- Reduce backtest data storage (the precomputed feathers stay full-universe; we just stop iterating them all).

## Lazy-build constraint

Universe builders fire from inside `_run_dispatch_path` (line 2143), which is called AFTER the build_df5_map loop completes. So `tag_map.active_symbols()` at the loop reflects the **previous bar's** state. Two-bar review:

| Bar T | tag_map state at loop | Builders firing in this bar's `_run_dispatch_path` | df5_by_symbol needs |
|---|---|---|---|
| 09:15 (first bar) | EMPTY (no prior call) | session_start (3-arg, no df5) + bar:09:15 (5-arg: gap_fade, long_panic) | **FULL** universe — 5-arg builders iterate it |
| 09:20 | session_start + bar:09:15 builders' universes already in tag_map | None (no later bar:HH:MM triggers in current registry) | active union only |
| 10:30 | All universes still in tag_map | None | active union only |
| 14:55 onward | Active windows closed → empty active_symbols → SCAN_SKIPPED | None | n/a |

Registry audit (2026-05-21) confirms: only `gap_fade_short` and `long_panic_gap_down` use 5-arg universe builders, both with `universe_trigger="bar:09:15"`. No 5-arg builders fire after 09:15.

So the narrowing is safe **as long as the fallback to full `core_symbols` activates when tag_map is empty** — exactly what the live pattern already does.

## Architecture

Single change in `services/screener_live.py`. Replace `for s in self.core_symbols:` (line ~1430) with a narrow_set loop that mirrors the live pattern.

### Current code (line 1426-1442)

```python
try:
    # Build df5_by_symbol from enriched data (API for paper, precomputed for backtest)
    with perf("scan", "build_df5_map", n_core=len(self.core_symbols)):
        df5_by_symbol: Dict[str, pd.DataFrame] = {}
        for s in self.core_symbols:
            # Data source:
            #  Paper/live: api_df5_cache (V3 Intraday API + enrichment)
            #  Backtest:   _precomputed_5m (enriched feather cache)
            if s in api_df5_cache:
                df5 = api_df5_cache[s]
            elif self._precomputed_5m and s in self._precomputed_5m:
                df5 = self._get_precomputed_5m(s, now, self.cfg.screener_store_5m_max)
            else:
                continue
            if validate_df(df5, min_rows=min_bars_for_processing):
                df5_by_symbol[s] = df5
```

### New code

```python
try:
    # Build df5_by_symbol from enriched data (API for paper, precomputed for backtest).
    # Narrow to active universes when tag_map is populated (matches live behavior at
    # line 1357). Falls back to full core_symbols when tag_map is empty (pre-09:15) to
    # preserve correctness for 5-arg universe builders that iterate df5_by_symbol.
    _univ_active = self._tag_map.active_symbols()
    narrow_set = (
        _univ_active & set(self.core_symbols)
        if _univ_active
        else set(self.core_symbols)
    )
    with perf("scan", "build_df5_map", n_core=len(narrow_set)):
        df5_by_symbol: Dict[str, pd.DataFrame] = {}
        for s in narrow_set:
            # Data source:
            #  Paper/live: api_df5_cache (V3 Intraday API + enrichment)
            #  Backtest:   _precomputed_5m (enriched feather cache)
            if s in api_df5_cache:
                df5 = api_df5_cache[s]
            elif self._precomputed_5m and s in self._precomputed_5m:
                df5 = self._get_precomputed_5m(s, now, self.cfg.screener_store_5m_max)
            else:
                continue
            if validate_df(df5, min_rows=min_bars_for_processing):
                df5_by_symbol[s] = df5
```

Changes:
1. Compute `narrow_set` from `tag_map.active_symbols()` with full-universe fallback (lines added before the loop).
2. Replace `for s in self.core_symbols:` with `for s in narrow_set:`.
3. Update the `perf` metric `n_core` arg to reflect actual loop size (not just metadata).

The data-source dispatch (`api_df5_cache` vs `_precomputed_5m`) and `validate_df` filter are unchanged.

## Components

### `services/screener_live.py:_run_5m_scan`
- Reads `self._tag_map.active_symbols()` (existing).
- Reads `self.core_symbols` (existing).
- Computes `narrow_set` as intersection or full-universe fallback (new ~5 lines).
- Loops over `narrow_set` instead of `core_symbols` (1-line change).

### Dependencies (unchanged but called out)
- `services/dispatch/tag_map.py:TagMap.active_symbols()` — already returns union of currently-open universes (verified in earlier audit).
- `services/dispatch/setup_registry.py:SetupRegistry` — registers each setup's `universe_trigger` and `universe_builder`. Determines when universes get built.
- `services/dispatch/planner.py` — used inside `_run_dispatch_path`. Filters per-symbol tags to OPEN-window setups only. Independent of this change.

## Data flow

```
Bar T:
  ┌─ Read self._tag_map.active_symbols()                            ← new
  │      └─ If empty: narrow_set = self.core_symbols  (first bar)
  │      └─ Else:     narrow_set = active_symbols ∩ core_symbols
  │
  ┌─ For s in narrow_set:                                           ← changed
  │      ├─ api_df5_cache.get(s)  OR
  │      ├─ self._precomputed_5m + _get_precomputed_5m(s)
  │      └─ validate + add to df5_by_symbol
  │
  ├─ _compute_orb_levels_once(now, df5_by_symbol)                   ← unchanged but reduced input
  │
  └─ _run_dispatch_path(now, df5_by_symbol, ...)                    ← unchanged
       ├─ Process calendar events (build universes, open/close windows)
       ├─ active_syms = self._tag_map.active_symbols()
       ├─ active_syms_with_data = active_syms & df5_by_symbol.keys() ← still correct
       └─ Dispatch per-symbol with active_tags(sym)
```

## Error handling

No new error paths. Existing fallbacks remain:
- If `narrow_set` is empty after intersection (active universes contain no `core_symbols` match): the loop body executes zero times, `df5_by_symbol` ends up empty, the existing `if not df5_by_symbol: return` (line 1443-1447) handles it silently.
- If `_get_precomputed_5m` raises or returns invalid df: existing `continue` path.

## Testing

### Correctness regression (must pass)

Run backtest 2026-04-21 (the smoke date) before and after the change. Compare `events_decisions.jsonl`:

```bash
# Before:
python main.py --dry-run --session-date 2026-04-21
cp logs/backtest_*/events_decisions.jsonl /tmp/before.jsonl

# Apply change.

# After:
python main.py --dry-run --session-date 2026-04-21
diff /tmp/before.jsonl logs/backtest_*/events_decisions.jsonl
```

**Pass criterion:** byte-identical events_decisions.jsonl. Any difference means the narrowing dropped a symbol that should have been dispatched.

### Performance verification (expected)

Compare `SCANNER_COMPLETE | data_loaded=X/Y ... TIME: Zs` lines before/after.

Expected:
- 09:15 bar: unchanged (~2266, ~5s).
- 09:20-14:55 bars: data_loaded drops from 2266 to active-union size; TIME drops 50-80%.
- 14:55-15:25: unchanged (already SCAN_SKIPPED).

### Multi-day spot-check

Run a different date with different active-universe composition (e.g., 2025-09-11 from the OOS window, 2026-01-23 from HO). Same byte-diff and timing checks. Catches dates where calendar event ordering differs.

### Live-mode unaffected check

The change only modifies code that runs when `_precomputed_5m` is the data source. In live, `api_df5_cache` is already narrowed at line 1357, so the new narrow_set ends up the same as before (narrow_set ⊇ api_df5_cache keys → loop body runs ≤ narrow_set times → same dict built). Verify by inspection only; no live test required.

## Rollout

1. Implement the 1-file change.
2. Run pytest on `tests/services/` and `tests/structures/` (existing suites; should not be affected).
3. Run the byte-diff regression on 2026-04-21.
4. If clean, run 3 multi-day spot-checks (covering Disc, OOS, HO windows).
5. Commit. No config change. No registry change.

## Out of scope (deferred)

- **Open-position bar inclusion in backtest:** in live, exit_executor uses tick subscription. In backtest, the simulator walks 5m bars for SL/T1/T2 exits via `self.broker.get_ltp_with_level(sym, check_level=...)` at `services/execution/exit_executor.py:485` — the broker abstraction owns its own intrabar data source, independent of the screener's `df5_by_symbol`. **Verified during design** (2026-05-21): narrowing the build_df5_map loop does NOT affect exit tracking. No implementation-time check needed.
- **Live-mode narrowing review:** out of scope — already correct.
- **Stage-0 energy scan skip when only static-cap setups are active:** different optimization, different code path. Separate spec if pursued.
