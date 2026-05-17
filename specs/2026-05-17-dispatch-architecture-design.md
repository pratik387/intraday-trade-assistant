# Per-Detector Dispatch Architecture + Source-of-Truth Cleanup

**Date:** 2026-05-17
**Status:** Design — pending user review
**Authors:** pratik387 + Claude

## Motivation

The current structure-detection pipeline does redundant work and has fragmented sources of truth.

**Performance pain (live):** Each 5-min bar runs `for symbol in 739: for detector in 8: detector.detect()` = 5,912 evaluations per bar. ~99% return "not my pattern" because each detector's first line re-applies the cap_segment/universe filter the universe-builder already applied. Structure phase consumes 11-14s per bar; the live timer runs every 10s, so scans drift. Most of the day has 0-2 active detectors (windowed setups), not 8 — but the dispatch loop pays the full cross-product cost regardless.

**Correctness pain (drift bugs):** Setup metadata lives in 7 places, 4 of which are duplicated. Adding a setup requires synchronizing `plan_orchestrator.py:_DETECTOR_REGISTRY`, `main_detector.py:detector_configs`, `screener_live.py:_universe_union()`, `screener_live.py` lazy-build if-blocks, `configuration.json`, `setup_categories.py`, and (for OCI) `sub8_oci_overrides.json`. The `long_panic_gap_down` setup was shipped with one of these missing — required `d0caa7d fix(plan_orchestrator)` after the bug was caught.

**Config rot:** `configuration.json` has 1,098 keys / 217 functional top-level sections; an automated audit identifies ~137 candidates as dead (no `.get("KEY")` reference anywhere in code). The audit has known false negatives, but the SCALE is real: most of the file is sediment from sub-7/sub-8 experiments.

**API-fetch waste (post-2026-04 parity fix):** Live now fetches 5m bars via REST per bar at 40 RPS over the universe-union (~573-739 syms). Idle windows (10:35-11:00, 15:05+) fetch for symbols no detector cares about — wasted API budget that delays subsequent scans.

## Design at a glance

Replace `(symbol × all-detectors)` dispatch with **symbol-tagged dispatch**: each symbol carries a set of active-detector tags. Tags evolve at known calendar timestamps (universe builds + active-window open/close), not per bar. Workers iterate only tagged detectors per symbol. The fetch set shrinks to the active-tag set, with on-demand backfill at re-entry.

Single source of truth: a new `SetupRegistry` loaded from `configuration.json`. All other "registries" (dual detector maps, hardcoded if-blocks, OCI override file) deleted.

Ships in three phases, each independently mergeable.

## Scope

**In scope:**
- Cleanup pass on `configuration.json` (Phase 0)
- Fold `sub8_oci_overrides.json` into a `mode_profiles` block inside `configuration.json`; delete the override file + merge tooling
- Delete dual detector registries; replace with one `SetupRegistry` module
- Replace per-symbol dispatch loop with per-detector-tag dispatch
- Trim per-bar API fetch to the active-tag set; add backfill at re-entry
- Strip dead cap_segment filters inside detectors (single source = the universe builder)
- Big-bang rollout validated by 3-year OCI backtest diff

**Out of scope:**
- Worker count tuning (`structure_detection_workers_local`) — stays at current values
- Gates / orchestrator / executor / exit logic — untouched
- Backtest cloud orchestration (Modal) — uses the same code, inherits the speedup
- New detector additions
- WebSocket plumbing (already lean; only enqueued/traded symbols subscribe)

## Phase 0 — Source-of-truth cleanup (PREREQUISITE)

Phase 0 ships first as two standalone PRs (PR A, PR B). No new dispatch code until both merge. Each PR is reversible. A third cleanup PR (PR C) is bundled with Phase 1 because it depends on the new `SetupRegistry`.

### PR A — configuration.json audit

For every functional top-level key in `configuration.json`:

1. Grep `"<key>"`, `'<key>'`, `.get("<key>"`, `["<key>"]`, and bare `<key>` references across `services/`, `structures/`, `pipelines/`, `gates/`, `tools/`, `oci/`, `main.py`
2. Check `config/pipelines/*.json` cross-references
3. Verify each candidate-dead key in code (humans validate; grep is a starting point, not a verdict — the audit caught one false negative for `entry_cutoff_hhmm`)
4. Delete confirmed-dead keys
5. Record the deletion list in `docs/config_keys_removed_2026-05-17.md` so a future engineer can `git log` the rationale

Target deletion: ~100 keys (subject to verification). No behavior change expected — these keys are dead by definition.

### PR B — fold sub8_oci_overrides.json into mode_profiles

`sub8_oci_overrides.json` (47 lines) contains 5 real OCI deltas + 42 lines of duplicates and comment noise. Inline the 5 deltas into `configuration.json`:

```json
{
  "mode": "production",
  "mode_profiles": {
    "production": {},
    "oci_research": {
      "entry_cutoff_hhmm": "15:25",
      "last_scan_hhmm": "15:25",
      "eod_squareoff_hhmm": "15:25",
      "max_trades_per_cycle": 10000,
      "gate_input_logging": { "enabled": true }
    }
  }
}
```

The selected profile's keys override the base config at load time. Mode selection: top-level `mode` key, overridable via env var `RUN_MODE=oci_research` (set by OCI entrypoint).

Deletions in this PR:
- `config/sub8_oci_overrides.json`
- `tools/apply_oci_override.py`
- The merge call in `oci/docker/entrypoint.py` (lines 112-153, 698-705)

Test: OCI Discovery run with `RUN_MODE=oci_research` produces a `trade_report.csv` byte-identical to a pre-PR OCI run (modulo `run_id`/timestamps).

### PR C (ships with Phase 1) — delete dead registries and hardcoded blocks

Listed here for completeness; technically bundled with the Phase 1 PR because it depends on `SetupRegistry`. Files touched:

- `services/plan_orchestrator.py:99-127` — delete `_DETECTOR_REGISTRY` + `ACTIVE_SETUPS`; replace references with `SetupRegistry`
- `structures/main_detector.py:62-104` — delete `detector_configs` list + ICT dead code (`ict_derived_setups`, `ict_base_config`, ICT merge if-block) + stale "only 2 detectors" comment
- `services/screener_live.py:741-759` — delete `_universe_union()`
- `services/screener_live.py:1438-1444` — delete universe-union if-blocks (inline construction of `universe` set)
- `services/screener_live.py:1485-1541` — delete hardcoded lazy-build if-blocks for gap_fade / long_panic / circuit_release

## Phase 1 — SetupRegistry and per-detector-tag dispatch

### New module: `services/dispatch/setup_registry.py`

```python
@dataclass(frozen=True)
class SetupSpec:
    name: str                           # "gap_fade_short"
    enabled: bool
    detector_class: type                # imported from detector_class config field
    universe_builder: Callable          # imported from universe_builder config field
    universe_trigger: Trigger           # SessionStart() | BarTime(time(9,15))
    active_window: tuple[time, time]    # inclusive (start, end)
    raw_config: dict                    # the full setup dict from configuration.json

class SetupRegistry:
    def __init__(self, root_config: dict): ...
    def enabled(self) -> list[SetupSpec]: ...
    def get(self, name: str) -> SetupSpec: ...
    def detector_for(self, setup_name: str) -> BaseStructure: ...
    def validate(self) -> None:
        """Fail-fast on missing detector_class, universe_builder, or invalid window."""
```

Loaded once at startup. The single source of truth for "what setups exist, what they need, what time they need it."

### New module: `services/dispatch/transition_calendar.py`

Derived from `SetupRegistry` at startup. A sorted list of events:

```python
class TransitionEvent:
    at: time                            # IST-naive
    kind: Literal["build_universe", "open_window", "close_window"]
    setup: str

class TransitionCalendar:
    def events_in(self, after: time, until: time) -> list[TransitionEvent]: ...
```

For an 8-setup config, the calendar has ~16-20 events spread across the session. Walked at each bar close to evolve `TagMap` state.

### New module: `services/dispatch/tag_map.py`

```python
class TagMap:
    def add_universe(self, setup: str, syms: set[str]) -> None: ...   # tag syms as eligible (inactive)
    def open_window(self, setup: str) -> None: ...                     # flip active flag on
    def close_window(self, setup: str) -> None: ...                    # flip active flag off
    def active_tags(self, sym: str) -> set[str]: ...                   # detector names active for sym now
    def active_symbols(self) -> set[str]: ...                          # union over all syms with ≥1 active tag
    def dormant_since(self, sym: str) -> Optional[datetime]: ...       # for backfill decision
```

In-memory state. Mutated only by `DispatchPlanner` walking the calendar; never read or written by detectors.

### New module: `services/dispatch/fetch_scope.py`

```python
class FetchScopeManager:
    def fetch_set(self, bar_ts: datetime, tag_map: TagMap) -> set[str]: ...
    def is_backfill_needed(self, sym: str, df5_by_symbol: dict) -> bool: ...
```

Returns the union of currently-active-tagged symbols. For symbols newly activated (re-entry after dormancy), `is_backfill_needed` returns `True` if the existing `df_5m` does not include all bars since session open. Backfill uses one REST call with `start_ts = session_open`.

### New module: `services/dispatch/planner.py`

```python
@dataclass
class Batch:
    items: list[tuple[str, pd.DataFrame, dict, set[str]]]   # (sym, df5, levels, tags)

class DispatchPlanner:
    def plan(
        self,
        bar_ts: datetime,
        tag_map: TagMap,
        df5_by_symbol: dict[str, pd.DataFrame],
        levels_by_symbol: dict[str, dict],
    ) -> list[Batch]: ...
```

Each `Batch` is ≤50 symbol-items. Workers receive a Batch, iterate `for sym, df5, levels, tags in batch.items`, build a `MarketContext`, then iterate `for det_name in tags: detector_instance.detect(ctx)`.

### Modified module: `services/screener_live.py`

`_run_5m_scan(bar_ts)` body rewritten:

```python
def _run_5m_scan(self, bar_ts):
    # 1. Walk calendar; evolve tag map
    for ev in self.transition_calendar.events_in(self._last_scan_ts, bar_ts):
        if ev.kind == "build_universe":
            spec = self.registry.get(ev.setup)
            syms = spec.universe_builder(...)
            self.tag_map.add_universe(ev.setup, syms)
        elif ev.kind == "open_window":
            self.tag_map.open_window(ev.setup)
        elif ev.kind == "close_window":
            self.tag_map.close_window(ev.setup)

    # 2. Check if anything to do
    active = self.tag_map.active_symbols()
    if not active:
        logger.info("SCAN_SKIPPED | no active detectors at bar %s", bar_ts)
        self._last_scan_ts = bar_ts
        return

    # 3. Fetch only what we need (with backfill)
    fetch_syms = self.fetch_scope.fetch_set(bar_ts, self.tag_map)
    for sym in fetch_syms:
        if self.fetch_scope.is_backfill_needed(sym, self.df5_by_symbol):
            self.api_fetch_history(sym, start_ts=session_open)
        else:
            self.api_fetch_single_bar(sym, bar_ts)

    # 4. Compute features for active syms only
    features = compute_bar_features(self.df5_by_symbol, active, bar_ts, self.levels_by_symbol)

    # 5. Plan and dispatch
    plan = self.planner.plan(bar_ts, self.tag_map, self.df5_by_symbol, self.levels_by_symbol)
    futures = [self.executor.submit(dispatch_worker_batch, batch) for batch in plan]
    events = []
    for fut in futures:
        events.extend(fut.result())

    # 6. Downstream (unchanged)
    self.gates.process(events)
    self._last_scan_ts = bar_ts
```

### Modified module: `structures/main_detector.py`

`MainDetector` class deleted entirely. Replaced by a stateless worker entry point:

```python
# In worker process. Module-level cache populated on first call.
_detector_cache: dict[str, BaseStructure] = {}

def dispatch_worker_batch(batch: Batch) -> list[StructureEvent]:
    out = []
    for sym, df5, levels, tags in batch.items:
        ctx = build_market_context(sym, df5, levels)
        for det_name in tags:
            det = _detector_cache.get(det_name)
            if det is None:
                spec = _registry_cache.get(det_name)
                det = spec.detector_class(spec.raw_config)
                _detector_cache[det_name] = det
            analysis = det.detect(ctx)
            out.extend(analysis.events)
            # accept/reject JSONL emission unchanged
    return out
```

### Modified module: each detector in `structures/`

Strip the first-line cap_segment / universe filter:

```python
# Before:
if context.cap_segment not in self.allowed_caps:
    return _empty(...)

# After:
# (deleted — universe builder guarantees only qualifying symbols reach here)
```

The `allowed_cap_segments` config key stays — it's still used by the universe builder. Only the redundant runtime check is removed.

### Config schema additions

Per setup in `configuration.json`:

```json
"gap_fade_short": {
  "enabled": true,
  "detector_class": "structures.gap_fade_short_structure.GapFadeShortStructure",
  "universe_builder": "services.setup_universe.gap_fade_universe",
  "universe_trigger": "bar:09:15",
  "active_window_start": "09:15",
  "active_window_end": "09:30",
  ...existing keys
}
```

`detector_class` and `universe_builder` are fully-qualified import paths — no naming-convention magic. Startup validation imports both; missing or wrong path errors at startup, not mid-session.

`universe_trigger` accepted values: `"session_start"` or `"bar:HH:MM"`. Other values reject at startup.

### Backfill behavior

When a symbol becomes newly active after dormancy:

- `tag_map.dormant_since(sym)` returns the last-active bar timestamp
- `fetch_scope.is_backfill_needed(sym, df5)` returns `True` if `df5[sym].index[-1] < bar_ts - 5min`
- Backfill fetch: one REST call with `start_ts = session_open`, returns all bars since open in one response
- Cost: 1 extra API call at re-entry vs (re-entry_bar - dormant_bar)/5 = up to 18 fetches saved during dormancy. Net 17:1 favorable

### Data lifecycle summary

| Element | Granularity | Lifecycle |
|---|---|---|
| WebSocket subs | Per trade | Add on enqueue, drop on close (unchanged) |
| REST 5m fetch scope | Per bar | = active-tag set + backfill for re-entries |
| Levels (PDH/PDL/PDC) | Per session | Computed at warmup, immutable |
| ORB (ORH/ORL) | Per session | Computed at 09:30, immutable |
| Per-bar features (VWAP/RSI/vol_ratio) | Per bar | Computed only for active-tag symbols |
| TagMap | Per session, mutated at calendar events | In-memory |
| Detector instances | Per worker process lifetime | Cached on first use |

## Phase 2 — Verification and rollout

### Verification

OCI 3-year backtest is the gold standard. Process:

1. Lock in a baseline: re-run the existing OCI Discovery on current `main` branch. Capture `trade_report.csv` per session (489 sessions).
2. Apply Phase 0 + Phase 1 to a feature branch. Re-run OCI Discovery on the same 489 sessions.
3. Per-session diff:
   - Set of `trade_id`s should match
   - Per `trade_id`: `entry_ts`, `entry_price`, `exit_ts`, `exit_price`, `last_exit_reason`, `realized_pnl` should match
4. Acceptable differences:
   - **None.** This is a refactor, not a behavior change. Any diff is a bug.
5. If diffs appear: triage (probably a missed universe-builder hookup or a calendar-event timing error), fix, re-run.

### Smoke validation

Before OCI:

- One-day backtest (`python main.py --dry-run --session-date 2024-05-03`) on `main` vs feature branch → diff `trade_report.csv`
- Three smoke days covering different regimes (trend, range, gap): 2024-05-03, 2024-09-10, 2024-11-25

### Rollout

Sequence:
1. Phase 0 PR A (config audit) → merge after self-review
2. Phase 0 PR B (mode_profiles fold) → merge after OCI smoke confirms no behavior change
3. Phase 1 PR (dispatch refactor, includes PR C cleanup) → merge after full OCI Discovery passes byte-diff
4. Live cut-over on a low-volatility trading day, with paper-trading run beforehand on the same day

If any post-cut-over anomaly: revert to the prior commit (the feature is a pure code refactor; no schema migration to undo).

## Error handling

| Failure | Detection | Response |
|---|---|---|
| Setup config missing `detector_class` | Startup validation | Hard exit with message; cannot run |
| Setup config missing `universe_builder` | Startup validation | Hard exit |
| `universe_trigger` malformed | Startup validation | Hard exit |
| `detector_class` import fails | Startup | Hard exit |
| Universe builder raises mid-session | Per-event try/except in `_run_5m_scan` calendar walk | Log + skip the build event; that setup is dormant for the rest of the day; alert via existing log channel |
| Worker batch raises | `executor.submit().result()` try/except | Log + drop the batch; continue with other batches |
| Backfill API fails | API client retry policy (existing) | If unavailable, log + skip the symbol this bar; retry next bar |
| TagMap state corrupt (negative ref count, etc.) | Defensive asserts inside TagMap methods | Hard exit (would indicate a code bug, not data issue) |

## Testing strategy

| Layer | Tests |
|---|---|
| Unit | `tests/dispatch/test_setup_registry.py` — valid/invalid configs, import resolution |
| Unit | `tests/dispatch/test_transition_calendar.py` — event ordering, edge cases (overlapping events at same time) |
| Unit | `tests/dispatch/test_tag_map.py` — add_universe, open/close window, active_symbols correctness |
| Unit | `tests/dispatch/test_fetch_scope.py` — fetch set computation, backfill detection |
| Unit | `tests/dispatch/test_planner.py` — batch chunking, empty plan when no active tags |
| Integration | `tests/dispatch/test_dispatch_e2e.py` — feed a synthetic bar stream, assert dispatch matches expected (det, sym) pairs |
| Regression | Existing detector tests in `tests/structures/` — must still pass (no detector logic change) |
| End-to-end | OCI 3-year backtest byte-diff (gold standard) |

## Open questions

None at this stage — all major decisions captured during brainstorming. If the OCI byte-diff surfaces unexpected differences, this design will need revision and re-review.

## Migration risks and mitigations

| Risk | Mitigation |
|---|---|
| Config audit deletes a key actually-used in obscure path | Per-key human verification before delete; deletion list committed for post-hoc audit |
| OCI byte-diff surfaces real divergence | Triage + fix before merge; do not ship if diff is non-zero |
| Live cut-over surprise | Paper-trade run same day before live switch; revert is one `git revert` |
| Worker-side detector cache memory grows unbounded | Detector instances are stateless once initialized — cache size = number of detectors = ~8 per worker = negligible |
| Calendar event ordering bug (e.g., close before open at same minute) | Unit tests enforce ordering rules at calendar construction |
| Backfill failures cascade | Single-symbol backfill failure doesn't block other symbols' dispatch; retry next bar |

## File-level deltas (summary)

| File | Change |
|---|---|
| `config/configuration.json` | Audit-delete ~100 dead keys; add `mode` + `mode_profiles` block; add per-setup `detector_class` + `universe_builder` + `universe_trigger` keys |
| `config/sub8_oci_overrides.json` | **DELETE** |
| `tools/apply_oci_override.py` | **DELETE** |
| `oci/docker/entrypoint.py` | Remove `apply_oci_config_override()` call + function; set `RUN_MODE=oci_research` env var |
| `services/dispatch/setup_registry.py` | **NEW** |
| `services/dispatch/transition_calendar.py` | **NEW** |
| `services/dispatch/tag_map.py` | **NEW** |
| `services/dispatch/fetch_scope.py` | **NEW** |
| `services/dispatch/planner.py` | **NEW** |
| `services/dispatch/worker.py` | **NEW** (houses `dispatch_worker_batch` + worker-side detector cache) |
| `services/screener_live.py` | Rewrite `_run_5m_scan`; delete `_universe_union()`, hardcoded universe-union and lazy-build if-blocks |
| `services/plan_orchestrator.py` | Delete `_DETECTOR_REGISTRY` and `ACTIVE_SETUPS`; replace references with registry reads |
| `structures/main_detector.py` | **DELETE** (replaced by worker entry point) |
| `structures/*_structure.py` | Strip dead cap_segment early-reject lines |
| `tests/dispatch/*.py` | **NEW** unit + integration tests |
| `docs/config_keys_removed_2026-05-17.md` | **NEW** audit trail |

## Done criteria

Refactor is done when:

1. Phase 0 PRs A + B merged; live and OCI continue to operate
2. Phase 1 + PR C merged; OCI 3-year backtest byte-diffs to zero against pre-refactor baseline
3. One day of paper trading on post-refactor code matches expectations (no unhandled errors, no missing signals vs concurrent OCI run)
4. One day of live trading clean
5. All deleted files / keys recorded in commit messages + `docs/config_keys_removed_2026-05-17.md`
