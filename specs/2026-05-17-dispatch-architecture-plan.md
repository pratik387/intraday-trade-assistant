# Per-Detector Dispatch Architecture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `(symbol × all-detectors)` dispatch with symbol-tagged dispatch backed by a single `SetupRegistry`, after cleaning out ~100 dead config keys and folding OCI override file into mode_profiles. Net wins: 18-22x fewer detector evaluations per bar, 40-50% fewer API calls per session, single source of truth for setup metadata.

**Architecture:** Calendar-driven tag map mutated only at known transition timestamps (universe builds + active-window edges). Worker batches process `(sym, df5, levels, tags)` tuples — only tagged detectors run per symbol. Spec: `specs/2026-05-17-dispatch-architecture-design.md`.

**Tech Stack:** Python 3.10, pandas, ProcessPoolExecutor, existing pytest setup.

---

## File Structure

**New files (Phase 1):**
- `services/dispatch/__init__.py` — package marker
- `services/dispatch/setup_registry.py` — `SetupSpec` + `SetupRegistry` + `Trigger` types; single source of truth for setup metadata
- `services/dispatch/transition_calendar.py` — `TransitionEvent` + `TransitionCalendar`; derived from registry, sorted by IST time
- `services/dispatch/tag_map.py` — `TagMap`; per-symbol active-detector state, mutated only by calendar walks
- `services/dispatch/fetch_scope.py` — `FetchScopeManager`; decides what symbols to API-fetch per bar + backfill detection
- `services/dispatch/planner.py` — `Batch` + `DispatchPlanner`; assembles per-bar work plan in ≤50-symbol chunks
- `services/dispatch/worker.py` — `dispatch_worker_batch` worker-side entry + module-level detector cache

**New test files:**
- `tests/dispatch/__init__.py`
- `tests/dispatch/test_setup_registry.py`
- `tests/dispatch/test_transition_calendar.py`
- `tests/dispatch/test_tag_map.py`
- `tests/dispatch/test_fetch_scope.py`
- `tests/dispatch/test_planner.py`
- `tests/dispatch/test_dispatch_e2e.py`

**Modified files:**
- `config/configuration.json` — Phase 0 A: delete ~100 dead keys; Phase 0 B: add `mode` + `mode_profiles`; Phase 1: add per-setup `detector_class`, `universe_builder`, `universe_trigger` keys
- `services/screener_live.py` — Phase 1: rewrite `_run_5m_scan`; delete `_universe_union()`, hardcoded universe-union and lazy-build if-blocks
- `services/plan_orchestrator.py` — Phase 1: delete `_DETECTOR_REGISTRY` + `ACTIVE_SETUPS`; replace references with `SetupRegistry`
- `structures/gap_fade_short_structure.py` (and 7 sibling detectors) — Phase 1: strip dead `cap_segment` early-reject lines
- `oci/docker/entrypoint.py` — Phase 0 B: remove `apply_oci_config_override` call + function; set `RUN_MODE=oci_research` env var

**Deleted files:**
- `structures/main_detector.py` — Phase 1: replaced by `worker.py`
- `config/sub8_oci_overrides.json` — Phase 0 B
- `tools/apply_oci_override.py` — Phase 0 B

**New docs:**
- `docs/config_keys_removed_2026-05-17.md` — Phase 0 A audit trail

---

## Phase 0 PR A — configuration.json audit + delete dead keys

This PR has zero behavior change. The ~137 grep-candidate dead keys are verified per-key, then deleted in one commit with a documented audit trail.

### Task A1: Generate the candidate-dead key list

**Files:**
- Create: `tools/audit_config_keys.py` (one-shot script, deleted after PR A merges)

- [ ] **Step 1: Write the audit script**

```python
# tools/audit_config_keys.py
"""One-shot config audit: emit candidate-dead keys for human verification."""
import json
import re
import subprocess
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config" / "configuration.json"
SEARCH_DIRS = ["services", "structures", "pipelines", "gates", "tools", "oci", "main.py", "broker", "config/pipelines"]


def top_level_functional_keys():
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return [k for k in cfg.keys() if not k.startswith("_")]


def references_for_key(key: str) -> list[str]:
    """Grep all .py + .json files in SEARCH_DIRS for any reference to `key`."""
    patterns = [
        rf'"{re.escape(key)}"',
        rf"'{re.escape(key)}'",
        rf'\.get\("{re.escape(key)}"',
        rf'\["{re.escape(key)}"\]',
    ]
    hits = []
    for pat in patterns:
        try:
            result = subprocess.run(
                ["grep", "-rln", "-E", pat] + SEARCH_DIRS,
                cwd=ROOT, capture_output=True, text=True, timeout=30,
            )
            for line in result.stdout.strip().split("\n"):
                if line and line not in hits:
                    hits.append(line)
        except Exception as e:
            print(f"  WARN: grep failed for {pat}: {e}")
    return hits


def main():
    keys = top_level_functional_keys()
    print(f"Auditing {len(keys)} functional top-level keys...\n")
    candidate_dead = []
    used = []
    for k in keys:
        refs = references_for_key(k)
        if refs:
            used.append((k, len(refs)))
        else:
            candidate_dead.append(k)
    print(f"USED ({len(used)} keys):")
    for k, n in sorted(used):
        print(f"  {k} ({n} files)")
    print(f"\nCANDIDATE-DEAD ({len(candidate_dead)} keys):")
    for k in sorted(candidate_dead):
        print(f"  {k}")
    print(f"\n--- NEXT: humans verify each candidate-dead key before deletion. ---")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the audit**

Run: `.venv/Scripts/python tools/audit_config_keys.py > /tmp/audit_candidates.txt 2>&1`

Expected: file `/tmp/audit_candidates.txt` lists USED (~80 keys) and CANDIDATE-DEAD (~137 keys). Counts may vary slightly.

- [ ] **Step 3: Commit the audit script**

```bash
git add tools/audit_config_keys.py
git commit -m "audit: one-shot script to enumerate dead config keys"
```

### Task A2: Per-key verification and deletion

This is the **manual verification gate**. For each candidate-dead key, the engineer must check:
1. Bare word references: `grep -rn "<key>" services structures tools` (catches `cfg[KEY_CONST]` where KEY_CONST is a module-level string)
2. Substring patterns: `grep -rn "[\\\"']<key>[\\\"']" .` (broader than the audit script)
3. Imports/re-exports in `config/pipelines/*.json`
4. Comments referencing it (those CAN stay, but if the comment is the only reference, the key is dead)

Known false negative: the audit flagged `entry_cutoff_hhmm` as dead, but `services/execution/trigger_aware_executor.py:543` reads it. **Treat the grep result as a starting point, not a verdict.**

- [ ] **Step 1: Open `/tmp/audit_candidates.txt` and the audit doc side-by-side**

Create the audit doc:

```bash
cat > docs/config_keys_removed_2026-05-17.md <<'EOF'
# Config Keys Removed 2026-05-17

Per Phase 0 PR A of `specs/2026-05-17-dispatch-architecture-plan.md`. Each entry is a key that was confirmed dead by human verification (not just grep): no `.py`, `.json`, or other reference outside comments.

Format: `key_name` — last-known purpose / git commit that added it / why dead

## Removed top-level keys

(populated during the audit)

## Removed nested keys

(populated during the audit; only nested keys removed if their parent is otherwise alive)
EOF
```

- [ ] **Step 2: Verify each candidate-dead key (engineer-driven, iterative)**

For each `<key>` in the candidate-dead list:

```bash
# Three greps. If ALL return zero hits outside comments, the key is dead.
grep -rn -F '"<key>"' services structures tools gates oci pipelines main.py 2>/dev/null
grep -rn -F "'<key>'" services structures tools gates oci pipelines main.py 2>/dev/null
grep -rn -wF "<key>" services structures tools gates oci pipelines main.py 2>/dev/null
```

If all three return zero hits (or hits only in `_comment_*` strings), append to `docs/config_keys_removed_2026-05-17.md` and proceed to remove. If any grep hits real code, mark the key USED and skip.

- [ ] **Step 3: Remove confirmed-dead keys from `config/configuration.json`**

Edit `config/configuration.json` directly. Remove each confirmed-dead key AND any `_comment_<key>` paired comment key (those become orphan documentation if the key is gone).

After each batch of ~20 removals: validate JSON still parses:

```bash
.venv/Scripts/python -c "import json; json.load(open('config/configuration.json', encoding='utf-8')); print('OK')"
```

Expected: `OK`.

- [ ] **Step 4: Run existing test suite to verify nothing breaks**

Run: `.venv/Scripts/python -m pytest tests/ -x --tb=short 2>&1 | tail -30`

Expected: all green. If any test fails accessing a removed key, restore that key and mark it USED in the audit doc.

- [ ] **Step 5: One-day smoke backtest sanity**

Run: `.venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_phase0a.log 2>&1`

Expected: exit code 0. `logs/backtest_*/trade_report.csv` (if generated) or `logs/backtest_*/events.jsonl` shows trades fire normally.

- [ ] **Step 6: Commit the deletion**

```bash
git add config/configuration.json docs/config_keys_removed_2026-05-17.md
git commit -m "$(cat <<'EOF'
config: remove ~N confirmed-dead keys (Phase 0 PR A)

Each removed key verified by human grep (not just audit script) to have
zero references in services/, structures/, gates/, tools/, oci/, pipelines/.
Audit trail in docs/config_keys_removed_2026-05-17.md.

No behavior change expected. Smoke 2024-05-03 + pytest suite both green.
EOF
)"
```

Replace `N` with the actual count.

- [ ] **Step 7: Delete the one-shot audit script**

```bash
git rm tools/audit_config_keys.py
git commit -m "audit: remove one-shot config-audit script (job done)"
```

---

## Phase 0 PR B — fold sub8_oci_overrides.json into mode_profiles

### Task B1: Add mode_profiles block to configuration.json

**Files:**
- Modify: `config/configuration.json`

- [ ] **Step 1: Add the `mode` + `mode_profiles` keys**

At the top of `configuration.json` (just after `"$schema"` if present, otherwise as the first keys), add:

```json
{
  "mode": "production",
  "_comment_mode": "Selects which mode_profiles entry's keys are applied on top of base config at load time. Override via env var RUN_MODE.",
  "mode_profiles": {
    "production": {},
    "oci_research": {
      "entry_cutoff_hhmm": "15:25",
      "last_scan_hhmm": "15:25",
      "eod_squareoff_hhmm": "15:25",
      "max_trades_per_cycle": 10000,
      "gate_input_logging": { "enabled": true }
    }
  },
  ...existing keys
}
```

- [ ] **Step 2: Verify JSON still parses**

Run: `.venv/Scripts/python -c "import json; cfg = json.load(open('config/configuration.json', encoding='utf-8')); print('mode:', cfg.get('mode')); print('profiles:', list(cfg.get('mode_profiles', {}).keys()))"`

Expected: `mode: production` + `profiles: ['production', 'oci_research']`.

### Task B2: Apply mode profile at config load time

**Files:**
- Modify: `config/filters_setup.py` (or wherever `load_filters` / config loader lives)
- Test: `tests/config/test_mode_profile_merge.py`

- [ ] **Step 1: Locate the config loader**

Run: `grep -rn "def load_filters\|def load_config\|json.load.*configuration.json" config/ services/ main.py | head -5`

Note the file and function that opens `configuration.json`. The merge logic goes there.

- [ ] **Step 2: Write the failing test**

```python
# tests/config/test_mode_profile_merge.py
import os
import pytest
from config.filters_setup import load_filters  # adjust import if loader lives elsewhere


def test_default_mode_is_production_no_override():
    """When mode=production, no overrides applied (empty profile)."""
    cfg = load_filters()
    assert cfg["mode"] == "production"
    # entry_cutoff_hhmm in base config (production value) is unaffected
    assert cfg["entry_cutoff_hhmm"] == "15:10"


def test_oci_research_mode_overrides_via_env(monkeypatch):
    """RUN_MODE=oci_research env var picks the oci_research profile."""
    monkeypatch.setenv("RUN_MODE", "oci_research")
    cfg = load_filters()
    assert cfg["entry_cutoff_hhmm"] == "15:25"
    assert cfg["last_scan_hhmm"] == "15:25"
    assert cfg["eod_squareoff_hhmm"] == "15:25"
    assert cfg["max_trades_per_cycle"] == 10000
    assert cfg["gate_input_logging"]["enabled"] is True


def test_unknown_mode_raises(monkeypatch):
    """Typo in RUN_MODE fails fast at startup."""
    monkeypatch.setenv("RUN_MODE", "oci_resaerch")  # typo
    with pytest.raises(ValueError, match="unknown mode"):
        load_filters()
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/config/test_mode_profile_merge.py -v`

Expected: 3 tests FAIL (loader doesn't apply profile yet).

- [ ] **Step 4: Implement the merge in the config loader**

In the loader function (after `json.load`), add:

```python
# Apply mode_profiles overrides on top of base config.
# Mode = env RUN_MODE (if set), else cfg["mode"], else "production".
import os
_mode = os.environ.get("RUN_MODE") or cfg.get("mode", "production")
_profiles = cfg.get("mode_profiles", {})
if _mode not in _profiles:
    raise ValueError(f"unknown mode {_mode!r}; available: {list(_profiles.keys())}")
_overrides = _profiles[_mode]
# Shallow merge: profile keys replace top-level keys
for k, v in _overrides.items():
    cfg[k] = v
cfg["_effective_mode"] = _mode  # for debugging
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/config/test_mode_profile_merge.py -v`

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add config/configuration.json config/filters_setup.py tests/config/test_mode_profile_merge.py
git commit -m "config: add mode_profiles + env-driven profile merge

Inlines the 5 real OCI Research deltas from sub8_oci_overrides.json
into a mode_profiles.oci_research block. Profile selection via
top-level 'mode' key (default 'production'), overridable by RUN_MODE
env var. Loader merges profile keys on top of base config at load.

Unknown mode fails fast. Tests cover the three branches."
```

### Task B3: Update OCI entrypoint to use RUN_MODE

**Files:**
- Modify: `oci/docker/entrypoint.py`

- [ ] **Step 1: Replace merge call with env var**

In `oci/docker/entrypoint.py`, find the `apply_oci_config_override()` function (around line 112-153) and the call to it (around line 698-705).

Replace the call site:

```python
# OLD (line ~698-705):
#   # Apply OCI config override (sub8_oci_overrides.json -> configuration.json).
#   ...
#   apply_oci_config_override()

# NEW:
os.environ["RUN_MODE"] = "oci_research"
log("OCI mode: set RUN_MODE=oci_research")
```

Delete the `apply_oci_config_override()` function definition entirely.

- [ ] **Step 2: Delete the merge tool**

```bash
git rm tools/apply_oci_override.py
git rm config/sub8_oci_overrides.json
```

- [ ] **Step 3: Verify no other code references the deleted files**

Run: `grep -rn "sub8_oci_overrides\|apply_oci_override" services/ structures/ oci/ tools/ main.py 2>&1`

Expected: zero hits (only the lines you just deleted should have appeared).

- [ ] **Step 4: Smoke test — production mode (no env)**

Run: `.venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_b3_prod.log 2>&1`

Expected: exit 0; log includes `_effective_mode: production`. `entry_cutoff_hhmm` in the runtime config is `15:10` (base value).

- [ ] **Step 5: Smoke test — oci_research mode**

Run: `RUN_MODE=oci_research .venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_b3_oci.log 2>&1`

Expected: exit 0. Grep the log to confirm OCI cutoffs applied: `grep -i "15:25\|entry_cutoff_hhmm" /tmp/smoke_b3_oci.log | head -3`.

- [ ] **Step 6: Commit**

```bash
git add oci/docker/entrypoint.py
git commit -m "oci: replace sub8_oci_overrides merge with RUN_MODE env var

Sets RUN_MODE=oci_research in the OCI entrypoint; the config loader
(previous commit) reads it and applies the oci_research profile.
sub8_oci_overrides.json + apply_oci_override.py deleted — their 5
real deltas now live in configuration.json mode_profiles.oci_research."
```

---

## Phase 1 — SetupRegistry and per-detector-tag dispatch (bundled with PR C cleanup)

Build new modules bottom-up with TDD, then wire into `screener_live.py`, then delete the old paths in the same PR.

### Task 1: Create the dispatch package + Trigger type

**Files:**
- Create: `services/dispatch/__init__.py`
- Create: `services/dispatch/setup_registry.py` (initial — Trigger + SetupSpec only)
- Create: `tests/dispatch/__init__.py`
- Create: `tests/dispatch/test_setup_registry.py` (initial)

- [ ] **Step 1: Create package markers**

```bash
mkdir -p services/dispatch tests/dispatch
touch services/dispatch/__init__.py tests/dispatch/__init__.py
```

- [ ] **Step 2: Write the failing test for Trigger parsing**

```python
# tests/dispatch/test_setup_registry.py
import pytest
from datetime import time
from services.dispatch.setup_registry import Trigger, parse_trigger


def test_parse_session_start_trigger():
    t = parse_trigger("session_start")
    assert t == Trigger.session_start()


def test_parse_bar_trigger():
    t = parse_trigger("bar:09:15")
    assert t.kind == "bar"
    assert t.at == time(9, 15)


def test_parse_bar_trigger_malformed_raises():
    with pytest.raises(ValueError, match="malformed"):
        parse_trigger("bar:9:15")  # missing leading zero
    with pytest.raises(ValueError, match="malformed"):
        parse_trigger("bar:09")
    with pytest.raises(ValueError, match="malformed"):
        parse_trigger("foo")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_setup_registry.py -v`

Expected: ImportError / module not found.

- [ ] **Step 4: Implement Trigger + parse_trigger**

```python
# services/dispatch/setup_registry.py
"""Single source of truth for setup metadata.

Loaded from configuration.json setups.* at startup. Every other module that
needs to know "what setups exist, what they need, when" reads from here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time
from typing import Callable, Optional


@dataclass(frozen=True)
class Trigger:
    """When a setup's universe is built.

    kind="session_start" — built at scanner warmup (no `at`).
    kind="bar" — built when the 5m bar at `at` closes.
    """
    kind: str
    at: Optional[time] = None

    @classmethod
    def session_start(cls) -> "Trigger":
        return cls(kind="session_start", at=None)

    @classmethod
    def bar(cls, at: time) -> "Trigger":
        return cls(kind="bar", at=at)


_BAR_RE = re.compile(r"^bar:(\d{2}):(\d{2})$")


def parse_trigger(spec: str) -> Trigger:
    """Parse a trigger spec string from config.

    Accepted: "session_start" | "bar:HH:MM" (zero-padded).
    """
    if spec == "session_start":
        return Trigger.session_start()
    m = _BAR_RE.match(spec)
    if not m:
        raise ValueError(f"malformed trigger spec {spec!r}; expected 'session_start' or 'bar:HH:MM'")
    return Trigger.bar(time(int(m.group(1)), int(m.group(2))))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_setup_registry.py -v`

Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add services/dispatch/ tests/dispatch/__init__.py tests/dispatch/test_setup_registry.py
git commit -m "dispatch: Trigger type + parse_trigger (setup registry foundation)"
```

### Task 2: SetupSpec dataclass

**Files:**
- Modify: `services/dispatch/setup_registry.py`
- Modify: `tests/dispatch/test_setup_registry.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/dispatch/test_setup_registry.py`:

```python
from services.dispatch.setup_registry import SetupSpec


def test_setup_spec_creation():
    spec = SetupSpec(
        name="gap_fade_short",
        enabled=True,
        detector_class_path="structures.gap_fade_short_structure.GapFadeShortStructure",
        universe_builder_path="services.setup_universe.gap_fade_universe",
        universe_trigger=Trigger.bar(time(9, 15)),
        active_window=(time(9, 15), time(9, 30)),
        raw_config={"foo": "bar"},
    )
    assert spec.name == "gap_fade_short"
    assert spec.enabled is True
    assert spec.active_window == (time(9, 15), time(9, 30))


def test_setup_spec_rejects_inverted_window():
    with pytest.raises(ValueError, match="active_window_start <= active_window_end"):
        SetupSpec(
            name="bad", enabled=True,
            detector_class_path="x.Y",
            universe_builder_path="x.y",
            universe_trigger=Trigger.session_start(),
            active_window=(time(10, 0), time(9, 0)),  # inverted
            raw_config={},
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_setup_registry.py -v`

Expected: 2 new FAILs (`SetupSpec` not defined).

- [ ] **Step 3: Implement SetupSpec**

Append to `services/dispatch/setup_registry.py`:

```python
@dataclass(frozen=True)
class SetupSpec:
    """All metadata for one setup. Single source of truth."""
    name: str
    enabled: bool
    detector_class_path: str        # "structures.gap_fade_short_structure.GapFadeShortStructure"
    universe_builder_path: str      # "services.setup_universe.gap_fade_universe"
    universe_trigger: Trigger
    active_window: tuple            # (start: time, end: time), inclusive
    raw_config: dict

    def __post_init__(self):
        start, end = self.active_window
        if start > end:
            raise ValueError(
                f"setup {self.name!r}: active_window_start <= active_window_end required; "
                f"got start={start} end={end}"
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_setup_registry.py -v`

Expected: 5 PASS (3 from Task 1 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/setup_registry.py tests/dispatch/test_setup_registry.py
git commit -m "dispatch: SetupSpec dataclass with window validation"
```

### Task 3: SetupRegistry loader + validation

**Files:**
- Modify: `services/dispatch/setup_registry.py`
- Modify: `tests/dispatch/test_setup_registry.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dispatch/test_setup_registry.py`:

```python
from services.dispatch.setup_registry import SetupRegistry


@pytest.fixture
def sample_config():
    return {
        "setups": {
            "gap_fade_short": {
                "enabled": True,
                "detector_class": "structures.gap_fade_short_structure.GapFadeShortStructure",
                "universe_builder": "services.setup_universe.gap_fade_universe",
                "universe_trigger": "bar:09:15",
                "active_window_start": "09:15",
                "active_window_end": "09:30",
            },
            "disabled_setup": {
                "enabled": False,
                "detector_class": "structures.foo.Bar",
                "universe_builder": "services.foo.bar",
                "universe_trigger": "session_start",
                "active_window_start": "09:30",
                "active_window_end": "10:30",
            },
        },
    }


def test_registry_loads_enabled_setups(sample_config):
    reg = SetupRegistry.load_from_config(sample_config)
    enabled = reg.enabled()
    assert len(enabled) == 1
    assert enabled[0].name == "gap_fade_short"


def test_registry_get_by_name(sample_config):
    reg = SetupRegistry.load_from_config(sample_config)
    spec = reg.get("gap_fade_short")
    assert spec.universe_trigger.kind == "bar"


def test_registry_missing_required_key_raises():
    bad_cfg = {"setups": {"x": {"enabled": True}}}  # missing detector_class etc.
    with pytest.raises(ValueError, match="missing required key"):
        SetupRegistry.load_from_config(bad_cfg)


def test_registry_validates_class_importable(sample_config):
    """validate() imports detector_class + universe_builder; fails fast if missing."""
    reg = SetupRegistry.load_from_config(sample_config)
    # gap_fade_short paths are real, should import
    reg.validate()


def test_registry_validation_fails_on_bad_import_path():
    bad_cfg = {
        "setups": {
            "x": {
                "enabled": True,
                "detector_class": "structures.does_not_exist.NopeStructure",
                "universe_builder": "services.setup_universe.gap_fade_universe",
                "universe_trigger": "session_start",
                "active_window_start": "09:15",
                "active_window_end": "09:30",
            },
        },
    }
    reg = SetupRegistry.load_from_config(bad_cfg)
    with pytest.raises(ImportError):
        reg.validate()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_setup_registry.py -v`

Expected: 5 new FAILs (`SetupRegistry` not defined).

- [ ] **Step 3: Implement SetupRegistry**

Append to `services/dispatch/setup_registry.py`:

```python
import importlib


def _import_path(path: str):
    """Import 'module.submodule.Symbol' and return the Symbol."""
    module_path, _, symbol = path.rpartition(".")
    if not module_path:
        raise ImportError(f"invalid import path {path!r}; expected 'module.Symbol'")
    mod = importlib.import_module(module_path)
    if not hasattr(mod, symbol):
        raise ImportError(f"{path!r}: module {module_path!r} has no attribute {symbol!r}")
    return getattr(mod, symbol)


def _parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


REQUIRED_KEYS = (
    "detector_class",
    "universe_builder",
    "universe_trigger",
    "active_window_start",
    "active_window_end",
)


class SetupRegistry:
    """Single source of truth for setup metadata."""

    def __init__(self, specs: dict[str, SetupSpec]):
        self._specs = specs

    @classmethod
    def load_from_config(cls, root_config: dict) -> "SetupRegistry":
        setups = root_config.get("setups", {})
        specs: dict[str, SetupSpec] = {}
        for name, raw in setups.items():
            if not isinstance(raw, dict):
                continue  # skip non-dict entries (e.g., comments at this level)
            for k in REQUIRED_KEYS:
                if k not in raw:
                    raise ValueError(f"setup {name!r}: missing required key {k!r}")
            specs[name] = SetupSpec(
                name=name,
                enabled=bool(raw.get("enabled", False)),
                detector_class_path=raw["detector_class"],
                universe_builder_path=raw["universe_builder"],
                universe_trigger=parse_trigger(raw["universe_trigger"]),
                active_window=(_parse_hhmm(raw["active_window_start"]), _parse_hhmm(raw["active_window_end"])),
                raw_config=raw,
            )
        return cls(specs)

    def enabled(self) -> list[SetupSpec]:
        return [s for s in self._specs.values() if s.enabled]

    def get(self, name: str) -> SetupSpec:
        return self._specs[name]

    def validate(self) -> None:
        """Import every enabled setup's detector_class + universe_builder. Fail fast."""
        for spec in self.enabled():
            _import_path(spec.detector_class_path)
            _import_path(spec.universe_builder_path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_setup_registry.py -v`

Expected: 10 PASS (all).

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/setup_registry.py tests/dispatch/test_setup_registry.py
git commit -m "dispatch: SetupRegistry loader + validate() fail-fast import check"
```

### Task 4: Add per-setup metadata keys to configuration.json

**Files:**
- Modify: `config/configuration.json` (8 setup blocks)

This task adds the three new keys (`detector_class`, `universe_builder`, `universe_trigger`) to each enabled setup. Values mirror what's currently hardcoded in `plan_orchestrator.py` (detector_class) and `screener_live.py` (universe_trigger).

- [ ] **Step 1: Compile the mapping table from current code**

| setup_name | detector_class (from plan_orchestrator.py) | universe_builder (from setup_universe.py) | universe_trigger (from screener_live.py lazy-build blocks) |
|---|---|---|---|
| gap_fade_short | structures.gap_fade_short_structure.GapFadeShortStructure | services.setup_universe.gap_fade_universe | bar:09:15 |
| circuit_t1_fade_short | structures.circuit_t1_fade_short_structure.CircuitT1FadeShortStructure | services.setup_universe.circuit_t1_universe | session_start |
| delivery_pct_anomaly_short | structures.delivery_pct_anomaly_short_structure.DeliveryPctAnomalyShortStructure | services.setup_universe.delivery_pct_universe | session_start |
| long_panic_gap_down | structures.long_panic_gap_down_structure.LongPanicGapDownStructure | services.setup_universe.long_panic_gap_down_universe | bar:09:15 |
| circuit_release_fade_short | structures.circuit_release_fade_short_structure.CircuitReleaseFadeShortStructure | services.setup_universe.circuit_release_fade_short_universe | bar:10:30 |
| round_number_sweep_short | structures.round_number_sweep_short_structure.RoundNumberSweepShortStructure | services.setup_universe.round_number_sweep_short_universe | session_start |
| or_window_failure_fade_short | structures.or_window_failure_fade_short_structure.OrWindowFailureFadeShortStructure | services.setup_universe.or_window_failure_fade_short_universe | session_start |
| mis_unwind_vwap_revert_short | structures.mis_unwind_vwap_revert_short_structure.MisUnwindVwapRevertShortStructure | services.setup_universe.mis_unwind_vwap_revert_short_universe | session_start |

- [ ] **Step 2: Edit `config/configuration.json`**

For each of the 8 setups, add the three keys just inside the setup block. Example for `gap_fade_short`:

```json
"gap_fade_short": {
  "enabled": true,
  "detector_class": "structures.gap_fade_short_structure.GapFadeShortStructure",
  "universe_builder": "services.setup_universe.gap_fade_universe",
  "universe_trigger": "bar:09:15",
  ...existing keys
}
```

Use the table from Step 1 for the other 7. Use `Edit` with replace_all=false against each setup block's `"enabled":` line as the anchor.

- [ ] **Step 3: Validate registry loads**

Run:

```bash
.venv/Scripts/python -c "
from config.filters_setup import load_filters
from services.dispatch.setup_registry import SetupRegistry
cfg = load_filters()
reg = SetupRegistry.load_from_config(cfg)
reg.validate()
print('OK — enabled setups:')
for s in reg.enabled():
    print(f'  {s.name}: detector={s.detector_class_path.split(\".\")[-1]}, trigger={s.universe_trigger.kind}')
"
```

Expected: prints 8 enabled setups, each with its detector class and trigger kind.

- [ ] **Step 4: Commit**

```bash
git add config/configuration.json
git commit -m "config: add detector_class + universe_builder + universe_trigger per setup

Three new keys per enabled setup, populating SetupRegistry's required
fields. Values mirror what's currently hardcoded in
plan_orchestrator.py:_DETECTOR_REGISTRY and screener_live.py
lazy-build if-blocks (those duplicates are deleted in a later task)."
```

### Task 5: TagMap

**Files:**
- Create: `services/dispatch/tag_map.py`
- Create: `tests/dispatch/test_tag_map.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dispatch/test_tag_map.py
import pytest
from services.dispatch.tag_map import TagMap


def test_empty_tag_map():
    tm = TagMap()
    assert tm.active_symbols() == set()
    assert tm.active_tags("NSE:RML") == set()


def test_add_universe_then_open_window():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A", "NSE:B", "NSE:C"})
    # Tagged but not active yet
    assert tm.active_symbols() == set()
    tm.open_window("gap_fade_short")
    assert tm.active_symbols() == {"NSE:A", "NSE:B", "NSE:C"}
    assert tm.active_tags("NSE:A") == {"gap_fade_short"}


def test_close_window_drops_from_active():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    tm.close_window("gap_fade_short")
    assert tm.active_symbols() == set()
    assert tm.active_tags("NSE:A") == set()


def test_multiple_setups_same_symbol():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:RML"})
    tm.add_universe("mis_unwind_vwap_revert_short", {"NSE:RML", "NSE:OTHER"})
    tm.open_window("gap_fade_short")
    tm.open_window("mis_unwind_vwap_revert_short")
    assert tm.active_tags("NSE:RML") == {"gap_fade_short", "mis_unwind_vwap_revert_short"}
    # Close one; symbol still has the other
    tm.close_window("gap_fade_short")
    assert tm.active_tags("NSE:RML") == {"mis_unwind_vwap_revert_short"}
    assert "NSE:RML" in tm.active_symbols()


def test_reopen_window_after_close():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    tm.close_window("gap_fade_short")
    tm.open_window("gap_fade_short")  # reopen (uncommon but legal)
    assert tm.active_symbols() == {"NSE:A"}


def test_close_unopened_window_is_noop():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.close_window("gap_fade_short")  # was never opened
    assert tm.active_symbols() == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_tag_map.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement TagMap**

```python
# services/dispatch/tag_map.py
"""Per-symbol active-detector state.

Mutated only by DispatchPlanner walking TransitionCalendar. Read by
FetchScopeManager and DispatchPlanner to build per-bar fetch + dispatch
plans.
"""
from __future__ import annotations

from collections import defaultdict


class TagMap:
    def __init__(self):
        # setup_name -> set[sym]
        self._universe: dict[str, set[str]] = {}
        # setup_name -> bool (is its active window currently open?)
        self._window_open: dict[str, bool] = {}

    def add_universe(self, setup: str, syms: set[str]) -> None:
        """Register the universe of qualifying symbols for a setup.

        Called once when the universe builder fires (session_start or bar:HH:MM
        trigger). Does NOT open the active window.
        """
        self._universe[setup] = set(syms)

    def open_window(self, setup: str) -> None:
        """Mark this setup's active window as open. Symbols in its universe
        become eligible for dispatch."""
        self._window_open[setup] = True

    def close_window(self, setup: str) -> None:
        """Mark this setup's active window as closed. No-op if never opened."""
        self._window_open[setup] = False

    def active_tags(self, sym: str) -> set[str]:
        """Detector names whose universe contains `sym` AND whose window is open."""
        return {
            setup
            for setup, univ in self._universe.items()
            if sym in univ and self._window_open.get(setup, False)
        }

    def active_symbols(self) -> set[str]:
        """Union of universes whose windows are currently open."""
        out: set[str] = set()
        for setup, univ in self._universe.items():
            if self._window_open.get(setup, False):
                out |= univ
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_tag_map.py -v`

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/tag_map.py tests/dispatch/test_tag_map.py
git commit -m "dispatch: TagMap — per-symbol active-detector state"
```

### Task 6: TransitionCalendar

**Files:**
- Create: `services/dispatch/transition_calendar.py`
- Create: `tests/dispatch/test_transition_calendar.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dispatch/test_transition_calendar.py
import pytest
from datetime import time
from services.dispatch.setup_registry import SetupSpec, Trigger, SetupRegistry
from services.dispatch.transition_calendar import TransitionCalendar, TransitionEvent


def _mk_spec(name: str, trigger: Trigger, win_start: time, win_end: time) -> SetupSpec:
    return SetupSpec(
        name=name, enabled=True,
        detector_class_path="x.Y", universe_builder_path="x.y",
        universe_trigger=trigger,
        active_window=(win_start, win_end),
        raw_config={},
    )


def _mk_registry(*specs: SetupSpec) -> SetupRegistry:
    reg = SetupRegistry({})
    reg._specs = {s.name: s for s in specs}
    return reg


def test_events_sorted_by_time():
    reg = _mk_registry(
        _mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)),
        _mk_spec("c8", Trigger.session_start(), time(14, 30), time(15, 0)),
    )
    cal = TransitionCalendar.from_registry(reg)
    times = [ev.at for ev in cal.all_events()]
    assert times == sorted(times)


def test_session_start_trigger_emits_build_at_market_open():
    reg = _mk_registry(_mk_spec("c8", Trigger.session_start(), time(14, 30), time(15, 0)))
    cal = TransitionCalendar.from_registry(reg)
    builds = [ev for ev in cal.all_events() if ev.kind == "build_universe"]
    assert len(builds) == 1
    # session_start universes built before first scan (use 09:15 as the conventional point)
    assert builds[0].at == time(9, 15)
    assert builds[0].setup == "c8"


def test_bar_trigger_emits_build_at_that_bar():
    reg = _mk_registry(_mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)))
    cal = TransitionCalendar.from_registry(reg)
    builds = [ev for ev in cal.all_events() if ev.kind == "build_universe"]
    assert len(builds) == 1
    assert builds[0].at == time(9, 15)


def test_each_setup_emits_open_and_close_window():
    reg = _mk_registry(_mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)))
    cal = TransitionCalendar.from_registry(reg)
    kinds = [(ev.at, ev.kind) for ev in cal.all_events()]
    assert (time(9, 15), "open_window") in kinds
    assert (time(9, 30), "close_window") in kinds


def test_events_in_range_inclusive_exclusive():
    """events_in(after, until) returns events where after < ev.at <= until."""
    reg = _mk_registry(_mk_spec("gap_fade_short", Trigger.bar(time(9, 15)), time(9, 15), time(9, 30)))
    cal = TransitionCalendar.from_registry(reg)
    events = cal.events_in(after=time(9, 10), until=time(9, 15))
    kinds = [ev.kind for ev in events]
    # 09:15 boundary INCLUDED in (9:10, 9:15] → build_universe + open_window both fire
    assert "build_universe" in kinds
    assert "open_window" in kinds

    events_late = cal.events_in(after=time(9, 15), until=time(9, 30))
    kinds_late = [ev.kind for ev in events_late]
    assert "close_window" in kinds_late
    assert "open_window" not in kinds_late  # 09:15 was in the prior range


def test_close_strictly_after_open_at_same_minute():
    """If a setup's window starts AND ends at the same minute (one-shot like circuit_t1),
    ordering still well-defined: open before close."""
    reg = _mk_registry(_mk_spec("circuit_t1", Trigger.session_start(), time(10, 30), time(10, 30)))
    cal = TransitionCalendar.from_registry(reg)
    same_minute = [ev for ev in cal.all_events() if ev.at == time(10, 30)]
    kinds = [ev.kind for ev in same_minute]
    # open must come before close for one-shot windows
    assert kinds.index("open_window") < kinds.index("close_window")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_transition_calendar.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement TransitionCalendar**

```python
# services/dispatch/transition_calendar.py
"""Time-keyed event list derived from SetupRegistry.

Walked at each bar close to evolve TagMap state. Events are sorted by
(time, ordering_key) where ordering_key ensures open_window precedes
close_window at the same minute (for one-shot setups).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Literal

from services.dispatch.setup_registry import SetupRegistry


# Conventional session-open time. session_start-trigger universes
# build here (before the 09:15 scan).
SESSION_OPEN = time(9, 15)


@dataclass(frozen=True)
class TransitionEvent:
    at: time
    kind: Literal["build_universe", "open_window", "close_window"]
    setup: str


# Sort ordering within the same minute: build first, then open, then close.
_KIND_ORDER = {"build_universe": 0, "open_window": 1, "close_window": 2}


class TransitionCalendar:
    def __init__(self, events: list[TransitionEvent]):
        self._events = sorted(events, key=lambda e: (e.at, _KIND_ORDER[e.kind]))

    @classmethod
    def from_registry(cls, registry: SetupRegistry) -> "TransitionCalendar":
        events: list[TransitionEvent] = []
        for spec in registry.enabled():
            # build_universe event
            build_at = (
                SESSION_OPEN
                if spec.universe_trigger.kind == "session_start"
                else spec.universe_trigger.at
            )
            events.append(TransitionEvent(at=build_at, kind="build_universe", setup=spec.name))
            # open_window event
            events.append(TransitionEvent(at=spec.active_window[0], kind="open_window", setup=spec.name))
            # close_window event
            events.append(TransitionEvent(at=spec.active_window[1], kind="close_window", setup=spec.name))
        return cls(events)

    def all_events(self) -> list[TransitionEvent]:
        return list(self._events)

    def events_in(self, after: time, until: time) -> list[TransitionEvent]:
        """Return events where after < ev.at <= until, in calendar order."""
        return [ev for ev in self._events if after < ev.at <= until]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_transition_calendar.py -v`

Expected: 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/transition_calendar.py tests/dispatch/test_transition_calendar.py
git commit -m "dispatch: TransitionCalendar — time-keyed events from registry"
```

### Task 7: FetchScopeManager

**Files:**
- Create: `services/dispatch/fetch_scope.py`
- Create: `tests/dispatch/test_fetch_scope.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dispatch/test_fetch_scope.py
import pandas as pd
from datetime import datetime, time
from services.dispatch.tag_map import TagMap
from services.dispatch.fetch_scope import FetchScopeManager


def test_fetch_set_is_active_symbols():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A", "NSE:B"})
    tm.open_window("gap_fade_short")
    scope = FetchScopeManager()
    assert scope.fetch_set(datetime(2024, 5, 3, 9, 30), tm) == {"NSE:A", "NSE:B"}


def test_fetch_set_empty_when_no_active_tags():
    tm = TagMap()
    scope = FetchScopeManager()
    assert scope.fetch_set(datetime(2024, 5, 3, 10, 30), tm) == set()


def test_backfill_needed_when_df_missing():
    scope = FetchScopeManager()
    assert scope.is_backfill_needed("NSE:A", df5_by_symbol={}, bar_ts=datetime(2024, 5, 3, 12, 0))


def test_backfill_needed_when_df_stale():
    """If last bar in df is more than 5 minutes before bar_ts, backfill."""
    scope = FetchScopeManager()
    df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-05-03 10:25:00")])
    assert scope.is_backfill_needed("NSE:A", df5_by_symbol={"NSE:A": df}, bar_ts=datetime(2024, 5, 3, 12, 0))


def test_no_backfill_when_df_current():
    scope = FetchScopeManager()
    df = pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp("2024-05-03 11:55:00")])
    assert not scope.is_backfill_needed("NSE:A", df5_by_symbol={"NSE:A": df}, bar_ts=datetime(2024, 5, 3, 12, 0))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_fetch_scope.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement FetchScopeManager**

```python
# services/dispatch/fetch_scope.py
"""Decides what symbols to API-fetch per bar + backfill detection on re-entry."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from services.dispatch.tag_map import TagMap


_ONE_BAR = timedelta(minutes=5)


class FetchScopeManager:
    def fetch_set(self, bar_ts: datetime, tag_map: TagMap) -> set[str]:
        """Symbols to API-fetch this bar = currently-active-tag set.

        Dormant symbols (tag was active earlier but window closed) drop out
        until they re-enter via open_window. Re-entries trigger backfill.
        """
        return tag_map.active_symbols()

    def is_backfill_needed(
        self,
        sym: str,
        df5_by_symbol: dict[str, pd.DataFrame],
        bar_ts: datetime,
    ) -> bool:
        """True if `sym` has no df_5m OR last bar is older than (bar_ts - 5min).

        Caller uses this to decide between a single-bar fetch (cheap, normal case)
        and a history-from-session-open fetch (one extra call, populates the gap).
        """
        df = df5_by_symbol.get(sym)
        if df is None or df.empty:
            return True
        last_ts = df.index[-1]
        # Convert pandas Timestamp to datetime if needed
        if hasattr(last_ts, "to_pydatetime"):
            last_ts = last_ts.to_pydatetime()
        return last_ts < bar_ts - _ONE_BAR
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_fetch_scope.py -v`

Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/fetch_scope.py tests/dispatch/test_fetch_scope.py
git commit -m "dispatch: FetchScopeManager — active-tag fetch set + backfill detection"
```

### Task 8: DispatchPlanner

**Files:**
- Create: `services/dispatch/planner.py`
- Create: `tests/dispatch/test_planner.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/dispatch/test_planner.py
import pandas as pd
from datetime import datetime
from services.dispatch.tag_map import TagMap
from services.dispatch.planner import Batch, DispatchPlanner


def _df(): return pd.DataFrame({"close": [100.0]})
def _lvl(): return {"PDC": 100.0, "ORH": 102.0, "ORL": 98.0}


def test_empty_plan_when_no_active_tags():
    tm = TagMap()
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 10, 30), tm, df5_by_symbol={}, levels_by_symbol={})
    assert plan == []


def test_plan_chunks_at_batch_size():
    tm = TagMap()
    syms = {f"NSE:S{i}" for i in range(120)}
    tm.add_universe("gap_fade_short", syms)
    tm.open_window("gap_fade_short")
    df5 = {s: _df() for s in syms}
    levels = {s: _lvl() for s in syms}
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 9, 20), tm, df5_by_symbol=df5, levels_by_symbol=levels)
    # 120 syms / 50 = 3 batches (50 + 50 + 20)
    assert len(plan) == 3
    assert sum(len(b.items) for b in plan) == 120


def test_plan_items_carry_tag_set():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A"})
    tm.add_universe("mis_unwind_vwap_revert_short", {"NSE:A"})
    tm.open_window("gap_fade_short")
    tm.open_window("mis_unwind_vwap_revert_short")
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 14, 30), tm,
                        df5_by_symbol={"NSE:A": _df()},
                        levels_by_symbol={"NSE:A": _lvl()})
    assert len(plan) == 1
    sym, df5, levels, tags = plan[0].items[0]
    assert sym == "NSE:A"
    assert tags == {"gap_fade_short", "mis_unwind_vwap_revert_short"}


def test_plan_skips_symbols_without_df5():
    tm = TagMap()
    tm.add_universe("gap_fade_short", {"NSE:A", "NSE:B"})
    tm.open_window("gap_fade_short")
    planner = DispatchPlanner(batch_size=50)
    plan = planner.plan(datetime(2024, 5, 3, 9, 20), tm,
                        df5_by_symbol={"NSE:A": _df()},  # B missing
                        levels_by_symbol={"NSE:A": _lvl()})
    syms_in_plan = [item[0] for b in plan for item in b.items]
    assert syms_in_plan == ["NSE:A"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_planner.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement DispatchPlanner**

```python
# services/dispatch/planner.py
"""Per-bar work plan assembly.

For each currently-active symbol with df_5m + levels, build a (sym, df5, levels, tags)
tuple. Chunk into batches of `batch_size`. Workers receive a Batch each.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple

import pandas as pd

from services.dispatch.tag_map import TagMap


# A single dispatch unit: one symbol, its data, its level cache, and the set of
# detector names that should run against it this bar.
BatchItem = Tuple[str, pd.DataFrame, dict, set]


@dataclass
class Batch:
    items: list[BatchItem] = field(default_factory=list)


class DispatchPlanner:
    def __init__(self, batch_size: int):
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        self._batch_size = batch_size

    def plan(
        self,
        bar_ts: datetime,
        tag_map: TagMap,
        df5_by_symbol: dict,
        levels_by_symbol: dict,
    ) -> list[Batch]:
        items: list[BatchItem] = []
        for sym in sorted(tag_map.active_symbols()):
            df5 = df5_by_symbol.get(sym)
            if df5 is None or df5.empty:
                continue  # data not ready this bar — skip
            levels = levels_by_symbol.get(sym, {})
            tags = tag_map.active_tags(sym)
            if not tags:
                continue  # defensive: active_symbols() implies tags non-empty
            items.append((sym, df5, levels, tags))

        # Chunk
        batches: list[Batch] = []
        for i in range(0, len(items), self._batch_size):
            batches.append(Batch(items=items[i : i + self._batch_size]))
        return batches
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_planner.py -v`

Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/planner.py tests/dispatch/test_planner.py
git commit -m "dispatch: DispatchPlanner — assemble (sym, df5, levels, tags) batches"
```

### Task 9: Worker entry point + detector cache

**Files:**
- Create: `services/dispatch/worker.py`
- Create: `tests/dispatch/test_worker.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dispatch/test_worker.py
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from services.dispatch.planner import Batch
from services.dispatch.worker import dispatch_worker_batch, _detector_cache, _registry_cache, init_worker


@pytest.fixture(autouse=True)
def clear_caches():
    _detector_cache.clear()
    yield
    _detector_cache.clear()


def test_worker_returns_empty_for_empty_batch():
    init_worker(registry=MagicMock())
    out = dispatch_worker_batch(Batch(items=[]))
    assert out == []


def test_worker_calls_detector_for_each_tag(monkeypatch):
    # Build a fake registry that returns a fake detector class
    fake_spec = MagicMock()
    fake_detector_instance = MagicMock()
    fake_detector_instance.detect.return_value = MagicMock(events=["evt1"])
    fake_spec.detector_class_path = "fake.path"
    fake_spec.raw_config = {}
    fake_registry = MagicMock()
    fake_registry.get.return_value = fake_spec

    def fake_import_path(path):
        return MagicMock(return_value=fake_detector_instance)

    monkeypatch.setattr("services.dispatch.worker._import_path", fake_import_path)
    init_worker(registry=fake_registry)

    df = pd.DataFrame({"close": [100.0]})
    levels = {"PDC": 100.0}
    batch = Batch(items=[("NSE:A", df, levels, {"gap_fade_short", "mis_unwind"})])
    events = dispatch_worker_batch(batch)
    # 2 tags x 1 event each
    assert len(events) == 2
    # Detector cached after first use per name
    assert "gap_fade_short" in _detector_cache
    assert "mis_unwind" in _detector_cache
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_worker.py -v`

Expected: ImportError.

- [ ] **Step 3: Implement the worker**

```python
# services/dispatch/worker.py
"""Worker-side entry point. Runs in ProcessPoolExecutor workers.

Holds a module-level detector instance cache keyed by setup name. Cache
populated lazily on first use within the worker process; survives across
batches for the worker's lifetime (= scan-loop lifetime).
"""
from __future__ import annotations

from typing import Optional

from services.dispatch.planner import Batch
from services.dispatch.setup_registry import SetupRegistry, _import_path


# Per-worker-process caches.
_detector_cache: dict = {}
_registry_cache: Optional[SetupRegistry] = None


def init_worker(registry: SetupRegistry) -> None:
    """Initialize worker-process state.

    Called from the parent on worker spawn, or in-test setup.
    """
    global _registry_cache
    _registry_cache = registry
    _detector_cache.clear()


def _get_detector(name: str):
    if name in _detector_cache:
        return _detector_cache[name]
    if _registry_cache is None:
        raise RuntimeError("worker not initialized — call init_worker first")
    spec = _registry_cache.get(name)
    cls = _import_path(spec.detector_class_path)
    instance = cls(spec.raw_config)
    _detector_cache[name] = instance
    return instance


def _build_market_context(sym: str, df5, levels: dict):
    """Build the MarketContext object detectors expect.

    Inlined here to avoid importing structures.data_models at worker-init time
    (keeps spawn-mode startup fast).
    """
    from structures.data_models import MarketContext
    return MarketContext(
        symbol=sym,
        df_5m=df5,
        levels=levels,
        # Other fields filled by detector-specific call paths; keep this
        # minimal — production wiring may need to plumb cap_segment, regime,
        # session_date through Batch.items if detectors require them.
    )


def dispatch_worker_batch(batch: Batch) -> list:
    """Process one batch: for each (sym, df5, levels, tags), run each tagged detector."""
    out = []
    for sym, df5, levels, tags in batch.items:
        ctx = _build_market_context(sym, df5, levels)
        for det_name in tags:
            det = _get_detector(det_name)
            analysis = det.detect(ctx)
            out.extend(analysis.events)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_worker.py -v`

Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add services/dispatch/worker.py tests/dispatch/test_worker.py
git commit -m "dispatch: worker entry point + per-process detector cache"
```

### Task 10: Wire dispatch into screener_live._run_5m_scan

**Files:**
- Modify: `services/screener_live.py` (rewrite `_run_5m_scan`; add registry/calendar/tagmap/scope/planner instances to ScreenerLive)

This is the largest single task — the scanner rewrite. Break it into substeps.

- [ ] **Step 1: Add registry/calendar/tagmap state to ScreenerLive.__init__**

Locate `class ScreenerLive` in `services/screener_live.py` and find its `__init__`. After existing init lines, add:

```python
from services.dispatch.setup_registry import SetupRegistry
from services.dispatch.transition_calendar import TransitionCalendar
from services.dispatch.tag_map import TagMap
from services.dispatch.fetch_scope import FetchScopeManager
from services.dispatch.planner import DispatchPlanner

# Dispatch state (Phase 1 refactor)
self.registry = SetupRegistry.load_from_config(self.raw_cfg)
self.registry.validate()
self.transition_calendar = TransitionCalendar.from_registry(self.registry)
self.tag_map = TagMap()
self.fetch_scope = FetchScopeManager()
self.planner = DispatchPlanner(batch_size=50)
self._last_scan_ts = None  # tracks calendar walk progress
```

- [ ] **Step 2: Locate and read the existing `_run_5m_scan` body**

Run: `grep -n "_run_5m_scan\|def _on_5m_close" services/screener_live.py`

Read enough surrounding context to understand the current method (the read tool with offset+limit on the method block).

- [ ] **Step 3: Rewrite `_run_5m_scan` to use dispatch pipeline**

Replace the body of `_run_5m_scan(self, bar_ts)`:

```python
def _run_5m_scan(self, bar_ts: datetime) -> None:
    """Per-bar scan using calendar-driven tag dispatch."""
    bar_t = bar_ts.time() if hasattr(bar_ts, "time") else bar_ts
    last_t = self._last_scan_ts.time() if self._last_scan_ts else _time_mod_min  # use a sentinel "before-open"

    # 1. Walk calendar — evolve tag map
    for ev in self.transition_calendar.events_in(after=last_t, until=bar_t):
        if ev.kind == "build_universe":
            spec = self.registry.get(ev.setup)
            builder = _import_path(spec.universe_builder_path)
            try:
                syms = builder(
                    self._df5_by_symbol,
                    getattr(self, "_daily_dict_cache", {}) or {},
                    bar_ts.date() if hasattr(bar_ts, "date") else bar_ts,
                    spec.raw_config,
                    self._cap_map,
                )
            except Exception as e:
                logger.warning("UNIVERSE_BUILD_FAILED | %s | %s", spec.name, e)
                syms = set()
            self.tag_map.add_universe(ev.setup, set(syms or []))
        elif ev.kind == "open_window":
            self.tag_map.open_window(ev.setup)
        elif ev.kind == "close_window":
            self.tag_map.close_window(ev.setup)

    # 2. Skip scan entirely if no active detectors
    active = self.tag_map.active_symbols()
    if not active:
        logger.info("SCAN_SKIPPED | no active detectors at bar %s", bar_ts)
        self._last_scan_ts = bar_ts
        return

    # 3. API fetch only for active syms (with backfill on re-entry)
    fetch_syms = self.fetch_scope.fetch_set(bar_ts, self.tag_map)
    self._fetch_with_backfill(fetch_syms, bar_ts)

    # 4. ORB levels (existing, cached at 09:30)
    if not self._levels_by_symbol:
        self._levels_by_symbol = self._compute_orb_levels_once(bar_ts, self._df5_by_symbol)

    # 5. Features for active syms only
    features = compute_bar_features(self._df5_by_symbol, active, bar_ts, self._levels_by_symbol)

    # 6. Plan + dispatch
    plan = self.planner.plan(bar_ts, self.tag_map, self._df5_by_symbol, self._levels_by_symbol)
    if not plan:
        logger.info("PLAN_EMPTY | bar %s", bar_ts)
        self._last_scan_ts = bar_ts
        return

    futures = [self._executor.submit(dispatch_worker_batch, batch) for batch in plan]
    all_events = []
    for fut in futures:
        try:
            all_events.extend(fut.result(timeout=30))
        except Exception as e:
            logger.exception("DISPATCH_BATCH_FAILED | %s", e)

    # 7. Downstream: gates → orchestrator → exec (unchanged)
    self._process_events_downstream(all_events, bar_ts)
    self._last_scan_ts = bar_ts
```

Add imports at top of file:

```python
from services.dispatch.worker import dispatch_worker_batch
from services.dispatch.setup_registry import _import_path
```

And the time-sentinel:

```python
_time_mod_min = time(0, 0)
```

- [ ] **Step 4: Add `_fetch_with_backfill` helper**

Inside `ScreenerLive`:

```python
def _fetch_with_backfill(self, fetch_syms: set, bar_ts: datetime) -> None:
    """Fetch single-bar for fresh syms, history-from-open for re-entries."""
    if not fetch_syms:
        return
    needs_backfill = [s for s in fetch_syms if self.fetch_scope.is_backfill_needed(s, self._df5_by_symbol, bar_ts)]
    fresh = [s for s in fetch_syms if s not in needs_backfill]

    if fresh:
        self._api_fetch_single_bar(fresh, bar_ts)  # existing helper
    if needs_backfill:
        logger.info("BACKFILL | %d syms re-entering active set", len(needs_backfill))
        self._api_fetch_history(needs_backfill, start_ts=bar_ts.replace(hour=9, minute=15, second=0))
```

If `_api_fetch_single_bar` / `_api_fetch_history` don't exist with that exact name, refactor the existing `async_fetch_intraday_5m_batch` call site (around `screener_live.py:1367`) into these helpers first.

- [ ] **Step 5: Initialize worker pool with init_worker**

Locate where `self._executor = ProcessPoolExecutor(...)` is created. After creation:

```python
from services.dispatch.worker import init_worker
# Initialize each worker with the registry (call via initializer kwarg)
self._executor = ProcessPoolExecutor(
    max_workers=worker_count,
    initializer=init_worker,
    initargs=(self.registry,),
)
```

- [ ] **Step 6: One-day smoke test**

Run: `.venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_task10.log 2>&1`

Expected: exit 0. Grep the log for new markers:

```bash
grep -E "SCAN_SKIPPED|PLAN_EMPTY|BACKFILL|DISPATCH_BATCH_FAILED" /tmp/smoke_task10.log | head -20
```

Expected: `SCAN_SKIPPED` lines for idle bars (10:35-10:55, 15:05+). `BACKFILL` lines if any symbol re-enters. No `DISPATCH_BATCH_FAILED`.

- [ ] **Step 7: Run dispatch unit tests**

Run: `.venv/Scripts/python -m pytest tests/dispatch/ -v`

Expected: all green.

- [ ] **Step 8: Commit**

```bash
git add services/screener_live.py
git commit -m "$(cat <<'EOF'
screener_live: rewrite _run_5m_scan to use calendar-driven dispatch

Per-bar pipeline:
  1. Walk TransitionCalendar -> mutate TagMap
  2. Skip scan entirely if no active detectors (idle windows)
  3. API-fetch only the active-tag set (backfill on re-entry)
  4. Compute features for active syms only
  5. Plan + dispatch (det, sym, chunk) batches to ProcessPool
  6. Reduce events -> existing gates/orchestrator/exec path

ProcessPoolExecutor workers initialized with SetupRegistry via
initializer kwarg; detector instances cached per-worker process.
EOF
)"
```

### Task 11: Strip dead cap_segment early-rejects from detectors

**Files:**
- Modify: `structures/gap_fade_short_structure.py`
- Modify: `structures/circuit_t1_fade_short_structure.py`
- Modify: `structures/delivery_pct_anomaly_short_structure.py`
- Modify: `structures/long_panic_gap_down_structure.py`
- Modify: `structures/circuit_release_fade_short_structure.py`
- Modify: `structures/round_number_sweep_short_structure.py`
- Modify: `structures/or_window_failure_fade_short_structure.py`
- Modify: `structures/mis_unwind_vwap_revert_short_structure.py`

- [ ] **Step 1: Locate the dead checks**

For each detector file, find the early-reject pattern. Example from `mis_unwind_vwap_revert_short_structure.py`:

```python
if not _wide_open and context.cap_segment not in self.allowed_caps:
    return _empty(f"Cap segment {context.cap_segment!r} not in allowed set")
```

Run: `grep -n "cap_segment not in\|not in self.allowed_caps" structures/*.py`

- [ ] **Step 2: Delete each occurrence**

For each match, delete the if-block (and its `not _wide_open` qualifier — wide_open_mode no longer needs to bypass this check since dispatch already filters).

The `self.allowed_caps = set(config["allowed_cap_segments"])` line in `__init__` STAYS — it's still used by the universe builder. Only the runtime check in `detect()` is removed.

- [ ] **Step 3: Run all structure tests**

Run: `.venv/Scripts/python -m pytest tests/structures/ -v`

Expected: all green. (Detector behavior is unchanged for symbols in their universe; the deleted check was only for off-universe symbols which dispatch now prevents.)

- [ ] **Step 4: Smoke test**

Run: `.venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_task11.log 2>&1`

Expected: same trade count + same trade outcomes as Task 10's smoke.

- [ ] **Step 5: Commit**

```bash
git add structures/
git commit -m "structures: strip dead cap_segment early-rejects from 8 detectors

Universe builders now filter by cap_segment before dispatch reaches the
detector. The runtime check in detect() was defensive-redundant and is
removed. allowed_cap_segments config key stays (used by builders)."
```

### Task 12: Delete the dual detector registry + main_detector.py

**Files:**
- Modify: `services/plan_orchestrator.py` (delete `_DETECTOR_REGISTRY` + `ACTIVE_SETUPS`)
- Delete: `structures/main_detector.py`

- [ ] **Step 1: Replace `ACTIVE_SETUPS` references in plan_orchestrator.py**

Find usages:

```bash
grep -n "ACTIVE_SETUPS\|_DETECTOR_REGISTRY" services/plan_orchestrator.py
```

Replace the module-level `_DETECTOR_REGISTRY` + `ACTIVE_SETUPS` blocks with:

```python
# Setup name validation comes from SetupRegistry (services/dispatch/setup_registry.py).
# Constructed at runtime from configuration.json — no dual source of truth.
from services.dispatch.setup_registry import SetupRegistry

_registry: Optional[SetupRegistry] = None


def _get_registry() -> SetupRegistry:
    """Lazy-load registry to avoid circular imports."""
    global _registry
    if _registry is None:
        from config.filters_setup import load_filters
        _registry = SetupRegistry.load_from_config(load_filters())
    return _registry


def _is_active_setup(setup_type: str) -> bool:
    """Replacement for `setup_type not in ACTIVE_SETUPS`."""
    try:
        spec = _get_registry().get(setup_type)
        return spec.enabled
    except KeyError:
        return False
```

Update the usage site (around line 671):

```python
# OLD:
#   if setup_type not in ACTIVE_SETUPS:
#       logger.warning(f"[ORCH] {symbol}: setup_type={setup_type} not in ACTIVE_SETUPS")
# NEW:
if not _is_active_setup(setup_type):
    logger.warning(f"[ORCH] {symbol}: setup_type={setup_type} not enabled in registry")
```

And around line 226 (the `_DETECTOR_REGISTRY.get(setup_type)` use):

```python
# OLD: cls = _DETECTOR_REGISTRY.get(setup_type)
# NEW:
try:
    spec = _get_registry().get(setup_type)
    cls = _import_path(spec.detector_class_path) if spec.enabled else None
except KeyError:
    cls = None
```

Add import:

```python
from services.dispatch.setup_registry import _import_path
```

- [ ] **Step 2: Run pytest**

Run: `.venv/Scripts/python -m pytest tests/ -x --tb=short 2>&1 | tail -30`

Expected: all green.

- [ ] **Step 3: Delete main_detector.py**

```bash
git rm structures/main_detector.py
```

- [ ] **Step 4: Find and fix any imports of main_detector**

Run: `grep -rn "from structures.main_detector\|import structures.main_detector\|MainDetector" services/ tools/ tests/ main.py 2>&1`

For each hit:
- If in a worker import path (likely `services/screener_live.py:_init_worker_with_factories`), replace with `from services.dispatch.worker import dispatch_worker_batch, init_worker`
- If in a test that mocks MainDetector, rewrite the test to mock `services.dispatch.worker.dispatch_worker_batch` instead (or delete the test if it's redundant with the new dispatch unit tests)

- [ ] **Step 5: Smoke test**

Run: `.venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_task12.log 2>&1`

Expected: exit 0. Trade count + outcomes same as Task 11's smoke.

- [ ] **Step 6: Commit**

```bash
git add -A services/plan_orchestrator.py structures/
git commit -m "$(cat <<'EOF'
delete dual detector registry + main_detector.py

Single source of truth is now SetupRegistry (services/dispatch/setup_registry.py)
loaded from configuration.json. plan_orchestrator + worker dispatch both
read from it. Eliminates the dual-registry drift bug class
(d0caa7d-style: adding setup to one but not the other).
EOF
)"
```

### Task 13: Delete hardcoded scanner blocks from screener_live

**Files:**
- Modify: `services/screener_live.py`

- [ ] **Step 1: Delete `_universe_union()` method**

Locate `def _universe_union(self)` (around line 741) and delete the whole method. References to it should already be gone from Task 10 (the new `_run_5m_scan` uses `self.tag_map.active_symbols()`).

Verify no remaining references:

```bash
grep -n "_universe_union" services/screener_live.py
```

Expected: zero hits.

- [ ] **Step 2: Delete hardcoded universe-union if-blocks (lines ~1438-1444)**

These lines inline the universe computation:

```python
universe = set()
for u in (self._setup_universes or {}).values():
    universe.update(u)
if self._gap_fade_universe:
    universe.update(self._gap_fade_universe)
if self._long_panic_gap_down_universe:
    universe.update(self._long_panic_gap_down_universe)
```

All dead now — replaced by `self.tag_map.active_symbols()` in the rewritten scanner. Delete the block + the surrounding code that used `universe` (the `feats_df = compute_bar_features(...)` call already moved to Task 10's `_run_5m_scan`).

- [ ] **Step 3: Delete hardcoded lazy-build if-blocks (lines ~1485-1541)**

These three blocks build gap_fade / long_panic_gap_down / circuit_release universes at specific bars. All dead now — `TransitionCalendar` walks the registry and fires `build_universe` events instead.

Delete:
- `if self._gap_fade_universe is None and ...` block
- `if self._long_panic_gap_down_universe is None and ...` block
- `if self._circuit_release_fade_universe is None and ...` block

Also delete the class-level state fields they populated (if no other code reads them):

```bash
grep -n "_gap_fade_universe\|_long_panic_gap_down_universe\|_circuit_release_fade_universe" services/screener_live.py
```

Delete the `self._..._universe: Optional[Set[str]] = None` init lines too.

- [ ] **Step 4: Verify no remaining references**

```bash
grep -n "_gap_fade_universe\|_long_panic_gap_down_universe\|_circuit_release_fade_universe\|_universe_union" services/screener_live.py
```

Expected: zero hits.

- [ ] **Step 5: Run full test suite**

Run: `.venv/Scripts/python -m pytest tests/ -x --tb=short 2>&1 | tail -30`

Expected: all green.

- [ ] **Step 6: Smoke test**

Run: `.venv/Scripts/python main.py --dry-run --session-date 2024-05-03 > /tmp/smoke_task13.log 2>&1`

Expected: exit 0. Trade count + outcomes match Task 12's smoke.

- [ ] **Step 7: Commit**

```bash
git add services/screener_live.py
git commit -m "screener_live: delete hardcoded universe blocks (now via SetupRegistry)

_universe_union(), the universe-union if-blocks, and the three hardcoded
lazy-build if-blocks all replaced by TransitionCalendar driving the
TagMap. SetupRegistry is the sole source of truth for which setups exist,
their universe builders, and their trigger times."
```

### Task 14: End-to-end integration test

**Files:**
- Create: `tests/dispatch/test_dispatch_e2e.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/dispatch/test_dispatch_e2e.py
"""End-to-end: feed a synthetic bar stream, assert dispatch matches expected (det, sym) pairs."""
import pandas as pd
import pytest
from datetime import datetime, time
from unittest.mock import MagicMock, patch

from services.dispatch.setup_registry import SetupSpec, SetupRegistry, Trigger
from services.dispatch.transition_calendar import TransitionCalendar
from services.dispatch.tag_map import TagMap
from services.dispatch.planner import DispatchPlanner


def test_e2e_morning_dispatch_pattern():
    """At 09:30, only gap_fade (closing) + delivery_pct (opening) should be active."""
    # Build minimal registry
    reg = SetupRegistry({})
    reg._specs = {
        "gap_fade_short": SetupSpec(
            "gap_fade_short", True,
            "structures.gap_fade_short_structure.GapFadeShortStructure",
            "services.setup_universe.gap_fade_universe",
            Trigger.bar(time(9, 15)),
            (time(9, 15), time(9, 30)),
            {},
        ),
        "delivery_pct_anomaly_short": SetupSpec(
            "delivery_pct_anomaly_short", True,
            "structures.delivery_pct_anomaly_short_structure.DeliveryPctAnomalyShortStructure",
            "services.setup_universe.delivery_pct_universe",
            Trigger.session_start(),
            (time(9, 30), time(10, 30)),
            {},
        ),
    }
    cal = TransitionCalendar.from_registry(reg)
    tm = TagMap()

    # Simulate walking from session-open through to 09:30 bar
    for ev in cal.events_in(after=time(0, 0), until=time(9, 30)):
        if ev.kind == "build_universe":
            # mock builder output
            syms = {"NSE:GAP1", "NSE:GAP2"} if ev.setup == "gap_fade_short" else {"NSE:DEL1"}
            tm.add_universe(ev.setup, syms)
        elif ev.kind == "open_window":
            tm.open_window(ev.setup)
        elif ev.kind == "close_window":
            tm.close_window(ev.setup)

    # At 09:30 (close of gap_fade + open of delivery_pct), expected active:
    # - gap_fade still open (close happens AT 09:30, BUT events_in returns it; verify order)
    # - delivery_pct just opened
    active = tm.active_symbols()
    # close_window for gap_fade fires AT 09:30 too. Our calendar sorts open before close at same minute,
    # so by end-of-walk: gap_fade is CLOSED.
    assert "NSE:GAP1" not in active
    assert "NSE:DEL1" in active


def test_e2e_idle_bar_skips_scan():
    """At 10:35 (between or_window_failure close and round_number open), zero active."""
    reg = SetupRegistry({})
    reg._specs = {
        "or_window_failure_fade_short": SetupSpec(
            "or_window_failure_fade_short", True,
            "x.Y", "x.y", Trigger.session_start(),
            (time(9, 30), time(10, 30)), {},
        ),
        "round_number_sweep_short": SetupSpec(
            "round_number_sweep_short", True,
            "x.Y", "x.y", Trigger.session_start(),
            (time(11, 0), time(12, 30)), {},
        ),
    }
    cal = TransitionCalendar.from_registry(reg)
    tm = TagMap()
    for ev in cal.events_in(after=time(0, 0), until=time(10, 35)):
        if ev.kind == "build_universe":
            tm.add_universe(ev.setup, {"NSE:A"})
        elif ev.kind == "open_window":
            tm.open_window(ev.setup)
        elif ev.kind == "close_window":
            tm.close_window(ev.setup)

    # or_window_failure closed at 10:30. round_number opens at 11:00. At 10:35: idle.
    assert tm.active_symbols() == set()
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python -m pytest tests/dispatch/test_dispatch_e2e.py -v`

Expected: 2 PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/dispatch/test_dispatch_e2e.py
git commit -m "dispatch: e2e tests for morning + idle-bar dispatch patterns"
```

---

## Phase 2 — Verification and rollout

### Task V1: Baseline OCI run on pre-refactor commit

- [ ] **Step 1: Identify the pre-refactor commit**

```bash
git log --oneline main..research/post-sebi-edge-setups | tail -5
```

Note the parent of the first Phase 0 commit (the last `main`-merged commit before the refactor branch).

- [ ] **Step 2: Run OCI Discovery on baseline**

User-driven. The OCI run is a manual cloud submission (per project conventions). Submit it on the parent-of-Phase-0 commit hash. When complete, capture the output dir (e.g., `20260518-XXXXXX_baseline`).

- [ ] **Step 3: Record baseline location**

Add a one-line note:

```bash
mkdir -p docs
echo "Baseline OCI for Phase 1 byte-diff: 20260518-XXXXXX_baseline (commit <HASH>)" > docs/phase1_oci_baseline.txt
```

(This is local-only — no commit needed; cleanup happens at end of Phase 2.)

### Task V2: Refactored OCI run

- [ ] **Step 1: Run OCI Discovery on the refactored branch**

User-driven. Submit OCI run on the tip of the Phase 1 PR branch.

- [ ] **Step 2: Diff trade_report.csv per session**

Create a one-shot diff script:

```python
# tools/diff_oci_runs.py
"""Per-session trade-by-trade diff of two OCI runs."""
import sys
import glob
from pathlib import Path
import pandas as pd


KEY_COLS = ["symbol", "entry_ts", "entry_price", "exit_ts", "exit_price",
            "last_exit_reason", "realized_pnl", "setup_type"]


def diff_run(baseline_dir, refactor_dir):
    baseline_sessions = {Path(f).parent.name for f in glob.glob(f"{baseline_dir}/*/trade_report.csv")}
    refactor_sessions = {Path(f).parent.name for f in glob.glob(f"{refactor_dir}/*/trade_report.csv")}
    common = baseline_sessions & refactor_sessions
    diffs = []
    for session in sorted(common):
        b = pd.read_csv(f"{baseline_dir}/{session}/trade_report.csv")
        r = pd.read_csv(f"{refactor_dir}/{session}/trade_report.csv")
        if len(b) != len(r):
            diffs.append((session, f"row count {len(b)} -> {len(r)}"))
            continue
        # Sort both by trade_id then compare key columns
        if "trade_id" in b.columns:
            b = b.sort_values("trade_id").reset_index(drop=True)
            r = r.sort_values("trade_id").reset_index(drop=True)
        for col in KEY_COLS:
            if col not in b.columns:
                continue
            if not b[col].equals(r[col]):
                n_diff = (b[col] != r[col]).sum()
                diffs.append((session, f"{col}: {n_diff} cell differences"))
    return diffs


if __name__ == "__main__":
    baseline = sys.argv[1]
    refactor = sys.argv[2]
    diffs = diff_run(baseline, refactor)
    if not diffs:
        print("PASS — no differences across all sessions")
    else:
        print(f"FAIL — {len(diffs)} session(s) with differences:")
        for session, desc in diffs[:50]:
            print(f"  {session}: {desc}")
        if len(diffs) > 50:
            print(f"  ... and {len(diffs) - 50} more")
```

- [ ] **Step 3: Run the diff**

```bash
.venv/Scripts/python tools/diff_oci_runs.py 20260518-XXXXXX_baseline 20260518-YYYYYY_refactor
```

Expected: `PASS — no differences across all sessions`.

- [ ] **Step 4: If diffs surface, triage**

For each diff: open both `trade_report.csv` files, find the differing row by `trade_id`, identify the missing/extra trade. Common causes:
- Universe builder hookup missed in Task 4 config keys → check the setup's `universe_builder` value
- TransitionCalendar event timing off-by-one minute → revisit calendar sort
- Backfill failed to fetch full history for a re-entering symbol → check fetch_scope.is_backfill_needed
Fix, push to the Phase 1 PR branch, re-run V2 from Step 1.

- [ ] **Step 5: Once diff is PASS, delete the one-shot diff script**

```bash
git rm tools/diff_oci_runs.py
rm docs/phase1_oci_baseline.txt 2>/dev/null
git commit -m "verify: remove one-shot OCI diff script (Phase 1 verified byte-identical)"
```

### Task V3: Paper-trading dress rehearsal

- [ ] **Step 1: Paper-trade for one full session on the refactored branch**

```bash
.venv/Scripts/python main.py --paper-trading
```

Run for one normal trading day. Monitor for:
- `SCAN_SKIPPED` log lines during idle windows (good — confirms dispatch is skipping)
- `BACKFILL` log lines when symbols re-enter active set (good)
- Zero `DISPATCH_BATCH_FAILED` (mandatory)
- Trade fires match what the OCI run for the same day showed

- [ ] **Step 2: At end-of-day, compare paper output to OCI output for the same day**

```bash
.venv/Scripts/python tools/diff_oci_runs.py 20260519_paper logs/paper_<session>
```

(Or eye-diff the `trade_report.csv` if `diff_oci_runs.py` was already deleted in V2.)

Expected: paper trades match same-day OCI Discovery 1:1 (modulo OCI's wider 15:25 cutoff which paper doesn't use unless `RUN_MODE=oci_research`).

### Task V4: Live cut-over

- [ ] **Step 1: Pick a low-volatility trading day**

Avoid expiry days, FOMC days, budget announcements. Communicate with user before flipping.

- [ ] **Step 2: Merge the Phase 1 PR**

User-driven git operation.

- [ ] **Step 3: Live-monitor the first 30 minutes**

Watch agent.log for:
- Successful scan completion each 5-min bar
- Expected trade fires for active setups
- Zero unhandled exceptions

- [ ] **Step 4: Revert plan ready**

If anything anomalous: `git revert <Phase 1 PR commit>`, push, restart. The refactor is pure code; no data migration to roll back.

---

## Done criteria

This refactor is done when:

1. Phase 0 PR A merged: ~100 dead config keys removed; smoke + pytest green
2. Phase 0 PR B merged: mode_profiles inline; OCI behavior unchanged; sub8_oci_overrides.json + apply_oci_override.py deleted
3. Phase 1 PR merged (includes PR C cleanup): SetupRegistry is the single source of truth; OCI byte-diff PASS against baseline
4. One day of paper trading clean on refactored branch
5. One day of live trading clean
6. All deleted files / config keys recorded in commit messages + `docs/config_keys_removed_2026-05-17.md`
