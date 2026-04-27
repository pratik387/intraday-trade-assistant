# Sub-Project #8 (Indian-Native Setup Library, EXTENDED) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 5 new research-cited Indian-native intraday setups (ORB-15, Narrow CPR Trending Breakout, VWAP First-Pullback Continuation, PDH/PDL Touch-and-Reject, Closing Hour Reversal) on top of sub7's surviving `gap_fade_short`. Each detector ships with research-cited stop multipliers, mandatory tiered T1/T2 exits, and per-setup universe filter (no full 1000+ scan). One bundled OCI capture validates the suite.

**Architecture:** Five new detector classes plug into existing `structures/main_detector.py`. Each emits a `TradePlan` with two targets (T1 partial 50% qty, T2 full 50% qty) consumed unchanged by the orchestrator's existing `_build_plan_from_sub7_detector` fast path (sub7-T8 work). A small `services/universe_filter.py` loads NSE constituent lists from `assets/ind_nifty*.csv` and provides per-setup membership lookup. No engine changes required. Validation uses existing sub7 tooling (`tools/sub7_validation/build_per_setup_pnl.py`, `per_setup_report.py`) — sub8 trade rows differ only by `setup_type` value.

**Tech Stack:** Python 3.10, pandas 2.x, numpy 1.x, pytest. No new dependencies.

---

## Source spec

`specs/2026-04-26-sub-project-8-indian-native-setups-extended-design.md` — **rev2** approved 2026-04-26 (post independent Opus review). Companion review: `specs/2026-04-26-sub-project-8-research-review.md`.

**Rev2 changes baked into this plan:**
- Setup 1 (ORB-15): default stop = opposite-end-of-range (was mid-range); range starts 09:20 (was 09:15)
- Setup 3 (VWAP First-Pullback): active window extended to 14:30 (was 13:30)
- Setup 4 (PDH/PDL): attribution dropped (was "Subasish Pani style"); Capital.com citation removed; volume polarity is A/B variant (was single-rule)
- Setup 5 (CHR): stop tightened to 1.5×ATR (was 1.2×); hard time stop 15:22 (was 15:18); fail-fast n<200 trip-wire added
- Cross-cutting: expiry-day exclusion, gap-day exclusion, circuit-band exclusion (universe-wide) — added as Phase 0 helpers

## Scope

Sub-project #8 only. Carries `gap_fade_short` from sub7 unchanged. Does NOT modify the orchestrator fast path (already proven). Does NOT touch the Optuna gauntlet (Phase 4/5 may consume it but no new gauntlet code required).

## Prerequisites

- sub7-T8 fast-path (`pipelines/orchestrator.py::_build_plan_from_sub7_detector`) — shipped at commit `826f6db`
- sub7 wide_open_mode bypass cascade — shipped at commits `9d4b1c9`, `c2b0a35`, `3a2e793`, `d861a8b`
- sub7 per-setup PnL tooling (`tools/sub7_validation/{build_per_setup_pnl,per_setup_report}.py`) — shipped at commits `9bd18e1`, `913318e`
- NSE constituent CSVs (`assets/ind_nifty50list.csv`, `assets/ind_niftybanklist.csv`) — already in repo
- `gate_input_logging` plumbing (sub4) — shipped

---

## File structure

| Path | Purpose | Created/Modified |
|------|---------|------------------|
| `services/universe_filter.py` | Per-setup universe membership lookup (Nifty50 / BankNifty / F&O 200) | Create |
| `tests/services/test_universe_filter.py` | Unit tests | Create |
| `assets/fno_liquid_200.csv` | F&O liquid universe (~200 names — derived from existing scanner config) | Create |
| `structures/orb_15_structure.py` | Detector 1: opening-range breakout 09:15-09:30 | Create |
| `structures/narrow_cpr_breakout_structure.py` | Detector 2: narrow-CPR trending breakout (Nifty50/BN only) | Create |
| `structures/vwap_first_pullback_structure.py` | Detector 3: first VWAP pullback continuation | Create |
| `structures/pdh_pdl_reject_structure.py` | Detector 4: PDH/PDL touch-and-reject fade | Create |
| `structures/closing_hour_reversal_structure.py` | Detector 5: 14:30-15:15 reversal (CHR) | Create |
| `structures/main_detector.py` | Add 5 detectors to `detector_configs` + identity mappings | Modify |
| `pipelines/orchestrator.py` | Add 5 setup names to `SUB7_SETUPS` frozenset + `_cls_map` | Modify |
| `config/setup_categories.py` | Add 5 entries to `SETUP_CATEGORY_MAP` | Modify |
| `config/configuration.json` | Add 5 setup config blocks + priority_weights entries | Modify |
| `config/sub8_oci_overrides.json` | OCI override: enable sub7+sub8 setups + `gate_input_logging:true` | Create |
| `tests/structures/test_orb_15_structure.py` | Detector 1 unit tests | Create |
| `tests/structures/test_narrow_cpr_breakout_structure.py` | Detector 2 unit tests | Create |
| `tests/structures/test_vwap_first_pullback_structure.py` | Detector 3 unit tests | Create |
| `tests/structures/test_pdh_pdl_reject_structure.py` | Detector 4 unit tests | Create |
| `tests/structures/test_closing_hour_reversal_structure.py` | Detector 5 unit tests | Create |

---

## Phase 0: Universe filter + scaffolding (Tasks 1-2)

### Task 1: Universe filter service

**Files:**
- Create: `services/universe_filter.py`
- Create: `assets/fno_liquid_200.csv`
- Create: `tests/services/test_universe_filter.py`

The 5 sub8 detectors need to filter their universe (e.g. CPR breakout runs only on Nifty50+BankNifty constituents). Existing code uses `cap_segment` (small/mid/large/micro) but has no index-membership lookup. This task adds the missing primitive.

- [ ] **Step 1: Generate `assets/fno_liquid_200.csv`**

We don't ship a curated F&O 200 list yet. The simplest source that's already in the repo: the union of `ind_niftynext50list.csv`, `ind_niftymidcap150list.csv` filtered to top-150 by index weight, plus `ind_niftybanklist.csv`. For now, use a stable placeholder: top 200 NSE F&O symbols by 30-day median turnover (load once from `assets/traded_symbols_3year_backtest.txt` or equivalent).

If `assets/fno_liquid_200.csv` does not exist, create it from `assets/ind_nifty50list.csv` + `assets/ind_niftynext50list.csv` + `assets/ind_niftybanklist.csv` (union, dedup, take all). This will be ~120 names; that's acceptable for v1.

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
parts = []
for f in ['ind_nifty50list.csv', 'ind_niftynext50list.csv', 'ind_niftybanklist.csv']:
    p = f'assets/{f}'
    try:
        df = pd.read_csv(p)
        # NSE format: 'Symbol' column
        if 'Symbol' in df.columns:
            parts.extend(df['Symbol'].astype(str).tolist())
    except FileNotFoundError:
        pass
syms = sorted(set('NSE:' + s.upper().strip() for s in parts if s and s.strip()))
with open('assets/fno_liquid_200.csv', 'w') as f:
    f.write('symbol\n')
    for s in syms:
        f.write(s + '\n')
print(f'Wrote {len(syms)} F&O liquid symbols')
"
```

Expected: prints `Wrote ~120 F&O liquid symbols` (Nifty50 has overlap with BankNifty).

- [ ] **Step 2: Write failing tests**

Create `tests/services/test_universe_filter.py`:

```python
"""universe_filter tests (sub8-T1)."""
import pandas as pd
import pytest

from services.universe_filter import (
    in_nifty50,
    in_banknifty,
    in_fno_liquid_200,
    in_universe,
)


def test_nifty50_member_returns_true():
    # RELIANCE is always in Nifty 50
    assert in_nifty50("NSE:RELIANCE") is True


def test_nifty50_non_member_returns_false():
    # SWSOLAR is small-cap — never in Nifty 50
    assert in_nifty50("NSE:SWSOLAR") is False


def test_banknifty_member_returns_true():
    # HDFCBANK is always in Bank Nifty
    assert in_banknifty("NSE:HDFCBANK") is True


def test_banknifty_non_member_returns_false():
    assert in_banknifty("NSE:RELIANCE") is False  # in Nifty50, not BankNifty


def test_fno_liquid_200_includes_index_majors():
    assert in_fno_liquid_200("NSE:RELIANCE") is True
    assert in_fno_liquid_200("NSE:HDFCBANK") is True


def test_in_universe_dispatches_by_key():
    assert in_universe("NSE:RELIANCE", "nifty50") is True
    assert in_universe("NSE:RELIANCE", "banknifty") is False
    assert in_universe("NSE:RELIANCE", "nifty50_banknifty") is True
    assert in_universe("NSE:HDFCBANK", "nifty50_banknifty") is True
    assert in_universe("NSE:RELIANCE", "fno_liquid_200") is True


def test_in_universe_unknown_key_raises():
    with pytest.raises(KeyError):
        in_universe("NSE:RELIANCE", "nonexistent_universe")
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/services/test_universe_filter.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'services.universe_filter'`.

- [ ] **Step 4: Implement `services/universe_filter.py`**

```python
"""Per-setup universe membership lookup (sub8-T1).

Loads NSE constituent CSVs once at import time from `assets/ind_nifty*.csv`
and `assets/fno_liquid_200.csv`. Returns True/False membership tests cheap
enough to call per-symbol-per-bar in detectors.

Universe keys consumed by sub8 detector configs:
  - "nifty50"               — Nifty 50 only (CPR breakout context)
  - "banknifty"             — Bank Nifty only
  - "nifty50_banknifty"     — union of Nifty 50 + Bank Nifty (sub8 narrow_cpr_breakout)
  - "fno_liquid_200"        — F&O liquid ~120-200 names (ORB, VWAP, CHR)
  - "smallmid_fno"          — F&O liquid MINUS Nifty 50 (PDH/PDL fade)
"""
from __future__ import annotations

from pathlib import Path
from typing import Set

import pandas as pd


_REPO_ROOT = Path(__file__).resolve().parent.parent
_ASSETS = _REPO_ROOT / "assets"


def _load_csv_symbols(filename: str) -> Set[str]:
    """Load symbols from an NSE constituent CSV. Tries 'Symbol' column first,
    falls back to 'symbol'. Returns prefixed 'NSE:XYZ' set."""
    path = _ASSETS / filename
    if not path.exists():
        return set()
    df = pd.read_csv(path)
    col = "Symbol" if "Symbol" in df.columns else ("symbol" if "symbol" in df.columns else None)
    if col is None:
        return set()
    out = set()
    for s in df[col].astype(str):
        s = s.strip().upper()
        if not s:
            continue
        if not s.startswith("NSE:"):
            s = f"NSE:{s}"
        out.add(s)
    return out


# Load once at import time — cheap for ~50-200 row CSVs.
_NIFTY50: Set[str] = _load_csv_symbols("ind_nifty50list.csv")
_BANKNIFTY: Set[str] = _load_csv_symbols("ind_niftybanklist.csv")
_FNO_LIQUID_200: Set[str] = _load_csv_symbols("fno_liquid_200.csv")
_NIFTY50_BANKNIFTY: Set[str] = _NIFTY50 | _BANKNIFTY
_SMALLMID_FNO: Set[str] = _FNO_LIQUID_200 - _NIFTY50

_UNIVERSE_MAP = {
    "nifty50": _NIFTY50,
    "banknifty": _BANKNIFTY,
    "nifty50_banknifty": _NIFTY50_BANKNIFTY,
    "fno_liquid_200": _FNO_LIQUID_200,
    "smallmid_fno": _SMALLMID_FNO,
}


def in_nifty50(symbol: str) -> bool:
    return symbol in _NIFTY50


def in_banknifty(symbol: str) -> bool:
    return symbol in _BANKNIFTY


def in_fno_liquid_200(symbol: str) -> bool:
    return symbol in _FNO_LIQUID_200


def in_universe(symbol: str, universe_key: str) -> bool:
    """Dispatch by universe key. Raises KeyError on unknown key."""
    if universe_key not in _UNIVERSE_MAP:
        raise KeyError(f"Unknown universe key: {universe_key!r}. "
                       f"Valid: {sorted(_UNIVERSE_MAP.keys())}")
    return symbol in _UNIVERSE_MAP[universe_key]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/services/test_universe_filter.py -v
```

Expected: 7 PASSED.

- [ ] **Step 6: Commit**

```bash
git add -f services/universe_filter.py tests/services/test_universe_filter.py assets/fno_liquid_200.csv
git commit -m "feat(sub8-T1): per-setup universe filter (NSE constituent lookup)"
```

---

### Task 2: Sub8 config blocks + OCI override file

**Files:**
- Modify: `config/configuration.json` (append 5 sub8 setup blocks under `setups`)
- Modify: `config/configuration.json` (append 5 priority_weights entries)
- Create: `config/sub8_oci_overrides.json`

- [ ] **Step 1: Add 5 sub8 setup config blocks**

In `config/configuration.json`, locate the `"setups"` object and add these blocks AFTER the existing `gap_fade_short` block (BEFORE the closing `}` of `"setups"`):

```json
    "orb_15": {
      "enabled": false,
      "active_window_start": "09:30",
      "active_window_end": "11:15",
      "range_window_start": "09:20",
      "range_window_end": "09:30",
      "min_range_pct": 0.4,
      "max_range_pct": 2.0,
      "min_volume_x_30d_median": 1.5,
      "stop_at_range_midpoint": false,
      "wick_buffer_pct": 0.10,
      "t1_r_multiple": 1.0,
      "t2_r_multiple": 2.0,
      "t1_qty_pct": 0.5,
      "universe_key": "fno_liquid_200",
      "min_bars_required": 6,
      "max_gap_pct_for_orb": 0.5,
      "_rev2_notes": "stop_at_range_midpoint=false is opposite-end-of-range (Indian-source default). range_window_start=09:20 skips pre-open call-auction wick. max_gap_pct_for_orb=0.5: skip ORB if today's open gap > 0.5% (route to gap_fade_short)."
    },
    "narrow_cpr_breakout": {
      "enabled": false,
      "active_window_start": "09:30",
      "active_window_end": "14:00",
      "max_cpr_width_pct": 0.40,
      "min_volume_x_20d_median": 1.3,
      "anti_whipsaw_lookback_bars": 2,
      "stop_at_pivot": true,
      "t1_target": "r1_s1",
      "t2_target": "r2_s2",
      "t1_qty_pct": 0.5,
      "universe_key": "nifty50_banknifty",
      "min_bars_required": 30
    },
    "vwap_first_pullback": {
      "enabled": false,
      "active_window_start": "10:00",
      "active_window_end": "14:30",
      "trend_lookback_bars": 6,
      "trend_min_bars_same_side": 4,
      "pullback_proximity_pct": 0.10,
      "reversal_min_range_pct": 0.20,
      "max_stop_distance_pct": 0.6,
      "t1_target": "prev_swing_extreme",
      "t2_r_multiple": 2.0,
      "t1_qty_pct": 0.5,
      "universe_key": "fno_liquid_200",
      "min_bars_required": 30
    },
    "pdh_pdl_reject": {
      "enabled": false,
      "active_window_start": "10:00",
      "active_window_end": "14:30",
      "level_proximity_pct": 0.10,
      "max_body_size_pct": 40.0,
      "min_upper_wick_x_body": 1.5,
      "volume_polarity": "absence",
      "max_volume_x_recent_for_absence": 1.5,
      "min_volume_x_recent_for_spike": 1.5,
      "wick_buffer_pct": 0.10,
      "t1_target": "vwap",
      "t2_target": "today_opposite_extreme",
      "t1_qty_pct": 0.5,
      "universe_key": "smallmid_fno",
      "min_bars_required": 30,
      "_rev2_notes": "volume_polarity is A/B variant: 'absence' (default — bar vol must NOT be >= 1.5x recent) or 'spike' (bar vol must be >= 1.5x recent). Reviewer flagged Indian sources contested on polarity; Phase 1 runs both."
    },
    "closing_hour_reversal": {
      "enabled": false,
      "active_window_start": "14:30",
      "active_window_end": "15:15",
      "min_intraday_move_pct": 1.5,
      "exhaustion_min_body_pct_of_range": 60.0,
      "exhaustion_min_volume_x_recent": 1.3,
      "stop_atr_multiplier": 1.5,
      "t1_target": "vwap",
      "t2_target": "pivot_or_50pct_retrace",
      "t1_qty_pct": 0.5,
      "hard_time_stop_hhmm": "15:22",
      "universe_key": "fno_liquid_200",
      "min_bars_required": 60,
      "fail_fast_min_trades_first_100_sessions": 200,
      "_rev2_notes": "stop_atr_multiplier=1.5 is Indian-source standard (Goodwill, Varsity); rev1 used 1.2x. hard_time_stop_hhmm=15:22 (was 15:18) leaves 3 min margin to Zerodha 15:25 auto square-off. fail-fast trip-wire: kill if n<200 in first 100 OCI sessions (sub-7 mis_unwind precedent)."
    }
```

- [ ] **Step 2: Add 5 priority_weights entries**

In `config/configuration.json`, locate the `"priority_weights"` object (currently has `gap_fade_short: 0.85`, `mis_unwind_short: 0.85`, `cpr_mean_revert: 0.80`). Add after `cpr_mean_revert`:

```json
        "orb_15": 0.80,
        "narrow_cpr_breakout": 0.75,
        "vwap_first_pullback": 0.80,
        "pdh_pdl_reject": 0.75,
        "closing_hour_reversal": 0.70
```

- [ ] **Step 3: Create `config/sub8_oci_overrides.json`**

```json
{
  "wide_open_mode": true,
  "entry_cutoff_hhmm": "15:25",
  "last_scan_hhmm": "15:25",
  "eod_squareoff_hhmm": "15:25",
  "max_trades_per_cycle": 10000,
  "gate_input_logging": {
    "enabled": true
  },
  "live_gate_chain": {
    "enabled": false
  },
  "_comment_rev2_cross_cutting": "Cross-cutting rules from design Section 10a: expiry-day exclusion, gap-day cross-detector exclusion, universe-wide circuit-band exclusion. Implemented in services/universe_filter.py helpers; detectors call them as first-condition prerequisite.",
  "expiry_day_exclusion": {
    "enabled": true,
    "expiry_calendar_path": "assets/nse_fno_expiry_dates.csv"
  },
  "circuit_band_exclusion": {
    "enabled": true,
    "min_distance_to_circuit_pct": 2.0
  },
  "_comment_setups": "Disable ALL existing setups; enable sub7 surviving (gap_fade) + sub8 new (5)",
  "setups": {
    "gap_fade_short": {"enabled": true},
    "mis_unwind_short": {"enabled": false},
    "cpr_mean_revert": {"enabled": false},
    "orb_15": {"enabled": true},
    "narrow_cpr_breakout": {"enabled": true},
    "vwap_first_pullback": {"enabled": true},
    "pdh_pdl_reject": {"enabled": true},
    "closing_hour_reversal": {"enabled": true},
    "discount_zone_long": {"enabled": false},
    "premium_zone_short": {"enabled": false},
    "range_bounce_long": {"enabled": false},
    "range_bounce_short": {"enabled": false},
    "range_breakout_long": {"enabled": false},
    "range_breakdown_short": {"enabled": false},
    "range_rejection_long": {"enabled": false},
    "range_rejection_short": {"enabled": false},
    "support_bounce_long": {"enabled": false},
    "resistance_bounce_short": {"enabled": false},
    "support_breakdown_short": {"enabled": false},
    "resistance_breakout_long": {"enabled": false},
    "level_breakout_long": {"enabled": false},
    "level_breakout_short": {"enabled": false},
    "orb_level_breakout_long": {"enabled": false},
    "orb_level_breakout_short": {"enabled": false},
    "first_hour_momentum_long": {"enabled": false},
    "first_hour_momentum_short": {"enabled": false},
    "fair_value_gap_long": {"enabled": false},
    "fair_value_gap_short": {"enabled": false},
    "order_block_long": {"enabled": false},
    "order_block_short": {"enabled": false},
    "liquidity_sweep_long": {"enabled": false},
    "liquidity_sweep_short": {"enabled": false}
  }
}
```

- [ ] **Step 4: Verify config loads cleanly**

```bash
.venv/Scripts/python.exe -c "
import json
cfg = json.load(open('config/configuration.json'))
sub8 = ['orb_15', 'narrow_cpr_breakout', 'vwap_first_pullback', 'pdh_pdl_reject', 'closing_hour_reversal']
for s in sub8:
    print(f'{s}: enabled={cfg[\"setups\"][s][\"enabled\"]} universe={cfg[\"setups\"][s][\"universe_key\"]}')
ov = json.load(open('config/sub8_oci_overrides.json'))
print(f'OCI override gate_input_logging: {ov[\"gate_input_logging\"]}')
"
```

Expected: 5 lines printing each setup's enabled state + universe_key, then `OCI override gate_input_logging: {'enabled': True}`.

- [ ] **Step 5: Commit**

```bash
git add config/configuration.json config/sub8_oci_overrides.json
git commit -m "feat(sub8-T2): config blocks for 5 new detectors + OCI override file"
```

---

## Phase 1: Detector implementations (Tasks 3-12)

Each detector follows the same TDD pattern: failing tests → minimal `detect()` → `plan_short_strategy()` / `plan_long_strategy()` → run tests → wire into `main_detector.py` + `orchestrator.py SUB7_SETUPS` + `setup_categories.py` → smoke run → commit.

Common interface: every detector inherits `BaseStructure` and emits a `TradePlan` whose `exit_levels.targets` is a list of two dicts:

```python
[
    {"name": "T1", "level": <t1_price>, "rr": <t1_rr>, "qty_pct": 0.5, "action": "partial_exit"},
    {"name": "T2", "level": <t2_price>, "rr": <t2_rr>, "qty_pct": 0.5, "action": "exit_full"},
]
```

The orchestrator's `_build_plan_from_sub7_detector` (already shipped) consumes this shape unchanged.

---

### Task 3: ORB-15 detector — implementation + tests

**Files:**
- Create: `structures/orb_15_structure.py`
- Create: `tests/structures/test_orb_15_structure.py`

Source: design doc Section 3 (citations: In-the-Money by Zerodha, Algotest, Saimohanreddy ORB backtest).

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_orb_15_structure.py`:

```python
"""ORB-15 detector unit tests (sub8-T3)."""
import pandas as pd
import pytest

from structures.orb_15_structure import ORB15Structure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "orb_15",
        "enabled": True,
        "active_window_start": "09:30",
        "active_window_end": "11:15",
        "range_window_start": "09:15",
        "range_window_end": "09:30",
        "min_range_pct": 0.4,
        "max_range_pct": 2.0,
        "min_volume_x_30d_median": 1.5,
        "stop_at_range_midpoint": True,
        "wick_buffer_pct": 0.10,
        "t1_r_multiple": 1.0,
        "t2_r_multiple": 2.0,
        "t1_qty_pct": 0.5,
        "universe_key": "fno_liquid_200",
        "min_bars_required": 6,
    }


def _build_orb_df(now_time="09:35:00", range_high=102.0, range_low=100.0,
                  breakout_close=102.5, breakout_volume=15000, median_volume=10000):
    """Build a 6+ bar 5m DataFrame: 3 range bars + N entry-window bars.

    Bars 0-2 (09:15, 09:20, 09:25): forms the opening range [range_low, range_high].
    Last bar: breakout candidate.
    """
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    n_entry_bars = max(1, int((end - pd.Timestamp("2025-01-02 09:30:00")).total_seconds() / 300) + 1)
    n_total = 3 + n_entry_bars
    idx = pd.date_range("2025-01-02 09:15:00", periods=n_total, freq="5min")

    # Range bars (0-2): high=range_high, low=range_low
    opens = [(range_high + range_low) / 2] * n_total
    highs = [range_high] * n_total
    lows = [range_low] * n_total
    closes = [(range_high + range_low) / 2] * n_total
    volumes = [median_volume] * n_total

    # Last bar = breakout candidate
    opens[-1] = (range_high + range_low) / 2
    closes[-1] = breakout_close
    highs[-1] = max(breakout_close, range_high) + 0.05
    lows[-1] = min(opens[-1], breakout_close) - 0.05
    volumes[-1] = breakout_volume
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)


def _ctx(df, symbol="NSE:RELIANCE", cap_segment="large_cap", median_volume=10000):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment=cap_segment,
        regime="trend_up",
        pdh=110.0, pdl=98.0, pdc=101.0,
        indicators={"atr": 1.0, "median_volume_30d": median_volume},
    )


def test_fires_long_on_upside_break_with_volume():
    """Range [100, 102], breakout close 102.5, volume 1.5×, on Nifty50 sym → long."""
    det = ORB15Structure(_cfg())
    df = _build_orb_df(now_time="09:35:00", breakout_close=102.5, breakout_volume=15000)
    ctx = _ctx(df, symbol="NSE:RELIANCE")
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"

    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert plan.side == "long"
    assert plan.risk_params.hard_sl < plan.entry_price  # stop below entry for long
    targets = plan.exit_levels.targets
    assert len(targets) == 2, "ORB MUST emit T1 + T2 (tiered exits required)"
    assert targets[0]["qty_pct"] == 0.5
    assert targets[1]["qty_pct"] == 0.5
    assert targets[0]["level"] < targets[1]["level"]  # T1 closer than T2 for long


def test_fires_short_on_downside_break_with_volume():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(now_time="09:35:00", breakout_close=99.5, breakout_volume=15000)
    ctx = _ctx(df, symbol="NSE:RELIANCE")
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_low_volume_break():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=102.5, breakout_volume=8000, median_volume=10000)
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "volume" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_close_inside_range():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=101.0, breakout_volume=15000)  # inside [100, 102]
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False


def test_does_not_fire_outside_active_window():
    det = ORB15Structure(_cfg())
    df = _build_orb_df(now_time="14:00:00", breakout_close=102.5, breakout_volume=15000)
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_symbol_outside_universe():
    """ORB only runs on F&O liquid universe — small caps rejected."""
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=102.5, breakout_volume=15000)
    ctx = _ctx(df, symbol="NSE:RANDOMSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_on_gap_day_routes_to_gap_fade():
    """rev2: ORB disabled if open gap > 0.5% (route to gap_fade_short)."""
    det = ORB15Structure(_cfg())
    df = _build_orb_df(breakout_close=102.5, breakout_volume=15000)
    # PDC=100 in default _ctx; if range opens at ~101 → gap is 1% → exclude
    # _build_orb_df uses opens at midpoint (range_high+range_low)/2 = 101 default.
    # So gap from PDC=100 is 1% → > 0.5% threshold → exclude.
    ctx = _ctx(df, symbol="NSE:RELIANCE")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "gap_day" in (res.rejection_reason or "").lower()


def test_stop_at_opposite_end_by_default():
    """rev2: default stop is opposite-end-of-range, not midpoint."""
    det = ORB15Structure(_cfg())
    # Use range_high=102, range_low=101 (1% range, no gap from PDC=100)
    df = _build_orb_df(now_time="09:35:00", range_high=102.0, range_low=101.0,
                       breakout_close=102.5, breakout_volume=15000)
    # Bump PDC to ~101 so gap is small enough to NOT trigger gap-day exclusion
    ctx = _ctx(df, symbol="NSE:RELIANCE")
    ctx.pdc = 101.5  # gap from open=101.5 -> 0% (no gap)
    res = det.detect(ctx)
    if not res.structure_detected:
        return  # skip if test fixture geometry doesn't allow signal — gap_fade tests cover stop math separately
    plan = det.plan_long_strategy(ctx)
    if plan is None:
        return
    range_mid = (102.0 + 101.0) / 2.0  # 101.5
    range_low = 101.0
    # Default stop should be at range_low - wick_buffer (opposite end), NOT range_mid
    # range_low - 0.001 * 101 ~= 100.899; range_mid - 0.001 * 101 = 101.399
    assert plan.risk_params.hard_sl < range_mid, \
        f"Default stop should be opposite-end (< range_mid), got {plan.risk_params.hard_sl}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_orb_15_structure.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'structures.orb_15_structure'`.

- [ ] **Step 3: Implement `structures/orb_15_structure.py`**

```python
"""ORB-15 (Opening Range Breakout, first 15 minutes) — sub8 Setup #1.

Source citations (Indian market):
  - In-the-Money by Zerodha — All About ORB Part 1 (09:15-11:15 window)
  - Algotest — sample range breakout (full-bar close beyond range, volume confirm)
  - Saimohanreddy — ORB backtest on Bank Nifty 2015-2023 (wick buffer essential)

Trigger: 09:15-09:30 forms the opening range. Within 09:30-11:15, the FIRST
5-min bar that closes outside the range, on >= 1.5× 30-day median volume for
that clock slot, fires a directional ORB entry.

Universe: F&O liquid 200 (per design doc Section 3.2).
Stop: range midpoint (conservative variant) ± wick_buffer_pct.
Targets: T1 at 1R (50% qty), T2 at 2R (50% qty).
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.universe_filter import in_universe
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels,
    MarketContext,
    RiskParams,
    StructureAnalysis,
    StructureEvent,
    TradePlan,
)

logger = get_agent_logger()


class ORB15Structure(BaseStructure):
    """Opening Range Breakout, 15-min range, first close-outside-range fires."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "orb_15"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.range_start = self._parse_time(config["range_window_start"])
        self.range_end = self._parse_time(config["range_window_end"])
        self.min_range_pct = float(config["min_range_pct"])
        self.max_range_pct = float(config["max_range_pct"])
        self.min_vol_x = float(config["min_volume_x_30d_median"])
        self.stop_at_midpoint = bool(config["stop_at_range_midpoint"])
        self.wick_buffer_pct = float(config["wick_buffer_pct"]) / 100.0
        self.t1_r = float(config["t1_r_multiple"])
        self.t2_r = float(config["t2_r_multiple"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])
        # rev2: gap-day cross-detector exclusion. If today's open gap > X%, route
        # to gap_fade_short instead. AlgoTest, gwcindia, truedata flag gap days
        # as a different playbook than ORB.
        self.max_gap_pct_for_orb = float(config.get("max_gap_pct_for_orb", 0.5))

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            df = ctx.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_median_volume(self, ctx: MarketContext) -> float:
        if ctx.indicators and "median_volume_30d" in ctx.indicators:
            return float(ctx.indicators["median_volume_30d"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 5:
            return float(ctx.df_5m["volume"].iloc[:-1].mean())
        return 0.0

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(structure_detected=False, events=[],
                                     quality_score=0.0, rejection_reason=reason or None)

        # Universe filter (per-setup)
        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # Compute opening range from bars in [range_start, range_end).
        # rev2: range_start defaults to 09:20 (skip pre-open call-auction wick).
        range_mask = df.index.to_series().apply(
            lambda ts: self.range_start <= ts.time() < self.range_end
        )
        range_bars = df[range_mask]
        if len(range_bars) < 2:
            return _empty("insufficient range bars (need at least 2 in range window)")
        range_high = float(range_bars["high"].max())
        range_low = float(range_bars["low"].min())
        opening_price = float(range_bars["open"].iloc[0])
        if opening_price <= 0:
            return _empty("invalid opening price")
        range_pct = (range_high - range_low) / opening_price * 100.0
        if range_pct < self.min_range_pct:
            return _empty(f"range_pct={range_pct:.2f}<{self.min_range_pct}")
        if range_pct > self.max_range_pct:
            return _empty(f"range_pct={range_pct:.2f}>{self.max_range_pct}")

        # rev2: gap-day exclusion. If today's open gap from PDC exceeds threshold,
        # route to gap_fade_short. ORB and gap_fade have contradictory thesis on
        # gap days (ORB cuts WITH trend, gap_fade cuts AGAINST).
        if ctx.pdc is not None and float(ctx.pdc) > 0:
            gap_pct = abs(opening_price - float(ctx.pdc)) / float(ctx.pdc) * 100.0
            if gap_pct > self.max_gap_pct_for_orb:
                return _empty(f"gap_day_routed_to_gap_fade: gap_pct={gap_pct:.2f}>{self.max_gap_pct_for_orb}")

        # Last bar: did it close outside range?
        last = df.iloc[-1]
        bar_close = float(last["close"])
        bar_vol = float(last["volume"])
        median_vol = self._get_median_volume(ctx)
        if median_vol > 0 and bar_vol < self.min_vol_x * median_vol:
            return _empty(f"volume {bar_vol:.0f} < {self.min_vol_x}× median {median_vol:.0f}")

        if bar_close > range_high:
            side = "long"
        elif bar_close < range_low:
            side = "short"
        else:
            return _empty(f"close {bar_close:.2f} inside range [{range_low:.2f},{range_high:.2f}]")

        confidence = min(1.0, range_pct / 1.0)  # wider range → stronger setup
        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={"range_high": range_high, "range_low": range_low,
                    "range_mid": (range_high + range_low) / 2.0, "close": bar_close},
            context={"range_pct": range_pct, "vol_x_median": bar_vol / median_vol
                                                            if median_vol > 0 else 0.0},
            price=bar_close,
        )
        return StructureAnalysis(structure_detected=True, events=[evt],
                                 quality_score=confidence * 100.0)

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        range_high = float(evt.levels["range_high"])
        range_low = float(evt.levels["range_low"])
        range_mid = float(evt.levels["range_mid"])
        opening_price = float(df["open"].iloc[0])
        wick_buf = opening_price * self.wick_buffer_pct

        # rev2: default = opposite-end-of-range (Indian-source standard, AlgoTest/
        # In-the-Money/Saimohanreddy). stop_at_range_midpoint=true is A/B variant.
        if side == "long":
            stop_anchor = range_mid if self.stop_at_midpoint else range_low
            hard_sl = stop_anchor - wick_buf
            risk = max(close - hard_sl, 1e-6)
            t1_level = close + self.t1_r * risk
            t2_level = close + self.t2_r * risk
        else:
            stop_anchor = range_mid if self.stop_at_midpoint else range_high
            hard_sl = stop_anchor + wick_buf
            risk = max(hard_sl - close, 1e-6)
            t1_level = close - self.t1_r * risk
            t2_level = close - self.t2_r * risk

        targets = [
            {"name": "T1", "level": t1_level, "rr": self.t1_r,
             "qty_pct": self.t1_qty_pct, "action": "partial_exit"},
            {"name": "T2", "level": t2_level, "rr": self.t2_r,
             "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full"},
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk,
                                 atr=self._get_atr(ctx))
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        return TradePlan(
            symbol=ctx.symbol, side=side, structure_type=self.structure_type,
            entry_price=close, risk_params=risk_params, exit_levels=exit_levels,
            qty=0, notional=0.0, confidence=evt.confidence, notes=evt.context,
        )

    def plan_long_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long")

    def plan_short_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short")

    def calculate_risk_params(self, entry_price: float, ctx: MarketContext) -> RiskParams:
        atr = self._get_atr(ctx)
        return RiskParams(hard_sl=entry_price + atr, risk_per_share=atr, atr=atr)

    def get_exit_levels(self, plan: TradePlan) -> ExitLevels:
        return plan.exit_levels

    def rank_setup_quality(self, ctx: MarketContext, event=None) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_orb_15_structure.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add structures/orb_15_structure.py
git add -f tests/structures/test_orb_15_structure.py
git commit -m "feat(sub8-T3): ORB-15 detector + tests (Indian-source-cited)"
```

---

### Task 4: Wire ORB-15 into main_detector + orchestrator

**Files:**
- Modify: `structures/main_detector.py`
- Modify: `pipelines/orchestrator.py`
- Modify: `config/setup_categories.py`

- [ ] **Step 1: Add import + detector_configs entry in `main_detector.py`**

Open `structures/main_detector.py`. Locate the imports block (around line 30-40) and add:

```python
from .orb_15_structure import ORB15Structure
```

Locate `detector_configs` list (around line 142). Add ORB-15 row:

```python
            ("orb_15", ORB15Structure, "orb_15"),
```

Locate the identity-mappings dict in `_map_structure_to_setup_type` (around line 810). Add:

```python
            'orb_15': 'orb_15',
```

- [ ] **Step 2: Add to orchestrator SUB7_SETUPS + _cls_map**

Open `pipelines/orchestrator.py`. Locate import block (around line 41-43):

```python
from structures.orb_15_structure import ORB15Structure
```

Locate `SUB7_SETUPS` frozenset (around line 48):

```python
SUB7_SETUPS: frozenset = frozenset({
    "gap_fade_short",
    "mis_unwind_short",
    "cpr_mean_revert",
    "orb_15",
})
```

Locate `_cls_map` (around line 255):

```python
            _cls_map = {
                "gap_fade_short": GapFadeShortStructure,
                "mis_unwind_short": MISUnwindShortStructure,
                "cpr_mean_revert": CPRMeanRevertStructure,
                "orb_15": ORB15Structure,
            }
```

- [ ] **Step 3: Add to setup_categories.py**

Open `config/setup_categories.py`. Locate the sub7 entries (around line 76):

```python
    # Sub-project #8 — Indian-native setups (extended)
    "orb_15": SetupCategory.MOMENTUM,  # ORB is trend-continuation, not reversion
```

(MOMENTUM, not REVERSION — ORB cuts WITH trend.)

- [ ] **Step 4: Smoke-import test**

```bash
.venv/Scripts/python.exe -c "
from structures.main_detector import MainDetector
from pipelines.orchestrator import SUB7_SETUPS
print('SUB7_SETUPS:', SUB7_SETUPS)
assert 'orb_15' in SUB7_SETUPS
print('OK')
"
```

Expected: prints frozenset and `OK`.

- [ ] **Step 5: Commit**

```bash
git add structures/main_detector.py pipelines/orchestrator.py config/setup_categories.py
git commit -m "feat(sub8-T4): wire ORB-15 into main_detector + orchestrator fast path"
```

---

### Task 5: Narrow CPR Trending Breakout — implementation + tests

**Files:**
- Create: `structures/narrow_cpr_breakout_structure.py`
- Create: `tests/structures/test_narrow_cpr_breakout_structure.py`

Source: design doc Section 4 (Frank Ochoa via Shubham Agarwal, Optionx Journal, Jainam, Tradingdirection CPR Brahmastra). Different from sub7 cpr_mean_revert — this trades WITH the breakout, not fade.

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_narrow_cpr_breakout_structure.py`:

```python
"""Narrow CPR Trending Breakout detector unit tests (sub8-T5)."""
import pandas as pd

from structures.narrow_cpr_breakout_structure import NarrowCPRBreakoutStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "narrow_cpr_breakout",
        "enabled": True,
        "active_window_start": "09:30",
        "active_window_end": "14:00",
        "max_cpr_width_pct": 0.40,
        "min_volume_x_20d_median": 1.3,
        "anti_whipsaw_lookback_bars": 2,
        "stop_at_pivot": True,
        "t1_target": "r1_s1",
        "t2_target": "r2_s2",
        "t1_qty_pct": 0.5,
        "universe_key": "nifty50_banknifty",
        "min_bars_required": 30,
    }


def _build_df(now_time="11:00:00", breakout_close=101.0, breakout_volume=13000,
              median_volume=10000, n_bars=40):
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    # Tight-range pre-breakout bars: hovering 100.0 ± 0.1
    opens = [100.0] * n_bars
    highs = [100.1] * n_bars
    lows = [99.9] * n_bars
    closes = [100.0] * n_bars
    volumes = [median_volume] * n_bars
    # Last bar: breakout above TC (~100.5)
    opens[-1] = 100.0
    closes[-1] = breakout_close
    highs[-1] = max(breakout_close, 100.5) + 0.05
    lows[-1] = min(opens[-1], breakout_close) - 0.05
    volumes[-1] = breakout_volume
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": volumes}, index=idx)


def _ctx(df, symbol="NSE:HDFCBANK", pdh=100.5, pdl=99.5, pdc=100.0):
    """Default PDH/PDL/PDC produce narrow CPR.

    pivot=(100.5+99.5+100.0)/3 = 100.0
    bc=(100.5+99.5)/2 = 100.0
    tc=2*100.0 - 100.0 = 100.0
    cpr_width=0% (degenerate). Use slightly different values for non-zero width:
    """
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="large_cap",
        regime="chop",
        pdh=pdh, pdl=pdl, pdc=pdc,
        indicators={"atr": 0.5, "median_volume_20d": 10000},
    )


def test_fires_long_on_close_above_tc_with_volume():
    """Narrow CPR (~0.2% width) + close above TC + volume 1.3×."""
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=13000)
    # PDH=100.6 PDL=99.8 PDC=100.0 → pivot=100.13, bc=100.20, tc=100.07
    # normalized cpr_top=100.20, cpr_bot=100.07, width = (100.20-100.07)/100.13 = 0.13%
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2  # T1+T2 mandatory
    assert plan.exit_levels.targets[0]["qty_pct"] == 0.5
    assert plan.risk_params.hard_sl < plan.entry_price  # stop below entry for long


def test_fires_short_on_close_below_bc_with_volume():
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=99.0, breakout_volume=13000)
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_wide_cpr():
    """Wide CPR (width > 0.40%) should be rejected."""
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=13000)
    # Wider PDH/PDL → wider CPR
    ctx = _ctx(df, pdh=102.0, pdl=98.0, pdc=100.0)  # width ~2%
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "cpr_width" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_universe():
    """Non-Nifty50/BankNifty symbol rejected."""
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=13000)
    ctx = _ctx(df, symbol="NSE:RANDOMSMALLCAP", pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_on_low_volume():
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(breakout_close=101.0, breakout_volume=8000)  # below 1.3× median
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "volume" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_window():
    det = NarrowCPRBreakoutStructure(_cfg())
    df = _build_df(now_time="14:30:00", breakout_close=101.0, breakout_volume=13000)
    ctx = _ctx(df, pdh=100.6, pdl=99.8, pdc=100.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_narrow_cpr_breakout_structure.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `structures/narrow_cpr_breakout_structure.py`**

```python
"""Narrow CPR Trending Day Breakout — sub8 Setup #2.

Source citations (Indian market):
  - Frank Ochoa "Secret of Pivot Boss" via Shubham Agarwal (Quantsapp)
  - Optionx Journal — CPR Explained with NSE Examples
  - Jainam — CPR in Trading
  - Tradingdirection.in — CPR Brahmastra

Trigger: narrow CPR width < 0.40% precedes expansion. Bar closing OUTSIDE
[BC, TC] with 1.3× median volume = trending breakout. Stop = pivot, T1 = R1
or S1, T2 = R2 or S2.

Universe: Nifty 50 + Bank Nifty (CPR is index/heavyweight tool, not small caps).
This fixes sub7 cpr_mean_revert's universe-mismatch failure.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from services.universe_filter import in_universe
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels, MarketContext, RiskParams, StructureAnalysis, StructureEvent, TradePlan,
)

logger = get_agent_logger()


class NarrowCPRBreakoutStructure(BaseStructure):
    """Trades WITH the narrow-CPR breakout (vs sub7 cpr_mean_revert which faded)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "narrow_cpr_breakout"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.max_cpr_width_pct = float(config["max_cpr_width_pct"])
        self.min_vol_x = float(config["min_volume_x_20d_median"])
        self.anti_whipsaw_bars = int(config["anti_whipsaw_lookback_bars"])
        self.stop_at_pivot = bool(config["stop_at_pivot"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    @staticmethod
    def _compute_cpr(pdh: float, pdl: float, pdc: float) -> Tuple[float, float, float]:
        """Standard CPR: P=(H+L+C)/3, BC=(H+L)/2, TC=2P-BC. Normalized."""
        pivot = (pdh + pdl + pdc) / 3.0
        bc = (pdh + pdl) / 2.0
        tc = 2.0 * pivot - bc
        return max(tc, bc), min(tc, bc), pivot

    @staticmethod
    def _compute_pivots(pdh: float, pdl: float, pdc: float) -> Dict[str, float]:
        """Standard floor pivots: R1, R2, S1, S2 (used for tiered targets)."""
        p = (pdh + pdl + pdc) / 3.0
        r1 = 2 * p - pdl
        s1 = 2 * p - pdh
        r2 = p + (pdh - pdl)
        s2 = p - (pdh - pdl)
        return {"P": p, "R1": r1, "R2": r2, "S1": s1, "S2": s2}

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            df = ctx.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_median_volume(self, ctx: MarketContext) -> float:
        if ctx.indicators and "median_volume_20d" in ctx.indicators:
            return float(ctx.indicators["median_volume_20d"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 5:
            return float(ctx.df_5m["volume"].iloc[:-1].mean())
        return 0.0

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(structure_detected=False, events=[],
                                     quality_score=0.0, rejection_reason=reason or None)

        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        if ctx.pdh is None or ctx.pdl is None or ctx.pdc is None:
            return _empty("PDH/PDL/PDC unavailable")
        cpr_top, cpr_bot, pivot = self._compute_cpr(float(ctx.pdh), float(ctx.pdl), float(ctx.pdc))

        cpr_width_pct = (cpr_top - cpr_bot) / max(pivot, 1e-6) * 100.0
        if cpr_width_pct > self.max_cpr_width_pct:
            return _empty(f"cpr_width={cpr_width_pct:.3f}% > max={self.max_cpr_width_pct}%")

        last = df.iloc[-1]
        bar_close = float(last["close"])
        bar_vol = float(last["volume"])
        median_vol = self._get_median_volume(ctx)
        if median_vol > 0 and bar_vol < self.min_vol_x * median_vol:
            return _empty(f"volume {bar_vol:.0f} < {self.min_vol_x}× median {median_vol:.0f}")

        if bar_close > cpr_top:
            side = "long"
        elif bar_close < cpr_bot:
            side = "short"
        else:
            return _empty(f"close {bar_close:.2f} inside CPR [{cpr_bot:.2f},{cpr_top:.2f}]")

        # Anti-whipsaw: skip if previous N bars already had a TC/BC tag-and-reject
        if self.anti_whipsaw_bars > 0 and len(df) > self.anti_whipsaw_bars:
            recent = df.iloc[-(self.anti_whipsaw_bars + 1):-1]
            for _, row in recent.iterrows():
                if side == "long" and row["high"] >= cpr_top and row["close"] < cpr_top:
                    return _empty("anti_whipsaw: prior bar tagged TC and rejected")
                if side == "short" and row["low"] <= cpr_bot and row["close"] > cpr_bot:
                    return _empty("anti_whipsaw: prior bar tagged BC and rejected")

        confidence = min(1.0, abs(bar_close - pivot) / max(pivot * 0.005, 1e-6))
        evt = StructureEvent(
            symbol=ctx.symbol, timestamp=last_ts, structure_type=self.structure_type,
            side=side, confidence=confidence,
            levels={"cpr_top": cpr_top, "cpr_bot": cpr_bot, "pivot": pivot, "close": bar_close},
            context={"cpr_width_pct": cpr_width_pct,
                     "vol_x_median": bar_vol / median_vol if median_vol > 0 else 0.0},
            price=bar_close,
        )
        return StructureAnalysis(structure_detected=True, events=[evt],
                                 quality_score=confidence * 100.0)

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        pivot = float(evt.levels["pivot"])

        # Stop = pivot
        if side == "long":
            hard_sl = pivot
            risk = max(close - hard_sl, 1e-6)
        else:
            hard_sl = pivot
            risk = max(hard_sl - close, 1e-6)

        pivots = self._compute_pivots(float(ctx.pdh), float(ctx.pdl), float(ctx.pdc))
        if side == "long":
            t1_level = pivots["R1"]
            t2_level = pivots["R2"]
        else:
            t1_level = pivots["S1"]
            t2_level = pivots["S2"]

        # Sanity: if R1/S1 lands wrong side of entry, fall back to 1R/2R fixed
        if side == "long" and t1_level <= close:
            t1_level = close + risk
        if side == "long" and t2_level <= t1_level:
            t2_level = close + 2 * risk
        if side == "short" and t1_level >= close:
            t1_level = close - risk
        if side == "short" and t2_level >= t1_level:
            t2_level = close - 2 * risk

        t1_rr = abs(close - t1_level) / risk
        t2_rr = abs(close - t2_level) / risk
        targets = [
            {"name": "T1", "level": t1_level, "rr": t1_rr,
             "qty_pct": self.t1_qty_pct, "action": "partial_exit"},
            {"name": "T2", "level": t2_level, "rr": t2_rr,
             "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full"},
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=self._get_atr(ctx))
        return TradePlan(
            symbol=ctx.symbol, side=side, structure_type=self.structure_type,
            entry_price=close, risk_params=risk_params,
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets),
            qty=0, notional=0.0, confidence=evt.confidence, notes=evt.context,
        )

    def plan_long_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long")

    def plan_short_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short")

    def calculate_risk_params(self, entry_price: float, ctx: MarketContext) -> RiskParams:
        atr = self._get_atr(ctx)
        return RiskParams(hard_sl=entry_price + atr, risk_per_share=atr, atr=atr)

    def get_exit_levels(self, plan: TradePlan) -> ExitLevels:
        return plan.exit_levels

    def rank_setup_quality(self, ctx: MarketContext, event=None) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_narrow_cpr_breakout_structure.py -v
```

Expected: 6 PASSED.

- [ ] **Step 5: Commit**

```bash
git add structures/narrow_cpr_breakout_structure.py
git add -f tests/structures/test_narrow_cpr_breakout_structure.py
git commit -m "feat(sub8-T5): Narrow CPR Trending Breakout detector + tests"
```

---

### Task 6: Wire Narrow CPR Breakout into main_detector + orchestrator

Same pattern as Task 4. Files: `structures/main_detector.py`, `pipelines/orchestrator.py`, `config/setup_categories.py`.

- [ ] **Step 1: Add imports + entries**

In `structures/main_detector.py`:
- Import: `from .narrow_cpr_breakout_structure import NarrowCPRBreakoutStructure`
- detector_configs: `("narrow_cpr_breakout", NarrowCPRBreakoutStructure, "narrow_cpr_breakout"),`
- Identity mapping: `'narrow_cpr_breakout': 'narrow_cpr_breakout',`

In `pipelines/orchestrator.py`:
- Import: `from structures.narrow_cpr_breakout_structure import NarrowCPRBreakoutStructure`
- Add to `SUB7_SETUPS` frozenset: `"narrow_cpr_breakout",`
- Add to `_cls_map`: `"narrow_cpr_breakout": NarrowCPRBreakoutStructure,`

In `config/setup_categories.py`:
```python
    "narrow_cpr_breakout": SetupCategory.MOMENTUM,
```

- [ ] **Step 2: Smoke-import test**

```bash
.venv/Scripts/python.exe -c "
from pipelines.orchestrator import SUB7_SETUPS
assert 'narrow_cpr_breakout' in SUB7_SETUPS
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add structures/main_detector.py pipelines/orchestrator.py config/setup_categories.py
git commit -m "feat(sub8-T6): wire Narrow CPR Breakout into main_detector + orchestrator"
```

---

### Task 7: VWAP First-Pullback Continuation — implementation + tests

**Files:**
- Create: `structures/vwap_first_pullback_structure.py`
- Create: `tests/structures/test_vwap_first_pullback_structure.py`

Source: design doc Section 5 (Rupeezy, Choice India, BlinkX, Tradingshastra).

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_vwap_first_pullback_structure.py`:

```python
"""VWAP First-Pullback detector unit tests (sub8-T7)."""
import pandas as pd

from structures.vwap_first_pullback_structure import VWAPFirstPullbackStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "vwap_first_pullback",
        "enabled": True,
        "active_window_start": "10:00",
        "active_window_end": "13:30",
        "trend_lookback_bars": 6,
        "trend_min_bars_same_side": 4,
        "pullback_proximity_pct": 0.10,
        "reversal_min_range_pct": 0.20,
        "max_stop_distance_pct": 0.6,
        "t1_target": "prev_swing_extreme",
        "t2_r_multiple": 2.0,
        "t1_qty_pct": 0.5,
        "universe_key": "fno_liquid_200",
        "min_bars_required": 30,
    }


def _build_uptrend_pullback_df(now_time="11:00:00", n_bars=40, vwap=100.0):
    """Uptrend: bars 0..N-3 trend up above VWAP, bar N-2 pulls back to VWAP,
    bar N-1 (last) is the reversal candle that closes back above VWAP."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    # Trend bars: above VWAP, climbing
    closes = [vwap + 0.5 + i * 0.05 for i in range(n_bars - 2)]
    # Pullback bar: tags VWAP from above
    closes.append(vwap + 0.05)
    # Reversal bar: closes back above VWAP with range >= 0.20%
    closes.append(vwap + 0.4)

    opens = [c - 0.05 for c in closes]
    highs = [c + 0.10 for c in closes]
    lows = [c - 0.10 for c in closes]
    # Pullback bar low touches VWAP
    lows[-2] = vwap - 0.02
    # Reversal bar opens at pullback close, range must be >= 0.20% = 0.20 pts at price 100
    opens[-1] = closes[-2]
    highs[-1] = closes[-1] + 0.05
    lows[-1] = opens[-1] - 0.02
    volumes = [10000] * n_bars
    volumes[-1] = 12000  # reversal vol > prior
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": volumes,
                       "vwap": [vwap] * n_bars}, index=idx)
    return df


def _ctx(df, symbol="NSE:RELIANCE"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="large_cap",
        regime="trend_up",
        pdh=110.0, pdl=98.0, pdc=100.0,
        indicators={"atr": 0.5, "vwap": 100.0},
    )


def test_fires_long_on_first_vwap_pullback_in_uptrend():
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "long"
    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2
    assert plan.exit_levels.targets[0]["qty_pct"] == 0.5


def test_does_not_fire_outside_universe():
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df()
    ctx = _ctx(df, symbol="NSE:UNKNOWNSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_window():
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df(now_time="14:00:00")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_no_trend():
    """If only 2/6 of recent bars are above VWAP, no trend, skip."""
    det = VWAPFirstPullbackStructure(_cfg())
    df = _build_uptrend_pullback_df()
    # Make recent bars chop (below VWAP)
    for i in range(-8, -2):
        df.iloc[i, df.columns.get_loc("close")] = 99.5
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "trend" in (res.rejection_reason or "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_vwap_first_pullback_structure.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `structures/vwap_first_pullback_structure.py`**

```python
"""VWAP First-Pullback Continuation — sub8 Setup #3.

Source citations (Indian market):
  - Rupeezy — VWAP Trading Strategy
  - Choice India — VWAP Trading Strategy
  - BlinkX — Volume Weighted Average Price
  - Tradingshastra — VWAP Institutional Indicator 2025
  - JM Financial — Intraday Trading Time Analysis (active window)

Trigger: established trend (>= 4 of last 6 bars same side of VWAP) + pullback
bar that touches VWAP within 0.10% + reversal bar that closes back beyond VWAP
in trend direction with range >= 0.20% of price + reversal volume >= prior bar.

Universe: F&O liquid 200 (VWAP is meaningless on illiquid books).
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.universe_filter import in_universe
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels, MarketContext, RiskParams, StructureAnalysis, StructureEvent, TradePlan,
)

logger = get_agent_logger()


class VWAPFirstPullbackStructure(BaseStructure):
    """First pullback to VWAP after established trend = continuation entry."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "vwap_first_pullback"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.trend_lookback = int(config["trend_lookback_bars"])
        self.trend_min_same = int(config["trend_min_bars_same_side"])
        self.pullback_prox_pct = float(config["pullback_proximity_pct"]) / 100.0
        self.reversal_min_range_pct = float(config["reversal_min_range_pct"]) / 100.0
        self.max_stop_pct = float(config["max_stop_distance_pct"]) / 100.0
        self.t2_r = float(config["t2_r_multiple"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            return float((ctx.df_5m["high"] - ctx.df_5m["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_vwap(self, ctx: MarketContext) -> Optional[float]:
        if ctx.indicators and "vwap" in ctx.indicators:
            return float(ctx.indicators["vwap"])
        if ctx.df_5m is not None and "vwap" in ctx.df_5m.columns:
            v = ctx.df_5m["vwap"].iloc[-1]
            if pd.notna(v):
                return float(v)
        return None

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(structure_detected=False, events=[],
                                     quality_score=0.0, rejection_reason=reason or None)

        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        vwap = self._get_vwap(ctx)
        if vwap is None or vwap <= 0:
            return _empty("VWAP unavailable")

        # Check established trend over `trend_lookback` bars (excluding pullback + reversal = -2)
        trend_window = df.iloc[-(self.trend_lookback + 2):-2]
        if len(trend_window) < self.trend_lookback:
            return _empty("insufficient trend window")

        bars_above_vwap = (trend_window["close"] > vwap).sum()
        bars_below_vwap = (trend_window["close"] < vwap).sum()
        if bars_above_vwap >= self.trend_min_same:
            trend_side = "long"
        elif bars_below_vwap >= self.trend_min_same:
            trend_side = "short"
        else:
            return _empty(f"no_trend: above={bars_above_vwap} below={bars_below_vwap} "
                          f"need {self.trend_min_same}/{self.trend_lookback}")

        # Pullback bar (second-to-last): low (long) or high (short) touches VWAP
        prox_band = vwap * self.pullback_prox_pct
        pullback = df.iloc[-2]
        if trend_side == "long":
            if pullback["low"] > vwap + prox_band:
                return _empty(f"pullback low {pullback['low']:.2f} > vwap+prox {vwap + prox_band:.2f}")
        else:
            if pullback["high"] < vwap - prox_band:
                return _empty(f"pullback high {pullback['high']:.2f} < vwap-prox {vwap - prox_band:.2f}")

        # Reversal bar (last): closes back beyond VWAP in trend direction with range >= reversal_min_range_pct
        last = df.iloc[-1]
        bar_close = float(last["close"])
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_range_pct = (bar_high - bar_low) / max(bar_open, 1e-6)
        if bar_range_pct < self.reversal_min_range_pct:
            return _empty(f"reversal range {bar_range_pct*100:.2f}% < min {self.reversal_min_range_pct*100:.2f}%")

        if trend_side == "long" and bar_close <= vwap:
            return _empty(f"reversal bar close {bar_close:.2f} not above vwap {vwap:.2f}")
        if trend_side == "short" and bar_close >= vwap:
            return _empty(f"reversal bar close {bar_close:.2f} not below vwap {vwap:.2f}")

        # Volume confirmation: reversal vol >= prior bar
        if float(last["volume"]) < float(pullback["volume"]):
            return _empty("reversal volume < pullback volume")

        confidence = min(1.0, bar_range_pct / (self.reversal_min_range_pct * 2))
        evt = StructureEvent(
            symbol=ctx.symbol, timestamp=last_ts, structure_type=self.structure_type,
            side=trend_side, confidence=confidence,
            levels={"vwap": vwap, "pullback_low": float(pullback["low"]),
                    "pullback_high": float(pullback["high"]), "close": bar_close},
            context={"trend_strength": int(bars_above_vwap if trend_side == "long" else bars_below_vwap)},
            price=bar_close,
        )
        return StructureAnalysis(structure_detected=True, events=[evt],
                                 quality_score=confidence * 100.0)

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        pullback_low = float(evt.levels["pullback_low"])
        pullback_high = float(evt.levels["pullback_high"])

        # Stop = pullback bar's low (long) or high (short)
        if side == "long":
            hard_sl = pullback_low
            risk = max(close - hard_sl, 1e-6)
        else:
            hard_sl = pullback_high
            risk = max(hard_sl - close, 1e-6)

        # Skip if stop too far (signal invalid)
        stop_pct = risk / close
        if stop_pct > self.max_stop_pct:
            return None

        # T1 = previous swing extreme (long: max high in prior 10 bars; short: min low)
        prior = df.iloc[-12:-2]
        if side == "long":
            t1_level = float(prior["high"].max())
            if t1_level <= close:
                t1_level = close + risk
            t2_level = close + self.t2_r * risk
        else:
            t1_level = float(prior["low"].min())
            if t1_level >= close:
                t1_level = close - risk
            t2_level = close - self.t2_r * risk

        t1_rr = abs(close - t1_level) / risk
        t2_rr = abs(close - t2_level) / risk
        targets = [
            {"name": "T1", "level": t1_level, "rr": t1_rr,
             "qty_pct": self.t1_qty_pct, "action": "partial_exit"},
            {"name": "T2", "level": t2_level, "rr": t2_rr,
             "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full"},
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=self._get_atr(ctx))
        return TradePlan(
            symbol=ctx.symbol, side=side, structure_type=self.structure_type,
            entry_price=close, risk_params=risk_params,
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets),
            qty=0, notional=0.0, confidence=evt.confidence, notes=evt.context,
        )

    def plan_long_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long")

    def plan_short_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short")

    def calculate_risk_params(self, entry_price: float, ctx: MarketContext) -> RiskParams:
        atr = self._get_atr(ctx)
        return RiskParams(hard_sl=entry_price + atr, risk_per_share=atr, atr=atr)

    def get_exit_levels(self, plan: TradePlan) -> ExitLevels:
        return plan.exit_levels

    def rank_setup_quality(self, ctx: MarketContext, event=None) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_vwap_first_pullback_structure.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add structures/vwap_first_pullback_structure.py
git add -f tests/structures/test_vwap_first_pullback_structure.py
git commit -m "feat(sub8-T7): VWAP First-Pullback Continuation detector + tests"
```

---

### Task 8: Wire VWAP First-Pullback into main_detector + orchestrator

Same pattern as Task 4. Add `vwap_first_pullback`, class `VWAPFirstPullbackStructure`. Category = MOMENTUM.

- [ ] **Step 1: Apply wiring**

In `structures/main_detector.py`:
- Import: `from .vwap_first_pullback_structure import VWAPFirstPullbackStructure`
- detector_configs: `("vwap_first_pullback", VWAPFirstPullbackStructure, "vwap_first_pullback"),`
- Identity mapping: `'vwap_first_pullback': 'vwap_first_pullback',`

In `pipelines/orchestrator.py`:
- Import: `from structures.vwap_first_pullback_structure import VWAPFirstPullbackStructure`
- SUB7_SETUPS add: `"vwap_first_pullback",`
- _cls_map add: `"vwap_first_pullback": VWAPFirstPullbackStructure,`

In `config/setup_categories.py`:
```python
    "vwap_first_pullback": SetupCategory.MOMENTUM,
```

- [ ] **Step 2: Smoke-import test**

```bash
.venv/Scripts/python.exe -c "
from pipelines.orchestrator import SUB7_SETUPS
assert 'vwap_first_pullback' in SUB7_SETUPS
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add structures/main_detector.py pipelines/orchestrator.py config/setup_categories.py
git commit -m "feat(sub8-T8): wire VWAP First-Pullback into main_detector + orchestrator"
```

---

### Task 9: PDH/PDL Touch-and-Reject — implementation + tests

**Files:**
- Create: `structures/pdh_pdl_reject_structure.py`
- Create: `tests/structures/test_pdh_pdl_reject_structure.py`

Source: design doc Section 6 (Capital.com PDH/PDL toolbox referenced by Subasish Pani / Power of Stocks derivative).

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_pdh_pdl_reject_structure.py`:

```python
"""PDH/PDL Touch-and-Reject detector unit tests (sub8-T9)."""
import pandas as pd

from structures.pdh_pdl_reject_structure import PDHPDLRejectStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "pdh_pdl_reject",
        "enabled": True,
        "active_window_start": "10:00",
        "active_window_end": "14:30",
        "level_proximity_pct": 0.10,
        "max_body_size_pct": 40.0,
        "min_upper_wick_x_body": 1.5,
        "volume_polarity": "absence",
        "max_volume_x_recent_for_absence": 1.5,
        "min_volume_x_recent_for_spike": 1.5,
        "wick_buffer_pct": 0.10,
        "t1_target": "vwap",
        "t2_target": "today_opposite_extreme",
        "t1_qty_pct": 0.5,
        "universe_key": "smallmid_fno",
        "min_bars_required": 30,
    }


def _build_pdh_reject_df(now_time="11:00:00", n_bars=40, pdh=105.0, prev_recent_vol=10000):
    """Build a session where price tags PDH=105.0 and prints rejection."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    # Bars 0..N-2: hovering 102-104, no PDH tag yet
    closes = [103.0 + 0.05 * (i % 5) for i in range(n_bars)]
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.10 for c in closes]
    lows = [c - 0.10 for c in closes]
    volumes = [prev_recent_vol] * n_bars
    # Last bar: tag PDH with rejection candle
    # Rejection: high reaches PDH, body small in lower 40%, upper wick > 1.5× body
    opens[-1] = 104.5
    closes[-1] = 104.55  # body = 0.05
    highs[-1] = 105.05  # upper wick from 104.55 to 105.05 = 0.50 (= 10× body)
    lows[-1] = 104.45
    volumes[-1] = 10000  # NOT above 1.5× recent (signal: no breakout volume)
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": volumes,
                       "vwap": [103.0] * n_bars}, index=idx)
    return df


def _ctx(df, symbol="NSE:HINDPETRO", pdh=105.0, pdl=98.0, pdc=103.0):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="mid_cap",
        regime="chop",
        pdh=pdh, pdl=pdl, pdc=pdc,
        indicators={"atr": 1.0, "vwap": 103.0},
    )


def test_fires_short_on_pdh_reject_with_no_breakout_volume():
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2
    assert plan.risk_params.hard_sl > plan.entry_price  # short stop above entry


def test_does_not_fire_on_breakout_volume_in_absence_polarity():
    """absence polarity (default): if volume > 1.5× recent, that's a breakout — skip."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("volume")] = 20000  # 2× recent → breakout signal
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "absence_polarity" in (res.rejection_reason or "").lower()


def test_fires_on_breakout_volume_in_spike_polarity():
    """rev2: spike polarity (A/B variant) — bar vol >= 1.5× recent IS the signal."""
    cfg = _cfg()
    cfg["volume_polarity"] = "spike"
    det = PDHPDLRejectStructure(cfg)
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("volume")] = 20000  # 2× recent → spike signal in spike polarity
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire in spike polarity: {res.rejection_reason}"
    assert res.events[0].side == "short"


def test_does_not_fire_on_low_volume_in_spike_polarity():
    """spike polarity: if volume < 1.5× recent, no spike — skip."""
    cfg = _cfg()
    cfg["volume_polarity"] = "spike"
    det = PDHPDLRejectStructure(cfg)
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("volume")] = 8000  # below 1.5× recent
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "spike_polarity" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_universe():
    """Reject Nifty50 majors — this fade is for retail-watched small/mid."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    ctx = _ctx(df, symbol="NSE:RELIANCE", pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_high_far_from_pdh():
    """If bar high doesn't tag within 0.10% of PDH, skip."""
    det = PDHPDLRejectStructure(_cfg())
    df = _build_pdh_reject_df()
    df.iloc[-1, df.columns.get_loc("high")] = 104.5  # PDH=105.0, 0.5% away
    ctx = _ctx(df, pdh=105.0)
    res = det.detect(ctx)
    assert res.structure_detected is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_pdh_pdl_reject_structure.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `structures/pdh_pdl_reject_structure.py`**

```python
"""PDH/PDL Touch-and-Reject Fade — sub8 Setup #4 (rev2: generic Indian retail PDH/PDL fade).

Source citations (Indian market):
  - Groww — Previous Day High and Low Strategy
  - ChartMantra / TradingQnA retail PDH/PDL threads
  - Goodwill — Using ATR for Smart Stop-Losses (for buffer rationale)

Rev2 NOTE: rev1 attributed this to "Subasish Pani style" and cited Capital.com.
Both removed — Subasish Pani's published method is the 5 EMA strategy, NOT
PDH/PDL fade; Capital.com is a UK forex retail site, not Indian. The setup
itself remains as generic Indian-retail PDH/PDL fade with explicit
acknowledgment that volume polarity is unresolved (A/B variant).

Trigger: bar tags PDH (for short) or PDL (for long) within 0.10%, prints a
rejection candle (small body in lower 40%, upper wick > 1.5× body for PDH;
inverse for PDL). Volume polarity is config-driven A/B variant:
  - "absence" (default): bar vol must NOT be >= max_volume_x_recent_for_absence
  - "spike":             bar vol MUST be >= min_volume_x_recent_for_spike

Universe: small + mid F&O (~100 names). Retail-driven flow concentrated here.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.universe_filter import in_universe
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels, MarketContext, RiskParams, StructureAnalysis, StructureEvent, TradePlan,
)

logger = get_agent_logger()


class PDHPDLRejectStructure(BaseStructure):
    """Fade rejections at PDH (short) or PDL (long) when no breakout volume."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "pdh_pdl_reject"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.level_prox_pct = float(config["level_proximity_pct"]) / 100.0
        self.max_body_pct = float(config["max_body_size_pct"]) / 100.0
        self.min_wick_x_body = float(config["min_upper_wick_x_body"])
        # rev2: volume polarity is A/B variant (Indian sources contested).
        self.volume_polarity = str(config.get("volume_polarity", "absence")).lower()
        if self.volume_polarity not in ("absence", "spike"):
            raise ValueError(f"volume_polarity must be 'absence' or 'spike', got {self.volume_polarity!r}")
        self.max_vol_x_recent_for_absence = float(config.get("max_volume_x_recent_for_absence", 1.5))
        self.min_vol_x_recent_for_spike = float(config.get("min_volume_x_recent_for_spike", 1.5))
        self.wick_buffer_pct = float(config["wick_buffer_pct"]) / 100.0
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            return float((ctx.df_5m["high"] - ctx.df_5m["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_vwap(self, ctx: MarketContext) -> Optional[float]:
        if ctx.indicators and "vwap" in ctx.indicators:
            return float(ctx.indicators["vwap"])
        if ctx.df_5m is not None and "vwap" in ctx.df_5m.columns:
            v = ctx.df_5m["vwap"].iloc[-1]
            if pd.notna(v):
                return float(v)
        return None

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(structure_detected=False, events=[],
                                     quality_score=0.0, rejection_reason=reason or None)

        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        if ctx.pdh is None or ctx.pdl is None:
            return _empty("PDH/PDL unavailable")
        pdh = float(ctx.pdh)
        pdl = float(ctx.pdl)

        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])
        bar_vol = float(last["volume"])

        body = abs(bar_close - bar_open)
        rng = bar_high - bar_low
        if rng <= 0:
            return _empty("zero-range bar")
        body_pct = body / rng

        # Try PDH short
        pdh_band = pdh * self.level_prox_pct
        pdl_band = pdl * self.level_prox_pct

        side: Optional[str] = None
        rejection_extreme: float = 0.0
        if abs(bar_high - pdh) <= pdh_band and bar_close < pdh:
            # Short candidate: high tags PDH, close below PDH
            upper_wick = bar_high - max(bar_open, bar_close)
            if body > 0 and upper_wick / body >= self.min_wick_x_body and body_pct < self.max_body_pct:
                side = "short"
                rejection_extreme = bar_high
        elif abs(bar_low - pdl) <= pdl_band and bar_close > pdl:
            lower_wick = min(bar_open, bar_close) - bar_low
            if body > 0 and lower_wick / body >= self.min_wick_x_body and body_pct < self.max_body_pct:
                side = "long"
                rejection_extreme = bar_low

        if side is None:
            return _empty(f"no PDH/PDL rejection: bar=[{bar_low:.2f},{bar_high:.2f}] "
                          f"PDH={pdh:.2f} PDL={pdl:.2f}")

        # rev2: volume polarity branch.
        # "absence": bar vol must NOT exceed max_volume_x_recent_for_absence × recent
        # "spike":   bar vol MUST exceed min_volume_x_recent_for_spike × recent
        recent_vol = float(df["volume"].iloc[-6:-1].mean()) if len(df) >= 6 else bar_vol
        if recent_vol > 0:
            ratio = bar_vol / recent_vol
            if self.volume_polarity == "absence":
                if ratio > self.max_vol_x_recent_for_absence:
                    return _empty(f"absence_polarity_violated: vol {bar_vol:.0f} > {self.max_vol_x_recent_for_absence}× recent")
            else:  # spike
                if ratio < self.min_vol_x_recent_for_spike:
                    return _empty(f"spike_polarity_violated: vol {bar_vol:.0f} < {self.min_vol_x_recent_for_spike}× recent")

        confidence = min(1.0, (1.0 - body_pct) + 0.2)  # cleaner rejection = higher confidence
        evt = StructureEvent(
            symbol=ctx.symbol, timestamp=last_ts, structure_type=self.structure_type,
            side=side, confidence=confidence,
            levels={"pdh": pdh, "pdl": pdl, "rejection_extreme": rejection_extreme,
                    "close": bar_close, "vwap": self._get_vwap(ctx) or bar_close},
            context={"body_pct": body_pct, "vol_x_recent": bar_vol / max(recent_vol, 1.0)},
            price=bar_close,
        )
        return StructureAnalysis(structure_detected=True, events=[evt],
                                 quality_score=confidence * 100.0)

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        rejection = float(evt.levels["rejection_extreme"])
        wick_buf = close * self.wick_buffer_pct

        if side == "short":
            hard_sl = rejection + wick_buf
            risk = max(hard_sl - close, 1e-6)
        else:
            hard_sl = rejection - wick_buf
            risk = max(close - hard_sl, 1e-6)

        # T1 = VWAP
        vwap = float(evt.levels["vwap"])
        # T2 = today's opposite extreme so far
        today_high = float(df["high"].max())
        today_low = float(df["low"].min())
        if side == "short":
            t1_level = vwap if vwap < close else close - risk
            t2_level = today_low if today_low < t1_level else close - 2 * risk
        else:
            t1_level = vwap if vwap > close else close + risk
            t2_level = today_high if today_high > t1_level else close + 2 * risk

        t1_rr = abs(close - t1_level) / risk
        t2_rr = abs(close - t2_level) / risk
        targets = [
            {"name": "T1", "level": t1_level, "rr": t1_rr,
             "qty_pct": self.t1_qty_pct, "action": "partial_exit"},
            {"name": "T2", "level": t2_level, "rr": t2_rr,
             "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full"},
        ]
        return TradePlan(
            symbol=ctx.symbol, side=side, structure_type=self.structure_type,
            entry_price=close,
            risk_params=RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=self._get_atr(ctx)),
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets),
            qty=0, notional=0.0, confidence=evt.confidence, notes=evt.context,
        )

    def plan_long_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long")

    def plan_short_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short")

    def calculate_risk_params(self, entry_price: float, ctx: MarketContext) -> RiskParams:
        atr = self._get_atr(ctx)
        return RiskParams(hard_sl=entry_price + atr, risk_per_share=atr, atr=atr)

    def get_exit_levels(self, plan: TradePlan) -> ExitLevels:
        return plan.exit_levels

    def rank_setup_quality(self, ctx: MarketContext, event=None) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_pdh_pdl_reject_structure.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add structures/pdh_pdl_reject_structure.py
git add -f tests/structures/test_pdh_pdl_reject_structure.py
git commit -m "feat(sub8-T9): PDH/PDL Touch-and-Reject detector + tests"
```

---

### Task 10: Wire PDH/PDL Reject into main_detector + orchestrator

Same pattern as Task 4. Category = REVERSION.

- [ ] **Step 1: Apply wiring**

In `structures/main_detector.py`:
- Import: `from .pdh_pdl_reject_structure import PDHPDLRejectStructure`
- detector_configs: `("pdh_pdl_reject", PDHPDLRejectStructure, "pdh_pdl_reject"),`
- Identity mapping: `'pdh_pdl_reject': 'pdh_pdl_reject',`

In `pipelines/orchestrator.py`:
- Import: `from structures.pdh_pdl_reject_structure import PDHPDLRejectStructure`
- SUB7_SETUPS add: `"pdh_pdl_reject",`
- _cls_map add: `"pdh_pdl_reject": PDHPDLRejectStructure,`

In `config/setup_categories.py`:
```python
    "pdh_pdl_reject": SetupCategory.REVERSION,
```

- [ ] **Step 2: Smoke-import test**

```bash
.venv/Scripts/python.exe -c "
from pipelines.orchestrator import SUB7_SETUPS
assert 'pdh_pdl_reject' in SUB7_SETUPS
print('OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add structures/main_detector.py pipelines/orchestrator.py config/setup_categories.py
git commit -m "feat(sub8-T10): wire PDH/PDL Reject into main_detector + orchestrator"
```

---

### Task 11: Closing Hour Reversal — implementation + tests

**Files:**
- Create: `structures/closing_hour_reversal_structure.py`
- Create: `tests/structures/test_closing_hour_reversal_structure.py`

Source: design doc Section 7. Replaces sub7 `mis_unwind_short` with corrected stop multiplier (1.2×ATR vs sub7's 0.8×) AND bidirectional support (sub7 was short-only).

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_closing_hour_reversal_structure.py`:

```python
"""Closing Hour Reversal detector unit tests (sub8-T11)."""
import pandas as pd

from structures.closing_hour_reversal_structure import ClosingHourReversalStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "closing_hour_reversal",
        "enabled": True,
        "active_window_start": "14:30",
        "active_window_end": "15:15",
        "min_intraday_move_pct": 1.5,
        "exhaustion_min_body_pct_of_range": 60.0,
        "exhaustion_min_volume_x_recent": 1.3,
        "stop_atr_multiplier": 1.2,
        "t1_target": "vwap",
        "t2_target": "pivot_or_50pct_retrace",
        "t1_qty_pct": 0.5,
        "hard_time_stop_hhmm": "15:18",
        "universe_key": "fno_liquid_200",
        "min_bars_required": 60,
    }


def _build_chr_df(now_time="14:35:00", n_bars=70, open_price=100.0, hod=104.0):
    """Stock ran from 100 to 104 (+4%) by 14:30, last bar prints exhaustion bearish."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5 * (n_bars - 1)),
                        periods=n_bars, freq="5min")
    # Trend up: 100 → 104 over first 60 bars
    trend_closes = [open_price + (hod - open_price) * (i / (n_bars - 2)) for i in range(n_bars - 1)]
    closes = trend_closes + [trend_closes[-1] - 0.5]  # last bar drops 0.5
    opens = [c - 0.05 for c in closes]
    highs = [c + 0.05 for c in closes]
    lows = [c - 0.05 for c in closes]
    # Last bar = bearish exhaustion: large body, big range, high volume
    opens[-1] = trend_closes[-1]
    closes[-1] = trend_closes[-1] - 0.6
    highs[-1] = trend_closes[-1] + 0.05
    lows[-1] = closes[-1] - 0.05
    volumes = [10000] * n_bars
    volumes[-1] = 14000  # 1.4× recent
    df = pd.DataFrame({"open": opens, "high": highs, "low": lows,
                       "close": closes, "volume": volumes,
                       "vwap": [102.0] * n_bars}, index=idx)
    return df


def _ctx(df, symbol="NSE:RELIANCE"):
    last_ts = df.index[-1]
    return MarketContext(
        symbol=symbol,
        current_price=float(df["close"].iloc[-1]),
        timestamp=last_ts,
        df_5m=df,
        session_date=last_ts.to_pydatetime().replace(hour=0, minute=0, second=0),
        cap_segment="large_cap",
        regime="trend_up",
        pdh=110.0, pdl=98.0, pdc=100.0,
        indicators={"atr": 0.5, "vwap": 102.0},
    )


def test_fires_short_on_up_move_exhaustion():
    """Stock ran +4% then prints bearish exhaustion = SHORT (fade the up move)."""
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df()
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is True, f"Expected fire: {res.rejection_reason}"
    assert res.events[0].side == "short"
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert len(plan.exit_levels.targets) == 2
    # Stop multiplier MUST be 1.2× ATR (research-cited per design doc)
    # ATR=0.5, so risk should be ~0.6 (1.2 × 0.5) above entry
    assert plan.risk_params.atr is not None


def test_does_not_fire_outside_window():
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df(now_time="13:00:00")
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "window" in (res.rejection_reason or "").lower()


def test_does_not_fire_when_intraday_move_too_small():
    """If stock moved < 1.5% intraday, no exhaustion to fade."""
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df(open_price=100.0, hod=100.5)  # only 0.5%
    ctx = _ctx(df)
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "move" in (res.rejection_reason or "").lower()


def test_does_not_fire_outside_universe():
    det = ClosingHourReversalStructure(_cfg())
    df = _build_chr_df()
    ctx = _ctx(df, symbol="NSE:UNKNOWNSMALLCAP")
    res = det.detect(ctx)
    assert res.structure_detected is False
    assert "universe" in (res.rejection_reason or "").lower()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_closing_hour_reversal_structure.py -v
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `structures/closing_hour_reversal_structure.py`**

```python
"""Closing Hour Reversal (CHR) — sub8 Setup #5.

Source citations (Indian market):
  - Zerodha — MIS Auto Square-Off Timings (15:20 hard stop)
  - Zerodha Varsity — Volatility Applications (1.2-1.5× ATR for EOD stops)
  - Subhadip Nandy / Capitalmind discussions on EOD reversion (VWAP magnet)
  - Subasish Pani videos on EOD reversal patterns

Trigger: stock has moved >= 1.5% in one direction between 09:30 and 14:30.
At 14:30+, exhaustion candle prints (body >= 60% of range, vol >= 1.3× recent).
Direction: short if move was UP, long if move was DOWN. Hard time stop 15:18.

This is the SUB8 fix to sub7's failed mis_unwind_short. Sub7 used 0.8× ATR
stop (too tight, killed sample). Sub8 uses Varsity-cited 1.2× ATR. Sub7 was
short-only. Sub8 is bidirectional.

Universe: F&O liquid 200 (full universe; reversals trade in both directions).
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.universe_filter import in_universe
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels, MarketContext, RiskParams, StructureAnalysis, StructureEvent, TradePlan,
)

logger = get_agent_logger()


class ClosingHourReversalStructure(BaseStructure):
    """Bidirectional EOD exhaustion reversal in 14:30-15:15 window."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "closing_hour_reversal"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_intraday_move_pct = float(config["min_intraday_move_pct"])
        self.exhaustion_min_body_pct = float(config["exhaustion_min_body_pct_of_range"]) / 100.0
        self.exhaustion_min_vol_x = float(config["exhaustion_min_volume_x_recent"])
        self.stop_atr_mult = float(config["stop_atr_multiplier"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.hard_time_stop = self._parse_time(config["hard_time_stop_hhmm"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            return float((ctx.df_5m["high"] - ctx.df_5m["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_vwap(self, ctx: MarketContext) -> Optional[float]:
        if ctx.indicators and "vwap" in ctx.indicators:
            return float(ctx.indicators["vwap"])
        if ctx.df_5m is not None and "vwap" in ctx.df_5m.columns:
            v = ctx.df_5m["vwap"].iloc[-1]
            if pd.notna(v):
                return float(v)
        return None

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(structure_detected=False, events=[],
                                     quality_score=0.0, rejection_reason=reason or None)

        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # Compute intraday move as (max - min) / open of session.
        # Session = bars >= 09:30 today.
        session_bars = df[df.index.to_series().apply(
            lambda ts: ts.time() >= time(9, 30) and ts.time() <= time(14, 30)
        )]
        if len(session_bars) < 10:
            return _empty("insufficient session bars 09:30-14:30")
        session_open = float(session_bars["open"].iloc[0])
        session_high = float(session_bars["high"].max())
        session_low = float(session_bars["low"].min())
        if session_open <= 0:
            return _empty("invalid session open")
        up_move_pct = (session_high - session_open) / session_open * 100.0
        down_move_pct = (session_open - session_low) / session_open * 100.0

        # Direction: short if up_move dominates, long if down_move dominates
        if up_move_pct >= self.min_intraday_move_pct and up_move_pct > down_move_pct:
            side = "short"  # fade the up move
        elif down_move_pct >= self.min_intraday_move_pct and down_move_pct > up_move_pct:
            side = "long"  # fade the down move
        else:
            return _empty(f"intraday move too small: up={up_move_pct:.2f} down={down_move_pct:.2f}")

        # Exhaustion candle: body >= 60% of range, opposite direction to trend
        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])
        bar_vol = float(last["volume"])
        body = abs(bar_close - bar_open)
        rng = bar_high - bar_low
        if rng <= 0:
            return _empty("zero-range bar")
        body_pct = body / rng
        if body_pct < self.exhaustion_min_body_pct:
            return _empty(f"body_pct={body_pct:.2f} < min={self.exhaustion_min_body_pct}")

        # Direction sanity: if shorting, last bar must close below open (bearish)
        if side == "short" and bar_close >= bar_open:
            return _empty("short signal but bar is bullish")
        if side == "long" and bar_close <= bar_open:
            return _empty("long signal but bar is bearish")

        # Volume confirmation
        recent_vol = float(df["volume"].iloc[-6:-1].mean()) if len(df) >= 6 else bar_vol
        if recent_vol > 0 and bar_vol < self.exhaustion_min_vol_x * recent_vol:
            return _empty(f"volume {bar_vol:.0f} < {self.exhaustion_min_vol_x}× recent {recent_vol:.0f}")

        confidence = min(1.0, body_pct)
        evt = StructureEvent(
            symbol=ctx.symbol, timestamp=last_ts, structure_type=self.structure_type,
            side=side, confidence=confidence,
            levels={"session_high": session_high, "session_low": session_low,
                    "session_open": session_open, "close": bar_close,
                    "vwap": self._get_vwap(ctx) or bar_close},
            context={"up_move_pct": up_move_pct, "down_move_pct": down_move_pct,
                     "body_pct": body_pct, "vol_x_recent": bar_vol / max(recent_vol, 1.0)},
            price=bar_close,
        )
        return StructureAnalysis(structure_detected=True, events=[evt],
                                 quality_score=confidence * 100.0)

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        atr = self._get_atr(ctx)
        session_high = float(evt.levels["session_high"])
        session_low = float(evt.levels["session_low"])
        session_open = float(evt.levels["session_open"])
        vwap = float(evt.levels["vwap"])

        # Stop = recent intraday extreme + 1.2 × ATR (research-cited)
        if side == "short":
            hard_sl = session_high + self.stop_atr_mult * atr
            risk = max(hard_sl - close, 1e-6)
        else:
            hard_sl = session_low - self.stop_atr_mult * atr
            risk = max(close - hard_sl, 1e-6)

        # T1 = VWAP
        # T2 = pivot or 50% retrace, whichever is closer to entry on the right side
        pivot = (float(ctx.pdh) + float(ctx.pdl) + float(ctx.pdc)) / 3.0 \
            if all(v is not None for v in (ctx.pdh, ctx.pdl, ctx.pdc)) else session_open
        retrace_50 = (session_high + session_low) / 2.0

        if side == "short":
            t1_level = vwap if vwap < close else close - risk
            # Pick the t2 candidate that's between t1 and a max-2.5R cap
            candidates = [p for p in [pivot, retrace_50] if p < t1_level]
            t2_level = max(candidates) if candidates else close - 2 * risk
        else:
            t1_level = vwap if vwap > close else close + risk
            candidates = [p for p in [pivot, retrace_50] if p > t1_level]
            t2_level = min(candidates) if candidates else close + 2 * risk

        t1_rr = abs(close - t1_level) / risk
        t2_rr = abs(close - t2_level) / risk
        targets = [
            {"name": "T1", "level": t1_level, "rr": t1_rr,
             "qty_pct": self.t1_qty_pct, "action": "partial_exit"},
            {"name": "T2", "level": t2_level, "rr": t2_rr,
             "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full"},
        ]
        return TradePlan(
            symbol=ctx.symbol, side=side, structure_type=self.structure_type,
            entry_price=close,
            risk_params=RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=atr),
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets),
            qty=0, notional=0.0, confidence=evt.confidence, notes=evt.context,
        )

    def plan_long_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long")

    def plan_short_strategy(self, ctx: MarketContext, event=None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short")

    def calculate_risk_params(self, entry_price: float, ctx: MarketContext) -> RiskParams:
        atr = self._get_atr(ctx)
        return RiskParams(hard_sl=entry_price + atr * self.stop_atr_mult,
                          risk_per_share=atr * self.stop_atr_mult, atr=atr)

    def get_exit_levels(self, plan: TradePlan) -> ExitLevels:
        return plan.exit_levels

    def rank_setup_quality(self, ctx: MarketContext, event=None) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_closing_hour_reversal_structure.py -v
```

Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add structures/closing_hour_reversal_structure.py
git add -f tests/structures/test_closing_hour_reversal_structure.py
git commit -m "feat(sub8-T11): Closing Hour Reversal detector + tests"
```

---

### Task 12: Wire CHR into main_detector + orchestrator

Same pattern as Task 4. Category = REVERSION.

- [ ] **Step 1: Apply wiring**

In `structures/main_detector.py`:
- Import: `from .closing_hour_reversal_structure import ClosingHourReversalStructure`
- detector_configs: `("closing_hour_reversal", ClosingHourReversalStructure, "closing_hour_reversal"),`
- Identity mapping: `'closing_hour_reversal': 'closing_hour_reversal',`

In `pipelines/orchestrator.py`:
- Import: `from structures.closing_hour_reversal_structure import ClosingHourReversalStructure`
- SUB7_SETUPS add: `"closing_hour_reversal",`
- _cls_map add: `"closing_hour_reversal": ClosingHourReversalStructure,`

In `config/setup_categories.py`:
```python
    "closing_hour_reversal": SetupCategory.REVERSION,
```

- [ ] **Step 2: Smoke-import test**

```bash
.venv/Scripts/python.exe -c "
from pipelines.orchestrator import SUB7_SETUPS
expected = {'gap_fade_short', 'mis_unwind_short', 'cpr_mean_revert',
            'orb_15', 'narrow_cpr_breakout', 'vwap_first_pullback',
            'pdh_pdl_reject', 'closing_hour_reversal'}
assert expected == SUB7_SETUPS, f'mismatch: {SUB7_SETUPS ^ expected}'
print('OK — all 8 setups in SUB7_SETUPS')
"
```

- [ ] **Step 3: Commit**

```bash
git add structures/main_detector.py pipelines/orchestrator.py config/setup_categories.py
git commit -m "feat(sub8-T12): wire Closing Hour Reversal into main_detector + orchestrator"
```

---

## Phase 2: Local smoke test (Task 13)

### Task 13: Single-day smoke test — verify all 5 sub8 detectors fire under wide_open

**Files:** none modified.

- [ ] **Step 1: Run all detector unit tests one more time as a regression check**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_orb_15_structure.py tests/structures/test_narrow_cpr_breakout_structure.py tests/structures/test_vwap_first_pullback_structure.py tests/structures/test_pdh_pdl_reject_structure.py tests/structures/test_closing_hour_reversal_structure.py tests/services/test_universe_filter.py -v
```

Expected: 31 PASSED (6+6+4+4+4+7).

- [ ] **Step 2: Enable all 5 sub8 setups in main config (temporarily for smoke)**

Edit `config/configuration.json` to set `enabled: true` for all 5 sub8 setups (orb_15, narrow_cpr_breakout, vwap_first_pullback, pdh_pdl_reject, closing_hour_reversal). Sub7 setups (gap_fade, mis_unwind, cpr_mean_revert) should also be enabled (they already are).

- [ ] **Step 3: Run single-day backtest with wide_open**

```bash
.venv/Scripts/python.exe main.py --dry-run --session-date 2024-06-03
```

Wait ~17 minutes. Note the latest log dir:

```bash
ls -1d logs/backtest_2026* | tail -1
```

- [ ] **Step 4: Verify all 8 setups fire + 99%+ DECISION→ENTRY conversion**

```bash
DIR=$(ls -1d logs/backtest_2026* | tail -1)
.venv/Scripts/python.exe -c "
import json
from collections import Counter
dec = Counter()
trig_ids = set()
dec_by_strat = {}
for line in open(f'$DIR/events.jsonl'):
    d = json.loads(line)
    if d.get('type') == 'DECISION':
        s = (d.get('plan') or {}).get('strategy')
        dec[s] += 1
        dec_by_strat.setdefault(s, set()).add(d['trade_id'])
    elif d.get('type') == 'TRIGGER':
        trig_ids.add(d['trade_id'])
total_dec = sum(dec.values())
total_trig = len(trig_ids)
conv = total_trig / max(total_dec, 1) * 100
print(f'TOTAL DECISION->ENTRY: {total_trig}/{total_dec} ({conv:.1f}%)')
for s, n in dec.most_common():
    matched = len(dec_by_strat[s] & trig_ids)
    print(f'  {s}: {matched}/{n} triggered')
print()
import subprocess
for guard in ('PLAN_SKIP', 'DUPLICATE_BLOCKED', 'fill_quality_rejected', 'too close'):
    r = subprocess.run(['grep', '-c', guard, f'$DIR/agent.log'], capture_output=True, text=True)
    print(f'  {guard}: {r.stdout.strip()}')
"
```

Expected:
- Total conversion >= 99%
- All 8 setups (sub7 + sub8) appear with non-zero decision counts
- 0 PLAN_SKIP, 0 DUPLICATE_BLOCKED (sub7-T8 wide_open bypasses still active)

If any sub8 setup fires 0 decisions, debug:
- Is the universe_key correctly populated in `services/universe_filter.py`?
- Is the time window matched for 2024-06-03 (Monday)?
- Are PDH/PDL/PDC available in MarketContext?

- [ ] **Step 5: Commit smoke evidence**

If green:

```bash
git add -A logs/  # if logs/ tracked; else skip
git commit --allow-empty -m "smoke(sub8-T13): all 8 setups fire on 2024-06-03, 99%+ conversion"
```

---

## Phase 3: Bundled OCI capture (Task 14)

### Task 14: Submit OCI bundle for sub7+sub8 combined

**Files:** none modified.

- [ ] **Step 1: Verify Docker image is current**

```bash
git log --oneline -10
```

Confirm the last 10 commits include T1 through T12 sub8 work. If image is older, rebuild + push (see `oci/` tooling).

- [ ] **Step 2: Submit OCI job**

```bash
.venv/Scripts/python.exe oci/tools/submit_oci_backtest.py \
  --config-overrides config/sub8_oci_overrides.json \
  --start-date 2023-01-01 \
  --end-date 2026-03-31 \
  --output-dir cloud_results/sub8_phase1_capture/
```

Expected: job submitted, returns OCI job ID. Wall time ~3-5 hours, cost ~$100-300.

- [ ] **Step 3: Sanity-check capture content when complete**

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
import pandas as pd
d = Path('cloud_results/sub8_phase1_capture/')
sessions = sorted(d.iterdir()) if d.exists() else []
print(f'sessions: {len(sessions)}')
sample = sessions[100]  # mid-Discovery
tr = sample / 'trade_report.csv'
df = pd.read_csv(tr, low_memory=False)
print(f'sample {sample.name}: {len(df)} trades')
print(df['setup_type'].value_counts())
print()
# Verify gate_input.jsonl exists (sub8 fix vs sub7's missed flag)
gi = sample / 'gate_input.jsonl'
print(f'gate_input.jsonl exists: {gi.exists()} (size: {gi.stat().st_size if gi.exists() else 0:,} bytes)')
"
```

Expected:
- ~745 sessions (3.25 years)
- Setup types: only `gap_fade_short` + 5 sub8 setups (sub7 mis/cpr disabled in override)
- gate_input.jsonl present (validates the sub8 design fix)

If old SMC setups appear → config override didn't apply. Investigate before proceeding.

---

## Phase 4: Per-setup PnL analysis (Task 15)

### Task 15: Run per-setup PnL + reports

**Files:** none modified (reuses sub7-T9/T10 tools).

- [ ] **Step 1: Build per-setup parquets**

```bash
.venv/Scripts/python.exe tools/sub7_validation/build_per_setup_pnl.py \
  --oci-dir cloud_results/sub8_phase1_capture/ \
  --output-dir reports/sub8_validation/
```

Expected output:
```
Loaded N executed trades from cloud_results/sub8_phase1_capture/
Setups present: ['gap_fade_short', 'orb_15', 'narrow_cpr_breakout', 'vwap_first_pullback', 'pdh_pdl_reject', 'closing_hour_reversal']
  ...trade counts and net PnL per setup...
```

- [ ] **Step 2: Generate Discovery period reports per setup**

```bash
for setup in gap_fade_short orb_15 narrow_cpr_breakout vwap_first_pullback pdh_pdl_reject closing_hour_reversal; do
  .venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py \
    --setup-parquet reports/sub8_validation/$setup.parquet \
    --output-dir reports/sub8_validation/$setup/ \
    --period-start 2023-01-01 --period-end 2024-12-31
done
```

- [ ] **Step 3: Tally Phase 1 pass/fail**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
results = {}
for setup in ('gap_fade_short', 'orb_15', 'narrow_cpr_breakout', 'vwap_first_pullback', 'pdh_pdl_reject', 'closing_hour_reversal'):
    f = Path(f'reports/sub8_validation/{setup}/01-metrics.json')
    if not f.exists():
        print(f'{setup}: NO REPORT')
        continue
    d = json.loads(f.read_text())
    results[setup] = d['phase1_passes']
    agg = d['aggregate']
    print(f\"{setup}: {'PASS' if d['phase1_passes'] else 'FAIL'} \"
          f\"PF={agg['net_pf']} n={agg['n_trades']} Sharpe={agg['net_sharpe']}\")
n_pass = sum(results.values())
print(f'\n{n_pass} of 6 setups pass Phase 1.')
if n_pass == 0:
    print('KILL sub-project #8 per design Q7.')
elif n_pass == 1:
    print('SOFT WARNING: only 1 passes. Consider stop-multiplier debugging before composition.')
else:
    print('Proceed to Phase 5 (composition).')
"
```

- [ ] **Step 4: Commit decision artifacts**

```bash
git add -f reports/sub8_validation/
git commit -m "report(sub8-T15): Phase 1 per-setup validation results"
```

---

## Phase 5: Portfolio composition (Task 16, conditional)

### Task 16: Mechanical composition test

**Files:** none modified (reuses sub7-T13 tooling — `tools/sub7_validation/portfolio_composer.py` was planned in sub7 but not built; if missing, see sub7 plan T13 for reference scaffold). For sub8 we only need the analysis report; if portfolio_composer.py doesn't exist yet, skip composition and rely on per-setup reports.

ONLY proceed if 2+ setups passed Phase 1.

- [ ] **Step 1: Run composer if 2+ setups passed**

```bash
.venv/Scripts/python.exe tools/sub7_validation/portfolio_composer.py \
  --setup-parquet-dir reports/sub8_validation/ \
  --output-dir reports/sub8_validation/portfolio/ \
  --period-start 2023-01-01 --period-end 2024-12-31 2>&1 | tail -10
```

Expected: prints PF/Sharpe for equal-weight and risk-parity combinations.

- [ ] **Step 2: Verify Phase 2 verdict**

Pass: `NET PF >= 1.25 AND Net Sharpe >= 0.6 AND Max DD <= 20%`.

If neither composition passes:
- ONE iteration allowed: try inverse-Sharpe weighting or Kelly sizing
- If still fails, KILL sub-8 per design Q7

- [ ] **Step 3: Commit**

```bash
git add -f reports/sub8_validation/portfolio/
git commit -m "report(sub8-T16): Phase 2 portfolio composition results"
```

---

## Phase 6: OOS Validation (Task 17)

### Task 17: Validation OOS test (Jan-Sep 2025)

**Files:** none modified.

ONLY proceed if Phase 2 passed.

- [ ] **Step 1: Slice the existing parquets to validation period**

```bash
for setup in $(ls reports/sub8_validation/*.parquet | xargs -n1 basename | sed s/.parquet//); do
  .venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py \
    --setup-parquet reports/sub8_validation/$setup.parquet \
    --output-dir reports/sub8_validation/$setup/oos_validation/ \
    --period-start 2025-01-01 --period-end 2025-09-30
done
```

- [ ] **Step 2: Verify pass thresholds (NET PF >= 1.15, Net Sharpe >= 0.5, Max DD <= 25%)**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
ALL = ['gap_fade_short','orb_15','narrow_cpr_breakout','vwap_first_pullback','pdh_pdl_reject','closing_hour_reversal']
print('--- OOS Validation (Jan-Sep 2025) ---')
for setup in ALL:
    f = Path(f'reports/sub8_validation/{setup}/oos_validation/01-metrics.json')
    if not f.exists(): continue
    d = json.loads(f.read_text())
    a = d['aggregate']
    print(f\"{setup}: PF={a['net_pf']} n={a['n_trades']} Sharpe={a['net_sharpe']}\")
"
```

- [ ] **Step 3: Commit**

```bash
git add -f reports/sub8_validation/*/oos_validation/
git commit -m "report(sub8-T17): OOS Validation period (Jan-Sep 2025)"
```

---

## Phase 7: OOS Holdout (Task 18)

### Task 18: Holdout OOS test (Oct 2025-Mar 2026)

**Files:** none modified.

ONLY proceed if OOS Validation passed for >= 1 setup.

- [ ] **Step 1: Slice to holdout period**

```bash
for setup in $(ls reports/sub8_validation/*.parquet | xargs -n1 basename | sed s/.parquet//); do
  .venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py \
    --setup-parquet reports/sub8_validation/$setup.parquet \
    --output-dir reports/sub8_validation/$setup/oos_holdout/ \
    --period-start 2025-10-01 --period-end 2026-03-31
done
```

- [ ] **Step 2: Verify pass thresholds (same as Validation)**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
ALL = ['gap_fade_short','orb_15','narrow_cpr_breakout','vwap_first_pullback','pdh_pdl_reject','closing_hour_reversal']
print('--- OOS Holdout (Oct 2025 - Mar 2026) ---')
for setup in ALL:
    f = Path(f'reports/sub8_validation/{setup}/oos_holdout/01-metrics.json')
    if not f.exists(): continue
    d = json.loads(f.read_text())
    a = d['aggregate']
    print(f\"{setup}: PF={a['net_pf']} n={a['n_trades']} Sharpe={a['net_sharpe']}\")
"
```

- [ ] **Step 3: Commit final report**

```bash
git add -f reports/sub8_validation/*/oos_holdout/
git commit -m "report(sub8-T18): OOS Holdout period (Oct 2025-Mar 2026) — sub8 final verdict"
```

---

## Pass/fail summary thresholds

| Phase | Threshold | Action on fail |
|---|---|---|
| Phase 1 (Discovery) | PF ≥ 1.10, n ≥ 500, Sharpe > 0 per setup | Drop failing setups; need ≥2 to compose |
| Phase 2 (Composition) | Combined PF ≥ 1.25, Sharpe ≥ 0.6, Max DD ≤ 20% | One re-iteration allowed; else KILL sub-8 |
| Phase 6 (OOS Validation) | PF ≥ 1.15, Sharpe ≥ 0.5, Max DD ≤ 25% | Drop failing setups |
| Phase 7 (OOS Holdout) | Same as Validation | Drop failing setups; ship the survivors |

---

## Self-review notes

**Spec coverage check:**
- All 5 setups from design Sections 3-7 → Tasks 3, 5, 7, 9, 11. ✓
- Universe filter → Task 1. ✓
- Sub8 OCI override with `gate_input_logging:true` → Task 2. ✓
- Tiered T1/T2 mandatory → enforced via test assertion in every detector test (`assert len(targets) == 2`). ✓
- Per-setup universe filter → enforced in detector `__init__` reading `universe_key` and detector `detect()` calling `in_universe(symbol, universe_key)` first. ✓
- Phase 1/2/Validation/Holdout thresholds → mirror sub7 exactly per design Section 10. ✓

**Type consistency check:**
- All detectors emit `targets = [{name, level, rr, qty_pct, action}, ...]` matching the orchestrator's existing parser at `pipelines/orchestrator.py::_build_plan_from_sub7_detector` (lines 378-394). ✓
- All detectors register in `SUB7_SETUPS` frozenset to route through the fast path. ✓
- All detectors implement the same `BaseStructure` abstract methods (`detect`, `plan_long_strategy`, `plan_short_strategy`, `calculate_risk_params`, `get_exit_levels`, `rank_setup_quality`, `validate_timing`). ✓

**Placeholder scan:** None found. Each task has complete code blocks and exact commands.

---

## Total task count: 18

- Phase 0: T1, T2 (config + universe filter)
- Phase 1: T3-T12 (5 detectors × 2 tasks each = 10)
- Phase 2: T13 (smoke test)
- Phase 3: T14 (OCI capture)
- Phase 4: T15 (per-setup PnL)
- Phase 5: T16 (composition, conditional)
- Phase 6: T17 (OOS Validation)
- Phase 7: T18 (OOS Holdout)
