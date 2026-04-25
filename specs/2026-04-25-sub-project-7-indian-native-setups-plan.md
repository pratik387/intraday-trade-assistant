# Sub-Project #7 (Indian-Native Setup Library) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portfolio of three Indian-native intraday equity setups (MIS unwind shorts, opening gap fade shorts, CPR mean reversion), validated independently before composition. Replace SMC-style detectors which sub-project #5 proved structurally mismatched to Indian intraday markets.

**Architecture:** Three new detector classes plug into existing `structures/main_detector.py`. Per-setup validation runs through a single bundled OCI capture (wide_open_mode + only new detectors enabled) and re-uses the existing parity_simulator + build_pnl_index pipeline with a NET fee-aware analysis layer added on top. Composition is mechanical (equal-weight, then risk-parity), NOT joint Optuna optimization. OOS validation uses sub-project #5's existing val/holdout periods (semantically fresh per design Q6).

**Tech Stack:** Python 3.10, pandas, pyarrow (existing). No new dependencies.

---

## Source spec

`specs/2026-04-25-sub-project-7-indian-native-setups-design.md` — approved.

## Scope

Sub-project #7 only. Does NOT delete existing detector files (kept in repo, disabled via config). Does NOT add new data feeds (FII/DII, India VIX) — those are deferred Phase 2 candidates. Does NOT deploy live (requires separate sub-project #8 decision).

## Prerequisites

- Sub-project #5 OCI infrastructure (already exists)
- Sub-project #5 wide_open_mode cascade (already shipped — pipelines/level_pipeline.py, services/gates/trade_decision_gate.py, services/gate_chain/live_gate_chain.py)
- Sub-project #5 cycle_limit fix (max_trades_per_cycle override) — verified via `oci_runner.py` config

## File structure

| Path | Purpose | Created/Modified |
|------|---------|------------------|
| `structures/mis_unwind_short_structure.py` | Detector 1: shorts during MIS auto-square window | Create |
| `structures/gap_fade_short_structure.py` | Detector 2: shorts opening-bell gap-up exhaustion | Create |
| `structures/cpr_mean_revert_structure.py` | Detector 3: long+short mean reversion in lunch lull | Create |
| `structures/main_detector.py` | Add 3 new setups to `detector_configs` list | Modify |
| `config/configuration.json` | Add 3 new setup config blocks (enabled=false default) | Modify |
| `config/sub7_oci_overrides.json` | OCI run override: disable all existing setups, enable only 3 new | Create |
| `tools/sub7_validation/__init__.py` | Package marker | Create |
| `tools/sub7_validation/build_per_setup_pnl.py` | Splits trade_report by setup_type, applies Indian intraday fees | Create |
| `tools/sub7_validation/per_setup_report.py` | Per-setup metrics + monthly/cap-segment/regime breakdowns | Create |
| `tools/sub7_validation/portfolio_composer.py` | Mechanical composition (equal-weight + risk-parity) | Create |
| `tools/sub7_validation/oos_validator.py` | One-shot Validation + Holdout OOS test (uses sub5 trial.py) | Create |
| `tests/structures/test_mis_unwind_short_structure.py` | Detector 1 unit tests | Create |
| `tests/structures/test_gap_fade_short_structure.py` | Detector 2 unit tests | Create |
| `tests/structures/test_cpr_mean_revert_structure.py` | Detector 3 unit tests | Create |
| `tests/sub7_validation/__init__.py` | Package marker | Create |
| `tests/sub7_validation/test_build_per_setup_pnl.py` | Fee math + grouping tests | Create |
| `tests/sub7_validation/test_portfolio_composer.py` | Composition tests | Create |

---

## Phase 0: Config + scaffolding (Task 1)

### Task 1: Add new setup config blocks + OCI override file

**Files:**
- Modify: `config/configuration.json` (append new setup blocks under `setups`)
- Create: `config/sub7_oci_overrides.json`

- [ ] **Step 1: Add 3 new setup config blocks to configuration.json**

In `config/configuration.json`, locate the `"setups"` object and add these three blocks at the end (BEFORE the closing brace of `"setups"`):

```json
"mis_unwind_short": {
  "enabled": false,
  "active_window_start": "14:55",
  "active_window_end": "15:15",
  "min_distance_above_vwap_pct": 0.5,
  "min_intraday_high_recency_min": 30,
  "max_momentum_3bar_pct": 0.0,
  "min_rvol": 1.2,
  "allowed_cap_segments": ["small_cap", "mid_cap", "micro_cap"],
  "stop_atr_buffer": 0.8,
  "target_type": "vwap",
  "time_stop_min_before_close": 5,
  "min_bars_required": 30
},
"gap_fade_short": {
  "enabled": false,
  "active_window_start": "09:15",
  "active_window_end": "09:30",
  "min_gap_pct_above_pdc": 1.5,
  "max_gap_pct_above_pdc": 8.0,
  "min_upper_wick_ratio": 0.5,
  "max_body_size_pct": 30.0,
  "require_volume_decline_after_gap": true,
  "allowed_cap_segments": ["small_cap", "mid_cap", "micro_cap"],
  "stop_above_gap_high_atr": 0.3,
  "target_type": "pdc_or_open",
  "time_stop_at": "10:00",
  "min_bars_required": 1
},
"cpr_mean_revert": {
  "enabled": false,
  "active_window_start": "11:30",
  "active_window_end": "13:30",
  "min_distance_atr_from_cpr": 1.0,
  "max_volume_pct_of_intraday_avg": 30.0,
  "require_reversion_candle": true,
  "reversion_patterns": ["hammer", "doji", "shooting_star"],
  "allowed_cap_segments": ["small_cap", "mid_cap", "large_cap"],
  "stop_at_extreme_atr_buffer": 0.2,
  "target_type": "cpr_midpoint",
  "time_stop_at": "13:45",
  "min_bars_required": 30
}
```

- [ ] **Step 2: Create OCI override file**

Create `config/sub7_oci_overrides.json`:

```json
{
  "wide_open_mode": true,
  "max_trades_per_cycle": 10000,
  "gate_input_logging": {
    "enabled": true
  },
  "live_gate_chain": {
    "enabled": false
  },
  "_comment_setups": "Disable ALL existing setups; enable only sub-project #7 detectors",
  "setups": {
    "mis_unwind_short": {"enabled": true},
    "gap_fade_short": {"enabled": true},
    "cpr_mean_revert": {"enabled": true},
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
    "liquidity_sweep_short": {"enabled": false},
    "break_of_structure_long": {"enabled": false},
    "break_of_structure_short": {"enabled": false},
    "change_of_character_long": {"enabled": false},
    "change_of_character_short": {"enabled": false},
    "vwap_lose_short": {"enabled": false},
    "vwap_reclaim_long": {"enabled": false},
    "vwap_mean_reversion_long": {"enabled": false},
    "vwap_mean_reversion_short": {"enabled": false},
    "gap_fill_long": {"enabled": false},
    "gap_fill_short": {"enabled": false},
    "gap_breakout_long": {"enabled": false},
    "gap_breakout_short": {"enabled": false},
    "orb_breakout_long": {"enabled": false},
    "orb_breakout_short": {"enabled": false},
    "orb_pullback_long": {"enabled": false},
    "orb_pullback_short": {"enabled": false},
    "trend_pullback_long": {"enabled": false},
    "trend_pullback_short": {"enabled": false},
    "trend_continuation_long": {"enabled": false},
    "trend_continuation_short": {"enabled": false},
    "momentum_breakout_long": {"enabled": false},
    "momentum_breakout_short": {"enabled": false},
    "failure_fade_long": {"enabled": false},
    "failure_fade_short": {"enabled": false},
    "squeeze_release_long": {"enabled": false},
    "squeeze_release_short": {"enabled": false},
    "flag_continuation_long": {"enabled": false},
    "flag_continuation_short": {"enabled": false},
    "volume_spike_reversal_long": {"enabled": false},
    "volume_spike_reversal_short": {"enabled": false},
    "volume_breakout_long": {"enabled": false},
    "volume_breakout_short": {"enabled": false}
  }
}
```

- [ ] **Step 3: Verify config loads**

Run:
```bash
.venv/Scripts/python.exe -c "import json; cfg=json.load(open('config/configuration.json')); print('mis_unwind_short:', cfg['setups'].get('mis_unwind_short', {}).get('enabled', 'MISSING')); print('gap_fade_short:', cfg['setups'].get('gap_fade_short', {}).get('enabled', 'MISSING')); print('cpr_mean_revert:', cfg['setups'].get('cpr_mean_revert', {}).get('enabled', 'MISSING'))"
```

Expected output:
```
mis_unwind_short: False
gap_fade_short: False
cpr_mean_revert: False
```

- [ ] **Step 4: Verify OCI override file parses**

Run:
```bash
.venv/Scripts/python.exe -c "import json; o=json.load(open('config/sub7_oci_overrides.json')); print('wide_open:', o['wide_open_mode']); print('cycle_limit:', o['max_trades_per_cycle']); print('new setups enabled:', sum(1 for k,v in o['setups'].items() if v.get('enabled'))); print('old setups disabled:', sum(1 for k,v in o['setups'].items() if not v.get('enabled')))"
```

Expected output:
```
wide_open: True
cycle_limit: 10000
new setups enabled: 3
old setups disabled: 56
```

- [ ] **Step 5: Commit**

```bash
git add -f config/configuration.json config/sub7_oci_overrides.json
git commit -m "feat(sub7-T1): add config blocks for 3 Indian-native detectors + OCI override"
```

---

## Phase 1: Detector implementations (Tasks 2-7, two tasks per detector)

### Task 2: MIS Unwind Short — unit tests + scaffold

**Files:**
- Create: `structures/mis_unwind_short_structure.py`
- Create: `tests/structures/test_mis_unwind_short_structure.py`

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_mis_unwind_short_structure.py`:

```python
"""MIS unwind short detector unit tests (sub7-T2)."""
import pandas as pd
from datetime import datetime, time
from structures.mis_unwind_short_structure import MISUnwindShortStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "mis_unwind_short",
        "enabled": True,
        "active_window_start": "14:55",
        "active_window_end": "15:15",
        "min_distance_above_vwap_pct": 0.5,
        "min_intraday_high_recency_min": 30,
        "max_momentum_3bar_pct": 0.0,
        "min_rvol": 1.2,
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "stop_atr_buffer": 0.8,
        "target_type": "vwap",
        "time_stop_min_before_close": 5,
        "min_bars_required": 30,
    }


def _build_df(now_time, n_bars=40, last_close=105.0, vwap=100.0,
              recent_high_offset_min=15, momentum_3bar_pct=-0.3):
    """Build a minimal df where the last bar is at `now_time`,
    with a fresh intraday high `recent_high_offset_min` ago and weakening momentum."""
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5*(n_bars-1)), periods=n_bars, freq="5min")
    closes = [vwap] * n_bars
    highs = [vwap + 0.5] * n_bars
    # Set a fresh intraday high N minutes ago
    high_bar = max(0, n_bars - 1 - (recent_high_offset_min // 5))
    highs[high_bar] = last_close + 1.0  # the fresh high
    closes[-1] = last_close             # current close above VWAP
    # Momentum_3bar_pct calculated from closes[-4] to closes[-1]
    closes[-4] = last_close - (momentum_3bar_pct / 100.0) * last_close
    df = pd.DataFrame({
        "open": closes, "high": highs,
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": [10000] * n_bars,
        "vwap": [vwap] * n_bars,
    }, index=idx)
    return df


def test_fires_in_window_with_valid_setup():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", recent_high_offset_min=15, momentum_3bar_pct=-0.5)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is True
    assert any(e.structure_type == "mis_unwind_short" for e in result.events)


def test_does_not_fire_outside_window():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("11:00:00")  # well before active window
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False


def test_does_not_fire_below_vwap():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", last_close=99.0, vwap=100.0)  # below VWAP
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False


def test_does_not_fire_with_positive_momentum():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", momentum_3bar_pct=+0.5)  # still going up
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False


def test_does_not_fire_in_disallowed_cap_segment():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00")
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={},
                        atr=1.0, cap_segment="large_cap", regime="trend_up", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is False
```

- [ ] **Step 2: Create scaffold (returns no detections)**

Create `structures/mis_unwind_short_structure.py`:

```python
"""MIS Unwind Short detector — sub-project #7.

Thesis: SEBI requires MIS positions to square off by 3:20 PM. Retail intraday
flow is structurally net-long. The forced unwind in the last 60-90 minutes
creates asymmetric net-sell pressure. Pros short into this.

Active window: 14:55-15:15 IST.
"""
from __future__ import annotations
from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (MarketContext, StructureAnalysis, StructureEvent,
                          TradePlan, RiskParams, ExitLevels)

logger = get_agent_logger()


class MISUnwindShortStructure(BaseStructure):
    """Detects late-day short opportunities driven by MIS auto-square unwind."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "mis_unwind_short"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_dist_vwap_pct = float(config["min_distance_above_vwap_pct"])
        self.min_high_recency_min = int(config["min_intraday_high_recency_min"])
        self.max_momentum_3bar_pct = float(config["max_momentum_3bar_pct"])
        self.min_rvol = float(config["min_rvol"])
        self.allowed_caps = set(config.get("allowed_cap_segments", []))
        self.stop_atr_buffer = float(config["stop_atr_buffer"])
        self.target_type = str(config["target_type"])
        self.time_stop_min_before_close = int(config["time_stop_min_before_close"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        """Stub: returns no detections (scaffolding for tests)."""
        return StructureAnalysis(
            structure_detected=False, events=[], quality_score=0.0,
            structure_type=self.structure_type,
        )

    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        return None  # short-only setup

    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        return None  # implemented in Task 3

    def calculate_risk_params(self, entry_price: float,
                               market_context: MarketContext) -> RiskParams:
        return RiskParams(
            stop_loss=entry_price + market_context.atr * self.stop_atr_buffer,
            risk_per_share=market_context.atr * self.stop_atr_buffer,
            risk_amount=1000.0,  # fixed Rs risk per design
        )
```

- [ ] **Step 3: Run failing tests**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_mis_unwind_short_structure.py -v
```

Expected: tests run; first 4 tests assert `structure_detected is True` or check window/cap/VWAP/momentum logic — most should FAIL because `detect()` always returns False.

- [ ] **Step 4: Implement detect() logic**

Replace the `detect()` method in `structures/mis_unwind_short_structure.py`:

```python
    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        empty = StructureAnalysis(
            structure_detected=False, events=[], quality_score=0.0,
            structure_type=self.structure_type,
        )
        df = market_context.df_5m
        if df is None or len(df) < self.min_bars_required:
            return empty

        # Cap segment filter
        if market_context.cap_segment not in self.allowed_caps:
            return empty

        # Active time-of-day window (current bar's time)
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return empty

        last = df.iloc[-1]
        vwap = float(last.get("vwap", 0.0))
        close = float(last.get("close", 0.0))
        if vwap <= 0:
            return empty

        # Must be above VWAP by min_distance_above_vwap_pct
        dist_vwap_pct = ((close - vwap) / vwap) * 100.0
        if dist_vwap_pct < self.min_dist_vwap_pct:
            return empty

        # RVOL filter
        if (market_context.rvol or 0.0) < self.min_rvol:
            return empty

        # Fresh intraday high within last min_high_recency_min minutes
        recent_window_bars = max(1, self.min_high_recency_min // 5)
        recent_highs = df["high"].iloc[-recent_window_bars:]
        intraday_max = df["high"].max()
        if recent_highs.max() < intraday_max - 1e-9:
            # Fresh high not in recent window
            return empty

        # Momentum_3bar_pct must be NEGATIVE (weakening)
        if len(df) < 4:
            return empty
        c_now = float(df["close"].iloc[-1])
        c_3ago = float(df["close"].iloc[-4])
        if c_3ago <= 0:
            return empty
        mom_3 = ((c_now - c_3ago) / c_3ago) * 100.0
        if mom_3 > self.max_momentum_3bar_pct:
            return empty

        # All conditions met
        evt = StructureEvent(
            structure_type=self.structure_type,
            timestamp=last_ts,
            confidence=min(1.0, abs(mom_3) / 1.0),  # confidence scales with momentum decay
            metadata={"dist_vwap_pct": dist_vwap_pct, "momentum_3bar_pct": mom_3,
                      "rvol": market_context.rvol},
        )
        return StructureAnalysis(
            structure_detected=True, events=[evt],
            quality_score=evt.confidence, structure_type=self.structure_type,
        )
```

- [ ] **Step 5: Run tests until all pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_mis_unwind_short_structure.py -v
```

Expected: all 5 tests PASS.

If any failure, inspect the test, refine `detect()` logic. Don't commit until green.

- [ ] **Step 6: Commit**

```bash
git add -f structures/mis_unwind_short_structure.py tests/structures/test_mis_unwind_short_structure.py
git commit -m "feat(sub7-T2): MIS unwind short detector + tests"
```

---

### Task 3: MIS Unwind Short — plan_short_strategy + integration

**Files:**
- Modify: `structures/mis_unwind_short_structure.py` (implement plan_short_strategy)
- Modify: `structures/main_detector.py` (add to detector_configs)
- Modify: `tests/structures/test_mis_unwind_short_structure.py` (add plan tests)

- [ ] **Step 1: Add plan tests**

Append to `tests/structures/test_mis_unwind_short_structure.py`:

```python
def test_plan_short_strategy_returns_valid_plan():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00", last_close=105.0, vwap=100.0)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={"VWAP_curr": 100.0},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert plan.bias == "short"
    assert plan.strategy == "mis_unwind_short"
    assert plan.entry_zone[0] <= 105.0 <= plan.entry_zone[1]
    # Stop should be ABOVE entry (short)
    assert plan.exit_levels.hard_sl > 105.0
    # Target should be at or near VWAP (100.0)
    assert plan.exit_levels.targets[0]["level"] <= 102.0


def test_plan_long_strategy_returns_none():
    cfg = _cfg()
    det = MISUnwindShortStructure(cfg)
    df = _build_df("15:00:00")
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df, levels={},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    assert det.plan_long_strategy(ctx) is None
```

- [ ] **Step 2: Run tests, verify failures**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_mis_unwind_short_structure.py::test_plan_short_strategy_returns_valid_plan -v
```

Expected: FAIL (plan_short_strategy returns None).

- [ ] **Step 3: Implement plan_short_strategy()**

Replace the `plan_short_strategy()` method:

```python
    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        analysis = self.detect(market_context)
        if not analysis.structure_detected:
            return None

        df = market_context.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(market_context.atr or 1.0)
        recent_high = float(df["high"].max())

        # Entry zone: at current close (immediate), small slippage tolerance
        entry_low = close * 0.999
        entry_high = close * 1.001

        # Stop: ABOVE recent intraday high + ATR buffer
        hard_sl = recent_high + atr * self.stop_atr_buffer
        risk_per_share = hard_sl - close

        # Target: VWAP or PDC (config target_type)
        vwap = float(last.get("vwap", close))
        pdc = float(market_context.levels.get("PDC", vwap)) if market_context.levels else vwap
        target_level = vwap if self.target_type == "vwap" else pdc
        # Ensure target makes sense for short (target < entry)
        if target_level >= close:
            target_level = vwap  # fallback

        rr = (close - target_level) / max(risk_per_share, 1e-6)
        targets = [{"name": "T1", "level": target_level, "rr": rr,
                    "qty_pct": 1.0, "action": "exit_full"}]

        return TradePlan(
            symbol=market_context.symbol,
            strategy="mis_unwind_short",
            bias="short",
            eligible=True,
            entry_zone=(entry_low, entry_high),
            entry_reference=close,
            entry_mode="immediate",
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets, trail=None),
            risk_params=self.calculate_risk_params(close, market_context),
            confidence=analysis.quality_score,
            timestamp=df.index[-1],
            metadata=analysis.events[0].metadata if analysis.events else {},
        )
```

- [ ] **Step 4: Run plan tests**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_mis_unwind_short_structure.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Wire into main_detector.py**

In `structures/main_detector.py`, locate the `detector_configs` list (around line 64). Find a sensible insertion point (after the existing imports of structure classes — around line 33). At top of file, add:

```python
from .mis_unwind_short_structure import MISUnwindShortStructure
```

Then inside the `detector_configs` list (around line 145, BEFORE the closing bracket and after the last entry), append:

```python
            ("mis_unwind_short", MISUnwindShortStructure, "mis_unwind_short"),
```

- [ ] **Step 6: Smoke test main_detector loads new detector**

```bash
.venv/Scripts/python.exe -c "
import json
from structures.main_detector import MainDetector
cfg = json.load(open('config/configuration.json'))
# Force-enable just our detector
cfg['setups']['mis_unwind_short']['enabled'] = True
md = MainDetector(cfg)
assert 'mis_unwind_short' in md.detectors, f'detector not loaded; got: {list(md.detectors.keys())[:5]}...'
print('mis_unwind_short detector loaded:', md.detectors['mis_unwind_short'].__class__.__name__)
"
```

Expected output:
```
mis_unwind_short detector loaded: MISUnwindShortStructure
```

- [ ] **Step 7: Commit**

```bash
git add -f structures/mis_unwind_short_structure.py structures/main_detector.py tests/structures/test_mis_unwind_short_structure.py
git commit -m "feat(sub7-T3): wire MIS unwind short into main_detector + plan_short_strategy"
```

---

### Task 4: Gap Fade Short — unit tests + scaffold + detect()

**Files:**
- Create: `structures/gap_fade_short_structure.py`
- Create: `tests/structures/test_gap_fade_short_structure.py`

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_gap_fade_short_structure.py`:

```python
"""Gap Fade Short detector unit tests (sub7-T4)."""
import pandas as pd
from datetime import datetime
from structures.gap_fade_short_structure import GapFadeShortStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "gap_fade_short",
        "enabled": True,
        "active_window_start": "09:15",
        "active_window_end": "09:30",
        "min_gap_pct_above_pdc": 1.5,
        "max_gap_pct_above_pdc": 8.0,
        "min_upper_wick_ratio": 0.5,
        "max_body_size_pct": 30.0,
        "require_volume_decline_after_gap": True,
        "allowed_cap_segments": ["small_cap", "mid_cap"],
        "stop_above_gap_high_atr": 0.3,
        "target_type": "pdc_or_open",
        "time_stop_at": "10:00",
        "min_bars_required": 1,
    }


def _build_gap_df(now_time, gap_pct, upper_wick_ratio, body_size_pct,
                   pdc=100.0, n_bars=2, vol_after_gap=8000):
    """Build df: bar 0 = opening gap-up bar with given pattern, bar 1 = current."""
    open_price = pdc * (1 + gap_pct/100.0)
    body = open_price * (body_size_pct / 100.0)
    close_price = open_price + body if body_size_pct > 0 else open_price
    upper_wick = body * upper_wick_ratio if body > 0 else open_price * (upper_wick_ratio / 100.0)
    high_price = close_price + upper_wick
    low_price = open_price * 0.998

    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5*(n_bars-1)), periods=n_bars, freq="5min")
    rows = []
    for i in range(n_bars):
        if i == 0:
            rows.append({"open": open_price, "high": high_price, "low": low_price,
                         "close": close_price, "volume": 15000, "vwap": open_price})
        else:
            rows.append({"open": close_price, "high": close_price + 0.1, "low": close_price - 0.5,
                         "close": close_price - 0.2, "volume": vol_after_gap, "vwap": open_price})
    return pd.DataFrame(rows, index=idx)


def test_fires_with_valid_gap_fade_pattern():
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    df = _build_gap_df("09:25:00", gap_pct=2.5, upper_wick_ratio=0.8,
                        body_size_pct=20.0, vol_after_gap=8000)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"PDC": 100.0, "OPEN": df.iloc[0]["open"]},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    result = det.detect(ctx)
    assert result.structure_detected is True


def test_does_not_fire_outside_window():
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    df = _build_gap_df("11:00:00", gap_pct=2.5, upper_wick_ratio=0.8, body_size_pct=20.0)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"PDC": 100.0, "OPEN": df.iloc[0]["open"]},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    assert det.detect(ctx).structure_detected is False


def test_does_not_fire_below_gap_threshold():
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    df = _build_gap_df("09:25:00", gap_pct=0.5,  # below 1.5% min
                        upper_wick_ratio=0.8, body_size_pct=20.0)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"PDC": 100.0, "OPEN": df.iloc[0]["open"]},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    assert det.detect(ctx).structure_detected is False


def test_does_not_fire_above_max_gap():
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    df = _build_gap_df("09:25:00", gap_pct=10.0,  # above 8% max (likely halted)
                        upper_wick_ratio=0.8, body_size_pct=20.0)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"PDC": 100.0, "OPEN": df.iloc[0]["open"]},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    assert det.detect(ctx).structure_detected is False


def test_does_not_fire_with_strong_body():
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    df = _build_gap_df("09:25:00", gap_pct=2.5, upper_wick_ratio=0.1,
                        body_size_pct=80.0)  # strong body, no exhaustion
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"PDC": 100.0, "OPEN": df.iloc[0]["open"]},
                        atr=1.0, cap_segment="small_cap", regime="trend_up", rvol=1.5)
    assert det.detect(ctx).structure_detected is False


def test_does_not_fire_in_disallowed_cap_segment():
    cfg = _cfg()
    det = GapFadeShortStructure(cfg)
    df = _build_gap_df("09:25:00", gap_pct=2.5, upper_wick_ratio=0.8, body_size_pct=20.0)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"PDC": 100.0, "OPEN": df.iloc[0]["open"]},
                        atr=1.0, cap_segment="large_cap", regime="trend_up", rvol=1.5)
    assert det.detect(ctx).structure_detected is False
```

- [ ] **Step 2: Implement scaffold + detect() + plan_short_strategy()**

Create `structures/gap_fade_short_structure.py`:

```python
"""Gap Fade Short detector — sub-project #7.

Thesis: FII gap-up + retail chase → exhaustion at first 5m bars in mid/small
caps. Stocks over-extend, then mean-revert.

Active window: 09:15-09:30 IST (first three 5m bars).
"""
from __future__ import annotations
from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (MarketContext, StructureAnalysis, StructureEvent,
                          TradePlan, RiskParams, ExitLevels)

logger = get_agent_logger()


class GapFadeShortStructure(BaseStructure):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "gap_fade_short"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_gap_pct = float(config["min_gap_pct_above_pdc"])
        self.max_gap_pct = float(config["max_gap_pct_above_pdc"])
        self.min_upper_wick_ratio = float(config["min_upper_wick_ratio"])
        self.max_body_size_pct = float(config["max_body_size_pct"])
        self.require_volume_decline = bool(config["require_volume_decline_after_gap"])
        self.allowed_caps = set(config.get("allowed_cap_segments", []))
        self.stop_above_gap_high_atr = float(config["stop_above_gap_high_atr"])
        self.target_type = str(config["target_type"])
        self.time_stop_at = self._parse_time(config["time_stop_at"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        empty = StructureAnalysis(
            structure_detected=False, events=[], quality_score=0.0,
            structure_type=self.structure_type,
        )
        df = market_context.df_5m
        if df is None or len(df) < self.min_bars_required:
            return empty

        if market_context.cap_segment not in self.allowed_caps:
            return empty

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return empty

        # Need PDC level
        pdc = float(market_context.levels.get("PDC", 0.0)) if market_context.levels else 0.0
        if pdc <= 0:
            return empty

        # Use the OPENING bar (first 5m of session) for gap measurement
        opening_bar = df.iloc[0]
        open_price = float(opening_bar["open"])
        gap_pct = ((open_price - pdc) / pdc) * 100.0
        if not (self.min_gap_pct <= gap_pct <= self.max_gap_pct):
            return empty

        # Exhaustion candle on opening or recent bar
        cur = df.iloc[-1]
        body = abs(float(cur["close"]) - float(cur["open"]))
        upper_wick = float(cur["high"]) - max(float(cur["close"]), float(cur["open"]))
        if body <= 0:
            return empty  # doji is OK but our wick ratio needs body
        wick_ratio = upper_wick / body
        body_pct = (body / float(cur["open"])) * 100.0
        if wick_ratio < self.min_upper_wick_ratio:
            return empty
        if body_pct > self.max_body_size_pct:
            return empty

        # Volume decline check (volume on current bar < first bar)
        if self.require_volume_decline and len(df) >= 2:
            if float(cur["volume"]) >= float(opening_bar["volume"]):
                return empty

        evt = StructureEvent(
            structure_type=self.structure_type,
            timestamp=last_ts,
            confidence=min(1.0, wick_ratio / 2.0),
            metadata={"gap_pct": gap_pct, "wick_ratio": wick_ratio,
                      "body_pct": body_pct},
        )
        return StructureAnalysis(
            structure_detected=True, events=[evt],
            quality_score=evt.confidence, structure_type=self.structure_type,
        )

    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        return None

    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        analysis = self.detect(market_context)
        if not analysis.structure_detected:
            return None

        df = market_context.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(market_context.atr or 1.0)

        # Stop: above the opening gap high + ATR buffer
        gap_high = float(df.iloc[0]["high"])
        hard_sl = gap_high + atr * self.stop_above_gap_high_atr
        risk_per_share = max(hard_sl - close, atr * 0.1)

        # Target: PDC (most aggressive) or opening price (conservative)
        pdc = float(market_context.levels.get("PDC", 0.0))
        opening = float(df.iloc[0]["open"])
        target_level = pdc if self.target_type == "pdc_or_open" else opening
        if target_level >= close:
            target_level = opening * 0.999  # ensure short target

        rr = (close - target_level) / max(risk_per_share, 1e-6)
        targets = [{"name": "T1", "level": target_level, "rr": rr,
                    "qty_pct": 1.0, "action": "exit_full"}]

        return TradePlan(
            symbol=market_context.symbol,
            strategy="gap_fade_short",
            bias="short",
            eligible=True,
            entry_zone=(close * 0.999, close * 1.001),
            entry_reference=close,
            entry_mode="immediate",
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets, trail=None),
            risk_params=RiskParams(stop_loss=hard_sl, risk_per_share=risk_per_share, risk_amount=1000.0),
            confidence=analysis.quality_score,
            timestamp=df.index[-1],
            metadata=analysis.events[0].metadata if analysis.events else {},
        )

    def calculate_risk_params(self, entry_price: float,
                               market_context: MarketContext) -> RiskParams:
        atr = float(market_context.atr or 1.0)
        return RiskParams(
            stop_loss=entry_price + atr * self.stop_above_gap_high_atr,
            risk_per_share=atr * self.stop_above_gap_high_atr,
            risk_amount=1000.0,
        )
```

- [ ] **Step 3: Run tests until all pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_gap_fade_short_structure.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add -f structures/gap_fade_short_structure.py tests/structures/test_gap_fade_short_structure.py
git commit -m "feat(sub7-T4): Gap Fade Short detector + tests"
```

---

### Task 5: Gap Fade Short — wire into main_detector

**Files:**
- Modify: `structures/main_detector.py`

- [ ] **Step 1: Add import + detector_configs entry**

In `structures/main_detector.py`, add at top (with other detector imports around line 33):

```python
from .gap_fade_short_structure import GapFadeShortStructure
```

In `detector_configs` list (after the mis_unwind_short entry from Task 3):

```python
            ("gap_fade_short", GapFadeShortStructure, "gap_fade_short"),
```

- [ ] **Step 2: Smoke test**

```bash
.venv/Scripts/python.exe -c "
import json
from structures.main_detector import MainDetector
cfg = json.load(open('config/configuration.json'))
cfg['setups']['mis_unwind_short']['enabled'] = True
cfg['setups']['gap_fade_short']['enabled'] = True
md = MainDetector(cfg)
assert 'mis_unwind_short' in md.detectors
assert 'gap_fade_short' in md.detectors
print('Both detectors loaded:', sorted([k for k in md.detectors if k in ('mis_unwind_short','gap_fade_short')]))
"
```

Expected:
```
Both detectors loaded: ['gap_fade_short', 'mis_unwind_short']
```

- [ ] **Step 3: Commit**

```bash
git add -f structures/main_detector.py
git commit -m "feat(sub7-T5): wire Gap Fade Short into main_detector"
```

---

### Task 6: CPR Mean Revert — unit tests + scaffold + detect() + plans

**Files:**
- Create: `structures/cpr_mean_revert_structure.py`
- Create: `tests/structures/test_cpr_mean_revert_structure.py`

- [ ] **Step 1: Write failing tests**

Create `tests/structures/test_cpr_mean_revert_structure.py`:

```python
"""CPR Mean Revert detector unit tests (sub7-T6)."""
import pandas as pd
from structures.cpr_mean_revert_structure import CPRMeanRevertStructure
from structures.data_models import MarketContext


def _cfg():
    return {
        "_setup_name": "cpr_mean_revert",
        "enabled": True,
        "active_window_start": "11:30",
        "active_window_end": "13:30",
        "min_distance_atr_from_cpr": 1.0,
        "max_volume_pct_of_intraday_avg": 30.0,
        "require_reversion_candle": True,
        "reversion_patterns": ["hammer", "doji", "shooting_star"],
        "allowed_cap_segments": ["small_cap", "mid_cap", "large_cap"],
        "stop_at_extreme_atr_buffer": 0.2,
        "target_type": "cpr_midpoint",
        "time_stop_at": "13:45",
        "min_bars_required": 30,
    }


def _build_lull_df(now_time, last_close, cpr_mid=100.0, atr=1.0,
                    cur_volume=2000, intraday_avg_volume=10000,
                    candle_pattern="hammer"):
    """Build df for 11:30-13:30 lull period.
    last_close > cpr_mid + atr → setup for SHORT reversion (toward CPR).
    last_close < cpr_mid - atr → setup for LONG reversion.
    Volume must be < max_volume_pct_of_intraday_avg of intraday avg."""
    n_bars = 40
    end = pd.Timestamp(f"2025-01-02 {now_time}")
    idx = pd.date_range(end - pd.Timedelta(minutes=5*(n_bars-1)), periods=n_bars, freq="5min")

    closes = [cpr_mid] * (n_bars - 1) + [last_close]
    highs = [c + 0.3 for c in closes]
    lows = [c - 0.3 for c in closes]
    volumes = [intraday_avg_volume] * (n_bars - 1) + [cur_volume]

    if candle_pattern == "hammer" and last_close > cpr_mid:
        # Hammer at top — small body, long lower wick
        opens = closes[:-1] + [last_close + 0.4]  # open above close (red small body)
        lows[-1] = last_close - 1.2  # long lower wick
        highs[-1] = last_close + 0.5
    elif candle_pattern == "shooting_star" and last_close < cpr_mid:
        # Shooting star at bottom — small body, long upper wick
        opens = closes[:-1] + [last_close - 0.4]
        highs[-1] = last_close + 1.2
        lows[-1] = last_close - 0.5
    else:
        opens = list(closes)

    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes, "vwap": closes,
    }, index=idx)


def test_fires_short_reversion_above_cpr():
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)
    df = _build_lull_df("12:00:00", last_close=102.0, cpr_mid=100.0,
                        atr=1.0, cur_volume=2000, candle_pattern="hammer")
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"CPR_TOP": 101.0, "CPR_BOTTOM": 99.0, "PDC": 100.0},
                        atr=1.0, cap_segment="small_cap", regime="chop", rvol=0.8)
    result = det.detect(ctx)
    assert result.structure_detected is True
    plan = det.plan_short_strategy(ctx)
    assert plan is not None
    assert plan.bias == "short"


def test_fires_long_reversion_below_cpr():
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)
    df = _build_lull_df("12:00:00", last_close=98.0, cpr_mid=100.0,
                        atr=1.0, cur_volume=2000, candle_pattern="shooting_star")
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"CPR_TOP": 101.0, "CPR_BOTTOM": 99.0, "PDC": 100.0},
                        atr=1.0, cap_segment="small_cap", regime="chop", rvol=0.8)
    result = det.detect(ctx)
    assert result.structure_detected is True
    plan = det.plan_long_strategy(ctx)
    assert plan is not None
    assert plan.bias == "long"


def test_does_not_fire_outside_window():
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)
    df = _build_lull_df("10:00:00", last_close=102.0, cpr_mid=100.0)
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"CPR_TOP": 101.0, "CPR_BOTTOM": 99.0, "PDC": 100.0},
                        atr=1.0, cap_segment="small_cap", regime="chop", rvol=0.8)
    assert det.detect(ctx).structure_detected is False


def test_does_not_fire_close_to_cpr():
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)
    df = _build_lull_df("12:00:00", last_close=100.5, cpr_mid=100.0)  # only 0.5 ATR away
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"CPR_TOP": 101.0, "CPR_BOTTOM": 99.0, "PDC": 100.0},
                        atr=1.0, cap_segment="small_cap", regime="chop", rvol=0.8)
    assert det.detect(ctx).structure_detected is False


def test_does_not_fire_with_high_volume():
    cfg = _cfg()
    det = CPRMeanRevertStructure(cfg)
    df = _build_lull_df("12:00:00", last_close=102.0, cpr_mid=100.0,
                        cur_volume=8000)  # 80% of avg, above 30% threshold
    ctx = MarketContext(symbol="NSE:SYM", df_5m=df,
                        levels={"CPR_TOP": 101.0, "CPR_BOTTOM": 99.0, "PDC": 100.0},
                        atr=1.0, cap_segment="small_cap", regime="chop", rvol=0.8)
    assert det.detect(ctx).structure_detected is False
```

- [ ] **Step 2: Implement detector**

Create `structures/cpr_mean_revert_structure.py`:

```python
"""CPR Mean Revert detector — sub-project #7.

Thesis: Lunch lull (11:30-13:30) has low volume → range-trading dominates →
stocks revert to CPR midpoint. Both long (rebound from below) and short
(fade from above) setups.

Active window: 11:30-13:30 IST.
"""
from __future__ import annotations
from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (MarketContext, StructureAnalysis, StructureEvent,
                          TradePlan, RiskParams, ExitLevels)

logger = get_agent_logger()


class CPRMeanRevertStructure(BaseStructure):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "cpr_mean_revert"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_dist_atr = float(config["min_distance_atr_from_cpr"])
        self.max_vol_pct = float(config["max_volume_pct_of_intraday_avg"])
        self.require_reversion = bool(config["require_reversion_candle"])
        self.reversion_patterns = set(config.get("reversion_patterns", []))
        self.allowed_caps = set(config.get("allowed_cap_segments", []))
        self.stop_atr_buffer = float(config["stop_at_extreme_atr_buffer"])
        self.target_type = str(config["target_type"])
        self.time_stop_at = self._parse_time(config["time_stop_at"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    @staticmethod
    def _candle_pattern(bar) -> str:
        o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
        body = abs(c - o)
        rng = h - l
        if rng <= 0:
            return "none"
        body_pct = body / rng
        upper_wick = h - max(o, c)
        lower_wick = min(o, c) - l
        if body_pct < 0.1:
            return "doji"
        if lower_wick > body * 2 and upper_wick < body * 0.5:
            return "hammer"
        if upper_wick > body * 2 and lower_wick < body * 0.5:
            return "shooting_star"
        return "none"

    def detect(self, market_context: MarketContext) -> StructureAnalysis:
        empty = StructureAnalysis(
            structure_detected=False, events=[], quality_score=0.0,
            structure_type=self.structure_type,
        )
        df = market_context.df_5m
        if df is None or len(df) < self.min_bars_required:
            return empty

        if market_context.cap_segment not in self.allowed_caps:
            return empty

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return empty

        levels = market_context.levels or {}
        cpr_top = float(levels.get("CPR_TOP", 0.0))
        cpr_bot = float(levels.get("CPR_BOTTOM", 0.0))
        if cpr_top <= 0 or cpr_bot <= 0:
            return empty
        cpr_mid = (cpr_top + cpr_bot) / 2.0

        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(market_context.atr or 1.0)
        dist_atr = abs(close - cpr_mid) / max(atr, 1e-6)
        if dist_atr < self.min_dist_atr:
            return empty

        # Volume must be a small % of intraday average
        intraday_vol = df["volume"].iloc[:-1]
        if intraday_vol.empty:
            return empty
        avg_vol = float(intraday_vol.mean())
        cur_vol = float(last["volume"])
        if avg_vol <= 0:
            return empty
        vol_pct = (cur_vol / avg_vol) * 100.0
        if vol_pct > self.max_vol_pct:
            return empty

        # Reversion candle
        if self.require_reversion:
            pattern = self._candle_pattern(last)
            if pattern not in self.reversion_patterns:
                return empty
            # Pattern must align with mean-revert direction:
            # above CPR → expect SHORT → hammer/doji/shooting_star (any reversal)
            # below CPR → expect LONG → hammer/doji (any bullish reversal)

        bias = "short" if close > cpr_mid else "long"
        evt = StructureEvent(
            structure_type=self.structure_type,
            timestamp=last_ts,
            confidence=min(1.0, dist_atr / 3.0),
            metadata={"bias": bias, "dist_atr_from_cpr": dist_atr,
                      "vol_pct_of_avg": vol_pct, "cpr_mid": cpr_mid},
        )
        return StructureAnalysis(
            structure_detected=True, events=[evt],
            quality_score=evt.confidence, structure_type=self.structure_type,
        )

    def _build_plan(self, market_context: MarketContext, bias: str) -> Optional[TradePlan]:
        analysis = self.detect(market_context)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.metadata.get("bias") != bias:
            return None

        df = market_context.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        atr = float(market_context.atr or 1.0)
        cpr_mid = float(evt.metadata.get("cpr_mid", close))

        if bias == "short":
            extreme = float(last["high"])
            hard_sl = extreme + atr * self.stop_atr_buffer
            target_level = cpr_mid
        else:
            extreme = float(last["low"])
            hard_sl = extreme - atr * self.stop_atr_buffer
            target_level = cpr_mid

        risk_per_share = abs(hard_sl - close)
        rr = abs(close - target_level) / max(risk_per_share, 1e-6)
        targets = [{"name": "T1", "level": target_level, "rr": rr,
                    "qty_pct": 1.0, "action": "exit_full"}]

        return TradePlan(
            symbol=market_context.symbol,
            strategy="cpr_mean_revert",
            bias=bias,
            eligible=True,
            entry_zone=(close * 0.999, close * 1.001),
            entry_reference=close,
            entry_mode="immediate",
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets, trail=None),
            risk_params=RiskParams(stop_loss=hard_sl, risk_per_share=risk_per_share, risk_amount=1000.0),
            confidence=analysis.quality_score,
            timestamp=df.index[-1],
            metadata=evt.metadata,
        )

    def plan_long_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        return self._build_plan(market_context, bias="long")

    def plan_short_strategy(self, market_context: MarketContext) -> Optional[TradePlan]:
        return self._build_plan(market_context, bias="short")

    def calculate_risk_params(self, entry_price: float,
                               market_context: MarketContext) -> RiskParams:
        atr = float(market_context.atr or 1.0)
        return RiskParams(
            stop_loss=entry_price * 1.01,  # placeholder; real plan computes from extreme
            risk_per_share=atr * self.stop_atr_buffer,
            risk_amount=1000.0,
        )
```

- [ ] **Step 3: Run tests until all pass**

```bash
.venv/Scripts/python.exe -m pytest tests/structures/test_cpr_mean_revert_structure.py -v
```

Expected: all 5 tests PASS.

If a test fails because data_models.MarketContext doesn't accept a kwarg you used, inspect `structures/data_models.py` and adapt the test fixture.

- [ ] **Step 4: Commit**

```bash
git add -f structures/cpr_mean_revert_structure.py tests/structures/test_cpr_mean_revert_structure.py
git commit -m "feat(sub7-T6): CPR Mean Revert detector + tests"
```

---

### Task 7: CPR Mean Revert — wire into main_detector

**Files:**
- Modify: `structures/main_detector.py`

- [ ] **Step 1: Add import + detector_configs entry**

In `structures/main_detector.py`, add at top (with other detector imports):

```python
from .cpr_mean_revert_structure import CPRMeanRevertStructure
```

In `detector_configs` list (after gap_fade_short entry from Task 5):

```python
            ("cpr_mean_revert", CPRMeanRevertStructure, "cpr_mean_revert"),
```

- [ ] **Step 2: Smoke test all 3 detectors load**

```bash
.venv/Scripts/python.exe -c "
import json
from structures.main_detector import MainDetector
cfg = json.load(open('config/configuration.json'))
for s in ('mis_unwind_short', 'gap_fade_short', 'cpr_mean_revert'):
    cfg['setups'][s]['enabled'] = True
md = MainDetector(cfg)
expected = {'mis_unwind_short', 'gap_fade_short', 'cpr_mean_revert'}
loaded = expected & set(md.detectors.keys())
assert loaded == expected, f'missing: {expected - loaded}'
print('All 3 sub7 detectors loaded:', sorted(loaded))
"
```

Expected:
```
All 3 sub7 detectors loaded: ['cpr_mean_revert', 'gap_fade_short', 'mis_unwind_short']
```

- [ ] **Step 3: Commit**

```bash
git add -f structures/main_detector.py
git commit -m "feat(sub7-T7): wire CPR Mean Revert into main_detector"
```

---

## Phase 2: Local subset validation (Task 8)

### Task 8: Local single-day smoke test

**Files:** none modified — runs existing main.py

- [ ] **Step 1: Pick a Discovery test session**

Use 2024-06-03 (a typical Monday). Verify it exists in local cache:

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
import os
cache = Path('cache/zerodha_1m_archive')
if cache.exists():
    sample = sorted(cache.glob('*'))[:3]
    print('cache exists, sample:', [p.name for p in sample])
else:
    print('NO cache — local backtest will fail. Need bar data first.')
"
```

If no local cache, this task can be SKIPPED — Task 9 (OCI run) is authoritative. Document the skip and proceed.

- [ ] **Step 2: Create temp config with new setups enabled**

```bash
.venv/Scripts/python.exe -c "
import json
cfg = json.load(open('config/configuration.json'))
# Disable everything else, enable just our 3
for k, v in cfg['setups'].items():
    if isinstance(v, dict):
        cfg['setups'][k]['enabled'] = False
for s in ('mis_unwind_short', 'gap_fade_short', 'cpr_mean_revert'):
    cfg['setups'][s]['enabled'] = True
cfg['wide_open_mode'] = True
cfg['max_trades_per_cycle'] = 10000
cfg['live_gate_chain']['enabled'] = False
json.dump(cfg, open('/tmp/sub7_smoke_cfg.json', 'w'), indent=2)
print('temp config written')
"
```

- [ ] **Step 3: Run main.py --dry-run for one session**

```bash
.venv/Scripts/python.exe main.py --dry-run --session-date 2024-06-03 --config /tmp/sub7_smoke_cfg.json 2>&1 | tail -30
```

Expected: completes without crash. Trades may be 0 if 2024-06-03 is a quiet day; that's OK as a smoke test (proves detectors load and run).

If it crashes on missing config keys / detector errors, fix in detector code, recommit, re-run.

- [ ] **Step 4: Inspect output**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
import pandas as pd
# Find latest log dir created
latest = sorted(Path('logs').glob('20*'))[-1]
print('latest log dir:', latest)
tr = latest / 'trade_report.csv'
if tr.exists():
    df = pd.read_csv(tr, low_memory=False)
    if 'setup_type' in df.columns:
        print('Setups in trade_report:')
        print(df['setup_type'].value_counts())
    else:
        print('No setup_type col')
else:
    print('NO trade_report.csv produced')
"
```

Expected: setups column shows only sub7 setups (no SMC ones). Numbers may be small (it's one day).

- [ ] **Step 5: Commit anything fixed during this task** (no separate commit if nothing changed)

If you fixed bugs during this task, commit them with:

```bash
git add -f structures/
git commit -m "fix(sub7-T8): smoke-test bug fixes for local run"
```

---

## Phase 3: Per-setup validation harness (Tasks 9-10)

### Task 9: build_per_setup_pnl.py

**Files:**
- Create: `tools/sub7_validation/__init__.py`
- Create: `tools/sub7_validation/build_per_setup_pnl.py`
- Create: `tests/sub7_validation/__init__.py`
- Create: `tests/sub7_validation/test_build_per_setup_pnl.py`

- [ ] **Step 1: Create package markers**

```bash
mkdir -p tools/sub7_validation tests/sub7_validation
echo '"""sub7 validation harness."""' > tools/sub7_validation/__init__.py
echo '"""sub7 validation tests."""' > tests/sub7_validation/__init__.py
```

- [ ] **Step 2: Write failing test for fee calculation**

Create `tests/sub7_validation/test_build_per_setup_pnl.py`:

```python
"""build_per_setup_pnl tests (sub7-T9)."""
import pandas as pd
from tools.sub7_validation.build_per_setup_pnl import calc_fee, build_net_per_setup


def test_calc_fee_long_trade_basic():
    # Long trade: BUY at 100, SELL at 102, qty 100
    fee = calc_fee(entry_price=100.0, exit_price=102.0, qty=100, side="BUY")
    # Brokerage: min(0.0003*10000, 20) + min(0.0003*10200, 20) = 3.0 + 3.06 = 6.06
    # STT: exit_to * 0.00025 = 10200 * 0.00025 = 2.55 (sell-side)
    # Stamp: entry_to * 0.00003 = 10000 * 0.00003 = 0.30 (buy-side)
    # Exchange: (10000+10200)*0.0000297 = 0.5999
    # SEBI: (20200)*0.000001 = 0.0202
    # IPFT: same = 0.0202
    # GST: (6.06 + 0.5999 + 0.0202 + 0.0202)*0.18 = 1.2204
    # Total ~10.7
    assert 10.0 < fee < 12.0


def test_calc_fee_short_trade_basic():
    # Short trade: SELL at 100, BUY at 98, qty 100
    fee = calc_fee(entry_price=100.0, exit_price=98.0, qty=100, side="SELL")
    # Brokerage: ~6.0
    # STT: entry_to * 0.00025 = 10000 * 0.00025 = 2.50 (sell-side = entry for short)
    # Stamp: exit_to * 0.00003 = 9800 * 0.00003 = 0.294 (buy-side = exit for short)
    # Total ~10.5
    assert 10.0 < fee < 12.0


def test_calc_fee_zero_qty():
    assert calc_fee(100.0, 102.0, 0, "BUY") == 0.0


def test_build_net_per_setup_groups_by_setup():
    df = pd.DataFrame({
        'session_date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        'setup_type': ['mis_unwind_short', 'gap_fade_short', 'mis_unwind_short'],
        'entry_price': [100.0, 50.0, 200.0],
        'e1_price': [98.0, 52.0, 195.0],
        'qty': [100, 200, 50],
        'side': ['SELL', 'SELL', 'SELL'],
        'realized_pnl': [200.0, -400.0, 250.0],
        'executed': [True, True, True],
    })
    out = build_net_per_setup(df)
    # 2 mis_unwind_short trades, 1 gap_fade_short
    assert set(out['setup_type'].unique()) == {'mis_unwind_short', 'gap_fade_short'}
    assert (out['fee'] > 0).all()
    assert 'net_pnl' in out.columns
    # Net = realized_pnl - fee
    for _, row in out.iterrows():
        assert abs(row['net_pnl'] - (row['realized_pnl'] - row['fee'])) < 0.01
```

- [ ] **Step 3: Implement build_per_setup_pnl.py**

Create `tools/sub7_validation/build_per_setup_pnl.py`:

```python
"""Build per-setup net PnL from trade_report.csv files (sub7-T9).

For each session's trade_report.csv, applies Indian intraday fee schedule
and groups by setup_type. Writes one parquet per setup with NET PnL.

CLI:
    python tools/sub7_validation/build_per_setup_pnl.py \\
        --oci-dir <path-to-OCI-output> \\
        --output-dir reports/sub7_validation/
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Indian intraday fee schedule (services/logging/trading_logger.py)
BROK_RATE = 0.0003
BROK_CAP = 20.0
STT_RATE = 0.00025
EXCH_RATE = 0.0000297
SEBI_RATE = 0.000001
IPFT_RATE = 0.000001
STAMP_RATE = 0.00003
GST_RATE = 0.18


def calc_fee(entry_price: float, exit_price: float, qty: int, side: str) -> float:
    """Compute round-trip fees for one Indian intraday equity trade."""
    if qty <= 0 or entry_price is None or exit_price is None:
        return 0.0
    if pd.isna(entry_price) or pd.isna(exit_price):
        return 0.0
    entry_to = float(entry_price) * int(qty)
    exit_to = float(exit_price) * int(qty)

    eb = min(BROK_RATE * entry_to, BROK_CAP)
    xb = min(BROK_RATE * exit_to, BROK_CAP)
    brok = eb + xb

    if side == "BUY":
        stt = exit_to * STT_RATE
        stamp = entry_to * STAMP_RATE
    else:
        stt = entry_to * STT_RATE
        stamp = exit_to * STAMP_RATE

    leg = entry_to + exit_to
    exch = leg * EXCH_RATE
    sebi = leg * SEBI_RATE
    ipft = leg * IPFT_RATE
    gst = (brok + exch + sebi + ipft) * GST_RATE

    return brok + stt + exch + sebi + ipft + stamp + gst


def build_net_per_setup(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to executed trades, compute fee + net PnL per row, return."""
    if df.empty or "executed" not in df.columns:
        return pd.DataFrame()
    mask = df["executed"] == True
    sub = df[mask].copy()
    if sub.empty:
        return sub
    sub["fee"] = sub.apply(
        lambda r: calc_fee(r.get("entry_price"), r.get("e1_price"),
                           int(r.get("qty", 0) or 0), r.get("side", "")),
        axis=1,
    )
    sub["net_pnl"] = sub["realized_pnl"].astype(float) - sub["fee"]
    return sub


def aggregate_oci_dir(oci_dir: Path) -> pd.DataFrame:
    """Walk OCI dir, load all trade_reports, return aggregated net DataFrame."""
    parts = []
    for f in sorted(glob.glob(f"{oci_dir}/*/trade_report.csv")):
        sess = Path(f).parent.name
        df = pd.read_csv(f, low_memory=False)
        if "realized_pnl" not in df.columns:
            continue
        sub = build_net_per_setup(df)
        if sub.empty:
            continue
        sub["session_date"] = sess
        parts.append(sub[["session_date", "setup_type", "realized_pnl",
                          "fee", "net_pnl", "qty", "entry_price", "e1_price",
                          "side", "decision_ts", "symbol",
                          "regime", "cap_segment", "rank_score"]])
    if not parts:
        raise SystemExit(f"[build_per_setup_pnl] no trade_reports under {oci_dir}")
    return pd.concat(parts, ignore_index=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oci-dir", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()

    oci_dir = Path(args.oci_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    big = aggregate_oci_dir(oci_dir)
    print(f"Loaded {len(big):,} executed trades from {oci_dir}")
    print(f"Setups present: {sorted(big['setup_type'].unique())}")
    for setup, grp in big.groupby("setup_type"):
        out_path = out_dir / f"{setup}.parquet"
        grp.to_parquet(out_path, index=False)
        print(f"  {setup}: {len(grp)} trades  net=Rs {int(grp['net_pnl'].sum()):,}  → {out_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/sub7_validation/test_build_per_setup_pnl.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add -f tools/sub7_validation/__init__.py tools/sub7_validation/build_per_setup_pnl.py tests/sub7_validation/__init__.py tests/sub7_validation/test_build_per_setup_pnl.py
git commit -m "feat(sub7-T9): build_per_setup_pnl with Indian intraday fee model"
```

---

### Task 10: per_setup_report.py

**Files:**
- Create: `tools/sub7_validation/per_setup_report.py`

- [ ] **Step 1: Implement metric report**

Create `tools/sub7_validation/per_setup_report.py`:

```python
"""Per-setup metrics + breakdowns report (sub7-T10).

Loads <setup>.parquet from build_per_setup_pnl output, splits by Discovery
period (2023-01-01 to 2024-12-31 by default), computes:
  - Aggregate net metrics (PF, Sharpe, total PnL, n_trades, WR)
  - Per-month breakdown (decay check)
  - Per-cap-segment breakdown
  - Per-regime breakdown
  - Per-day-of-week breakdown
Writes JSON + markdown report.

CLI:
    python tools/sub7_validation/per_setup_report.py \\
        --setup-parquet reports/sub7_validation/mis_unwind_short.parquet \\
        --output-dir reports/sub7_validation/mis_unwind_short/ \\
        --period-start 2023-01-01 --period-end 2024-12-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0, "n_sessions": 0, "net_pnl": 0.0,
                "net_pf": 0.0, "net_sharpe": 0.0, "wr": 0.0,
                "trades_per_day": 0.0, "losing_days_pct": 0.0,
                "max_dd": 0.0}
    n = df["net_pnl"]
    wins = n[n > 0].sum()
    losses = n[n < 0].abs().sum()
    pf = float(wins / losses) if losses > 0 else float("inf")
    daily = df.groupby("session_date")["net_pnl"].sum()
    sharpe = float(daily.mean() / daily.std()) if daily.std() > 0 else 0.0
    cumret = daily.cumsum()
    max_dd = float((cumret - cumret.cummax()).min())
    return {
        "n_trades": int(len(df)),
        "n_sessions": int(daily.size),
        "net_pnl": float(n.sum()),
        "net_pf": round(pf, 3) if pf != float("inf") else 999.0,
        "net_sharpe": round(sharpe, 3),
        "wr": round(float((n > 0).mean()), 3),
        "trades_per_day": round(len(df) / daily.size, 2) if daily.size else 0.0,
        "losing_days_pct": round(100 * (daily < 0).sum() / daily.size, 1) if daily.size else 0.0,
        "max_dd": round(max_dd, 0),
    }


def breakdown_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for k, grp in df.groupby(col):
        m = compute_metrics(grp)
        m[col] = k
        rows.append(m)
    return pd.DataFrame(rows).sort_values("net_pnl", ascending=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup-parquet", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--period-start", default="2023-01-01")
    p.add_argument("--period-end", default="2024-12-31")
    args = p.parse_args()

    df = pd.read_parquet(args.setup_parquet)
    setup_name = Path(args.setup_parquet).stem
    df = df[(df["session_date"] >= args.period_start) & (df["session_date"] <= args.period_end)]
    print(f"Loaded {len(df)} trades for {setup_name} in {args.period_start} to {args.period_end}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    agg = compute_metrics(df)
    df["month"] = df["session_date"].astype(str).str[:7]
    by_month = breakdown_by(df, "month")
    by_cap = breakdown_by(df, "cap_segment") if "cap_segment" in df.columns else pd.DataFrame()
    by_regime = breakdown_by(df, "regime") if "regime" in df.columns else pd.DataFrame()

    # Pass/fail per Phase 1 bar
    bar = {"net_pf_min": 1.10, "n_trades_min": 500, "net_sharpe_min": 0.0}
    passes = (
        agg["net_pf"] >= bar["net_pf_min"]
        and agg["n_trades"] >= bar["n_trades_min"]
        and agg["net_sharpe"] >= bar["net_sharpe_min"]
    )

    result = {
        "setup": setup_name,
        "period": {"start": args.period_start, "end": args.period_end},
        "aggregate": agg,
        "phase1_pass_criteria": bar,
        "phase1_passes": bool(passes),
    }
    (out_dir / "01-metrics.json").write_text(json.dumps(result, indent=2))

    by_month.to_csv(out_dir / "02-by-month.csv", index=False)
    if not by_cap.empty:
        by_cap.to_csv(out_dir / "03-by-cap-segment.csv", index=False)
    if not by_regime.empty:
        by_regime.to_csv(out_dir / "04-by-regime.csv", index=False)

    md = [f"# {setup_name} — Per-setup Report",
          f"\n**Period:** {args.period_start} to {args.period_end}",
          f"\n**Phase 1 verdict:** {'PASS' if passes else 'FAIL'}",
          "\n## Aggregate Metrics", "```json", json.dumps(agg, indent=2), "```",
          "\n## Per-month breakdown", by_month.to_markdown(index=False)]
    if not by_cap.empty:
        md.extend(["\n## Per-cap-segment", by_cap.to_markdown(index=False)])
    if not by_regime.empty:
        md.extend(["\n## Per-regime", by_regime.to_markdown(index=False)])
    (out_dir / "05-report.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Pass: {passes}  PF={agg['net_pf']}  n={agg['n_trades']}  Sharpe={agg['net_sharpe']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Quick smoke test against fake data**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd, numpy as np
from pathlib import Path
np.random.seed(42)
n = 1000
df = pd.DataFrame({
    'session_date': np.random.choice(pd.date_range('2023-01-01', '2024-12-31', freq='B').strftime('%Y-%m-%d'), n),
    'setup_type': ['mis_unwind_short'] * n,
    'realized_pnl': np.random.normal(150, 600, n),
    'fee': np.random.uniform(50, 200, n),
    'cap_segment': np.random.choice(['small_cap', 'mid_cap'], n),
    'regime': np.random.choice(['trend_up', 'chop'], n),
    'qty': [100]*n, 'entry_price': [100.0]*n, 'e1_price': [100.0]*n,
    'side': ['SELL']*n, 'decision_ts': ['2024-01-01']*n, 'symbol': ['NSE:X']*n,
    'rank_score': [1.0]*n,
})
df['net_pnl'] = df['realized_pnl'] - df['fee']
out = Path('/tmp/sub7_smoke')
out.mkdir(exist_ok=True)
df.to_parquet(out / 'mis_unwind_short.parquet', index=False)
print('smoke fixture written')
"
.venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py --setup-parquet /tmp/sub7_smoke/mis_unwind_short.parquet --output-dir /tmp/sub7_smoke/mis_unwind_short/ 2>&1 | tail -5
```

Expected: prints "Pass: True/False  PF=... n=... Sharpe=..." with a real number, no crash.

- [ ] **Step 3: Commit**

```bash
git add -f tools/sub7_validation/per_setup_report.py
git commit -m "feat(sub7-T10): per_setup_report — net metrics + breakdowns"
```

---

## Phase 4: OCI capture run (Task 11)

### Task 11: Submit bundled OCI run

**Files:** none modified

This task is OPERATOR-DRIVEN (requires OCI access). It can run in parallel with later tasks once submitted.

- [ ] **Step 1: Verify Docker image is current**

The OCI Docker image must include all sub7-T2 through sub7-T7 commits. Verify:

```bash
git log --oneline -10
```

Confirm last 5 commits include the 3 detector tasks. If image is older, rebuild + push first (see oci/ tooling).

- [ ] **Step 2: Submit OCI job**

```bash
.venv/Scripts/python.exe oci/tools/submit_oci_backtest.py \
  --config-overrides config/sub7_oci_overrides.json \
  --start-date 2023-01-01 \
  --end-date 2026-03-31 \
  --output-dir cloud_results/sub7_phase1_capture/
```

Expected: job submitted, returns OCI job ID. Wall time ~3 hours, cost ~$50-200.

- [ ] **Step 3: Monitor and download results**

While running, the OCI job will populate the output dir. When complete, download to local using existing OCI tooling (or rsync from object storage). Confirm:

```bash
.venv/Scripts/python.exe -c "
from pathlib import Path
d = Path('cloud_results/sub7_phase1_capture/')
sessions = sorted(d.iterdir()) if d.exists() else []
print(f'sessions: {len(sessions)}')
if sessions:
    print(f'first: {sessions[0].name}, last: {sessions[-1].name}')
"
```

Expected: ~745 session dirs (one per trading day from 2023-01-01 to 2026-03-31).

- [ ] **Step 4: Sanity check capture content**

```bash
.venv/Scripts/python.exe -c "
import pandas as pd
from pathlib import Path
sample = sorted(Path('cloud_results/sub7_phase1_capture/').iterdir())[100]  # ~mid-Discovery
tr = sample / 'trade_report.csv'
if tr.exists():
    df = pd.read_csv(tr, low_memory=False)
    print(f'session: {sample.name}, trades: {len(df)}')
    if 'setup_type' in df.columns:
        print(df['setup_type'].value_counts())
"
```

Expected: setup_type column should show ONLY mis_unwind_short, gap_fade_short, cpr_mean_revert. NO old SMC setups.

If old setups appear → config override didn't apply correctly. Investigate before proceeding.

---

## Phase 5: Per-setup analysis + decision (Task 12)

### Task 12: Run per-setup pass/fail analysis

**Files:** none modified

- [ ] **Step 1: Build per-setup parquets**

```bash
.venv/Scripts/python.exe tools/sub7_validation/build_per_setup_pnl.py \
  --oci-dir cloud_results/sub7_phase1_capture/ \
  --output-dir reports/sub7_validation/
```

Expected output:
```
Loaded N executed trades from cloud_results/sub7_phase1_capture/
Setups present: ['cpr_mean_revert', 'gap_fade_short', 'mis_unwind_short']
  cpr_mean_revert: ... trades  net=Rs ...
  gap_fade_short:  ... trades  net=Rs ...
  mis_unwind_short: ... trades  net=Rs ...
```

- [ ] **Step 2: Generate per-setup reports for Discovery 2023-2024 only**

```bash
for setup in mis_unwind_short gap_fade_short cpr_mean_revert; do
  .venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py \
    --setup-parquet reports/sub7_validation/$setup.parquet \
    --output-dir reports/sub7_validation/$setup/ \
    --period-start 2023-01-01 --period-end 2024-12-31
done
```

Expected: 3 sets of report files. Each prints PASS or FAIL.

- [ ] **Step 3: Decide based on Phase 1 bar**

Read each `reports/sub7_validation/<setup>/01-metrics.json`. Tally how many pass:

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
results = {}
for setup in ('mis_unwind_short', 'gap_fade_short', 'cpr_mean_revert'):
    f = Path(f'reports/sub7_validation/{setup}/01-metrics.json')
    if f.exists():
        d = json.loads(f.read_text())
        results[setup] = d['phase1_passes']
        print(f\"{setup}: {'PASS' if d['phase1_passes'] else 'FAIL'} (PF={d['aggregate']['net_pf']}, n={d['aggregate']['n_trades']}, Sharpe={d['aggregate']['net_sharpe']})\")
n_pass = sum(results.values())
print(f'\\n{n_pass} of 3 detectors pass Phase 1.')
if n_pass == 0:
    print('KILL sub-project #7 per design Q7.')
elif n_pass == 1:
    print('SOFT WARNING: only 1 passes. Add Phase 2 candidates (FII/DII, VIX) before composition.')
else:
    print('Proceed to Phase 6 (composition).')
"
```

- [ ] **Step 4: Commit decision artifacts**

```bash
git add -f reports/sub7_validation/
git commit -m "report(sub7-T12): Phase 1 per-setup validation results"
```

If Phase 1 verdict is KILL or SOFT WARNING, STOP HERE and re-plan with the user.

---

## Phase 6: Portfolio composition (Task 13)

### Task 13: Mechanical composition test

**Files:**
- Create: `tools/sub7_validation/portfolio_composer.py`
- Create: `tests/sub7_validation/test_portfolio_composer.py`

ONLY proceed if 2+ detectors passed Phase 1.

- [ ] **Step 1: Write tests**

Create `tests/sub7_validation/test_portfolio_composer.py`:

```python
"""portfolio_composer tests (sub7-T13)."""
import pandas as pd
from tools.sub7_validation.portfolio_composer import compose_equal_weight, compose_risk_parity


def _setup_pnl(setup, n_days=10, daily_pnl=100):
    return pd.DataFrame({
        'session_date': pd.date_range('2024-01-01', periods=n_days*5, freq='5min').strftime('%Y-%m-%d')[:n_days*5],
        'setup_type': [setup] * (n_days*5),
        'net_pnl': [daily_pnl/5] * (n_days*5),
        'realized_pnl': [daily_pnl/5 + 30] * (n_days*5),
        'fee': [30] * (n_days*5),
    })


def test_equal_weight_combines_setups():
    a = _setup_pnl('a', daily_pnl=100)
    b = _setup_pnl('b', daily_pnl=200)
    out = compose_equal_weight({'a': a, 'b': b})
    # Should contain both setups
    assert set(out['setup_type'].unique()) == {'a', 'b'}
    # Total net = sum of components
    assert abs(out['net_pnl'].sum() - (a['net_pnl'].sum() + b['net_pnl'].sum())) < 0.01


def test_risk_parity_normalizes_by_vol():
    # a has lower volatility than b → should get HIGHER weight
    a = pd.DataFrame({
        'session_date': pd.date_range('2024-01-01', periods=20, freq='B').strftime('%Y-%m-%d'),
        'setup_type': ['a'] * 20, 'net_pnl': [100]*10 + [-50]*10,  # std=75 mean=25
        'realized_pnl': [100]*10 + [-50]*10, 'fee': [0]*20,
    })
    b = pd.DataFrame({
        'session_date': pd.date_range('2024-01-01', periods=20, freq='B').strftime('%Y-%m-%d'),
        'setup_type': ['b'] * 20, 'net_pnl': [500]*10 + [-450]*10,  # std=475 mean=25
        'realized_pnl': [500]*10 + [-450]*10, 'fee': [0]*20,
    })
    out = compose_risk_parity({'a': a, 'b': b})
    a_total = out[out['setup_type']=='a']['net_pnl'].abs().sum()
    b_total = out[out['setup_type']=='b']['net_pnl'].abs().sum()
    # a should contribute MORE total exposure (lower vol → higher weight)
    assert a_total > b_total
```

- [ ] **Step 2: Implement portfolio_composer.py**

Create `tools/sub7_validation/portfolio_composer.py`:

```python
"""Mechanical portfolio composition (sub7-T13).

NOT joint optimization. Just two recipes:
  - equal_weight: each setup contributes its raw net PnL
  - risk_parity: setups scaled to equalize daily volatility contribution

CLI:
    python tools/sub7_validation/portfolio_composer.py \\
        --setup-parquet-dir reports/sub7_validation/ \\
        --output-dir reports/sub7_validation/portfolio/ \\
        --period-start 2023-01-01 --period-end 2024-12-31
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from tools.sub7_validation.per_setup_report import compute_metrics


def compose_equal_weight(setups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate all setups' trades — equal weight by trade count."""
    return pd.concat(list(setups.values()), ignore_index=True)


def compose_risk_parity(setups: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Scale each setup's pnl by inverse of its daily-PnL std."""
    weighted_parts = []
    for name, df in setups.items():
        if df.empty:
            continue
        daily = df.groupby("session_date")["net_pnl"].sum()
        std = float(daily.std()) if len(daily) > 1 else 1.0
        if std <= 0:
            std = 1.0
        weight = 1.0 / std
        scaled = df.copy()
        scaled["net_pnl"] = scaled["net_pnl"] * weight
        scaled["realized_pnl"] = scaled["realized_pnl"] * weight
        scaled["fee"] = scaled["fee"] * weight
        weighted_parts.append(scaled)
    if not weighted_parts:
        return pd.DataFrame()
    return pd.concat(weighted_parts, ignore_index=True)


def evaluate_portfolio(combined: pd.DataFrame) -> dict:
    """Use existing per-setup metric computation."""
    return compute_metrics(combined)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--setup-parquet-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--period-start", default="2023-01-01")
    p.add_argument("--period-end", default="2024-12-31")
    args = p.parse_args()

    setup_dir = Path(args.setup_parquet_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setups = {}
    for f in sorted(setup_dir.glob("*.parquet")):
        df = pd.read_parquet(f)
        df = df[(df["session_date"] >= args.period_start) & (df["session_date"] <= args.period_end)]
        # Only include setups that pass Phase 1 (their report.json says so)
        report_f = setup_dir / f.stem / "01-metrics.json"
        if report_f.exists():
            r = json.loads(report_f.read_text())
            if r.get("phase1_passes"):
                setups[f.stem] = df
                print(f"Including {f.stem}: {len(df)} trades")
            else:
                print(f"Excluding {f.stem}: failed Phase 1")

    if len(setups) < 2:
        print("Less than 2 setups passed Phase 1. Composition not meaningful. Exiting.")
        return

    eq = compose_equal_weight(setups)
    rp = compose_risk_parity(setups)
    eq_m = evaluate_portfolio(eq)
    rp_m = evaluate_portfolio(rp)

    bar = {"net_pf_min": 1.25, "net_sharpe_min": 0.6,
           "max_dd_max_abs_pct": 20.0}
    # max_dd is in ₹; need pct vs total notional which we don't have. Use absolute Rs vs daily-avg-pnl as proxy.

    result = {
        "setups_included": sorted(setups.keys()),
        "period": {"start": args.period_start, "end": args.period_end},
        "equal_weight": eq_m,
        "risk_parity": rp_m,
        "phase2_pass_criteria": bar,
        "phase2_eq_passes": eq_m["net_pf"] >= bar["net_pf_min"] and eq_m["net_sharpe"] >= bar["net_sharpe_min"],
        "phase2_rp_passes": rp_m["net_pf"] >= bar["net_pf_min"] and rp_m["net_sharpe"] >= bar["net_sharpe_min"],
    }
    (out_dir / "06-portfolio.json").write_text(json.dumps(result, indent=2))

    print(f"\nEqual-weight: PF={eq_m['net_pf']}  Sharpe={eq_m['net_sharpe']}  net=Rs {int(eq_m['net_pnl']):,}")
    print(f"Risk-parity:  PF={rp_m['net_pf']}  Sharpe={rp_m['net_sharpe']}  net=Rs {int(rp_m['net_pnl']):,}")
    print(f"\nPhase 2 EQ pass: {result['phase2_eq_passes']}")
    print(f"Phase 2 RP pass: {result['phase2_rp_passes']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run tests**

```bash
.venv/Scripts/python.exe -m pytest tests/sub7_validation/test_portfolio_composer.py -v
```

Expected: 2 tests PASS.

- [ ] **Step 4: Run composition on real data**

```bash
.venv/Scripts/python.exe tools/sub7_validation/portfolio_composer.py \
  --setup-parquet-dir reports/sub7_validation/ \
  --output-dir reports/sub7_validation/portfolio/ \
  --period-start 2023-01-01 --period-end 2024-12-31
```

Expected: prints PF/Sharpe for both compositions, writes `06-portfolio.json`.

- [ ] **Step 5: Verify Phase 2 verdict**

If neither composition passes minimum (PF ≥ 1.25, Sharpe ≥ 0.6):
- ONE iteration allowed: try inverse-Sharpe weighting or Kelly sizing
- If still fails, KILL per design Q7

If at least one passes, proceed to Phase 4 (OOS).

- [ ] **Step 6: Commit**

```bash
git add -f tools/sub7_validation/portfolio_composer.py tests/sub7_validation/test_portfolio_composer.py reports/sub7_validation/portfolio/
git commit -m "feat(sub7-T13): portfolio composer + Phase 2 results"
```

---

## Phase 7: OOS validation + holdout (Tasks 14-15)

### Task 14: Validation OOS (Jan-Sep 2025)

**Files:** none modified — uses existing per_setup_report.py with different period

- [ ] **Step 1: Build per-setup parquet for Validation period**

The OCI capture from Task 11 already covers 2025-01-01 to 2026-03-31. Just re-run per-setup analysis with new period:

```bash
for setup in mis_unwind_short gap_fade_short cpr_mean_revert; do
  .venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py \
    --setup-parquet reports/sub7_validation/$setup.parquet \
    --output-dir reports/sub7_validation/$setup/_validation_oos/ \
    --period-start 2025-01-01 --period-end 2025-09-30
done
```

- [ ] **Step 2: Run portfolio composer for Validation period**

```bash
.venv/Scripts/python.exe tools/sub7_validation/portfolio_composer.py \
  --setup-parquet-dir reports/sub7_validation/ \
  --output-dir reports/sub7_validation/portfolio/_validation_oos/ \
  --period-start 2025-01-01 --period-end 2025-09-30
```

NOTE: composer currently checks Discovery's `01-metrics.json` for inclusion. That's correct — we want the SAME setup mix as Discovery (no peeking at OOS to choose setups).

- [ ] **Step 3: Pass/fail check against OOS bar**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
r = json.loads(Path('reports/sub7_validation/portfolio/_validation_oos/06-portfolio.json').read_text())
eq = r['equal_weight']
rp = r['risk_parity']
print('=== VALIDATION OOS (Jan-Sep 2025) ===')
print(f\"Equal-weight: PF={eq['net_pf']}  Sharpe={eq['net_sharpe']}  net=Rs {int(eq['net_pnl']):,}  LD={eq['losing_days_pct']}%\")
print(f\"Risk-parity:  PF={rp['net_pf']}  Sharpe={rp['net_sharpe']}  net=Rs {int(rp['net_pnl']):,}  LD={rp['losing_days_pct']}%\")
bar = {'pf_min': 1.15, 'sharpe_min': 0.5, 'dd_max_pct': 25.0}
eq_pass = eq['net_pf'] >= bar['pf_min'] and eq['net_sharpe'] >= bar['sharpe_min']
rp_pass = rp['net_pf'] >= bar['pf_min'] and rp['net_sharpe'] >= bar['sharpe_min']
print(f\"\\nValidation thresholds: PF>={bar['pf_min']}, Sharpe>={bar['sharpe_min']}\")
print(f\"Equal-weight: {'PASS' if eq_pass else 'FAIL'}\")
print(f\"Risk-parity:  {'PASS' if rp_pass else 'FAIL'}\")
print('\\nPick BETTER of (eq, rp) on Validation: this is the FROZEN portfolio for Holdout.')
"
```

- [ ] **Step 4: Commit results**

```bash
git add -f reports/sub7_validation/portfolio/_validation_oos/ reports/sub7_validation/*/_validation_oos/
git commit -m "report(sub7-T14): Validation OOS results"
```

If validation FAILS for both compositions → KILL sub-project per Q7. STOP HERE.

---

### Task 15: Holdout OOS (Oct 2025-Mar 2026) — final binary test

**Files:** none modified

This is the final one-shot test. Per master plan discipline, results inform pass/fail only — no further tuning permitted.

- [ ] **Step 1: Run per-setup + composer on Holdout period**

```bash
for setup in mis_unwind_short gap_fade_short cpr_mean_revert; do
  .venv/Scripts/python.exe tools/sub7_validation/per_setup_report.py \
    --setup-parquet reports/sub7_validation/$setup.parquet \
    --output-dir reports/sub7_validation/$setup/_holdout_oos/ \
    --period-start 2025-10-01 --period-end 2026-03-31
done

.venv/Scripts/python.exe tools/sub7_validation/portfolio_composer.py \
  --setup-parquet-dir reports/sub7_validation/ \
  --output-dir reports/sub7_validation/portfolio/_holdout_oos/ \
  --period-start 2025-10-01 --period-end 2026-03-31
```

- [ ] **Step 2: Pass/fail check (holdout thresholds)**

```bash
.venv/Scripts/python.exe -c "
import json
from pathlib import Path
r = json.loads(Path('reports/sub7_validation/portfolio/_holdout_oos/06-portfolio.json').read_text())
eq = r['equal_weight']
rp = r['risk_parity']
print('=== HOLDOUT OOS (Oct 2025-Mar 2026) ===')
print(f\"Equal-weight: PF={eq['net_pf']}  Sharpe={eq['net_sharpe']}  net=Rs {int(eq['net_pnl']):,}  LD={eq['losing_days_pct']}%\")
print(f\"Risk-parity:  PF={rp['net_pf']}  Sharpe={rp['net_sharpe']}  net=Rs {int(rp['net_pnl']):,}  LD={rp['losing_days_pct']}%\")
bar = {'pf_min': 1.15, 'sharpe_min': 0.5, 'ld_max_pct': 25.0}
eq_pass = eq['net_pf'] >= bar['pf_min'] and eq['net_sharpe'] >= bar['sharpe_min'] and eq['losing_days_pct'] <= bar['ld_max_pct']
rp_pass = rp['net_pf'] >= bar['pf_min'] and rp['net_sharpe'] >= bar['sharpe_min'] and rp['losing_days_pct'] <= bar['ld_max_pct']
print(f\"\\nHoldout thresholds: PF>={bar['pf_min']}, Sharpe>={bar['sharpe_min']}, LD<={bar['ld_max_pct']}%\")
print(f\"Equal-weight: {'PASS' if eq_pass else 'FAIL'}\")
print(f\"Risk-parity:  {'PASS' if rp_pass else 'FAIL'}\")
"
```

- [ ] **Step 3: Commit results + write summary**

```bash
git add -f reports/sub7_validation/portfolio/_holdout_oos/ reports/sub7_validation/*/_holdout_oos/
git commit -m "report(sub7-T15): Holdout OOS results — final"
```

- [ ] **Step 4: Document outcome**

Create `reports/sub7_validation/00-final-verdict.md` with:
- Phase 1 results per detector
- Phase 2 composition results
- Phase 4 Validation OOS results
- Phase 5 Holdout OOS results
- Final verdict: PASS / FAIL with rationale
- If PASS: hand off to deployment decision (sub-project #8)
- If FAIL: kill sub-project, document learnings for future iterations

```bash
git add -f reports/sub7_validation/00-final-verdict.md
git commit -m "docs(sub7-T15): final verdict — sub-project #7 outcome"
```

---

## Acceptance criteria summary

| Criterion | Measure |
|---|---|
| Phase 1 per-setup | NET PF ≥ 1.10, n ≥ 500, Net Sharpe > 0 (per detector) |
| Phase 2 composition | NET PF ≥ 1.25, Net Sharpe ≥ 0.6 on Discovery |
| Phase 4 validation | NET PF ≥ 1.15, Net Sharpe ≥ 0.5 on Jan-Sep 2025 |
| Phase 5 holdout | NET PF ≥ 1.15, Net Sharpe ≥ 0.5, LD ≤ 25% on Oct 2025-Mar 2026 |

## Self-review

- **Spec coverage:** Q1 stay intraday → Tasks 2-7 build intraday detectors. Q2 portfolio of independent → T9 per-setup analysis + T13 mechanical composition (no joint Optuna). Q3 success bars → encoded in T10 + T13 + T14 + T15 thresholds. Q4 three detectors → T2-7. Q5 bundled OCI → T11. Q6 OOS reservation → T14 + T15 use existing 2025+ captures. Q7 kill conditions → T12 step 3, T13 step 5, T14 step 3, T15 step 2.
- **Placeholder scan:** all steps include code or exact commands. No "TBD," no "implement appropriately." Detector default thresholds are configured (subject to refinement during T8 smoke testing).
- **Type consistency:** `setup_type` string identifiers consistent (`mis_unwind_short`, `gap_fade_short`, `cpr_mean_revert`) across config, detector files, main_detector wiring, and analysis code. `MarketContext`, `TradePlan`, `RiskParams`, `ExitLevels`, `StructureEvent`, `StructureAnalysis` all reference existing data_models.py types unchanged. `compute_metrics()` signature consistent across per_setup_report.py and portfolio_composer.py (latter imports the former).

## Cost summary

```
Active engineering: ~5-7 weeks
OCI cost:           1-2 bundled runs = $50-400
Total elapsed:      ~6-8 weeks
```

## Out-of-scope (deferred to follow-on)

- FII/DII flow signal ingestion (Phase 2 candidate)
- India VIX integration
- Sector rotation
- Circuit-limit detection
- F&O dynamics
- Live deployment (sub-project #8 if pursued)
