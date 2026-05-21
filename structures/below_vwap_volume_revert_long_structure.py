"""Below-VWAP volume-revert LONG detector — paper-trade phase.

Phase 5 SHIPPABLE per cell-sweep (2026-05-21). 3D cell locked to:
  cap_segment=unknown × vol_ratio_bin=gte_10 × hhmm_bucket=afternoon_1300_1500
Disc PF 1.587 / OOS 1.782 / HO 1.606 (3+ years, n=3,712 pooled).

Spec:      specs/2026-05-21-below_vwap_volume_revert_long-paper-trade-spec.md
Brief:     specs/2026-05-21-brief-below_vwap_volume_revert_long.md
Lock JSON: tools/sub9_research/below_vwap_volume_revert_long_cell_lock.json

DO NOT enable in live (`enabled: true`) until paper-trade phase acceptance
gates pass — see spec Section "Paper-trade acceptance gates".
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
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


class BelowVwapVolumeRevertLongStructure(BaseStructure):
    """LONG entry on >=2% below-VWAP bars with elevated volume in afternoon."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "below_vwap_volume_revert_long"
        self.configured_setup_type = config.get("_setup_name")

        # All params from config — NO hardcoded defaults (project Rule #1)
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.vwap_dev_pct_max = float(config["vwap_dev_pct_max"])
        self.vol_ratio_min = float(config["vol_ratio_min"])
        self.cell_cap_segment = str(config["cell_lock_cap_segment"])
        self.cell_vol_ratio_min = float(config["cell_lock_vol_ratio_min"])
        self.cell_hhmm_min = self._parse_time(config["cell_lock_hhmm_min"])
        self.cell_hhmm_max = self._parse_time(config["cell_lock_hhmm_max"])
        self.min_signal_bar_notional_rs = float(config["min_signal_bar_notional_rs"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.sl_buffer_below_bar_low_pct = float(config["sl_buffer_below_bar_low_pct"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])
        self.min_bars_required = int(config["min_bars_required"])

        # First-fire-per-(symbol, date) latch
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def detect(self, context: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False, events=[], quality_score=0.0,
                rejection_reason=reason or None,
            )

        df = context.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        session_date = (
            context.session_date if context.session_date is not None
            else pd.Timestamp(last_ts).date()
        )
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return _empty("No bars for session date")

        # Session VWAP up through current (last) bar — cumulative since 09:15.
        pv = (today_bars["close"].astype("float64") * today_bars["volume"].astype("float64")).cumsum()
        vol_cum = today_bars["volume"].astype("float64").cumsum()
        if float(vol_cum.iloc[-1]) <= 0:
            return _empty("Zero session volume")
        session_vwap = float(pv.iloc[-1] / vol_cum.iloc[-1])
        last_bar = today_bars.iloc[-1]
        bar_close = float(last_bar["close"])
        vwap_dev_pct = (bar_close - session_vwap) / session_vwap * 100.0

        if vwap_dev_pct > self.vwap_dev_pct_max:
            return _empty(
                f"vwap_dev_pct={vwap_dev_pct:.3f} > max={self.vwap_dev_pct_max}"
            )

        # Cell-lock hhmm: 13:00 <= bar_hhmm <= 14:55
        if not (self.cell_hhmm_min <= cur_t <= self.cell_hhmm_max):
            return _empty(
                f"cell_hhmm window: {cur_t} not in "
                f"[{self.cell_hhmm_min}, {self.cell_hhmm_max}]"
            )

        # Cell-lock cap_segment
        ctx_cap = (context.cap_segment or "").strip()
        if ctx_cap != self.cell_cap_segment:
            return _empty(
                f"cap_segment={ctx_cap!r} != cell_lock={self.cell_cap_segment!r}"
            )

        return _empty("not_implemented_beyond_cap_segment")

    def plan_long_strategy(self, context, event=None):
        return None

    def plan_short_strategy(self, context, event=None):
        return None

    def calculate_risk_params(self, entry_price, market_context):
        return RiskParams(
            hard_sl=entry_price * (1.0 - self.min_stop_pct / 100.0),
            risk_per_share=entry_price * (self.min_stop_pct / 100.0),
            atr=entry_price * (self.min_stop_pct / 100.0),
        )

    def get_exit_levels(self, trade_plan):
        return ExitLevels(hard_sl=trade_plan.risk_params.hard_sl, targets=[])

    def rank_setup_quality(self, context, event=None):
        return 0.0

    def validate_timing(self, current_time):
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
