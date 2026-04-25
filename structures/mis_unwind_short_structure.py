"""MIS Unwind Short detector — sub-project #7.

Thesis: SEBI requires MIS positions to square off by 3:20 PM. Retail intraday
flow is structurally net-long. The forced unwind in the last 60-90 minutes
creates asymmetric net-sell pressure. Pros short into this.

Active window: 14:55-15:15 IST.
"""
from __future__ import annotations
from datetime import time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from .base_structure import BaseStructure
from .data_models import (
    MarketContext, StructureAnalysis, StructureEvent,
    TradePlan, RiskParams, ExitLevels,
)

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

    def _get_rvol(self, context: MarketContext) -> float:
        """Extract rvol from indicators dict (stored there since MarketContext has no rvol field)."""
        if context.indicators and "rvol" in context.indicators:
            return float(context.indicators["rvol"])
        return 0.0

    def _get_atr(self, context: MarketContext) -> float:
        """Extract ATR from indicators dict with fallback."""
        if context.indicators and "atr" in context.indicators:
            return float(context.indicators["atr"])
        if context.df_5m is not None and len(context.df_5m) >= 14:
            df = context.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return context.current_price * 0.01

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect MIS unwind short setups in the 14:55-15:15 window.

        Six conditions must ALL be true:
        1. Current 5m bar is within active_window_start..active_window_end
        2. Cap segment is in allowed_cap_segments
        3. Close is above VWAP by >= min_distance_above_vwap_pct
        4. RVOL >= min_rvol
        5. Intraday high was made within the last min_intraday_high_recency_min minutes
        6. 3-bar momentum is <= max_momentum_3bar_pct (negative — weakening)
        """
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        df = context.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        # --- Condition 2: Cap segment ---
        if context.cap_segment not in self.allowed_caps:
            return _empty(f"Cap segment {context.cap_segment} not in allowed set")

        # --- Condition 1: Active time-of-day window ---
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        last = df.iloc[-1]
        vwap = float(last.get("vwap", 0.0))
        close = float(last.get("close", 0.0))

        if vwap <= 0:
            return _empty("VWAP unavailable")

        # --- Condition 3: Price above VWAP ---
        dist_vwap_pct = ((close - vwap) / vwap) * 100.0
        if dist_vwap_pct < self.min_dist_vwap_pct:
            return _empty(f"dist_vwap_pct={dist_vwap_pct:.2f} < {self.min_dist_vwap_pct}")

        # --- Condition 4: RVOL ---
        rvol = self._get_rvol(context)
        if rvol < self.min_rvol:
            return _empty(f"rvol={rvol:.2f} < {self.min_rvol}")

        # --- Condition 5: Fresh intraday high within recency window ---
        recent_window_bars = max(1, self.min_high_recency_min // 5)
        recent_highs = df["high"].iloc[-recent_window_bars:]
        intraday_max = df["high"].max()
        if recent_highs.max() < intraday_max - 1e-9:
            return _empty("Intraday high not in recent window")

        # --- Condition 6: 3-bar momentum is negative (weakening) ---
        if len(df) < 4:
            return _empty("Need at least 4 bars for momentum calculation")
        c_now = float(df["close"].iloc[-1])
        c_3ago = float(df["close"].iloc[-4])
        if c_3ago <= 0:
            return _empty("c_3ago is zero/negative")
        mom_3 = ((c_now - c_3ago) / c_3ago) * 100.0
        if mom_3 > self.max_momentum_3bar_pct:
            return _empty(f"momentum_3bar_pct={mom_3:.2f} > {self.max_momentum_3bar_pct}")

        # All conditions met — build event
        confidence = min(1.0, abs(mom_3) / 1.0)  # scales with momentum decay strength
        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={"vwap": vwap, "close": close},
            context={
                "dist_vwap_pct": dist_vwap_pct,
                "momentum_3bar_pct": mom_3,
                "rvol": rvol,
            },
            price=close,
        )
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ---------- Required abstract method implementations ----------

    def plan_long_strategy(self, context: MarketContext,
                           event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        """Short-only setup — no long trades."""
        return None

    def plan_short_strategy(self, context: MarketContext,
                            event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        """Placeholder — implemented in Task 3."""
        return None

    def calculate_risk_params(self, context: MarketContext,
                              event: Optional[StructureEvent] = None,
                              side: str = "short") -> RiskParams:
        """Stop above current price by ATR * stop_atr_buffer (short trade)."""
        entry_price = context.current_price
        atr = self._get_atr(context)
        hard_sl = entry_price + atr * self.stop_atr_buffer
        risk_per_share = atr * self.stop_atr_buffer
        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """VWAP-based target for short (Task 3 will flesh this out fully)."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        t1 = entry - risk  # 1R target
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": 1.0}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(self, context: MarketContext,
                           event: Optional[StructureEvent] = None) -> float:
        """Proxy: re-run detect and return quality_score."""
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """Return True if current_time falls within the active window."""
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
