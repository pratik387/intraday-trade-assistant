"""CPR Mean Revert detector — sub-project #7, Task 6.

Thesis: During the midday lunch lull (11:30-13:30 IST), low-volume ranging
conditions cause price to oscillate around the Central Pivot Range (CPR). When
price extends >= 1 ATR from CPR midpoint on low volume with a reversal candle,
the path of least resistance is mean-reversion back to CPR mid. Both long and
short directions are supported.

Active window: 11:30-13:30 IST (lunch lull, low-volume range-trading).
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


class CPRMeanRevertStructure(BaseStructure):
    """Detects bidirectional mean-reversion setups around CPR midpoint during lunch lull."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "cpr_mean_revert"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_distance_atr = float(config["min_distance_atr_from_cpr"])
        self.max_volume_pct = float(config["max_volume_pct_of_intraday_avg"])
        self.require_reversion_candle = bool(config["require_reversion_candle"])
        self.reversion_patterns = list(config["reversion_patterns"])
        self.allowed_caps = set(config.get("allowed_cap_segments", []))
        self.stop_atr_buffer = float(config["stop_at_extreme_atr_buffer"])
        self.target_type = str(config["target_type"])
        self.min_bars_required = int(config["min_bars_required"])
        self.min_cpr_width_pct = float(config.get("min_cpr_width_pct", 0.0))

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, context: MarketContext) -> float:
        """Extract ATR from indicators dict with fallback."""
        if context.indicators and "atr" in context.indicators:
            return float(context.indicators["atr"])
        if context.df_5m is not None and len(context.df_5m) >= 14:
            df = context.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return context.current_price * 0.01

    @staticmethod
    def _compute_cpr(pdh, pdl, pdc) -> Optional[tuple]:
        """Standard CPR formula (Frank Ochoa / Zerodha Varsity reference).

        Pivot = (H + L + C) / 3
        BC    = (H + L) / 2
        TC    = 2 * Pivot - BC

        TC and BC can be in either order — always normalize via max/min.

        Returns (cpr_top, cpr_bottom, cpr_mid) or None if inputs are None.
        """
        if pdh is None or pdl is None or pdc is None:
            return None
        pivot = (pdh + pdl + pdc) / 3.0
        bc = (pdh + pdl) / 2.0
        tc = 2.0 * pivot - bc
        cpr_top = max(tc, bc)
        cpr_bottom = min(tc, bc)
        cpr_mid = pivot  # = (cpr_top + cpr_bottom) / 2 by construction
        return cpr_top, cpr_bottom, cpr_mid

    def _get_cpr_levels(self, context: MarketContext) -> Optional[tuple]:
        """Compute CPR levels from previous-day pdh/pdl/pdc on MarketContext.

        Production MarketContext provides pdh, pdl, pdc as direct fields (set in
        main_detector.py). We compute CPR from those rather than from indicators
        (which only populate 'vol_z' and 'atr' in production).

        Returns (cpr_top, cpr_bottom, cpr_mid) or None if unavailable.
        """
        pdh = getattr(context, 'pdh', None)
        pdl = getattr(context, 'pdl', None)
        pdc = getattr(context, 'pdc', None)
        return self._compute_cpr(pdh, pdl, pdc)

    @staticmethod
    def _candle_pattern(bar) -> str:
        """Classify the bar's candle pattern.

        Returns one of: "doji", "hammer", "shooting_star", "none".
        """
        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])
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

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect CPR mean revert setups in the 11:30-13:30 window.

        All conditions must be true:
        1. Current bar is within active_window_start..active_window_end
        2. Cap segment is in allowed_cap_segments
        3. CPR_TOP and CPR_BOTTOM levels are available
        4. abs(close - cpr_mid) / atr >= min_distance_atr_from_cpr
        5. current bar volume / mean(intraday volume excl. last bar) <= max_volume_pct / 100
        6. (If require_reversion_candle) current bar candle pattern is in reversion_patterns
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
            return _empty(f"Cap segment {context.cap_segment!r} not in allowed set")

        # --- Condition 1: Active time-of-day window ---
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # --- Condition 3: CPR levels available ---
        cpr_result = self._get_cpr_levels(context)
        if cpr_result is None:
            return _empty("CPR levels (CPR_TOP/CPR_BOTTOM) unavailable")
        cpr_top, cpr_bot, cpr_mid = cpr_result

        # --- CPR width filter (research: Wright Research, Groww) ---
        # Mean reversion only works on wide-CPR days (CPR width > X% of price).
        # Narrow CPR = trending day → fade fails systematically.
        if self.min_cpr_width_pct > 0:
            cpr_width_pct = ((cpr_top - cpr_bot) / max(cpr_mid, 1e-6)) * 100.0
            if cpr_width_pct < self.min_cpr_width_pct:
                return _empty(f"cpr_too_narrow:{cpr_width_pct:.2f}%<{self.min_cpr_width_pct}%")

        last = df.iloc[-1]
        close = float(last["close"])
        atr = self._get_atr(context)

        # --- Condition 4: Distance from CPR midpoint ---
        if atr <= 0:
            return _empty("ATR is zero or negative")
        dist_atr = abs(close - cpr_mid) / atr
        if dist_atr < self.min_distance_atr:
            return _empty(
                f"distance from CPR mid={dist_atr:.2f} ATR < min={self.min_distance_atr}"
            )

        # --- Condition 5: Volume filter ---
        if len(df) < 2:
            return _empty("Need at least 2 bars for volume comparison")
        intraday_vols = df["volume"].iloc[:-1]
        if len(intraday_vols) == 0:
            return _empty("No prior bars for volume average")
        mean_vol = float(intraday_vols.mean())
        cur_vol = float(last["volume"])
        if mean_vol > 0:
            vol_ratio_pct = (cur_vol / mean_vol) * 100.0
            if vol_ratio_pct > self.max_volume_pct:
                return _empty(
                    f"volume ratio={vol_ratio_pct:.1f}% > max={self.max_volume_pct}% (high volume)"
                )

        # --- Condition 6: Reversion candle pattern ---
        if self.require_reversion_candle:
            pattern = self._candle_pattern(last)
            if pattern not in self.reversion_patterns:
                return _empty(
                    f"Candle pattern={pattern!r} not in reversion_patterns={self.reversion_patterns}"
                )
        else:
            pattern = self._candle_pattern(last)

        # --- Determine bias direction ---
        bias = "short" if close > cpr_mid else "long"
        side = bias

        # Confidence: distance from CPR scaled (more distance = stronger revert setup)
        confidence = min(1.0, dist_atr / (self.min_distance_atr * 2.0))

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "cpr_top": cpr_top,
                "cpr_bottom": cpr_bot,
                "cpr_mid": cpr_mid,
                "close": close,
            },
            context={
                "bias": bias,
                "dist_atr": dist_atr,
                "candle_pattern": pattern,
                "cur_vol": cur_vol,
                "mean_vol": mean_vol,
                "vol_ratio_pct": (cur_vol / mean_vol * 100.0) if mean_vol > 0 else 0.0,
            },
            price=close,
        )
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ---------- Shared plan builder ----------

    def _build_plan(self, context: MarketContext, bias: str) -> Optional[TradePlan]:
        """Build a TradePlan for the given bias ("long" or "short").

        Returns None if detect() fails or the detected bias doesn't match.
        """
        analysis = self.detect(context)
        if not analysis.structure_detected:
            return None

        evt = analysis.events[0]
        if evt.context.get("bias") != bias:
            return None

        df = context.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        atr = self._get_atr(context)
        cpr_mid = evt.levels["cpr_mid"]

        if bias == "short":
            # Stop: ABOVE current bar high + ATR buffer
            hard_sl = bar_high + atr * self.stop_atr_buffer
            risk_per_share = hard_sl - close
            target_level = cpr_mid
            # Ensure target is below entry for short
            if target_level >= close:
                target_level = close - risk_per_share  # fallback: 1R below entry
        else:  # long
            # Stop: BELOW current bar low - ATR buffer
            hard_sl = bar_low - atr * self.stop_atr_buffer
            risk_per_share = close - hard_sl
            target_level = cpr_mid
            # Ensure target is above entry for long
            if target_level <= close:
                target_level = close + risk_per_share  # fallback: 1R above entry

        rr = abs(close - target_level) / max(abs(risk_per_share), 1e-6)
        targets = [
            {
                "name": "T1",
                "level": target_level,
                "rr": rr,
                "qty_pct": 1.0,
                "action": "exit_full",
            }
        ]

        risk_params = RiskParams(
            hard_sl=hard_sl,
            risk_per_share=risk_per_share,
            atr=atr,
        )
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        notes = dict(evt.context)  # includes bias, dist_atr, candle_pattern, etc.

        return TradePlan(
            symbol=context.symbol,
            side=bias,
            structure_type=self.structure_type,
            entry_price=close,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,       # Pipeline overrides with proper sizing
            notional=0.0,
            confidence=evt.confidence,
            notes=notes,
        )

    # ---------- Required abstract method implementations ----------

    def plan_long_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a TradePlan for a long mean-reversion back to CPR midpoint."""
        return self._build_plan(context, bias="long")

    def plan_short_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a TradePlan for a short mean-reversion back to CPR midpoint."""
        return self._build_plan(context, bias="short")

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext
    ) -> RiskParams:
        """Compute risk params. Direction-agnostic stub — used by pipeline sizing."""
        atr = self._get_atr(market_context)
        stop_distance = atr * self.stop_atr_buffer
        return RiskParams(
            hard_sl=entry_price + stop_distance,  # placeholder direction
            risk_per_share=stop_distance,
            atr=atr,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """Return exit levels derived from the trade plan's existing risk params."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        if trade_plan.side == "short":
            t1 = entry - risk
        else:
            t1 = entry + risk
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": 1.0}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> float:
        """Proxy: re-run detect and return quality_score."""
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """Return True if current_time falls within the active window."""
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
