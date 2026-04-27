"""Narrow CPR Trending Day Breakout — sub8 Setup #2.

Source citations (Indian market):
  - Frank Ochoa "Secret of Pivot Boss" via Shubham Agarwal (Quantsapp)
  - Optionx Journal — CPR Explained with NSE Examples
  - Jainam — CPR in Trading
  - Tradingdirection.in — CPR Brahmastra

Trigger: narrow CPR width < 0.40% precedes expansion. Bar closing OUTSIDE
[BC, TC] with 1.3x median volume = trending breakout. Stop = pivot, T1 = R1
or S1, T2 = R2 or S2.

Universe: Nifty 50 + Bank Nifty (CPR is index/heavyweight tool, not small caps).
This fixes sub7 cpr_mean_revert's universe-mismatch failure.

NOTE: This is the OPPOSITE direction of sub7's cpr_mean_revert. Sub7 faded
TC/BC rejections (mean reversion); this setup trades WITH the breakout.
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
            return _empty(f"volume {bar_vol:.0f} < {self.min_vol_x}x median {median_vol:.0f}")

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
