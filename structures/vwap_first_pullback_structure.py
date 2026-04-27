"""VWAP First-Pullback Continuation — sub8 Setup #3.

Source citations (Indian market):
  - Rupeezy — VWAP Trading Strategy
  - Choice India — VWAP Trading Strategy
  - BlinkX — Volume Weighted Average Price
  - Tradingshastra — VWAP Institutional Indicator 2025
  - Fyers — VWAP intraday strategy
  - JM Financial — Intraday Trading Time Analysis (rev2 active window)

Trigger: established trend (>= 4 of last 6 bars same side of VWAP) + pullback
bar that touches VWAP within 0.10% + reversal bar that closes back beyond VWAP
in trend direction with range >= 0.20% of price + reversal volume >= prior bar.

rev2: active window extended to 14:30 (from rev1's 13:30) to capture JM
Financial's afternoon golden window 13:30-15:00. Cut at 14:30 to leave Setup 5
(CHR) clear.

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

        # Established trend over `trend_lookback` bars (excluding pullback + reversal = -2)
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

        # Pullback bar (second-to-last): touches VWAP within proximity
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

        # Stop = pullback bar's low (long) or high (short) — Rupeezy verbatim
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

        # T1 = previous swing extreme; T2 = entry + 2R
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
