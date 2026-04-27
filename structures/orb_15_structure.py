"""ORB-15 (Opening Range Breakout, first 15 minutes) — sub8 Setup #1.

Source citations (Indian market):
  - In-the-Money by Zerodha — All About ORB Part 1 (09:15-11:15 window)
  - Algotest — sample range breakout (full-bar close beyond range, volume confirm)
  - Saimohanreddy — ORB backtest on Bank Nifty 2015-2023 (wick buffer essential)

Trigger: 09:20-09:30 forms the opening range (rev2: skips 09:15-09:20 pre-open
call-auction wick). Within 09:30-11:15, the FIRST 5-min bar that closes outside
the range, on >= 1.5× 30-day median volume for that clock slot, fires a
directional ORB entry.

Universe: F&O liquid 200 (per design doc Section 3.2).
Stop: opposite-end-of-range (rev2 default; mid-range A/B variant) ± wick_buffer.
Targets: T1 at 1R (50% qty), T2 at 2R (50% qty).
Gap-day exclusion: if open gap from PDC > max_gap_pct_for_orb, route to gap_fade_short.
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


def _is_wide_open() -> bool:
    """Read top-level wide_open_mode flag from base config."""
    try:
        from pipelines.base_pipeline import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


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
        # rev2: gap-day cross-detector exclusion. AlgoTest, gwcindia, truedata
        # flag gap days as a different playbook than ORB.
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

        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # rev2: design-inferred filters bypass under wide_open_mode
        _wide_open = _is_wide_open()

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

        # rev2: gap-day exclusion. ORB and gap_fade have contradictory thesis on
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
        if not _wide_open and median_vol > 0 and bar_vol < self.min_vol_x * median_vol:
            return _empty(f"volume {bar_vol:.0f} < {self.min_vol_x}× median {median_vol:.0f}")

        if bar_close > range_high:
            side = "long"
        elif bar_close < range_low:
            side = "short"
        else:
            return _empty(f"close {bar_close:.2f} inside range [{range_low:.2f},{range_high:.2f}]")

        confidence = min(1.0, range_pct / 1.0)
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

        # rev2: default = opposite-end-of-range (Indian-source standard).
        # stop_at_range_midpoint=true is A/B variant.
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
