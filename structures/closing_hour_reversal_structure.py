"""Closing Hour Reversal (CHR) — sub8 Setup #5.

Source citations (Indian market):
  - Zerodha — MIS Auto Square-Off Timings (15:25 hard stop)
  - Zerodha Varsity — Volatility Applications (1.5x ATR for EOD stops)
  - Goodwill — Using ATR for Smart Stop-Losses
  - Subhadip Nandy / Capitalmind on EOD reversion (VWAP magnet)
  - StockGro — Intraday Closing Time (bidirectional sharp moves)

Trigger: stock has moved >= 1.5% in one direction between 09:30 and 14:30.
At 14:30+, exhaustion candle prints (body >= 60% of range, vol >= 1.3x recent).
Direction: short if move was UP, long if move was DOWN. Hard time stop 15:22.

rev2 vs sub7 mis_unwind_short:
  - Bidirectional (sub7 was short-only)
  - Stop multiplier 1.5x ATR (sub7 used 0.8x — too tight, killed sample)
  - Hard time stop 15:22 (sub7 had 15:18 — too conservative; Zerodha auto-square at 15:25)

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


def _is_wide_open() -> bool:
    """Read top-level wide_open_mode flag from base config."""
    try:
        from pipelines.base_pipeline import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


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

        # rev2: design-inferred filters bypass under wide_open_mode
        _wide_open = _is_wide_open()

        # Compute intraday move from session bars 09:30-14:30
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

        if up_move_pct >= self.min_intraday_move_pct and up_move_pct > down_move_pct:
            side = "short"  # fade the up move
        elif down_move_pct >= self.min_intraday_move_pct and down_move_pct > up_move_pct:
            side = "long"   # fade the down move
        else:
            return _empty(f"intraday move too small: up={up_move_pct:.2f} down={down_move_pct:.2f}")

        # Exhaustion candle: body >= 60% of range, opposite direction, volume confirmed
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
        if not _wide_open and body_pct < self.exhaustion_min_body_pct:
            return _empty(f"body_pct={body_pct:.2f} < min={self.exhaustion_min_body_pct}")

        if side == "short" and bar_close >= bar_open:
            return _empty("short signal but bar is bullish")
        if side == "long" and bar_close <= bar_open:
            return _empty("long signal but bar is bearish")

        recent_vol = float(df["volume"].iloc[-6:-1].mean()) if len(df) >= 6 else bar_vol
        if not _wide_open and recent_vol > 0 and bar_vol < self.exhaustion_min_vol_x * recent_vol:
            return _empty(f"volume {bar_vol:.0f} < {self.exhaustion_min_vol_x}x recent {recent_vol:.0f}")

        confidence = min(1.0, body_pct)
        evt = StructureEvent(
            symbol=ctx.symbol, timestamp=last_ts, structure_type=self.structure_type,
            side=side, confidence=confidence,
            levels={"session_high": session_high, "session_low": session_low,
                    "session_open": session_open, "close": bar_close,
                    "vwap": self._get_vwap(ctx) or bar_close},
            context={"up_move_pct": up_move_pct, "down_move_pct": down_move_pct,
                     "body_pct": body_pct,
                     "vol_x_recent": bar_vol / max(recent_vol, 1.0)},
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

        # rev2: stop = recent intraday extreme + 1.5x ATR (Indian-source standard)
        if side == "short":
            hard_sl = session_high + self.stop_atr_mult * atr
            risk = max(hard_sl - close, 1e-6)
        else:
            hard_sl = session_low - self.stop_atr_mult * atr
            risk = max(close - hard_sl, 1e-6)

        # T1 = VWAP
        # T2 = pivot or 50% retrace (whichever lies between t1 and entry on the right side)
        pivot = (float(ctx.pdh) + float(ctx.pdl) + float(ctx.pdc)) / 3.0 \
            if all(v is not None for v in (ctx.pdh, ctx.pdl, ctx.pdc)) else session_open
        retrace_50 = (session_high + session_low) / 2.0

        if side == "short":
            t1_level = vwap if vwap < close else close - risk
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
