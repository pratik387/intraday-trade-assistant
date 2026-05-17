"""MIS-Unwind VWAP-Mean-Revert SHORT detector.

Discovered via tools/sub9_research/sanity_mis_unwind_vwap_revert.py (2026-05-16).
Aggregate ships - no cell filter required.

3-WINDOW VALIDATION:
  Discovery (2023-24)        n=41,068  PF=1.92  WR=51.9%  NET=+Rs.15.7M  Sharpe 4.66
  OOS       (2025 Q1-Q3)     n=20,665  PF=1.70  WR=50.7%  NET=+Rs. 6.4M  Sharpe 3.79
  Holdout   (2025-10 to 26-04) n=15,459 PF=1.60  WR=49.1%  NET=+Rs. 4.2M  Sharpe 3.33

  Standalone ship gate ALL PASS:
    Disc >= 1.30 [PASS 1.92] | OOS >= 1.20 [PASS 1.70] | Hold >= 1.15 [PASS 1.60]
    All samples >= 200; WR delta within 10pp across all windows.

THESIS: SEBI rules require MIS positions to be auto-squared-off by 15:20 IST,
with brokers typically starting forced liquidations from 15:00 onwards. Retail
long positions on small/mid-cap stocks trading materially ABOVE their session
VWAP (>=0.5%) with RSI overbought (>=65) AND with high recent volume (>=2x
session cumulative average) face concentrated forced-sell flow during this
window. The mean-reversion back toward VWAP is the captured edge.

CELL-MINE FINDING (Discovery, MONOTONIC mechanism gradient):
  - VWAP extension 0.5-0.7% (PF 1.52) -> 3.0+% (PF 2.36): more extension = stronger
  - RSI 65-70 (PF 1.55) -> 80+ (PF 2.44): more overbought = stronger
  - Vol ratio 2-2.5x (PF 1.43) -> 7+x (PF 2.34): more volume = stronger
  - Hour 14:30-14:45 (PF 1.55) -> 15:00-15:10 (PF 3.11): closer to MIS-square = stronger
  - small_cap PF 2.08 > mid_cap PF 1.75 (slight)

  Every dimension shows positive monotonic relationship with the MIS-unwind thesis.
  This is mechanism-validated cell structure - NOT data-mined.

  Aggregate already passes ship gate (no cell filter needed at launch). Cell-based
  size-tilt is a post-OCI optimization candidate.

Active window: 14:30-15:10 IST. One fire per (symbol, day) - latched.

Distinct from retired `mis_unwind_short` (sub-7, retired). Same THESIS, different
MECHANIC:
  - Retired version used VWAP-cross signal + momentum gate. Failed Phase-1.
  - This version uses VWAP-EXTENSION (price meaningfully ABOVE VWAP) + RSI overbought.
    Different signal geometry - exhaustion at extension vs cross signal.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.config_loader import load_base_config
from services.plan_helpers import (
    PlanRejected,
    assert_sl_outside_entry_zone,
    compute_entry_zone,
    enforce_min_stop_distance,
)
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
    try:
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class MisUnwindVwapRevertShortStructure(BaseStructure):
    """SHORT entry on VWAP-extension + RSI-overbought + volume-spike in MIS-unwind window."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "mis_unwind_vwap_revert_short"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])

        self.vwap_extension_pct = float(config["vwap_extension_pct"])
        self.rsi_overbought = float(config["rsi_overbought"])
        self.vol_ratio_min = float(config["vol_ratio_min"])

        self.allowed_caps = set(config["allowed_cap_segments"])

        self.sl_pct_above_entry = float(config["sl_pct_above_entry"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        # NOTE: t1_partial_qty_pct = 0 in config (T1 informational only).
        # Sanity used ride-to-T2 geometry (no T1 partial); detector mirrors that.
        # T1 retained as event metadata for monitoring.
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])

        self.min_bars_required = int(config["min_bars_required"])

        # One-fire-per-day latch.
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def detect(self, context: MarketContext) -> StructureAnalysis:
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

        _wide_open = _is_wide_open()

        if not _wide_open and context.cap_segment not in self.allowed_caps:
            return _empty(f"Cap segment {context.cap_segment!r} not in allowed set")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        session_date = (
            context.session_date
            if context.session_date is not None
            else pd.Timestamp(last_ts).date()
        )
        latch_key = (context.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return _empty("No bars for session date")

        # Need VWAP and RSI columns (5m_enriched feathers have them)
        if "vwap" not in today_bars.columns or "rsi" not in today_bars.columns:
            return _empty("VWAP/RSI columns missing")

        current_bar = today_bars.iloc[-1]
        vwap = current_bar.get("vwap")
        rsi = current_bar.get("rsi")
        if pd.isna(vwap) or pd.isna(rsi) or float(vwap) <= 0:
            return _empty("VWAP/RSI unavailable on signal bar")

        close_px = float(current_bar["close"])
        vwap_val = float(vwap)
        rsi_val = float(rsi)
        bar_volume = float(current_bar["volume"])

        # VWAP-extension check (price meaningfully ABOVE VWAP)
        ext_pct = (close_px / vwap_val - 1.0) * 100.0
        if ext_pct < self.vwap_extension_pct:
            return _empty(f"VWAP extension {ext_pct:.3f}% < min {self.vwap_extension_pct}%")

        # RSI overbought check
        if rsi_val < self.rsi_overbought:
            return _empty(f"RSI {rsi_val:.1f} < overbought threshold {self.rsi_overbought}")

        # Volume confirmation: bar volume >= vol_ratio_min * session-cumulative-mean
        prior_bars = today_bars.iloc[:-1]
        if len(prior_bars) < 2:
            return _empty("Not enough prior bars for volume baseline")
        cum_vol_mean = float(prior_bars["volume"].mean())
        if cum_vol_mean <= 0:
            return _empty("Volume baseline invalid")
        vol_ratio = bar_volume / cum_vol_mean
        if vol_ratio < self.vol_ratio_min:
            return _empty(f"vol_ratio {vol_ratio:.2f} < min {self.vol_ratio_min}")

        # Signal qualifies - emit SHORT event
        # Confidence proxy: higher VWAP extension + higher RSI = higher confidence
        ext_score = min(1.0, ext_pct / 2.0)  # normalize to 0-1 at 2% extension
        rsi_score = min(1.0, (rsi_val - 65) / 20.0)  # 65->0, 85->1
        confidence = (ext_score + rsi_score) / 2.0

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "vwap": vwap_val,
                "entry_price": close_px,
            },
            context={
                "vwap_ext_pct": ext_pct,
                "rsi": rsi_val,
                "vol_ratio": vol_ratio,
            },
            price=close_px,
        )
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    def plan_long_strategy(self, context: MarketContext, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        """SHORT-only - no long trades."""
        return None

    def plan_short_strategy(self, context: MarketContext, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        """Generate a SHORT TradePlan.

        Entry: signal bar close.
        SL: entry * (1 + sl_pct/100).
        T1: entry - 1R (informational only; qty_pct=0)
        T2: entry - 2R (FULL exit)
        Time stop: 15:10 IST.
        """
        if event is None or event.side != "short":
            return None

        entry = float(event.price)
        if entry <= 0:
            return None

        hard_sl = entry * (1.0 + self.sl_pct_above_entry / 100.0)
        # Enforce min_stop_pct floor
        sl_from_min = entry * (1.0 + self.min_stop_pct / 100.0)
        hard_sl = max(hard_sl, sl_from_min)
        risk_per_share = hard_sl - entry
        if risk_per_share <= 0:
            return None

        t1_level = entry - self.t1_r_multiple * risk_per_share
        t2_level = entry - self.t2_r_multiple * risk_per_share

        # Ride-to-T2 geometry (sanity tested this; T1 stored for monitoring).
        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": self.t1_r_multiple,
                "qty_pct": 0.0,
                "action": "exit_full",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": self.t2_r_multiple,
                "qty_pct": 1.0,
                "action": "exit_full",
            },
        ]

        risk_params = self.calculate_risk_params(entry, context)
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

        try:
            _zone = compute_entry_zone(
                entry=entry, bias="short",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, risk_params.hard_sl, "short")
            enforce_min_stop_distance(
                entry, risk_params.hard_sl,
                self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{context.symbol}] mis_unwind_vwap_revert_short plan rejected: {e.reason} {e.details}"
            )
            return None

        return TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=event.structure_type,
            entry_price=entry,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context,
            trade_id=event.trade_id,
            target_anchor_type="r_multiple",
        )

    def calculate_risk_params(self, entry_price: float, market_context: MarketContext) -> RiskParams:
        df = market_context.df_5m
        hard_sl = entry_price * (1.0 + self.sl_pct_above_entry / 100.0)
        sl_from_min = entry_price * (1.0 + self.min_stop_pct / 100.0)
        hard_sl = max(hard_sl, sl_from_min)
        stop_distance = max(hard_sl - entry_price, entry_price * 0.001)
        if df is None or df.empty:
            atr_proxy = entry_price * 0.01
        else:
            session_date = market_context.session_date
            if session_date is None:
                session_date = pd.Timestamp(df.index[-1]).date()
            _sd = session_date.date() if hasattr(session_date, "date") else session_date
            today_bars = df[df.index.date == _sd]
            if today_bars.empty:
                atr_proxy = entry_price * 0.01
            else:
                atr_proxy = float((today_bars["high"] - today_bars["low"]).mean())
        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=stop_distance,
            atr=atr_proxy,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        t2 = entry - self.t2_r_multiple * risk
        return ExitLevels(
            targets=[{"level": t2, "qty_pct": 100, "rr": self.t2_r_multiple}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(self, context: MarketContext, event: Optional[StructureEvent] = None) -> float:
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
