"""Circuit-Release-Fade SHORT detector - small/mid cap MIS-eligible SHORT mean-revert.

Discovered via tools/sub9_research/sanity_circuit_release_fade.py + the cell
sweep in tools/sub9_research/_circuit_release_fade_sweep_cellmine.py.

Aggregate validated across 3 windows (2026-05-16):
  Disc    PF=2.12  n=1,323  WR=58.5%  NET=+Rs.440K  (T1=1.0R / T2=2.0R, TS=15:10)
  OOS     PF=3.13  n=  493  WR=68.0%  NET=+Rs.226K
  Hold    PF=4.53  n=  223  WR=72.7%  NET=+Rs.119K

Thesis: Indian NSE small/mid-caps that pin upper circuit-band early (morning
retail FOMO surge) often see sellers appear mid-day, price drops 1-2% from
the pin, retail buyers re-engage and price RE-TESTS the day's high from below.
When the re-test FAILS (price can't break day high; new buyers exhausted), the
cascade-down of trapped FOMO buyers panic-selling produces a strong intraday
SHORT setup.

Distinct from circuit_t1_fade_short (active in prod) which fades T+1 next-day.
This is the SAME UNDERLYING EDGE captured INTRADAY rather than next-session.

Active window: 12:00-15:10 IST (after morning pin established, before MIS auto-square).
Multi-bar detection: any 5m bar in the active window that meets the failed-retest
signature triggers a SHORT entry (one fire per (symbol, session) - latched).

NO cell filter applied at detector level. Aggregate edge passes ship-gate on its own.
Cell-specific filters (day_gain_bucket, cap_segment, rejection_hhmm) can be added
post-OCI if aggregate fires too widely.

See specs/2026-05-16-new-setup-candidates.md -> C-03 PASSED SANITY for full details.
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
    """Bypass cell filters under wide_open_mode (OCI capture pattern)."""
    try:
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class CircuitReleaseFadeShortStructure(BaseStructure):
    """SHORT entry on failed re-test of morning circuit-pin in small/mid-cap names."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "circuit_release_fade_short"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])

        self.min_day_gain_pct = float(config["min_day_gain_pct"])
        self.morning_high_by_hhmm = self._parse_time(config["morning_high_by_hhmm"])
        self.morning_high_tolerance_pct = float(config["morning_high_tolerance_pct"])

        self.retest_tol_pct = float(config["retest_tol_pct"])
        self.rejection_close_pct = float(config["rejection_close_pct"])
        self.volume_confirm_lookback = int(config["volume_confirm_lookback"])

        self.allowed_caps = set(config["allowed_cap_segments"])

        self.sl_buffer_above_rejection_high_pct = float(config["sl_buffer_above_rejection_high_pct"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])

        self.min_bars_required = int(config["min_bars_required"])

        # First-trigger latch - one fire per (symbol, session_date).
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

        pdc = float(context.pdc) if context.pdc is not None else None
        if pdc is None or pdc <= 0:
            return _empty("PDC unavailable")

        # Warmup bars from prior sessions can pollute df.iloc; filter to today.
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return _empty("No bars for session date")

        # ----- Morning circuit-pin signature check -----
        # 1. Session high so far must be >= min_day_gain_pct above PDC (circuit-pin proxy)
        session_high_so_far = float(today_bars["high"].max())
        day_gain_pct = (session_high_so_far / pdc - 1.0) * 100.0
        if day_gain_pct < self.min_day_gain_pct:
            return _empty(f"day_gain={day_gain_pct:.2f}% < min={self.min_day_gain_pct}")

        # 2. The session high so far must have been reached BY morning_high_by_hhmm
        #    (i.e., no new high made after that time)
        morning_bars = today_bars[today_bars.index.time <= self.morning_high_by_hhmm]
        if morning_bars.empty:
            return _empty("No morning bars before cutoff")
        morning_high = float(morning_bars["high"].max())
        morning_floor = session_high_so_far * (1.0 - self.morning_high_tolerance_pct / 100.0)
        if morning_high < morning_floor:
            return _empty(
                f"morning_high={morning_high:.2f} < session_high*tol={morning_floor:.2f} "
                f"(new high made AFTER morning - not a morning pin)"
            )

        # ----- Failed re-test signal on current bar -----
        current_bar = today_bars.iloc[-1]
        bar_high = float(current_bar["high"])
        bar_close = float(current_bar["close"])
        bar_volume = float(current_bar["volume"])

        # Re-test: bar high reaches within retest_tol_pct of session high
        retest_threshold = session_high_so_far * (1.0 - self.retest_tol_pct / 100.0)
        if bar_high < retest_threshold:
            return _empty(
                f"bar_high={bar_high:.2f} < retest_threshold={retest_threshold:.2f} "
                f"(no re-test of day high)"
            )

        # Rejection: close meaningfully below bar high
        rejection_threshold = bar_high * (1.0 - self.rejection_close_pct / 100.0)
        if bar_close > rejection_threshold:
            return _empty(
                f"bar_close={bar_close:.2f} > rejection_threshold={rejection_threshold:.2f} "
                f"(no rejection wick)"
            )

        # Volume confirmation: bar.volume >= median of prior `lookback` bars' volume
        if len(today_bars) > self.volume_confirm_lookback + 1:
            lookback_bars = today_bars.iloc[-(self.volume_confirm_lookback + 1):-1]
            recent_vol_median = float(lookback_bars["volume"].median())
            if recent_vol_median > 0 and bar_volume < recent_vol_median:
                return _empty(
                    f"bar_volume={bar_volume:.0f} < recent_median={recent_vol_median:.0f}"
                )

        # ----- Signal passes - emit SHORT event -----
        confidence = min(1.0, (bar_high - bar_close) / max(bar_high - float(current_bar["low"]), 1e-9))

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "pdc": pdc,
                "session_high": session_high_so_far,
                "morning_high": morning_high,
                "rejection_high": bar_high,
                "rejection_close": bar_close,
            },
            context={
                "day_gain_pct": day_gain_pct,
                "retest_tol_pct": self.retest_tol_pct,
                "rejection_pct": (bar_high - bar_close) / bar_high * 100.0,
                "bar_volume": bar_volume,
            },
            price=bar_close,
        )
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ---------- abstract method implementations ----------

    def plan_long_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Short-only setup - no long trades."""
        return None

    def plan_short_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a SHORT TradePlan.

        Entry: close of the rejection bar (= current_price at detection).
        Stop: rejection_bar.high * (1 + sl_buffer_pct/100), capped by min_stop_pct.
        Targets: T1 = entry - t1_r_multiple*R (partial), T2 = entry - t2_r_multiple*R (remainder).
        Time stop: from config 'time_stop_at' (executor honors plan).
        """
        if event is None:
            return None
        evt = event
        if evt.side != "short":
            return None

        rejection_high = float(evt.levels.get("rejection_high", 0))
        entry = float(evt.price)
        if rejection_high <= 0 or entry <= 0:
            return None

        sl_from_high = rejection_high * (1.0 + self.sl_buffer_above_rejection_high_pct / 100.0)
        sl_from_min = entry * (1.0 + self.min_stop_pct / 100.0)
        # For SHORT, SL is ABOVE entry; pick the FARTHER (larger) of the two
        # (deeper SL = safer against immediate stop-out)
        hard_sl = max(sl_from_high, sl_from_min)
        if hard_sl <= entry:
            return None
        risk_per_share = hard_sl - entry

        t1_level = entry - self.t1_r_multiple * risk_per_share
        t2_level = entry - self.t2_r_multiple * risk_per_share
        rr_t1 = self.t1_r_multiple
        rr_t2 = self.t2_r_multiple

        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": rr_t1,
                "qty_pct": self.t1_partial_qty_pct,
                "action": "partial_exit",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": rr_t2,
                "qty_pct": round(1.0 - self.t1_partial_qty_pct, 4),
                "action": "exit_full",
            },
        ]

        risk_params = self.calculate_risk_params(entry, context)
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

        # Plan-geometry validation
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
                f"[{context.symbol}] circuit_release_fade_short plan rejected: {e.reason} {e.details}"
            )
            return None

        return TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=evt.structure_type,
            entry_price=entry,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=evt.confidence,
            notes=evt.context,
            trade_id=evt.trade_id,
            target_anchor_type="r_multiple",
        )

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext
    ) -> RiskParams:
        df = market_context.df_5m
        if df is None or df.empty:
            return RiskParams(
                hard_sl=entry_price * (1.0 + self.min_stop_pct / 100.0),
                risk_per_share=entry_price * (self.min_stop_pct / 100.0),
                atr=entry_price * 0.01,
            )
        session_date = market_context.session_date
        if session_date is None:
            session_date = pd.Timestamp(df.index[-1]).date()
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return RiskParams(
                hard_sl=entry_price * (1.0 + self.min_stop_pct / 100.0),
                risk_per_share=entry_price * (self.min_stop_pct / 100.0),
                atr=entry_price * 0.01,
            )
        # Use current (rejection) bar's high for SL anchor
        rejection_high = float(today_bars.iloc[-1]["high"])
        sl_from_high = rejection_high * (1.0 + self.sl_buffer_above_rejection_high_pct / 100.0)
        sl_from_min = entry_price * (1.0 + self.min_stop_pct / 100.0)
        hard_sl = max(sl_from_high, sl_from_min)
        stop_distance = max(hard_sl - entry_price, entry_price * 0.001)
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
        t1 = entry - self.t1_r_multiple * risk
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": self.t1_r_multiple}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> float:
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
