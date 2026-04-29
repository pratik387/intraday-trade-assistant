"""5 EMA Alert-Candle Pullback detector — sub-project #8.

Subasish Pani / Power of Stocks 5 EMA strategy per
specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 4).

Mechanic (LONG; mirror SHORT):
  In an established 5m UPTREND (5 EMA strictly increasing for the last
  `trend_lookback_bars` increments), the FIRST candle whose entire
  body+wick HIGH closes BELOW the 5 EMA is the "Alert Candle". When the
  next 5m candle's HIGH breaks the alert candle's HIGH, fire LONG with
  stop at alert candle's LOW and 1:3 R:R target. If the bar after the
  alert breaks the alert's LOW instead, abort (drop pending — the trap
  thesis is invalidated).

Subasish-specified entry-by-10:00-AM time gate. Active window 09:30-10:00.

Sources cited:
  - specs/2026-04-29-ema5_alert_pullback-plan.md
  - specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 4)
  - Subasish Pani / Power of Stocks (via TradersCarnival, MyAlgomate)
  - TradingQnA threads, Streak.tech / MyAlgomate published 5 EMA backtests
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional, Set, Tuple

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


class EMA5AlertPullbackStructure(BaseStructure):
    """5 EMA Alert-Candle Pullback (Subasish Pani named method).

    Three-state pipeline per side per session:
      1) Trend prerequisite — `ema_slope_positive` (default) or
         `price_above_ema` over `trend_lookback_bars` increments.
      2) Alert candle latch — first bar whose HIGH (long) / LOW (short)
         lies entirely beyond the 5 EMA in the counter-trend direction.
      3) Confirmation entry — when the bar AFTER the alert breaks alert's
         HIGH (long) or LOW (short). If it breaks the OPPOSITE extreme,
         the pending alert is dropped (trap thesis invalidated).

    First-trigger latch keyed on (symbol, side, session_date_iso); cleared
    at session boundary by _maybe_reset_state.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "ema5_alert_pullback"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config (KeyError on missing).
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.ema_period = int(config["ema_period"])
        self.trend_lookback_bars = int(config["trend_lookback_bars"])
        self.trend_definition = str(config["trend_definition"])
        self.target_rr = float(config["target_rr"])
        self.t1_rr = float(config["t1_rr"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.allowed_sides: Set[str] = set(config["allowed_sides"])
        self.allowed_caps: Set[str] = set(config["allowed_cap_segments"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"]) / 100.0

        # State containers — keyed by (symbol, side, session_date_iso).
        # _pending_alerts: alert candle bookkeeping while waiting for confirm.
        # _fired_today: latch — once fired, no second fire same session.
        self._pending_alerts: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._fired_today: Set[Tuple[str, str, str]] = set()
        self._latch_session_date: Optional[str] = None

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

    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Standard exponential moving average via pandas ewm."""
        return series.ewm(span=period, adjust=False).mean()

    def _maybe_reset_state(self, session_date_iso: str) -> None:
        """Clear pending alerts + fired_today if session boundary crossed."""
        if session_date_iso != self._latch_session_date:
            self._pending_alerts.clear()
            self._fired_today.clear()
            self._latch_session_date = session_date_iso

    def _check_trend_prerequisite(
        self,
        ema: pd.Series,
        closes: pd.Series,
        side: str,
    ) -> bool:
        """Return True if the trend prerequisite holds for this side.

        For `ema_slope_positive` (long): EMA strictly increasing for the
        last `trend_lookback_bars` increments BEFORE the current bar
        (i.e., excluding the current bar itself, which may be the alert).
        Mirror for short (strictly decreasing).

        For `price_above_ema` (long): every close > EMA in the window.
        Mirror for short (every close < EMA).
        """
        # Window: last trend_lookback_bars+1 values up to and INCLUDING the
        # bar before the current one (current bar may be the alert candle).
        # We need at least trend_lookback_bars + 1 prior bars.
        n_required = self.trend_lookback_bars + 1
        if len(ema) < n_required + 1:
            return False
        # Window excludes the current (last) bar.
        ema_win = ema.iloc[-(n_required + 1) : -1]
        close_win = closes.iloc[-(n_required + 1) : -1]
        if self.trend_definition == "ema_slope_positive":
            diffs = ema_win.diff().iloc[1:]   # drop the first NaN
            if side == "long":
                return bool((diffs > 0).all())
            else:
                return bool((diffs < 0).all())
        elif self.trend_definition == "price_above_ema":
            if side == "long":
                return bool((close_win > ema_win).all())
            else:
                return bool((close_win < ema_win).all())
        else:
            return False

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Three-state pipeline: trend → alert → confirm."""
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        # ---- Universe + cap_segment + bars + active window ----
        # Universe + cap_segment are design-inferred — bypassed under wide_open.
        if not _wide_open and not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")
        if not _wide_open and ctx.cap_segment not in self.allowed_caps:
            return _empty(f"cap_segment {ctx.cap_segment!r} not in allowed set")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- Latch session-boundary reset ----
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        self._maybe_reset_state(session_date_iso)

        # ---- Compute EMA over closes ----
        ema = self._compute_ema(df["close"], self.ema_period)
        ema_at_last = float(ema.iloc[-1])

        last = df.iloc[-1]
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])
        bar_open = float(last["open"])

        # ---- Iterate sides deterministically: short then long ----
        # (Order is arbitrary; deterministic reproducibility is what matters.)
        for side in ("long", "short"):
            if side not in self.allowed_sides:
                continue

            latch_key = (ctx.symbol, side, session_date_iso)
            if latch_key in self._fired_today:
                continue

            # ---- Step A: Confirmation check ----
            pending = self._pending_alerts.get(latch_key)
            if pending is not None:
                # Pending must be from a PRIOR bar (not the current one — that's
                # the bar we'd be confirming on). detect() may be called multiple
                # times per bar via sub7 fast path.
                pending_ts = pending.get("alert_bar_ts")
                if pending_ts is not None and pending_ts >= last_ts:
                    continue   # current bar IS the alert; wait for next bar

                alert_high = float(pending["alert_high"])
                alert_low = float(pending["alert_low"])
                ema_at_alert = float(pending["ema_at_alert"])

                fired = False
                aborted = False
                if side == "long":
                    if bar_high > alert_high:
                        fired = True
                    elif bar_low < alert_low:
                        aborted = True
                else:   # short
                    if bar_low < alert_low:
                        fired = True
                    elif bar_high > alert_high:
                        aborted = True

                if fired:
                    self._fired_today.add(latch_key)
                    self._pending_alerts.pop(latch_key, None)
                    evt = self._build_event(
                        ctx, side, alert_high, alert_low, ema_at_alert,
                        bar_close, ema_at_last, session_date_iso,
                    )
                    return StructureAnalysis(
                        structure_detected=True,
                        events=[evt],
                        quality_score=evt.confidence * 100.0,
                    )
                elif aborted:
                    # Drop the pending — the trap thesis is invalidated.
                    self._pending_alerts.pop(latch_key, None)
                    # Continue to allow trend-prereq + alert-latch check on this
                    # bar, in case the same bar is both an aborted-pending-target
                    # and a fresh alert candidate.
                # else: pending stays alive (neither fired nor aborted this bar)
                else:
                    continue

            # ---- Step B: Trend prerequisite ----
            if not self._check_trend_prerequisite(ema, df["close"], side):
                continue

            # ---- Step C: Alert candle latch ----
            # Long: bar's entire HIGH must be < EMA at this bar (strict
            # separation — body+wick fully below EMA). Mirror short: LOW > EMA.
            if side == "long":
                is_alert = bar_high < ema_at_last
            else:
                is_alert = bar_low > ema_at_last

            if is_alert and latch_key not in self._pending_alerts:
                self._pending_alerts[latch_key] = {
                    "alert_bar_ts": last_ts,
                    "alert_high": bar_high,
                    "alert_low": bar_low,
                    "alert_close": bar_close,
                    "alert_open": bar_open,
                    "ema_at_alert": ema_at_last,
                    "session_date_iso": session_date_iso,
                }

        return _empty("no alert+confirm pattern this bar")

    def _build_event(
        self,
        ctx: MarketContext,
        side: str,
        alert_high: float,
        alert_low: float,
        ema_at_alert: float,
        entry_close: float,
        ema_at_entry: float,
        session_date_iso: str,
    ) -> StructureEvent:
        """Construct StructureEvent for a confirmed alert+entry pair."""
        last_ts = ctx.df_5m.index[-1]
        atr = self._get_atr(ctx)
        # Confidence proxy: separation between alert's bar and EMA, normalized
        # by ATR. A larger separation = stronger pullback = higher conviction.
        if side == "long":
            sep = ema_at_alert - alert_high
        else:
            sep = alert_low - ema_at_alert
        confidence = min(1.0, max(0.0, sep / max(atr, 1e-6)))
        return StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "alert_high": alert_high,
                "alert_low": alert_low,
                "ema_at_alert": ema_at_alert,
                "entry_close": entry_close,
                "ema_at_entry": ema_at_entry,
            },
            context={
                "trend_definition": self.trend_definition,
                "trend_lookback_bars": self.trend_lookback_bars,
                "session_date_iso": session_date_iso,
            },
            price=entry_close,
        )

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        """Build a TradePlan from a confirmed alert+entry pair."""
        analysis = self.detect(ctx)
        if not analysis.structure_detected or not analysis.events:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        levels = evt.levels
        alert_high = float(levels["alert_high"])
        alert_low = float(levels["alert_low"])
        entry = float(evt.price)
        atr = self._get_atr(ctx)

        if side == "long":
            # Subasish stop: alert candle's LOW (with no buffer — Indian-source
            # standard for this method). Project floor: min_stop_distance_pct.
            hard_sl = alert_low
            risk = max(entry - hard_sl, atr * 0.1)
            min_risk = entry * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = entry - risk
            t1_level = entry + risk * self.t1_rr
            t2_level = entry + risk * self.target_rr
        else:   # short
            hard_sl = alert_high
            risk = max(hard_sl - entry, atr * 0.1)
            min_risk = entry * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = entry + risk
            t1_level = entry - risk * self.t1_rr
            t2_level = entry - risk * self.target_rr

        targets = [
            {
                "name": "T1", "level": t1_level, "rr": self.t1_rr,
                "qty_pct": self.t1_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_level, "rr": self.target_rr,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full",
            },
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=atr)
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        return TradePlan(
            symbol=ctx.symbol,
            side=side,
            structure_type=evt.structure_type,
            entry_price=entry,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=evt.confidence,
            notes=evt.context,
            trade_id=evt.trade_id,
        )

    def plan_long_strategy(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        if "long" not in self.allowed_sides:
            return None
        return self._build_plan(ctx, "long")

    def plan_short_strategy(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        if "short" not in self.allowed_sides:
            return None
        return self._build_plan(ctx, "short")

    def calculate_risk_params(
        self,
        entry_price: float,
        market_context: MarketContext,
    ) -> RiskParams:
        """Placeholder using ATR (real risk computed inside _build_plan from
        the alert candle's low/high)."""
        atr = self._get_atr(market_context)
        stop_distance = max(atr * 1.5, entry_price * self.min_stop_distance_pct)
        return RiskParams(
            hard_sl=entry_price + stop_distance,
            risk_per_share=stop_distance,
            atr=atr,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        return trade_plan.exit_levels

    def rank_setup_quality(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
