"""PDH/PDL Sweep+Reclaim detector — sub-project #8 (replaces pdh_pdl_reject).

Indian-pro mechanic per
specs/2026-04-28-research-pdh_pdl_reject-indian-pro-mechanics.md (IPM).

The dropped pdh_pdl_reject detector used a "first-touch + rejection wick"
mechanic which Indian SMC sources unanimously identified as mechanically
wrong. The canonical entry is **sweep + reclaim**: price first PENETRATES
the level (a liquidity-grab / stop-hunt above PDH or below PDL), THEN closes
back inside the prior session range — the sweep candle. The next bar then
confirms by closing further beyond the sweep candle's opposite extreme.
Fade the false breakout.

PDH-fade-short (mirror for PDL-fade-long):
  bar t-1: high > pdh*(1+pen) AND close < pdh*(1-rec)  → SWEEP latches
  bar t:   close < pdh AND close < bar(t-1).low        → FIRE short

Configurable mechanics (per IPM):
  M1 — sweep+reclaim core trigger (always on; this is the mechanic)
  M2 — multi-day confluence: only latch if today's PDH is within
       multi_day_confluence_pct of any of last `multi_day_lookback` daily
       highs (mirror PDL for longs). Default OFF (Phase 2 enhancement).
  M5 — POC gating (Phase 2; not in this implementation).
  M8 — gap-context side selector: on gap-up days (open vs PDC > threshold)
       prefer PDH-fade-short; gap-down prefer PDL-fade-long; flat-open allow
       both. Default ON.

State machine: pending sweep candles are stored on the instance keyed by
(symbol, side, session_date) until the next bar confirms or the session
rolls over. First-trigger latch keyed on the same tuple prevents
double-fire within a session.

Wide-open mode bypass: when wide_open_mode is True, design-inferred filters
(multi-day confluence, gap-context selector) are skipped. Trigger geometry
(sweep penetration / reclaim / confirm continuation) STAYS active — those
are mechanical conditions, not quality filters.

Sources cited:
  - specs/2026-04-29-pdh_pdl_sweep_reclaim-plan.md
  - specs/2026-04-28-research-pdh_pdl_reject-findings.md (FIND)
  - specs/2026-04-28-research-pdh_pdl_reject-indian-pro-mechanics.md (IPM)
  - Indian-pro corpus: Sahi.com, DailyPriceAction (sweep+reclaim);
    Vtrender / WavesStrategy (POC, deferred); ICT-India community.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, List, Optional, Set, Tuple

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


class PDHPDLSweepReclaimStructure(BaseStructure):
    """Sweep+reclaim fade at PDH (short) or PDL (long).

    Two-bar state machine: pending sweep candles wait on the instance until
    the next bar confirms (fire) or aborts (drop pending). First-trigger
    latch prevents double-fire same (symbol, side, session_date).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "pdh_pdl_sweep_reclaim"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config (KeyError on missing).
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.sweep_pen_pct = float(config["sweep_penetration_pct"]) / 100.0
        self.reclaim_buf_pct = float(config["reclaim_buffer_pct"]) / 100.0
        self.confirm_close_beyond_sweep_low = bool(config["confirm_close_beyond_sweep_low"])

        self.multi_day_enabled = bool(config["multi_day_confluence_enabled"])
        self.multi_day_pct = float(config["multi_day_confluence_pct"]) / 100.0
        self.multi_day_lookback = int(config["multi_day_lookback"])

        self.gap_context_enabled = bool(config["gap_context_enabled"])
        self.gap_threshold_pct = float(config["gap_threshold_pct"]) / 100.0

        self.allowed_sides: Set[str] = set(config["allowed_sides"])
        self.allowed_caps: Set[str] = set(config["allowed_cap_segments"])
        self.universe_key = str(config["universe_key"])

        self.stop_atr_buffer = float(config["stop_atr_buffer"])
        self.wick_buf_pct = float(config["wick_buffer_pct"]) / 100.0
        self.t1_target = str(config["t1_target"])
        self.t2_target = str(config["t2_target"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.min_bars_required = int(config["min_bars_required"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"]) / 100.0

        # State machine: pending sweeps keyed by (symbol, side, session_date_iso).
        # Value carries sweep extreme/close/bar_low/bar_high so the confirmation
        # check on the next bar can verify continuation.
        self._pending_sweeps: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        # First-trigger latch: set of (symbol, side, session_date_iso) tuples
        # that have already fired today. Cleared at session boundary.
        self._fired_today: Set[Tuple[str, str, str]] = set()
        # Cheap session-boundary detection: when this differs from the
        # current detect()'s session_date_iso, both maps get cleared.
        self._latch_session_date: Optional[str] = None

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        """Extract ATR with fallback (mirror gap_fade_short)."""
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            df = ctx.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_vwap(self, ctx: MarketContext) -> Optional[float]:
        """Extract VWAP with fallback to df_5m['vwap'] last value."""
        if ctx.indicators and "vwap" in ctx.indicators:
            try:
                return float(ctx.indicators["vwap"])
            except (TypeError, ValueError):
                pass
        if ctx.df_5m is not None and "vwap" in ctx.df_5m.columns:
            v = ctx.df_5m["vwap"].iloc[-1]
            if pd.notna(v):
                return float(v)
        return None

    def _maybe_reset_state(self, session_date_iso: str) -> None:
        """Clear pending sweeps + fired_today if session boundary crossed."""
        if session_date_iso != self._latch_session_date:
            self._pending_sweeps.clear()
            self._fired_today.clear()
            self._latch_session_date = session_date_iso

    def _check_multi_day_confluence(self, ctx: MarketContext, side: str) -> bool:
        """Return True if multi-day confluence holds (or is disabled).

        M2 per IPM§3: a sweep at PDH is only meaningful if PDH itself
        coincides with a recent daily high (within multi_day_confluence_pct).
        Single-day-extreme sweeps are weak signals.

        Returns True if `multi_day_enabled` is False (no-op gate).
        Returns False if df_daily is unavailable or has insufficient history
        (fail-closed — don't latch on uncertain confluence).
        """
        if not self.multi_day_enabled:
            return True
        daily = ctx.df_daily
        if daily is None or len(daily) < self.multi_day_lookback:
            return False
        today = pd.Timestamp(ctx.session_date).normalize()
        # Filter to dates strictly before today (df_daily indexed by date)
        try:
            prior = daily[daily.index < today].tail(self.multi_day_lookback)
        except TypeError:
            prior = daily.tail(self.multi_day_lookback)
        if len(prior) < self.multi_day_lookback:
            return False
        if side == "short":
            level = float(ctx.pdh) if ctx.pdh is not None else 0.0
            if level <= 0:
                return False
            for h in prior["high"].astype(float):
                if abs(h - level) / level <= self.multi_day_pct:
                    return True
            return False
        else:  # long
            level = float(ctx.pdl) if ctx.pdl is not None else 0.0
            if level <= 0:
                return False
            for low in prior["low"].astype(float):
                if abs(low - level) / level <= self.multi_day_pct:
                    return True
            return False

    # ---------- abstract method stubs (filled in subsequent tasks) ----------

    def _build_event(
        self,
        ctx: MarketContext,
        side: str,
        sweep: Dict[str, Any],
        bar_close: float,
    ) -> StructureEvent:
        """Construct StructureEvent for a confirmed sweep+reclaim+confirm.

        Levels populated:
          pdh / pdl: prior-day extremes (always present)
          sweep_extreme: the sweep bar's high (short) or low (long)
          sweep_close: the sweep bar's close
          sweep_bar_low / sweep_bar_high: the sweep bar's full extent
          close: the confirm bar's close
          vwap: current VWAP for T1 reference (None-safe)
        """
        last_ts = ctx.df_5m.index[-1]
        pdh = float(ctx.pdh) if ctx.pdh is not None else 0.0
        pdl = float(ctx.pdl) if ctx.pdl is not None else 0.0
        vwap = self._get_vwap(ctx)
        # Confidence: ratio of sweep depth vs ATR proxy. Larger sweep = higher
        # confidence the trap was a real institutional liquidity grab.
        atr = self._get_atr(ctx)
        if side == "short":
            sweep_depth = float(sweep["sweep_extreme"]) - pdh
        else:
            sweep_depth = pdl - float(sweep["sweep_extreme"])
        confidence = min(1.0, max(0.0, sweep_depth / max(atr, 1e-6)))
        return StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "pdh": pdh,
                "pdl": pdl,
                "sweep_extreme": float(sweep["sweep_extreme"]),
                "sweep_close": float(sweep["sweep_close"]),
                "sweep_bar_low": float(sweep["sweep_bar_low"]),
                "sweep_bar_high": float(sweep["sweep_bar_high"]),
                "close": float(bar_close),
                "vwap": float(vwap) if vwap is not None else float(bar_close),
            },
            context={
                "sweep_bar_ts": str(sweep.get("sweep_bar_ts")),
                "session_date_iso": sweep.get("session_date_iso"),
            },
            price=float(bar_close),
        )

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Sweep+reclaim+confirm state machine.

        Three-bar canonical sequence (mirror for long-side off PDL):
          bar t-1: high > pdh*(1+pen) AND close < pdh*(1-rec)   → latch
          bar t:   close < pdh AND close < bar(t-1).low          → fire short

        Per-bar logic:
          1) Cheap rejections (universe, cap_segment, bars, active window).
          2) Reset state at session boundary.
          3) Step A — Confirmation check: if a pending sweep exists for
             (sym, side, session_date) and was set on a bar BEFORE the
             current one, check if THIS bar confirms. Whether it does or
             not, the pending has had its only chance — drop it.
          4) Step B — Sweep latching: if THIS bar is a sweep candle for a
             non-fired side, latch a fresh pending entry.
          5) Return _empty if neither fired nor latched.
        """
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        # ---- Universe + cap_segment + bars + active window ----
        if not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        if ctx.cap_segment not in self.allowed_caps:
            return _empty(f"cap_segment {ctx.cap_segment!r} not in allowed set")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- PDH/PDL availability ----
        if ctx.pdh is None or ctx.pdl is None:
            return _empty("PDH/PDL unavailable")
        pdh = float(ctx.pdh)
        pdl = float(ctx.pdl)

        # ---- Latch session-boundary reset ----
        # session_date is an IST-naive datetime; convert to ISO date string
        # so the latch key is hashable + stable across detect() re-runs.
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        self._maybe_reset_state(session_date_iso)

        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])

        # ---- Determine sides allowed today ----
        # M8 (gap-context side selector, IPM§3): on gap-up days continuation
        # is the prevailing flow, so PDH-fade-shorts are statistically more
        # likely to work than PDL-fade-longs (and vice-versa for gap-down).
        # On flat-open days, allow both sides.
        # Skipped under wide_open_mode — design-inferred filter, not trigger
        # geometry.
        allowed_sides_today = set(self.allowed_sides)
        _wide_open = _is_wide_open()
        if self.gap_context_enabled and not _wide_open and ctx.pdc and float(ctx.pdc) > 0:
            open_today = float(df.iloc[0]["open"])
            gap_frac = (open_today - float(ctx.pdc)) / float(ctx.pdc)
            if gap_frac > self.gap_threshold_pct:
                allowed_sides_today.discard("long")    # gap-up → drop long
            elif gap_frac < -self.gap_threshold_pct:
                allowed_sides_today.discard("short")   # gap-down → drop short
            # else: |gap| within threshold → flat-open, both sides allowed

        # ---- Step A: Confirmation check ----
        # Iterate sides deterministically. If a pending sweep exists from a
        # PRIOR bar, this current bar is the only chance to confirm.
        for side in ("short", "long"):
            if side not in allowed_sides_today:
                continue
            latch_key = (ctx.symbol, side, session_date_iso)
            if latch_key in self._fired_today:
                continue
            pending = self._pending_sweeps.get(latch_key)
            if pending is None:
                continue
            # Pending must be from an earlier bar — same-bar pending is the
            # current sweep we just latched in a prior detect() call within
            # the same bar (sub7 fast path may re-run detect()). Skip in that
            # case so we don't fire on the latch bar itself.
            sweep_ts = pending.get("sweep_bar_ts")
            if sweep_ts is not None and sweep_ts >= last_ts:
                continue

            # Check confirmation continuation
            sweep_low = float(pending["sweep_bar_low"])
            sweep_high = float(pending["sweep_bar_high"])
            fires = False
            if side == "short":
                # Confirm: close < pdh*(1-rec) AND (if required) close < sweep_low
                cond_close = bar_close < pdh * (1.0 - self.reclaim_buf_pct)
                cond_continuation = (
                    bar_close < sweep_low if self.confirm_close_beyond_sweep_low else True
                )
                fires = cond_close and cond_continuation
            else:  # long
                cond_close = bar_close > pdl * (1.0 + self.reclaim_buf_pct)
                cond_continuation = (
                    bar_close > sweep_high if self.confirm_close_beyond_sweep_low else True
                )
                fires = cond_close and cond_continuation

            # Either way, pending has had its chance — drop it.
            self._pending_sweeps.pop(latch_key, None)

            if fires:
                self._fired_today.add(latch_key)
                evt = self._build_event(ctx, side, pending, bar_close)
                return StructureAnalysis(
                    structure_detected=True,
                    events=[evt],
                    quality_score=evt.confidence * 100.0,
                )

        # ---- Step B: Sweep latching ----
        # Detect if THIS bar is a sweep candle for a non-fired side.

        # Short-side sweep: high penetrates above PDH, close back inside the
        # prior session range (close <= pdh*(1-rec)). Body color is
        # intentionally not required — a doji/small-body bar with rejection
        # wick is the canonical trap candle.
        # Multi-day confluence gate (M2) is bypassed under wide_open_mode.
        pen_short = bar_high > pdh * (1.0 + self.sweep_pen_pct)
        rec_short = bar_close <= pdh * (1.0 - self.reclaim_buf_pct)
        if "short" in allowed_sides_today and pen_short and rec_short:
            confluence_ok = _wide_open or self._check_multi_day_confluence(ctx, "short")
            if confluence_ok:
                latch_key = (ctx.symbol, "short", session_date_iso)
                if latch_key not in self._fired_today and latch_key not in self._pending_sweeps:
                    self._pending_sweeps[latch_key] = {
                        "sweep_bar_ts": last_ts,
                        "sweep_extreme": bar_high,
                        "sweep_close": bar_close,
                        "sweep_bar_low": bar_low,
                        "sweep_bar_high": bar_high,
                        "session_date_iso": session_date_iso,
                    }

        # Long-side sweep: low penetrates below PDL, close back inside.
        pen_long = bar_low < pdl * (1.0 - self.sweep_pen_pct)
        rec_long = bar_close >= pdl * (1.0 + self.reclaim_buf_pct)
        if "long" in allowed_sides_today and pen_long and rec_long:
            confluence_ok = _wide_open or self._check_multi_day_confluence(ctx, "long")
            if confluence_ok:
                latch_key = (ctx.symbol, "long", session_date_iso)
                if latch_key not in self._fired_today and latch_key not in self._pending_sweeps:
                    self._pending_sweeps[latch_key] = {
                        "sweep_bar_ts": last_ts,
                        "sweep_extreme": bar_low,
                        "sweep_close": bar_close,
                        "sweep_bar_low": bar_low,
                        "sweep_bar_high": bar_high,
                        "session_date_iso": session_date_iso,
                    }

        return _empty("no sweep+reclaim+confirm pattern this bar")

    def _compute_two_day_50pct_retrace(
        self,
        ctx: MarketContext,
        side: str,
        close: float,
        risk: float,
    ) -> float:
        """T2 target: midpoint of last 2 days' high-low range.

        Falls back to 2R from entry if df_daily missing or midpoint is
        wrong-direction relative to entry (e.g., midpoint above entry for
        short trade).
        """
        daily = ctx.df_daily
        if daily is not None and len(daily) >= 2:
            tail = daily.tail(2)
            try:
                hi = float(tail["high"].max())
                lo = float(tail["low"].min())
                mid = (hi + lo) / 2.0
                if side == "short" and mid < close:
                    return mid
                if side == "long" and mid > close:
                    return mid
            except (KeyError, ValueError, TypeError):
                pass
        # Fallback: 2R from entry
        return close - 2.0 * risk if side == "short" else close + 2.0 * risk

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        """Build a TradePlan after a confirmed sweep+reclaim+confirm fire."""
        analysis = self.detect(ctx)
        if not analysis.structure_detected or not analysis.events:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        levels = evt.levels
        sweep_extreme = float(levels["sweep_extreme"])
        close = float(ctx.df_5m["close"].iloc[-1])
        atr = self._get_atr(ctx)
        wick_buf = close * self.wick_buf_pct

        # Stop placement: sweep_extreme + ATR*buffer + wick_buf for short
        # (sweep_high is the extreme penetration above PDH; stop above it).
        if side == "short":
            hard_sl = sweep_extreme + atr * self.stop_atr_buffer + wick_buf
            risk = max(hard_sl - close, atr * 0.1)
            # Enforce min_stop_distance_pct (widen hard_sl if too tight)
            min_risk = close * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = close + risk
        else:  # long
            hard_sl = sweep_extreme - atr * self.stop_atr_buffer - wick_buf
            risk = max(close - hard_sl, atr * 0.1)
            min_risk = close * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = close - risk

        # T1 target: VWAP retest (if direction-correct, else 1R fallback).
        vwap = self._get_vwap(ctx)
        if vwap is not None:
            if side == "short" and vwap < close:
                t1_level = float(vwap)
            elif side == "long" and vwap > close:
                t1_level = float(vwap)
            else:
                t1_level = close - risk if side == "short" else close + risk
        else:
            t1_level = close - risk if side == "short" else close + risk

        # T2 target: 2-day-range 50% retrace, fallback 2R.
        t2_level = self._compute_two_day_50pct_retrace(ctx, side, close, risk)
        # Ensure T2 is beyond T1 (more aggressive direction)
        if side == "short":
            if t2_level >= t1_level:
                t2_level = close - 2.0 * risk
        else:
            if t2_level <= t1_level:
                t2_level = close + 2.0 * risk

        rr_t1 = abs(close - t1_level) / max(risk, 1e-6)
        rr_t2 = abs(close - t2_level) / max(risk, 1e-6)
        targets = [
            {
                "name": "T1", "level": t1_level, "rr": rr_t1,
                "qty_pct": self.t1_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_level, "rr": rr_t2,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full",
            },
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=atr)
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        return TradePlan(
            symbol=ctx.symbol,
            side=side,
            structure_type=evt.structure_type,
            entry_price=close,
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
        """Placeholder used by sub7 fast-path before plan_*_strategy fires.

        Real risk params are computed inside _build_plan once the sweep
        extreme is known. This helper returns a conservative ATR-based
        approximation for upstream sizing checks.
        """
        atr = self._get_atr(market_context)
        # Conservative stop: 1.5×ATR away from entry (direction-agnostic
        # placeholder; the real stop is placed at sweep_extreme + buffer).
        stop_distance = max(atr * 1.5, entry_price * self.min_stop_distance_pct)
        return RiskParams(
            hard_sl=entry_price + stop_distance,    # caller may flip sign
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
