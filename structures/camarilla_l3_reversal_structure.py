"""Camarilla L3/H3 Reversal detector — sub-project #8 (Phase 0 2026-04-29).

Statistical mean-revert at narrow-band Camarilla pivot levels per
specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 1).

Indian sources cited:
  - market-pulse.in: "H3 and L3 are the levels to go against the trend
    with stop loss around H4 or L4"
  - Pivottrading.co.in (Indian Camarilla calculator) — formula source
  - Bhaskar Das (Medium) — Indian intraday on 15m chart
  - Jainam, Fisdom, OptionX — retail Indian Camarilla guidance

CANONICAL Camarilla pivot formulas (NOT Floor Pivot — distinct system):
    Range = PDH - PDL
    P     = (PDH + PDL + PDC) / 3
    H3    = PDC + 0.275 * Range   (= 1.1 / 4)
    L3    = PDC - 0.275 * Range
    H4    = PDC + 0.55  * Range   (= 1.1 / 2)
    L4    = PDC - 0.55  * Range

Mechanic (long; mirror short):
  Two-bar state machine on the L3 level.
    bar t-1 (sweep candle): low penetrates L3 by sweep_penetration_pct AND
      close back above L3 by reclaim_buffer_pct (close >= L3*(1+rec_buf)).
    bar t (confirmation):   close > L3*(1+rec_buf) AND > pending.sweep_close
    -> fire LONG
  Stop at L4 (no extra ATR buffer — L4 IS the wide stop per Camarilla
  literature). T1 at pivot P, T2 at opposite-side L3-or-H3 (long: T2 = H3;
  short: T2 = L3).

Filters:
  - Active window 10:00-14:00 (skip first 30 min for level visibility, skip
    last 90 min for closing volatility).
  - Regime gate: ADX(14) on 5m < max_adx_for_revert (default 25).
    Camarilla literature explicit on trending days running through L3 to L4.
  - Mutual exclusion vs pdh_pdl_sweep_reclaim: when |L3-PDL|/PDL <= 0.3%
    (or |H3-PDH|/PDH <= 0.3% for short side), cede to that detector to
    avoid double-trade on the same level.

Wide-open mode bypass:
  - Bypassed (design-inferred): ADX regime gate, mutual-exclusion check,
    universe + cap_segment filters
  - Always enforced (mechanical): active window, min_bars, sweep+reclaim
    geometry (penetration ≥ pen_pct, reclaim ≥ rec_buf), confirmation bar
    direction
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional, Set, Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from services.symbol_metadata import in_universe
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
        from services.config_loader import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


def _camarilla_levels(pdh: float, pdl: float, pdc: float) -> Dict[str, float]:
    """Compute canonical Camarilla pivot levels.

    H3 = PDC + 0.275 * Range  (= 1.1 / 4)
    L3 = PDC - 0.275 * Range
    H4 = PDC + 0.55  * Range  (= 1.1 / 2)
    L4 = PDC - 0.55  * Range
    P  = (PDH + PDL + PDC) / 3
    """
    rng = pdh - pdl
    p = (pdh + pdl + pdc) / 3.0
    h3 = pdc + 0.275 * rng
    l3 = pdc - 0.275 * rng
    h4 = pdc + 0.55 * rng
    l4 = pdc - 0.55 * rng
    return {"P": p, "H3": h3, "L3": l3, "H4": h4, "L4": l4}


class CamarillaL3ReversalStructure(BaseStructure):
    """Sweep+reclaim+confirm at Camarilla L3 (long) / H3 (short).

    Two-bar state machine (mirror pdh_pdl_sweep_reclaim pattern). Pending
    sweeps stored on the instance keyed by (symbol, side, session_date_iso)
    until the next bar confirms (fire) or aborts (drop pending — only one
    chance). First-trigger latch keyed identically prevents double-fire.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "camarilla_l3_reversal"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config (KeyError on missing).
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.sweep_pen_pct = float(config["sweep_penetration_pct"]) / 100.0
        self.reclaim_buf_pct = float(config["reclaim_buffer_pct"]) / 100.0
        self.max_adx = float(config["max_adx_for_revert"])
        self.pdh_pdl_skip_pct = float(config["pdh_pdl_proximity_skip_pct"]) / 100.0
        self.t1_target = str(config["t1_target"])
        self.t2_target = str(config["t2_target"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.stop_atr_buf = float(config["stop_atr_buffer"])
        self.wick_buf_pct = float(config["wick_buffer_pct"]) / 100.0
        self.allowed_sides: Set[str] = set(config["allowed_sides"])
        self.allowed_caps: Set[str] = set(config["allowed_cap_segments"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"]) / 100.0

        # State machine.
        self._pending_sweeps: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
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

    def _get_vwap(self, ctx: MarketContext) -> Optional[float]:
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

    def _get_adx(self, ctx: MarketContext) -> Optional[float]:
        """Read ADX(14) on 5m chart from indicators or df['adx'] last value."""
        if ctx.indicators and "adx" in ctx.indicators:
            try:
                v = float(ctx.indicators["adx"])
                if pd.notna(v):
                    return v
            except (TypeError, ValueError):
                pass
        if ctx.df_5m is not None and "adx" in ctx.df_5m.columns:
            v = ctx.df_5m["adx"].iloc[-1]
            if pd.notna(v):
                return float(v)
        return None

    def _maybe_reset_state(self, session_date_iso: str) -> None:
        if session_date_iso != self._latch_session_date:
            self._pending_sweeps.clear()
            self._fired_today.clear()
            self._latch_session_date = session_date_iso

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        # ---- Universe + cap_segment + bars + active window ----
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

        # ---- PDH/PDL/PDC availability ----
        if ctx.pdh is None or ctx.pdl is None or ctx.pdc is None:
            return _empty("PDH/PDL/PDC unavailable")
        pdh = float(ctx.pdh)
        pdl = float(ctx.pdl)
        pdc = float(ctx.pdc)
        if pdh <= 0 or pdl <= 0 or pdc <= 0 or pdh <= pdl:
            return _empty("PDH/PDL/PDC invalid (non-positive or PDH<=PDL)")

        # ---- Compute Camarilla levels ----
        levels = _camarilla_levels(pdh, pdl, pdc)
        l3 = levels["L3"]
        h3 = levels["H3"]
        l4 = levels["L4"]
        h4 = levels["H4"]

        # ---- Latch session-boundary reset ----
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        self._maybe_reset_state(session_date_iso)

        # ---- Determine sides allowed today ----
        # Mutual-exclusion vs pdh_pdl_sweep_reclaim: when our level is too
        # close to a PDH/PDL level, cede to the other detector.
        # Bypassed under wide_open (design-inferred filter).
        allowed_sides_today = set(self.allowed_sides)
        if not _wide_open:
            if "long" in allowed_sides_today and pdl > 0:
                if abs(l3 - pdl) / pdl <= self.pdh_pdl_skip_pct:
                    allowed_sides_today.discard("long")
            if "short" in allowed_sides_today and pdh > 0:
                if abs(h3 - pdh) / pdh <= self.pdh_pdl_skip_pct:
                    allowed_sides_today.discard("short")

        # ---- Regime gate: ADX < max_adx_for_revert (skipped under wide_open) ----
        if not _wide_open:
            adx = self._get_adx(ctx)
            if adx is None:
                return _empty("ADX unavailable")
            if adx >= self.max_adx:
                return _empty(f"adx={adx:.2f} >= max_adx_for_revert={self.max_adx}")

        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])

        # ---- Step A: Confirmation check ----
        # Per side, if a pending sweep exists from a PRIOR bar, this current
        # bar is the only chance to confirm. Either fires or aborts (drops
        # pending). Mirror pdh_pdl_sweep_reclaim's logic.
        for side in ("short", "long"):
            if side not in allowed_sides_today:
                continue
            latch_key = (ctx.symbol, side, session_date_iso)
            if latch_key in self._fired_today:
                continue
            pending = self._pending_sweeps.get(latch_key)
            if pending is None:
                continue
            sweep_ts = pending.get("sweep_bar_ts")
            if sweep_ts is not None and sweep_ts >= last_ts:
                continue   # current bar IS the sweep we just latched

            sweep_bar_low = float(pending["sweep_bar_low"])
            sweep_bar_high = float(pending["sweep_bar_high"])
            sweep_close = float(pending["sweep_close"])
            fires = False
            if side == "long":
                # Confirm: close > L3*(1+rec_buf) AND close > pending.sweep_close
                cond_close = bar_close > l3 * (1.0 + self.reclaim_buf_pct)
                cond_continuation = bar_close > sweep_close
                fires = cond_close and cond_continuation
            else:   # short — confirm at H3
                cond_close = bar_close < h3 * (1.0 - self.reclaim_buf_pct)
                cond_continuation = bar_close < sweep_close
                fires = cond_close and cond_continuation

            # Either way, pending has had its chance — drop it.
            self._pending_sweeps.pop(latch_key, None)

            if fires:
                # NOTE: latch NOT added here. detect() must remain
                # idempotent. _build_plan commits the latch on success.
                evt = self._build_event(
                    ctx, side, levels, pending, bar_close, session_date_iso,
                )
                return StructureAnalysis(
                    structure_detected=True,
                    events=[evt],
                    quality_score=evt.confidence * 100.0,
                )

        # ---- Step B: Sweep latching ----
        # Long-side L3 sweep: bar low penetrates below L3, close back above
        # L3 by reclaim_buf, bar bullish (close > open).
        pen_long = bar_low < l3 * (1.0 - self.sweep_pen_pct)
        rec_long = bar_close >= l3 * (1.0 + self.reclaim_buf_pct)
        if "long" in allowed_sides_today and pen_long and rec_long and bar_close > bar_open:
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

        # Short-side H3 sweep: bar high penetrates above H3, close back below
        # H3 by reclaim_buf, bar bearish (close < open).
        pen_short = bar_high > h3 * (1.0 + self.sweep_pen_pct)
        rec_short = bar_close <= h3 * (1.0 - self.reclaim_buf_pct)
        if "short" in allowed_sides_today and pen_short and rec_short and bar_close < bar_open:
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

        return _empty("no sweep+reclaim+confirm pattern this bar")

    def _build_event(
        self,
        ctx: MarketContext,
        side: str,
        levels: Dict[str, float],
        sweep: Dict[str, Any],
        bar_close: float,
        session_date_iso: str,
    ) -> StructureEvent:
        last_ts = ctx.df_5m.index[-1]
        atr = self._get_atr(ctx)
        # Confidence proxy: sweep depth (penetration distance) normalized by
        # the L3-L4 distance (or H3-H4 for short). A deeper trap = stronger
        # institutional liquidity grab signal.
        if side == "long":
            sweep_depth = float(levels["L3"]) - float(sweep["sweep_extreme"])
            band_width = float(levels["L3"]) - float(levels["L4"])
        else:
            sweep_depth = float(sweep["sweep_extreme"]) - float(levels["H3"])
            band_width = float(levels["H4"]) - float(levels["H3"])
        confidence = min(1.0, max(0.0, sweep_depth / max(band_width, 1e-6)))
        return StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "P": float(levels["P"]),
                "H3": float(levels["H3"]),
                "L3": float(levels["L3"]),
                "H4": float(levels["H4"]),
                "L4": float(levels["L4"]),
                "sweep_extreme": float(sweep["sweep_extreme"]),
                "sweep_close": float(sweep["sweep_close"]),
                "close": float(bar_close),
            },
            context={
                "sweep_bar_ts": str(sweep.get("sweep_bar_ts")),
                "session_date_iso": session_date_iso,
                "atr": atr,
            },
            price=float(bar_close),
        )

    def _build_plan(self, ctx: MarketContext, side: str, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        # Architectural rule (2026-04-30): no re-detect; event REQUIRED.
        if event is None:
            return None
        evt = event
        if evt.side != side:
            return None
        # Re-entry guard
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        latch_key = (ctx.symbol, side, session_date_iso)
        if latch_key in self._fired_today:
            return None

        levels = evt.levels
        p = float(levels["P"])
        h3 = float(levels["H3"])
        l3 = float(levels["L3"])
        h4 = float(levels["H4"])
        l4 = float(levels["L4"])
        close = float(ctx.df_5m["close"].iloc[-1])
        atr = self._get_atr(ctx)
        wick_buf = close * self.wick_buf_pct

        # Stop placement: AT L4 (long) / H4 (short). Camarilla literature
        # explicit — L4 IS the wide stop, no extra ATR buffer needed. We
        # apply a tiny wick buffer below L4 (configurable) to absorb a
        # spike-and-recover wick. stop_atr_buf is configurable for A/B
        # variants but defaults to 0.0.
        if side == "long":
            hard_sl = l4 - wick_buf - atr * self.stop_atr_buf
            risk = max(close - hard_sl, atr * 0.1)
            min_risk = close * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = close - risk
            # T1 = pivot P (with fallback to 1R if direction-wrong)
            t1_level = p if p > close else close + risk
            # T2 = opposite side (long: H3; short: L3)
            t2_level = h3 if h3 > t1_level else close + 2.0 * risk
        else:   # short
            hard_sl = h4 + wick_buf + atr * self.stop_atr_buf
            risk = max(hard_sl - close, atr * 0.1)
            min_risk = close * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = close + risk
            t1_level = p if p < close else close - risk
            t2_level = l3 if l3 < t1_level else close - 2.0 * risk

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

        # Commit-on-success latch
        self._fired_today.add(latch_key)

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
        return self._build_plan(ctx, "long", event=event)

    def plan_short_strategy(
        self,
        ctx: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        if "short" not in self.allowed_sides:
            return None
        return self._build_plan(ctx, "short", event=event)

    def calculate_risk_params(
        self,
        entry_price: float,
        market_context: MarketContext,
    ) -> RiskParams:
        """Placeholder using ATR (real risk computed inside _build_plan from
        Camarilla L4/H4)."""
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
