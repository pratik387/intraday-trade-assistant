"""Gap-and-Go Continuation detector — sub-project #8 (Phase 0 2026-04-29).

REGIME-COMPLEMENT to the live `gap_fade_short` setup:
  - gap_fade_short FADES gaps in small/mid/micro_cap (gap-up exhausts on
    retail-driven momentum; pros short the failed continuation).
  - gap_and_go_continuation CONTINUES gaps in large/mid_cap that align with
    the daily trend (PDC vs 20-day SMA). Different cap-segment, different
    direction, mechanically opposite.

Indian-pro mechanic per
specs/2026-04-29-research-new-indian-setup-candidates.md Candidate 5
(Sources: Enrich Money, ICICI Direct, TrueData, GWC India, Motilal Oswal).

Two-bar state machine:
  bar 0 (09:15): qualifier — must satisfy
    (a) gap_pct >= gap_threshold_pct (default 1.0%) — minimum gap qualifier
    (b) PDC aligned with 20-day SMA in same direction as gap (gap-up needs
        PDC > 20SMA; gap-down needs PDC < 20SMA) — daily-trend filter
    (c) bar 0's high > bar 0's open (long; print a new intraday high above
        the gap-open). Mirror low < open for short.
    (d) bar 0's volume >= min_first_bar_volume_ratio × prior 14d 09:15-09:20
        baseline volume — institutional participation
  bar 1+ (within active_window 09:15-09:30): trigger — first bar whose
    high tags or exceeds bar 0's high (long; mirror low <= bar 0's low for
    short) fires the entry.

First-trigger latch keyed on (symbol, side, session_date_iso) prevents
same-session double-fire.

Mutual exclusion with orb_15: orb_15 self-excludes on
abs(gap_pct) > max_gap_pct_for_orb (0.5%); this detector self-excludes on
abs(gap_pct) < gap_threshold_pct (1.0%). The 0.5-1.0% band is dead — by
design — so the two never compete.

Wide-open mode bypass: under wide_open, daily-trend filter and first-bar
volume filter are skipped (design-inferred, not trigger geometry).
Trigger geometry (gap-pct, new-extreme on bar 0, trigger bar tagging
bar 0's extreme) STAYS active.

Sources cited:
  - specs/2026-04-29-gap_and_go_continuation-plan.md
  - specs/2026-04-29-research-new-indian-setup-candidates.md (§ Candidate 5)
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


class GapAndGoContinuationStructure(BaseStructure):
    """Bidirectional gap-and-go continuation in large/mid-cap."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "gap_and_go_continuation"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config (KeyError on missing).
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.gap_threshold_pct = float(config["gap_threshold_pct"])
        self.daily_trend_lookback = int(config["daily_trend_lookback_days"])
        self.daily_trend_min_distance_pct = float(config["daily_trend_min_distance_pct"])
        self.min_first_bar_vol_ratio = float(config["min_first_bar_volume_ratio"])
        self.volume_baseline_lookback = int(config["volume_baseline_lookback_days"])
        self.t1_target_r = float(config["t1_target_r"])
        self.t2_target_r = float(config["t2_target_r"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.stop_below_first_bar_low_buffer_pct = float(
            config["stop_below_first_bar_low_buffer_pct"]
        )
        self.allowed_sides: Set[str] = set(config["allowed_sides"])
        self.allowed_caps: Set[str] = set(config["allowed_cap_segments"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"]) / 100.0

        # First-trigger latch: set of (symbol, side, session_date_iso) tuples
        # that have already fired today. Cleared at session boundary.
        self._fired_today: Set[Tuple[str, str, str]] = set()
        self._latch_session_date: Optional[str] = None

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        """ATR with fallback (mirror gap_fade_short)."""
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            df = ctx.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _maybe_reset_latch(self, session_date_iso: str) -> None:
        """Clear fired_today if session boundary crossed."""
        if session_date_iso != self._latch_session_date:
            self._fired_today.clear()
            self._latch_session_date = session_date_iso

    def _get_volume_baseline(self, ctx: MarketContext) -> Optional[float]:
        """Return prior 14d mean of 09:15-09:20 (first 5m bar) volume.

        Reads from ctx.indicators["volume_baseline_open_5m"] first
        (orchestrator-populated). Fallback: estimate from
        ctx.df_daily["volume"].tail(N).mean() / 75.0 (75 ≈ 5m bars per
        6.25-hour session); approximation only.

        Returns None if neither source available.
        """
        if ctx.indicators is not None:
            v = ctx.indicators.get("volume_baseline_open_5m")
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        if ctx.df_daily is not None and len(ctx.df_daily) >= self.volume_baseline_lookback:
            try:
                daily_vol_mean = float(
                    ctx.df_daily["volume"].tail(self.volume_baseline_lookback).mean()
                )
                # 5m bars per 6.25h session ≈ 75
                return daily_vol_mean / 75.0
            except (KeyError, ValueError, TypeError):
                pass
        return None

    def _check_daily_trend(
        self,
        ctx: MarketContext,
        side: str,
    ) -> Optional[str]:
        """Validate PDC vs 20-day SMA alignment with the gap direction.

        Returns None on pass, or a rejection reason string.
        """
        daily = ctx.df_daily
        if daily is None or len(daily) < self.daily_trend_lookback:
            return "daily trend data unavailable"
        try:
            sma20 = float(
                daily["close"].iloc[-self.daily_trend_lookback :].mean()
            )
        except (KeyError, ValueError, TypeError):
            return "daily trend data invalid"
        pdc = float(ctx.pdc) if ctx.pdc is not None else 0.0
        if pdc <= 0:
            return "PDC unavailable for trend filter"
        threshold = sma20 * (1.0 + self.daily_trend_min_distance_pct / 100.0)
        threshold_neg = sma20 * (1.0 - self.daily_trend_min_distance_pct / 100.0)
        if side == "long":
            if pdc < threshold:
                return f"daily_trend_long: pdc={pdc:.2f} < 20sma+{self.daily_trend_min_distance_pct}%={threshold:.2f}"
        else:
            if pdc > threshold_neg:
                return f"daily_trend_short: pdc={pdc:.2f} > 20sma-{self.daily_trend_min_distance_pct}%={threshold_neg:.2f}"
        return None

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Two-bar state machine: bar 0 qualifier + bar 1+ trigger.

        Order of checks (cheap first):
          1. Universe + cap_segment + bars + active window
          2. Latch session-boundary reset
          3. PDC availability
          4. Gap qualification (abs(gap_pct) >= gap_threshold_pct)
          5. Side selection from gap direction; check allowed_sides
          6. First-trigger latch
          7. Wide-open-bypassable: daily-trend filter
          8. Wide-open-bypassable: first-bar volume filter
          9. First-bar new-extreme check (bar 0 high > bar 0 open for long;
             low < open for short)
          10. Trigger bar check (last bar tags bar 0's high for long; bar 0's
              low for short). Trigger MUST NOT be bar 0 itself (len >= 2).
          11. Build event + return
        """
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        # ---- Universe + cap_segment (design-inferred — bypassed under wide_open) ----
        # Per master plan: wide-open OCI capture must see ALL symbols / cap
        # segments so the gauntlet can decide which slice the detector works in.
        if not _wide_open and not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        if not _wide_open and ctx.cap_segment not in self.allowed_caps:
            return _empty(f"cap_segment {ctx.cap_segment!r} not in allowed set")

        # ---- Bars + active window (always enforced — mechanical) ----
        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- Latch session-boundary reset ----
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        self._maybe_reset_latch(session_date_iso)

        # ---- PDC availability ----
        if ctx.pdc is None or float(ctx.pdc) <= 0:
            return _empty("PDC unavailable")
        pdc = float(ctx.pdc)

        # ---- Gap qualification ----
        bar0 = df.iloc[0]
        bar0_open = float(bar0["open"])
        gap_pct = ((bar0_open - pdc) / pdc) * 100.0
        if abs(gap_pct) < self.gap_threshold_pct:
            return _empty(
                f"gap_pct={gap_pct:.2f} below threshold ±{self.gap_threshold_pct}"
            )

        # ---- Side selection from gap direction ----
        side = "long" if gap_pct > 0 else "short"
        if side not in self.allowed_sides:
            return _empty(
                f"side {side!r} not in allowed_sides {sorted(self.allowed_sides)}"
            )

        # ---- First-trigger latch ----
        latch_key = (ctx.symbol, side, session_date_iso)
        if latch_key in self._fired_today:
            return _empty(f"already_fired: {ctx.symbol}/{side}/{session_date_iso}")

        # ---- Daily-trend filter (wide-open bypass) ----
        if not _wide_open:
            trend_reject = self._check_daily_trend(ctx, side)
            if trend_reject is not None:
                return _empty(trend_reject)

        # ---- First-bar volume filter (wide-open bypass) ----
        if not _wide_open:
            baseline = self._get_volume_baseline(ctx)
            if baseline is None:
                return _empty("volume_baseline_unavailable")
            if baseline <= 0:
                return _empty("volume_baseline_invalid")
            first_bar_vol = float(bar0["volume"])
            vol_ratio = first_bar_vol / baseline
            if vol_ratio < self.min_first_bar_vol_ratio:
                return _empty(
                    f"first_bar_vol_ratio={vol_ratio:.2f} < {self.min_first_bar_vol_ratio}"
                )

        # ---- First-bar new-extreme check ----
        bar0_high = float(bar0["high"])
        bar0_low = float(bar0["low"])
        if side == "long":
            if bar0_high <= bar0_open:
                return _empty(
                    f"first_bar_no_new_high_long: high={bar0_high:.2f} <= open={bar0_open:.2f}"
                )
        else:
            if bar0_low >= bar0_open:
                return _empty(
                    f"first_bar_no_new_low_short: low={bar0_low:.2f} >= open={bar0_open:.2f}"
                )

        # ---- Trigger bar check ----
        # Trigger bar must NOT be bar 0 itself; need at least bar 0 + bar 1.
        if len(df) < 2:
            return _empty("trigger requires bar 1+ (bar 0 is qualifier only)")
        last = df.iloc[-1]
        last_high = float(last["high"])
        last_low = float(last["low"])
        last_close = float(last["close"])
        if side == "long":
            if last_high < bar0_high:
                return _empty(
                    f"trigger_bar_below_first_bar_high: last_high={last_high:.2f} < bar0_high={bar0_high:.2f}"
                )
        else:
            if last_low > bar0_low:
                return _empty(
                    f"trigger_bar_above_first_bar_low: last_low={last_low:.2f} > bar0_low={bar0_low:.2f}"
                )

        # ---- All gates pass — build event ----
        confidence = min(1.0, abs(gap_pct) / 3.0)
        # entry_trigger_level = bar 0 extreme (the level the trigger bar tagged)
        entry_trigger_level = bar0_high if side == "long" else bar0_low
        # daily_trend_distance_pct: how far PDC is from 20SMA (positive = above)
        if ctx.df_daily is not None and len(ctx.df_daily) >= self.daily_trend_lookback:
            try:
                sma20 = float(
                    ctx.df_daily["close"].iloc[-self.daily_trend_lookback :].mean()
                )
                trend_dist_pct = (pdc - sma20) / sma20 * 100.0
            except (KeyError, ValueError, TypeError):
                trend_dist_pct = 0.0
        else:
            trend_dist_pct = 0.0
        first_bar_vol = float(bar0["volume"])
        baseline = self._get_volume_baseline(ctx)
        first_bar_vol_ratio = (
            first_bar_vol / baseline if (baseline is not None and baseline > 0) else 0.0
        )

        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "pdc": pdc,
                "gap_open": bar0_open,
                "first_bar_high": bar0_high,
                "first_bar_low": bar0_low,
                "close": last_close,
                "entry_trigger_level": entry_trigger_level,
            },
            context={
                "gap_pct": gap_pct,
                "first_bar_volume_ratio": first_bar_vol_ratio,
                "daily_trend_distance_pct": trend_dist_pct,
                "session_date_iso": session_date_iso,
            },
            price=last_close,
        )
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    def _build_plan(self, ctx: MarketContext, side: str, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        # Architectural rule (2026-04-30): no re-detect; event REQUIRED.
        if event is None:
            return None
        evt = event
        if evt.side != side:
            return None

        levels = evt.levels
        first_bar_high = float(levels["first_bar_high"])
        first_bar_low = float(levels["first_bar_low"])
        close = float(ctx.df_5m["close"].iloc[-1])
        atr = self._get_atr(ctx)

        buffer_frac = self.stop_below_first_bar_low_buffer_pct / 100.0
        if side == "long":
            # Entry at trigger level (slight slippage tolerance)
            entry = first_bar_high * (1.0 + 0.0001)
            hard_sl = first_bar_low * (1.0 - buffer_frac)
            risk = max(entry - hard_sl, atr * 0.1)
            min_risk = entry * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = entry - risk
            t1_level = entry + risk * self.t1_target_r
            t2_level = entry + risk * self.t2_target_r
        else:  # short
            entry = first_bar_low * (1.0 - 0.0001)
            hard_sl = first_bar_high * (1.0 + buffer_frac)
            risk = max(hard_sl - entry, atr * 0.1)
            min_risk = entry * self.min_stop_distance_pct
            if risk < min_risk:
                risk = min_risk
                hard_sl = entry + risk
            t1_level = entry - risk * self.t1_target_r
            t2_level = entry - risk * self.t2_target_r

        targets = [
            {
                "name": "T1", "level": t1_level, "rr": self.t1_target_r,
                "qty_pct": self.t1_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_level, "rr": self.t2_target_r,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full",
            },
        ]
        risk_params = RiskParams(hard_sl=hard_sl, risk_per_share=risk, atr=atr)
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        # Latch on plan-build success (mirrors orb_15 pattern — keeps detect()
        # idempotent; only commit-to-trade actions register the latch).
        session_date_iso = evt.context.get("session_date_iso")
        if session_date_iso:
            self._fired_today.add((ctx.symbol, side, session_date_iso))

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
        bar 0's first-bar low/high)."""
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
