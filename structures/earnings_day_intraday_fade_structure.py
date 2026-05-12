"""Earnings-Day Intraday Fade detector — sub-project #9 (ship #5).

Indian-microstructure-specific asymmetry: NSE single-stock options + low
retail tolerance for earnings-day uncertainty produces a T+0 09:15 gap
(positive or negative) that retail / option-buyers extend in the first
hour — only for inst-flow to fade it in mid-session as the earnings
beat/miss gets priced rationally by 10:30+.

Mechanic
--------
T+0 earnings: BMO (Before Market Open) or AMC (After Market Close)
on today; the morning 09:15 5m open already prices the announcement.
Single fire per (symbol, session_date). Date matching is T+0 ONLY —
see services/earnings_calendar_loader.py module docstring for the
production/sanity alignment rationale (T-1 lookback was removed
2026-05-12 after producing unvalidated extra fires).

T+0 09:15 gap classification:
  gap_pct = (open_09:15 - T-1_daily_close) / T-1_daily_close
  in [-6%, -0.5%] -> LONG  (fade panic capitulation)
  in [+0.5%, +6%] -> SHORT (fade FOMO continuation)

T+0 10:30-15:00 5m bar confirmation candle:
  SHORT: close < open (red bar)  AND  close >= T-1 close (not already mean-reverted past prior close)
  LONG : close > open (green bar) AND close <= T-1 close (not already mean-reverted past prior close)

Stop:
  SHORT: hard_sl = max(T+0_high * (1 + sl_buffer), entry * (1 + min_stop_pct))
  LONG : hard_sl = min(T+0_low  * (1 - sl_buffer), entry * (1 - min_stop_pct))
  Then scale by stop_r_multiple (2.5× per validated 3D sweep).

Targets (locked from 3D sweep — DO NOT modify without re-running gauntlet):
  T1 = 1.0R, 50% qty
  T2 = 1.0R (same as T1; behaves as a single tier — kept for compatibility)
  Breakeven trail after T1.
  Time stop: 15:10 IST (5 min before MIS auto-square).

Validation history:
  Discovery (in-sample): PF 1.64, n=1569
  OOS (Jan-Sep 2025):    PF 1.53, n=510
  Holdout (Oct'25-Apr'26): PF 1.25, n=298 (regime-adaptive across India war Jan'26)

Universe & cell:
  - MIS-eligible (nse_all.json)
  - ADV >= 10 cr / day (cell-locked at 10cr to capture mid-tier earnings reactors)
  - cap_segment in {large_cap, mid_cap}  (small_cap dropped to PF 0.87 OOS — exclude)
  - BMO or AMC announce_time_class

Brief: specs/2026-05-06-sub-project-9-brief-earnings_day_intraday_fade.md
Sanity: tools/sub9_research/sanity_earnings_day_intraday_fade.py
"""
from __future__ import annotations

from datetime import time, date, timedelta
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from services.earnings_calendar_loader import has_earnings_today_or_yesterday
from services.plan_helpers import (
    PlanRejected,
    assert_sl_outside_entry_zone,
    compute_entry_zone,
    enforce_min_stop_distance,
)
from services.symbol_metadata import get_mis_info
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


class EarningsDayIntradayFadeStructure(BaseStructure):
    """Earnings-Day Intraday Fade — bidirectional.

    Cross-day state: requires (a) T-1 daily close (read from `ctx.df_daily`)
    to compute the T+0 09:15 opening gap, and (b) BMO/AMC earnings event
    on today (T+0 only, read from the earnings_calendar parquet via
    `services.earnings_calendar_loader`).

    Side selection is data-driven from the gap direction. One fire per
    (symbol, session_date).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "earnings_day_intraday_fade"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config — KeyError on missing.
        # Gap classification
        self.gap_pct_min = float(config["gap_pct_min"])
        self.gap_pct_max = float(config["gap_pct_max"])
        # Entry window
        self.entry_window_start = self._parse_time(config["entry_window_start_hhmm"])
        self.entry_window_end = self._parse_time(config["entry_window_end_hhmm"])
        self.time_stop = self._parse_time(config["time_stop_hhmm"])
        # Risk
        self.sl_buffer_pct = float(config["sl_buffer_pct"])
        self.min_stop_pct = float(config["min_stop_pct"])
        # Locked R-multiples (do not modify — 3D-sweep validated)
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.stop_r_multiple = float(config["stop_r_multiple"])
        self.use_breakeven_trail_after_t1 = bool(config["use_breakeven_trail_after_t1"])
        # Per-target qty split — plan-as-source-of-truth (2026-05-12 architectural
        # refactor). qty_pct=0 → executor skips T1 partial, full qty rides to T2.
        self.t1_qty_pct = float(config.get("t1_qty_pct", 0.5))
        # Universe gates
        self.min_adv_cr = float(config["min_adv_cr"])
        self.allowed_caps = set(config["allowed_caps"])
        self.announce_time_classes = tuple(config["announce_time_classes"])
        # Plumbing
        self.min_bars_required = int(config.get("min_bars_required", 4))
        self.risk_per_trade_inr = float(config["risk_per_trade_inr"])

        # First-trigger latch: one fire per (symbol, session_date).
        self._fired_today: set = set()
        # Per-(symbol, session_date) cache for the qualifying T-1 close / day_open / day_high / day_low.
        self._t0_cache: Dict[Tuple[str, date], Optional[Dict[str, float]]] = {}

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_prev_close(
        self, df_daily: pd.DataFrame, session_date: date,
    ) -> Optional[float]:
        """Return the close of the most-recent daily bar strictly before
        `session_date`. None on any failure (missing daily / no prior bar).
        """
        if df_daily is None or df_daily.empty:
            return None
        d_idx = df_daily.index
        try:
            d_dates = (
                d_idx.date if hasattr(d_idx, "date")
                else pd.to_datetime(d_idx).date
            )
        except Exception:
            return None
        prior = [d for d in d_dates if d < session_date]
        if not prior:
            return None
        last_prior = max(prior)
        try:
            pos = list(d_dates).index(last_prior)
            return float(df_daily.iloc[pos]["close"])
        except (KeyError, ValueError, IndexError):
            return None

    def _compute_adv_cr(
        self, df_daily: pd.DataFrame, session_date: date,
    ) -> Optional[float]:
        """20-day average daily traded value (close × volume) in INR Cr,
        computed from `df_daily` rows strictly before `session_date`.
        None on insufficient data.
        """
        if df_daily is None or df_daily.empty:
            return None
        try:
            d_idx = df_daily.index
            d_dates = (
                d_idx.date if hasattr(d_idx, "date")
                else pd.to_datetime(d_idx).date
            )
            mask = [d < session_date for d in d_dates]
            prior = df_daily[mask]
            if prior.empty:
                return None
            tail = prior.iloc[-20:]
            if tail.empty:
                return None
            tv = (tail["close"].astype(float) * tail["volume"].astype(float)).mean()
            return float(tv) / 1e7
        except Exception:
            return None

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Fire LONG or SHORT on a 10:30-15:00 5m bar after a qualifying
        BMO/AMC earnings event + |gap|in[0.5%, 6%] + confirmation candle.
        """

        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        # ---- Cap segment guard (bypassed under wide_open) ----
        if not _wide_open and ctx.cap_segment not in self.allowed_caps:
            return _empty(
                f"cap_segment {ctx.cap_segment!r} not in allowed set "
                f"{sorted(self.allowed_caps)}"
            )

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        # ---- Active window: 10:30 - 15:00 ----
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.entry_window_start <= cur_t <= self.entry_window_end):
            return _empty(f"Outside entry window: {cur_t}")

        # ---- Latch (per session_date) ----
        session_date = ctx.session_date
        if session_date is None:
            session_date = pd.Timestamp(last_ts).date()
        latch_key = (ctx.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        # ---- MIS eligibility (bypassed under wide_open) ----
        if not _wide_open:
            mis = get_mis_info(ctx.symbol)
            if not mis.get("mis_enabled"):
                return _empty(f"MIS not enabled for {ctx.symbol}")

        # ---- Earnings event gate (BMO/AMC T+0 only) ----
        # Bypassed under wide_open so capture runs see fires without the
        # earnings filter — that lets the gauntlet sanity-check the
        # confounders. In production this is a hard gate.
        # NOTE: T+0 only — matches the validated sanity script join logic
        # (PF 1.64 / n=1569 Discovery). DO NOT extend to T-1 lookback
        # without re-running the gauntlet. See services/earnings_calendar_
        # loader.py module docstring for the full rationale.
        if not _wide_open:
            if not has_earnings_today_or_yesterday(
                ctx.symbol, session_date, self.announce_time_classes,
            ):
                return _empty(
                    f"no {list(self.announce_time_classes)} earnings on "
                    f"{session_date}"
                )

        # ---- Cross-day daily bars (T-1 close + ADV) ----
        df_daily = ctx.df_daily
        if df_daily is None or df_daily.empty:
            return _empty("daily bars unavailable")

        # Cache the cross-day meta per (symbol, session_date).
        cache_key = (ctx.symbol, session_date)
        if cache_key in self._t0_cache:
            cross_day = self._t0_cache[cache_key]
        else:
            prev_close = self._get_prev_close(df_daily, session_date)
            adv_cr = self._compute_adv_cr(df_daily, session_date)
            if prev_close is None or prev_close <= 0:
                cross_day = None
            elif adv_cr is None:
                cross_day = None
            else:
                cross_day = {"prev_close": prev_close, "adv_cr": adv_cr}
            self._t0_cache[cache_key] = cross_day
        if cross_day is None:
            return _empty("prev_close or ADV unavailable")

        # ---- ADV gate (bypassed under wide_open) ----
        adv_cr = cross_day["adv_cr"]
        if not _wide_open and adv_cr < self.min_adv_cr:
            return _empty(f"adv_cr={adv_cr:.2f} < {self.min_adv_cr}")

        # ---- Today's bars + day_open / day_high / day_low ----
        today_bars = df[df.index.date == session_date]
        if today_bars.empty:
            return _empty("no bars for session_date")
        day_open = float(today_bars.iloc[0]["open"])
        day_high = float(today_bars["high"].max())
        day_low = float(today_bars["low"].min())
        if day_open <= 0:
            return _empty("day_open invalid")

        # ---- Gap classification ----
        prev_close = cross_day["prev_close"]
        gap_pct = (day_open / prev_close) - 1.0
        abs_gap = abs(gap_pct)
        if abs_gap < self.gap_pct_min:
            return _empty(
                f"|gap_pct|={abs_gap:.4f} < {self.gap_pct_min} (min)"
            )
        if abs_gap > self.gap_pct_max:
            return _empty(
                f"|gap_pct|={abs_gap:.4f} > {self.gap_pct_max} (max)"
            )

        # ---- Confirmation candle ----
        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_close = float(last["close"])

        if gap_pct > 0:
            # SHORT (fade FOMO continuation)
            side = "short"
            # red candle AND close >= prev_close (not already mean-reverted past T-1 close)
            if bar_close >= bar_open:
                return _empty(
                    f"SHORT requires red bar: open={bar_open}, close={bar_close}"
                )
            if bar_close < prev_close:
                return _empty(
                    f"SHORT already mean-reverted: close={bar_close} < "
                    f"prev_close={prev_close}"
                )
        else:
            # LONG (fade panic capitulation)
            side = "long"
            if bar_close <= bar_open:
                return _empty(
                    f"LONG requires green bar: open={bar_open}, close={bar_close}"
                )
            if bar_close > prev_close:
                return _empty(
                    f"LONG already mean-reverted: close={bar_close} > "
                    f"prev_close={prev_close}"
                )

        # ---- Build event ----
        # Confidence scaled by |gap_pct| within the band — larger surviving
        # gaps have proportionally more mean-reversion potential.
        gap_range = max(self.gap_pct_max - self.gap_pct_min, 1e-6)
        confidence = max(0.0, min(1.0, (abs_gap - self.gap_pct_min) / gap_range))

        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "prev_close": prev_close,
                "day_open": day_open,
                "day_high": day_high,
                "day_low": day_low,
                "entry_close": bar_close,
            },
            context={
                "gap_pct": gap_pct,
                "abs_gap_pct": abs_gap,
                "adv_cr": adv_cr,
                "session_date_iso": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
            },
            price=bar_close,
        )
        # Set latch HERE in detect() — not in plan_*_strategy. Per lessons.md:
        # plan_*_strategy runs in PlanOrchestrator (main process) under a
        # SEPARATE detector instance, so a latch set there never propagates
        # back to the workers' detect() loop. Setting the latch here is the
        # canonical pattern used by every other sub-9 detector.
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ------------------------------------------------------------------
    # plan_long_strategy() / plan_short_strategy()
    # ------------------------------------------------------------------

    def _build_plan(
        self,
        ctx: MarketContext,
        event: StructureEvent,
        side: str,
    ) -> Optional[TradePlan]:
        """Build TradePlan with R-multiple targets (T1=1R, T2=1R, SL=2.5×
        structural). Returns None on invalid geometry.
        """
        if event is None or event.side != side:
            return None

        df = ctx.df_5m
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        close = float(last["close"])
        levels = event.levels or {}
        day_high = float(levels.get("day_high", close))
        day_low = float(levels.get("day_low", close))

        # ---- Structural stop ----
        if side == "short":
            sl_struct = day_high * (1.0 + self.sl_buffer_pct)
            sl_min = close * (1.0 + self.min_stop_pct)
            hard_sl_raw = max(sl_struct, sl_min)
            stop_distance_raw = hard_sl_raw - close
        else:
            sl_struct = day_low * (1.0 - self.sl_buffer_pct)
            sl_min = close * (1.0 - self.min_stop_pct)
            hard_sl_raw = min(sl_struct, sl_min)
            stop_distance_raw = close - hard_sl_raw

        if stop_distance_raw <= 0:
            logger.debug(
                f"[{ctx.symbol}] earnings_day_intraday_fade skip: invalid "
                f"raw stop_distance={stop_distance_raw} (side={side}, "
                f"close={close}, day_high={day_high}, day_low={day_low})"
            )
            return None

        # Apply the locked stop_r_multiple (2.5×) to scale the structural stop.
        stop_distance = stop_distance_raw * self.stop_r_multiple
        if side == "short":
            hard_sl = close + stop_distance
        else:
            hard_sl = close - stop_distance

        risk_per_share = stop_distance

        # ---- R-multiple targets (LOCKED: T1=1R, T2=1R) ----
        if side == "short":
            t1_target = close - self.t1_r_multiple * stop_distance
            t2_target = close - self.t2_r_multiple * stop_distance
        else:
            t1_target = close + self.t1_r_multiple * stop_distance
            t2_target = close + self.t2_r_multiple * stop_distance

        # T1 / T2 qty split from config (plan-as-source-of-truth 2026-05-12).
        targets = [
            {
                "name": "T1", "level": t1_target, "rr": self.t1_r_multiple,
                "qty_pct": self.t1_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_target, "rr": self.t2_r_multiple,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full",
            },
        ]

        # ATR for risk_params metadata only.
        atr_val = None
        try:
            if ctx.indicators and "atr" in ctx.indicators:
                atr_val = float(ctx.indicators["atr"])
            elif "atr" in df.columns and not pd.isna(df["atr"].iloc[-1]):
                atr_val = float(df["atr"].iloc[-1])
        except Exception:
            atr_val = None

        risk_params = RiskParams(
            hard_sl=hard_sl, risk_per_share=risk_per_share, atr=atr_val,
        )
        trail_to = "breakeven" if self.use_breakeven_trail_after_t1 else None
        exit_levels = ExitLevels(
            hard_sl=hard_sl, targets=targets, trail_to=trail_to,
            time_exit=self.time_stop.strftime("%H:%M"),
        )

        # Plan-geometry validation
        try:
            _zone = compute_entry_zone(
                entry=close, bias=side,
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, hard_sl, side)
            enforce_min_stop_distance(
                close, hard_sl, self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{ctx.symbol}] earnings_day_intraday_fade plan rejected: "
                f"{e.reason} {e.details}"
            )
            return None

        # Latch is set in detect() (worker-side) — not here.
        return TradePlan(
            symbol=ctx.symbol,
            side=side,
            structure_type=event.structure_type,
            entry_price=close,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context,
            trade_id=event.trade_id,
            # R-multiple targets — preserved across late fills by
            # services/target_recalc.py recomputing T1/T2 from the actual
            # entry using stored R multiples.
            target_anchor_type="r_multiple",
        )

    def plan_long_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        return self._build_plan(ctx, event, "long") if event is not None else None

    def plan_short_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        return self._build_plan(ctx, event, "short") if event is not None else None

    # ------------------------------------------------------------------
    # BaseStructure abstract methods (legacy ABC contract)
    # ------------------------------------------------------------------

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext,
    ) -> RiskParams:
        """ABC contract — production stop comes from _build_plan."""
        return RiskParams(
            hard_sl=entry_price * (1.0 + self.min_stop_pct),
            risk_per_share=entry_price * self.min_stop_pct,
            atr=None,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """ABC contract — default 1R exit; production targets from _build_plan."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        if trade_plan.side == "short":
            t1 = entry - self.t1_r_multiple * risk
        else:
            t1 = entry + self.t1_r_multiple * risk
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": self.t1_r_multiple}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(
        self, context: MarketContext, event: Optional[StructureEvent] = None,
    ) -> float:
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.entry_window_start <= t <= self.entry_window_end
