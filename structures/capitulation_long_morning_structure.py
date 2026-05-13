"""Capitulation LONG morning fade detector — sub-project #9 round-6.

Indian-microstructure-specific asymmetry: NSE 09:15 gap-DOWN of 1.5-8% on a
mid_cap symbol with NIFTY in trend_down regime → retail panic-capitulation
exhausts in 09:25-10:00 window → fade UP. Mirror image of gap_fade_short
(which fades retail FOMO on gap-UPS).

Mechanic
--------
T+0 09:15 5m bar gap-down: gap_pct = (open_09:15 - prev_close) / prev_close
in [-8%, -1.5%] (gap-down, capped at -8% to exclude fundamental-news
shocks).

T+0 09:25-10:00 5m bar exhaustion-candle confirmation:
  - lower_wick / body >= 0.5  (retail-stop-cleanout signature)
  - body_size_pct <= 30%       (small body — capitulation, not continuation)
  - close > open               (green = bid-side stepping in)
  - low > 09:15 low            (no fresh low — bottom forming)
Fire LONG at next 5m bar's open.

Stop:
  stop_a = gap_low * (1 - 0.005)   (just below 09:15 capitulation low)
  stop_b = entry - ATR * 1.5         (volatility-scaled floor)
  hard_sl = min(stop_a, stop_b)     (whichever is further below entry)
  enforce min_stop_distance_pct (config, default 0.3%)

T1 = entry × (1 + stop_pct × 1R), 50% qty (locked param)
T2 = entry × (1 + stop_pct × 2R), 50% qty
Time stop: 10:15 IST (10 min after exhaustion-window close — same as
gap_fade_short's symmetric mirror).

Cell-locked filters (per round-6 cell selection on n=7,471 sanity, where
trend_down × mid_cap × liq=10-30cr produced n=443 PF 1.238 WR 62.5%):
  - allowed_regimes = ["trend_down"]
  - allowed_cap_segments = ["mid_cap"]
  - allowed_liquidity_band_cr = [10.0, 30.0]   (20-day ADV in Cr)

Brief: specs/2026-05-07-sub-project-9-brief-capitulation_long_morning.md
Sanity: reports/sub9_sanity/capitulation_long_morning_trades.csv (n=7,471
aggregate PF 0.813; trend_down × mid_cap × liq=10-30cr cell ships at PF
1.238).
"""
from __future__ import annotations

from datetime import time, date
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.plan_helpers import (
    PlanRejected,
    assert_sl_outside_entry_zone,
    compute_entry_zone,
    enforce_min_stop_distance,
)
from services.symbol_metadata import in_universe
from .base_structure import BaseStructure


# NIFTY 50 daily regime cache — computed once per process from the daily
# index feather. Matches `tools/sub9_research/sanity_first_hour_momentum.py`'s
# `load_nifty_daily_regime()` classifier (5-day return thresholded at +/-1.5%)
# which is the convention used for sub9 cell-mining. The shipped cell
# (trend_down x mid_cap x liq=10-30cr, PF 1.238 n=443) was validated against
# THIS classifier — not the ADX-based regime in services/gates/regime_gate.py.
_NIFTY_DAILY_REGIME_CACHE: Optional[Dict[date, str]] = None


def _load_nifty_daily_regime() -> Dict[date, str]:
    """Load NIFTY 50 daily feather and compute 5-day-return-based regime.

    Returns dict mapping session_date -> regime in {trend_up, trend_down,
    chop, unknown}. Cached per process.

    Source: backtest-cache-download/index_ohlcv/NSE_NIFTY_50/NSE_NIFTY_50_1days.feather
    """
    global _NIFTY_DAILY_REGIME_CACHE
    if _NIFTY_DAILY_REGIME_CACHE is not None:
        return _NIFTY_DAILY_REGIME_CACHE
    base = Path(__file__).resolve().parents[1]
    fp = base / "backtest-cache-download" / "index_ohlcv" / "NSE_NIFTY_50" / "NSE_NIFTY_50_1days.feather"
    if not fp.exists():
        logger.warning(f"NIFTY 50 daily feather not found at {fp} — regime defaults to 'unknown'")
        _NIFTY_DAILY_REGIME_CACHE = {}
        return _NIFTY_DAILY_REGIME_CACHE
    df = pd.read_feather(fp)
    if "date" in df.columns:
        if df["date"].dt.tz is not None:
            df["date"] = df["date"].dt.tz_localize(None)
        df["d"] = df["date"].dt.date
    df = df.sort_values("d").reset_index(drop=True)
    df["nifty_5d_ret"] = df["close"].pct_change(5) * 100.0

    def _r(r: float) -> str:
        if pd.isna(r):
            return "unknown"
        if r >= 1.5:
            return "trend_up"
        if r <= -1.5:
            return "trend_down"
        return "chop"

    df["regime"] = df["nifty_5d_ret"].apply(_r)
    _NIFTY_DAILY_REGIME_CACHE = dict(zip(df["d"], df["regime"]))
    return _NIFTY_DAILY_REGIME_CACHE
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
        from services.config_loader import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class CapitulationLongMorningStructure(BaseStructure):
    """LONG-only capitulation-fade detector at NSE morning open.

    Fires once per (symbol, session_date) on a 09:25-10:00 5m bar that
    confirms exhaustion-bottom after a 09:15 gap-down. Cell-locked to
    trend_down × mid_cap × liq=10-30cr per round-6 cell selection.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "capitulation_long_morning"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config — KeyError on missing.
        self.gap_min_pct = float(config["gap_min_pct"])              # 1.5
        self.gap_max_pct = float(config["gap_max_pct"])              # 8.0
        self.lower_wick_ratio_min = float(config["lower_wick_ratio_min"])  # 0.5
        self.body_size_max_pct = float(config["body_size_max_pct"])  # 30.0
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.stop_pct = float(config["stop_pct"])                    # 1.0
        self.gap_low_buffer_pct = float(config["gap_low_buffer_pct"])  # 0.5
        self.atr_stop_multiple = float(config["atr_stop_multiple"])  # 1.5
        self.t1_r_multiple = float(config["t1_r_multiple"])          # 1.0
        self.t2_r_multiple = float(config["t2_r_multiple"])          # 2.0
        self.t1_qty_pct = float(config["t1_qty_pct"])                # 0.5
        # Cell filters (locked from round-6 cell selection):
        self.allowed_caps = set(config["allowed_cap_segments"])
        ar = config.get("allowed_regimes")
        self.allowed_regimes: Optional[set] = set(ar) if ar else None
        liq = config.get("allowed_liquidity_band_cr")
        self.liquidity_band: Optional[tuple] = (
            (float(liq[0]), float(liq[1])) if liq and len(liq) == 2 else None
        )
        uk = config.get("universe_key")
        self.universe_key = str(uk) if uk else None
        self.min_bars_required = int(config["min_bars_required"])

        # First-trigger latch: one fire per (symbol, session_date) per session.
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Fire LONG only on 09:20-10:00 bar that confirms exhaustion-bottom
        after a 09:15 gap-down within [-8%, -1.5%]."""

        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        if not _wide_open and ctx.cap_segment not in self.allowed_caps:
            return _empty(f"Cap segment {ctx.cap_segment!r} not in allowed set")

        if (
            not _wide_open
            and self.universe_key is not None
            and not in_universe(ctx.symbol, self.universe_key)
        ):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        # Regime classification: use NIFTY 50 daily 5-day-return rule
        # (matches sub9 cell-mining convention — see _load_nifty_daily_regime
        # docstring). Production's services/gates/regime_gate.py uses an
        # ADX-based classifier with different sensitivity; that produces a
        # DIFFERENT "trend_down" classification than what was validated.
        session_date_for_regime = ctx.session_date
        if session_date_for_regime is None and ctx.df_5m is not None and len(ctx.df_5m) > 0:
            session_date_for_regime = pd.Timestamp(ctx.df_5m.index[-1]).date()
        nifty_regime = _load_nifty_daily_regime().get(session_date_for_regime, "unknown")

        if (
            not _wide_open
            and self.allowed_regimes is not None
            and nifty_regime not in self.allowed_regimes
        ):
            return _empty(
                f"nifty_regime {nifty_regime!r} not in allowed set {sorted(self.allowed_regimes)}"
            )

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        session_date = ctx.session_date
        if session_date is None:
            session_date = pd.Timestamp(last_ts).date()
        latch_key = (ctx.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        # ---- Filter to today's bars only ----
        today = df[df.index.date == session_date]
        if today.empty:
            return _empty("no bars for session_date")

        # ---- Gap-down gate (09:15 first bar vs prior daily close) ----
        # Use df_daily for prev close; if missing, fall back to PDC from levels.
        prev_close = None
        if ctx.df_daily is not None and not ctx.df_daily.empty:
            try:
                d_idx = ctx.df_daily.index
                d_dates = (d_idx.date if hasattr(d_idx, "date")
                           else pd.to_datetime(d_idx).date)
                priors = [d for d in d_dates if d < session_date]
                if priors:
                    last_prior = max(priors)
                    pos = list(d_dates).index(last_prior)
                    prev_close = float(ctx.df_daily.iloc[pos]["close"])
            except Exception:
                prev_close = None
        if prev_close is None and ctx.pdc is not None:
            prev_close = float(ctx.pdc)
        if prev_close is None or prev_close <= 0:
            return _empty("prev_close unavailable")

        first_bar = today.iloc[0]
        first_open = float(first_bar["open"])
        gap_pct = (first_open / prev_close - 1.0) * 100.0
        if gap_pct > -self.gap_min_pct or gap_pct < -self.gap_max_pct:
            return _empty(
                f"gap_pct={gap_pct:.2f}% not in [-{self.gap_max_pct}, "
                f"-{self.gap_min_pct}]"
            )
        gap_low = float(first_bar["low"])

        # ---- Liquidity gate (20-day ADV in Cr) — bypassed under wide_open ----
        if not _wide_open and self.liquidity_band is not None:
            adv_cr = None
            if "adv_20d_cr" in df.columns and not pd.isna(df["adv_20d_cr"].iloc[-1]):
                adv_cr = float(df["adv_20d_cr"].iloc[-1])
            elif ctx.df_daily is not None and not ctx.df_daily.empty:
                # Fall back: compute from daily close × volume tail
                try:
                    pos = len(ctx.df_daily) - 1
                    tail = ctx.df_daily.iloc[max(0, pos - 20): pos]
                    if not tail.empty:
                        tv = (tail["close"] * tail["volume"]).mean() / 1e7
                        adv_cr = float(tv)
                except Exception:
                    adv_cr = None
            if adv_cr is None or not (
                self.liquidity_band[0] <= adv_cr <= self.liquidity_band[1]
            ):
                return _empty(
                    f"adv_20d_cr={adv_cr} not in band {self.liquidity_band}"
                )

        # ---- Exhaustion-candle confirmation on the current bar ----
        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])

        if bar_close <= bar_open:
            return _empty(f"not green: open={bar_open}, close={bar_close}")
        if bar_low <= gap_low:
            return _empty(f"fresh low (bar_low {bar_low} <= gap_low {gap_low})")

        body = abs(bar_close - bar_open)
        candle_bot = min(bar_open, bar_close)
        lower_wick = candle_bot - bar_low
        if body < 1e-8:
            lower_wick_ratio = float("inf")  # doji bottom = strong exhaustion
        else:
            lower_wick_ratio = lower_wick / body
        if lower_wick_ratio < self.lower_wick_ratio_min:
            return _empty(
                f"lower_wick_ratio={lower_wick_ratio:.3f} < "
                f"{self.lower_wick_ratio_min}"
            )

        body_size_pct = (body / bar_open * 100.0) if bar_open > 0 else 0.0
        if body_size_pct > self.body_size_max_pct:
            return _empty(
                f"body_size_pct={body_size_pct:.2f}% > {self.body_size_max_pct}"
            )

        # ---- ATR for stop floor ----
        atr_val = (ctx.indicators or {}).get("atr")
        if atr_val is None and "atr" in df.columns:
            try:
                atr_val = float(df["atr"].iloc[-1])
            except Exception:
                atr_val = None
        if atr_val is None or pd.isna(atr_val):
            # Fallback: bar-range-based proxy
            atr_val = float((today["high"] - today["low"]).tail(14).mean())

        # ---- All conditions met ----
        confidence = min(1.0, max(0.0, abs(gap_pct) / self.gap_max_pct))
        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="long",
            confidence=confidence,
            levels={
                "entry_close": bar_close,
                "bar_low": bar_low,
                "bar_high": bar_high,
                "gap_low": gap_low,
                "first_open": first_open,
                "prev_close": prev_close,
            },
            context={
                "gap_pct": gap_pct,
                "lower_wick_ratio": lower_wick_ratio,
                "body_size_pct": body_size_pct,
                "atr": atr_val,
                "session_date_iso": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
            },
            price=bar_close,
        )
        # Set latch HERE in detect() — not in plan_long_strategy. detect()
        # runs in the cached MainDetector instance per worker process so the
        # latch survives across bars within that worker. plan_long_strategy
        # runs in PlanOrchestrator (main process) — a separate detector
        # instance, so a latch set there never propagated back to the
        # workers' detect() loop. Caused 1.67x re-fire on (symbol, day)
        # within the 09:25-10:00 active window.
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ------------------------------------------------------------------
    # plan_long_strategy()
    # ------------------------------------------------------------------

    def plan_short_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """LONG-only setup — no short trades."""
        return None

    def plan_long_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        if event is None or event.side != "long":
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        levels = event.levels or {}
        gap_low = float(levels.get("gap_low", close))
        atr_val = float((event.context or {}).get("atr", close * 0.01))

        # Stop: min(gap_low_buffer, entry - ATR*1.5)
        stop_a = gap_low * (1.0 - self.gap_low_buffer_pct / 100.0)
        stop_b = close - atr_val * self.atr_stop_multiple
        hard_sl = min(stop_a, stop_b)
        risk_per_share = max(close - hard_sl, close * 1e-4)

        t1_target = close + self.t1_r_multiple * risk_per_share
        t2_target = close + self.t2_r_multiple * risk_per_share

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

        risk_params = RiskParams(
            hard_sl=hard_sl, risk_per_share=risk_per_share, atr=atr_val,
        )
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        # Plan-geometry validation
        try:
            _zone = compute_entry_zone(
                entry=close, bias="long",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, hard_sl, "long")
            enforce_min_stop_distance(
                close, hard_sl, self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{ctx.symbol}] capitulation_long_morning plan rejected: "
                f"{e.reason} {e.details}"
            )
            return None

        # Latch is set in detect() (worker-side) — not here.
        return TradePlan(
            symbol=ctx.symbol,
            side="long",
            structure_type=event.structure_type,
            entry_price=close,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context,
            trade_id=event.trade_id,
            target_anchor_type="r_multiple",  # was "arithmetic" — silent dispatch typo (2026-05-13)
        )

    # ------------------------------------------------------------------
    # BaseStructure abstract methods
    # ------------------------------------------------------------------

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext,
    ) -> RiskParams:
        return RiskParams(
            hard_sl=entry_price * (1.0 - self.stop_pct / 100.0),
            risk_per_share=entry_price * self.stop_pct / 100.0,
            atr=None,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
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
        return self.active_start <= t <= self.active_end
