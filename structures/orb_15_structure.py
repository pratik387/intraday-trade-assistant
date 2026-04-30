"""ORB-15 (Opening Range Breakout, first 15 minutes) — sub8 Setup #1 (REDESIGN).

Redesign date: 2026-04-29 per specs/2026-04-29-orb_15-redesign-plan.md.

The original mechanic ("first 5m bar that closes outside the opening range")
was empirically broken: 47,754 trades over 2023-2024 produced gross_PF 1.052
but net_PF 0.797 — the thin gross edge couldn't survive 9.4-bps fee drag.
Re-fire spam (same symbol-day fired 2-20× per the broken trigger) plus
universal regime weakness made it un-rescuable via filtering alone (Optuna
30-trial whiff confirmed).

This redesign replaces the trigger with sweep+reclaim and adds three
documented Indian-pro pre-conditions:

  - Sweep+reclaim trigger (Sahi.com / DailyPriceAction liquidity-sweep
    literature): a bar must FIRST penetrate the OR (high > range_high for
    long, low < range_low for short) AND close back inside the range — the
    sweep candle. The CURRENT bar then closing back beyond the range fires
    the trigger. Standalone close-outside-range no longer fires.
  - NR7 pre-condition (Crabel; Indian productizations HDFC Sky /
    eLearnMarkets / Streak): today's OR range must be ≤ nr7_multiplier ×
    min(prior 7 sessions' OR ranges).
  - Relative-volume "in play" filter (QuantConnect 2.4-Sharpe finding,
    adapted for NSE F&O liquid 200): cumulative volume in the OR window
    must be ≥ min_rvol_at_or_close × prior 14-day mean of OR-window volume.

Plus three environmental gates:
  - Friday exclusion (research findings §4.1 — gross_PF 0.925 on Fri vs >1.05
    on every other DOW).
  - Regime allowlist (explicit-allowlist semantics — empty list = none).
  - Side allowlist (same semantics — supports SELL-only experiments).

First-trigger latch eliminates re-fire spam: per (symbol, side, session_date)
only the FIRST trigger fires. Latch is REGISTERED in plan_*_strategy on
success (not in detect()) so internal re-runs of detect() from _build_plan
do not self-block.

Sources:
  - specs/2026-04-28-research-orb_15-findings.md (47K-trade diagnosis,
    SELL × squeeze × ≤09:50 cell at gross_PF 1.624 / net_PF 1.298 / n=206
    OOS-stable)
  - specs/2026-04-28-research-orb_15-indian-pro-mechanics.md (mechanic
    citations: sweep+reclaim, NR7, rvol)

Universe: F&O liquid 200 (per design doc Section 3.2).
Stop: opposite-end-of-range (rev2 default; mid-range A/B variant) ± wick_buffer.
Targets: T1 at 1R (50% qty), T2 at 2R (50% qty).
Gap-day exclusion: if open gap from PDC > max_gap_pct_for_orb, route to gap_fade_short.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, List, Optional, Set, Tuple

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


class ORB15Structure(BaseStructure):
    """Opening Range Breakout, 15-min range, sweep+reclaim trigger (redesign 2026-04-29)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "orb_15"
        self.configured_setup_type = config.get("_setup_name")

        # --- Existing keys (preserved) ---
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

        # --- Redesign keys (Phase 0, 2026-04-29) ---
        # Per CLAUDE.md rule 1: every parameter from config (KeyError on missing).
        self.sweep_reclaim_lookback_bars = int(config["sweep_reclaim_lookback_bars"])
        self.nr7_lookback_days = int(config["nr7_lookback_days"])
        self.nr7_multiplier = float(config["nr7_multiplier"])
        self.min_rvol_at_or_close = float(config["min_rvol_at_or_close"])
        self.rvol_baseline_lookback_days = int(config["rvol_baseline_lookback_days"])
        # Day-of-week exclusion: list of integers (Mon=0, Sun=6).
        self.exclude_dow: Set[int] = set(int(d) for d in config["exclude_dow"])
        # Explicit-allowlist semantics: empty list = NOTHING allowed (safer
        # for live trading; misconfiguration fails closed). Use ["trend_up",
        # "trend_down", "chop", "squeeze"] for "all regimes" / ["long",
        # "short"] for "both sides".
        self.allowed_regimes: Set[str] = set(config["allowed_regimes"])
        self.allowed_sides: Set[str] = set(config["allowed_sides"])

        # --- First-trigger latch state ---
        # Key = (symbol, side, session_date_iso). Cleared at session boundary
        # (when session_date changes between detect() calls). Registered in
        # plan_*_strategy on success — NOT in detect() — so internal re-runs
        # of detect() from _build_plan don't self-block.
        self._fired_keys: Set[Tuple[str, str, str]] = set()
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

    def _get_median_volume(self, ctx: MarketContext) -> float:
        if ctx.indicators and "median_volume_30d" in ctx.indicators:
            return float(ctx.indicators["median_volume_30d"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 5:
            return float(ctx.df_5m["volume"].iloc[:-1].mean())
        return 0.0

    def _get_prior_or_ranges(self, ctx: MarketContext) -> Optional[List[float]]:
        """Return prior `nr7_lookback_days` OR-window high-low ranges.

        Reads from `ctx.indicators["prior_or_ranges_7d"]` first (orchestrator
        is expected to populate this from a precomputed feature). Falls back
        to computing from `ctx.df_daily` if it has 09:15-09:30 high/low data
        (uncommon — daily bars usually don't preserve intraday OR window).
        Returns None if neither source is available.
        """
        if ctx.indicators is not None:
            v = ctx.indicators.get("prior_or_ranges_7d")
            if isinstance(v, (list, tuple)) and len(v) >= self.nr7_lookback_days:
                return [float(x) for x in v[: self.nr7_lookback_days]]
        # df_daily fallback is intentionally limited — daily bars don't carry
        # intraday OR. We don't synthesize from non-OR-bounded data.
        return None

    def _get_or_volume_baseline(self, ctx: MarketContext) -> Optional[float]:
        """Return prior `rvol_baseline_lookback_days` mean of OR-window volume.

        Reads from `ctx.indicators["or_window_volume_baseline_14d"]` first.
        Returns None if missing.
        """
        if ctx.indicators is not None:
            v = ctx.indicators.get("or_window_volume_baseline_14d")
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return None
        return None

    def _maybe_reset_latch(self, session_date_iso: str) -> None:
        """Clear fired_keys if session boundary crossed since last call."""
        if session_date_iso != self._latch_session_date:
            self._fired_keys.clear()
            self._latch_session_date = session_date_iso

    def _register_latch(self, symbol: str, side: str, session_date_iso: str) -> None:
        """Mark (symbol, side, session_date) as fired — called from plan_*_strategy."""
        self._fired_keys.add((symbol, side, session_date_iso))

    def _detect_sweep_reclaim(
        self,
        df: pd.DataFrame,
        range_high: float,
        range_low: float,
    ) -> Optional[str]:
        """Scan for sweep+reclaim sequence ending on the CURRENT bar.

        Returns "long" if an upside sweep+reclaim fires, "short" if downside,
        None otherwise. The current bar (df.iloc[-1]) is the reclaim trigger;
        a sweep candle must exist within the prior `sweep_reclaim_lookback_bars`.

        Sweep bar (long): high > range_high AND range_low <= close <= range_high.
        Reclaim trigger (long): current bar close > range_high.
        Mirror for short.
        """
        if df is None or len(df) < 2:
            return None
        last = df.iloc[-1]
        last_close = float(last["close"])

        # Determine reclaim direction from the current bar's close.
        if last_close > range_high:
            # Look for an upside sweep candle in the lookback window (excluding the current bar).
            lookback = df.iloc[-(self.sweep_reclaim_lookback_bars + 1):-1]
            for _, bar in lookback.iterrows():
                bar_high = float(bar["high"])
                bar_close = float(bar["close"])
                if bar_high > range_high and range_low <= bar_close <= range_high:
                    return "long"
            return None
        if last_close < range_low:
            lookback = df.iloc[-(self.sweep_reclaim_lookback_bars + 1):-1]
            for _, bar in lookback.iterrows():
                bar_low = float(bar["low"])
                bar_close = float(bar["close"])
                if bar_low < range_low and range_low <= bar_close <= range_high:
                    return "short"
            return None
        return None

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        # ---- Universe (design-inferred — bypassed under wide_open) ----
        # Per master plan: wide-open OCI capture must see ALL symbols so the
        # gauntlet can decide which universe slice the detector works in.
        if not _wide_open and not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        # ---- Bars + active window (always enforced — mechanical) ----
        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- Latch session-boundary reset ----
        # session_date format: ISO date string (IST-naive — no tzinfo).
        session_date_iso = pd.Timestamp(ctx.session_date).strftime("%Y-%m-%d")
        self._maybe_reset_latch(session_date_iso)

        # ---- DOW exclusion (design-inferred — bypassed under wide_open) ----
        # Use session_date.weekday() rather than last_ts.weekday() to be robust
        # against pre-session-open ticks (session_date is the trading-day, not
        # the bar timestamp).
        if not _wide_open:
            try:
                dow = pd.Timestamp(ctx.session_date).weekday()
            except Exception:
                dow = last_ts.weekday()
            if dow in self.exclude_dow:
                return _empty(f"dow_excluded: dow={dow} in {sorted(self.exclude_dow)}")

        # ---- Regime gate (design-inferred; explicit-allowlist semantics) ----
        if not _wide_open:
            regime = ctx.regime
            if regime not in self.allowed_regimes:
                return _empty(
                    f"regime_not_allowed: regime={regime!r} not in {sorted(self.allowed_regimes)}"
                )

        # ---- Compute opening range from bars in [range_start, range_end) ----
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

        # ---- Gap-day exclusion (always enforced — routes to gap_fade_short) ----
        if ctx.pdc is not None and float(ctx.pdc) > 0:
            gap_pct = abs(opening_price - float(ctx.pdc)) / float(ctx.pdc) * 100.0
            if gap_pct > self.max_gap_pct_for_orb:
                return _empty(
                    f"gap_day_routed_to_gap_fade: gap_pct={gap_pct:.2f}>{self.max_gap_pct_for_orb}"
                )

        # ---- NR7 pre-condition (design-inferred) ----
        if not _wide_open:
            prior_ranges = self._get_prior_or_ranges(ctx)
            if prior_ranges is None or len(prior_ranges) < self.nr7_lookback_days:
                return _empty("nr7_data_unavailable")
            today_or_range = range_high - range_low
            nr7_floor = min(prior_ranges)
            if today_or_range > self.nr7_multiplier * nr7_floor:
                return _empty(
                    f"nr7_violated: today={today_or_range:.4f} > "
                    f"{self.nr7_multiplier}×min_prior({nr7_floor:.4f})"
                )

        # ---- Relative-volume "in play" filter (design-inferred) ----
        if not _wide_open:
            rvol_baseline = self._get_or_volume_baseline(ctx)
            if rvol_baseline is None:
                return _empty("rvol_baseline_unavailable")
            if rvol_baseline <= 0:
                return _empty("rvol_baseline_invalid")
            today_or_vol = float(range_bars["volume"].sum())
            rvol = today_or_vol / rvol_baseline
            if rvol < self.min_rvol_at_or_close:
                return _empty(f"rvol={rvol:.2f}<{self.min_rvol_at_or_close}")

        # ---- Sweep+reclaim trigger (always enforced — core mechanic) ----
        side = self._detect_sweep_reclaim(df, range_high, range_low)
        if side is None:
            last_close = float(df.iloc[-1]["close"])
            return _empty(
                f"no_sweep_reclaim: close={last_close:.2f} range=[{range_low:.2f},{range_high:.2f}]"
            )

        # ---- Side allowlist (design-inferred; check AFTER side determined) ----
        if not _wide_open and side not in self.allowed_sides:
            return _empty(
                f"side_not_allowed: side={side!r} not in {sorted(self.allowed_sides)}"
            )

        # ---- First-trigger latch CHECK (always enforced) ----
        # Note: registration happens in plan_*_strategy on success, so detect()
        # called multiple times by _build_plan returns the same event.
        latch_key = (ctx.symbol, side, session_date_iso)
        if latch_key in self._fired_keys:
            return _empty(f"already_fired: {latch_key[0]}/{latch_key[1]}/{latch_key[2]}")

        # ---- Build event ----
        last = df.iloc[-1]
        bar_close = float(last["close"])
        bar_vol = float(last["volume"])
        median_vol = self._get_median_volume(ctx)
        confidence = min(1.0, range_pct / 1.0)
        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "range_high": range_high,
                "range_low": range_low,
                "range_mid": (range_high + range_low) / 2.0,
                "close": bar_close,
            },
            context={
                "range_pct": range_pct,
                "vol_x_median": bar_vol / median_vol if median_vol > 0 else 0.0,
                "session_date": session_date_iso,
            },
            price=bar_close,
        )
        return StructureAnalysis(
            structure_detected=True, events=[evt], quality_score=confidence * 100.0,
        )

    def _build_plan(self, ctx: MarketContext, side: str, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        # Architectural rule (2026-04-30): no re-detect inside _build_plan.
        # Caller (orchestrator → plan_*_strategy) MUST pass the StructureEvent
        # produced by MainDetector. None is a programming bug — fail fast.
        if event is None:
            return None
        evt = event
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
        risk_params = RiskParams(
            hard_sl=hard_sl, risk_per_share=risk, atr=self._get_atr(ctx),
        )
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        # === Plan-geometry validation (Phase-C-Commit-2) ===
        try:
            _zone = compute_entry_zone(
                entry=close, bias=side,
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, hard_sl, side)
            enforce_min_stop_distance(close, hard_sl, self.config.get("min_stop_distance_pct"))
        except PlanRejected as e:
            logger.warning(
                f"[{ctx.symbol}] orb_15 plan rejected: {e.reason} {e.details}"
            )
            return None

        plan = TradePlan(
            symbol=ctx.symbol, side=side, structure_type=self.structure_type,
            entry_price=close, risk_params=risk_params, exit_levels=exit_levels,
            qty=0, notional=0.0, confidence=evt.confidence, notes=evt.context,
            trade_id=evt.trade_id,
            # ORB targets are ORH/ORL ± k×OR_range — structurally meaningful
            # levels, not arithmetic R-multiples. Preserve target levels on
            # actual-entry re-anchor. Switch to "or_range" later if/when we
            # decide to re-enable explicit OR-range recalc with config knobs.
            target_anchor_type="structural",
        )

        # Latch registration on plan success — separated from detect() so
        # internal re-runs of detect() from _build_plan don't self-block.
        # session_date_iso lives in evt.context (set at detect time).
        session_date_iso = evt.context.get("session_date")
        if session_date_iso:
            self._register_latch(ctx.symbol, side, session_date_iso)

        return plan

    def plan_long_strategy(self, ctx: MarketContext, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long", event=event)

    def plan_short_strategy(self, ctx: MarketContext, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short", event=event)

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
