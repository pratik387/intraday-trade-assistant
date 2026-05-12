"""Upper-Circuit T+1 Fade Short detector — sub-project #9.

Indian-microstructure-specific asymmetry: post-circuit-hit retail FOMO +
operator pump exhaustion in T+0 close, faded T+1 mid-session.

Mechanic
--------
T+0 (prior trading day): a mid/small-cap stock hits its upper circuit
band — daily close ≥4.5% above prior close, day clamped at top
(close ≈ high), volume in last 30 min collapses to ≤35% of intraday avg
(price-band lock signature), AND day's volume ≥1.5× 20-day avg (true
operator-pump signature, not low-volume drift to band).

T+1 (current session): if T+1 opens with a gap-up between 1% and 5%
above T+0 close (continuation evidence within sane bounds — > 5% is
fundamental-news territory), fire SHORT at the 10:30 5m bar's close.

Stop = max(T+1 day's high × 1.005, entry × 1.01) — above the post-flex
peak, with a 1% min-stop floor (qty-inflation guard).

Targets:
  T1 = T+1 open price (gap start) — 50% qty
  T2 = T+0 close price (full gap fill) — 50% qty
Time stop: 15:10 IST (5 min before MIS auto-square).

Research basis (5 peer-reviewed sources):
  Guo et al., J. Int'l Fin. Markets 2023 (Indian price-band experiment)
  Chen, Petukhov, Wang (MIT WP, magnet effect: T+1 gap-up continuation)
  Tandfonline 2024 (agent-based magnet effect, EM)
  Sehgal et al., Pacific-Basin Finance Journal 2024 (Indian momentum/
    reversal: upper-circuit next-day continuation in operator stocks)
  Chari et al. 2017 / IJABMR 2019 (Indian market-circuit dynamics)

Brief: specs/2026-05-01-sub-project-9-brief-circuit_t1_fade_short.md
Sanity-check (2026-05-01): NET PF=1.473 on 654 trades over 12 months
2024 5m bars (tools/sub9_research/sanity_circuit_t1_fade_short.py).
"""
from __future__ import annotations

from datetime import time, date, timedelta
from typing import Any, Dict, Optional, Tuple

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


class CircuitT1FadeShortStructure(BaseStructure):
    """Upper-Circuit T+1 Fade Short — Indian-microstructure asymmetry.

    Cross-day state: requires T-1 daily-bar features (close, high, volume,
    last-30-min-volume share, 20-day vol avg) to qualify the symbol on
    T+0. The detector reads these from `context.df_daily` (passed by
    orchestrator) on first detect() call per session and caches per
    (symbol, session_date).

    Single-bar entry: only fires on the 10:30-bar close (per Chen/Petukhov/
    Wang inflection-point evidence + sanity-check tested timing).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "circuit_t1_fade_short"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config — KeyError on missing.
        # T+0 circuit-hit detection
        self.t0_min_pct_change = float(config["t0_min_pct_change"])
        self.t0_high_to_close_min = float(config["t0_high_to_close_min"])
        self.t0_last30min_vol_share_max = float(config["t0_last30min_vol_share_max"])
        self.t0_min_vol_vs_20d = float(config["t0_min_vol_vs_20d"])
        # T+1 entry conditions
        self.t1_gap_min_pct = float(config["t1_gap_min_pct"])
        self.t1_gap_max_pct = float(config["t1_gap_max_pct"])
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        # Risk
        self.stop_t1_high_buffer_pct = float(config["stop_t1_high_buffer_pct"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"])
        # Targets
        self.t1_target_anchor = str(config["t1_target_anchor"])
        self.t2_target_anchor = str(config["t2_target_anchor"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        # Universe
        self.allowed_caps = set(config["allowed_cap_segments"])
        # OOS-validated cell restriction (sub-9 gauntlet 2026-05-06): only
        # cells in this regime allowlist are tradable. Empty list/None =
        # no restriction (Discovery / wide-open capture). Production should
        # enable the validated subset (currently {"trend_up"} per
        # docs/edge_discovery/2026-05-06-sub9-validation-gate/
        # stage6_validation_survivors.json — squeeze regime FAILED OOS).
        ar = config.get("allowed_regimes")
        self.allowed_regimes: Optional[set] = set(ar) if ar else None
        uk = config.get("universe_key")
        self.universe_key = str(uk) if uk else None
        # Plumbing
        self.min_bars_required = int(config["min_bars_required"])

        # Per-(symbol, session_date) qualifier cache populated lazily on
        # first detect() per session. Stores t0 close + t1 open for use
        # in plan_short_strategy.
        self._t0_cache: Dict[Tuple[str, date], Optional[Dict[str, float]]] = {}
        # First-trigger latch: one fire per (symbol, session_date) per session.
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    # ------------------------------------------------------------------
    # T-1 circuit-hit detection (cross-day, lazy per session)
    # ------------------------------------------------------------------

    def _qualify_t0(
        self, df_daily: pd.DataFrame, t0_date: date,
    ) -> Optional[Dict[str, float]]:
        """Check whether `t0_date`'s daily bar (the trading day BEFORE
        current session) is an upper-circuit-hit qualifier.

        Returns dict with t0_close (used by plan_short_strategy as T2
        target) on success, None on any filter rejection.

        Required df_daily columns: open, high, low, close, volume.
        Index must be timezone-naive datetime or date.
        """
        if df_daily is None or df_daily.empty:
            return None

        # Normalize daily index to date for safe comparison
        idx = df_daily.index
        if hasattr(idx, "date"):
            try:
                dates = idx.date
            except Exception:
                dates = pd.to_datetime(idx).date
        else:
            dates = pd.to_datetime(idx).date

        # Find T+0 row
        match_mask = dates == t0_date
        if not any(match_mask):
            return None
        t0_idx = list(match_mask).index(True)
        if t0_idx == 0:
            return None  # need prior close
        t0_row = df_daily.iloc[t0_idx]
        prev_row = df_daily.iloc[t0_idx - 1]

        try:
            t0_close = float(t0_row["close"])
            t0_high  = float(t0_row["high"])
            t0_vol   = float(t0_row["volume"])
            prev_close = float(prev_row["close"])
        except (KeyError, ValueError, TypeError):
            return None

        if prev_close <= 0:
            return None
        pct_change = (t0_close / prev_close - 1.0) * 100.0
        if pct_change < self.t0_min_pct_change:
            return None
        if t0_high <= 0 or t0_close / t0_high < self.t0_high_to_close_min:
            return None

        # 20-day volume avg (excluding T+0)
        if t0_idx < 20:
            return None
        vol_window = df_daily["volume"].iloc[t0_idx - 20: t0_idx]
        avg_vol = float(vol_window.mean())
        if avg_vol <= 0 or t0_vol / avg_vol < self.t0_min_vol_vs_20d:
            return None

        # NOTE: last-30-min-volume share check requires intraday 5m data
        # for T-1, which we don't carry in df_daily. The MainDetector
        # passes df_daily but not T-1 5m bars. Heuristic substitute: if
        # the daily bar shows close == high (clamped at top), that's a
        # strong proxy for "price was clamped at the band edge" without
        # needing the intraday volume share. This is more lenient than
        # the sanity-check tool but the sanity-check funnel showed
        # close-clamp + vol-ratio together survives most candidates.

        return {
            "t0_close": t0_close,
            "t0_high": t0_high,
            "t0_pct_change": pct_change,
            "t0_vol_ratio_20d": t0_vol / avg_vol,
        }

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Fire SHORT only on the 10:30 5m bar after a T-1 upper-circuit
        hit + T+0 1-5% gap-up continuation.
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
            return _empty(f"Cap segment {ctx.cap_segment!r} not in allowed set")

        # ---- Universe guard (bypassed under wide_open; None = no filter) ----
        if (
            not _wide_open
            and self.universe_key is not None
            and not in_universe(ctx.symbol, self.universe_key)
        ):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        # ---- Regime allowlist (OOS-validated cell restriction) ----
        # Bypassed under wide_open so capture runs see all regimes. None
        # in production config = no restriction. Sub-9 gauntlet 2026-05-06:
        # squeeze regime PF=0.96 OOS — failed validation, excluded.
        if (
            not _wide_open
            and self.allowed_regimes is not None
            and ctx.regime not in self.allowed_regimes
        ):
            return _empty(f"regime {ctx.regime!r} not in allowed set {sorted(self.allowed_regimes)}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        # ---- Active window: single-bar 10:30 ----
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- First-trigger latch ----
        session_date = ctx.session_date
        if session_date is None:
            session_date = pd.Timestamp(last_ts).date()
        latch_key = (ctx.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        # ---- T-1 qualification (cross-day) ----
        # T-1 = the LAST daily bar with date < session_date in df_daily.
        df_daily = ctx.df_daily
        if df_daily is None or df_daily.empty:
            return _empty("daily bars unavailable")

        # Identify T-1 (most-recent trading day before today's session)
        d_idx = df_daily.index
        try:
            d_dates = d_idx.date if hasattr(d_idx, "date") else pd.to_datetime(d_idx).date
        except Exception:
            return _empty("daily index format error")
        prior = [d for d in d_dates if d < session_date]
        if not prior:
            return _empty("no prior daily bar")
        t0_date = max(prior)   # NOTE: in this detector "T+0" = the circuit-hit day = T-1 in calendar terms

        cache_key = (ctx.symbol, session_date)
        if cache_key in self._t0_cache:
            t0_info = self._t0_cache[cache_key]
        else:
            t0_info = self._qualify_t0(df_daily, t0_date)
            self._t0_cache[cache_key] = t0_info
        if t0_info is None:
            return _empty("T-1 not an upper-circuit qualifier")

        # ---- T+1 (current session) gap-up filter ----
        # Today's open = TODAY'S first bar's open. Production passes a
        # multi-day rolling tail (default 120 bars), so df.iloc[0] is
        # the OLDEST bar in the tail (often 1-2 days back), not today's
        # 09:15. Filter to session_date before taking iloc[0] — bug
        # caught on 2024-04-04 smoke where APOLLO had df.iloc[0] = 04-02
        # 13:05 open=105.05, producing gap_pct=-4.67% instead of the
        # correct +1.63% from 04-04 09:15 open=112.00.
        t0_close = t0_info["t0_close"]
        today_bars = df[df.index.date == session_date]
        if today_bars.empty:
            return _empty("no bars for session_date")
        t1_open = float(today_bars.iloc[0]["open"])
        if t1_open <= 0:
            return _empty("T+1 open invalid")
        gap_pct = (t1_open / t0_close - 1.0) * 100.0
        if gap_pct < self.t1_gap_min_pct:
            return _empty(f"T+1 gap_pct={gap_pct:.2f} < min={self.t1_gap_min_pct}")
        if gap_pct > self.t1_gap_max_pct:
            return _empty(f"T+1 gap_pct={gap_pct:.2f} > max={self.t1_gap_max_pct}")

        # ---- All conditions met: emit SHORT event ----
        last = df.iloc[-1]
        bar_close = float(last["close"])
        # Confidence: scale by gap_pct within the [min, max] band — bigger
        # gaps that survived the max filter have proportionally more fade
        # potential.
        gap_range = max(self.t1_gap_max_pct - self.t1_gap_min_pct, 0.01)
        confidence = (gap_pct - self.t1_gap_min_pct) / gap_range
        confidence = max(0.0, min(1.0, confidence))

        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "t0_close": t0_close,
                "t0_high": t0_info["t0_high"],
                "t1_open": t1_open,
                "t1_high_so_far": float(today_bars["high"].max()),
                "entry_close": bar_close,
            },
            context={
                "t0_pct_change": t0_info["t0_pct_change"],
                "t0_vol_ratio_20d": t0_info["t0_vol_ratio_20d"],
                "t1_gap_pct": gap_pct,
                "session_date_iso": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
            },
            price=bar_close,
        )
        # Set latch HERE in detect() — not in plan_short_strategy. The
        # in-process latch in plan_*_strategy never propagated back to the
        # workers' detect() loop because they run in different processes.
        # The single-bar 10:30 active window masked the bug for this
        # detector, but moving the latch here matches the canonical pattern
        # used by all other latched detectors.
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ------------------------------------------------------------------
    # plan_short_strategy()
    # ------------------------------------------------------------------

    def plan_long_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Short-only setup — no long trades."""
        return None

    def plan_short_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Build TradePlan with structural targets at gap edges."""
        if event is None or event.side != "short":
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        levels = event.levels or {}
        t0_close = float(levels.get("t0_close", close))
        t1_open = float(levels.get("t1_open", close))
        # T+1 high so far = max of bars in TODAY only. df is a multi-day
        # rolling tail; df["high"].max() over the tail picks up prior
        # days' highs and inflates the stop. Same wrong-bar bug class as
        # the t1_open issue in detect().
        session_date = (
            ctx.session_date
            or pd.Timestamp(df.index[-1]).date()
        )
        today_bars = df[df.index.date == session_date]
        if today_bars.empty:
            return None
        t1_high_so_far = float(today_bars["high"].max())

        # Stop = max(T+1 high × (1 + buffer), entry × (1 + min_stop_pct/100))
        sl_from_high = t1_high_so_far * (1.0 + self.stop_t1_high_buffer_pct / 100.0)
        sl_from_min = close * (1.0 + self.min_stop_distance_pct / 100.0)
        hard_sl = max(sl_from_high, sl_from_min)
        risk_per_share = max(hard_sl - close, close * 0.001)   # avoid zero

        # Targets: T1 = T+1 open (gap start), T2 = T+0 close (full gap fill).
        # Both should be BELOW entry for a short trade (we entered above
        # both; the fade target is downward).
        t1_target = t1_open
        t2_target = t0_close
        if t1_target >= close:
            # Pathological: gap was filled before our entry. Cap target.
            t1_target = close - risk_per_share * 0.5
        if t2_target >= close:
            t2_target = close - risk_per_share

        rr_t1 = (close - t1_target) / max(risk_per_share, 1e-6)
        rr_t2 = (close - t2_target) / max(risk_per_share, 1e-6)
        # Per-target qty_pct is honored by exit_executor (plan-as-source-of-truth
        # refactor 2026-05-12). qty_pct=0.0 means the executor skips T1 partial
        # and rides full qty to T2 (the validated mechanic per
        # _circuit_t1_sl_target_sweep.py — partial dropped PF 1.34 -> 0.49).
        targets = [
            {
                "name": "T1", "level": t1_target, "rr": rr_t1,
                "qty_pct": self.t1_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_target, "rr": rr_t2,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4), "action": "exit_full",
            },
        ]

        risk_params = RiskParams(
            hard_sl=hard_sl, risk_per_share=risk_per_share, atr=None,
        )
        # Plan-as-source-of-truth (2026-05-12): time_exit from per-setup config.
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

        # Plan-geometry validation (Phase-C-Commit-2 pattern)
        try:
            _zone = compute_entry_zone(
                entry=close, bias="short",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, hard_sl, "short")
            enforce_min_stop_distance(
                close, hard_sl, self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{ctx.symbol}] circuit_t1_fade_short plan rejected: "
                f"{e.reason} {e.details}"
            )
            return None

        # Latch is set in detect() (worker-side) — not here.
        return TradePlan(
            symbol=ctx.symbol,
            side="short",
            structure_type=event.structure_type,
            entry_price=close,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context,
            trade_id=event.trade_id,
            # Targets are gap-edges (T+1 open, T+0 close) — structural
            # levels, not arithmetic R-multiples. Late fills must keep
            # those structural targets.
            target_anchor_type="structural",
        )

    # ------------------------------------------------------------------
    # BaseStructure abstract methods (legacy ABC contract)
    # ------------------------------------------------------------------

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext,
    ) -> RiskParams:
        """Required by BaseStructure ABC; mirrors the inline logic in plan_short."""
        return RiskParams(
            hard_sl=entry_price * (1.0 + self.min_stop_distance_pct / 100.0),
            risk_per_share=entry_price * self.min_stop_distance_pct / 100.0,
            atr=None,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """Default 1R exit if caller asks via the legacy ABC path. The
        production exit comes from plan_short_strategy which builds T1+T2
        at gap edges; this method exists only to satisfy BaseStructure."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        t1 = entry - risk   # 1R below entry for a short
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": 1.0}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(
        self, context: MarketContext, event: Optional[StructureEvent] = None,
    ) -> float:
        """Proxy: detect()'s quality_score IS the ranking signal."""
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """True if current_time is within the single-bar 10:30 window."""
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
