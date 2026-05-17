"""Long Panic-Gap-Down detector — small/mid cap MIS-eligible LONG mean-revert.

Discovered via tools/sub9_research/sanity_long_panic_gap_down.py + the cell
sweep in tools/sub9_research/_long_panic_gap_down_sweep_cellmine.py.

Cell B + regime guard validated across 3 windows (2026-05-15):
  Disc    PF=1.45  n=1,412  pos_months=17-18/24   (T1=1.5R / T2=2.5R)
  OOS     PF=1.40  n=  333  pos_months=  6-7/9
  Hold    PF=1.72  n=  126  pos_months=    6/7

Thesis: in normal mean-reverting regimes, small/mid MIS-eligible names that
gap down hard (>=1%) at open AND close the first 5m bar 3-5% below the prior
day low AND 5.5%+ below prior day high tend to bounce back toward the prior
day's range — driven by stop-loss cascades exhausting and dip-buyer demand.
In panic regimes (e.g. war months) the gap-downs continue trending; that
failure mode is the reason for the upstream regime guard (see WARNING below).

Active window: 09:15-09:20 IST (first 5m bar only — entry at 09:15 bar close).

WARNING — regime guard is NOT in this detector.
  The sanity result (PF 0.73 -> 1.35 on Holdout) depends on a UNIVERSE-wide
  regime guard: "by 09:20, count gap-down triggers (gap<=-1%, dist_pdh<=-5.5%,
  dist_pdl in [-5%,-3%]) across the small/mid MIS universe; skip ALL entries
  if count > 80". This requires cross-symbol coordination at the orchestrator
  level — a single detector instance only sees one symbol's context.
  Until that plumbing exists, this detector ships with enabled=false and
  documents the requirement in its config _status block. Do not enable in
  live without wiring the regime guard upstream.
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
from services import regime_density_tracker
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


class LongPanicGapDownStructure(BaseStructure):
    """Long entry on small/mid-cap deep panic gap-down (Cell B + 1.5R/2.5R targets)."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "long_panic_gap_down"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])

        self.gap_pct_max = float(config["gap_pct_max"])
        self.dist_from_pdh_pct_max = float(config["dist_from_pdh_pct_max"])
        self.dist_from_pdl_pct_min = float(config["dist_from_pdl_pct_min"])
        self.dist_from_pdl_pct_max = float(config["dist_from_pdl_pct_max"])
        self.allowed_caps = set(config["allowed_cap_segments"])

        self.sl_buffer_below_bar_low_pct = float(config["sl_buffer_below_bar_low_pct"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])
        self.min_bars_required = int(config["min_bars_required"])

        # Regime guard: dist_from_pdl threshold for "broader universe" count
        # (looser than narrow Cell B band, matches sanity n_triggers_today).
        self.broader_dist_from_pdl_pct_max = float(config["broader_dist_from_pdl_pct_max"])
        self.regime_guard_max_density = int(config["regime_guard_n_triggers_today_max"])

        # First-trigger latch — one fire per (symbol, session_date).
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
        pdh = float(context.pdh) if context.pdh is not None else None
        pdl = float(context.pdl) if context.pdl is not None else None
        if pdc is None or pdc <= 0:
            return _empty("PDC unavailable")
        if pdh is None or pdh <= 0:
            return _empty("PDH unavailable")
        if pdl is None or pdl <= 0:
            return _empty("PDL unavailable")

        # df_5m may contain warmup bars from prior sessions. Filter to today's
        # bars before reading the 09:15 bar — without this, df.iloc[0] is a
        # stale prior-session bar and SL/gap geometry breaks.
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return _empty("No bars for session date")
        first_bar = today_bars.iloc[0]
        bar_open = float(first_bar["open"])
        bar_low = float(first_bar["low"])
        bar_close = float(first_bar["close"])

        gap_pct = ((bar_open - pdc) / pdc) * 100.0
        if gap_pct > self.gap_pct_max:
            return _empty(f"gap_pct={gap_pct:.2f} > max={self.gap_pct_max}")

        dist_from_pdh_pct = ((bar_close - pdh) / pdh) * 100.0
        if dist_from_pdh_pct > self.dist_from_pdh_pct_max:
            return _empty(
                f"dist_from_pdh={dist_from_pdh_pct:.2f} > max={self.dist_from_pdh_pct_max}"
            )

        dist_from_pdl_pct = ((bar_close - pdl) / pdl) * 100.0

        # ---- Regime density guard ----
        # Note this symbol if it passes the BROADER filter (looser dist_pdl
        # band than the narrow Cell B). Sanity counted the broader universe
        # to threshold panic-density at 80; do the same here.
        broader_qualifies = dist_from_pdl_pct <= self.broader_dist_from_pdl_pct_max
        if broader_qualifies:
            regime_density_tracker.note(
                self.structure_type, session_date, context.symbol,
            )
            # Suppress fires once the day's count crosses threshold. The check
            # happens AFTER note() so density reflects this symbol's contribution
            # — symmetric across all symbols regardless of processing order.
            density = regime_density_tracker.get_density(
                self.structure_type, session_date,
            )
            if density > self.regime_guard_max_density:
                return _empty(
                    f"regime guard: density={density} > max={self.regime_guard_max_density}"
                )

        if not (self.dist_from_pdl_pct_min <= dist_from_pdl_pct <= self.dist_from_pdl_pct_max):
            return _empty(
                f"dist_from_pdl={dist_from_pdl_pct:.2f} not in "
                f"[{self.dist_from_pdl_pct_min}, {self.dist_from_pdl_pct_max}]"
            )

        # Confidence proxy: deeper-below-PDL within the configured band => higher.
        depth_in_band = (self.dist_from_pdl_pct_max - dist_from_pdl_pct) / max(
            self.dist_from_pdl_pct_max - self.dist_from_pdl_pct_min, 1e-9
        )
        confidence = max(0.0, min(1.0, depth_in_band))

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="long",
            confidence=confidence,
            levels={
                "pdc": pdc,
                "pdh": pdh,
                "pdl": pdl,
                "entry_bar_low": bar_low,
                "entry_bar_close": bar_close,
            },
            context={
                "gap_pct": gap_pct,
                "dist_from_pdh_pct": dist_from_pdh_pct,
                "dist_from_pdl_pct": dist_from_pdl_pct,
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

    def plan_short_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Long-only setup — no short trades."""
        return None

    def plan_long_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a LONG TradePlan.

        Entry: close of 09:15 bar (=09:20 mark, current_price).
        Stop:  the deeper of (entry_bar_low × (1 - buf), entry × (1 - min_stop_pct/100)).
        Targets: T1 = entry + t1_r_multiple*R (50% qty), T2 = entry + t2_r_multiple*R (remainder).
        Time stop: from config 'time_stop_at' (executor honors plan).
        """
        if event is None:
            return None
        evt = event
        if evt.side != "long":
            return None

        df = context.df_5m
        if df is None or df.empty:
            return None
        # Filter df_5m to today's bars only — warmup bars from prior days
        # otherwise pollute df.iloc[0]/[-1] and break SL/target geometry.
        session_date = context.session_date
        if session_date is None:
            session_date = pd.Timestamp(df.index[-1]).date()
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return None
        first_bar = today_bars.iloc[0]
        entry_bar_low = float(first_bar["low"])
        # Entry is at the 09:15 bar's close (= 09:20 price). The setup fires
        # only on the first bar of the session per active_window, so the
        # 'first bar of today' IS the entry bar.
        close = float(first_bar["close"])

        sl_from_low = entry_bar_low * (1.0 - self.sl_buffer_below_bar_low_pct / 100.0)
        sl_from_min = close * (1.0 - self.min_stop_pct / 100.0)
        hard_sl = min(sl_from_low, sl_from_min)
        if hard_sl >= close:
            return None
        risk_per_share = close - hard_sl

        t1_level = close + self.t1_r_multiple * risk_per_share
        t2_level = close + self.t2_r_multiple * risk_per_share
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

        risk_params = self.calculate_risk_params(close, context)
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

        # Plan-geometry validation
        try:
            _zone = compute_entry_zone(
                entry=close, bias="long",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, risk_params.hard_sl, "long")
            enforce_min_stop_distance(
                close, risk_params.hard_sl,
                self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{context.symbol}] long_panic_gap_down plan rejected: {e.reason} {e.details}"
            )
            return None

        return TradePlan(
            symbol=context.symbol,
            side="long",
            structure_type=evt.structure_type,
            entry_price=close,
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
                hard_sl=entry_price * (1.0 - self.min_stop_pct / 100.0),
                risk_per_share=entry_price * (self.min_stop_pct / 100.0),
                atr=entry_price * 0.01,
            )
        # Filter to today's bars before reading the entry bar.
        session_date = market_context.session_date
        if session_date is None:
            session_date = pd.Timestamp(df.index[-1]).date()
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return RiskParams(
                hard_sl=entry_price * (1.0 - self.min_stop_pct / 100.0),
                risk_per_share=entry_price * (self.min_stop_pct / 100.0),
                atr=entry_price * 0.01,
            )
        entry_bar_low = float(today_bars.iloc[0]["low"])
        sl_from_low = entry_bar_low * (1.0 - self.sl_buffer_below_bar_low_pct / 100.0)
        sl_from_min = entry_price * (1.0 - self.min_stop_pct / 100.0)
        hard_sl = min(sl_from_low, sl_from_min)
        stop_distance = max(entry_price - hard_sl, entry_price * 0.001)
        atr_proxy = float((today_bars["high"] - today_bars["low"]).mean())
        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=stop_distance,
            atr=atr_proxy,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(entry - hard_sl)
        t1 = entry + self.t1_r_multiple * risk
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
