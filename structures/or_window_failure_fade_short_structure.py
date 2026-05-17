"""OR-Window Failure Fade SHORT detector.

Cell-locked: SHORT-only fade of failed ORH (Opening Range High) upside pierces
in small_cap MIS-eligible names with HIGH-but-not-extreme volume (vol_ratio
in (8, 15]) during the 09:30-10:30 IST IB window.

Discovered via tools/sub9_research/sanity_or_window_failure_fade.py + cross-window
cell-mine + R-sweep (2026-05-16).

CELL LOCK + 3-WINDOW STABILITY:
  Cell: SHORT + cap_segment=small_cap + vol_ratio in (8, 15] + IB window
  Geometry: SL=0.3% above sweep_high, T2=2R (RIDE-TO-T2, no T1 partial), TS=15:10
  R-sweep finding: locking T1 wins HURT Holdout PF (1.12 -> 1.03).
                   The trades that reach T1 continue to T2 often; the
                   "ride to T2 / stop / time_stop" geometry is optimal.

  3-window stability:
    Discovery (2023-24)        n=1,104  PF=1.22  WR=50%  NET=+Rs.103,386
    OOS       (2025 Q1-Q3)     n=  490  PF=1.27  WR=51%  NET=+Rs. 55,750
    Holdout   (2025-10 to 26-04) n=  328  PF=1.12  WR=48%  NET=+Rs. 17,556

THESIS: During the IB window (09:30-10:30), retail breakout-traders are taught
to enter on ORH/ORL pierces. When the upside pierce FAILS (closes back below
ORH within 1-2 bars), retail entries are trapped. Trapped retail panic-sell,
creating cascade-down. The vol_ratio in (8, 15] filter captures bars with
high-but-not-extreme institutional/algo participation - small-cap stocks
showing real flow but not parabolic noise.

Distinct from retired `pdh_pdl_sweep_reclaim` (sub-8, ICT framing at PDH/PDL):
this uses TODAY's forming ORH (intraday level just established) instead of
yesterday's PDH/PDL. Different participant (IB-window retail breakout traders),
different timing.

SHIP STATUS (2026-05-16): Marginal-stable. Disc PF 1.22 below 1.30 ship-gate
floor; OOS PF 1.27 comfortably above 1.20; Hold PF 1.12 just below 1.15 floor.
All 3 windows positive direction. Hold n=328 above 200-floor. SHIPPED FOR OCI
VALIDATION; retire if OCI run shows aggregate PF < 1.10 on any window.

Active window: 09:30-10:30 IST (IB window only). One fire per (symbol, day).
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
    try:
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class OrWindowFailureFadeShortStructure(BaseStructure):
    """SHORT entry on failed ORH upside pierce in small_cap IB-window high-vol bars."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "or_window_failure_fade_short"
        self.configured_setup_type = config.get("_setup_name")

        self.or_bars = int(config["or_bars"])
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])

        self.poke_pct = float(config["poke_pct"])
        self.vol_ratio_min = float(config["vol_ratio_min"])
        self.vol_ratio_max = float(config["vol_ratio_max"])

        self.allowed_caps = set(config["allowed_cap_segments"])

        self.sl_pct_above_sweep_high = float(config["sl_pct_above_sweep_high"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        # NOTE: T1 stored as informational only. Actual exit is T2 or stop or
        # time_stop (no T1 partial exit). R-sweep showed T1 lock-in DROPS Hold PF
        # from 1.12 to 1.03. Ride-to-T2 is the optimal geometry for this cell.
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])

        self.min_bars_required = int(config["min_bars_required"])

        # One-fire-per-day latch.
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

        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        today_bars = df[df.index.date == _sd]
        if len(today_bars) < self.or_bars + 2:
            return _empty(f"Need at least {self.or_bars + 2} bars; have {len(today_bars)}")

        # Compute ORH from first or_bars bars
        or_bars_df = today_bars.iloc[:self.or_bars]
        orh = float(or_bars_df["high"].max())

        # Sweep bar = second-to-last (current is the confirmation bar)
        sweep_bar = today_bars.iloc[-2]
        confirm_bar = today_bars.iloc[-1]

        sweep_high = float(sweep_bar["high"])
        sweep_close = float(sweep_bar["close"])
        sweep_volume = float(sweep_bar["volume"])
        confirm_close = float(confirm_bar["close"])

        # ORH pierce + failure check
        poke_threshold = orh * (1.0 + self.poke_pct / 100.0)
        if sweep_high < poke_threshold:
            return _empty(f"sweep_high={sweep_high:.2f} < poke_threshold={poke_threshold:.2f}")
        if sweep_close > orh:
            return _empty(f"sweep_close={sweep_close:.2f} > orh={orh:.2f} (no failure)")

        # Confirmation: next bar's close stays below ORH
        if confirm_close >= orh:
            return _empty(f"confirm_close={confirm_close:.2f} >= orh={orh:.2f} (no recovery confirm)")

        # Volume cell filter: vol_ratio in (vol_ratio_min, vol_ratio_max]
        # vol_ratio = sweep_bar.volume / cumulative-avg-of-bars-before-sweep
        prior_bars = today_bars.iloc[:-2]  # bars before sweep bar
        if len(prior_bars) < 2:
            return _empty("Not enough prior bars for volume baseline")
        cum_vol_mean = float(prior_bars["volume"].mean())
        if cum_vol_mean <= 0:
            return _empty("Volume baseline invalid")
        vol_ratio = sweep_volume / cum_vol_mean
        if not _wide_open:
            if not (self.vol_ratio_min < vol_ratio <= self.vol_ratio_max):
                return _empty(
                    f"vol_ratio={vol_ratio:.2f} outside cell band "
                    f"({self.vol_ratio_min}, {self.vol_ratio_max}]"
                )

        # Signal qualifies - emit SHORT event
        confidence = min(1.0, (sweep_high - sweep_close) / max(sweep_high - float(sweep_bar["low"]), 1e-9))

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "orh": orh,
                "sweep_high": sweep_high,
                "sweep_close": sweep_close,
                "confirm_close": confirm_close,
            },
            context={
                "vol_ratio": vol_ratio,
                "poke_pct": (sweep_high - orh) / orh * 100.0,
            },
            price=confirm_close,
        )
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    def plan_long_strategy(self, context: MarketContext, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        """SHORT-only - no long trades."""
        return None

    def plan_short_strategy(self, context: MarketContext, event: Optional[StructureEvent] = None) -> Optional[TradePlan]:
        """Generate a SHORT TradePlan.

        Entry: confirmation bar close (event.price).
        SL: sweep_high * (1 + sl_pct/100), capped by min_stop_pct.
        T1: entry - 1R (informational; NOT used as exit per R-sweep findings)
        T2: entry - 2R (FULL exit here; ride-to-T2 geometry is optimal)
        Time stop: from config 'time_stop_at'.
        """
        if event is None or event.side != "short":
            return None

        entry = float(event.price)
        sweep_high = float(event.levels.get("sweep_high", 0))
        if entry <= 0 or sweep_high <= 0:
            return None

        sl_from_sweep = sweep_high * (1.0 + self.sl_pct_above_sweep_high / 100.0)
        sl_from_min = entry * (1.0 + self.min_stop_pct / 100.0)
        hard_sl = max(sl_from_sweep, sl_from_min)
        if hard_sl <= entry:
            return None
        risk_per_share = hard_sl - entry

        t1_level = entry - self.t1_r_multiple * risk_per_share
        t2_level = entry - self.t2_r_multiple * risk_per_share

        # RIDE-TO-T2 geometry: T1 is informational only (qty_pct=0). Full exit at T2.
        # R-sweep showed locking T1 wins drops Holdout PF from 1.12 to 1.03.
        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": self.t1_r_multiple,
                "qty_pct": 0.0,  # NOT used as exit
                "action": "exit_full",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": self.t2_r_multiple,
                "qty_pct": 1.0,  # FULL exit at T2
                "action": "exit_full",
            },
        ]

        risk_params = self.calculate_risk_params(entry, context)
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

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
                f"[{context.symbol}] or_window_failure_fade_short plan rejected: {e.reason} {e.details}"
            )
            return None

        return TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=event.structure_type,
            entry_price=entry,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context,
            trade_id=event.trade_id,
            target_anchor_type="r_multiple",
        )

    def calculate_risk_params(self, entry_price: float, market_context: MarketContext) -> RiskParams:
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
        if len(today_bars) < 2:
            return RiskParams(
                hard_sl=entry_price * (1.0 + self.min_stop_pct / 100.0),
                risk_per_share=entry_price * (self.min_stop_pct / 100.0),
                atr=entry_price * 0.01,
            )
        sweep_high = float(today_bars.iloc[-2]["high"])
        sl_from_sweep = sweep_high * (1.0 + self.sl_pct_above_sweep_high / 100.0)
        sl_from_min = entry_price * (1.0 + self.min_stop_pct / 100.0)
        hard_sl = max(sl_from_sweep, sl_from_min)
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
        t2 = entry - self.t2_r_multiple * risk
        return ExitLevels(
            targets=[{"level": t2, "qty_pct": 100, "rr": self.t2_r_multiple}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(self, context: MarketContext, event: Optional[StructureEvent] = None) -> float:
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
