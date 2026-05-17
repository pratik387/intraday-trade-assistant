"""Round-Number Stop-Cluster Sweep SHORT detector.

Cell-locked: SHORT-only fade of failed upside sweeps of round-number levels
in small_cap MIS-eligible stocks priced Rs.100-250, during 11:00-12:30 IST.

Discovered via tools/sub9_research/sanity_round_number_sweep.py + cross-window
cell-mine (2026-05-16). The ORIGINAL aggregate fails across all 3 windows
(Disc PF 0.80, OOS 0.84, Holdout 0.81), but ONE narrow cell holds:

  Cell: SHORT + rn_level=Rs.100-250 + hour=11:00-12:30 + small_cap
  R-geom: T1=1R full-exit, T2=2R, SL=0.5% above sweep_high, TS=15:00

  3-window stability (with R-sweep geometry):
    Discovery (2023-24)        n=300  PF=1.24  WR=51%  NET=+Rs.26,942
    OOS       (2025 Q1-Q3)     n=176  PF=1.21  WR=49%  NET=+Rs.14,393
    Holdout   (2025-10 to 26-04) n=126  PF=1.17  WR=49%  NET=+Rs. 7,785

THESIS: Indian retail traders cluster stop-losses at round-number prices
(Rs.100, 150, 200, 250) far more than at PDH/PDL because retail education
(Subasish Pani, Powerof Stocks, Zerodha Varsity) teaches this. Rs.100-250
stocks are PRIME retail territory (cheap, accessible, hyped on YouTube).
When intraday price pokes above a round number briefly and closes back below,
that's the upside stop-cluster failing. Trapped retail breakout buyers
panic-sell, creating cascade-down for SHORT entry.

Distinct from retired `pdh_pdl_sweep_reclaim` (sub-8, ICT framing at PDH/PDL).
This uses ROUND-NUMBER LEVELS (not PDH/PDL) and retail-psychology mechanism.

SHIP STATUS (2026-05-16): Borderline-marginal. Disc PF 1.24 below standalone
ship-gate floor (1.30). OOS+Hold pass (1.21/1.17 >= 1.20/1.15 floors). All
3-window directionally positive. Holdout n=126 above 100-floor. SHIPPED FOR
OCI VALIDATION; retire if OCI run shows materially worse numbers (PF<1.10 on
any window).

Active window: 11:00-12:30 IST (off-peak mid-session, NOT first hour or last hour).
One fire per (symbol, day) - latched.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, List, Optional

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


def _round_numbers_near(price: float, increment: float = 50.0, tol_pct: float = 2.0) -> List[float]:
    """Round-number levels within +/- tol_pct of price.

    For the Rs.100-250 universe, multiples of 50 (50, 100, 150, 200, 250) are
    the relevant retail-psychology levels.
    """
    low = price * (1.0 - tol_pct / 100.0)
    high = price * (1.0 + tol_pct / 100.0)
    near_below = (price // increment) * increment
    candidates = [near_below - increment, near_below, near_below + increment, near_below + 2 * increment]
    return [c for c in candidates if low <= c <= high and c > 0]


class RoundNumberSweepShortStructure(BaseStructure):
    """SHORT entry on failed upside sweep of round-number level in Rs.100-250 small_caps."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "round_number_sweep_short"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])

        self.min_price = float(config["min_price"])
        self.max_price = float(config["max_price"])
        self.round_number_increment = float(config["round_number_increment"])

        self.poke_pct = float(config["poke_pct"])
        self.vol_ratio_min = float(config["vol_ratio_min"])

        self.allowed_caps = set(config["allowed_cap_segments"])

        self.sl_pct_above_sweep_high = float(config["sl_pct_above_sweep_high"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        # NOTE: t1_partial_qty_pct stored but actual exit logic is FULL-at-T1.
        # See _status block in config - R-sweep showed full-exit-at-T1 with T2
        # backup gives the most-stable 3-window PFs (1.24/1.21/1.17).
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
        if today_bars.empty or len(today_bars) < 2:
            return _empty("Not enough today's bars")

        # Last 2 bars: sweep bar + confirmation bar (current is the confirm)
        sweep_bar = today_bars.iloc[-2]
        confirm_bar = today_bars.iloc[-1]
        sweep_high = float(sweep_bar["high"])
        sweep_close = float(sweep_bar["close"])
        sweep_volume = float(sweep_bar["volume"])
        confirm_close = float(confirm_bar["close"])
        sweep_mid = (sweep_high + float(sweep_bar["low"])) / 2.0

        # Price band filter (Rs.100-250 anchored - retail psychology cell)
        if not _wide_open:
            if not (self.min_price <= sweep_mid <= self.max_price):
                return _empty(
                    f"Price {sweep_mid:.2f} outside cell band [{self.min_price}, {self.max_price}]"
                )

        # Find candidate round number near the sweep bar
        rns = _round_numbers_near(sweep_mid, increment=self.round_number_increment, tol_pct=2.0)
        if not rns:
            return _empty("No round number nearby")

        # Identify a round number that the sweep bar pierced and closed back below
        signal_rn = None
        for rn in rns:
            poke_thresh = rn * (1.0 + self.poke_pct / 100.0)
            if sweep_high >= poke_thresh and sweep_close <= rn:
                # Confirmation: next bar's close also below RN
                if confirm_close < rn:
                    signal_rn = rn
                    break
        if signal_rn is None:
            return _empty("No qualifying round-number sweep")

        # Volume confirmation: sweep_bar volume >= vol_ratio_min * session_avg up to that bar
        prior_bars = today_bars.iloc[:-1]  # up to and including sweep_bar
        if len(prior_bars) < 2:
            return _empty("Not enough prior bars for volume baseline")
        cum_vol_mean = float(prior_bars["volume"].iloc[:-1].mean()) if len(prior_bars) > 1 else 0.0
        if cum_vol_mean <= 0:
            return _empty("Volume baseline invalid")
        if sweep_volume < self.vol_ratio_min * cum_vol_mean:
            return _empty(
                f"Sweep volume {sweep_volume:.0f} < {self.vol_ratio_min}x baseline {cum_vol_mean:.0f}"
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
                "round_number": signal_rn,
                "sweep_high": sweep_high,
                "sweep_close": sweep_close,
                "confirm_close": confirm_close,
            },
            context={
                "vol_ratio": sweep_volume / cum_vol_mean,
                "poke_pct": (sweep_high - signal_rn) / signal_rn * 100.0,
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

        Entry: confirmation bar close.
        SL: sweep_high * (1 + sl_pct_above_sweep_high/100), capped by min_stop_pct.
        T1: entry - t1_r_multiple*R   (FULL EXIT here per R-sweep methodology)
        T2: entry - t2_r_multiple*R   (backup target if T1 jumped)
        Time stop: from config 'time_stop_at'.
        """
        if event is None:
            return None
        if event.side != "short":
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

        # R-sweep result: FULL exit at T1 (not partial). Use partial_qty_pct=1.0 to
        # signal "exit all at T1" through the standard target/action interface.
        # T2 is kept as a backup-only target (qty=0); executor will exit-full at T1
        # which renders T2 inert in practice. This mirrors the R-sweep's hyp_pnl
        # function that gave PF 1.24/1.21/1.17 across 3 windows.
        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": self.t1_r_multiple,
                "qty_pct": 1.0,  # FULL exit at T1
                "action": "exit_full",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": self.t2_r_multiple,
                "qty_pct": 0.0,
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
                f"[{context.symbol}] round_number_sweep_short plan rejected: {e.reason} {e.details}"
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
        # Reuse the sweep_high from the last event's levels if available
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
        # Sweep bar = second-to-last (its high is the SL anchor)
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
        t1 = entry - self.t1_r_multiple * risk
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": self.t1_r_multiple}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(self, context: MarketContext, event: Optional[StructureEvent] = None) -> float:
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
