"""Delivery-Percentage Anomaly Short detector — sub-project #9.

Indian-microstructure-specific asymmetry: NSE bhavcopy `delivery_pct` < 20%
on a session whose daily return > +3% is a textbook retail / operator-pump
signature (low actual ownership transfer + large positive return). The
asymmetry reverses at the next 09:30-10:00 5m bar when fresh-day flow
fails to follow through (close < open, close < session VWAP, cross-day
RVOL >= 1.0). Fire SHORT on confirmation.

Mechanic
--------
T-1 (prior trading day, end-of-day signals from NSE bhavcopy MTO file):
  - delivery_pct < 20.0     (fraction of traded volume actually delivered)
  - daily_return_pct > +3.0  (close vs prev_close)
  - 100 cr <= ADV_20d × close <= 1000 cr  (load-bearing liquidity band)

T+0 (current session 09:30-10:00 5m bar, intraday confirmation):
  - gap_pct from PDC in [-2.0, +3.0]
  - bar close < bar open
  - bar close < session VWAP (to-date)
  - cross-day RVOL >= 1.0 (today's bar volume / 20-prior-session same-time-of-day mean)
Entry = bar's close (SHORT). Latch one fire per (symbol, session_date).

Stop:
  hard_sl = min(open_high_09_15 * 1.005, t_day_close * 1.012)
  (bracketed by either the morning swing high + 50 bps OR T-1 close +
  120 bps — whichever is closer above. If hard_sl <= entry: invalid
  geometry, skip.)

Targets — TIGHT (locked from validated cell):
  T1 = entry - 0.25R, 50% qty
  T2 = entry - 0.75R, 50% qty
  BE trail: stop moves to entry after T1 hit.
  Time stop: 13:00 IST.

Validation history — passed all three sub-9 gates at TIGHT targets:
  Discovery 2023-2024: n=244, PF=1.435, WR=78.7%
  OOS Jan-Sep 2025:    n=135, PF=1.897, WR=84.4%
  Holdout Oct'25-Apr'26: n=102, PF=1.132, WR=75.5%
  War period Feb 28-Apr 8 2026: PF=5.028 (n=10) — benefits from volatility.

Cell-locked filters (cell that survived all 3 periods):
  - cap_segment: ANY (cell is ADV-bucketed not cap-bucketed; trades cluster
    in mid_cap × ADV 100-1000cr territory)
  - adv_20d_cr in [100, 1000]
  - active window 09:30-10:00 IST

UPSTREAM DEPENDENCY (not in this file)
--------------------------------------
The detector reads `delivery_pct` from `ctx.df_daily`. Upstream pipeline
must enrich `df_daily` with a `delivery_pct` column populated from the NSE
bhavcopy MTO file:
  - Backtest: load from `data/delivery_pct/delivery_history.parquet`
    (1.82M rows already populated).
  - Live: scrape NSE MTO daily file pre-market (~7:00 AM IST). Re-use
    `tools/delivery_pct/fetch_delivery.py` (8-worker ThreadPool, ~66s
    for full year).
This pipeline is implemented in a separate task; the detector assumes the
column is present and silently skips when it's missing.

Brief: specs/2026-05-08-sub-project-9-brief-nse_delivery_pct_anomaly.md
Sanity: tools/sub9_research/sanity_nse_delivery_pct_anomaly.py
"""
from __future__ import annotations

from datetime import time, date
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config.logging_config import get_agent_logger
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
    """Read top-level wide_open_mode flag from base config."""
    try:
        from services.config_loader import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class DeliveryPctAnomalyShortStructure(BaseStructure):
    """Delivery-percentage anomaly SHORT — Indian-microstructure asymmetry.

    Cross-day state: requires T-1 daily-bar features (delivery_pct, daily
    return, 20d ADV in Cr) supplied via `context.df_daily`. Fires once per
    (symbol, session_date) on a 09:30-10:00 5m bar that confirms downside
    follow-through with cross-day RVOL >= 1.0 and close < session VWAP.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "delivery_pct_anomaly_short"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config — KeyError on missing.
        # T-1 anomaly thresholds
        self.delivery_pct_max = float(config["delivery_pct_max"])
        self.min_prior_day_return_pct = float(config["min_prior_day_return_pct"])
        self.min_adv_inr_cr = float(config["min_adv_inr_cr"])
        self.max_adv_inr_cr = float(config["max_adv_inr_cr"])
        # T+0 entry conditions
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_gap_pct = float(config["min_gap_pct"])
        self.max_gap_pct = float(config["max_gap_pct"])
        self.min_volume_ratio_to_20d_avg = float(
            config["min_volume_ratio_to_20d_avg"]
        )
        # Risk
        self.stop_open_high_buffer_pct = float(config["stop_open_high_buffer_pct"])
        self.stop_tday_close_buffer_pct = float(config["stop_tday_close_buffer_pct"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"])
        # Targets — TIGHT (load-bearing R-multiples)
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])
        # Plumbing
        self.min_bars_required = int(config["min_bars_required"])

        # Per-(symbol, session_date) qualifier cache populated lazily on
        # first detect() per session.
        self._t_minus_1_cache: Dict[Tuple[str, date], Optional[Dict[str, float]]] = {}
        # First-trigger latch: one fire per (symbol, session_date) per session.
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    # ------------------------------------------------------------------
    # T-1 anomaly qualification (cross-day, lazy per session)
    # ------------------------------------------------------------------

    def _qualify_t_minus_1(
        self, df_daily: pd.DataFrame, t_minus_1_date: date,
    ) -> Optional[Dict[str, float]]:
        """Check whether T-1 daily bar matches the delivery-anomaly signature.

        Required df_daily columns: open, high, low, close, volume, delivery_pct.
        Index must be timezone-naive datetime or date.

        Returns dict with t_minus_1_close (used by plan_short_strategy as
        stop anchor) on success, None on any filter rejection.
        """
        if df_daily is None or df_daily.empty:
            return None
        if "delivery_pct" not in df_daily.columns:
            return None

        # Normalize daily index to date for safe comparison
        idx = df_daily.index
        try:
            d_dates = (
                idx.date if hasattr(idx, "date")
                else pd.to_datetime(idx).date
            )
        except Exception:
            return None

        match_mask = [d == t_minus_1_date for d in d_dates]
        if not any(match_mask):
            return None
        t1_idx = match_mask.index(True)
        if t1_idx == 0:
            return None  # need prior close for daily return
        t1_row = df_daily.iloc[t1_idx]
        prev_row = df_daily.iloc[t1_idx - 1]

        try:
            t1_close = float(t1_row["close"])
            t1_vol = float(t1_row["volume"])
            prev_close = float(prev_row["close"])
            delivery_pct = float(t1_row["delivery_pct"])
        except (KeyError, ValueError, TypeError):
            return None

        if prev_close <= 0 or t1_close <= 0:
            return None
        if pd.isna(delivery_pct):
            return None

        # ---- Anomaly filters ----
        # 1) Low delivery (operator/retail pump signature)
        if delivery_pct >= self.delivery_pct_max:
            return None

        # 2) Same-day pump > threshold
        daily_return_pct = (t1_close / prev_close - 1.0) * 100.0
        if daily_return_pct <= self.min_prior_day_return_pct:
            return None

        # 3) Liquidity band (ADV × close in Cr)
        if t1_idx < 20:
            return None
        vol_window = df_daily["volume"].iloc[t1_idx - 20: t1_idx]
        close_window = df_daily["close"].iloc[t1_idx - 20: t1_idx]
        try:
            adv_inr_cr = float((vol_window * close_window).mean()) / 1e7
        except Exception:
            return None
        if not (self.min_adv_inr_cr <= adv_inr_cr <= self.max_adv_inr_cr):
            return None

        return {
            "t_minus_1_close": t1_close,
            "t_minus_1_volume": t1_vol,
            "delivery_pct": delivery_pct,
            "daily_return_pct": daily_return_pct,
            "adv_inr_cr": adv_inr_cr,
        }

    # ------------------------------------------------------------------
    # Cross-day RVOL (today bar vol / 20-prior-session same-time-of-day mean)
    # ------------------------------------------------------------------

    @staticmethod
    def _cross_day_rvol(
        df_5m: pd.DataFrame, session_date: date, symbol: str
    ) -> Optional[float]:
        """Return today's last-bar volume divided by the prior-20-session
        same-time-of-day mean for (symbol, session_date, hhmm).

        Reads baseline from `services/cross_day_rvol_enrichment.py` (a static
        precomputed parquet keyed by (symbol, date, hhmm)). The previous
        implementation filtered df_5m for prior same-tod bars — but the
        screener caps df_5m at `screener_store_5m_max=120` (~1.5 days), so
        20 prior same-tod bars are never present. That made this function
        always return None in production, which silenced every otherwise-
        qualifying delivery_pct_anomaly_short fire (verified by 0 trades
        across 501 Discovery days in OCI run 20260508-230433_full vs
        sanity's 261).

        Falls back to None when the lookup parquet is missing — caller
        rejects with "cross-day RVOL unavailable" so the operator sees the
        problem without crashing live.
        """
        if df_5m is None or df_5m.empty:
            return None
        try:
            last_ts = df_5m.index[-1]
            last_vol = float(df_5m.iloc[-1]["volume"])
            if last_vol <= 0:
                return None
            tod = last_ts.time() if hasattr(last_ts, "time") else None
            if tod is None:
                return None
            hhmm = tod.hour * 100 + tod.minute
            from services.cross_day_rvol_enrichment import get_baseline_vol
            baseline = get_baseline_vol(symbol, session_date, hhmm)
            if baseline is None or baseline <= 0:
                return None
            return last_vol / baseline
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Session VWAP-to-date (using today's bars only)
    # ------------------------------------------------------------------

    @staticmethod
    def _session_vwap(today_bars: pd.DataFrame) -> Optional[float]:
        """Cumulative VWAP over today's bars (typical-price × volume) /
        cumulative volume. None on degenerate input."""
        if today_bars is None or today_bars.empty:
            return None
        try:
            tp = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
            cum_v = float(today_bars["volume"].sum())
            if cum_v <= 0:
                return None
            return float((tp * today_bars["volume"]).sum() / cum_v)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Fire SHORT only on a 09:30-10:00 5m bar after a T-1 delivery-pct
        anomaly + same-day pump, gated by gap range + downside confirmation.
        """

        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        # ---- Active window ----
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # ---- Latch ----
        session_date = ctx.session_date
        if session_date is None:
            session_date = pd.Timestamp(last_ts).date()
        latch_key = (ctx.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        # ---- T-1 anomaly qualification (cross-day) ----
        # NOTE on wide_open: the prior version of this detector bypassed all
        # signal gates below under wide_open_mode. That was wrong — without
        # the delivery_pct/daily_return/ADV gates this setup has no identity,
        # and it caused ~7000x trade-count inflation in OCI capture (every
        # symbol fired every bar). wide_open is meant to bypass meta-filters
        # (cap_segment, regime, universe, historical PnL gates), NOT the
        # detector signal itself. This detector has no meta-filters to
        # bypass, so wide_open has no effect here — which is correct.
        df_daily = ctx.df_daily
        if df_daily is None or df_daily.empty:
            return _empty("daily bars unavailable")

        d_idx = df_daily.index
        try:
            d_dates = (
                d_idx.date if hasattr(d_idx, "date")
                else pd.to_datetime(d_idx).date
            )
        except Exception:
            return _empty("daily index format error")
        prior = [d for d in d_dates if d < session_date]
        if not prior:
            return _empty("no prior daily bar")
        t_minus_1_date = max(prior)

        cache_key = (ctx.symbol, session_date)
        if cache_key in self._t_minus_1_cache:
            t1_info = self._t_minus_1_cache[cache_key]
        else:
            t1_info = self._qualify_t_minus_1(df_daily, t_minus_1_date)
            self._t_minus_1_cache[cache_key] = t1_info
        if t1_info is None:
            return _empty("T-1 not a delivery-pct anomaly qualifier")

        # ---- Today's bars ----
        today_bars = df[df.index.date == session_date]
        if today_bars.empty:
            return _empty("no bars for session_date")

        # ---- PDC + gap_pct gate ----
        pdc = float(t1_info["t_minus_1_close"])
        if pdc <= 0:
            return _empty("invalid PDC")
        first_open = float(today_bars.iloc[0]["open"])
        if first_open <= 0:
            return _empty("session open invalid")
        gap_pct = (first_open / pdc - 1.0) * 100.0
        if gap_pct < self.min_gap_pct:
            return _empty(f"gap_pct={gap_pct:.2f} < min={self.min_gap_pct}")
        if gap_pct > self.max_gap_pct:
            return _empty(f"gap_pct={gap_pct:.2f} > max={self.max_gap_pct}")

        # ---- Confirmation candle (current bar) ----
        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_close = float(last["close"])
        if bar_close >= bar_open:
            return _empty(f"not red bar: open={bar_open}, close={bar_close}")

        # close < session VWAP
        vwap_val = self._session_vwap(today_bars)
        if vwap_val is None:
            return _empty("VWAP unavailable")
        if bar_close >= vwap_val:
            return _empty(f"close {bar_close} >= VWAP {vwap_val}")

        # cross-day RVOL >= threshold (uses precomputed baseline parquet —
        # df_5m alone doesn't have 20 prior same-tod bars in production).
        rvol = self._cross_day_rvol(df, session_date, ctx.symbol)
        if rvol is None:
            return _empty("cross-day RVOL unavailable")
        if rvol < self.min_volume_ratio_to_20d_avg:
            return _empty(
                f"rvol={rvol:.2f} < {self.min_volume_ratio_to_20d_avg}"
            )

        # ---- Open-high (09:15) for stop construction ----
        # Use today's first bar high as the morning swing high anchor.
        open_high_09_15 = float(today_bars.iloc[0]["high"])

        # ---- Confidence (proxy: how anomalous T-1 was) ----
        # Bigger same-day return + lower delivery_pct = higher confidence.
        delivery_pct = t1_info.get("delivery_pct")
        daily_ret = t1_info.get("daily_return_pct")
        try:
            d_score = max(0.0, 1.0 - float(delivery_pct) / self.delivery_pct_max) \
                if delivery_pct is not None and not pd.isna(delivery_pct) else 0.5
        except Exception:
            d_score = 0.5
        try:
            r_score = min(1.0, max(0.0, (float(daily_ret) - self.min_prior_day_return_pct) / 5.0)) \
                if daily_ret is not None and not pd.isna(daily_ret) else 0.5
        except Exception:
            r_score = 0.5
        confidence = max(0.0, min(1.0, 0.5 * d_score + 0.5 * r_score))

        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "t_minus_1_close": pdc,
                "open_high_09_15": open_high_09_15,
                "session_vwap": vwap_val,
                "entry_close": bar_close,
                "first_open": first_open,
            },
            context={
                "delivery_pct": float(delivery_pct) if delivery_pct is not None
                                 and not pd.isna(delivery_pct) else None,
                "daily_return_pct": float(daily_ret) if daily_ret is not None
                                     and not pd.isna(daily_ret) else None,
                "adv_inr_cr": float(t1_info.get("adv_inr_cr"))
                              if t1_info.get("adv_inr_cr") is not None
                              and not pd.isna(t1_info.get("adv_inr_cr")) else None,
                "gap_pct": gap_pct,
                "rvol_cross_day": rvol,
                "session_date_iso": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
            },
            price=bar_close,
        )
        # Set latch HERE in detect() — not in plan_short_strategy. detect()
        # runs in the cached MainDetector instance per worker process, so
        # the latch state survives across bars within that worker. Setting
        # it in plan_short_strategy was broken because plan_*_strategy runs
        # in PlanOrchestrator (main process) — a separate detector instance,
        # so the latch never propagated back to the workers' detect() loop.
        # Caused ~6.5x re-fire on every (symbol, day) in the active window.
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
        """Build TradePlan with R-multiple targets at 0.25R / 0.75R."""
        if event is None or event.side != "short":
            return None

        df = ctx.df_5m
        if df is None or df.empty:
            return None
        last = df.iloc[-1]
        close = float(last["close"])
        levels = event.levels or {}

        open_high_09_15 = float(levels.get("open_high_09_15", close))
        t_minus_1_close = float(levels.get("t_minus_1_close", close))

        # Stop = min(open_high * (1 + buffer), t_minus_1_close * (1 + buffer))
        sl_a = open_high_09_15 * (1.0 + self.stop_open_high_buffer_pct / 100.0)
        sl_b = t_minus_1_close * (1.0 + self.stop_tday_close_buffer_pct / 100.0)
        hard_sl = min(sl_a, sl_b)

        # Invalid geometry: stop must be strictly above entry for SHORT.
        if hard_sl <= close:
            logger.debug(
                f"[{ctx.symbol}] delivery_pct_anomaly_short skip: "
                f"hard_sl {hard_sl} <= entry {close} (sl_a={sl_a}, sl_b={sl_b})"
            )
            return None

        risk_per_share = max(hard_sl - close, close * 1e-4)

        # R-multiple targets (TIGHT — load-bearing).
        t1_target = close - self.t1_r_multiple * risk_per_share
        t2_target = close - self.t2_r_multiple * risk_per_share

        targets = [
            {
                "name": "T1", "level": t1_target, "rr": self.t1_r_multiple,
                "qty_pct": self.t1_partial_qty_pct, "action": "partial_exit",
            },
            {
                "name": "T2", "level": t2_target, "rr": self.t2_r_multiple,
                "qty_pct": round(1.0 - self.t1_partial_qty_pct, 4),
                "action": "exit_full",
            },
        ]

        # ATR proxy for risk_params metadata only (not used in stop calc here).
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
        # Plan-as-source-of-truth (2026-05-12): time_exit from per-setup config.
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(
            hard_sl=hard_sl, targets=targets, trail_to="breakeven",
            time_exit=time_exit_str,
        )

        # Plan-geometry validation
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
                f"[{ctx.symbol}] delivery_pct_anomaly_short plan rejected: "
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
            # R-multiple (arithmetic) targets — preserved across late fills
            # by services/target_recalc.py recomputing T1/T2 from the actual
            # entry using stored R multiples.
            target_anchor_type="arithmetic",
        )

    # ------------------------------------------------------------------
    # BaseStructure abstract methods
    # ------------------------------------------------------------------

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext,
    ) -> RiskParams:
        """ABC contract — production stop comes from plan_short_strategy."""
        return RiskParams(
            hard_sl=entry_price * (1.0 + self.min_stop_distance_pct / 100.0),
            risk_per_share=entry_price * self.min_stop_distance_pct / 100.0,
            atr=None,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """ABC contract — default 1R exit; production targets come from
        plan_short_strategy with the load-bearing 0.25R/0.75R."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        t1 = entry - risk
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": 1.0}],
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
