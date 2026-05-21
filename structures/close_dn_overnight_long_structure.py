"""close_dn_overnight_long detector — Phase 5 SHIPPABLE Cell #5.

Mechanism: post-rally EOD-flush overnight reversion. Fires at 15:25 IST
when the closing 5 bars (15:00-15:20) show heavy sell-volume on a
post-up-3% day, then BUYs MOC (MTF or CNC) and queues AMO SELL for
next-day pre-open auction.

Spec:      specs/2026-05-21-close_dn_overnight_long-SHIPPABLE-cell-5.md
Cell-lock: tools/sub9_research/close_dn_overnight_long_cell_lock.json
Brief:     specs/2026-05-21-brief-close_dn_overnight_long.md

CRITICAL look-ahead correction: the 5-bar signal uses bars 15:00-15:20
ONLY. The 15:25 bar (which closes at 15:30 IST) is excluded because in
live execution, MOC orders fire at 15:25-15:30 — we cannot use the 15:25
bar's volume/direction to make a 15:25 trading decision.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config.logging_config import get_agent_logger
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


# 5-bar signal window (look-ahead-safe; 15:25 bar excluded)
_SIGNAL_BAR_HHMMS = ("15:00", "15:05", "15:10", "15:15", "15:20")
# Active window single bar — only fire at 15:25
_ACTIVE_HHMM = "15:25"


class CloseDnOvernightLongStructure(BaseStructure):
    """LONG overnight setup. Fires at 15:25 IST after EOD sell-flush on post-up-rally day."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "close_dn_overnight_long"
        self.configured_setup_type = config.get("_setup_name")

        # Primary filter thresholds (brief section 5/6)
        self.signed_vol_ratio_max = float(config["signed_vol_ratio_max"])
        self.closing_volume_z_min = float(config["closing_25m_volume_z_min"])
        self.min_bar_count = int(config["min_signal_bar_count"])
        # Cell filter thresholds (cell-lock JSON: closing_30m_volume_z >= 2.0, prior_ret >= 3.0)
        self.cell_volume_z_min = float(config["cell_volume_z_min"])
        self.cell_prior_day_return_pct_min = float(config["cell_prior_day_return_pct_min"])
        # Rolling baseline window for volume z (typically 20 days)
        self.baseline_rolling_days = int(config["baseline_rolling_days"])
        # MTF universe loader for product routing
        mtf_cfg = config.get("mtf", {})
        self._mtf_snapshot_path = Path(str(mtf_cfg["approved_list_snapshot_path"]))
        self._mtf_exclude_etf = bool(mtf_cfg.get("exclude_etf", True))
        # Per-day latch (key = (symbol, date))
        self._fired_today: set = set()
        self._mtf_universe = None  # lazy-loaded on first detect

        logger.info(
            "close_dn_overnight_long_structure: initialized with signed_vol<=%.2f, "
            "closing_volume_z>=%.2f, cell_volume_z>=%.2f, cell_prior_ret>=%.2f%%",
            self.signed_vol_ratio_max, self.closing_volume_z_min,
            self.cell_volume_z_min, self.cell_prior_day_return_pct_min,
        )

    def _load_mtf_universe(self):
        """Lazy-load MTF universe (avoid loading at import time for fast tests)."""
        if self._mtf_universe is None:
            from services.mtf_universe import MtfUniverse
            self._mtf_universe = MtfUniverse(self._mtf_snapshot_path)
        return self._mtf_universe

    def detect(self, context: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False, events=[], quality_score=0.0,
                rejection_reason=reason or None,
            )

        df = context.df_5m
        if df is None or df.empty:
            return _empty("No 5m bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        cur_hhmm = cur_t.strftime("%H:%M") if hasattr(cur_t, "strftime") else str(cur_t)

        # 1. Active window: single-bar 15:25
        if cur_hhmm != _ACTIVE_HHMM:
            return _empty(f"outside active window: {cur_hhmm} != {_ACTIVE_HHMM}")

        session_date = (
            context.session_date if context.session_date is not None
            else pd.Timestamp(last_ts).date()
        )
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        latch_key = (context.symbol, _sd)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        today_bars = df[df.index.date == _sd] if hasattr(df.index, "date") else df
        if today_bars.empty:
            return _empty("No bars for session date")

        # 2. Extract the 5 signal bars (15:00-15:20)
        signal_bars = today_bars[today_bars.index.map(lambda ts: ts.strftime("%H:%M") in _SIGNAL_BAR_HHMMS)]
        if len(signal_bars) < self.min_bar_count:
            return _empty(
                f"insufficient signal bars: have {len(signal_bars)} of 5 "
                f"({_SIGNAL_BAR_HHMMS}); min_required={self.min_bar_count}"
            )

        # 3. Compute signed_vol_ratio = sum(vol * sign(close-open)) / sum(vol)
        opens = signal_bars["open"].astype("float64")
        closes = signal_bars["close"].astype("float64")
        vols = signal_bars["volume"].astype("float64")
        total_vol = float(vols.sum())
        if total_vol <= 0:
            return _empty("zero total volume across signal bars")
        bar_dirs = np.sign(closes - opens)
        signed_vol = (vols * bar_dirs).sum()
        signed_vol_ratio = float(signed_vol / total_vol)
        if signed_vol_ratio > self.signed_vol_ratio_max:
            return _empty(
                f"signed_vol_ratio={signed_vol_ratio:.3f} > max={self.signed_vol_ratio_max} "
                f"(needs heavy sell-volume in closing 25m)"
            )

        # 4. Compute closing_25m_volume_z = (today_5bar_total - prior_20d_mean) / prior_20d_std
        baseline_mean, baseline_std = self._closing_baseline(context, _sd)
        if baseline_mean is None or baseline_std is None or baseline_std <= 0:
            return _empty("closing-25m baseline unavailable (insufficient prior history)")
        volume_z = (total_vol - baseline_mean) / baseline_std
        if volume_z < self.closing_volume_z_min:
            return _empty(
                f"closing_25m_volume_z={volume_z:.2f} < min={self.closing_volume_z_min}"
            )

        # 5. Cell filter — extreme volume_z
        if volume_z < self.cell_volume_z_min:
            return _empty(
                f"closing_volume_z={volume_z:.2f} < cell_min={self.cell_volume_z_min} "
                f"(extreme bucket required)"
            )

        # 6. Cell filter — prior day return >= 3%
        prior_day_return_pct = self._prior_day_return_pct(context, _sd)
        if prior_day_return_pct is None:
            return _empty("prior_day_return_pct unavailable")
        if prior_day_return_pct < self.cell_prior_day_return_pct_min:
            return _empty(
                f"prior_day_return_pct={prior_day_return_pct:.2f} < "
                f"cell_min={self.cell_prior_day_return_pct_min} (up_gt_3pct cell required)"
            )

        # 7. MTF eligibility + product routing
        bare = context.symbol.replace("NSE:", "")
        mtf_universe = self._load_mtf_universe()
        mtf_info = mtf_universe.lookup(bare)
        # ETF guard: skip entirely (mechanism mismatch)
        if mtf_info is not None and mtf_info.category == "etf" and self._mtf_exclude_etf:
            return _empty(f"symbol is ETF (category=etf in MTF list); mechanism mismatch")
        # Route by eligibility
        if mtf_info is not None and (not self._mtf_exclude_etf or mtf_info.category != "etf"):
            product = "MTF"
            leverage = float(mtf_info.leverage)
        else:
            product = "CNC"
            leverage = 1.0

        # 8. Compute entry-bar reference price (15:25 bar OPEN, since we only have bars
        #    up through 15:25 — the 15:25 bar represents 15:25-15:30; its OPEN is the
        #    first price observed at 15:25 IST). The actual MOC fill will be at the
        #    15:25 bar's CLOSE (= 15:30 IST close), but at signal time we don't know
        #    that yet. Use the entry-bar OPEN as the reference; the executor logs the
        #    actual MOC fill price separately.
        entry_bar = today_bars[today_bars.index.map(lambda ts: ts.strftime("%H:%M") == _ACTIVE_HHMM)]
        if entry_bar.empty:
            return _empty("entry bar (15:25) not present in df_5m")
        entry_price = float(entry_bar["open"].iloc[-1])

        # 9. Build StructureEvent (the planner uses this to construct TradePlan)
        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="long",
            confidence=min(1.0, max(0.0, abs(signed_vol_ratio))),  # |svr| in [0.5, 1.0] → [0.5, 1.0]
            levels={
                "entry_bar_open": entry_price,
                "signal_total_volume": float(total_vol),
            },
            context={
                "signed_vol_ratio": signed_vol_ratio,
                "closing_volume_z": float(volume_z),
                "prior_day_return_pct": float(prior_day_return_pct),
                "product": product,
                "leverage": leverage,
                "cap_segment": context.cap_segment or "unknown",
            },
            price=entry_price,
        )
        self._fired_today.add(latch_key)
        logger.info(
            "close_dn_overnight_long fired | symbol=%s svr=%.3f vol_z=%.2f prior_ret=%.2f%% product=%s lev=%.2f",
            context.symbol, signed_vol_ratio, volume_z, prior_day_return_pct, product, leverage,
        )
        return StructureAnalysis(
            structure_detected=True, events=[evt],
            quality_score=evt.confidence * 100.0,
        )

    def _closing_baseline(self, context: MarketContext, session_date: date) -> tuple:
        """Compute (mean, std) of prior-N-day closing-25m total volume for this symbol.

        Returns (None, None) if insufficient history.
        """
        df = context.df_5m
        if df is None or df.empty:
            return (None, None)
        # All bars before session_date, filtered to 15:00-15:20 hhmms
        hist = df[df.index.date < session_date]
        if hist.empty:
            return (None, None)
        signal_hist = hist[hist.index.map(lambda ts: ts.strftime("%H:%M") in _SIGNAL_BAR_HHMMS)]
        if signal_hist.empty:
            return (None, None)
        # Per-prior-session totals
        per_session = signal_hist.groupby(signal_hist.index.date)["volume"].sum().astype("float64")
        # Take the most recent N sessions
        if len(per_session) < max(10, self.baseline_rolling_days // 2):
            return (None, None)
        recent = per_session.iloc[-self.baseline_rolling_days:]
        mean = float(recent.mean())
        std = float(recent.std(ddof=1))
        return (mean, std)

    def _prior_day_return_pct(self, context: MarketContext, session_date: date) -> Optional[float]:
        """Return (prior_close - prev_prior_close) / prev_prior_close * 100.

        Reads from context.df_daily if available, else derives from df_5m
        prior-session close.
        """
        # Try daily first
        if context.df_daily is not None and not context.df_daily.empty:
            ddf = context.df_daily
            # Filter to dates strictly before session_date
            if hasattr(ddf.index, "date"):
                before = ddf[ddf.index.date < session_date]
            else:
                before = ddf
            if len(before) >= 2:
                prev_close = float(before["close"].iloc[-1])
                prev_prev_close = float(before["close"].iloc[-2])
                if prev_prev_close > 0:
                    return (prev_close - prev_prev_close) / prev_prev_close * 100.0

        # Fallback: derive from df_5m prior session closes (use 15:25 close per day)
        df = context.df_5m
        if df is None or df.empty:
            return None
        hist = df[df.index.date < session_date]
        if hist.empty:
            return None
        # Take the LAST bar of each prior session as "close"
        per_session_close = hist.groupby(hist.index.date)["close"].last().astype("float64")
        if len(per_session_close) < 2:
            return None
        prev_close = float(per_session_close.iloc[-1])
        prev_prev_close = float(per_session_close.iloc[-2])
        if prev_prev_close <= 0:
            return None
        return (prev_close - prev_prev_close) / prev_prev_close * 100.0

    def plan_long_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Build the overnight LONG TradePlan.

        Entry price is the signal-bar OPEN (informational); actual MOC fill is
        recorded by the executor at the 15:25 bar CLOSE (~15:30 IST close price).
        Exit is scheduled at next-trading-day 09:15 via broker AMO.
        No SL, no T1/T2, no intraday management.
        """
        if event is None or event.side != "long":
            return None
        evt = event
        product = evt.context.get("product", "CNC")
        leverage = float(evt.context.get("leverage", 1.0))
        margin_per_slot = float(self.config["capital_allocation"]["margin_per_slot_inr"])
        notional = margin_per_slot * leverage
        entry_price = float(evt.price)
        if entry_price <= 0:
            return None
        qty = max(1, int(notional / entry_price))

        # Next-trading-day 09:15 IST as the scheduled exit
        # NOTE: weekend/holiday handling is best done downstream where a
        # trading-calendar utility is available. For the detector's view, use
        # naive "next calendar day at 09:15"; the executor adjusts if needed.
        evt_ts = pd.Timestamp(evt.timestamp)
        next_day = (evt_ts + pd.Timedelta(days=1)).normalize() + pd.Timedelta(hours=9, minutes=15)

        exit_levels = ExitLevels(
            targets=[],            # no T1/T2 — single exit at scheduled time
            hard_sl=0.0,           # sentinel: scheduled_amo mode ignores SL
            trail_to=None,
            time_exit=None,
            structure_exit=None,
            exit_mode="scheduled_amo",
            scheduled_exit_at=next_day,
        )
        risk_params = RiskParams(
            hard_sl=0.0,
            risk_per_share=0.0,
        )
        return TradePlan(
            symbol=context.symbol,
            side="long",
            structure_type=self.structure_type,
            entry_price=entry_price,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=qty,
            notional=notional,
            confidence=evt.confidence,
            timestamp=pd.Timestamp(evt.timestamp),
            market_context=context,
            notes={
                "product": product,
                "leverage": leverage,
                "signed_vol_ratio": evt.context["signed_vol_ratio"],
                "closing_volume_z": evt.context["closing_volume_z"],
                "prior_day_return_pct": evt.context["prior_day_return_pct"],
            },
        )

    def plan_short_strategy(self, context, event=None):
        """Not applicable — overnight LONG setup only."""
        return None

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext
    ) -> RiskParams:
        """Overnight scheduled_amo mode — no price-triggered SL. Sentinel zeros."""
        return RiskParams(hard_sl=0.0, risk_per_share=0.0)

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """Return the trade plan's own exit levels (already built by plan_long_strategy)."""
        return trade_plan.exit_levels

    def rank_setup_quality(self, context: MarketContext, event: Optional[StructureEvent] = None) -> float:
        """Quality proxy — uses event confidence if available, else 0."""
        if event is not None:
            return float(event.confidence) * 100.0
        return 0.0

    def validate_timing(self, current_time) -> bool:
        """Single-bar 15:25 window."""
        t = current_time.time() if hasattr(current_time, "time") else current_time
        hhmm = t.strftime("%H:%M") if hasattr(t, "strftime") else str(t)
        return hhmm == _ACTIVE_HHMM
