"""Below-VWAP volume-revert LONG detector — paper-trade phase.

Phase 5 SHIPPABLE per cell-sweep (2026-05-21). 3D cell locked to:
  cap_segment=unknown × vol_ratio_bin=gte_10 × hhmm_bucket=afternoon_1300_1500
Disc PF 1.587 / OOS 1.782 / HO 1.606 (3+ years, n=3,712 pooled).

Spec:      specs/2026-05-21-below_vwap_volume_revert_long-paper-trade-spec.md
Brief:     specs/2026-05-21-brief-below_vwap_volume_revert_long.md
Lock JSON: tools/sub9_research/below_vwap_volume_revert_long_cell_lock.json

DO NOT enable in live (`enabled: true`) until paper-trade phase acceptance
gates pass — see spec Section "Paper-trade acceptance gates".
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

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
from services import cross_day_rvol_enrichment
from services.plan_helpers import (
    PlanRejected,
    assert_sl_outside_entry_zone,
    compute_entry_zone,
    enforce_min_stop_distance,
)


logger = get_agent_logger()


class BelowVwapVolumeRevertLongStructure(BaseStructure):
    """LONG entry on >=2% below-VWAP bars with elevated volume in afternoon."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "below_vwap_volume_revert_long"
        self.configured_setup_type = config.get("_setup_name")

        # All params from config — NO hardcoded defaults (project Rule #1)
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.vwap_dev_pct_max = float(config["vwap_dev_pct_max"])
        self.vol_ratio_min = float(config["vol_ratio_min"])
        self.cell_cap_segment = str(config["cell_lock_cap_segment"])
        self.cell_vol_ratio_min = float(config["cell_lock_vol_ratio_min"])
        self.cell_hhmm_min = self._parse_time(config["cell_lock_hhmm_min"])
        self.cell_hhmm_max = self._parse_time(config["cell_lock_hhmm_max"])
        self.min_signal_bar_notional_rs = float(config["min_signal_bar_notional_rs"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.sl_buffer_below_bar_low_pct = float(config["sl_buffer_below_bar_low_pct"])
        self.min_stop_pct = float(config["min_stop_pct"])
        self.t1_partial_qty_pct = float(config["t1_partial_qty_pct"])
        self.min_bars_required = int(config["min_bars_required"])

        # First-fire-per-(symbol, date) latch
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def detect(self, context: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False, events=[], quality_score=0.0,
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
            context.session_date if context.session_date is not None
            else pd.Timestamp(last_ts).date()
        )
        _sd = session_date.date() if hasattr(session_date, "date") else session_date
        latch_key = (context.symbol, _sd)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        today_bars = df[df.index.date == _sd]
        if today_bars.empty:
            return _empty("No bars for session date")

        # Session VWAP (cumulative since 09:15)
        pv = (today_bars["close"].astype("float64") * today_bars["volume"].astype("float64")).cumsum()
        vol_cum = today_bars["volume"].astype("float64").cumsum()
        if float(vol_cum.iloc[-1]) <= 0:
            return _empty("Zero session volume")
        session_vwap = float(pv.iloc[-1] / vol_cum.iloc[-1])
        last_bar = today_bars.iloc[-1]
        bar_open = float(last_bar["open"])
        bar_high = float(last_bar["high"])
        bar_low = float(last_bar["low"])
        bar_close = float(last_bar["close"])
        bar_volume = float(last_bar["volume"])
        vwap_dev_pct = (bar_close - session_vwap) / session_vwap * 100.0

        if vwap_dev_pct > self.vwap_dev_pct_max:
            return _empty(f"vwap_dev_pct={vwap_dev_pct:.3f} > max={self.vwap_dev_pct_max}")

        if not (self.cell_hhmm_min <= cur_t <= self.cell_hhmm_max):
            return _empty(
                f"cell_hhmm window: {cur_t} not in "
                f"[{self.cell_hhmm_min}, {self.cell_hhmm_max}]"
            )

        ctx_cap = (context.cap_segment or "").strip()
        if ctx_cap != self.cell_cap_segment:
            return _empty(
                f"cap_segment={ctx_cap!r} != cell_lock={self.cell_cap_segment!r}"
            )

        bar_hhmm_int = int(cur_t.strftime("%H%M"))
        baseline_vol = cross_day_rvol_enrichment.get_baseline_vol(
            context.symbol, session_date, bar_hhmm_int,
        )
        if baseline_vol is None or baseline_vol <= 0:
            return _empty("baseline volume unavailable")
        vol_ratio = bar_volume / baseline_vol
        if vol_ratio < self.vol_ratio_min:
            return _empty(f"vol_ratio={vol_ratio:.2f} < brief_min={self.vol_ratio_min}")
        if vol_ratio < self.cell_vol_ratio_min:
            return _empty(f"vol_ratio={vol_ratio:.2f} < cell_lock={self.cell_vol_ratio_min}")

        notional = bar_close * bar_volume
        if notional < self.min_signal_bar_notional_rs:
            return _empty(
                f"notional={notional:,.0f} < min={self.min_signal_bar_notional_rs:,.0f}"
            )

        # Confidence proxy: deeper-below-VWAP within the band gives higher confidence.
        # Clamp to [0, 1]: -2% -> 0.0; -6% -> 1.0.
        depth_band_floor = -6.0
        depth_band_ceil = self.vwap_dev_pct_max  # -2.0
        depth = (depth_band_ceil - vwap_dev_pct) / max(depth_band_ceil - depth_band_floor, 1e-9)
        confidence = max(0.0, min(1.0, depth))

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="long",
            confidence=confidence,
            levels={
                "session_vwap": session_vwap,
                "signal_bar_open": bar_open,
                "signal_bar_high": bar_high,
                "signal_bar_low": bar_low,
                "signal_bar_close": bar_close,
            },
            context={
                "vwap_dev_pct": vwap_dev_pct,
                "vol_ratio": vol_ratio,
                "notional": notional,
                "cap_segment": ctx_cap,
            },
            price=bar_close,
        )
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True, events=[evt],
            quality_score=confidence * 100.0,
        )

    def plan_long_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a LONG TradePlan (Mode B entry placeholder).

        Entry: signal-bar close (Mode B = next-bar open in live; the executor
        applies the actual fill at next bar OPEN). Targets: T1=1.5R, T2=2.0R.
        Hard SL: min(signal_bar_low * (1 - sl_buffer/100), entry * (1 - min_stop_pct/100)).
        """
        if event is None or event.side != "long":
            return None
        evt = event

        bar_low = float(evt.levels["signal_bar_low"])
        entry = float(evt.levels["signal_bar_close"])

        sl_from_low = bar_low * (1.0 - self.sl_buffer_below_bar_low_pct / 100.0)
        sl_from_min = entry * (1.0 - self.min_stop_pct / 100.0)
        hard_sl = min(sl_from_low, sl_from_min)
        if hard_sl >= entry:
            return None
        risk_per_share = entry - hard_sl

        t1_level = entry + self.t1_r_multiple * risk_per_share
        t2_level = entry + self.t2_r_multiple * risk_per_share
        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": self.t1_r_multiple,
                "qty_pct": self.t1_partial_qty_pct,
                "action": "partial_exit",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": self.t2_r_multiple,
                "qty_pct": round(1.0 - self.t1_partial_qty_pct, 4),
                "action": "exit_full",
            },
        ]

        risk_params = self.calculate_risk_params(entry, context)
        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

        try:
            zone = compute_entry_zone(
                entry=entry, bias="long",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(zone, risk_params.hard_sl, "long")
            enforce_min_stop_distance(
                entry, risk_params.hard_sl, self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{context.symbol}] below_vwap_volume_revert_long plan rejected: "
                f"{e.reason} {e.details}"
            )
            return None

        return TradePlan(
            symbol=context.symbol,
            side="long",
            structure_type=evt.structure_type,
            entry_price=entry,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=evt.confidence,
            notes=evt.context,
            trade_id=evt.trade_id,
            target_anchor_type="r_multiple",
        )

    def plan_short_strategy(self, context, event=None):
        return None

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
        bar_low = float(today_bars.iloc[-1]["low"])
        sl_from_low = bar_low * (1.0 - self.sl_buffer_below_bar_low_pct / 100.0)
        sl_from_min = entry_price * (1.0 - self.min_stop_pct / 100.0)
        hard_sl = min(sl_from_low, sl_from_min)
        risk_per_share = max(entry_price - hard_sl, entry_price * 0.001)
        atr_proxy = float((today_bars["high"] - today_bars["low"]).mean())
        return RiskParams(hard_sl=hard_sl, risk_per_share=risk_per_share, atr=atr_proxy)

    def get_exit_levels(self, trade_plan):
        return ExitLevels(hard_sl=trade_plan.risk_params.hard_sl, targets=[])

    def rank_setup_quality(self, context, event=None):
        return 0.0

    def validate_timing(self, current_time):
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
