"""Gap Fade Short detector — sub-project #7, Task 4.

Thesis: When a small/mid/micro-cap stock gaps up 1.5-8% above PDC on the opening
bar, retail-driven momentum often exhausts quickly. An exhaustion candle (large
upper wick, small body) combined with declining volume signals that buyers are
running out of steam. Pros fade the gap back toward the PDC.

Active window: 09:15-09:30 IST (first three 5m bars of session).
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


def _is_wide_open() -> bool:
    """Read top-level wide_open_mode flag from base config.

    When true, bypass cell filters (cap_segment) so the OCI capture run
    sees every gap-up candidate. The gauntlet replays offline with the
    cap filter re-applied. See commit 65648f1 for sub8 precedent.
    """
    try:
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False
from .data_models import (
    ExitLevels,
    MarketContext,
    RiskParams,
    StructureAnalysis,
    StructureEvent,
    TradePlan,
)

logger = get_agent_logger()


class GapFadeShortStructure(BaseStructure):
    """Detects early-session short opportunities when a gap-up exhausts."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "gap_fade_short"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.min_gap_pct = float(config["min_gap_pct_above_pdc"])
        self.max_gap_pct = float(config["max_gap_pct_above_pdc"])
        self.min_upper_wick_ratio = float(config["min_upper_wick_ratio"])
        self.max_body_size_pct = float(config["max_body_size_pct"])
        self.require_vol_decline = bool(config["require_volume_decline_after_gap"])
        self.allowed_caps = set(config.get("allowed_cap_segments", []))
        self.stop_atr_buffer = float(config["stop_above_gap_high_atr"])
        # Cap-aware buffer above gap_high. Indian sources (StockManiacs, smallcase):
        # micro-caps need wider buffer than small/mid because of illiquid spreads.
        self.stop_buffer_above_gap_high_pct_small_mid = float(
            config.get("stop_buffer_above_gap_high_pct_small_mid", 0.0025)
        )
        self.stop_buffer_above_gap_high_pct_micro = float(
            config.get("stop_buffer_above_gap_high_pct_micro", 0.005)
        )
        self.target_type = str(config["target_type"])
        self.min_bars_required = int(config["min_bars_required"])
        # T1 partial exit at 50% gap fill (TradingQnA: small-cap gap-close rate
        # only 50-60%; capture half-fill at ~1R, let runners go to PDC).
        self.t1_partial_qty_pct = float(config.get("t1_partial_qty_pct", 0.5))
        # First-trigger latch: one fire per (symbol, session_date). Without
        # this, the detector re-fires on every 5m bar in 09:15-09:30 that
        # still meets the exhaustion-candle conditions (verified in OCI
        # run 20260508-195237_full: 30% of qualifying symbols fired 2-3
        # bars consecutively, mean 1.31 fires/symbol).
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, context: MarketContext) -> float:
        """Extract ATR from indicators dict with fallback."""
        if context.indicators and "atr" in context.indicators:
            return float(context.indicators["atr"])
        if context.df_5m is not None and len(context.df_5m) >= 14:
            df = context.df_5m
            return float((df["high"] - df["low"]).tail(14).mean())
        return context.current_price * 0.01

    def detect(self, context: MarketContext) -> StructureAnalysis:
        """Detect gap-fade short setups in the 09:15-09:30 window.

        Six conditions must ALL be true:
        1. Current 5m bar is within active_window_start..active_window_end
        2. Cap segment is in allowed_cap_segments
        3. Opening bar's open is above PDC by min_gap_pct..max_gap_pct
        4. PDC level is available
        5. Exhaustion candle: upper_wick/body >= min_upper_wick_ratio AND
           body_size_pct <= max_body_size_pct
        6. (If require_volume_decline_after_gap) current bar volume < opening bar volume
        """
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

        # --- Condition 2: Cap segment (bypassed under wide_open) ---
        if not _wide_open and context.cap_segment not in self.allowed_caps:
            return _empty(f"Cap segment {context.cap_segment!r} not in allowed set")

        # --- Condition 1: Active time-of-day window ---
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        # --- Latch: one fire per (symbol, session_date) ---
        session_date = (
            context.session_date
            if context.session_date is not None
            else pd.Timestamp(last_ts).date()
        )
        latch_key = (context.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        # --- Condition 4: PDC available ---
        pdc = float(context.pdc) if context.pdc is not None else None
        if pdc is None or pdc <= 0:
            return _empty("PDC unavailable")

        # --- Condition 3: Gap-up on opening bar ---
        # The opening bar is bar 0 (first bar of df, assumed to be 09:15 bar).
        opening_bar = df.iloc[0]
        gap_open = float(opening_bar["open"])
        gap_pct = ((gap_open - pdc) / pdc) * 100.0
        if gap_pct < self.min_gap_pct:
            return _empty(
                f"gap_pct={gap_pct:.2f} < min_gap_pct_above_pdc={self.min_gap_pct}"
            )
        if gap_pct > self.max_gap_pct:
            return _empty(
                f"gap_pct={gap_pct:.2f} > max_gap_pct_above_pdc={self.max_gap_pct}"
            )

        # --- Condition 5: Exhaustion candle on current bar ---
        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_close = float(last["close"])

        body = abs(bar_close - bar_open)
        candle_top = max(bar_open, bar_close)
        upper_wick = bar_high - candle_top

        # Avoid division by zero — if body is tiny, treat upper_wick_ratio as large
        if body < 1e-8:
            wick_ratio = float("inf")
        else:
            wick_ratio = upper_wick / body

        body_size_pct = (body / bar_open) * 100.0 if bar_open > 0 else 0.0

        if wick_ratio < self.min_upper_wick_ratio:
            return _empty(
                f"upper_wick_ratio={wick_ratio:.2f} < min={self.min_upper_wick_ratio}"
            )
        if body_size_pct > self.max_body_size_pct:
            return _empty(
                f"body_size_pct={body_size_pct:.2f} > max={self.max_body_size_pct}"
            )

        # --- Condition 6: Volume decline after opening bar ---
        if self.require_vol_decline and len(df) >= 2:
            opening_vol = float(df["volume"].iloc[0])
            current_vol = float(last["volume"])
            if current_vol >= opening_vol:
                return _empty(
                    f"Volume not declining: current={current_vol} >= opening={opening_vol}"
                )

        # All conditions met — build event
        gap_high = float(opening_bar["high"])
        confidence = min(1.0, wick_ratio / 2.0)   # stronger wick → higher confidence

        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "pdc": pdc,
                "gap_open": gap_open,
                "gap_high": gap_high,
                "close": bar_close,
            },
            context={
                "gap_pct": gap_pct,
                "upper_wick_ratio": wick_ratio,
                "body_size_pct": body_size_pct,
            },
            price=bar_close,
        )
        # Set latch in detect() — runs in MainDetector worker process and
        # survives across bars within that worker. Setting it in
        # plan_short_strategy would not propagate back because plan_*
        # runs in PlanOrchestrator (main process) — different instance.
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ---------- Required abstract method implementations ----------

    def plan_long_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Short-only setup — no long trades."""
        return None

    def plan_short_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a TradePlan for the gap-fade short setup.

        Entry: at current close (0.1% slippage tolerance).
        Stop:  above opening gap high + ATR * stop_above_gap_high_atr.
        Target: PDC (if target_type == "pdc_or_open") else opening price.

        Architectural rule (2026-04-30): no re-detect; event REQUIRED.
        Caller (orchestrator → plan_short_strategy) MUST pass the
        StructureEvent produced by MainDetector.
        """
        if event is None:
            return None
        evt = event
        if evt.side != "short":
            return None

        df = context.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        atr_val = self._get_atr(context)

        opening_bar = df.iloc[0]
        gap_high = float(opening_bar["high"])
        gap_open = float(opening_bar["open"])

        # Stop: ABOVE opening gap high (short, so stop is above entry)
        # Cap-segment-aware buffer: micro_cap needs wider buffer (0.5%) than
        # small/mid_cap (0.25%) due to higher noise.
        # Source: TradingQnA, StockManiacs — micro-cap gap fades stopped on normal
        # intrabar spikes at 0.25%; 0.5% provides research-aligned cushion.
        cap_seg = getattr(context, "cap_segment", None) or ""
        if cap_seg == "micro_cap":
            stop_a = gap_high * 1.005                 # gap_high + 0.5% for micro_cap
        else:
            stop_a = gap_high * 1.0025                # gap_high + 0.25% for small/mid_cap
        stop_b = close + atr_val * 1.5                # entry + 1.5 ATR
        hard_sl = max(stop_a, stop_b)
        risk_per_share = max(hard_sl - close, atr_val * 0.1)

        # Tiered targets: T1 at 50% gap fill, T2 at full PDC.
        # Research: TradingQnA — small-cap gap-close rate only 50-60%; tiered
        # exits capture half-fills that never reach PDC.
        pdc = float(context.pdc) if context.pdc is not None else close
        t1_level = (close + pdc) / 2.0                # 50% gap fill
        t2_level = pdc                                 # full PDC

        # Ensure targets are below entry for short
        if t1_level >= close:
            t1_level = close - risk_per_share * 0.5
        if t2_level >= close:
            t2_level = close - risk_per_share

        rr_t1 = (close - t1_level) / max(risk_per_share, 1e-6)
        rr_t2 = (close - t2_level) / max(risk_per_share, 1e-6)
        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": rr_t1,
                "qty_pct": 0.5,
                "action": "partial_exit",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": rr_t2,
                "qty_pct": 0.5,
                "action": "exit_full",
            },
        ]

        risk_params = self.calculate_risk_params(close, context)
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

        # === Plan-geometry validation (Phase-C-Commit-2) ===
        # Reject geometrically-broken plans at the source. The orchestrator
        # used to catch these LATE — Bugs A-E (NSIL/CAPLIPOINT class) all
        # surfaced because by the time the orchestrator ran the SL-zone
        # check, the detector had already minted a trade_id and emitted an
        # accept event. Catching here = cleaner audit + earlier short-circuit.
        try:
            _zone = compute_entry_zone(
                entry=close, bias="short",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, risk_params.hard_sl, "short")
            enforce_min_stop_distance(
                close, risk_params.hard_sl,
                self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{context.symbol}] gap_fade_short plan rejected: {e.reason} {e.details}"
            )
            return None

        confidence = evt.confidence
        notes = evt.context
        structure_type = evt.structure_type
        event_trade_id = evt.trade_id

        return TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=structure_type,
            entry_price=close,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,        # Pipeline overrides with proper sizing
            notional=0.0,
            confidence=confidence,
            notes=notes,
            trade_id=event_trade_id,
            # gap_fade_short targets are anchored to gap structure (gap_low /
            # PDC / VWAP). A late fill should NOT push targets further away —
            # price respects the level, not arithmetic distance from entry.
            target_anchor_type="structural",
        )

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext
    ) -> RiskParams:
        """Compute risk params for a SHORT entry at entry_price.

        Stop is ABOVE entry (since shorting). Based on gap_high + ATR buffer.
        """
        atr = self._get_atr(market_context)
        df = market_context.df_5m
        gap_high = float(df.iloc[0]["high"]) if df is not None and len(df) >= 1 else entry_price
        cap_seg = getattr(market_context, "cap_segment", None) or ""
        gap_buf = 1.005 if cap_seg == "micro_cap" else 1.0025
        stop_a = gap_high * gap_buf
        stop_b = entry_price + atr * 1.5
        hard_sl = max(stop_a, stop_b)
        stop_distance = max(hard_sl - entry_price, atr * 0.1)
        return RiskParams(
            hard_sl=hard_sl,
            risk_per_share=stop_distance,
            atr=atr,
        )

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        """PDC-based target for short (1R default)."""
        entry = trade_plan.entry_price
        hard_sl = trade_plan.risk_params.hard_sl
        risk = abs(hard_sl - entry)
        t1 = entry - risk   # 1R target (below entry for short)
        return ExitLevels(
            targets=[{"level": t1, "qty_pct": 100, "rr": 1.0}],
            hard_sl=hard_sl,
        )

    def rank_setup_quality(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> float:
        """Proxy: re-run detect and return quality_score."""
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        """Return True if current_time falls within the active window."""
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
