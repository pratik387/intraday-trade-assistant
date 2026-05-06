"""Options-vol IV-rank-revert SHORT detector — sub-project #9.

Indian-microstructure-specific asymmetry: Indian single-stock-options
ATM IV-rank ≥ threshold → underlying mean-reverts down intraday on T+0.

Mechanic
--------
T-1 EOD: bhavcopy publishes settlement IV per (symbol, expiry, strike).
A daily batch job (tools/option_chain/build_iv_rank.py) computes ATM IV
for each F&O-listed stock, then a 252-day rolling min-max IV-rank.

T+0 11:00 IST 5m bar: if iv_rank ≥ iv_rank_high_threshold (typically 0.85
per round-4 cell-selection 2026-05-06) AND the 11:00 5m bar is a red
confirmation candle below intraday VWAP, fire SHORT at next 5m bar's open.

Stop = entry × (1 + stop_pct)  (default 1%)
T1 = entry × (1 - 1R) at 50% qty
T2 = entry × (1 - 2R) for remaining 50%
Time stop: 15:10 IST (5 min before MIS auto-square)

Research basis
--------------
- SEBI FY23 retail-F&O loss study (91-93% lose, structural retail option-buyer
  loss): https://www.sebi.gov.in/reports-and-statistics/research/sep-2024/
- Bollen & Whaley 2004, Journal of Finance — net buying pressure shapes IV
- Hill, Balasubramanian, Gregory-Allen — Indian VRP study (SSRN 2495568)
- Indian retail-algo precedent: Stratzy, Choice India, Sensibull, Quantsapp
  publish IV-rank-based strategies at IV-rank ≥ 80 (we use ≥ 85 per cell-
  selection result; 0.95 is the "extreme" cell with PF 1.19)

Brief: specs/2026-05-06-sub-project-9-brief-options_vol_iv_rank_revert.md
Sanity-check (2026-05-06): aggregate PF 0.843 RETIRE; cell-select rescued
SHORT × iv_rank ≥ 0.85 cell at PF 1.131, n=448 over 2yr (2023-2024).
Cell-selection report: reports/sub9_sanity/round4_iv_rank_cell_selection.md
"""
from __future__ import annotations

from datetime import time, date
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.iv_rank_service import get_iv_rank_service
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
    try:
        from services.config_loader import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class OptionsVolIvRankRevertStructure(BaseStructure):
    """SHORT-only IV-rank-revert detector.

    Reads daily IV-rank cache via iv_rank_service. Fires once per
    (symbol, session_date) on the 11:00 5m bar when iv_rank ≥ threshold
    AND the bar is a red candle below intraday VWAP (confirmation).

    LONG side intentionally not implemented — sanity-check found LONG-cell
    PF=0.819 with NIFTY-up gate, dragging aggregate PF to 0.843. Per cell-
    selection 2026-05-06 only the SHORT × iv_rank≥0.85 cell ships.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "options_vol_iv_rank_revert"
        self.configured_setup_type = config.get("_setup_name")

        # Per CLAUDE.md rule 1: every parameter from config — KeyError on missing.
        self.iv_rank_high_threshold = float(config["iv_rank_high_threshold"])
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.stop_pct = float(config["stop_pct"])
        self.t1_r_multiple = float(config["t1_r_multiple"])
        self.t2_r_multiple = float(config["t2_r_multiple"])
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.allowed_caps = set(config["allowed_cap_segments"])
        uk = config.get("universe_key")
        self.universe_key = str(uk) if uk else None
        ar = config.get("allowed_regimes")
        self.allowed_regimes: Optional[set] = set(ar) if ar else None
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
        """Fire SHORT only on the active-window 5m bar when:
          - iv_rank ≥ threshold (T-1 EOD signal)
          - bar is red candle (close < open)
          - bar close < intraday VWAP (downside conviction)
        """

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

        if (
            not _wide_open
            and self.allowed_regimes is not None
            and ctx.regime not in self.allowed_regimes
        ):
            return _empty(
                f"regime {ctx.regime!r} not in allowed set {sorted(self.allowed_regimes)}"
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

        # ---- IV-rank lookup (T-1 EOD signal) ----
        iv_rank = get_iv_rank_service().get_iv_rank(ctx.symbol, session_date)
        if iv_rank is None:
            return _empty(f"no IV-rank for {ctx.symbol} on/before {session_date}")
        if iv_rank < self.iv_rank_high_threshold:
            return _empty(
                f"iv_rank={iv_rank:.3f} < threshold={self.iv_rank_high_threshold}"
            )

        # ---- Confirmation candle: red bar below intraday VWAP ----
        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_close = float(last["close"])
        if bar_close >= bar_open:
            return _empty(f"not a red candle: open={bar_open}, close={bar_close}")

        # VWAP — prefer enriched-feather column; fall back to compute on the fly
        bar_vwap = last.get("vwap")
        if bar_vwap is None or pd.isna(bar_vwap):
            today_bars = df[df.index.date == session_date]
            if today_bars.empty:
                return _empty("no bars for session_date")
            tp = (today_bars["high"] + today_bars["low"] + today_bars["close"]) / 3.0
            cum_tpv = (tp * today_bars["volume"]).cumsum()
            cum_v = today_bars["volume"].cumsum().replace(0, 1e-9)
            bar_vwap = float((cum_tpv / cum_v).iloc[-1])
        else:
            bar_vwap = float(bar_vwap)
        if bar_close >= bar_vwap:
            return _empty(f"close={bar_close} >= VWAP={bar_vwap}")

        # ---- All conditions met ----
        # Confidence: 0.5 at threshold, 1.0 at iv_rank=1.0, linearly scaled.
        denom = max(1.0 - self.iv_rank_high_threshold, 1e-6)
        confidence = 0.5 + 0.5 * (iv_rank - self.iv_rank_high_threshold) / denom
        confidence = max(0.0, min(1.0, confidence))

        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "entry_close": bar_close,
                "bar_vwap": bar_vwap,
                "bar_open": bar_open,
                "bar_high": float(last["high"]),
            },
            context={
                "iv_rank": iv_rank,
                "iv_rank_threshold": self.iv_rank_high_threshold,
                "session_date_iso": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
            },
            price=bar_close,
        )
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
        """SHORT-only setup — no long trades."""
        return None

    def plan_short_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Build TradePlan with arithmetic R-multiple targets."""
        if event is None or event.side != "short":
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])

        hard_sl = close * (1.0 + self.stop_pct)
        risk_per_share = max(hard_sl - close, close * 1e-4)

        t1_target = close - self.t1_r_multiple * risk_per_share
        t2_target = close - self.t2_r_multiple * risk_per_share

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
            hard_sl=hard_sl, risk_per_share=risk_per_share, atr=None,
        )
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets)

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
                f"[{ctx.symbol}] options_vol_iv_rank_revert plan rejected: "
                f"{e.reason} {e.details}"
            )
            return None

        session_date = ctx.session_date or pd.Timestamp(df.index[-1]).date()
        self._fired_today.add((ctx.symbol, session_date))

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
            # Targets are R-multiples — arithmetic, not gap-edges.
            target_anchor_type="arithmetic",
        )

    # ------------------------------------------------------------------
    # BaseStructure abstract methods
    # ------------------------------------------------------------------

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext,
    ) -> RiskParams:
        return RiskParams(
            hard_sl=entry_price * (1.0 + self.stop_pct),
            risk_per_share=entry_price * self.stop_pct,
            atr=None,
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

    def rank_setup_quality(
        self, context: MarketContext, event: Optional[StructureEvent] = None,
    ) -> float:
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
