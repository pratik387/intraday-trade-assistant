"""PDH/PDL Touch-and-Reject Fade -- sub8 Setup #4 (rev2: generic Indian retail PDH/PDL fade).

Source citations (Indian market):
  - Groww -- Previous Day High and Low Strategy
  - ChartMantra / TradingQnA retail PDH/PDL threads
  - Goodwill -- Using ATR for Smart Stop-Losses (for buffer rationale)

Rev2 NOTE: rev1 attributed this to "Subasish Pani style" and cited Capital.com.
Both removed -- Subasish Pani's published method is the 5 EMA strategy, NOT
PDH/PDL fade; Capital.com is a UK forex retail site, not Indian. The setup
itself remains as generic Indian-retail PDH/PDL fade with explicit
acknowledgment that volume polarity is unresolved (A/B variant).

Trigger: bar tags PDH (for short) or PDL (for long) within 0.10%, prints a
rejection candle (small body in lower 40%, upper wick > 1.5x body for PDH;
inverse for PDL). Volume polarity is config-driven A/B variant:
  - "absence" (default): bar vol must NOT be >= max_volume_x_recent_for_absence
  - "spike":             bar vol MUST be >= min_volume_x_recent_for_spike

Universe: small + mid F&O (~100 names). Retail-driven flow concentrated here.
"""
from __future__ import annotations

from datetime import time
from typing import Any, Dict, Optional

import pandas as pd

from config.logging_config import get_agent_logger
from services.universe_filter import in_universe
from .base_structure import BaseStructure
from .data_models import (
    ExitLevels, MarketContext, RiskParams, StructureAnalysis, StructureEvent, TradePlan,
)

logger = get_agent_logger()


def _is_wide_open() -> bool:
    """Read top-level wide_open_mode flag from base config."""
    try:
        from pipelines.base_pipeline import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class PDHPDLRejectStructure(BaseStructure):
    """Fade rejections at PDH (short) or PDL (long) with config-driven volume polarity."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "pdh_pdl_reject"
        self.configured_setup_type = config.get("_setup_name")

        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.level_prox_pct = float(config["level_proximity_pct"]) / 100.0
        self.max_body_pct = float(config["max_body_size_pct"]) / 100.0
        self.min_wick_x_body = float(config["min_upper_wick_x_body"])
        # rev2: volume polarity is A/B variant (Indian sources contested on polarity).
        self.volume_polarity = str(config["volume_polarity"]).lower()
        if self.volume_polarity not in ("absence", "spike"):
            raise ValueError(
                f"volume_polarity must be 'absence' or 'spike', got {self.volume_polarity!r}"
            )
        self.max_vol_x_recent_for_absence = float(config["max_volume_x_recent_for_absence"])
        self.min_vol_x_recent_for_spike = float(config["min_volume_x_recent_for_spike"])
        self.wick_buffer_pct = float(config["wick_buffer_pct"]) / 100.0
        self.t1_qty_pct = float(config["t1_qty_pct"])
        self.universe_key = str(config["universe_key"])
        self.min_bars_required = int(config["min_bars_required"])

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = s.split(":")
        return time(int(h), int(m))

    def _get_atr(self, ctx: MarketContext) -> float:
        if ctx.indicators and "atr" in ctx.indicators:
            return float(ctx.indicators["atr"])
        if ctx.df_5m is not None and len(ctx.df_5m) >= 14:
            return float((ctx.df_5m["high"] - ctx.df_5m["low"]).tail(14).mean())
        return ctx.current_price * 0.01

    def _get_vwap(self, ctx: MarketContext) -> Optional[float]:
        if ctx.indicators and "vwap" in ctx.indicators:
            return float(ctx.indicators["vwap"])
        if ctx.df_5m is not None and "vwap" in ctx.df_5m.columns:
            v = ctx.df_5m["vwap"].iloc[-1]
            if pd.notna(v):
                return float(v)
        return None

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        # rev2: design-inferred filters bypass under wide_open_mode
        _wide_open = _is_wide_open()

        # ---- Universe (design-inferred — bypassed under wide_open) ----
        # Per master plan: wide-open OCI capture must see ALL symbols so the
        # gauntlet can decide which universe slice the detector works in.
        if not _wide_open and not in_universe(ctx.symbol, self.universe_key):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        if ctx.pdh is None or ctx.pdl is None:
            return _empty("PDH/PDL unavailable")
        pdh = float(ctx.pdh)
        pdl = float(ctx.pdl)

        last = df.iloc[-1]
        bar_open = float(last["open"])
        bar_high = float(last["high"])
        bar_low = float(last["low"])
        bar_close = float(last["close"])
        bar_vol = float(last["volume"])

        body = abs(bar_close - bar_open)
        rng = bar_high - bar_low
        if rng <= 0:
            return _empty("zero-range bar")
        body_pct = body / rng

        # PDH proximity for short, PDL proximity for long
        pdh_band = pdh * self.level_prox_pct
        pdl_band = pdl * self.level_prox_pct

        side: Optional[str] = None
        rejection_extreme: float = 0.0
        if abs(bar_high - pdh) <= pdh_band and bar_close < pdh:
            upper_wick = bar_high - max(bar_open, bar_close)
            # wide_open: bypass inferred wick/body ratio and body% filters; keep body > 0
            if _wide_open:
                if body > 0:
                    side = "short"
                    rejection_extreme = bar_high
            elif body > 0 and upper_wick / body >= self.min_wick_x_body and body_pct < self.max_body_pct:
                side = "short"
                rejection_extreme = bar_high
        elif abs(bar_low - pdl) <= pdl_band and bar_close > pdl:
            lower_wick = min(bar_open, bar_close) - bar_low
            # wide_open: bypass inferred wick/body ratio and body% filters; keep body > 0
            if _wide_open:
                if body > 0:
                    side = "long"
                    rejection_extreme = bar_low
            elif body > 0 and lower_wick / body >= self.min_wick_x_body and body_pct < self.max_body_pct:
                side = "long"
                rejection_extreme = bar_low

        if side is None:
            return _empty(
                f"no PDH/PDL rejection: bar=[{bar_low:.2f},{bar_high:.2f}] "
                f"PDH={pdh:.2f} PDL={pdl:.2f}"
            )

        # rev2: volume polarity branch (Indian sources contested on polarity).
        recent_vol = float(df["volume"].iloc[-6:-1].mean()) if len(df) >= 6 else bar_vol
        if not _wide_open and recent_vol > 0:
            ratio = bar_vol / recent_vol
            if self.volume_polarity == "absence":
                if ratio > self.max_vol_x_recent_for_absence:
                    return _empty(
                        f"absence_polarity_violated: vol {bar_vol:.0f} > "
                        f"{self.max_vol_x_recent_for_absence}x recent"
                    )
            else:  # spike
                if ratio < self.min_vol_x_recent_for_spike:
                    return _empty(
                        f"spike_polarity_violated: vol {bar_vol:.0f} < "
                        f"{self.min_vol_x_recent_for_spike}x recent"
                    )

        confidence = min(1.0, (1.0 - body_pct) + 0.2)
        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side=side,
            confidence=confidence,
            levels={
                "pdh": pdh,
                "pdl": pdl,
                "rejection_extreme": rejection_extreme,
                "close": bar_close,
                "vwap": self._get_vwap(ctx) or bar_close,
            },
            context={
                "body_pct": body_pct,
                "vol_x_recent": bar_vol / max(recent_vol, 1.0),
                "polarity": self.volume_polarity,
            },
            price=bar_close,
        )
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    def _build_plan(self, ctx: MarketContext, side: str) -> Optional[TradePlan]:
        analysis = self.detect(ctx)
        if not analysis.structure_detected:
            return None
        evt = analysis.events[0]
        if evt.side != side:
            return None

        df = ctx.df_5m
        last = df.iloc[-1]
        close = float(last["close"])
        rejection = float(evt.levels["rejection_extreme"])
        wick_buf = close * self.wick_buffer_pct

        if side == "short":
            hard_sl = rejection + wick_buf
            risk = max(hard_sl - close, 1e-6)
        else:
            hard_sl = rejection - wick_buf
            risk = max(close - hard_sl, 1e-6)

        # T1 = VWAP; T2 = today's opposite extreme
        vwap = float(evt.levels["vwap"])
        today_high = float(df["high"].max())
        today_low = float(df["low"].min())
        if side == "short":
            t1_level = vwap if vwap < close else close - risk
            t2_level = today_low if today_low < t1_level else close - 2 * risk
        else:
            t1_level = vwap if vwap > close else close + risk
            t2_level = today_high if today_high > t1_level else close + 2 * risk

        t1_rr = abs(close - t1_level) / risk
        t2_rr = abs(close - t2_level) / risk
        targets = [
            {
                "name": "T1",
                "level": t1_level,
                "rr": t1_rr,
                "qty_pct": self.t1_qty_pct,
                "action": "partial_exit",
            },
            {
                "name": "T2",
                "level": t2_level,
                "rr": t2_rr,
                "qty_pct": round(1.0 - self.t1_qty_pct, 4),
                "action": "exit_full",
            },
        ]
        return TradePlan(
            symbol=ctx.symbol,
            side=side,
            structure_type=self.structure_type,
            entry_price=close,
            risk_params=RiskParams(
                hard_sl=hard_sl,
                risk_per_share=risk,
                atr=self._get_atr(ctx),
            ),
            exit_levels=ExitLevels(hard_sl=hard_sl, targets=targets),
            qty=0,
            notional=0.0,
            confidence=evt.confidence,
            notes=evt.context,
            trade_id=evt.trade_id,
        )

    def plan_long_strategy(
        self, ctx: MarketContext, event=None
    ) -> Optional[TradePlan]:
        return self._build_plan(ctx, "long")

    def plan_short_strategy(
        self, ctx: MarketContext, event=None
    ) -> Optional[TradePlan]:
        return self._build_plan(ctx, "short")

    def calculate_risk_params(
        self, entry_price: float, ctx: MarketContext
    ) -> RiskParams:
        atr = self._get_atr(ctx)
        return RiskParams(
            hard_sl=entry_price + atr,
            risk_per_share=atr,
            atr=atr,
        )

    def get_exit_levels(self, plan: TradePlan) -> ExitLevels:
        return plan.exit_levels

    def rank_setup_quality(
        self, ctx: MarketContext, event=None
    ) -> float:
        return self.detect(ctx).quality_score

    def validate_timing(self, current_time: pd.Timestamp) -> bool:
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
