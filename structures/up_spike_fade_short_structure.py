"""Up Spike Fade SHORT detector — sub-project #9 (T1 up-spike fade).

Thesis: A +5.5%-over-15min intraday surge (trailing-3-bar 5m return >= +5.5%)
on a heavy volume burst (> 5x causal trailing-mean) in a thin small/micro/
unknown-cap is retail FOMO with no institutional supply to meet it -> the
spike over-extends and fades into the close. Mirror of panic_crash_revert_long;
same illiquid over-extension -> reversion family as the LIVE gap_fade_short.

Cell-locked (Discovery-only selection; pure EOD-hold, no T1/T2):
  cap_segment in {small_cap, micro_cap, unknown}
  x trailing-3-bar return >= +5.5%
  x current-bar volume > 5x causal trailing-mean bar volume (strictly-prior
    bars that day; >= min_bars_before_signal prior bars — NO look-ahead, Lesson #25)
  x 09:45-14:00 IST
  x per-day turnover floor >= Rs 1.5cr
  x next-bar-open entry (Mode B), EOD square-off, MIS short.
  x MIS-short eligibility gate: broker intraday leverage > 1 (enforced in the
    universe builder, NOT here — see up_spike_fade_short_universe).

Lock JSON: tools/sub9_research/up_spike_fade_short_cell_selection_locked.json
Brief:     specs/2026-06-03-brief-up_spike_fade_short.md

DO NOT enable in live (`enabled: true`) until the OCI-pipeline coverage gate +
upper-circuit guard + paper A/B pass — see brief Section 12.
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
from services.plan_helpers import (
    PlanRejected,
    assert_sl_outside_entry_zone,
    compute_entry_zone,
    enforce_min_stop_distance,
)

logger = get_agent_logger()


class UpSpikeFadeShortStructure(BaseStructure):
    """SHORT entry on a deep intraday up-spike (>=+5.5% / 15min) + heavy burst."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "up_spike_fade_short"
        self.configured_setup_type = config.get("_setup_name")

        # All params from config — NO hardcoded defaults (project Rule #1).
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.spike_r3_min = float(config["spike_r3_pct_min"])
        self.vol_burst_min = float(config["vol_burst_min"])
        self.min_bars_before_signal = int(config["min_bars_before_signal"])
        self.per_day_turnover_floor_rs = float(config["per_day_turnover_floor_rs"])
        self.allowed_caps = set(config["allowed_cap_segments"])
        self.catastrophe_stop_pct = float(config["catastrophe_stop_pct"])
        self.min_bars_required = int(config["min_bars_required"])

        # First-fire-per-(symbol, date) latch — one entry per day, first
        # qualifying spike (mirrors gap_fade_short).
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

        # Cap segment gate (illiquid cohort). MIS-short eligibility (leverage>1)
        # is enforced upstream in the universe builder, not here.
        ctx_cap = (context.cap_segment or "").strip()
        if ctx_cap not in self.allowed_caps:
            return _empty(f"cap_segment={ctx_cap!r} not in {sorted(self.allowed_caps)}")

        # df_5m may carry warmup bars from prior sessions. Filter to TODAY's
        # bars before computing trailing returns / causal volume baseline.
        today_bars = df[df.index.date == _sd]
        if len(today_bars) < (self.min_bars_before_signal + 1):
            return _empty(
                f"only {len(today_bars)} today-bars < "
                f"min_bars_before_signal+1={self.min_bars_before_signal + 1}"
            )
        if len(today_bars) < 4:
            return _empty("need >=4 bars for trailing-3-bar return")

        closes = today_bars["close"].astype("float64")
        vols = today_bars["volume"].astype("float64")
        cur_close = float(closes.iloc[-1])
        close_3ago = float(closes.iloc[-4])
        if close_3ago <= 0:
            return _empty("non-positive close[-4]")

        # --- Trigger 1: trailing-3-bar (15-min) up-spike ---
        r3 = (cur_close / close_3ago) - 1.0
        if r3 < self.spike_r3_min:
            return _empty(f"r3={r3*100:.2f}% < spike_r3_pct_min={self.spike_r3_min*100:.2f}%")

        # --- Trigger 2: causal volume burst ---
        # Mean of STRICTLY-PRIOR bars that day (excludes current bar). NEVER
        # the whole-day mean (look-ahead — Lesson #25).
        cur_vol = float(vols.iloc[-1])
        prior_vols = vols.iloc[:-1]
        if len(prior_vols) < self.min_bars_before_signal:
            return _empty(
                f"only {len(prior_vols)} prior bars < min_bars_before_signal="
                f"{self.min_bars_before_signal}"
            )
        causal_mean_vol = float(prior_vols.mean())
        if causal_mean_vol <= 0:
            return _empty("non-positive causal mean volume")
        vol_burst = cur_vol / causal_mean_vol
        # Cell is "burst > 5x" (STRICTLY greater) — reject == boundary too.
        # (panic_crash_revert_long uses ">= 2x" so its check is "<"; the
        # asymmetry is intentional and matches each locked cell.)
        if vol_burst <= self.vol_burst_min:
            return _empty(f"vol_burst={vol_burst:.2f} <= min={self.vol_burst_min}")

        # --- Liquidity floor: per-day turnover (sum close*volume today) ---
        day_turnover = float((closes * vols).sum())
        if day_turnover < self.per_day_turnover_floor_rs:
            return _empty(
                f"day_turnover={day_turnover:,.0f} < floor="
                f"{self.per_day_turnover_floor_rs:,.0f}"
            )

        # Confidence proxy: bigger spike -> higher confidence. spike floor maps
        # to 0.0; 2x the floor maps to 1.0. Clamp to [0, 1].
        excess = (r3 - self.spike_r3_min) / max(abs(self.spike_r3_min), 1e-9)
        confidence = max(0.0, min(1.0, excess))

        last_bar = today_bars.iloc[-1]
        evt = StructureEvent(
            symbol=context.symbol,
            timestamp=last_ts,
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "signal_bar_open": float(last_bar["open"]),
                "signal_bar_high": float(last_bar["high"]),
                "signal_bar_low": float(last_bar["low"]),
                "signal_bar_close": cur_close,
                "close_3_bars_ago": close_3ago,
            },
            context={
                "r3_pct": r3 * 100.0,
                "vol_burst": vol_burst,
                "day_turnover": day_turnover,
                "cap_segment": ctx_cap,
            },
            price=cur_close,
        )
        # Latch in detect() — runs in MainDetector worker process (see
        # gap_fade_short rationale).
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True, events=[evt],
            quality_score=confidence * 100.0,
        )

    def plan_long_strategy(self, context, event=None):
        """Short-only setup — no long trades."""
        return None

    def plan_short_strategy(
        self,
        context: MarketContext,
        event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Generate a SHORT TradePlan (Mode B = next-bar-open entry).

        Entry: signal-bar close. The orchestrator/executor applies the actual
        fill at the NEXT 5m bar's OPEN (Mode B convention used across this
        codebase — see below_vwap_volume_revert_long_structure). The detector
        records signal-bar close as entry_price; target_anchor_type="structural"
        so a next-bar-open fill does not re-anchor the (single, far) exit.

        Exit structure (validated research = PURE EOD-HOLD, no T1/T2):
          - time_exit at session square-off (config time_stop_at).
          - hard_sl = WIDE catastrophe stop ABOVE entry
            (entry * (1 + catastrophe_stop_pct)) — a blowup guard to cap MIS
            tail risk, NOT a strategy exit. Wide enough to almost never trigger.
          - targets: single far/EOD target (priced at the catastrophe-stop
            distance BELOW entry) so the plan is well-formed; the real exit is
            the time_stop square-off.
        """
        if event is None or event.side != "short":
            return None
        evt = event

        entry = float(evt.levels["signal_bar_close"])
        risk_params = self.calculate_risk_params(entry, context)
        hard_sl = risk_params.hard_sl  # ABOVE entry for short
        if hard_sl <= entry:
            return None
        risk_per_share = hard_sl - entry

        # Single far/EOD target at -catastrophe distance below entry (profit
        # side for a short). EOD-hold means the time_stop normally squares off
        # before this is touched; it exists only so the plan is well-formed.
        eod_target = entry - risk_per_share
        targets = [
            {
                "name": "EOD",
                "level": eod_target,
                "rr": 1.0,
                "qty_pct": 1.0,
                "action": "exit_full",
            },
        ]

        time_exit_str = str(self.config.get("time_stop_at") or "").strip() or None
        exit_levels = ExitLevels(hard_sl=hard_sl, targets=targets, time_exit=time_exit_str)

        try:
            zone = compute_entry_zone(
                entry=entry, bias="short",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(zone, hard_sl, "short")
            enforce_min_stop_distance(
                entry, hard_sl, self.config.get("min_stop_distance_pct"),
            )
        except PlanRejected as e:
            logger.warning(
                f"[{context.symbol}] up_spike_fade_short plan rejected: "
                f"{e.reason} {e.details}"
            )
            return None

        return TradePlan(
            symbol=context.symbol,
            side="short",
            structure_type=evt.structure_type,
            entry_price=entry,
            risk_params=risk_params,
            exit_levels=exit_levels,
            qty=0,
            notional=0.0,
            confidence=evt.confidence,
            notes=evt.context,
            trade_id=evt.trade_id,
            target_anchor_type="structural",
        )

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext
    ) -> RiskParams:
        """WIDE catastrophe stop ABOVE entry (short). Blowup guard, not a
        strategy exit — see plan_short_strategy docstring."""
        hard_sl = entry_price * (1.0 + self.catastrophe_stop_pct / 100.0)
        risk_per_share = max(hard_sl - entry_price, entry_price * 1e-4)
        atr_proxy = entry_price * (self.catastrophe_stop_pct / 100.0)
        df = market_context.df_5m
        if df is not None and not df.empty:
            try:
                atr_proxy = float((df["high"] - df["low"]).mean())
            except Exception:
                pass
        return RiskParams(hard_sl=hard_sl, risk_per_share=risk_per_share, atr=atr_proxy)

    def get_exit_levels(self, trade_plan: TradePlan) -> ExitLevels:
        return ExitLevels(hard_sl=trade_plan.risk_params.hard_sl, targets=[])

    def rank_setup_quality(self, context, event=None):
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time):
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
