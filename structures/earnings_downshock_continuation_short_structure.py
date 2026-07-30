"""Earnings Down-Shock Continuation Short detector.

Mechanic
--------
T-1 (the reaction session): an analyst-uncovered, retail-held NSE small/mid-cap
reports a quarterly result the market punishes with a close-to-close move
<= `shock_threshold_pct` (-8%).
T+1 (this session): the selling is not finished — SHORT at the close of the
09:15-09:20 5m bar and cover on a hard time exit at 15:15.

Geometry is deliberately NONE/NONE: no profit target and no working stop.
Phase 5 swept 11 alternatives inside the locked cell and **zero** were
strictly better in BOTH eras — every stop degraded it (sl2 PF 1.36->1.07,
sl3 1.11, sl4 1.17) and every target degraded it (tp1.5 0.99, tp2.5 1.21).
The only stop present is a catastrophe backstop (`catastrophe_stop_pct`,
15%) that exists because production cannot represent a stop-less plan; it
never binds — max adverse excursion across all 220 Discovery trades was
12.97%. Do not "improve" this by adding a tighter stop.

The 15:15 exit is LOCKED BY BROKER MECHANICS, not by performance. Zerodha
auto-squares MIS equity from ~15:20 (Upstox/Angel 15:15, ICICI 15:15-20),
so a session-close exit is unobtainable on an MIS short. 15:20 scored
better in testing (+0.650% vs +0.552%) and was deliberately NOT taken.

Validation chain (all in specs/2026-07-29-brief-earnings_downshock_continuation_short.md)
  Stage 4 real-P&L, Discovery n=220: gross +1.007%/trade PF 2.19;
    net +0.552% PF 1.55 at measured CENTRAL slippage (18.7bp/side)
  Phase 5: 324 combinations, 1 lockable cell, incumbent geometry held;
    demoted-window one-shot n=219 PF 1.44/1.21, PF gap 0.11-0.15 vs the
    0.30 overfit threshold
  Fresh-pool one-shot n=77: PASS by 1.4bp (+0.164% at CONSERVATIVE) —
    ONE MONTH carries it (May +0.983%, June -1.487%, July -2.034%), so the
    supported forward expectation is ~ZERO. Paper is a genuine test.

Division of labour
------------------
The UNIVERSE BUILDER (`services.setup_universe.earnings_downshock_continuation_short_universe`)
owns per-date eligibility: ADV tercile tier, cap segment, MIS-shortability,
and the ASM/GSM surveillance exclusion. This detector owns per-bar signal
logic only: the T-1 earnings-reaction qualification, the shock threshold,
the single-bar entry window, and the circuit-blocked-open guard.

Data dependency: `ctx.df_daily` must carry `is_earnings_reaction_day`,
populated by `services.earnings_reaction_enrichment` from
`data/earnings_calendar/earnings_events.parquet` (repaired 2026-07-28).
Same contract `delivery_pct_anomaly_short` uses for `delivery_pct`. A
missing/stale calendar yields no fires with a loud reason, never a crash —
and is watched by the data-health monitor in `jobs/check_edge_integrity.py`.

All parameters come from config (CLAUDE.md rule 1 — KeyError on missing).
All timestamps IST-naive (rule 2). No `datetime.now()` in any decision path
(rule 3): every time comparison uses the bar timestamp.
"""
from __future__ import annotations

from datetime import date, time
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config.logging_config import get_agent_logger
from services.earnings_reaction_enrichment import enrich as enrich_earnings
from services.earnings_reaction_enrichment import load_error as earnings_load_error
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
    """Read top-level wide_open_mode flag from base config."""
    try:
        from services.config_loader import load_base_config
        return bool(load_base_config().get("wide_open_mode", False))
    except Exception:
        return False


class EarningsDownshockContinuationShortStructure(BaseStructure):
    """Post-earnings down-shock continuation, T+1 intraday SHORT.

    Cross-day state: requires the T-1 daily bar to be an earnings reaction
    day with a move <= shock_threshold_pct. Read from `ctx.df_daily`.

    Single-bar entry: fires only on the 09:15-labelled 5m bar, whose CLOSE
    is the 09:20 print. Production 5m bars are START-labelled (see
    services/ingest/live_tick_handler.py::_attempt_close_5m), so the
    09:15 label and the 09:20 fill are the same bar.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.structure_type = "earnings_downshock_continuation_short"
        self.configured_setup_type = config.get("_setup_name")

        # --- signal (CLAUDE.md rule 1: no defaults, KeyError on missing) ---
        self.shock_threshold_pct = float(config["shock_threshold_pct"])
        self.reaction_same_day_classes = list(config["reaction_same_day_classes"])
        self.reaction_same_day_hour_max = int(config["reaction_same_day_hour_max"])
        self.reaction_max_staleness_days = int(config["reaction_max_staleness_days"])

        # --- entry window (single bar) + hard time exit ---
        self.active_start = self._parse_time(config["active_window_start"])
        self.active_end = self._parse_time(config["active_window_end"])
        self.time_stop_at = str(config["time_stop_at"])

        # --- risk: catastrophe backstop only (geometry is none/none) ---
        self.catastrophe_stop_pct = float(config["catastrophe_stop_pct"])
        self.min_stop_distance_pct = float(config["min_stop_distance_pct"])

        # --- intraday guards ---
        self.exclude_circuit_blocked_open = bool(config["exclude_circuit_blocked_open"])
        self.circuit_block_min_gap_pct = float(config["circuit_block_min_gap_pct"])

        # --- plumbing ---
        self.min_bars_required = int(config["min_bars_required"])
        # Execution-working contract (fail-fast like every other trading
        # parameter — this one gates whether fills are economically valid at all).
        self.entry_order_working_required = bool(config["entry_order_working_required"])
        self.entry_order_working_supported = bool(config["entry_order_working_supported"])
        uk = config.get("universe_key")
        self.universe_key = str(uk) if uk else None

        # Per-(symbol, session) reaction cache + one-fire-per-session latch.
        self._reaction_cache: Dict[Tuple[str, date], Optional[Dict[str, float]]] = {}
        self._fired_today: set = set()

    @staticmethod
    def _parse_time(s: str) -> time:
        h, m = str(s).split(":")
        return time(int(h), int(m))

    # ------------------------------------------------------------------
    # T-1 earnings-reaction qualification (cross-day, cached per session)
    # ------------------------------------------------------------------

    def _qualify_reaction(
        self, df_daily: pd.DataFrame, symbol: str, session_date: date,
    ) -> Optional[Dict[str, float]]:
        """Return reaction stats if the previous session was an earnings
        reaction day with a move <= shock_threshold_pct, else None.

        `df_daily` must contain open/high/low/close/volume. The
        `is_earnings_reaction_day` column is added here if absent, so the
        detector works whether or not the upstream pipeline pre-enriched.
        """
        if df_daily is None or df_daily.empty:
            return None

        if "is_earnings_reaction_day" not in df_daily.columns:
            df_daily = enrich_earnings(
                df_daily,
                symbol,
                same_day_classes=self.reaction_same_day_classes,
                same_day_hour_max=self.reaction_same_day_hour_max,
            )
            if "is_earnings_reaction_day" not in df_daily.columns:
                return None
            if not bool(df_daily["is_earnings_reaction_day"].any()):
                err = earnings_load_error()
                if err:
                    logger.warning(
                        "[%s] earnings calendar unavailable (%s) — no fires possible",
                        symbol, err,
                    )
                return None

        dates = pd.to_datetime(pd.Index(df_daily.index)).normalize().date
        prior = [(i, d) for i, d in enumerate(dates) if d < session_date]
        if len(prior) < 2:
            return None
        t1_idx, t1_date = prior[-1]          # T-1 = the reaction session
        t2_idx, _ = prior[-2]                # T-2 = the pre-reaction close

        # Staleness: a reaction more than N calendar days back is a data gap
        # (suspension/holiday run), not a tradeable T+1.
        if (session_date - t1_date).days > self.reaction_max_staleness_days:
            return None

        row_t1 = df_daily.iloc[t1_idx]
        if not bool(row_t1.get("is_earnings_reaction_day", False)):
            return None

        try:
            close_t1 = float(row_t1["close"])
            close_t2 = float(df_daily.iloc[t2_idx]["close"])
        except (KeyError, ValueError, TypeError):
            return None
        if close_t2 <= 0 or close_t1 <= 0:
            return None

        reaction_move_pct = (close_t1 / close_t2 - 1.0) * 100.0
        # Epsilon guards the INCLUSIVE boundary against float representation:
        # a genuine -8.00% close computes as -7.999999999999996 in IEEE-754 and
        # would otherwise be rejected. The spec is "<= shock_threshold_pct", so
        # this makes the implementation match the spec rather than change it.
        if reaction_move_pct > self.shock_threshold_pct + 1e-9:
            return None  # not a deep-enough down-shock

        return {
            "reaction_move_pct": reaction_move_pct,
            "reaction_close": close_t1,
            "pre_reaction_close": close_t2,
            "reaction_date_iso": pd.Timestamp(t1_date).strftime("%Y-%m-%d"),
        }

    # ------------------------------------------------------------------
    # detect()
    # ------------------------------------------------------------------

    def detect(self, ctx: MarketContext) -> StructureAnalysis:
        """Fire SHORT only on the 09:15-labelled 5m bar (09:20 close)."""

        def _empty(reason: str = "") -> StructureAnalysis:
            return StructureAnalysis(
                structure_detected=False,
                events=[],
                quality_score=0.0,
                rejection_reason=reason or None,
            )

        _wide_open = _is_wide_open()

        if (
            not _wide_open
            and self.universe_key is not None
            and not in_universe(ctx.symbol, self.universe_key)
        ):
            return _empty(f"universe_filter:{ctx.symbol} not in {self.universe_key}")

        df = ctx.df_5m
        if df is None or len(df) < self.min_bars_required:
            return _empty("Insufficient bars")

        # ---- Active window: the single 09:15-labelled bar ----
        last_ts = df.index[-1]
        cur_t = last_ts.time() if hasattr(last_ts, "time") else last_ts
        if not (self.active_start <= cur_t <= self.active_end):
            return _empty(f"Outside active window: {cur_t}")

        session_date = ctx.session_date or pd.Timestamp(last_ts).date()
        latch_key = (ctx.symbol, session_date)
        if latch_key in self._fired_today:
            return _empty("already fired this session")

        # ---- T-1 earnings-reaction qualification (cached per session) ----
        cache_key = (ctx.symbol, session_date)
        if cache_key in self._reaction_cache:
            info = self._reaction_cache[cache_key]
        else:
            info = self._qualify_reaction(ctx.df_daily, ctx.symbol, session_date)
            self._reaction_cache[cache_key] = info
        if info is None:
            return _empty("no qualifying T-1 earnings down-shock")

        today = df[pd.to_datetime(pd.Index(df.index)).normalize().date == session_date]
        if today.empty:
            return _empty("no bars for this session")
        entry_bar = today.iloc[-1]
        bar_close = float(entry_bar["close"])
        bar_open = float(today.iloc[0]["open"])
        if bar_close <= 0:
            return _empty("bad entry price")

        # ---- Circuit-blocked open guard ----
        # A name gapped hard against us and pinned with no range cannot be
        # shorted at the mark. Research measured this at only 0.26-0.28% of
        # events (the T+1 open is ~flat vs the reaction close), so this is a
        # cheap safety net rather than a load-bearing filter.
        if self.exclude_circuit_blocked_open:
            hi = float(entry_bar["high"])
            lo = float(entry_bar["low"])
            vol = float(entry_bar.get("volume", 0.0) or 0.0)
            gap_pct = (bar_open / info["reaction_close"] - 1.0) * 100.0
            # circuit_block_min_gap_pct is NEGATIVE (-1.5): reject only when the
            # session opened PINNED AT A LOWER BAND, i.e. gap_pct <= threshold.
            # A previous `abs(gap_pct) >= threshold` made this always-true for a
            # negative threshold, collapsing the guard to "reject every
            # zero-range bar" and over-rejecting vs the validated construction
            # (research measured this filter at 0.26-0.28% of events).
            if hi == lo and vol > 0 and gap_pct <= self.circuit_block_min_gap_pct:
                return _empty(
                    f"circuit-blocked open (zero-range bar, gap {gap_pct:.2f}% "
                    f"<= {self.circuit_block_min_gap_pct}%)"
                )

        # The config documents that this warning fires on EVERY fire so the
        # order-layer gap can never be silent. At Rs 1L this order is ~25% of the
        # entry MINUTE's turnover in the adv_low tier; a single MARKET order
        # costs materially more than the entire 1.4bp fresh-pool margin.
        if self.entry_order_working_required and not self.entry_order_working_supported:
            logger.warning(
                "EXECUTION_WORKING_UNSUPPORTED | %s | %s | order must be WORKED "
                "across the 5m block but the order layer only sends MARKET; "
                "fills will be worse than every number in the validation chain",
                self.structure_type, ctx.symbol,
            )

        confidence = 0.6
        evt = StructureEvent(
            symbol=ctx.symbol,
            timestamp=pd.Timestamp(last_ts),
            structure_type=self.structure_type,
            side="short",
            confidence=confidence,
            levels={
                "entry_close": bar_close,
                "reaction_close": info["reaction_close"],
                "pre_reaction_close": info["pre_reaction_close"],
            },
            context={
                "reaction_move_pct": info["reaction_move_pct"],
                "reaction_date": info["reaction_date_iso"],
                "session_date_iso": pd.Timestamp(session_date).strftime("%Y-%m-%d"),
                # Surfaces the measured execution constraint to the order
                # layer: at Rs 1L this order is ~25% of the entry MINUTE's
                # turnover in the illiquid tier, so it must be WORKED across
                # the 5m block, never sent as one market order.
                "entry_order_working_required": self.entry_order_working_required,
            },
            price=bar_close,
        )
        # Latch here in detect() (worker-side) — plan_*_strategy runs in a
        # different process and its latch never propagates back.
        self._fired_today.add(latch_key)
        return StructureAnalysis(
            structure_detected=True,
            events=[evt],
            quality_score=confidence * 100.0,
        )

    # ------------------------------------------------------------------
    # plan_*_strategy()
    # ------------------------------------------------------------------

    def plan_long_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Short-only setup — no long trades."""
        return None

    def plan_short_strategy(
        self, ctx: MarketContext, event: Optional[StructureEvent] = None,
    ) -> Optional[TradePlan]:
        """Time-exit-only plan: no targets, catastrophe stop, 15:15 exit."""
        if event is None or event.side != "short":
            return None

        df = ctx.df_5m
        if df is None or df.empty:
            return None
        close = float(df.iloc[-1]["close"])
        if close <= 0:
            return None

        # Catastrophe backstop ONLY. The validated geometry is none/none;
        # production cannot represent a stop-less plan, so the stop sits far
        # enough out that it never binds (max Discovery MAE 12.97% vs 15%).
        hard_sl = close * (1.0 + self.catastrophe_stop_pct / 100.0)
        risk_per_share = max(hard_sl - close, close * 0.001)

        try:
            _zone = compute_entry_zone(
                entry=close, bias="short",
                zone_pct=float(self.config["entry_zone_pct"]),
                zone_mode=str(self.config["entry_zone_mode"]),
            )
            assert_sl_outside_entry_zone(_zone, hard_sl, "short")
            enforce_min_stop_distance(
                close, hard_sl, self.min_stop_distance_pct,
            )
        except PlanRejected as e:
            logger.warning(
                "[%s] earnings_downshock_continuation_short plan rejected: %s %s",
                ctx.symbol, e.reason, e.details,
            )
            return None

        return TradePlan(
            symbol=ctx.symbol,
            side="short",
            structure_type=event.structure_type,
            entry_price=close,
            risk_params=RiskParams(
                hard_sl=hard_sl, risk_per_share=risk_per_share, atr=None,
            ),
            # targets=[] — single exit on the 15:15 time stop, matching the
            # validated construction (cf. close_dn_overnight_long).
            exit_levels=ExitLevels(
                hard_sl=hard_sl, targets=[], time_exit=self.time_stop_at,
            ),
            qty=0,
            notional=0.0,
            confidence=event.confidence,
            notes=event.context,
            trade_id=event.trade_id,
            # No R-multiple targets to recompute; the stop is a fixed pct of
            # entry and the exit is a wall-clock time.
            target_anchor_type="structural",
        )

    # ------------------------------------------------------------------
    # BaseStructure abstract methods (legacy ABC contract)
    # Mirrors up_spike_fade_short, which uses the same
    # catastrophe-stop + no-targets + time-exit shape.
    # ------------------------------------------------------------------

    def calculate_risk_params(
        self, entry_price: float, market_context: MarketContext
    ) -> RiskParams:
        """WIDE catastrophe stop ABOVE entry (short). Blowup guard, not a
        strategy exit — the validated geometry is none/none and this level
        never bound on any Discovery trade (max MAE 12.97% vs 15%)."""
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
        return ExitLevels(
            hard_sl=trade_plan.risk_params.hard_sl,
            targets=[],
            time_exit=self.time_stop_at,
        )

    def rank_setup_quality(self, context, event=None):
        result = self.detect(context)
        return result.quality_score

    def validate_timing(self, current_time):
        t = current_time.time() if hasattr(current_time, "time") else current_time
        return self.active_start <= t <= self.active_end
