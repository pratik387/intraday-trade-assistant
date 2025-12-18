# services/execution/exit_executor.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, Tuple, List
import math
import time
import pandas as pd  # for Timestamp typing only

from config.logging_config import get_execution_loggers
from config.filters_setup import load_filters
from utils.time_util import _to_naive_ist, _now_naive_ist, _minute_of_day, _parse_hhmm_to_md
from diagnostics.diag_event_log import diag_event_log
import uuid

# Both loggers (as requested)
logger, trade_logger = get_execution_loggers()


# ---------- Data model (matches earlier executors) ----------

@dataclass
class Position:
    symbol: str
    side: str                 # "BUY" or "SELL"
    qty: int
    avg_price: float
    plan: Dict[str, Any] = field(default_factory=dict)


class PositionStore:
    """Minimal store interface used by ExitExecutor."""
    def __init__(self) -> None:
        self._pos: Dict[str, Position] = {}

    def list_open(self) -> Dict[str, Position]:
        return dict(self._pos)

    def upsert(self, p: Position) -> None:
        self._pos[p.symbol] = p

    def close(self, sym: str) -> None:
        self._pos.pop(sym, None)

    def reduce(self, sym: str, qty_exit: int) -> None:
        p = self._pos.get(sym)
        if not p:
            return
        nq = int(p.qty) - int(qty_exit)
        if nq <= 0:
            self._pos.pop(sym, None)
        else:
            p.qty = nq
            self._pos[sym] = p


# ---------- Helpers ----------


# ---------- ExitExecutor (LTP-only exits + tick-TS EOD) ----------

class ExitExecutor:
    """
    LTP-only exit engine with:
      • EOD square-off based on tick timestamp (NOT wall clock)
      • Hard SL (static, from plan)
      • T2 (full exit)
      • T1 (one-time partial; then SL -> BE once)
      • Optional trail from cached plan (points/ticks/ATR_cached*mult) — still LTP-only
      • Optional precomputed indicator kills (ORH/ORL, custom kill_levels)
      • Optional score-drop & time-stop
      • Enhanced logging for trade lifecycle tracking

    Single interface:
      - get_ltp_ts(symbol) -> (ltp, ts)

    Position.plan uses:
      - "hard_sl" (or "stop": {"hard": ...})
      - "targets": [{"level": T1}, {"level": T2}]
      - optional "trail": {"points"|("ticks","tick_size")|("atr_cached"/"atr","atr_mult")}
      - optional "orh"/"orl", optional "kill_levels":[{"name","level","direction"}]
      - optional "entry_ts" or "entry_epoch_ms" (for time-stop)
    """

    def __init__(
        self,
        broker,
        positions: PositionStore,
        get_ltp_ts: Callable[[str], Tuple[Optional[float], Optional[pd.Timestamp]]],
        bar_builder=None,  # For tick-level exit validation
        trading_logger=None,  # Enhanced logging service
        capital_manager=None,  # Capital release on exits
        persistence=None,  # Position persistence for crash recovery
    ) -> None:
        self.broker = broker
        self.positions = positions
        self.get_ltp_ts = get_ltp_ts
        self.bar_builder = bar_builder
        self.trading_logger = trading_logger
        self.capital_manager = capital_manager
        self.persistence = persistence  # For updating/removing positions on exit

        # Hook into tick stream for live/paper mode instant exits
        if bar_builder is not None:
            self.original_on_tick = bar_builder.on_tick
            bar_builder.on_tick = self._enhanced_on_tick
            logger.info("ExitExecutor initialized with tick-level validation")
        else:
            self.original_on_tick = None
            logger.info("ExitExecutor initialized (no tick hook - backtest polling only)")

        cfg = load_filters()
        # Config knobs
        self.eod_hhmm = str(cfg.get("eod_squareoff_hhmm", "")) or str(cfg.get("exit_eod_squareoff_hhmm", "1512"))
        self.eod_md = _parse_hhmm_to_md(self.eod_hhmm)

        # Enhanced exit configuration - KeyError if missing trading parameters
        exits_config = cfg.get("exits", {})

        self.score_drop_enabled = bool(exits_config.get("score_drop_enabled", cfg["exit_score_drop_enabled"]))
        self.score_drop_bpct = float(exits_config.get("score_drop_bpct", cfg["exit_score_drop_bpct"]))
        self.time_stop_min = float(exits_config.get("time_stop_min", cfg["exit_time_stop_min"]))
        self.time_stop_req_rr = float(exits_config.get("time_stop_req_rr", cfg["exit_time_stop_req_rr"]))

        self.breakout_short_risk_control = bool(exits_config.get("breakout_short_risk_control", cfg["breakout_short_risk_control"]))
        self.breakout_short_initial_stop_mult = float(exits_config["breakout_short_initial_stop_mult"])
        self.breakout_short_partial_rr = float(exits_config["breakout_short_partial_rr"])
        self.breakout_short_partial_pct = float(exits_config["breakout_short_partial_pct"])
        self.breakout_short_sl_to_neg = float(exits_config["breakout_short_sl_to_neg"])
        self.breakout_short_time_stop_min = float(exits_config["breakout_short_time_stop_min"])
        self.breakout_short_time_stop_max = float(exits_config["breakout_short_time_stop_max"])
        self.breakout_short_time_stop_rr = float(exits_config["breakout_short_time_stop_rr"])

        self.eod_scale_out = bool(exits_config.get("eod_scale_out", cfg["eod_scale_out"]))
        self.eod_scale_out_time1 = str(exits_config["eod_scale_out_time1"])
        self.eod_scale_out_time2 = str(exits_config["eod_scale_out_time2"])
        self.eod_scale_out_rr1 = float(exits_config["eod_scale_out_rr1"])
        self.eod_scale_out_rr2 = float(exits_config["eod_scale_out_rr2"])
        self.eod_scale_out_pct1 = float(exits_config["eod_scale_out_pct1"])

        # T1 behavior - PHASE 2.5: No defaults, must be in config
        self.t1_book_pct = float(cfg["exit_t1_book_pct"])
        self.t1_move_sl_to_be = bool(cfg.get("exit_t1_move_sl_to_be", True))

        # PHASE 2.5: T2 and trailing stop behavior - No defaults, must be in config
        self.t2_book_pct = float(cfg["exit_t2_book_pct"])
        self.trail_atr_mult = float(cfg["exit_trail_atr_mult"])

        # PHASE 3: Time-based trail tightening
        self.trail_time_tighten = str(cfg.get("exit_trail_time_tighten", "14:30"))
        self.trail_atr_mult_late = float(cfg.get("exit_trail_atr_mult_late", 1.5))

        # Time-based SL widening (Pro Trader Standard - reduces morning whipsaw)
        self.sl_time_widening_enabled = bool(cfg["sl_time_widening_enabled"])
        self.sl_time_widening_after_minutes = float(cfg["sl_time_widening_after_minutes"])
        self.sl_time_widening_atr_add = float(cfg["sl_time_widening_atr_add"])
        self.sl_time_widening_max_r_from_entry = float(cfg["sl_time_widening_max_r_from_entry"])
        self.sl_time_widening_second_after_minutes = float(cfg["sl_time_widening_second_after_minutes"])
        self.sl_time_widening_second_atr_add = float(cfg["sl_time_widening_second_atr_add"])

        # PRO TRADER: ORB-specific max hold time (Crabel: ideal trade profits instantly)
        # Pro traders hold ORB for 30-90 min max. Exit if no target hit within this time.
        self.orb_max_hold_minutes = float(cfg.get("orb_max_hold_minutes", 60))

        # execution params
        self.exec_product = str(cfg.get("exec_product", "MIS")).upper()
        self.exec_variety = str(cfg.get("exec_variety", "regular")).lower()
        self.exec_mode    = str(cfg.get("exec_order_mode", "MARKET")).upper()

        logger.info(
            f"exit_executor: init eod={self.eod_hhmm} "
            f"score_drop={self.score_drop_enabled}:{self.score_drop_bpct}% "
            f"time_stop={self.time_stop_min}m@RR<{self.time_stop_req_rr} "
            f"t1_pct={self.t1_book_pct}% t2_pct={self.t2_book_pct}% "
            f"trail={self.trail_atr_mult}x->{self.trail_atr_mult_late}x@{self.trail_time_tighten} "
            f"sl2be={self.t1_move_sl_to_be}"
        )

        self._peak_price: Dict[str, float] = {}
        self._trail_price: Dict[str, float] = {}
        self._closing_state = {}

        # OR_kill enhancement state tracking
        self._or_kill_observation: Dict[str, Dict] = {}  # Track OR observation states
        self._last_momentum_check: Dict[str, float] = {}  # Cache momentum calculations

        # ISSUE 1 FIX: Enable intrabar inference for T1/SL race condition
        self.intrabar_inference_enabled = bool(cfg.get("exit_intrabar_inference_enabled", True))

        # OR_kill enhancement config
        self.or_kill_time_adaptive = bool(cfg.get("or_kill_time_adaptive", True))
        self.or_kill_volume_confirmation = bool(cfg.get("or_kill_volume_confirmation", True))
        self.or_kill_momentum_filter = bool(cfg.get("or_kill_momentum_filter", True))
        self.or_kill_partial_exit_pct = float(cfg.get("or_kill_partial_exit_pct", 30.0))
        self.or_kill_observation_minutes = float(cfg.get("or_kill_observation_minutes", 15.0))
        self.or_kill_volume_multiplier = float(cfg.get("or_kill_volume_multiplier", 1.5))
        self.or_kill_early_buffer_mult = float(cfg.get("or_kill_early_buffer_mult", 1.0))
        self.or_kill_mid_buffer_mult = float(cfg.get("or_kill_mid_buffer_mult", 0.75))
        self.or_kill_late_buffer_mult = float(cfg.get("or_kill_late_buffer_mult", 0.25))
        self.or_kill_major_break_mult = float(cfg.get("or_kill_major_break_mult", 2.0))

    def _enhanced_on_tick(self, symbol: str, price: float, volume: float, ts) -> None:
        """
        Enhanced tick handler that checks exits for open positions.

        Note: 'price' parameter is required by bar_builder.on_tick signature but not used here.
        We call broker.get_ltp_with_level() instead for polymorphic behavior.
        """
        # Call original on_tick first
        if callable(self.original_on_tick):
            try:
                self.original_on_tick(symbol, price, volume, ts)
            except Exception as e:
                logger.exception(f"Original on_tick failed for {symbol}: {e}")

        # Check exits on tick - broker.get_ltp_with_level() handles live vs backtest polymorphically
        self._check_tick_exits(symbol, ts)

    def _check_tick_exits(self, symbol: str, ts) -> None:
        """
        Check if tick hits SL/targets for open positions.

        Uses broker.get_ltp_with_level() for polymorphic behavior:
        - Live/paper: Returns current LTP (real-time tick)
        - Backtest: Checks if bar OHLC touched level, returns level or close
        """
        try:
            open_pos = self.positions.list_open()
            pos = open_pos.get(symbol)
            if not pos:
                return  # No open position for this symbol

            # Convert timestamp
            ts_pd = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts

            # Check SL - broker handles intrabar accuracy polymorphically
            plan_sl = self._get_plan_sl(pos.plan)
            if not math.isnan(plan_sl):
                sl_px = self.broker.get_ltp_with_level(symbol, check_level=plan_sl)
                if sl_px is not None and self._breach_sl(pos.side, sl_px, plan_sl):
                    logger.info(f"TICK_SL_HIT: {symbol} {pos.side} price={sl_px:.2f} sl={plan_sl:.2f}")
                    self._exit(symbol, pos, sl_px, ts_pd, "tick_sl")
                    return

            # Check targets - broker handles intrabar accuracy polymorphically
            t1, t2 = self._get_targets(pos.plan)
            st = pos.plan.get("_state") or {}
            t1_done = bool(st.get("t1_done", False))

            # T2 (full exit)
            if not math.isnan(t2):
                t2_px = self.broker.get_ltp_with_level(symbol, check_level=t2)
                if t2_px is not None and self._target_hit(pos.side, t2_px, t2):
                    logger.info(f"TICK_T2_HIT: {symbol} {pos.side} price={t2_px:.2f} t2={t2:.2f}")
                    self._exit(symbol, pos, t2_px, ts_pd, "tick_target_t2")
                    return

            # T1 (partial exit)
            if (not t1_done) and not math.isnan(t1):
                t1_px = self.broker.get_ltp_with_level(symbol, check_level=t1)
                if t1_px is not None and self._target_hit(pos.side, t1_px, t1):
                    logger.info(f"TICK_T1_HIT: {symbol} {pos.side} price={t1_px:.2f} t1={t1:.2f}")
                    self._partial_exit_t1(symbol, pos, t1_px, ts_pd)
                    return

        except Exception as e:
            logger.exception(f"Tick exit check failed for {symbol}: {e}")

    def run_once(self) -> None:
        open_pos = self.positions.list_open()
        if not open_pos:
            return

        for sym, pos in open_pos.items():
            try:
                px, ts = self._get_px_ts(sym)
                if px is None or ts is None:
                    continue

                # Track MAE/MFE (Maximum Adverse/Favorable Excursion) for exit diagnostics
                st = pos.plan.get('_state', {})
                entry_price = pos.avg_price
                side = pos.side.upper()

                # Calculate current excursion
                if side == 'BUY':
                    current_excursion = px - entry_price
                else:
                    current_excursion = entry_price - px

                # Update MAE (worst drawdown)
                current_mae = st.get('mae', 0.0)
                if current_excursion < current_mae:
                    st['mae'] = current_excursion
                    pos.plan['_state'] = st

                # Update MFE (best profit)
                current_mfe = st.get('mfe', 0.0)
                if current_excursion > current_mfe:
                    st['mfe'] = current_excursion
                    pos.plan['_state'] = st

                # 0) EOD square-off by tick timestamp
                if self.eod_md is not None and _minute_of_day(ts) >= self.eod_md:
                    self._exit(sym, pos, float(px), ts, f"eod_squareoff_{self.eod_hhmm}")
                    continue

                # 0.5) PRO TRADER: Failed breakout exit for ORB trades
                # If candle closes back inside OR, exit immediately (Crabel standard)
                setup_type = pos.plan.get("setup_type", "")
                if setup_type in ("orb_breakout_long", "orb_breakdown_short"):
                    levels = pos.plan.get("levels", {})
                    orh = levels.get("ORH") or levels.get("orh")
                    orl = levels.get("ORL") or levels.get("orl")

                    if orh is not None and orl is not None:
                        # Check if candle CLOSE is back inside OR (using px as close price)
                        candle_close = float(px)  # At bar end, px is the close

                        if setup_type == "orb_breakout_long" and candle_close < orh:
                            # Long breakout failed - price closed back below ORH
                            logger.info(f"FAILED_BREAKOUT | {sym} | LONG closed below ORH | Close: {candle_close:.2f} < ORH: {orh:.2f}")
                            self._exit(sym, pos, candle_close, ts, "failed_breakout_back_inside_or")
                            continue

                        elif setup_type == "orb_breakdown_short" and candle_close > orl:
                            # Short breakdown failed - price closed back above ORL
                            logger.info(f"FAILED_BREAKOUT | {sym} | SHORT closed above ORL | Close: {candle_close:.2f} > ORL: {orl:.2f}")
                            self._exit(sym, pos, candle_close, ts, "failed_breakout_back_inside_or")
                            continue

                # 0.6) PRO TRADER: ORB max hold time (Crabel: ideal trade profits instantly)
                # Exit ORB trades if held longer than max hold time without hitting target
                if setup_type in ("orb_breakout_long", "orb_breakdown_short"):
                    entry_ts_str = pos.plan.get("entry_ts") or pos.plan.get("decision_ts")
                    if entry_ts_str:
                        try:
                            entry_ts = pd.Timestamp(entry_ts_str)
                            time_held_minutes = (ts - entry_ts).total_seconds() / 60.0

                            if time_held_minutes >= self.orb_max_hold_minutes:
                                # Check if still no T1 hit
                                st = pos.plan.get("_state") or {}
                                t1_done = bool(st.get("t1_done", False))

                                if not t1_done:
                                    logger.info(f"ORB_TIME_STOP | {sym} | Held {time_held_minutes:.0f}m >= {self.orb_max_hold_minutes}m | No T1 hit | Exiting")
                                    self._exit(sym, pos, float(px), ts, f"orb_time_stop_{time_held_minutes:.0f}m")
                                    continue
                        except Exception as e:
                            logger.debug(f"ORB time stop check failed for {sym}: {e}")

                # 1) Hard SL first, but anchor to ATR sanity if available
                plan_sl = self._get_plan_sl(pos.plan)
                original_sl = plan_sl  # Store original for logging
                if not math.isnan(plan_sl):
                    atr_min_mult = float(load_filters().get("exit_sl_atr_mult_min", 1.0))
                    atr_cached = float(pos.plan.get("indicators", {}).get("atr", float("nan")))
                    if (not math.isnan(atr_cached)) and atr_min_mult > 0:
                        # Calculate ATR-based minimum SL
                        if pos.side.upper() == "BUY":
                            atr_based_sl = float(pos.avg_price) - atr_min_mult * atr_cached
                            new_plan_sl = min(plan_sl, atr_based_sl)
                        else:
                            atr_based_sl = float(pos.avg_price) + atr_min_mult * atr_cached
                            new_plan_sl = max(plan_sl, atr_based_sl)

                        # Log ATR expansion if it changed the SL
                        if abs(new_plan_sl - plan_sl) > 1e-6:  # Check for meaningful change
                            expansion_amount = abs(new_plan_sl - plan_sl)
                            expansion_direction = "expanded" if (
                                (pos.side.upper() == "BUY" and new_plan_sl < plan_sl) or
                                (pos.side.upper() == "SELL" and new_plan_sl > plan_sl)
                            ) else "tightened"

                            logger.info(
                                f"ATR_SL_ADJUSTMENT | {sym} | {pos.side} | "
                                f"Original_SL: {plan_sl:.2f} | ATR_Based_SL: {atr_based_sl:.2f} | "
                                f"Final_SL: {new_plan_sl:.2f} | ATR: {atr_cached:.3f} | "
                                f"Entry: {pos.avg_price:.2f} | Adjustment: {expansion_direction} by {expansion_amount:.2f}"
                            )

                        plan_sl = new_plan_sl

                # Apply time-based SL widening (Pro Trader Standard - reduces morning whipsaw)
                plan_sl = self._apply_time_based_sl_widening(sym, pos, plan_sl, float(px), ts)

                # Get state early to check if T1/T2 were already hit (for better exit reason labeling)
                st = pos.plan.get("_state") or {}
                t1_done = bool(st.get("t1_done", False))
                t2_done = bool(st.get("t2_done", False))

                # ISSUE 1 FIX: Get T1 early for intrabar inference
                t1_early, _ = self._get_targets(pos.plan)

                # ISSUE 1 FIX: Check T1 FIRST if intrabar inference suggests T1 was hit before SL
                # This prevents the race condition where SL is checked first even though T1 was hit first
                if (not t1_done) and not math.isnan(t1_early) and not math.isnan(plan_sl):
                    if self._intrabar_t1_first(sym, pos.side, t1_early, plan_sl):
                        # T1 was likely hit first - process T1 before SL
                        t1_px = self.broker.get_ltp_with_level(sym, check_level=t1_early)
                        if t1_px is not None and self._target_hit(pos.side, t1_px, t1_early):
                            t1_ltp = t1_px
                            self._partial_exit_t1(sym, pos, t1_ltp, ts)
                            continue  # T1 partial done, SL will be BE now

                # Check SL with intrabar accuracy - broker handles live vs backtest polymorphically
                if not math.isnan(plan_sl):
                    sl_px = self.broker.get_ltp_with_level(sym, check_level=plan_sl)
                    if sl_px is not None and self._breach_sl(pos.side, sl_px, plan_sl):
                        sl_ltp = sl_px
                        # Enhanced SL exit logging with T1/T2 awareness
                        slippage = abs(sl_ltp - plan_sl)
                        # Differentiate SL hit: after T2 partial > after T1 partial > initial SL
                        if t2_done:
                            exit_reason = "sl_post_t2"
                        elif t1_done:
                            exit_reason = "sl_post_t1"
                        else:
                            exit_reason = "hard_sl"

                        logger.info(
                            f"SL_BREACH | {sym} | {pos.side} | "
                            f"Exit_Price: {sl_ltp:.2f} | Final_SL: {plan_sl:.2f} | "
                            f"Original_SL: {original_sl:.2f} | Slippage: {slippage:.2f} | "
                            f"Entry: {pos.avg_price:.2f} | T1_Done: {t1_done} | T2_Done: {t2_done}"
                        )
                        self._exit(sym, pos, sl_ltp, ts, exit_reason)
                        continue

                # Phase 2.5: Fast Scalp Lane time-based stops
                if self._check_fast_scalp_time_stop(sym, pos, float(px), ts):
                    continue  # Exit handled internally

                # Phase 2.5: Auto-breakeven for fast scalp after favorable move
                if self._check_fast_scalp_auto_be(sym, pos, float(px)):
                    pass  # SL updated, continue monitoring

                # 2) Targets & state (already retrieved above for SL check)
                t1, t2 = self._get_targets(pos.plan)

                # Debug: Log target levels for validation (Option B fix verification)
                if not math.isnan(t1) and not math.isnan(t2):
                    entry = pos.plan.get("entry", 0)
                    sl = pos.plan.get("sl", 0)
                    risk = abs(entry - sl) if entry and sl else 0
                    t1_r = abs(t1 - entry) / risk if risk > 0 else 0
                    t2_r = abs(t2 - entry) / risk if risk > 0 else 0
                    logger.debug(f"TARGET_CHECK: {sym} entry={entry:.2f} sl={sl:.2f} risk={risk:.2f} | T1={t1:.2f} ({t1_r:.2f}R) T2={t2:.2f} ({t2_r:.2f}R)")

                # 2a) T2 (PHASE 2: partial exit - 40% of remaining, leave 20% for trail)
                # Note: t1_done and t2_done already retrieved at top of loop (line 325-326)
                if (not t2_done) and not math.isnan(t2):
                    t2_px = self.broker.get_ltp_with_level(sym, check_level=t2)
                    if t2_px is not None and self._target_hit(pos.side, t2_px, t2):
                        t2_ltp = t2_px
                        self._partial_exit_t2(sym, pos, t2_ltp, ts)
                        continue

                # 2b) T1 (one-time partial) - broker handles intrabar accuracy polymorphically
                if (not t1_done) and not math.isnan(t1):
                    t1_px = self.broker.get_ltp_with_level(sym, check_level=t1)
                    if t1_px is not None and self._target_hit(pos.side, t1_px, t1):
                        t1_ltp = t1_px
                        self._partial_exit_t1(sym, pos, t1_ltp, ts)
                        continue

                # 3) Dynamic trail (tighten-only) - broker handles intrabar accuracy polymorphically
                if self._has_trail(pos.plan) and self._trail_allowed(pos, ts):
                    level, why = self._trail_from_plan(sym, pos, float(px), ts)
                    if not math.isnan(level):
                        trail_px = self.broker.get_ltp_with_level(sym, check_level=level)
                        if trail_px is not None and self._breach_sl(pos.side, trail_px, level):
                            trail_ltp = trail_px
                            self._exit(sym, pos, trail_ltp, ts, f"trail_stop({why})")
                            continue

                # 4) Indicator kill-switches (precomputed levels)
                if self._or_kill(sym, pos.side, float(px), pos.plan):
                    self._exit(sym, pos, float(px), ts, "or_kill")
                    continue
                reason = self._custom_kill(pos.side, float(px), pos.plan)
                if reason:
                    self._exit(sym, pos, float(px), ts, reason)
                    continue

                # 5) Score-drop (% from peak since entry)
                sdrop = self._score_drop_price(sym, pos.side, float(px))
                if sdrop:
                    self._exit(sym, pos, float(px), ts, sdrop)
                    continue

                # 6) Breakout short risk control
                if self._breakout_short_risk_control_triggered(sym, pos, float(px), ts):
                    continue  # Handled internally

                # 7) EOD scale-out
                if self._eod_scale_out_triggered(sym, pos, float(px), ts):
                    continue  # Handled internally

                # 8) Time-stop (tick timestamp)
                if self.time_stop_min > 0 and self._time_stop_triggered(pos, float(px), plan_sl, ts):
                    self._exit(sym, pos, float(px), ts, f"time_stop_{self.time_stop_min}m_rr<{self.time_stop_req_rr}")
                    continue

            except Exception as e:
                logger.exception(f"exit_executor: run_once error sym={sym}: {e}")

    def run_forever(self, sleep_ms: int = 200) -> None:
        try:
            while True:
                self.run_once()
                time.sleep(max(0.0, sleep_ms / 1000.0))
        except KeyboardInterrupt:
            logger.info("exit_executor: stop (KeyboardInterrupt)")

    # ---------- Price & timestamp ----------

    def _get_px_ts(self, sym: str) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
        try:
            px, ts = self.get_ltp_ts(sym)
            if ts is not None:
                ts = pd.Timestamp(ts)
            return (None if px is None else float(px)), ts
        except Exception:
            return None, None

    # ---------- Static levels ----------

    def _is_eod(self, ts: pd.Timestamp) -> bool:
        if self.eod_md is None:
            return False
        try:
            return _minute_of_day(ts) >= int(self.eod_md)
        except Exception:
            return False

    def _get_plan_sl(self, plan: Dict[str, Any]) -> float:
        """
        Extract SL from plan using standard format.

        Pipeline contract: plan["stop"]["hard"]
        """
        try:
            return float(plan["stop"]["hard"])
        except (KeyError, TypeError, ValueError):
            return float("nan")

    def _apply_time_based_sl_widening(self, sym: str, pos: Position, plan_sl: float, current_price: float, ts: pd.Timestamp) -> float:
        """
        Apply time-based SL widening to avoid morning volatility whipsaw.

        Pro trader standard: After 30-60 minutes in trade, if price is within 0.5R of entry,
        widen SL by 0.3-0.5 ATR to give trade room to work.

        Returns: Widened SL price (or original if widening not applicable)
        """
        if not self.sl_time_widening_enabled:
            return plan_sl

        if math.isnan(plan_sl):
            return plan_sl

        # Get entry timestamp
        entry_ts_raw = pos.plan.get("entry_ts") or pos.plan.get("entry_epoch_ms")
        if not entry_ts_raw:
            return plan_sl

        try:
            if isinstance(entry_ts_raw, (int, float)) and entry_ts_raw > 1e10:
                entry_ts = pd.Timestamp(entry_ts_raw, unit='ms')
            else:
                entry_ts = pd.Timestamp(entry_ts_raw)
        except Exception:
            return plan_sl

        # Calculate time in trade
        try:
            time_in_trade_minutes = (ts - entry_ts).total_seconds() / 60.0
        except Exception:
            return plan_sl

        # Get ATR for widening calculation
        atr_cached = float(pos.plan.get("indicators", {}).get("atr", float("nan")))
        if math.isnan(atr_cached) or atr_cached <= 0:
            return plan_sl

        # Calculate current R from entry
        entry_price = pos.avg_price
        side = pos.side.upper()
        risk = abs(entry_price - plan_sl)
        if risk <= 0:
            return plan_sl

        if side == "BUY":
            current_r = (current_price - entry_price) / risk
        else:
            current_r = (entry_price - current_price) / risk

        # Check state to avoid re-widening
        st = pos.plan.get("_state") or {}
        first_widening_done = st.get("sl_time_widening_1_done", False)
        second_widening_done = st.get("sl_time_widening_2_done", False)

        widened_sl = plan_sl

        # First widening: after 30 minutes (configurable)
        if not first_widening_done and time_in_trade_minutes >= self.sl_time_widening_after_minutes:
            # Only widen if trade is within max_r from entry
            if abs(current_r) <= self.sl_time_widening_max_r_from_entry:
                if side == "BUY":
                    widened_sl = plan_sl - (self.sl_time_widening_atr_add * atr_cached)
                else:
                    widened_sl = plan_sl + (self.sl_time_widening_atr_add * atr_cached)

                # Update plan with widened SL
                if isinstance(pos.plan.get("stop"), dict):
                    pos.plan["stop"]["hard"] = widened_sl
                pos.plan["hard_sl"] = widened_sl

                # Mark first widening done
                st["sl_time_widening_1_done"] = True
                st["sl_time_widening_1_at"] = str(ts)
                pos.plan["_state"] = st

                logger.info(
                    f"SL_TIME_WIDENING_1 | {sym} | {side} | "
                    f"Time_In_Trade: {time_in_trade_minutes:.1f}min | Current_R: {current_r:.2f} | "
                    f"Original_SL: {plan_sl:.2f} | Widened_SL: {widened_sl:.2f} | "
                    f"ATR_Add: {self.sl_time_widening_atr_add * atr_cached:.2f}"
                )
                return widened_sl

        # Second widening: after 60 minutes (configurable)
        if first_widening_done and not second_widening_done and time_in_trade_minutes >= self.sl_time_widening_second_after_minutes:
            # Only widen if trade is within max_r from entry
            if abs(current_r) <= self.sl_time_widening_max_r_from_entry:
                if side == "BUY":
                    widened_sl = plan_sl - (self.sl_time_widening_second_atr_add * atr_cached)
                else:
                    widened_sl = plan_sl + (self.sl_time_widening_second_atr_add * atr_cached)

                # Update plan with widened SL
                if isinstance(pos.plan.get("stop"), dict):
                    pos.plan["stop"]["hard"] = widened_sl
                pos.plan["hard_sl"] = widened_sl

                # Mark second widening done
                st["sl_time_widening_2_done"] = True
                st["sl_time_widening_2_at"] = str(ts)
                pos.plan["_state"] = st

                logger.info(
                    f"SL_TIME_WIDENING_2 | {sym} | {side} | "
                    f"Time_In_Trade: {time_in_trade_minutes:.1f}min | Current_R: {current_r:.2f} | "
                    f"Previous_SL: {plan_sl:.2f} | Widened_SL: {widened_sl:.2f} | "
                    f"ATR_Add: {self.sl_time_widening_second_atr_add * atr_cached:.2f}"
                )
                return widened_sl

        return plan_sl

    def _breach_sl(self, side: str, price: float, sl: float) -> bool:
        if math.isnan(sl) or math.isnan(price):
            return False
        side = side.upper()
        return (price <= sl) if side == "BUY" else (price >= sl)

    def _get_targets(self, plan: Dict[str, Any]) -> Tuple[float, float]:
        """
        Extract T1 and T2 from plan using standard format.

        Pipeline contract: plan["targets"][n]["level"]
        """
        t1 = t2 = float("nan")
        targets = plan.get("targets") or []

        if len(targets) > 0:
            try:
                t1 = float(targets[0]["level"])
            except (KeyError, TypeError, ValueError):
                pass
        if len(targets) > 1:
            try:
                t2 = float(targets[1]["level"])
            except (KeyError, TypeError, ValueError):
                pass

        return t1, t2

    def _target_hit(self, side: str, px: float, tgt: float) -> bool:
        if math.isnan(tgt) or math.isnan(px):
            return False
        side = side.upper()
        return (px >= tgt) if side == "BUY" else (px <= tgt)

    def _intrabar_t1_first(self, sym: str, side: str, t1: float, sl: float) -> bool:
        """
        ISSUE 1 FIX: Intrabar inference for T1/SL race condition.

        When both T1 and SL are within the current bar's [low, high] range,
        we infer which was likely hit first based on the bar's direction:

        For LONG positions:
          - Bullish bar (close > open): Price went UP first → T1 likely hit first
          - Bearish bar (close < open): Price went DOWN first → SL likely hit first

        For SHORT positions:
          - Bearish bar (close < open): Price went DOWN first → T1 likely hit first (favorable)
          - Bullish bar (close > open): Price went UP first → SL likely hit first

        Returns True if T1 was likely hit first, False otherwise.
        """
        if not self.intrabar_inference_enabled:
            return False

        # Get bar OHLC from broker's cache
        bar_ohlc = None
        try:
            with self.broker._lp_lock:
                bar_ohlc = self.broker._last_bar_ohlc.get(sym)
        except (AttributeError, KeyError):
            pass

        if not bar_ohlc:
            return False

        try:
            low = float(bar_ohlc["low"])
            high = float(bar_ohlc["high"])
            open_px = float(bar_ohlc["open"])
            close_px = float(bar_ohlc["close"])
        except (KeyError, TypeError, ValueError):
            return False

        # Check if both T1 and SL are within bar range
        t1_in_range = low <= t1 <= high
        sl_in_range = low <= sl <= high

        if not (t1_in_range and sl_in_range):
            # Only one level touched, no race condition
            return False

        # Both levels within bar range - use direction inference
        side = side.upper()
        is_bullish = close_px > open_px

        if side == "BUY":
            # LONG: Bullish bar = T1 first, Bearish bar = SL first
            t1_first = is_bullish
        else:
            # SHORT: Bearish bar = T1 first, Bullish bar = SL first
            t1_first = not is_bullish

        if t1_first:
            logger.info(
                f"INTRABAR_INFERENCE | {sym} | {side} | T1_FIRST | "
                f"Bar: O={open_px:.2f} H={high:.2f} L={low:.2f} C={close_px:.2f} | "
                f"T1={t1:.2f} SL={sl:.2f} | Bullish={is_bullish}"
            )

        return t1_first

    # ---------- Trail ----------

    def _has_trail(self, plan: Dict[str, Any]) -> bool:
        tr = plan.get("trail")
        return isinstance(tr, dict) and len(tr) > 0

    def _trail_from_plan(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp] = None) -> Tuple[float, str]:
        """
        PHASE 2 & 3: Calculate trailing stop with time-based tightening.

        After 14:30, NSE liquidity drops, so we tighten trail from 2.0× ATR to 1.5× ATR.
        This captures profit before EOD volatility while giving room for runners.
        """
        tr = pos.plan.get("trail") or {}
        side = pos.side.upper()
        best = self._trail_price.get(sym)
        level = float("nan"); why = "trail_none"
        try:
            if "points" in tr:
                pts = float(tr["points"])
                level = (px - pts) if side == "BUY" else (px + pts)
                why = f"trail_points{pts}"
            elif "ticks" in tr and "tick_size" in tr:
                ticks = float(tr["ticks"]); tsz = float(tr["tick_size"])
                pts = ticks * tsz
                level = (px - pts) if side == "BUY" else (px + pts)
                why = f"trail_ticks{ticks}x{tsz}"
            elif ("atr_cached" in tr or "atr" in tr) and "atr_mult" in tr:
                atr = float(tr.get("atr_cached", tr.get("atr")))

                # PHASE 3: Time-based trail tightening
                # Check if we're past the tightening time (default 14:30)
                # Use ts (backtest time) if available, otherwise current time (live trading)
                now = ts if ts is not None else pd.Timestamp.now()
                tighten_time_str = self.trail_time_tighten  # e.g., "14:30"
                try:
                    hh, mm = tighten_time_str.split(":")
                    tighten_time = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
                    is_late = now >= tighten_time
                except Exception:
                    is_late = False

                # Use tighter multiplier after tighten_time
                if is_late:
                    mult = self.trail_atr_mult_late  # 1.5× ATR after 14:30
                    why_suffix = "_LATE"
                else:
                    mult = self.trail_atr_mult  # 2.0× ATR before 14:30
                    why_suffix = ""

                pts = atr * mult
                level = (px - pts) if side == "BUY" else (px + pts)
                why = f"trail_ATRx{mult}{why_suffix}"
        except Exception:
            pass

        if not math.isnan(level):
            if best is None:
                tight = level
            else:
                tight = max(best, level) if side == "BUY" else min(best, level)
            self._trail_price[sym] = tight
            return tight, why

        if best is not None:
            return best, "trail_prev"
        return float("nan"), "trail_none"

    # ---------- OR / custom kill ----------

    def _or_kill(self, symbol: str, side: str, px: float, plan: Dict[str, Any]) -> bool:
        """
        Enhanced OR kill logic with time-adaptive buffers, volume confirmation,
        momentum filters, and graduated exit strategy.
        """
        try:
            # Get ORH/ORL from levels (correct structure)
            levels = plan.get("levels", {})
            orh = levels.get("ORH")
            orl = levels.get("ORL")
            orh = None if orh is None else float(orh)
            orl = None if orl is None else float(orl)

            # Get ATR for buffer calculation - CRITICAL SAFETY FIX
            atr_raw = plan.get("indicators", {}).get("atr")
            if atr_raw is None or atr_raw <= 0:
                # CRITICAL: Use conservative fallback instead of arbitrary 1%
                # Conservative approach: Use 0.5% of price as minimum safe buffer
                # This prevents both under-buffering (unsafe) and over-buffering (prevents exits)
                atr = abs(px * 0.005)  # 0.5% fallback - more conservative than 1%
                # Log only once per symbol to avoid spam
                if not hasattr(self, '_atr_fallback_logged'):
                    self._atr_fallback_logged = set()
                if symbol not in self._atr_fallback_logged:
                    logger.debug(f"OR_KILL: {symbol} ATR missing/invalid ({atr_raw}), using conservative 0.5% fallback: {atr:.2f}")
                    self._atr_fallback_logged.add(symbol)
            else:
                atr = float(atr_raw)

        except Exception:
            return False

        if orh is None and orl is None:
            return False

        side = side.upper()

        # Enhanced buffer calculation with time-of-day adaptation
        buffer_mult = self._get_time_adaptive_buffer_mult()

        # Regime-specific adjustments
        regime = plan.get("regime", "").lower()
        if regime in ["chop", "choppy"]:
            buffer_mult *= 1.5  # Larger buffer in choppy markets
        elif regime in ["trend", "trending"]:
            buffer_mult *= 0.8  # Smaller buffer in trending markets

        buffer = atr * buffer_mult

        # Determine OR levels and break severity
        if side == "BUY" and orl is not None:
            or_level = orl
            price_break = orl - px  # positive = breaking below ORL
            kill_level = orl - buffer
        elif side == "SELL" and orh is not None:
            or_level = orh
            price_break = px - orh  # positive = breaking above ORH
            kill_level = orh + buffer
        else:
            return False

        # Check if price is near or breaking OR level
        is_major_break = price_break > (buffer * self.or_kill_major_break_mult)
        is_minor_break = 0 < price_break <= buffer
        is_touching = abs(price_break) <= (buffer * 0.5)

        # Fast path: Major breaks kill immediately (with volume confirmation if enabled)
        if is_major_break:
            if self.or_kill_volume_confirmation and not self._check_volume_confirmation(symbol, plan):
                logger.debug(f"OR_KILL: {symbol} major break but no volume confirmation")
                return False
            logger.info(f"OR_KILL: {symbol} major {side} break {px:.2f} vs {or_level:.2f} (break={price_break:.2f}, buffer={buffer:.2f})")
            return True

        # Graduated path: Minor breaks trigger observation logic
        if is_minor_break or is_touching:
            return self._handle_or_observation(symbol, side, px, or_level, price_break, buffer, plan)

        return False

    def _get_time_adaptive_buffer_mult(self) -> float:
        """Calculate time-of-day adaptive buffer multiplier"""
        if not self.or_kill_time_adaptive:
            return 0.75  # Default fallback

        try:
            current_time = _now_naive_ist()
            minute_of_day = current_time.hour * 60 + current_time.minute

            # Session time windows (IST)
            early_end = 11 * 60    # 11:00 AM
            late_start = 14 * 60   # 2:00 PM

            if minute_of_day < early_end:
                return self.or_kill_early_buffer_mult  # 1.0 - larger buffer early
            elif minute_of_day > late_start:
                return self.or_kill_late_buffer_mult   # 0.25 - smaller buffer late
            else:
                return self.or_kill_mid_buffer_mult    # 0.75 - standard buffer mid-day
        except Exception:
            return 0.75

    def _check_volume_confirmation(self, symbol: str, plan: Dict[str, Any]) -> bool:
        """Check if current volume supports the OR break"""
        try:

            # Get recent daily volume data for comparison
            daily_df = self.broker.get_daily(symbol, days=10)
            if daily_df.empty or len(daily_df) < 5:
                return True  # Not enough data, allow the exit

            # Calculate average daily volume from recent 5 days (excluding today)
            recent_volumes = daily_df['volume'].tail(5)
            avg_volume = recent_volumes.mean()

            # For OR breaks, we expect above-average volume activity
            # Current tick should show elevated volume compared to recent average
            # Since we only have LTP access, we use a conservative threshold
            # Real implementation would check current 5m volume vs 5m average

            # For now, use a relaxed volume requirement for OR breaks
            # This is a conservative placeholder that allows most exits
            return True  # Allow OR exits - volume confirmation would need intraday data

        except Exception as e:
            # CRITICAL FIX: Log volume confirmation failures and use conservative approach
            logger.error(f"VOLUME_CHECK: Volume confirmation failed for {symbol}: {e}")
            # Conservative approach: If we can't verify volume, don't allow exit without explicit override
            return False  # Default to blocking exit on error - safety first

    def _check_momentum_filter(self, symbol: str, side: str, plan: Dict[str, Any]) -> bool:
        """Check momentum indicators to filter false OR breaks"""
        if not self.or_kill_momentum_filter:
            return True

        try:
            # Cache momentum checks (expensive calculation)
            cache_key = f"{symbol}_{side}"
            current_time = time.time()

            if cache_key in self._last_momentum_check:
                last_check = self._last_momentum_check[cache_key]
                if current_time - last_check < 10:  # Cache for 10 seconds
                    return True

            # Update cache
            self._last_momentum_check[cache_key] = current_time

            # Get indicators from plan (these should be calculated elsewhere)
            indicators = plan.get("indicators", {})
            rsi = float(indicators.get("rsi", 50))
            adx = float(indicators.get("adx", 20))

            # Momentum filter logic
            if side == "BUY":
                # For longs being killed on ORL break, check if oversold with momentum
                if rsi < 30 and adx > 25:  # Strong oversold momentum
                    return False  # Don't kill, might bounce
            else:  # SELL
                # For shorts being killed on ORH break, check if overbought with momentum
                if rsi > 70 and adx > 25:  # Strong overbought momentum
                    return False  # Don't kill, might reverse

            return True

        except Exception as e:
            logger.debug(f"Momentum filter error for {symbol}: {e}")
            return False  # Conservative: don't kill on data errors

    def _handle_or_observation(self, symbol: str, side: str, px: float, or_level: float,
                              price_break: float, buffer: float, plan: Dict[str, Any]) -> bool:
        """Handle graduated exit strategy for OR touches/minor breaks"""

        # Get the current position
        pos = self.positions.list_open().get(symbol)
        if not pos:
            # CRITICAL FIX: Don't default to kill when no position found - this masks tracking bugs
            logger.error(f"OR_KILL: No position found for {symbol} - possible position tracking error")
            return False  # Conservative: don't kill if we can't find the position

        # Check if already in observation for this symbol
        obs_key = f"{symbol}_{side}"
        observation = self._or_kill_observation.get(obs_key, {})

        current_time = time.time()

        if not observation:
            # First touch/minor break - start observation
            observation = {
                "start_time": current_time,
                "or_level": or_level,
                "initial_price": px,
                "partial_exit_done": False,
                "side": side
            }
            self._or_kill_observation[obs_key] = observation

            # Trigger partial exit if configured
            if self.or_kill_partial_exit_pct > 0:
                partial_qty = int(pos.qty * self.or_kill_partial_exit_pct / 100.0)
                if partial_qty > 0:
                    logger.info(f"OR_KILL: {symbol} OR touch - starting observation, partial exit {self.or_kill_partial_exit_pct}%")

                    # Execute partial exit
                    try:
                        self._place_and_log_exit(symbol, pos, px, partial_qty, None, "OR_KILL_PARTIAL")
                        exit_result = True
                    except Exception as e:
                        logger.warning(f"OR_KILL: {symbol} partial exit failed: {e}")
                        exit_result = False

                    if exit_result:
                        trade_logger.info(f"OR_KILL_PARTIAL | {symbol} | {pos.side} {partial_qty} @ {px:.2f} | remaining: {pos.qty - partial_qty}")
                        self.positions.reduce(symbol, partial_qty)  # Update position quantity
                        # Update persistence with new qty (crash recovery)
                        if self.persistence:
                            new_qty = pos.qty - partial_qty
                            self.persistence.update_position(symbol, new_qty=new_qty)
                        observation["partial_exit_done"] = True
                        observation["partial_qty_exited"] = partial_qty
                    else:
                        logger.warning(f"OR_KILL: {symbol} partial exit failed")
                        observation["partial_exit_done"] = False

            return False  # Don't kill yet, start observing

        # Check if observation period expired
        elapsed_minutes = (current_time - observation["start_time"]) / 60.0
        if elapsed_minutes >= self.or_kill_observation_minutes:

            # Check momentum filter before final kill
            if not self._check_momentum_filter(symbol, side, plan):
                logger.debug(f"OR_KILL: {symbol} observation expired but momentum filter blocks kill")
                # Reset observation for another cycle
                del self._or_kill_observation[obs_key]
                return False

            logger.info(f"OR_KILL: {symbol} observation period expired ({elapsed_minutes:.1f}m) - final exit")
            del self._or_kill_observation[obs_key]  # Clean up
            return True  # Final kill

        # Still in observation period
        return False   

    def _custom_kill(self, side: str, px: float, plan: Dict[str, Any]) -> Optional[str]:
        kills: List[Dict[str, Any]] = plan.get("kill_levels") or []
        for k in kills:
            try:
                name = str(k.get("name") or "kill")
                lvl = float(k.get("level"))
                if (side.upper() == "BUY" and px < lvl) or (side.upper() == "SELL" and px > lvl):
                    return f"kill_{name}"
            except Exception:
                continue
        return None

    # ---------- Score drop ----------

    def _score_drop_price(self, sym: str, side: str, px: float) -> Optional[str]:
        if not self.score_drop_enabled or self.score_drop_bpct <= 0 or math.isnan(px):
            return None

        cur = self._peak_price.get(sym)
        if cur is None:
            self._peak_price[sym] = px
            return None

        side = side.upper()
        if side == "BUY":
            if px > cur:
                self._peak_price[sym] = px
                return None
            dd = (cur - px) / cur * 100.0
            if dd >= self.score_drop_bpct:
                return f"score_drop_dd{dd:.1f}%>= {self.score_drop_bpct:.1f}%"
        else:
            if px < cur:
                self._peak_price[sym] = px
                return None
            dd = (px - cur) / cur * 100.0
            if dd >= self.score_drop_bpct:
                return f"score_drop_rally{dd:.1f}%>= {self.score_drop_bpct:.1f}%"
        return None

    # ---------- Time stop (tick timestamp) ----------

    def _time_stop_triggered(self, pos: Position, px: float, sl: float, ts: Optional[pd.Timestamp]) -> bool:
        try:
            entry_ts = pos.plan.get("entry_ts")
            if entry_ts:
                start = pd.Timestamp(entry_ts)
            else:
                epoch_ms = pos.plan.get("entry_epoch_ms")
                if epoch_ms is None:
                    return False
                start = pd.Timestamp(epoch_ms, unit="ms")
        except Exception:
            return False

        ref_ts = pd.Timestamp(ts) if ts is not None else _now_naive_ist()
        mins_live = max(0.0, (ref_ts - start).total_seconds() / 60.0)
        if mins_live < float(self.time_stop_min):
            return False

        if math.isnan(sl):
            return False
        side = pos.side.upper()
        r_ps = abs(float(pos.avg_price) - float(sl))
        if r_ps <= 0:
            return False

        rr = ((px - float(pos.avg_price)) / r_ps) if side == "BUY" else ((float(pos.avg_price) - px) / r_ps)
        return rr < float(self.time_stop_req_rr)

    # ---------- Place exits + logging ----------

    def _place_and_log_exit(self, sym: str, pos: Position, exit_px: float, qty_exit: int, ts: Optional[pd.Timestamp], reason: str) -> None:
        try:
            cur = self.positions.list_open().get(sym)
            if cur is not None:
                # CRITICAL FIX: Don't default to 0 - log position tracking issues
                cur_qty_raw = getattr(cur, "qty", None)
                if cur_qty_raw is None:
                    logger.error(f"EXIT: Position tracking error for {sym} - qty attribute missing")
                    cur_qty = int(pos.qty)  # Use original position qty as safer fallback
                else:
                    cur_qty = int(cur_qty_raw or 0)
                    if cur_qty == 0:
                        logger.warning(f"EXIT: Position {sym} shows 0 quantity, using original: {pos.qty}")
                        cur_qty = int(pos.qty)
            else:
                cur_qty = int(pos.qty)
        except Exception as e:
            logger.error(f"EXIT: Failed to get current quantity for {sym}: {e}")
            cur_qty = int(pos.qty)
        qty_exit = int(max(0, min(int(qty_exit), cur_qty)))
        if qty_exit <= 0:
            return

        try:
            exit_side = "SELL" if pos.side.upper() == "BUY" else "BUY"
            # Get trade_id from plan if available (for tagging exit orders)
            trade_id = pos.plan.get("trade_id") if pos.plan else None
            args = {
                "symbol": sym,
                "side": exit_side,
                "qty": int(qty_exit),
                "order_type": self.exec_mode,
                "product": self.exec_product,
                "variety": self.exec_variety,
                "trade_id": trade_id,  # For order tagging (ITDA_xxx) - identifies app orders
            }
            self.broker.place_order(**args)
        except Exception as e:
            logger.warning("exit.place_order failed sym=%s qty=%s reason=%s err=%s", sym, qty_exit, reason, e)

        entry_price = float(pos.avg_price)
        pnl = ((exit_px - entry_price) if pos.side.upper() == "BUY" else (entry_price - exit_px)) * int(qty_exit)

        # REMOVED duplicate trade_logger.info() call for EXIT
        # Reason: Both trade_logger.info() (removed) and trading_logger.log_exit() (below)
        #         write to the SAME trade_logs.log file, creating duplicate EXIT entries
        #
        # Evidence from logs/run_bb5bf6d6_20251013_084000/trade_logs.log:
        #   - Line 3: trade_logger format (basic PnL summary)
        #   - Line 4: trading_logger format (from log_exit with diagnostics)
        #
        # Decision: Use trading_logger.log_exit() as single source of truth
        # Benefits:
        #   - No duplicates in trade_logs.log
        #   - Rich diagnostics (R-multiple, MAE/MFE, time in trade, remaining qty)
        #   - Consistent with TRIGGER logging (also uses trading_logger only)

        # Enhanced logging: Log EXIT to events.jsonl via trading_logger
        # This ensures all exits (partial and full) are captured with rich diagnostics
        if self.trading_logger:
            # Calculate R-multiple (PnL in units of initial risk)
            plan_sl = pos.plan.get('stop', {}).get('hard') if isinstance(pos.plan.get('stop'), dict) else pos.plan.get('hard_sl')
            r_multiple = None
            if plan_sl:
                risk_per_unit = abs(entry_price - plan_sl)
                if risk_per_unit > 0:
                    r_multiple = pnl / (qty_exit * risk_per_unit)

            # Get MAE/MFE from state if tracked
            state = pos.plan.get('_state', {})
            mae = state.get('mae')
            mfe = state.get('mfe')

            # Calculate time in trade
            entry_ts = pos.plan.get('entry_ts')
            time_since_entry_mins = None
            if entry_ts and ts:
                try:
                    entry_time = pd.Timestamp(entry_ts)
                    time_delta = ts - entry_time
                    time_since_entry_mins = time_delta.total_seconds() / 60
                except:
                    pass

            # Determine remaining quantity after this exit
            remaining_qty = max(0, pos.qty - qty_exit)

            exit_data = {
                'symbol': sym,
                'trade_id': pos.plan.get('trade_id', ''),
                'qty': qty_exit,
                'entry_price': entry_price,
                'exit_price': exit_px,
                'pnl': round(pnl, 2),
                'reason': reason,
                'timestamp': str(ts) if ts else str(pd.Timestamp.now()),
                # Edge diagnostic fields for exit analysis
                'diagnostics': {
                    'exit_type': 'partial' if qty_exit < pos.qty else 'full',
                    'remaining_qty': remaining_qty,
                    'r_multiple': round(r_multiple, 2) if r_multiple is not None else None,
                    'mae': round(mae, 2) if mae is not None else None,
                    'mfe': round(mfe, 2) if mfe is not None else None,
                    'time_since_entry_mins': round(time_since_entry_mins, 1) if time_since_entry_mins is not None else None,
                    'regime': pos.plan.get('regime'),
                    'setup_type': pos.plan.get('setup_type'),
                    'acceptance_status': pos.plan.get('quality', {}).get('acceptance_status')
                }
            }
            self.trading_logger.log_exit(exit_data)

        # REMOVED diag_event_log.log_exit() call to eliminate duplicate EXIT events
        # Reason: Both trading_logger.log_exit() (above) and diag_event_log.log_exit() (removed)
        #         write to the SAME events.jsonl file, creating duplicate EXIT entries
        #
        # Evidence from logs/run_3d495b1f_20251013_002316/events.jsonl:
        #   - Line 5: trading_logger format with rich diagnostics (pnl, mae, mfe, r_multiple)
        #   - Line 6: diag_event_log format with minimal data (just reason, qty, price)
        #
        # Decision: Use trading_logger as single source of truth for EXIT events
        # Benefits:
        #   - No duplicates in events.jsonl
        #   - Rich diagnostics (R-multiple, MAE/MFE, time in trade)
        #   - Consistent with TRIGGER event logging (also uses trading_logger only)
        #
        # Note: diag_event_log is legacy system, trading_logger is the new enhanced system

        logger.debug(f"exit_executor: {sym} qty={qty_exit} reason={reason}")

    def _exit(self, sym: str, pos: Position, exit_px: float, ts: Optional[pd.Timestamp], reason: str) -> None:
        # re-read current qty from store just before the full exit
        try:
            cur = self.positions.list_open().get(sym)
            if cur is not None:
                # CRITICAL FIX: Don't default to 0 - log position tracking issues
                qty_raw = getattr(cur, "qty", None)
                if qty_raw is None:
                    logger.error(f"EXIT: Position tracking error for {sym} - qty attribute missing during full exit")
                    qty_now = int(pos.qty)
                else:
                    qty_now = int(qty_raw or 0)
                    if qty_now == 0:
                        logger.warning(f"EXIT: Position {sym} shows 0 quantity during full exit, using original: {pos.qty}")
                        qty_now = int(pos.qty)
            else:
                qty_now = int(pos.qty)
        except Exception as e:
            logger.error(f"EXIT: Failed to get current quantity for full exit {sym}: {e}")
            qty_now = int(pos.qty)

        if qty_now <= 0:
            return

        # Calculate PnL and diagnostics for enhanced logging
        pnl = qty_now * (exit_px - pos.avg_price) if pos.side.upper() == "BUY" else qty_now * (pos.avg_price - exit_px)

        # Extract edge diagnostic fields for exit analysis
        entry_price = pos.avg_price
        side = pos.side.upper()

        # Calculate R-multiple (PnL in units of initial risk)
        plan_sl = pos.plan.get('stop', {}).get('hard') if isinstance(pos.plan.get('stop'), dict) else pos.plan.get('hard_sl')
        r_multiple = None
        if plan_sl:
            risk_per_unit = abs(entry_price - plan_sl)
            if risk_per_unit > 0:
                r_multiple = pnl / (qty_now * risk_per_unit)

        # Get MAE/MFE from state if tracked
        state = pos.plan.get('_state', {})
        mae = state.get('mae')
        mfe = state.get('mfe')

        # Calculate time in trade
        entry_ts = pos.plan.get('entry_ts')
        time_since_entry_mins = None
        if entry_ts and ts:
            try:
                entry_time = pd.Timestamp(entry_ts)
                time_delta = ts - entry_time
                time_since_entry_mins = time_delta.total_seconds() / 60
            except:
                pass

        # Get remaining qty (0 for full exit, >0 for partial)
        state_after = pos.plan.get('_state', {})
        remaining_qty = 0  # Will be 0 for full exits in this function

        # NOTE: _place_and_log_exit() will handle trading_logger.log_exit() call
        # We don't call it here to avoid duplicate EXIT events in events.jsonl
        # _place_and_log_exit() has all the necessary logging logic

        self._place_and_log_exit(sym, pos, float(exit_px), qty_now, ts, reason)
        self.positions.close(sym)

        # Remove from persistence (crash recovery)
        if self.persistence:
            self.persistence.remove_position(sym)

        # Release capital (free margin) on full exit
        if self.capital_manager:
            self.capital_manager.exit_position(sym)

        try:
            st = pos.plan.get("_state") or {}
            st.pop("t1_done", None); st.pop("sl_moved_to_be", None)
            pos.plan["_state"] = st
        except Exception:
            pass
        self._peak_price.pop(sym, None); self._trail_price.pop(sym, None)

        # Clean up OR_kill observation states for this symbol
        obs_keys_to_remove = [k for k in self._or_kill_observation.keys() if k.startswith(f"{sym}_")]
        for obs_key in obs_keys_to_remove:
            self._or_kill_observation.pop(obs_key, None)

        # Clean up momentum check cache
        momentum_keys_to_remove = [k for k in self._last_momentum_check.keys() if k.startswith(f"{sym}_")]
        for momentum_key in momentum_keys_to_remove:
            self._last_momentum_check.pop(momentum_key, None)

        rem = 0
        try:
            cur = self.positions.list_open().get(sym)
            if cur is not None:
                # CRITICAL FIX: Log position tracking issues in remaining quantity check
                rem_raw = getattr(cur, "qty", None)
                if rem_raw is None:
                    logger.error(f"EXIT: Position tracking error for {sym} - qty attribute missing for remaining check")
                else:
                    rem = int(rem_raw or 0)
        except Exception as e:
            logger.error(f"EXIT: Failed to check remaining quantity for {sym}: {e}")
            pass

        # REMOVED third duplicate trade_logger.info() call for EXIT with remaining qty
        # This was adding yet another EXIT log to trade_logs.log (line 5 in evidence)
        # The remaining qty info is already included in trading_logger.log_exit() diagnostics
        # via the 'remaining_qty' field in exit_data['diagnostics']

    def _partial_exit_t1(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp]) -> None:
        # Enhanced partial exit logic - always use partial exits for better R:R
        current_qty = int(pos.qty)
        if current_qty <= 0:
            return

        # Check if T2 is infeasible (T1-only scalp mode) or Fast Scalp Lane
        t2_exit_mode = pos.plan.get("quality", {}).get("t2_exit_mode", None)
        if t2_exit_mode in ("T1_only_scalp", "fast_scalp_T1_only"):
            # Exit 100% at T1 if T2 is not feasible or in Fast Scalp Lane
            reason_suffix = "t2_infeasible" if t2_exit_mode == "T1_only_scalp" else "fast_scalp_lane"
            logger.info(f"exit_executor: {sym} {t2_exit_mode} mode - exiting 100% at T1 ({reason_suffix})")
            self._exit(sym, pos, float(px), ts, f"target_t1_full_{reason_suffix}")
            return

        # Get or store original entry quantity (for consistent 60-40 split)
        # This ensures T1 uses correct qty even if T2 fired first
        st = pos.plan.get("_state") or {}
        if "entry_qty" not in st:
            st["entry_qty"] = current_qty  # Store original entry qty on first access
            pos.plan["_state"] = st
        original_qty = st["entry_qty"]

        # Use config-driven percentage (60-40-0 split from config)
        # FIX: Calculate based on ORIGINAL entry qty, not current remaining
        actual_pct = max(1.0, self.t1_book_pct)

        qty_exit = int(max(1, round(original_qty * (actual_pct / 100.0))))
        qty_exit = min(qty_exit, current_qty)  # Can't exit more than we have

        # Enhanced logic for small quantities
        if qty_exit >= current_qty:
            if current_qty > 2:  # Changed from 1 to 2 - allow partial even for small positions
                qty_exit = max(1, current_qty // 2)  # Take 50% minimum
            else:
                self._exit(sym, pos, float(px), ts, "target_t1_full")
                return

        # Log enhanced partial exit info
        profit_booked = qty_exit * (px - pos.avg_price)
        logger.info(f"exit_executor: {sym} T1_PARTIAL booking {qty_exit}/{original_qty} ({actual_pct:.1f}%) → profit Rs.{profit_booked:.2f} [CONFIG: {actual_pct:.0f}%-{self.t2_book_pct:.0f}%-0% split]")

        self._place_and_log_exit(sym, pos, float(px), int(qty_exit), ts, "t1_partial")
        self.positions.reduce(sym, int(qty_exit))

        st["t1_done"] = True
        st["t1_booked_qty"] = qty_exit
        st["t1_booked_price"] = px
        st["t1_profit"] = profit_booked
        
        # Enhanced breakeven logic - always move to BE after T1
        if not st.get("sl_moved_to_be"):
            try:
                be = float(pos.avg_price)
                # Add small buffer above BE for long positions, below for short
                if pos.side.upper() == "BUY":
                    be_buffer = be + (be * 0.001)  # 0.1% above BE
                else:
                    be_buffer = be - (be * 0.001)  # 0.1% below BE
                
                if isinstance(pos.plan.get("stop"), dict):
                    pos.plan["stop"]["hard"] = be_buffer
                pos.plan["hard_sl"] = be_buffer
                st["sl_moved_to_be"] = True
                st["be_price"] = be_buffer
                logger.debug(f"exit_executor: {sym} T1 hit — moved SL to BE+ @{be_buffer:.2f} (buffer: {be_buffer-be:.3f})")
            except Exception:
                pass
        pos.plan["_state"] = st

        # Update persistence with new qty and state (crash recovery)
        if self.persistence:
            new_qty = current_qty - int(qty_exit)
            self.persistence.update_position(sym, new_qty=new_qty, state_updates={"t1_done": True})

        if self.eod_md is not None and ts is not None:
            try:
                if _minute_of_day(ts) >= int(self.eod_md):
                    cur = self.positions.list_open().get(sym)
                    if cur and int(cur.qty) > 0:
                        self._exit(sym, cur, float(px), ts, f"eod_squareoff_{self.eod_hhmm}")
                        return
            except Exception:
                pass

    def _partial_exit_t2(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp]) -> None:
        """
        T2 partial exit - book config-driven percentage of ORIGINAL entry position.

        FIX: Changed from percentage of remaining (which gave 40-40-20 behavior)
        to percentage of original entry qty (which gives configured 60-40-0 behavior).

        Example with 60-40-0 config:
        - Entry: 100 qty
        - T1: 60% of 100 = 60 qty (leaves 40)
        - T2: 40% of 100 = 40 qty (leaves 0 for trail) ✓ CORRECT

        Old buggy behavior:
        - T2: 40% of 40 remaining = 16 qty (leaves 24) ✗ WRONG (gave 60-16-24 = 40-40-20)
        """
        qty = int(pos.qty)
        if qty <= 0:
            return

        # If no trail configured (T1+T2 >= 100%), exit ALL remaining at T2
        # This avoids rounding issues and matches 60-40-0 intent
        no_trail = (self.t1_book_pct + self.t2_book_pct) >= 100.0
        if no_trail:
            logger.info(f"exit_executor: {sym} T2_FULL exit (no trail configured: {self.t1_book_pct:.0f}%-{self.t2_book_pct:.0f}%-0%)")
            self._exit(sym, pos, float(px), ts, "target_t2_full")
            return

        # Trail is configured - do partial T2 exit
        t2_pct = self.t2_book_pct

        # Get or store original entry quantity (for consistent split)
        st = pos.plan.get("_state") or {}
        if "entry_qty" not in st:
            st["entry_qty"] = qty  # Store original entry qty if T2 fires first
            pos.plan["_state"] = st
        original_entry_qty = st["entry_qty"]

        # Calculate T2 qty as percentage of ORIGINAL entry
        qty_exit = int(max(1, round(original_entry_qty * (t2_pct / 100.0))))
        qty_exit = min(qty_exit, qty)  # Cap at current position size

        # Exit fully if calculated qty covers remaining position
        if qty_exit >= qty:
            self._exit(sym, pos, float(px), ts, "target_t2_full")
            return

        # Log T2 partial exit with original qty context
        profit_booked = qty_exit * (px - pos.avg_price)
        t2_pct_of_original = (qty_exit / original_entry_qty * 100) if original_entry_qty > 0 else 0
        remaining_qty = qty - qty_exit
        trail_pct = 100.0 - self.t1_book_pct - self.t2_book_pct
        logger.info(f"exit_executor: {sym} T2_PARTIAL booking {qty_exit}/{original_entry_qty} orig ({t2_pct_of_original:.1f}%) → profit Rs.{profit_booked:.2f} [CONFIG: {self.t1_book_pct:.0f}%-{self.t2_book_pct:.0f}%-{trail_pct:.0f}%, leaving {remaining_qty} for trail]")

        self._place_and_log_exit(sym, pos, float(px), int(qty_exit), ts, "t2_partial")
        self.positions.reduce(sym, int(qty_exit))

        # Mark T2 as done
        st["t2_done"] = True
        st["t2_booked_qty"] = qty_exit
        st["t2_booked_price"] = px
        st["t2_profit"] = profit_booked
        pos.plan["_state"] = st

        # Update persistence with new qty and state (crash recovery)
        if self.persistence:
            self.persistence.update_position(sym, new_qty=remaining_qty, state_updates={"t1_done": True, "t2_done": True})

    def _fabricate_eod_ts(self, pos: Position) -> pd.Timestamp:
        try:
            ts_src = pos.plan.get("entry_ts") or pos.plan.get("decision_ts")
            d = pd.Timestamp(ts_src).date() if ts_src else None
        except Exception:
            d = None
        try:
            hh, mm = self.eod_hhmm.split(":")
            hh = int(hh); mm = int(mm)
        except Exception:
            hh = 15; mm = 10
        if d is not None:
            return pd.Timestamp(f"{d} {hh:02d}:{mm:02d}:00")
        return _now_naive_ist()

    def square_off_all_open_positions(self, reason_prefix: Optional[str] = None) -> None:
        reason = reason_prefix or f"eod_squareoff_{self.eod_hhmm}"
        try:
            open_pos = self.positions.list_open()
        except Exception:
            open_pos = {}

        for sym, pos in list(open_pos.items()):
            try:
                ltp, ts = (self.get_ltp_ts(sym) if callable(self.get_ltp_ts) else (None, None))
                if ltp is None:
                    try:
                        ltp = float(self.broker.get_ltp(sym))
                    except Exception:
                        ltp = None
                use_px = float(ltp) if ltp is not None else float(pos.avg_price)
                use_reason = reason if ltp is not None else reason + "_mkt_noltp"
                ts_obj = pd.Timestamp(ts) if ts is not None else self._fabricate_eod_ts(pos)
                self._flatten_to_closed(sym, pos, use_px, ts_obj, use_reason)
            except Exception as e:
                logger.warning("square_off_all_open_positions: sym=%s err=%s", sym, e)
                
    def _is_open(self, sym: str) -> bool:
        try:
            open_map = self.positions.list_open() or {}
            return sym in open_map
        except Exception:
            try:
                pos = self.positions.list_open().get(sym)
            except Exception:
                pos = None
            return bool(pos is not None and getattr(pos, "qty", 0) > 0)

    def _flatten_to_closed(self, sym: str, pos, px: float, ts, reason: str) -> None:
        intent_id = uuid.uuid4().hex[:8]
        self._closing_state[sym] = {"state": "closing", "intent_id": intent_id, "reason": reason}
        attempts = 0
        while self._is_open(sym):
            self._exit(sym, pos, float(px), ts, reason if attempts == 0 else f"{reason}_retry{attempts}")
            attempts += 1
            if attempts >= 12:
                break
            try:
                pos = self.positions.list_open().get(sym) or pos
            except Exception:
                pass
        self._closing_state[sym] = {"state": "closed", "intent_id": intent_id, "reason": reason}
        
    def _bars_since_entry(self, pos: Position, now_ts: pd.Timestamp) -> int:
        try:
            ets = pos.plan.get("entry_ts")
            if ets is None: return 9999
            i = pd.to_datetime(ets)
            return int(max(0, (now_ts - i).total_seconds() // 300))  # 5m bars
        except Exception:
            return 9999

    def _trail_allowed(self, pos: Position, now_ts: pd.Timestamp) -> bool:
        cfg = load_filters()
        need_t1 = bool(cfg.get("exit_trail_requires_t1", True))
        delay_bars = int(cfg.get("exit_trail_delay_bars", 4))
        st = pos.plan.get("_state") or {}
        if need_t1 and not st.get("t1_done", False):
            return False
        return self._bars_since_entry(pos, now_ts) >= delay_bars

    def _calculate_rr(self, pos: Position, current_price: float) -> float:
        """Calculate risk-reward ratio for current position."""
        try:
            entry_price = float(pos.avg_price)
            plan_sl = self._get_plan_sl(pos.plan)

            if math.isnan(plan_sl):
                return 0.0

            risk_per_share = abs(entry_price - plan_sl)
            if risk_per_share <= 0:
                return 0.0

            if pos.side.upper() == "BUY":
                reward_per_share = current_price - entry_price
            else:
                reward_per_share = entry_price - current_price

            return reward_per_share / risk_per_share
        except Exception:
            return 0.0

    def _breakout_short_risk_control_triggered(self, sym: str, pos: Position, px: float, ts: pd.Timestamp) -> bool:
        """
        Breakout short risk control: High WR but net negative → losses too fat.
        Initial stop = 0.7× recent swing, partial at 0.6R, move stop to –0.1R after partial,
        add time-stop 30–45m at <0.4R
        """
        if not self.breakout_short_risk_control:
            return False

        # Only apply to breakout_short strategies
        strategy_type = pos.plan.get("strategy_type", "")
        if "breakout_short" not in strategy_type:
            return False

        st = pos.plan.get("_state") or {}
        rr = self._calculate_rr(pos, px)

        # Partial exit at 0.6R
        if not st.get("breakout_short_partial_done", False) and rr >= self.breakout_short_partial_rr:
            qty = int(pos.qty)
            qty_exit = int(max(1, round(qty * (self.breakout_short_partial_pct / 100.0))))
            qty_exit = min(qty_exit, qty)

            if qty_exit > 0:
                logger.info(f"BREAKOUT_SHORT_RISK: {sym} partial exit at {rr:.2f}R - booking {qty_exit}/{qty}")
                self._place_and_log_exit(sym, pos, px, qty_exit, ts, f"breakout_short_partial_{rr:.2f}R")
                self.positions.reduce(sym, qty_exit)

                # Move stop to -0.1R
                try:
                    entry_price = float(pos.avg_price)
                    original_sl = self._get_plan_sl(pos.plan)
                    risk_per_share = abs(entry_price - original_sl)
                    new_sl = entry_price + (self.breakout_short_sl_to_neg * risk_per_share)  # -0.1R

                    if isinstance(pos.plan.get("stop"), dict):
                        pos.plan["stop"]["hard"] = new_sl
                    pos.plan["hard_sl"] = new_sl

                    st["breakout_short_partial_done"] = True
                    st["breakout_short_new_sl"] = new_sl
                    pos.plan["_state"] = st

                    # Update persistence with new qty (crash recovery)
                    if self.persistence:
                        new_qty = qty - qty_exit
                        self.persistence.update_position(sym, new_qty=new_qty, state_updates={"breakout_short_partial_done": True})

                    logger.info(f"BREAKOUT_SHORT_RISK: {sym} moved SL to {self.breakout_short_sl_to_neg}R @ {new_sl:.2f}")
                except Exception as e:
                    logger.warning(f"BREAKOUT_SHORT_RISK: {sym} failed to move SL: {e}")

                return True

        # Time stop for breakout shorts (30-45m)
        try:
            entry_ts = pos.plan.get("entry_ts")
            if entry_ts:
                start = pd.Timestamp(entry_ts)
                mins_live = max(0.0, (ts - start).total_seconds() / 60.0)

                # Check if we're in the time stop window and RR is poor
                if (self.breakout_short_time_stop_min <= mins_live <= self.breakout_short_time_stop_max
                    and rr < self.breakout_short_time_stop_rr):

                    logger.info(f"BREAKOUT_SHORT_RISK: {sym} time stop {mins_live:.1f}m @ RR={rr:.2f} < {self.breakout_short_time_stop_rr}")
                    self._exit(sym, pos, px, ts, f"breakout_short_time_stop_{mins_live:.1f}m_rr{rr:.2f}")
                    return True
        except Exception:
            pass

        return False

    def _eod_scale_out_triggered(self, sym: str, pos: Position, px: float, ts: pd.Timestamp) -> bool:
        """
        EOD scale-out: If still <0.7R by 15:00, exit 50%; if <0.4R by 15:07, close the rest.
        """
        if not self.eod_scale_out:
            return False

        try:
            minute_of_day = _minute_of_day(ts)
            eod_time1_md = _parse_hhmm_to_md(self.eod_scale_out_time1)  # 15:00
            eod_time2_md = _parse_hhmm_to_md(self.eod_scale_out_time2)  # 15:07

            if eod_time1_md is None or eod_time2_md is None:
                return False

            st = pos.plan.get("_state") or {}
            rr = self._calculate_rr(pos, px)

            # First scale-out at 15:00 if <0.7R
            if (minute_of_day >= eod_time1_md and
                not st.get("eod_scale_out_first_done", False) and
                rr < self.eod_scale_out_rr1):

                qty = int(pos.qty)
                qty_exit = int(max(1, round(qty * (self.eod_scale_out_pct1 / 100.0))))
                qty_exit = min(qty_exit, qty)

                if qty_exit > 0:
                    logger.info(f"EOD_SCALE_OUT: {sym} first scale @ {self.eod_scale_out_time1} RR={rr:.2f} < {self.eod_scale_out_rr1} - exit {qty_exit}/{qty}")
                    self._place_and_log_exit(sym, pos, px, qty_exit, ts, f"eod_scale_out_1st_{rr:.2f}R")
                    self.positions.reduce(sym, qty_exit)

                    st["eod_scale_out_first_done"] = True
                    pos.plan["_state"] = st

                    # Update persistence with new qty (crash recovery)
                    if self.persistence:
                        new_qty = qty - qty_exit
                        self.persistence.update_position(sym, new_qty=new_qty, state_updates={"eod_scale_out_first_done": True})

                    return True

            # Final exit at 15:07 if <0.4R
            if (minute_of_day >= eod_time2_md and
                rr < self.eod_scale_out_rr2):

                logger.info(f"EOD_SCALE_OUT: {sym} final exit @ {self.eod_scale_out_time2} RR={rr:.2f} < {self.eod_scale_out_rr2}")
                self._exit(sym, pos, px, ts, f"eod_scale_out_final_{rr:.2f}R")
                return True

        except Exception as e:
            logger.warning(f"EOD_SCALE_OUT: {sym} error: {e}")

        return False

    def _check_fast_scalp_time_stop(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp]) -> bool:
        """
        Phase 2.5: Time-based stop for Fast Scalp Lane.

        Exit if T1 not hit within N bars (default 5 bars = 25 minutes on 5m chart).
        This prevents capital being tied up in stagnant fast scalps.
        """
        lane_type = pos.plan.get("context", {}).get("lane_type")
        if lane_type != "fast_scalp_lane":
            return False

        # Check if already past T1
        st = pos.plan.get("_state") or {}
        if st.get("t1_done", False):
            return False  # Already past T1, let normal exit logic handle

        # Get entry timestamp
        entry_ts = pos.plan.get("entry_timestamp")
        if entry_ts is None or ts is None:
            return False

        # Calculate bars elapsed (5m bars)
        bars_elapsed = (ts - entry_ts).total_seconds() / 300.0  # 300 seconds = 5 minutes

        # Time stop threshold (default 5 bars = 25 minutes)
        time_stop_bars = 5

        if bars_elapsed >= time_stop_bars:
            logger.info(
                f"FAST_SCALP_TIME_STOP | {sym} | "
                f"Bars elapsed: {bars_elapsed:.1f} >= {time_stop_bars} | "
                f"No T1 hit, exiting at market"
            )
            self._exit(sym, pos, px, ts, f"fast_scalp_time_stop_{bars_elapsed:.0f}bars")
            return True

        return False

    def _check_fast_scalp_auto_be(self, sym: str, pos: Position, px: float) -> bool:
        """
        Phase 2.5: Auto-breakeven for Fast Scalp Lane.

        Move SL to breakeven after favorable move (>0.5 RPS profit).
        This locks in zero-loss quickly for fast scalps.
        """
        lane_type = pos.plan.get("context", {}).get("lane_type")
        if lane_type != "fast_scalp_lane":
            return False

        # Check if already moved to BE
        st = pos.plan.get("_state") or {}
        if st.get("sl_moved_to_be_fast_scalp", False):
            return False

        # Calculate profit in terms of RPS (risk per share)
        rps = pos.plan.get("sizing", {}).get("risk_per_share", 0.0)
        if rps <= 0:
            return False

        # Current P&L
        side = pos.side.upper()
        if side == "BUY":
            pnl_per_share = px - pos.avg_price
        else:
            pnl_per_share = pos.avg_price - px

        # Auto-BE threshold: >0.5 RPS profit
        auto_be_threshold = 0.5 * rps

        if pnl_per_share >= auto_be_threshold:
            # Move SL to breakeven
            new_sl = pos.avg_price
            old_sl = self._get_plan_sl(pos.plan)

            logger.info(
                f"FAST_SCALP_AUTO_BE | {sym} | "
                f"Profit: {pnl_per_share:.2f} >= {auto_be_threshold:.2f} ({0.5:.1f}R) | "
                f"Moving SL: {old_sl:.2f} -> BE {new_sl:.2f}"
            )

            # Update plan SL
            pos.plan["hard_sl"] = new_sl
            st["sl_moved_to_be_fast_scalp"] = True
            pos.plan["_state"] = st

            return True

        return False


# (end)
