# services/execution/exit_executor.py
from __future__ import annotations

from typing import Dict, Any, Callable, Optional, Tuple, List
import math
import time
import pandas as pd  # for Timestamp typing only

from config.logging_config import get_execution_loggers
from config.filters_setup import load_filters
from utils.time_util import _to_naive_ist, _now_naive_ist, _minute_of_day, _parse_hhmm_to_md
from utils.price_utils import round_to_tick
from diagnostics.diag_event_log import diag_event_log, SCHEMA_VERSION as _EVT_SCHEMA
import uuid

from services.execution.models import Position
from services.state.position_store import PositionStore

# Both loggers (as requested)
logger, trade_logger = get_execution_loggers()


# ---------- Helpers ----------


def _pos_tid(pos: Position) -> str:
    """Extract trade_id from a Position for PositionStore reduce/close/get calls.
    Mirrors the trade_id resolution in services.state.position_store._trade_id_of
    so identical positions hash to the same key in both modules."""
    tid = (pos.plan or {}).get("trade_id") if hasattr(pos, "plan") else None
    return str(tid) if tid else f"sym:{pos.symbol}"


def _r_multiple(entry_price, pnl, qty, decision_sl, current_sl):
    """PnL in units of the trade's INITIAL (decision-time) risk.

    Must use the decision SL, not the live stop: after T1 the stop is trailed /
    moved to breakeven, which collapses the risk denominator and grossly inflates
    R on scaled winners. Fall back to the current stop only when the decision SL
    was never snapshotted (older positions). Returns None if risk is undefined.
    """
    risk_sl = decision_sl if decision_sl is not None else current_sl
    if not risk_sl:
        return None
    risk_per_unit = abs(float(entry_price) - float(risk_sl))
    if risk_per_unit <= 0:
        return None
    return pnl / (qty * risk_per_unit)


_STALE_TS_THRESHOLD_MIN = 5  # ticks older than this are not authoritative for time-stop


def _is_stale_ts_for_live(ts) -> bool:
    """Return True if `ts` is stale relative to wall-clock IST in live/paper.

    Upstox WebSocket can deliver "last_known_price" / snapshot ticks shortly
    after subscribe whose payload timestamp is from the previous trading
    session. If such a ts is used for time-stop decisions, _minute_of_day(ts)
    can land far past 13:00/15:15 and falsely flatten today's positions
    seconds after entry.

    In DRY_RUN we *want* to use the tick ts (it IS the simulated wall clock),
    so this returns False there. The check is wall-clock-aware only when the
    process isn't a backtest replay.
    """
    try:
        from config.env_setup import env  # local import to avoid cycle
        if getattr(env, "DRY_RUN", False):
            return False
        if ts is None:
            return True
        tsp = pd.Timestamp(ts)
        if tsp.tzinfo is not None:
            tsp = _to_naive_ist(tsp)
        now = _now_naive_ist()
        delta_min = (now - tsp).total_seconds() / 60.0
        # Accept slight future drift (≤1 min) without flagging stale.
        return delta_min > _STALE_TS_THRESHOLD_MIN or delta_min < -1.0
    except Exception:
        return False


def classify_sl_exit_reason(st: dict, *, default_no_t1: str = "hard_sl") -> str:
    """Classify SL exit reason based on T1/T2 booking state.

    `sl_post_t1` and `sl_post_t2` must reflect ACTUAL partial bookings. Both
    flags also fire when partials were SKIPPED (low-R or plan-zero qty_pct=0),
    in which case the SL fires at the ORIGINAL hard_sl level — semantically
    a hard_sl, not a post-T1 stop.

    Args:
        st: position._state dict
        default_no_t1: label when no T1 done at all. "tick_sl" for per-tick
            checks, "hard_sl" for bar-boundary checks.

    Bug context (2026-05-20): prior inline logic checked t1_skipped_low_r but
    missed t1_skipped_plan_zero. ~150 OCI trades for setups with
    t1_partial_qty_pct=0 (circuit_release, or_window_failure) got labeled
    sl_post_t1 despite NEVER booking a T1 partial. Aggregate -Rs 170K was
    expected -1R stop losses, mis-attributed as post-T1 BE-trail failures.
    """
    if st.get("t2_done"):
        return "sl_post_t2"
    t1_done = st.get("t1_done", False)
    if t1_done and not (
        st.get("t1_skipped_low_r") or st.get("t1_skipped_plan_zero")
    ):
        return "sl_post_t1"
    if t1_done:
        return "hard_sl"
    return default_no_t1


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
        api_server=None,  # For processing exit requests from API
    ) -> None:
        self.broker = broker
        self.positions = positions
        self.get_ltp_ts = get_ltp_ts
        self.bar_builder = bar_builder
        self.trading_logger = trading_logger
        self.capital_manager = capital_manager
        self.persistence = persistence  # For updating/removing positions on exit
        self.api_server = api_server  # For exit request queue

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


        self.breakout_short_risk_control = bool(exits_config.get("breakout_short_risk_control", cfg["breakout_short_risk_control"]))
        self.breakout_short_initial_stop_mult = float(exits_config["breakout_short_initial_stop_mult"])
        self.breakout_short_partial_rr = float(exits_config["breakout_short_partial_rr"])
        self.breakout_short_partial_pct = float(exits_config["breakout_short_partial_pct"])
        self.breakout_short_sl_to_neg = float(exits_config["breakout_short_sl_to_neg"])
        self.breakout_short_time_stop_min = float(exits_config["breakout_short_time_stop_min"])
        self.breakout_short_time_stop_max = float(exits_config["breakout_short_time_stop_max"])
        self.breakout_short_time_stop_rr = float(exits_config["breakout_short_time_stop_rr"])


        # T1 behavior - PHASE 2.5: No defaults, must be in config
        self.t1_min_partial_r = float(cfg["exit_t1_min_partial_r"])
        self.t1_book_pct = float(cfg["exit_t1_book_pct"])
        self.t1_move_sl_to_be = bool(cfg.get("exit_t1_move_sl_to_be", True))
        self.be_buffer_pct = float(cfg["exit_be_buffer_pct"]) / 100.0  # Convert 0.1 -> 0.001

        # PHASE 2.5: T2 and trailing stop behavior - No defaults, must be in config
        self.t2_book_pct = float(cfg["exit_t2_book_pct"])
        self.trail_atr_mult = float(cfg["exit_trail_atr_mult"])

        # PHASE 3: Time-based trail tightening
        self.trail_time_tighten = str(cfg.get("exit_trail_time_tighten", "14:30"))
        self.trail_atr_mult_late = float(cfg.get("exit_trail_atr_mult_late", 1.5))

        # Directional bias exit modifiers
        db_cfg = cfg.get("directional_bias", {})
        self._exit_modifiers = db_cfg.get("exit_modifiers", {}) if db_cfg.get("enabled", False) else {}


        # PRO TRADER: ORB-specific max hold time (Crabel: ideal trade profits instantly)
        # Pro traders hold ORB for 30-90 min max. Exit if no target hit within this time.
        self.orb_max_hold_minutes = float(cfg.get("orb_max_hold_minutes", 60))

        # execution params
        self.exec_product = str(cfg.get("exec_product", "MIS")).upper()
        self.exec_variety = str(cfg.get("exec_variety", "regular")).lower()
        self.exec_mode    = str(cfg.get("exec_order_mode", "MARKET")).upper()

        logger.info(
            f"exit_executor: init eod={self.eod_hhmm} "
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
        self.or_kill_enabled = bool(cfg["or_kill_enabled"])
        self.or_kill_time_adaptive = bool(cfg["or_kill_time_adaptive"])
        self.or_kill_volume_confirmation = bool(cfg["or_kill_volume_confirmation"])
        self.or_kill_momentum_filter = bool(cfg["or_kill_momentum_filter"])
        self.or_kill_partial_exit_pct = float(cfg["or_kill_partial_exit_pct"])
        self.or_kill_observation_minutes = float(cfg["or_kill_observation_minutes"])
        self.or_kill_volume_multiplier = float(cfg["or_kill_volume_multiplier"])
        self.or_kill_early_buffer_mult = float(cfg["or_kill_early_buffer_mult"])
        self.or_kill_mid_buffer_mult = float(cfg["or_kill_mid_buffer_mult"])
        self.or_kill_late_buffer_mult = float(cfg["or_kill_late_buffer_mult"])
        self.or_kill_major_break_mult = float(cfg["or_kill_major_break_mult"])

    def _enhanced_on_tick(self, symbol: str, price: float, volume: float, ts) -> None:
        """
        Enhanced tick handler that checks exits for open positions.

        Uses the tick price directly for instant exit checking in live/paper mode.
        This avoids race conditions with ltp_cache (which is updated after this callback)
        and eliminates REST API calls that would be rate-limited.
        """
        # Call original on_tick first
        if callable(self.original_on_tick):
            try:
                self.original_on_tick(symbol, price, volume, ts)
            except Exception as e:
                logger.exception(f"Original on_tick failed for {symbol}: {e}")

        # Check exits using the tick price directly (no API call needed)
        self._check_tick_exits(symbol, price, ts)

    def _check_tick_exits(self, symbol: str, tick_price: float, ts) -> None:
        """
        Check if tick hits SL/targets for open positions.

        In live/paper mode, uses the tick price directly for instant exit checking.
        This eliminates race conditions with ltp_cache and avoids rate-limited API calls.

        Args:
            symbol: Trading symbol
            tick_price: Current tick price from websocket
            ts: Tick timestamp
        """
        try:
            # Multi-position per symbol: tick price applies to ALL open positions
            # on this symbol. Iterate each (e.g., concurrent orb_15 long + later
            # vwap_first_pullback short on the same symbol under wide_open_mode).
            positions_for_symbol = self.positions.list_open_by_symbol(symbol)
            if not positions_for_symbol:
                return  # No open position for this symbol

            # Convert timestamp once
            ts_pd = pd.Timestamp(ts) if not isinstance(ts, pd.Timestamp) else ts

            for pos in positions_for_symbol:
                # RACE CONDITION FIX: Check if full exit already in progress
                st = pos.plan.get("_state") or {}
                if st.get("exit_pending", False):
                    logger.debug(f"EXIT_SKIP | {symbol} | Exit already pending, skipping tick check")
                    continue

                # Check SL using tick price directly
                plan_sl = self._get_plan_sl(pos.plan)
                if not math.isnan(plan_sl):
                    if self._breach_sl(pos.side, tick_price, plan_sl):
                        # Set exit_pending BEFORE placing order to prevent race
                        st["exit_pending"] = True
                        pos.plan["_state"] = st
                        # Classify SL reason — see classify_sl_exit_reason() docstring.
                        sl_reason = classify_sl_exit_reason(st, default_no_t1="tick_sl")
                        logger.info(f"TICK_SL_HIT: {symbol} {pos.side} price={tick_price:.2f} sl={plan_sl:.2f} reason={sl_reason}")
                        self._exit(symbol, pos, tick_price, ts_pd, sl_reason)
                        continue

                # Check targets using tick price directly
                t1, t2 = self._get_targets(pos.plan)
                # Re-read state (may have been updated by SL check above)
                st = pos.plan.get("_state") or {}
                t1_done = bool(st.get("t1_done", False))

                # T2 (full exit)
                if not math.isnan(t2):
                    if self._target_hit(pos.side, tick_price, t2):
                        # Set exit_pending BEFORE placing order to prevent race
                        st["exit_pending"] = True
                        pos.plan["_state"] = st
                        logger.info(f"TICK_T2_HIT: {symbol} {pos.side} price={tick_price:.2f} t2={t2:.2f}")
                        self._exit(symbol, pos, tick_price, ts_pd, "tick_target_t2")
                        continue

                # T1 (partial exit)
                if (not t1_done) and not math.isnan(t1):
                    if self._target_hit(pos.side, tick_price, t1):
                        logger.info(f"TICK_T1_HIT: {symbol} {pos.side} price={tick_price:.2f} t1={t1:.2f}")
                        self._partial_exit_t1(symbol, pos, tick_price, ts_pd)
                        continue

        except Exception as e:
            logger.exception(f"Tick exit check failed for {symbol}: {e}")

    def run_once(self) -> None:
        # Process any pending API exit requests first
        self._process_api_exits()

        # list_open() now returns {trade_id: Position}. We iterate by trade_id so
        # multiple concurrent positions on the same symbol (wide_open_mode) each
        # get their own SL/T1/T2/EOD-flat lifecycle. `sym` for tick lookup comes
        # from the position itself.
        open_pos = self.positions.list_open()
        if not open_pos:
            return

        for _tid, pos in open_pos.items():
            sym = pos.symbol
            try:
                px, ts = self._get_px_ts(sym)
                if px is None or ts is None:
                    continue

                # Track MAE/MFE (Maximum Adverse/Favorable Excursion) for exit diagnostics
                st = pos.plan.get('_state', {})

                # RACE CONDITION FIX: Skip entire bar processing if exit already pending
                if st.get("exit_pending", False):
                    logger.info(f"EXIT_SKIP | {sym} | Exit already pending, skipping bar check")
                    continue

                entry_price = pos.avg_price
                side = pos.side.upper()

                # Calculate current excursion
                if side == 'BUY':
                    current_excursion = px - entry_price
                else:
                    current_excursion = entry_price - px

                # Update MAE (worst drawdown) — always initialize to 0.0
                current_mae = st.get('mae', 0.0)
                if current_excursion < current_mae:
                    st['mae'] = current_excursion
                elif 'mae' not in st:
                    st['mae'] = 0.0

                # Update MFE (best profit) — always initialize to 0.0
                current_mfe = st.get('mfe', 0.0)
                if current_excursion > current_mfe:
                    st['mfe'] = current_excursion
                elif 'mfe' not in st:
                    st['mfe'] = 0.0

                # Track bars held (only increment on new bar timestamp)
                bar_ts_key = str(ts)
                if st.get('_last_bar_ts') != bar_ts_key:
                    st['bars_held'] = st.get('bars_held', 0) + 1
                    st['_last_bar_ts'] = bar_ts_key
                pos.plan['_state'] = st

                # 0) Time-stop / EOD square-off by tick timestamp.
                # Plan-as-source-of-truth (2026-05-12): respect per-setup
                # time_stop_hhmm if set, capped at the global EOD safety floor.
                # 2026-05-29 fix: in live/paper, reject stale tick timestamps
                # (e.g. Upstox WS snapshot/last-known-price ticks replayed at
                # subscribe time can carry yesterday's ts; those would compute
                # _minute_of_day = 946 and falsely trigger time_stop at 09:27).
                effective_md = self._effective_eod_md(pos)
                if effective_md is not None and not _is_stale_ts_for_live(ts) \
                        and _minute_of_day(ts) >= effective_md:
                    plan_ts = (pos.plan.get("exits") or {}).get("time_stop_hhmm")
                    reason = (f"time_stop_{plan_ts}"
                              if (plan_ts and effective_md != self.eod_md)
                              else f"eod_squareoff_{self.eod_hhmm}")
                    self._exit(sym, pos, float(px), ts, reason)
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
                    atr_min_mult = float(load_filters()["exit_sl_atr_mult_min"])
                    _atr_val = pos.plan.get("indicators", {}).get("atr")
                    atr_cached = float(_atr_val) if _atr_val is not None else float("nan")
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

                            # Per-tick: debug only (ATR and plan SL are constant — identical every tick)
                            logger.debug(
                                f"ATR_SL_ADJUSTMENT | {sym} | {pos.side} | "
                                f"Original_SL: {plan_sl:.2f} | ATR_Based_SL: {atr_based_sl:.2f} | "
                                f"Final_SL: {new_plan_sl:.2f} | ATR: {atr_cached:.3f} | "
                                f"Entry: {pos.avg_price:.2f} | Adjustment: {expansion_direction} by {expansion_amount:.2f}"
                            )

                            # Once per position: INFO log + events.jsonl emit (same pattern as SL_WIDENING)
                            _atr_st = pos.plan.get("_state") or {}
                            if not _atr_st.get("atr_sl_logged"):
                                logger.info(
                                    f"ATR_SL_ADJUSTMENT | {sym} | {pos.side} | "
                                    f"Original_SL: {plan_sl:.2f} | ATR_Based_SL: {atr_based_sl:.2f} | "
                                    f"Final_SL: {new_plan_sl:.2f} | ATR: {atr_cached:.3f} | "
                                    f"Entry: {pos.avg_price:.2f} | Adjustment: {expansion_direction} by {expansion_amount:.2f}"
                                )
                                try:
                                    diag_event_log._emit({
                                        "schema_version": _EVT_SCHEMA,
                                        "type": "ATR_SL_ADJUSTMENT",
                                        "run_id": diag_event_log.run_id,
                                        "trade_id": pos.plan.get("trade_id", ""),
                                        "symbol": sym,
                                        "ts": ts.isoformat() if hasattr(ts, 'isoformat') else str(ts),
                                        "atr_sl_adjustment": {
                                            "side": pos.side.upper(),
                                            "original_sl": round(plan_sl, 2),
                                            "atr_based_sl": round(atr_based_sl, 2),
                                            "final_sl": round(new_plan_sl, 2),
                                            "atr": round(atr_cached, 3),
                                            "atr_mult": atr_min_mult,
                                            "entry_price": round(float(pos.avg_price), 2),
                                            "direction": expansion_direction,
                                            "amount": round(expansion_amount, 2),
                                        },
                                    })
                                except Exception:
                                    pass  # Best-effort logging
                                _atr_st["atr_sl_logged"] = True
                                pos.plan["_state"] = _atr_st

                        plan_sl = new_plan_sl

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
                    # RACE CONDITION FIX: Check if exit already pending from tick-level check
                    if st.get("exit_pending", False):
                        logger.info(f"EXIT_SKIP | {sym} | Exit already pending, skipping bar SL check")
                        continue

                    sl_px = self.broker.get_ltp_with_level(sym, check_level=plan_sl)
                    if sl_px is not None and self._breach_sl(pos.side, sl_px, plan_sl):
                        sl_ltp = sl_px
                        # Enhanced SL exit logging with T1/T2 awareness
                        slippage = abs(sl_ltp - plan_sl)
                        # Classify SL reason — see classify_sl_exit_reason() docstring.
                        exit_reason = classify_sl_exit_reason(st, default_no_t1="hard_sl")

                        # Set exit_pending BEFORE placing order to prevent race
                        st["exit_pending"] = True
                        pos.plan["_state"] = st

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
                if self._or_kill(sym, pos.side, float(px), pos.plan, ts):
                    self._exit(sym, pos, float(px), ts, "or_kill")
                    continue
                reason = self._custom_kill(pos.side, float(px), pos.plan)
                if reason:
                    self._exit(sym, pos, float(px), ts, reason)
                    continue

                # 5) Breakout short risk control
                if self._breakout_short_risk_control_triggered(sym, pos, float(px), ts):
                    continue  # Handled internally

            except Exception as e:
                logger.exception(f"exit_executor: run_once error sym={sym}: {e}")

    def _process_api_exits(self) -> None:
        """
        Process pending exit requests from API server queue.

        API exits go through proper channels:
        - Logging via trading_logger
        - Persistence updates
        - Capital release
        """
        if not self.api_server:
            return

        try:
            requests = self.api_server.get_pending_exits()
            if not requests:
                return

            for req in requests:
                symbol = req.get("symbol")
                qty = req.get("qty")  # None means full exit
                reason = req.get("reason", "manual_exit")

                # Get position(s) for this symbol — under wide_open_mode there
                # may be multiple. API exits operate on the FIRST open position
                # for the symbol (ambiguous under multi-position; consider passing
                # trade_id from the API for explicit targeting).
                positions_for_symbol = self.positions.list_open_by_symbol(symbol)
                if not positions_for_symbol:
                    logger.warning(f"[API_EXIT] Position not found: {symbol}")
                    continue
                pos = positions_for_symbol[0]

                # Get current price
                px, ts = self._get_px_ts(symbol)
                if px is None:
                    # Try broker LTP as fallback
                    try:
                        px = self.broker.get_ltp(symbol)
                        ts = pd.Timestamp.now()
                    except Exception:
                        logger.error(f"[API_EXIT] Cannot get price for {symbol}")
                        continue

                # Execute exit through normal channels
                if qty is None or qty >= pos.qty:
                    # Full exit
                    logger.info(f"[API_EXIT] Full exit: {symbol} qty={pos.qty} reason={reason}")
                    self._exit(symbol, pos, float(px), ts, reason)
                else:
                    # Partial exit
                    logger.info(f"[API_EXIT] Partial exit: {symbol} qty={qty}/{pos.qty} reason={reason}")
                    self._partial_exit_api(symbol, pos, float(px), ts, qty, reason)

        except Exception as e:
            logger.exception(f"[API_EXIT] Error processing API exits: {e}")

    def _partial_exit_api(self, sym: str, pos, px: float, ts, qty: int, reason: str) -> None:
        """Handle partial exit from API request.

        Stores partial exit profit in position state so it's included in total P&L
        when the remaining position is closed.
        """
        current_qty = int(pos.qty)
        qty_exit = int(min(qty, current_qty))

        if qty_exit <= 0:
            return

        # Place exit order and get actual fill price
        actual_exit_px = self._place_and_log_exit(sym, pos, px, qty_exit, ts, reason)

        # Calculate profit for this partial exit
        entry_price = float(pos.avg_price)
        if pos.side.upper() == "BUY":
            partial_profit = qty_exit * (actual_exit_px - entry_price)
        else:
            partial_profit = qty_exit * (entry_price - actual_exit_px)

        # Store partial exit profit in position state (like T1 does)
        # This ensures it's included in total P&L when remaining position closes
        st = pos.plan.get("_state") or {}
        existing_partial_profit = st.get("manual_partial_profit", 0) or 0
        existing_partial_qty = st.get("manual_partial_qty", 0) or 0
        st["manual_partial_profit"] = existing_partial_profit + partial_profit
        st["manual_partial_qty"] = existing_partial_qty + qty_exit
        st["manual_partial_price"] = actual_exit_px  # Store broker fill for weighted avg exit
        st["manual_partial_time"] = ts.isoformat() if ts else None
        pos.plan["_state"] = st

        logger.info(f"[API_EXIT] Partial profit stored: {sym} qty={qty_exit} profit=Rs.{partial_profit:.2f} (cumulative: Rs.{st['manual_partial_profit']:.2f})")

        # Reduce position qty
        self.positions.reduce(_pos_tid(pos), qty_exit)

        # Release partial margin
        new_qty = current_qty - qty_exit
        if self.capital_manager:
            self.capital_manager.reduce_position(sym, qty_exit, new_qty)

        # Update persistence with new qty and state
        if self.persistence:
            self.persistence.update_position(sym, new_qty=new_qty, state_updates={"manual_partial_profit": st["manual_partial_profit"]})

        # Manual partial profit is tracked in _state (like T1 partial), NOT as a
        # separate dashboard row. The final _exit() call will include manual_partial_profit
        # and manual_partial_qty in the total trade summary — one trade = one dashboard row.

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

    def _effective_eod_md(self, pos) -> Optional[int]:
        """Plan-as-source-of-truth (2026-05-12): per-position effective EOD
        = min(plan.exits.time_stop_hhmm, global eod). Plan can pull EOD
        EARLIER but not LATER than the global MIS auto-square cutoff.

        Research-mode bypass (2026-05-13): under setups.<strategy>.wide_open=true
        the plan's time_stop_hhmm is IGNORED — capture mode lets the trade
        ride to natural SL/T2/EOD so cell-mining can discover the optimal
        time_stop from data (e.g. delivery_pct's 13:00 was brief-asserted,
        never swept). Global EOD (MIS auto-square) still binds as a hard cap.
        """
        plan_md = None
        try:
            strategy = (pos.plan.get("strategy") if pos and pos.plan else None)
            from services.config_loader import is_wide_open_for_setup
            if is_wide_open_for_setup(strategy):
                plan_md = None  # bypass plan time_stop in research-mode capture
            else:
                plan_ts = (pos.plan.get("exits") or {}).get("time_stop_hhmm") if pos and pos.plan else None
                if plan_ts:
                    plan_md = _parse_hhmm_to_md(str(plan_ts).replace(":", ""))
        except Exception:
            plan_md = None
        if plan_md is None:
            return self.eod_md
        if self.eod_md is None:
            return plan_md
        return min(plan_md, self.eod_md)

    def _is_eod(self, ts: pd.Timestamp, pos=None) -> bool:
        eff_md = self._effective_eod_md(pos) if pos is not None else self.eod_md
        if eff_md is None:
            return False
        try:
            return _minute_of_day(ts) >= int(eff_md)
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

                bias_mods = self._get_bias_exit_modifiers(pos)
                trail_bias_mult = bias_mods.get("trail_atr_mult", 1.0)
                pts = atr * mult * trail_bias_mult
                level = (px - pts) if side == "BUY" else (px + pts)
                why = f"trail_ATRx{mult}{why_suffix}"
                if trail_bias_mult != 1.0:
                    why += f"_bias{trail_bias_mult}"
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

    def _or_kill(self, symbol: str, side: str, px: float, plan: Dict[str, Any], ts=None) -> bool:
        """
        Enhanced OR kill logic with time-adaptive buffers, volume confirmation,
        momentum filters, and graduated exit strategy.
        """
        if not self.or_kill_enabled:
            return False
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
        buffer_mult = self._get_time_adaptive_buffer_mult(ts)

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
            return self._handle_or_observation(symbol, side, px, or_level, price_break, buffer, plan, ts)

        return False

    def _get_time_adaptive_buffer_mult(self, ts=None) -> float:
        """Calculate time-of-day adaptive buffer multiplier"""
        if not self.or_kill_time_adaptive:
            return 0.75  # Default fallback

        try:
            current_time = pd.Timestamp(ts) if ts is not None else _now_naive_ist()
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
            logger.error(f"VOLUME_CHECK: Volume confirmation failed for {symbol}: {e}")
            return True  # Allow protective exit when volume data unavailable

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

            # Get indicators from plan (these should be calculated elsewhere).
            # Guard against indicators[key]=None — `.get(k, default)` returns None
            # when the key is present with a None value, not the default.
            indicators = plan.get("indicators", {})
            _rsi = indicators.get("rsi")
            _adx = indicators.get("adx")
            rsi = float(_rsi) if _rsi is not None else 50.0
            adx = float(_adx) if _adx is not None else 20.0

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
                              price_break: float, buffer: float, plan: Dict[str, Any], ts=None) -> bool:
        """Handle graduated exit strategy for OR touches/minor breaks"""

        # Resolve position via plan trade_id (multi-position safe). Falls back to
        # the first open position on the symbol if trade_id is missing.
        plan_tid = (plan or {}).get("trade_id")
        pos = self.positions.get_by_trade_id(str(plan_tid)) if plan_tid else None
        if pos is None:
            sym_positions = self.positions.list_open_by_symbol(symbol)
            pos = sym_positions[0] if sym_positions else None
        if not pos:
            # CRITICAL FIX: Don't default to kill when no position found - this masks tracking bugs
            logger.error(f"OR_KILL: No position found for {symbol} - possible position tracking error")
            return False  # Conservative: don't kill if we can't find the position

        # Check if already in observation for this symbol
        obs_key = f"{symbol}_{side}"
        observation = self._or_kill_observation.get(obs_key, {})

        # Use tick timestamp for backtest compatibility (not wall clock)
        current_time = pd.Timestamp(ts) if ts is not None else pd.Timestamp(_now_naive_ist())

        if not observation:
            # First touch/minor break - start observation (store tick timestamp)
            observation = {
                "start_ts": current_time,
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
                        logger.info(f"OR_KILL_PARTIAL | {symbol} | {pos.side} {partial_qty} @ {px:.2f} | remaining: {pos.qty - partial_qty}")
                        self.positions.reduce(_pos_tid(pos), partial_qty)  # Update position quantity
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

        # Check if observation period expired (tick timestamp diff for backtest compatibility)
        elapsed_minutes = (current_time - observation["start_ts"]).total_seconds() / 60.0
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

    # ---------- Directional bias exit modifiers ----------

    def _get_bias_exit_modifiers(self, pos: Position) -> dict:
        """Get directional bias exit modifiers for current position.
        Queries LIVE tracker state — automatically handles bias flips during hold.
        Returns empty dict if no modifiers apply (neutral/chop/disabled).
        """
        if not self._exit_modifiers:
            return {}

        from services.gates.directional_bias import get_tracker
        tracker = get_tracker()
        if tracker is None:
            return {}

        alignment = tracker.classify_alignment(pos.side)
        if alignment == "neutral":
            return {}

        return self._exit_modifiers.get(f"{alignment}_trend", {})

    # ---------- Place exits + logging ----------

    def _place_and_log_exit(self, sym: str, pos: Position, exit_px: float, qty_exit: int, ts: Optional[pd.Timestamp], reason: str) -> float:
        try:
            cur = self.positions.get_by_trade_id(_pos_tid(pos))
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
            return exit_px  # No exit, return assumed price

        actual_exit_px = exit_px  # Default to assumed price
        try:
            # Check if this is a shadow trade (simulated, no real orders)
            is_shadow = pos.plan.get("shadow", False) if pos.plan else False

            if is_shadow:
                # SHADOW TRADE: Don't place real exit order, just simulate
                logger.info(f"SHADOW_EXIT_ORDER | {sym} | Simulated exit (no broker call) | qty={qty_exit} reason={reason}")
            else:
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
                order_id = self.broker.place_order(**args)

                # Reconcile exit with broker - verify order filled and get actual price
                if order_id and hasattr(self.broker, 'reconcile_exit'):
                    try:
                        reconciled = self.broker.reconcile_exit(
                            symbol=sym,
                            order_id=order_id,
                            expected_qty=qty_exit,
                            position_qty_before=cur_qty,
                            timeout=0.5
                        )

                        if reconciled is None:
                            # Exit order not filled - position unchanged at broker
                            logger.error(f"EXIT_RECONCILE | {sym} | Order {order_id} not filled - exit failed, position still open at broker")
                            raise RuntimeError(f"Exit order {order_id} not filled - position still open at broker")

                        # Use actual fill price from broker
                        broker_fill = reconciled.get("avg_price", exit_px)
                        if broker_fill and broker_fill > 0:
                            slippage = broker_fill - exit_px
                            slippage_bps = (slippage / exit_px) * 10000 if exit_px > 0 else 0
                            logger.info(f"EXIT_FILL | {sym} | Broker: {broker_fill:.2f} | Assumed: {exit_px:.2f} | Slippage: {slippage:+.2f} ({slippage_bps:+.1f} bps)")
                            actual_exit_px = broker_fill

                        logger.info(f"EXIT_RECONCILE | {sym} | Exit verified | Order: {order_id} | Remaining: {reconciled.get('remaining_qty', 'unknown')}")
                    except RuntimeError:
                        raise  # Re-raise exit failed
                    except Exception as e:
                        logger.warning(f"Failed to reconcile exit for {sym}: {e}")
        except Exception as e:
            logger.warning("exit.place_order failed sym=%s qty=%s reason=%s err=%s", sym, qty_exit, reason, e)

        entry_price = float(pos.avg_price)
        pnl = ((actual_exit_px - entry_price) if pos.side.upper() == "BUY" else (entry_price - actual_exit_px)) * int(qty_exit)

        # Compute exit diagnostics for events.jsonl
        plan_sl = pos.plan.get('stop', {}).get('hard') if isinstance(pos.plan.get('stop'), dict) else pos.plan.get('hard_sl')
        # R uses the INITIAL (decision-time) risk, not the trailed/BE stop.
        r_multiple = _r_multiple(entry_price, pnl, qty_exit,
                                 pos.plan.get('_decision_sl'), plan_sl)

        state = pos.plan.get('_state', {})
        mae = state.get('mae', 0.0)
        mfe = state.get('mfe', 0.0)
        bars_held = state.get('bars_held', 0)

        mae_pct = round(mae / entry_price * 100, 4) if mae is not None and entry_price > 0 else None
        mfe_pct = round(mfe / entry_price * 100, 4) if mfe is not None and entry_price > 0 else None

        entry_ts_val = pos.plan.get('entry_ts')
        time_since_entry_mins = None
        if entry_ts_val and ts:
            try:
                time_delta = ts - pd.Timestamp(entry_ts_val)
                time_since_entry_mins = time_delta.total_seconds() / 60
            except Exception:
                pass

        remaining_qty = max(0, pos.qty - qty_exit)

        # Decision-time SL (stored before target recalculation in trigger_aware_executor)
        decision_sl = pos.plan.get('_decision_sl')

        exit_diagnostics = {
            'exit_type': 'partial' if qty_exit < pos.qty else 'full',
            'remaining_qty': remaining_qty,
            'r_multiple': round(r_multiple, 2) if r_multiple is not None else None,
            'mae': round(mae, 2) if mae is not None else None,
            'mfe': round(mfe, 2) if mfe is not None else None,
            'mae_pct': mae_pct,
            'mfe_pct': mfe_pct,
            'bars_held': bars_held,
            'time_since_entry_mins': round(time_since_entry_mins, 1) if time_since_entry_mins is not None else None,
            'regime': pos.plan.get('regime'),
            'setup_type': pos.plan.get('setup_type'),
            'acceptance_status': pos.plan.get('quality', {}).get('status'),
            # Actual SL/targets for audit trail (post ATR adjustment + widening + trail)
            'actual_sl': round(plan_sl, 2) if plan_sl is not None else None,
            'original_sl': round(decision_sl, 2) if decision_sl is not None else None,
            'actual_targets': [round(t.get('level', 0), 2) for t in pos.plan.get('targets', []) if t.get('level') is not None],
            'decision_targets': pos.plan.get('_decision_targets'),
        }

        # Log EXIT to events.jsonl (single writer: diag_event_log)
        try:
            diag_event_log.log_exit(
                symbol=sym,
                plan=pos.plan,
                reason=reason,
                exit_price=actual_exit_px,
                exit_qty=qty_exit,
                ts=ts,
                pnl=round(pnl, 2),
                diagnostics=exit_diagnostics,
            )
        except Exception as _diag_err:
            logger.warning("diag_event_log.log_exit failed for %s: %s", sym, _diag_err)

        # Log EXIT to trade_logs.log (human-readable)
        if self.trading_logger:
            self.trading_logger.log_exit({
                'symbol': sym, 'qty': qty_exit, 'entry_price': entry_price,
                'exit_price': actual_exit_px, 'pnl': round(pnl, 2), 'reason': reason,
            })

        logger.debug(f"exit_executor: {sym} qty={qty_exit} reason={reason}")

        return actual_exit_px  # Return actual broker fill price for API logging

    def _exit(self, sym: str, pos: Position, exit_px: float, ts: Optional[pd.Timestamp], reason: str) -> None:
        # EARLY GUARD: Check if position has quantity to exit before any processing
        # This prevents double exits when multiple threads/paths call _exit() concurrently
        if pos.qty <= 0:
            logger.info(f"EXIT_SKIP | {sym} | No qty to exit (pos.qty={pos.qty})")
            return

        # re-read current qty from store just before the full exit
        try:
            cur = self.positions.get_by_trade_id(_pos_tid(pos))
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

        # Calculate R-multiple (PnL in units of INITIAL decision-time risk, not the
        # trailed/BE stop — see _r_multiple).
        plan_sl = pos.plan.get('stop', {}).get('hard') if isinstance(pos.plan.get('stop'), dict) else pos.plan.get('hard_sl')
        r_multiple = _r_multiple(entry_price, pnl, qty_now,
                                 pos.plan.get('_decision_sl'), plan_sl)

        # Get MAE/MFE from state if tracked
        # Default to 0.0 (not None) — trade may exit via tick before run_once tracks bars
        state = pos.plan.get('_state', {})
        mae = state.get('mae', 0.0)
        mfe = state.get('mfe', 0.0)

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

        # _place_and_log_exit() handles both diag_event_log + trading_logger logging

        # Capture actual broker fill price for API server logging
        actual_exit_px = self._place_and_log_exit(sym, pos, float(exit_px), qty_now, ts, reason)
        self.positions.close(_pos_tid(pos))

        # Remove from persistence (crash recovery)
        if self.persistence:
            self.persistence.remove_position(sym)

        # Release capital (free margin) on full exit
        # Shadow trades have no margin to release
        if self.capital_manager:
            is_shadow = pos.plan.get("shadow", False)
            self.capital_manager.exit_position(sym, shadow=is_shadow)

        # Log closed trade to API server for dashboard display
        if self.api_server:
            # Extract SL and targets from plan
            stop_data = pos.plan.get("stop", {})
            sl = stop_data.get("hard") if isinstance(stop_data, dict) else pos.plan.get("sl")
            targets = pos.plan.get("targets", [])
            t1 = targets[0].get("level") if targets and len(targets) > 0 else pos.plan.get("t1")
            t2 = targets[1].get("level") if targets and len(targets) > 1 else pos.plan.get("t2")

            # Recalculate PnL using actual broker fill price (not assumed price)
            actual_pnl = qty_now * (actual_exit_px - pos.avg_price) if pos.side.upper() == "BUY" else qty_now * (pos.avg_price - actual_exit_px)

            # Calculate TOTAL trade PnL including ALL partial exit profits (T1, T2, manual, EOD)
            state = pos.plan.get("_state", {})
            t1_profit = state.get("t1_profit", 0) or 0
            t1_booked_qty = state.get("t1_booked_qty", 0) or 0
            t2_profit = state.get("t2_profit", 0) or 0
            t2_booked_qty = state.get("t2_booked_qty", 0) or 0
            manual_partial_profit = state.get("manual_partial_profit", 0) or 0
            manual_partial_qty = state.get("manual_partial_qty", 0) or 0
            eod_partial_profit = state.get("eod_partial_profit", 0) or 0
            eod_partial_qty = state.get("eod_partial_qty", 0) or 0
            total_pnl = t1_profit + t2_profit + manual_partial_profit + eod_partial_profit + actual_pnl

            # Use original entry qty (all partials + remaining), not just final exit qty
            original_qty = t1_booked_qty + t2_booked_qty + manual_partial_qty + eod_partial_qty + qty_now

            # Get entry time - try multiple sources
            entry_time = (
                pos.plan.get("entry_ts") or
                pos.plan.get("trigger_ts") or
                state.get("entry_time")
            )

            # Weighted average exit price from all broker fills (T1, T2, manual, EOD, final)
            exit_legs = []
            t1_px = state.get("t1_booked_price")
            if t1_booked_qty > 0 and t1_px:
                exit_legs.append((t1_booked_qty, float(t1_px)))
            t2_px = state.get("t2_booked_price")
            if t2_booked_qty > 0 and t2_px:
                exit_legs.append((t2_booked_qty, float(t2_px)))
            manual_px = state.get("manual_partial_price")
            if manual_partial_qty > 0 and manual_px:
                exit_legs.append((manual_partial_qty, float(manual_px)))
            eod_px = state.get("eod_partial_price")
            if eod_partial_qty > 0 and eod_px:
                exit_legs.append((eod_partial_qty, float(eod_px)))
            exit_legs.append((qty_now, actual_exit_px))  # Final exit (always has broker fill)
            total_exit_qty = sum(q for q, _ in exit_legs)
            weighted_exit_px = sum(q * p for q, p in exit_legs) / total_exit_qty if total_exit_qty > 0 else actual_exit_px

            is_shadow = pos.plan.get("shadow", False)
            closed_trade = {
                "symbol": sym,
                "side": pos.side.upper(),
                "qty": original_qty,  # Total trade qty, not just final exit qty
                "entry_price": round(pos.avg_price, 2),
                "exit_price": round(weighted_exit_px, 2),  # Weighted avg across all broker fills
                "pnl": round(total_pnl, 2),  # Total trade PnL using actual broker price
                "exit_reason": reason,
                "setup": pos.plan.get("setup_type", "unknown"),
                "exit_time": ts.isoformat() if ts else None,
                "entry_time": entry_time,
                "sl": round(sl, 2) if sl else None,
                "t1": round(t1, 2) if t1 else None,
                "t2": round(t2, 2) if t2 else None,
                "t1_profit": round(t1_profit, 2) if t1_profit else None,  # Breakdown for debugging
                "t2_profit": round(t2_profit, 2) if t2_profit else None,
                "t1_exit_time": state.get("t1_exit_time"),  # When T1 was taken
                "manual_partial_profit": round(manual_partial_profit, 2) if manual_partial_profit else None,  # Manual partial exits
                "shadow": is_shadow,  # Shadow trade flag (for filtering in dashboard)
            }
            self.api_server.log_closed_trade(closed_trade)
            # Broadcast closed trade to WebSocket clients for real-time dashboard
            # Skip shadow trades - they're for internal analysis only
            if not is_shadow:
                self.api_server.broadcast_ws("closed_trade", closed_trade)

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
            cur = self.positions.get_by_trade_id(_pos_tid(pos))
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

    def _partial_exit_t1(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp]) -> None:
        # Enhanced partial exit logic - always use partial exits for better R:R
        current_qty = int(pos.qty)
        if current_qty <= 0:
            return

        # RACE CONDITION FIX: Use _t1_processing lock flag to prevent duplicate T1 exits
        # We defer setting t1_done until t1_profit is also ready, so HTTP polls and
        # WebSocket broadcasts never see t1_done=True with booked_pnl=0.
        st = pos.plan.get("_state") or {}
        if st.get("t1_done", False) or st.get("_t1_processing", False):
            logger.info(f"T1_SKIP | {sym} | T1 already done/processing, skipping duplicate")
            return
        st["_t1_processing"] = True
        pos.plan["_state"] = st

        # Check if T2 is infeasible (T1-only scalp mode) or Fast Scalp Lane
        t2_exit_mode = pos.plan.get("quality", {}).get("t2_exit_mode", None)
        if t2_exit_mode in ("T1_only_scalp", "fast_scalp_T1_only"):
            # Exit 100% at T1 if T2 is not feasible or in Fast Scalp Lane
            reason_suffix = "t2_infeasible" if t2_exit_mode == "T1_only_scalp" else "fast_scalp_lane"
            logger.info(f"exit_executor: {sym} {t2_exit_mode} mode - exiting 100% at T1 ({reason_suffix})")
            st.pop("_t1_processing", None)  # Clean up lock before full exit
            self._exit(sym, pos, float(px), ts, f"target_t1_full_{reason_suffix}")
            return

        # Get or store original entry quantity (for consistent 60-40 split)
        # This ensures T1 uses correct qty even if T2 fired first
        st = pos.plan.get("_state") or {}
        if "entry_qty" not in st:
            st["entry_qty"] = current_qty  # Store original entry qty on first access
            pos.plan["_state"] = st
        original_qty = st["entry_qty"]

        # Plan-as-source-of-truth (2026-05-12): use plan.targets[0].qty_pct
        # (set by detector from setup config) over global t1_book_pct.
        # Falls back to global only if plan doesn't carry a target qty_pct.
        bias_mods = self._get_bias_exit_modifiers(pos)
        t1_add = bias_mods.get("t1_book_pct_add", 0)
        plan_t1_qty_pct = None
        try:
            tgts = pos.plan.get("targets") or []
            if tgts and isinstance(tgts[0], dict) and "qty_pct" in tgts[0]:
                # Plan stores fraction (0.5 = 50%); convert to pct
                plan_t1_qty_pct = float(tgts[0]["qty_pct"]) * 100.0
        except Exception:
            plan_t1_qty_pct = None
        base_pct = plan_t1_qty_pct if plan_t1_qty_pct is not None else self.t1_book_pct
        actual_pct = max(0.0, base_pct + t1_add)

        # Plan-override: qty_pct=0 means "skip T1 partial, full qty rides to T2"
        if actual_pct <= 0.0:
            st["t1_done"] = True
            st["t1_booked_qty"] = 0
            st["t1_booked_price"] = px
            st["t1_profit"] = 0.0
            st["t1_skipped_plan_zero"] = True
            st.pop("_t1_processing", None)
            pos.plan["_state"] = st
            logger.info(f"T1_SKIP_PLAN_ZERO | {sym} | plan qty_pct=0; full qty rides to T2")
            return

        qty_exit = int(max(1, round(original_qty * (actual_pct / 100.0))))
        qty_exit = min(qty_exit, current_qty)  # Can't exit more than we have

        # Enhanced logic for small quantities
        if qty_exit >= current_qty:
            if current_qty > 2:  # Changed from 1 to 2 - allow partial even for small positions
                qty_exit = max(1, current_qty // 2)  # Take 50% minimum
            else:
                st.pop("_t1_processing", None)  # Clean up lock before full exit
                self._exit(sym, pos, float(px), ts, "target_t1_full")
                return

        # Calculate profit based on position direction
        if pos.side.upper() == "BUY":
            profit_booked = qty_exit * (px - pos.avg_price)
        else:
            profit_booked = qty_exit * (pos.avg_price - px)  # SHORT: profit when price goes down

        # Skip T1 partial if profit relative to risk is too low
        # When entry overshoots past T1, the partial books near-zero R but pays full charges.
        # "Free trade" condition (Minervini/Grimes): partial profit must cover remaining risk.
        # At <0.3R, charges eat 87%+ of gross profit on NSE — negative expectancy partial.
        # SL stays at original level: moving to BE on a tiny favorable move = noise stop-out.
        rps = pos.plan.get("risk_per_share") or pos.plan.get("stop", {}).get("risk_per_share", 0)
        if rps > 0:
            partial_r = profit_booked / (rps * qty_exit)
            if partial_r < self.t1_min_partial_r:
                # Mark T1 as done — stop re-checking every tick
                st["t1_done"] = True
                st["t1_booked_qty"] = 0
                st["t1_booked_price"] = px
                st["t1_profit"] = 0.0
                st["t1_skipped_low_r"] = True
                st.pop("_t1_processing", None)
                pos.plan["_state"] = st
                logger.info(
                    f"T1_SKIP_LOW_R | {sym} | T1 partial {partial_r:.2f}R "
                    f"< min {self.t1_min_partial_r}R (profit Rs.{profit_booked:.0f}, "
                    f"rps={rps:.2f}) — full qty rides to T2, SL unchanged"
                )
                return

        adjusted_t2 = self.t2_book_pct - bias_mods.get("t1_book_pct_add", 0)
        logger.info(f"exit_executor: {sym} T1_PARTIAL booking {qty_exit}/{original_qty} ({actual_pct:.1f}%) → profit Rs.{profit_booked:.2f} [CONFIG: {actual_pct:.0f}%-{adjusted_t2:.0f}%-0% split]")

        # Calculate new qty for capital manager
        new_qty = current_qty - int(qty_exit)

        # UPDATE STATE BEFORE reduce() - so WebSocket broadcast has correct t1_profit/t1_done
        # (positions.reduce triggers _broadcast_positions which reads from pos.plan["_state"])
        # Set t1_done and t1_profit ATOMICALLY so HTTP polls never see t1_done=True with booked=0
        st["t1_done"] = True
        st["t1_booked_qty"] = qty_exit
        st["t1_booked_price"] = px
        st["t1_profit"] = profit_booked
        st["t1_exit_time"] = ts.isoformat() if ts else None
        st.pop("_t1_processing", None)  # Clean up lock flag

        # Enhanced breakeven logic - always move to BE after T1
        if not st.get("sl_moved_to_be"):
            try:
                be = float(pos.avg_price)
                # Add small buffer above BE for long positions, below for short
                if pos.side.upper() == "BUY":
                    be_buffer = be + (be * self.be_buffer_pct)
                else:
                    be_buffer = be - (be * self.be_buffer_pct)

                be_buffer = round_to_tick(be_buffer)
                if isinstance(pos.plan.get("stop"), dict):
                    pos.plan["stop"]["hard"] = be_buffer
                pos.plan["hard_sl"] = be_buffer
                st["sl_moved_to_be"] = True
                st["be_price"] = be_buffer
                logger.debug(f"exit_executor: {sym} T1 hit — moved SL to BE+ @{be_buffer:.2f} (buffer: {be_buffer-be:.3f})")
            except Exception:
                pass
        pos.plan["_state"] = st

        # Place exit order AFTER state update
        actual_t1_fill = self._place_and_log_exit(sym, pos, float(px), int(qty_exit), ts, "t1_partial")

        # Update state with actual broker fill price (initial state used signal price for fast WebSocket)
        if actual_t1_fill != px:
            st["t1_booked_price"] = actual_t1_fill
            if pos.side.upper() == "BUY":
                st["t1_profit"] = qty_exit * (actual_t1_fill - pos.avg_price)
            else:
                st["t1_profit"] = qty_exit * (pos.avg_price - actual_t1_fill)
            pos.plan["_state"] = st

        # Reduce position qty (triggers WebSocket broadcast with updated state)
        self.positions.reduce(_pos_tid(pos), int(qty_exit))

        # Release partial margin for the exited quantity
        if self.capital_manager:
            self.capital_manager.reduce_position(sym, int(qty_exit), new_qty)

        # Update persistence with new qty and state (crash recovery)
        if self.persistence:
            self.persistence.update_position(sym, new_qty=new_qty, state_updates={
                "t1_done": True,
                "t1_profit": round(profit_booked, 2),
            })

        if ts is not None:
            try:
                eff_md = self._effective_eod_md(pos)
                # 2026-05-29 fix: same stale-ts guard as run_once time-stop
                # — prevent post-T1 stale-tick time-stop misfire.
                if eff_md is not None and not _is_stale_ts_for_live(ts) \
                        and _minute_of_day(ts) >= int(eff_md):
                    cur = self.positions.get_by_trade_id(_pos_tid(pos))
                    if cur and int(cur.qty) > 0:
                        plan_ts = (pos.plan.get("exits") or {}).get("time_stop_hhmm")
                        reason = (f"time_stop_{plan_ts}"
                                  if (plan_ts and eff_md != self.eod_md)
                                  else f"eod_squareoff_{self.eod_hhmm}")
                        self._exit(sym, cur, float(px), ts, reason)
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

        # RACE CONDITION FIX: Use _t2_processing lock flag to prevent duplicate T2 exits
        # We defer setting t2_done until t2_profit is also ready (same pattern as T1).
        st = pos.plan.get("_state") or {}
        if st.get("t2_done", False) or st.get("_t2_processing", False):
            logger.debug(f"T2_SKIP | {sym} | T2 already done/processing, skipping duplicate")
            return
        st["_t2_processing"] = True
        pos.plan["_state"] = st

        # Plan-as-source-of-truth (2026-05-12): use plan target qty_pct over
        # globals. Each plan target carries its own qty_pct from the detector.
        bias_mods = self._get_bias_exit_modifiers(pos)
        t1_add = bias_mods.get("t1_book_pct_add", 0)
        plan_t1_pct = None
        plan_t2_pct = None
        try:
            tgts = pos.plan.get("targets") or []
            if tgts and isinstance(tgts[0], dict) and "qty_pct" in tgts[0]:
                plan_t1_pct = float(tgts[0]["qty_pct"]) * 100.0
            if len(tgts) >= 2 and isinstance(tgts[1], dict) and "qty_pct" in tgts[1]:
                plan_t2_pct = float(tgts[1]["qty_pct"]) * 100.0
        except Exception:
            plan_t1_pct = plan_t2_pct = None
        base_t1 = plan_t1_pct if plan_t1_pct is not None else self.t1_book_pct
        base_t2 = plan_t2_pct if plan_t2_pct is not None else self.t2_book_pct
        adjusted_t1 = base_t1 + t1_add
        adjusted_t2 = base_t2 - t1_add  # T2 absorbs what T1 gives up/takes

        # If no trail configured (T1+T2 >= 100%), exit ALL remaining at T2
        # This avoids rounding issues and matches 60-40-0 intent
        no_trail = (adjusted_t1 + adjusted_t2) >= 100.0
        if no_trail:
            logger.info(f"exit_executor: {sym} T2_FULL exit (no trail configured: {adjusted_t1:.0f}%-{adjusted_t2:.0f}%-0%)")
            st.pop("_t2_processing", None)
            self._exit(sym, pos, float(px), ts, "target_t2_full")
            return

        # Trail is configured - do partial T2 exit
        t2_pct = adjusted_t2

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
            st.pop("_t2_processing", None)
            self._exit(sym, pos, float(px), ts, "target_t2_full")
            return

        # Log T2 partial exit with original qty context
        # FIX: Calculate profit based on position direction (same as T1)
        if pos.side.upper() == "BUY":
            profit_booked = qty_exit * (px - pos.avg_price)
        else:
            profit_booked = qty_exit * (pos.avg_price - px)  # SHORT: profit when price goes down

        t2_pct_of_original = (qty_exit / original_entry_qty * 100) if original_entry_qty > 0 else 0
        remaining_qty = qty - qty_exit
        trail_pct = 100.0 - adjusted_t1 - adjusted_t2
        logger.info(f"exit_executor: {sym} T2_PARTIAL booking {qty_exit}/{original_entry_qty} orig ({t2_pct_of_original:.1f}%) → profit Rs.{profit_booked:.2f} [CONFIG: {adjusted_t1:.0f}%-{adjusted_t2:.0f}%-{trail_pct:.0f}%, leaving {remaining_qty} for trail]")

        # UPDATE STATE BEFORE reduce() - so WebSocket broadcast has correct t2_profit/t2_done
        # (positions.reduce triggers _broadcast_positions which reads from pos.plan["_state"])
        # Set t2_done and t2_profit ATOMICALLY (same pattern as T1)
        st["t2_done"] = True
        st["t2_booked_qty"] = qty_exit
        st["t2_booked_price"] = px
        st["t2_profit"] = profit_booked
        st.pop("_t2_processing", None)  # Clean up lock flag
        pos.plan["_state"] = st

        # Place exit order AFTER state update
        actual_t2_fill = self._place_and_log_exit(sym, pos, float(px), int(qty_exit), ts, "t2_partial")

        # Update state with actual broker fill price (initial state used signal price for fast WebSocket)
        if actual_t2_fill != px:
            st["t2_booked_price"] = actual_t2_fill
            if pos.side.upper() == "BUY":
                st["t2_profit"] = qty_exit * (actual_t2_fill - pos.avg_price)
            else:
                st["t2_profit"] = qty_exit * (pos.avg_price - actual_t2_fill)
            pos.plan["_state"] = st

        # Reduce position qty (triggers WebSocket broadcast with updated state)
        self.positions.reduce(_pos_tid(pos), int(qty_exit))

        # Release partial margin for the exited quantity
        if self.capital_manager:
            self.capital_manager.reduce_position(sym, int(qty_exit), remaining_qty)

        # Update persistence with new qty and state (crash recovery)
        if self.persistence:
            self.persistence.update_position(sym, new_qty=remaining_qty, state_updates={
                "t1_done": True,
                "t2_done": True,
                "t2_profit": round(profit_booked, 2),
            })

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
            open_pos = self.positions.list_open()  # {trade_id: Position}
        except Exception:
            open_pos = {}

        for _tid, pos in list(open_pos.items()):
            sym = pos.symbol
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

    def _is_open_tid(self, trade_id: str) -> bool:
        """Check if a specific trade_id is still in the open positions store."""
        try:
            return self.positions.get_by_trade_id(trade_id) is not None
        except Exception:
            return False

    def _flatten_to_closed(self, sym: str, pos, px: float, ts, reason: str) -> None:
        intent_id = uuid.uuid4().hex[:8]
        # _closing_state is keyed by symbol — fine since multiple symbols don't
        # collide here; the retry loop targets a specific Position by trade_id.
        self._closing_state[sym] = {"state": "closing", "intent_id": intent_id, "reason": reason}
        tid = _pos_tid(pos)
        attempts = 0
        while self._is_open_tid(tid):
            self._exit(sym, pos, float(px), ts, reason if attempts == 0 else f"{reason}_retry{attempts}")
            attempts += 1
            if attempts >= 12:
                break
            try:
                pos = self.positions.get_by_trade_id(tid) or pos
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
                self.positions.reduce(_pos_tid(pos), qty_exit)

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
        # Prefer top-level (post-recalc) over sizing.rps (decision-time stale value)
        rps = pos.plan.get("risk_per_share") or pos.plan.get("stop", {}).get("risk_per_share") or pos.plan.get("sizing", {}).get("risk_per_share", 0.0)
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
