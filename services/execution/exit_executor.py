# services/execution/exit_executor.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, Tuple, List
import math
import time
import pandas as pd  # for Timestamp typing only

from config.logging_config import get_execution_loggers
from config.filters_setup import load_filters
from utils.time_util import _to_naive_ist, _now_naive_ist
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

def _parse_eod_to_md(eod: str | int) -> Optional[int]:
    """
    Accept "15:12", "1512" or minute-of-day int. Return minute-of-day [0..1439].
    """
    if eod is None:
        return None
    try:
        if isinstance(eod, int):
            v = eod
            if v >= 1000:  # assume HHMM
                hh, mm = divmod(v, 100)
                return hh * 60 + mm
            return v
        s = str(eod).strip()
        if ":" in s:
            hh, mm = s.split(":")
            return int(hh) * 60 + int(mm)
        # "1512" form
        if len(s) >= 3:
            hh = int(s[:-2]); mm = int(s[-2:])
            return hh * 60 + mm
        return int(s)
    except Exception:
        return None

def _minute_of_day(ts: pd.Timestamp) -> int:
    t = _to_naive_ist(ts)
    return t.hour * 60 + t.minute


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
        trading_logger=None,  # Enhanced logging service
    ) -> None:
        self.broker = broker
        self.positions = positions
        self.get_ltp_ts = get_ltp_ts
        self.trading_logger = trading_logger

        cfg = load_filters()
        # Config knobs
        self.eod_hhmm = str(cfg.get("eod_squareoff_hhmm", "")) or str(cfg.get("exit_eod_squareoff_hhmm", "1512"))
        self.eod_md = _parse_eod_to_md(self.eod_hhmm)

        self.score_drop_enabled = bool(cfg.get("exit_score_drop_enabled", False))
        self.score_drop_bpct = float(cfg.get("exit_score_drop_bpct", 0.0))  # % drop from peak before exit
        self.time_stop_min = float(cfg.get("exit_time_stop_min", 0.0))      # minutes after entry to check RR
        self.time_stop_req_rr = float(cfg.get("exit_time_stop_req_rr", 0.0))

        # T1 behavior
        self.t1_book_pct = float(cfg.get("exit_t1_book_pct", 50.0))
        self.t1_move_sl_to_be = bool(cfg.get("exit_t1_move_sl_to_be", True))

        # execution params
        self.exec_product = str(cfg.get("exec_product", "MIS")).upper()
        self.exec_variety = str(cfg.get("exec_variety", "regular")).lower()
        self.exec_mode    = str(cfg.get("exec_order_mode", "MARKET")).upper()

        logger.info(
            f"exit_executor: init eod={self.eod_hhmm} "
            f"score_drop={self.score_drop_enabled}:{self.score_drop_bpct}% "
            f"time_stop={self.time_stop_min}m@RR<{self.time_stop_req_rr} "
            f"t1_pct={self.t1_book_pct}% sl2be={self.t1_move_sl_to_be}"
        )

        self._peak_price: Dict[str, float] = {}
        self._trail_price: Dict[str, float] = {}
        self._closing_state = {} 

    def run_once(self) -> None:
        open_pos = self.positions.list_open()
        if not open_pos:
            return

        for sym, pos in open_pos.items():
            try:
                px, ts = self._get_px_ts(sym)
                if px is None or ts is None:
                    continue

                # 0) EOD square-off by tick timestamp
                if self.eod_md is not None and _minute_of_day(ts) >= self.eod_md:
                    self._exit(sym, pos, float(px), ts, f"eod_squareoff_{self.eod_hhmm}")
                    continue

                # 1) Hard SL first, but anchor to ATR sanity if available
                plan_sl = self._get_plan_sl(pos.plan)
                if not math.isnan(plan_sl):
                    atr_min_mult = float(load_filters().get("exit_sl_atr_mult_min", 1.0))
                    atr_cached = float(pos.plan.get("indicators", {}).get("atr", float("nan")))
                    if (not math.isnan(atr_cached)) and atr_min_mult > 0:
                        # expand stop slightly if it's inside 1.0x ATR noise
                        if pos.side.upper() == "BUY":
                            plan_sl = min(plan_sl, float(pos.avg_price) - atr_min_mult * atr_cached)
                        else:
                            plan_sl = max(plan_sl, float(pos.avg_price) + atr_min_mult * atr_cached)

                if self._breach_sl(pos.side, float(px), plan_sl):
                    self._exit(sym, pos, float(px), ts, "hard_sl")
                    continue

                # 2) Targets & state
                t1, t2 = self._get_targets(pos.plan)
                st = pos.plan.get("_state") or {}
                t1_done = bool(st.get("t1_done", False))

                # 2a) T2 (full exit first)
                if self._target_hit(pos.side, float(px), t2):
                    self._exit(sym, pos, float(px), ts, "target_t2")
                    continue

                # 2b) T1 (one-time partial)
                if (not t1_done) and self._target_hit(pos.side, float(px), t1):
                    self._partial_exit_t1(sym, pos, float(px), ts)
                    continue

                # 3) Dynamic trail (tighten-only)
                if self._has_trail(pos.plan) and self._trail_allowed(pos, ts):
                    level, why = self._trail_from_plan(sym, pos, float(px))
                    if self._breach_sl(pos.side, float(px), level):
                        self._exit(sym, pos, float(px), ts, f"trail_stop({why})")
                        continue

                # 4) Indicator kill-switches (precomputed levels)
                if self._or_kill(pos.side, float(px), pos.plan):
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

                # 6) Time-stop (tick timestamp)
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
        sl = plan.get("hard_sl", plan.get("stop"))
        try:
            if isinstance(sl, dict) and "hard" in sl:
                return float(sl["hard"])
            return float(sl) if sl is not None else float("nan")
        except Exception:
            return float("nan")

    def _breach_sl(self, side: str, price: float, sl: float) -> bool:
        if math.isnan(sl) or math.isnan(price):
            return False
        side = side.upper()
        return (price <= sl) if side == "BUY" else (price >= sl)

    def _get_targets(self, plan: Dict[str, Any]) -> Tuple[float, float]:
        t1 = t2 = float("nan")
        try:
            ts = plan.get("targets") or []
            if len(ts) > 0 and ts[0] and "level" in ts[0]:
                t1 = float(ts[0]["level"])
            if len(ts) > 1 and ts[1] and "level" in ts[1]:
                t2 = float(ts[1]["level"])
        except Exception:
            pass
        return t1, t2

    def _target_hit(self, side: str, px: float, tgt: float) -> bool:
        if math.isnan(tgt) or math.isnan(px):
            return False
        side = side.upper()
        return (px >= tgt) if side == "BUY" else (px <= tgt)

    # ---------- Trail ----------

    def _has_trail(self, plan: Dict[str, Any]) -> bool:
        tr = plan.get("trail")
        return isinstance(tr, dict) and len(tr) > 0

    def _trail_from_plan(self, sym: str, pos: Position, px: float) -> Tuple[float, str]:
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
                mult = float(tr["atr_mult"])
                pts = atr * mult
                level = (px - pts) if side == "BUY" else (px + pts)
                why = f"trail_ATRx{mult}"
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

    def _or_kill(self, side: str, px: float, plan: Dict[str, Any]) -> bool:
        """
        Professional OR kill logic with ATR-based buffers.
        Only kills on significant range breaks, not minor slippage.
        """
        try:
            orh = plan.get("orh"); orl = plan.get("orl")
            orh = None if orh is None else float(orh)
            orl = None if orl is None else float(orl)

            # Get ATR for buffer calculation
            atr = float(plan.get("indicators", {}).get("atr", 0))
            if atr <= 0:
                # Fallback: estimate ATR as 1% of price
                atr = abs(px * 0.01)

        except Exception:
            return False

        if orh is None and orl is None:
            return False

        side = side.upper()

        # Professional approach: Add ATR-based buffer to avoid noise kills
        # Regime-specific buffer multipliers
        regime = plan.get("regime", "").lower()
        if regime in ["chop", "choppy"]:
            buffer_mult = 0.75  # Larger buffer in choppy markets (3/4 ATR)
        else:
            buffer_mult = 0.5   # Smaller buffer in trending markets (1/2 ATR)

        buffer = atr * buffer_mult

        # Apply buffers - only kill on convincing breaks beyond normal noise
        if side == "BUY" and orl is not None:
            # For longs: only kill if price breaks significantly below ORL
            kill_level = orl - buffer
            if px < kill_level:
                logger.debug(f"OR_KILL: {plan.get('symbol')} LONG breach {px:.2f} < {kill_level:.2f} (ORL-{buffer:.2f})")
                return True

        if side == "SELL" and orh is not None:
            # For shorts: only kill if price breaks significantly above ORH
            kill_level = orh + buffer
            if px > kill_level:
                logger.debug(f"OR_KILL: {plan.get('symbol')} SHORT breach {px:.2f} > {kill_level:.2f} (ORH+{buffer:.2f})")
                return True

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
            cur_qty = int(getattr(cur, "qty", 0) or 0) if cur is not None else int(pos.qty)
        except Exception:
            cur_qty = int(pos.qty)
        qty_exit = int(max(0, min(int(qty_exit), cur_qty)))
        if qty_exit <= 0:
            return

        try:
            exit_side = "SELL" if pos.side.upper() == "BUY" else "BUY"
            args = {
                "symbol": sym,
                "side": exit_side,
                "qty": int(qty_exit),
                "order_type": self.exec_mode,
                "product": self.exec_product,
                "variety": self.exec_variety,
            }
            self.broker.place_order(**args)
        except Exception as e:
            logger.warning("exit.place_order failed sym=%s qty=%s reason=%s err=%s", sym, qty_exit, reason, e)

        entry_price = float(pos.avg_price)
        pnl = ((exit_px - entry_price) if pos.side.upper() == "BUY" else (entry_price - exit_px)) * int(qty_exit)
        trade_logger.info(
            f"EXIT | {sym} | Qty: {qty_exit} | Entry: Rs.{entry_price:.2f} | Exit: Rs.{exit_px:.2f} | PnL: Rs.{pnl:.2f} {reason}"
        )
        try:
            diag_event_log.log_exit(
                symbol=sym,
                plan=pos.plan,          # same plan with trade_id & entry stamps
                reason=str(reason),     # e.g., hard_sl, t1_partial, target_t2, trail_stop(...), or_kill, time_stop_.., eod_squareoff_HHMM
                exit_price=float(exit_px),
                exit_qty=int(qty_exit),
                ts=ts,                  # tick-ts (time-naive)
            )
        except Exception as _e:
            logger.warning("diag_event_log.log_exit failed sym=%s err=%s", sym, _e)
        logger.debug(f"exit_executor: {sym} qty={qty_exit} reason={reason}")

    def _exit(self, sym: str, pos: Position, exit_px: float, ts: Optional[pd.Timestamp], reason: str) -> None:
        # re-read current qty from store just before the full exit
        try:
            cur = self.positions.list_open().get(sym)
            qty_now = int(getattr(cur, "qty", 0) or 0) if cur is not None else int(pos.qty)
        except Exception:
            qty_now = int(pos.qty)

        if qty_now <= 0:
            return

        # Calculate PnL for enhanced logging
        pnl = qty_now * (exit_px - pos.avg_price) if pos.side.upper() == "BUY" else qty_now * (pos.avg_price - exit_px)

        # Enhanced logging: Log exit execution
        if self.trading_logger:
            exit_data = {
                'symbol': sym,
                'trade_id': pos.plan.get('trade_id', ''),
                'qty': qty_now,
                'entry_price': pos.avg_price,
                'exit_price': exit_px,
                'pnl': round(pnl, 2),
                'reason': reason,
                'timestamp': str(ts) if ts else str(pd.Timestamp.now())
            }
            self.trading_logger.log_exit(exit_data)

        self._place_and_log_exit(sym, pos, float(exit_px), qty_now, ts, reason)
        self.positions.close(sym)
        try:
            st = pos.plan.get("_state") or {}
            st.pop("t1_done", None); st.pop("sl_moved_to_be", None)
            pos.plan["_state"] = st
        except Exception:
            pass
        self._peak_price.pop(sym, None); self._trail_price.pop(sym, None)

        rem = 0
        try:
            cur = self.positions.list_open().get(sym)
            rem = int(getattr(cur, "qty", 0) or 0)
        except Exception:
            pass
        trade_logger.info(f"EXIT | {sym} | qty {qty_now} @ Rs.{float(exit_px):.2f} → remaining {rem}")

    def _partial_exit_t1(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp]) -> None:
        # Enhanced partial exit logic - always use partial exits for better R:R
        pct = max(1.0, float(getattr(self, "t1_book_pct", 50.0)))
        qty = int(pos.qty)

        # Calculate partial exit quantity - ensure minimum 30% book
        min_book_pct = 30.0  # Minimum partial booking percentage
        actual_pct = max(min_book_pct, pct)
        
        qty_exit = int(max(1, round(qty * (actual_pct / 100.0))))
        qty_exit = min(qty_exit, qty)

        # Enhanced logic for small quantities
        if qty_exit >= qty:
            if qty > 2:  # Changed from 1 to 2 - allow partial even for small positions
                qty_exit = max(1, qty // 2)  # Take 50% minimum
            else:
                self._exit(sym, pos, float(px), ts, "target_t1_full")
                return

        # Log enhanced partial exit info  
        profit_booked = qty_exit * (px - pos.avg_price)
        logger.info(f"exit_executor: {sym} T1_PARTIAL booking {qty_exit}/{qty} ({actual_pct:.1f}%) → profit Rs.{profit_booked:.2f}")
        
        self._place_and_log_exit(sym, pos, float(px), int(qty_exit), ts, "t1_partial")
        self.positions.reduce(sym, int(qty_exit))

        st = pos.plan.get("_state") or {}
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

        if self.eod_md is not None and ts is not None:
            try:
                if _minute_of_day(ts) >= int(self.eod_md):
                    cur = self.positions.list_open().get(sym)
                    if cur and int(cur.qty) > 0:
                        self._exit(sym, cur, float(px), ts, f"eod_squareoff_{self.eod_hhmm}")
                        return
            except Exception:
                pass

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
                pos = self.positions.get(sym)
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
                pos = self.positions.get(sym) or pos
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


# (end)
