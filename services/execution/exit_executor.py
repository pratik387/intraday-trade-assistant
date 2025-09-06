# services/execution/exit_executor.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, Tuple, List
import math
import time
import pandas as pd  # for Timestamp typing only

from config.logging_config import get_loggers
from config.filters_setup import load_filters
from utils.time_util import _to_naive_ist, _now_naive_ist

# Both loggers (as requested)
logger, trade_logger = get_loggers()


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
    ) -> None:
        self.broker = broker
        self.positions = positions
        self.get_ltp_ts = get_ltp_ts

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

        # per-symbol runtime state
        self._peak_price: Dict[str, float] = {}
        self._trail_price: Dict[str, float] = {}

    # ------------------- loop -------------------

    def run_once(self) -> None:
        open_pos = self.positions.list_open()
        if not open_pos:
            return

        for sym, pos in open_pos.items():
            try:
                px, ts = self._get_px_ts(sym)
                if px is None or ts is None:
                    # Need both price and tick timestamp (for EOD, time-stop)
                    continue

                # 0) EOD square-off by tick timestamp
                if self.eod_md is not None and _minute_of_day(ts) >= self.eod_md:
                    self._exit(sym, pos, float(px), ts, f"eod_squareoff_{self.eod_hhmm}")
                    continue

                # 1) Hard SL first
                plan_sl = self._get_plan_sl(pos.plan)
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
                    # After partial, do not evaluate further rules this tick
                    continue

                # 3) Dynamic trail (tighten-only)
                if self._has_trail(pos.plan):
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
            # support nested {"stop":{"hard":...}}
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
        try:
            orh = plan.get("orh"); orl = plan.get("orl")
            orh = None if orh is None else float(orh)
            orl = None if orl is None else float(orl)
        except Exception:
            return False
        side = side.upper()
        if side == "BUY" and (orl is not None):
            return px < orl
        if side == "SELL" and (orh is not None):
            return px > orh
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
            f"EXIT | {sym} | Qty: {qty_exit} | Entry: ₹{entry_price:.2f} | Exit: ₹{exit_px:.2f} | PnL: ₹{pnl:.2f} {reason}"
        )
        logger.info(f"exit_executor: {sym} qty={qty_exit} reason={reason}")

    def _exit(self, sym: str, pos: Position, exit_px: float, ts: Optional[pd.Timestamp], reason: str) -> None:
        # Full close
        self._place_and_log_exit(sym, pos, exit_px, int(pos.qty), ts, reason)
        # Update store & cleanup
        self.positions.close(sym)
        try:
            st = pos.plan.get("_state") or {}
            st.pop("t1_done", None)
            st.pop("sl_moved_to_be", None)
            pos.plan["_state"] = st
        except Exception:
            pass
        self._peak_price.pop(sym, None)
        self._trail_price.pop(sym, None)

    def _partial_exit_t1(self, sym: str, pos: Position, px: float, ts: Optional[pd.Timestamp]) -> None:
        # book configurable percent at T1
        pct = max(1.0, float(getattr(self, "t1_book_pct", 50.0)))
        qty_exit = int(max(1, round(int(pos.qty) * (pct / 100.0))))
        qty_exit = min(qty_exit, int(pos.qty))

        # place partial
        self._place_and_log_exit(sym, pos, px, qty_exit, ts, "t1_partial")

        # reduce in store
        self.positions.reduce(sym, qty_exit)

        # mark state + move SL->BE once
        st = pos.plan.get("_state") or {}
        st["t1_done"] = True
        if getattr(self, "t1_move_sl_to_be", True) and not st.get("sl_moved_to_be"):
            try:
                be = float(pos.avg_price)
                if isinstance(pos.plan.get("stop"), dict):
                    pos.plan["stop"]["hard"] = be
                pos.plan["hard_sl"] = be
                st["sl_moved_to_be"] = True
                logger.info(f"exit_executor: {sym} T1 hit — moved SL->BE @{be}")
            except Exception:
                pass
        pos.plan["_state"] = st

# (end)
