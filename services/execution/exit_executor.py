# services/execution/exit_executor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, Optional, Tuple
import math
import time
import pandas as pd

from config.logging_config import get_loggers
from config.filters_setup import load_filters
from utils.time_util import _now_naive_ist, _to_naive_ist

logger, _ = get_loggers()


# --- minimal position store interface -----------------------------------------
@dataclass
class Position:
    symbol: str
    side: str                 # "BUY" or "SELL"
    qty: int
    avg_price: float
    plan: Dict[str, Any] = field(default_factory=dict)  # includes hard_sl, targets, entry_zone, etc.


class PositionStore:
    """
    Very thin in-memory store. Replace with your real store if you already have one.
    """
    def __init__(self) -> None:
        self._pos: Dict[str, Position] = {}

    def list_open(self) -> Dict[str, Position]:
        return dict(self._pos)

    def upsert(self, p: Position) -> None:
        self._pos[p.symbol] = p

    def close(self, sym: str) -> None:
        self._pos.pop(sym, None)


# --- exit executor ------------------------------------------------------------
class ExitExecutor:
    """
    Pulls open positions from a PositionStore, applies exit policies, and places exit orders.
    Designed to run alongside your trade executor.

    Dependencies you must inject:
      - broker with place_order(symbol, side, qty, order_type="MARKET", product="MIS", variety="regular", price=None)
      - get_df5m: Callable[[str, int], pd.DataFrame]  -> latest 5m DF (closed bars)
      - get_ltp : Callable[[str], float]              -> current LTP (or None)
    """
    def __init__(
        self,
        broker,
        positions: PositionStore,
        get_df5m: Callable[[str, int], pd.DataFrame],
        get_ltp: Callable[[str], Optional[float]],
    ) -> None:
        self.broker = broker
        self.positions = positions
        self.get_df5m = get_df5m
        self.get_ltp = get_ltp
        self.cfg = load_filters()   # merged entry/exit config

        # Optional knobs (if missing in config, the check auto-disables)
        self.eod_hhmm = str(self.cfg.get("eod_squareoff_hhmm", "")) or str(self.cfg.get("exit_eod_squareoff_hhmm", ""))
        self.score_drop_enabled = bool(self.cfg.get("exit_score_drop_enabled", False))
        self.score_drop_bpct = float(self.cfg.get("exit_score_drop_bpct", 0.0))  # e.g., 25 -> drop by 25% from recent max
        self.trail_atr_mult = float(self.cfg.get("exit_trail_atr_mult", 0.0))   # 0 disables
        self.use_vwap_break = bool(self.cfg.get("exit_use_vwap_break", False))
        self.use_or_break   = bool(self.cfg.get("exit_use_or_break", False))

        # product/variety/mode for exit orders
        self.exec_product = str(self.cfg.get("exec_product", "MIS")).upper()
        self.exec_variety = str(self.cfg.get("exec_variety", "regular")).lower()
        self.exec_mode    = str(self.cfg.get("exec_order_mode", "MARKET")).upper()

        logger.info(f"exit_executor: init eod={self.eod_hhmm} score_drop={self.score_drop_enabled}:{self.score_drop_bpct}% "
                    f"trail_atr_mult={self.trail_atr_mult} vwap_break={self.use_vwap_break} or_break={self.use_or_break}")

        # per-symbol runtime state (peak for score-drop; last trail)
        self._peak_price: Dict[str, float] = {}
        self._trail_price: Dict[str, float] = {}

    # ------------------- public loop -------------------

    def run_once(self) -> None:
        open_pos = self.positions.list_open()
        if not open_pos:
            return

        now = _now_naive_ist()
        for sym, pos in open_pos.items():
            try:
                action, why = self._decide_exit(sym, pos, now)
                if action:
                    self._place_exit(sym, pos, why)
            except Exception as e:
                logger.exception(f"exit_executor: run_once error sym={sym}: {e}")

    def run_forever(self, sleep_ms: int = 400) -> None:
        try:
            while True:
                self.run_once()
                time.sleep(max(0.0, sleep_ms / 1000.0))
        except KeyboardInterrupt:
            logger.info("exit_executor: stop (KeyboardInterrupt)")

    # ------------------- decisions -------------------

    def _decide_exit(self, sym: str, pos: Position, now: pd.Timestamp) -> Tuple[bool, str]:
        """
        Returns (should_exit, reason)
        """
        # 0) EOD square-off
        if self._is_eod(now):
            return True, f"eod_squareoff_{self.eod_hhmm}"

        df = self.get_df5m(sym, 60)
        if df is None or df.empty:
            return False, "no_df5m"

        # Latest closed bar
        last = df.iloc[-1]
        ltp = self.get_ltp(sym)
        px = float(ltp if ltp is not None else last.get("close", float("nan")))

        # 1) Hard SL from plan (always on if provided)
        plan_sl = self._get_plan_sl(pos.plan, pos.side)
        if self._breach_sl(pos.side, px, plan_sl):
            return True, f"hard_sl({plan_sl})"

        # 2) Target hits (T1/T2). You can customize partial exits. Here: full exit on T2, partial on T1 (optional).
        t1, t2 = self._get_targets(pos.plan)
        hit, which = self._target_hit(pos.side, px, t1, t2)
        if hit:
            return True, f"target_{which}"

        # 3) Trailing stop using ATR (if enabled and ATR present)
        if self.trail_atr_mult > 0:
            trail, why = self._trail_price_for(pos.side, df)
            if not math.isnan(trail):
                # persist the tightest trail depending on side
                prev = self._trail_price.get(sym)
                if prev is None:
                    self._trail_price[sym] = trail
                else:
                    self._trail_price[sym] = max(prev, trail) if pos.side == "BUY" else min(prev, trail)
                if self._breach_sl(pos.side, px, self._trail_price[sym]):
                    return True, f"trail_stop({self._trail_price[sym]}:{why})"

        # 4) Score-drop by price (peak drawdown in % since entry or since peak)
        if self.score_drop_enabled and self.score_drop_bpct > 0:
            reason = self._score_drop_price(sym, pos.side, px)
            if reason:
                return True, reason

        # 5) VWAP / OR breaks (if enabled and metrics available)
        if self.use_vwap_break and self._vwap_break(pos.side, last):
            return True, "vwap_break"
        if self.use_or_break and self._or_break(pos.side, df):
            return True, "or_break"

        return False, "hold"

    # ------------------- rules -------------------

    def _is_eod(self, now: pd.Timestamp) -> bool:
        if not self.eod_hhmm:
            return False
        hhmm = _to_naive_ist(now).strftime("%H:%M").replace(":", "")
        # Allow both "1512" and "15:12" styles in config
        eod = self.eod_hhmm.replace(":", "")
        return hhmm >= eod

    def _get_plan_sl(self, plan: Dict[str, Any], side: str) -> float:
        sl = plan.get("hard_sl")
        if sl is None:
            sl = plan.get("stop")  # backwards compatibility
        try:
            return float(sl) if sl is not None else float("nan")
        except Exception:
            return float("nan")

    def _breach_sl(self, side: str, price: float, sl: float) -> bool:
        if math.isnan(sl) or math.isnan(price):
            return False
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

    def _target_hit(self, side: str, px: float, t1: float, t2: float) -> Tuple[bool, str]:
        if math.isnan(px):
            return False, ""
        if side == "BUY":
            if not math.isnan(t2) and px >= t2:
                return True, "t2"
            if not math.isnan(t1) and px >= t1:
                return True, "t1"
        else:
            if not math.isnan(t2) and px <= t2:
                return True, "t2"
            if not math.isnan(t1) and px <= t1:
                return True, "t1"
        return False, ""

    def _trail_price_for(self, side: str, df5: pd.DataFrame) -> Tuple[float, str]:
        """
        ATR-based trailing from latest bar:
          BUY : trail = close - ATR * mult
          SELL: trail = close + ATR * mult
        Expects an 'ATR' column (or computes from high/low/close if you have a helper).
        """
        mult = self.trail_atr_mult
        if mult <= 0:
            return float("nan"), "trail_disabled"
        last = df5.iloc[-1]
        atr = last.get("ATR")
        try:
            atr = float(atr) if atr is not None else float("nan")
        except Exception:
            atr = float("nan")
        if math.isnan(atr):
            return float("nan"), "no_atr"

        close = float(last["close"])
        if side == "BUY":
            return close - atr * mult, f"close({close})-atr*{mult}({atr})"
        else:
            return close + atr * mult, f"close({close})+atr*{mult}({atr})"

    def _score_drop_price(self, sym: str, side: str, px: float) -> Optional[str]:
        """
        Simple price-based 'score drop': if price falls by X% from the peak since we started tracking,
        we exit. For SELL, peak is the *lowest* price (we track it inverted).
        """
        if self.score_drop_bpct <= 0 or math.isnan(px):
            return None

        cur = self._peak_price.get(sym)
        if cur is None:
            # initialize
            self._peak_price[sym] = px
            return None

        if side == "BUY":
            if px > cur:
                self._peak_price[sym] = px
                return None
            dd = (cur - px) / cur * 100.0
            if dd >= self.score_drop_bpct:
                return f"score_drop_dd{dd:.1f}%>= {self.score_drop_bpct:.1f}%"
        else:
            # For SELL, invert peak tracking (track trough)
            if px < cur:
                self._peak_price[sym] = px
                return None
            dd = (px - cur) / cur * 100.0  # price rising vs trough
            if dd >= self.score_drop_bpct:
                return f"score_drop_rally{dd:.1f}%>= {self.score_drop_bpct:.1f}%"
        return None

    def _vwap_break(self, side: str, last_row: pd.Series) -> bool:
        vwap = last_row.get("VWAP", last_row.get("vwap"))
        try:
            vwap = float(vwap) if vwap is not None else float("nan")
        except Exception:
            vwap = float("nan")
        if math.isnan(vwap):
            return False
        close = float(last_row["close"])
        return (close < vwap) if side == "BUY" else (close > vwap)

    def _or_break(self, side: str, df5: pd.DataFrame) -> bool:
        """
        Crude OR break: compute today's opening range from first N bars (default 3),
        and exit if close crosses back through the opposite side.
        """
        n = int(self.cfg.get("opening_range_bars", 3))
        today = df5.iloc[:n]
        if len(today) < n:
            return False
        orh = float(today["high"].max())
        orl = float(today["low"].min())
        close = float(df5.iloc[-1]["close"])
        return (close < orl) if side == "BUY" else (close > orh)

    # ------------------- side-effects -------------------

    def _place_exit(self, sym: str, pos: Position, reason: str) -> None:
        side = "SELL" if pos.side == "BUY" else "BUY"
        qty = int(pos.qty)
        try:
            args = {
                "symbol": sym,
                "side": side,
                "qty": qty,
                "order_type": self.exec_mode,
                "product": self.exec_product,
                "variety": self.exec_variety,
            }
            order_id = self.broker.place_order(**args)
            logger.info(f"exit_executor: EXIT placed sym={sym} side={side} qty={qty} reason={reason} id={order_id}")
            # Update position store
            self.positions.close(sym)
        except Exception as e:
            logger.exception(f"exit_executor: place exit failed sym={sym} reason={reason}: {e}")
