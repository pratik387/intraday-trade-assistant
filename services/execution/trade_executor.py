# services/execution/trade_executor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import time
import math
import pandas as pd

from config.logging_config import get_loggers
from config.filters_setup import load_filters
from services.orders.order_queue import OrderQueue
from utils.time_util import _now_naive_ist, _minute_of_day, _parse_hhmm_to_md

logger, trade_logger = get_loggers()


@dataclass
class Position:
    symbol: str
    side: str                 # "BUY" or "SELL"
    qty: int
    avg_price: float
    plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskState:
    max_concurrent: int
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gross_exposure_rupees: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    correlation_buckets: Dict[str, List[str]] = field(default_factory=dict)

    def can_open_more(self) -> bool:
        return len(self.open_positions) < int(self.max_concurrent)


class TradeExecutor:
    """
    Places entry orders and updates basic risk state.

    Implements:
      • Entry cutoff (default "14:45") and EOD guard (e.g., "15:10") using tick timestamps if available.
      • Entry-time sanitization of SL/T1/T2 against the actual fill.
      • ENTRY logging via trade_logger.
      • Plan stamping of entry_ts and entry_epoch_ms.

    Optional external hook (can also be passed into __init__):
      get_ltp_ts(symbol) -> (ltp: float|None, ts: pd.Timestamp|None)
    """

    def __init__(
        self,
        broker,
        order_queue: OrderQueue,
        risk_state: RiskState,
        positions: Optional[Any] = None,   # must expose .upsert(Position) if provided
        get_ltp_ts=None,                   # <— NEW: accept optional hook for tick LTP + timestamp
    ) -> None:
        self.broker = broker
        self.oq = order_queue
        self.risk = risk_state
        self.positions = positions
        self.get_ltp_ts = get_ltp_ts      # <— store the hook if provided

        self.cfg = load_filters()
        # STRICT config (KeyError if missing – by design)
        self.order_mode = str(self.cfg["exec_order_mode"]).upper()   # "MARKET" or "LIMIT"
        self.order_product = str(self.cfg["exec_product"]).upper()   # e.g., "MIS"
        self.order_variety = str(self.cfg["exec_variety"]).lower()   # e.g., "regular"

        # Entry cutoff + EOD (IST minute-of-day)
        self.entry_cutoff_hhmm = str(self.cfg.get("entry_cutoff_hhmm", "14:45"))
        self._entry_cutoff_md = _parse_hhmm_to_md(self.entry_cutoff_hhmm)

        self.eod_hhmm = str(self.cfg.get("eod_squareoff_hhmm", "15:10"))
        self._eod_md = _parse_hhmm_to_md(self.eod_hhmm)

    # -------- helpers: extract/patch exits on plan -------- #
    @staticmethod
    def _extract_plan_exits(plan: Dict[str, Any]) -> Tuple[float, float, float]:
        """Return (sl, t1, t2) as floats or NaN if unavailable."""
        sl = float("nan")
        for k in ("hard_sl", "hard_stop"):
            if k in plan and plan[k] is not None:
                try:
                    sl = float(plan[k]); break
                except Exception:
                    pass
        if math.isnan(sl):
            stop = plan.get("stop")
            if isinstance(stop, dict) and stop.get("hard") is not None:
                try: sl = float(stop["hard"])
                except Exception: sl = float("nan")
            elif isinstance(stop, (int, float)):
                sl = float(stop)

        t1 = t2 = float("nan")
        tgts = plan.get("targets") or []
        if len(tgts) > 0 and tgts[0] and "level" in tgts[0] and tgts[0]["level"] is not None:
            try: t1 = float(tgts[0]["level"])
            except Exception: t1 = float("nan")
        if len(tgts) > 1 and tgts[1] and "level" in tgts[1] and tgts[1]["level"] is not None:
            try: t2 = float(tgts[1]["level"])
            except Exception: t2 = float("nan")
        return sl, t1, t2

    @staticmethod
    def _inject_plan_exits(plan: Dict[str, Any], sl: float, t1: float, t2: float) -> None:
        """Write (sl, t1, t2) back into plan in a tolerant way."""
        if isinstance(plan.get("stop"), dict):
            plan["stop"]["hard"] = round(float(sl), 2)
        else:
            plan["stop"] = {"hard": round(float(sl), 2)}
        plan["hard_sl"] = round(float(sl), 2)

        if not isinstance(plan.get("targets"), list):
            plan["targets"] = []
        if len(plan["targets"]) < 1:
            plan["targets"].append({"name": "T1", "level": round(float(t1), 2), "rr": None})
        else:
            plan["targets"][0]["name"] = plan["targets"][0].get("name", "T1")
            plan["targets"][0]["level"] = round(float(t1), 2)
        if len(plan["targets"]) < 2:
            plan["targets"].append({"name": "T2", "level": round(float(t2), 2), "rr": None})
        else:
            plan["targets"][1]["name"] = plan["targets"][1].get("name", "T2")
            plan["targets"][1]["level"] = round(float(t2), 2)

    def _sanitize_exits(self, sym: str, side: str, entry_px: float, plan: Dict[str, Any]) -> None:
        """
        Enforce:
          LONG:  SL < entry < T1 <= T2
          SHORT: SL > entry > T1 >= T2
        If invalid, correct orientation/order; if needed, rebuild T1/T2 from R with defaults T1=1.0R, T2=1.6R.
        Log one-liner when sanitization occurs.
        """
        sideU = side.upper()
        sl, t1, t2 = self._extract_plan_exits(plan)

        changed = False
        reasons: List[str] = []

        # Fallback R if SL invalid/absent -> 0.5% of price or 0.05 whichever larger
        fallback_R = max(entry_px * 0.005, 0.05)

        if sideU == "BUY":
            if math.isnan(sl) or sl >= entry_px:
                sl = entry_px - fallback_R; changed = True; reasons.append("fix_sl_side_or_nan")
            R = max(entry_px - sl, 0.0)
            if R <= 0:
                sl = entry_px - fallback_R; R = fallback_R; changed = True; reasons.append("fix_sl_zero_R")

            valid_t1 = (not math.isnan(t1)) and (t1 > entry_px)
            valid_t2 = (not math.isnan(t2)) and (t2 >= (t1 if not math.isnan(t1) else entry_px))
            if not (valid_t1 and valid_t2):
                t1 = entry_px + 1.0 * R; t2 = entry_px + 1.6 * R
                changed = True; reasons.append("rebuild_targets_from_R")
            elif t1 > t2:
                t1, t2 = t2, t1; changed = True; reasons.append("swap_t1_t2_order")

        else:  # SELL / SHORT
            if math.isnan(sl) or sl <= entry_px:
                sl = entry_px + fallback_R; changed = True; reasons.append("fix_sl_side_or_nan")
            R = max(sl - entry_px, 0.0)
            if R <= 0:
                sl = entry_px + fallback_R; R = fallback_R; changed = True; reasons.append("fix_sl_zero_R")

            valid_t1 = (not math.isnan(t1)) and (t1 < entry_px)
            valid_t2 = (not math.isnan(t2)) and (t2 <= (t1 if not math.isnan(t1) else entry_px))
            if not (valid_t1 and valid_t2):
                t1 = entry_px - 1.0 * R; t2 = entry_px - 1.6 * R
                changed = True; reasons.append("rebuild_targets_from_R")
            elif t1 < t2:
                t1, t2 = t2, t1; changed = True; reasons.append("swap_t1_t2_order")

        if changed:
            self._inject_plan_exits(plan, sl, t1, t2)
            logger.info(
                f"executor.sanitize_exits {sym} side={sideU} entry={entry_px:.2f} "
                f"SL={sl:.2f} T1={t1:.2f} T2={t2:.2f} reason={'|'.join(reasons)}"
            )

    # -------- Plan extraction (STRICT) -------- #
    def _extract_plan_fields(self, sym: str, plan: Dict[str, Any]) -> Tuple[str, int, Optional[float]]:
        side = plan.get("side") or plan.get("bias")
        if not side:
            raise ValueError(f"plan missing 'side'/'bias' for {sym}")
        side = str(side).upper()
        if side not in ("BUY", "SELL", "LONG", "SHORT"):
            raise ValueError(f"invalid side '{side}' for {sym}")
        if side == "LONG": side = "BUY"
        elif side == "SHORT": side = "SELL"

        if "qty" not in plan:
            raise ValueError(f"plan missing 'qty' for {sym}")
        qty = int(plan["qty"])
        if qty <= 0:
            raise ValueError(f"invalid qty {qty} for {sym}")

        price_hint: Optional[float] = None
        zone = plan.get("entry_zone") or plan.get("entry")
        if isinstance(zone, (list, tuple)) and len(zone) == 2 and zone[0] is not None and zone[1] is not None:
            price_hint = (float(zone[0]) + float(zone[1])) / 2.0
        elif plan.get("price") is not None:
            price_hint = float(plan["price"])

        return side, qty, price_hint

    # -------- Order placement (no upsert here) -------- #
    def _place_order(
        self, sym: str, side: str, qty: int, price_hint: Optional[float]
    ) -> Tuple[Optional[float], Optional[pd.Timestamp]]:
        """
        Place the order and return (fill_price, entry_ts).
        MARKET fill prefers get_ltp_ts() for consistency with exit tick-clock.
        """
        # LIMIT requires a price; no silent fallback
        if self.order_mode == "LIMIT" and price_hint is None:
            logger.info(f"executor.block {sym} reason=limit_without_price_hint")
            return None, None

        args = {
            "symbol": sym,
            "side": side,
            "qty": int(qty),
            "order_type": self.order_mode,
            "product": self.order_product,
            "variety": self.order_variety,
        }
        if self.order_mode == "LIMIT":
            args["price"] = float(price_hint)

        order_id = self.broker.place_order(**args)
        logger.info(f"executor.order_placed sym={sym} side={side} qty={qty} mode={self.order_mode} id={order_id}")

        # Fill price & ts
        ts: Optional[pd.Timestamp] = None
        px: Optional[float] = None

        get_ts = getattr(self, "get_ltp_ts", None)
        if callable(get_ts):
            try:
                px, ts = get_ts(sym)  # (ltp, ts)
            except Exception:
                px, ts = None, None

        if self.order_mode == "MARKET":
            if px is None:
                try:
                    px = float(self.broker.get_ltp(sym, ltp=price_hint))  # MockBroker honors hint
                except Exception:
                    px = float(price_hint) if price_hint is not None else None
        else:  # LIMIT
            px = float(price_hint) if price_hint is not None else px

        if ts is None:
            ts = _now_naive_ist()

        return px, ts

    def _update_risk_on_fill(self, sym: str, side: str, qty: int, price: float) -> None:
        try:
            self.risk.open_positions[sym] = {"side": side, "qty": qty, "avg_price": float(price)}
            self.risk.gross_exposure_rupees += abs(qty * float(price))
        except Exception as e:
            logger.warning(f"executor: risk update failed sym={sym}: {e}")

    # -------- Main loop -------- #
    def run_once(self) -> None:
        item = self.oq.get_next()
        if not item:
            return
        try:
            sym = str(item.get("symbol"))
            plan: Dict[str, Any] = dict(item.get("plan") or {})
            if not sym or not plan:
                logger.info("executor.skip empty item")
                return

            side, qty, price_hint = self._extract_plan_fields(sym, plan)

            if not self.risk.can_open_more():
                logger.info(f"executor.block {sym} reason=max_concurrent positions={len(self.risk.open_positions)}")
                return

            # --- TICK-CLOCK TIME NOW (for gates) ---
            ts_now = None
            get_ts = getattr(self, "get_ltp_ts", None)
            if callable(get_ts):
                try:
                    _px_now, ts_now = get_ts(sym)
                except Exception:
                    ts_now = None
            if ts_now is None:
                ts_now = _now_naive_ist()

            # --- HARD GATES: Entry cutoff + EOD ---
            try:
                md = _minute_of_day(pd.Timestamp(ts_now))
                if self._entry_cutoff_md is not None and md >= int(self._entry_cutoff_md):
                    logger.info(f"executor.block {sym} reason=after_entry_cutoff_{self.entry_cutoff_hhmm}")
                    return
                if self._eod_md is not None and md >= int(self._eod_md):
                    logger.info(f"executor.block {sym} reason=after_eod_{self.eod_hhmm}")
                    return
            except Exception:
                md_now = _minute_of_day(_now_naive_ist())
                if self._entry_cutoff_md is not None and md_now >= int(self._entry_cutoff_md):
                    logger.info(f"executor.block {sym} reason=after_entry_cutoff_{self.entry_cutoff_hhmm}")
                    return
                if self._eod_md is not None and md_now >= int(self._eod_md):
                    logger.info(f"executor.block {sym} reason=after_eod_{self.eod_hhmm}")
                    return

            # --- Place order (fill price + tick ts) ---
            fill_price, entry_ts = self._place_order(sym, side, qty, price_hint)
            if fill_price is None:
                return

            # --- Stamp entry time on plan ---
            plan["entry_ts"] = str(pd.Timestamp(entry_ts))
            plan["entry_epoch_ms"] = int(pd.Timestamp(entry_ts).value // 1_000_000)

            # --- Sanitize exits at entry (prevents instant-exit anomalies) ---
            try:
                self._sanitize_exits(sym, side, float(fill_price), plan)
            except Exception as e:
                logger.warning(f"executor.sanitize_exits error sym={sym}: {e}")

            # --- Persist position & log entry ---
            if self.positions is not None:
                try:
                    self.positions.upsert(Position(symbol=sym, side=side, qty=qty, avg_price=float(fill_price), plan=plan))
                except Exception as e:
                    logger.warning(f"executor: positions.upsert failed sym={sym}: {e}")

            trade_logger.info(f"ENTRY | {sym} | Side: {side} | Qty: {qty} | Price: ₹{float(fill_price):.2f}")

            # --- Update risk ---
            self._update_risk_on_fill(sym, side, qty, float(fill_price))

        except Exception as e:
            logger.exception(f"executor.run_once error item={item}: {e}")

    def run_forever(self, sleep_ms: int = 200) -> None:
        try:
            while True:
                self.run_once()
                time.sleep(max(0.0, sleep_ms / 1000.0))
        except KeyboardInterrupt:
            logger.info("executor.stop (KeyboardInterrupt)")
