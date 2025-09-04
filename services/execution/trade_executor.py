# services/execution/trade_executor.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple
import time

from config.logging_config import get_loggers
from config.filters_setup import load_filters
from services.orders.order_queue import OrderQueue

logger, _ = get_loggers()


@dataclass
class Position:
    symbol: str
    side: str                 # "BUY" or "SELL"
    qty: int
    avg_price: float
    plan: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RiskState:
    max_concurrent: int                         # STRICT: required
    open_positions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gross_exposure_rupees: float = 0.0
    sector_exposure: Dict[str, float] = field(default_factory=dict)
    correlation_buckets: Dict[str, List[str]] = field(default_factory=dict)

    def can_open_more(self) -> bool:
        return len(self.open_positions) < int(self.max_concurrent)


class TradeExecutor:
    """
    STRICT (no in-code defaults). Portfolio-level gate runs here (if present).
    """

    def __init__(
        self,
        broker,
        order_queue: OrderQueue,
        risk_state: RiskState,
        positions: Optional[Any] = None,  # must expose .upsert(Position) if provided
    ) -> None:
        self.broker = broker
        self.oq = order_queue
        self.cfg = load_filters()

        # STRICT config (KeyError if missing – by design)
        self.order_mode = str(self.cfg["exec_order_mode"]).upper()   # "MARKET" or "LIMIT"
        self.order_product = str(self.cfg["exec_product"]).upper()   # e.g., "MIS"
        self.order_variety = str(self.cfg["exec_variety"]).lower()   # e.g., "regular"

        self.risk = risk_state
        self.positions = positions


    # -------- Plan extraction (STRICT) -------- #
    def _extract_plan_fields(self, sym: str, plan: Dict[str, Any]) -> Tuple[str, int, Optional[float]]:
        side = plan.get("side") or plan.get("bias")
        if not side:
            raise ValueError(f"plan missing 'side'/'bias' for {sym}")
        side = str(side).upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"invalid side '{side}' for {sym}")

        if "qty" not in plan:
            raise ValueError(f"plan missing 'qty' for {sym}")
        qty = int(plan["qty"])
        if qty <= 0:
            raise ValueError(f"invalid qty {qty} for {sym}")

        price_hint = None
        zone = plan.get("entry_zone") or plan.get("entry")
        if isinstance(zone, (list, tuple)) and len(zone) == 2 and zone[0] is not None and zone[1] is not None:
            price_hint = (float(zone[0]) + float(zone[1])) / 2.0
        elif "price" in plan and plan["price"] is not None:
            price_hint = float(plan["price"])

        return side, qty, price_hint

    # -------- Order placement (STRICT) -------- #
    def _place_order(self, sym: str, side: str, qty: int, price_hint: Optional[float], plan: Dict[str, Any]) -> Optional[float]:
        # LIMIT requires a price; no silent fallback
        if self.order_mode == "LIMIT" and price_hint is None:
            logger.info(f"executor.block {sym} reason=limit_without_price_hint")
            return None

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

        if self.order_mode == "MARKET":
            fill_price = float(self.broker.get_ltp(sym, ltp=price_hint))
        else:
            fill_price = float(price_hint)

        if self.positions is not None:
            try:
                self.positions.upsert(Position(symbol=sym, side=side, qty=qty, avg_price=fill_price, plan=plan))
            except Exception as e:
                logger.warning(f"executor: positions.upsert failed sym={sym}: {e}")

        return fill_price

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

            fill_price = self._place_order(sym, side, qty, price_hint, plan)
            if fill_price is None:
                return

            self._update_risk_on_fill(sym, side, qty, fill_price)

        except Exception as e:
            logger.exception(f"executor.run_once error item={item}: {e}")

    def run_forever(self, sleep_ms: int = 200) -> None:
        try:
            while True:
                self.run_once()
                time.sleep(max(0.0, sleep_ms / 1000.0))
        except KeyboardInterrupt:
            logger.info("executor.stop (KeyboardInterrupt)")
