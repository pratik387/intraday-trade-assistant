# close_dn_overnight_long Live Activation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Take the paper-validated `close_dn_overnight_long` overnight/MTF setup to real-money live on Zerodha, with a hybrid broker, AMO-window-correct cron split, and a GTT catastrophe stop.

**Architecture:** Live reuses the paper-validated Upstox data path verbatim and only flips the order sink to Kite via a composite `LiveOvernightBroker`. The single 15:26 entry run is split so the exit AMO + GTT stop are placed in a post-16:00 cron (Zerodha AMO window opens 4 PM), and `verify-exit` cancels the GTT after the AMO fills.

**Tech Stack:** Python 3.10, pytest, pandas, kiteconnect (Kite Connect v4), Upstox data client.

**Source spec:** `specs/2026-06-22-close_dn_overnight_long-live-activation-design.md`

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `services/capital_manager.py` | `OvernightSlot` gains `gtt_id`; `from_dict` tolerates legacy keys | Modify |
| `broker/kite/kite_broker.py` | `get_order_status`, `place_gtt_stop`, `cancel_gtt` | Modify |
| `broker/live_overnight_broker.py` | Composite: Upstox data + Kite orders | Create |
| `services/execution/overnight_handlers.py` | `run_entry` BUY-only; new `run_place_exit`; `run_verify_exit` cancels GTT; un-gate baseline build | Modify |
| `main.py` | `--action place-exit` + live hybrid broker construction | Modify |
| `config/configuration.json` | `catastrophe_stop_pct`, `gtt` block, live-pilot caps | Modify |
| `scripts/cron-place-exit.sh` | 16:05 exit-placement cron wrapper | Create |
| `docs/runbooks/close_dn_overnight_live.md` | Go-live runbook (caps swap, token, pilot) | Create |

Tests live under `tests/` mirroring source paths.

---

## Task 1: `OvernightSlot.gtt_id` field + robust `from_dict`

**Files:**
- Modify: `services/capital_manager.py` (`OvernightSlot` dataclass ~line 666; imports ~line 662)
- Test: `tests/services/test_capital_manager_overnight.py` (append)

- [ ] **Step 1: Write the failing test**

```python
# tests/services/test_capital_manager_overnight.py  (append)
def test_slot_roundtrips_gtt_id(tmp_path):
    from services.capital_manager import OvernightSlotPool
    state = tmp_path / "slots.json"
    pool = OvernightSlotPool(state, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=__import__("datetime").date(2026, 6, 22))
    slot.gtt_id = "GTT_123"
    pool.persist()
    pool2 = OvernightSlotPool(state, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert pool2._get_slot(slot.slot_id).gtt_id == "GTT_123"


def test_from_dict_ignores_unknown_legacy_keys():
    from services.capital_manager import OvernightSlot
    d = {"slot_id": 1, "status": "free", "paper_variant_b": None}  # legacy key not on the dataclass
    slot = OvernightSlot.from_dict(d)
    assert slot.slot_id == 1
    assert slot.gtt_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/test_capital_manager_overnight.py::test_slot_roundtrips_gtt_id tests/services/test_capital_manager_overnight.py::test_from_dict_ignores_unknown_legacy_keys -v`
Expected: FAIL — `OvernightSlot` has no attribute `gtt_id`; `from_dict` raises `TypeError` on `paper_variant_b`.

- [ ] **Step 3: Implement**

In `services/capital_manager.py`, change the import line `from dataclasses import dataclass, asdict` to:
```python
from dataclasses import dataclass, asdict, fields
```

Add the field to `OvernightSlot` (after `reserved_today`):
```python
    reserved_today: Optional[str] = None     # ISO date when reserve() was called
    gtt_id: Optional[str] = None             # broker GTT trigger id for the catastrophe stop
```

Replace `from_dict`:
```python
    @classmethod
    def from_dict(cls, d: dict) -> "OvernightSlot":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/services/test_capital_manager_overnight.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add services/capital_manager.py tests/services/test_capital_manager_overnight.py
git commit -m "feat(overnight): add gtt_id to OvernightSlot + tolerant from_dict"
```

---

## Task 2: `KiteBroker.get_order_status`

The overnight handler's `_live_poll_fill` / `_live_check_amo_fill` call `broker.get_order_status(order_id)` expecting `{"status", "average_price"}`. `KiteBroker` lacks it.

**Files:**
- Modify: `broker/kite/kite_broker.py` (add method after `get_order_fill_price` ~line 447)
- Test: `tests/broker/test_kite_broker_order_status.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/broker/test_kite_broker_order_status.py
import os, sys
from unittest.mock import patch
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def mock_env():
    with patch('broker.kite.kite_broker.env') as m:
        m.KITE_API_KEY = "k"; m.KITE_ACCESS_TOKEN = "t"
        yield m


@pytest.fixture
def live_broker():
    with patch("broker.kite.kite_broker.KiteConnect") as MockKC:
        inst = MockKC.return_value
        from broker.kite.kite_broker import KiteBroker
        b = KiteBroker(api_key="k", access_token="t", dry_run=False)
        b.kc = inst
        yield b, inst


def test_get_order_status_returns_status_and_price(live_broker):
    b, inst = live_broker
    inst.orders.return_value = [
        {"order_id": "111", "status": "COMPLETE", "average_price": 145.8},
        {"order_id": "222", "status": "OPEN", "average_price": 0},
    ]
    out = b.get_order_status("111")
    assert out["status"] == "COMPLETE"
    assert out["average_price"] == 145.8


def test_get_order_status_unknown_order(live_broker):
    b, inst = live_broker
    inst.orders.return_value = []
    out = b.get_order_status("999")
    assert out["status"] == "UNKNOWN"
    assert out["average_price"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/broker/test_kite_broker_order_status.py -v`
Expected: FAIL — `KiteBroker` has no attribute `get_order_status`.

- [ ] **Step 3: Implement**

In `broker/kite/kite_broker.py`, add after `get_order_fill_price`:
```python
    def get_order_status(self, order_id: str) -> Dict[str, Optional[float]]:
        """Return {'order_id','status','average_price'} for an order.

        Used by the overnight handler's fill-polling. status is upper-cased;
        'UNKNOWN' when the order is not found. average_price is 0.0 until filled.
        """
        if self.dry_run:
            for o in self._paper_orders:
                if str(o.get("order_id")) == str(order_id):
                    return {"order_id": str(order_id),
                            "status": str(o.get("status", "COMPLETE")).upper(),
                            "average_price": float(o.get("average_price") or 0.0)}
            return {"order_id": str(order_id), "status": "UNKNOWN", "average_price": 0.0}
        try:
            for o in self.kc.orders():
                if str(o.get("order_id")) == str(order_id):
                    return {"order_id": str(order_id),
                            "status": str(o.get("status") or "").upper(),
                            "average_price": float(o.get("average_price") or 0.0)}
        except Exception as e:
            logger.warning(f"get_order_status failed for {order_id}: {e}")
        return {"order_id": str(order_id), "status": "UNKNOWN", "average_price": 0.0}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/broker/test_kite_broker_order_status.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add broker/kite/kite_broker.py tests/broker/test_kite_broker_order_status.py
git commit -m "feat(kite): add get_order_status for overnight fill polling"
```

---

## Task 3: `KiteBroker.place_gtt_stop` + `cancel_gtt`

Single-leg GTT stop-loss SELL (verified MTF-supported) + cancel.

**Files:**
- Modify: `broker/kite/kite_broker.py`
- Test: `tests/broker/test_kite_broker_gtt.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/broker/test_kite_broker_gtt.py
import os, sys
from unittest.mock import patch, MagicMock
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@pytest.fixture(autouse=True)
def mock_env():
    with patch('broker.kite.kite_broker.env') as m:
        m.KITE_API_KEY = "k"; m.KITE_ACCESS_TOKEN = "t"
        yield m


@pytest.fixture
def live_broker():
    with patch("broker.kite.kite_broker.KiteConnect") as MockKC:
        inst = MockKC.return_value
        inst.GTT_TYPE_SINGLE = "single"
        inst.TRANSACTION_TYPE_SELL = "SELL"
        inst.ORDER_TYPE_LIMIT = "LIMIT"
        inst.PRODUCT_MTF = "MTF"
        inst.PRODUCT_CNC = "CNC"
        from broker.kite.kite_broker import KiteBroker
        b = KiteBroker(api_key="k", access_token="t", dry_run=False)
        b.kc = inst
        b.get_ltp = MagicMock(return_value=145.0)
        yield b, inst


def test_place_gtt_stop_calls_kc_with_single_leg(live_broker):
    b, inst = live_broker
    inst.place_gtt.return_value = {"trigger_id": 777}
    gid = b.place_gtt_stop(symbol="NSE:RELIANCE", qty=68, trigger_price=138.5, limit_price=137.0, product="MTF")
    assert gid == "777"
    kwargs = inst.place_gtt.call_args.kwargs
    assert kwargs["trigger_type"] == "single"
    assert kwargs["trigger_values"] == [138.5]
    assert kwargs["orders"][0]["transaction_type"] == "SELL"
    assert kwargs["orders"][0]["product"] == "MTF"
    assert kwargs["orders"][0]["price"] == 137.0


def test_cancel_gtt_calls_delete(live_broker):
    b, inst = live_broker
    assert b.cancel_gtt("777") is True
    inst.delete_gtt.assert_called_once_with(777)


def test_cancel_gtt_swallows_error(live_broker):
    b, inst = live_broker
    inst.delete_gtt.side_effect = RuntimeError("already gone")
    assert b.cancel_gtt("777") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/broker/test_kite_broker_gtt.py -v`
Expected: FAIL — no `place_gtt_stop` / `cancel_gtt`.

- [ ] **Step 3: Implement**

In `broker/kite/kite_broker.py`, add:
```python
    def place_gtt_stop(self, *, symbol: str, qty: int, trigger_price: float,
                       limit_price: float, product: str = "MTF") -> str:
        """Place a single-leg GTT stop-loss SELL. Returns the GTT trigger_id (str).

        Catastrophe failsafe for an overnight position whose AMO did not fill.
        On trigger, a LIMIT SELL is submitted at `limit_price` (set a small buffer
        below `trigger_price` to ensure fill).
        """
        exch, tsym = _split_symbol(symbol)
        prod = product.upper()
        kc_product = getattr(self.kc, "PRODUCT_MTF", "MTF") if prod == "MTF" else self.kc.PRODUCT_CNC
        last_price = float(self.get_ltp(symbol) or 0.0)
        orders = [{
            "exchange": exch, "tradingsymbol": tsym,
            "transaction_type": self.kc.TRANSACTION_TYPE_SELL,
            "quantity": int(qty), "order_type": self.kc.ORDER_TYPE_LIMIT,
            "product": kc_product, "price": float(limit_price),
        }]
        if self.dry_run:
            self._paper_order_counter += 1
            gid = f"PAPER_GTT_{self._paper_order_counter:08d}"
            logger.info(f"[PAPER] GTT stop {symbol} trig={trigger_price} lim={limit_price} -> {gid}")
            return gid
        resp = self.kc.place_gtt(
            trigger_type=self.kc.GTT_TYPE_SINGLE,
            tradingsymbol=tsym, exchange=exch,
            trigger_values=[float(trigger_price)],
            last_price=last_price, orders=orders,
        )
        trigger_id = resp.get("trigger_id") if isinstance(resp, dict) else resp
        logger.info(f"GTT stop placed: {symbol} trig={trigger_price} -> trigger_id={trigger_id}")
        return str(trigger_id)

    def cancel_gtt(self, gtt_id: str) -> bool:
        """Delete a GTT by trigger_id. Returns True on success, False on failure."""
        if self.dry_run:
            return True
        try:
            self.kc.delete_gtt(int(gtt_id))
            return True
        except Exception as e:
            logger.warning(f"cancel_gtt failed for {gtt_id}: {e}")
            return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/broker/test_kite_broker_gtt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add broker/kite/kite_broker.py tests/broker/test_kite_broker_gtt.py
git commit -m "feat(kite): single-leg GTT stop place + cancel for overnight failsafe"
```

---

## Task 4: `LiveOvernightBroker` composite

Data → Upstox (`_data_sdk`); orders/GTT → Kite. Exposes the exact surface `overnight_handlers` consumes.

**Files:**
- Create: `broker/live_overnight_broker.py`
- Test: `tests/broker/test_live_overnight_broker.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/broker/test_live_overnight_broker.py
import os, sys
from unittest.mock import MagicMock
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _make():
    from broker.live_overnight_broker import LiveOvernightBroker
    data_sdk = MagicMock()
    data_sdk._equity_instruments = ["NSE:RELIANCE", "NSE:TCS"]
    kite = MagicMock()
    return LiveOvernightBroker(data_sdk=data_sdk, kite=kite), data_sdk, kite


def test_data_methods_route_to_data_sdk():
    b, data_sdk, kite = _make()
    b.get_intraday_5m("NSE:RELIANCE")
    data_sdk.get_intraday_5m.assert_called_once_with("NSE:RELIANCE")
    b.get_daily("NSE:RELIANCE", days=30)
    data_sdk.get_daily.assert_called_once_with("NSE:RELIANCE", days=30)
    assert b.list_symbols() == ["NSE:RELIANCE", "NSE:TCS"]
    assert b._data_sdk is data_sdk
    # never used the kite client for data
    kite.get_intraday_5m.assert_not_called()


def test_order_methods_route_to_kite():
    b, data_sdk, kite = _make()
    kite.place_order.return_value = "ORD1"
    oid = b.place_order(symbol="NSE:RELIANCE", side="BUY", qty=10, product="MTF", variety="regular")
    assert oid == "ORD1"
    kite.place_order.assert_called_once()
    b.get_order_status("ORD1"); kite.get_order_status.assert_called_once_with("ORD1")
    b.place_gtt_stop(symbol="NSE:RELIANCE", qty=10, trigger_price=1.0, limit_price=0.9, product="MTF")
    kite.place_gtt_stop.assert_called_once()
    b.cancel_gtt("G1"); kite.cancel_gtt.assert_called_once_with("G1")
    # data sdk never asked to place orders
    data_sdk.place_order.assert_not_called()


def test_has_no_dry_session_date_attr():
    # _build_market_context treats presence of _dry_session_date as backtest mode.
    b, _, _ = _make()
    assert getattr(b, "_dry_session_date", None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/broker/test_live_overnight_broker.py -v`
Expected: FAIL — module `broker.live_overnight_broker` does not exist.

- [ ] **Step 3: Implement**

```python
# broker/live_overnight_broker.py
from __future__ import annotations
"""Composite broker for LIVE overnight trading.

Market DATA comes from Upstox (the paper-validated path); ORDERS, fills, and
GTTs go to Kite. The signal pipeline is byte-identical to paper — only the
order sink changes. Constructed by main.py for `--mode overnight` live runs.
"""
from typing import Any, Dict, List, Optional

import pandas as pd


class LiveOvernightBroker:
    def __init__(self, data_sdk: Any, kite: Any) -> None:
        # `_data_sdk` name is load-bearing: overnight_handlers reads
        # broker._data_sdk for the async 5m batch + baseline build.
        self._data_sdk = data_sdk
        self._kite = kite

    # ---------------- market data (Upstox) ----------------
    def get_intraday_5m(self, symbol: str) -> Optional[pd.DataFrame]:
        return self._data_sdk.get_intraday_5m(symbol)

    def get_daily(self, symbol: str, days: int) -> Optional[pd.DataFrame]:
        return self._data_sdk.get_daily(symbol, days=days)

    def list_symbols(self, exchange: str = "NSE", instrument_type: str = "EQ") -> List[str]:
        syms = getattr(self._data_sdk, "_equity_instruments", None)
        return list(syms) if syms else []

    def get_symbol_map(self) -> Dict[str, int]:
        return self._data_sdk.get_symbol_map()

    def resolve_tokens(self, symbols) -> List[int]:
        return self._data_sdk.resolve_tokens(symbols)

    # ---------------- orders / GTT (Kite) ----------------
    def place_order(self, **kwargs) -> str:
        return self._kite.place_order(**kwargs)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        return self._kite.get_order_status(order_id)

    def get_ltp(self, symbol: str) -> float:
        return self._kite.get_ltp(symbol)

    def place_gtt_stop(self, **kwargs) -> str:
        return self._kite.place_gtt_stop(**kwargs)

    def cancel_gtt(self, gtt_id: str) -> bool:
        return self._kite.cancel_gtt(gtt_id)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/broker/test_live_overnight_broker.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add broker/live_overnight_broker.py tests/broker/test_live_overnight_broker.py
git commit -m "feat(broker): LiveOvernightBroker (Upstox data + Kite orders)"
```

---

## Task 5: Split entry — `run_entry` BUY-only + new `run_place_exit`

`run_entry` currently places the BUY *and* the AMO SELL. The AMO SELL must move to a post-16:00 cron (`run_place_exit`) that also places the GTT stop.

**Files:**
- Modify: `services/execution/overnight_handlers.py`
- Test: `tests/services/execution/test_overnight_place_exit.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/services/execution/test_overnight_place_exit.py
import os, sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.capital_manager import OvernightSlotPool


def _seed_t0_open_slot(state_path):
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 22))
    pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-22T15:26:00", order_id="BUY1")
    pool.persist()
    return slot.slot_id


def _config(state_path):
    return {"setups": {"close_dn_overnight_long": {
        "mode": "overnight", "enabled": True, "paper_enabled": True,
        "capital_allocation": {"state_file": str(state_path), "max_concurrent_slots": 2,
                               "margin_per_slot_inr": 10000, "max_new_positions_per_day": 2},
        "catastrophe_stop_pct": 5.0, "gtt_limit_buffer_pct": 0.5,
    }}}


@pytest.fixture
def patched_registry(monkeypatch, state_path):
    # _select_overnight_setups builds SetupRegistry from config; stub it to a
    # single overnight spec so the test doesn't need the full registry.
    spec = MagicMock()
    spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = _config(state_path)["setups"]["close_dn_overnight_long"]
    import services.execution.overnight_handlers as oh
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    return spec


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "slots.json"


def test_run_place_exit_places_amo_and_gtt(state_path, patched_registry):
    import services.execution.overnight_handlers as oh
    sid = _seed_t0_open_slot(state_path)
    broker = MagicMock()
    broker.place_order.return_value = "AMO1"
    broker.place_gtt_stop.return_value = "GTT1"
    summary = oh.run_place_exit(_config(state_path), broker,
                                now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=False)
    assert summary["placed_count"] == 1
    # AMO SELL placed as variety=amo product=MTF
    amo_kwargs = broker.place_order.call_args.kwargs
    assert amo_kwargs["side"] == "SELL" and amo_kwargs["variety"] == "amo" and amo_kwargs["product"] == "MTF"
    # GTT trigger = 140 * 0.95 = 133.0
    gtt_kwargs = broker.place_gtt_stop.call_args.kwargs
    assert round(gtt_kwargs["trigger_price"], 2) == 133.0
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool._get_slot(sid)
    assert slot.amo_sell_order_id == "AMO1"
    assert slot.gtt_id == "GTT1"


def test_run_place_exit_idempotent(state_path, patched_registry):
    import services.execution.overnight_handlers as oh
    _seed_t0_open_slot(state_path)
    broker = MagicMock(); broker.place_order.return_value = "AMO1"; broker.place_gtt_stop.return_value = "GTT1"
    cfg = _config(state_path)
    oh.run_place_exit(cfg, broker, now_ist=pd.Timestamp("2026-06-22 16:05:00"), paper_mode=False)
    broker.place_order.reset_mock()
    summary2 = oh.run_place_exit(cfg, broker, now_ist=pd.Timestamp("2026-06-22 16:06:00"), paper_mode=False)
    assert summary2["placed_count"] == 0           # slot already has amo_sell_order_id
    broker.place_order.assert_not_called()


def test_run_place_exit_refuses_before_amo_window(state_path, patched_registry):
    import services.execution.overnight_handlers as oh
    _seed_t0_open_slot(state_path)
    broker = MagicMock()
    summary = oh.run_place_exit(_config(state_path), broker,
                                now_ist=pd.Timestamp("2026-06-22 15:30:00"), paper_mode=False)
    assert summary.get("refused_amo_window") is True
    broker.place_order.assert_not_called()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_overnight_place_exit.py -v`
Expected: FAIL — `run_place_exit` does not exist.

- [ ] **Step 3a: Remove the AMO block from `run_entry`**

In `services/execution/overnight_handlers.py`, delete the AMO-placement block in `run_entry` (the `# Place AMO SELL for next trading day` section through `pool.attach_amo_sell(...)`), so the per-symbol loop ends right after `pool.attach_buy_fill(...)` with the success bookkeeping:
```python
            pool.attach_buy_fill(
                slot.slot_id,
                fill_price=float(fill_price),
                fill_ts_iso=now.isoformat(),
                order_id=str(buy_order_id),
            )

            summary["fired_count"] += 1
            summary["events"].append({
                "symbol": symbol, "qty": plan.qty,
                "product": evt.context["product"],
                "buy_fill_price": float(fill_price),
            })
```

- [ ] **Step 3b: Add `run_place_exit`**

Add to `services/execution/overnight_handlers.py` (after `run_entry`):
```python
def run_place_exit(
    config: dict,
    broker,
    *,
    now_ist: Optional[pd.Timestamp] = None,
    paper_mode: bool = True,
) -> dict:
    """At ~16:05 IST (T0): for each t0_open slot place the exit AMO SELL + a
    GTT catastrophe stop. Idempotent — slots already carrying an
    amo_sell_order_id are skipped, so a re-run (or the morning fallback)
    is safe. Refuses to run before the 16:00 AMO window opens.
    """
    from utils.time_util import _now_naive_ist
    from services.capital_manager import OvernightSlotPool

    now = pd.Timestamp(now_ist) if now_ist is not None else _now_naive_ist()
    summary: dict = {"now_ist": str(now), "paper_mode": paper_mode,
                     "placed_count": 0, "gtt_failed_count": 0, "events": []}

    # AMO window opens 16:00 IST (Zerodha NSE equity). Refuse earlier so we
    # never submit an AMO that the exchange would reject during market hours.
    if (now.hour, now.minute) < (16, 0):
        logger.warning("run_place_exit: before 16:00 AMO window (now=%s); refusing", now)
        summary["refused_amo_window"] = True
        return summary

    setups = _select_overnight_setups(config, paper_mode=paper_mode)
    if not setups:
        logger.info("run_place_exit: no overnight setups active; exit")
        return summary
    sc = setups[0].raw_config
    slot_cfg = sc["capital_allocation"]
    state_path = Path(slot_cfg["state_file"])
    if not state_path.exists():
        logger.info("run_place_exit: no state file; nothing to place")
        return summary
    pool = OvernightSlotPool(
        state_path,
        max_slots=int(slot_cfg["max_concurrent_slots"]),
        margin_per_slot=float(slot_cfg["margin_per_slot_inr"]),
        max_new_per_day=int(slot_cfg["max_new_positions_per_day"]),
    )
    catastrophe_pct = float(sc["catastrophe_stop_pct"])
    gtt_buffer_pct = float(sc["gtt_limit_buffer_pct"])

    for slot in list(pool.active()):
        if slot.status != "t0_open":
            continue
        if slot.amo_sell_order_id is not None:
            continue  # already placed — idempotent skip
        if slot.buy_fill_price is None or slot.notional_inr <= 0:
            logger.warning("run_place_exit: slot %d has no buy fill; skipping", slot.slot_id)
            continue
        qty = int(round(slot.notional_inr / slot.buy_fill_price))
        next_day = _next_trading_day(date.fromisoformat(slot.reserved_today))
        # AMO SELL for next-day pre-open
        amo_id = _place_amo_sell(
            broker, symbol=slot.symbol, qty=qty,
            product=slot.product or "CNC", paper_mode=paper_mode,
            trade_id=f"OVERNIGHT_AMO_{slot.reserved_today}_{slot.slot_id}",
        )
        pool.attach_amo_sell(slot.slot_id, str(amo_id), next_day)
        # GTT catastrophe stop (failsafe if the AMO does not fill)
        trigger = slot.buy_fill_price * (1.0 - catastrophe_pct / 100.0)
        limit = trigger * (1.0 - gtt_buffer_pct / 100.0)
        try:
            gid = broker.place_gtt_stop(
                symbol=slot.symbol, qty=qty,
                trigger_price=round(trigger, 2), limit_price=round(limit, 2),
                product=slot.product or "CNC",
            )
            slot.gtt_id = str(gid)
        except Exception as e:
            logger.error("run_place_exit: GTT place failed for %s: %s "
                         "(AMO still queued; morning failsafe covers it)", slot.symbol, e)
            summary["gtt_failed_count"] += 1
        summary["placed_count"] += 1
        summary["events"].append({"slot_id": slot.slot_id, "symbol": slot.symbol,
                                  "amo_sell_order_id": str(amo_id), "gtt_id": slot.gtt_id,
                                  "expected_exit_date": next_day.isoformat()})

    pool.persist()
    logger.info("run_place_exit: complete | placed=%d gtt_failed=%d",
                summary["placed_count"], summary["gtt_failed_count"])
    return summary
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_overnight_place_exit.py tests/services/execution/test_overnight_handlers.py -v`
Expected: PASS (new file passes; existing `run_entry` tests still pass — update any that asserted AMO placement inside entry to assert it's now absent).

- [ ] **Step 5: Commit**

```bash
git add services/execution/overnight_handlers.py tests/services/execution/test_overnight_place_exit.py
git commit -m "feat(overnight): split exit into run_place_exit (AMO+GTT post-16:00)"
```

---

## Task 6: `run_verify_exit` cancels the GTT after settle

After the AMO fill is confirmed and the slot is settled, the dangling GTT must be cancelled — otherwise a later trigger opens a naked short.

**Files:**
- Modify: `services/execution/overnight_handlers.py` (`run_verify_exit`, after `pool.settle(...)`)
- Test: `tests/services/execution/test_overnight_verify_cancels_gtt.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/services/execution/test_overnight_verify_cancels_gtt.py
import os, sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock
import pandas as pd
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from services.capital_manager import OvernightSlotPool


@pytest.fixture
def state_path(tmp_path):
    return tmp_path / "slots.json"


def _config(state_path):
    return {"setups": {"close_dn_overnight_long": {
        "mode": "overnight", "enabled": True, "paper_enabled": True,
        "capital_allocation": {"state_file": str(state_path), "max_concurrent_slots": 2,
                               "margin_per_slot_inr": 10000, "max_new_positions_per_day": 2},
    }}}


def _seed_ready_to_settle(state_path):
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    slot = pool.reserve(symbol="NSE:RELIANCE", product="MTF", leverage=2.5, today=date(2026, 6, 22))
    pool.attach_buy_fill(slot.slot_id, fill_price=140.0, fill_ts_iso="2026-06-22T15:26:00", order_id="BUY1")
    pool.attach_amo_sell(slot.slot_id, "AMO1", date(2026, 6, 23))
    slot.gtt_id = "GTT1"
    pool.persist()
    return slot.slot_id


def test_verify_exit_cancels_gtt_on_settle(monkeypatch, state_path):
    import services.execution.overnight_handlers as oh
    sid = _seed_ready_to_settle(state_path)
    spec = MagicMock(); spec.name = "close_dn_overnight_long"; spec.mode = "overnight"; spec.enabled = True
    spec.raw_config = _config(state_path)["setups"]["close_dn_overnight_long"]
    monkeypatch.setattr(oh, "_select_overnight_setups", lambda config, *, paper_mode: [spec])
    # Force a known sell fill price and skip baseline build.
    monkeypatch.setattr(oh, "_paper_fill_price_exit", lambda b, s, d: 150.0)
    broker = MagicMock(); broker.cancel_gtt.return_value = True
    oh.run_verify_exit(_config(state_path), broker,
                       now_ist=pd.Timestamp("2026-06-23 09:30:00"), paper_mode=True)
    broker.cancel_gtt.assert_called_once_with("GTT1")
    pool = OvernightSlotPool(state_path, max_slots=2, margin_per_slot=10000, max_new_per_day=2)
    assert pool._get_slot(sid).gtt_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_overnight_verify_cancels_gtt.py -v`
Expected: FAIL — `cancel_gtt` never called.

- [ ] **Step 3: Implement**

In `services/execution/overnight_handlers.py`, in `run_verify_exit`, immediately after the `pool.settle(...)` call, add:
```python
        # Cancel the dangling catastrophe GTT now that the AMO has filled and
        # the slot is flat — a later GTT trigger would open a NAKED SHORT.
        if slot.gtt_id and hasattr(broker, "cancel_gtt"):
            ok = broker.cancel_gtt(slot.gtt_id)
            if not ok:
                logger.warning("run_verify_exit: GTT %s cancel returned False for slot %d "
                               "(verify manually — risk of naked short)", slot.gtt_id, slot.slot_id)
            slot.gtt_id = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_overnight_verify_cancels_gtt.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/execution/overnight_handlers.py tests/services/execution/test_overnight_verify_cancels_gtt.py
git commit -m "fix(overnight): cancel catastrophe GTT on settle to avoid naked short"
```

---

## Task 7: Un-gate the candidate baseline build for live

The baseline/candidate pre-filter (consumed by `run_entry`) is gated `if paper_mode:`. It keys off `broker._data_sdk`, which the live hybrid broker now provides — so it should run in live too.

**Files:**
- Modify: `services/execution/overnight_handlers.py` (`run_verify_exit`, the `if paper_mode:` baseline block ~line 642)
- Test: `tests/services/execution/test_overnight_baseline_gating.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/services/execution/test_overnight_baseline_gating.py
import inspect
import services.execution.overnight_handlers as oh

def test_baseline_build_gated_on_data_sdk_not_paper_mode():
    src = inspect.getsource(oh.run_verify_exit)
    # The baseline build must trigger on data_sdk presence, not paper_mode.
    assert 'getattr(broker, "_data_sdk", None) is not None' in src
    # Guard against regression to the paper-only gate wrapping the baseline build.
    assert "if paper_mode:\n        try:\n            data_sdk = getattr(broker" not in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_overnight_baseline_gating.py -v`
Expected: FAIL — still gated on `if paper_mode:`.

- [ ] **Step 3: Implement**

In `run_verify_exit`, change the baseline-build guard from:
```python
    if paper_mode:
        try:
            data_sdk = getattr(broker, "_data_sdk", None)
            if data_sdk is not None:
```
to:
```python
    if getattr(broker, "_data_sdk", None) is not None:
        try:
            data_sdk = broker._data_sdk
            if data_sdk is not None:
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/services/execution/test_overnight_baseline_gating.py tests/services/execution/test_overnight_handlers.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add services/execution/overnight_handlers.py tests/services/execution/test_overnight_baseline_gating.py
git commit -m "feat(overnight): build candidate baseline whenever a data SDK is wired (live too)"
```

---

## Task 8: `main.py` — `--action place-exit` + live hybrid broker

**Files:**
- Modify: `main.py` (arg validation ~688-712; live broker construction ~746-753; overnight routing ~755-771)
- Test: `tests/test_main_overnight_place_exit_wiring.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_main_overnight_place_exit_wiring.py
import inspect
import main

def test_place_exit_is_a_valid_overnight_action():
    src = inspect.getsource(main)
    assert '"place-exit"' in src  # added to --action choices / routing
    assert "run_place_exit" in src

def test_live_overnight_uses_hybrid_broker():
    src = inspect.getsource(main)
    assert "LiveOvernightBroker" in src
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_main_overnight_place_exit_wiring.py -v`
Expected: FAIL.

- [ ] **Step 3a: Add `place-exit` to the `--action` choices** (the `ap.add_argument("--action", ...)` line):
```python
    ap.add_argument("--action", choices=["run", "entry", "verify-exit", "place-exit", "exit", "verify-entry"],
                    default="run", ...)
```
(keep the existing help text; add `place-exit` to the choices list only.)

- [ ] **Step 3b: Allow `place-exit` for overnight** in the validation block. Where `--mode overnight` currently permits `entry`/`verify-exit`, ensure `place-exit` is accepted (no parser_error for `overnight` + `place-exit`). Concretely, after the existing overnight checks, no extra error is needed — just confirm the `args.mode == "overnight" and args.action == "exit"` guard still rejects `exit` for overnight, and that `place-exit` is not caught by the `--action=run` guard.

- [ ] **Step 3c: Build the live hybrid broker.** Replace the live-mode broker construction:
```python
        else:
            # Live mode: Upstox for data (paper-validated path) + Kite for orders.
            from broker.kite.kite_broker import KiteBroker  # noqa: WPS433
            from broker.upstox.upstox_data_client import UpstoxDataClient
            from broker.live_overnight_broker import LiveOvernightBroker
            kite = KiteBroker(
                api_key=os.environ.get("KITE_API_KEY"),
                access_token=os.environ.get("KITE_ACCESS_TOKEN"),
            )
            broker = LiveOvernightBroker(data_sdk=UpstoxDataClient(), kite=kite)
            paper_mode = False
```

- [ ] **Step 3d: Route `place-exit`.** In the `if args.mode == "overnight":` block, add a branch:
```python
            from services.execution.overnight_handlers import run_entry, run_verify_exit, run_place_exit
            if args.action == "entry":
                summary = run_entry(cfg, broker, paper_mode=paper_mode)
                print(f"[overnight entry] fired={summary['fired_count']} "
                      f"skipped={summary['skipped_count']} rejected={summary['rejected_count']}", file=sys.stderr)
            elif args.action == "place-exit":
                summary = run_place_exit(cfg, broker, paper_mode=paper_mode)
                print(f"[overnight place-exit] placed={summary['placed_count']} "
                      f"gtt_failed={summary['gtt_failed_count']}", file=sys.stderr)
            else:  # verify-exit
                summary = run_verify_exit(cfg, broker, paper_mode=paper_mode)
                print(f"[overnight verify-exit] settled={summary['settled_count']} "
                      f"released={summary['released_count']} orphan_t0={summary['orphan_t0_count']}", file=sys.stderr)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_main_overnight_place_exit_wiring.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_main_overnight_place_exit_wiring.py
git commit -m "feat(main): --action place-exit + live hybrid overnight broker"
```

---

## Task 9: Config — `catastrophe_stop_pct`, `gtt` keys, live-pilot caps

**Files:**
- Modify: `config/configuration.json` (`setups.close_dn_overnight_long`)
- Test: `tests/config/test_close_dn_live_config.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/config/test_close_dn_live_config.py
from config.filters_setup import load_filters

def test_close_dn_has_gtt_and_live_pilot_keys():
    cfg = load_filters()
    s = cfg["setups"]["close_dn_overnight_long"]
    assert float(s["catastrophe_stop_pct"]) == 5.0
    assert float(s["gtt_limit_buffer_pct"]) > 0.0
    ca = s["capital_allocation"]
    # Live-pilot caps documented for the 1-slot first run.
    assert int(ca["_live_pilot_max_concurrent_slots"]) == 1
    assert int(ca["_live_pilot_margin_per_slot_inr"]) == 10000
    # Full live caps reflect Rs10k/slot, Rs2L total.
    assert int(ca["_live_margin_per_slot_inr"]) == 10000
    assert int(ca["_live_active_margin_inr"]) == 200000
    assert int(ca["_live_max_concurrent_slots"]) == 20
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/config/test_close_dn_live_config.py -v`
Expected: FAIL — keys absent.

- [ ] **Step 3: Implement**

In `config/configuration.json`, under `setups.close_dn_overnight_long`, add alongside the existing keys:
```json
      "catastrophe_stop_pct": 5.0,
      "gtt_limit_buffer_pct": 0.5,
```
and update the `capital_allocation` `_live_*` shadow keys + add pilot keys:
```json
        "_live_active_margin_inr": 200000,
        "_live_cushion_inr": 50000,
        "_live_max_concurrent_slots": 20,
        "_live_margin_per_slot_inr": 10000,
        "_live_max_new_positions_per_day": 20,
        "_live_pilot_max_concurrent_slots": 1,
        "_live_pilot_max_new_positions_per_day": 1,
        "_live_pilot_margin_per_slot_inr": 10000,
        "_live_rationale": "Rs10k base margin/trade (MTF-leveraged to ~Rs26k notional). Rs2L total => 20 slots; 2-day T0->T+2 settlement lock means effective concurrency ~ fires/day x 2. Pilot starts at 1 slot to validate live fills + GTT before scaling. PF 2.44 / HO 1.52 measured under <=4 slots; re-run the confidence card before exceeding 20.",
```
Leave `enabled: false` and the inflated *active* keys unchanged — the cap swap + enable flip is an operational go-live step (Task 10 runbook), not a code change.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/config/test_close_dn_live_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add config/configuration.json tests/config/test_close_dn_live_config.py
git commit -m "feat(config): close_dn live GTT keys + Rs10k/slot live-pilot caps"
```

---

## Task 10: Cron wrapper + go-live runbook

**Files:**
- Create: `scripts/cron-place-exit.sh`
- Modify: `scripts/cron-entry.sh` (drop the AMO note — entry no longer places AMO)
- Create: `docs/runbooks/close_dn_overnight_live.md`
- Test: `tests/test_cron_place_exit_script.py` (Create)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cron_place_exit_script.py
import os, stat
from pathlib import Path

def test_place_exit_cron_exists_and_executable():
    p = Path("scripts/cron-place-exit.sh")
    assert p.exists(), "scripts/cron-place-exit.sh missing"
    text = p.read_text()
    assert "--mode overnight --action place-exit" in text
    assert text.startswith("#!/bin/bash")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_cron_place_exit_script.py -v`
Expected: FAIL — script missing.

- [ ] **Step 3a: Create `scripts/cron-place-exit.sh`**

```bash
#!/bin/bash
# Triggered at 16:05 IST every weekday by cron (after the 16:00 AMO window opens).
# Places the exit AMO SELL + GTT catastrophe stop for each overnight position
# opened by today's 15:26 entry run.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
    if [[ -x ".venv/bin/python" ]]; then PYTHON_BIN=".venv/bin/python"; else PYTHON_BIN=".venv/Scripts/python"; fi
fi

# Default = paper (Upstox data + simulated orders). For LIVE, set MODE_FLAGS=""
# in the crontab line so main.py builds the Kite hybrid broker.
MODE_FLAGS="${MODE_FLAGS:---paper-trading --data-source upstox --session-date $(date +%F)}"

LOG_DIR="logs"; mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight_place_exit_$(date +%Y-%m-%d).log"

"$PYTHON_BIN" main.py --mode overnight --action place-exit $MODE_FLAGS >> "$LOG_FILE" 2>&1
```

Then mark executable:
```bash
git update-index --chmod=+x scripts/cron-place-exit.sh 2>/dev/null || chmod +x scripts/cron-place-exit.sh
```

- [ ] **Step 3b: Create `docs/runbooks/close_dn_overnight_live.md`** with the go-live procedure:

```markdown
# close_dn_overnight_long — Go-Live Runbook

## Live crontab (IST; CRON_TZ=Asia/Kolkata). MODE_FLAGS="" forces live + Kite.
26 15 * * 1-5  cd /opt/intraday-trade-assistant && MODE_FLAGS="" scripts/cron-entry.sh
05 16 * * 1-5  cd /opt/intraday-trade-assistant && MODE_FLAGS="" scripts/cron-place-exit.sh
30 09 * * 1-5  cd /opt/intraday-trade-assistant && MODE_FLAGS="" scripts/cron-verify-exit.sh

## Preconditions (do ALL before flipping enabled)
1. Confirm the VM forward-paper ledger exists and PF is acceptable (local state/ is backtest-seeded).
2. Confirm KITE_ACCESS_TOKEN daily refresh runs before 15:26 (reuse the intraday daemon's refresh).
3. Dry-run E2E on a known Cell-#5 fire date: entry -> place-exit -> verify-exit produces a clean trade.

## Pilot (1 slot) then scale
- Set capital_allocation active keys to the _live_pilot_* values; set enabled=true. Run >= several sessions.
- Watch: real BUY fill, 16:05 AMO + GTT placement, 09:30 AMO fill + GTT cancel. No dangling GTTs.
- Then swap active keys to the full _live_* values (20 slots / Rs2L).

## Rollback
- Set enabled=false. In-flight positions still settle via verify-exit (idempotent); GTTs auto-cancel on settle.
```

- [ ] **Step 3c: Update `scripts/cron-entry.sh`** — remove the line in its header comment that says it "places AMO SELL"; entry now places the BUY only (AMO moved to cron-place-exit.sh). Adjust the comment block accordingly.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/Scripts/python -m pytest tests/test_cron_place_exit_script.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/cron-place-exit.sh scripts/cron-entry.sh docs/runbooks/close_dn_overnight_live.md tests/test_cron_place_exit_script.py
git commit -m "feat(overnight): 16:05 place-exit cron + go-live runbook"
```

---

## Task 11: Full-suite + dry-run E2E verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full overnight + broker test suites**

Run:
```
.venv/Scripts/python -m pytest tests/services/test_capital_manager_overnight.py tests/services/execution/ tests/broker/ tests/config/test_close_dn_live_config.py tests/test_main_overnight_place_exit_wiring.py tests/test_cron_place_exit_script.py -v
```
Expected: PASS, no regressions.

- [ ] **Step 2: Dry-run E2E on a known Cell-#5 fire date (paper)**

Run (entry → place-exit → verify-exit across two sessions):
```
.venv/Scripts/python main.py --mode overnight --action entry       --paper-trading --data-source upstox --session-date 2024-03-15
.venv/Scripts/python main.py --mode overnight --action place-exit   --paper-trading --data-source upstox --session-date 2024-03-15
.venv/Scripts/python main.py --mode overnight --action verify-exit  --paper-trading --data-source upstox --session-date 2024-03-18
```
Expected: entry fires ≥1; place-exit reports `placed>=1`; verify-exit settles and reports no orphans. Inspect `state/overnight_slots.json` between runs.

- [ ] **Step 3: Confirm no GTT/AMO leak**

Verify in the verify-exit log that each settled slot logged a GTT cancel and the slot's `gtt_id` is null afterward.

- [ ] **Step 4: Final commit (if any test fixups were needed)**

```bash
git add -A
git commit -m "test(overnight): full-suite + dry-run E2E green for live activation"
```

---

## Self-review notes

- **Spec coverage:** Hybrid broker → Task 4/8; cron split (15:26/16:05/09:30) → Tasks 5/8/10; ₹10k sizing → Task 9 (math already in `plan_long_strategy`, unchanged); GTT place + cancel-on-settle → Tasks 3/5/6; `get_order_status` → Task 2; baseline build in live → Task 7; `gtt_id` persistence → Task 1; preconditions/runbook → Task 10/11. Token refresh = operational (runbook precondition 2), not code.
- **Live-vs-paper entry-price delta** (spec F): monitored in pilot (runbook), no code task — correct (it's an observation gate, not a build item).
- **Method-name consistency:** `place_gtt_stop` / `cancel_gtt` / `get_order_status` / `run_place_exit` used identically across broker, handler, main, and tests.
