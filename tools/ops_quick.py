"""Fast ops checks — direct kiteconnect, NO instrument-dump load.

KiteBroker's constructor pulls ~96k instruments from Upstox (~10-20s) — far too
slow for a 2-line status query. This helper talks to Kite directly (sub-second)
for the routine live-ops questions. READ-ONLY.

Usage (VM):  .venv/bin/python tools/ops_quick.py orders [SYMBOL]
             .venv/bin/python tools/ops_quick.py funds
             .venv/bin/python tools/ops_quick.py slots
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _kc():
    from kiteconnect import KiteConnect
    from config.env_setup import env
    kc = KiteConnect(api_key=env.KITE_API_KEY)
    kc.set_access_token(env.KITE_ACCESS_TOKEN)
    return kc


def cmd_orders(symbol: str | None) -> None:
    for o in _kc().orders():
        if symbol and o["tradingsymbol"] != symbol.upper().replace("NSE:", ""):
            continue
        print("%s %-12s %-4s %-6s qty=%-5s filled=%-5s avg=%-9s %-18s %s" % (
            o["order_id"], o["tradingsymbol"], o["transaction_type"],
            o["order_type"], o["quantity"], o.get("filled_quantity"),
            o.get("average_price"), o["status"], o.get("status_message") or ""))


def cmd_funds() -> None:
    eq = _kc().margins().get("equity", {})
    av, ut = eq.get("available", {}) or {}, eq.get("utilised", {}) or {}
    print("cash=%.0f live_balance=%.0f utilised_debits=%.0f" % (
        av.get("cash") or 0, av.get("live_balance") or 0, ut.get("debits") or 0))


def cmd_slots() -> None:
    d = json.loads((ROOT / "state" / "overnight_slots.json").read_text())
    for s in d["slots"]:
        if s["status"] != "free":
            print("slot %s %-12s %-14s fill=%-10s qty=%-6s amo=%s" % (
                s["slot_id"], s["status"], s.get("symbol"),
                s.get("buy_fill_price"), s.get("buy_qty"),
                s.get("amo_sell_order_id")))
    from collections import Counter
    print("statuses:", dict(Counter(s["status"] for s in d["slots"])))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    po = sub.add_parser("orders"); po.add_argument("symbol", nargs="?")
    sub.add_parser("funds")
    sub.add_parser("slots")
    a = p.parse_args()
    if a.cmd == "orders":
        cmd_orders(a.symbol)
    elif a.cmd == "funds":
        cmd_funds()
    else:
        cmd_slots()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
