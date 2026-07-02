"""Append one funds/margin snapshot line to logs/funds_timeline.jsonl.

READ-ONLY. Cron-driven every 15 min during market hours to build the intraday
timeline of WHEN freed MTF margin becomes spendable again (the T+1-vs-T+2
slot-release question, 2026-07-02). Fields:
  - cash / live_balance / utilised_debits: Kite equity margins view
  - mtf_initial_margin: sum of holdings' mtf.initial_margin (margin currently
    blocked in open MTF positions — drops when the broker releases it)

Never raises: a token/API failure logs an error row instead (cron-safe).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "logs" / "funds_timeline.jsonl"


def main() -> int:
    from utils.time_util import _now_naive_ist
    row = {"ts": _now_naive_ist().isoformat()}
    try:
        from broker.kite.kite_broker import KiteBroker
        kc = KiteBroker(dry_run=False).kc
        eq = kc.margins().get("equity", {})
        av = eq.get("available", {}) or {}
        ut = eq.get("utilised", {}) or {}
        row.update(
            cash=float(av.get("cash") or 0.0),
            live_balance=float(av.get("live_balance") or 0.0),
            utilised_debits=float(ut.get("debits") or 0.0),
        )
        mtf_margin = 0.0
        for h in kc.holdings():
            mtf_margin += float((h.get("mtf") or {}).get("initial_margin") or 0.0)
        row["mtf_initial_margin"] = mtf_margin
    except Exception as e:  # noqa: BLE001 - cron-safe: record the failure, exit 0
        row["error"] = f"{type(e).__name__}: {e}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
