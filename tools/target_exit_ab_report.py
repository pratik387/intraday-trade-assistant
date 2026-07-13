"""A/B report: multiday target_touch exits vs their hold-to-close counterfactual.

For every ledger row with exit_reason="target_touch" (owner rows only — mirrors
are excluded), the position's SCHEDULED exit was entry_date + hold_days trading
days (the setup's K). Once that day has passed, its close is fetched and the
counterfactual computed with the same fee model (hold-to-close pays the full
K+1 days of MTF interest; the target exit paid fewer).

    verdict per trade:  actual_net (booked at target)  vs  counterfactual_net
    aggregate:          sum(actual - counterfactual) = rupees the rule added

Run on the VM (needs Upstox creds for recent closes):
    .venv/bin/python tools/target_exit_ab_report.py
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SETUPS = {  # setup -> hold_days K (mirrors config; read from config at runtime)
    "mtf_capitulation_revert_long": None,
    "low52_capitulation_revert_long": None,
    "zscore_oversold_revert_long": None,
    "crash2d_revert_long": None,
}


def main() -> int:
    from services.execution.overnight_handlers import _is_trading_day
    from services.execution.mtf_capitulation_handlers import _add_trading_days
    from tools.sub7_validation.build_per_setup_pnl import calc_fee_mtf, calc_fee_cnc

    cfg = json.loads((ROOT / "config" / "configuration.json").read_text(encoding="utf-8"))
    for s in SETUPS:
        SETUPS[s] = int(cfg["setups"][s]["hold_days"])

    sdk = None

    def close_on(symbol: str, d: date):
        nonlocal sdk
        if sdk is None:
            from broker.upstox.upstox_data_client import UpstoxDataClient
            sdk = UpstoxDataClient()
        df = sdk.get_daily(symbol, days=40)
        if df is None or df.empty:
            return None
        import pandas as pd
        row = df[pd.to_datetime(df.index).normalize() == pd.Timestamp(d)]
        return float(row["close"].iloc[0]) if not row.empty else None

    today = date.today()
    rows, tot_actual, tot_cf = [], 0.0, 0.0
    for setup, K in SETUPS.items():
        f = ROOT / "state" / f"decay_tripwire_{setup}.json"
        if not f.exists():
            continue
        for t in json.loads(f.read_text(encoding="utf-8")).get("trades", []):
            if not isinstance(t, dict) or t.get("exit_reason") != "target_touch":
                continue
            if t.get("attributed"):
                continue  # mirror row — the owner row carries the book trade
            sym, qty = t["symbol"], int(t["qty"])
            entry, tgt_px = float(t["entry_price"]), float(t["exit_price"])
            actual_net = float(t["net_pnl_inr"])
            # scheduled hold-to-close exit day (entry_date not persisted on the
            # row; reconstruct: actual exit ts date - hold shortfall is unsafe,
            # so derive from diagnostics if present else entry via ledger date).
            entry_d = t.get("_entry_date") or t.get("entry_date")
            if entry_d is None:
                # ts_iso is the ACTUAL (early) exit day; entry day is not
                # recoverable from the tripwire row alone before this tool ran —
                # events.jsonl diagnostics carry it. Fallback: skip with note.
                rows.append((setup, sym, "?", "entry_date missing — see events.jsonl", None, actual_net, None))
                continue
            sched = _add_trading_days(date.fromisoformat(str(entry_d)[:10]), K)
            if sched >= today or not _is_trading_day(sched):
                rows.append((setup, sym, sched.isoformat(), "pending (scheduled day not closed yet)", tgt_px, actual_net, None))
                continue
            cf_close = close_on(sym, sched)
            if cf_close is None:
                rows.append((setup, sym, sched.isoformat(), "close unavailable", tgt_px, actual_net, None))
                continue
            notional = entry * qty
            hold_cal = max(1, (sched - date.fromisoformat(str(entry_d)[:10])).days)
            fees_cf = calc_fee_mtf(notional, cf_close * qty, notional / 2.79, hold_cal)
            cf_net = (cf_close - entry) * qty - fees_cf
            tot_actual += actual_net
            tot_cf += cf_net
            rows.append((setup, sym, sched.isoformat(), f"cf_close={cf_close}", tgt_px, actual_net, cf_net))

    print(f"{'setup':<16}{'symbol':<14}{'sched_exit':<12}{'note':<38}{'booked_net':>11}{'hold_net':>10}{'rule +/-':>9}")
    for setup, sym, sched, note, tgt, a, c in rows:
        delta = f"{a - c:+.0f}" if c is not None else "—"
        cf = f"{c:.0f}" if c is not None else "—"
        print(f"{setup[:15]:<16}{sym.replace('NSE:',''):<14}{sched:<12}{note:<38}{a:>11.0f}{cf:>10}{delta:>9}")
    if tot_cf or tot_actual:
        print(f"\nAGGREGATE: booked {tot_actual:+.0f} vs hold-to-close {tot_cf:+.0f}  ->  rule added {tot_actual - tot_cf:+.0f}")
    else:
        print("\n(no resolved counterfactuals yet — scheduled exit days still pending)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
