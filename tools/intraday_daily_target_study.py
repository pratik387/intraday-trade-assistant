"""Would a daily profit target have helped the intraday book? Reconstructs the curve.

The book has no live notion of "how much am I up right now". P&L is only known
per trade at exit, so a rule like "stop for the day at +Rs5,000" cannot be
evaluated — or run — without first rebuilding the intraday equity curve.

Method, which is arithmetic rather than modelling:

  1. Every final exit gives side, qty, entry price, and the entry/exit clock
     (exit timestamp minus time_in_trade_minutes).
  2. Fetch each symbol's 5-minute closes for that session.
  3. At every 5m stamp a position contributes:
         0                                  before its entry
         (entry - close) * qty  [SHORT]     while held
         (close - entry) * qty  [LONG]
         its realised P&L                   after its exit
  4. Sum across positions, subtract the day's fees, and the running total is the
     book's mark-to-market. The peak is the largest value it reached.

Bar CLOSES are used, so a spike that round-trips inside one 5m bar is invisible.
That biases the measured peak DOWN, never up. The per-trade MFE recorded live by
the engine bounds it from above: summing MFEs assumes every trade peaked at the
same instant, which is the ceiling.

2026-09-08 is why this matters: the book reached +Rs16,155 (gross, bar close
labelled 14:05 = the 14:10 print) and settled at -Rs4,518 net, because
up_spike_fade_short is deliberately stop-less (its 9%
catastrophe stop is a blowup guard, and Phase 5 found 0 of 11 stop/target
variants improved it in both eras). No per-trade rule would have caught that
give-back. A book-level daily target is the only mechanism that would.

Usage:
    python tools/intraday_daily_target_study.py
    python tools/intraday_daily_target_study.py --targets 3000 5000 7500 10000
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import glob
import json
import os
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

ARCHIVE_GLOB = "/tmp/oci_an/*.jsonl"
SESSION_GLOB = "logs/paper_2026*"


def _active_intraday_setups(cfg: dict) -> set:
    return {n for n, r in (cfg.get("setups") or {}).items()
            if str(r.get("horizon")) != "multi_day"
            and (r.get("enabled") or r.get("paper_enabled"))}


def _rows_from(fh, active: set) -> list:
    out = []
    for ln in fh:
        try:
            t = json.loads(ln)
        except Exception:
            continue
        if t.get("setup_type") not in active or not t.get("is_final_exit"):
            continue
        ep, q, ts = t.get("actual_entry_price"), t.get("qty"), t.get("timestamp")
        if ep is None or not q or not ts:
            continue
        end = pd.to_datetime(str(ts).replace("Z", ""))
        # analytics.jsonl semantics, verified 2026-09-08 on 167 paper rows:
        #   total_trade_pnl == pnl  -> GROSS (reconciles with (entry-exit)*qty)
        #   net_pnl                 -> after Zerodha MIS costs
        #   gross_pnl               -> absent on these rows
        # An earlier revision preferred total_trade_pnl and CALLED it net, which
        # overstated every session by the day's fees (median Rs32.79/trade).
        net = t.get("net_pnl")
        gross = t.get("pnl") if t.get("pnl") is not None else t.get("total_trade_pnl")
        if net is None:
            net = gross
        out.append(dict(
            day=end.strftime("%Y-%m-%d"), sym=t["symbol"], qty=int(q), ep=float(ep),
            xp=float(t.get("exit_price") or ep), end=end,
            start=end - dt.timedelta(minutes=float(t.get("time_in_trade_minutes") or 0)),
            net=float(net or 0), gross=(float(gross) if gross is not None else None),
            short=("short" in str(t.get("bias", "")).lower()
                   or "short" in str(t.get("setup_type", "")).lower())))
    return out


def load_trades(active: set) -> dict:
    """Archive plus local session dirs, deduped by (day, symbol, entry)."""
    seen, days = set(), defaultdict(list)
    srcs = sorted(glob.glob(ARCHIVE_GLOB))
    srcs += [os.path.join(d, "analytics.jsonl") for d in sorted(glob.glob(SESSION_GLOB))]
    for p in srcs:
        if not os.path.exists(p):
            continue
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                for r in _rows_from(fh, active):
                    k = (r["day"], r["sym"], round(r["ep"], 4))
                    if k in seen:
                        continue
                    seen.add(k)
                    days[r["day"]].append(r)
        except Exception:
            continue
    return days


def fetch_bars(sdk, day: str, syms: list) -> dict:
    """Historical for past sessions; the intraday endpoint for today."""
    today = dt.date.today().isoformat()
    out = {}
    if day == today:
        for s in syms:
            try:
                out[s] = sdk.get_intraday_5m(s)
            except Exception:
                out[s] = None
        return out
    try:
        out = asyncio.run(sdk.async_fetch_historical_5m_batch(
            syms, day, day, concurrency=4, rps=5)) or {}
    except Exception:
        out = {}
    return out


def day_curve(rows: list, bars: dict) -> pd.Series | None:
    grid = None
    have = [r for r in rows if bars.get(r["sym"]) is not None and len(bars[r["sym"]])]
    if not have:
        return None
    for i, r in enumerate(have):
        df = bars[r["sym"]].copy()
        df.index = pd.to_datetime(df.index)
        px = df["close"].astype(float)
        s = ((r["ep"] - px) * r["qty"]) if r["short"] else ((px - r["ep"]) * r["qty"])
        s[df.index < r["start"]] = 0.0
        realised = ((r["ep"] - r["xp"]) if r["short"] else (r["xp"] - r["ep"])) * r["qty"]
        s[df.index > r["end"]] = realised
        s = s.rename("p%d" % i)
        grid = s.to_frame() if grid is None else grid.join(s, how="outer")
    if grid is None:
        return None
    grid = grid.ffill().fillna(0.0)
    fees = sum((r["gross"] - r["net"]) for r in have if r["gross"] is not None)
    return grid.sum(axis=1) - fees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[3000, 5000, 7500, 10000, 15000])
    args = ap.parse_args()

    cfg = json.load(open("config/configuration.json", encoding="utf-8"))
    active = _active_intraday_setups(cfg)
    days = load_trades(active)
    print("sessions with intraday exits: %d  (%s .. %s)\n" % (
        len(days), min(days), max(days)))

    from broker.upstox.upstox_data_client import UpstoxDataClient
    sdk = UpstoxDataClient()

    recs, missing = [], 0
    for day in sorted(days):
        rows = days[day]
        bars = fetch_bars(sdk, day, sorted({r["sym"] for r in rows}))
        missing += sum(1 for r in rows
                       if bars.get(r["sym"]) is None or not len(bars[r["sym"]]))
        cur = day_curve(rows, bars)
        if cur is None or cur.empty:
            continue
        recs.append(dict(day=day, n=len(rows), peak=float(cur.max()),
                         ptime=cur.idxmax(), realised=sum(r["net"] for r in rows),
                         curve=[float(v) for v in cur.values]))
    print("  reconstructed %d sessions | bars missing for %d trades\n" % (len(recs), missing))

    print("  %-12s %3s %11s %7s %11s" % ("day", "n", "peak(net)", "@", "realised"))
    for r in sorted(recs, key=lambda x: -x["peak"])[:14]:
        print("  %-12s %3d %+11.0f %7s %+11.0f" % (
            r["day"], r["n"], r["peak"], r["ptime"].strftime("%H:%M"), r["realised"]))
    print("  ... (%d sessions total)\n" % len(recs))

    base = sum(r["realised"] for r in recs)
    print("  baseline realised over %d sessions: Rs%+.0f\n" % (len(recs), base))
    print("  %8s %6s %13s %11s %9s" % ("target", "fires", "book", "delta", "red days"))
    for t in args.targets:
        tot, fires = 0.0, 0
        red = 0
        for r in recs:
            hit = next((v for v in r["curve"] if v >= t), None)
            val = t if hit is not None else r["realised"]
            if hit is not None:
                fires += 1
            tot += val
            red += (val < 0)
        print("  %8.0f %6d %+13.0f %+11.0f %8.0f%%" % (
            t, fires, tot, tot - base, 100 * red / len(recs)))
    red0 = sum(1 for r in recs if r["realised"] < 0)
    print("  %8s %6s %+13.0f %+11.0f %8.0f%%" % (
        "none", 0, base, 0, 100 * red0 / len(recs)))

    print("\n  leave-one-out on the best target (fragility check)")
    best = max(args.targets, key=lambda t: sum(
        (t if any(v >= t for v in r["curve"]) else r["realised"]) for r in recs))
    firing = [r for r in recs if any(v >= best for v in r["curve"])]
    print("    target Rs%.0f fires on %d sessions" % (best, len(firing)))
    for r0 in firing:
        sub = [r for r in recs if r["day"] != r0["day"]]
        t2 = sum((best if any(v >= best for v in r["curve"]) else r["realised"]) for r in sub)
        b2 = sum(r["realised"] for r in sub)
        print("      without %s (realised %+8.0f): delta Rs%+.0f" % (
            r0["day"], r0["realised"], t2 - b2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
