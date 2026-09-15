"""Rebuild the intraday book's mark-to-market from a session's events.jsonl at
1-minute resolution, both as the dashboard shows it (GROSS, tick-marked) and net.

Why this exists: tools/intraday_daily_target_study.py used 5m bar CLOSES minus
fees. The Live tab sums closed-trade gross P&L + booked partials + LTP-marked
unrealised P&L, i.e. GROSS on every tick. Comparing a net/5m-close number with
what the screen showed understates the screen's peak twice over.

Curves per session:
  gross_close    : every position marked at the 1m bar close       (floor)
  gross_ceiling  : every position at its best 1m print (high/low)  (upper bound:
                   assumes all positions peaked in the same minute)
  net_close      : gross_close minus fees, charged per exit leg at its time
  gross_close5m  : the old 5m-close sampling, for reference

Fees per leg come from analytics.jsonl (pnl - net_pnl) when the leg is there,
else from the MIS fee model (open legs on a session still running).

Usage:
    python tools/intraday_mtm_peak_from_events.py --sessions "logs/paper_202609*"
    python tools/intraday_mtm_peak_from_events.py --analytics "/tmp/oci_an/*.jsonl" \
        --sessions "logs/paper_2026*" --targets 3000 5000 7500 10000

This supersedes tools/intraday_daily_target_study.py, which read final-exit rows
only and therefore carried every T1-partial trade at its leftover size.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

from tools.intraday_daily_target_backtest import intraday_fees  # noqa: E402


def load_session(d: str) -> dict:
    """trade_id -> dict(sym, side, qty, ep, start, legs=[{ts, qty, px, pnl, fee}])"""
    trades = {}
    ev = os.path.join(d, "events.jsonl")
    for ln in open(ev, encoding="utf-8", errors="replace"):
        try:
            e = json.loads(ln)
        except Exception:
            continue
        t = e.get("type")
        tid = e.get("trade_id")
        if t == "ENTRY":
            en = e["entry"]
            trades[tid] = dict(sym=e["symbol"], side=en["side"], qty=int(en["qty"]),
                               ep=float(en["price"]), start=pd.Timestamp(e["ts"]), legs=[],
                               setup=_setup_of(d, tid))
        elif t == "EXIT" and tid in trades:
            x = e["exit"]
            trades[tid]["legs"].append(dict(ts=pd.Timestamp(e["ts"]), qty=int(x["qty"]),
                                            px=float(x["price"]), pnl=float(x.get("pnl") or 0),
                                            fee=None))
    an = os.path.join(d, "analytics.jsonl")
    if os.path.exists(an):
        fees = {}
        for ln in open(an, encoding="utf-8", errors="replace"):
            try:
                a = json.loads(ln)
            except Exception:
                continue
            if a.get("pnl") is not None and a.get("net_pnl") is not None:
                fees[(a["symbol"], str(a["timestamp"])[:19])] = float(a["pnl"]) - float(a["net_pnl"])
        for tr in trades.values():
            for leg in tr["legs"]:
                leg["fee"] = fees.get((tr["sym"], str(leg["ts"])[:19]))
    for tr in trades.values():
        for leg in tr["legs"]:
            if leg["fee"] is None:
                leg["fee"] = intraday_fees(tr["ep"], leg["px"], leg["qty"])
    return trades


def load_analytics(path: str, active: set | None) -> dict:
    """Same shape as load_session(), from an analytics.jsonl.

    Every exit leg is a row (is_final_exit False for partials). The final row's
    `qty` is only the LEFTOVER quantity, so a trade's size is the sum of its
    legs. The earlier study read final rows only, which carried every T1-partial
    trade at half size for its whole life - about 19% of live trades."""
    legs = {}
    for ln in open(path, encoding="utf-8", errors="replace"):
        try:
            a = json.loads(ln)
        except Exception:
            continue
        if active is not None and a.get("setup_type") not in active:
            continue
        ep, q, ts = a.get("actual_entry_price"), a.get("qty"), a.get("timestamp")
        if ep is None or not q or not ts or a.get("pnl") is None:
            continue
        end = pd.Timestamp(str(ts).replace("Z", ""))
        key = (a["symbol"], round(float(ep), 4), end.strftime("%Y-%m-%d"))
        short = ("short" in str(a.get("bias", "")).lower()
                 or "short" in str(a.get("setup_type", "")).lower())
        tr = legs.setdefault(key, dict(sym=a["symbol"], side="SELL" if short else "BUY",
                                       qty=0, ep=float(ep), start=None, legs=[],
                                       setup=str(a.get("setup_type"))))
        start = end - pd.Timedelta(minutes=float(a.get("time_in_trade_minutes") or 0))
        tr["start"] = start if tr["start"] is None else min(tr["start"], start)
        tr["qty"] += int(q)
        net = a.get("net_pnl")
        fee = (float(a["pnl"]) - float(net)) if net is not None else None
        tr["legs"].append(dict(ts=end, qty=int(q), px=float(a.get("exit_price") or ep),
                               pnl=float(a["pnl"]), fee=fee))
    for tr in legs.values():
        for leg in tr["legs"]:
            if leg["fee"] is None:
                leg["fee"] = intraday_fees(tr["ep"], leg["px"], leg["qty"])
    return {"%s|%s" % (k[0], k[1]): v for k, v in legs.items()}


def curves(trades: dict, bars: dict) -> dict | None:
    idx = None
    for tr in trades.values():
        b = bars.get(tr["sym"])
        if b is None or b.empty:
            continue
        idx = b.index if idx is None else idx.union(b.index)
    if idx is None:
        return None
    out = {k: pd.Series(0.0, index=idx) for k in ("gross_close", "gross_ceiling", "net_close")}
    for tr in trades.values():
        b = bars.get(tr["sym"])
        if b is None or b.empty:
            continue
        b = b.reindex(idx).ffill()
        sgn = 1.0 if tr["side"] == "BUY" else -1.0
        best = b["high"] if sgn > 0 else b["low"]
        q = pd.Series(float(tr["qty"]), index=idx)
        q[idx < tr["start"]] = 0.0
        realised = pd.Series(0.0, index=idx)
        fees = pd.Series(0.0, index=idx)
        for leg in tr["legs"]:
            q[idx >= leg["ts"]] -= leg["qty"]
            realised[idx >= leg["ts"]] += leg["pnl"]
            fees[idx >= leg["ts"]] += leg["fee"]
        q = q.clip(lower=0.0)
        unreal_close = (sgn * (b["close"] - tr["ep"]) * q).fillna(0.0)
        unreal_best = (sgn * (best - tr["ep"]) * q).fillna(0.0)
        out["gross_close"] += unreal_close + realised
        out["gross_ceiling"] += unreal_best + realised
        out["net_close"] += unreal_close + realised - fees
    g = out["gross_close"]
    out["gross_close5m"] = g[(g.index.minute % 5 == 4)]
    return out


def fetch_1m(sdk, sym: str, day: dt.date) -> pd.DataFrame | None:
    s = sym if sym.startswith("NSE:") else "NSE:" + sym  # client keys need the prefix
    f = dt.datetime.combine(day, dt.time(9, 15))
    t = dt.datetime.combine(day, dt.time(15, 30))
    try:
        return sdk.get_historical_1m(s, f, t)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", default=None, help="glob of session dirs (events.jsonl)")
    ap.add_argument("--analytics", default=None,
                    help="glob of analytics.jsonl files (one session each), e.g. the OCI archive")
    ap.add_argument("--target", type=float, default=5000.0)
    ap.add_argument("--targets", type=float, nargs="+", default=[3000, 5000, 7500, 10000],
                    help="daily profit targets to sweep (stop for the day at +T)")
    ap.add_argument("--all-setups", action="store_true",
                    help="keep retired setups too (default: active intraday only)")
    ap.add_argument("--by-setup", action="store_true",
                    help="also sweep the target per setup on that setup's own daily curve")
    ap.add_argument("--since", default=None,
                    help="sweep only sessions on/after this date (YYYY-MM-DD). The book "
                         "was resized ~10x on 2026-08-14; a rupee target is not "
                         "comparable across that boundary")
    args = ap.parse_args()
    if not args.sessions and not args.analytics:
        ap.error("need --sessions and/or --analytics")
    cfg = json.load(open(_REPO_ROOT / "config" / "configuration.json", encoding="utf-8"))
    active = None if args.all_setups else {
        n for n, r in (cfg.get("setups") or {}).items()
        if str(r.get("horizon")) != "multi_day"
        and (r.get("enabled") or r.get("paper_enabled"))
        and n != "close_dn_overnight_long"}

    from broker.upstox.upstox_data_client import UpstoxDataClient
    sdk = UpstoxDataClient()

    # events-based sessions win over analytics-based ones for the same day
    sources = []
    for d in sorted(glob.glob(args.sessions or "")):
        if os.path.exists(os.path.join(d, "events.jsonl")):
            sources.append(("events", d))
    for f in sorted(glob.glob(args.analytics or "")):
        sources.append(("analytics", f))
    rows, seen_days = [], set()
    for kind, d in sources:
        trades = load_session(d) if kind == "events" else load_analytics(d, active)
        if kind == "events" and active is not None:
            trades = {k: v for k, v in trades.items() if v["setup"] in active}
        if not trades:
            continue
        day0 = min(tr["start"] for tr in trades.values()).date()
        if day0 in seen_days:
            continue
        seen_days.add(day0)
        day = min(tr["start"] for tr in trades.values()).date()
        bars = {tr["sym"]: fetch_1m(sdk, tr["sym"], day) for tr in trades.values()}
        missing = [s for s, b in bars.items() if b is None or b.empty]
        c = curves(trades, bars)
        if c is None:
            print("%s: no bars" % day)
            continue
        done = all(tr["legs"] and sum(l["qty"] for l in tr["legs"]) >= tr["qty"]
                   for tr in trades.values())
        r = dict(day=str(day), n=len(trades), done=done, missing=len(missing),
                 realised_gross=float(c["gross_close"].iloc[-1]),
                 realised_net=float(c["net_close"].iloc[-1]),
                 fees=sum(l["fee"] for tr in trades.values() for l in tr["legs"]),
                 curve_gross_close=[float(v) for v in c["gross_close"].values],
                 per_setup={})
        if args.by_setup:
            for su in sorted({tr["setup"] for tr in trades.values()}):
                sub = {k: v for k, v in trades.items() if v["setup"] == su}
                cs = curves(sub, bars)
                if cs is None:
                    continue
                r["per_setup"][su] = dict(
                    n=len(sub), fees=sum(l["fee"] for tr in sub.values() for l in tr["legs"]),
                    peak=float(cs["gross_close"].max()),
                    realised=float(cs["net_close"].iloc[-1]),
                    curve=[float(v) for v in cs["gross_close"].values])
        for k in ("gross_close", "gross_ceiling", "net_close", "gross_close5m"):
            s = c[k]
            r["peak_" + k] = float(s.max())
            r["t_" + k] = s.idxmax().strftime("%H:%M")
            hit = s[s >= args.target]
            r["hit_" + k] = hit.index[0].strftime("%H:%M") if len(hit) else "-"
        rows.append(r)
        print("%s  n=%d  %s  bars-missing=%d  [%s]" % (
            day, len(trades), "closed" if done else "OPEN", len(missing), kind))
        for tr in trades.values():
            legs = ", ".join("%s q%d @%.2f pnl%+.0f" % (
                l["ts"].strftime("%H:%M"), l["qty"], l["px"], l["pnl"]) for l in tr["legs"])
            print("     %-12s %-4s qty=%-5d ep=%-8.2f legs=%s" % (
                tr["sym"].replace("NSE:", ""), tr["side"], tr["qty"], tr["ep"], legs))

    print("\n=== peak of the book mark-to-market per session (Rs) ===")
    print("target Rs%.0f. hit = first minute at/above target\n" % args.target)
    hdr = "%-11s %2s %10s %10s | %9s %5s %5s | %9s %5s %5s | %9s %5s %5s | %9s %5s"
    print(hdr % ("day", "n", "end gross", "end net",
                 "1m close", "@", "hit", "1m ceil", "@", "hit",
                 "net 1m", "@", "hit", "5m close", "hit"))
    for r in rows:
        print(hdr % (
            r["day"] + ("" if r["done"] else "*"), r["n"],
            format(r["realised_gross"], "+,.0f"), format(r["realised_net"], "+,.0f"),
            format(r["peak_gross_close"], "+,.0f"), r["t_gross_close"], r["hit_gross_close"],
            format(r["peak_gross_ceiling"], "+,.0f"), r["t_gross_ceiling"], r["hit_gross_ceiling"],
            format(r["peak_net_close"], "+,.0f"), r["t_net_close"], r["hit_net_close"],
            format(r["peak_gross_close5m"], "+,.0f"), r["hit_gross_close5m"]))
    print("\n* = session still open; end = last bar seen, not the close")

    closed = [r for r in rows if r["done"] and (not args.since or r["day"] >= args.since)]
    if len(closed) < 2:
        return 0
    print("\n=== daily profit target on %d closed sessions%s ===" % (
        len(closed), (" since %s" % args.since) if args.since else ""))
    print("stop for the day the first minute the GROSS 1m-close curve is at/above T;")
    print("that day then books T minus the day's fees. Otherwise it books its realised net.\n")
    base = sum(r["realised_net"] for r in closed)
    print("  %8s %6s %13s %11s %9s" % ("target", "fires", "book (net)", "delta", "red days"))
    for t in args.targets:
        tot = fires = red = 0
        for r in closed:
            hit = any(v >= t for v in r["curve_gross_close"])
            val = (t - r["fees"]) if hit else r["realised_net"]
            fires += hit
            tot += val
            red += val < 0
        print("  %8.0f %6d %13s %11s %8.0f%%" % (
            t, fires, format(tot, "+,.0f"), format(tot - base, "+,.0f"), 100 * red / len(closed)))
    print("  %8s %6s %13s %11s %8.0f%%" % (
        "none", "-", format(base, "+,.0f"), "+0",
        100 * sum(1 for r in closed if r["realised_net"] < 0) / len(closed)))

    if args.by_setup:
        print("\n=== PER-SETUP: stop THIS setup once its own day is at +T (gross 1m close) ===")
        setups = sorted({su for r in closed for su in r["per_setup"]})
        for su in setups:
            srecs = [r["per_setup"][su] for r in closed if su in r["per_setup"]]
            days = [r["day"] for r in closed if su in r["per_setup"]]
            b = sum(x["realised"] for x in srecs)
            print("\n  %s | %d days | baseline %s | red %d/%d" % (
                su, len(srecs), format(b, "+,.0f"),
                sum(1 for x in srecs if x["realised"] < 0), len(srecs)))
            for t in args.targets:
                f = [(d, x) for d, x in zip(days, srecs) if any(v >= t for v in x["curve"])]
                if not f:
                    continue
                d_ = sum((t - x["fees"]) for _, x in f) - sum(x["realised"] for _, x in f)
                print("    Rs%-7s fires %2d  delta %10s  firing days end: %s" % (
                    format(int(t), ","), len(f), format(d_, "+,.0f"),
                    ", ".join("%s %s" % (d[5:], format(x["realised"], "+,.0f")) for d, x in f)))
    return 0


def _setup_of(session_dir: str, trade_id: str) -> str | None:
    """setup_type for a trade_id from the session's DECISION events (cached)."""
    cache = _setup_of.__dict__.setdefault("cache", {})
    if session_dir not in cache:
        m = {}
        for ln in open(os.path.join(session_dir, "events.jsonl"), encoding="utf-8", errors="replace"):
            try:
                e = json.loads(ln)
            except Exception:
                continue
            if e.get("type") == "DECISION":
                m[e.get("trade_id")] = (e.get("decision") or {}).get("setup_type")
        cache[session_dir] = m
    return cache[session_dir].get(trade_id)


if __name__ == "__main__":
    raise SystemExit(main())
