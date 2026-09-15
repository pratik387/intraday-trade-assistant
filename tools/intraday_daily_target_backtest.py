"""Daily profit-target / loss-cap sweep on the July-2026 backtest runs (current setups, 2023-2026).

The live version (tools/intraday_mtm_peak_from_events.py) has only ~60 sessions.
This runs the identical reconstruction over the OCI backtest runs, which carry
the CURRENT setup roster across 868 sessions.

Traps this had to avoid, each of which bit on an earlier attempt:

  1. Run selection. The `backtest-results` bucket holds runs from May and July
     2026. The May runs contain ONLY retired setups. Only the 20260730/31 runs
     have the live roster, and the bucket must be listed with delimiter="/"
     (object paging silently truncates at ~61k objects).

  2. P&L fields. `pnl` on a row is that exit LEG's gross; `net_pnl` is the leg
     after costs; `total_trade_pnl` (final row only) is the trade's gross.

  3. Partial exits. A T1 partial writes its own row (is_final_exit False) and
     the final row's `qty` is only the LEFTOVER. The first version of this tool
     kept final rows only, so every T1-partial trade (407 of 4,131, ~10%) was
     carried at half size for its whole life and its T1 profit was dropped from
     the baseline. The extract now keeps every leg (bt_active_legs.jsonl) and a
     trade's size is the sum of its legs.

  4. Resolution. 5m bar closes miss spikes that round-trip inside a bar. Bars
     are now the 1m feathers (also the family the engine replays; the 5m_enriched
     files carry 32 contaminated symbol-days).

Curve: at every 1m stamp a position contributes 0 before entry, side*(close-entry)*
open_qty while held, and its realised leg P&L after each exit. The GROSS curve
is what the dashboard shows (LTP-marked, no fees). The rule fires on the gross
curve; a day that fires books T minus that day's fees.

Usage:
    python tools/intraday_daily_target_backtest.py --rule both
    python tools/intraday_daily_target_backtest.py --targets 3000 5000 --book-scale 9.92
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

MONTHLY = _REPO_ROOT / "backtest-cache-download" / "monthly"

# Zerodha intraday equity, post Oct-2024 (services/logging/trading_logger.py)
BROKERAGE_RATE, BROKERAGE_CAP = 0.0003, 20.0
STT_RATE = 0.00025            # sell side only
EXCHANGE_RATE = 0.0000307
SEBI_RATE = IPFT_RATE = 0.000001
STAMP_DUTY_RATE = 0.00003     # buy side only
GST_RATE = 0.18


def intraday_fees(entry_px: float, exit_px: float, qty: int) -> float:
    """Round-trip MIS cost. Brokerage is min(0.03%, Rs20) PER ORDER, which is why
    a bigger book cannot be modelled by scaling a smaller book's P&L: at Rs50k
    notional brokerage is Rs15/order, at Rs500k it is capped at Rs20. Fees are
    strongly sublinear in size, so a 10x book keeps far more than 10x the net."""
    et, xt = entry_px * qty, exit_px * qty
    brokerage = min(BROKERAGE_RATE * et, BROKERAGE_CAP) + min(BROKERAGE_RATE * xt, BROKERAGE_CAP)
    stt = xt * STT_RATE
    leg = et + xt
    exch, sebi, ipft = leg * EXCHANGE_RATE, leg * SEBI_RATE, leg * IPFT_RATE
    stamp = et * STAMP_DUTY_RATE
    gst = (brokerage + exch + sebi + ipft) * GST_RATE
    return brokerage + stt + exch + sebi + ipft + stamp + gst


def load_trades(path: Path) -> dict:
    """session -> [trade]; trade = sym, short, ep, raw_qty, start, legs=[(end, raw_qty, xp)]"""
    by_key = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        ep, q, ts = t.get("actual_entry_price"), t.get("qty"), t.get("timestamp")
        if ep is None or not q or not ts:
            continue
        end = pd.to_datetime(str(ts))
        key = (t["session"], t["symbol"], round(float(ep), 4))
        tr = by_key.setdefault(key, dict(
            session=t["session"], sym=str(t["symbol"]).replace("NSE:", ""), ep=float(ep),
            raw_qty=0, start=None, legs=[],
            short=("short" in str(t.get("bias", "")).lower()
                   or "short" in str(t.get("setup_type", "")).lower())))
        start = end - pd.Timedelta(minutes=float(t.get("time_in_trade_minutes") or 0))
        tr["start"] = start if tr["start"] is None else min(tr["start"], start)
        tr["raw_qty"] += int(q)
        tr["legs"].append((end, int(q), float(t.get("exit_price") or ep)))
    days = collections.defaultdict(list)
    for tr in by_key.values():
        tr["legs"].sort()
        days[tr["session"]].append(tr)
    return days


def scale_trade(tr: dict, scale: float, max_notional: float) -> None:
    """Re-size a 1x backtest trade the way the live book sizes it.

    Live sizing (services/risk/intraday_sizing.py): the 1x notional times
    `book_size_multiplier`, then clamped to `max_notional_pct_of_capital` x
    capital. Both sizing modes scale the same way, so qty_live =
    min(qty_1x * multiplier, max_notional // entry). The clamp matters: at 10x a
    Rs50k+ 1x trade hits the Rs500k ceiling, so the live book is NOT a uniform
    10x - measured median is ~4.6-6.4x. An earlier version used a flat 9.92x,
    which oversized every clamped trade. Each leg's gross and fee are recomputed
    at the new size."""
    sgn = -1.0 if tr["short"] else 1.0
    tr["qty"] = max(1, min(int(round(tr["raw_qty"] * scale)), int(max_notional // tr["ep"])))
    left, legs = tr["qty"], []
    for i, (end, rq, xp) in enumerate(tr["legs"]):
        q = left if i == len(tr["legs"]) - 1 else max(1, int(round(rq * scale)))
        q = min(q, left)
        left -= q
        legs.append(dict(end=end, qty=q, xp=xp,
                         gross=sgn * (xp - tr["ep"]) * q,
                         fee=intraday_fees(tr["ep"], xp, q)))
    tr["slegs"] = legs
    tr["gross"] = sum(l["gross"] for l in legs)
    tr["fees"] = sum(l["fee"] for l in legs)


def day_curves(rows: list, sub: pd.DataFrame) -> tuple | None:
    """(gross_close_curve, net_close_curve, n_used) on the union of the symbols' 1m stamps."""
    idx = None
    for r in rows:
        b = sub[sub["symbol"] == r["sym"]]
        if b.empty:
            continue
        idx = b["ts"] if idx is None else pd.Index(idx).union(b["ts"])
    if idx is None:
        return None
    idx = pd.DatetimeIndex(sorted(set(idx)))
    gross = pd.Series(0.0, index=idx)
    net = pd.Series(0.0, index=idx)
    used = 0
    for r in rows:
        b = sub[sub["symbol"] == r["sym"]]
        if b.empty:
            continue
        used += 1
        px = b.set_index("ts")["close"].astype(float).reindex(idx).ffill()
        sgn = -1.0 if r["short"] else 1.0
        q = pd.Series(float(r["qty"]), index=idx)
        q[idx < r["start"]] = 0.0
        realised = pd.Series(0.0, index=idx)
        fees = pd.Series(0.0, index=idx)
        for l in r["slegs"]:
            q[idx >= l["end"]] -= l["qty"]
            realised[idx >= l["end"]] += l["gross"]
            fees[idx >= l["end"]] += l["fee"]
        q = q.clip(lower=0.0)
        unreal = (sgn * (px - r["ep"]) * q).fillna(0.0)
        gross += unreal + realised
        net += unreal + realised - fees
    return gross, net, used


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=str(
        Path.home() / "AppData/Local/Temp/claude"
        / "E--Codebase-intraday-trade-assistant"
        / "d9c67968-368c-45aa-a25e-bd4d1cfb4906/scratchpad/bt_active_legs.jsonl"))
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[1000, 2000, 3000, 5000, 7500, 10000, 15000, 25000, 50000])
    ap.add_argument("--rule", choices=["target", "loss", "both"], default="target",
                    help="target: stop for the day at +T. loss: stop at -T (a daily "
                         "loss cap). both: sweep each separately from the same curves.")
    ap.add_argument("--book-scale", dest="scale", type=float, default=None,
                    help="live/backtest size multiplier. Default: config "
                         "intraday_sizing.book_size_multiplier (10 = Rs10k stop-risk "
                         "on Rs5L vs the backtest's Rs1k). QUANTITIES are multiplied "
                         "and fees RECOMPUTED at the larger turnover - dividing the "
                         "threshold instead would be wrong, because the Rs20/order "
                         "brokerage cap makes fees strongly sublinear in size. "
                         "Pass 1.0 to evaluate the book as the backtest traded it.")
    ap.add_argument("--max-notional", type=float, default=None,
                    help="per-trade notional ceiling applied after scaling. Default: "
                         "config max_notional_pct_of_capital x paper_initial_capital "
                         "(= Rs500k), the clamp the live sizer applies.")
    ap.add_argument("--dump", default=None,
                    help="write one JSON line per session (day, n, fees, peak, realised, "
                         "curve) so follow-up questions do not need a 10-minute rerun")
    ap.add_argument("--bars", choices=["1m", "5m"], default="1m",
                    help="1m = *_1m.feather (engine's family). 5m = *_5m_enriched.feather "
                         "(the earlier method; misses intra-bar spikes, has bad prints)")
    args = ap.parse_args()
    cfg = json.load(open(_REPO_ROOT / "config" / "configuration.json", encoding="utf-8"))
    if args.scale is None:
        args.scale = float(cfg["intraday_sizing"]["book_size_multiplier"])
    if args.max_notional is None:
        args.max_notional = (float(cfg["intraday_sizing"]["max_notional_pct_of_capital"])
                             * float(cfg["capital_management"]["paper_initial_capital"]))

    days = load_trades(Path(args.trades))
    print("sessions with trades: %d  (%s .. %s)" % (len(days), min(days), max(days)))
    n_tr = sum(len(v) for v in days.values())
    n_multi = sum(1 for v in days.values() for t in v if len(t["legs"]) > 1)
    print("trades: %d  (%d with a partial exit, carried at FULL size until the partial)\n"
          % (n_tr, n_multi))

    by_month = collections.defaultdict(list)
    for d in days:
        by_month[d[:7].replace("-", "_")].append(d)

    recs, no_bars = [], 0
    suffix = "_1m.feather" if args.bars == "1m" else "_5m_enriched.feather"
    for month in sorted(by_month):
        f = MONTHLY / (month + suffix)
        if not f.exists():
            no_bars += sum(len(days[d]) for d in by_month[month])
            continue
        need = {r["sym"] for d in by_month[month] for r in days[d]}
        cols = ["symbol", "close"] + (["ts"] if args.bars == "1m" else ["date"])
        df = pd.read_feather(f, columns=cols)
        df = df[df["symbol"].isin(need)]
        if args.bars != "1m":
            df = df.rename(columns={"date": "ts"})
            df["ts"] = pd.to_datetime(df["ts"])
            if getattr(df["ts"].dt, "tz", None) is not None:
                df["ts"] = df["ts"].dt.tz_localize(None)
        df["d"] = df["ts"].dt.strftime("%Y-%m-%d")
        for day in sorted(by_month[month]):
            rows = days[day]
            sub = df[df["d"] == day]
            if sub.empty:
                no_bars += len(rows)
                continue
            for r in rows:
                scale_trade(r, args.scale, args.max_notional)
            out = day_curves(rows, sub)
            if out is None:
                no_bars += len(rows)
                continue
            gross, net, used = out
            no_bars += len(rows) - used
            recs.append(dict(day=day, n=len(rows),
                             fees=sum(r["fees"] for r in rows),
                             peak=float(gross.max()), ptime=gross.idxmax(),
                             realised=sum(r["gross"] - r["fees"] for r in rows),
                             curve=[float(v) for v in gross.values]))
        print("  %s: %d sessions done" % (month, len(by_month[month])), flush=True)

    print("\nreconstructed %d sessions | %d trades without bars\n" % (len(recs), no_bars))
    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(dict(r, ptime=str(r["ptime"]))) + "\n")
        print("  per-session records -> %s\n" % args.dump)
    base = sum(r["realised"] for r in recs)
    print("  baseline realised (net): Rs%s over %d sessions" % (format(base, "+,.0f"), len(recs)))
    print("  red days: %.0f%%\n" % (100 * sum(1 for r in recs if r["realised"] < 0) / len(recs)))

    scaled = [r for v in days.values() for r in v if "qty" in r]
    med1x = sorted(r["raw_qty"] * r["ep"] for r in scaled)[len(scaled) // 2]
    medlv = sorted(r["qty"] * r["ep"] for r in scaled)[len(scaled) // 2]
    at_cap = sum(1 for r in scaled if r["raw_qty"] * r["ep"] * args.scale >= args.max_notional)
    print("  sizing: x%.1f then clamp Rs%s | median notional 1x Rs%s -> live Rs%s (%.1fx) | "
          "%d of %d trades at the clamp" % (
              args.scale, format(args.max_notional, ",.0f"), format(med1x, ",.0f"),
              format(medlv, ",.0f"), medlv / med1x, at_cap, len(scaled)))
    print("  thresholds in TODAY's rupees; rule fires on the GROSS %s-close curve; a firing "
          "day books T minus its fees" % args.bars)
    rules = ["target", "loss"] if args.rule == "both" else [args.rule]

    def book(r, t, sign):
        hit = any((v >= t) if sign > 0 else (v <= t) for v in r["curve"])
        return (t - r["fees"] if hit else r["realised"]), hit

    for rule in rules:
        sign = 1.0 if rule == "target" else -1.0
        print("\n  === RULE: %s ===" % (
            "PROFIT TARGET - stop for the day at +T" if rule == "target"
            else "LOSS CAP - stop for the day at -T"))
        print("  %-14s %6s %14s %13s %8s %9s" % (
            "threshold", "fires", "book", "delta", "red", "worst day"))
        for lt in args.targets:
            t = sign * lt
            tot = fires = red = 0
            worst = 0.0
            for r in recs:
                val, hit = book(r, t, sign)
                fires += hit
                tot += val
                red += (val < 0)
                worst = min(worst, val)
            print("  Rs%-12s %6d %14s %13s %7.0f%% %9s" % (
                format(int(lt), ","), fires, format(tot, "+,.0f"),
                format(tot - base, "+,.0f"), 100 * red / len(recs), format(worst, "+,.0f")))
        base_worst = min(r["realised"] for r in recs)
        print("  %-14s %6s %14s %13s %7.0f%% %9s" % (
            "none", "-", format(base, "+,.0f"), "+0",
            100 * sum(1 for r in recs if r["realised"] < 0) / len(recs),
            format(base_worst, "+,.0f")))

        best = max(args.targets, key=lambda lt: sum(book(r, sign * lt, sign)[0] for r in recs))
        t = sign * best
        print("\n  best %s Rs%.0f - yearly stability:" % (rule, best))
        for yr in sorted({r["day"][:4] for r in recs}):
            sub = [r for r in recs if r["day"][:4] == yr]
            b = sum(r["realised"] for r in sub)
            t2 = sum(book(r, t, sign)[0] for r in sub)
            print("    %s  n=%-4d baseline %14s -> rule %14s  delta %13s" % (
                yr, len(sub), format(b, "+,.0f"), format(t2, "+,.0f"), format(t2 - b, "+,.0f")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
