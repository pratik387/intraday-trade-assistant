"""Daily profit-target sweep on the July-2026 backtest runs (current setups, 2023-2026).

The paper/live version of this study (tools/intraday_daily_target_study.py) has
only ~56 sessions. This runs the identical reconstruction over the OCI backtest
runs, which carry the CURRENT setup roster across 857 sessions.

Two traps this had to avoid, both of which bit on the first attempt:

  1. Run selection. The `backtest-results` bucket holds runs from May and July
     2026. The May runs contain ONLY retired setups (gap_fade_short,
     capitulation_long_morning, circuit_t1_fade_short) — filtering them to
     currently-active setups keeps 0 of 105,104 rows. Only the 20260730/31 runs
     have the live roster. Listing the bucket by paging objects also silently
     truncates (61k objects, and the July prefixes sort last), so prefixes must
     be listed with delimiter="/".

  2. P&L fields. `total_trade_pnl` is GROSS — it reconciles with
     (entry-exit)*qty on 90% of trades — and `net_pnl` is after costs.
     `gross_pnl` is absent in these runs. Day fees are therefore
     sum(total_trade_pnl - net_pnl), and the curve is gross MTM minus that.

Curve construction is the same arithmetic as the live study: at each 5m stamp a
position contributes 0 before entry, (entry-close)*qty while held for a short
(sign flipped for a long), and its realised P&L after exit. Bars come from the
local monthly 5m feathers rather than Upstox.

Bar CLOSES are used, so a spike that round-trips inside one 5m bar is invisible.
That biases the measured peak DOWN, never up.

Usage:
    python tools/intraday_daily_target_backtest.py
    python tools/intraday_daily_target_backtest.py --targets 2000 3000 5000
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
    days = collections.defaultdict(list)
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        t = json.loads(ln)
        ep, q, ts = t.get("actual_entry_price"), t.get("qty"), t.get("timestamp")
        if ep is None or not q or not ts:
            continue
        end = pd.to_datetime(str(ts))
        gross = t.get("total_trade_pnl")
        net = t.get("net_pnl")
        days[t["session"]].append(dict(
            sym=str(t["symbol"]).replace("NSE:", ""), qty=int(q), ep=float(ep),
            raw_qty=int(q),
            xp=float(t.get("exit_price") or ep), end=end,
            start=end - pd.Timedelta(minutes=float(t.get("time_in_trade_minutes") or 0)),
            gross=(float(gross) if gross is not None else None),
            net=(float(net) if net is not None else None),
            short=("short" in str(t.get("bias", "")).lower()
                   or "short" in str(t.get("setup_type", "")).lower())))
    return days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades", default=str(
        Path.home() / "AppData/Local/Temp/claude"
        / "E--Codebase-intraday-trade-assistant"
        / "d9c67968-368c-45aa-a25e-bd4d1cfb4906/scratchpad/bt_active_trades.jsonl"))
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[1000, 2000, 3000, 5000, 7500, 10000, 15000, 25000, 50000])
    ap.add_argument("--book-scale", dest="scale", type=float, default=9.92,
                    help="live book size / backtest book size. MEASURED: backtest "
                         "median notional Rs50,123 vs post-2026-08-14 live median "
                         "Rs497,198 = 9.92x. QUANTITIES are multiplied by this and "
                         "fees RECOMPUTED at the larger turnover - dividing the "
                         "threshold instead would be wrong, because the Rs20/order "
                         "brokerage cap makes fees strongly sublinear in size. "
                         "Pass 1.0 to evaluate the book as the backtest traded it.")
    args = ap.parse_args()

    days = load_trades(Path(args.trades))
    print("sessions with trades: %d  (%s .. %s)" % (
        len(days), min(days), max(days)))
    print("trades: %d\n" % sum(len(v) for v in days.values()))

    by_month = collections.defaultdict(list)
    for d in days:
        by_month[d[:7].replace("-", "_")].append(d)

    recs, no_bars = [], 0
    for month in sorted(by_month):
        f = MONTHLY / ("%s_5m_enriched.feather" % month)
        if not f.exists():
            no_bars += sum(len(days[d]) for d in by_month[month])
            continue
        need = {r["sym"] for d in by_month[month] for r in days[d]}
        df = pd.read_feather(f, columns=["date", "symbol", "close"])
        df = df[df["symbol"].isin(need)]
        df["d"] = df["date"].dt.strftime("%Y-%m-%d")
        for day in sorted(by_month[month]):
            rows = days[day]
            sub = df[df["d"] == day]
            if sub.empty:
                no_bars += len(rows)
                continue
            for r in rows:
                r["qty"] = max(1, int(round(r["raw_qty"] * args.scale)))
            grid, used = None, []
            for i, r in enumerate(rows):
                b = sub[sub["symbol"] == r["sym"]]
                if b.empty:
                    continue
                b = b.set_index("date")["close"].astype(float).sort_index()
                s = ((r["ep"] - b) * r["qty"]) if r["short"] else ((b - r["ep"]) * r["qty"])
                s[b.index < r["start"]] = 0.0
                realised = ((r["ep"] - r["xp"]) if r["short"] else (r["xp"] - r["ep"])) * r["qty"]
                # Charge each trade's round-trip cost AT ITS EXIT. Charging the
                # whole day's fees from the first bar (the earlier convention)
                # understated early-session peaks by ~Rs221/session at backtest
                # scale (~Rs2,200 at 9.92x), which biases a profit target to
                # look BETTER than it is — fewer crossings detected than real.
                s[b.index > r["end"]] = realised - intraday_fees(r["ep"], r["xp"], r["qty"])
                s = s.rename("p%d" % i)
                grid = s.to_frame() if grid is None else grid.join(s, how="outer")
                used.append(r)
            if grid is None:
                no_bars += len(rows)
                continue
            no_bars += len(rows) - len(used)
            cur = grid.ffill().fillna(0.0).sum(axis=1)
            recs.append(dict(day=day, n=len(rows),
                             peak=float(cur.max()), ptime=cur.idxmax(),
                             realised=sum(
                                 (((r["ep"] - r["xp"]) if r["short"] else (r["xp"] - r["ep"]))
                                  * r["qty"]) - intraday_fees(r["ep"], r["xp"], r["qty"])
                                 for r in rows),
                             curve=[float(v) for v in cur.values]))
        print("  %s: %d sessions done" % (month, len(by_month[month])), flush=True)

    print("\nreconstructed %d sessions | %d trades without bars\n" % (len(recs), no_bars))
    base = sum(r["realised"] for r in recs)
    print("  baseline realised: Rs%s over %d sessions" % (format(base, "+,.0f"), len(recs)))
    print("  red days: %.0f%%\n" % (100 * sum(1 for r in recs if r["realised"] < 0) / len(recs)))

    print("  book-size scale: %.2fx  (targets quoted in TODAY's rupees)" % args.scale)
    print()
    print("  %-14s %-13s %6s %14s %13s %8s" % (
        "target(today)", "=backtest", "fires", "book", "delta", "red"))
    for lt in args.targets:
        t = lt
        tot = fires = red = 0
        for r in recs:
            hit = any(v >= t for v in r["curve"])
            val = t if hit else r["realised"]
            fires += hit
            tot += val
            red += (val < 0)
        print("  Rs%-12s %-13s %6d %14s %13s %7.0f%%" % (
            format(int(lt), ","), "", fires, format(tot, "+,.0f"),
            format(tot - base, "+,.0f"), 100 * red / len(recs)))

    best = max(args.targets, key=lambda lt: sum(
        (lt if any(v >= lt for v in r["curve"]) else r["realised"]) for r in recs))
    print("\n  best target Rs%.0f — yearly stability:" % best)
    for yr in sorted({r["day"][:4] for r in recs}):
        sub = [r for r in recs if r["day"][:4] == yr]
        b = sum(r["realised"] for r in sub)
        t2 = sum((best if any(v >= best for v in r["curve"]) else r["realised"]) for r in sub)
        print("    %s  n=%-4d baseline %14s -> target %14s  delta %13s" % (
            yr, len(sub), format(b, "+,.0f"), format(t2, "+,.0f"), format(t2 - b, "+,.0f")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
