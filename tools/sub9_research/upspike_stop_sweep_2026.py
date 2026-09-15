"""Stop sweep for up_spike_fade_short on the population where the adverse move
tripled: 2026 backtest trades (Jan-Jul, 1m feathers) and the paper trades since
2026-08-14 (Upstox 1m). 2023-25 is run too, as the cost in the old regime.

Phase 5 rejected every stop/target variant on a population whose median MAE was
-1.65%. Median MAE is -1.92% in 2026 H1, -3.23% in Jul-2026, -5.4% in Aug-Sep.

Variants, applied to the REMAINING quantity on top of the actual exit legs:
  fixed  s%  : short is stopped when the 1m HIGH >= entry*(1+s); filled at the stop
               level +slip bp (illiquid names gap through stops; slip is a knob)
  trail  a/g : once the trade has been >= a% in profit (1m LOW), exit at the 1m
               CLOSE if profit falls to <= best_profit - g pp
  time   hh:mm: exit remaining at that bar's close
Baseline = the trade as it happened (9% catastrophe stop, 15:10 time stop).

Sized like live: Rs300k notional per trade (0.06 x 5L x 10), fees per leg.

Usage:
    python tools/sub9_research/upspike_stop_sweep_2026.py <scratch dir>  [--slip-bp 20]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from tools.intraday_daily_target_backtest import intraday_fees, flagged_symbol_days  # noqa: E402

SETUP = "up_spike_fade_short"
NOTIONAL = 300_000.0
MONTHLY = _REPO_ROOT / "backtest-cache-download" / "monthly"


def load_trades(S: str) -> list:
    bad = flagged_symbol_days()
    by = collections.defaultdict(list)
    for ln in open(S + "/bt_active_legs.jsonl", encoding="utf-8"):
        t = json.loads(ln)
        if t["setup_type"] != SETUP or (t["symbol"].replace("NSE:", ""), t["session"]) in bad:
            continue
        by[(t["session"], t["symbol"], t["actual_entry_price"])].append(t)
    for f in glob.glob(S + "/oci_an/*.jsonl") + glob.glob(S + "/vm_sessions/intraday_fixed/intraday-trade-assistant/logs/paper_2026*/analytics.jsonl"):
        for ln in open(f, encoding="utf-8", errors="replace"):
            try:
                a = json.loads(ln)
            except Exception:
                continue
            d = str(a["timestamp"])[:10]
            if a.get("setup_type") != SETUP or d < "2026-08-14" or a.get("pnl") is None:
                continue
            a = dict(a, session=d, paper=True)
            by[(d, a["symbol"], a["actual_entry_price"])].append(a)
    out = []
    for (d, sym, ep), legs in by.items():
        ep = float(ep)
        legs = sorted(legs, key=lambda l: str(l["timestamp"]))
        q1 = sum(int(l["qty"]) for l in legs)
        fin = legs[-1]
        end = pd.Timestamp(str(fin["timestamp"]))
        start = end - pd.Timedelta(minutes=float(fin.get("time_in_trade_minutes") or 0))
        era = "paper_aug_sep" if legs[0].get("paper") else ("bt_2026H1" if d >= "2026-01-01" else "bt_2023_25")
        out.append(dict(day=d, sym=sym.replace("NSE:", ""), ep=ep, start=start, era=era,
                        legs=[(pd.Timestamp(str(l["timestamp"])), int(l["qty"]) / q1, float(l.get("exit_price") or ep)) for l in legs]))
    return out


def bars_for(trades: list, sdk) -> dict:
    """(day, sym) -> 1m DataFrame indexed by ts with high/low/close."""
    out = {}
    by_month = collections.defaultdict(set)
    for t in trades:
        if t["era"] == "paper_aug_sep":
            continue
        by_month[t["day"][:7].replace("-", "_")].add(t["sym"])
    for m, syms in sorted(by_month.items()):
        f = MONTHLY / (m + "_1m.feather")
        if not f.exists():
            continue
        df = pd.read_feather(f, columns=["ts", "symbol", "high", "low", "close"])
        df = df[df["symbol"].isin(syms)]
        df["d"] = df["ts"].dt.strftime("%Y-%m-%d")
        for (d, s), g in df.groupby(["d", "symbol"]):
            out[(d, s)] = g.set_index("ts")[["high", "low", "close"]].sort_index()
        print("  bars %s: %d symbol-days" % (m, len(syms)), flush=True)
    for t in trades:
        if t["era"] != "paper_aug_sep":
            continue
        day = dt.date.fromisoformat(t["day"])
        try:
            b = sdk.get_historical_1m("NSE:" + t["sym"], dt.datetime.combine(day, dt.time(9, 15)),
                                      dt.datetime.combine(day, dt.time(15, 30)))
        except Exception:
            b = None
        if b is not None and len(b):
            out[(t["day"], t["sym"])] = b[["high", "low", "close"]]
    return out


def simulate(t: dict, b: pd.DataFrame, rule: tuple, slip_bp: float) -> float | None:
    """Return trade P&L % of entry notional (gross), applying `rule` to the
    quantity still open, on top of the actual legs. None if no bars."""
    ep = t["ep"]
    path = b[b.index >= t["start"]]
    if path.empty:
        return None
    kind = rule[0]
    stop_ts, stop_px = None, None
    if kind == "fixed":
        s = rule[1] / 100.0
        lvl = ep * (1 + s)
        hit = path[path["high"] >= lvl]
        if len(hit):
            stop_ts, stop_px = hit.index[0], lvl * (1 + slip_bp / 1e4)
    elif kind == "trail":
        arm, gb = rule[1] / 100.0, rule[2] / 100.0
        best = 0.0
        armed = False
        for ts_, r in path.iterrows():
            prof_low = (ep - r["low"]) / ep
            best = max(best, prof_low)
            if best >= arm:
                armed = True
            prof_close = (ep - r["close"]) / ep
            if armed and prof_close <= best - gb:
                stop_ts, stop_px = ts_, r["close"] * (1 + slip_bp / 1e4)
                break
    elif kind == "time":
        hh, mm = rule[1]
        cut = path[(path.index.hour > hh) | ((path.index.hour == hh) & (path.index.minute >= mm))]
        if len(cut):
            stop_ts, stop_px = cut.index[0], cut["close"].iloc[0]
    pnl, left = 0.0, 1.0
    for (lts, frac, xp) in t["legs"]:
        if stop_ts is not None and lts > stop_ts:
            break
        pnl += (ep - xp) / ep * frac
        left -= frac
    if left > 1e-9:
        if stop_ts is None:
            # legs did not close it (should not happen) - mark at last close
            stop_px = path["close"].iloc[-1]
        pnl += (ep - stop_px) / ep * left
    return 100 * pnl


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("--slip-bp", type=float, default=20.0,
                    help="adverse fill vs stop level, bp. Thin names gap through stops.")
    args = ap.parse_args()
    trades = load_trades(args.scratch)
    print("trades: %s" % dict(collections.Counter(t["era"] for t in trades)))
    from broker.upstox.upstox_data_client import UpstoxDataClient
    sdk = UpstoxDataClient()
    bars = bars_for(trades, sdk)
    trades = [t for t in trades if (t["day"], t["sym"]) in bars]
    print("with bars: %s\n" % dict(collections.Counter(t["era"] for t in trades)))

    rules = [("baseline",)] + [("fixed", s) for s in (2.0, 3.0, 4.0, 5.0, 6.0, 7.0)] + \
            [("trail", a, g) for a in (1.0, 2.0) for g in (1.0, 1.5, 2.0, 3.0)] + \
            [("time", (14, 0)), ("time", (14, 30))]

    def label(r):
        return {"baseline": "baseline (9% cat, 15:10)", "fixed": "fixed stop %.0f%%" % r[1] if r[0] == "fixed" else "",
                "trail": "trail arm %.0f%% give-back %.1fpp" % (r[1], r[2]) if r[0] == "trail" else "",
                "time": "time exit %02d:%02d" % r[1] if r[0] == "time" else ""}[r[0]]

    results = {}
    for r in rules:
        per = collections.defaultdict(list)
        for t in trades:
            b = bars[(t["day"], t["sym"])]
            p = simulate(t, b, ("baseline",), 0.0) if r[0] == "baseline" else simulate(t, b, r, args.slip_bp)
            if p is None:
                continue
            q = int(NOTIONAL // t["ep"])
            xp_eff = t["ep"] * (1 - p / 100)
            net = p / 100 * t["ep"] * q - intraday_fees(t["ep"], xp_eff, q)
            per[t["era"]].append(dict(ret=p, net=net))
        results[r] = per

    eras = ["bt_2023_25", "bt_2026H1", "paper_aug_sep"]
    base = results[("baseline",)]
    print("slip on stop fills: %.0f bp | Rs300k per trade\n" % args.slip_bp)
    print("%-34s" % "variant" + "".join(" | %-30s" % e for e in eras))
    print("%-34s" % "" + "".join(" | %8s %6s %6s %6s" % ("net", "PF", "mean%", "worst") for _ in eras))
    for r in rules:
        line = "%-34s" % label(r)
        for e in eras:
            xs = results[r].get(e, [])
            if not xs:
                line += " | %30s" % "-"
                continue
            gw = sum(x["net"] for x in xs if x["net"] > 0)
            gl = -sum(x["net"] for x in xs if x["net"] < 0)
            line += " | %8s %6.2f %+6.2f %6.1f" % (
                format(sum(x["net"] for x in xs) / 1000, "+,.0f") + "k", gw / gl if gl else 9.99,
                st.mean(x["ret"] for x in xs), min(x["ret"] for x in xs))
        print(line)
    print("\n  delta vs baseline (net, Rs k):")
    for r in rules[1:]:
        line = "  %-32s" % label(r)
        for e in eras:
            a = sum(x["net"] for x in results[r].get(e, []))
            b = sum(x["net"] for x in base.get(e, []))
            line += " | %+9.0fk %s" % ((a - b) / 1000, "better" if a > b else "worse ")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
