"""Should or_window_failure_fade_short lose its stop? Its 2026 break is entirely
in stop-hit trades (PF 0.74 -> 0.39) while EOD trades are unchanged (PF ~3.9),
and up_spike runs stop-less. This replays every trade on 1m paths.

Shipped rules: SL 0.3% above the sweep high (min 0.5%), T1 at 1R with 0%
partial, T2 at 2R full, time stop 15:10. The stop distance d is inferred per
stop-hit trade from its exit (fill slippage included); target = entry*(1-2d).
Trades that exited on target or at 15:10 never touched the stop, so widening
or removing it changes nothing for them - they are kept as traded.

Variants, applied to trades that WERE stopped:
  stop x1      : stop at entry*(1+d)       (should reproduce baseline)
  stop x2, x3  : stop at entry*(1+2d), (1+3d); target and time stop kept
  no SL, cat 9%: no stop below a 9% catastrophe level; target and time stop kept
  no SL, no T2 : hold to 15:10 with only the 9% catastrophe (pure fade, like up_spike)
Fills at the level +slip bp. Sized like live (x10, Rs500k clamp), fees per leg.

Result 2026-09-15 (slip 20bp): NO. The like-for-like reference is "stop x1
replay" (the replay is ~0.2pp/trade pessimistic vs as-traded because inferred
d already carries fill slippage). Against it, on the stopped trades: removing
the SL is +120k on 2023-25 but -37k on 2026 H1 (-49k with the target dropped
too); x2/x3 are +40k/+79k and +25k/+35k with worst-trade going -6% -> -12%.
Nothing moves 2026 H1 off PF ~0.72-0.79. Stopped trades lose -1.3% to -1.6%
mean under every rule: in 2026 the failed pierce keeps going rather than
reversing, so holding longer loses more. The exit is not the problem; the
entry population is. Pause stands.

Usage:
    python tools/sub9_research/or_window_stop_sweep.py <scratch dir> [--slip-bp 20]
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

SETUP = "or_window_failure_fade_short"
MULT, CAP, CAT = 10.0, 500_000.0, 0.09
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
            if a.get("setup_type") != SETUP or a.get("pnl") is None:
                continue
            d = str(a["timestamp"])[:10]
            by[(d, a["symbol"], a["actual_entry_price"])].append(dict(a, session=d, paper=True))
    out = []
    for (d, sym, ep), legs in by.items():
        ep = float(ep)
        legs = sorted(legs, key=lambda l: str(l["timestamp"]))
        fin = legs[-1]
        q1 = sum(int(l["qty"]) for l in legs)
        end = pd.Timestamp(str(fin["timestamp"]))
        start = end - pd.Timedelta(minutes=float(fin.get("time_in_trade_minutes") or 0))
        xp = float(fin.get("exit_price") or ep)
        eod = end.strftime("%H:%M") >= "15:05"
        kind = "eod" if eod else ("stop" if xp > ep else "target")
        q = q1 if legs[0].get("paper") else max(1, min(int(round(q1 * MULT)), int(CAP // ep)))
        era = "paper" if legs[0].get("paper") else ("bt_2026H1" if d >= "2026-01-01" else "bt_2023_25")
        out.append(dict(day=d, sym=sym.replace("NSE:", ""), ep=ep, start=start, end=end, xp=xp, q=q,
                        kind=kind, d=(xp - ep) / ep if kind == "stop" else None, era=era))
    return out


def bars_for(trades: list, sdk) -> dict:
    out = {}
    by_month = collections.defaultdict(set)
    for t in trades:
        if t["era"] != "paper":
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
        print("  bars %s" % m, flush=True)
    for t in trades:
        if t["era"] != "paper":
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


def replay(t: dict, b: pd.DataFrame, stop_mult: float | None, keep_target: bool, slip: float) -> float:
    """P&L % for a STOPPED trade under a new stop rule. stop_mult None = no SL (cat 9%)."""
    ep, d = t["ep"], t["d"]
    stop_lvl = ep * (1 + (stop_mult * d if stop_mult is not None else CAT))
    tgt_lvl = ep * (1 - 2 * d) if keep_target else None
    path = b[(b.index >= t["start"])]
    for ts_, r in path.iterrows():
        if ts_.strftime("%H:%M") >= "15:10":
            return 100 * (ep - r["close"]) / ep
        if r["high"] >= stop_lvl:
            return 100 * (ep - stop_lvl * (1 + slip / 1e4)) / ep
        if tgt_lvl is not None and r["low"] <= tgt_lvl:
            return 100 * (ep - tgt_lvl * (1 - slip / 1e4)) / ep
    return 100 * (ep - path["close"].iloc[-1]) / ep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("scratch")
    ap.add_argument("--slip-bp", type=float, default=20.0)
    args = ap.parse_args()
    trades = load_trades(args.scratch)
    from broker.upstox.upstox_data_client import UpstoxDataClient
    bars = bars_for(trades, UpstoxDataClient())
    trades = [t for t in trades if (t["day"], t["sym"]) in bars]
    print("trades with bars: %s | exit kinds: %s\n" % (
        dict(collections.Counter(t["era"] for t in trades)), dict(collections.Counter(t["kind"] for t in trades))))
    print("stop-hit share by era: " + ", ".join("%s %.0f%%" % (
        e, 100 * sum(1 for t in trades if t["era"] == e and t["kind"] == "stop") / max(1, sum(1 for t in trades if t["era"] == e)))
        for e in ("bt_2023_25", "bt_2026H1", "paper")))
    print("inferred stop distance d (stop-hit trades): median %.2f%%  p75 %.2f%%\n" % (
        st.median(100 * t["d"] for t in trades if t["kind"] == "stop"),
        sorted(100 * t["d"] for t in trades if t["kind"] == "stop")[int(0.75 * sum(1 for t in trades if t["kind"] == "stop"))]))

    variants = [("baseline (as traded)", None), ("stop x1 (replay check)", (1.0, True)),
                ("stop x2, target kept", (2.0, True)), ("stop x3, target kept", (3.0, True)),
                ("no SL (cat 9%), target kept", (None, True)), ("no SL (cat 9%), no target", (None, False))]
    eras = ["bt_2023_25", "bt_2026H1", "paper"]
    res = {}
    for lab, v in variants:
        per = collections.defaultdict(list)
        for t in trades:
            if v is None or t["kind"] != "stop":
                p = 100 * (t["ep"] - t["xp"]) / t["ep"]
            else:
                p = replay(t, bars[(t["day"], t["sym"])], v[0], v[1], args.slip_bp)
            xp_eff = t["ep"] * (1 - p / 100)
            net = p / 100 * t["ep"] * t["q"] - intraday_fees(t["ep"], xp_eff, t["q"])
            per[t["era"]].append(dict(ret=p, net=net, stopped=t["kind"] == "stop"))
        res[lab] = per
    print("%-30s" % "variant" + "".join(" | %-32s" % e for e in eras))
    print("%-30s" % "" + "".join(" | %8s %5s %6s %6s %5s" % ("net", "PF", "mean%", "worst", "win%") for _ in eras))
    for lab, _ in variants:
        line = "%-30s" % lab
        for e in eras:
            xs = res[lab].get(e, [])
            if not xs:
                line += " | %32s" % "-"; continue
            gw = sum(x["net"] for x in xs if x["net"] > 0); gl = -sum(x["net"] for x in xs if x["net"] < 0)
            line += " | %8s %5.2f %+6.2f %6.1f %4.0f%%" % (
                format(sum(x["net"] for x in xs) / 1000, "+,.0f") + "k", gw / gl if gl else 9.99,
                st.mean(x["ret"] for x in xs), min(x["ret"] for x in xs), 100 * sum(1 for x in xs if x["net"] > 0) / len(xs))
        print(line)
    print("\n  delta vs baseline (net):")
    base = res["baseline (as traded)"]
    for lab, _ in variants[1:]:
        line = "  %-28s" % lab
        for e in eras:
            a = sum(x["net"] for x in res[lab].get(e, [])); b = sum(x["net"] for x in base.get(e, []))
            line += " | %+9.0fk %s" % ((a - b) / 1000, "better" if a > b else "worse ")
        print(line)
    print("\n  the stopped trades only, by era (what the stop rule is actually deciding):")
    for lab, _ in variants:
        line = "  %-28s" % lab
        for e in eras:
            xs = [x for x in res[lab].get(e, []) if x["stopped"]]
            line += " | n=%-3d net %8s mean %+5.2f%%" % (len(xs), format(sum(x["net"] for x in xs) / 1000, "+,.0f") + "k", st.mean(x["ret"] for x in xs) if xs else 0)
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
