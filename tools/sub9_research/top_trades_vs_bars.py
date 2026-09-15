"""Are the top backtest trades real? Entry/exit vs the 1m feather's day range and
the previous day's close, per setup top-5 (+ all RMDRIP trades)."""
import json, sys, collections
import pandas as pd
S = sys.argv[1]
MONTHLY = "backtest-cache-download/monthly"

by = collections.defaultdict(list)
for ln in open(S + "/bt_active_legs.jsonl", encoding="utf-8"):
    t = json.loads(ln)
    by[(t["session"], t["symbol"], t["actual_entry_price"], t["setup_type"])].append(t)
trades = []
for (d, sym, ep, su), legs in by.items():
    q = sum(int(l["qty"]) for l in legs); g = sum(float(l.get("pnl") or 0) for l in legs)
    fin = max(legs, key=lambda l: l["timestamp"])
    trades.append(dict(day=d, sym=sym.replace("NSE:", ""), setup=su, ep=float(ep), ret=100 * g / (float(ep) * q),
                       xp=float(fin["exit_price"]), t_exit=fin["timestamp"][11:16],
                       tit=float(fin.get("time_in_trade_minutes") or 0), mfe=fin.get("mfe_pct"), mae=fin.get("mae_pct")))
want = []
for su in sorted({t["setup"] for t in trades}):
    ts = sorted([t for t in trades if t["setup"] == su], key=lambda t: -abs(t["ret"]))
    want += ts[:5]
want += [t for t in trades if t["sym"] == "RMDRIP" and t not in want]

cache = {}
def day_bars(sym, day):
    m = day[:7].replace("-", "_")
    if m not in cache:
        df = pd.read_feather("%s/%s_1m.feather" % (MONTHLY, m), columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
        cache[m] = df
    df = cache[m]
    return df[(df["symbol"] == sym) & (df["ts"].dt.strftime("%Y-%m-%d") == day)]

print("%-10s %-12s %-30s %7s %8s %8s | %8s %8s %8s %8s %9s | %s" % (
    "day", "sym", "setup", "ret%", "entry", "exit", "d.open", "d.high", "d.low", "d.close", "turnover", "flag"))
for t in sorted(want, key=lambda t: (t["setup"], t["day"])):
    b = day_bars(t["sym"], t["day"])
    if b.empty:
        print("%-10s %-12s %-30s %+6.1f%%  no 1m bars" % (t["day"], t["sym"], t["setup"], t["ret"])); continue
    o, h, l, c = b["open"].iloc[0], b["high"].max(), b["low"].min(), b["close"].iloc[-1]
    turn = (b["close"] * b["volume"]).sum()
    flag = ""
    if not (l * 0.995 <= t["ep"] <= h * 1.005) or not (l * 0.995 <= t["xp"] <= h * 1.005):
        flag += "PRICE_OUTSIDE_DAY_RANGE "
    if h / l > 1.4:
        flag += "RANGE>40%% "
    print("%-10s %-12s %-30s %+6.1f%% %8.2f %8.2f | %8.2f %8.2f %8.2f %8.2f %9s | %s" % (
        t["day"], t["sym"], t["setup"][:30], t["ret"], t["ep"], t["xp"], o, h, l, c, format(turn / 1e5, ",.0f") + "L", flag))
