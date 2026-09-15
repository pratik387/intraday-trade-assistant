"""Day-of-week x setup on the backtest legs (per-trade return %, size-free) and on
the per-(day, setup) curves at live size. A pattern is only reported as a
pattern if it holds in each year separately."""
import json, sys, collections, statistics as st, datetime as dt
S = sys.argv[1]
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri"]

by = collections.defaultdict(list)
for ln in open(S + "/bt_active_legs.jsonl", encoding="utf-8"):
    t = json.loads(ln)
    by[(t["session"], t["symbol"], t["actual_entry_price"], t["setup_type"])].append(t)
trades = []
for (d, sym, ep, su), legs in by.items():
    q = sum(int(l["qty"]) for l in legs)
    gross = sum(float(l.get("pnl") or 0) for l in legs)
    net = sum(float(l.get("net_pnl") or 0) for l in legs)
    trades.append(dict(day=d, setup=su, ret=100 * gross / (float(ep) * q), win=gross > 0,
                       dow=dt.date.fromisoformat(d).weekday(), yr=d[:4], net=net))

def se(xs):
    return (st.pstdev(xs) / len(xs) ** 0.5) if len(xs) > 1 else float("nan")

print("=== BACKTEST per-trade return %% by weekday (all setups pooled) ===")
print("  %-4s %6s %9s %8s %6s %8s" % ("dow", "n", "mean ret", "t", "win%", "median"))
for w in range(5):
    xs = [t["ret"] for t in trades if t["dow"] == w]
    print("  %-4s %6d %+8.3f%% %8.2f %5.0f%% %+7.3f%%" % (
        DOW[w], len(xs), st.mean(xs), st.mean(xs) / se(xs), 100 * st.mean(t["win"] for t in trades if t["dow"] == w), st.median(xs)))

print("\n=== per setup x weekday: mean ret %% (n) | then per-year sign check ===")
for su in sorted({t["setup"] for t in trades}):
    ts = [t for t in trades if t["setup"] == su]
    print("\n  %s  n=%d  overall %+.3f%%" % (su, len(ts), st.mean(t["ret"] for t in ts)))
    print("    %-4s %5s %9s %6s %6s | %s" % ("dow", "n", "mean", "t", "win%", "by year (mean ret %, n)"))
    for w in range(5):
        xs = [t for t in ts if t["dow"] == w]
        if len(xs) < 10:
            continue
        r = [t["ret"] for t in xs]
        yrs = []
        for yr in ("2023", "2024", "2025", "2026"):
            ys = [t["ret"] for t in xs if t["yr"] == yr]
            yrs.append("%s %+.2f (%d)" % (yr, st.mean(ys), len(ys)) if ys else "%s   -" % yr)
        print("    %-4s %5d %+8.3f%% %6.2f %5.0f%% | %s" % (
            DOW[w], len(xs), st.mean(r), st.mean(r) / se(r), 100 * st.mean(t["win"] for t in xs), "  ".join(yrs)))

# live
lv = collections.defaultdict(list)
import glob
for f in glob.glob(S + "/oci_an/*.jsonl") + glob.glob(S + "/vm_sessions/intraday_fixed/intraday-trade-assistant/logs/paper_2026*/analytics.jsonl"):
    for ln in open(f, encoding="utf-8", errors="replace"):
        try:
            a = json.loads(ln)
        except Exception:
            continue
        if a.get("actual_entry_price") is None or not a.get("qty") or a.get("pnl") is None:
            continue
        d = str(a["timestamp"])[:10]
        lv[(d, a["symbol"], a["actual_entry_price"], a.get("setup_type"))].append(a)
lt = []
for (d, sym, ep, su), legs in lv.items():
    q = sum(int(l["qty"]) for l in legs); gross = sum(float(l["pnl"]) for l in legs)
    lt.append(dict(day=d, setup=su, ret=100 * gross / (float(ep) * q), dow=dt.date.fromisoformat(d).weekday(), gross=gross))
print("\n=== LIVE/PAPER 2026-06-01.. per-trade return %% by weekday (all setups) ===")
for w in range(5):
    xs = [t for t in lt if t["dow"] == w]
    post = [t for t in xs if t["day"] >= "2026-08-14"]
    if xs:
        print("  %-4s n=%-4d mean %+.3f%%  win %.0f%%  | since 08-14: n=%d mean %+.3f%% gross %s" % (
            DOW[w], len(xs), st.mean(t["ret"] for t in xs), 100 * st.mean(t["ret"] > 0 for t in xs),
            len(post), st.mean(t["ret"] for t in post) if post else float("nan"),
            format(sum(t["gross"] for t in post), "+,.0f")))
print("\n  live up_spike by weekday since 08-14:")
for w in range(5):
    xs = [t for t in lt if t["dow"] == w and t["setup"] == "up_spike_fade_short" and t["day"] >= "2026-08-14"]
    if xs:
        print("    %-4s n=%-3d mean %+.2f%%  %s" % (DOW[w], len(xs), st.mean(t["ret"] for t in xs), " ".join("%+.1f" % t["ret"] for t in xs)))

# session-level at live size from the dump
print("\n=== BACKTEST session net at live size by weekday (book) ===")
recs = [json.loads(l) for l in open(S + "/bt_clamp_1m_days.jsonl", encoding="utf-8")]
for w in range(5):
    xs = [r["realised"] for r in recs if dt.date.fromisoformat(r["day"]).weekday() == w]
    print("  %-4s n=%-4d mean %s  median %s  red %.0f%%" % (
        DOW[w], len(xs), format(st.mean(xs), "+,.0f"), format(st.median(xs), "+,.0f"), 100 * sum(1 for x in xs if x < 0) / len(xs)))
