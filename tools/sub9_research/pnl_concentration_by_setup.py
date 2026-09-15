"""P&L concentration per setup: how much of the edge sits in the top few trades?

Per-trade rupee net at LIVE size (qty = min(10 x qty_1x, 500k // entry), fees
from the MIS model), plus size-free return %. Reports share of total net from
the top 1%/5%/10% of trades, and what is left after removing the top 5 / 10 /
top 5% winners. Then the same on the live post-resize trades.
"""
import json, sys, glob, collections, statistics as st
sys.path.insert(0, ".")
from tools.intraday_daily_target_backtest import intraday_fees
S = sys.argv[1]
MULT, CAP = 10.0, 500_000.0

def build(rows, live_size):
    by = collections.defaultdict(list)
    for t in rows:
        by[(str(t["timestamp"])[:10], t["symbol"], t["actual_entry_price"], t.get("setup_type"))].append(t)
    out = []
    for (d, sym, ep, su), legs in by.items():
        ep = float(ep); q1 = sum(int(l["qty"]) for l in legs)
        gross1 = sum(float(l.get("pnl") or 0) for l in legs)
        ret = gross1 / (ep * q1)
        if live_size:
            q = max(1, min(int(round(q1 * MULT)), int(CAP // ep)))
        else:
            q = q1
        # exit price implied per leg
        net = 0.0
        for l in legs:
            lq = int(round(int(l["qty"]) * (q / q1)))
            xp = float(l.get("exit_price") or ep)
            sgn = -1.0 if ("short" in str(l.get("bias", "")).lower() or "short" in str(su).lower()) else 1.0
            net += sgn * (xp - ep) * lq - intraday_fees(ep, xp, lq)
        out.append(dict(day=d, sym=sym, setup=su, ret=100 * ret, net=net, yr=d[:4]))
    return out

def report(tag, ts, top_ns=(1, 5, 10)):
    ts = sorted(ts, key=lambda t: -t["net"])
    n = len(ts); tot = sum(t["net"] for t in ts)
    wins = [t for t in ts if t["net"] > 0]
    gw = sum(t["net"] for t in wins); gl = -sum(t["net"] for t in ts if t["net"] < 0)
    pf = gw / gl if gl else float("inf")
    def share(k):
        return 100 * sum(t["net"] for t in ts[:k]) / tot if tot else float("nan")
    k1, k5, k10 = max(1, n // 100), max(1, n // 20), max(1, n // 10)
    line = "  %-38s n=%-5d net %12s  PF %.2f | top1%% (%d tr) %4.0f%%  top5%% (%d) %4.0f%%  top10%% (%d) %4.0f%% |" % (
        tag, n, format(tot, "+,.0f"), pf, k1, share(k1), k5, share(k5), k10, share(k10))
    for k in top_ns:
        rest = ts[k:]
        r_tot = sum(t["net"] for t in rest)
        r_gw = sum(t["net"] for t in rest if t["net"] > 0); r_gl = -sum(t["net"] for t in rest if t["net"] < 0)
        line += " -top%d: %s PF %.2f |" % (k, format(r_tot, "+,.0f"), (r_gw / r_gl) if r_gl else float("inf"))
    # without the top 5% of winners
    rest = ts[k5:]
    r_gw = sum(t["net"] for t in rest if t["net"] > 0); r_gl = -sum(t["net"] for t in rest if t["net"] < 0)
    line += " -top5%%: %s PF %.2f" % (format(sum(t["net"] for t in rest), "+,.0f"), (r_gw / r_gl) if r_gl else float("inf"))
    print(line)
    return ts

bt_rows = [json.loads(l) for l in open(S + "/bt_active_legs.jsonl", encoding="utf-8")]
bt = build(bt_rows, live_size=True)
print("=== BACKTEST 857 sessions, per-trade net at LIVE size (x10, Rs500k clamp) ===")
print("  (share = % of the setup's total net that its top-k winners contribute; -topN = what remains after deleting the N best trades)")
allts = report("ALL SETUPS", bt)
for su in sorted({t["setup"] for t in bt}):
    ts = [t for t in bt if t["setup"] == su]
    sts = report(su, ts)
    top = sts[:5]
    print("      top5: " + "; ".join("%s %s %s" % (t["day"], t["sym"].replace("NSE:", ""), format(t["net"], "+,.0f")) for t in top))
    # median trade and typical winner vs the top
    w = [t["net"] for t in ts if t["net"] > 0]
    print("      median winner %s  p90 winner %s  max %s  | median loser %s  worst %s" % (
        format(st.median(w), "+,.0f"), format(sorted(w)[int(.9 * len(w))], "+,.0f"), format(max(w), "+,.0f"),
        format(st.median([t["net"] for t in ts if t["net"] < 0]), "+,.0f"), format(min(t["net"] for t in ts), "+,.0f")))

print("\n=== per year: does each setup survive deleting its top 5% of trades? (net after removal, PF after removal) ===")
for su in sorted({t["setup"] for t in bt}):
    row = "  %-38s" % su
    for yr in ("2023", "2024", "2025", "2026"):
        ts = sorted([t for t in bt if t["setup"] == su and t["yr"] == yr], key=lambda t: -t["net"])
        if len(ts) < 10:
            row += " %s: -            " % yr; continue
        k = max(1, len(ts) // 20); rest = ts[k:]
        gw = sum(t["net"] for t in rest if t["net"] > 0); gl = -sum(t["net"] for t in rest if t["net"] < 0)
        row += " %s: %10s PF %.2f |" % (yr, format(sum(t["net"] for t in rest), "+,.0f"), gw / gl if gl else 9.99)
    print(row)

# live
lv_rows = []
for f in glob.glob(S + "/oci_an/*.jsonl") + glob.glob(S + "/vm_sessions/intraday_fixed/intraday-trade-assistant/logs/paper_2026*/analytics.jsonl"):
    for ln in open(f, encoding="utf-8", errors="replace"):
        try:
            a = json.loads(ln)
        except Exception:
            continue
        if a.get("actual_entry_price") is None or not a.get("qty") or a.get("pnl") is None:
            continue
        if str(a["timestamp"])[:10] >= "2026-08-14":
            lv_rows.append(a)
lv = build(lv_rows, live_size=False)
print("\n=== LIVE since 2026-08-14 (actual size) ===")
report("ALL SETUPS", lv, top_ns=(1, 3))
for su in sorted({t["setup"] for t in lv}):
    sts = report(su, [t for t in lv if t["setup"] == su], top_ns=(1, 3))
    print("      " + "; ".join("%s %s %s" % (t["day"][5:], t["sym"].replace("NSE:", ""), format(t["net"], "+,.0f")) for t in sts))
