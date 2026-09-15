"""up_spike_fade_short: is the live post-peak give-back inside the backtest's distribution?

Written 2026-09-15 after the per-setup daily-target sweep put the whole live
give-back on this one setup. Compares, per trade, engine MFE/MAE, final return,
give-back (MFE - return) and exit clock between the OCI backtest legs and the
post-resize live sessions, then block-bootstraps 10 consecutive backtest
up_spike days at Rs300k/trade to place the live 10-day window.

Finding: exit timing is identical (94-95% EOD). MFE is identical. MAE is 3x
worse live (median -5.4% vs -1.65%; 53% of trades <= -5% vs 17%). The backtest
already shows MAE drifting from -1.06% (Jan 2026) to -3.23% (Jul 2026); the
live window continues that drift. The live 10-day total sits at the 8.5th
percentile of all backtest windows and the 0.9th percentile of 2026 windows.

Usage:
    python tools/sub9_research/upspike_giveback_parity.py <dir with bt_active_legs.jsonl, oci_an/, vm_sessions/>
"""

import json, sys, glob, collections, random, statistics as st
S = sys.argv[1]
SETUP = "up_spike_fade_short"

def trades_from(rows):
    by = collections.defaultdict(list)
    for a in rows:
        if a.get("setup_type") != SETUP or a.get("actual_entry_price") is None or not a.get("qty"):
            continue
        by[(str(a["timestamp"])[:10], a["symbol"], a["actual_entry_price"])].append(a)
    out = []
    for (d, sym, ep), legs in by.items():
        q = sum(int(l["qty"]) for l in legs)
        gross = sum(float(l.get("pnl") or 0) for l in legs)
        fin = max(legs, key=lambda l: str(l["timestamp"]))
        mfe = fin.get("mfe_pct"); mae = fin.get("mae_pct")
        if mfe is None:
            continue
        ret = 100 * gross / (float(ep) * q)
        out.append(dict(day=d, sym=sym, ep=float(ep), q=q, ret=ret, mfe=float(mfe), mae=float(mae or 0),
                        gb=float(mfe) - ret, exit_hm=str(fin["timestamp"])[11:16],
                        tit=float(fin.get("time_in_trade_minutes") or 0), gross=gross))
    return out

bt_rows = [json.loads(l) for l in open(S + "/bt_active_legs.jsonl", encoding="utf-8")]
for r in bt_rows:
    r["timestamp"] = r["timestamp"]
bt = trades_from(bt_rows)
lv_rows = []
for f in glob.glob(S + "/oci_an/*.jsonl") + glob.glob(S + "/vm_sessions/intraday_fixed/intraday-trade-assistant/logs/paper_2026*/analytics.jsonl"):
    for ln in open(f, encoding="utf-8", errors="replace"):
        try:
            a = json.loads(ln)
        except Exception:
            continue
        if str(a.get("timestamp"))[:10] >= "2026-08-14":
            lv_rows.append(a)
lv = trades_from(lv_rows)
bt26 = [t for t in bt if t["day"] >= "2026-01-01"]

def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))]

def summary(tag, ts):
    if not ts:
        return
    eod = sum(1 for t in ts if t["exit_hm"] >= "15:05")
    print("  %-14s n=%-4d ret%%: mean %+.2f med %+.2f | mfe%%: med %.2f | give-back(mfe-ret)pp: med %.2f p75 %.2f p90 %.2f | "
          "mae%% med %.2f | EOD exits %.0f%% | win %.0f%%" % (
              tag, len(ts), st.mean(t["ret"] for t in ts), st.median(t["ret"] for t in ts),
              st.median(t["mfe"] for t in ts), st.median(t["gb"] for t in ts), q([t["gb"] for t in ts], .75),
              q([t["gb"] for t in ts], .9), st.median(t["mae"] for t in ts), 100 * eod / len(ts),
              100 * sum(1 for t in ts if t["ret"] > 0) / len(ts)))

print("=== per-trade, %s ===" % SETUP)
summary("backtest all", bt); summary("backtest 2026", bt26); summary("LIVE >=08-14", lv)
print("\n  live trades:")
for t in sorted(lv, key=lambda t: t["day"]):
    print("    %s %-12s ret %+6.2f%%  mfe %5.2f%%  mae %6.2f%%  give-back %5.2f pp  exit %s  %3.0f min  gross %+8.0f" % (
        t["day"], t["sym"].replace("NSE:", ""), t["ret"], t["mfe"], t["mae"], t["gb"], t["exit_hm"], t["tit"], t["gross"]))

# trades that were up >=1% at some point (mfe>=1): how do they end?
print("\n=== conditional on mfe >= 1%%: how the trade ends ===")
for tag, ts in (("backtest all", bt), ("backtest 2026", bt26), ("LIVE", lv)):
    f = [t for t in ts if t["mfe"] >= 1.0]
    if f:
        print("  %-14s n=%-4d end ret mean %+.2f%% med %+.2f%%  ended negative %.0f%%  kept <25%% of mfe %.0f%%" % (
            tag, len(f), st.mean(t["ret"] for t in f), st.median(t["ret"] for t in f),
            100 * sum(1 for t in f if t["ret"] < 0) / len(f),
            100 * sum(1 for t in f if t["ret"] < 0.25 * t["mfe"]) / len(f)))

# block bootstrap: 10 consecutive up_spike days in the backtest, sized like live (Rs300k notional)
print("\n=== block bootstrap: 10 consecutive backtest up_spike days at Rs300k/trade vs live ===")
days = collections.defaultdict(float)
for t in bt:
    days[t["day"]] += t["ret"] / 100 * 300000
dl = sorted(days)
live_days = collections.defaultdict(float)
for t in lv:
    live_days[t["day"]] += t["ret"] / 100 * 300000
lv_tot = sum(live_days.values()); lv_red = sum(1 for v in live_days.values() if v < 0)
print("  live: %d days, gross total %s, red %d/%d" % (len(live_days), format(lv_tot, "+,.0f"), lv_red, len(live_days)))
wins = [(sum(days[d] for d in dl[i:i + 10]), sum(1 for d in dl[i:i + 10] if days[d] < 0)) for i in range(len(dl) - 9)]
print("  backtest windows: %d | window total: p5 %s  p25 %s  median %s | share of windows <= live total: %.1f%% | share with >=%d red days: %.1f%%" % (
    len(wins), format(q([w[0] for w in wins], .05), "+,.0f"), format(q([w[0] for w in wins], .25), "+,.0f"), format(q([w[0] for w in wins], .5), "+,.0f"),
    100 * sum(1 for w in wins if w[0] <= lv_tot) / len(wins), lv_red,
    100 * sum(1 for w in wins if w[1] >= lv_red) / len(wins)))
w26 = [w for i, w in enumerate(wins) if dl[i] >= "2026-01-01"]
if w26:
    print("  2026-only windows: %d | share <= live total: %.1f%%" % (len(w26), 100 * sum(1 for w in w26 if w[0] <= lv_tot) / len(w26)))

print("")
print("=== adverse excursion: share of trades with MAE <= -5%% ===")
for tag, ts in (("backtest all", bt), ("backtest 2026", bt26), ("LIVE", lv)):
    print("  %-14s %.0f%%  (MAE p25 %.2f  p10 %.2f)" % (tag, 100 * sum(1 for t in ts if t["mae"] <= -5) / len(ts), q([t["mae"] for t in ts], .25), q([t["mae"] for t in ts], .10)))
print("")
print("=== backtest by year: MAE median / give-back median / win%% ===")
for yr in ("2023", "2024", "2025", "2026"):
    ts = [t for t in bt if t["day"].startswith(yr)]
    print("  %s n=%-4d mae med %.2f  gb med %.2f  ret mean %+.2f  win %.0f%%  MAE<=-5%%: %.0f%%" % (
        yr, len(ts), st.median(t["mae"] for t in ts), st.median(t["gb"] for t in ts), st.mean(t["ret"] for t in ts),
        100 * sum(1 for t in ts if t["ret"] > 0) / len(ts), 100 * sum(1 for t in ts if t["mae"] <= -5) / len(ts)))
print("")
print("=== backtest 2026 by month ===")
for m in sorted({t["day"][:7] for t in bt26}):
    ts = [t for t in bt26 if t["day"].startswith(m)]
    print("  %s n=%-3d mae med %.2f  ret mean %+.2f  win %.0f%%  MAE<=-5%%: %.0f%%" % (
        m, len(ts), st.median(t["mae"] for t in ts), st.mean(t["ret"] for t in ts),
        100 * sum(1 for t in ts if t["ret"] > 0) / len(ts), 100 * sum(1 for t in ts if t["mae"] <= -5) / len(ts)))
