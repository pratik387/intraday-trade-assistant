"""Retirement test for or_window_failure_fade_short (Stage 14 of docs/setup_lifecycle.md).

Rule: retire when the PF interval crosses 1.0 on accumulated OCI data; decay
monitor pauses at rolling-6mo PF < 1.2 and retires at < 1.0. Also runs the
inverse-edge check (feedback_inverse_edge_signature) and breaks 2026 down so a
cell-level break can be told from a population-level one.

Trades come from the July-2026 OCI runs (all exit legs), sized like live
(x10 then Rs500k clamp, fees recomputed), scanner-flagged symbol-days dropped.

Usage:
    python tools/sub9_research/or_window_retire_test.py <scratch dir with bt_active_legs.jsonl>
"""
from __future__ import annotations

import collections
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
from tools.intraday_daily_target_backtest import intraday_fees, flagged_symbol_days  # noqa: E402

SETUP = "or_window_failure_fade_short"
MULT, CAP = 10.0, 500_000.0


def load(S: str) -> list:
    bad = flagged_symbol_days()
    by = collections.defaultdict(list)
    for ln in open(S + "/bt_active_legs.jsonl", encoding="utf-8"):
        t = json.loads(ln)
        if t["setup_type"] != SETUP or (t["symbol"].replace("NSE:", ""), t["session"]) in bad:
            continue
        by[(t["session"], t["symbol"], t["actual_entry_price"])].append(t)
    out = []
    for (d, sym, ep), legs in by.items():
        ep = float(ep)
        q1 = sum(int(l["qty"]) for l in legs)
        q = max(1, min(int(round(q1 * MULT)), int(CAP // ep)))
        net = 0.0
        gross = 0.0
        for l in legs:
            lq = int(round(int(l["qty"]) * (q / q1)))
            xp = float(l.get("exit_price") or ep)
            gross += (ep - xp) * lq
            net += (ep - xp) * lq - intraday_fees(ep, xp, lq)
        fin = max(legs, key=lambda l: l["timestamp"])
        out.append(dict(day=d, sym=sym, ep=ep, q=q, net=net, ret=100 * gross / (ep * q),
                        eod=fin["timestamp"][11:16] >= "15:05",
                        mfe=float(fin.get("mfe_pct") or 0), mae=float(fin.get("mae_pct") or 0),
                        tit=float(fin.get("time_in_trade_minutes") or 0)))
    return sorted(out, key=lambda t: t["day"])


def pf(ts):
    gw = sum(t["net"] for t in ts if t["net"] > 0)
    gl = -sum(t["net"] for t in ts if t["net"] < 0)
    return gw / gl if gl else float("inf")


def pf_ci(ts, n_boot=2000, seed=7):
    rnd = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        s = [rnd.choice(ts) for _ in ts]
        vals.append(pf(s))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.5 * n_boot)], vals[int(0.975 * n_boot)]


def main() -> int:
    S = sys.argv[1]
    ts = load(S)
    print("%s | %d trades 2023-01..2026-07 | net at live size %s | PF %.2f" % (
        SETUP, len(ts), format(sum(t["net"] for t in ts), "+,.0f"), pf(ts)))

    print("\n=== Stage-14 rule: PF interval on ACCUMULATED data (bootstrap 95%) ===")
    lo, md, hi = pf_ci(ts)
    print("  all 857 sessions : PF %.2f  [%.2f, %.2f]  -> %s" % (
        md, lo, hi, "crosses 1.0: RETIRE" if lo < 1.0 else "above 1.0: keep"))
    for yrs, lab in ((("2025", "2026"), "trailing ~19 months (2025-01..)"), (("2026",), "2026 only")):
        sub = [t for t in ts if t["day"][:4] in yrs]
        lo, md, hi = pf_ci(sub)
        print("  %-30s: n=%-4d PF %.2f  [%.2f, %.2f]  net %s" % (
            lab, len(sub), md, lo, hi, format(sum(t["net"] for t in sub), "+,.0f")))

    print("\n=== decay monitor: rolling 6-month PF (pause < 1.2, retire < 1.0) ===")
    months = sorted({t["day"][:7] for t in ts})
    for i in range(5, len(months)):
        win = months[i - 5:i + 1]
        sub = [t for t in ts if t["day"][:7] in win]
        p = pf(sub)
        flag = "RETIRE" if p < 1.0 else ("pause" if p < 1.2 else "")
        if months[i] >= "2025-01" or flag:
            print("  %s..%s  n=%-4d PF %.2f  net %10s  %s" % (
                win[0], win[-1], len(sub), p, format(sum(t["net"] for t in sub), "+,.0f"), flag))

    print("\n=== quarterly ===")
    q = collections.defaultdict(list)
    for t in ts:
        q[t["day"][:4] + "Q%d" % ((int(t["day"][5:7]) - 1) // 3 + 1)].append(t)
    for k in sorted(q):
        v = q[k]
        print("  %s n=%-4d net %10s PF %.2f win %.0f%%  mean ret %+.3f%%" % (
            k, len(v), format(sum(t["net"] for t in v), "+,.0f"), pf(v),
            100 * sum(1 for t in v if t["net"] > 0) / len(v), st.mean(t["ret"] for t in v)))

    print("\n=== inverse-edge check (feedback_inverse_edge_signature) ===")
    y26 = [t for t in ts if t["day"] >= "2026-01-01"]
    m26 = collections.defaultdict(float)
    for t in y26:
        m26[t["day"][:7]] += t["net"]
    print("  2026: n=%d PF %.2f winning months %d/%d  -> %s" % (
        len(y26), pf(y26), sum(1 for v in m26.values() if v > 0), len(m26),
        "INVERSE-EDGE signature (PF<<1, ~0 winning months on large n): test the flip"
        if pf(y26) < 0.7 and sum(1 for v in m26.values() if v > 0) <= 1 else
        "not inverse: mild PF, mixed months = decay / no edge, not a reversed edge"))
    flipped = [dict(t, net=-t["net"] - 2 * 0) for t in y26]
    print("  flipped 2026 gross return mean %+.3f%% (fees would still be paid)" % (-st.mean(t["ret"] for t in y26)))

    print("\n=== where did 2026 break? ===")
    print("  by exit type (EOD time-stop vs early stop/target):")
    for lab, key in (("EOD", True), ("early", False)):
        for yrs, tag in ((("2023", "2024", "2025"), "2023-25"), (("2026",), "2026")):
            sub = [t for t in ts if t["eod"] == key and t["day"][:4] in yrs]
            if sub:
                print("    %-5s %-8s n=%-4d share %3.0f%%  mean ret %+.3f%%  PF %.2f  mae med %.2f  mfe med %.2f" % (
                    lab, tag, len(sub), 100 * len(sub) / sum(1 for t in ts if t["day"][:4] in yrs),
                    st.mean(t["ret"] for t in sub), pf(sub), st.median(t["mae"] for t in sub),
                    st.median(t["mfe"] for t in sub)))
    print("  MFE / MAE drift by year (median %):")
    for yr in ("2023", "2024", "2025", "2026"):
        sub = [t for t in ts if t["day"][:4] == yr]
        print("    %s n=%-4d mfe %.2f  mae %.2f  ret mean %+.3f  win %.0f%%  mae<=-3%%: %.0f%%" % (
            yr, len(sub), st.median(t["mfe"] for t in sub), st.median(t["mae"] for t in sub),
            st.mean(t["ret"] for t in sub), 100 * sum(1 for t in sub if t["ret"] > 0) / len(sub),
            100 * sum(1 for t in sub if t["mae"] <= -3) / len(sub)))
    print("  2026 by month:")
    for m in sorted(m26):
        sub = [t for t in y26 if t["day"][:7] == m]
        print("    %s n=%-3d net %9s PF %.2f win %.0f%%" % (
            m, len(sub), format(m26[m], "+,.0f"), pf(sub), 100 * sum(1 for t in sub if t["net"] > 0) / len(sub)))

    # live/paper since June 2026
    lv = collections.defaultdict(list)
    for f in glob.glob(S + "/oci_an/*.jsonl") + glob.glob(S + "/vm_sessions/intraday_fixed/intraday-trade-assistant/logs/paper_2026*/analytics.jsonl"):
        for ln in open(f, encoding="utf-8", errors="replace"):
            try:
                a = json.loads(ln)
            except Exception:
                continue
            if a.get("setup_type") == SETUP and a.get("pnl") is not None and a.get("actual_entry_price"):
                lv[(str(a["timestamp"])[:10], a["symbol"], a["actual_entry_price"])].append(a)
    if lv:
        rows = []
        for (d, sym, ep), legs in lv.items():
            q = sum(int(l["qty"]) for l in legs)
            g = sum(float(l["pnl"]) for l in legs)
            rows.append(dict(day=d, ret=100 * g / (float(ep) * q), net=sum(float(l.get("net_pnl") or 0) for l in legs)))
        print("\n=== paper/live 2026-06.. (actual size) ===")
        print("  n=%d  net %s  PF %.2f  win %.0f%%  mean ret %+.3f%%" % (
            len(rows), format(sum(r["net"] for r in rows), "+,.0f"), pf(rows),
            100 * sum(1 for r in rows if r["net"] > 0) / len(rows), st.mean(r["ret"] for r in rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
