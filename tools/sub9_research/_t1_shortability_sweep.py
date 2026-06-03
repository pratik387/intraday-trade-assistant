"""T1 (up_spike_fade_short) shortability sweep.

Authoritative SELL-MIS eligibility via Kite order_margins, applied to the unique
T1 signal symbols (illiquid up-spike fade, realistic next-bar-open entry), then
re-measure the OOS/HO net edge on only the genuinely-shortable subset.

Run with: KITE_API_KEY=<key> python tools/sub9_research/_t1_shortability_sweep.py
"""
import logging
logging.disable(logging.WARNING)  # silence broker INFO/WARNING per-symbol noise
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("E:/Codebase/intraday-trade-assistant")
sys.path.insert(0, str(BASE))  # repo root on path so `broker` imports when run as a file
SIGF = BASE / "reports/sub9_research/_t1_signals.feather"
ILLIQ = {"small_cap", "micro_cap", "unknown"}
OOS = [f"2025_{m:02d}" for m in range(1, 10)]
HO = [f"2025_{m:02d}" for m in (10, 11, 12)] + [f"2026_{m:02d}" for m in (1, 2, 3, 4)]


def compute_signals():
    caps = json.load(open(BASE / "data/cap_segments/cap_segments_latest.json"))
    rows = []
    for W, tags in [("OOS", OOS), ("HO", HO)]:
        for t in tags:
            f = BASE / f"backtest-cache-download/monthly/{t}_5m_enriched.feather"
            if not f.exists():
                continue
            d = pd.read_feather(f, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
            d = d[d["symbol"].map(lambda s: caps.get(s, "unknown")).isin(ILLIQ)]
            if d.empty:
                continue
            d["date"] = pd.to_datetime(d["date"])
            d["day"] = d["date"].dt.normalize()
            d["t"] = d["date"].dt.strftime("%H:%M")
            d["turn"] = d.close * d.volume
            d = d.sort_values(["symbol", "date"])
            k = ["symbol", "day"]
            gk = d.groupby(k)
            d["c3"] = gk["close"].shift(3)
            d["r3"] = d.close / d.c3 - 1
            cs = gk["volume"].cumsum()
            d["nbar"] = gk.cumcount()
            d["tav"] = (cs - d["volume"]) / d["nbar"].replace(0, np.nan)
            d["eod"] = gk["close"].transform("last")
            d["dturn"] = gk["turn"].transform("sum")
            d["nxt_open"] = gk["open"].shift(-1)
            liq = d.dturn >= 1.5e7
            win = (d.t >= "09:45") & (d.t <= "14:00")
            brk = (d.volume >= 2 * d.tav) & (d.nbar >= 6)
            up = d[liq & win & brk & (d.r3 >= 0.04)].drop_duplicates(k, keep="first").dropna(subset=["nxt_open"])
            rows += list(zip(up.symbol, [W] * len(up), up.date.dt.strftime("%Y-%m"),
                             (-(up.eod / up.nxt_open - 1)).values))
            print(f"  {t}: cum {len(rows)}", flush=True)
    sig = pd.DataFrame(rows, columns=["sym", "W", "month", "ret"])
    sig.to_feather(SIGF)
    return sig


def main():
    if SIGF.exists():
        sig = pd.read_feather(SIGF)
        print(f"loaded cached signals: {len(sig)}", flush=True)
    else:
        print("computing signals...", flush=True)
        sig = compute_signals()
    syms = sorted(sig.sym.unique())
    print(f"unique T1 symbols: {len(syms)}", flush=True)

    from broker.kite.kite_broker import KiteBroker
    b = KiteBroker(dry_run=True)
    # live LTPs in batches (needed for order_margins leverage field)
    nsyms = ["NSE:" + s for s in syms]
    ltp = {}
    for j in range(0, len(nsyms), 400):
        chunk = nsyms[j:j + 400]
        try:
            r = b.kc.ltp(chunk)
            for kk, vv in r.items():
                ltp[kk.split(":")[-1]] = vv.get("last_price")
        except Exception as e:
            print(f"  ltp chunk {j} err {e}", flush=True)
        time.sleep(0.3)
    print(f"  got LTP for {sum(1 for s in syms if ltp.get(s))}/{len(syms)}", flush=True)

    # leverage field = real MIS gate (>1 => intraday leverage granted; 1 => 100% margin/surveillance; 0 => invalid/delisted)
    lev = {}
    for i, s in enumerate(syms):
        px = ltp.get(s)
        try:
            m = b.kc.order_margins([{"exchange": "NSE", "tradingsymbol": s, "transaction_type": "SELL",
                                     "variety": "regular", "product": "MIS", "order_type": "MARKET",
                                     "quantity": 1, "price": float(px or 0)}])
            lev[s] = (m[0].get("leverage") if m else None)
        except Exception:
            lev[s] = None
        time.sleep(0.12)
        if (i + 1) % 100 == 0:
            ok = sum(1 for v in lev.values() if v and v > 1)
            print(f"  {i+1}/{len(syms)} checked, lev>1 so far {ok}", flush=True)

    short_ok = {s: bool(v and v > 1) for s, v in lev.items()}
    pd.Series(lev).to_json(BASE / "reports/sub9_research/t1_short_mis_eligibility.json")
    sig["short_ok"] = sig.sym.map(short_ok)
    nshort = sum(short_ok.values())
    # leverage breakdown
    from collections import Counter
    levc = Counter(round(v) if v is not None else "none" for v in lev.values())
    print(f"\nleverage distribution across {len(syms)} symbols: {dict(levc)}")
    print(f"SHORTABLE (leverage>1): {nshort}/{len(syms)} symbols = {nshort/len(syms)*100:.0f}%")
    print(f"signal-weighted capturable: {sig.short_ok.mean()*100:.0f}% of trades\n")
    for W in ["OOS", "HO"]:
        g = sig[sig.W == W]
        for lab, sub in [("ALL (research)", g), ("SHORTABLE only", g[g.short_ok]), ("non-shortable", g[~g.short_ok])]:
            x = sub.ret.values * 100
            if len(x) == 0:
                print(f"{W} {lab}: n=0")
                continue
            mm = sub.groupby("month").ret.mean() * 100
            print(f"{W:4} {lab:16} n={len(x):4d} net@0.30%={x.mean()-0.30:+.3f}% "
                  f"hit={(x>0).mean()*100:.0f}% +mo={int((mm>0).sum())}/{len(mm)}")
        print()
    print("saved -> reports/sub9_research/t1_short_mis_eligibility.json")


if __name__ == "__main__":
    main()
