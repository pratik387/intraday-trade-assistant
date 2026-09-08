"""Scan every 5m_enriched feather for symbol-days with implausible bars.

Trigger: RMDRIP 2026-04-08 prints Rs46-50 across 13 of 75 bars on a stock whose
other 62 bars sit at Rs26-30, one of them on 407 shares. The 1m replay the
backtest engine used never saw those prices (its MFE stayed at 0.24%), so the
contamination is in the 5m_enriched family specifically.

Flag a symbol-day when any bar's high or low is more than 40% away from that
day's MEDIAN close. Real moves that large on an intraday basis exist but are
rare; a median-anchored test is robust to the one-bar spike shape seen here.

First run, 2026-09-08, over all 43 monthly files (~1.2M symbol-days): 32
flagged. Three shapes in the output:
  - exact 2.0x (VBL 2023-06-14, ASMS 2023-02-09): unadjusted split
  - single-bar spike / crater (TBOTEK 2024-05-15 low 301 vs median 1373)
  - RMDRIP: EIGHT sessions 2026-03-20 .. 2026-04-09 at a consistent ~1.7x,
    which is not a stray tick but two instruments merged under one symbol.
The backtest engine replays the 1m family and did not see these (RMDRIP
2026-04-08 engine MFE 0.24% vs the 5m file's +73% print), so the 5m_enriched
family is contaminated where the 1m family is not.

Usage:
    python tools/scan_5m_feather_badprints.py
"""
import glob, os, sys
from pathlib import Path
import pandas as pd
_ROOT = Path(__file__).resolve().parents[1]
out = []
files = sorted(glob.glob(r"E:\Codebase\intraday-trade-assistant\backtest-cache-download\monthly\*_5m_enriched.feather"))
for f in files:
    mon = os.path.basename(f)[:7]
    try:
        df = pd.read_feather(f, columns=["date", "symbol", "high", "low", "close", "volume"])
    except Exception as e:
        print("  %s unreadable: %s" % (mon, e), flush=True); continue
    df["d"] = df["date"].dt.date
    g = df.groupby(["symbol", "d"], observed=True)
    med = g["close"].median().rename("med")
    hi = g["high"].max().rename("hi")
    lo = g["low"].min().rename("lo")
    n = g.size().rename("bars")
    s = pd.concat([med, hi, lo, n], axis=1)
    s = s[s["bars"] >= 20]
    s["hi_dev"] = s["hi"] / s["med"] - 1.0
    s["lo_dev"] = 1.0 - s["lo"] / s["med"]
    bad = s[(s["hi_dev"] > 0.40) | (s["lo_dev"] > 0.40)]
    for (sym, d), r in bad.iterrows():
        out.append(dict(month=mon, symbol=sym, day=str(d), med=r["med"], hi=r["hi"], lo=r["lo"],
                        hi_dev=r["hi_dev"], lo_dev=r["lo_dev"], bars=int(r["bars"])))
    print("  %s: %d symbol-days scanned, %d flagged" % (mon, len(s), len(bad)), flush=True)
res = pd.DataFrame(out)
outp = r"E:\Codebase\intraday-trade-assistant\reports\data_health\_5m_enriched_badprint_suspects.csv"
os.makedirs(os.path.dirname(outp), exist_ok=True)
res.to_csv(outp, index=False)
print("\nTOTAL flagged symbol-days: %d -> %s" % (len(res), outp))
if len(res):
    print("\nby month:"); print(res.groupby("month").size().to_string())
    print("\nworst 12 by deviation:")
    res["dev"] = res[["hi_dev", "lo_dev"]].max(axis=1)
    print(res.sort_values("dev", ascending=False).head(12)[["day", "symbol", "med", "hi", "lo", "dev", "bars"]].to_string(index=False))
    print("\nRMDRIP 2026-04-08 caught:", bool(((res["symbol"] == "RMDRIP") & (res["day"] == "2026-04-08")).any()))
