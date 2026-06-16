"""Phase-2 signature (Stage 2, docs/setup_lifecycle.md) for the candidate-#5/#6
extension of the 2-3 day CNC/MTF reversion batch: C5 range_exhaustion / C6 crash2d.

DISCOVERY ONLY (2023-01-02..2024-12-31). Production-faithful universe:
MTF-eligible (excl ETF) x ADV>=20L x ADV-tier-1. Measures the RAW mechanism
footprint (no fees, no leverage) as signal_mean - tier-1-universe_baseline_mean
(market-relative) over a 2-3 day hold (T+1 open -> T+1+K close). Kill gate:
|delta| < 0.1% drift.

Distinct from the existing 4 triggers:
  A2 raw-5d-%, C1 252d-low level, C3 1d-decile (killed), C4 z-score(20d band).
  C5 keys on the INTRADAY RANGE (high-low)/prev_close + close<open (none do).
  C6 keys on 2-DAY cumulative return decile (between C3's 1d and A2's 5d).

NOT a ship test. No OOS/HO touched. Prints df.shape + per-trigger n + symbol
concentration (post-filter distribution + bad-print discipline, lessons #26 /
data-cleaning). Mirrors phase2_2to3day_reversion_signature.py exactly; SEPARATE
script (does not touch the shared batch script).
"""
from datetime import date
from pathlib import Path
import pandas as pd, numpy as np
from collections import Counter

REPO = Path(__file__).resolve().parents[2]
DD = REPO / "cache/preaggregate/clean_daily_from5m.feather"
MTF = REPO / "data/mtf_universe/approved_mtf_securities_2026-05-21.json"
DISC_START, DISC_END = pd.Timestamp(2023, 1, 2), pd.Timestamp(2024, 12, 31)
ADV_FLOOR = 2_000_000
SHOCK = 2.0
HOLDS = [2, 3]

def main():
    import json
    fo = json.load(open(MTF, encoding="utf-8"))
    eligible = {str(e["tradingsymbol"]).upper().strip() for e in fo
                if str(e.get("category")) != "etf"}
    print(f"MTF-eligible (excl ETF): {len(eligible)}")

    dd = pd.read_feather(DD)
    dd = dd.rename(columns={"ts": "date"}) if "ts" in dd.columns else dd
    dd["date"] = pd.to_datetime(dd["date"]).dt.normalize()
    dd["symbol"] = dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    dd = dd[dd["symbol"].isin(eligible)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = dd.groupby("symbol", sort=False)
    dd["pc"] = g["close"].shift(1)
    dd["ret1"] = dd["close"] / dd["pc"] - 1.0
    dd["ret2"] = dd["close"] / g["close"].shift(2) - 1.0        # C6: 2-day cumulative
    dd["turnover"] = dd["close"] * dd["volume"]
    dd["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    # causal turnover shock baseline (shift(1).rolling(20)) — no look-ahead (lesson #25)
    dd["adv20_prior"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20).mean())
    dd["tshock"] = dd["turnover"] / dd["adv20_prior"]
    # C5: intraday range / prev_close + closed weak
    dd["range_pct"] = (dd["high"] - dd["low"]) / dd["pc"]
    dd["close_weak"] = dd["close"] < dd["open"]
    dd["o_next"] = g["open"].shift(-1)
    for h in HOLDS:
        dd[f"fwd{h}"] = g["close"].shift(-(h + 1)) / dd["o_next"] - 1.0

    # Discovery window + universe gate
    d = dd[(dd["date"] >= DISC_START) & (dd["date"] <= DISC_END) &
           (dd["adv20"] >= ADV_FLOOR) & (dd["close"] >= 5) & (dd["tshock"].notna())].copy()
    d["tier"] = d.groupby("date")["adv20"].transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop") if s.nunique() >= 5 else np.nan).astype(float)
    t1 = d[d["tier"] == 1].copy()
    print(f"clean_daily Discovery rows (eligible+ADV-floor+close>=5): {len(d)}  | tier-1 rows: {len(t1)}")
    print(f"tier-1 date span: {t1['date'].min().date()}..{t1['date'].max().date()}  symbols: {t1['symbol'].nunique()}")

    # cross-sectional decile ranks (per day, within tier-1)
    t1["range_rk"] = t1.groupby("date")["range_pct"].rank(pct=True)   # high range = high rank
    t1["ret2_rk"] = t1.groupby("date")["ret2"].rank(pct=True)         # low return = low rank

    triggers = {
        "C5_range_exhaustion": (t1["range_rk"] >= 0.90) & (t1["close_weak"]) & (t1["tshock"] >= SHOCK),
        "C6_crash2d":          (t1["ret2_rk"] <= 0.10) & (t1["tshock"] >= SHOCK),
    }

    print("\n=== PHASE-2 RAW FOOTPRINT (Discovery only; signal vs tier-1 universe baseline = market-relative) ===")
    print("kill gate: |delta| < 0.10%  (no fees, raw drift)\n")
    for name, mask in triggers.items():
        sig = t1[mask]
        print(f"--- {name}: n={len(sig)} ({len(sig)/max(1,len(t1))*100:.1f}% of tier-1 rows) ---")
        if len(sig) < 50:
            print("   n<50: too thin to read footprint\n"); continue
        # symbol concentration (bad-print / single-name artifact guard)
        top = Counter(sig["symbol"]).most_common(5)
        print(f"   top-5 symbols: {top}  | unique symbols: {sig['symbol'].nunique()}")
        for h in HOLDS:
            s = sig.dropna(subset=[f"fwd{h}"])
            sig_dates = s["date"].unique()
            base = t1[t1["date"].isin(sig_dates)].dropna(subset=[f"fwd{h}"])
            sig_m, base_m = s[f"fwd{h}"].mean(), base[f"fwd{h}"].mean()
            delta = sig_m - base_m
            verdict = "PASS" if abs(delta) >= 0.001 else "KILL(<0.1%)"
            print(f"   hold{h}: signal {sig_m:+.3%}  baseline {base_m:+.3%}  "
                  f"delta {delta:+.3%}  [{verdict}]  win {(s[f'fwd{h}']>0).mean():.0%}  n={len(s)}")
        # Discovery sub-year stationarity (2023 vs 2024) — still Discovery-only
        for yr in (2023, 2024):
            sy = sig[sig["date"].dt.year == yr].dropna(subset=["fwd3"])
            by = t1[(t1["date"].dt.year == yr) & t1["date"].isin(sy["date"].unique())].dropna(subset=["fwd3"])
            if len(sy) >= 20:
                print(f"     {yr} hold3 delta: {sy['fwd3'].mean()-by['fwd3'].mean():+.3%} (n{len(sy)})")
        print()

if __name__ == "__main__":
    main()
