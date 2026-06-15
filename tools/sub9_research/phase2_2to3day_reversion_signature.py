"""Phase-2 signature (Stage 2, docs/setup_lifecycle.md) for the 2-3 day CNC/MTF
reversion batch: low52 / downstreak / crash1d.

DISCOVERY ONLY (2023-01-02..2024-12-31). Production-faithful universe:
MTF-eligible (excl ETF) ∩ ADV>=20L ∩ ADV-tier-1. Measures the RAW mechanism
footprint (no fees, no leverage) as signal_mean - universe_baseline_mean over a
2-3 day hold (T+1 open -> T+1+K close). Kill gate: |delta| < 0.1% drift.

NOT a ship test. No OOS/HO touched. Prints df.shape + per-mechanism n
(post-filter distribution discipline, lesson #26).
"""
from datetime import date
from pathlib import Path
import pandas as pd, numpy as np

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
    dd["date"] = pd.to_datetime(dd["date"]).dt.normalize()
    dd["symbol"] = dd["symbol"].astype(str).str.replace("NSE:", "", regex=False).str.upper()
    dd = dd[dd["symbol"].isin(eligible)].sort_values(["symbol", "date"]).reset_index(drop=True)
    g = dd.groupby("symbol", sort=False)
    dd["pc"] = g["close"].shift(1)
    dd["ret1"] = dd["close"] / dd["pc"] - 1.0
    dd["turnover"] = dd["close"] * dd["volume"]
    dd["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    dd["vbase20"] = g["volume"].transform(lambda s: s.shift(1).rolling(20).mean())
    dd["vshock"] = dd["volume"] / dd["vbase20"]
    dd["low252"] = g["low"].transform(lambda s: s.rolling(252, min_periods=60).min())
    dd["dist_low"] = dd["close"] / dd["low252"] - 1.0
    dd["mean20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    dd["std20"] = g["close"].transform(lambda s: s.rolling(20).std())
    dd["zscore"] = (dd["close"] - dd["mean20"]) / dd["std20"]
    dn = (dd["ret1"] < 0).astype(int)
    dd["streak"] = g.apply(lambda x: (x["ret1"] < 0).astype(int) *
        ((x["ret1"] < 0).astype(int).groupby(((x["ret1"] < 0).astype(int).diff() != 0).cumsum()).cumcount() + 1)
    ).reset_index(level=0, drop=True)
    dd["o_next"] = g["open"].shift(-1)
    for h in HOLDS:
        dd[f"fwd{h}"] = g["close"].shift(-(h + 1)) / dd["o_next"] - 1.0

    # Discovery window + universe gate
    d = dd[(dd["date"] >= DISC_START) & (dd["date"] <= DISC_END) & (dd["adv20"] >= ADV_FLOOR)].copy()
    d["tier"] = d.groupby("date")["adv20"].transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop") if s.nunique() >= 5 else np.nan).astype(float)
    t1 = d[d["tier"] == 1].copy()
    print(f"clean_daily Discovery rows (eligible+ADV-floor): {len(d)}  | tier-1 rows: {len(t1)}")
    print(f"tier-1 date span: {t1['date'].min().date()}..{t1['date'].max().date()}  symbols: {t1['symbol'].nunique()}")

    # cross-sectional rank for crash1d
    t1["ret1_rk"] = t1.groupby("date")["ret1"].rank(pct=True)

    triggers = {
        "C1_low52":     (t1["dist_low"] <= 0.03) & (t1["vshock"] >= SHOCK),
        "C2_downstreak":(t1["streak"] >= 3) & (t1["vshock"] >= SHOCK),
        "C3_crash1d":   (t1["ret1_rk"] <= 0.10) & (t1["vshock"] >= SHOCK),
        "C4_zscore":    (t1["zscore"] <= -2.0) & (t1["vshock"] >= SHOCK),  # new candidate
    }

    print("\n=== PHASE-2 RAW FOOTPRINT (Discovery only; signal vs tier-1 universe baseline) ===")
    print("kill gate: |delta| < 0.10%  (no fees, raw drift)\n")
    for name, mask in triggers.items():
        sig = t1[mask]
        print(f"--- {name}: n={len(sig)} ({len(sig)/max(1,len(t1))*100:.1f}% of tier-1 rows) ---")
        if len(sig) < 50:
            print("   n<50: too thin to read footprint\n"); continue
        for h in HOLDS:
            s = sig.dropna(subset=[f"fwd{h}"])
            # baseline = tier-1 universe mean forward on the SAME signal dates
            sig_dates = s["date"].unique()
            base = t1[t1["date"].isin(sig_dates)].dropna(subset=[f"fwd{h}"])
            sig_m, base_m = s[f"fwd{h}"].mean(), base[f"fwd{h}"].mean()
            delta = sig_m - base_m
            verdict = "PASS" if abs(delta) >= 0.001 else "KILL(<0.1%)"
            print(f"   hold{h}: signal {sig_m:+.3%}  baseline {base_m:+.3%}  "
                  f"delta {delta:+.3%}  [{verdict}]  win {(s[f'fwd{h}']>0).mean():.0%}  n={len(s)}")
        # Discovery sub-period stationarity (2023 vs 2024) — sanity, still Discovery-only
        for yr in (2023, 2024):
            sy = sig[sig["date"].dt.year == yr].dropna(subset=["fwd3"])
            by = t1[(t1["date"].dt.year == yr) & t1["date"].isin(sy["date"].unique())].dropna(subset=["fwd3"])
            if len(sy) >= 20:
                print(f"     {yr} hold3 delta: {sy['fwd3'].mean()-by['fwd3'].mean():+.3%} (n{len(sy)})")
        print()

if __name__ == "__main__":
    main()
