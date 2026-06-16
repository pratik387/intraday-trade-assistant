"""Stage-6 confidence card (Lesson #15 framework) for candidate #5/#6.

Stage 2 + Stage 4-5 verdict:
  - C5 range_exhaustion: KILL (Stage-2 delta NEGATIVE for a long; 0 ship-eligible
    cells in the mine). Continuation/inverse footprint, not a long-revert edge.
  - C6 crash2d: PASS. This card is for the LOCKED C6 cell.

Locked C6 cell (Discovery, most-stable by min dPF):
  MTF tier1 x 2-day-return cross-sectional rank <= 0.10 x turnover-shock >= 2.0x
  x K=3 hold, LONG, T+1-open entry, T+1+K close exit.

Card on DISCOVERY ONLY (OOS+HO used in most-stable selection); OOS/HO as one-shot
validators. Harvey-Liu at true M=72. net@(0.347%+20bp) delivery on Rs50k notional/trade.

Mirrors confidence_card_zscore.py; SEPARATE candidate-specific script.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
_DAILY = ROOT / "cache" / "preaggregate" / "clean_daily_from5m.feather"
_MTF = ROOT / "data" / "mtf_universe" / "approved_mtf_securities_2026-05-21.json"

def load():
    df = pd.read_feather(_DAILY).rename(columns={"ts": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if getattr(df["date"].dt, "tz", None) is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    return df.sort_values(["symbol", "date"])

def load_mtf_eligible():
    import json
    return {str(r["tradingsymbol"]).strip().upper()
            for r in json.load(open(_MTF, encoding="utf-8"))
            if str(r.get("category", "")).lower() != "etf" and r.get("tradingsymbol")}

from tools.methodology.confidence.bootstrap_ci import compute_aggregate_ci
from tools.methodology.confidence.regime_breakdown import compute_per_regime_stats, format_regime_table
from tools.methodology.confidence.selection_bias import build_daily_equity_curve, harvey_liu_haircut

K, TIER, SHOCK, RET2_MAX = 3, 1, 2.0, 0.10
COST = 0.00347 + 0.0020
NOTIONAL = 50000.0
M_TESTED = 72

def build():
    df = load(); elig = load_mtf_eligible()
    df["bare"] = df["symbol"].str.replace("NSE:", "", regex=False).str.upper()
    df = df[df["bare"].isin(elig)].sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    df["ret2"] = g["close"].transform(lambda s: s / s.shift(2) - 1)
    df["open_next"] = g["open"].shift(-1)
    df["fwd"] = g["close"].shift(-(1 + K)) / df["open_next"] - 1
    df["turnover"] = df["close"] * df["volume"]
    df["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    df["adv20_prior"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20).mean())
    df["tshock"] = df["turnover"] / df["adv20_prior"]
    df["yr"] = df["date"].dt.year
    m = df[(df.fwd.notna()) & (df.adv20 >= 2e6) & (df.close >= 5) & (df.tshock.notna()) & (df.ret2.notna())].copy()
    m["tier"] = m.groupby("date")["adv20"].transform(lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
    m["ret2_rk"] = m.groupby("date")["ret2"].rank(pct=True)
    cell = m[(m.ret2_rk <= RET2_MAX) & (m.tier == TIER) & (m.tshock >= SHOCK)].copy()
    cell["pnl"] = (cell["fwd"] - COST) * NOTIONAL
    cell["signal_date"] = pd.to_datetime(cell["date"]).dt.date
    return cell[["signal_date", "bare", "pnl", "yr"]].rename(columns={"bare": "symbol"})

def main():
    t = build()
    disc = t[t.yr.isin([2023, 2024])].copy()
    print(f"C6 crash2d confidence card | DISCOVERY n={len(disc)} (OOS n={(t.yr==2025).sum()}, HO n={(t.yr==2026).sum()}), net@(0.347%+20bp) Rs{int(NOTIONAL):,}/trade\n")
    print("== 1. Bootstrap BCa 95% CI (Discovery only) ==")
    ci = compute_aggregate_ci(disc, pnl_column="pnl")
    for k, v in (ci.items() if isinstance(ci, dict) else [("result", ci)]):
        print(f"  {k}: {v}")
    print("\n== 2. Per-regime breakdown (Discovery) ==")
    rs = compute_per_regime_stats(disc, date_column="signal_date", pnl_column="pnl")
    print(format_regime_table(rs))
    print("\n== 3. Harvey-Liu haircut (Discovery, true M=72) ==")
    daily = build_daily_equity_curve(disc, pnl_column="pnl", date_column="signal_date")
    for M in (10, M_TESTED):
        print(f"  M={M:>3} (Bonferroni): {harvey_liu_haircut('crash2d_revert', daily, M=M, method='Bonferroni')}")
    print("\n== 4. One-shot validators (cell locked on Discovery) ==")
    def pf(fr):
        w = fr[fr > 0].sum(); l = -fr[fr < 0].sum(); return (w / l) if l > 0 else float('nan')
    for lbl, yrs in [("OOS 2025", [2025]), ("HO 2026", [2026])]:
        sub = t[t.yr.isin(yrs)]
        print(f"  {lbl}: PF {pf(sub['pnl'].values):.2f}  mean Rs{sub['pnl'].mean():,.0f}/trade  win {(sub['pnl']>0).mean():.0%}  n={len(sub)}")

if __name__ == "__main__":
    main()
