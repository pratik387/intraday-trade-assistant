"""Stage 4-5 (docs/setup_lifecycle.md) cell-mine for the 2-3 day CNC/MTF reversion
batch: C1 low52, C2 downstreak, C3 crash1d. Faithfully mirrors A2's daily
cell-mine (_tmp_cnc_cellmine.py): Discovery-only full-grid sweep -> ship-gate
(n>=200, netPF>=1.20) -> cross-apply ALL ship-eligible cells to OOS(2025)+HO(2026)
-> pick MOST-STABLE (min dPF), net-positive all 3 periods -> report TRUE M for
Harvey-Liu (feedback_cell_sweep_stability_over_top_pf).

Trade ledger = sanity (Stage 4) with anti-bias guards:
  - signal at day-t close; ENTRY T+1 open; EXIT T+1+K close (forward shifts)
  - turnover-shock baseline is CAUSAL (shift(1).rolling(20)) — no look-ahead
  - universe = MTF-eligible (excl ETF) x ADV>=20L x ADV-tier (cross-sectional/day)
  - REAL delivery cost 0.347% + 20bp slip (Zerodha CNC brokerage Rs0, STT %-symmetric)
K in {2,3} only (user 2-3 day hold constraint).
"""
from __future__ import annotations
import sys, itertools
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

COST = 0.00347 + 0.0020
K_GRID = [2, 3]
TIER_GRID = [1, 2, 3]
SHOCK_GRID = [1.0, 1.5, 2.0, 3.0]
SHIP_MIN_N = 200
SHIP_MIN_PF = 1.20

# candidate-specific selection dims
LOW52_GRID = [0.01, 0.03, 0.05, 0.10]   # close within X of 252d low
STREAK_GRID = [3, 4, 5]
DECILE_GRID = [0.05, 0.10, 0.20]
ZSCORE_GRID = [-1.5, -2.0, -2.5]        # close this many sd below 20d mean

def net_pf(fr):
    net = fr - COST
    w = net[net > 0].sum(); l = -net[net < 0].sum()
    return (w / l) if l > 0 else np.nan

def prep():
    df = load(); elig = load_mtf_eligible()
    df["bare"] = df["symbol"].str.replace("NSE:", "", regex=False).str.upper()
    df = df[df["bare"].isin(elig)].sort_values(["symbol", "date"]).copy()
    g = df.groupby("symbol", sort=False)
    df["ret1"] = g["close"].transform(lambda s: s / s.shift(1) - 1)
    df["open_next"] = g["open"].shift(-1)
    for k in K_GRID:
        df[f"fwd{k}"] = g["close"].shift(-(1 + k)) / df["open_next"] - 1
    df["turnover"] = df["close"] * df["volume"]
    df["adv20"] = g["turnover"].transform(lambda s: s.rolling(20).mean())
    df["adv20_prior"] = g["turnover"].transform(lambda s: s.shift(1).rolling(20).mean())
    df["tshock"] = df["turnover"] / df["adv20_prior"]
    df["low252"] = g["low"].transform(lambda s: s.rolling(252, min_periods=60).min())
    df["dist_low"] = df["close"] / df["low252"] - 1.0
    df["mean20"] = g["close"].transform(lambda s: s.rolling(20).mean())
    df["std20"] = g["close"].transform(lambda s: s.rolling(20).std())
    df["zscore"] = (df["close"] - df["mean20"]) / df["std20"]
    dni = (df["ret1"] < 0).astype(int)
    df["streak"] = g["ret1"].transform(
        lambda s: (s < 0).astype(int) * ((s < 0).astype(int).groupby(((s < 0).astype(int).diff() != 0).cumsum()).cumcount() + 1))
    df["yr"] = df["date"].dt.year
    base = df[(df.adv20 >= 2e6) & (df.close >= 5) & (df.tshock.notna())].copy()
    base["tier"] = base.groupby("date")["adv20"].transform(
        lambda s: pd.qcut(s, 5, labels=[1, 2, 3, 4, 5], duplicates="drop"))
    base["ret1_rk"] = base.groupby("date")["ret1"].rank(pct=True)
    return base

def sweep(base, name, dim_name, dim_grid, sel_fn):
    cells, M = [], 0
    for val, K, tier, shock in itertools.product(dim_grid, K_GRID, TIER_GRID, SHOCK_GRID):
        m = base[base[f"fwd{K}"].notna()]
        sel = sel_fn(m, val, tier, shock)
        disc = sel[sel.yr.isin([2023, 2024])][f"fwd{K}"].values
        if len(disc) < SHIP_MIN_N:
            continue
        M += 1
        dpf = net_pf(disc)
        if not (dpf >= SHIP_MIN_PF):
            continue
        oos = sel[sel.yr == 2025][f"fwd{K}"].values
        y26 = sel[sel.yr == 2026][f"fwd{K}"].values
        opf = net_pf(oos) if len(oos) >= 50 else np.nan
        pf26 = net_pf(y26) if len(y26) >= 30 else np.nan
        pfs = [p for p in [dpf, opf, pf26] if not np.isnan(p)]
        cells.append({dim_name: val, "K": K, "tier": tier, "shock": shock,
                      "n_disc": len(disc), "pf_disc": dpf, "pf_oos": opf, "pf_2026": pf26,
                      "dPF": (max(pfs) - min(pfs)) if len(pfs) >= 2 else np.nan,
                      "all_pos": all(p > 1.0 for p in pfs) and len(pfs) == 3})
    res = pd.DataFrame(cells)
    print(f"\n########## {name} ##########")
    print(f"cells swept (n_disc>={SHIP_MIN_N}): M={M} | ship-eligible (Disc netPF>={SHIP_MIN_PF}): {len(res)}")
    if not len(res):
        print("  NO ship-eligible cell on Discovery -> KILL (lesson #2 anti-salvage)"); return
    elig3 = res[res.all_pos].sort_values("dPF")
    print(f"  net-positive ALL 3 periods: {len(elig3)}")
    pd.set_option("display.width", 200)
    show = (elig3 if len(elig3) else res.sort_values("pf_disc", ascending=False)).head(10)
    print(show.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if len(elig3):
        b = elig3.iloc[0]
        print(f"  >>> MOST-STABLE: {dim_name}={b[dim_name]} K={int(b.K)} tier={int(b.tier)} shock={b.shock} "
              f"| dPF={b.dPF:.3f} PF Disc/OOS/26={b.pf_disc:.2f}/{b.pf_oos:.2f}/{b.pf_2026:.2f} | Harvey-Liu M={M}")
    else:
        print(f"  no all-3-positive cell -> marginal; Harvey-Liu M={M}")

def main():
    base = prep()
    print(f"universe rows (eligible+ADV-floor): {len(base)} | tier-1: {(base.tier==1).sum()} | yrs: {sorted(base.yr.unique())}")
    sweep(base, "C1_low52", "low52", LOW52_GRID,
          lambda m, v, t, s: m[(m.dist_low <= v) & (m.tier == t) & (m.tshock >= s)])
    sweep(base, "C2_downstreak", "streak_min", STREAK_GRID,
          lambda m, v, t, s: m[(m.streak >= v) & (m.tier == t) & (m.tshock >= s)])
    sweep(base, "C3_crash1d (NB: overlaps A2's N=1 mine cell)", "decile", DECILE_GRID,
          lambda m, v, t, s: m[(m.ret1_rk <= v) & (m.tier == t) & (m.tshock >= s)])
    sweep(base, "C4_zscore", "zscore_max", ZSCORE_GRID,
          lambda m, v, t, s: m[(m.zscore <= v) & (m.tier == t) & (m.tshock >= s)])

if __name__ == "__main__":
    main()
