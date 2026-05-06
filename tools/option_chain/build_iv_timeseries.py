"""Build ATM-IV time series from option_chain bhavcopy parquets.

For each (session_date, symbol, expiry):
  - Find ATM strike (closest to spot from consolidated_daily.feather)
  - Compute Black-Scholes IV for ATM CE and PE from their LTPs
  - Average CE+PE IV → atm_iv
  - Use nearest expiry ≥ 7 days out (avoid expiry-day distortions)

Output: data/options_iv/<YYYY>_iv_timeseries.parquet
Schema: session_date, symbol, expiry_date, ttm_days, atm_strike,
        atm_iv, spot, oi_atm_ce, oi_atm_pe

Usage:
    python tools/option_chain/build_iv_timeseries.py --from 2023-01-01 --to 2024-12-31
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.option_chain.compute_iv import implied_vol_bs  # noqa: E402

_OPTION_CHAIN_DIR = _REPO_ROOT / "data" / "option_chain"
_OUT_DIR = _REPO_ROOT / "data" / "options_iv"
_SPOT_PATH = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
_FNO_UNIVERSE_PATH = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
_RISK_FREE_RATE = 0.07


def load_spot_lookup() -> pd.DataFrame:
    """Load consolidated daily, return DataFrame indexed by (symbol, date) → close."""
    print(f"  loading spot prices from {_SPOT_PATH.name} ...")
    df = pd.read_feather(_SPOT_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    df = df.rename(columns={"close": "spot"})
    print(f"  spot: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df[["symbol", "date", "spot"]].set_index(["symbol", "date"])["spot"]


def load_fno_universe() -> set:
    df = pd.read_csv(_FNO_UNIVERSE_PATH)
    return set(df["symbol"].astype(str).str.replace("NSE:", "", regex=False))


def parquet_paths_in_range(from_d: date, to_d: date) -> list[Path]:
    paths: list[Path] = []
    for year_dir in sorted(_OPTION_CHAIN_DIR.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if not month_dir.is_dir():
                continue
            for p in sorted(month_dir.glob("*.parquet")):
                try:
                    d = datetime.strptime(p.stem, "%Y-%m-%d").date()
                except ValueError:
                    continue
                if from_d <= d <= to_d:
                    paths.append(p)
    return paths


def process_session(session_path: Path, spot_lookup: pd.Series, universe: set,
                    rf: float = _RISK_FREE_RATE) -> list[dict]:
    """Process one session-date parquet. Returns list of (symbol, expiry, atm_iv) rows."""
    try:
        df = pd.read_parquet(session_path)
    except Exception as e:
        print(f"  skip {session_path.name}: {e}", file=sys.stderr)
        return []
    df = df[df["symbol"].isin(universe)]
    if df.empty:
        return []
    session_date = pd.to_datetime(df["session_date"].iloc[0]).date()
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    df["ttm_days"] = (pd.to_datetime(df["expiry_date"]) - pd.Timestamp(session_date)).dt.days

    # nearest expiry ≥ 7 days out for each symbol (avoid expiry-week noise)
    df_eligible = df[df["ttm_days"] >= 7].copy()
    if df_eligible.empty:
        return []

    rows: list[dict] = []
    for sym, g_sym in df_eligible.groupby("symbol"):
        spot = spot_lookup.get((sym, session_date))
        if spot is None or not np.isfinite(spot) or spot <= 0:
            continue

        # pick the nearest eligible expiry
        nearest_exp = g_sym["expiry_date"].min()
        g_exp = g_sym[g_sym["expiry_date"] == nearest_exp]
        ttm = g_exp["ttm_days"].iloc[0]

        # ATM strike = closest to spot
        strikes = g_exp["strike"].unique()
        if len(strikes) == 0:
            continue
        atm_strike = strikes[np.argmin(np.abs(strikes - spot))]

        ce_rows = g_exp[(g_exp["strike"] == atm_strike) & (g_exp["option_type"] == "CE")]
        pe_rows = g_exp[(g_exp["strike"] == atm_strike) & (g_exp["option_type"] == "PE")]
        if ce_rows.empty and pe_rows.empty:
            continue

        ivs = []
        oi_ce = oi_pe = 0
        if not ce_rows.empty:
            ce = ce_rows.iloc[0]
            ce_iv = implied_vol_bs(float(ce["ltp"]), float(spot), float(atm_strike),
                                    float(ttm), risk_free_rate=rf, option_type="CE")
            if np.isfinite(ce_iv):
                ivs.append(ce_iv)
            oi_ce = int(ce.get("oi", 0) or 0)
        if not pe_rows.empty:
            pe = pe_rows.iloc[0]
            pe_iv = implied_vol_bs(float(pe["ltp"]), float(spot), float(atm_strike),
                                    float(ttm), risk_free_rate=rf, option_type="PE")
            if np.isfinite(pe_iv):
                ivs.append(pe_iv)
            oi_pe = int(pe.get("oi", 0) or 0)

        if not ivs:
            continue
        atm_iv = float(np.mean(ivs))

        rows.append({
            "session_date": session_date,
            "symbol": sym,
            "expiry_date": nearest_exp,
            "ttm_days": int(ttm),
            "atm_strike": float(atm_strike),
            "atm_iv": atm_iv,
            "spot": float(spot),
            "oi_atm_ce": oi_ce,
            "oi_atm_pe": oi_pe,
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_date", type=str, required=True)
    p.add_argument("--to", dest="to_date", type=str, required=True)
    args = p.parse_args()

    from_d = datetime.strptime(args.from_date, "%Y-%m-%d").date()
    to_d = datetime.strptime(args.to_date, "%Y-%m-%d").date()

    spot_lookup = load_spot_lookup()
    universe = load_fno_universe()
    print(f"  F&O universe: {len(universe)} symbols")

    paths = parquet_paths_in_range(from_d, to_d)
    print(f"  option_chain parquets in range: {len(paths)}")

    all_rows: list[dict] = []
    for i, path in enumerate(paths):
        rows = process_session(path, spot_lookup, universe)
        all_rows.extend(rows)
        if (i + 1) % 50 == 0 or (i + 1) == len(paths):
            print(f"  [{i+1}/{len(paths)}] {path.stem} | running: {len(all_rows):,} rows")

    if not all_rows:
        print("no rows generated", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(all_rows).sort_values(["symbol", "session_date"]).reset_index(drop=True)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{from_d.year}_{to_d.year}_iv_timeseries.parquet"
    out_df.to_parquet(out_path)
    print()
    print(f"wrote: {out_path}")
    print(f"  rows: {len(out_df):,} | symbols: {out_df['symbol'].nunique()} "
          f"| date range: {out_df['session_date'].min()} → {out_df['session_date'].max()}")
    print(f"  IV stats: median={out_df['atm_iv'].median():.4f}, "
          f"P10={out_df['atm_iv'].quantile(0.10):.4f}, "
          f"P90={out_df['atm_iv'].quantile(0.90):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
