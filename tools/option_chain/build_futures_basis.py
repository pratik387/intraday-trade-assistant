"""Build daily stock-futures basis time-series.

For each (session_date, symbol) on which we hold a stock-futures bhavcopy
row AND a daily spot close, compute:

    basis_pct  = (futures_close - spot_close) / spot_close * 100
    basis_bps  = basis_pct * 100

Only the **nearest expiry** futures contract per symbol per session is
retained (matches the §3.3 mechanic — front-month single-stock futures).
Index futures (FUTIDX) are excluded — there's no spot leg analog the
sub-9 candidate trades.

Sources:
  - data/futures/<YYYY>/<MM>/<YYYY-MM-DD>.parquet
        symbol, instrument_type ('FUTSTK'/'FUTIDX'), expiry_date,
        contract_type, strike, open/high/low/close, settle, oi, vol,
        session_date
  - cache/preaggregate/consolidated_daily.feather
        ts (datetime64[ns]), open/high/low/close/volume, symbol

Output:
    data/futures_basis/<from_year>_<to_year>_basis.parquet
    schema: session_date, symbol, expiry_date, days_to_expiry,
            futures_close, spot_close, basis_pct, basis_bps

Usage:
    python tools/option_chain/build_futures_basis.py --from 2023-01-01 --to 2026-04-30

Per specs/2026-05-07-sub-project-9-brief-stock_futures_basis_convergence.md §11
data-engineering plan, sub-step 1 (EOD-only first pass).
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

_FUTURES_DIR = _REPO_ROOT / "data" / "futures"
_OUT_DIR = _REPO_ROOT / "data" / "futures_basis"
_SPOT_PATH = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"


def load_spot_lookup() -> pd.Series:
    """Load consolidated daily, return Series indexed by (symbol, date) → close."""
    print(f"  loading spot prices from {_SPOT_PATH.name} ...")
    df = pd.read_feather(_SPOT_PATH)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["date"] = df["ts"].dt.date
    df = df.rename(columns={"close": "spot"})
    print(f"  spot: {len(df):,} rows, {df['symbol'].nunique()} symbols")
    return df[["symbol", "date", "spot"]].set_index(["symbol", "date"])["spot"]


def parquet_paths_in_range(from_d: date, to_d: date) -> list[Path]:
    paths: list[Path] = []
    if not _FUTURES_DIR.exists():
        return paths
    for year_dir in sorted(_FUTURES_DIR.iterdir()):
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


def process_session(path: Path, spot_lookup: pd.Series) -> list[dict]:
    """Process one futures parquet → list of basis rows (nearest-expiry only)."""
    try:
        df = pd.read_parquet(path)
    except Exception as e:   # noqa: BLE001
        print(f"  skip {path.name}: {e}", file=sys.stderr)
        return []
    if df.empty:
        return []

    # Stock-futures only — index futures have no retail spot-leg analog.
    df = df[df["instrument_type"] == "FUTSTK"].copy()
    if df.empty:
        return []

    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    session_date = pd.to_datetime(df["session_date"].iloc[0]).date()
    df["days_to_expiry"] = (
        pd.to_datetime(df["expiry_date"]) - pd.Timestamp(session_date)
    ).dt.days

    # Drop expired / same-day rows (basis convergence is exactly 0 there).
    df = df[df["days_to_expiry"] > 0]
    if df.empty:
        return []

    # Nearest expiry per symbol = front-month contract.
    df = df.sort_values(["symbol", "days_to_expiry"])
    df = df.drop_duplicates(subset=["symbol"], keep="first")

    rows: list[dict] = []
    for r in df.itertuples(index=False):
        sym = r.symbol
        spot = spot_lookup.get((sym, session_date))
        if spot is None or not np.isfinite(spot) or spot <= 0:
            continue
        fut_close = float(r.close) if r.close is not None and np.isfinite(r.close) else None
        if fut_close is None or fut_close <= 0:
            continue
        basis_pct = (fut_close - float(spot)) / float(spot) * 100.0
        rows.append({
            "session_date": session_date,
            "symbol": sym,
            "expiry_date": r.expiry_date,
            "days_to_expiry": int(r.days_to_expiry),
            "futures_close": fut_close,
            "spot_close": float(spot),
            "basis_pct": basis_pct,
            "basis_bps": basis_pct * 100.0,
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
    paths = parquet_paths_in_range(from_d, to_d)
    print(f"  futures parquets in range: {len(paths)}")

    all_rows: list[dict] = []
    for i, path in enumerate(paths):
        rows = process_session(path, spot_lookup)
        all_rows.extend(rows)
        if (i + 1) % 100 == 0 or (i + 1) == len(paths):
            print(f"  [{i+1}/{len(paths)}] {path.stem} | running: {len(all_rows):,} rows")

    if not all_rows:
        print("no rows generated", file=sys.stderr)
        return 1

    out_df = (
        pd.DataFrame(all_rows)
        .sort_values(["symbol", "session_date"])
        .reset_index(drop=True)
    )
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _OUT_DIR / f"{from_d.year}_{to_d.year}_basis.parquet"
    out_df.to_parquet(out_path, index=False)
    print()
    print(f"wrote: {out_path}")
    print(
        f"  rows: {len(out_df):,} | symbols: {out_df['symbol'].nunique()} "
        f"| date range: {out_df['session_date'].min()} -> {out_df['session_date'].max()}"
    )
    print(
        f"  basis_bps: median={out_df['basis_bps'].median():.2f}, "
        f"P10={out_df['basis_bps'].quantile(0.10):.2f}, "
        f"P90={out_df['basis_bps'].quantile(0.90):.2f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
