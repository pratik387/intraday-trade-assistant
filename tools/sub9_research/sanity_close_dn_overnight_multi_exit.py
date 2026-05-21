"""Phase 5 R-analog: multi-exit-time sweep for close_dn_overnight_long.

The base sanity (sanity_close_dn_overnight_long.py) exits at next-day 09:15 open.
This variant computes net_pnl for MULTIPLE candidate exit times to find the
optimal exit-time × cell combination.

Exit candidates (all on the next trading day):
  - 09:15 open  (baseline — Phase 4 sanity uses this)
  - 09:30 close (15-min hold past open)
  - 10:00 close (45-min hold)
  - 11:00 close (1h45 hold)
  - 12:30 close (mid-day, ~3h)
  - 15:25 close (full-day hold — exits at MIS auto-square boundary)

For each (signal, exit_time) we produce a row. The sweep script can then
group by (cell_dims, exit_time) and find the best combination.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"

sys.path.insert(0, str(_REPO_ROOT))

# Reuse helpers from the base sanity
from tools.sub9_research.sanity_close_dn_overnight_long import (
    WINDOWS, CLOSING_30M_HHMM_LIST, SIGNED_VOL_RATIO_MAX,
    CLOSING_30M_VOLUME_Z_MIN, ROLLING_DAYS, POSITION_NOTIONAL_INR,
    MIN_TRADING_DAYS_COVERAGE, MIN_DAILY_AVG_VOLUME, ALLOWED_CAP_SEGMENTS,
    calc_fee_cnc, _get_cap_segment, _load_earnings_set, _has_earnings,
    _signed_vol_ratio_bin, _volume_z_bin, _prior_day_return_bin,
)


# Exit candidates: (label, hhmm, use_open_or_close)
EXITS = [
    ("exit_0915_open",  "09:15", "open"),
    ("exit_0930_close", "09:30", "close"),
    ("exit_1000_close", "10:00", "close"),
    ("exit_1100_close", "11:00", "close"),
    ("exit_1230_close", "12:30", "close"),
    ("exit_1525_close", "15:25", "close"),
]
EXIT_HHMM_LIST = [h for _, h, _ in EXITS]


def _months_between(d0: date, d1: date) -> List[Tuple[int, int]]:
    out = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


def _load_window(d0: date, d1: date) -> pd.DataFrame:
    """Load closing-30m bars + every exit-candidate bar from next trading day."""
    keep_hhmm = set(CLOSING_30M_HHMM_LIST + EXIT_HHMM_LIST)
    chunks = []
    load_start = d0 - timedelta(days=45)
    load_end = d1 + timedelta(days=7)
    for (yyyy, mm) in _months_between(load_start, load_end):
        p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            continue
        df = pd.read_feather(p, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype("float32")
        chunks.append(df)
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df = df[df["hhmm"].isin(keep_hhmm)].reset_index(drop=True)
    df["symbol"] = df["symbol"].astype("category")
    df = df.drop(columns=["date"])
    return df


def _compute_signals_and_exits(df_window: pd.DataFrame, d0: date, d1: date) -> pd.DataFrame:
    """Compute signals + multi-exit prices."""
    closing = df_window[df_window["hhmm"].isin(CLOSING_30M_HHMM_LIST)].copy()
    # Per-exit bars: pivot to wide format (symbol, d, exit_hhmm -> open/close)
    exits_long = df_window[df_window["hhmm"].isin(EXIT_HHMM_LIST)][
        ["symbol", "d", "hhmm", "open", "close"]
    ].copy()

    # Aggregate closing 30m
    print("  Aggregating closing 30m per (symbol, date)...", flush=True)
    closing["bar_dir"] = np.sign(closing["close"] - closing["open"]).astype("int8")
    closing["signed_vol"] = closing["volume"].astype("float64") * closing["bar_dir"]
    closing_agg = closing.groupby(["symbol", "d"], observed=True).agg(
        signed_vol_sum=("signed_vol", "sum"),
        total_vol=("volume", "sum"),
        bar_count=("hhmm", "count"),
        last_close=("close", "last"),
    ).reset_index()
    closing_agg["signed_vol_ratio"] = (
        closing_agg["signed_vol_sum"] / closing_agg["total_vol"].replace(0, np.nan)
    )

    # closing_30m_volume_z via prior-20d baseline
    print("  Computing closing_30m_volume_z...", flush=True)
    closing_agg = closing_agg.sort_values(["symbol", "d"])
    grp = closing_agg.groupby("symbol", observed=True)["total_vol"]
    closing_agg["close30_mean20"] = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_DAYS, min_periods=10).mean()
    )
    closing_agg["close30_std20"] = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_DAYS, min_periods=10).std()
    )
    closing_agg["closing_30m_volume_z"] = (
        (closing_agg["total_vol"] - closing_agg["close30_mean20"]) /
        closing_agg["close30_std20"].replace(0, np.nan)
    )

    # prior_day_return_pct
    closing_agg["prev_close"] = closing_agg.groupby("symbol", observed=True)["last_close"].shift(1)
    closing_agg["prior_day_return_pct"] = (
        (closing_agg["last_close"] - closing_agg["prev_close"]) /
        closing_agg["prev_close"].replace(0, np.nan) * 100.0
    )

    # Restrict to signal_date in window
    closing_agg = closing_agg[
        (closing_agg["d"] >= d0) & (closing_agg["d"] <= d1)
    ]

    # Apply primary filters
    print("  Applying primary filters...", flush=True)
    mask = (
        (closing_agg["bar_count"] >= 5)
        & (closing_agg["signed_vol_ratio"] <= SIGNED_VOL_RATIO_MAX)
        & (closing_agg["closing_30m_volume_z"] >= CLOSING_30M_VOLUME_Z_MIN)
        & (closing_agg["prior_day_return_pct"].notna())
    )
    signals = closing_agg[mask].copy()
    print(f"  Pre-exit-match signals: {len(signals):,}", flush=True)
    if signals.empty:
        return signals

    # Build exit-bar lookup: dict (symbol, d, hhmm) -> (open, close)
    print("  Building exit-bar lookup...", flush=True)
    exits_long["key"] = list(zip(exits_long["symbol"].astype(str), exits_long["d"], exits_long["hhmm"]))
    exit_lookup_open = dict(zip(exits_long["key"], exits_long["open"].astype("float64")))
    exit_lookup_close = dict(zip(exits_long["key"], exits_long["close"].astype("float64")))

    # For each signal, find next trading day and look up all exit prices
    print("  Matching next-trading-day per signal...", flush=True)
    # Group all exit-bar dates by symbol
    exit_dates_by_sym: Dict[str, List[date]] = {}
    for sym, g in exits_long.groupby("symbol", observed=True):
        exit_dates_by_sym[str(sym)] = sorted(g["d"].unique())

    next_day_map: Dict[Tuple[str, date], date] = {}
    for row in signals[["symbol", "d"]].itertuples(index=False):
        sym = str(row.symbol)
        dlist = exit_dates_by_sym.get(sym, [])
        fwd = [d for d in dlist if d > row.d]
        if fwd and (fwd[0] - row.d).days <= 7:
            next_day_map[(sym, row.d)] = fwd[0]

    print(f"  Signals with valid next-day: {len(next_day_map):,}", flush=True)
    signals = signals[signals.apply(
        lambda r: (str(r["symbol"]), r["d"]) in next_day_map, axis=1
    )].copy()
    signals["next_trading_day"] = signals.apply(
        lambda r: next_day_map[(str(r["symbol"]), r["d"])], axis=1
    )

    # Compute cell dims
    print("  Computing cell dims...", flush=True)
    signals["cap_segment"] = signals["symbol"].astype(str).apply(
        lambda s: _get_cap_segment(s) or "unknown"
    )
    signals = signals[signals["cap_segment"].isin(ALLOWED_CAP_SEGMENTS + ("unknown",))]
    signals["signed_vol_ratio_bin"] = signals["signed_vol_ratio"].apply(_signed_vol_ratio_bin)
    signals["closing_30m_volume_z_bin"] = signals["closing_30m_volume_z"].apply(_volume_z_bin)
    signals["prior_day_return_bin"] = signals["prior_day_return_pct"].apply(_prior_day_return_bin)
    signals["news_proximity"] = signals.apply(
        lambda r: "within_1day_earnings" if _has_earnings(str(r["symbol"]), r["next_trading_day"]) else "clear",
        axis=1,
    )

    # For each signal, look up each exit price
    print("  Looking up exit prices for each candidate...", flush=True)
    for label, hhmm, side in EXITS:
        col_open = f"{label}_raw_open"
        col_close = f"{label}_raw_close"
        signals[col_open] = signals.apply(
            lambda r: exit_lookup_open.get((str(r["symbol"]), r["next_trading_day"], hhmm)),
            axis=1,
        )
        signals[col_close] = signals.apply(
            lambda r: exit_lookup_close.get((str(r["symbol"]), r["next_trading_day"], hhmm)),
            axis=1,
        )
        # Effective exit price
        if side == "open":
            signals[label] = signals[col_open]
        else:
            signals[label] = signals[col_close]

    # Drop signals where the 09:15 open is missing (baseline exit must exist)
    signals = signals[signals[EXITS[0][0]].notna()].reset_index(drop=True)
    print(f"  Final signals (with 09:15 open): {len(signals):,}", flush=True)
    return signals


def _compute_trades_multi_exit(signals: pd.DataFrame) -> pd.DataFrame:
    """One row per signal, with net_pnl_inr per exit candidate as columns."""
    if signals.empty:
        return pd.DataFrame()

    rows = []
    for s in signals.itertuples(index=False):
        entry = float(s.last_close)
        if entry <= 0:
            continue
        qty = max(1, int(POSITION_NOTIONAL_INR / entry))
        buy_value = entry * qty
        row = {
            "signal_date": s.d,
            "symbol": f"NSE:{s.symbol}",
            "side": "LONG",
            "entry_price": entry,
            "qty": qty,
            "next_trading_day": s.next_trading_day,
            "signed_vol_ratio": float(s.signed_vol_ratio),
            "closing_30m_volume_z": float(s.closing_30m_volume_z),
            "prior_day_return_pct": float(s.prior_day_return_pct),
            "cap_segment": s.cap_segment,
            "signed_vol_ratio_bin": s.signed_vol_ratio_bin,
            "closing_30m_volume_z_bin": s.closing_30m_volume_z_bin,
            "prior_day_return_bin": s.prior_day_return_bin,
            "news_proximity": s.news_proximity,
        }
        for label, _, _ in EXITS:
            exit_p = getattr(s, label, None)
            if exit_p is None or pd.isna(exit_p) or exit_p <= 0:
                row[f"{label}_net_pnl_inr"] = None
                continue
            exit_p = float(exit_p)
            sell_value = exit_p * qty
            gross = sell_value - buy_value
            fee = calc_fee_cnc(buy_value, sell_value)
            net = gross - fee
            row[f"{label}_net_pnl_inr"] = float(net)
            row[f"{label}_exit_price"] = exit_p
        rows.append(row)
    return pd.DataFrame(rows)


def run_window(window_label: str) -> pd.DataFrame:
    d0, d1 = WINDOWS[window_label]
    print(f"\n=== Window: {window_label} ({d0} to {d1}) ===", flush=True)

    df = _load_window(d0, d1)
    print(f"  Loaded {len(df):,} bars (closing-30m + multi-exit candidates)", flush=True)

    # Universe filter: trading-days coverage + daily-avg-vol (same as base sanity)
    days_per_sym = df[df["hhmm"] == "15:25"].groupby("symbol", observed=True)["d"].nunique()
    total_window_days = df[df["hhmm"] == "15:25"]["d"].nunique()
    min_days = int(total_window_days * MIN_TRADING_DAYS_COVERAGE)
    keep_d = days_per_sym[days_per_sym >= min_days].index
    daily_vol = df.groupby(["symbol", "d"], observed=True)["volume"].sum()
    avg_vol = daily_vol.groupby(level=0, observed=True).mean()
    keep_v = avg_vol[avg_vol >= MIN_DAILY_AVG_VOLUME / 10].index
    keep = set(keep_d) & set(keep_v)
    df = df[df["symbol"].isin(keep)]
    print(f"  Universe: {len(keep)} symbols", flush=True)

    signals = _compute_signals_and_exits(df, d0, d1)
    if signals.empty:
        print("  No signals — empty output")
        return pd.DataFrame()
    trades = _compute_trades_multi_exit(signals)
    print(f"  Trades: {len(trades):,}", flush=True)

    # Quick stats per exit candidate
    print("\n  Per-exit-time aggregate stats:")
    for label, hhmm, _ in EXITS:
        col = f"{label}_net_pnl_inr"
        if col not in trades.columns:
            continue
        s = trades[col].dropna()
        if s.empty:
            continue
        g = s[s > 0].sum()
        l = -s[s < 0].sum()
        pf = (g / l) if l > 0 else float("inf")
        wr = (s > 0).mean() * 100
        print(f"    {label:>18}: n={len(s):>5}  PF={pf:.3f}  WR={wr:.1f}%  mean=Rs{s.mean():+.0f}/trade")

    return trades


def main():
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    for window_label in ("discovery", "oos", "holdout"):
        trades_df = run_window(window_label)
        out_path = out_dir / f"_close_dn_overnight_long_multi_exit_{window_label}.csv"
        if not trades_df.empty:
            trades_df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
