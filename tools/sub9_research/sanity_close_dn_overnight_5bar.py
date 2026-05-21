"""Phase 5 robustness check: 5-bar partial signal vs 6-bar full signal.

Background: the base sanity computes the closing flush signal over the
FULL last 30min (six 5m bars: 15:00, 15:05, 15:10, 15:15, 15:20, 15:25).
The 15:25 bar represents 15:25-15:30 IST and doesn't fully close until
15:30 — but our backtest "entry" uses that bar's close (= 15:30 IST price).

This is a LOOK-AHEAD artifact for live execution: in live we can only
know bars that have CLOSED. At 15:25 IST (when we'd submit a MOC order),
only bars 15:00-15:20 are closed. The 15:25-15:30 bar is still forming.

This script recomputes the signal using ONLY the first 5 bars (15:00-15:20,
25-min closing window), keeps entry at the 15:25 bar's close (= MOC fill
at 15:30 IST), and exits at next-day 09:15 open as before.

The PF delta between 5-bar and 6-bar signals quantifies the look-ahead
bias. Output mirrors the base sanity schema so downstream cell-sweep
+ confidence-card scripts can re-run on the 5-bar output for an
apples-to-apples comparison.
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

from tools.sub9_research.sanity_close_dn_overnight_long import (
    WINDOWS, SIGNED_VOL_RATIO_MAX, CLOSING_30M_VOLUME_Z_MIN, ROLLING_DAYS,
    POSITION_NOTIONAL_INR, MIN_TRADING_DAYS_COVERAGE, MIN_DAILY_AVG_VOLUME,
    ALLOWED_CAP_SEGMENTS, calc_fee_cnc, _get_cap_segment, _has_earnings,
    _signed_vol_ratio_bin, _volume_z_bin, _prior_day_return_bin,
)


# 5-bar partial signal: ONLY bars known at 15:25 IST when MOC order is submitted
SIGNAL_HHMM_LIST = ["15:00", "15:05", "15:10", "15:15", "15:20"]
# Entry bar (the 6th, last bar before market close — used for entry price only)
ENTRY_BAR_HHMM = "15:25"
EXIT_BAR_HHMM = "09:15"


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
    keep_hhmm = set(SIGNAL_HHMM_LIST + [ENTRY_BAR_HHMM, EXIT_BAR_HHMM])
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


def _compute_5bar_signals(df_window: pd.DataFrame, d0: date, d1: date) -> pd.DataFrame:
    # 5-bar signal aggregate (bars 15:00-15:20)
    signal_bars = df_window[df_window["hhmm"].isin(SIGNAL_HHMM_LIST)].copy()
    entry_bars = df_window[df_window["hhmm"] == ENTRY_BAR_HHMM][["symbol", "d", "close"]].copy()
    entry_bars = entry_bars.rename(columns={"close": "entry_close_15_25"})
    opening = df_window[df_window["hhmm"] == EXIT_BAR_HHMM][["symbol", "d", "open"]].copy()
    opening = opening.rename(columns={"open": "next_open"})

    print("  Aggregating 5-bar signal (15:00-15:20)...", flush=True)
    signal_bars["bar_dir"] = np.sign(signal_bars["close"] - signal_bars["open"]).astype("int8")
    signal_bars["signed_vol"] = signal_bars["volume"].astype("float64") * signal_bars["bar_dir"]
    sig_agg = signal_bars.groupby(["symbol", "d"], observed=True).agg(
        signed_vol_sum=("signed_vol", "sum"),
        total_vol=("volume", "sum"),
        bar_count=("hhmm", "count"),
        last_close_15_20=("close", "last"),  # not used for entry, just diagnostic
    ).reset_index()
    sig_agg["signed_vol_ratio"] = (
        sig_agg["signed_vol_sum"] / sig_agg["total_vol"].replace(0, np.nan)
    )

    # Closing-25m volume z (using 5 bars) — baseline is prior-20d 5-bar total
    print("  Computing 5-bar volume z (prior-20d baseline)...", flush=True)
    sig_agg = sig_agg.sort_values(["symbol", "d"])
    grp = sig_agg.groupby("symbol", observed=True)["total_vol"]
    sig_agg["close25_mean20"] = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_DAYS, min_periods=10).mean()
    )
    sig_agg["close25_std20"] = grp.transform(
        lambda s: s.shift(1).rolling(ROLLING_DAYS, min_periods=10).std()
    )
    sig_agg["closing_30m_volume_z"] = (  # keep same column name for downstream compat
        (sig_agg["total_vol"] - sig_agg["close25_mean20"]) /
        sig_agg["close25_std20"].replace(0, np.nan)
    )

    # Attach entry price (15:25 bar's close = 15:30 IST MOC fill)
    sig_agg = sig_agg.merge(entry_bars, on=["symbol", "d"], how="left")
    sig_agg = sig_agg.rename(columns={"entry_close_15_25": "last_close"})

    # prior_day_return_pct from prior session's last_close
    sig_agg["prev_close"] = sig_agg.groupby("symbol", observed=True)["last_close"].shift(1)
    sig_agg["prior_day_return_pct"] = (
        (sig_agg["last_close"] - sig_agg["prev_close"]) /
        sig_agg["prev_close"].replace(0, np.nan) * 100.0
    )

    # Restrict to signal_date window
    sig_agg = sig_agg[(sig_agg["d"] >= d0) & (sig_agg["d"] <= d1)]

    # Next-trading-day 09:15 open
    print("  Matching next-trading-day 09:15 open...", flush=True)
    opening = opening.sort_values(["symbol", "d"]).reset_index(drop=True)
    open_by_sym = {sym: g.reset_index(drop=True) for sym, g in opening.groupby("symbol", observed=True)}
    next_opens = []
    next_dates = []
    for row in sig_agg.itertuples(index=False):
        sym_g = open_by_sym.get(row.symbol)
        if sym_g is None or sym_g.empty:
            next_opens.append(None); next_dates.append(None); continue
        fwd = sym_g[sym_g["d"] > row.d]
        if fwd.empty:
            next_opens.append(None); next_dates.append(None)
        else:
            first = fwd.iloc[0]
            if (first["d"] - row.d).days <= 7:
                next_opens.append(float(first["next_open"])); next_dates.append(first["d"])
            else:
                next_opens.append(None); next_dates.append(None)
    sig_agg["next_open"] = next_opens
    sig_agg["next_trading_day"] = next_dates

    # Apply signal filters (same thresholds)
    print("  Applying signal filters (5-bar variant)...", flush=True)
    mask = (
        (sig_agg["bar_count"] >= 4)   # at least 4 of 5 bars present (was 5 of 6)
        & (sig_agg["signed_vol_ratio"] <= SIGNED_VOL_RATIO_MAX)
        & (sig_agg["closing_30m_volume_z"] >= CLOSING_30M_VOLUME_Z_MIN)
        & (sig_agg["next_open"].notna())
        & (sig_agg["prior_day_return_pct"].notna())
        & (sig_agg["last_close"].notna())
    )
    signals = sig_agg[mask].copy()
    print(f"  Signals (5-bar): {len(signals):,}", flush=True)
    if signals.empty:
        return signals

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
    return signals.reset_index(drop=True)


def _compute_trades(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    rows = []
    for s in signals.itertuples(index=False):
        entry = float(s.last_close)
        exit_p = float(s.next_open)
        if entry <= 0 or exit_p <= 0:
            continue
        qty = max(1, int(POSITION_NOTIONAL_INR / entry))
        buy_value = entry * qty
        sell_value = exit_p * qty
        gross = sell_value - buy_value
        fee = calc_fee_cnc(buy_value, sell_value)
        net = gross - fee
        pnl_pct = (exit_p - entry) / entry * 100.0
        rows.append({
            "signal_date": s.d,
            "symbol": f"NSE:{s.symbol}",
            "side": "LONG",
            "entry_price": entry,
            "exit_price": exit_p,
            "qty": qty,
            "pnl_pct": pnl_pct,
            "exit_reason": "next_day_open",
            "cap_segment": s.cap_segment,
            "signal_ts": f"{s.d}T15:25:00",
            "entry_ts": f"{s.d}T15:25:00",  # MOC fill at 15:30 IST, signal known at 15:25
            "exit_ts": f"{s.next_trading_day}T09:15:00",
            "realized_pnl_inr": float(gross),
            "fee_inr": float(fee),
            "net_pnl_inr": float(net),
            "signed_vol_ratio": float(s.signed_vol_ratio),
            "closing_30m_volume_z": float(s.closing_30m_volume_z),
            "prior_day_return_pct": float(s.prior_day_return_pct),
            "next_trading_day": s.next_trading_day,
            "signed_vol_ratio_bin": s.signed_vol_ratio_bin,
            "closing_30m_volume_z_bin": s.closing_30m_volume_z_bin,
            "prior_day_return_bin": s.prior_day_return_bin,
            "news_proximity": s.news_proximity,
        })
    return pd.DataFrame(rows)


def run_window(window_label: str) -> pd.DataFrame:
    d0, d1 = WINDOWS[window_label]
    print(f"\n=== Window: {window_label} ({d0} to {d1}) — 5-bar signal variant ===", flush=True)
    df = _load_window(d0, d1)
    print(f"  Loaded {len(df):,} bars", flush=True)

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

    signals = _compute_5bar_signals(df, d0, d1)
    if signals.empty:
        return pd.DataFrame()
    trades = _compute_trades(signals)
    print(f"  Trades: {len(trades):,}", flush=True)
    if not trades.empty:
        n_win = (trades["net_pnl_inr"] > 0).sum()
        net = trades["net_pnl_inr"].sum()
        print(f"  Aggregate: WR={n_win/len(trades)*100:.1f}%  NET=Rs{net:+,.0f}  mean=Rs{net/len(trades):+.0f}/trade")

        # Spot check Cell #2 (neg0.9_to_neg1.0)
        cell2 = trades[trades["signed_vol_ratio_bin"] == "neg0.9_to_neg1.0"]
        if len(cell2):
            s = cell2["net_pnl_inr"]
            pf = s[s>0].sum() / max(1, -s[s<0].sum())
            print(f"  Cell #2 (neg0.9): n={len(cell2):,}  PF={pf:.3f}  WR={(s>0).mean()*100:.1f}%  mean=Rs{s.mean():+.0f}")
    return trades


def main():
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    for window_label in ("discovery", "oos", "holdout"):
        trades_df = run_window(window_label)
        out_path = out_dir / f"_close_dn_overnight_long_5bar_trades_{window_label}.csv"
        if not trades_df.empty:
            trades_df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
