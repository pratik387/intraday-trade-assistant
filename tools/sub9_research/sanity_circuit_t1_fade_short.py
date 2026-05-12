"""Pre-coding sanity check for circuit_t1_fade_short candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-01-sub-project-9-brief-circuit_
t1_fade_short.md): BEFORE writing detector code, simulate the T+1 10:30
short on 6 months of 2024 5m bar data with simple circuit-hit detection.

Decision criterion (from brief):
  PF >= 1.10  → strong proceed
  1.0-1.10    → marginal, proceed with caveat
  PF < 1.0    → retire candidate, do NOT write detector

Usage:
    python tools/sub9_research/sanity_circuit_t1_fade_short.py

Circuit-hit detection (heuristic, no price-band CSV needed for sanity):
  - Daily close ≥ 4.5% above prior close (catches stocks hitting 5% / 10% bands)
  - Day high ≈ day close (≥ 99.5% of high, i.e. clamped at top)
  - Last-30-min volume drops to ≤ 35% of intraday-avg (price clamp signature)
This is a heuristic substitute for the proper price-band CSV lookup; the
real detector (post-approval) will use the official band CSV. Trades
precision for sanity-check speed.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Config knobs (matching the §3.3 brief) ----
ALLOWED_CAPS = {"mid_cap", "small_cap"}    # exclude large (rare circuits) + micro (no short liquidity)
MIN_DAY_GAIN_PCT = 4.5                     # circuit-hit detection floor (catches 5% / 10% bands)
HIGH_TO_CLOSE_RATIO_MIN = 0.995            # day high ≈ day close (clamped at top)
LAST30MIN_VOL_RATIO_MAX = 0.35             # last-30-min vol ≤ 35% of intraday avg (clamp signature)
MIN_VOL_VS_20D = 1.5                       # T+0 day's volume ≥ 1.5× 20-day avg (true pump)

T1_GAP_MIN_PCT = 1.0                       # T+1 must gap up ≥ 1% (continuation evidence)
T1_GAP_MAX_PCT = 5.0                       # but not > 5% (fundamental news territory)

ENTRY_BAR_HHMM = "10:30"                    # T+1 inflection per Chen/Petukhov/Wang
EXIT_BAR_HHMM  = "15:10"                    # last bar before MIS auto-square

STOP_T1_HIGH_BUFFER_PCT = 0.5              # SL = T+1 high × (1 + 0.005)
MIN_STOP_PCT = 1.0                         # qty-inflation guard (1% min)
RISK_PER_TRADE_RUPEES = 1000


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


import os as _os
_PERIOD = _os.environ.get("SANITY_PERIOD", "discovery").lower()
if _PERIOD == "oos":
    _Y_M_PAIRS = [(2025, m) for m in range(1, 10)]
    _DATE_LO, _DATE_HI = date(2025, 1, 1), date(2025, 9, 30)
    _OUT_SUFFIX = "_oos"
elif _PERIOD == "holdout":
    _Y_M_PAIRS = [(2025, m) for m in range(10, 13)] + [(2026, m) for m in range(1, 5)]
    _DATE_LO, _DATE_HI = date(2025, 10, 1), date(2026, 4, 30)
    _OUT_SUFFIX = "_holdout"
else:
    _Y_M_PAIRS = [(2024, m) for m in range(1, 13)]
    _DATE_LO, _DATE_HI = date(2024, 1, 1), date(2024, 12, 31)
    _OUT_SUFFIX = ""


def build_full_year_5m() -> pd.DataFrame:
    """Concatenate monthly 5m feathers for the configured SANITY_PERIOD."""
    print(f"  loading {len(_Y_M_PAIRS)} monthly 5m feathers ({_PERIOD}) ...")
    parts: List[pd.DataFrame] = []
    for yyyy, m in _Y_M_PAIRS:
        mdf = _load_5m_for_month(yyyy, m)
        if not mdf.empty:
            parts.append(mdf)
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_production_daily() -> pd.DataFrame:
    """Load 1day OHLCV directly from the SAME source production reads.

    Earlier versions of this script aggregated 5m bars to daily
    (groupby + max-high + last-close), which produced a phantom daily
    series where close == high for circuit-hit days. The actual 1day
    data from Upstox/Zerodha (cached in consolidated_daily.feather)
    reports a different close — typically slightly lower than the
    intraday high — because the API close is the post-close auction /
    settlement price, not the last 5m bar's close.

    Production reads consolidated_daily.feather. Sanity must too,
    otherwise the brief gate measures a setup that doesn't exist in
    production. Closes the data divergence found while diagnosing
    why circuit_t1_fade_short fired 0 trades on a date this script
    expected 11 fires.
    """
    print("  loading production 1day data from consolidated_daily.feather ...")
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(
            f"{daily_path} missing. Run "
            f"`python tools/create_preaggregated_cache.py "
            f"--from 2023-01-01 --to 2024-12-31` to build."
        )
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    # Sanity period: full calendar year 2024 (T+0 events 2024-01 through 2024-12;
    # T+1 entry trades land in 2024-01 through 2025-01 — the 5m feather is
    # 2024-only so any 2024-12-31 hits won't simulate, accepted truncation).
    df = df[(df["d"] >= _DATE_LO) & (df["d"] <= _DATE_HI)]
    df = df.rename(columns={"open": "open", "high": "high", "low": "low",
                            "close": "close", "volume": "volume"})
    df = df[["symbol", "d", "open", "high", "low", "close", "volume"]].copy()
    print(f"  daily rows (2024 only): {len(df):,} | symbols: {df['symbol'].nunique()}")
    return df


def detect_circuit_hits(daily: pd.DataFrame) -> pd.DataFrame:
    """Apply the heuristic upper-circuit detector + cap filter + volume filter.

    Returns one row per (symbol, T+0 date) circuit-hit event.
    """
    df = daily.sort_values(["symbol", "d"]).copy()
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["pct_change"] = (df["close"] / df["prev_close"] - 1.0) * 100.0
    df["high_to_close"] = df["close"] / df["high"]
    # 20-day avg volume (excluding T+0 itself)
    df["vol_avg_20d"] = df.groupby("symbol")["volume"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    )
    df["vol_ratio_20d"] = df["volume"] / df["vol_avg_20d"]

    print("  applying circuit-hit heuristic + cap + volume filters ...")
    print(f"    raw daily rows:                            {len(df):,}")
    df = df.dropna(subset=["pct_change", "vol_avg_20d"])
    print(f"    with prior-close + 20d vol history:         {len(df):,}")

    df = df[df["pct_change"] >= MIN_DAY_GAIN_PCT]
    print(f"    pct_change ≥ {MIN_DAY_GAIN_PCT}%:                       {len(df):,}")

    df = df[df["high_to_close"] >= HIGH_TO_CLOSE_RATIO_MIN]
    print(f"    close ≈ day high (clamped):                {len(df):,}")

    # last-30-min volume share check dropped to match production detector.
    # Production reads only 1day OHLCV (no intraday volume share available);
    # the close ≈ high check above is the working proxy for "price clamped
    # at the band edge." Re-introducing it here would re-create the
    # divergence we just fixed.

    df = df[df["vol_ratio_20d"] >= MIN_VOL_VS_20D]
    print(f"    day vol ≥ {MIN_VOL_VS_20D}× 20d avg:                  {len(df):,}")

    # cap_segment filter
    df["nse_symbol"] = "NSE:" + df["symbol"].astype(str)
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)]
    print(f"    cap_segment ∈ {sorted(ALLOWED_CAPS)}:    {len(df):,}")

    return df.reset_index(drop=True)


def build_daily_atr_table(daily: pd.DataFrame) -> pd.Series:
    """14-day ATR per (symbol, date). Same approach as bulk-block sanity."""
    df = daily.sort_values(["symbol", "d"]).copy()
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["tr"] = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"]  - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = df.groupby("symbol")["tr"].transform(
        lambda s: s.rolling(14).mean()
    )
    return df.set_index(["symbol", "d"])["atr14"]


def simulate_t1_short(
    hits: pd.DataFrame,
    big5m: pd.DataFrame,
    atr_table: pd.Series,
) -> pd.DataFrame:
    """For each T+0 circuit-hit, find T+1 (next trading day in 5m data),
    check gap-up ≥1% and ≤5%, enter SHORT at 10:30 close, exit at SL or 15:10."""
    trades: List[dict] = []

    # Index 5m by (symbol, d) for fast lookup of T+1 day's bars
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date") for sym, g in big5m.groupby("symbol")
    }

    n_no_t1 = n_gap_fail = n_no_entry_bar = n_traded = 0
    for _, hit in hits.iterrows():
        sym = hit["symbol"]
        sd = hit["d"]
        t0_close = float(hit["close"])

        # Find T+1: next trading day in big5m for this symbol
        sym_df = days_per_sym.get(sym)
        if sym_df is None or sym_df.empty:
            n_no_t1 += 1; continue
        future = sym_df[sym_df["d"] > sd]
        if future.empty:
            n_no_t1 += 1; continue
        t1 = future["d"].iloc[0]
        t1_df = future[future["d"] == t1].sort_values("date")
        if t1_df.empty:
            n_no_t1 += 1; continue

        t1_open = float(t1_df.iloc[0]["open"])
        t1_high = float(t1_df["high"].max())
        gap_pct = (t1_open / t0_close - 1.0) * 100.0

        if gap_pct < T1_GAP_MIN_PCT or gap_pct > T1_GAP_MAX_PCT:
            n_gap_fail += 1; continue

        # Entry bar: 10:30
        entry_bar = t1_df[t1_df["date"].dt.strftime("%H:%M") == ENTRY_BAR_HHMM]
        if entry_bar.empty:
            n_no_entry_bar += 1; continue
        entry_ts = entry_bar.iloc[0]["date"]
        entry_price = float(entry_bar.iloc[0]["close"])

        # Stop = max(t1_high × (1 + STOP_BUFFER), entry × (1 + MIN_STOP_PCT/100))
        sl_from_high = t1_high * (1.0 + STOP_T1_HIGH_BUFFER_PCT / 100.0)
        sl_from_min = entry_price * (1.0 + MIN_STOP_PCT / 100.0)
        hard_sl = max(sl_from_high, sl_from_min)
        stop_distance = hard_sl - entry_price

        # T1 = t1_open (gap start), T2 = t0_close (full gap fill)
        t1_target = t1_open
        t2_target = t0_close

        # Walk forward: find first of (stop hit, T2 hit, EXIT_BAR_HHMM)
        after = t1_df[t1_df["date"] >= entry_ts].copy()
        exit_ts = None
        exit_price = None
        exit_reason = None
        for _, bar in after.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            # Stop check first (worst-case fill)
            if high >= hard_sl:
                exit_ts = ts
                exit_price = hard_sl
                exit_reason = "stop"
                break
            # T2 (full gap fill — best for short)
            if low <= t2_target:
                exit_ts = ts
                exit_price = t2_target
                exit_reason = "t2_full_gap_fill"
                break
            # Time stop
            if ts.strftime("%H:%M") >= EXIT_BAR_HHMM:
                exit_ts = ts
                exit_price = float(bar["close"])
                exit_reason = "eod"
                break
        if exit_price is None:
            last = after.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "last_bar"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        # SHORT: pnl = (entry - exit) × qty
        realized_pnl = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "SELL")
        net_pnl = realized_pnl - fee

        trades.append({
            "T0_signal_date": sd,
            "T1_entry_date": t1,
            "symbol": "NSE:" + sym,
            "cap_segment": hit["cap_segment"],
            "t0_pct_change": hit["pct_change"],
            "t0_vol_ratio_20d": hit["vol_ratio_20d"],
            "t0_close": t0_close,
            "t1_open": t1_open,
            "t1_high": t1_high,
            "gap_pct": gap_pct,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"\n  no T+1 data:          {n_no_t1}")
    print(f"  gap fail (<1% or >5%): {n_gap_fail}")
    print(f"  no 10:30 entry bar:   {n_no_entry_bar}")
    print(f"  traded:               {n_traded}")
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        return
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)

    print("\n=== circuit_t1_fade_short — pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: ₹{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      ₹{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   ₹{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>4} PF={pf2:>5} netPnL=₹{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=₹{int(grp['net_pnl'].mean()):>6,}")

    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"PF={pf} >= 1.10 → STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) → marginal. Proceed with caveat.")
    else:
        print(f"PF={pf} < 1.00 → RETIRE candidate. Do not write detector code.")


def main():
    # T-1 circuit-hit classification reads production 1day data
    # (consolidated_daily.feather) — same source mock_broker.get_daily
    # returns to circuit_t1_fade_short.detect() at runtime. T+1
    # simulation continues to use 5m bars.
    big5m = build_full_year_5m()
    daily = load_production_daily()
    hits = detect_circuit_hits(daily)
    print(f"\nFiltered to {len(hits)} T+0 upper-circuit-hit events.")
    if hits.empty:
        return

    atr_table = build_daily_atr_table(daily)

    print("\nSimulating T+1 10:30 short entry → 15:10 exit (or stop / gap-fill T2):")
    trades = simulate_t1_short(hits, big5m, atr_table)
    report(trades)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / f"circuit_t1_fade_short_trades{_OUT_SUFFIX}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
