"""Pre-coding sanity check for index_stock_divergence_revert candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-06-sub-project-9-brief-index_
stock_divergence_revert.md): BEFORE writing detector code, simulate
the cross-asset spread mean-reversion fade on 12 months of 2024 5m
enriched feathers + NIFTY 50 1m feather (resampled to 5m).

Decision criterion (from brief):
  PF >= 1.10  -> strong proceed
  1.0-1.10    -> marginal, proceed with caveat
  PF < 1.0    -> retire candidate, do NOT write detector

Long-side ship gate: LONG-side PF must be >= SHORT-side PF * 0.85.

Usage:
    python tools/sub9_research/sanity_index_stock_divergence_revert.py

Mechanic (per locked brief params):
  - Compute rolling 30-day daily beta of stock vs NIFTY 50 from
    consolidated_daily.feather; one beta per (sym, day).
  - Per 5m bar in [11:00, 14:30] IST:
      stock_intraday_ret = (close[t] - today_open) / today_open
      nifty_intraday_ret = (NIFTY_close[t] - NIFTY_today_open) / NIFTY_today_open
      spread = stock_intraday_ret - beta * nifty_intraday_ret
      spread_z = (spread - rolling_30bar_mean) / rolling_30bar_stdev (within day)
  - Trigger: |spread_z| >= 2.0 + reversal-candle on SAME bar
  - LONG side gated on NIFTY uptrend (NIFTY 5m close > NIFTY 5m EMA20)
  - Entry: confirmation bar's CLOSE
  - Hard SL: trigger high/low + 0.3% buffer; min stop 0.6% of entry
  - T1: spread_z reverts to 0; T2: overshoots to -/+ 0.5
  - Time stop: 12 bars (~60 min) or 15:10 IST hard stop
  - Latch per (symbol, day, side)
  - Universe: F&O 200 (no cap-segment exclusion -- only candidate without)

Note on intraday return formulation: brief Mechanic step 2 specifies
intraday return (close - today_open)/today_open. The user task description
mentioned 30-bar rolling return; we follow the BRIEF (intraday cumulative
ret) because it is the locked-param reference. Spread z-score is then
computed on a rolling 30-bar window within the session per brief.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (research-defensible per brief §3.3 / Mechanic) ----
SPREAD_Z_THRESHOLD = 2.0                  # +/-2.0 sigma — locked threshold (brief Mechanic step 2)
BETA_WINDOW_DAYS = 30                     # 30-day rolling daily beta (brief Mechanic step 1)
SPREAD_ZSCORE_WINDOW_BARS = 30            # 30 5m bars rolling stdev within session
ENTRY_WINDOW_START_HHMM = "11:00"         # 11:00 IST — NIFTY accumulated >= 21 5m bars
ENTRY_WINDOW_END_HHMM = "14:30"           # 14:30 IST — earlier than candidate 1 (cross-asset rationale)
TIME_STOP_BARS = 12                       # 12 bars (~60 min) per brief
TIME_STOP_HARD_HHMM = "15:10"             # absolute hard stop
SL_BUFFER_PCT = 0.3                       # 0.3% buffer
MIN_STOP_PCT = 0.6                        # min stop 0.6%
T2_OVERSHOOT_STDEV_MULT = 0.5             # T2: spread_z = -/+ 0.5
NIFTY_UPTREND_EMA_BARS = 20               # NIFTY 5m EMA20 for LONG-side cross-asset gate
RISK_PER_TRADE_RUPEES = 1000              # match circuit_t1 sanity
REVERSAL_WICK_MIN_FRAC = 0.5              # >=50% rejection wick on reversal-confirmation bar


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_full_year_5m() -> pd.DataFrame:
    print("  loading 12 monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    for m in range(1, 13):
        mdf = _load_5m_for_month(2024, m)
        if not mdf.empty:
            parts.append(mdf)
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_fno_universe() -> set:
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_nifty_5m() -> pd.DataFrame:
    """Load NIFTY 50 1m feather, strip tz to IST-naive, resample to 5m."""
    print("  loading NIFTY 50 1m feather + resampling to 5m ...")
    p = _REPO_ROOT / "backtest-cache-download" / "index_ohlcv" / "NSE_NIFTY_50" / "NSE_NIFTY_50_1minutes.feather"
    if not p.exists():
        raise FileNotFoundError(f"{p} missing — required for cross-asset sanity check")
    df = pd.read_feather(p)
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= pd.Timestamp("2024-01-01")) & (df["date"] < pd.Timestamp("2025-01-01"))]
    df = df.set_index("date")
    # 5m resample (right-closed/right-labeled to match production 5m bar 09:15..09:19 -> 09:15)
    agg = df.resample("5min", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last"
    }).dropna()
    agg = agg.reset_index().rename(columns={"date": "ts"})
    agg["d"] = agg["ts"].dt.date
    # daily open: first 5m bar's open per day
    first_open = agg.groupby("d")["open"].transform("first")
    agg["nifty_today_open"] = first_open
    agg["nifty_intraday_ret"] = (agg["close"] / agg["nifty_today_open"]) - 1.0
    # NIFTY 5m EMA20 for LONG-side gate
    agg["nifty_ema20"] = agg["close"].ewm(span=NIFTY_UPTREND_EMA_BARS, adjust=False).mean()
    print(f"    NIFTY 5m bars: {len(agg):,}")
    return agg[["ts", "d", "open", "high", "low", "close", "nifty_today_open", "nifty_intraday_ret", "nifty_ema20"]]


def load_daily_for_beta() -> pd.DataFrame:
    """Daily OHLCV for 30-day rolling beta vs NIFTY 50."""
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing")
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date(2023, 9, 1)) & (df["d"] <= date(2024, 12, 31))]
    return df[["symbol", "d", "close"]].copy()


def load_nifty_daily() -> pd.DataFrame:
    """Daily NIFTY close for beta computation."""
    p = _REPO_ROOT / "backtest-cache-download" / "index_ohlcv" / "NSE_NIFTY_50" / "NSE_NIFTY_50_1days.feather"
    df = pd.read_feather(p)
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    return df[["d", "close"]].rename(columns={"close": "nifty_close"})


def compute_beta_table(daily: pd.DataFrame, nifty_daily: pd.DataFrame) -> pd.DataFrame:
    """30-day rolling daily beta per (symbol, d) — using prior 30 days
    (shift(1) to avoid look-ahead).
    """
    print("  computing 30-day rolling betas per (symbol, d) ...")
    df = daily.merge(nifty_daily, on="d", how="inner").sort_values(["symbol", "d"]).reset_index(drop=True)
    df["sret"] = df.groupby("symbol")["close"].transform(lambda s: s.pct_change())
    df["nret"] = df.groupby("symbol")["nifty_close"].transform(lambda s: s.pct_change())

    out_rows: List[dict] = []
    for sym, g in df.groupby("symbol"):
        g = g.sort_values("d").reset_index(drop=True)
        # rolling cov / var with min_periods=BETA_WINDOW_DAYS, then shift(1) to use prior window
        cov = g["sret"].rolling(BETA_WINDOW_DAYS, min_periods=BETA_WINDOW_DAYS).cov(g["nret"])
        var = g["nret"].rolling(BETA_WINDOW_DAYS, min_periods=BETA_WINDOW_DAYS).var()
        beta = (cov / var).shift(1)  # use prior-day rolling window (no look-ahead)
        for d, b in zip(g["d"], beta):
            if pd.notna(b):
                out_rows.append({"symbol": sym, "d": d, "beta_30d": float(b)})
    bdf = pd.DataFrame(out_rows)
    print(f"    beta rows: {len(bdf):,} | symbols: {bdf['symbol'].nunique()}")
    return bdf


def find_triggers(big5m: pd.DataFrame, universe: set, nifty_5m: pd.DataFrame, beta_table: pd.DataFrame) -> pd.DataFrame:
    """Build spread z-score per (sym, day) bar; identify trigger bars."""
    print("  filtering 5m bars to F&O 200 universe ...")
    df = big5m[big5m["symbol"].isin(universe)].copy()
    print(f"    universe-filtered 5m bars: {len(df):,}")

    # cap segment (recorded for reporting, NOT filtered — brief: no cap exclusion)
    df["nse_symbol"] = "NSE:" + df["symbol"].astype(str)
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)

    # join NIFTY 5m features by ts
    nifty_idx = nifty_5m.set_index("ts")[["nifty_today_open", "nifty_intraday_ret", "nifty_ema20", "close"]]
    nifty_idx = nifty_idx.rename(columns={"close": "nifty_close_bar"})
    df = df.set_index("date").join(nifty_idx, how="left").reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["nifty_intraday_ret", "nifty_ema20"])
    print(f"    after NIFTY join: {len(df):,}")

    # join beta
    beta_idx = beta_table.set_index(["symbol", "d"])["beta_30d"]
    df["beta_30d"] = df.set_index(["symbol", "d"]).index.map(beta_idx).values
    df = df.dropna(subset=["beta_30d"])
    print(f"    after beta join: {len(df):,}")

    # Q2 user decision 2026-05-06: ROLLING N-BAR closing-to-closing returns
    # (NOT intraday-since-open). Aligns with QuantInsti EPAT pairs work +
    # global stat-arb convention; regime-stable + comparable across times of
    # day. Window N = SPREAD_ZSCORE_WINDOW_BARS bars (5m each).
    df = df.sort_values(["symbol", "d", "date"])
    N = SPREAD_ZSCORE_WINDOW_BARS

    # Stock N-bar return: (close[t]/close[t-N] - 1). NIFTY same on its 5m.
    df["stock_close_lagN"] = df.groupby(["symbol", "d"])["close"].shift(N)
    df["stock_ret_Nbar"] = df["close"] / df["stock_close_lagN"] - 1.0
    df = df.dropna(subset=["stock_ret_Nbar"])

    # NIFTY N-bar return joined on the same bar timestamp
    nifty_close_idx = nifty_5m.set_index("ts")["close"].rename("nifty_close")
    df = df.set_index("date").join(nifty_close_idx, how="left").reset_index().rename(columns={"index": "date"})
    # NIFTY lag: shift by N bars within day. NIFTY index has one series, so
    # build the lag once globally on the resampled NIFTY series.
    nifty_lag = nifty_5m.set_index("ts")["close"].shift(N).rename("nifty_close_lagN")
    df = df.set_index("date").join(nifty_lag, how="left").reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["nifty_close", "nifty_close_lagN"])
    df["nifty_ret_Nbar"] = df["nifty_close"] / df["nifty_close_lagN"] - 1.0

    df["spread"] = df["stock_ret_Nbar"] - df["beta_30d"] * df["nifty_ret_Nbar"]

    # rolling N-bar mean/std of spread within session for z-score
    df["spread_mean"] = df.groupby(["symbol", "d"])["spread"].transform(
        lambda s: s.rolling(N, min_periods=N).mean()
    )
    df["spread_std"] = df.groupby(["symbol", "d"])["spread"].transform(
        lambda s: s.rolling(N, min_periods=N).std()
    )
    df["spread_z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]
    df = df.dropna(subset=["spread_std", "spread_z"])
    df = df[df["spread_std"] > 0]

    # bar geometry for wick check on trigger bar
    df["range"] = (df["high"] - df["low"]).clip(lower=1e-9)
    df["upper_wick_frac"] = (df["high"] - df["close"]) / df["range"]
    df["lower_wick_frac"] = (df["close"] - df["low"]) / df["range"]

    # active window
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= ENTRY_WINDOW_START_HHMM) & (df["hhmm"] <= ENTRY_WINDOW_END_HHMM)]

    # NIFTY uptrend gate (LONG side)
    df["nifty_uptrend"] = df["nifty_close_bar"] > df["nifty_ema20"]

    short = df[(df["spread_z"] >= SPREAD_Z_THRESHOLD)].copy()
    short["side"] = "SHORT"
    long_ = df[(df["spread_z"] <= -SPREAD_Z_THRESHOLD) & (df["nifty_uptrend"])].copy()
    long_["side"] = "LONG"

    print(f"    SHORT triggers: {len(short):,}")
    print(f"    LONG triggers (NIFTY-uptrend gated): {len(long_):,}")
    return pd.concat([short, long_], ignore_index=True)


def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame, nifty_5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating entries -> targets/stop/time-stop ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }
    nifty_by_d = {d: g.sort_values("ts").reset_index(drop=True) for d, g in nifty_5m.groupby("d")}

    # latch
    triggers = triggers.sort_values(["symbol", "d", "side", "date"])
    triggers = triggers.drop_duplicates(subset=["symbol", "d", "side"], keep="first")

    trades: List[dict] = []
    n_no_next = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]; sd = t["d"]; side = t["side"]
        trig_ts = t["date"]
        trig_high = float(t["high"]); trig_low = float(t["low"])
        beta = float(t["beta_30d"])
        spread_mean = float(t["spread_mean"])
        spread_std = float(t["spread_std"])

        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            n_no_next += 1; continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_next += 1; continue

        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr:
            n_no_next += 1; continue
        trig_idx = idx_arr[0]

        if trig_idx + 1 >= len(day_df):
            n_no_next += 1; continue
        conf = day_df.iloc[trig_idx + 1]

        c_open = float(conf["open"]); c_close = float(conf["close"])
        c_high = float(conf["high"]); c_low = float(conf["low"])
        c_range = max(c_high - c_low, 1e-9)
        prior_close = float(t["close"])
        if side == "SHORT":
            ok = (c_close < c_open) and (c_close < prior_close) and ((c_high - c_close) / c_range >= REVERSAL_WICK_MIN_FRAC)
        else:
            ok = (c_close > c_open) and (c_close > prior_close) and ((c_close - c_low) / c_range >= REVERSAL_WICK_MIN_FRAC)
        if not ok:
            continue

        # Q1 user decision 2026-05-06: ENTRY at NEXT-BAR OPEN (not conf close).
        # Streak/AlgoTest "Signal candle = Trade candle - 1" — confirmation
        # completes at idx+1 close, trade enters at idx+2 open.
        if trig_idx + 2 >= len(day_df):
            n_no_next += 1; continue
        entry_bar = day_df.iloc[trig_idx + 2]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        # hard SL
        if side == "SHORT":
            sl_struct = trig_high * (1.0 + SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 + MIN_STOP_PCT / 100.0)
            hard_sl = max(sl_struct, sl_min)
            stop_distance = hard_sl - entry_price
        else:
            sl_struct = trig_low * (1.0 - SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 - MIN_STOP_PCT / 100.0)
            hard_sl = min(sl_struct, sl_min)
            stop_distance = entry_price - hard_sl
        if stop_distance <= 0:
            continue

        # NIFTY 5m bars indexed for forward walk (need spread z to detect T1/T2)
        nday = nifty_by_d.get(sd)
        if nday is None:
            continue
        nday_idx = nday.set_index("ts")["close"]

        # Walk forward from entry bar (idx+2) — Q1 entry is at its open.
        forward = day_df.iloc[trig_idx + 2: trig_idx + 2 + TIME_STOP_BARS].copy().reset_index(drop=True)
        if forward.empty:
            continue

        exit_ts = None; exit_price = None; exit_reason = None
        hit_t1 = False; t1_exit_price = None; t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]; high = float(bar["high"]); low = float(bar["low"])
            close = float(bar["close"])
            # Q8 breakeven trail: after T1 fills, SL on T2 leg moves to entry_price.
            active_sl = entry_price if hit_t1 else hard_sl
            # SL first
            if side == "SHORT" and high >= active_sl:
                exit_ts = ts; exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break
            if side == "LONG" and low <= active_sl:
                exit_ts = ts; exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            # Q2 user decision: rolling N-bar closing-to-closing returns.
            # spread_z = (spread - trigger_bar_mean) / trigger_bar_std,
            # where mean/std are FROZEN at trigger time (used as fixed
            # reference for the duration of the trade — same as a static
            # z-band reference in pairs trading).
            nifty_close_bar = nday_idx.get(ts)
            if pd.isna(nifty_close_bar) or nifty_close_bar is None:
                nifty_close_bar = float(nday["close"].iloc[-1])
            # find the index in day_df / nday at lag-N for current bar ts
            cur_idx_in_day = trig_idx + 2 + list(forward["date"]).index(ts)
            if cur_idx_in_day - SPREAD_ZSCORE_WINDOW_BARS < 0:
                continue  # not enough history for N-bar return
            stock_close_lagN = float(day_df.iloc[cur_idx_in_day - SPREAD_ZSCORE_WINDOW_BARS]["close"])
            nifty_lag_row = nday[nday["ts"] <= ts].iloc[-(SPREAD_ZSCORE_WINDOW_BARS + 1)] if len(nday[nday["ts"] <= ts]) > SPREAD_ZSCORE_WINDOW_BARS else None
            if nifty_lag_row is None:
                continue
            nifty_close_lagN = float(nifty_lag_row["close"])
            stock_ret = (close / stock_close_lagN) - 1.0
            nifty_ret = (float(nifty_close_bar) / nifty_close_lagN) - 1.0
            spread = stock_ret - beta * nifty_ret
            spread_z_bar = (spread - spread_mean) / spread_std

            # T1 = spread reverts to 0 (z<=0 SHORT, z>=0 LONG)
            # T2 = spread overshoots to -/+ 0.5
            if side == "SHORT":
                if not hit_t1 and spread_z_bar <= 0.0:
                    hit_t1 = True; t1_exit_price = close; t1_exit_ts = ts
                if hit_t1 and spread_z_bar <= -T2_OVERSHOOT_STDEV_MULT:
                    exit_ts = ts; exit_price = close; exit_reason = "t2"
                    break
            else:
                if not hit_t1 and spread_z_bar >= 0.0:
                    hit_t1 = True; t1_exit_price = close; t1_exit_ts = ts
                if hit_t1 and spread_z_bar >= T2_OVERSHOOT_STDEV_MULT:
                    exit_ts = ts; exit_price = close; exit_reason = "t2"
                    break

            if ts.strftime("%H:%M") >= TIME_STOP_HARD_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "time_stop_hard"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"]); exit_reason = "time_stop_bars"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if hit_t1:
            qty_t1 = qty // 2; qty_t2 = qty - qty_t1
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL" if side == "SHORT" else "BUY")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL" if side == "SHORT" else "BUY")
            fee = fee_t1 + fee_t2
            blended_exit = (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            if side == "SHORT":
                realized_pnl = (entry_price - exit_price) * qty
            else:
                realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL" if side == "SHORT" else "BUY")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": sd,
            "symbol": "NSE:" + sym,
            "cap_segment": t["cap_segment"],
            "side": side,
            "trigger_ts": trig_ts,
            "trigger_spread_z": float(t["spread_z"]),
            "beta_30d": beta,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": "spread_z=0",
            "t2_target": f"spread_z=+/-{T2_OVERSHOOT_STDEV_MULT}",
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"\n  no next bar:        {n_no_next}")
    print(f"  traded:             {n_traded}")
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

    print("\n=== index_stock_divergence_revert -- pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer side:")
    short_pf = None; long_pf = None
    for sd, grp in trades.groupby("side"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"  {sd:<6} n={n2:>4} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net:>10,}")
        if sd == "SHORT": short_pf = pf2
        if sd == "LONG":  long_pf = pf2

    print("\nPer cap_segment (diagnostic — universe is full F&O 200):")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>4} PF={pf2:>5} netPnL=Rs.{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{int(grp['net_pnl'].mean()):>6,}")

    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"PF={pf} >= 1.10 -> STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> marginal. Proceed with caveat.")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE candidate. Do not write detector code.")

    if short_pf is not None and long_pf is not None and short_pf > 0:
        ratio = round(long_pf / short_pf, 3) if isinstance(long_pf, (int, float)) else 0.0
        print(f"\nLong/Short PF ratio: {ratio} (gate: >= 0.85 for bidirectional ship)")
        if isinstance(long_pf, (int, float)) and long_pf >= short_pf * 0.85:
            print("  LONG-side passes ship gate.")
        else:
            print("  LONG-side FAILS ship gate -> SHORT-only ship.")


def main():
    big5m = build_full_year_5m()
    universe = load_fno_universe()
    nifty_5m = load_nifty_5m()
    daily = load_daily_for_beta()
    nifty_daily = load_nifty_daily()
    beta_table = compute_beta_table(daily, nifty_daily)

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, nifty_5m, beta_table)
    print(f"\nTotal triggers (both sides, before latch): {len(triggers)}")
    if triggers.empty:
        return

    print("\nSimulating entries -> exits:")
    trades = simulate(triggers, big5m, nifty_5m)
    report(trades)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "index_stock_divergence_revert_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
