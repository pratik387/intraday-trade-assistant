"""Pre-coding sanity check for midsession_momentum_continuation_long candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-
midsession_momentum_continuation_long.md): BEFORE writing detector code,
simulate the 11:00 confirmation-hold LONG continuation with hard 14:00
exit on 24 months (2023-01 .. 2024-12) of 5m enriched feather data.

Mechanic (per brief, §Mechanic):
  At 11:00 IST 5m bar close, evaluate every NSE-liquid stock in the
  data-broad universe. Take a LONG entry on the next bar's open ONLY if
  ALL seven gates G1-G7 hold simultaneously:
    G1 - Intraday return: (close@11:00 - open@09:15) / open@09:15
         in [+2.0%, +4.0%].
    G2 - VWAP-distance: close@11:00 >= vwap@11:00 * 1.005.
    G3 - Sustained-above-VWAP duration: >= 80% of 5m bars from 09:30 to
         11:00 (>= 14 of 17 bars) closed above intraday VWAP.
    G4 - Declining-volume pullback (the disambiguator from breakouts):
         mean(vol_5m, 10:30-11:00) <= 0.7 * mean(vol_5m, 09:30-10:30).
    G5 - No fresh selling: min(low_5m, 10:30-11:00) >= vwap@11:00.
    G6 - Cross-detector exclusion: not a circuit-band T+1 day (heuristic
         mirroring circuit_t1 sanity), not a gap-day with |gap| > 1.5%
         (gap_fade exclusion). Earnings-event exclusion deferred per
         brief Data Engineering plan.
    G7 - Liquidity floor: rolling 20-day median daily volume >= 100K
         shares.
  Entry: NEXT 5m bar open (Streak/AlgoTest convention - eliminates look-
         ahead). Single-bar entry, no re-entry, latch per (symbol, day).
  Stop: vwap@entry-bar * 0.992 (0.8% below entry-bar VWAP). VWAP break
        flips the institutional thesis.
  Target / partials: NONE (no T1/T2). Brief explicitly disables price
                     targets - the trade is timing-driven, hold to 14:00.
  Hard exit: 14:00 IST 5m bar close, market exit regardless of P&L.
  Position sizing: 1R = 0.8%; risk_per_trade = Rs 1,000.

Universe: data-broad NSE-liquid stocks (NO F&O 200 lock, NO cap_segment
pre-lock per brief §Universe). Liquidity floor (G7) is the only universe
filter at sanity.

Decision gate (from brief):
  Net PF >= 1.10 AND n >= 500 AND |WR delta| <= 10pp -> APPROVED
  Net PF in [1.00, 1.10)                            -> marginal, tune
  Net PF < 1.00                                      -> RETIRE
  n < 500                                            -> STRUCTURAL RETIRE

Independence story (acceptance criterion):
  Symbol-day overlap with gap_fade_short  expected < 2%
  Symbol-day overlap with circuit_t1_fade_short expected ~ 0% (G6 forces it)

Usage:
    python tools/sub9_research/sanity_midsession_momentum_continuation_long.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (research-defensible per brief §Mechanic) ----
G1_INTRADAY_RETURN_LO_PCT = 2.0           # G1 lower bound (rules out range-bound noise)
G1_INTRADAY_RETURN_HI_PCT = 4.0           # G1 upper bound (rules out blow-off exhaustion)
G2_VWAP_DISTANCE_RATIO = 1.005            # G2: close >= vwap * 1.005 (>= 0.5% above VWAP)
G3_SUSTAINED_VWAP_FRAC = 0.80             # G3: >= 80% of 09:30-11:00 bars above VWAP
G3_BARS_TOTAL_0930_1100 = 17              # 09:30..10:55 inclusive of 11:00 entry bar = 17 5m bars
G3_BARS_REQUIRED = 14                     # round(0.80 * 17) = 14 bars must close above VWAP
G4_VOL_PULLBACK_RATIO = 0.70              # G4: mean(10:30-11:00 vol) <= 0.70 * mean(09:30-10:30)
G6_CIRCUIT_PRIOR_GAIN_PCT = 4.5           # G6: heuristic mirror of circuit_t1 prior-day gain floor
G6_CIRCUIT_HIGH_TO_CLOSE_RATIO = 0.995    # G6: close ~ high on prior day
G6_GAP_MAX_ABS_PCT = 1.5                  # G6: today's gap > 1.5% absolute -> exclude
G7_LIQUIDITY_MEDIAN_SHARES = 100_000      # G7: rolling 20-day median daily volume >= 100K shares

ENTRY_BAR_HHMM = "11:00"                  # confirmation bar timestamp
HARD_EXIT_HHMM = "14:00"                  # hard market exit timestamp
STOP_VWAP_RATIO = 0.992                   # SL = entry-bar vwap * 0.992 (0.8% below VWAP)
RISK_PER_TRADE_RUPEES = 1000              # match other sub9 sanity scripts


# Brief uses "next bar open" convention (Streak/AlgoTest "Signal candle =
# Trade candle - 1"). 11:00 close is the confirmation; 11:05 open is the fill.
ENTRY_ON_NEXT_BAR_OPEN = True


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_full_5m(years: Tuple[int, ...] = (2023, 2024)) -> pd.DataFrame:
    """DEPRECATED in current sanity flow: full-period concat blows out 16GB
    RAM (24 monthly feathers ~2.4 GB on disk inflate to ~10x in memory).
    Use process_month_by_month() instead. Kept for downstream tools that
    call it explicitly."""
    print(f"  loading {12 * len(years)} monthly 5m feathers ({years[0]}-01 .. {years[-1]}-12) ...")
    parts: List[pd.DataFrame] = []
    for yyyy in years:
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if not mdf.empty:
                parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    big["hhmm"] = big["date"].dt.strftime("%H:%M")
    print(f"  total 5m bars: {len(big):,}")
    return big


def _prepare_month(mdf: pd.DataFrame) -> pd.DataFrame:
    """Strip to columns we need + add d/hhmm, restrict to bars <= 14:00."""
    keep = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap"]
    mdf = mdf[keep]
    # Compute hhmm without copying twice; filter first, then add columns
    hhmm = mdf["date"].dt.strftime("%H:%M")
    mask = hhmm <= "14:00"
    mdf = mdf.loc[mask].reset_index(drop=True)
    mdf["d"] = mdf["date"].dt.date
    mdf["hhmm"] = mdf["date"].dt.strftime("%H:%M")
    return mdf


def load_daily_for_g6_g7() -> pd.DataFrame:
    """Daily OHLCV for G6 prior-day circuit detection + G7 liquidity floor."""
    print("  loading consolidated_daily.feather for G6 (circuit prior-day) + G7 (liquidity) ...")
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing")
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date(2022, 11, 1)) & (df["d"] <= date(2024, 12, 31))]
    df = df[["symbol", "d", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values(["symbol", "d"])

    # G6 prior-day circuit-hit heuristic (mirrors circuit_t1 sanity)
    df["prev_close"] = df.groupby("symbol")["close"].shift(1)
    df["prev_pct_change"] = (df["close"] / df["prev_close"] - 1.0) * 100.0
    df["prev_high_to_close"] = df["close"] / df["high"]
    df["circuit_hit_today"] = (
        (df["prev_pct_change"] >= G6_CIRCUIT_PRIOR_GAIN_PCT)
        & (df["prev_high_to_close"] >= G6_CIRCUIT_HIGH_TO_CLOSE_RATIO)
    )
    # Mark NEXT trading day as "T+1 of a circuit hit" (G6 exclusion target)
    df["circuit_t1_day"] = df.groupby("symbol")["circuit_hit_today"].shift(1).fillna(False)

    # G7 rolling 20-day median daily volume (excluding today, lookback only)
    df["vol_median_20d"] = df.groupby("symbol")["volume"].transform(
        lambda v: v.shift(1).rolling(20).median()
    )

    return df[["symbol", "d", "close", "vol_median_20d", "circuit_t1_day"]]


# ---------------------------------------------------------------------------
# Gate evaluation
# ---------------------------------------------------------------------------
def evaluate_gates(big5m: pd.DataFrame, daily: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """For each (symbol, day), evaluate G1..G7 at 11:00 IST. Return one row
    per (symbol, day) with all gate-pass booleans + bar-level metrics, and
    a funnel-counts dict.

    Funnel here is for in-month evaluation; main() aggregates across months.
    """
    funnel: Dict[str, int] = {}

    # work is already trimmed to <= 14:00 by _prepare_month()
    work = big5m
    print(f"    bars up to 14:00:                          {len(work):,}")

    # ---- Build per-(symbol, day) features for gate evaluation ----
    # Open at 09:15
    open_0915 = (
        work[work["hhmm"] == "09:15"]
        .groupby(["symbol", "d"])["open"].first()
        .rename("open_0915")
    )
    # Close + vwap at 11:00 (entry confirmation bar)
    bar_1100 = (
        work[work["hhmm"] == "11:00"]
        .groupby(["symbol", "d"])
        .agg(close_1100=("close", "first"), vwap_1100=("vwap", "first"))
    )
    # Bars 09:30 -> 10:55 inclusive (G3 sustained-above-VWAP window).
    # Bar at 09:30 is the bar that closes at 09:30 (i.e., 09:30 timestamp = 09:30..09:34
    # close-of-bar in 5m feathers; brief language "09:30 to 11:00" means we count
    # bars whose timestamp is in [09:30, 10:55] which is 17 bars).
    g3_window = work[(work["hhmm"] >= "09:30") & (work["hhmm"] <= "10:55")].copy()
    g3_window["above_vwap"] = (g3_window["close"] > g3_window["vwap"]).astype(int)
    g3_agg = (
        g3_window.groupby(["symbol", "d"])
        .agg(g3_bars_total=("above_vwap", "size"),
             g3_bars_above=("above_vwap", "sum"))
    )
    # G4 volume windows
    g4_morn = work[(work["hhmm"] >= "09:30") & (work["hhmm"] <= "10:25")]
    g4_pull = work[(work["hhmm"] >= "10:30") & (work["hhmm"] <= "10:55")]
    g4_morn_mean = g4_morn.groupby(["symbol", "d"])["volume"].mean().rename("g4_morn_vol")
    g4_pull_mean = g4_pull.groupby(["symbol", "d"])["volume"].mean().rename("g4_pull_vol")
    # G5 lowest low in 10:30-11:00 (inclusive of 10:55 bar; 11:00 itself excluded
    # because it's the trigger bar - we want pullback bars only)
    g5_pull = work[(work["hhmm"] >= "10:30") & (work["hhmm"] <= "10:55")]
    g5_low = g5_pull.groupby(["symbol", "d"])["low"].min().rename("g5_min_low")

    # Today's gap @ 09:15 vs prior-day close (G6 gap exclusion)
    daily_close = daily.set_index(["symbol", "d"])["close"]
    open_0915_df = open_0915.to_frame()
    open_0915_df = open_0915_df.reset_index()
    # prior trading day's close per symbol from daily
    prev_close_map = (
        daily.assign(d_next=daily.groupby("symbol")["d"].shift(-1))
        .dropna(subset=["d_next"])
    )
    prev_close_map["d_next"] = pd.to_datetime(prev_close_map["d_next"]).dt.date
    prev_close_map = prev_close_map.rename(columns={"close": "prev_close"})
    prev_close_map = prev_close_map[["symbol", "d_next", "prev_close"]].rename(columns={"d_next": "d"})

    # Merge everything onto open_0915
    feat = open_0915_df.merge(bar_1100.reset_index(), on=["symbol", "d"], how="inner")
    feat = feat.merge(g3_agg.reset_index(), on=["symbol", "d"], how="left")
    feat = feat.merge(g4_morn_mean.reset_index(), on=["symbol", "d"], how="left")
    feat = feat.merge(g4_pull_mean.reset_index(), on=["symbol", "d"], how="left")
    feat = feat.merge(g5_low.reset_index(), on=["symbol", "d"], how="left")
    feat = feat.merge(prev_close_map, on=["symbol", "d"], how="left")
    feat = feat.merge(
        daily[["symbol", "d", "vol_median_20d", "circuit_t1_day"]],
        on=["symbol", "d"], how="left",
    )

    n0 = len(feat)
    funnel["candidates_with_11:00_bar"] = n0
    print(f"    candidates with 09:15 open + 11:00 bar:    {n0:,}")

    # ---- G7 (liquidity floor) — drops illiquid micro-caps first ----
    g7_pass = feat["vol_median_20d"].fillna(0) >= G7_LIQUIDITY_MEDIAN_SHARES
    feat = feat.loc[g7_pass].copy()
    funnel["G7_liquidity_pass"] = len(feat)
    print(f"    G7 (median 20d vol >= {G7_LIQUIDITY_MEDIAN_SHARES:,}): {len(feat):,}")

    # ---- G1 (intraday return at 11:00) ----
    feat["intraday_pct"] = (feat["close_1100"] / feat["open_0915"] - 1.0) * 100.0
    g1_pass = (feat["intraday_pct"] >= G1_INTRADAY_RETURN_LO_PCT) & (
        feat["intraday_pct"] <= G1_INTRADAY_RETURN_HI_PCT
    )
    feat = feat.loc[g1_pass].copy()
    funnel["G1_intraday_return_in_band"] = len(feat)
    print(f"    G1 (return in [+2%, +4%]):                 {len(feat):,}")

    # ---- G2 (VWAP distance) ----
    g2_pass = feat["close_1100"] >= feat["vwap_1100"] * G2_VWAP_DISTANCE_RATIO
    feat = feat.loc[g2_pass].copy()
    funnel["G2_vwap_distance_pass"] = len(feat)
    print(f"    G2 (close >= vwap * 1.005):                {len(feat):,}")

    # ---- G3 (sustained-above-VWAP) ----
    g3_pass = feat["g3_bars_above"].fillna(0) >= G3_BARS_REQUIRED
    feat = feat.loc[g3_pass].copy()
    funnel["G3_sustained_vwap_pass"] = len(feat)
    print(f"    G3 (>= {G3_BARS_REQUIRED}/{G3_BARS_TOTAL_0930_1100} bars above VWAP):     {len(feat):,}")

    # ---- G4 (declining-volume pullback - the disambiguator) ----
    g4_pass = feat["g4_pull_vol"].fillna(np.inf) <= G4_VOL_PULLBACK_RATIO * feat["g4_morn_vol"].fillna(0)
    # guard: if morn vol is 0 the test is moot - skip
    g4_pass &= feat["g4_morn_vol"].fillna(0) > 0
    feat = feat.loc[g4_pass].copy()
    funnel["G4_declining_vol_pullback"] = len(feat)
    print(f"    G4 (10:30-11:00 vol <= 0.70 * 09:30-10:30): {len(feat):,}")

    # ---- G5 (no fresh selling) ----
    g5_pass = feat["g5_min_low"].fillna(-np.inf) >= feat["vwap_1100"]
    feat = feat.loc[g5_pass].copy()
    funnel["G5_no_fresh_selling"] = len(feat)
    print(f"    G5 (min(low, 10:30-11:00) >= vwap@11:00):  {len(feat):,}")

    # ---- G6 (cross-detector exclusions) ----
    feat["gap_pct"] = (feat["open_0915"] / feat["prev_close"] - 1.0) * 100.0
    g6_circuit_ok = ~feat["circuit_t1_day"].fillna(False)
    g6_gap_ok = feat["gap_pct"].abs().fillna(0) <= G6_GAP_MAX_ABS_PCT
    g6_pass = g6_circuit_ok & g6_gap_ok
    feat = feat.loc[g6_pass].copy()
    funnel["G6_cross_detector_exclusion"] = len(feat)
    print(f"    G6 (no circuit T+1, |gap|<=1.5%):           {len(feat):,}")

    return feat, funnel


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(qualified: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    """For each qualifying (symbol, day), enter LONG at 11:05 open (or
    11:00 close if ENTRY_ON_NEXT_BAR_OPEN=False) and hold to 14:00 close.
    Stop = entry-bar VWAP * 0.992. No targets - hard exit at 14:00.
    """
    print("  simulating LONG entries -> 14:00 hard exit (or VWAP-stop) ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    n_no_entry_bar = n_no_exit_bar = n_traded = 0

    for _, row in qualified.iterrows():
        sym = row["symbol"]; sd = row["d"]
        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            n_no_entry_bar += 1; continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_entry_bar += 1; continue

        idx_arr = day_df.index[day_df["hhmm"] == ENTRY_BAR_HHMM].tolist()
        if not idx_arr:
            n_no_entry_bar += 1; continue
        sig_idx = idx_arr[0]

        if ENTRY_ON_NEXT_BAR_OPEN:
            if sig_idx + 1 >= len(day_df):
                n_no_entry_bar += 1; continue
            entry_bar = day_df.iloc[sig_idx + 1]
            entry_price = float(entry_bar["open"])
            entry_ts = entry_bar["date"]
        else:
            entry_bar = day_df.iloc[sig_idx]
            entry_price = float(entry_bar["close"])
            entry_ts = entry_bar["date"]

        # Stop = entry-bar VWAP * 0.992 (brief Mechanic step 4)
        entry_vwap = float(entry_bar["vwap"])
        hard_sl = entry_vwap * STOP_VWAP_RATIO
        stop_distance = entry_price - hard_sl
        if stop_distance <= 0:
            # Entry below VWAP - thesis broken before fill; skip
            continue

        # Walk forward to 14:00 close (inclusive). Hard-exit dominates.
        forward = day_df[
            (day_df["date"] >= entry_ts) & (day_df["hhmm"] <= HARD_EXIT_HHMM)
        ].copy()
        if forward.empty:
            n_no_exit_bar += 1; continue

        exit_ts = None; exit_price = None; exit_reason = None
        for _, bar in forward.iterrows():
            ts = bar["date"]
            low = float(bar["low"])
            close = float(bar["close"])
            # Stop-check first (intrabar low <= hard_sl)
            if low <= hard_sl:
                exit_ts = ts; exit_price = hard_sl; exit_reason = "vwap_stop"
                break
            # 14:00 hard exit (close-of-bar)
            if bar["hhmm"] == HARD_EXIT_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "hard_exit_1400"
                break

        if exit_price is None:
            # Last bar of session before 14:00 (data truncation safeguard)
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"])
            exit_reason = "last_bar_safeguard"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        realized_pnl = (exit_price - entry_price) * qty   # LONG
        fee = calc_fee(entry_price, exit_price, qty, "BUY")
        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": sd,
            "symbol": "NSE:" + sym,
            "cap_segment": get_cap_segment("NSE:" + sym),
            "side": "LONG",
            "open_0915": float(row["open_0915"]),
            "close_1100": float(row["close_1100"]),
            "vwap_1100": float(row["vwap_1100"]),
            "intraday_pct": float(row["intraday_pct"]),
            "gap_pct": float(row["gap_pct"]) if pd.notna(row["gap_pct"]) else float("nan"),
            "g3_bars_above": int(row["g3_bars_above"]) if pd.notna(row["g3_bars_above"]) else 0,
            "vol_median_20d": float(row["vol_median_20d"]) if pd.notna(row["vol_median_20d"]) else 0.0,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "entry_vwap": entry_vwap,
            "hard_sl": hard_sl,
            "exit_ts": exit_ts,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"\n  no entry bar:   {n_no_entry_bar}")
    print(f"  no exit bar:    {n_no_exit_bar}")
    print(f"  traded:         {n_traded}")
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def report(trades: pd.DataFrame, funnel: Dict[str, int]) -> None:
    print("\n=== midsession_momentum_continuation_long -- pre-coding sanity check ===")
    print("\nGate funnel (per-gate drop):")
    prev = None
    for k, v in funnel.items():
        drop_pct = ""
        if prev is not None and prev > 0:
            drop_pct = f"  (-{(prev - v) / prev * 100:5.1f}%)"
        print(f"  {k:<32} n={v:>9,}{drop_pct}")
        prev = v

    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades - STRUCTURAL RETIRE")
        return
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)

    print(f"\nPeriod: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net = int(grp["net_pnl"].sum())
        print(f"  {str(cap):<12} n={n2:>5} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net:>10,}")

    print("\nPer month:")
    trades["ym"] = pd.to_datetime(trades["T1_entry_date"]).dt.strftime("%Y-%m")
    for ym, grp in trades.groupby("ym"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {ym:<8} n={n2:>4} PF={pf2:>5} netPnL=Rs.{net:>9,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        n2 = len(grp)
        avg = int(grp["net_pnl"].mean())
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        print(f"  {rsn:<22} n={n2:>5} avg_net=Rs.{avg:>6,} wr={wr2}%")

    print("\n--- SAMPLE-SIZE DIAGNOSTIC ---")
    if n >= 500:
        print(f"n={n} >= 500 floor -> sample-size OK")
    else:
        print(f"n={n} < 500 floor -> STRUCTURAL RETIRE (gates over-restrictive or asymmetry too thin)")

    print("\n--- VERDICT ---")
    if n < 500:
        print(f"STRUCTURAL RETIRE: n={n} < 500.")
    elif pf >= 1.10:
        print(f"PF={pf} >= 1.10 AND n={n} >= 500 -> APPROVED for detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> marginal; tune G1 band or G4 threshold and re-run.")
    else:
        print(f"PF={pf} < 1.00 -> THESIS RETIRE. Institutional-accumulation thesis fails net-fee.")


def overlap_check(trades: pd.DataFrame) -> None:
    """Compute (symbol, day) overlap with circuit_t1_fade_short and a
    reconstructed gap_fade_short approximation. Brief expects:
      vs gap_fade_short:    < 2%
      vs circuit_t1_fade_short: ~ 0% (G6 forces it)
    """
    if trades.empty:
        return
    print("\n--- INDEPENDENCE / SYMBOL-DAY OVERLAP CHECK ---")
    ours = set(zip(trades["symbol"].astype(str), pd.to_datetime(trades["T1_entry_date"]).dt.date))
    n_ours = len(ours)

    # circuit_t1_fade_short: read trades CSV directly
    ct1_path = _REPO_ROOT / "reports" / "sub9_sanity" / "circuit_t1_fade_short_trades.csv"
    if ct1_path.exists():
        ct1 = pd.read_csv(ct1_path)
        ct1["T1_entry_date"] = pd.to_datetime(ct1["T1_entry_date"]).dt.date
        ct1_set = set(zip(ct1["symbol"].astype(str), ct1["T1_entry_date"]))
        inter = ours & ct1_set
        pct = (len(inter) / n_ours * 100.0) if n_ours else 0.0
        print(f"  vs circuit_t1_fade_short: {len(inter)}/{n_ours} ({pct:.2f}%) "
              f"[expected ~0%; brief G6 explicitly excludes circuit T+1 days]")
    else:
        print(f"  vs circuit_t1_fade_short: skipped (no trades CSV at {ct1_path})")

    # gap_fade_short: no trades CSV exists, reconstruct from disk.
    # gap_fade_short fires at 09:15-09:30 on a gap-down >= 1.5% with
    # confirmation; for OVERLAP measurement we use a permissive proxy:
    # gap-down >= 1.5% (negative). G6 already excludes |gap| > 1.5% so
    # the overlap should be ~0% by construction. Verify empirically.
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if daily_path.exists():
        d = pd.read_feather(daily_path)
        d["ts"] = pd.to_datetime(d["ts"])
        if d["ts"].dt.tz is not None:
            d["ts"] = d["ts"].dt.tz_localize(None)
        d["d"] = d["ts"].dt.date
        d = d.sort_values(["symbol", "d"]).copy()
        d["prev_close"] = d.groupby("symbol")["close"].shift(1)
        d["gap_pct"] = (d["open"] / d["prev_close"] - 1.0) * 100.0
        gap_set = set(
            zip(d.loc[d["gap_pct"] <= -1.5, "symbol"].astype(str).map(lambda s: "NSE:" + s),
                d.loc[d["gap_pct"] <= -1.5, "d"])
        )
        inter2 = ours & gap_set
        pct2 = (len(inter2) / n_ours * 100.0) if n_ours else 0.0
        print(f"  vs gap_fade_short (proxy: gap-down>=1.5%): {len(inter2)}/{n_ours} "
              f"({pct2:.2f}%) [expected <2%; brief G6 excludes |gap|>1.5%]")
    else:
        print(f"  vs gap_fade_short: skipped (no daily feather at {daily_path})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    daily = load_daily_for_g6_g7()

    all_trades: List[pd.DataFrame] = []
    agg_funnel: Dict[str, int] = {}

    months = [(y, m) for y in (2023, 2024) for m in range(1, 13)]
    import gc
    for (yyyy, mm) in months:
        path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not path.exists():
            continue
        print(f"\n========== {yyyy:04d}-{mm:02d} ==========")
        mdf_raw = _load_5m_for_month(yyyy, mm)
        if mdf_raw.empty:
            continue
        mdf = _prepare_month(mdf_raw)
        del mdf_raw
        gc.collect()
        print(f"  bars (<=14:00): {len(mdf):,}  symbols: {mdf['symbol'].nunique()}")

        # Subset daily to symbols in this month + the month's date range +
        # 30 prior days to keep G7 rolling-20d valid at the start.
        d_min = mdf["d"].min(); d_max = mdf["d"].max()
        d_lookback = d_min - pd.Timedelta(days=45)
        d_lookback = d_lookback.date() if hasattr(d_lookback, "date") else d_lookback
        sym_set = set(mdf["symbol"].unique())
        d_subset = daily[(daily["symbol"].isin(sym_set)) & (daily["d"] <= d_max)].copy()

        print("  evaluating gates G1-G7 ...")
        qualified, funnel = evaluate_gates(mdf, d_subset)
        for k, v in funnel.items():
            agg_funnel[k] = agg_funnel.get(k, 0) + int(v)
        print(f"  qualified (symbol, day) after G1-G7: {len(qualified):,}")

        if qualified.empty:
            del mdf, qualified
            continue

        # Simulate using this month's bars only (intraday trades, no cross-month dep)
        trades_m = simulate(qualified, mdf)
        if not trades_m.empty:
            all_trades.append(trades_m)
        del mdf, qualified, trades_m
        gc.collect()

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    report(trades, agg_funnel)
    overlap_check(trades)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "midsession_momentum_continuation_long_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
