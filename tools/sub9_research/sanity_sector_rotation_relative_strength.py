"""Pre-coding sanity check for sector_rotation_relative_strength candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-06-sub-project-9-brief-sector_
rotation_relative_strength.md). Mechanic:
  - T-1 EOD sector strength rank: 5-day return on each of 10 NSE sectoral indices,
    rank top 3 / bottom 3
  - T+0 5m bar 11:00-15:00: stock_5m_ret vs sector_5m_ret divergence
  - LONG: top quartile + sector positive + stock < sector*0.5 (under-performer in strong sector)
  - SHORT: bottom quartile + sector negative + stock > sector*0.5 (over-performer in weak sector)
  - 0.7% hard stop, T1=1R, T2=2R, breakeven trail, time-stop 14:45
  - Anti-noise: |sector_5m_ret| >= 0.0015 (15 bps)
  - Confirmation: vol >= 1.2 x rolling 30-bar avg + reversal candle
  - Latch per (symbol, day, side)

Universe: F&O 200 mid+small_cap, mappable via assets/stock_sector_map.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment              # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee     # noqa: E402


# Round-4 loosened params 2026-05-06. Original brief restrictions caused
# n=59 across 2yr (only ENERGY + REALTY had any triggers because F&O 200
# mid+small_cap = 4 stocks total). Loosened:
#   - ALLOWED_CAPS = all 3 (F&O 200 is large_cap-dominant by structure)
#   - VOL_CONFIRM_MULTIPLIER = 1.0 (drop the 1.2x volume gate; uncited in brief)
#   - SECTOR_NOISE_THRESHOLD = 0.001 (10bps; was 15bps)
#   - DIVERGENCE_RATIO = 0.5 KEPT (loosening risks losing signal entirely)
#   - TOP/BOTTOM quartile 3 each KEPT (strict published convention)
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}
ENTRY_WINDOW_START_HHMM = "11:00"
ENTRY_WINDOW_END_HHMM = "15:00"
TIME_STOP_HHMM = "14:45"
STOP_PCT = 0.007
T1_R_MULTIPLE = 1.0
T2_R_MULTIPLE = 2.0
STOP_R_MULTIPLE = 1.0   # scales STOP_PCT (1.0 = locked 0.7%); CLI overrides
SECTOR_STRENGTH_LOOKBACK_DAYS = 5
TOP_QUARTILE_N = 3
BOTTOM_QUARTILE_N = 3
DIVERGENCE_RATIO = 0.5
SECTOR_NOISE_THRESHOLD = 0.001              # was 0.0015
VOL_CONFIRM_MULTIPLIER = 1.0                # was 1.2 (drop)
VOL_CONFIRM_WINDOW = 30
USE_BREAKEVEN_TRAIL_AFTER_T1 = True
RISK_PER_TRADE_RUPEES = 1000

# 10 sectoral indices that NIFTY publishes
SECTOR_INDICES = [
    "NSE_NIFTY_AUTO", "NSE_NIFTY_BANK", "NSE_NIFTY_ENERGY",
    "NSE_NIFTY_FIN_SERVICE", "NSE_NIFTY_FMCG", "NSE_NIFTY_IT",
    "NSE_NIFTY_METAL", "NSE_NIFTY_PHARMA", "NSE_NIFTY_PSU_BANK", "NSE_NIFTY_REALTY",
]


_MIN_COLS_5M = ["symbol", "date", "open", "high", "low", "close", "volume", "vwap"]


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    # Project to minimal column set; full enriched has ~14 columns and the
    # consolidated copy during sort+reset_index OOMs at ~3GB on 47M rows.
    return pd.read_feather(path, columns=_MIN_COLS_5M)


def build_full_period_5m() -> pd.DataFrame:
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")
    parts: List[pd.DataFrame] = []
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if not mdf.empty:
                parts.append(mdf)
    # Skip the global sort+reset_index — both find_triggers and simulate re-sort
    # by symbol+date downstream, so the global sort is redundant and OOMed at ~3GB.
    big = pd.concat(parts, ignore_index=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_fno_universe() -> set:
    df = pd.read_csv(_REPO_ROOT / "assets" / "fno_liquid_200.csv")
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_sector_map() -> Dict[str, str]:
    path = _REPO_ROOT / "assets" / "stock_sector_map.json"
    m = json.load(open(path))
    # Strip NSE: prefix to match 5m feather symbol column. Skip NIFTY 50 (broad-market).
    out = {k.replace("NSE:", ""): v for k, v in m.items() if v in SECTOR_INDICES}
    print(f"  stock_sector_map: {len(out)} stocks mapped to 10 sub-sectors (NIFTY 50 excluded)")
    return out


def _load_index_5m(idx: str) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "index_ohlcv" / idx / f"{idx}_1minutes.feather"
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= pd.Timestamp("2022-12-15"))
            & (df["date"] < pd.Timestamp("2025-01-01"))].set_index("date")
    return df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna().reset_index()


def build_sector_strength() -> pd.DataFrame:
    """For each (date, sector), compute T-1 5-day return. Rank top 3 / bottom 3 each day."""
    print("  building T-1 sector strength rank ...")
    parts = []
    for idx in SECTOR_INDICES:
        d5 = _load_index_5m(idx)
        d5["d"] = d5["date"].dt.date
        daily = d5.groupby("d")["close"].last().reset_index()
        daily["sector"] = idx
        parts.append(daily)
    daily = pd.concat(parts, ignore_index=True).sort_values(["sector", "d"])
    daily["close_lag"] = daily.groupby("sector")["close"].shift(SECTOR_STRENGTH_LOOKBACK_DAYS)
    daily["ret_5d"] = (daily["close"] / daily["close_lag"]) - 1.0
    daily = daily.dropna(subset=["ret_5d"])

    # Per-day, rank sectors and assign tag. Use T-1 rank applied to T+0.
    daily["rank"] = daily.groupby("d")["ret_5d"].rank(method="first", ascending=False)

    def tag(r):
        if r <= TOP_QUARTILE_N:    return "top_quartile"
        if r >= 11 - BOTTOM_QUARTILE_N: return "bottom_quartile"
        return "middle"
    daily["strength_tag"] = daily["rank"].apply(tag)

    # Use T-1 rank applied to T+0: shift the strength_tag forward by 1 day
    daily = daily.sort_values(["sector", "d"])
    daily["strength_tag_t0"] = daily.groupby("sector")["strength_tag"].shift(1)
    daily = daily.dropna(subset=["strength_tag_t0"])
    print(f"    sector-day rows: {len(daily):,}")
    return daily[["sector", "d", "strength_tag_t0", "rank"]]


def build_sector_5m_returns() -> pd.DataFrame:
    """For each sector, compute 5m bar return = (close - open) / open."""
    print("  building sector 5m returns ...")
    parts = []
    for idx in SECTOR_INDICES:
        d5 = _load_index_5m(idx)
        d5["sector"] = idx
        d5["sector_5m_ret"] = (d5["close"] - d5["open"]) / d5["open"]
        d5["d"] = d5["date"].dt.date
        d5["hhmm"] = d5["date"].dt.strftime("%H:%M")
        parts.append(d5[["sector", "date", "d", "hhmm", "sector_5m_ret"]])
    out = pd.concat(parts, ignore_index=True)
    print(f"    sector 5m bars: {len(out):,}")
    return out


def find_triggers(big5m: pd.DataFrame, universe: set, sector_map: Dict[str, str],
                  strength: pd.DataFrame, sector_5m: pd.DataFrame) -> pd.DataFrame:
    print("  filtering bars + computing 5m returns + joining sector context ...")
    df = big5m[big5m["symbol"].isin(universe)].copy()
    df["nse_symbol"] = "NSE:" + df["symbol"]
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)].copy()
    df["sector"] = df["symbol"].map(sector_map)
    df = df.dropna(subset=["sector"])
    print(f"    universe + cap_segment + sector-mapped bars: {len(df):,}")

    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= ENTRY_WINDOW_START_HHMM) & (df["hhmm"] <= ENTRY_WINDOW_END_HHMM)].copy()

    # stock 5m return = (close - open) / open of that bar
    df["stock_5m_ret"] = (df["close"] - df["open"]) / df["open"]

    # rolling 30-bar avg vol per (symbol, day) excl current bar
    df = df.sort_values(["symbol", "d", "date"]).reset_index(drop=True)
    df["vol_avg_30"] = df.groupby(["symbol", "d"])["volume"].transform(
        lambda s: s.shift(1).rolling(VOL_CONFIRM_WINDOW, min_periods=VOL_CONFIRM_WINDOW).mean()
    )
    df = df.dropna(subset=["vol_avg_30"])
    df = df[df["volume"] >= df["vol_avg_30"] * VOL_CONFIRM_MULTIPLIER].copy()
    print(f"    after volume confirm (>=1.2x 30-bar avg): {len(df):,}")

    # join sector strength tag
    sm = strength.set_index(["sector", "d"])["strength_tag_t0"]
    df["strength_tag"] = df.set_index(["sector", "d"]).index.map(sm).values
    df = df[df["strength_tag"].isin(["top_quartile", "bottom_quartile"])].copy()
    print(f"    after sector strength filter (top/bottom quartile): {len(df):,}")

    # join sector 5m return on (sector, date)
    sec5 = sector_5m.set_index(["sector", "date"])["sector_5m_ret"]
    df["sector_5m_ret"] = df.set_index(["sector", "date"]).index.map(sec5).values
    df = df.dropna(subset=["sector_5m_ret"])
    df = df[df["sector_5m_ret"].abs() >= SECTOR_NOISE_THRESHOLD].copy()
    print(f"    after sector 5m noise gate (>=15bps): {len(df):,}")

    # prior bar high/low for reversal-candle confirmation
    df["prev_low"] = df.groupby(["symbol", "d"])["low"].shift(1)
    df["prev_high"] = df.groupby(["symbol", "d"])["high"].shift(1)
    df = df.dropna(subset=["prev_low", "prev_high"])

    # LONG: top_quartile + sector positive + stock under-performs + bar low > prev low (no fresh low)
    long_ = df[
        (df["strength_tag"] == "top_quartile")
        & (df["sector_5m_ret"] > 0)
        & (df["stock_5m_ret"] < df["sector_5m_ret"] * DIVERGENCE_RATIO)
        & (df["low"] > df["prev_low"])
    ].copy()
    long_["side"] = "LONG"

    # SHORT: bottom_quartile + sector negative + stock over-performs + bar high < prev high (no fresh high)
    short = df[
        (df["strength_tag"] == "bottom_quartile")
        & (df["sector_5m_ret"] < 0)
        & (df["stock_5m_ret"] > df["sector_5m_ret"] * DIVERGENCE_RATIO)
        & (df["high"] < df["prev_high"])
    ].copy()
    short["side"] = "SHORT"

    print(f"    LONG triggers: {len(long_):,} | SHORT triggers: {len(short):,}")
    return pd.concat([long_, short], ignore_index=True)


def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating exits ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    triggers = triggers.sort_values(["symbol", "d", "side", "date"])
    triggers = triggers.drop_duplicates(subset=["symbol", "d", "side"], keep="first")

    trades: List[dict] = []
    for _, t in triggers.iterrows():
        sym = t["symbol"]; sd = t["d"]; side = t["side"]
        trig_ts = t["date"]

        sym_df = days_per_sym.get(sym)
        if sym_df is None: continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty: continue
        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr or idx_arr[0] + 1 >= len(day_df):
            continue
        trig_idx = idx_arr[0]

        entry_bar = day_df.iloc[trig_idx + 1]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        eff_stop_pct = STOP_PCT * STOP_R_MULTIPLE
        if side == "SHORT":
            hard_sl = entry_price * (1.0 + eff_stop_pct)
            stop_distance = hard_sl - entry_price
            t1_target = entry_price - T1_R_MULTIPLE * stop_distance
            t2_target = entry_price - T2_R_MULTIPLE * stop_distance
        else:
            hard_sl = entry_price * (1.0 - eff_stop_pct)
            stop_distance = entry_price - hard_sl
            t1_target = entry_price + T1_R_MULTIPLE * stop_distance
            t2_target = entry_price + T2_R_MULTIPLE * stop_distance
        if stop_distance <= 0:
            continue

        forward = day_df.iloc[trig_idx + 1:].copy()
        forward["hhmm"] = forward["date"].dt.strftime("%H:%M")

        exit_ts = None; exit_price = None; exit_reason = None
        hit_t1 = False; t1_exit_price = None

        for _, bar in forward.iterrows():
            ts = bar["date"]; high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
            active_sl = (entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl)
            if side == "SHORT":
                if high >= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"; break
                if not hit_t1 and low <= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target
                if hit_t1 and low <= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"; break
            else:
                if low <= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"; break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target
                if hit_t1 and high >= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"; break
            if bar["hhmm"] >= TIME_STOP_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "time_stop"; break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"]); exit_reason = "time_stop"

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
            fee = (calc_fee(entry_price, t1_exit_price, qty_t1, "SELL" if side == "SHORT" else "BUY")
                   + calc_fee(entry_price, exit_price, qty_t2, "SELL" if side == "SHORT" else "BUY"))
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
            "sector": t["sector"],
            "strength_tag": t["strength_tag"],
            "side": side,
            "stock_5m_ret": float(t["stock_5m_ret"]),
            "sector_5m_ret": float(t["sector_5m_ret"]),
            "trigger_ts": trig_ts,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "t2_target": t2_target,
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
    print(f"  simulated trades: {len(trades)}")
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES]")
        return
    n = len(trades); npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum(); losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)
    print("\n=== sector_rotation_relative_strength -- pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n} | WR = {wr}% | Gross = Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees: Rs.{int(trades['fee'].sum()):,} | NET PnL: Rs.{int(npnl.sum()):,}")
    print(f"NET PF: {pf} | NET Sharpe (daily): {sharpe}")

    short_pf = None; long_pf = None
    print("\nPer side:")
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

    print("\nPer sector:")
    for sec, grp in trades.groupby("sector"):
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {sec:<22} n={len(grp):>4} PF={pf2:>5} netPnL=Rs.{net:>10,}")

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={len(grp):>4} PF={pf2:>5} netPnL=Rs.{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{int(grp['net_pnl'].mean()):>6,}")

    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"PF={pf} >= 1.10 -> STRONG PROCEED.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> marginal.")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE.")

    if short_pf is not None and long_pf is not None and short_pf > 0:
        ratio = round(long_pf / short_pf, 3)
        print(f"\nLong/Short PF ratio: {ratio} (gate: >= 0.85 for bidirectional ship)")


def _pf(s: pd.Series) -> float:
    g = s[s > 0].sum()
    l = s[s < 0].abs().sum()
    return float(g / l) if l > 0 else float("inf")


def _hl_bucket(t) -> str:
    # trigger_ts hour-minute bucket
    h = t.hour * 100 + t.minute
    if h < 1130:
        return "morning"
    if h < 1330:
        return "midday"
    return "afternoon"


def write_artifacts(trades: pd.DataFrame, out_dir: Path,
                    start_d: date, end_d: date) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_dir / "trades.csv", index=False)

    if trades.empty:
        summary = {
            "period_start": str(start_d), "period_end": str(end_d),
            "n_trades": 0, "gross_pf": None, "net_pf": None,
            "sharpe_annualized": 0.0, "n_cells_passing_mining_gate": 0,
            "top_cell_net_pf": None, "passing_cells": [],
            "params": {"t1_r_mult": T1_R_MULTIPLE, "t2_r_mult": T2_R_MULTIPLE,
                       "stop_r_mult": STOP_R_MULTIPLE, "stop_pct_base": STOP_PCT},
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        return summary

    # Add cell tags (sector + hl_bucket from trigger time + side + cap_segment)
    trades = trades.copy()
    trades["entry_ts_dt"] = pd.to_datetime(trades["entry_ts"])
    trades["hl_bucket"] = trades["entry_ts_dt"].apply(_hl_bucket)

    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    if daily.std(ddof=0) > 0:
        sharpe = float(daily.mean() / daily.std(ddof=0) * np.sqrt(252))
    else:
        sharpe = 0.0

    gross_pf = _pf(trades["realized_pnl"])
    net_pf = _pf(trades["net_pnl"])

    # Cell mining: sector x hl_bucket x side x cap_segment, n>=30
    cell_rows = []
    for (sec, hl, side, cap), g in trades.groupby(
            ["sector", "hl_bucket", "side", "cap_segment"]):
        if len(g) < 30:
            continue
        d = g.groupby("T1_entry_date")["net_pnl"].sum()
        sh = float(d.mean() / d.std(ddof=0) * np.sqrt(252)) if d.std(ddof=0) > 0 else 0.0
        cell_rows.append({
            "sector": sec, "hl_bucket": hl, "side": side, "cap_segment": cap,
            "n_trades": int(len(g)),
            "gross_pf": _pf(g["realized_pnl"]),
            "net_pf": _pf(g["net_pnl"]),
            "net_pnl": float(g["net_pnl"].sum()),
            "sharpe": sh,
            "win_rate": float((g["net_pnl"] > 0).mean()),
        })
    if cell_rows:
        cell_df = pd.DataFrame(cell_rows).sort_values("net_pf", ascending=False)
    else:
        cell_df = pd.DataFrame(columns=["sector", "hl_bucket", "side", "cap_segment",
                                         "n_trades", "gross_pf", "net_pf", "net_pnl",
                                         "sharpe", "win_rate"])
    cell_df.to_csv(out_dir / "by_cell.csv", index=False)

    cell_mining = cell_df[
        (cell_df["n_trades"] >= 200) & (cell_df["net_pf"] >= 1.20) & (cell_df["sharpe"] > 0.5)
    ] if not cell_df.empty else cell_df
    top_cell_net_pf = float(cell_df["net_pf"].iloc[0]) if not cell_df.empty else None

    summary = {
        "period_start": str(start_d),
        "period_end": str(end_d),
        "n_trades": int(len(trades)),
        "gross_pf": gross_pf,
        "net_pf": net_pf,
        "gross_pnl_inr": float(trades["realized_pnl"].sum()),
        "net_pnl_inr": float(trades["net_pnl"].sum()),
        "sharpe_annualized": sharpe,
        "win_rate": float((trades["net_pnl"] > 0).mean()),
        "n_cells_passing_mining_gate": int(len(cell_mining)),
        "top_cell_net_pf": top_cell_net_pf,
        "passing_cells": cell_mining.head(30).to_dict("records") if not cell_mining.empty else [],
        "params": {
            "t1_r_mult": T1_R_MULTIPLE, "t2_r_mult": T2_R_MULTIPLE,
            "stop_r_mult": STOP_R_MULTIPLE, "stop_pct_base": STOP_PCT,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def main():
    global T1_R_MULTIPLE, T2_R_MULTIPLE, STOP_R_MULTIPLE
    p = argparse.ArgumentParser(description="sector_rotation_relative_strength sanity")
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--t1-r-mult", type=float, default=T1_R_MULTIPLE,
                   help="T1 target as R-multiple (default 1.0)")
    p.add_argument("--t2-r-mult", type=float, default=T2_R_MULTIPLE,
                   help="T2 target as R-multiple (default 2.0)")
    p.add_argument("--stop-r-mult", type=float, default=STOP_R_MULTIPLE,
                   help="Hard-stop R scale (1.0 = locked 0.7 pct)")
    p.add_argument("--out-suffix", default="",
                   help="suffix appended to output dir name")
    args = p.parse_args()

    T1_R_MULTIPLE = float(args.t1_r_mult)
    T2_R_MULTIPLE = float(args.t2_r_mult)
    STOP_R_MULTIPLE = float(args.stop_r_mult)

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end_d > date(2024, 12, 31):
        print("ERROR: end past Discovery (2024-12-31); refusing OOS/Holdout reads")
        return 2
    print(f"params: t1={T1_R_MULTIPLE}R, t2={T2_R_MULTIPLE}R, "
          f"stop={STOP_R_MULTIPLE}x{STOP_PCT*100:.2f}%")
    print(f"period: {start_d}..{end_d}")

    out_dir_name = "sector_rotation_relative_strength"
    if args.out_suffix:
        out_dir_name += args.out_suffix
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity" / out_dir_name

    big5m = build_full_period_5m()
    # Sentinel: clip to the requested Discovery window
    big5m = big5m[(big5m["d"] >= start_d) & (big5m["d"] <= end_d)].copy()
    print(f"  bars in window: {len(big5m):,}")

    universe = load_fno_universe()
    sector_map = load_sector_map()
    strength = build_sector_strength()
    sector_5m = build_sector_5m_returns()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, sector_map, strength, sector_5m)
    print(f"\nTotal triggers (before latch): {len(triggers)}")

    if triggers.empty:
        write_artifacts(pd.DataFrame(), out_dir, start_d, end_d)
        print(f"\nNO TRIGGERS. Wrote empty summary to {out_dir}")
        return 0

    trades = simulate(triggers, big5m)
    report(trades)
    summary = write_artifacts(trades, out_dir, start_d, end_d)

    print(f"\n=== ARTIFACTS ===")
    print(f"  out_dir       : {out_dir}")
    print(f"  trades.csv    : {len(trades)} rows")
    print(f"  by_cell.csv   : {summary['n_cells_passing_mining_gate']} cells passing mining gate")
    print(f"  net_pf        : {summary['net_pf']}")
    print(f"  sharpe        : {summary['sharpe_annualized']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
