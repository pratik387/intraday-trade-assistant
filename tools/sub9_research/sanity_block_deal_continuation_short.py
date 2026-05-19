"""Pre-coding sanity check for `block_deal_followthrough_short`.

Hypothesis (sub-9 §4 candidate H — sell-block CONTINUATION variant):
  NSE block-deal SELL disclosures on illiquid mid/small caps are informed-
  selling events. When block_qty is meaningful relative to the session's
  traded volume (>=5%), the institutional seller's remaining inventory
  continues to be worked into T+1, producing a predictable T+1 negative
  drift.

This is structurally DIFFERENT from `sanity_nse_block_deal_counter_flow.py`:
  - Counter-flow: large-cap F&O 200, both sides, fade direction. PF 0.74-0.86.
  - Continuation: mid/small-cap, SELL-only, follow direction.

Pre-registered cells (locked, per
`specs/2026-05-14-research-block_deal_followthrough.md`):
  - buy_or_sell == "SELL"
  - side_total_cr >= 5.0 (notional floor)
  - block_qty / avg_T0_session_volume >= 5% (volume-shock floor)
  - cap_segment in ["mid_cap", "small_cap"]
  - latch: one fire per (symbol, T+1, side="SHORT")
  - F&O 200 NOT required

Ship gates (gauntlet-v2):
  - n >= 100
  - net PF >= 1.20
  - Sharpe (daily) > 0
  - per-month stability: majority of trading months net-PF > 1.0
  - cap-segment cross-check: mid_cap AND small_cap pass PF >= 1.15

Falsifiers:
  1. T+0 EOD SHORT control PF >= T+1 PF -> leakage, RETIRE
  2. n < 100 post-latch -> DATA-UNAVAILABLE
  3. Discovery year-over-year |PF drift| > 0.4 -> regime-unstable, defer
  4. large_cap rerun PF >= small/mid PF -> not illiquidity-driven, RETIRE

Usage:
    .venv/Scripts/python tools/sub9_research/sanity_block_deal_continuation_short.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params per brief ----
DISCOVERY_START = date(2023, 1, 5)
DISCOVERY_END   = date(2024, 12, 31)

# Cell filters
MIN_SIDE_TOTAL_CR = 5.0       # notional floor (Cr)
MIN_VOL_RATIO_PCT = 5.0       # block_qty / day_volume >= 5%
ALLOWED_CAP_SEGMENTS = {"mid_cap", "small_cap"}

# Entry mechanic
T1_FIRST_BAR_HHMM = "09:15"
T1_ENTRY_BAR_HHMM = "09:25"
TIME_STOP_HHMM    = "14:30"
T0_CONTROL_EOD_HHMM   = "15:10"
T0_CONTROL_HOLD_BARS  = 1

HARD_STOP_PCT = 1.5
MIN_STOP_PCT  = 1.0

T1_R_MULTIPLE = 1.0
T2_R_MULTIPLE = 2.0
T1_PARTIAL_FRACTION = 0.5
USE_BREAKEVEN_TRAIL_AFTER_T1 = True

RISK_PER_TRADE_RUPEES = 1000

# Ship gates
SHIP_MIN_N = 100
SHIP_MIN_PF = 1.20
SHIP_CAP_PF = 1.15

# Buckets
VOL_RATIO_BUCKETS = [(5.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, float("inf"))]
SIZE_CR_BUCKETS   = [(5.0, 25.0), (25.0, 100.0), (100.0, 500.0), (500.0, float("inf"))]


# ----- Data loaders -----

def load_block_deals() -> pd.DataFrame:
    path = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[(df["trade_date"] >= DISCOVERY_START) & (df["trade_date"] <= DISCOVERY_END)]
    # NSE only (BSE deals are on different listing — bar data uses NSE prefix)
    df = df[df["exchange"] == "NSE"]
    return df.reset_index(drop=True)


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_5m_for_events(events: pd.DataFrame) -> pd.DataFrame:
    needed_months: set = set()
    for d in events["trade_date"].unique():
        needed_months.add((d.year, d.month))
        next_d = d + timedelta(days=7)
        needed_months.add((next_d.year, next_d.month))

    print(f"  loading {len(needed_months)} monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    universe_syms = set(events["raw_symbol"].astype(str).unique())
    for yyyy, mm in sorted(needed_months):
        mdf = _load_5m_for_month(yyyy, mm)
        if mdf.empty:
            continue
        mdf = mdf[mdf["symbol"].isin(universe_syms)]
        if not mdf.empty:
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total event-relevant 5m bars: {len(big):,}")
    return big


# ----- Signal aggregation -----

def aggregate_to_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Build per (trade_date, symbol) SELL-block signals with cell filters.

    Cells locked: SELL only, side_total >= 5cr, cap_segment in {mid,small}.
    Vol-ratio filter applied later (needs T0 day volume from 5m bars).
    """
    print("\n-- Funnel (signal aggregation, pre-volume-filter) --")
    print(f"  raw NSE Discovery events                  : {len(df):>6}")

    sells = df[df["buy_or_sell"] == "SELL"].copy()
    print(f"  after SELL-only filter                    : {len(sells):>6}")

    # Aggregate per (date, symbol) — sum value, sum qty across multi-line disclosures
    agg = (
        sells.groupby(["trade_date", "symbol"], as_index=False)
        .agg(
            side_total_cr=("trade_value_cr", "sum"),
            total_qty=("qty", "sum"),
            n_lines=("trade_value_cr", "size"),
            avg_price=("trade_price", "mean"),
            client_names=("client_name", lambda s: " / ".join(sorted(set(s.astype(str))))),
        )
    )
    print(f"  after dedup (date,symbol)                 : {len(agg):>6}")

    agg = agg[agg["side_total_cr"] >= MIN_SIDE_TOTAL_CR].copy()
    print(f"  after side_total_cr >= {MIN_SIDE_TOTAL_CR} Cr             : {len(agg):>6}")

    # Strip NSE: prefix to match raw symbol in feathers
    agg["raw_symbol"] = agg["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    agg["nse_symbol"] = "NSE:" + agg["raw_symbol"]
    agg["cap_segment"] = agg["nse_symbol"].apply(get_cap_segment)
    print(f"  cap_segment dist                           :")
    print("   ", agg["cap_segment"].value_counts().to_dict())

    agg = agg[agg["cap_segment"].isin(ALLOWED_CAP_SEGMENTS)].copy()
    print(f"  after cap_segment in {ALLOWED_CAP_SEGMENTS}: {len(agg):>6}")

    agg["side"] = "SHORT"
    return agg.reset_index(drop=True)


# ----- T+1 simulation -----

def _next_trading_day_with_data(sym_df: pd.DataFrame, t0: date) -> Optional[date]:
    future_days = sym_df[sym_df["d"] > t0]["d"].unique()
    if len(future_days) == 0:
        return None
    return min(future_days)


def simulate_t1(
    signals: pd.DataFrame,
    big5m: pd.DataFrame,
    apply_vol_ratio_filter: bool = True,
) -> tuple[pd.DataFrame, dict]:
    print("\n-- Simulating T+1 SHORT entries (mid/small cap, SELL-block) --")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    funnel = {
        "signals_in": len(signals),
        "no_5m_data": 0,
        "no_t0_day": 0,
        "no_t0_volume": 0,
        "fail_vol_ratio": 0,
        "no_t1_day": 0,
        "no_first_bar": 0,
        "no_entry_bar": 0,
        "fired_pre_latch": 0,
        "fired_post_latch": 0,
    }

    raw_trades: List[dict] = []
    for _, sig in signals.iterrows():
        raw_sym = sig["raw_symbol"]
        sym_df = days_per_sym.get(raw_sym)
        if sym_df is None or sym_df.empty:
            funnel["no_5m_data"] += 1
            continue

        t0 = sig["trade_date"]

        # T+0 day volume for vol-ratio filter
        t0_df = sym_df[sym_df["d"] == t0]
        if t0_df.empty:
            funnel["no_t0_day"] += 1
            continue
        t0_volume = float(t0_df["volume"].sum())
        if t0_volume <= 0:
            funnel["no_t0_volume"] += 1
            continue
        vol_ratio_pct = float(sig["total_qty"]) / t0_volume * 100.0

        if apply_vol_ratio_filter and vol_ratio_pct < MIN_VOL_RATIO_PCT:
            funnel["fail_vol_ratio"] += 1
            continue

        t0_close = float(t0_df.iloc[-1]["close"])

        t1 = _next_trading_day_with_data(sym_df, t0)
        if t1 is None:
            funnel["no_t1_day"] += 1
            continue
        t1_df = sym_df[sym_df["d"] == t1].sort_values("date").reset_index(drop=True)
        if t1_df.empty:
            funnel["no_t1_day"] += 1
            continue
        t1_df["hhmm"] = t1_df["date"].dt.strftime("%H:%M")

        first_bar_rows = t1_df[t1_df["hhmm"] == T1_FIRST_BAR_HHMM]
        if first_bar_rows.empty:
            funnel["no_first_bar"] += 1
            continue
        first_bar = first_bar_rows.iloc[0]
        first_open = float(first_bar["open"])
        gap_pct = (first_open / t0_close - 1.0) * 100.0

        entry_rows = t1_df[t1_df["hhmm"] == T1_ENTRY_BAR_HHMM]
        if entry_rows.empty:
            funnel["no_entry_bar"] += 1
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_price = float(entry_row["open"])

        # SHORT-only
        sl_pct_used = max(HARD_STOP_PCT, MIN_STOP_PCT)
        hard_sl = entry_price * (1.0 + sl_pct_used / 100.0)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            continue
        t1_target = entry_price - T1_R_MULTIPLE * stop_distance
        t2_target = entry_price - T2_R_MULTIPLE * stop_distance

        entry_idx_arr = t1_df.index[t1_df["date"] == entry_ts].tolist()
        if not entry_idx_arr:
            funnel["no_entry_bar"] += 1
            continue
        entry_idx = entry_idx_arr[0]
        forward = t1_df.iloc[entry_idx + 1:].copy()
        forward = forward[forward["hhmm"] <= TIME_STOP_HHMM]
        if forward.empty:
            funnel["no_entry_bar"] += 1
            continue

        exit_ts = None
        exit_price = None
        exit_reason = None
        hit_t1 = False
        t1_exit_price = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]
            high = float(bar["high"])
            low = float(bar["low"])
            close = float(bar["close"])
            hhmm = bar["hhmm"]

            # CORRECT BE trail: active_sl = entry_price if t1_hit else hard_sl
            active_sl = entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl

            if high >= active_sl:
                exit_ts = ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break
            if not hit_t1 and low <= t1_target:
                hit_t1 = True
                t1_exit_price = t1_target
                t1_exit_ts = ts
            if hit_t1 and low <= t2_target:
                exit_ts = ts
                exit_price = t2_target
                exit_reason = "t2"
                break

            if hhmm >= TIME_STOP_HHMM:
                exit_ts = ts
                exit_price = close
                exit_reason = "time_stop"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "time_stop"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        if hit_t1:
            qty_t1 = max(int(qty * T1_PARTIAL_FRACTION), 1)
            qty_t2 = max(qty - qty_t1, 0)
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_t2 = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL") if qty_t2 > 0 else 0.0
            fee = fee_t1 + fee_t2
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
                if qty_t2 > 0 else t1_exit_price
            )
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        raw_trades.append({
            "T0_signal_date": t0,
            "T1_entry_date": t1,
            "symbol": "NSE:" + raw_sym,
            "cap_segment": sig["cap_segment"],
            "side": "SHORT",
            "side_total_cr": float(sig["side_total_cr"]),
            "total_qty": int(sig["total_qty"]),
            "t0_volume": int(t0_volume),
            "vol_ratio_pct": vol_ratio_pct,
            "client_names": sig["client_names"],
            "t0_close": t0_close,
            "t1_first_open": first_open,
            "gap_pct": gap_pct,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "stop_distance": stop_distance,
            "t1_target": t1_target,
            "t2_target": t2_target,
            "hit_t1": hit_t1,
            "t1_exit_price": t1_exit_price,
            "t1_exit_ts": t1_exit_ts,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })

    funnel["fired_pre_latch"] = len(raw_trades)
    if raw_trades:
        trades_df = pd.DataFrame(raw_trades)
        trades_df = (
            trades_df.sort_values(["symbol", "T1_entry_date", "side_total_cr"],
                                  ascending=[True, True, False])
            .drop_duplicates(subset=["symbol", "T1_entry_date"], keep="first")
            .reset_index(drop=True)
        )
    else:
        trades_df = pd.DataFrame()
    funnel["fired_post_latch"] = len(trades_df)
    return trades_df, funnel


# ----- T+0 control simulation (leakage check) -----

def simulate_t0_control(
    signals: pd.DataFrame,
    big5m: pd.DataFrame,
    apply_vol_ratio_filter: bool = True,
) -> pd.DataFrame:
    """T+0 leakage check: SHORT at 15:10 close on the disclosure day itself,
    exit at next bar's close (~15:15 = EOD). If T+0 PF >= T+1 PF, the
    asymmetry is pre-disclosure leakage, NOT continuation."""
    print("\n-- T+0 leakage control simulation --")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    raw_trades: List[dict] = []
    for _, sig in signals.iterrows():
        raw_sym = sig["raw_symbol"]
        sym_df = days_per_sym.get(raw_sym)
        if sym_df is None or sym_df.empty:
            continue
        t0 = sig["trade_date"]
        t0_df = sym_df[sym_df["d"] == t0].sort_values("date").reset_index(drop=True)
        if t0_df.empty:
            continue
        t0_volume = float(t0_df["volume"].sum())
        if t0_volume <= 0:
            continue
        vol_ratio_pct = float(sig["total_qty"]) / t0_volume * 100.0
        if apply_vol_ratio_filter and vol_ratio_pct < MIN_VOL_RATIO_PCT:
            continue
        t0_df["hhmm"] = t0_df["date"].dt.strftime("%H:%M")
        entry_rows = t0_df[t0_df["hhmm"] == T0_CONTROL_EOD_HHMM]
        if entry_rows.empty:
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_price = float(entry_row["close"])

        entry_idx = int(entry_rows.index[0])
        if entry_idx + T0_CONTROL_HOLD_BARS < len(t0_df):
            exit_row = t0_df.iloc[entry_idx + T0_CONTROL_HOLD_BARS]
        else:
            exit_row = t0_df.iloc[-1]
        exit_ts = exit_row["date"]
        exit_price = float(exit_row["close"])

        sl_pct_used = max(HARD_STOP_PCT, MIN_STOP_PCT)
        hard_sl = entry_price * (1.0 + sl_pct_used / 100.0)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            continue
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        realized_pnl = (entry_price - exit_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "SELL")
        net_pnl = realized_pnl - fee

        raw_trades.append({
            "T0_signal_date": t0,
            "symbol": "NSE:" + raw_sym,
            "cap_segment": sig["cap_segment"],
            "side": "SHORT",
            "side_total_cr": float(sig["side_total_cr"]),
            "vol_ratio_pct": vol_ratio_pct,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "exit_ts": exit_ts,
            "exit_price": exit_price,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })

    if not raw_trades:
        return pd.DataFrame()
    df = pd.DataFrame(raw_trades)
    df = (
        df.sort_values(["symbol", "T0_signal_date", "side_total_cr"],
                       ascending=[True, True, False])
        .drop_duplicates(subset=["symbol", "T0_signal_date"], keep="first")
        .reset_index(drop=True)
    )
    return df


# ----- Reporting -----

def _pf_wr(grp: pd.DataFrame) -> tuple[float, float, int, float]:
    n = len(grp)
    if n == 0:
        return float("nan"), float("nan"), 0, 0.0
    npnl = grp["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(float(wins / losses), 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    net = float(npnl.sum())
    return pf, wr, n, net


def _bucket(v: float, edges: list) -> str:
    av = abs(v)
    for lo, hi in edges:
        if av >= lo and av < hi:
            return f"[{lo:.1f},{hi if hi != float('inf') else 'inf'})"
    return "out"


def report(trades: pd.DataFrame, label: str = "T+1 entries") -> dict:
    if trades.empty:
        print(f"\n[{label}] no trades")
        return {"n": 0, "pf": None, "wr": None, "sharpe": None, "monthly_stable_pct": None}

    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(float(wins / losses), 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    daily_col = "T1_entry_date" if "T1_entry_date" in trades.columns else "T0_signal_date"
    daily = trades.groupby(daily_col)["net_pnl"].sum()
    sharpe = round(float(daily.mean() / daily.std()), 3) if daily.std() > 0 else 0.0

    print(f"\n=== {label} — sanity report ===")
    print(f"Period       : {trades[daily_col].min()} .. {trades[daily_col].max()}")
    print(f"Trades n     : {n}")
    print(f"Win rate     : {wr}%")
    print(f"Gross PnL    : Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees         : Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL      : Rs.{int(npnl.sum()):,}")
    print(f"NET PF       : {pf}")
    print(f"Sharpe (d)   : {sharpe}")

    cap_summary = {}
    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        pf2, wr2, n2, net = _pf_wr(grp)
        cap_summary[cap] = {"pf": pf2, "wr": wr2, "n": n2, "net": net}
        print(f"  {cap:<12} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    monthly_stable_pct = None
    if "T1_entry_date" in trades.columns or "T0_signal_date" in trades.columns:
        print("\nPer month:")
        td = trades.copy()
        td["month"] = pd.to_datetime(td[daily_col]).dt.to_period("M").astype(str)
        stable_months = 0
        total_months = 0
        for m, grp in td.groupby("month"):
            pf2, wr2, n2, net = _pf_wr(grp)
            total_months += 1
            if isinstance(pf2, (int, float)) and pf2 > 1.0:
                stable_months += 1
            print(f"  {m:<8} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")
        monthly_stable_pct = (stable_months / total_months * 100) if total_months > 0 else 0.0
        print(f"  Stable months (PF>1.0): {stable_months}/{total_months} ({monthly_stable_pct:.1f}%)")

    # Per-year stability
    if daily_col in trades.columns:
        print("\nPer year:")
        td = trades.copy()
        td["year"] = pd.to_datetime(td[daily_col]).dt.year
        for y, grp in td.groupby("year"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {y}  n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    if "exit_reason" in trades.columns:
        print("\nExit-reason breakdown:")
        for rsn, grp in trades.groupby("exit_reason"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {rsn:<18} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  avgNet=Rs.{int(grp['net_pnl'].mean()):>6,}")

    if "vol_ratio_pct" in trades.columns:
        print("\nPer vol_ratio_pct bucket:")
        td = trades.copy()
        td["vrb"] = td["vol_ratio_pct"].apply(lambda v: _bucket(v, VOL_RATIO_BUCKETS))
        for b, grp in td.groupby("vrb"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {b:<16} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    if "side_total_cr" in trades.columns:
        print("\nPer side_total_cr bucket:")
        td = trades.copy()
        td["scb"] = td["side_total_cr"].apply(lambda v: _bucket(v, SIZE_CR_BUCKETS))
        for b, grp in td.groupby("scb"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {b:<16} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    return {
        "n": n, "pf": pf, "wr": wr, "sharpe": sharpe,
        "monthly_stable_pct": monthly_stable_pct,
        "cap_summary": cap_summary,
        "net_pnl": float(npnl.sum()),
    }


def _print_funnel(funnel: dict) -> None:
    print("\n-- Trade-funnel --")
    for k, v in funnel.items():
        print(f"  {k:<28}: {v}")


def _verdict(t1_summary: dict, t0_summary: dict, large_cap_summary: Optional[dict] = None) -> None:
    print("\n" + "=" * 70)
    print(f"VERDICT (gauntlet-v2 gates: n>={SHIP_MIN_N}, PF>={SHIP_MIN_PF}, Sharpe>0, monthly stable)")
    print("=" * 70)

    if t1_summary["n"] == 0:
        print("VERDICT: no T+1 trades fired. DATA_UNAVAILABLE / RETIRE.")
        return

    pf = t1_summary["pf"]
    n = t1_summary["n"]
    sharpe = t1_summary["sharpe"]
    monthly_stable_pct = t1_summary.get("monthly_stable_pct")
    cap_summary = t1_summary.get("cap_summary", {})

    print(f"  Aggregate: n={n}, PF={pf}, Sharpe={sharpe}, monthly stable={monthly_stable_pct}%")
    for cap, s in cap_summary.items():
        print(f"  {cap:<12}: n={s['n']}, PF={s['pf']}, WR={s['wr']}%")

    # T+0 leakage check
    leakage_clean = True
    if t0_summary and t0_summary.get("n", 0) > 0:
        t0_pf = t0_summary["pf"]
        print(f"\n  T+0 leakage control PF={t0_pf} | T+1 PF={pf}")
        if isinstance(t0_pf, (int, float)) and isinstance(pf, (int, float)):
            if t0_pf >= pf:
                print("  *** T+0 PF >= T+1 PF: pre-disclosure leakage; continuation thesis INVALIDATED. ***")
                leakage_clean = False
            else:
                print("  T+0 underperforms T+1 -> continuation mechanism consistent with thesis.")

    # Large-cap diagnostic
    if large_cap_summary and large_cap_summary.get("n", 0) > 0:
        lc_pf = large_cap_summary["pf"]
        print(f"\n  Large-cap diagnostic (relaxed universe): n={large_cap_summary['n']}, PF={lc_pf}")
        if isinstance(lc_pf, (int, float)) and isinstance(pf, (int, float)):
            if lc_pf >= pf:
                print("  *** large_cap PF >= mid/small PF: NOT illiquidity-driven. Data-mined. ***")

    # Gate evaluation
    print("\nGate eval:")
    pass_n = n >= SHIP_MIN_N
    pass_pf = isinstance(pf, (int, float)) and pf >= SHIP_MIN_PF
    pass_sharpe = isinstance(sharpe, (int, float)) and sharpe > 0
    pass_monthly = (monthly_stable_pct is not None) and monthly_stable_pct >= 50.0

    mid_ok = False
    small_ok = False
    if cap_summary.get("mid_cap"):
        s = cap_summary["mid_cap"]
        mid_ok = isinstance(s["pf"], (int, float)) and s["pf"] >= SHIP_CAP_PF and s["n"] >= 30
    if cap_summary.get("small_cap"):
        s = cap_summary["small_cap"]
        small_ok = isinstance(s["pf"], (int, float)) and s["pf"] >= SHIP_CAP_PF and s["n"] >= 30

    print(f"  n >= {SHIP_MIN_N}                 : {'PASS' if pass_n else 'FAIL'}")
    print(f"  PF >= {SHIP_MIN_PF}              : {'PASS' if pass_pf else 'FAIL'}")
    print(f"  Sharpe > 0              : {'PASS' if pass_sharpe else 'FAIL'}")
    print(f"  Monthly stable >= 50%   : {'PASS' if pass_monthly else 'FAIL'}")
    print(f"  mid_cap PF >= {SHIP_CAP_PF}      : {'PASS' if mid_ok else 'FAIL'}")
    print(f"  small_cap PF >= {SHIP_CAP_PF}    : {'PASS' if small_ok else 'FAIL'}")
    print(f"  T+0 leakage clean       : {'PASS' if leakage_clean else 'FAIL'}")

    all_pass = pass_n and pass_pf and pass_sharpe and pass_monthly and mid_ok and small_ok and leakage_clean
    if all_pass:
        print("\n>>> ALL GAUNTLET-V2 GATES PASS -> STRONG PROCEED (full brief + OOS)")
    elif pass_n and pass_pf and leakage_clean:
        print("\n>>> Core gates pass but secondary gates fail -> MARGINAL (narrow universe + revisit)")
    elif not pass_n:
        print("\n>>> n < threshold -> DATA_UNAVAILABLE / defer until 2025+ backfill")
    else:
        print("\n>>> Gate FAIL -> RETIRE candidate.")


def main():
    print("=== block_deal_followthrough_short — pre-coding sanity ===")
    print(f"Discovery window: {DISCOVERY_START} .. {DISCOVERY_END}")

    print("\nLoading block-deal events (NSE, SELL only, Discovery)...")
    df = load_block_deals()
    print(f"  raw NSE Discovery events: {len(df)}")

    signals = aggregate_to_signals(df)
    if signals.empty:
        print("No signals after filter; aborting.")
        return

    # Also build a "large_cap" diagnostic signal set (same filters but cap=large)
    print("\n-- Building large_cap diagnostic signals (for falsifier #4) --")
    sells_all = df[df["buy_or_sell"] == "SELL"].copy()
    lc_agg = (
        sells_all.groupby(["trade_date", "symbol"], as_index=False)
        .agg(side_total_cr=("trade_value_cr", "sum"),
             total_qty=("qty", "sum"),
             client_names=("client_name", lambda s: " / ".join(sorted(set(s.astype(str))))))
    )
    lc_agg = lc_agg[lc_agg["side_total_cr"] >= MIN_SIDE_TOTAL_CR].copy()
    lc_agg["raw_symbol"] = lc_agg["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    lc_agg["nse_symbol"] = "NSE:" + lc_agg["raw_symbol"]
    lc_agg["cap_segment"] = lc_agg["nse_symbol"].apply(get_cap_segment)
    lc_signals = lc_agg[lc_agg["cap_segment"] == "large_cap"].reset_index(drop=True)
    lc_signals["side"] = "SHORT"
    print(f"  large_cap signals: {len(lc_signals)}")

    # Need 5m bars covering BOTH mid/small and large-cap diagnostic universes
    all_signals = pd.concat([signals, lc_signals], ignore_index=True)

    print("\nLoading 5m feathers for event months ...")
    big5m = build_5m_for_events(all_signals)
    if big5m.empty:
        print("No 5m bars; aborting.")
        return

    # Run T+1 simulation (target cells)
    trades, funnel = simulate_t1(signals, big5m, apply_vol_ratio_filter=True)
    _print_funnel(funnel)
    if trades.empty:
        print("\nNo T+1 trades fired. DATA_UNAVAILABLE.")
        return
    t1_summary = report(trades, label="T+1 block_deal_followthrough_short (mid/small)")

    # T+0 leakage control
    t0_trades = simulate_t0_control(signals, big5m, apply_vol_ratio_filter=True)
    t0_summary = report(t0_trades, label="T+0 leakage control") if not t0_trades.empty else {"n": 0}

    # Large-cap diagnostic
    lc_trades, _ = simulate_t1(lc_signals, big5m, apply_vol_ratio_filter=True)
    lc_summary = report(lc_trades, label="Large-cap diagnostic (falsifier #4)") if not lc_trades.empty else {"n": 0}

    _verdict(t1_summary, t0_summary, lc_summary)

    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "block_deal_followthrough_short_trades.csv"
    trades.to_csv(out_csv, index=False)
    print(f"\nT+1 trade log: {out_csv}")
    if not t0_trades.empty:
        t0_trades.to_csv(out_dir / "block_deal_followthrough_short_t0_control.csv", index=False)
        print(f"T+0 control log: {out_dir / 'block_deal_followthrough_short_t0_control.csv'}")
    if not lc_trades.empty:
        lc_trades.to_csv(out_dir / "block_deal_followthrough_short_large_cap_diag.csv", index=False)
        print(f"Large-cap diag log: {out_dir / 'block_deal_followthrough_short_large_cap_diag.csv'}")


if __name__ == "__main__":
    main()
