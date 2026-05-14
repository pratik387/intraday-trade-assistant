"""Pre-coding sanity check for `block_deal_t0_short`.

Hypothesis (sub-9, derived from block_deal_continuation_short post-mortem):
  The T+0 EOD SHORT "leakage control" from
  ``sanity_block_deal_continuation_short.py`` showed PF=1.137 (n=132) vs the
  T+1 thesis PF=1.131 (n=130). The T+0 PF >= T+1 PF was used to RETIRE the
  continuation hypothesis (leakage). But the residual T+0 PF itself is
  >1.0 net of fees — worth investigating as a standalone setup.

Critical data audit finding (2026-05-14):
  ``data/block_deals/block_deals_events.parquet`` has columns:
    trade_date, symbol, raw_symbol, client_name, buy_or_sell, qty,
    trade_price, trade_value_cr, exchange, company_name
  **There is NO timestamp column.** NSE disclosures are EOD-only
  (published ~17:00-18:30 IST in a daily report). The block-deal special
  window is 15:05-15:30, but execution within it is not in our data.

Implication: A pure "T+0 SHORT after disclosure" is not actionable —
disclosure happens after close. The only way the original 15:10 control
shows PF > 1 is if the disclosed block is itself executed in the special
window (15:05-15:30) and causes intraday price impact a trader could ride
IF they could detect the block from tape (volume spike + offered).

This sanity therefore does TWO things:
  1. STABILITY SCAN: sweep entry time across 14:00, 14:30, 15:00, 15:05,
     15:10 to test whether the PF 1.137 finding is time-stable (real
     mechanism) or time-dependent (data-mined noise).
  2. PROPER GAUNTLET-V2: apply tight SL (50 bps), proper R:R (T1 30bps,
     T2 50bps), BE trail, and ship gates.

If PF is time-stable across 14:00-15:10 with PF >= 1.20: there's likely
an intraday-drift component to disclosed-block-sell names. If PF is only
high at 15:10 and collapses at earlier times, the original signal was
data-mined noise.

Pre-registered cells (locked, mirrored from continuation-short brief):
  - buy_or_sell == "SELL"
  - side_total_cr >= 5.0 (notional floor)
  - block_qty / T0_day_volume >= 5% (volume-shock floor)
  - cap_segment in ["mid_cap", "small_cap"]
  - latch: one fire per (symbol, T0)

Ship gates (gauntlet-v2):
  - n >= 100 (or n >= 125 per power calc)
  - net PF >= 1.30 (ship) or PF >= 1.20 (survivor)
  - Sharpe (daily) > 0
  - per-month winning >= 55%
  - top month NET contribution < 40%
  - PF must be stable across entry-time sweep (max-min spread < 0.25)

Entry mechanic (brief):
  - SHORT at next 5m bar close after disclosure timestamp; fallback 15:10
  - SL: entry * 1.005 (50 bps — tight, we're entering AT impact not before)
  - T1 (50%): entry * 0.997 (30 bps move)
  - T2 (50%): entry * 0.995 (50 bps move)
  - Time stop: 15:25 IST close
  - BE trail after T1: active_sl = entry if t1_hit else hard_sl

Usage:
    .venv/Scripts/python tools/sub9_research/sanity_block_deal_t0_short.py
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
MIN_SIDE_TOTAL_CR = 5.0
MIN_VOL_RATIO_PCT = 5.0
ALLOWED_CAP_SEGMENTS = {"mid_cap", "small_cap"}

# T+0 entry/exit mechanic (brief)
# Sweep entry times for stability check; primary = 15:10 (mirrors original control)
ENTRY_TIME_SWEEP = ["14:00", "14:30", "15:00", "15:05", "15:10"]
PRIMARY_ENTRY_HHMM = "15:10"
TIME_STOP_HHMM = "15:25"

# Tight SL and R:R per brief
HARD_STOP_PCT = 0.5     # 50 bps
T1_TARGET_PCT = 0.3     # 30 bps (50% off)
T2_TARGET_PCT = 0.5     # 50 bps (rest)
T1_PARTIAL_FRACTION = 0.5
USE_BREAKEVEN_TRAIL_AFTER_T1 = True

RISK_PER_TRADE_RUPEES = 1000

# Ship gates (brief)
SHIP_MIN_N = 100
SHIP_SURVIVOR_PF = 1.20
SHIP_PROCEED_PF  = 1.30
SHIP_MONTHLY_WIN_PCT = 55.0
SHIP_TOP_MONTH_PCT = 40.0
SHIP_PF_SWEEP_SPREAD = 0.25

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


# ----- Signal aggregation (identical to T+1 sanity for apples-to-apples) -----

def aggregate_to_signals(df: pd.DataFrame) -> pd.DataFrame:
    print("\n-- Funnel (signal aggregation, pre-volume-filter) --")
    print(f"  raw NSE Discovery events                  : {len(df):>6}")

    sells = df[df["buy_or_sell"] == "SELL"].copy()
    print(f"  after SELL-only filter                    : {len(sells):>6}")

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

    agg["raw_symbol"] = agg["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    agg["nse_symbol"] = "NSE:" + agg["raw_symbol"]
    agg["cap_segment"] = agg["nse_symbol"].apply(get_cap_segment)
    print(f"  cap_segment dist                           :")
    print("   ", agg["cap_segment"].value_counts().to_dict())

    agg = agg[agg["cap_segment"].isin(ALLOWED_CAP_SEGMENTS)].copy()
    print(f"  after cap_segment in {ALLOWED_CAP_SEGMENTS}: {len(agg):>6}")

    agg["side"] = "SHORT"
    return agg.reset_index(drop=True)


# ----- T+0 simulation (variable entry time) -----

def simulate_t0(
    signals: pd.DataFrame,
    big5m: pd.DataFrame,
    entry_hhmm: str,
    apply_vol_ratio_filter: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """T+0 SHORT entry at given HH:MM bar (close), tight SL/TP and BE trail."""
    print(f"\n-- T+0 simulation @ entry={entry_hhmm}, SL={HARD_STOP_PCT}%, T1={T1_TARGET_PCT}%, T2={T2_TARGET_PCT}% --")
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
        "no_entry_bar": 0,
        "no_forward_bars": 0,
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
        t0_df = sym_df[sym_df["d"] == t0].sort_values("date").reset_index(drop=True)
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

        t0_df["hhmm"] = t0_df["date"].dt.strftime("%H:%M")
        entry_rows = t0_df[t0_df["hhmm"] == entry_hhmm]
        if entry_rows.empty:
            funnel["no_entry_bar"] += 1
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_price = float(entry_row["close"])
        if entry_price <= 0:
            funnel["no_entry_bar"] += 1
            continue

        # Tight SL / T1 / T2 — brief spec
        hard_sl = entry_price * (1.0 + HARD_STOP_PCT / 100.0)
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            continue
        t1_target = entry_price * (1.0 - T1_TARGET_PCT / 100.0)
        t2_target = entry_price * (1.0 - T2_TARGET_PCT / 100.0)

        entry_idx = int(entry_rows.index[0])
        forward = t0_df.iloc[entry_idx + 1:].copy()
        # cap at 15:25 close
        forward = forward[forward["hhmm"] <= TIME_STOP_HHMM]
        if forward.empty:
            funnel["no_forward_bars"] += 1
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

            active_sl = entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl

            # SHORT: stop hit if high >= active_sl
            if high >= active_sl:
                exit_ts = ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break
            # T1 hit if low <= t1_target (before T2)
            if not hit_t1 and low <= t1_target:
                hit_t1 = True
                t1_exit_price = t1_target
                t1_exit_ts = ts
            # T2 only after T1
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
            "symbol": "NSE:" + raw_sym,
            "cap_segment": sig["cap_segment"],
            "side": "SHORT",
            "side_total_cr": float(sig["side_total_cr"]),
            "total_qty": int(sig["total_qty"]),
            "t0_volume": int(t0_volume),
            "vol_ratio_pct": vol_ratio_pct,
            "client_names": sig["client_names"],
            "entry_hhmm": entry_hhmm,
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
            trades_df.sort_values(["symbol", "T0_signal_date", "side_total_cr"],
                                  ascending=[True, True, False])
            .drop_duplicates(subset=["symbol", "T0_signal_date"], keep="first")
            .reset_index(drop=True)
        )
    else:
        trades_df = pd.DataFrame()
    funnel["fired_post_latch"] = len(trades_df)
    return trades_df, funnel


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


def report(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        print(f"\n[{label}] no trades")
        return {"n": 0, "pf": None, "wr": None, "sharpe": None,
                "monthly_winning_pct": None, "top_month_net_pct": None}

    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(float(wins / losses), 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    daily = trades.groupby("T0_signal_date")["net_pnl"].sum()
    sharpe = round(float(daily.mean() / daily.std()), 3) if daily.std() > 0 else 0.0
    net_total = float(npnl.sum())

    print(f"\n=== {label} ===")
    print(f"Period       : {trades['T0_signal_date'].min()} .. {trades['T0_signal_date'].max()}")
    print(f"Trades n     : {n}")
    print(f"Win rate     : {wr}%")
    print(f"Gross PnL    : Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees         : Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL      : Rs.{int(net_total):,}")
    print(f"NET PF       : {pf}")
    print(f"Sharpe (d)   : {sharpe}")

    cap_summary = {}
    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        pf2, wr2, n2, net = _pf_wr(grp)
        cap_summary[cap] = {"pf": pf2, "wr": wr2, "n": n2, "net": net}
        print(f"  {cap:<12} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    # Per-month winning % and top month concentration
    td = trades.copy()
    td["month"] = pd.to_datetime(td["T0_signal_date"]).dt.to_period("M").astype(str)
    monthly_pf_list = []
    monthly_net_list = []
    print("\nPer month:")
    for m, grp in td.groupby("month"):
        pf2, wr2, n2, net = _pf_wr(grp)
        monthly_pf_list.append((m, pf2, n2, net))
        monthly_net_list.append((m, net))
        print(f"  {m:<8} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    winning_months = sum(1 for _, pf2, _, _ in monthly_pf_list
                         if isinstance(pf2, (int, float)) and pf2 > 1.0)
    total_months = len(monthly_pf_list)
    monthly_winning_pct = (winning_months / total_months * 100) if total_months > 0 else 0.0
    print(f"  Winning months (PF>1.0): {winning_months}/{total_months} ({monthly_winning_pct:.1f}%)")

    top_month_net_pct = None
    if net_total > 0:
        top_net = max((net for _, net in monthly_net_list if net > 0), default=0.0)
        top_month_net_pct = (top_net / net_total * 100) if net_total > 0 else None
        print(f"  Top winning-month NET share: {top_month_net_pct:.1f}% of total")

    print("\nPer year:")
    td["year"] = pd.to_datetime(td["T0_signal_date"]).dt.year
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
        td2 = trades.copy()
        td2["vrb"] = td2["vol_ratio_pct"].apply(lambda v: _bucket(v, VOL_RATIO_BUCKETS))
        for b, grp in td2.groupby("vrb"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {b:<16} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    return {
        "n": n, "pf": pf, "wr": wr, "sharpe": sharpe,
        "monthly_winning_pct": monthly_winning_pct,
        "top_month_net_pct": top_month_net_pct,
        "cap_summary": cap_summary,
        "net_pnl": net_total,
    }


def _print_funnel(funnel: dict) -> None:
    print("\n-- Trade-funnel --")
    for k, v in funnel.items():
        print(f"  {k:<28}: {v}")


def _verdict(primary: dict, sweep_summaries: dict) -> None:
    print("\n" + "=" * 72)
    print(f"VERDICT (gauntlet-v2: n>={SHIP_MIN_N}, PF>={SHIP_SURVIVOR_PF}/{SHIP_PROCEED_PF}, "
          f"Sharpe>0, MonthlyWin>={SHIP_MONTHLY_WIN_PCT}%, TopMonthNet<{SHIP_TOP_MONTH_PCT}%, "
          f"PFsweepSpread<{SHIP_PF_SWEEP_SPREAD})")
    print("=" * 72)

    if primary["n"] == 0:
        print("VERDICT: no T+0 trades fired. DATA_UNAVAILABLE / RETIRE.")
        return

    pf = primary["pf"]
    n = primary["n"]
    sharpe = primary["sharpe"]
    monthly_winning_pct = primary.get("monthly_winning_pct")
    top_month_net_pct = primary.get("top_month_net_pct")

    print(f"  Primary @ {PRIMARY_ENTRY_HHMM}: n={n}, PF={pf}, Sharpe={sharpe}, "
          f"MonthlyWin={monthly_winning_pct}%, TopMonthNet={top_month_net_pct}%")

    # Stability sweep
    print("\nEntry-time stability sweep:")
    sweep_pfs = []
    for t, s in sweep_summaries.items():
        if s["n"] > 0:
            print(f"  entry={t}  n={s['n']:>4}  PF={s['pf']:>6}  WR={s['wr']:>5}%  "
                  f"Sharpe={s['sharpe']:>6}  MonWin={s.get('monthly_winning_pct', 0):.1f}%")
            if isinstance(s["pf"], (int, float)) and not np.isinf(s["pf"]):
                sweep_pfs.append(s["pf"])
        else:
            print(f"  entry={t}  no trades")

    pf_spread = (max(sweep_pfs) - min(sweep_pfs)) if len(sweep_pfs) >= 2 else 0.0
    print(f"  PF sweep spread: {pf_spread:.3f} (gauntlet limit: {SHIP_PF_SWEEP_SPREAD})")

    # Gate eval
    print("\nGate eval:")
    pass_n = n >= SHIP_MIN_N
    pass_pf_survivor = isinstance(pf, (int, float)) and pf >= SHIP_SURVIVOR_PF
    pass_pf_proceed  = isinstance(pf, (int, float)) and pf >= SHIP_PROCEED_PF
    pass_sharpe = isinstance(sharpe, (int, float)) and sharpe > 0
    pass_monthly = (monthly_winning_pct is not None) and monthly_winning_pct >= SHIP_MONTHLY_WIN_PCT
    pass_top_month = (top_month_net_pct is not None) and top_month_net_pct < SHIP_TOP_MONTH_PCT
    pass_stability = pf_spread < SHIP_PF_SWEEP_SPREAD

    print(f"  n >= {SHIP_MIN_N}                  : {'PASS' if pass_n else 'FAIL'}")
    print(f"  PF >= {SHIP_SURVIVOR_PF} (survivor)      : {'PASS' if pass_pf_survivor else 'FAIL'}")
    print(f"  PF >= {SHIP_PROCEED_PF} (ship)          : {'PASS' if pass_pf_proceed else 'FAIL'}")
    print(f"  Sharpe > 0               : {'PASS' if pass_sharpe else 'FAIL'}")
    print(f"  Monthly win >= {SHIP_MONTHLY_WIN_PCT}%    : {'PASS' if pass_monthly else 'FAIL'}")
    print(f"  TopMonth NET < {SHIP_TOP_MONTH_PCT}%    : {'PASS' if pass_top_month else 'FAIL'}")
    print(f"  PF sweep spread < {SHIP_PF_SWEEP_SPREAD}: {'PASS' if pass_stability else 'FAIL'}")

    # Comparison vs continuation-short T+1 finding
    print("\nReference (block_deal_continuation_short brief output):")
    print("  T+1 SHORT @09:25 entry  : n=130, PF=1.131 (RETIRED — leakage)")
    print("  T+0 control @15:10 (old): n=132, PF=1.137, WR=43.2%, MonStable=33.3%")

    # Final verdict
    all_proceed = pass_n and pass_pf_proceed and pass_sharpe and pass_monthly and pass_top_month and pass_stability
    all_survivor = pass_n and pass_pf_survivor and pass_sharpe and pass_monthly and pass_top_month and pass_stability
    if all_proceed:
        print("\n>>> ALL SHIP GATES PASS -> STRONG PROCEED (ship-eligible, write brief)")
    elif all_survivor:
        print("\n>>> SURVIVOR-LEVEL PASS (PF>=1.20) -> MARGINAL (more cells, OOS test)")
    elif not pass_n:
        print("\n>>> n < threshold -> DATA_UNAVAILABLE / defer")
    elif not pass_stability:
        print("\n>>> PF time-unstable -> DATA-MINED NOISE / RETIRE")
    else:
        print("\n>>> Gate FAIL -> RETIRE candidate.")


def main():
    print("=== block_deal_t0_short — pre-coding sanity ===")
    print(f"Discovery window: {DISCOVERY_START} .. {DISCOVERY_END}")
    print(f"Entry sweep     : {ENTRY_TIME_SWEEP}  (primary={PRIMARY_ENTRY_HHMM})")
    print(f"Mechanic        : SL={HARD_STOP_PCT}%, T1={T1_TARGET_PCT}%/50%, T2={T2_TARGET_PCT}%/50%, BE-trail, TimeStop={TIME_STOP_HHMM}")

    print("\n[1/4] data audit — block_deals_events.parquet schema")
    bd_path = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
    bd_df_full = pd.read_parquet(bd_path)
    print(f"  columns: {list(bd_df_full.columns)}")
    print(f"  total rows: {len(bd_df_full)}")
    has_time = any(c.lower() in ("trade_time", "report_time", "disclosure_ts", "timestamp")
                   for c in bd_df_full.columns)
    print(f"  HAS_INTRADAY_TIMING: {has_time}")
    print(f"  -> NSE publishes block-deal report EOD; no intraday timing column => fallback to entry-time sweep.")

    print("\n[2/4] Loading block-deal events (NSE, SELL, Discovery)")
    df = load_block_deals()
    print(f"  raw NSE Discovery events: {len(df)}")

    signals = aggregate_to_signals(df)
    if signals.empty:
        print("No signals after filter; aborting.")
        return

    print("\n[3/4] Loading 5m feathers for event months")
    big5m = build_5m_for_events(signals)
    if big5m.empty:
        print("No 5m bars; aborting.")
        return

    # Run sweep
    print(f"\n[4/4] Running T+0 SHORT sweep across {ENTRY_TIME_SWEEP}")
    sweep_summaries: Dict[str, dict] = {}
    sweep_trades: Dict[str, pd.DataFrame] = {}
    for ehhmm in ENTRY_TIME_SWEEP:
        trades_e, funnel_e = simulate_t0(signals, big5m, ehhmm, apply_vol_ratio_filter=True)
        _print_funnel(funnel_e)
        if trades_e.empty:
            print(f"  No trades at entry={ehhmm}")
            sweep_summaries[ehhmm] = {"n": 0}
            continue
        s = report(trades_e, label=f"T+0 SHORT @entry={ehhmm}")
        sweep_summaries[ehhmm] = s
        sweep_trades[ehhmm] = trades_e

    primary = sweep_summaries.get(PRIMARY_ENTRY_HHMM, {"n": 0})
    _verdict(primary, sweep_summaries)

    # Write outputs
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ehhmm, td in sweep_trades.items():
        slug = ehhmm.replace(":", "")
        out_csv = out_dir / f"block_deal_t0_short_entry_{slug}.csv"
        td.to_csv(out_csv, index=False)
        print(f"\nTrade log @{ehhmm}: {out_csv}")


if __name__ == "__main__":
    main()
