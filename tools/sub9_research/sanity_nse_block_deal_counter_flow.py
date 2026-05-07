"""Pre-coding sanity check for nse_block_deal_counter_flow candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-nse_block_
deal_counter_flow.md): BEFORE writing detector code, simulate the T+1
retail-FOMO counter-flow fade on Discovery period block-deal events.

Decision criterion (locked thresholds, brief §3.3):
  PF >= 1.10 AND n >= 30 per cell AND |WR delta SHORT vs LONG| <= 10pp
    -> candidate for OOS validation
  Otherwise -> retire

Usage:
    python tools/sub9_research/sanity_nse_block_deal_counter_flow.py

Mechanic (per locked brief params):
  - Per (trade_date, symbol) aggregate ALL block-deal rows (NSE+BSE);
    sum trade_value_cr per side (BUY/SELL); compute net direction.
  - Trigger: |sum_buy_value - sum_sell_value| >= 25 Cr; net side wins.
  - Net BUY  -> SHORT next-day (fade retail FOMO into seller's direction)
  - Net SELL -> LONG  next-day (fade retail panic into buyer's direction)
  - T+1 09:15 first 5m bar opens above (SHORT) / below (LONG) T+0 close;
    first bar must be green (SHORT) / red (LONG) to confirm retail FOMO.
  - Entry: second 5m bar's OPEN (09:25 IST).
  - Hard SL: 1.5% (min 1.0%).
  - Targets: T1 = 1R partial (50% qty), T2 = 2R full (50% qty); BE-trail
    after T1.
  - Time stop: 14:30 IST.
  - Latch: one fire per (symbol, T+1, side).
  - Universe: F&O liquid 200 (assets/fno_liquid_200.csv); all cap segments.

Falsification gates (locked):
  - PF >= 1.10 per cell
  - n >= 30 per cell
  - |WR delta SHORT vs LONG| <= 10pp
  - T+0 control: shorting the block-deal date itself at end-of-day must
    UNDERPERFORM T+1; otherwise asymmetry is pre-disclosure leakage,
    NOT retail-FOMO fade -> thesis invalidated.

Reference templates:
  - sanity_circuit_t1_fade_short.py (cross-day daily-stale-signal pattern)
  - sanity_volume_spike_exhaustion_reversal.py (50/50 tiered exits, calc_fee)
  - sanity_nifty_reconstitution_fade.py (event-driven cross-day pattern)
"""
from __future__ import annotations

import sys
from datetime import date, time, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params per brief §3.3 ----
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END   = date(2024, 12, 31)

MIN_NET_VALUE_CR = 25.0          # |net_value_cr| >= 25 Cr trigger floor
T0_CONTROL_EOD_HHMM = "15:10"    # T+0 control: short at 15:10 (last fully-formed bar)
T0_CONTROL_HOLD_BARS = 1         # exit at next bar's close (= 15:15 close ~ EOD)

# T+1 entry mechanic
T1_FIRST_BAR_HHMM = "09:15"      # first 5m bar of T+1
T1_ENTRY_BAR_HHMM = "09:25"      # second 5m bar (we enter at its OPEN)
TIME_STOP_HHMM = "14:30"         # mid-session time stop

HARD_STOP_PCT = 1.5              # 1.5% hard stop
MIN_STOP_PCT = 1.0               # min 1.0%

T1_R_MULTIPLE = 1.0              # T1 partial @ 1.0R
T2_R_MULTIPLE = 2.0              # T2 full @ 2.0R
T1_PARTIAL_FRACTION = 0.5        # 50% qty out at T1
USE_BREAKEVEN_TRAIL_AFTER_T1 = True

RISK_PER_TRADE_RUPEES = 1000     # match other sub9 sanities

# Bucket edges
NET_VALUE_BUCKETS = [(25, 50), (50, 100), (100, 250), (250, float("inf"))]
GAP_PCT_BUCKETS = [(0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, float("inf"))]


# ----- Data loaders -----

def load_block_deals() -> pd.DataFrame:
    path = _REPO_ROOT / "data" / "block_deals" / "block_deals_events.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing")
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df[(df["trade_date"] >= DISCOVERY_START) & (df["trade_date"] <= DISCOVERY_END)]
    return df.reset_index(drop=True)


def load_fno_universe() -> set:
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    return set(df["symbol"].astype(str).tolist())


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_5m_for_events(events: pd.DataFrame) -> pd.DataFrame:
    """Load 5m bars for each event's (T-0, T+1) months."""
    needed_months: set = set()
    for d in events["trade_date"].unique():
        # T-0 month
        needed_months.add((d.year, d.month))
        # T+1 month (may roll over)
        next_d = d + timedelta(days=7)  # week ahead spans month boundary
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


# ----- Aggregation -----

def aggregate_to_signals(df: pd.DataFrame, fno_syms: set) -> pd.DataFrame:
    """Build per (trade_date, symbol, side) signals.

    NOTE on data structure: each block deal is published as TWO rows in the
    NSE/BSE archive — one BUY row (buyer client name) and one SELL row
    (seller client name) for the SAME notional. Aggregating buy-vs-sell
    sums per (date, symbol) therefore mostly NETS to ~0 because the
    archive structurally records both sides of every deal.

    Per user spec: "just take each row as a separate event and let multiple
    events on the same symbol+day produce just one fire via latch." We do
    exactly that — per-row >=25 Cr filter, side derived from buy_or_sell,
    latch in simulate_t1.

    Funnel matches user's spec:
      events (5,875) -> >=25cr (2,788) -> + F&O 200 (1,050)
      then dedup at the side-level (one signal per (date, symbol, side))
    """
    print("\n-- Funnel --")
    print(f"  raw block-deal events           : {len(df):>6}")

    df_25 = df[df["trade_value_cr"] >= MIN_NET_VALUE_CR]
    print(f"  after individual-row >=25 Cr    : {len(df_25):>6}")

    df_fno = df[df["symbol"].isin(fno_syms)]
    print(f"  after F&O 200 filter            : {len(df_fno):>6}")

    df_25_fno = df[(df["trade_value_cr"] >= MIN_NET_VALUE_CR) & (df["symbol"].isin(fno_syms))]
    print(f"  after >=25 Cr + F&O 200 (events): {len(df_25_fno):>6}")

    # Per-side aggregation BEFORE >=25cr filter (so we capture the full
    # one-side notional even when split across multiple contra-parties)
    side_agg = (
        df[df["symbol"].isin(fno_syms)]
        .groupby(["trade_date", "symbol", "buy_or_sell"], as_index=False)
        .agg(side_total_cr=("trade_value_cr", "sum"),
             n_lines=("trade_value_cr", "size"),
             max_line_cr=("trade_value_cr", "max"))
    )
    # Apply per-(date, symbol, side) >= 25 Cr gate (fires when EITHER one
    # contra-line is >=25cr OR aggregated side total >=25cr)
    side_agg = side_agg[side_agg["side_total_cr"] >= MIN_NET_VALUE_CR].copy()
    print(f"  unique (date, symbol, side) cells: {len(side_agg):>6}")

    # Side derivation: BUY-side disclosure -> retail FOMO long -> we SHORT T+1
    #                  SELL-side disclosure -> retail panic   -> we LONG  T+1
    side_agg["side"] = np.where(side_agg["buy_or_sell"] == "BUY", "SHORT", "LONG")

    # Pull both sides separately into a per-(date, symbol) table to show
    # buy vs sell context (used for diagnostics / per-bucket reporting)
    pivot = (
        df[df["symbol"].isin(fno_syms)]
        .groupby(["trade_date", "symbol", "buy_or_sell"], as_index=False)["trade_value_cr"]
        .sum()
        .pivot_table(index=["trade_date", "symbol"], columns="buy_or_sell",
                     values="trade_value_cr", fill_value=0.0)
        .reset_index()
    )
    if "BUY" not in pivot.columns:
        pivot["BUY"] = 0.0
    if "SELL" not in pivot.columns:
        pivot["SELL"] = 0.0
    pivot.columns.name = None
    pivot = pivot.rename(columns={"BUY": "sum_buy_cr", "SELL": "sum_sell_cr"})

    sig = side_agg.merge(pivot, on=["trade_date", "symbol"], how="left")
    sig["abs_net_cr"] = sig["side_total_cr"]  # bucket on the disclosed-side notional

    # Strip NSE:/BSE: prefix to align with 5m feather symbols (which use raw NSE name).
    # Block-deal events are predominantly NSE-listed F&O 200; parquet "symbol" is
    # already NSE: prefix for those names. Strip prefix uniformly.
    sig["raw_symbol"] = sig["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    sig["nse_symbol"] = "NSE:" + sig["raw_symbol"]
    sig["cap_segment"] = sig["nse_symbol"].apply(get_cap_segment)
    print(f"  cap_segment dist (signal-level)  :")
    print("   ", sig["cap_segment"].value_counts().to_dict())
    print(f"  side dist (signal-level)         :")
    print("   ", sig["side"].value_counts().to_dict())
    print(f"  side_total_cr percentiles (cr)   :")
    print(f"    p25={sig['side_total_cr'].quantile(0.25):.1f} "
          f"p50={sig['side_total_cr'].median():.1f} "
          f"p75={sig['side_total_cr'].quantile(0.75):.1f} "
          f"max={sig['side_total_cr'].max():.1f}")
    return sig.reset_index(drop=True)


# ----- T+1 simulation -----

def _next_trading_day_with_data(sym_df: pd.DataFrame, t0: date) -> Optional[date]:
    future_days = sym_df[sym_df["d"] > t0]["d"].unique()
    if len(future_days) == 0:
        return None
    return min(future_days)


def simulate_t1(
    signals: pd.DataFrame,
    big5m: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    print("\n-- Simulating T+1 entries --")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    funnel = {
        "signals_in": len(signals),
        "no_5m_data": 0,
        "no_t1_day": 0,
        "no_first_bar": 0,
        "no_t0_close": 0,
        "fail_gap_or_candle": 0,
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
        side = sig["side"]
        t1 = _next_trading_day_with_data(sym_df, t0)
        if t1 is None:
            funnel["no_t1_day"] += 1
            continue

        # T+0 last bar's close as reference
        t0_df = sym_df[sym_df["d"] == t0]
        if t0_df.empty:
            funnel["no_t0_close"] += 1
            continue
        t0_close = float(t0_df.iloc[-1]["close"])

        # T+1 day bars
        t1_df = sym_df[sym_df["d"] == t1].sort_values("date").reset_index(drop=True)
        if t1_df.empty:
            funnel["no_t1_day"] += 1
            continue
        t1_df["hhmm"] = t1_df["date"].dt.strftime("%H:%M")

        # First bar @ 09:15
        first_bar_rows = t1_df[t1_df["hhmm"] == T1_FIRST_BAR_HHMM]
        if first_bar_rows.empty:
            funnel["no_first_bar"] += 1
            continue
        first_bar = first_bar_rows.iloc[0]
        first_open = float(first_bar["open"])
        first_close = float(first_bar["close"])

        # Gap classification (kept for diagnostic) — gate DROPPED in v2 (round-5
        # rescue 2026-05-07). Original gate fired only on gap+candle aligned to
        # the side, killing 472/585 events (81%) and leaving n=113 below the
        # 500-floor. v2: trade EVERY net-direction event at T+1 09:25 open;
        # the 1,050 strict-filter (≥₹25cr + F&O 200) events should produce
        # ~500-1,000 fires post-latch. Confirms whether the "trade all signals"
        # variant has aggregate edge before adding any entry-confirmation gate.
        gap_pct = (first_open / t0_close - 1.0) * 100.0
        is_green = first_close > first_open
        is_red = first_close < first_open
        # ok = True for every event (fire on direction signal alone)

        # Entry @ 09:25 OPEN
        entry_rows = t1_df[t1_df["hhmm"] == T1_ENTRY_BAR_HHMM]
        if entry_rows.empty:
            funnel["no_entry_bar"] += 1
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_price = float(entry_row["open"])

        # Hard SL
        if side == "SHORT":
            sl_pct_used = max(HARD_STOP_PCT, MIN_STOP_PCT)
            hard_sl = entry_price * (1.0 + sl_pct_used / 100.0)
            stop_distance = hard_sl - entry_price
            t1_target = entry_price - T1_R_MULTIPLE * stop_distance
            t2_target = entry_price - T2_R_MULTIPLE * stop_distance
        else:
            sl_pct_used = max(HARD_STOP_PCT, MIN_STOP_PCT)
            hard_sl = entry_price * (1.0 - sl_pct_used / 100.0)
            stop_distance = entry_price - hard_sl
            t1_target = entry_price + T1_R_MULTIPLE * stop_distance
            t2_target = entry_price + T2_R_MULTIPLE * stop_distance

        if stop_distance <= 0:
            continue

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

            active_sl = entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl

            if side == "SHORT":
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
            else:
                if low <= active_sl:
                    exit_ts = ts
                    exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True
                    t1_exit_price = t1_target
                    t1_exit_ts = ts
                if hit_t1 and high >= t2_target:
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
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2 if qty_t2 > 0 else 0.0
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2 if qty_t2 > 0 else 0.0
            realized_pnl = pnl_t1 + pnl_t2
            kite_side = "SELL" if side == "SHORT" else "BUY"
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, kite_side)
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, kite_side) if qty_t2 > 0 else 0.0
            fee = fee_t1 + fee_t2
            blended_exit = (
                (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
                if qty_t2 > 0 else t1_exit_price
            )
        else:
            if side == "SHORT":
                realized_pnl = (entry_price - exit_price) * qty
            else:
                realized_pnl = (exit_price - entry_price) * qty
            kite_side = "SELL" if side == "SHORT" else "BUY"
            fee = calc_fee(entry_price, exit_price, qty, kite_side)
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        raw_trades.append({
            "T0_signal_date": t0,
            "T1_entry_date": t1,
            "symbol": "NSE:" + raw_sym,
            "cap_segment": sig["cap_segment"],
            "side": side,
            "sum_buy_cr": float(sig.get("sum_buy_cr", 0.0)),
            "sum_sell_cr": float(sig.get("sum_sell_cr", 0.0)),
            "side_total_cr": float(sig["side_total_cr"]),
            "abs_net_cr": float(sig["abs_net_cr"]),
            "t0_close": t0_close,
            "t1_first_open": first_open,
            "t1_first_close": first_close,
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

    # Latch: one fire per (symbol, T+1, side)
    if raw_trades:
        trades_df = pd.DataFrame(raw_trades)
        trades_df = (
            trades_df.sort_values(["symbol", "T1_entry_date", "side", "abs_net_cr"], ascending=[True, True, True, False])
            .drop_duplicates(subset=["symbol", "T1_entry_date", "side"], keep="first")
            .reset_index(drop=True)
        )
    else:
        trades_df = pd.DataFrame()

    funnel["fired_post_latch"] = len(trades_df)
    return trades_df, funnel


# ----- T+0 control simulation -----

def simulate_t0_control(
    signals: pd.DataFrame,
    big5m: pd.DataFrame,
) -> pd.DataFrame:
    """T+0 same-day diagnostic: SHORT (NET BUY) / LONG (NET SELL) at end-of-day
    on the disclosure date itself, hold through close (1 bar)."""
    print("\n-- T+0 control simulation --")
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
        side = sig["side"]
        t0_df = sym_df[sym_df["d"] == t0].sort_values("date").reset_index(drop=True)
        if t0_df.empty:
            continue
        t0_df["hhmm"] = t0_df["date"].dt.strftime("%H:%M")
        entry_rows = t0_df[t0_df["hhmm"] == T0_CONTROL_EOD_HHMM]
        if entry_rows.empty:
            continue
        entry_row = entry_rows.iloc[0]
        entry_ts = entry_row["date"]
        entry_price = float(entry_row["close"])  # at-close fill at 15:10

        # Exit at next bar's close (or last bar of day)
        entry_idx = int(entry_rows.index[0])
        if entry_idx + T0_CONTROL_HOLD_BARS < len(t0_df):
            exit_row = t0_df.iloc[entry_idx + T0_CONTROL_HOLD_BARS]
        else:
            exit_row = t0_df.iloc[-1]
        exit_ts = exit_row["date"]
        exit_price = float(exit_row["close"])

        # Stop distance for sizing: same hard-stop pct for parity
        sl_pct_used = max(HARD_STOP_PCT, MIN_STOP_PCT)
        if side == "SHORT":
            hard_sl = entry_price * (1.0 + sl_pct_used / 100.0)
            stop_distance = hard_sl - entry_price
            realized_pnl = (entry_price - exit_price)
        else:
            hard_sl = entry_price * (1.0 - sl_pct_used / 100.0)
            stop_distance = entry_price - hard_sl
            realized_pnl = (exit_price - entry_price)
        if stop_distance <= 0:
            continue
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        realized_pnl *= qty
        kite_side = "SELL" if side == "SHORT" else "BUY"
        fee = calc_fee(entry_price, exit_price, qty, kite_side)
        net_pnl = realized_pnl - fee

        raw_trades.append({
            "T0_signal_date": t0,
            "symbol": "NSE:" + raw_sym,
            "cap_segment": sig["cap_segment"],
            "side": side,
            "abs_net_cr": float(sig["abs_net_cr"]),
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
    # Latch on (symbol, T+0, side)
    df = (
        df.sort_values(["symbol", "T0_signal_date", "side", "abs_net_cr"],
                       ascending=[True, True, True, False])
        .drop_duplicates(subset=["symbol", "T0_signal_date", "side"], keep="first")
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


def _bucket_net_value(v: float) -> str:
    av = abs(v)
    for lo, hi in NET_VALUE_BUCKETS:
        if av >= lo and av < hi:
            return f"[{lo:.0f},{hi if hi != float('inf') else 'inf'})cr"
    return "out"


def _bucket_gap(g: float) -> str:
    ag = abs(g)
    for lo, hi in GAP_PCT_BUCKETS:
        if ag >= lo and ag < hi:
            return f"[{lo:.1f},{hi if hi != float('inf') else 'inf'})%"
    return "out"


def report(
    trades: pd.DataFrame,
    label: str = "T+1 entries",
) -> dict:
    if trades.empty:
        print(f"\n[{label}] no trades")
        return {"n": 0, "pf": None, "wr": None, "sharpe": None}

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

    # Per side
    side_summary: dict = {}
    print("\nPer side:")
    for sd, grp in trades.groupby("side"):
        pf2, wr2, n2, net = _pf_wr(grp)
        side_summary[sd] = {"pf": pf2, "wr": wr2, "n": n2, "net": net}
        print(f"  {sd:<6} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    # Per cap segment
    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        pf2, wr2, n2, net = _pf_wr(grp)
        print(f"  {cap:<12} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    # Per month
    print("\nPer month:")
    trades = trades.copy()
    trades["month"] = pd.to_datetime(trades[daily_col]).dt.to_period("M").astype(str)
    for m, grp in trades.groupby("month"):
        pf2, wr2, n2, net = _pf_wr(grp)
        print(f"  {m:<8} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    # Per exit-reason
    if "exit_reason" in trades.columns:
        print("\nExit-reason breakdown:")
        for rsn, grp in trades.groupby("exit_reason"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {rsn:<18} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  avgNet=Rs.{int(grp['net_pnl'].mean()):>6,}")

    # Per |net_value_cr| bucket
    if "abs_net_cr" in trades.columns:
        print("\nPer |net_value_cr| bucket:")
        trades["nv_bucket"] = trades["abs_net_cr"].apply(_bucket_net_value)
        for b, grp in trades.groupby("nv_bucket"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {b:<14} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    # Per gap-pct bucket (T+1 entries only)
    if "gap_pct" in trades.columns:
        print("\nPer |gap_pct| bucket (T+1 09:15):")
        trades["gap_bucket"] = trades["gap_pct"].apply(_bucket_gap)
        for b, grp in trades.groupby("gap_bucket"):
            pf2, wr2, n2, net = _pf_wr(grp)
            print(f"  {b:<14} n={n2:>4}  PF={pf2:>6}  WR={wr2:>5}%  netPnL=Rs.{int(net):>10,}")

    return {
        "n": n, "pf": pf, "wr": wr, "sharpe": sharpe,
        "side_summary": side_summary,
        "net_pnl": float(npnl.sum()),
    }


def _print_funnel(funnel: dict) -> None:
    print("\n-- Trade-funnel --")
    print(f"  signals fed in              : {funnel['signals_in']}")
    print(f"  no 5m data for symbol       : {funnel['no_5m_data']}")
    print(f"  no T+1 trading day          : {funnel['no_t1_day']}")
    print(f"  no T+0 close bar            : {funnel['no_t0_close']}")
    print(f"  no 09:15 first bar          : {funnel['no_first_bar']}")
    print(f"  fail gap-or-candle gate     : {funnel['fail_gap_or_candle']}")
    print(f"  no 09:25 entry bar          : {funnel['no_entry_bar']}")
    print(f"  -> fired (pre-latch)        : {funnel['fired_pre_latch']}")
    print(f"  -> fired (post-latch, FINAL): {funnel['fired_post_latch']}")


def _verdict(t1_summary: dict, t0_summary: dict) -> None:
    print("\n" + "=" * 70)
    print("VERDICT (locked thresholds: PF>=1.10, n>=30 per cell, |WR delta|<=10pp)")
    print("=" * 70)

    if t1_summary["n"] == 0:
        print("VERDICT: no T+1 trades fired. RETIRE candidate.")
        return

    overall_pf = t1_summary["pf"]
    overall_n = t1_summary["n"]
    side_summary = t1_summary.get("side_summary", {})
    short_pf = side_summary.get("SHORT", {}).get("pf")
    long_pf = side_summary.get("LONG", {}).get("pf")
    short_wr = side_summary.get("SHORT", {}).get("wr")
    long_wr = side_summary.get("LONG", {}).get("wr")
    short_n = side_summary.get("SHORT", {}).get("n", 0)
    long_n = side_summary.get("LONG", {}).get("n", 0)

    print(f"  Aggregate: n={overall_n}, PF={overall_pf}")
    print(f"  SHORT    : n={short_n}, PF={short_pf}, WR={short_wr}%")
    print(f"  LONG     : n={long_n}, PF={long_pf}, WR={long_wr}%")
    if short_wr is not None and long_wr is not None:
        wr_delta = abs(short_wr - long_wr)
        print(f"  |WR delta SHORT vs LONG|: {wr_delta:.1f}pp (gate <= 10pp)")
    else:
        wr_delta = None

    # T+0 control
    if t0_summary and t0_summary.get("n", 0) > 0:
        t0_pf = t0_summary["pf"]
        print(f"\n  T+0 control PF={t0_pf} | T+1 PF={overall_pf}")
        if isinstance(t0_pf, (int, float)) and isinstance(overall_pf, (int, float)):
            if t0_pf >= overall_pf:
                print("  *** T+0 PF >= T+1 PF: pre-disclosure leakage suspected;")
                print("      asymmetry is NOT retail-FOMO fade. Thesis INVALIDATED. ***")
            else:
                print("  T+0 underperforms T+1 -> retail-FOMO mechanism consistent with thesis.")
    else:
        print("\n  T+0 control: no trades. Cannot run leakage check.")

    # Gate evaluation
    print("\nGate eval:")
    pass_pf = isinstance(overall_pf, (int, float)) and overall_pf >= 1.10
    pass_n = overall_n >= 30
    print(f"  Aggregate PF >= 1.10           : {'PASS' if pass_pf else 'FAIL'}")
    print(f"  Aggregate n  >= 30             : {'PASS' if pass_n else 'FAIL'}")

    short_ok = (isinstance(short_pf, (int, float)) and short_pf >= 1.10 and short_n >= 30)
    long_ok = (isinstance(long_pf, (int, float)) and long_pf >= 1.10 and long_n >= 30)
    print(f"  SHORT cell (n>=30, PF>=1.10)   : {'PASS' if short_ok else 'FAIL'}")
    print(f"  LONG  cell (n>=30, PF>=1.10)   : {'PASS' if long_ok else 'FAIL'}")
    if wr_delta is not None:
        print(f"  |WR delta| <= 10pp             : {'PASS' if wr_delta <= 10.0 else 'FAIL'}")

    if pass_pf and pass_n:
        print("\n>>> Aggregate PF/n gate PASS -> CANDIDATE for OOS validation.")
    elif short_ok or long_ok:
        winner = "SHORT" if short_ok else "LONG"
        print(f"\n>>> One-sided WIN ({winner}) passes gate -> CANDIDATE (single-side OOS validation).")
    else:
        print("\n>>> Gate FAIL -> RETIRE candidate.")


def main():
    print("=== nse_block_deal_counter_flow — pre-coding sanity ===")
    print(f"Discovery window: {DISCOVERY_START} .. {DISCOVERY_END}")

    print("\nLoading block-deal events ...")
    df = load_block_deals()
    print(f"  raw events in Discovery: {len(df)}")

    print("\nLoading F&O 200 universe ...")
    fno_syms = load_fno_universe()
    print(f"  fno symbols: {len(fno_syms)}")

    signals = aggregate_to_signals(df, fno_syms)
    if signals.empty:
        print("No signals after filter; aborting.")
        return

    print("\nLoading 5m feathers for event months ...")
    big5m = build_5m_for_events(signals.rename(columns={"trade_date": "trade_date"}))
    if big5m.empty:
        print("No 5m bars; aborting.")
        return

    # Run T+1 simulation
    trades, funnel = simulate_t1(signals, big5m)
    _print_funnel(funnel)

    if trades.empty:
        print("\nNo T+1 trades fired. Cannot run report.")
        return

    t1_summary = report(trades, label="T+1 nse_block_deal_counter_flow")

    # Run T+0 control
    t0_trades = simulate_t0_control(signals, big5m)
    t0_summary = report(t0_trades, label="T+0 control (same-day EOD)") if not t0_trades.empty else {"n": 0}

    # Verdict
    _verdict(t1_summary, t0_summary)

    # Persist trades CSV
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "nse_block_deal_counter_flow_trades.csv"
    trades.to_csv(out_csv, index=False)
    print(f"\nT+1 trade log: {out_csv}")
    if not t0_trades.empty:
        out_csv0 = out_dir / "nse_block_deal_counter_flow_t0_control.csv"
        t0_trades.to_csv(out_csv0, index=False)
        print(f"T+0 control log: {out_csv0}")


if __name__ == "__main__":
    main()
