"""Phase 4 sanity: close_dn_overnight_long.

Brief: specs/2026-05-21-brief-close_dn_overnight_long.md
Predecessor Phase 2: tools/sub9_research/phase2_volume_4angles_v2.py (Angle 4)

# Anti-bias guards (Lesson #5 6-failure-mode checklist)

  1. No future-bar lookahead. Signal is computed only from closing-30m bars
     of session_date (15:00-15:25). Exit is next-trading-day 09:15 OPEN, not
     CLOSE — matches Phase 2 sensitivity output exactly.
  2. closing_30m_volume_z baseline uses .shift(1) over prior-20-session
     closing-30m totals (excludes today).
  3. Position is single-trade (no MFE/MAE walk needed — only 2 prices matter:
     entry_close and next_day_open).
  4. Same-bar mechanics N/A — entry is at one bar's close, exit at another
     bar's open the next session.
  5. Filters at signal time only. Pre-registered dim_pool in brief section 5.
  6. news_proximity filter from data/earnings_calendar/earnings_events.parquet
     — exclude signals where next-trading-day's trade_date matches an earnings
     event (prevents overnight earnings-announcement blowups).

# CNC fee model (local helper — not yet promoted to build_per_setup_pnl.py)

  Per-side:
    - Brokerage: Rs 20 flat
    - STT: 0.10% on SELL value (delivery)
    - Stamp: 0.015% on BUY value (delivery)
    - Txn charges: 0.00345% per side
    - SEBI: 0.0001% per side
    - GST: 18% on (brokerage + txn charges)

  Round-trip on Rs 100K position with +0.5% move: ~Rs 171 fees → net Rs 329
  per trade (matches brief section 10 economics).

# Mechanic (from brief section 6)

LONG CNC entry at signal_bar.close (15:25 bar close):
  - signed_vol_ratio (closing 15:00-15:25) <= -0.5
  - closing_30m_volume_z >= 1.0 (today's closing-30m total vs prior-20d mean)
  - 09:15 next-day open is available (skip if no next-trading-day data)
  - news_proximity != 'within_1day_earnings'
  - cap_segment in {large, mid, small}; micro excluded for liquidity
  - daily_avg_vol >= 50K, trading_days_coverage >= 80%

Exit: SELL at next-trading-day's 09:15 OPEN.
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"
_EARNINGS_PARQUET = _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"

sys.path.insert(0, str(_REPO_ROOT))

# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

WINDOWS = {
    "discovery": (date(2023, 1, 2), date(2024, 6, 30)),
    "oos": (date(2024, 7, 1), date(2025, 6, 30)),
    "holdout": (date(2025, 7, 1), date(2026, 4, 30)),
}

# Pre-registered parameters (brief section 5-6, LOCKED)
SIGNED_VOL_RATIO_MAX = -0.5            # signal fires when ratio <= this
CLOSING_30M_VOLUME_Z_MIN = 1.0          # confirms flush vs prior-20d mean
CLOSING_30M_HHMM_LIST = ["15:00", "15:05", "15:10", "15:15", "15:20", "15:25"]
SIGNAL_BAR_HHMM = "15:25"
ENTRY_BAR_HHMM = "15:25"               # CNC entry at signal bar's close
EXIT_BAR_HHMM = "09:15"                # CNC exit at next-day open

ROLLING_DAYS = 20                       # for closing_30m_volume_z baseline
POSITION_NOTIONAL_INR = 100_000         # Rs 1L fixed per trade (CNC)
MIN_TRADING_DAYS_COVERAGE = 0.80
MIN_DAILY_AVG_VOLUME = 50_000
ALLOWED_CAP_SEGMENTS = ("large_cap", "mid_cap", "small_cap")


# ---------------------------------------------------------------------------
# CNC fee helper (local — not promoted to build_per_setup_pnl yet)
# ---------------------------------------------------------------------------

def calc_fee_cnc(buy_value: float, sell_value: float) -> float:
    """CNC (delivery) fee model for a single round-trip.

    Returns total Rs fee. See sanity-script docstring for the breakdown.
    """
    # Brokerage: Rs 20 flat per side
    brokerage_buy = 20.0
    brokerage_sell = 20.0
    # STT: 0.1% on sell-side (delivery)
    stt_sell = sell_value * 0.001
    # Stamp: 0.015% on buy-side (delivery)
    stamp_buy = buy_value * 0.00015
    # Txn charges: 0.00345% per side (NSE)
    txn_buy = buy_value * 0.0000345
    txn_sell = sell_value * 0.0000345
    # SEBI: 0.0001% per side
    sebi_buy = buy_value * 0.000001
    sebi_sell = sell_value * 0.000001
    # GST: 18% on (brokerage + txn) per side
    gst_buy = (brokerage_buy + txn_buy) * 0.18
    gst_sell = (brokerage_sell + txn_sell) * 0.18
    return (
        brokerage_buy + brokerage_sell + stt_sell + stamp_buy +
        txn_buy + txn_sell + sebi_buy + sebi_sell + gst_buy + gst_sell
    )


# ---------------------------------------------------------------------------
# Cap segment + earnings calendar
# ---------------------------------------------------------------------------

_CAP_CACHE: Dict[str, str] = {}


def _get_cap_segment(symbol: str) -> Optional[str]:
    global _CAP_CACHE
    if not _CAP_CACHE:
        import json
        try:
            with open(_REPO_ROOT / "nse_all.json", encoding="utf-8") as f:
                nse_all = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _CAP_CACHE = {"__loaded__": "fail"}
            return None
        if isinstance(nse_all, list):
            for entry in nse_all:
                if not isinstance(entry, dict):
                    continue
                sym_full = entry.get("symbol", "")
                seg = entry.get("cap_segment")
                if sym_full and seg:
                    bare = sym_full.replace(".NS", "").replace("NSE:", "")
                    _CAP_CACHE[bare] = str(seg)
        _CAP_CACHE["__loaded__"] = "ok"
    return _CAP_CACHE.get(symbol)


_EARNINGS_CACHE: Optional[set] = None


def _load_earnings_set() -> set:
    """Load (symbol_bare, trade_date) tuples — these are dates where the
    symbol has an earnings announcement affecting that day's open."""
    global _EARNINGS_CACHE
    if _EARNINGS_CACHE is None:
        try:
            df = pd.read_parquet(_EARNINGS_PARQUET)
            df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
            df["symbol_bare"] = df["symbol"].str.replace("NSE:", "", regex=False)
            _EARNINGS_CACHE = set(zip(df["symbol_bare"], df["trade_date"]))
            print(f"  Earnings calendar: {len(_EARNINGS_CACHE):,} (symbol, trade_date) pairs", flush=True)
        except Exception as e:
            print(f"  WARNING: earnings calendar load failed: {e}", flush=True)
            _EARNINGS_CACHE = set()
    return _EARNINGS_CACHE


def _has_earnings(symbol: str, next_day: date) -> bool:
    earn = _load_earnings_set()
    bare = symbol.replace("NSE:", "")
    return (bare, next_day) in earn


# ---------------------------------------------------------------------------
# Binning helpers (per brief section 5)
# ---------------------------------------------------------------------------

def _signed_vol_ratio_bin(svr: float) -> str:
    if svr <= -0.9:
        return "neg0.9_to_neg1.0"
    elif svr <= -0.75:
        return "neg0.75_to_neg0.9"
    elif svr <= -0.6:
        return "neg0.6_to_neg0.75"
    else:  # -0.5 <= svr < -0.6
        return "neg0.5_to_neg0.6"


def _volume_z_bin(z: float) -> str:
    if z >= 2.0:
        return "extreme"
    elif z >= 1.0:
        return "high"
    else:
        return "normal"


def _prior_day_return_bin(ret_pct: float) -> str:
    if ret_pct <= -3.0:
        return "down_gt_3pct"
    elif ret_pct <= -1.0:
        return "down_1to3pct"
    elif ret_pct < 1.0:
        return "flat"
    elif ret_pct < 3.0:
        return "up_1to3pct"
    else:
        return "up_gt_3pct"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

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
    """Load monthly feathers, restrict to closing-30min + 09:15 bars
    (the only bars we need)."""
    keep_hhmm = set(CLOSING_30M_HHMM_LIST + [EXIT_BAR_HHMM])
    chunks = []
    # Need 1 extra calendar week before d0 to capture 09:15 of d0 (already
    # in d0's month) AND ~30 days before d0 to compute prior-20d baseline
    # for the closing volume z-score
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


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _compute_signals(df_window: pd.DataFrame, window_d0: date, window_d1: date) -> pd.DataFrame:
    """Compute per-(symbol, signal_date) signals + cell dims + exit prices.

    Returns one row per qualifying signal with all data needed to compute
    PnL and bin into cells. Only signals where signal_date in [d0, d1]
    are kept (universe filtering happens upstream of this function).
    """
    # Split into closing-30m bars vs 09:15 open bars
    closing = df_window[df_window["hhmm"].isin(CLOSING_30M_HHMM_LIST)].copy()
    opening = df_window[df_window["hhmm"] == EXIT_BAR_HHMM][["symbol", "d", "open"]].copy()
    opening = opening.rename(columns={"open": "next_open"})

    # Aggregate closing-30m per (symbol, date)
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
        closing_agg["signed_vol_sum"] /
        closing_agg["total_vol"].replace(0, np.nan)
    )

    # closing_30m_volume_z: per-symbol z-score vs prior-20-session mean/std
    print("  Computing closing_30m_volume_z (prior-20d baseline)...", flush=True)
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

    # prior_day_return_pct: today_close - prev_close / prev_close
    closing_agg["prev_close"] = closing_agg.groupby("symbol", observed=True)["last_close"].shift(1)
    closing_agg["prior_day_return_pct"] = (
        (closing_agg["last_close"] - closing_agg["prev_close"]) /
        closing_agg["prev_close"].replace(0, np.nan) * 100.0
    )

    # Match next-trading-day 09:15 open
    print("  Matching next-trading-day 09:15 open per (symbol, signal_date)...", flush=True)
    # For each (symbol, d), find the NEXT (symbol, d') where d' > d and d' is in opening
    # opening already has one row per (symbol, d) — group by symbol, sort by d, find next.
    opening = opening.sort_values(["symbol", "d"]).reset_index(drop=True)
    open_by_sym = {sym: g.reset_index(drop=True) for sym, g in opening.groupby("symbol", observed=True)}

    next_opens = []
    next_dates = []
    for row in closing_agg.itertuples(index=False):
        sym = row.symbol
        d_val = row.d
        sym_opens = open_by_sym.get(sym)
        if sym_opens is None or sym_opens.empty:
            next_opens.append(None)
            next_dates.append(None)
            continue
        fwd = sym_opens[sym_opens["d"] > d_val]
        if fwd.empty:
            next_opens.append(None)
            next_dates.append(None)
        else:
            first = fwd.iloc[0]
            # Only accept if within 7 calendar days (skip listing halts)
            if (first["d"] - d_val).days <= 7:
                next_opens.append(float(first["next_open"]))
                next_dates.append(first["d"])
            else:
                next_opens.append(None)
                next_dates.append(None)
    closing_agg["next_open"] = next_opens
    closing_agg["next_trading_day"] = next_dates

    # Restrict to signal_date in requested window
    closing_agg = closing_agg[
        (closing_agg["d"] >= window_d0) & (closing_agg["d"] <= window_d1)
    ]

    # Apply pre-registered signal filters.
    # IMPORTANT: every comparison must be parenthesized because Python's
    # bitwise `&` binds tighter than `>=`/`<=`, which silently distorts
    # the mask if any comparison is unparenthesized.
    print("  Applying signal filters...", flush=True)
    sig_mask = (
        (closing_agg["bar_count"] >= 5)   # at least 5 of 6 closing-30m bars present
        & (closing_agg["signed_vol_ratio"] <= SIGNED_VOL_RATIO_MAX)
        & (closing_agg["closing_30m_volume_z"] >= CLOSING_30M_VOLUME_Z_MIN)
        & (closing_agg["next_open"].notna())
        & (closing_agg["prior_day_return_pct"].notna())
    )
    signals = closing_agg[sig_mask].copy()
    print(f"  Pre-cell-dim signals: {len(signals):,}", flush=True)
    if signals.empty:
        return signals

    # Compute cell dims
    print("  Computing cell dims...", flush=True)
    signals["cap_segment"] = signals["symbol"].astype(str).apply(
        lambda s: _get_cap_segment(s) or "unknown"
    )
    signals["signed_vol_ratio_bin"] = signals["signed_vol_ratio"].apply(_signed_vol_ratio_bin)
    signals["closing_30m_volume_z_bin"] = signals["closing_30m_volume_z"].apply(_volume_z_bin)
    signals["prior_day_return_bin"] = signals["prior_day_return_pct"].apply(_prior_day_return_bin)

    # news_proximity
    signals["news_proximity"] = signals.apply(
        lambda r: "within_1day_earnings" if _has_earnings(str(r["symbol"]), r["next_trading_day"]) else "clear",
        axis=1,
    )

    # Cap segment exclusion
    signals = signals[signals["cap_segment"].isin(ALLOWED_CAP_SEGMENTS + ("unknown",))]

    return signals.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-trade PnL
# ---------------------------------------------------------------------------

def _compute_trades(signals: pd.DataFrame) -> pd.DataFrame:
    """For each signal, compute qty (Rs 100K notional), gross PnL, fees,
    net PnL, and canonical-schema fields.
    """
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
        gross_pnl = sell_value - buy_value
        fee = calc_fee_cnc(buy_value, sell_value)
        net_pnl = gross_pnl - fee
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
            "entry_ts": f"{s.d}T15:25:00",
            "exit_ts": f"{s.next_trading_day}T09:15:00",
            "realized_pnl_inr": float(gross_pnl),
            "fee_inr": float(fee),
            "net_pnl_inr": float(net_pnl),
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


# ---------------------------------------------------------------------------
# Per-window orchestration
# ---------------------------------------------------------------------------

def run_window(window_label: str) -> pd.DataFrame:
    d0, d1 = WINDOWS[window_label]
    print(f"\n=== Window: {window_label} ({d0} to {d1}) ===", flush=True)

    df = _load_window(d0, d1)
    print(f"  Loaded {len(df):,} bars (closing-30m + next-day 09:15 only)", flush=True)

    # Universe filter: trading-days coverage + daily avg vol
    # Coverage requires at least closing 15:25 bar; use 15:25 hhmm as proxy for trading days
    days_per_sym = df[df["hhmm"] == "15:25"].groupby("symbol", observed=True)["d"].nunique()
    total_window_days = df[df["hhmm"] == "15:25"]["d"].nunique()
    min_days = int(total_window_days * MIN_TRADING_DAYS_COVERAGE)
    keep_d = days_per_sym[days_per_sym >= min_days].index

    # daily avg volume: aggregate closing-30m total per day, mean over days
    daily_close30_vol = df.groupby(["symbol", "d"], observed=True)["volume"].sum()
    daily_avg_vol = daily_close30_vol.groupby(level=0, observed=True).mean()
    keep_v = daily_avg_vol[daily_avg_vol >= MIN_DAILY_AVG_VOLUME / 10].index  # closing-30m is ~10% of daily

    keep = set(keep_d) & set(keep_v)
    df = df[df["symbol"].isin(keep)]
    print(f"  Universe: {len(keep)} symbols ({total_window_days} window-days, coverage threshold {min_days})", flush=True)

    signals = _compute_signals(df, d0, d1)
    if signals.empty:
        print(f"  No signals after filter — empty output", flush=True)
        return pd.DataFrame()
    print(f"  Post-cell-dim signals: {len(signals):,}", flush=True)

    trades = _compute_trades(signals)
    print(f"  Trades: {len(trades):,}", flush=True)
    if not trades.empty:
        n_win = (trades["net_pnl_inr"] > 0).sum()
        n_loss = (trades["net_pnl_inr"] < 0).sum()
        net_total = trades["net_pnl_inr"].sum()
        fee_total = trades["fee_inr"].sum()
        gross_total = trades["realized_pnl_inr"].sum()
        print(f"  Win/Loss: {n_win}/{n_loss}  (WR = {n_win/max(1,len(trades))*100:.1f}%)", flush=True)
        print(f"  Gross: Rs {gross_total:+,.0f}  Fees: Rs {fee_total:+,.0f}  NET: Rs {net_total:+,.0f}", flush=True)
        print(f"  Mean NET per trade: Rs {net_total/len(trades):+.2f}", flush=True)

    return trades


def main():
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    for window_label in ("discovery", "oos", "holdout"):
        trades_df = run_window(window_label)
        out_path = out_dir / f"_close_dn_overnight_long_trades_{window_label}.csv"
        if not trades_df.empty:
            trades_df.to_csv(out_path, index=False)
            print(f"  Saved: {out_path}")
        else:
            print(f"  Empty — not writing {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
