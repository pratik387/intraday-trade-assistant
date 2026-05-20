"""Phase 4 sanity: extreme_vol_revert_long.

Brief: specs/2026-05-20-brief-extreme_vol_revert_long.md
Predecessor Phase 2: tools/sub9_research/phase2_attention_v2_baseline.py

# Anti-bias guards (Lesson #5 6-failure-mode checklist)

  1. No day_high / day_low / day_vwap / day_close at signal time. All
     features use bars[:i+1] only (session-to-date aggregates).
  2. Volume baseline excludes current bar (.shift(1) on rolling-20d mean
     per (symbol, hhmm)).
  3. Mode B entry at bars[i+1].open. Path walk starts at bars[i+1] (entry
     happened at OPEN of i+1, full intra-bar range is post-entry).
  4. Same-bar SL+T2 hits: SL wins (pessimistic).
  5. Filters at signal time only. NO post-hoc filter tuning -- all
     parameters pre-registered in brief section 5.
  6. Output canonical schema validated via tools.methodology.sanity_csv_schema.

# Mechanic (from brief sections 5-6)

LONG entry conditions (signal bar):
  - vol_ratio (today vs prior-20d same-bar mean) >= 5.0
  - bar_return_pct < 0 AND in DN_Q1..Q3 (mild down bars only;
    quintile edges from baseline pool, locked at session start)
  - 09:30 <= signal_bar_hhmm < 15:00
  - cap_segment in {large_cap, mid_cap, small_cap} (NO micro_cap; we
    measure cap_segment as a dim, but micro is excluded for liquidity)

Entry: bars[i+1].open
Hard SL: min(signal_bar.low * 0.998, entry * 0.99)  — deeper of two
T1: entry + 1.0R (1R = entry - hard_sl)
T2: entry + 2.0R
Time stop: 14:30 IST (sweep-locked from brief section 5)
"""
from __future__ import annotations

import sys
from datetime import date, datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MONTHLY_DIR = _REPO_ROOT / "backtest-cache-download" / "monthly"

sys.path.insert(0, str(_REPO_ROOT))
from tools.methodology import sanity_csv_schema  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---------------------------------------------------------------------------
# Window definitions
# ---------------------------------------------------------------------------

WINDOWS = {
    "discovery": (date(2023, 1, 2), date(2024, 6, 30)),     # 1.5yr Discovery
    "oos": (date(2024, 7, 1), date(2025, 6, 30)),            # 1yr OOS
    "holdout": (date(2025, 7, 1), date(2026, 4, 30)),        # 10mo Holdout
}

# Pre-registered parameters (brief section 5-6, LOCKED)
ROLLING_DAYS = 20
VOL_RATIO_MIN = 5.0
SIGNAL_HHMM_MIN = "09:30"
SIGNAL_HHMM_MAX = "14:55"     # last signal bar (entry at 15:00, exit at 15:10)
TIME_STOP_HHMM_LIST = [1430, 1500]    # 14:30 and 15:00 — sweep later
SL_PCT_FROM_LOW = 0.002          # 0.2% below signal bar low
SL_PCT_FROM_ENTRY = 0.01         # alternative 1% below entry
T1_R_MULT = 1.0
T2_R_MULT = 2.0
RISK_PER_TRADE_RUPEES = 1000     # standard

# Cap segment filter: micro_cap excluded for liquidity
ALLOWED_CAP_SEGMENTS = ("large_cap", "mid_cap", "small_cap")
MIN_TRADING_DAYS_PER_SYMBOL = 200
MIN_DAILY_AVG_VOLUME = 50_000

# DN quintile edges — computed at start of run from baseline pool
# (LOCKED from each window's first 90 days to prevent look-ahead within window)
DN_QUINTILE_BASELINE_DAYS = 90


def _months_between(d0: date, d1: date) -> List[Tuple[int, int]]:
    out = []
    y, m = d0.year, d0.month
    while (y, m) <= (d1.year, d1.month):
        out.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return out


# Lazy cap-segment lookup
_CAP_CACHE: Dict[str, str] = {}

def _get_cap_segment(symbol: str) -> Optional[str]:
    """Lookup symbol's cap_segment from cached nse_all.json.

    nse_all.json is a LIST of dicts: [{"symbol": "AAATECH.NS", "cap_segment":
    "unknown", ...}, ...]. Keys: symbol, market_cap_cr, cap_segment, mis_enabled,
    mis_leverage, mis_margin_pct.
    """
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
                    # Strip .NS suffix and NSE: prefix if present
                    bare = sym_full.replace(".NS", "").replace("NSE:", "")
                    _CAP_CACHE[bare] = str(seg)
        _CAP_CACHE["__loaded__"] = "ok"
    return _CAP_CACHE.get(symbol)


def _load_window(d0: date, d1: date) -> pd.DataFrame:
    chunks = []
    for (yyyy, mm) in _months_between(d0, d1):
        p = _MONTHLY_DIR / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
        if not p.exists():
            continue
        df = pd.read_feather(p, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        # Memory-efficient dtypes
        df["open"] = df["open"].astype("float32")
        df["high"] = df["high"].astype("float32")
        df["low"] = df["low"].astype("float32")
        df["close"] = df["close"].astype("float32")
        df["volume"] = df["volume"].astype("float32")
        chunks.append(df)
    df = pd.concat(chunks, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["d"] = df["date"].dt.date
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df["symbol"] = df["symbol"].astype("category")
    df = df.drop(columns=["date"])
    # Restrict to window
    mask = (df["d"] >= d0) & (df["d"] <= d1)
    return df[mask].reset_index(drop=True)


def _add_vol_ratio_and_bar_return(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["symbol", "hhmm", "d"])
    df["vol_mean20"] = (
        df.groupby(["symbol", "hhmm"], observed=True)["volume"]
        .transform(lambda s: s.rolling(ROLLING_DAYS, min_periods=ROLLING_DAYS).mean().shift(1))
        .astype("float32")
    )
    df["vol_ratio"] = (df["volume"] / df["vol_mean20"]).astype("float32")
    df["bar_return_pct"] = ((df["close"] - df["open"]) / df["open"] * 100.0).astype("float32")
    df = df.drop(columns=["vol_mean20"])
    return df


def _compute_dn_quintile_edges(df: pd.DataFrame, d0: date) -> np.ndarray:
    """Lock DN quintile edges from FIRST 90 days of the window only."""
    cutoff = pd.Timestamp(d0).date() + pd.Timedelta(days=DN_QUINTILE_BASELINE_DAYS).to_pytimedelta()
    early = df[(df["d"] <= cutoff) & (df["bar_return_pct"] < 0)]
    if early.empty:
        return np.array([-1.0, -0.5, -0.2, -0.05, 0.0])  # fallback
    edges = pd.qcut(
        early["bar_return_pct"].astype("float64"),
        q=5, retbins=True, duplicates="drop",
    )[1]
    return edges


def _identify_signals(df: pd.DataFrame, dn_edges: np.ndarray) -> pd.DataFrame:
    """Apply locked filter rules to identify LONG-entry candidates."""
    mask = (
        (df["vol_ratio"] >= VOL_RATIO_MIN) &
        (df["bar_return_pct"] < 0) &
        (df["bar_return_pct"] >= dn_edges[2]) &   # at least DN_Q3 (mild-to-mid)
        (df["hhmm"] >= SIGNAL_HHMM_MIN) &
        (df["hhmm"] <= SIGNAL_HHMM_MAX)
    )
    sig = df.loc[mask, ["symbol", "d", "hhmm", "open", "high", "low", "close",
                        "volume", "vol_ratio", "bar_return_pct"]]
    return sig.reset_index(drop=True)


def _hhmm_add(hhmm: str, minutes: int) -> str:
    h, m = hhmm.split(":")
    total = int(h) * 60 + int(m) + minutes
    return f"{total // 60:02d}:{total % 60:02d}"


def _build_session_bars_index(df: pd.DataFrame) -> Dict[Tuple[str, date], List[Tuple[str, float, float, float, float]]]:
    """Build (symbol, date) -> list of (hhmm, open, high, low, close) tuples,
    sorted by hhmm. One-time O(N) build for O(1) subsequent lookup.
    """
    print("  Building per-session bars index...", flush=True)
    df_sorted = df.sort_values(["symbol", "d", "hhmm"])
    idx: Dict[Tuple[str, date], List[Tuple[str, float, float, float, float]]] = {}
    # Extract via itertuples (fast)
    for r in df_sorted[["symbol", "d", "hhmm", "open", "high", "low", "close"]].itertuples(index=False):
        sym = str(r.symbol)
        key = (sym, r.d)
        if key not in idx:
            idx[key] = []
        idx[key].append((r.hhmm, float(r.open), float(r.high), float(r.low), float(r.close)))
    print(f"  Index built: {len(idx):,} (symbol, date) keys", flush=True)
    return idx


def _walk_to_exit_indexed(
    session_bars: List[Tuple[str, float, float, float, float]],
    start_hhmm: str,
    entry_price: float, hard_sl: float, t1: float, t2: float,
    time_stop_hhmm_int: int,
) -> Tuple[float, str, str, float, float]:
    """Walk forward through pre-indexed session bars to find exit.

    session_bars is a list of (hhmm, open, high, low, close) tuples sorted
    by hhmm. start_hhmm is the entry bar (Mode B: bars[i+1].open).
    """
    R = entry_price - hard_sl
    if R <= 0:
        return (entry_price, "invalid_R", start_hhmm, 0.0, 0.0)

    mfe = 0.0
    mae = 0.0
    exit_price = entry_price
    exit_reason = "time_stop"
    exit_hhmm = start_hhmm
    walked = False

    for (hhmm, op, hi, lo, cl) in session_bars:
        if hhmm < start_hhmm:
            continue
        walked = True
        # Update MFE/MAE
        if hi > entry_price:
            mfe = max(mfe, hi - entry_price)
        if lo < entry_price:
            mae = max(mae, entry_price - lo)
        # Check time stop
        cur_hhmm_int = int(hhmm.replace(":", ""))
        if cur_hhmm_int >= time_stop_hhmm_int:
            exit_price = cl
            exit_reason = "time_stop"
            exit_hhmm = hhmm
            break
        # Same-bar SL+target -> SL wins
        sl_hit = lo <= hard_sl
        t2_hit = hi >= t2
        t1_hit = hi >= t1
        if sl_hit and (t1_hit or t2_hit):
            exit_price = hard_sl
            exit_reason = "same_bar_sl"
            exit_hhmm = hhmm
            break
        if sl_hit:
            exit_price = hard_sl
            exit_reason = "sl"
            exit_hhmm = hhmm
            break
        if t2_hit:
            exit_price = t2
            exit_reason = "t2"
            exit_hhmm = hhmm
            break
        if t1_hit:
            exit_price = t1
            exit_reason = "t1"
            exit_hhmm = hhmm
            break

    if not walked:
        return (entry_price, "no_data", start_hhmm, 0.0, 0.0)
    mfe_r = mfe / R if R > 0 else 0.0
    mae_r = mae / R if R > 0 else 0.0
    return (exit_price, exit_reason, exit_hhmm, mfe_r, mae_r)


def _close_at_indexed(session_bars: List[Tuple[str, float, float, float, float]],
                       target_hhmm: str) -> float:
    """Get close price at specific hhmm in pre-indexed bars."""
    for (hhmm, op, hi, lo, cl) in session_bars:
        if hhmm == target_hhmm:
            return cl
    return float("nan")


def run_window(window_label: str, time_stop_hhmm_int: int = 1430) -> pd.DataFrame:
    d0, d1 = WINDOWS[window_label]
    print(f"\n=== Window: {window_label} ({d0} to {d1}) ===", flush=True)

    df = _load_window(d0, d1)
    print(f"  Loaded {len(df):,} bars", flush=True)

    # Filter universe
    days_per_sym = df.groupby("symbol", observed=True)["d"].nunique()
    keep_d = days_per_sym[days_per_sym >= MIN_TRADING_DAYS_PER_SYMBOL].index
    daily_vol = df.groupby(["symbol", "d"], observed=True)["volume"].sum().groupby(level=0, observed=True).mean()
    keep_v = daily_vol[daily_vol >= MIN_DAILY_AVG_VOLUME].index
    keep = set(keep_d) & set(keep_v)
    df = df[df["symbol"].isin(keep)]
    print(f"  Universe: {len(keep)} symbols, {len(df):,} bars", flush=True)

    df = _add_vol_ratio_and_bar_return(df)
    df = df[df["vol_ratio"].notna() & df["bar_return_pct"].notna()]
    print(f"  Valid baselines: {len(df):,}", flush=True)

    dn_edges = _compute_dn_quintile_edges(df, d0)
    print(f"  DN quintile edges (from first {DN_QUINTILE_BASELINE_DAYS}d): "
          f"{[f'{x:.3f}' for x in dn_edges]}", flush=True)

    signals = _identify_signals(df, dn_edges)
    print(f"  {len(signals):,} signal candidates identified", flush=True)

    # Pre-build session bars index — O(N) once, then O(1) per signal
    session_idx = _build_session_bars_index(df)

    # Generate trades
    trades = []
    n_total = len(signals)
    for i, sig in enumerate(signals.itertuples()):
        if i % 25000 == 0 and i > 0:
            print(f"    processed {i:,} / {n_total:,}", flush=True)
        sym = str(sig.symbol)
        d_val = sig.d
        hhmm = sig.hhmm
        # Get pre-indexed bars for this (symbol, date)
        session_bars = session_idx.get((sym, d_val))
        if not session_bars:
            continue
        # Find the bar AFTER signal bar (Mode B entry)
        entry_hhmm = _hhmm_add(hhmm, 5)
        entry_bar = next(((h, o, hi, lo, cl) for (h, o, hi, lo, cl) in session_bars if h == entry_hhmm), None)
        if entry_bar is None:
            continue
        entry_price = entry_bar[1]
        if entry_price <= 0:
            continue
        signal_low = float(sig.low)
        sl_from_low = signal_low * (1.0 - SL_PCT_FROM_LOW)
        sl_from_entry = entry_price * (1.0 - SL_PCT_FROM_ENTRY)
        hard_sl = min(sl_from_low, sl_from_entry)
        if hard_sl >= entry_price:
            continue
        R = entry_price - hard_sl
        t1 = entry_price + T1_R_MULT * R
        t2 = entry_price + T2_R_MULT * R

        exit_price, exit_reason, exit_hhmm, mfe_r, mae_r = _walk_to_exit_indexed(
            session_bars, entry_hhmm,
            entry_price, hard_sl, t1, t2, time_stop_hhmm_int,
        )
        if exit_reason in ("no_data", "invalid_R"):
            continue

        cap_seg = _get_cap_segment(sym) or "unknown"
        if cap_seg not in ALLOWED_CAP_SEGMENTS and cap_seg != "unknown":
            continue

        if R <= 0:
            continue
        qty = max(1, int(RISK_PER_TRADE_RUPEES / R))
        gross = (exit_price - entry_price) * qty
        fee = calc_fee(entry_price, exit_price, qty, "BUY")
        net_pnl = gross - fee
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0

        trades.append({
            "signal_date": d_val,
            "symbol": f"NSE:{sym}",
            "side": "LONG",
            "entry_price": entry_price,
            "exit_price": exit_price,
            "qty": qty,
            "pnl_pct": pnl_pct,
            "exit_reason": exit_reason,
            "same_bar": exit_reason == "same_bar_sl",
            "cap_segment": cap_seg,
            "signal_ts": f"{d_val}T{hhmm}:00",
            "entry_ts": f"{d_val}T{entry_hhmm}:00",
            "exit_ts": f"{d_val}T{exit_hhmm}:00",
            "realized_pnl_inr": float(gross),
            "fee_inr": float(fee),
            "net_pnl_inr": float(net_pnl),
            "r_multiple": (pnl_pct / 100.0 * entry_price) / R if R > 0 else 0.0,
            "t1_target": float(t1),
            "t2_target": float(t2),
            "hard_sl": float(hard_sl),
            "t1_partial_booked": False,
            "mfe_r": float(mfe_r),
            "mae_r": float(mae_r),
            "R_per_share": float(R),
            "vol_ratio": float(sig.vol_ratio),
            "hhmm_bucket": _hhmm_to_bucket(hhmm),
            "vol_ratio_bin": _vol_ratio_bin(sig.vol_ratio),
            "bar_return_bin": _bar_return_bin(sig.bar_return_pct, dn_edges),
            "close_at_1430": _close_at_indexed(session_bars, "14:30"),
            "close_at_1500": _close_at_indexed(session_bars, "15:00"),
        })

    trades_df = pd.DataFrame(trades)
    print(f"  Generated {len(trades_df):,} trades", flush=True)
    if not trades_df.empty:
        n_win = (trades_df["pnl_pct"] > 0).sum()
        n_loss = (trades_df["pnl_pct"] < 0).sum()
        total_net = trades_df["net_pnl_inr"].sum()
        print(f"  Trades: {n_win} winners, {n_loss} losers, NET Rs {total_net:+,.0f}", flush=True)
    return trades_df


def _hhmm_to_bucket(hhmm: str) -> str:
    h, m = hhmm.split(":")
    minutes = int(h) * 60 + int(m)
    if minutes < 11 * 60:
        return "morning_0930_1100"
    elif minutes < 13 * 60:
        return "midday_1100_1300"
    else:
        return "afternoon_1300_1500"


def _vol_ratio_bin(vr: float) -> str:
    if vr < 7.0:
        return "5_to_7"
    elif vr < 10.0:
        return "7_to_10"
    elif vr < 15.0:
        return "10_to_15"
    else:
        return "gte_15"


def _bar_return_bin(br: float, dn_edges: np.ndarray) -> str:
    """Bin negative bar_return into DN_Q1, DN_Q2, DN_Q3."""
    # dn_edges are negative, ordered from most-negative to least-negative
    if br <= dn_edges[1]:
        return "DN_Q5"  # most extreme down (filtered out)
    elif br <= dn_edges[2]:
        return "DN_Q4"
    elif br <= dn_edges[3]:
        return "DN_Q3"
    elif br <= dn_edges[4]:
        return "DN_Q2"
    else:
        return "DN_Q1"  # mildest down


def _get_close_at(df: pd.DataFrame, sym: str, d_val: date, hhmm: str) -> float:
    row = df[(df["symbol"] == sym) & (df["d"] == d_val) & (df["hhmm"] == hhmm)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0]["close"])


def main():
    out_dir = _REPO_ROOT / "reports" / "sub9_sanity"
    out_dir.mkdir(parents=True, exist_ok=True)

    for window_label in ("discovery", "oos", "holdout"):
        trades_df = run_window(window_label, time_stop_hhmm_int=1430)
        out_path = out_dir / f"_extreme_vol_revert_long_trades_{window_label}.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"  Saved: {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
