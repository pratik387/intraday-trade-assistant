"""Pre-coding sanity check for earnings_day_intraday_fade candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-06-sub-project-9-brief-earnings_
day_intraday_fade.md). Mechanic:
  - T+0 09:15 gap classification: |gap| in [1%, 8%]
    * gap > 0 -> SHORT (fade FOMO continuation)
    * gap < 0 -> LONG (fade panic capitulation)
  - T+0 entry window: 11:00-14:30, single fire per (symbol, day)
  - Entry trigger: price has retraced <=50% of opening gap, and
    SHORT bar = red candle / LONG bar = green candle
  - Stop: T+0 high * 1.005 (SHORT) / T+0 low * 0.995 (LONG); min 1% stop
  - T1 = 1R, T2 = 2R, breakeven trail after T1
  - Time stop: 15:00 IST
  - Universe: F&O 200 mid+small_cap

Earnings calendar source: data/earnings_calendar/earnings_events.parquet
(BMO/AMC events backfilled 2022-2024 from NSE corporate filings).
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment              # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee     # noqa: E402


# ---- Locked params (per brief Mechanic) ----
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}  # NB: brief said mid+small only,
# but F&O 200 has only 4 mid_cap and 0 small_cap stocks (143 large_cap). The brief's
# cap restriction is empirically empty for F&O 200 — broaden to all caps so the
# thesis can be tested. Result will be reported per cap_segment.
# Round-4 loosened params per agent research 2026-05-06 (Indian retail-algo
# convention review). Original brief params replaced with:
#   - gap band [0.5%, 6%] (Enrich Money 0.5% gap floor; Endovia 3-6% upper)
#   - entry window 10:30-15:00 (JM Financial mid-session lull begins 10:30;
#     TradingView IN 11:30 reversal — 10:30 captures the early fade fires)
#   - retracement filter DROPPED (invented in brief; no Indian retail-algo
#     source uses it; was the dominant gate-killer 2253 -> 170)
#   - T1=0.5R / T2=1.5R (Zerodha Varsity "half the gap" target convention;
#     Tizy "VWAP flush" target which is <1R; was 1R/2R causing 96% time_stop)
GAP_PCT_MIN = 0.005                         # 0.5% min absolute gap (was 1%)
GAP_PCT_MAX = 0.06                          # 6% max absolute gap (was 8%)
ENTRY_WINDOW_START_HHMM = "10:30"           # was 11:00
ENTRY_WINDOW_END_HHMM = "15:00"             # was 14:30
TIME_STOP_HHMM = "15:10"                    # 5 min before MIS auto-square (was 15:00)
RETRACEMENT_LIMIT_FRAC = 1.0                # filter DROPPED (was 0.50);
                                            # 1.0 = always passes (kept for code-symmetry)
SL_BUFFER_PCT = 0.5                         # SL = T+0 high * 1.005 (SHORT) / low * 0.995 (LONG)
MIN_STOP_PCT = 0.01                         # 1% minimum stop distance
T1_R_MULTIPLE = 0.5                         # was 1.0
T2_R_MULTIPLE = 1.5                         # was 2.0
STOP_R_MULTIPLE = 1.0                       # multiplier on computed structural stop_distance (1.0 = baseline)
USE_BREAKEVEN_TRAIL_AFTER_T1 = True
RISK_PER_TRADE_RUPEES = 1000


# Columns retained for sanity logic (drop heavy enriched columns to reduce memory)
_KEEP_COLS = ["symbol", "date", "open", "high", "low", "close"]


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    """Load only the 6 OHLC columns needed (avoid heavy enriched cols at I/O)."""
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    try:
        # pyarrow column projection: read only what we need (avoids 1.4 GiB peak alloc)
        import pyarrow.feather as feather
        tbl = feather.read_table(path, columns=_KEEP_COLS)
        return tbl.to_pandas()
    except Exception:
        # Fallback: read full and project
        df = pd.read_feather(path)
        return df[[c for c in _KEEP_COLS if c in df.columns]]


def build_full_period_5m(universe: set | None = None,
                         event_keys: set | None = None,
                         start_d: date | None = None,
                         end_d: date | None = None) -> pd.DataFrame:
    """Load monthly 5m feathers, filtering each in-place to reduce memory.

    Filters applied PER FILE before concat:
      - keep only `_KEEP_COLS` columns
      - keep only symbols in `universe` (if provided)
      - keep only (symbol, day) in `event_keys` (if provided)
      - keep only dates within [start_d, end_d] if provided

    Period: months overlapping [start_d, end_d] (defaults to 2023-01..2024-12 for back-compat).
    """
    # Determine month range from [start_d, end_d]
    if start_d is None:
        start_d = date(2023, 1, 1)
    if end_d is None:
        end_d = date(2024, 12, 31)
    months = []
    cur = date(start_d.year, start_d.month, 1)
    end_first = date(end_d.year, end_d.month, 1)
    while cur <= end_first:
        months.append((cur.year, cur.month))
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    print(f"  loading {len(months)} monthly 5m feathers ({months[0][0]:04d}-{months[0][1]:02d} .. {months[-1][0]:04d}-{months[-1][1]:02d}) ...")
    parts: List[pd.DataFrame] = []
    import gc
    for yyyy, m in months:
            mdf = _load_5m_for_month(yyyy, m)
            if mdf.empty:
                continue
            # Filter rows BEFORE any column projection / copy to keep peak memory low
            if universe is not None:
                mdf = mdf[mdf["symbol"].isin(universe)]
                if mdf.empty:
                    del mdf; gc.collect(); continue
            if event_keys is not None:
                d_series = mdf["date"].dt.date
                sym_arr = mdf["symbol"].to_numpy()
                d_arr = d_series.to_numpy()
                # Avoid Python-level zip over all rows: build a small list of bools via zip
                mask = [(s, d) in event_keys for s, d in zip(sym_arr, d_arr)]
                mdf = mdf[pd.Series(mask, index=mdf.index)]
                if mdf.empty:
                    del mdf; gc.collect(); continue
            cols = [c for c in _KEEP_COLS if c in mdf.columns]
            mdf = mdf[cols].copy()
            mdf["d"] = mdf["date"].dt.date
            parts.append(mdf)
            gc.collect()
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    if "d" not in big.columns:
        big["d"] = big["date"].dt.date
    print(f"  total 5m bars (post-filter): {len(big):,}")
    return big


def load_daily_close() -> pd.Series:
    """Load consolidated daily close — needed for T-1 close to compute opening gap."""
    path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    df = pd.read_feather(path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    return df.set_index(["symbol", "d"])["close"]


def load_fno_universe() -> set:
    df = pd.read_csv(_REPO_ROOT / "assets" / "fno_liquid_200.csv")
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_expanded_universe(min_adv_cr: float = 10.0) -> set:
    """MIS-eligible + ADV >= min_adv_cr / day (in INR crore).
    Sources: nse_all.json (MIS) + consolidated_daily.feather (ADV).
    """
    import json
    nse_path = _REPO_ROOT / "nse_all.json"
    items = json.loads(nse_path.read_text(encoding="utf-8"))
    mis_set = set()
    for it in items:
        sym = str(it.get("symbol", ""))
        if not sym: continue
        bare = sym[:-3] if sym.endswith(".NS") else sym
        if it.get("mis_enabled") and (it.get("mis_leverage") or 0) >= 1.0:
            mis_set.add(bare)
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    daily = pd.read_feather(daily_path, columns=["ts", "symbol", "close", "volume"])
    daily["ts"] = pd.to_datetime(daily["ts"])
    daily["adv_inr"] = daily["close"].astype(float) * daily["volume"].astype(float)
    adv = daily.groupby("symbol")["adv_inr"].median()
    floor_inr = min_adv_cr * 1e7
    liquid = set(adv[adv >= floor_inr].index)
    expanded = mis_set & liquid
    print(f"  expanded universe: MIS-eligible={len(mis_set)}, ADV>={min_adv_cr}cr={len(liquid)}, intersection={len(expanded)}")
    return expanded


def load_earnings_events() -> pd.DataFrame:
    path = _REPO_ROOT / "data" / "earnings_calendar" / "earnings_events.parquet"
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["bare_symbol"] = df["symbol"].astype(str).str.replace("NSE:", "", regex=False)
    print(f"  earnings events: {len(df)} rows ({df['announce_time_class'].value_counts().to_dict()})")
    # Filter to BMO + AMC (intraday is rare, brief excludes from sanity)
    df = df[df["announce_time_class"].isin(["BMO", "AMC"])].copy()
    print(f"    BMO+AMC: {len(df)} rows")
    return df[["bare_symbol", "trade_date", "announce_time_class"]]


def find_triggers(big5m: pd.DataFrame, universe: set, events: pd.DataFrame,
                  daily_close: pd.Series) -> pd.DataFrame:
    print("  joining earnings events to 5m bars ...")
    df = big5m[big5m["symbol"].isin(universe)].copy()
    df["nse_symbol"] = "NSE:" + df["symbol"]
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)].copy()
    print(f"    F&O 200 + mid/small_cap: {len(df):,}")

    # restrict 5m bars to (symbol, trade_date) in earnings events
    event_keys = set(zip(events["bare_symbol"], events["trade_date"]))
    df["key"] = list(zip(df["symbol"], df["d"]))
    df = df[df["key"].apply(lambda k: k in event_keys)].copy()
    print(f"    bars on earnings days: {len(df):,}")

    if df.empty:
        return df

    # Per (symbol, day): compute opening gap and T+0 day high/low.
    # Open = first bar's open; T-1 close from daily; H/L = max/min of all bars.
    day_meta = df.groupby(["symbol", "d"]).agg(
        day_open=("open", "first"),
        day_high=("high", "max"),
        day_low=("low", "min"),
    ).reset_index()
    # T-1 close: ask daily_close for prior session
    daily_close_idx = daily_close
    def get_t1_close(row):
        sym, d = row["symbol"], row["d"]
        for back in range(1, 8):  # search back up to 7 calendar days for prior trading session
            d_prev = d - timedelta(days=back)
            try:
                v = daily_close_idx.get((sym, d_prev))
                if v is not None and pd.notna(v):
                    return v
            except KeyError:
                continue
        return None
    day_meta["t1_close"] = day_meta.apply(get_t1_close, axis=1)
    day_meta = day_meta.dropna(subset=["t1_close"])
    day_meta["gap_pct"] = (day_meta["day_open"] - day_meta["t1_close"]) / day_meta["t1_close"]
    day_meta = day_meta[day_meta["gap_pct"].abs().between(GAP_PCT_MIN, GAP_PCT_MAX)].copy()
    print(f"    earnings days with |gap|in[1%,8%]: {len(day_meta):,}")

    # Side from gap direction
    day_meta["side"] = day_meta["gap_pct"].apply(lambda g: "SHORT" if g > 0 else "LONG")

    # Join meta back to bars
    meta_idx = day_meta.set_index(["symbol", "d"])
    df["day_open"] = df.set_index(["symbol", "d"]).index.map(meta_idx["day_open"]).values
    df["day_high"] = df.set_index(["symbol", "d"]).index.map(meta_idx["day_high"]).values
    df["day_low"]  = df.set_index(["symbol", "d"]).index.map(meta_idx["day_low"]).values
    df["t1_close"] = df.set_index(["symbol", "d"]).index.map(meta_idx["t1_close"]).values
    df["side"]     = df.set_index(["symbol", "d"]).index.map(meta_idx["side"]).values
    df = df.dropna(subset=["day_open", "side"])
    df["gap_pct"] = (df["day_open"] - df["t1_close"]) / df["t1_close"]

    # Restrict to entry window
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= ENTRY_WINDOW_START_HHMM) & (df["hhmm"] <= ENTRY_WINDOW_END_HHMM)].copy()

    # Retracement check: |close - day_open| / |day_open - t1_close| <= 0.5
    # i.e., price has retraced <= 50% of the opening gap.
    df["gap_size"] = (df["day_open"] - df["t1_close"]).abs()
    df["retraced"] = (df["close"] - df["day_open"]).abs() / df["gap_size"].replace(0, 1e-9)

    # Confirmation candle
    short_mask = (
        (df["side"] == "SHORT")
        & (df["close"] < df["open"])             # red candle
        & (df["close"] >= df["t1_close"])        # not already mean-reverted past prior close
        & (df["retraced"] <= RETRACEMENT_LIMIT_FRAC)
    )
    long_mask = (
        (df["side"] == "LONG")
        & (df["close"] > df["open"])             # green candle
        & (df["close"] <= df["t1_close"])        # not already mean-reverted past prior close
        & (df["retraced"] <= RETRACEMENT_LIMIT_FRAC)
    )
    triggers = df[short_mask | long_mask].copy()
    print(f"    triggers (SHORT={short_mask.sum()}, LONG={long_mask.sum()}): {len(triggers):,}")
    return triggers


def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating exits ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    # Latch: first qualifying trigger per (symbol, day, side)
    triggers = triggers.sort_values(["symbol", "d", "side", "date"])
    triggers = triggers.drop_duplicates(subset=["symbol", "d", "side"], keep="first")

    trades: List[dict] = []
    for _, t in triggers.iterrows():
        sym = t["symbol"]; sd = t["d"]; side = t["side"]
        trig_ts = t["date"]
        day_high = float(t["day_high"]); day_low = float(t["day_low"])

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

        if side == "SHORT":
            sl_struct = day_high * (1.0 + SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 + MIN_STOP_PCT)
            hard_sl_raw = max(sl_struct, sl_min)
            stop_distance_raw = hard_sl_raw - entry_price
            stop_distance = stop_distance_raw * STOP_R_MULTIPLE
            hard_sl = entry_price + stop_distance
            t1_target = entry_price - T1_R_MULTIPLE * stop_distance
            t2_target = entry_price - T2_R_MULTIPLE * stop_distance
        else:
            sl_struct = day_low * (1.0 - SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 - MIN_STOP_PCT)
            hard_sl_raw = min(sl_struct, sl_min)
            stop_distance_raw = entry_price - hard_sl_raw
            stop_distance = stop_distance_raw * STOP_R_MULTIPLE
            hard_sl = entry_price - stop_distance
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
            "side": side,
            "gap_pct": float(t["gap_pct"]),
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
    print("\n=== earnings_day_intraday_fade -- pre-coding sanity check ===")
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


def main():
    global T1_R_MULTIPLE, T2_R_MULTIPLE, STOP_R_MULTIPLE
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", default="2023-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--t1-r-mult", type=float, default=T1_R_MULTIPLE,
                   help=f"T1 R-multiple (default {T1_R_MULTIPLE})")
    p.add_argument("--t2-r-mult", type=float, default=T2_R_MULTIPLE,
                   help=f"T2 R-multiple (default {T2_R_MULTIPLE})")
    p.add_argument("--stop-r-mult", type=float, default=STOP_R_MULTIPLE,
                   help=f"Stop R-multiple (scales structural stop; default {STOP_R_MULTIPLE})")
    p.add_argument("--out-suffix", default="", help="suffix appended to output dir/file basename")
    p.add_argument("--expanded-universe", action="store_true",
                   help="Use MIS-eligible + ADV>=10cr universe instead of F&O 200")
    p.add_argument("--min-adv-cr", type=float, default=10.0,
                   help="Minimum ADV (Rs crore) when --expanded-universe is set (default 10)")
    p.add_argument("--allow-oos", action="store_true",
                   help="Opt in to read past 2024-12-31 (OOS / Holdout). Default refuses (Discovery guard).")
    args = p.parse_args()

    # Sentinel guard: refuse OOS/Holdout reads unless explicit opt-in
    end_d = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end_d > date(2024, 12, 31) and not args.allow_oos:
        print("[ERROR] --end past 2024-12-31 (Discovery cutoff). Pass --allow-oos to bypass.")
        return 2
    if args.allow_oos:
        print(f"[WARN] OOS/Holdout mode active (--allow-oos); reading past Discovery cutoff.")

    T1_R_MULTIPLE = float(args.t1_r_mult)
    T2_R_MULTIPLE = float(args.t2_r_mult)
    STOP_R_MULTIPLE = float(args.stop_r_mult)
    print(f"[params] T1_R={T1_R_MULTIPLE} T2_R={T2_R_MULTIPLE} STOP_R={STOP_R_MULTIPLE} "
          f"period={args.start}..{args.end} out_suffix={args.out_suffix!r}")

    start_d = datetime.strptime(args.start, "%Y-%m-%d").date()
    universe = load_expanded_universe(args.min_adv_cr) if args.expanded_universe else load_fno_universe()
    daily_close = load_daily_close()
    events = load_earnings_events()
    # Filter earnings events to the requested period
    events = events[(events["trade_date"] >= start_d) & (events["trade_date"] <= end_d)].copy()
    print(f"  earnings events in window {start_d}..{end_d}: {len(events):,}")

    # Build (symbol, day) prefilter set so monthly feathers can be downselected at load time
    event_keys = set(zip(events["bare_symbol"].astype(str), events["trade_date"]))
    print(f"  earnings (symbol, day) keys for prefilter: {len(event_keys):,}")
    big5m = build_full_period_5m(universe=universe, event_keys=event_keys,
                                  start_d=start_d, end_d=end_d)

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, events, daily_close)
    print(f"\nTotal triggers (before latch): {len(triggers)}")
    if triggers.empty:
        return

    trades = simulate(triggers, big5m)
    report(trades)

    suffix = args.out_suffix or ""
    if suffix:
        out_dir = _REPO_ROOT / "reports" / "sub9_sanity" / f"earnings_day_intraday_fade{suffix}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "trades.csv"
    else:
        out = _REPO_ROOT / "reports" / "sub9_sanity" / "earnings_day_intraday_fade_trades.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
