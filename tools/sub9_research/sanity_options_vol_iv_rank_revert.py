"""Pre-coding sanity check for options_vol_iv_rank_revert candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-06-sub-project-9-brief-options_
vol_iv_rank_revert.md): BEFORE writing detector code, simulate the IV-rank
extreme → T+0 underlying mean-reversion fade on 2 years of data.

Mechanic (per locked brief params):
  - T-1 EOD IV-rank lookup (data/options_iv/2022_2024_iv_rank.parquet)
  - SHORT: iv_rank >= 0.80 (high-IV crush thesis)
  - LONG: iv_rank <= 0.20 AND NIFTY 5m close > NIFTY 5m EMA(20) at 11:00
  - Entry at 11:00 IST 5m bar with confirmation candle
  - 1% hard stop, T1 = 1R, T2 = 2R, breakeven trail after T1
  - Time stop: 15:10 IST
  - Latch per (symbol, day, side)

Decision criterion (from brief):
  PF >= 1.10  -> strong proceed; 1.0-1.10 marginal; PF < 1.0 retire
LONG-side ship gate: LONG-side PF >= SHORT-side PF * 0.85.
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment              # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee     # noqa: E402


# ---- Locked params (per brief Mechanic) ----
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}  # F&O 200 universe; brief §Universe
IV_RANK_HIGH = 0.80                       # iv_rank >= 0.80 -> SHORT
IV_RANK_LOW = 0.20                        # iv_rank <= 0.20 -> LONG
ENTRY_HHMM = "11:00"                      # Entry at 11:00 5m bar
TIME_STOP_HHMM = "15:10"                  # 5 min before MIS auto-square
STOP_PCT = 0.01                           # 1% hard stop
T1_R_MULTIPLE = 1.0
T2_R_MULTIPLE = 2.0
USE_BREAKEVEN_TRAIL_AFTER_T1 = True       # Q8 design decision (round-3)
NIFTY_TREND_EMA_PERIOD = 20               # NIFTY 50 EMA(20) for LONG-side trend gate
RISK_PER_TRADE_RUPEES = 1000


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_full_period_5m() -> pd.DataFrame:
    print("  loading 24 monthly 5m feathers (2023-01 .. 2024-12) ...")
    parts: List[pd.DataFrame] = []
    for yyyy in (2023, 2024):
        for m in range(1, 13):
            mdf = _load_5m_for_month(yyyy, m)
            if not mdf.empty:
                parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_fno_universe() -> set:
    df = pd.read_csv(_REPO_ROOT / "assets" / "fno_liquid_200.csv")
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_iv_rank_lookup() -> pd.Series:
    path = _REPO_ROOT / "data" / "options_iv" / "2022_2024_iv_rank.parquet"
    df = pd.read_parquet(path)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df = df.dropna(subset=["iv_rank"])
    print(f"  iv_rank rows: {len(df):,} for {df['symbol'].nunique()} symbols")
    return df.set_index(["symbol", "session_date"])["iv_rank"]


def load_nifty_trend_at_1100() -> Dict[date, bool]:
    """Per session_date: True if NIFTY 50 5m close at 11:00 > 5m EMA(20) at 11:00."""
    path = _REPO_ROOT / "backtest-cache-download" / "index_ohlcv" / "NSE_NIFTY_50" / "NSE_NIFTY_50_1minutes.feather"
    df = pd.read_feather(path)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df = df[(df["date"] >= pd.Timestamp("2023-01-01"))
            & (df["date"] < pd.Timestamp("2025-01-01"))].set_index("date")
    agg = df.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    agg["ema20"] = agg["close"].ewm(span=NIFTY_TREND_EMA_PERIOD, adjust=False).mean()
    agg = agg.reset_index()
    agg["d"] = agg["date"].dt.date
    agg["hhmm"] = agg["date"].dt.strftime("%H:%M")
    at1100 = agg[agg["hhmm"] == ENTRY_HHMM].copy()
    at1100["trend_up"] = at1100["close"] > at1100["ema20"]
    print(f"  NIFTY trend gate computed for {len(at1100)} sessions")
    return dict(zip(at1100["d"], at1100["trend_up"]))


def find_triggers(big5m: pd.DataFrame, universe: set,
                  iv_lookup: pd.Series, nifty_trend: Dict[date, bool]) -> pd.DataFrame:
    print("  filtering 5m bars to F&O 200 + 11:00 entry bar ...")
    df = big5m[big5m["symbol"].isin(universe)].copy()
    df["nse_symbol"] = "NSE:" + df["symbol"].astype(str)
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)].copy()
    df["hhmm"] = df["date"].dt.strftime("%H:%M")

    entry_bars = df[df["hhmm"] == ENTRY_HHMM].copy()
    print(f"    11:00 bars (F&O 200, mid/small/large_cap): {len(entry_bars):,}")

    # T-1 IV rank: previous calendar day's iv_rank (iv_rank computed at EOD T-1)
    # Use the iv_rank with session_date = T-1 (not T+0). For sanity, look up
    # iv_rank on the trade day's prior session_date in the IV series.
    # Approach: iv_lookup is indexed (symbol, session_date). For trade day D,
    # find the most recent session_date < D in the IV series for that symbol.
    iv_df = iv_lookup.reset_index()
    iv_df = iv_df.sort_values(["symbol", "session_date"])
    # asof-merge: for each (symbol, d) entry bar, get iv_rank from session_date <= (d - 1 day)
    entry_bars["lookup_date"] = pd.to_datetime(entry_bars["d"])
    iv_df["lookup_date"] = pd.to_datetime(iv_df["session_date"])
    # Per-symbol asof merge
    merged_parts = []
    for sym, ebg in entry_bars.groupby("symbol"):
        iv_g = iv_df[iv_df["symbol"] == sym]
        if iv_g.empty:
            continue
        ebg = ebg.sort_values("lookup_date")
        iv_g = iv_g.sort_values("lookup_date")
        m = pd.merge_asof(
            ebg, iv_g[["lookup_date", "iv_rank"]],
            on="lookup_date", direction="backward", allow_exact_matches=False,
        )
        merged_parts.append(m)
    if not merged_parts:
        print("    no IV-rank coverage on F&O universe!")
        return pd.DataFrame()
    merged = pd.concat(merged_parts, ignore_index=True)
    merged = merged.dropna(subset=["iv_rank"])
    print(f"    11:00 bars with T-1 iv_rank coverage: {len(merged):,}")

    merged["nifty_trend_up"] = merged["d"].map(nifty_trend).fillna(False)

    # SHORT: iv_rank >= 0.80, with confirmation red+below VWAP
    short = merged[
        (merged["iv_rank"] >= IV_RANK_HIGH)
        & (merged["close"] < merged["open"])
        & (merged["close"] < merged["vwap"])
    ].copy()
    short["side"] = "SHORT"

    # LONG: iv_rank <= 0.20 + NIFTY trend up + green+above VWAP
    long_ = merged[
        (merged["iv_rank"] <= IV_RANK_LOW)
        & (merged["nifty_trend_up"])
        & (merged["close"] > merged["open"])
        & (merged["close"] > merged["vwap"])
    ].copy()
    long_["side"] = "LONG"

    print(f"    SHORT triggers (iv_rank>=0.80 + red + below VWAP): {len(short):,}")
    print(f"    LONG triggers (iv_rank<=0.20 + NIFTY uptrend + green + above VWAP): {len(long_):,}")
    return pd.concat([short, long_], ignore_index=True)


def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating exits ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    # latch
    triggers = triggers.sort_values(["symbol", "d", "side", "date"])
    triggers = triggers.drop_duplicates(subset=["symbol", "d", "side"], keep="first")

    trades: List[dict] = []
    for _, t in triggers.iterrows():
        sym = t["symbol"]; sd = t["d"]; side = t["side"]
        trig_ts = t["date"]; trig_close = float(t["close"])

        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            continue

        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr or idx_arr[0] + 1 >= len(day_df):
            continue
        trig_idx = idx_arr[0]

        # Entry at NEXT bar's open (Streak/AlgoTest convention, Q1 design decision)
        entry_bar = day_df.iloc[trig_idx + 1]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        if side == "SHORT":
            hard_sl = entry_price * (1.0 + STOP_PCT)
            stop_distance = hard_sl - entry_price
            t1_target = entry_price - T1_R_MULTIPLE * stop_distance
            t2_target = entry_price - T2_R_MULTIPLE * stop_distance
        else:
            hard_sl = entry_price * (1.0 - STOP_PCT)
            stop_distance = entry_price - hard_sl
            t1_target = entry_price + T1_R_MULTIPLE * stop_distance
            t2_target = entry_price + T2_R_MULTIPLE * stop_distance
        if stop_distance <= 0:
            continue

        forward = day_df.iloc[trig_idx + 1:].copy()
        forward["hhmm"] = forward["date"].dt.strftime("%H:%M")

        exit_ts = None; exit_price = None; exit_reason = None
        hit_t1 = False; t1_exit_price = None; t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]; high = float(bar["high"]); low = float(bar["low"]); close = float(bar["close"])
            active_sl = (entry_price if (hit_t1 and USE_BREAKEVEN_TRAIL_AFTER_T1) else hard_sl)
            if side == "SHORT":
                if high >= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and low <= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target; t1_exit_ts = ts
                if hit_t1 and low <= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                    break
            else:
                if low <= active_sl:
                    exit_ts = ts; exit_price = active_sl
                    exit_reason = "breakeven_trail" if hit_t1 else "stop"
                    break
                if not hit_t1 and high >= t1_target:
                    hit_t1 = True; t1_exit_price = t1_target; t1_exit_ts = ts
                if hit_t1 and high >= t2_target:
                    exit_ts = ts; exit_price = t2_target; exit_reason = "t2"
                    break
            if bar["hhmm"] >= TIME_STOP_HHMM:
                exit_ts = ts; exit_price = close; exit_reason = "time_stop"
                break

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
            "iv_rank": float(t["iv_rank"]),
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
    print("\n=== options_vol_iv_rank_revert -- pre-coding sanity check ===")
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
        print(f"PF={pf} >= 1.10 -> STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> marginal.")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE.")

    if short_pf is not None and long_pf is not None and short_pf > 0:
        print(f"\nLong/Short PF ratio: {round(long_pf/short_pf, 3)} (gate: >= 0.85 for bidirectional ship)")


def main():
    big5m = build_full_period_5m()
    universe = load_fno_universe()
    iv_lookup = load_iv_rank_lookup()
    nifty_trend = load_nifty_trend_at_1100()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, iv_lookup, nifty_trend)
    print(f"\nTotal triggers (before latch): {len(triggers)}")
    if triggers.empty:
        return

    trades = simulate(triggers, big5m)
    report(trades)
    out = _REPO_ROOT / "reports" / "sub9_sanity" / "options_vol_iv_rank_revert_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
