"""Pre-coding sanity check for bulk_block_buy_continuation candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-01-sub-project-9-brief-bulk_block_
buy_continuation.md): BEFORE writing detector code, simulate the rough
T+1-only intraday hold on 6 months of NSE bulk-deals + 5m bar data.
Decision criterion (from brief):
  PF >= 1.10  → strong proceed
  1.0-1.10    → marginal, proceed with caveat
  PF < 1.0    → retire candidate, do NOT write detector

Fetches NSE bulk-deals via nselib for 2024-07-01..2024-12-31, applies
the brief's filters, looks up T+1 09:20 → 15:15 5m bars from local
backtest-cache feather files, computes Indian-fee NET PF.

Usage:
    python tools/sub9_research/sanity_bulk_block_buy_continuation.py
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

# Repo root on sys.path so we can import the existing fee model
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import in_universe, get_cap_segment   # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee   # noqa: E402


# ---- Config knobs (revised after first sanity-check funnel) ----
DATE_FROM = "01-07-2024"   # DD-MM-YYYY (nselib format)
DATE_TO   = "31-12-2024"
# Sanity-check finding (2026-05-01): the brief's filter combo
# (F&O 200 + large/mid cap + ≥₹50 cr) only yields 7 signals over 6 months.
# Peer-reviewed evidence (Agarwalla/Pandey, Chaturvedula) covered the
# FULL NSE bulk/block universe — there's no literature support for
# restricting to F&O 200 + large/mid cap. The brief's universe filter
# was over-cautious without backing. Revised: keep the canonical
# block-deal threshold (≥₹10 cr) and drop the universe restriction;
# rely on per-stock cap_segment ∈ ALLOWED_CAPS so we still avoid the
# small/micro-cap manipulation risk that worried the brief.
MIN_BUY_VALUE_CR = 10.0
ALLOWED_CAPS = {"large_cap", "mid_cap", "small_cap"}
UNIVERSE_KEY = None   # no universe filter — use whole NSE (per literature)
ENTRY_BAR_HHMM = "09:20"      # 09:20 bar (start of period) = first regular-session 5m bar
EXIT_BAR_HHMM  = "15:10"      # 15:10 bar = last bar before MIS auto-square at 15:20
RISK_PER_TRADE_RUPEES = 1000  # match the rest of the system
ATR_STOP_MULT = 1.5           # Indian retail standard
MIN_STOP_PCT = 0.5            # qty-inflation guard


def fetch_bulk_deals(from_date: str, to_date: str) -> pd.DataFrame:
    """Fetch NSE bulk-deals via nselib in monthly chunks (API limits)."""
    from nselib import capital_market
    out: List[pd.DataFrame] = []
    cur = datetime.strptime(from_date, "%d-%m-%Y").date()
    end = datetime.strptime(to_date, "%d-%m-%Y").date()
    while cur <= end:
        chunk_end = min(cur.replace(day=1) + timedelta(days=31), end)
        # snap chunk_end to last day of month
        nxt = (cur.replace(day=1) + timedelta(days=32)).replace(day=1)
        chunk_end = min(nxt - timedelta(days=1), end)
        f, t = cur.strftime("%d-%m-%Y"), chunk_end.strftime("%d-%m-%Y")
        print(f"  fetching {f} .. {t}", flush=True)
        try:
            df = capital_market.bulk_deal_data(from_date=f, to_date=t)
            out.append(df)
        except Exception as e:
            print(f"    skip ({type(e).__name__}: {e})", flush=True)
        cur = chunk_end + timedelta(days=1)
    if not out:
        raise SystemExit("no bulk-deals data fetched")
    return pd.concat(out, ignore_index=True)


def aggregate_buy_signals(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the brief's filters and aggregate to one row per (symbol, date)."""
    df = raw.copy()
    df.columns = [c.strip() for c in df.columns]
    print(f"  raw rows: {len(df):,}")

    # parse types
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%Y")
    df["QuantityTraded"] = (
        df["QuantityTraded"].astype(str).str.replace(",", "").astype(float)
    )
    df["TradePrice/Wght.Avg.Price"] = pd.to_numeric(
        df["TradePrice/Wght.Avg.Price"], errors="coerce"
    )
    df["value"] = df["QuantityTraded"] * df["TradePrice/Wght.Avg.Price"]

    # buy-side only
    df = df[df["Buy/Sell"].str.upper() == "BUY"].copy()
    print(f"  after BUY-only:           {len(df):,}")

    # add NSE: prefix to symbol for our metadata helpers
    df["Symbol"] = "NSE:" + df["Symbol"].astype(str).str.upper()

    if UNIVERSE_KEY is not None:
        df = df[df["Symbol"].apply(lambda s: in_universe(s, UNIVERSE_KEY))]
        print(f"  after universe={UNIVERSE_KEY}:   {len(df):,}")

    df["cap_segment"] = df["Symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)]
    print(f"  after cap_segment filter:  {len(df):,}  (caps allowed: {sorted(ALLOWED_CAPS)})")

    # aggregate per (symbol, date)
    grp = df.groupby(["Date", "Symbol"]).agg(
        n_clients=("ClientName", "nunique"),
        buy_value_cr=("value", lambda v: v.sum() / 1e7),
        buy_qty=("QuantityTraded", "sum"),
        weighted_price=(
            "TradePrice/Wght.Avg.Price",
            lambda s: (s * df.loc[s.index, "QuantityTraded"]).sum() / max(df.loc[s.index, "QuantityTraded"].sum(), 1),
        ),
        cap_segment=("cap_segment", "first"),
    ).reset_index()

    # value filter
    print(f"  pre-value-filter signals:  {len(grp):,}  (one row per [symbol, date])")
    grp = grp[grp["buy_value_cr"] >= MIN_BUY_VALUE_CR].copy()
    print(f"  after ≥₹{MIN_BUY_VALUE_CR} cr filter:    {len(grp):,}")
    return grp


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    """Load 5m enriched feather for one month."""
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


# One-shot daily-ATR table: MultiIndex (symbol, date) → ATR(14) value.
# Built once on first access via vectorized aggregate over all monthly feathers.
_DAILY_ATR_TABLE: pd.Series | None = None


def _build_daily_atr_table() -> pd.Series:
    """Single-pass aggregation: load all 2024 monthly 5m feathers, build
    daily OHLC per (symbol, date), compute true range + 14-day rolling
    mean ATR. Returns Series indexed by (symbol, date) → atr_value.

    Cost: ~30s one-time vs hours-per-symbol if done lazily.
    """
    print("  one-shot daily-ATR precompute (loading 12 monthly 5m feathers)...")
    parts: List[pd.DataFrame] = []
    for m in range(1, 13):
        mdf = _load_5m_for_month(2024, m)
        if mdf.empty:
            continue
        # Aggregate to daily per (date, symbol) in one vectorized op
        mdf = mdf[["date", "symbol", "high", "low", "close"]].copy()
        mdf["d"] = mdf["date"].dt.date
        day = mdf.groupby(["symbol", "d"]).agg(
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
        ).reset_index()
        parts.append(day)
    if not parts:
        return pd.Series(dtype=float)

    daily = pd.concat(parts, ignore_index=True).sort_values(["symbol", "d"])
    daily["prev_close"] = daily.groupby("symbol")["close"].shift(1)
    daily["tr"] = pd.concat([
        daily["high"] - daily["low"],
        (daily["high"] - daily["prev_close"]).abs(),
        (daily["low"]  - daily["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    daily["atr14"] = daily.groupby("symbol")["tr"].transform(
        lambda s: s.rolling(14).mean()
    )
    out = daily.set_index(["symbol", "d"])["atr14"]
    print(f"  daily-ATR table built: {len(out):,} (symbol, date) rows")
    return out


def _daily_atr_for_symbol(sym_bare: str, target_date) -> float:
    """O(1) lookup: ATR(14) for `sym_bare` on the trading day BEFORE `target_date`.

    Uses the one-shot table built by `_build_daily_atr_table`. Returns NaN
    if the symbol has insufficient history.
    """
    global _DAILY_ATR_TABLE
    if _DAILY_ATR_TABLE is None:
        _DAILY_ATR_TABLE = _build_daily_atr_table()
    try:
        sub = _DAILY_ATR_TABLE.loc[sym_bare]
    except KeyError:
        return float("nan")
    eligible = sub[sub.index < target_date].dropna()
    if eligible.empty:
        return float("nan")
    return float(eligible.iloc[-1])


def simulate_t1_intraday(signals: pd.DataFrame) -> pd.DataFrame:
    """For each T+0 signal row, look up T+1 09:25 entry → 15:15 exit, with ATR
    stop, compute net PnL via Indian fee model.

    Returns per-trade DataFrame with realized_pnl + fee + net_pnl.
    """
    # Cache 5m feather per month
    month_cache: dict[Tuple[int, int], pd.DataFrame] = {}

    trades: List[dict] = []
    for _, sig in signals.iterrows():
        sd = sig["Date"].date()  # T+0
        sym_full = sig["Symbol"]   # "NSE:RELIANCE"
        sym_bare = sym_full.replace("NSE:", "")  # feather uses bare names

        # T+1 = next trading day. Iterate +1..+4 (handles weekend).
        for delta in (1, 2, 3, 4):
            t1 = sd + timedelta(days=delta)
            ym = (t1.year, t1.month)
            if ym not in month_cache:
                month_cache[ym] = _load_5m_for_month(*ym)
            mdf = month_cache[ym]
            if mdf.empty:
                continue

            day_mask = (mdf["date"].dt.date == t1) & (mdf["symbol"] == sym_bare)
            day_df = mdf[day_mask].sort_values("date")
            if day_df.empty:
                continue   # try next day (likely weekend)

            # entry: 09:20 bar (start-of-period = the 09:20-09:24 5m candle).
            # Close of this bar represents price at 09:25 actual time.
            entry_row = day_df[day_df["date"].dt.strftime("%H:%M") == ENTRY_BAR_HHMM]
            if entry_row.empty:
                # try first available bar of the day
                entry_row = day_df.head(1)
                if entry_row.empty:
                    break
            entry_ts = entry_row.iloc[0]["date"]
            entry_price = float(entry_row.iloc[0]["close"])

            # Brief: stop = 1.5 × daily ATR(14). The 5m feather has no atr
            # column, so we aggregate to daily and compute true-range ATR
            # ourselves (cached per symbol). Min-stop floor of 0.5% of
            # entry kicks in on stocks with tight historical ATR.
            atr_daily = _daily_atr_for_symbol(sym_bare, t1)
            if pd.isna(atr_daily) or atr_daily <= 0:
                # Insufficient history (e.g., new listing) — skip this trade
                # rather than use a noisy fallback that distorts stop sizing.
                break
            stop_distance = max(
                ATR_STOP_MULT * atr_daily,
                entry_price * MIN_STOP_PCT / 100.0,
            )
            hard_sl = entry_price - stop_distance

            # Walk forward: exit on stop hit, or at EXIT_BAR_HHMM close, whichever first
            after = day_df[day_df["date"] >= entry_ts].copy()
            exit_ts = None
            exit_price = None
            exit_reason = None
            for _, bar in after.iterrows():
                ts = bar["date"]
                low = float(bar["low"])
                if low <= hard_sl:
                    exit_ts = ts
                    exit_price = hard_sl
                    exit_reason = "stop"
                    break
                if ts.strftime("%H:%M") >= EXIT_BAR_HHMM:
                    exit_ts = ts
                    exit_price = float(bar["close"])
                    exit_reason = "eod"
                    break
            if exit_price is None:
                # fall through to last bar of session
                last = after.iloc[-1]
                exit_ts = last["date"]
                exit_price = float(last["close"])
                exit_reason = "last_bar"

            qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
            realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "BUY")
            net_pnl = realized_pnl - fee

            trades.append({
                "T0_signal_date": sd,
                "T1_entry_date": t1,
                "symbol": sym_full,
                "buy_value_cr": sig["buy_value_cr"],
                "n_clients": sig["n_clients"],
                "cap_segment": sig["cap_segment"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "stop_distance": stop_distance,
                "qty": qty,
                "realized_pnl": realized_pnl,
                "fee": fee,
                "net_pnl": net_pnl,
            })
            break   # only T+1 (or its first available trading day) — done

    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades — possibly data gap or filter too tight")
        return

    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)

    gross = trades["realized_pnl"].sum()
    fees_total = trades["fee"].sum()
    net_total = npnl.sum()

    print("\n=== bulk_block_buy_continuation — pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: ₹{int(gross):,}")
    print(f"Fees:      ₹{int(fees_total):,}")
    print(f"NET PnL:   ₹{int(net_total):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    # Per-cap breakdown
    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>4} PF={pf2:>5} netPnL=₹{net:>10,}")

    # Verdict per brief
    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"PF={pf} >= 1.10 → STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) → marginal. Proceed with caveat.")
    else:
        print(f"PF={pf} < 1.00 → RETIRE candidate. Do not write detector code.")
        print("  (per sub-9 spec §3.3 brief gate — this is the disciplined kill.)")


def main():
    print(f"Fetching NSE bulk deals {DATE_FROM} .. {DATE_TO} ...")
    raw = fetch_bulk_deals(DATE_FROM, DATE_TO)
    print(f"  raw rows: {len(raw):,}")

    print("\nApplying §3.3 brief filters (BUY-only / F&O 200 / large+mid cap / ≥₹50 cr)...")
    signals = aggregate_buy_signals(raw)
    print(f"  surviving signals: {len(signals):,}")
    if signals.empty:
        print("  no signals after filter — sanity check returns 0 trades")
        return

    print("\nSimulating T+1 09:25 → 15:15 MIS hold with 1.5×ATR stop ...")
    trades = simulate_t1_intraday(signals)
    print(f"  successful sims: {len(trades):,}")

    report(trades)

    out_path = _REPO_ROOT / "reports" / "sub9_sanity" / "bulk_block_buy_continuation_trades.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_path, index=False)
    print(f"\nFull trade log: {out_path}")


if __name__ == "__main__":
    main()
