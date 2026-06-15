"""Build the daily mark-to-market portfolio-return series for overlapping
multi-day CNC/MTF positions.

The deep-research recipe (specs/2026-06-15-cnc-confidence-card-methodology-research.md)
requires a SINGLE daily portfolio-return series as the input to the stationary
block bootstrap (PF/expectancy CI) and the Lo (2002) autocorrelation-corrected
Sharpe. Overlapping multi-day holds are precisely what induce the positive
autocorrelation those methods correct for — so the series must mark every open
position to market on every calendar trading day, not bucket a trade's whole P&L
on its entry day.

Position semantics mirror the cell-mine ledger EXACTLY (parity):
  signal at day-t close  ->  ENTRY at T+1 open  ->  EXIT at close of T+1+K.
Indexing is POSITIONAL per symbol on the sorted clean-daily panel (same as the
cell-mine's `g['open'].shift(-1)` / `close.shift(-(1+K))`).
"""
from __future__ import annotations

import pandas as pd

_PANEL_COLS = ("symbol", "date", "open", "close")


def expand_trade_to_daily(
    panel: pd.DataFrame,
    *,
    symbol: str,
    signal_date,
    k_hold: int,
    cost: float = 0.0,
) -> pd.Series:
    """Daily mark-to-market returns for ONE position over its hold window.

    Signal at `signal_date` close -> entry at the next panel row's open -> exit at
    the close `k_hold+1` rows forward (positional, within `symbol`). Returns a
    date-indexed Series of daily returns:
        entry day : close/open - 1            (minus round-trip `cost`)
        later days: close/prev_close - 1

    An empty Series is returned if the panel lacks a full hold window after the
    signal (e.g. the signal sits too close to the end of the data).
    """
    sym = panel[panel["symbol"] == symbol].sort_values("date").reset_index(drop=True)
    sig = pd.Timestamp(signal_date)
    pos = sym.index[sym["date"] == sig]
    if len(pos) == 0:
        return pd.Series(dtype=float)
    i = int(pos[0])
    entry_i = i + 1
    exit_i = i + 1 + k_hold
    if exit_i >= len(sym):
        return pd.Series(dtype=float)

    window = sym.iloc[entry_i:exit_i + 1]
    rets, dates = [], []
    prev_close = None
    for j, (_, row) in enumerate(window.iterrows()):
        if j == 0:
            r = row["close"] / row["open"] - 1.0 - cost
        else:
            r = row["close"] / prev_close - 1.0
        rets.append(r)
        dates.append(row["date"])
        prev_close = row["close"]
    return pd.Series(rets, index=pd.DatetimeIndex(dates))


def _index_panel(panel: pd.DataFrame) -> dict:
    """Pre-index the panel by symbol once (O(N)) so expanding many trades does
    not rescan the full panel per trade."""
    idx: dict = {}
    for sym, sub in panel.sort_values("date").groupby("symbol", sort=False):
        dates = pd.DatetimeIndex(sub["date"].to_numpy())
        idx[sym] = (
            {d: i for i, d in enumerate(dates)},  # date -> positional row
            sub["open"].to_numpy(dtype=float),
            sub["close"].to_numpy(dtype=float),
            dates,
        )
    return idx


def build_daily_portfolio_returns(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    k_hold: int,
    cost: float = 0.0,
    aggregate: str = "mean",
    signal_date_col: str = "signal_date",
    symbol_col: str = "symbol",
) -> pd.Series:
    """Daily portfolio-return series across all overlapping positions.

    Each position is marked to market every day of its hold window (entry day
    close/open; later days close/prev_close). The day's portfolio return is the
    aggregate of every position open that day:

      aggregate='mean' — equal-weight across the basket (models a fixed TOTAL book
        split equally over however many signals fire). Down-weights big-basket
        days.
      aggregate='sum'  — fixed Rs/slot book (each open position is a fixed-size
        slot). Scale-invariant in PF/Sharpe; big-basket days contribute more, as
        the system's slot model (`margin_per_slot_inr`) actually deploys capital.
        Combine with `simulate_slot_admission` for the capacity-capped book.

    Pre-indexes the panel by symbol so it scales to production-size panels.
    Returns a date-indexed Series sorted ascending; empty if no trade has a full
    hold window.
    """
    if aggregate not in ("mean", "sum"):
        raise ValueError(f"aggregate must be 'mean' or 'sum', got {aggregate!r}")
    if len(trades) == 0:
        return pd.Series(dtype=float)

    idx = _index_panel(panel)
    per_day: dict[pd.Timestamp, list] = {}
    for sym, sig in zip(trades[symbol_col].to_numpy(),
                        pd.to_datetime(trades[signal_date_col]).to_numpy()):
        info = idx.get(sym)
        if info is None:
            continue
        pos_map, opens, closes, dates = info
        i = pos_map.get(pd.Timestamp(sig))
        if i is None:
            continue
        entry_i, exit_i = i + 1, i + 1 + k_hold
        if exit_i >= len(dates):
            continue
        prev_close = None
        for j in range(entry_i, exit_i + 1):
            if j == entry_i:
                r = closes[j] / opens[j] - 1.0 - cost
            else:
                r = closes[j] / prev_close - 1.0
            per_day.setdefault(dates[j], []).append(r)
            prev_close = closes[j]

    if not per_day:
        return pd.Series(dtype=float)

    if aggregate == "mean":
        series = pd.Series({dt: sum(v) / len(v) for dt, v in per_day.items()})
    else:
        series = pd.Series({dt: sum(v) for dt, v in per_day.items()})
    return series.sort_index()


def simulate_slot_admission(
    trades: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    k_hold: int,
    max_slots: int,
    score_col: str,
    signal_date_col: str = "signal_date",
    symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Simulate a fixed-slot capacity book: at most `max_slots` concurrent
    positions, fixed Rs/slot. When more signals fire on a day than free slots,
    admit the best (LOWEST `score_col` = deepest loser = strongest signal); reject
    the rest. A position occupies its slot from entry (T+1) through exit (T+1+K),
    freeing it for later signals.

    Returns the admitted subset of `trades` (same columns), to be fed to
    `build_daily_portfolio_returns(..., aggregate='sum')`.
    """
    if len(trades) == 0:
        return trades.copy()

    idx = _index_panel(panel)
    rows = []
    for r in trades.to_dict("records"):
        info = idx.get(r[symbol_col])
        if info is None:
            continue
        pos_map, _opens, _closes, dates = info
        i = pos_map.get(pd.Timestamp(pd.Timestamp(r[signal_date_col])))
        if i is None or i + 1 + k_hold >= len(dates):
            continue
        rows.append({**r, "entry_dt": dates[i + 1], "exit_dt": dates[i + 1 + k_hold]})

    if not rows:
        return trades.iloc[0:0].copy()

    df = pd.DataFrame(rows).sort_values(["entry_dt", score_col]).reset_index(drop=True)
    open_exits: list = []  # exit dates of currently-held positions
    admitted_idx = []
    for pos in df.itertuples(index=True):
        entry = pos.entry_dt
        open_exits = [e for e in open_exits if e >= entry]  # free slots that exited before entry
        if len(open_exits) < max_slots:
            admitted_idx.append(pos.Index)
            open_exits.append(pos.exit_dt)

    return df.loc[admitted_idx].drop(columns=["entry_dt", "exit_dt"]).reset_index(drop=True)
