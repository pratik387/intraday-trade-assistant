"""DTE sweep for stock_futures_basis_convergence candidate.

Original sanity (sanity_stock_futures_basis_convergence.py) tested:
  - T-1 (dte=1): n=1,496, NET PF 0.753, Sharpe -0.284  -> FAIL
  - Control dte=7:  n=1,174, NET PF 1.182, Sharpe +0.156 -> incidental PASS

The brief assumed T-1 was the signal day; the falsification "control" at
dte=7 was actually where the edge appeared. This sweep tests every
realistic dte bucket to:
  1. Identify which day(s) carry the basis-dislocation edge.
  2. Validate continuity (adjacent dte buckets should also be positive
     if the signal is real; isolated single-bucket edge => overfit).
  3. Ship the cleanest cell (dte bucket meeting PF>=1.10, Sharpe>0,
     n>=30/side, AND adjacency).

Trigger logic, thresholds, and entry mechanic are IDENTICAL to the
original sanity script. Only the dte-of-entry varies across buckets.

Locked params (per brief §5/§6, unchanged):
  - LONG  spot when basis_bps - median_20d_bps > +25 bps
  - SHORT spot when basis_bps - median_20d_bps < -15 bps
  - 1% hard stop on T+0 spot
  - Median: 20-obs trailing on dte in [7..20] only
  - One fire per (symbol, expiry_date, side)
  - Universe: F&O 200
  - Discovery period: 2023-01-01 .. 2024-12-31
  - Entry: signal-day spot close; exit: next-trading-day spot
    close OR 1% stop / gap-through

Usage:
    python tools/sub9_research/sanity_stock_futures_basis_convergence_dte_sweep.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment           # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee  # noqa: E402


# ---- Locked params (must match sanity_stock_futures_basis_convergence.py) ----
LONG_TRIGGER_BPS = 25.0
SHORT_TRIGGER_BPS = -15.0
HARD_STOP_PCT = 1.0
MEDIAN_WINDOW = 20
MEDIAN_DTE_LO = 7
MEDIAN_DTE_HI = 20
RISK_PER_TRADE_RUPEES = 1000

DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END = date(2024, 12, 31)

# DTE buckets to sweep.
# Task spec calls for {1, 3, 5, 7, 10, 14, 21}. We additionally include
# 6 and 8 as TRUE adjacency neighbours of dte=7 — dte={4,5,11,12,18,19,25}
# are calendar-thin (weekend-boundary dtes) and have ~1/4 the row count of
# their neighbours, so they cannot serve as adjacency tests for dte=7.
DTE_BUCKETS = [1, 3, 5, 6, 7, 8, 10, 14, 21]

# Cell-selection thresholds (per task §1):
MIN_PF = 1.10
MIN_N_PER_SIDE = 30
MIN_SHARPE = 0.0


# ---------------------------------------------------------------------------
# Loaders / median computation — identical to base sanity script
# ---------------------------------------------------------------------------
def load_basis() -> pd.DataFrame:
    path = _REPO_ROOT / "data" / "futures_basis" / "2023_2026_basis.parquet"
    print(f"  loading basis parquet: {path}")
    df = pd.read_parquet(path)
    df["session_date"] = pd.to_datetime(df["session_date"]).dt.date
    df["expiry_date"] = pd.to_datetime(df["expiry_date"]).dt.date
    print(f"  basis rows: {len(df):,}  symbols: {df['symbol'].nunique()}  "
          f"period: {df['session_date'].min()} .. {df['session_date'].max()}")
    return df


def load_fno_universe() -> set:
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_daily_spot() -> pd.DataFrame:
    path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    print(f"  loading daily spot: {path}")
    df = pd.read_feather(path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[["symbol", "d", "open", "high", "low", "close", "volume"]].copy()
    print(f"  daily rows: {len(df):,}  symbols: {df['symbol'].nunique()}")
    return df


def compute_rolling_median(basis: pd.DataFrame) -> pd.DataFrame:
    print(f"  computing 20-obs rolling median basis (dte in [{MEDIAN_DTE_LO}..{MEDIAN_DTE_HI}]) ...")
    df = basis.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    eligible_mask = df["days_to_expiry"].between(MEDIAN_DTE_LO, MEDIAN_DTE_HI)
    elig = df[eligible_mask].copy()
    elig["_med_eligible_only"] = (
        elig.groupby("symbol")["basis_bps"]
            .transform(lambda s: s.shift(1).rolling(MEDIAN_WINDOW, min_periods=MEDIAN_WINDOW).median())
    )
    df = df.merge(
        elig[["symbol", "session_date", "_med_eligible_only"]],
        on=["symbol", "session_date"], how="left",
    )
    df = df.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    df["median_20d_bps"] = (
        df.groupby("symbol")["_med_eligible_only"].transform(lambda s: s.ffill())
    )
    df = df.drop(columns=["_med_eligible_only"])
    n_with_median = df["median_20d_bps"].notna().sum()
    print(f"  rows with valid median: {n_with_median:,} / {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Trigger / simulate — identical mechanic, parameterized by dte
# ---------------------------------------------------------------------------
def find_triggers(basis: pd.DataFrame, universe: set, target_dte: int) -> pd.DataFrame:
    cand = basis[basis["days_to_expiry"] == target_dte].copy()
    if target_dte == 1:
        # Monthly-expiry safety (matches base script)
        cand = cand[pd.to_datetime(cand["expiry_date"]).dt.day >= 22]
    cand = cand.dropna(subset=["median_20d_bps"])
    cand = cand[cand["symbol"].isin(universe)]
    cand["dislocation_bps"] = cand["basis_bps"] - cand["median_20d_bps"]
    cand["side"] = np.where(
        cand["dislocation_bps"] > LONG_TRIGGER_BPS, "LONG",
        np.where(cand["dislocation_bps"] < SHORT_TRIGGER_BPS, "SHORT", None),
    )
    triggered = cand[cand["side"].notna()].copy()
    triggered = triggered[
        (triggered["session_date"] >= DISCOVERY_START)
        & (triggered["session_date"] <= DISCOVERY_END)
    ]
    return triggered.reset_index(drop=True)


def simulate_eod(triggers: pd.DataFrame, by_sym: Dict[str, pd.DataFrame],
                 dte_bucket: int) -> pd.DataFrame:
    triggers = triggers.sort_values(["symbol", "expiry_date", "side", "session_date"])
    triggers = triggers.drop_duplicates(
        subset=["symbol", "expiry_date", "side"], keep="first"
    ).reset_index(drop=True)

    trades: List[dict] = []
    for _, t in triggers.iterrows():
        sym = t["symbol"]
        sig_d = t["session_date"]
        side = t["side"]

        sym_df = by_sym.get(sym)
        if sym_df is None or sym_df.empty:
            continue
        sd_row = sym_df[sym_df["d"] == sig_d]
        if sd_row.empty:
            continue
        entry_price = float(sd_row.iloc[0]["close"])
        future = sym_df[sym_df["d"] > sig_d]
        if future.empty:
            continue
        t0_row = future.iloc[0]
        t0_open = float(t0_row["open"])
        t0_high = float(t0_row["high"])
        t0_low = float(t0_row["low"])
        t0_close = float(t0_row["close"])
        t0_d = t0_row["d"]

        if side == "LONG":
            hard_sl = entry_price * (1.0 - HARD_STOP_PCT / 100.0)
        else:
            hard_sl = entry_price * (1.0 + HARD_STOP_PCT / 100.0)
        stop_distance = abs(entry_price - hard_sl)

        if side == "LONG":
            if t0_open <= hard_sl:
                exit_price, exit_reason = t0_open, "gap_through_stop"
            elif t0_low <= hard_sl:
                exit_price, exit_reason = hard_sl, "stop"
            else:
                exit_price, exit_reason = t0_close, "eod"
        else:
            if t0_open >= hard_sl:
                exit_price, exit_reason = t0_open, "gap_through_stop"
            elif t0_high >= hard_sl:
                exit_price, exit_reason = hard_sl, "stop"
            else:
                exit_price, exit_reason = t0_close, "eod"

        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)
        if side == "LONG":
            realized_pnl = (exit_price - entry_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "BUY")
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
        net_pnl = realized_pnl - fee

        nse_sym = "NSE:" + sym
        trades.append({
            "dte_bucket": dte_bucket,
            "signal_date": sig_d,
            "T0_entry_date": t0_d,
            "expiry_date": t["expiry_date"],
            "days_to_expiry_at_signal": int(t["days_to_expiry"]),
            "symbol": nse_sym,
            "cap_segment": get_cap_segment(nse_sym),
            "side": side,
            "futures_close": float(t["futures_close"]),
            "spot_close": float(t["spot_close"]),
            "basis_bps": float(t["basis_bps"]),
            "median_20d_bps": float(t["median_20d_bps"]),
            "dislocation_bps": float(t["dislocation_bps"]),
            "entry_price": entry_price,
            "hard_sl": hard_sl,
            "t0_open": t0_open,
            "t0_high": t0_high,
            "t0_low": t0_low,
            "t0_close": t0_close,
            "exit_price": exit_price,
            "exit_reason": exit_reason,
            "stop_distance": stop_distance,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def _summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"n": 0, "pf": 0.0, "wr": 0.0, "sharpe": 0.0, "net": 0}
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    wr = round(float((npnl > 0).mean()) * 100, 1)
    daily = trades.groupby("T0_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    return {"n": len(trades), "pf": pf, "wr": wr, "sharpe": sharpe, "net": int(npnl.sum())}


def per_dte_row(dte: int, trades: pd.DataFrame) -> dict:
    overall = _summary(trades)
    longs = trades[trades["side"] == "LONG"] if not trades.empty else trades
    shorts = trades[trades["side"] == "SHORT"] if not trades.empty else trades
    s_long = _summary(longs)
    s_short = _summary(shorts)
    return {
        "dte": dte,
        "n_total": overall["n"],
        "n_long": s_long["n"],
        "n_short": s_short["n"],
        "PF_long": s_long["pf"],
        "PF_short": s_short["pf"],
        "PF_combined": overall["pf"],
        "WR_combined": overall["wr"],
        "Sharpe": overall["sharpe"],
        "net_pnl": overall["net"],
    }


def cell_passes(row: dict) -> bool:
    pf = row["PF_combined"]
    if not isinstance(pf, (int, float)):
        return False
    if pf < MIN_PF:
        return False
    if min(row["n_long"], row["n_short"]) < MIN_N_PER_SIDE:
        return False
    if row["Sharpe"] <= MIN_SHARPE:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("== stock_futures_basis_convergence — DTE SWEEP ==\n")
    print("Loading inputs:")
    basis = load_basis()
    universe = load_fno_universe()
    daily = load_daily_spot()

    print("\nComputing per-symbol 20d rolling median basis ...")
    basis = compute_rolling_median(basis)

    print("\nIndexing daily spot by symbol ...")
    daily_sorted = daily.sort_values(["symbol", "d"]).reset_index(drop=True)
    by_sym: Dict[str, pd.DataFrame] = {
        sym: g.reset_index(drop=True) for sym, g in daily_sorted.groupby("symbol")
    }

    all_trades: List[pd.DataFrame] = []
    rows: List[dict] = []
    for dte in DTE_BUCKETS:
        print(f"\n--- DTE BUCKET dte=={dte} ---")
        trig = find_triggers(basis, universe, target_dte=dte)
        n_long = (trig["side"] == "LONG").sum() if not trig.empty else 0
        n_short = (trig["side"] == "SHORT").sum() if not trig.empty else 0
        print(f"  triggers: total={len(trig):,}  long={n_long:,}  short={n_short:,}")
        trades = simulate_eod(trig, by_sym, dte_bucket=dte)
        print(f"  traded: {len(trades):,}")
        if not trades.empty:
            all_trades.append(trades)
        row = per_dte_row(dte, trades)
        rows.append(row)
        print(f"  n={row['n_total']:>4}  "
              f"PF_long={row['PF_long']:>5}  PF_short={row['PF_short']:>5}  "
              f"PF_comb={row['PF_combined']:>5}  WR={row['WR_combined']:>5}%  "
              f"Sharpe={row['Sharpe']:>6}  net=Rs.{row['net_pnl']:>9,}")

    # ---- Sweep summary table ----
    print("\n\n========== DTE SWEEP — SUMMARY TABLE ==========")
    hdr = (f"{'dte':>4}  {'n_tot':>6}  {'n_lng':>6}  {'n_sht':>6}  "
           f"{'PF_lng':>7}  {'PF_sht':>7}  {'PF_comb':>8}  "
           f"{'WR%':>5}  {'Sharpe':>7}  {'net Rs.':>11}  pass?")
    print(hdr)
    print("-" * len(hdr))
    pass_set = set()
    for r in rows:
        passed = cell_passes(r)
        if passed:
            pass_set.add(r["dte"])
        marker = "PASS" if passed else "----"
        print(f"{r['dte']:>4}  {r['n_total']:>6}  {r['n_long']:>6}  {r['n_short']:>6}  "
              f"{str(r['PF_long']):>7}  {str(r['PF_short']):>7}  "
              f"{str(r['PF_combined']):>8}  {r['WR_combined']:>5}  "
              f"{str(r['Sharpe']):>7}  {r['net_pnl']:>11,}  {marker}")

    # ---- Adjacency check ----
    print("\n\n========== ADJACENCY (continuity) CHECK ==========")
    print(f"  Pass thresholds: PF_combined >= {MIN_PF}, min(n_long, n_short) >= "
          f"{MIN_N_PER_SIDE}, Sharpe > {MIN_SHARPE}")
    print(f"  Cells that passed: {sorted(pass_set) if pass_set else 'NONE'}")

    by_dte = {r["dte"]: r for r in rows}

    def adj_label(d: int) -> str:
        # closest swept neighbours either side of d
        idx = DTE_BUCKETS.index(d)
        lower = DTE_BUCKETS[idx - 1] if idx > 0 else None
        upper = DTE_BUCKETS[idx + 1] if idx < len(DTE_BUCKETS) - 1 else None
        return f"lower={lower}, upper={upper}"

    def is_positive(r: dict) -> bool:
        pf = r["PF_combined"]
        return isinstance(pf, (int, float)) and pf > 1.0

    if pass_set:
        for d in sorted(pass_set):
            idx = DTE_BUCKETS.index(d)
            lower = DTE_BUCKETS[idx - 1] if idx > 0 else None
            upper = DTE_BUCKETS[idx + 1] if idx < len(DTE_BUCKETS) - 1 else None
            l_pos = is_positive(by_dte[lower]) if lower is not None else None
            u_pos = is_positive(by_dte[upper]) if upper is not None else None
            print(f"\n  Winning bucket dte={d}: PF={by_dte[d]['PF_combined']}, "
                  f"Sharpe={by_dte[d]['Sharpe']}")
            print(f"    Adjacent buckets: {adj_label(d)}")
            if lower is not None:
                print(f"    dte={lower}: PF={by_dte[lower]['PF_combined']}, "
                      f"Sharpe={by_dte[lower]['Sharpe']} -> "
                      f"{'POSITIVE' if l_pos else 'NOT POSITIVE'}")
            if upper is not None:
                print(f"    dte={upper}: PF={by_dte[upper]['PF_combined']}, "
                      f"Sharpe={by_dte[upper]['Sharpe']} -> "
                      f"{'POSITIVE' if u_pos else 'NOT POSITIVE'}")
            adjacency_ok = (l_pos or u_pos)  # at least one side positive
            both_ok = (l_pos and u_pos)
            if both_ok:
                print(f"    ADJACENCY: STRONG (both sides positive) -> "
                      f"signal continuous; APPROVE dte={d}.")
            elif adjacency_ok:
                print(f"    ADJACENCY: PARTIAL (one side positive) -> "
                      f"signal partially continuous; PROCEED with caution.")
            else:
                print(f"    ADJACENCY: WEAK (no adjacent positive PF) -> "
                      f"isolated bucket, likely overfit. RETIRE.")
    else:
        print("\n  No bucket passed all thresholds. Showing best-three by PF anyway:")
        ranked = sorted(rows, key=lambda r: (r["PF_combined"] if isinstance(r["PF_combined"], (int, float)) else -1),
                        reverse=True)
        for r in ranked[:3]:
            print(f"    dte={r['dte']}: n={r['n_total']}  PF={r['PF_combined']}  "
                  f"Sharpe={r['Sharpe']}  WR={r['WR_combined']}%  "
                  f"long_n={r['n_long']}  short_n={r['n_short']}")

    # ---- Save trades CSV ----
    out = _REPO_ROOT / "reports" / "sub9_sanity" / "stock_futures_basis_convergence_dte_sweep_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if all_trades:
        full = pd.concat(all_trades, ignore_index=True)
    else:
        full = pd.DataFrame()
    full.to_csv(out, index=False)
    print(f"\nFull trade log: {out}  ({len(full):,} rows)")

    # ---- Final verdict ----
    print("\n\n========== FINAL VERDICT ==========")
    if not pass_set:
        print("VERDICT: NO dte bucket meets PF>=1.10 + n>=30/side + Sharpe>0.")
        print("         Original 'control' dte=7 PF=1.182 was likely random noise.")
        print("         RETIRE candidate.")
    else:
        # Best winning bucket (highest PF among passes)
        best = max((by_dte[d] for d in pass_set), key=lambda r: r["PF_combined"])
        d = best["dte"]
        idx = DTE_BUCKETS.index(d)
        lower = DTE_BUCKETS[idx - 1] if idx > 0 else None
        upper = DTE_BUCKETS[idx + 1] if idx < len(DTE_BUCKETS) - 1 else None
        l_pos = is_positive(by_dte[lower]) if lower is not None else None
        u_pos = is_positive(by_dte[upper]) if upper is not None else None
        adjacency_strong = (l_pos and u_pos)
        adjacency_partial = (l_pos or u_pos)
        if adjacency_strong:
            print(f"VERDICT: APPROVE dte={d} for detector implementation.")
            print(f"         PF={best['PF_combined']}, Sharpe={best['Sharpe']}, "
                  f"n={best['n_total']} (long={best['n_long']}, short={best['n_short']}).")
            print(f"         Both adjacent buckets positive -> signal continuous, "
                  f"dte cell is the stable signal.")
        elif adjacency_partial:
            print(f"VERDICT: PROCEED-WITH-CAUTION dte={d} (partial adjacency).")
            print(f"         PF={best['PF_combined']}, Sharpe={best['Sharpe']}, "
                  f"n={best['n_total']}.")
            print(f"         Only one adjacent bucket positive; consider a wider "
                  f"window dte+/-2 in production rather than locking single-day.")
        else:
            print(f"VERDICT: RETIRE — dte={d} passes thresholds but is isolated "
                  f"(no adjacent bucket positive).")
            print(f"         Likely overfit / random.")


if __name__ == "__main__":
    main()
