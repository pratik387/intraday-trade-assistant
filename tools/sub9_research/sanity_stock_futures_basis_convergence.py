"""Pre-coding sanity check for stock_futures_basis_convergence_T1_expiry candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-07-sub-project-9-brief-stock_
futures_basis_convergence.md): BEFORE writing detector code, simulate
the directional SPOT trade using futures basis as a SIGNAL on T-1
(day before monthly expiry).

Decision criterion (from brief §9):
  PF >= 1.10  -> EOD-pass; escalate to intraday Phase 2.
  PF < 1.10   -> RETIRE; do NOT invest in intraday ingestion.

Falsification gate (brief §9 risk #4): if T-1 PF and the same trigger
on a non-T-1 control day (days_to_expiry == 7) are within 0.10, the
basis-anomaly edge is NOT expiry-mechanical. Retire regardless of
T-1 PF level.

Two-phase sanity per brief §10:
  Phase 1 (this script, ENABLED): EOD-only. Trigger uses T-1 EOD basis;
    enter SPOT at T-1 EOD close, exit at T+0 EOD close. Cheap test that
    gates whether intraday ingestion is justified.
  Phase 2 (commented out below): Intraday — enter spot at T+0 09:15 open,
    exit at 15:10 close. Only run if Phase 1 PF >= 1.10.

Locked params (brief §6 / §5):
  - LONG  spot when T-1 basis_bps - median_20d_bps > +25 bps (futures expensive)
  - SHORT spot when T-1 basis_bps - median_20d_bps < -15 bps (futures cheap)
  - 1% hard stop on T+0 spot (per task description; brief §6 step 4 cites
    0.6% intraday stop, but EOD-Phase has no intraday bars to evaluate
    intrabar stop — use a daily-close 1% guard via T+0 OHLC).
  - Time stop = T+0 close (Phase 1) / T+0 15:10 (Phase 2).
  - Latch one fire per (symbol, expiry_date, side).
  - Universe: F&O 200 (assets/fno_liquid_200.csv) intersected with basis parquet.
  - 20d rolling median basis from days_to_expiry in [7..20] only (avoids
    expiry-edge contamination), per task spec.

Discovery period: 2023-01-01 .. 2024-12-31.

Usage:
    python tools/sub9_research/sanity_stock_futures_basis_convergence.py
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


# ---- Locked params (brief §5 / §6 / §9) ----
LONG_TRIGGER_BPS = 25.0     # basis - median > +25bps -> LONG spot
SHORT_TRIGGER_BPS = -15.0   # basis - median < -15bps -> SHORT spot
HARD_STOP_PCT = 1.0         # 1% hard stop on T+0 spot
MEDIAN_WINDOW = 20          # 20-day rolling median
MEDIAN_DTE_LO = 7           # rolling median uses only days_to_expiry in [7..20]
MEDIAN_DTE_HI = 20
RISK_PER_TRADE_RUPEES = 1000

# Falsification control: same trigger logic on non-T-1 day (days_to_expiry == 7).
# If control PF >= T-1 PF - 0.10, the edge is NOT expiry-mechanical -> RETIRE.
CONTROL_DTE = 7
FALSIFICATION_PF_DELTA = 0.10

# Discovery window (per task spec).
DISCOVERY_START = date(2023, 1, 1)
DISCOVERY_END = date(2024, 12, 31)

# Phase 2 (intraday) — DISABLED for cheap-EOD-first per brief §10.
# Switch on only after Phase 1 PF >= 1.10.
RUN_INTRADAY_PHASE_2 = False


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
    """Production-grade daily spot OHLC (consolidated_daily.feather)."""
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
    """Per (symbol), rolling median of basis_bps using ONLY rows where
    days_to_expiry is in [MEDIAN_DTE_LO..MEDIAN_DTE_HI] — avoids
    expiry-edge contamination.

    Implementation: extract eligible-only rows, compute trailing-20
    rolling median (shift(1) so today's value is excluded), forward-fill
    that as-of value back onto the full series. The median attached to
    every row is "the median of the last 20 eligible observations
    strictly before this session_date".

    With ~13-14 eligible rows per monthly cycle, 20 observations span
    ~1.5 months — the brief calls this "20-day rolling median basis"
    (§6 step 1) so we honor the 20-observation count, not 20 calendar
    days.
    """
    print(f"  computing 20-obs rolling median basis (dte in [{MEDIAN_DTE_LO}..{MEDIAN_DTE_HI}]) ...")
    df = basis.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    eligible_mask = df["days_to_expiry"].between(MEDIAN_DTE_LO, MEDIAN_DTE_HI)

    # Compute rolling median on eligible-only rows.
    elig = df[eligible_mask].copy()
    elig["_med_eligible_only"] = (
        elig.groupby("symbol")["basis_bps"]
            .transform(lambda s: s.shift(1).rolling(MEDIAN_WINDOW, min_periods=MEDIAN_WINDOW).median())
    )

    # Map back: for every row in df, take the most recent eligible row's
    # median value strictly before this row's session_date.
    df = df.merge(
        elig[["symbol", "session_date", "_med_eligible_only"]],
        on=["symbol", "session_date"], how="left",
    )
    df = df.sort_values(["symbol", "session_date"]).reset_index(drop=True)
    # Forward-fill within each symbol so non-eligible rows inherit the
    # latest computed median from preceding eligible rows.
    df["median_20d_bps"] = (
        df.groupby("symbol")["_med_eligible_only"].transform(lambda s: s.ffill())
    )
    df = df.drop(columns=["_med_eligible_only"])

    n_with_median = df["median_20d_bps"].notna().sum()
    print(f"  rows with valid median: {n_with_median:,} / {len(df):,}")
    return df


def find_triggers(
    basis: pd.DataFrame,
    universe: set,
    target_dte: int,
    label: str,
) -> pd.DataFrame:
    """Filter basis rows to (target_dte) entries, apply universe filter,
    then apply 25bps long / -15bps short trigger.

    Funnel printout: candidate -> after-trigger -> after-universe."""
    print(f"\n--- {label} (days_to_expiry == {target_dte}) ---")
    cand = basis[basis["days_to_expiry"] == target_dte].copy()
    if target_dte == 1:
        # Monthly-expiry safety: parquet should already exclude weeklies
        # (FUTSTK is monthly), but the brief explicitly states day>=22.
        cand = cand[pd.to_datetime(cand["expiry_date"]).dt.day >= 22]
    print(f"  candidate (symbol, T-{1 if target_dte==1 else target_dte}) pairs: {len(cand):,}")

    cand = cand.dropna(subset=["median_20d_bps"])
    print(f"  after symbol-level history coverage (median valid): {len(cand):,}")

    cand = cand[cand["symbol"].isin(universe)]
    print(f"  after F&O 200 universe filter: {len(cand):,}")

    cand["dislocation_bps"] = cand["basis_bps"] - cand["median_20d_bps"]
    cand["side"] = np.where(
        cand["dislocation_bps"] > LONG_TRIGGER_BPS, "LONG",
        np.where(cand["dislocation_bps"] < SHORT_TRIGGER_BPS, "SHORT", None),
    )
    triggered = cand[cand["side"].notna()].copy()
    print(f"  after trigger ({LONG_TRIGGER_BPS:+.0f}bps long / {SHORT_TRIGGER_BPS:+.0f}bps short): "
          f"{len(triggered):,}")
    print(f"    LONG  triggers: {(triggered['side']=='LONG').sum():,}")
    print(f"    SHORT triggers: {(triggered['side']=='SHORT').sum():,}")

    triggered = triggered[
        (triggered["session_date"] >= DISCOVERY_START)
        & (triggered["session_date"] <= DISCOVERY_END)
    ]
    print(f"  after discovery-period filter [{DISCOVERY_START}..{DISCOVERY_END}]: {len(triggered):,}")
    return triggered.reset_index(drop=True)


def simulate_eod(triggers: pd.DataFrame, daily: pd.DataFrame, label: str) -> pd.DataFrame:
    """Phase 1 EOD: enter at signal-day's spot close, exit at next-trading-day
    spot close OR at -1% (long) / +1% (short) intrabar stop on T+0 (using
    T+0 high/low, NOT shadow simulation).

    Latch: one fire per (symbol, expiry_date, side)."""
    print(f"\n  Phase 1 EOD simulate ({label}): {len(triggers):,} triggers ...")
    triggers = triggers.sort_values(["symbol", "expiry_date", "side", "session_date"])
    triggers = triggers.drop_duplicates(
        subset=["symbol", "expiry_date", "side"], keep="first"
    ).reset_index(drop=True)
    print(f"  after latch (one fire per symbol×expiry×side): {len(triggers):,}")

    # Index daily by (symbol, d) for fast forward-day lookup.
    daily_sorted = daily.sort_values(["symbol", "d"]).reset_index(drop=True)
    by_sym: Dict[str, pd.DataFrame] = {
        sym: g.reset_index(drop=True) for sym, g in daily_sorted.groupby("symbol")
    }

    trades: List[dict] = []
    n_no_t0 = n_no_signal_day = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]
        sig_d = t["session_date"]
        side = t["side"]

        sym_df = by_sym.get(sym)
        if sym_df is None or sym_df.empty:
            n_no_t0 += 1; continue

        # signal-day close = entry price proxy (EOD-Phase 1)
        sd_row = sym_df[sym_df["d"] == sig_d]
        if sd_row.empty:
            n_no_signal_day += 1; continue
        entry_price = float(sd_row.iloc[0]["close"])

        # T+0 = next trading day's row
        future = sym_df[sym_df["d"] > sig_d]
        if future.empty:
            n_no_t0 += 1; continue
        t0_row = future.iloc[0]
        t0_open = float(t0_row["open"])
        t0_high = float(t0_row["high"])
        t0_low = float(t0_row["low"])
        t0_close = float(t0_row["close"])
        t0_d = t0_row["d"]

        # 1% hard stop on T+0
        if side == "LONG":
            hard_sl = entry_price * (1.0 - HARD_STOP_PCT / 100.0)
        else:
            hard_sl = entry_price * (1.0 + HARD_STOP_PCT / 100.0)
        stop_distance = abs(entry_price - hard_sl)

        # exit logic — gap-through stop = filled at T+0 open;
        # intrabar stop hit = filled at hard_sl; otherwise = T+0 close
        if side == "LONG":
            if t0_open <= hard_sl:
                exit_price = t0_open
                exit_reason = "gap_through_stop"
            elif t0_low <= hard_sl:
                exit_price = hard_sl
                exit_reason = "stop"
            else:
                exit_price = t0_close
                exit_reason = "eod"
        else:  # SHORT
            if t0_open >= hard_sl:
                exit_price = t0_open
                exit_reason = "gap_through_stop"
            elif t0_high >= hard_sl:
                exit_price = hard_sl
                exit_reason = "stop"
            else:
                exit_price = t0_close
                exit_reason = "eod"

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
            "T_minus_1_signal_date": sig_d,
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
        n_traded += 1

    print(f"  no T+0 day:       {n_no_t0}")
    print(f"  no signal-day spot: {n_no_signal_day}")
    print(f"  traded:           {n_traded}")
    return pd.DataFrame(trades)


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


def report(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        print(f"\n[{label}] NO TRADES")
        return {"n": 0, "pf": 0.0, "wr": 0.0, "sharpe": 0.0, "net": 0}

    s = _summary(trades)
    print(f"\n=== {label} ===")
    print(f"Period: {trades['T0_entry_date'].min()} .. {trades['T0_entry_date'].max()}")
    print(f"Trades: n = {s['n']}")
    print(f"Win rate: {s['wr']}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{s['net']:,}")
    print(f"NET PF:    {s['pf']}")
    print(f"NET Sharpe (daily): {s['sharpe']}")

    print("\nPer side:")
    for sd, grp in trades.groupby("side"):
        ss = _summary(grp)
        print(f"  {sd:<6} n={ss['n']:>4} PF={ss['pf']:>5} WR={ss['wr']:>5}% netPnL=Rs.{ss['net']:>10,}")

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        ss = _summary(grp)
        print(f"  {str(cap):<12} n={ss['n']:>4} PF={ss['pf']:>5} netPnL=Rs.{ss['net']:>10,}")

    print("\nPer month:")
    months = pd.to_datetime(trades["T0_entry_date"]).dt.to_period("M")
    for mn, grp in trades.groupby(months):
        ss = _summary(grp)
        print(f"  {str(mn)} n={ss['n']:>4} PF={ss['pf']:>5} netPnL=Rs.{ss['net']:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        avg = int(grp["net_pnl"].mean()) if len(grp) > 0 else 0
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{avg:>6,}")

    print("\nTop-10 symbols by netPnL:")
    sym_grp = trades.groupby("symbol")["net_pnl"].agg(["count", "sum"]).reset_index()
    sym_grp.columns = ["symbol", "n", "net"]
    sym_grp = sym_grp.sort_values("net", ascending=False).head(10)
    for _, r in sym_grp.iterrows():
        print(f"  {r['symbol']:<22} n={int(r['n']):>3} netPnL=Rs.{int(r['net']):>9,}")

    return s


def sample_size_diagnostic(t1_trades: pd.DataFrame) -> None:
    print("\n--- SAMPLE-SIZE DIAGNOSTIC (brief §8 / §9 risk #1) ---")
    expected_per_side_2yr = 1100  # 24 monthly expiries × 153 syms × ~30%
    print(f"  Brief expectation: ~1,100 events/side over 2yr (24 expiries × 153 syms × 30%).")
    if t1_trades.empty:
        print("  ZERO trades — flag SEVERELY thin.")
        return
    for sd, grp in t1_trades.groupby("side"):
        n = len(grp)
        flag = "OK" if n >= 500 else ("MARGINAL" if n >= 100 else "TOO THIN")
        print(f"  {sd}: n={n}  vs expected ~{expected_per_side_2yr // 2}  -> {flag}")
    n_total = len(t1_trades)
    print(f"  TOTAL n={n_total}  brief floor = 500 (retire if below)")


def falsification_gate(t1_summary: dict, control_summary: dict) -> str:
    print("\n--- FALSIFICATION GATE (brief §9 risk #4: expiry-mechanic) ---")
    print(f"  T-1 PF (days_to_expiry==1):  {t1_summary['pf']}")
    print(f"  Control PF (dte=={CONTROL_DTE}):           {control_summary['pf']}")
    delta = (t1_summary["pf"] if isinstance(t1_summary["pf"], (int, float)) else 0.0) \
            - (control_summary["pf"] if isinstance(control_summary["pf"], (int, float)) else 0.0)
    print(f"  T-1 PF - Control PF = {round(delta, 3)}")
    print(f"  Required delta >= {FALSIFICATION_PF_DELTA} for expiry mechanic to be the source.")
    if delta >= FALSIFICATION_PF_DELTA:
        print("  PASS: T-1 edge materially exceeds non-expiry control. Mechanic plausible.")
        return "pass"
    print("  FAIL: control PF too close to T-1 PF -> edge is NOT expiry-mechanical. RETIRE.")
    return "fail"


def main():
    print("== stock_futures_basis_convergence_T1_expiry — Phase 1 EOD sanity ==\n")
    print("Loading inputs:")
    basis = load_basis()
    universe = load_fno_universe()
    daily = load_daily_spot()

    print("\nComputing per-symbol 20d rolling median basis ...")
    basis = compute_rolling_median(basis)

    # Phase 1 — T-1 (days_to_expiry == 1)
    t1_triggers = find_triggers(basis, universe, target_dte=1, label="T-1 (signal day)")
    print()
    t1_trades = simulate_eod(t1_triggers, daily, label="T-1")

    # Falsification control — same trigger logic on dte == 7
    ctrl_triggers = find_triggers(basis, universe, target_dte=CONTROL_DTE,
                                  label=f"CONTROL (dte=={CONTROL_DTE})")
    print()
    ctrl_trades = simulate_eod(ctrl_triggers, daily, label=f"control(dte={CONTROL_DTE})")

    # Reports
    t1_summary = report(t1_trades, label="T-1 EOD sanity (Phase 1)")
    ctrl_summary = report(ctrl_trades, label=f"CONTROL EOD (dte=={CONTROL_DTE}) — falsification")

    sample_size_diagnostic(t1_trades)
    gate_result = falsification_gate(t1_summary, ctrl_summary)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "stock_futures_basis_convergence_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not t1_trades.empty:
        t1_out = t1_trades.copy()
        t1_out["arm"] = "T_minus_1"
    else:
        t1_out = pd.DataFrame()
    if not ctrl_trades.empty:
        c_out = ctrl_trades.copy()
        c_out["arm"] = f"control_dte_{CONTROL_DTE}"
    else:
        c_out = pd.DataFrame()
    full = pd.concat([t1_out, c_out], ignore_index=True) if (not t1_out.empty or not c_out.empty) else pd.DataFrame()
    full.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")

    # Final verdict line — brief §9 + falsification gate
    print("\n=== VERDICT ===")
    pf_t1 = t1_summary["pf"]
    n_t1 = t1_summary["n"]
    sharpe_t1 = t1_summary["sharpe"]
    pf_ctrl = ctrl_summary["pf"]

    print(f"  T-1 n={n_t1}  PF={pf_t1}  WR={t1_summary['wr']}%  Sharpe={sharpe_t1}")
    print(f"  Control(dte={CONTROL_DTE}) n={ctrl_summary['n']}  PF={pf_ctrl}")
    print(f"  Falsification gate: {gate_result.upper()}")

    retire_reasons = []
    if isinstance(pf_t1, (int, float)) and pf_t1 < 1.10:
        retire_reasons.append(f"T-1 PF={pf_t1} < 1.10")
    if n_t1 < 500:
        retire_reasons.append(f"n={n_t1} < 500 (brief §9 floor)")
    if isinstance(sharpe_t1, (int, float)) and sharpe_t1 <= 0:
        retire_reasons.append(f"Sharpe={sharpe_t1} <= 0")
    if gate_result == "fail":
        retire_reasons.append("falsification gate FAIL (edge not expiry-mechanical)")

    if retire_reasons:
        print("VERDICT: RETIRE candidate. Reasons: " + "; ".join(retire_reasons))
    else:
        print("VERDICT: PROCEED to Phase 2 intraday escalation. T-1 PF clears 1.10, "
              "n>=500, Sharpe>0, falsification gate passes.")


# ---------------------------------------------------------------------------
# Phase 2 — INTRADAY (DISABLED until Phase 1 PF >= 1.10)
# ---------------------------------------------------------------------------
# Per brief §10 step 3: only run after EOD-Phase passes. Skeleton retained
# below for ease of activation.
#
# def simulate_intraday(triggers, big5m):
#     """Enter spot at T+0 09:15 open, exit at T+0 15:10 close OR 1% intrabar stop.
#     5m feathers: backtest-cache-download/monthly/{YYYY}_{MM}_5m_enriched.feather"""
#     pass
#
# if RUN_INTRADAY_PHASE_2:
#     # build_5m, simulate_intraday, report — same fee model.
#     ...
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
