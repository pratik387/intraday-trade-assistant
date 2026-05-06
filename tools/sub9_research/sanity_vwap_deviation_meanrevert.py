"""Pre-coding sanity check for vwap_deviation_meanrevert candidate.

Per sub-9 §3.3 brief gate (specs/2026-05-06-sub-project-9-brief-vwap_
deviation_meanrevert.md): BEFORE writing detector code, simulate the
intraday VWAP-deviation mean-reversion fade on 12 months of 2024 5m
enriched feathers with rolling-window z-score detection.

Decision criterion (from brief):
  PF >= 1.10  -> strong proceed
  1.0-1.10    -> marginal, proceed with caveat
  PF < 1.0    -> retire candidate, do NOT write detector

Long-side ship gate: LONG-side PF must be >= SHORT-side PF * 0.85.

Usage:
    python tools/sub9_research/sanity_vwap_deviation_meanrevert.py

Mechanic (per locked brief params):
  - For each 5m bar 11:00..15:00 IST: z = (close - vwap) / intraday_stdev
  - intraday_stdev = stdev of (close - vwap) over rolling 9-bar window
  - Trigger: |z| >= 2.0 + reversal-candle wick on SAME bar (>= 50% of range)
  - Entry: NEXT bar's close (5m delay; matches brief's "confirmation bar close")
  - Hard SL: trigger-bar high/low + 0.3% buffer; min stop 0.6% of entry
  - T1 (50% qty): VWAP touch (mean revert)
  - T2 (50% qty): VWAP +/- 0.5x intraday stdev (overshoot)
  - Time stop: 8 bars or 15:10 IST hard stop, whichever first
  - One fire per (symbol, day, side). Latch.
  - Universe: F&O 200, mid_cap + small_cap

Note on 5m delay (entry): the brief says "entry on confirmation bar close"
i.e. the bar AFTER the trigger bar. The user-supplied task description
uses "NEXT bar's open" — we follow the brief verbatim (close of the
reversal-confirmation bar) because that is the locked-param reference
and the open-vs-close distinction at 5m granularity is small in
expected impact.
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


# ---- Locked params (research-defensible per brief §3.3 / Mechanic) ----
ALLOWED_CAPS = {"mid_cap", "small_cap"}    # large_cap muted edge; micro_cap thin (brief Universe)
Z_THRESHOLD = 2.0                          # +/-2.0 sigma — locked threshold (brief Mechanic step 1)
STDEV_WINDOW_BARS = 9                      # rolling 9-bar stdev window (~45 min) per brief
REVERSAL_WICK_MIN_FRAC = 0.5               # >= 50% rejection wick on trigger bar (brief step 2)
ENTRY_WINDOW_START_HHMM = "11:00"          # 11:00 IST — VWAP stable after 21+ bars
ENTRY_WINDOW_END_HHMM = "15:00"            # 15:00 IST — exclude MIS auto-square contamination
TIME_STOP_BARS = 8                         # 8 bars (~40 min) per brief step 6
TIME_STOP_HARD_HHMM = "15:10"              # absolute hard stop before MIS auto-square 15:20
SL_BUFFER_PCT = 0.3                        # 0.3% buffer above trigger high / below trigger low
MIN_STOP_PCT = 0.6                         # min stop 0.6% of entry (qty-inflation guard)
T2_OVERSHOOT_STDEV_MULT = 0.5              # T2 = VWAP +/- 0.5 * intraday stdev
RISK_PER_TRADE_RUPEES = 1000               # match circuit_t1_fade_short sanity (ref template)
MIN_ADV_INR_CR = 3.0                       # 20-day avg traded value >= Rs 3 Cr (brief Universe liquidity)


def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = _REPO_ROOT / "backtest-cache-download" / "monthly" / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_feather(path)


def build_full_year_5m() -> pd.DataFrame:
    """Concatenate all 12 monthly 2024 5m feathers."""
    print("  loading 12 monthly 5m feathers ...")
    parts: List[pd.DataFrame] = []
    for m in range(1, 13):
        mdf = _load_5m_for_month(2024, m)
        if not mdf.empty:
            parts.append(mdf)
    big = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    big["d"] = big["date"].dt.date
    print(f"  total 5m bars: {len(big):,}")
    return big


def load_fno_universe() -> set:
    """Load F&O 200 universe; strip NSE: prefix to match feather symbols."""
    path = _REPO_ROOT / "assets" / "fno_liquid_200.csv"
    df = pd.read_csv(path)
    syms = df["symbol"].astype(str).str.replace("NSE:", "", regex=False).tolist()
    print(f"  F&O 200 universe: {len(syms)} symbols")
    return set(syms)


def load_daily_for_liquidity() -> pd.DataFrame:
    """Daily OHLCV for 20-day rolling traded-value liquidity gate."""
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing")
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date(2023, 11, 1)) & (df["d"] <= date(2024, 12, 31))]
    df = df[["symbol", "d", "close", "volume"]].copy()
    df["traded_value"] = df["close"] * df["volume"]
    df = df.sort_values(["symbol", "d"])
    df["adv_20d_cr"] = df.groupby("symbol")["traded_value"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) / 1e7  # Rs Cr
    return df[["symbol", "d", "adv_20d_cr"]]


def find_triggers(big5m: pd.DataFrame, universe: set, adv_table: pd.DataFrame) -> pd.DataFrame:
    """Scan 5m bars for VWAP-deviation triggers + reversal wick + cap/liq filters.

    Returns one row per trigger bar with side ('SHORT'/'LONG') and bar info.
    """
    print("  filtering 5m bars to F&O 200 universe ...")
    df = big5m[big5m["symbol"].isin(universe)].copy()
    print(f"    universe-filtered 5m bars: {len(df):,}")

    # cap-segment filter
    df["nse_symbol"] = "NSE:" + df["symbol"].astype(str)
    df["cap_segment"] = df["nse_symbol"].apply(get_cap_segment)
    df = df[df["cap_segment"].isin(ALLOWED_CAPS)].copy()
    print(f"    cap_segment in {sorted(ALLOWED_CAPS)}: {len(df):,}")

    # liquidity gate via prior-day 20d ADV
    adv_idx = adv_table.set_index(["symbol", "d"])["adv_20d_cr"]
    df["adv_20d_cr"] = df.set_index(["symbol", "d"]).index.map(adv_idx).values
    df = df[df["adv_20d_cr"] >= MIN_ADV_INR_CR].copy()
    print(f"    adv_20d >= Rs {MIN_ADV_INR_CR}Cr: {len(df):,}")

    df = df.sort_values(["symbol", "d", "date"]).reset_index(drop=True)

    # rolling stdev of (close - vwap) over STDEV_WINDOW_BARS, within (sym, d)
    df["dev"] = df["close"] - df["vwap"]
    df["dev_std"] = df.groupby(["symbol", "d"])["dev"].transform(
        lambda s: s.rolling(STDEV_WINDOW_BARS, min_periods=STDEV_WINDOW_BARS).std()
    )
    df["z"] = df["dev"] / df["dev_std"]

    # bar geometry for wick check
    df["range"] = (df["high"] - df["low"]).clip(lower=1e-9)
    df["upper_wick_frac"] = (df["high"] - df["close"]) / df["range"]
    df["lower_wick_frac"] = (df["close"] - df["low"]) / df["range"]

    # active window
    df["hhmm"] = df["date"].dt.strftime("%H:%M")
    df = df[(df["hhmm"] >= ENTRY_WINDOW_START_HHMM) & (df["hhmm"] <= ENTRY_WINDOW_END_HHMM)]
    df = df.dropna(subset=["dev_std", "z"])
    df = df[df["dev_std"] > 0]

    short = df[(df["z"] >= Z_THRESHOLD) & (df["upper_wick_frac"] >= REVERSAL_WICK_MIN_FRAC)].copy()
    short["side"] = "SHORT"
    long_ = df[(df["z"] <= -Z_THRESHOLD) & (df["lower_wick_frac"] >= REVERSAL_WICK_MIN_FRAC)].copy()
    long_["side"] = "LONG"

    print(f"    SHORT triggers (z>=+2.0 + upper-wick>=0.5): {len(short):,}")
    print(f"    LONG triggers  (z<=-2.0 + lower-wick>=0.5): {len(long_):,}")

    return pd.concat([short, long_], ignore_index=True)


def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    """Walk each trigger forward: entry on next bar close, then T1/T2/SL/time stop.

    Latch: one fire per (symbol, day, side).
    """
    print("  simulating entries -> targets/stop/time-stop ...")
    days_per_sym: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    # de-dup latch: keep the EARLIEST trigger per (symbol, d, side)
    triggers = triggers.sort_values(["symbol", "d", "side", "date"])
    triggers = triggers.drop_duplicates(subset=["symbol", "d", "side"], keep="first")

    trades: List[dict] = []
    n_no_next = n_no_bars_after = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]
        sd = t["d"]
        side = t["side"]
        trig_ts = t["date"]
        trig_high = float(t["high"])
        trig_low = float(t["low"])
        trig_vwap = float(t["vwap"])
        trig_dev_std = float(t["dev_std"])

        sym_df = days_per_sym.get(sym)
        if sym_df is None:
            n_no_next += 1; continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_next += 1; continue

        # find trigger bar's index in day
        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr:
            n_no_next += 1; continue
        trig_idx = idx_arr[0]

        # confirmation bar = next bar (5m later)
        if trig_idx + 1 >= len(day_df):
            n_no_next += 1; continue
        conf = day_df.iloc[trig_idx + 1]

        # confirmation bar reversal-candle filter (matches brief Mechanic step 2)
        c_open = float(conf["open"]); c_close = float(conf["close"])
        c_high = float(conf["high"]); c_low = float(conf["low"])
        c_range = max(c_high - c_low, 1e-9)
        prior_close = float(t["close"])
        if side == "SHORT":
            ok = (c_close < c_open) and (c_close < prior_close) and ((c_high - c_close) / c_range >= REVERSAL_WICK_MIN_FRAC)
        else:
            ok = (c_close > c_open) and (c_close > prior_close) and ((c_close - c_low) / c_range >= REVERSAL_WICK_MIN_FRAC)
        if not ok:
            continue

        # Q1 user decision 2026-05-06: ENTRY at NEXT-BAR OPEN (Streak/AlgoTest
        # convention "Signal candle = Trade candle - 1"). Confirmation bar =
        # idx+1; entry bar = idx+2 at its open.
        if trig_idx + 2 >= len(day_df):
            n_no_next += 1; continue
        entry_bar = day_df.iloc[trig_idx + 2]
        entry_price = float(entry_bar["open"])
        entry_ts = entry_bar["date"]

        # hard SL
        if side == "SHORT":
            sl_struct = trig_high * (1.0 + SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 + MIN_STOP_PCT / 100.0)
            hard_sl = max(sl_struct, sl_min)
            stop_distance = hard_sl - entry_price
        else:
            sl_struct = trig_low * (1.0 - SL_BUFFER_PCT / 100.0)
            sl_min = entry_price * (1.0 - MIN_STOP_PCT / 100.0)
            hard_sl = min(sl_struct, sl_min)
            stop_distance = entry_price - hard_sl

        if stop_distance <= 0:
            continue

        # T1 = VWAP touch on subsequent bar; T2 = VWAP +/- 0.5*stdev (overshoot)
        if side == "SHORT":
            t1_target = trig_vwap
            t2_target = trig_vwap - T2_OVERSHOOT_STDEV_MULT * trig_dev_std
        else:
            t1_target = trig_vwap
            t2_target = trig_vwap + T2_OVERSHOOT_STDEV_MULT * trig_dev_std

        # Walk forward from entry bar (Q1: entry at trig_idx+2 open) up to
        # TIME_STOP_BARS or 15:10 hard stop. Entry-bar high/low/close included.
        after = day_df.iloc[trig_idx + 2:].reset_index(drop=True)
        forward = after.iloc[: TIME_STOP_BARS].copy()
        if forward.empty:
            n_no_bars_after += 1; continue

        exit_ts = None; exit_price = None; exit_reason = None
        # 50% qty at T1, 50% at T2/SL/time
        # for sanity simplicity: simulate T1 partial as 50% qty exit at t1_target if hit;
        # remaining 50% rides to T2 / SL / time-stop. PnL aggregated.
        hit_t1 = False
        t1_exit_price = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            ts = bar["date"]; high = float(bar["high"]); low = float(bar["low"])
            # Q8 breakeven trail: after T1 fills, SL on T2 leg moves to entry_price.
            active_sl = entry_price if hit_t1 else hard_sl
            # SL first (worst-case fill)
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
            # 15:10 hard stop
            if ts.strftime("%H:%M") >= TIME_STOP_HARD_HHMM:
                exit_ts = ts; exit_price = float(bar["close"]); exit_reason = "time_stop_hard"
                break

        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]; exit_price = float(last["close"]); exit_reason = "time_stop_bars"

        # qty sizing: risk Rs 1000 / stop_distance (rounded down, min 1)
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        # if T1 was hit, model 50% qty exit at T1, 50% qty at exit_price
        if hit_t1:
            qty_t1 = qty // 2
            qty_t2 = qty - qty_t1
            if side == "SHORT":
                pnl_t1 = (entry_price - t1_exit_price) * qty_t1
                pnl_t2 = (entry_price - exit_price) * qty_t2
            else:
                pnl_t1 = (t1_exit_price - entry_price) * qty_t1
                pnl_t2 = (exit_price - entry_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            # fees on both legs
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL" if side == "SHORT" else "BUY")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL" if side == "SHORT" else "BUY")
            fee = fee_t1 + fee_t2
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
            "trigger_ts": trig_ts,
            "trigger_z": float(t["z"]),
            "trigger_dev_std": trig_dev_std,
            "trigger_vwap": trig_vwap,
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
        n_traded += 1

    print(f"\n  no next bar:        {n_no_next}")
    print(f"  no bars after entry:{n_no_bars_after}")
    print(f"  traded:             {n_traded}")
    return pd.DataFrame(trades)


def report(trades: pd.DataFrame) -> None:
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        return
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(wins / losses, 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = round(daily.mean() / daily.std(), 3) if daily.std() > 0 else 0.0
    wr = round(float((npnl > 0).mean()) * 100, 1)

    print("\n=== vwap_deviation_meanrevert -- pre-coding sanity check ===")
    print(f"Period: {trades['T1_entry_date'].min()} .. {trades['T1_entry_date'].max()}")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{int(npnl.sum()):,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer side:")
    short_pf = None; long_pf = None
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
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>4} PF={pf2:>5} netPnL=Rs.{net:>10,}")

    print("\nPer side x cap_segment:")
    for (sd, cap), grp in trades.groupby(["side", "cap_segment"]):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(w / l, 3) if l > 0 else float("inf")
        net = int(grp["net_pnl"].sum())
        print(f"  {sd:<6} {cap:<12} n={n2:>4} PF={pf2:>5} netPnL=Rs.{net:>10,}")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        print(f"  {rsn:<22} n={len(grp):>4} avg_net=Rs.{int(grp['net_pnl'].mean()):>6,}")

    print("\n--- VERDICT ---")
    if pf >= 1.10:
        print(f"PF={pf} >= 1.10 -> STRONG PROCEED. Move to detector implementation.")
    elif pf >= 1.00:
        print(f"PF={pf} in [1.00, 1.10) -> marginal. Proceed with caveat.")
    else:
        print(f"PF={pf} < 1.00 -> RETIRE candidate. Do not write detector code.")

    if short_pf is not None and long_pf is not None and short_pf > 0:
        ratio = round(long_pf / short_pf, 3) if isinstance(long_pf, (int, float)) else 0.0
        print(f"\nLong/Short PF ratio: {ratio} (gate: >= 0.85 for bidirectional ship)")
        if isinstance(long_pf, (int, float)) and long_pf >= short_pf * 0.85:
            print("  LONG-side passes ship gate.")
        else:
            print("  LONG-side FAILS ship gate -> SHORT-only ship.")


def main():
    big5m = build_full_year_5m()
    universe = load_fno_universe()
    adv_table = load_daily_for_liquidity()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, universe, adv_table)
    print(f"\nTotal triggers (both sides, before latch): {len(triggers)}")
    if triggers.empty:
        return

    print("\nSimulating entries -> exits:")
    trades = simulate(triggers, big5m)
    report(trades)

    out = _REPO_ROOT / "reports" / "sub9_sanity" / "vwap_deviation_meanrevert_trades.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
