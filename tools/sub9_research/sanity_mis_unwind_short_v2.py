"""Sanity check for mis_unwind_short v2 — tape-confirmed unwind retry.

Per `specs/2026-05-14-research-mis_unwind_retry.md`. Two prior attempts at this
asymmetry both failed because they shorted in ANTICIPATION of a 14:30-15:15 MIS
unwind that often didn't materialize (CNC-conversion squeeze):
  - Attempt 1 `mis_unwind_short`         : PF 0.355, n=304, WR 9.2%
  - Attempt 2 `mis_unwind_short_late_session`: PF 0.367, n=1008, WR 40.7%

This v2 mechanic shorts ONLY after the unwind has physically materialized on
tape — VWAP loss + red bar + structural break + volume spike — and on the
broad mid+small-cap universe where retail MIS leverage actually concentrates
(NOT F&O 200 which is 143/153 large_cap).

Mechanic — locked per research note:

  Universe: broad NSE equities, cap_segment in {mid_cap, small_cap}.
  Liquidity: 20-day ADV * close >= Rs 5 Cr.

  Day-level qualification (at 14:00 IST):
    - intraday_ret(14:00) in [+1.0%, +6.0%]  (long-bag exists; excludes runaways)
    - |gap_pct_at_open| <= 2.0%              (cross-detector vs gap_fade_short)

  Tape-confirmation entry trigger (5m bar in [14:30, 15:00]):
    ALL of:
      1. close < vwap                              (VWAP loss — flow rolled over)
      2. close < open                              (red bar — confirmed selling)
      3. close < min(prev 3 bars' lows)            (structural break of morning)
      4. volume / mean(prior 12 bars vol) >= 1.5   (institutional volume spike)
    Latch: first qualifying bar per (symbol, day).

  Entry:    SHORT at qualifying bar's close.
  Hard SL:  max(intraday_high_so_far, entry * 1.012).
  T1:       entry * 0.995  (50% partial; breakeven trail after).
  Time-stop: HARD exit 15:10 bar close (5 min before SEBI 15:15 auto-square).
  T2:       implicit — whatever the 15:10 bar close gives after T1 partial.

  Discovery window: 2024-09-01 .. 2025-09-30
  Universe + fees match other sub9 sanity scripts.

Pre-registered ship gates (per gauntlet-v2):
  Aggregate: n >= 500, NET PF >= 1.10, daily Sharpe > 0
  Per-month: >= 55% winning months, top month <= 40% of NET

Verdict:
  PF >= 1.30 AND n >= 100 AND stability passes -> STRONG PROCEED
  PF 1.10-1.29                                  -> MARGINAL
  Otherwise                                     -> RETIRE

Usage:
    python tools/sub9_research/sanity_mis_unwind_short_v2.py
"""
from __future__ import annotations

import io
import sys
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

# Ensure stdout can print rule_changes.csv unicode (arrow chars etc.)
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from services.symbol_metadata import get_cap_segment            # noqa: E402
from services.regime_break_detector import check_window, GauntletRegimeBreak  # noqa: E402
from tools.sub7_validation.build_per_setup_pnl import calc_fee   # noqa: E402


# ---- Locked params -----------------------------------------------------

DISCOVERY_START = date(2024, 9, 1)
DISCOVERY_END   = date(2025, 9, 30)

# Cap segments where retail MIS leverage concentrates (5x available).
ALLOWED_CAPS = {"mid_cap", "small_cap"}

# Liquidity floor.
MIN_ADV_INR_CR = 5.0

# Day-level qualification at 14:00 IST.
HHMM_1400 = 1400
INTRADAY_RET_LO_PCT = 1.0     # long-bag exists
INTRADAY_RET_HI_PCT = 6.0     # exclude runaway (CNC-convert magnets)

# Gap-day cross-detector exclusion.
MAX_ABS_GAP_PCT = 2.0

# Entry-window bars (5m bar closes in [14:30, 15:00] inclusive).
ENTRY_HHMM_LO = 1430
ENTRY_HHMM_HI = 1500

# Tape-confirmation gates.
STRUCT_BREAK_LOOKBACK = 3            # close < min(prev 3 bars' lows)
RVOL_LOOKBACK_BARS = 12              # prior 12 bars same session
RVOL_MIN = 1.5

# Stop / target.
STOP_MIN_PCT = 1.2                   # entry * 1.012 floor
T1_PCT = 0.5                         # entry * 0.995

# Hard time stop.
TIMESTOP_HHMM = 1510

# Risk per trade.
RISK_PER_TRADE_RUPEES = 1000

# Pre-registered ship gates.
N_FLOOR = 500
PF_MARGINAL = 1.10
PF_STRONG = 1.30
PER_MONTH_WIN_MIN_PCT = 55.0
TOP_MONTH_CONC_MAX_PCT = 40.0


# ---- Data loading ------------------------------------------------------

def _load_5m_for_month(yyyy: int, mm: int) -> pd.DataFrame:
    path = (
        _REPO_ROOT / "backtest-cache-download" / "monthly"
        / f"{yyyy:04d}_{mm:02d}_5m_enriched.feather"
    )
    if not path.exists():
        return pd.DataFrame()
    keep = ["date", "symbol", "open", "high", "low", "close", "volume", "vwap"]
    df = pd.read_feather(path, columns=keep)
    for c in ("open", "high", "low", "close", "vwap"):
        df[c] = df[c].astype("float32")
    df["volume"] = df["volume"].astype("float32")
    return df


def _months_in_range(start: date, end: date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m == 13:
            m = 1
            y += 1


def build_period_5m() -> pd.DataFrame:
    """Load 5m bars for Discovery window 2024-09 .. 2025-09."""
    print(f"  loading 5m feathers {DISCOVERY_START} .. {DISCOVERY_END} ...")
    parts: List[pd.DataFrame] = []
    for y, m in _months_in_range(DISCOVERY_START, DISCOVERY_END):
        mdf = _load_5m_for_month(y, m)
        if not mdf.empty:
            parts.append(mdf)
    if not parts:
        return pd.DataFrame()
    big = pd.concat(parts, ignore_index=True, copy=False)
    big["d"] = big["date"].dt.date
    _hh = big["date"].dt.hour.astype("int16")
    _mm = big["date"].dt.minute.astype("int16")
    big["hhmm"] = (_hh * 100 + _mm).astype("int16")
    big = big[(big["d"] >= DISCOVERY_START) & (big["d"] <= DISCOVERY_END)]
    print(f"  total 5m bars (after date trim): {len(big):,}")
    print(f"  unique symbols: {big['symbol'].nunique():,}")
    return big.reset_index(drop=True)


def annotate_caps_and_filter(big5m: pd.DataFrame) -> pd.DataFrame:
    """Map each symbol to cap_segment, keep only mid_cap + small_cap."""
    syms = big5m["symbol"].unique().tolist()
    sym_to_cap: Dict[str, str] = {
        s: get_cap_segment("NSE:" + s) or "unknown" for s in syms
    }
    big5m["cap_segment"] = big5m["symbol"].map(sym_to_cap)
    cap_dist = pd.Series(sym_to_cap).value_counts().to_dict()
    print("  cap_segment distribution (unique symbols):")
    for k, v in cap_dist.items():
        print(f"    {k:<14} {v:,}")
    pre = len(big5m)
    big5m = big5m[big5m["cap_segment"].isin(ALLOWED_CAPS)].copy()
    print(f"  after cap filter ({sorted(ALLOWED_CAPS)}): {len(big5m):,} "
          f"(dropped {pre - len(big5m):,})")
    return big5m


def load_daily_for_liquidity() -> pd.DataFrame:
    daily_path = _REPO_ROOT / "cache" / "preaggregate" / "consolidated_daily.feather"
    if not daily_path.exists():
        raise FileNotFoundError(f"{daily_path} missing")
    df = pd.read_feather(daily_path)
    df["ts"] = pd.to_datetime(df["ts"])
    if df["ts"].dt.tz is not None:
        df["ts"] = df["ts"].dt.tz_localize(None)
    df["d"] = df["ts"].dt.date
    df = df[(df["d"] >= date(2024, 6, 1)) & (df["d"] <= DISCOVERY_END)]
    df = df[["symbol", "d", "close", "volume"]].copy()
    df["traded_value"] = df["close"] * df["volume"]
    df = df.sort_values(["symbol", "d"]).reset_index(drop=True)
    df["adv_20d_cr"] = df.groupby("symbol")["traded_value"].transform(
        lambda v: v.shift(1).rolling(20).mean()
    ) / 1e7
    df["pdc"] = df.groupby("symbol")["close"].shift(1)
    return df[["symbol", "d", "adv_20d_cr", "pdc"]]


# ---- Trigger discovery -------------------------------------------------

def find_triggers(big5m: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Annotate per-(symbol, session) features and apply v2 entry gates."""
    # Pre-slim: only bars we need (09:15 for gap_pct, 13:55-14:00 for day_qual,
    # and 14:30..15:00 entry window).
    in_open = big5m["hhmm"] == 915
    in_qual = big5m["hhmm"] == HHMM_1400
    in_entry = (big5m["hhmm"] >= ENTRY_HHMM_LO) & (big5m["hhmm"] <= ENTRY_HHMM_HI)
    # We also need bars BEFORE entry-window for struct-break + rvol lookback;
    # keep all bars from 09:15 to 15:00 to compute features cleanly per session.
    keep = (big5m["hhmm"] >= 915) & (big5m["hhmm"] <= 1500)
    pre = len(big5m)
    big5m = big5m[keep].copy()
    print(f"  intraday slice [09:15..15:00]: {len(big5m):,} (from {pre:,})")

    # Merge daily ADV + PDC.
    big5m = big5m.merge(daily, on=["symbol", "d"], how="left", validate="many_to_one")
    pre = len(big5m)
    big5m = big5m.dropna(subset=["adv_20d_cr", "pdc"])
    big5m = big5m[big5m["adv_20d_cr"] >= MIN_ADV_INR_CR].copy()
    print(f"  after ADV >= Rs {MIN_ADV_INR_CR}Cr: {len(big5m):,} (dropped {pre - len(big5m):,})")

    # Sort + assign per-(symbol, session) bar index.
    big5m = big5m.sort_values(["symbol", "d", "date"]).reset_index(drop=True)

    # ---- Per-session features ----
    grp = big5m.groupby(["symbol", "d"], sort=False)

    # Open of 09:15 bar -> session anchor.
    big5m["open_anchor"] = grp["open"].transform("first")
    big5m["intraday_high_so_far"] = grp["high"].cummax()
    big5m["intraday_ret_pct"] = (
        big5m["close"] / big5m["open_anchor"] - 1.0
    ) * 100.0

    # gap_pct at session open: (open_anchor - pdc) / pdc * 100.
    big5m["gap_pct"] = (big5m["open_anchor"] - big5m["pdc"]) / big5m["pdc"] * 100.0

    # intraday return at 14:00 (forward-fill across rows of that session).
    # We compute the value on the 14:00 bar, then broadcast to all rows.
    ret_at_1400 = big5m.loc[big5m["hhmm"] == HHMM_1400, ["symbol", "d", "intraday_ret_pct"]]
    ret_at_1400 = ret_at_1400.rename(columns={"intraday_ret_pct": "ret_at_1400_pct"})
    big5m = big5m.merge(ret_at_1400, on=["symbol", "d"], how="left", validate="many_to_one")

    # Structural break feature: min(prior 3 bars' lows) per session, EXCLUSIVE of current.
    # = rolling(3).min().shift(1)
    big5m["prev3_min_low"] = grp["low"].transform(
        lambda v: v.rolling(STRUCT_BREAK_LOOKBACK, min_periods=STRUCT_BREAK_LOOKBACK).min().shift(1)
    )

    # RVOL: volume / mean(prior 12 bars vol same session) -- shift(1) so exclusive of current.
    big5m["vol_mean12"] = grp["volume"].transform(
        lambda v: v.shift(1).rolling(RVOL_LOOKBACK_BARS, min_periods=5).mean()
    )
    big5m["rvol"] = big5m["volume"] / big5m["vol_mean12"].replace(0, np.nan)

    # Apply day-level qualification + entry-window mask + tape-confirmation gates.
    in_entry = (big5m["hhmm"] >= ENTRY_HHMM_LO) & (big5m["hhmm"] <= ENTRY_HHMM_HI)
    day_qual = (
        (big5m["ret_at_1400_pct"] >= INTRADAY_RET_LO_PCT)
        & (big5m["ret_at_1400_pct"] <= INTRADAY_RET_HI_PCT)
        & (big5m["gap_pct"].abs() <= MAX_ABS_GAP_PCT)
    )
    confirm = (
        (big5m["close"] < big5m["vwap"])
        & (big5m["close"] < big5m["open"])
        & (big5m["close"] < big5m["prev3_min_low"])
        & (big5m["rvol"] >= RVOL_MIN)
        & big5m["rvol"].notna()
        & big5m["prev3_min_low"].notna()
        & big5m["vwap"].notna()
    )
    qual = in_entry & day_qual & confirm
    triggers = big5m[qual].copy()
    print(f"  raw qualifying entry bars: {len(triggers):,}")

    # Latch per (symbol, day): first qualifying bar only.
    triggers = triggers.sort_values(["symbol", "d", "date"]).drop_duplicates(
        subset=["symbol", "d"], keep="first"
    )
    print(f"  unique (symbol, day) latches: {len(triggers):,}")
    return triggers.reset_index(drop=True)


# ---- Trade simulation --------------------------------------------------

def simulate(triggers: pd.DataFrame, big5m: pd.DataFrame) -> pd.DataFrame:
    print("  simulating SHORT entries -> T1/SL/time-stop ...")
    # Group source bars by symbol once.
    sym_groups: Dict[str, pd.DataFrame] = {
        sym: g.sort_values("date").reset_index(drop=True)
        for sym, g in big5m.groupby("symbol")
    }

    trades: List[dict] = []
    n_no_session = n_no_entry_idx = n_zero_dist = n_traded = 0

    for _, t in triggers.iterrows():
        sym = t["symbol"]
        sd = t["d"]
        trig_ts = t["date"]

        sym_df = sym_groups.get(sym)
        if sym_df is None:
            n_no_session += 1
            continue
        day_df = sym_df[sym_df["d"] == sd].reset_index(drop=True)
        if day_df.empty:
            n_no_session += 1
            continue

        idx_arr = day_df.index[day_df["date"] == trig_ts].tolist()
        if not idx_arr:
            n_no_entry_idx += 1
            continue
        entry_idx = idx_arr[0]
        entry_bar = day_df.iloc[entry_idx]

        entry_price = float(entry_bar["close"])
        entry_ts = entry_bar["date"]
        intra_high_at_entry = float(t["intraday_high_so_far"])
        hard_sl = max(intra_high_at_entry, entry_price * (1.0 + STOP_MIN_PCT / 100.0))
        stop_distance = hard_sl - entry_price
        if stop_distance <= 0:
            n_zero_dist += 1
            continue
        t1_target = entry_price * (1.0 - T1_PCT / 100.0)

        forward = day_df.iloc[entry_idx + 1:]
        if forward.empty:
            continue

        exit_ts = None
        exit_price = None
        exit_reason = None
        hit_t1 = False
        t1_exit_price: Optional[float] = None
        t1_exit_ts = None

        for _, bar in forward.iterrows():
            bar_ts = bar["date"]
            bar_hhmm = int(bar["hhmm"])
            high = float(bar["high"])
            low = float(bar["low"])
            close_b = float(bar["close"])

            # Active SL: hard_sl until T1 fills, then breakeven (entry).
            active_sl = entry_price if hit_t1 else hard_sl

            # SL first (worst-case fill).
            if high >= active_sl:
                exit_ts = bar_ts
                exit_price = active_sl
                exit_reason = "breakeven_trail" if hit_t1 else "stop"
                break

            # T1 partial fill check.
            if (not hit_t1) and (low <= t1_target):
                hit_t1 = True
                t1_exit_price = t1_target
                t1_exit_ts = bar_ts

            # Hard time stop at 15:10.
            if bar_hhmm >= TIMESTOP_HHMM:
                exit_ts = bar_ts
                exit_price = close_b
                exit_reason = "time_stop_1510"
                break

        # Walked off end of session (shouldn't happen since 15:10 bar exists in mostly all sessions).
        if exit_price is None:
            last = forward.iloc[-1]
            exit_ts = last["date"]
            exit_price = float(last["close"])
            exit_reason = "eod"

        # Position size: risk Rs.1000 over stop_distance, qty = floor(1000 / stop_distance).
        qty = max(int(RISK_PER_TRADE_RUPEES / max(stop_distance, 1e-6)), 1)

        if hit_t1:
            qty_t1 = qty // 2
            qty_t2 = qty - qty_t1
            pnl_t1 = (entry_price - t1_exit_price) * qty_t1
            pnl_t2 = (entry_price - exit_price) * qty_t2
            realized_pnl = pnl_t1 + pnl_t2
            fee_t1 = calc_fee(entry_price, t1_exit_price, qty_t1, "SELL")
            fee_t2 = calc_fee(entry_price, exit_price, qty_t2, "SELL")
            fee = fee_t1 + fee_t2
            blended_exit = (t1_exit_price * qty_t1 + exit_price * qty_t2) / max(qty, 1)
        else:
            realized_pnl = (entry_price - exit_price) * qty
            fee = calc_fee(entry_price, exit_price, qty, "SELL")
            blended_exit = exit_price

        net_pnl = realized_pnl - fee

        trades.append({
            "T1_entry_date": sd,
            "symbol": "NSE:" + sym,
            "cap_segment": t["cap_segment"],
            "side": "SHORT",
            "entry_hhmm": int(t["hhmm"]),
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "intraday_ret_at_1400_pct": float(t["ret_at_1400_pct"]),
            "rvol": float(t["rvol"]),
            "intraday_high_at_entry": intra_high_at_entry,
            "hard_sl": hard_sl,
            "t1_target": t1_target,
            "stop_distance": stop_distance,
            "hit_t1": hit_t1,
            "exit_ts": exit_ts,
            "exit_price": blended_exit,
            "exit_reason": exit_reason,
            "qty": qty,
            "realized_pnl": realized_pnl,
            "fee": fee,
            "net_pnl": net_pnl,
        })
        n_traded += 1

    print(f"    no session in 5m:    {n_no_session}")
    print(f"    no entry bar idx:    {n_no_entry_idx}")
    print(f"    zero stop distance:  {n_zero_dist}")
    print(f"    traded:              {n_traded}")
    return pd.DataFrame(trades)


# ---- Reporting ---------------------------------------------------------

def report(trades: pd.DataFrame) -> str:
    """Run gates + verdict logic; return verdict label."""
    if trades.empty:
        print("\n[NO TRADES] sanity check returns 0 trades")
        return "RETIRE"
    n = len(trades)
    npnl = trades["net_pnl"]
    wins = npnl[npnl > 0].sum()
    losses = npnl[npnl < 0].abs().sum()
    pf = round(float(wins / losses), 3) if losses > 0 else float("inf")
    daily = trades.groupby("T1_entry_date")["net_pnl"].sum()
    sharpe = (
        round(float(daily.mean() / daily.std()), 3)
        if daily.std() > 0 else 0.0
    )
    wr = round(float((npnl > 0).mean()) * 100, 1)
    net = int(npnl.sum())

    print("\n=== mis_unwind_short v2 -- Discovery sanity (2024-09 .. 2025-09) ===")
    print(f"Trades: n = {n}")
    print(f"Win rate: {wr}%")
    print(f"Gross PnL: Rs.{int(trades['realized_pnl'].sum()):,}")
    print(f"Fees:      Rs.{int(trades['fee'].sum()):,}")
    print(f"NET PnL:   Rs.{net:,}")
    print(f"NET PF:    {pf}")
    print(f"NET Sharpe (daily): {sharpe}")

    print("\nPer cap_segment:")
    for cap, grp in trades.groupby("cap_segment"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(float(w / l), 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net2 = int(grp["net_pnl"].sum())
        print(f"  {cap:<12} n={n2:>5} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net2:>11,}")

    print("\nPer month:")
    months = pd.to_datetime(trades["T1_entry_date"]).dt.to_period("M").astype(str)
    monthly_rows = []
    for m, grp in trades.assign(_m=months.values).groupby("_m"):
        n2 = len(grp)
        w = grp["net_pnl"][grp["net_pnl"] > 0].sum()
        l = grp["net_pnl"][grp["net_pnl"] < 0].abs().sum()
        pf2 = round(float(w / l), 3) if l > 0 else float("inf")
        wr2 = round(float((grp["net_pnl"] > 0).mean()) * 100, 1)
        net2 = int(grp["net_pnl"].sum())
        monthly_rows.append((m, n2, pf2, wr2, net2))
        print(f"  {m}  n={n2:>4} PF={pf2:>5} WR={wr2:>5}% netPnL=Rs.{net2:>11,}")

    n_months = len(monthly_rows)
    winning_months = sum(1 for r in monthly_rows if r[4] > 0)
    pct_winning_months = round(winning_months / n_months * 100.0, 1) if n_months else 0.0
    top_month_net = max((abs(r[4]) for r in monthly_rows), default=0)
    top_month_pct_of_net = (
        round(top_month_net / abs(net) * 100.0, 1) if net != 0 else 0.0
    )
    print(f"\nWinning months: {winning_months}/{n_months} = {pct_winning_months}%")
    print(f"Top month |NET|: Rs.{top_month_net:,} = {top_month_pct_of_net}% of |aggregate NET|")

    print("\nExit-reason breakdown:")
    for rsn, grp in trades.groupby("exit_reason"):
        n2 = len(grp)
        avg = int(grp["net_pnl"].mean())
        print(f"  {rsn:<22} n={n2:>5} avg_net=Rs.{avg:>6,}")

    # ---- Verdict ----
    print("\n--- VERDICT (pre-registered gates) ---")
    fails: List[str] = []
    if n < N_FLOOR:
        fails.append(f"n {n} < {N_FLOOR}")
    if pf < PF_MARGINAL:
        fails.append(f"PF {pf} < {PF_MARGINAL}")
    if sharpe <= 0:
        fails.append(f"Sharpe {sharpe} <= 0")
    if pct_winning_months < PER_MONTH_WIN_MIN_PCT:
        fails.append(
            f"per-month winning {pct_winning_months}% < {PER_MONTH_WIN_MIN_PCT}%"
        )
    if top_month_pct_of_net > TOP_MONTH_CONC_MAX_PCT:
        fails.append(
            f"top-month concentration {top_month_pct_of_net}% > {TOP_MONTH_CONC_MAX_PCT}%"
        )

    if pf >= PF_STRONG and n >= 100 and not fails:
        verdict = "STRONG_PROCEED"
        print(f"PF={pf} >= {PF_STRONG} AND n={n} >= 100 AND stability passes")
        print("-> STRONG PROCEED. Write detailed brief + OOS test on 2025-10 .. 2026-04.")
    elif pf >= PF_MARGINAL and not fails:
        verdict = "MARGINAL"
        print(f"PF={pf} in [{PF_MARGINAL}, {PF_STRONG}) -> MARGINAL.")
        print("Log for later. Do not promote to detector yet.")
    else:
        verdict = "RETIRE"
        if not fails and pf < PF_MARGINAL:
            fails.append(f"PF {pf} < {PF_MARGINAL}")
        print(f"PF={pf} | FAILED gates -> RETIRE")
        for f in fails:
            print(f"  - {f}")
    return verdict


def run_regime_break_preflight() -> None:
    """Pre-flight: declare depends_on tags and check Discovery window for breaks."""
    print("\n--- regime_break_detector pre-flight ---")
    depends_on = ["MIS_leverage", "F&O_speculation"]
    print(f"  strategy: mis_unwind_short_v2")
    print(f"  depends_on: {depends_on}")
    print(f"  window: Discovery {DISCOVERY_START} .. {DISCOVERY_END}")
    try:
        hits = check_window(
            strategy_name="mis_unwind_short_v2",
            depends_on=depends_on,
            window_label="Discovery",
            start=DISCOVERY_START,
            end=DISCOVERY_END,
            min_severity="high",
            raise_on_break=False,  # don't crash sanity; we want to enumerate
        )
        if not hits:
            print("  no high/critical rule changes in window. PROCEED.")
        else:
            print(f"  WARNING: {len(hits)} rule change(s) in window:")
            for r in hits:
                print(f"    - {r.effective_date} [{r.severity.upper()}] {r.description}")
            print("  Accepted with caveat (see specs/2026-05-14-research-mis_unwind_retry.md §6):")
            print("    setup is cash-equity MIS unwind, not F&O speculation; STT/option-premium")
            print("    rule changes do not affect retail cash-equity MIS long inventory dynamics.")
    except GauntletRegimeBreak as e:
        print(f"  REGIME BREAK: {e}")
        raise


def main():
    run_regime_break_preflight()
    big5m = build_period_5m()
    if big5m.empty:
        print("[ABORT] no 5m feathers found")
        return
    big5m = annotate_caps_and_filter(big5m)
    if big5m.empty:
        print("[ABORT] no bars after cap filter")
        return
    daily = load_daily_for_liquidity()

    print("\nFinding triggers ...")
    triggers = find_triggers(big5m, daily)
    if triggers.empty:
        print("[NO TRIGGERS] -- 6-gate filter selected zero bars.")
        return
    print(f"\nTotal triggers (after latch): {len(triggers)}")

    print("\nSimulating entries -> exits:")
    trades = simulate(triggers, big5m)
    report(trades)

    out = (
        _REPO_ROOT / "reports" / "sub9_sanity"
        / "mis_unwind_short_v2_trades.csv"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out, index=False)
    print(f"\nFull trade log: {out}")


if __name__ == "__main__":
    main()
